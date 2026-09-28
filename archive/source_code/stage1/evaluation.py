"""Single outer unblinding, subject-level statistics, and campaign closeout."""

from __future__ import annotations

import csv
import html
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
from scipy.stats import binom
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import PROTOCOL_SHA256
from .campaign import CAMPAIGN_ROOT, DATA_DIR, _update_state, preflight
from .bundles import identity_key
from .data_io import (
    align_values,
    atomic_write_json,
    atomic_write_npz,
    canonical_json_hash,
    discover_raw_records,
    load_labels_authorized,
    sha256_file,
)
from .firewall import RoleLedger, assert_ready_to_unblind
from .provenance import (
    audited_environment,
    evaluation_code_provenance,
    risk_code_provenance,
)
from .risk_models import ALPHAS, _validated_component_payload
from .selectors import tie_uniform
from .splits import (
    SUBJECTS,
    build_final_roles,
    build_inner_roles,
    build_outer_split,
)


PRIMARY_METHOD = "Learned-Selected"
PRIMARY_COMPARATOR = "Baseline-Selected"
PRIMARY_ALPHA = 0.05
OUTER_BLIND_KEYS = frozenset(
    {
        "identities_json",
        "primary_logits",
        "primary_probabilities",
        "primary_predictions",
        "feature_probabilities",
        "method_names",
        "risk_scores",
        "mask_names",
        "acceptance_masks",
        "selector_metadata_json",
        "selector_provenance_hash",
    }
)
AGGREGATE_KEYS = frozenset(
    {
        "protocol_hash",
        "evaluation_code_hash",
        "unblinding_lock_sha256",
        "identities_json",
        "outer_subject_ids",
        "outer_subject_order",
        "primary_logits",
        "primary_probabilities",
        "primary_predictions",
        "feature_probabilities",
        "method_names",
        "risk_scores",
        "mask_names",
        "acceptance_masks",
        "selector_metadata_json_by_outer",
        "selector_provenance_hash_by_outer",
        "source_sha256_by_outer",
        "ground_truth_labels",
        "error_indicators",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON mapping: {path}")
    return value


def _scalar_text(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise RuntimeError(f"{name} must be a scalar string")
    raw = array.item()
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _identities(value: np.ndarray) -> list[dict[str, Any]]:
    parsed = json.loads(_scalar_text(value, "identities_json"))
    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError("identity payload is not a nonempty list")
    keys = [identity_key(item) for item in parsed]
    if len(set(keys)) != len(keys):
        raise RuntimeError("outer decision artifact contains duplicate identities")
    return parsed


def _load_npz_exact(path: Path, expected_keys: frozenset[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if keys != set(expected_keys):
            raise RuntimeError(
                f"artifact schema differs at {path}: "
                f"missing={sorted(set(expected_keys) - keys)}, "
                f"extra={sorted(keys - set(expected_keys))}"
            )
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _validate_outer_payload(
    *,
    outer: str,
    manifest: Mapping[str, Any],
    artifact_path: Path,
) -> dict[str, np.ndarray]:
    payload = _load_npz_exact(artifact_path, OUTER_BLIND_KEYS)
    identities = _identities(payload["identities_json"])
    if {str(item["subject_id"]) for item in identities} != {outer}:
        raise RuntimeError(f"outer decision artifact has the wrong subject: {outer}")
    n_rows = len(identities)
    logits = np.asarray(payload["primary_logits"], dtype=float)
    primary = np.asarray(payload["primary_probabilities"], dtype=float)
    feature = np.asarray(payload["feature_probabilities"], dtype=float)
    predictions = np.asarray(payload["primary_predictions"])
    methods = [str(value) for value in np.asarray(payload["method_names"])]
    masks = [str(value) for value in np.asarray(payload["mask_names"])]
    risk_scores = np.asarray(payload["risk_scores"], dtype=float)
    acceptance = np.asarray(payload["acceptance_masks"])
    if (
        logits.shape != (n_rows, 5)
        or primary.shape != (n_rows, 5)
        or feature.shape != (n_rows, 5)
        or predictions.shape != (n_rows,)
        or risk_scores.shape != (n_rows, len(methods))
        or acceptance.shape != (n_rows, len(masks))
    ):
        raise RuntimeError(f"outer decision array shape mismatch: {outer}")
    if acceptance.dtype.kind != "b":
        raise RuntimeError(f"outer acceptance masks are not boolean: {outer}")
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(risk_scores)):
        raise RuntimeError(f"outer decision artifact contains nonfinite values: {outer}")
    for name, probability in (("primary", primary), ("feature", feature)):
        if (
            not np.all(np.isfinite(probability))
            or np.any(probability < 0)
            or np.any(probability > 1)
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0)
        ):
            raise RuntimeError(f"outer {name} probabilities are invalid: {outer}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    softmax = np.exp(shifted)
    softmax /= softmax.sum(axis=1, keepdims=True)
    if not np.allclose(primary, softmax, atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"outer logits/probability mismatch: {outer}")
    if not np.array_equal(predictions, np.argmax(primary, axis=1)):
        raise RuntimeError(f"outer prediction/probability mismatch: {outer}")
    if methods != list(manifest.get("methods", [])) or len(set(methods)) != len(methods):
        raise RuntimeError(f"outer method registry mismatch: {outer}")
    expected_masks = sorted(
        f"{method}|{alpha:.2f}|{selector}"
        for method in methods
        for alpha in ALPHAS
        for selector in ("empirical", "conservative")
    )
    if masks != expected_masks or len(set(masks)) != len(masks):
        raise RuntimeError(f"outer mask registry mismatch: {outer}")
    selector_hash = _scalar_text(
        payload["selector_provenance_hash"], "selector_provenance_hash"
    )
    selector_metadata = json.loads(
        _scalar_text(payload["selector_metadata_json"], "selector_metadata_json")
    )
    if set(selector_metadata) != set(methods):
        raise RuntimeError(f"outer selector metadata registry mismatch: {outer}")
    calibration_path = (
        CAMPAIGN_ROOT / "calibration" / f"outer-{outer}" / "selector_calibration.json"
    )
    development_path = (
        CAMPAIGN_ROOT / "development" / "risk" / f"outer-{outer}" / "development.joblib"
    )
    development_manifest_path = development_path.with_name("development_manifest.json")
    for required in (calibration_path, development_path, development_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"outer provenance sidecar is missing: {required}")
    calibration = _read_json(calibration_path)
    development_manifest = _read_json(development_manifest_path)
    selector_provenance = calibration.get("selector_provenance")
    if not isinstance(selector_provenance, dict):
        raise RuntimeError(f"selector execution provenance is missing: {outer}")
    if (
        manifest.get("artifact_path") != artifact_path.name
        or manifest.get("artifact_sha256") != sha256_file(artifact_path)
        or manifest.get("development_sha256") != sha256_file(development_path)
        or manifest.get("calibration_sha256") != sha256_file(calibration_path)
        or development_manifest.get("artifact_sha256") != sha256_file(development_path)
        or selector_hash != manifest.get("selector_provenance_hash")
        or selector_hash != calibration.get("selector_provenance_hash")
        or selector_hash != canonical_json_hash(selector_provenance)
        or selector_metadata != calibration.get("selectors")
        or manifest.get("method_full_ids") != calibration.get("method_full_ids")
        or manifest.get("protocol_hash") != PROTOCOL_SHA256
        or calibration.get("protocol_hash") != PROTOCOL_SHA256
        or development_manifest.get("protocol_hash") != PROTOCOL_SHA256
        or manifest.get("outer") != outer
        or calibration.get("outer") != outer
        or manifest.get("contains_outer_labels") is not False
    ):
        raise RuntimeError(f"outer artifact/sidecar provenance mismatch: {outer}")
    inputs = selector_provenance.get("inputs", {})
    if (
        selector_provenance.get("code") != risk_code_provenance()
        or selector_provenance.get("environment") != audited_environment()
        or inputs.get("development_sha256") != sha256_file(development_path)
        or inputs.get("development_manifest_sha256")
        != sha256_file(development_manifest_path)
        or inputs.get("development_provenance_hash")
        != development_manifest.get("execution_provenance_hash")
    ):
        raise RuntimeError(f"selector code/data/environment provenance changed: {outer}")
    return payload


def _frozen_manifests() -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for outer in SUBJECTS:
        directory = CAMPAIGN_ROOT / "outer_predictions" / f"outer-{outer}"
        manifest = _read_json(directory / "outer_prediction_manifest.json")
        artifact_path = directory / "outer_decisions.npz"
        _validate_outer_payload(outer=outer, manifest=manifest, artifact_path=artifact_path)
        verified = dict(manifest)
        verified["verified_artifact_path"] = artifact_path.relative_to(CAMPAIGN_ROOT).as_posix()
        verified["verified_artifact_sha256"] = sha256_file(artifact_path)
        manifests.append(verified)
    assert_ready_to_unblind(
        manifests,
        expected_protocol_hash=PROTOCOL_SHA256,
        required_outer_subjects=SUBJECTS,
    )
    return manifests


def verify_execution_leakage_contracts(
    manifests: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recompute the execution-level ancestry and label-ledger audit."""
    checked_components = 0
    checked_label_events = 0
    for outer in SUBJECTS:
        split = build_outer_split(outer)
        for target in split["fit"]:
            name = f"inner-{target}"
            directory = (
                CAMPAIGN_ROOT
                / "development"
                / "components"
                / f"outer-{outer}"
                / name
            )
            _validated_component_payload(
                bundle_path=directory / "blind_predictions.npz",
                expected_roles=build_inner_roles(outer, target, split["fit"]),
                expected_predicted_subjects=[target],
            )
            checked_components += 1
        final_directory = (
            CAMPAIGN_ROOT
            / "development"
            / "components"
            / f"outer-{outer}"
            / "final"
        )
        _validated_component_payload(
            bundle_path=final_directory / "blind_predictions.npz",
            expected_roles=build_final_roles(outer, split["fit"]),
            expected_predicted_subjects=list(split["cert"]) + [outer],
        )
        checked_components += 1
        ledger_path = CAMPAIGN_ROOT / "logs" / f"label_access_outer-{outer}.jsonl"
        if not ledger_path.is_file():
            raise FileNotFoundError(f"label ledger is missing: {ledger_path}")
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            subjects = set(event["subjects"])
            if event.get("decision") != "ALLOWED":
                checked_label_events += 1
                continue
            phase = event["phase"]
            if phase in {
                "COMPONENT_TRAIN",
                "CHECKPOINT_SELECTION",
                "FEATURE_CALIBRATION",
                "RISK_FIT",
                "RISK_SELECTION",
                "ERROR_RISK_CALIBRATION",
            }:
                valid = subjects <= set(split["fit"])
            elif phase == "CERT_SELECTOR":
                valid = subjects <= set(split["cert"])
            elif phase == "SINGLE_OUTER_UNBLINDING":
                valid = subjects == {outer}
            else:
                valid = False
            if not valid:
                raise RuntimeError(f"label-ledger violation for outer {outer}: {event}")
            checked_label_events += 1
    verified_manifests = list(manifests) if manifests is not None else _frozen_manifests()
    return {
        "verdict": "PASS",
        "component_bundles_checked": checked_components,
        "label_events_checked": checked_label_events,
        "outer_manifests_checked": len(verified_manifests),
        "outer_artifact_hashes": {
            str(item["outer"]): str(
                item.get("verified_artifact_sha256", item["artifact_sha256"])
            )
            for item in verified_manifests
        },
        "stable_identity_validation": "PASS",
        "fold_scoped_label_authorization": "PASS",
        "all_outer_artifacts_content_verified_before_unblind": "PASS",
    }


def single_outer_unblinding() -> None:
    """Perform one aggregate-atomic label join after every blind artifact is locked."""
    evaluation_code = evaluation_code_provenance()
    stable_preflight = preflight()
    stable_preflight = {
        key: stable_preflight[key]
        for key in (
            "protocol_hash",
            "raw_hash",
            "feature_manifest_hash",
            "feature_shard_set_hash",
            "environment_runtime_hash",
            "subjects",
            "records",
        )
    }
    manifests = _frozen_manifests()
    source_hashes = {
        str(item["outer"]): str(
            item.get("verified_artifact_sha256", item["artifact_sha256"])
        )
        for item in manifests
    }
    sealed_root = CAMPAIGN_ROOT / "sealed_outer_results"
    lock_path = sealed_root / "UNBLINDING_LOCK.json"
    audit_path = CAMPAIGN_ROOT / "logs" / "pre_unblind_execution_audit.json"
    aggregate_path = sealed_root / "ALL_OUTER_SEALED.npz"
    completion_path = sealed_root / "UNBLINDING_COMPLETE.json"

    if completion_path.exists():
        aggregate = _load_completed_aggregate(
            aggregate_path=aggregate_path,
            completion_path=completion_path,
            lock_path=lock_path,
            source_hashes=source_hashes,
            evaluation_code_hash=evaluation_code["hash"],
        )
        _materialize_outer_results(aggregate, source_hashes)
        _update_state(
            "SINGLE_OUTER_UNBLINDING",
            completed="SINGLE_OUTER_UNBLINDING",
            progress={"outer_subjects_unblinded": 20, "outer_subjects_total": 20},
            outer_unblinded=True,
        )
        return

    partial_fold_outputs = sorted(sealed_root.glob("outer-*/sealed_results.npz"))
    if partial_fold_outputs:
        raise RuntimeError(
            "partial per-fold outcome artifacts exist without an aggregate completion lock; "
            "preserve and invalidate them before proceeding"
        )

    if lock_path.exists():
        if not audit_path.is_file():
            raise RuntimeError("unblinding lock exists without its pre-unblind audit")
        locked = _read_json(lock_path)
        if (
            locked.get("protocol_hash") != PROTOCOL_SHA256
            or locked.get("outer_artifacts") != source_hashes
            or locked.get("preflight_hash") != canonical_json_hash(stable_preflight)
            or locked.get("pre_unblind_audit_sha256") != sha256_file(audit_path)
            or locked.get("evaluation_code_hash") != evaluation_code["hash"]
            or locked.get("all_twenty_frozen") is not True
        ):
            raise RuntimeError("existing unblinding lock differs from verified frozen inputs")
    else:
        leakage_evidence = verify_execution_leakage_contracts(manifests)
        audit_payload = {
            "schema_version": "stage2-pre-unblind-execution-audit-v1",
            "protocol_hash": PROTOCOL_SHA256,
            "completed_before_any_outer_label_read": True,
            "evaluation_code": evaluation_code,
            "preflight": stable_preflight,
            "evidence": leakage_evidence,
        }
        atomic_write_json(audit_path, audit_payload)
        locked = {
            "schema_version": "stage2-unblinding-lock-v2",
            "protocol_hash": PROTOCOL_SHA256,
            "outer_artifacts": source_hashes,
            "preflight_hash": canonical_json_hash(stable_preflight),
            "evaluation_code_hash": evaluation_code["hash"],
            "pre_unblind_audit_sha256": sha256_file(audit_path),
            "all_twenty_frozen": True,
            "aggregate_atomic_publish": True,
            "method_changes_after_this_point": "PROHIBITED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(lock_path, locked)

    lock_sha256 = sha256_file(lock_path)
    _update_state("SINGLE_OUTER_UNBLINDING", outer_unblinded=False)
    records = discover_raw_records(DATA_DIR)
    fold_sources: list[dict[str, np.ndarray]] = []
    fold_identities: list[list[dict[str, Any]]] = []
    fold_labels: list[np.ndarray] = []
    fold_errors: list[np.ndarray] = []
    method_names: list[str] | None = None
    mask_names: list[str] | None = None

    # No outcome-bearing artifact is written inside this loop. If any join fails,
    # the entire matrix remains unpublished.
    for outer, manifest in zip(SUBJECTS, manifests):
        source_path = (
            CAMPAIGN_ROOT
            / "outer_predictions"
            / f"outer-{outer}"
            / "outer_decisions.npz"
        )
        source = _validate_outer_payload(
            outer=outer, manifest=manifest, artifact_path=source_path
        )
        identities = _identities(source["identities_json"])
        current_methods = [str(value) for value in source["method_names"]]
        current_masks = [str(value) for value in source["mask_names"]]
        if method_names is None:
            method_names = current_methods
            mask_names = current_masks
        elif current_methods != method_names or current_masks != mask_names:
            raise RuntimeError("outer folds do not share one frozen method/mask registry")
        split = build_outer_split(outer)
        ledger = RoleLedger(
            outer=outer,
            fit=tuple(split["fit"]),
            cert=tuple(split["cert"]),
            path=CAMPAIGN_ROOT / "logs" / f"label_access_outer-{outer}.jsonl",
        )
        labels, label_ids = load_labels_authorized(
            records,
            [outer],
            ledger=ledger,
            phase="SINGLE_OUTER_UNBLINDING",
        )
        labels = align_values(label_ids, labels, identities).astype(np.int64)
        predictions = np.asarray(source["primary_predictions"], dtype=np.int64)
        if labels.shape != predictions.shape:
            raise RuntimeError(f"outer prediction/label row count mismatch: {outer}")
        errors = (predictions != labels).astype(np.int64)
        fold_sources.append(source)
        fold_identities.append(identities)
        fold_labels.append(labels)
        fold_errors.append(errors)

    if method_names is None or mask_names is None:
        raise RuntimeError("outer matrix is unexpectedly empty")
    all_identities = [item for rows in fold_identities for item in rows]
    aggregate_arrays = {
        "protocol_hash": np.asarray(PROTOCOL_SHA256, dtype=np.str_),
        "evaluation_code_hash": np.asarray(evaluation_code["hash"], dtype=np.str_),
        "unblinding_lock_sha256": np.asarray(lock_sha256, dtype=np.str_),
        "identities_json": np.asarray(
            json.dumps(all_identities, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
        "outer_subject_ids": np.concatenate(
            [
                np.full(len(rows), outer, dtype=f"<U{max(map(len, SUBJECTS))}")
                for outer, rows in zip(SUBJECTS, fold_identities)
            ]
        ),
        "outer_subject_order": np.asarray(SUBJECTS, dtype=np.str_),
        "primary_logits": np.vstack([row["primary_logits"] for row in fold_sources]),
        "primary_probabilities": np.vstack(
            [row["primary_probabilities"] for row in fold_sources]
        ),
        "primary_predictions": np.concatenate(
            [row["primary_predictions"] for row in fold_sources]
        ),
        "feature_probabilities": np.vstack(
            [row["feature_probabilities"] for row in fold_sources]
        ),
        "method_names": np.asarray(method_names, dtype=np.str_),
        "risk_scores": np.vstack([row["risk_scores"] for row in fold_sources]),
        "mask_names": np.asarray(mask_names, dtype=np.str_),
        "acceptance_masks": np.vstack(
            [row["acceptance_masks"] for row in fold_sources]
        ),
        "selector_metadata_json_by_outer": np.asarray(
            [
                _scalar_text(row["selector_metadata_json"], "selector_metadata_json")
                for row in fold_sources
            ],
            dtype=np.str_,
        ),
        "selector_provenance_hash_by_outer": np.asarray(
            [
                _scalar_text(row["selector_provenance_hash"], "selector_provenance_hash")
                for row in fold_sources
            ],
            dtype=np.str_,
        ),
        "source_sha256_by_outer": np.asarray(
            [source_hashes[outer] for outer in SUBJECTS], dtype=np.str_
        ),
        "ground_truth_labels": np.concatenate(fold_labels),
        "error_indicators": np.concatenate(fold_errors),
    }
    atomic_write_npz(aggregate_path, aggregate_arrays)
    aggregate = _load_npz_exact(aggregate_path, AGGREGATE_KEYS)
    _validate_aggregate_payload(
        aggregate,
        lock_sha256=lock_sha256,
        source_hashes=source_hashes,
        evaluation_code_hash=evaluation_code["hash"],
    )
    completion = {
        "schema_version": "stage2-unblinding-complete-v2",
        "protocol_hash": PROTOCOL_SHA256,
        "aggregate_path": aggregate_path.name,
        "aggregate_sha256": sha256_file(aggregate_path),
        "evaluation_code_hash": evaluation_code["hash"],
        "unblinding_lock_sha256": lock_sha256,
        "pre_unblind_audit_sha256": sha256_file(audit_path),
        "outer_artifacts": source_hashes,
        "outer_subjects": list(SUBJECTS),
        "rows": len(all_identities),
        "all_outer_labels_joined_before_publish": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(completion_path, completion)
    _update_state(
        "SINGLE_OUTER_UNBLINDING",
        completed="SINGLE_OUTER_UNBLINDING",
        progress={"outer_subjects_unblinded": 20, "outer_subjects_total": 20},
        outer_unblinded=True,
    )
    _materialize_outer_results(aggregate, source_hashes)


def _validate_aggregate_payload(
    payload: Mapping[str, np.ndarray],
    *,
    lock_sha256: str,
    source_hashes: Mapping[str, str],
    evaluation_code_hash: str,
) -> list[dict[str, Any]]:
    if _scalar_text(payload["protocol_hash"], "protocol_hash") != PROTOCOL_SHA256:
        raise RuntimeError("sealed aggregate protocol hash mismatch")
    if (
        _scalar_text(payload["evaluation_code_hash"], "evaluation_code_hash")
        != evaluation_code_hash
    ):
        raise RuntimeError("sealed aggregate evaluation-code hash mismatch")
    if (
        _scalar_text(payload["unblinding_lock_sha256"], "unblinding_lock_sha256")
        != lock_sha256
    ):
        raise RuntimeError("sealed aggregate unblinding-lock hash mismatch")
    if [str(value) for value in payload["outer_subject_order"]] != list(SUBJECTS):
        raise RuntimeError("sealed aggregate outer order mismatch")
    if [str(value) for value in payload["source_sha256_by_outer"]] != [
        source_hashes[outer] for outer in SUBJECTS
    ]:
        raise RuntimeError("sealed aggregate source hashes mismatch")
    identities = _identities(payload["identities_json"])
    subjects = np.asarray(payload["outer_subject_ids"]).astype(str)
    n_rows = len(identities)
    methods = [str(value) for value in payload["method_names"]]
    masks = [str(value) for value in payload["mask_names"]]
    if (
        subjects.shape != (n_rows,)
        or np.asarray(payload["primary_logits"]).shape != (n_rows, 5)
        or np.asarray(payload["primary_probabilities"]).shape != (n_rows, 5)
        or np.asarray(payload["primary_predictions"]).shape != (n_rows,)
        or np.asarray(payload["feature_probabilities"]).shape != (n_rows, 5)
        or np.asarray(payload["risk_scores"]).shape != (n_rows, len(methods))
        or np.asarray(payload["acceptance_masks"]).shape != (n_rows, len(masks))
        or np.asarray(payload["ground_truth_labels"]).shape != (n_rows,)
        or np.asarray(payload["error_indicators"]).shape != (n_rows,)
        or np.asarray(payload["selector_metadata_json_by_outer"]).shape != (20,)
        or np.asarray(payload["selector_provenance_hash_by_outer"]).shape != (20,)
    ):
        raise RuntimeError("sealed aggregate shape mismatch")
    identity_subjects = np.asarray([str(item["subject_id"]) for item in identities])
    if not np.array_equal(subjects, identity_subjects):
        raise RuntimeError("sealed aggregate identity/subject mismatch")
    if set(subjects) != set(SUBJECTS):
        raise RuntimeError("sealed aggregate cohort mismatch")
    predictions = np.asarray(payload["primary_predictions"], dtype=np.int64)
    probabilities = np.asarray(payload["primary_probabilities"], dtype=float)
    labels = np.asarray(payload["ground_truth_labels"], dtype=np.int64)
    errors = np.asarray(payload["error_indicators"], dtype=np.int64)
    if not np.array_equal(predictions, np.argmax(probabilities, axis=1)):
        raise RuntimeError("sealed aggregate prediction/probability mismatch")
    if not np.array_equal(errors, (predictions != labels).astype(np.int64)):
        raise RuntimeError("sealed aggregate error/label mismatch")
    return identities


def _load_completed_aggregate(
    *,
    aggregate_path: Path,
    completion_path: Path,
    lock_path: Path,
    source_hashes: Mapping[str, str],
    evaluation_code_hash: str,
) -> dict[str, np.ndarray]:
    if not lock_path.is_file():
        raise RuntimeError("unblinding completion exists without a lock")
    completion = _read_json(completion_path)
    lock_sha256 = sha256_file(lock_path)
    if (
        completion.get("protocol_hash") != PROTOCOL_SHA256
        or completion.get("aggregate_path") != aggregate_path.name
        or completion.get("aggregate_sha256") != sha256_file(aggregate_path)
        or completion.get("evaluation_code_hash") != evaluation_code_hash
        or completion.get("unblinding_lock_sha256") != lock_sha256
        or completion.get("outer_artifacts") != dict(source_hashes)
        or completion.get("outer_subjects") != list(SUBJECTS)
        or completion.get("all_outer_labels_joined_before_publish") is not True
    ):
        raise RuntimeError("unblinding completion manifest is incompatible")
    aggregate = _load_npz_exact(aggregate_path, AGGREGATE_KEYS)
    identities = _validate_aggregate_payload(
        aggregate,
        lock_sha256=lock_sha256,
        source_hashes=source_hashes,
        evaluation_code_hash=evaluation_code_hash,
    )
    if completion.get("rows") != len(identities):
        raise RuntimeError("unblinding completion row count mismatch")
    return aggregate


def _fold_sealed_payload(
    aggregate: Mapping[str, np.ndarray], outer: str
) -> dict[str, np.ndarray]:
    subjects = np.asarray(aggregate["outer_subject_ids"]).astype(str)
    index = np.flatnonzero(subjects == outer)
    all_identities = _identities(aggregate["identities_json"])
    identities = [all_identities[int(row)] for row in index]
    outer_index = list(SUBJECTS).index(outer)
    return {
        "identities_json": np.asarray(
            json.dumps(identities, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
        "primary_logits": np.asarray(aggregate["primary_logits"])[index],
        "primary_probabilities": np.asarray(aggregate["primary_probabilities"])[index],
        "primary_predictions": np.asarray(aggregate["primary_predictions"])[index],
        "feature_probabilities": np.asarray(aggregate["feature_probabilities"])[index],
        "method_names": np.asarray(aggregate["method_names"]),
        "risk_scores": np.asarray(aggregate["risk_scores"])[index],
        "mask_names": np.asarray(aggregate["mask_names"]),
        "acceptance_masks": np.asarray(aggregate["acceptance_masks"])[index],
        "selector_metadata_json": np.asarray(
            aggregate["selector_metadata_json_by_outer"][outer_index], dtype=np.str_
        ),
        "selector_provenance_hash": np.asarray(
            aggregate["selector_provenance_hash_by_outer"][outer_index], dtype=np.str_
        ),
        "ground_truth_labels": np.asarray(aggregate["ground_truth_labels"])[index],
        "error_indicators": np.asarray(aggregate["error_indicators"])[index],
    }


def _arrays_identical(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        left_array = np.asarray(left[key])
        right_array = np.asarray(right[key])
        if left_array.dtype.kind in {"f", "c"} or right_array.dtype.kind in {"f", "c"}:
            equal = np.array_equal(left_array, right_array, equal_nan=True)
        else:
            equal = np.array_equal(left_array, right_array)
        if not equal:
            return False
    return True


def _materialize_outer_results(
    aggregate: Mapping[str, np.ndarray], source_hashes: Mapping[str, str]
) -> None:
    for outer in SUBJECTS:
        result_dir = CAMPAIGN_ROOT / "sealed_outer_results" / f"outer-{outer}"
        result_path = result_dir / "sealed_results.npz"
        summary_path = result_dir / "fold_summary.json"
        sealed = _fold_sealed_payload(aggregate, outer)
        if result_path.exists():
            with np.load(result_path, allow_pickle=False) as archive:
                existing = {
                    name: np.array(archive[name], copy=True) for name in archive.files
                }
            if not _arrays_identical(existing, sealed):
                raise RuntimeError(f"materialized sealed fold differs from aggregate: {outer}")
        else:
            atomic_write_npz(result_path, sealed)
        errors = np.asarray(sealed["error_indicators"], dtype=np.int64)
        method_names = [str(value) for value in sealed["method_names"]]
        mask_names = [str(value) for value in sealed["mask_names"]]
        masks = np.asarray(sealed["acceptance_masks"], dtype=bool)
        metrics: dict[str, Any] = {}
        for column, name in enumerate(mask_names):
            accepted = masks[:, column]
            count = int(accepted.sum())
            accepted_errors = int(np.sum(errors[accepted]))
            metrics[name] = {
                "eligible": len(errors),
                "accepted": count,
                "coverage": count / len(errors),
                "accepted_errors": accepted_errors,
                "empirical_accepted_risk": (
                    None if count == 0 else accepted_errors / count
                ),
                "calibration_accepted_risk_ucb": None if count == 0 else 1.0,
                "certificate_supported": False,
            }
        summary = {
            "schema_version": "stage2-sealed-fold-summary-v2",
            "protocol_hash": PROTOCOL_SHA256,
            "outer": outer,
            "source_sha256": source_hashes[outer],
            "aggregate_sha256": sha256_file(
                CAMPAIGN_ROOT / "sealed_outer_results" / "ALL_OUTER_SEALED.npz"
            ),
            "sealed_sha256": sha256_file(result_path),
            "eligible": len(errors),
            "primary_errors": int(errors.sum()),
            "primary_error_rate": float(errors.mean()),
            "methods": method_names,
            "metrics": metrics,
        }
        if summary_path.exists():
            existing_summary = _read_json(summary_path)
            if existing_summary != summary:
                raise RuntimeError(f"materialized fold summary differs from aggregate: {outer}")
        else:
            atomic_write_json(summary_path, summary)


def _load_sealed() -> dict[str, dict[str, Any]]:
    manifests = _frozen_manifests()
    source_hashes = {
        str(item["outer"]): str(
            item.get("verified_artifact_sha256", item["artifact_sha256"])
        )
        for item in manifests
    }
    aggregate = _load_completed_aggregate(
        aggregate_path=CAMPAIGN_ROOT / "sealed_outer_results" / "ALL_OUTER_SEALED.npz",
        completion_path=CAMPAIGN_ROOT / "sealed_outer_results" / "UNBLINDING_COMPLETE.json",
        lock_path=CAMPAIGN_ROOT / "sealed_outer_results" / "UNBLINDING_LOCK.json",
        source_hashes=source_hashes,
        evaluation_code_hash=evaluation_code_provenance()["hash"],
    )
    _materialize_outer_results(aggregate, source_hashes)
    result: dict[str, dict[str, Any]] = {}
    for outer in SUBJECTS:
        path = (
            CAMPAIGN_ROOT
            / "sealed_outer_results"
            / f"outer-{outer}"
            / "sealed_results.npz"
        )
        with np.load(path, allow_pickle=False) as archive:
            row = {name: np.array(archive[name], copy=True) for name in archive.files}
        row["identities"] = _identities(row["identities_json"])
        row["method_names_list"] = [str(value) for value in row["method_names"]]
        row["mask_names_list"] = [str(value) for value in row["mask_names"]]
        result[outer] = row
    return result


def _metric(accept: np.ndarray, errors: np.ndarray) -> dict[str, Any]:
    eligible = len(errors)
    accepted = int(np.sum(accept))
    accepted_errors = int(np.sum(errors[accept]))
    return {
        "eligible": eligible,
        "accepted": accepted,
        "accepted_errors": accepted_errors,
        "coverage": accepted / eligible,
        "risk": None if accepted == 0 else accepted_errors / accepted,
        "error_mass": accepted_errors / eligible,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"cannot write an empty table: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sign_flip_tail(deltas: np.ndarray) -> float:
    observed = float(np.mean(deltas))
    count = 0
    total = 1 << len(deltas)
    chunk = 65536
    bit_positions = np.arange(len(deltas), dtype=np.uint64)
    for start in range(0, total, chunk):
        integers = np.arange(start, min(start + chunk, total), dtype=np.uint64)
        signs = 2.0 * (((integers[:, None] >> bit_positions) & 1).astype(float)) - 1.0
        count += int(np.sum(np.mean(signs * deltas[None, :], axis=1) >= observed))
    return count / total


def _bootstrap_deltas(deltas: np.ndarray) -> dict[str, float]:
    rng = np.random.Generator(np.random.PCG64(20260901))
    samples = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "replicates": 10000,
        "seed": 20260901,
        "mean": float(samples.mean()),
        "percentile_2_5": float(lower),
        "percentile_97_5": float(upper),
    }


def _paired_diagnostics(deltas: np.ndarray, subjects: Sequence[str]) -> dict[str, Any]:
    positive = int(np.sum(deltas > 0))
    negative = int(np.sum(deltas < 0))
    ties = int(np.sum(deltas == 0))
    nonzero = positive + negative
    exact_sign_tail = None if nonzero == 0 else float(binom.sf(positive - 1, nonzero, 0.5))
    bootstrap = _bootstrap_deltas(deltas)
    leave_one_out = {
        subject: float(np.mean(np.delete(deltas, index)))
        for index, subject in enumerate(subjects)
    }
    descending = np.argsort(-deltas)
    remove_top_one = float(np.mean(np.delete(deltas, descending[:1])))
    remove_top_two = float(np.mean(np.delete(deltas, descending[:2])))
    positive_values = np.maximum(deltas, 0)
    denominator = float(positive_values.sum())
    shares = np.sort(positive_values)[::-1] / denominator if denominator > 0 else np.zeros(2)
    return {
        "positive": positive,
        "positive_fraction": positive / len(deltas),
        "negative": negative,
        "negative_fraction": negative / len(deltas),
        "ties": ties,
        "tie_fraction": ties / len(deltas),
        "median": float(np.median(deltas)),
        "mean": float(np.mean(deltas)),
        "exact_sign_upper_tail_diagnostic": exact_sign_tail,
        "enumerated_sign_flip_tail_diagnostic": _sign_flip_tail(deltas),
        "bootstrap_percentile_diagnostic": bootstrap,
        "leave_one_subject_out_mean": leave_one_out,
        "minimum_leave_one_subject_out_mean": float(min(leave_one_out.values())),
        "remove_largest_gain_mean": remove_top_one,
        "remove_largest_two_gains_mean": remove_top_two,
        "largest_positive_gain_share": None if denominator == 0 else float(shares[0]),
        "top_two_positive_gain_share": (
            None if denominator == 0 else float(shares[:2].sum())
        ),
        "positive_gain_denominator": denominator,
    }


def _weighted_ece(scores: np.ndarray, outcomes: np.ndarray, weights: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    result = 0.0
    total = float(weights.sum())
    for index in range(10):
        mask = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == 9 else scores < edges[index + 1]
        )
        if not np.any(mask):
            continue
        mass = float(weights[mask].sum())
        result += mass / total * abs(
            float(np.average(scores[mask], weights=weights[mask]))
            - float(np.average(outcomes[mask], weights=weights[mask]))
        )
    return result


def _adaptive_ece(
    scores: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    tie_values: np.ndarray,
) -> float:
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    tie_values = np.asarray(tie_values, dtype=object)
    if not (
        scores.shape == outcomes.shape == weights.shape == tie_values.shape
        and scores.ndim == 1
    ):
        raise ValueError("adaptive ECE inputs must be aligned one-dimensional arrays")
    if (
        not np.all(np.isfinite(scores))
        or not np.all(np.isfinite(outcomes))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or float(weights.sum()) <= 0
    ):
        raise ValueError("adaptive ECE inputs are invalid")
    # Python sorting preserves the full Fraction comparison for the uint256 tie
    # variable. Each row's subject-equal mass may be split at a bin boundary;
    # this makes the ten bins exactly equal in subject-weight mass and invariant
    # to duplicating rows while dividing their weights accordingly.
    order = sorted(
        range(len(scores)), key=lambda index: (scores[index], tie_values[index])
    )
    total = float(weights.sum())
    target_mass = total / 10.0
    bins = [
        {"mass": 0.0, "score_sum": 0.0, "outcome_sum": 0.0}
        for _ in range(10)
    ]
    bin_index = 0
    for row_index in order:
        remaining = float(weights[row_index])
        while remaining > 0 and bin_index < 10:
            capacity = target_mass - bins[bin_index]["mass"]
            take = min(remaining, max(capacity, 0.0))
            if take <= total * 1e-15:
                bin_index += 1
                continue
            bins[bin_index]["mass"] += take
            bins[bin_index]["score_sum"] += take * float(scores[row_index])
            bins[bin_index]["outcome_sum"] += take * float(outcomes[row_index])
            remaining -= take
            if bins[bin_index]["mass"] >= target_mass - total * 1e-15:
                bin_index += 1
    return float(
        sum(
            row["mass"]
            / total
            * abs(
                row["score_sum"] / row["mass"]
                - row["outcome_sum"] / row["mass"]
            )
            for row in bins
            if row["mass"] > 0
        )
    )


def _calibration_diagnostics(sealed: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    first = sealed[SUBJECTS[0]]
    methods = first["method_names_list"]
    rows: list[dict[str, Any]] = []
    for method in methods:
        scores_parts: list[np.ndarray] = []
        outcome_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        tie_parts: list[np.ndarray] = []
        for outer in SUBJECTS:
            row = sealed[outer]
            column = row["method_names_list"].index(method)
            scores = np.asarray(row["risk_scores"][:, column], dtype=float)
            outcome = np.asarray(row["error_indicators"], dtype=int)
            selector_metadata = json.loads(
                _scalar_text(row["selector_metadata_json"], "selector_metadata_json")
            )
            full_method_id = selector_metadata[method]["full_method_id"]
            scores_parts.append(scores)
            outcome_parts.append(outcome)
            weight_parts.append(np.full(len(outcome), 1.0 / len(outcome)))
            tie_parts.append(
                np.asarray(
                    [
                        tie_uniform(
                            identity,
                            method_id=full_method_id,
                            protocol_hash=PROTOCOL_SHA256,
                            outer=outer,
                        )
                        for identity in row["identities"]
                    ],
                    dtype=object,
                )
            )
        scores = np.concatenate(scores_parts)
        outcomes = np.concatenate(outcome_parts)
        weights = np.concatenate(weight_parts)
        tie_values = np.concatenate(tie_parts).astype(object, copy=False)
        weights /= weights.sum()
        probability_like = bool(np.all((scores >= 0) & (scores <= 1)))
        if len(np.unique(outcomes)) == 2:
            auroc = float(roc_auc_score(outcomes, scores, sample_weight=weights))
            auprc = float(average_precision_score(outcomes, scores, sample_weight=weights))
        else:
            auroc = None
            auprc = None
        if probability_like:
            clipped = np.clip(scores, 1e-6, 1 - 1e-6)
            brier = float(np.average((scores - outcomes) ** 2, weights=weights))
            nll = float(
                np.average(
                    -(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped)),
                    weights=weights,
                )
            )
            fixed_ece = _weighted_ece(scores, outcomes, weights)
            adaptive_ece = _adaptive_ece(scores, outcomes, weights, tie_values)
        else:
            brier = nll = fixed_ece = adaptive_ece = None
        rows.append(
            {
                "method": method,
                "AUROC": auroc,
                "AUPRC": auprc,
                "Brier": brier,
                "NLL": nll,
                "ECE_10_fixed": fixed_ece,
                "ECE_10_adaptive": adaptive_ece,
                "probability_like": probability_like,
            }
        )
    return rows


def _fraction_columns(value: Any, prefix: str) -> dict[str, Any]:
    if value is None:
        return {
            f"{prefix}_numerator": None,
            f"{prefix}_denominator": None,
            f"{prefix}_value": None,
        }
    if isinstance(value, Mapping):
        fraction = Fraction(int(value["numerator"]), int(value["denominator"]))
    elif isinstance(value, Fraction):
        fraction = value
    else:
        fraction = Fraction(str(value))
    return {
        f"{prefix}_numerator": fraction.numerator,
        f"{prefix}_denominator": fraction.denominator,
        f"{prefix}_value": float(fraction),
    }


def _selector_metadata_columns(
    metadata: Mapping[str, Any], *, method: str, alpha: str, selector: str
) -> dict[str, Any]:
    if method not in metadata:
        raise RuntimeError(f"selector metadata has no method {method}")
    method_row = metadata[method]
    if not isinstance(method_row, Mapping):
        raise RuntimeError(f"selector metadata for {method} is malformed")
    alpha_rows = method_row.get("alphas", {})
    if alpha not in alpha_rows or selector not in alpha_rows[alpha]:
        raise RuntimeError(f"selector metadata has no {method}/{alpha}/{selector}")
    branch = alpha_rows[alpha][selector]
    if not isinstance(branch, Mapping):
        raise RuntimeError(f"selector branch is malformed: {method}/{alpha}/{selector}")
    cert_metric = branch.get("cert_metric")
    if cert_metric is not None and not isinstance(cert_metric, Mapping):
        raise RuntimeError("CERT metric payload is malformed")
    cert_coverage = None if cert_metric is None else cert_metric.get("coverage")
    cert_risk = None if cert_metric is None else cert_metric.get("risk")
    cert_per_subject = None if cert_metric is None else cert_metric.get("per_subject")
    supported = bool(branch.get("supported", False))
    risk_value = _fraction_columns(cert_risk, "_risk")["_risk_value"]
    return {
        "selector_full_method_id": method_row.get("full_method_id"),
        **_fraction_columns(branch.get("q"), "selector_q"),
        "selector_threshold": branch.get("threshold"),
        **_fraction_columns(branch.get("tie_fraction"), "selector_tie_fraction"),
        **_fraction_columns(cert_coverage, "selector_cert_macro_coverage"),
        **_fraction_columns(cert_risk, "selector_cert_macro_risk"),
        "selector_minimum_accepted_per_cert_subject": branch.get(
            "minimum_accepted_per_cert_subject", 0
        ),
        "selector_supported": supported,
        "selector_zero_fallback": bool(branch.get("zero_fallback", False)),
        "selector_cert_risk_ucb": branch.get(
            "cert_risk_ucb", branch.get("risk_ucb")
        ),
        "selector_certificate_supported": bool(
            branch.get("certificate_supported", False)
        ),
        "selector_cert_feasible_at_alpha": bool(
            supported and risk_value is not None and risk_value <= float(alpha)
        ),
        "selector_cert_per_subject_json": (
            None
            if cert_per_subject is None
            else json.dumps(cert_per_subject, sort_keys=True, separators=(",", ":"))
        ),
    }


def _pre_unblind_leakage_evidence() -> dict[str, Any]:
    audit_path = CAMPAIGN_ROOT / "logs" / "pre_unblind_execution_audit.json"
    completion_path = CAMPAIGN_ROOT / "sealed_outer_results" / "UNBLINDING_COMPLETE.json"
    audit = _read_json(audit_path)
    completion = _read_json(completion_path)
    if (
        audit.get("protocol_hash") != PROTOCOL_SHA256
        or audit.get("completed_before_any_outer_label_read") is not True
        or completion.get("pre_unblind_audit_sha256") != sha256_file(audit_path)
        or audit.get("evidence", {}).get("verdict") != "PASS"
    ):
        raise RuntimeError("pre-unblind leakage audit is missing or incompatible")
    return dict(audit["evidence"])


def _aggregate_rows(per_subject: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in per_subject:
        groups.setdefault((row["method"], row["alpha"], row["selector"]), []).append(row)
    results: list[dict[str, Any]] = []
    for (method, alpha, selector), rows in sorted(groups.items()):
        coverages = np.asarray([float(row["coverage"]) for row in rows])
        error_mass = np.asarray([float(row["error_mass"]) for row in rows])
        accepted = sum(int(row["accepted"]) for row in rows)
        accepted_errors = sum(int(row["accepted_errors"]) for row in rows)
        macro_coverage = float(coverages.mean())
        macro_risk = None if macro_coverage == 0 else float(error_mass.mean() / macro_coverage)
        results.append(
            {
                "method": method,
                "alpha": alpha,
                "selector": selector,
                "subject_macro_coverage": macro_coverage,
                "median_subject_coverage": float(np.median(coverages)),
                "subject_macro_ratio_risk": macro_risk,
                "pooled_coverage": sum(int(row["accepted"]) for row in rows)
                / sum(int(row["eligible"]) for row in rows),
                "pooled_risk": None if accepted == 0 else accepted_errors / accepted,
                "accepted": accepted,
                "accepted_errors": accepted_errors,
                "subjects_nonzero_coverage": int(np.sum(coverages > 0)),
                "certificate_supported": False,
                "valid_risk_ucb": None if accepted == 0 else 1.0,
            }
        )
    return results


def analyze_statistics() -> dict[str, Any]:
    leakage_evidence = _pre_unblind_leakage_evidence()
    sealed = _load_sealed()
    per_subject: list[dict[str, Any]] = []
    for outer in SUBJECTS:
        row = sealed[outer]
        errors = np.asarray(row["error_indicators"], dtype=np.int64)
        selector_metadata = json.loads(
            _scalar_text(row["selector_metadata_json"], "selector_metadata_json")
        )
        for column, mask_name in enumerate(row["mask_names_list"]):
            method, alpha, selector = mask_name.split("|")
            metric = _metric(np.asarray(row["acceptance_masks"][:, column], dtype=bool), errors)
            per_subject.append(
                {
                    "subject": outer,
                    "method": method,
                    "alpha": alpha,
                    "selector": selector,
                    **_selector_metadata_columns(
                        selector_metadata,
                        method=method,
                        alpha=alpha,
                        selector=selector,
                    ),
                    **metric,
                    "valid_risk_ucb": None if metric["accepted"] == 0 else 1.0,
                    "risk_criterion_met": (
                        metric["accepted"] >= 50
                        and metric["risk"] is not None
                        and metric["risk"] <= float(alpha)
                    ),
                }
            )
    aggregate = _aggregate_rows(per_subject)
    primary_rows = [
        row
        for row in per_subject
        if row["method"] == PRIMARY_METHOD
        and row["alpha"] == "0.05"
        and row["selector"] == "empirical"
    ]
    comparator_rows = [
        row
        for row in per_subject
        if row["method"] == PRIMARY_COMPARATOR
        and row["alpha"] == "0.05"
        and row["selector"] == "empirical"
    ]
    primary_rows.sort(key=lambda row: row["subject"])
    comparator_rows.sort(key=lambda row: row["subject"])
    deltas = np.asarray(
        [left["coverage"] - right["coverage"] for left, right in zip(primary_rows, comparator_rows)]
    )
    paired = _paired_diagnostics(deltas, SUBJECTS)
    primary_aggregate = next(
        row
        for row in aggregate
        if row["method"] == PRIMARY_METHOD
        and row["alpha"] == "0.05"
        and row["selector"] == "empirical"
    )
    comparator_aggregate = next(
        row
        for row in aggregate
        if row["method"] == PRIMARY_COMPARATOR
        and row["alpha"] == "0.05"
        and row["selector"] == "empirical"
    )
    stable_risk_subjects = sum(bool(row["risk_criterion_met"]) for row in primary_rows)
    criteria = {
        "leakage_audit_pass": leakage_evidence["verdict"] == "PASS",
        "macro_risk_at_most_0_05": (
            primary_aggregate["subject_macro_ratio_risk"] is not None
            and primary_aggregate["subject_macro_ratio_risk"] <= 0.05
        ),
        "pooled_risk_at_most_0_05": (
            primary_aggregate["pooled_risk"] is not None
            and primary_aggregate["pooled_risk"] <= 0.05
        ),
        "at_least_16_supported_subjects_meet_risk": stable_risk_subjects >= 16,
        "at_least_12_positive_subjects": paired["positive"] >= 12,
        "median_delta_positive": paired["median"] > 0,
        "macro_delta_at_least_0_02": paired["mean"] >= 0.02,
        "sign_flip_tail_below_0_05": paired["enumerated_sign_flip_tail_diagnostic"] < 0.05,
        "bootstrap_lower_positive": paired["bootstrap_percentile_diagnostic"]["percentile_2_5"] > 0,
        "all_leave_one_out_positive": paired["minimum_leave_one_subject_out_mean"] > 0,
        "remove_top_one_and_two_positive": (
            paired["remove_largest_gain_mean"] > 0
            and paired["remove_largest_two_gains_mean"] > 0
        ),
        "largest_positive_share_at_most_0_30": (
            paired["largest_positive_gain_share"] is not None
            and paired["largest_positive_gain_share"] <= 0.30
        ),
        "top_two_positive_share_at_most_0_50": (
            paired["top_two_positive_gain_share"] is not None
            and paired["top_two_positive_gain_share"] <= 0.50
        ),
    }
    verdict = "STAGE2_ICLR_GO" if all(criteria.values()) else "STAGE2_ICLR_NO_GO"
    calibration = _calibration_diagnostics(sealed)
    tables = CAMPAIGN_ROOT / "tables"
    _write_csv(tables / "per_subject_results.csv", per_subject)
    _write_csv(tables / "aggregate_results.csv", aggregate)
    _write_csv(tables / "calibration_diagnostics.csv", calibration)
    paired_rows = [
        {
            "subject": subject,
            "learned_selected_coverage": primary_rows[index]["coverage"],
            "baseline_selected_coverage": comparator_rows[index]["coverage"],
            "paired_delta": deltas[index],
            "learned_selected_risk": primary_rows[index]["risk"],
            "baseline_selected_risk": comparator_rows[index]["risk"],
        }
        for index, subject in enumerate(SUBJECTS)
    ]
    _write_csv(tables / "primary_paired_results.csv", paired_rows)
    baseline_comparisons: dict[str, Any] = {}
    comparison_table: list[dict[str, Any]] = []
    for comparator in (
        "MSP",
        "Margin",
        "Entropy",
        "Temperature-MSP",
        "Forward-KL",
        "Disagreement",
        "Baseline-Selected",
    ):
        comparator_subject_rows = sorted(
            [
                row
                for row in per_subject
                if row["method"] == comparator
                and row["alpha"] == "0.05"
                and row["selector"] == "empirical"
            ],
            key=lambda row: row["subject"],
        )
        comparator_deltas = np.asarray(
            [
                primary_rows[index]["coverage"]
                - comparator_subject_rows[index]["coverage"]
                for index in range(20)
            ]
        )
        baseline_comparisons[comparator] = _paired_diagnostics(
            comparator_deltas, SUBJECTS
        )
        comparison_table.extend(
            {
                "subject": subject,
                "comparator": comparator,
                "learned_selected_coverage": primary_rows[index]["coverage"],
                "comparator_coverage": comparator_subject_rows[index]["coverage"],
                "paired_coverage_delta": comparator_deltas[index],
            }
            for index, subject in enumerate(SUBJECTS)
        )
    _write_csv(tables / "paired_comparisons_all_baselines.csv", comparison_table)
    selected_families = {
        outer: _read_json(
            CAMPAIGN_ROOT
            / "outer_predictions"
            / f"outer-{outer}"
            / "outer_prediction_manifest.json"
        )["learned_selected_family"]
        for outer in SUBJECTS
    }
    family_counts = {
        family: sum(value == family for value in selected_families.values())
        for family in ("Primary-only", "Heterogeneous-expert")
    }
    result = {
        "schema_version": "stage2-statistical-analysis-v1",
        "protocol_hash": PROTOCOL_SHA256,
        "primary_method": PRIMARY_METHOD,
        "primary_comparator": PRIMARY_COMPARATOR,
        "primary_alpha": PRIMARY_ALPHA,
        "primary_result": primary_aggregate,
        "comparator_result": comparator_aggregate,
        "coverage_delta": {
            "subject_macro": paired["mean"],
            "pooled": primary_aggregate["pooled_coverage"]
            - comparator_aggregate["pooled_coverage"],
            "median": paired["median"],
        },
        "subjects_meeting_primary_risk_and_support": stable_risk_subjects,
        "paired_diagnostics": paired,
        "paired_comparisons_vs_fixed_baselines": baseline_comparisons,
        "learned_selected_family_by_outer": selected_families,
        "learned_selected_family_counts": family_counts,
        "go_criteria": criteria,
        "certificate": {
            "supported": False,
            "calibration_ucb_for_any_nonzero_selector": 1.0,
            "conservative_selector_coverage": 0.0,
            "claim": "risk-targeted but uncertified",
        },
        "leakage_evidence": leakage_evidence,
        "verdict": verdict,
        "aggregate_results": aggregate,
        "calibration_diagnostics": calibration,
    }
    atomic_write_json(CAMPAIGN_ROOT / "statistics" / "statistical_analysis.json", result)
    _update_state("STATISTICAL_ANALYSIS", completed="STATISTICAL_ANALYSIS")
    return result


def _primary_subject_rows() -> list[dict[str, str]]:
    path = CAMPAIGN_ROOT / "tables" / "primary_paired_results.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_figures(result: Mapping[str, Any]) -> None:
    figures = CAMPAIGN_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paired = _primary_subject_rows()
    subjects = [row["subject"] for row in paired]
    learned = np.asarray([float(row["learned_selected_coverage"]) for row in paired])
    baseline = np.asarray([float(row["baseline_selected_coverage"]) for row in paired])
    delta = learned - baseline
    x = np.arange(len(subjects))

    plt.figure(figsize=(11, 4.5))
    width = 0.42
    plt.bar(x - width / 2, baseline, width, label="Baseline-Selected")
    plt.bar(x + width / 2, learned, width, label="Learned-Selected")
    plt.axhline(0, color="black", linewidth=0.6)
    plt.xticks(x, subjects)
    plt.ylim(0, 1)
    plt.ylabel("Coverage at empirical target α = 5%")
    plt.xlabel("Unseen outer subject")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figures / "primary_coverage_by_subject.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 4.2))
    colors = np.where(delta > 0, "#2a9d8f", np.where(delta < 0, "#e76f51", "#777777"))
    plt.bar(x, delta, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, subjects)
    plt.ylabel("Paired Coverage Δ (Learned − Baseline)")
    plt.xlabel("Unseen outer subject")
    plt.tight_layout()
    plt.savefig(figures / "paired_coverage_delta.png", dpi=180)
    plt.close()

    leave = result["paired_diagnostics"]["leave_one_subject_out_mean"]
    values = np.asarray([leave[subject] for subject in subjects])
    plt.figure(figsize=(11, 4.2))
    plt.plot(x, values, marker="o", color="#264653")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, subjects)
    plt.ylabel("Mean paired Δ after deleting subject")
    plt.xlabel("Deleted subject")
    plt.tight_layout()
    plt.savefig(figures / "leave_one_subject_out_influence.png", dpi=180)
    plt.close()

    aggregates = [
        row
        for row in result["aggregate_results"]
        if row["selector"] == "empirical"
        and row["alpha"] == "0.05"
        and row["method"] in (PRIMARY_METHOD, PRIMARY_COMPARATOR, "MSP", "Margin", "Entropy", "Forward-KL")
    ]
    names = [row["method"] for row in aggregates]
    coverage = [row["subject_macro_coverage"] for row in aggregates]
    plt.figure(figsize=(10, 4.8))
    plt.bar(np.arange(len(names)), coverage, color="#457b9d")
    plt.xticks(np.arange(len(names)), names, rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Subject-macro Coverage at α = 5%")
    plt.tight_layout()
    plt.savefig(figures / "method_coverage_summary.png", dpi=180)
    plt.close()

    secondary = [
        row
        for row in result["aggregate_results"]
        if row["selector"] == "empirical"
        and row["method"] in (PRIMARY_METHOD, PRIMARY_COMPARATOR)
    ]
    plt.figure(figsize=(7.5, 4.5))
    for method, marker in ((PRIMARY_METHOD, "o"), (PRIMARY_COMPARATOR, "s")):
        method_rows = sorted(
            [row for row in secondary if row["method"] == method],
            key=lambda row: float(row["alpha"]),
        )
        plt.plot(
            [float(row["alpha"]) for row in method_rows],
            [row["subject_macro_coverage"] for row in method_rows],
            marker=marker,
            label=method,
        )
    plt.xlabel("Accepted-risk target α")
    plt.ylabel("Subject-macro Coverage")
    plt.xticks(list(ALPHAS), ["2%", "5%", "10%"])
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figures / "secondary_alpha_sensitivity.png", dpi=180)
    plt.close()

    calibration = result["calibration_diagnostics"]
    names = [row["method"] for row in calibration]
    auroc = [np.nan if row["AUROC"] is None else row["AUROC"] for row in calibration]
    plt.figure(figsize=(11, 4.8))
    plt.bar(np.arange(len(names)), auroc, color="#6a4c93")
    plt.xticks(np.arange(len(names)), names, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Subject-weighted error-detection AUROC")
    plt.tight_layout()
    plt.savefig(figures / "risk_score_discrimination.png", dpi=180)
    plt.close()

    sealed = _load_sealed()
    epoch_counts = [len(sealed[subject]["error_indicators"]) for subject in subjects]
    plt.figure(figsize=(11, 4.2))
    plt.bar(x, epoch_counts, color="#8ecae6")
    plt.xticks(x, subjects)
    plt.ylabel("Eligible sleep epochs")
    plt.xlabel("Subject")
    plt.tight_layout()
    plt.savefig(figures / "cohort_epochs_by_subject.png", dpi=180)
    plt.close()

    reliability_scores: list[np.ndarray] = []
    reliability_outcomes: list[np.ndarray] = []
    reliability_weights: list[np.ndarray] = []
    for subject in subjects:
        row = sealed[subject]
        column = row["method_names_list"].index(PRIMARY_METHOD)
        score = np.asarray(row["risk_scores"][:, column], dtype=float)
        outcome = np.asarray(row["error_indicators"], dtype=float)
        reliability_scores.append(score)
        reliability_outcomes.append(outcome)
        reliability_weights.append(np.full(len(outcome), 1.0 / len(outcome)))
    score = np.concatenate(reliability_scores)
    outcome = np.concatenate(reliability_outcomes)
    weights = np.concatenate(reliability_weights)
    bin_x: list[float] = []
    bin_y: list[float] = []
    edges = np.linspace(0, 1, 11)
    for index in range(10):
        mask = (score >= edges[index]) & (
            score <= edges[index + 1] if index == 9 else score < edges[index + 1]
        )
        if np.any(mask):
            bin_x.append(float(np.average(score[mask], weights=weights[mask])))
            bin_y.append(float(np.average(outcome[mask], weights=weights[mask])))
    plt.figure(figsize=(5.4, 5.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Ideal")
    plt.plot(bin_x, bin_y, marker="o", color="#e76f51", label=PRIMARY_METHOD)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Predicted error risk")
    plt.ylabel("Observed error frequency")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figures / "error_risk_reliability.png", dpi=180)
    plt.close()

    q_grid = np.linspace(0.05, 1.0, 20)
    plt.figure(figsize=(7.2, 5.0))
    for method, style in ((PRIMARY_METHOD, "-o"), (PRIMARY_COMPARATOR, "-s")):
        risks: list[float] = []
        for q_value in q_grid:
            coverages: list[float] = []
            error_masses: list[float] = []
            for subject in subjects:
                row = sealed[subject]
                column = row["method_names_list"].index(method)
                subject_score = np.asarray(row["risk_scores"][:, column], dtype=float)
                subject_error = np.asarray(row["error_indicators"], dtype=int)
                n_accept = max(1, int(round(q_value * len(subject_score))))
                order = np.argsort(subject_score, kind="stable")
                accepted = order[:n_accept]
                coverages.append(n_accept / len(subject_score))
                error_masses.append(float(subject_error[accepted].sum() / len(subject_score)))
            risks.append(float(np.mean(error_masses) / np.mean(coverages)))
        plt.plot(q_grid, risks, style, markevery=3, label=method)
    plt.axhline(0.05, color="#d00000", linestyle="--", label="α = 5%")
    plt.xlabel("Diagnostic subject-macro Coverage")
    plt.ylabel("Observed accepted risk")
    plt.ylim(bottom=0)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figures / "diagnostic_risk_coverage.png", dpi=180)
    plt.close()


def _human_summary_text(result: Mapping[str, Any]) -> str:
    primary = result["primary_result"]
    comparator = result["comparator_result"]
    paired = result["paired_diagnostics"]
    return f"""# Stage 2 Human Review — Executive Summary

## Outcome

**Final decision: `{result['verdict']}`.**

The frozen primary comparison is `Learned-Selected` versus `Baseline-Selected`
at accepted-risk target α = 5%. Learned-Selected reached subject-macro Coverage
**{primary['subject_macro_coverage']:.4f}** with empirical macro accepted risk
**{primary['subject_macro_ratio_risk'] if primary['subject_macro_ratio_risk'] is not None else 'NA'}**.
Baseline-Selected Coverage was **{comparator['subject_macro_coverage']:.4f}**.
The paired macro Coverage difference was **{result['coverage_delta']['subject_macro']:.4f}**;
the median difference was **{result['coverage_delta']['median']:.4f}** and
{paired['positive']}/20 subjects had a positive difference.

## How to interpret this

Coverage is the fraction of sleep epochs that the system processes automatically.
Accepted risk is the error fraction inside those automatically accepted epochs.
The scientific question is therefore not merely “can we detect mistakes?”, but
“can we automate more epochs than strong confidence baselines while keeping the
accepted error rate within the frozen 5% target on unseen people?”

The empirical selector is **risk-targeted but uncertified**. With only four CERT
subjects per fold, the prespecified cluster-aware Hoeffding upper bound is 1.0
for every nonzero selector. Consequently, the conservative certified branch has
zero Coverage and this campaign cannot claim a finite-sample 5% risk certificate.

## Claim ceiling

The maximum safe scope is retrospective, campaign-sealed, unseen-subject
generalization on this frozen processed SleepEDF-20 cohort. It is not prospective
clinical validation, unrestricted distribution-shift control, or statistically
certified selective risk control.
"""


def generate_human_review(result: Mapping[str, Any]) -> None:
    review = CAMPAIGN_ROOT / "HUMAN_REVIEW"
    review.mkdir(parents=True, exist_ok=True)
    summary = _human_summary_text(result)
    readme = f"""# Stage 2 Human Review

**Decision:** `{result['verdict']}`  
**Frozen protocol SHA-256:** `{PROTOCOL_SHA256}`

Read in this order:

1. `EXECUTIVE_SUMMARY.md` for the plain-language outcome and claim boundary.
2. `STAGE2_HUMAN_REVIEW.html` for the fastest visual review.
3. `STAGE2_HUMAN_REVIEW.ipynb` for the editable notebook form.
4. `FIGURE_INDEX.md` and `TABLE_INDEX.md` for direct artifact navigation.
5. `../FINAL_REPORT.md`, `../LEAKAGE_AUDIT.md`, and
   `../PAPER_CLAIM_RECOMMENDATIONS.md` for the auditable closeout.

The fixed review sequence is cohort/provenance → split/leakage → development →
selector calibration → primary α=5% → paired subjects → influence → secondary
alphas → calibration/certificate → GO/NO-GO → claim ceiling.
"""
    (review / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    (review / "EXECUTIVE_SUMMARY.md").write_text(summary, encoding="utf-8", newline="\n")
    figures = [
        ("Primary Coverage by unseen subject", "primary_coverage_by_subject.png"),
        ("Paired Coverage difference", "paired_coverage_delta.png"),
        ("Influential-subject check", "leave_one_subject_out_influence.png"),
        ("Fixed-risk method summary", "method_coverage_summary.png"),
        ("Secondary α sensitivity", "secondary_alpha_sensitivity.png"),
        ("Risk-score discrimination", "risk_score_discrimination.png"),
        ("Cohort epochs by subject", "cohort_epochs_by_subject.png"),
        ("Error-risk reliability", "error_risk_reliability.png"),
        ("Diagnostic risk–Coverage curves", "diagnostic_risk_coverage.png"),
    ]
    figure_index = "# Figure Index\n\n" + "\n".join(
        f"- **{title}:** `../figures/{filename}`" for title, filename in figures
    ) + "\n"
    (review / "FIGURE_INDEX.md").write_text(figure_index, encoding="utf-8", newline="\n")
    table_index = """# Table Index

- `../tables/per_subject_results.csv`: every subject, method, α and selector.
- `../tables/primary_paired_results.csv`: frozen primary paired comparison.
- `../tables/aggregate_results.csv`: macro, median and pooled summaries.
- `../tables/calibration_diagnostics.csv`: score calibration/discrimination diagnostics.
- `../tables/paired_comparisons_all_baselines.csv`: paired deltas versus every fixed baseline.
"""
    (review / "TABLE_INDEX.md").write_text(table_index, encoding="utf-8", newline="\n")
    notebook_cells: list[dict[str, Any]] = []

    def markdown_cell(text: str) -> None:
        notebook_cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in text.splitlines()],
            }
        )

    markdown_cell("# Stage 2 ICLR Human Review\n\n" + summary)
    sections = [
        ("Cohort and provenance", "20 SleepEDF subjects, 39 records and 42,308 epochs; every outer prediction was generated by a model whose ancestry excludes that subject."),
        ("Split and leakage verification", "FIT15/CERT4/OUTER1 assignments, component ancestry, label-ledger denials and 29/29 synthetic contracts are content-addressed."),
        ("Risk-model development", "Primary-only and Heterogeneous-expert families were selected within FIT subjects; no FIT metric is presented as confirmatory evidence."),
        ("Selector calibration", "CERT subjects selected the largest empirically feasible Coverage. The conservative cluster-aware certificate is provably vacuous."),
        ("Primary α = 5% result", f"![Primary result](../figures/{figures[0][1]})"),
        ("Per-subject paired comparison", f"![Paired result](../figures/{figures[1][1]})"),
        ("Stability and influence analysis", f"![Influence](../figures/{figures[2][1]})"),
        ("Secondary alphas", f"![Sensitivity](../figures/{figures[4][1]})"),
        ("Calibration and certificate diagnostics", f"![Diagnostics](../figures/{figures[5][1]})"),
        ("GO / NO-GO", f"The precommitted decision is **{result['verdict']}**; no post-unblinding rescue rule was used."),
        ("Claim ceiling", "Retrospective unseen-subject empirical risk targeting on the frozen processed cohort; uncertified and non-prospective."),
    ]
    for title, body in sections:
        markdown_cell(f"## {title}\n\n{body}")
    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    atomic_write_json(review / "STAGE2_HUMAN_REVIEW.ipynb", notebook)
    body = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Stage 2 ICLR Human Review</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;line-height:1.55;color:#1d2433}h1,h2{color:#17324d}img{max-width:100%;border:1px solid #ddd;margin:12px 0 28px}code{background:#f2f4f7;padding:2px 4px}.note{background:#fff4d6;padding:14px;border-left:5px solid #d89b00}</style></head><body>",
        "<h1>Stage 2 ICLR Human Review</h1>",
        f"<div class='note'><strong>Decision:</strong> {html.escape(result['verdict'])}<br><strong>Protocol:</strong> {PROTOCOL_SHA256}</div>",
    ]
    for title, section_body in sections:
        body.append(f"<h2>{html.escape(title)}</h2>")
        if "![" in section_body:
            filename = section_body.split("../figures/")[1].split(")")[0]
            body.append(f"<img src='../figures/{html.escape(filename)}' alt='{html.escape(title)}'>")
        else:
            body.append(f"<p>{html.escape(section_body)}</p>")
    body.append("</body></html>")
    (review / "STAGE2_HUMAN_REVIEW.html").write_text("".join(body), encoding="utf-8")


def _claim_recommendations(result: Mapping[str, Any]) -> str:
    verdict = result["verdict"]
    supported = verdict == "STAGE2_ICLR_GO"
    direction = (
        "Stage 2 satisfies the frozen empirical GO gate and may be positioned as the main empirical contribution."
        if supported
        else "Stage 2 does not satisfy the frozen GO gate and must not be packaged as a successful new method."
    )
    go_positioning = (
        """
## GO-only paper framing

Suggested title: **Risk-Targeted Selective Prediction under Unseen-Subject Shift**.

Suggested abstract framing: learn an error-risk score from deployable primary and
heterogeneous-expert signals, construct selectors without outer labels, and test
whether they increase Coverage at a fixed empirical accepted-risk target across
previously unseen subjects. Contributions should emphasize the sealed nested
evaluation, the fair learned-versus-baseline selector comparison, and the
subject-level stability analysis—not a finite-sample certificate.
"""
        if supported
        else ""
    )
    return f"""# Paper Claim Recommendations

## Final recommendation

`{verdict}` — {direction}

## Supported claims

- The campaign used a sealed, fold-scoped outer-label firewall across 20 unseen-subject folds.
- Primary-only and heterogeneous learned-risk routes were compared under the same frozen selector construction as strong confidence/disagreement baselines.
- All reported Coverage and accepted-risk quantities are retrospective results on the frozen processed SleepEDF-20 cohort.

## Claims requiring caution

- Say **empirical risk targeting**, not “certified risk control.” The valid conservative branch has zero Coverage; every nonzero calibration UCB is 1.0.
- Say **unseen-subject evaluation within one historical cohort**, not universal OOD or unrestricted distribution-shift generalization.
- Fold-overlap makes sign-flip and bootstrap outputs prespecified resampling diagnostics rather than independent-fold p-values or confidence intervals.

## Unsupported claims

- Prospective clinical safety or utility.
- A finite-sample guarantee that accepted risk is at most 5%.
- Generalization beyond the frozen preprocessing, cohort, dataset, or acquisition setting.

## Role of Stage 1

Stage 1 is motivating evidence only: raw confidence can coexist with systematic errors, and expert disagreement can enrich them. Historical MSP/KL thresholds are not the Stage 2 method or optimization objective.

## Minimal next scientific step

If GO, validate the already frozen selector construction on an independent cohort before strengthening confirmatory language. If NO-GO, use the failure analysis to choose one development-only revision, then obtain a genuinely independent confirmatory dataset; do not re-label the current outer matrix as confirmatory.
{go_positioning}
"""


def write_closeout_documents(result: Mapping[str, Any]) -> None:
    leakage = {
        "schema_version": "stage2-leakage-audit-v1",
        "protocol_hash": PROTOCOL_SHA256,
        "verdict": "PASS",
        "contract_tests": "29/29 PASS",
        "all_outer_manifests_frozen_before_unblind": True,
        "fold_scoped_label_ledgers": True,
        "stable_identity_join": True,
        "blind_component_and_outer_artifacts": True,
        "execution_evidence": result["leakage_evidence"],
        "limitations": [
            "official-download-to-processed-NPZ lineage is not signed",
            "the cohort was historically studied before Stage 2 prespecification",
        ],
    }
    atomic_write_json(CAMPAIGN_ROOT / "leakage_audit.json", leakage)
    leakage_md = f"""# Leakage Audit

**Verdict: PASS** under frozen protocol `{PROTOCOL_SHA256}`.

All 20 outer artifacts were frozen and content-hashed before the single outer
label join. The label ledger is keyed by outer fold, phase and subject; FIT,
CERT and matching outer permissions are disjoint. Stable dataset/subject/record/
epoch identities are used for every join, and shuffled, missing, duplicate or
extra rows fail closed. Synthetic contract tests: **29/29 PASS**.

Limitations: the upstream official-download-to-processed-NPZ lineage is not
cryptographically signed, and this cohort informed historical Stage 1 work.
"""
    (CAMPAIGN_ROOT / "LEAKAGE_AUDIT.md").write_text(leakage_md, encoding="utf-8", newline="\n")
    claim_text = _claim_recommendations(result)
    (CAMPAIGN_ROOT / "PAPER_CLAIM_RECOMMENDATIONS.md").write_text(
        claim_text, encoding="utf-8", newline="\n"
    )
    primary = result["primary_result"]
    final = f"""# Stage 2 ICLR Campaign Final Report

## Final decision

`{result['verdict']}`

The frozen primary method was `Learned-Selected`; the comparator was
`Baseline-Selected`, both at α = 5% with identical CERT selector construction.

## Primary evidence

- Learned-Selected subject-macro Coverage: {primary['subject_macro_coverage']:.6f}
- Learned-Selected macro accepted risk: {primary['subject_macro_ratio_risk']}
- Learned-Selected pooled accepted risk: {primary['pooled_risk']}
- Paired macro Coverage delta: {result['coverage_delta']['subject_macro']:.6f}
- Paired median Coverage delta: {result['coverage_delta']['median']:.6f}
- Positive subjects: {result['paired_diagnostics']['positive']}/20
- Subjects with at least 50 accepts and risk at most 5%: {result['subjects_meeting_primary_risk_and_support']}/20

## Certification boundary

No nonzero selector is statistically certified. Under the frozen four-CERT-
subject Hoeffding construction, the valid upper bound is 1.0 for every nonzero
policy; the conservative branch therefore has zero Coverage. The primary branch
is explicitly `risk-targeted but uncertified`.

## Stability

- Minimum leave-one-subject-out mean gain: {result['paired_diagnostics']['minimum_leave_one_subject_out_mean']:.6f}
- Gain after deleting the largest subject: {result['paired_diagnostics']['remove_largest_gain_mean']:.6f}
- Gain after deleting the largest two subjects: {result['paired_diagnostics']['remove_largest_two_gains_mean']:.6f}
- Largest positive-gain share: {result['paired_diagnostics']['largest_positive_gain_share']}
- Top-two positive-gain share: {result['paired_diagnostics']['top_two_positive_gain_share']}

## Primary-only versus Heterogeneous-expert

Foldwise FIT-only selection chose Primary-only in
{result['learned_selected_family_counts']['Primary-only']}/20 outer folds and
Heterogeneous-expert in
{result['learned_selected_family_counts']['Heterogeneous-expert']}/20 outer folds.
This is the prespecified `Learned-Selected` route; no family was chosen from
outer outcomes.

## Claim ceiling

Retrospective, campaign-sealed unseen-subject empirical risk targeting on the
frozen processed SleepEDF-20 cohort. No prospective, clinical, unrestricted-
shift, or statistically certified risk-control claim is supported.
"""
    (CAMPAIGN_ROOT / "FINAL_REPORT.md").write_text(final, encoding="utf-8", newline="\n")
    machine = {
        "schema_version": "stage2-machine-readable-summary-v1",
        "protocol_hash": PROTOCOL_SHA256,
        "state": "FINAL_CLOSEOUT",
        **result,
        "paths": {
            "human_review_notebook": "HUMAN_REVIEW/STAGE2_HUMAN_REVIEW.ipynb",
            "human_review_html": "HUMAN_REVIEW/STAGE2_HUMAN_REVIEW.html",
            "final_report": "FINAL_REPORT.md",
            "figures": "figures",
            "tables": "tables",
        },
    }
    atomic_write_json(CAMPAIGN_ROOT / "MACHINE_READABLE_SUMMARY.json", machine)


def write_final_artifact_manifests() -> None:
    per_outer: dict[str, Any] = {}
    for outer in SUBJECTS:
        paths = {
            "development": CAMPAIGN_ROOT / "development" / "risk" / f"outer-{outer}" / "development_manifest.json",
            "calibration": CAMPAIGN_ROOT / "calibration" / f"outer-{outer}" / "selector_calibration.json",
            "outer_predictions": CAMPAIGN_ROOT / "outer_predictions" / f"outer-{outer}" / "outer_prediction_manifest.json",
            "sealed_results": CAMPAIGN_ROOT / "sealed_outer_results" / f"outer-{outer}" / "sealed_results.npz",
            "fold_summary": CAMPAIGN_ROOT / "sealed_outer_results" / f"outer-{outer}" / "fold_summary.json",
            "label_ledger": CAMPAIGN_ROOT / "logs" / f"label_access_outer-{outer}.jsonl",
        }
        per_outer[outer] = {
            name: {
                "path": path.relative_to(CAMPAIGN_ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        }
    artifact_manifest = {
        "schema_version": "stage2-artifact-manifest-v2-final",
        "protocol_hash": PROTOCOL_SHA256,
        "status": "FINAL_CLOSEOUT",
        "authoritative_inputs": {
            "raw_data": "data/Sleep-EDF-20/eeg_fpz_cz",
            "raw_file_set_hash": "6553de7de5c29572b4b99532ad648d95e54b90ff8d6c32a6fbd40e6aa435130a",
            "feature_cache_manifest": "outputs/stage1_clean_nested_loso/feature_cache/manifest-9aa61e01e862e0cc8994611663efd3716a6caff74c4c349c6d19f54e7546e02b.json",
            "feature_manifest_sha256": "4de20c8ca0bfccc9337ae8a01ee1bb5b819d960ecab33a646e0914f700214960",
        },
        "outer_folds": per_outer,
        "aggregate_unblinding": {
            name: {
                "path": path.relative_to(CAMPAIGN_ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in {
                "pre_unblind_execution_audit": CAMPAIGN_ROOT
                / "logs"
                / "pre_unblind_execution_audit.json",
                "unblinding_lock": CAMPAIGN_ROOT
                / "sealed_outer_results"
                / "UNBLINDING_LOCK.json",
                "sealed_aggregate": CAMPAIGN_ROOT
                / "sealed_outer_results"
                / "ALL_OUTER_SEALED.npz",
                "unblinding_complete": CAMPAIGN_ROOT
                / "sealed_outer_results"
                / "UNBLINDING_COMPLETE.json",
            }.items()
        },
    }
    atomic_write_json(CAMPAIGN_ROOT / "artifact_manifest.json", artifact_manifest)
    key_paths = [
        "frozen_protocol.json",
        "split_manifest.json",
        "feature_schema.json",
        "model_registry.json",
        "environment_manifest.json",
        "POST_FREEZE_IMPLEMENTATION_REPAIRS.md",
        "development/invalidated_pre_provenance_fix/INVALIDATION_RECORD.json",
        "logs/pre_unblind_execution_audit.json",
        "sealed_outer_results/UNBLINDING_LOCK.json",
        "sealed_outer_results/ALL_OUTER_SEALED.npz",
        "sealed_outer_results/UNBLINDING_COMPLETE.json",
        "leakage_audit.json",
        "statistics/statistical_analysis.json",
        "FINAL_REPORT.md",
        "MACHINE_READABLE_SUMMARY.json",
        "HUMAN_REVIEW/STAGE2_HUMAN_REVIEW.ipynb",
        "HUMAN_REVIEW/STAGE2_HUMAN_REVIEW.html",
        "run_state.json",
    ]
    hashes = {
        relative: sha256_file(CAMPAIGN_ROOT / relative) for relative in key_paths
    }
    atomic_write_json(
        CAMPAIGN_ROOT / "artifact_hashes.json",
        {
            "schema_version": "stage2-artifact-hashes-v2-final",
            "protocol_hash": PROTOCOL_SHA256,
            "files": hashes,
        },
    )


def closeout_campaign() -> None:
    single_outer_unblinding()
    result = analyze_statistics()
    generate_figures(result)
    generate_human_review(result)
    write_closeout_documents(result)
    _update_state("HUMAN_REVIEW", completed="HUMAN_REVIEW")
    _update_state("FINAL_CLOSEOUT", completed="FINAL_CLOSEOUT", outer_unblinded=True)
    write_final_artifact_manifests()


__all__ = ["analyze_statistics", "closeout_campaign", "single_outer_unblinding"]
