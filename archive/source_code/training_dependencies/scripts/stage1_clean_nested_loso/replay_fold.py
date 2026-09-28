"""Deterministically replay and source-bind one completed Stage-1 outer fold.

The fold directory is never modified. The optional JSON report must be written
outside it. Artifact/provenance validation runs before either model is loaded for
replay, and the replay reuses the canonical runner's inference implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

if __package__ in {None, ""}:  # Make ``python path/to/replay_fold.py`` usable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.stage1_clean_nested_loso.contracts import (  # noqa: E402
    ContractError,
    PredictionBundle,
    align_one_to_one,
    atomic_write_json,
    sha256_file,
)
from scripts.stage1_clean_nested_loso.feature_cache import (  # noqa: E402
    load_feature_rows,
)
from scripts.stage1_clean_nested_loso.run_outer_fold import (  # noqa: E402
    discover_raw_records,
    files_for_subjects,
    primary_outer_inference,
    set_seed,
)
from scripts.stage1_clean_nested_loso.verify_fold import (  # noqa: E402
    PROJECT_ROOT,
    verify_fold,
)


REPLAY_SCHEMA_VERSION = "stage1-clean-source-replay-v1"
DEFAULT_ATOL = 1e-7
DEFAULT_RTOL = 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _validated_tolerance(value: float, name: str) -> float:
    converted = float(value)
    _require(math.isfinite(converted) and converted >= 0.0, f"{name} must be finite and non-negative")
    return converted


def _read_manifest(fold_dir: Path) -> dict[str, Any]:
    path = fold_dir / "run_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read validated run manifest {path}: {exc}") from exc
    _require(isinstance(payload, dict), "run manifest must contain one JSON object")
    return payload


def _declared_input_path(manifest: Mapping[str, Any], key: str) -> Path:
    relative = manifest["input_paths"][key]
    _require(isinstance(relative, str), f"input_paths.{key} must be a string")
    return (PROJECT_ROOT / Path(*relative.split("/"))).resolve()


def _artifact_path(fold_dir: Path, manifest: Mapping[str, Any], name: str) -> Path:
    relative = manifest["artifacts"][name]["path"]
    return (fold_dir / Path(*relative.split("/"))).resolve()


def _resolve_device(requested: str, manifest: Mapping[str, Any]) -> Any:
    import torch

    selected = manifest["environment"]["selected_device"] if requested == "recorded" else requested
    _require(selected in {"cpu", "cuda"}, "replay device must resolve to cpu or cuda")
    if selected == "cuda":
        _require(torch.cuda.is_available(), "recorded CUDA replay requested but CUDA is unavailable")
    return torch.device(selected)


def _validate_generated_predictions(
    probabilities: Any,
    predictions: Any,
    confidences: Any,
    *,
    role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities)
    pred = np.asarray(predictions)
    conf = np.asarray(confidences)
    _require(probs.ndim == 2 and probs.shape[0] > 0 and probs.shape[1] >= 2, f"{role} replay probabilities have invalid shape")
    _require(pred.shape == (probs.shape[0],), f"{role} replay predictions have invalid shape")
    _require(conf.shape == (probs.shape[0],), f"{role} replay confidences have invalid shape")
    _require(np.isfinite(probs).all() and np.isfinite(conf).all(), f"{role} replay contains non-finite values")
    _require(np.all((probs >= 0.0) & (probs <= 1.0)), f"{role} replay probabilities fall outside [0, 1]")
    _require(
        np.allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-6),
        f"{role} replay probability rows do not sum to one",
    )
    expected_pred = np.argmax(probs, axis=1)
    expected_conf = np.max(probs, axis=1)
    _require(np.array_equal(pred, expected_pred), f"{role} replay prediction != argmax(probability)")
    _require(
        np.allclose(conf, expected_conf, rtol=0.0, atol=0.0),
        f"{role} replay confidence != max(probability)",
    )
    return probs, pred.astype(np.int64, copy=False), conf


def _align_replay(
    saved: PredictionBundle,
    replay_identities: Sequence[Any],
    replay_labels: Any,
    *,
    role: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(replay_labels)
    try:
        order = align_one_to_one(
            saved.identities,
            tuple(replay_identities),
            reference_labels=saved.labels,
            candidate_labels=labels,
        )
    except ContractError as exc:
        raise ContractError(f"{role} replay identities/labels do not join saved rows: {exc}") from exc
    reordered_ids = tuple(replay_identities[index] for index in order)
    _require(reordered_ids == saved.identities, f"{role} replay stable identity order mismatch")
    return order, labels[order]


def _compare_role(
    saved: PredictionBundle,
    probabilities: Any,
    predictions: Any,
    confidences: Any,
    order: np.ndarray,
    replay_labels: np.ndarray,
    *,
    role: str,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    probs, pred, conf = _validate_generated_predictions(
        probabilities,
        predictions,
        confidences,
        role=role,
    )
    probs = probs[order]
    pred = pred[order]
    conf = conf[order]
    _require(np.array_equal(replay_labels, saved.labels), f"{role} replay labels differ from saved labels")
    _require(np.array_equal(pred, saved.predictions), f"{role} replay predictions differ from saved bundle")
    _require(
        probs.shape == saved.probabilities.shape,
        f"{role} replay probability shape differs from saved bundle",
    )
    probability_difference = np.abs(probs.astype(np.float64) - saved.probabilities.astype(np.float64))
    confidence_difference = np.abs(conf.astype(np.float64) - saved.confidences.astype(np.float64))
    maximum_probability_difference = float(np.max(probability_difference))
    maximum_confidence_difference = float(np.max(confidence_difference))
    _require(
        np.allclose(probs, saved.probabilities, rtol=rtol, atol=atol),
        f"{role} replay probabilities differ from saved bundle; max_abs_diff={maximum_probability_difference:.12g}",
    )
    _require(
        np.allclose(conf, saved.confidences, rtol=rtol, atol=atol),
        f"{role} replay confidences differ from saved bundle; max_abs_diff={maximum_confidence_difference:.12g}",
    )
    return {
        "row_count": int(len(saved.identities)),
        "max_abs_probability_difference": maximum_probability_difference,
        "max_abs_confidence_difference": maximum_confidence_difference,
        "prediction_match": "EXACT",
        "identity_label_join": "PASS",
    }


def replay_fold(
    fold_dir: Path,
    *,
    data_dir: Path | None = None,
    feature_manifest: Path | None = None,
    probability_atol: float = DEFAULT_ATOL,
    probability_rtol: float = DEFAULT_RTOL,
    device: str = "recorded",
) -> dict[str, Any]:
    """Replay both frozen models and compare their outputs with saved bundles."""

    atol = _validated_tolerance(probability_atol, "probability_atol")
    rtol = _validated_tolerance(probability_rtol, "probability_rtol")
    fold_dir = Path(fold_dir).resolve()

    strict_result = verify_fold(
        fold_dir,
        data_dir=data_dir,
        feature_manifest=feature_manifest,
    )
    _require(
        strict_result.get("status") == "PASS_ARTIFACT_CONTRACTS_ONLY",
        "strict artifact verification did not pass",
    )
    manifest = _read_manifest(fold_dir)
    outer_subject = str(manifest["outer_subject_id"])
    raw_dir = _declared_input_path(manifest, "raw_data_dir")
    feature_manifest_path = _declared_input_path(manifest, "feature_cache_manifest")
    primary_checkpoint_path = _artifact_path(fold_dir, manifest, "primary_checkpoint")
    primary_predictions_path = _artifact_path(fold_dir, manifest, "primary_predictions")
    feature_model_path = _artifact_path(fold_dir, manifest, "feature_model")
    feature_predictions_path = _artifact_path(fold_dir, manifest, "feature_predictions")

    replay_device = _resolve_device(device, manifest)
    set_seed(int(manifest["seed"]))
    records = discover_raw_records(raw_dir)
    outer_files = files_for_subjects(records, (outer_subject,))
    (
        primary_ids,
        primary_labels,
        primary_probabilities,
        primary_predictions,
        primary_confidences,
    ) = primary_outer_inference(
        checkpoint_path=primary_checkpoint_path,
        expected_run_config_hash=manifest["run_config_hash"],
        outer_files=outer_files,
        outer_fold=outer_subject,
        batch_size=int(manifest["primary_config"]["batch_size"]),
        device=replay_device,
    )
    saved_primary = PredictionBundle.from_npz(primary_predictions_path)
    primary_order, primary_replay_labels = _align_replay(
        saved_primary,
        primary_ids,
        primary_labels,
        role="primary",
    )
    primary_result = _compare_role(
        saved_primary,
        primary_probabilities,
        primary_predictions,
        primary_confidences,
        primary_order,
        primary_replay_labels,
        role="primary",
        atol=atol,
        rtol=rtol,
    )
    primary_result["checkpoint_sha256"] = sha256_file(primary_checkpoint_path)
    primary_result["prediction_bundle_sha256"] = sha256_file(primary_predictions_path)

    try:
        feature_payload = joblib.load(feature_model_path)
    except Exception as exc:
        raise ContractError(f"cannot load feature model for replay: {exc}") from exc
    _require(
        isinstance(feature_payload, dict) and set(feature_payload) == {"model", "metadata"},
        "feature model payload is not canonical",
    )
    feature_model = feature_payload["model"]
    _require(hasattr(feature_model, "predict_proba"), "feature model lacks predict_proba")
    x_outer, feature_labels, feature_ids, feature_names = load_feature_rows(
        feature_manifest_path,
        (outer_subject,),
        outer_fold=outer_subject,
    )
    _require(
        list(feature_names) == feature_payload["metadata"]["feature_config"]["feature_names"],
        "feature replay column schema differs from fitted model metadata",
    )
    feature_probabilities = np.asarray(feature_model.predict_proba(x_outer))
    _require(
        feature_probabilities.shape == (len(x_outer), 5),
        f"feature replay probability shape mismatch: {feature_probabilities.shape}",
    )
    feature_predictions = np.argmax(feature_probabilities, axis=1).astype(np.int64)
    feature_confidences = np.max(feature_probabilities, axis=1)
    saved_feature = PredictionBundle.from_npz(feature_predictions_path)
    feature_order, feature_replay_labels = _align_replay(
        saved_feature,
        feature_ids,
        feature_labels,
        role="feature",
    )
    feature_result = _compare_role(
        saved_feature,
        feature_probabilities,
        feature_predictions,
        feature_confidences,
        feature_order,
        feature_replay_labels,
        role="feature",
        atol=atol,
        rtol=rtol,
    )
    feature_result["model_sha256"] = sha256_file(feature_model_path)
    feature_result["prediction_bundle_sha256"] = sha256_file(feature_predictions_path)
    _require(
        saved_primary.identities == saved_feature.identities
        and np.array_equal(saved_primary.labels, saved_feature.labels),
        "primary and feature saved rows are not one-to-one",
    )

    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "PASS_SOURCE_REPLAY",
        "source_binding_replay": "PASS",
        "strict_artifact_validation": strict_result["status"],
        "fold_dir": str(fold_dir),
        "outer_subject": outer_subject,
        "row_count": int(len(saved_primary.identities)),
        "device": str(replay_device),
        "tolerance": {"atol": atol, "rtol": rtol},
        "primary": primary_result,
        "feature": feature_result,
        "confirmatory_eligibility": "NOT_DETERMINED_BY_SOURCE_REPLAY_ALONE",
    }


def _outside_fold(output_path: Path, fold_dir: Path) -> Path:
    resolved = output_path.resolve()
    fold = fold_dir.resolve()
    _require(
        resolved != fold and not resolved.is_relative_to(fold),
        "output JSON must be outside the read-only fold directory",
    )
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--feature-manifest", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--device", choices=("recorded", "cpu", "cuda"), default="recorded")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = (
        _outside_fold(args.output_json, args.fold_dir)
        if args.output_json is not None
        else None
    )
    result = replay_fold(
        args.fold_dir,
        data_dir=args.data_dir,
        feature_manifest=args.feature_manifest,
        probability_atol=args.atol,
        probability_rtol=args.rtol,
        device=args.device,
    )
    if output_path is not None:
        atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
