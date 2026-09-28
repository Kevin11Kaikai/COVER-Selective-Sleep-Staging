"""Fail-closed asset preflight for one clean Stage-1 outer fold.

Fold selection is intentionally blind to labels-as-outcomes, predictions, and
metrics.  It selects the numerically smallest subject whose raw records and
content hashes are structurally complete, validates the identity-bearing clean
feature manifest, then audits only canonical clean artifact paths for that subject.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Support both ``python -m ...preflight`` and direct script execution.
    from .contracts import (
        ContractError,
        PredictionBundle,
        RowIdentity,
        align_one_to_one,
        build_cache_key,
        canonical_json_hash,
        sha256_file,
    )
    from .feature_cache import validate_feature_cache, validate_feature_cache_against_raw
    from .splits import (
        ASSIGNMENT_RULE,
        DEFAULT_SPLIT_SEED,
        SplitProtocolError,
        SubjectSplit,
        build_subject_split,
        normalize_subject_id,
        validate_no_outer_leakage,
    )
except ImportError:  # pragma: no cover - exercised by CLI invocation, not import tests.
    from contracts import (  # type: ignore[no-redef]
        ContractError,
        PredictionBundle,
        RowIdentity,
        align_one_to_one,
        build_cache_key,
        canonical_json_hash,
        sha256_file,
    )
    from feature_cache import (  # type: ignore[no-redef]
        validate_feature_cache,
        validate_feature_cache_against_raw,
    )
    from splits import (  # type: ignore[no-redef]
        ASSIGNMENT_RULE,
        DEFAULT_SPLIT_SEED,
        SplitProtocolError,
        SubjectSplit,
        build_subject_split,
        normalize_subject_id,
        validate_no_outer_leakage,
    )


RUN_MANIFEST_SCHEMA_VERSION = "stage1-clean-run-manifest-v1"
PREDICTION_SCHEMA_VERSION = "stage1-clean-prediction-bundle-v1"
DATASET_ID = "sleepedf20"
FOLD_SELECTION_RULE = (
    "minimum numeric subject_id with non-empty raw records whose filename identities, "
    "row counts, labels, and content hashes pass raw inventory checks; "
    "no prediction or metric is read for fold selection"
)
RAW_RECORD_PATTERN = re.compile(r"^SC4(?P<subject>[0-9]{2})(?P<night>[0-9]+)E0[.]npz$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

CANONICAL_ARTIFACTS = {
    "primary_checkpoint": "primary/best_model.pth",
    "primary_recovery_checkpoint": "primary/last_checkpoint.pth",
    "primary_training_history": "primary/history.json",
    "primary_predictions": "primary/predictions.npz",
    "feature_predictions": "feature/predictions.npz",
    "metrics_fixed_kappa_015": "metrics_fixed_kappa_015.json",
}
CONTENT_ADDRESSED_FEATURE_PATTERN = re.compile(
    r"^feature/cache/(?P<cache_key>[0-9a-f]{64})[.]joblib$"
)
CONTENT_ADDRESSED_FEATURE_METADATA_PATTERN = re.compile(
    r"^feature/cache/(?P<cache_key>[0-9a-f]{64})[.]json$"
)
RUN_CONFIG_FIELDS = (
    "schema_version",
    "dataset",
    "outer_fold",
    "outer_subject_id",
    "seed",
    "assignment_rule",
    "subject_split",
    "checkpoint_selection_rule",
    "tau_h",
    "kappa",
    "kl_epsilon",
    "raw_data_hash",
    "feature_cache_hash",
    "preprocessing_hash",
    "feature_preprocessing_hash",
    "schema_hash",
    "feature_schema_hash",
    "feature_extractor_hash",
    "input_paths",
    "code_hash",
    "primary_config",
    "feature_smote_ratio",
    "metric_min_support",
    "environment",
)
PREDICTION_SCHEMA_HASH = canonical_json_hash(
    {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "identity": [
            "dataset",
            "subject_id",
            "record_id",
            "epoch_index",
            "outer_fold",
        ],
        "arrays": ["labels", "probabilities", "predictions", "confidences"],
    }
)
LEGACY_CONFIRMATORY_ROOTS = (
    PurePosixPath("outputs/sleepedf20"),
    PurePosixPath("outputs/mf_models"),
)


@dataclass(frozen=True)
class _RecordAsset:
    subject_id: str
    record_id: str
    filename: str
    row_count: int
    labels: np.ndarray
    sha256: str


@dataclass(frozen=True)
class _DatasetInspection:
    records: tuple[_RecordAsset, ...]
    raw_data_hash: str | None
    complete_subjects: tuple[str, ...]
    issues: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class _FeatureCacheInspection:
    path: Path
    manifest: Mapping[str, Any] | None
    summary: Mapping[str, Any] | None
    manifest_sha256: str | None
    preprocessing_hash: str | None
    issues: tuple[dict[str, str], ...]


def _issue(
    code: str,
    path: Path,
    detail: str,
    action: str,
    *,
    kind: str,
) -> dict[str, str]:
    return {
        "code": code,
        "kind": kind,
        "path": str(path.resolve()),
        "detail": detail,
        "minimal_recompute_action": action,
    }


def _sha256_is_valid(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _read_raw_record(path: Path, subject_id: str) -> _RecordAsset:
    before_hash = sha256_file(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "x" not in archive.files or "y" not in archive.files:
                raise ValueError("raw NPZ must contain x and y")
            features = archive["x"]
            labels = np.asarray(archive["y"])
    except Exception as error:
        raise ValueError(f"could not load raw record: {error}") from error
    if features.ndim < 1:
        raise ValueError("raw x must have a row dimension")
    if labels.ndim != 1:
        raise ValueError("raw y must be one-dimensional")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("raw y must use an integer dtype")
    if features.shape[0] == 0 or features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"raw row mismatch: x rows={features.shape[0]}, y rows={labels.shape[0]}"
        )
    labels = np.array(labels, copy=True)
    labels.setflags(write=False)
    if sha256_file(path) != before_hash:
        raise ValueError("raw record changed while being inspected")
    return _RecordAsset(
        subject_id=subject_id,
        record_id=path.stem,
        filename=path.name,
        row_count=int(labels.shape[0]),
        labels=labels,
        sha256=before_hash,
    )


def _inspect_dataset(
    data_dir: Path,
    expected_subjects: Sequence[Any],
) -> _DatasetInspection:
    issues: list[dict[str, str]] = []
    records: list[_RecordAsset] = []
    expected = tuple(normalize_subject_id(value) for value in expected_subjects)
    if len(set(expected)) != len(expected):
        raise ValueError("expected_subjects contains duplicate IDs")

    if not data_dir.is_dir():
        issues.append(
            _issue(
                "RAW_DATA_DIRECTORY_MISSING",
                data_dir,
                "canonical Sleep-EDF record directory is absent",
                "restore or preprocess the canonical Sleep-EDF record NPZ files",
                kind="missing",
            )
        )
    else:
        for path in sorted(data_dir.glob("SC*.npz"), key=lambda item: item.name):
            match = RAW_RECORD_PATTERN.fullmatch(path.name)
            if match is None:
                issues.append(
                    _issue(
                        "RAW_RECORD_NAME_INVALID",
                        path,
                        "record filename does not provide canonical subject/night identity",
                        "rename or regenerate the record using SC4{subject:02d}{night}E0.npz",
                        kind="invalid",
                    )
                )
                continue
            subject = normalize_subject_id(match.group("subject"))
            try:
                records.append(_read_raw_record(path, subject))
            except ValueError as error:
                issues.append(
                    _issue(
                        "RAW_RECORD_INVALID",
                        path,
                        str(error),
                        "regenerate this raw record from the frozen preprocessing pipeline",
                        kind="invalid",
                    )
                )

    by_filename = {record.filename: record for record in records}
    if len(by_filename) != len(records):
        issues.append(
            _issue(
                "RAW_RECORD_ID_DUPLICATE",
                data_dir,
                "canonical record identities are not unique",
                "regenerate records with one unique subject/night identity per file",
                kind="invalid",
            )
        )

    subjects_with_records = {record.subject_id for record in records}
    for subject in expected:
        if subject not in subjects_with_records:
            issues.append(
                _issue(
                    "SUBJECT_RAW_RECORDS_MISSING",
                    data_dir,
                    f"expected subject {subject} has no canonical raw record",
                    "restore or preprocess the missing subject before fold selection",
                    kind="missing",
                )
            )
    unexpected = sorted(subjects_with_records - set(expected))
    if unexpected:
        issues.append(
            _issue(
                "UNEXPECTED_SUBJECTS",
                data_dir,
                f"raw data contains subjects outside the declared cohort: {unexpected}",
                "freeze the intended cohort manifest and isolate unrelated records",
                kind="invalid",
            )
        )

    complete_subjects = []
    for subject in expected:
        subject_records = [record.filename for record in records if record.subject_id == subject]
        if subject_records:
            complete_subjects.append(subject)

    raw_hash: str | None = None
    if records:
        raw_hash = canonical_json_hash(
            [
                {
                    "name": record.filename,
                    "sha256": record.sha256,
                }
                for record in records
            ]
        )
    return _DatasetInspection(
        records=tuple(records),
        raw_data_hash=raw_hash,
        complete_subjects=tuple(complete_subjects),
        issues=tuple(issues),
    )


def _subject_sort_key(subject: str) -> tuple[int, int | str, str]:
    if subject.isdigit():
        return (0, int(subject), subject)
    return (1, subject, subject)


def select_dry_run_subject(complete_subjects: Sequence[Any]) -> str:
    """Select a dry-run fold using structural assets only, never metrics."""

    normalized = tuple(normalize_subject_id(value) for value in complete_subjects)
    if not normalized:
        raise ValueError("no subject has complete raw inventory assets")
    if len(set(normalized)) != len(normalized):
        raise ValueError("complete_subjects contains duplicates")
    return min(normalized, key=_subject_sort_key)


def _safe_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"could not parse JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError("JSON root must be an object")
    return value


def _inspect_feature_manifest(
    path: Path,
    dataset: _DatasetInspection,
    expected_subjects: Sequence[Any],
    raw_dir: Path,
) -> _FeatureCacheInspection:
    issues: list[dict[str, str]] = []
    if not path.is_file():
        issues.append(
            _issue(
                "FEATURE_MANIFEST_MISSING",
                path,
                "identity-bearing clean feature-cache manifest is absent",
                "build the clean 35-D cache from raw NPZs with feature_cache.py",
                kind="missing",
            )
        )
        return _FeatureCacheInspection(path, None, None, None, None, tuple(issues))

    try:
        summary = validate_feature_cache(path)
        raw_binding = validate_feature_cache_against_raw(path, raw_dir)
        manifest = _safe_json_mapping(path)
        manifest_sha256 = sha256_file(path)
        if summary.get("valid") is not True:
            raise ValueError("feature-cache validator did not return valid=true")
        if raw_binding.get("raw_binding") != "PASS":
            raise ValueError("feature-cache/raw binding validator did not return PASS")
        if Path(str(summary.get("manifest", ""))).resolve() != path.resolve():
            raise ValueError("feature-cache validator returned a different manifest path")
        if manifest.get("dataset") != DATASET_ID:
            raise ValueError(f"feature-cache dataset must be {DATASET_ID}")

        raw_sources = manifest.get("raw_sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError("feature-cache raw_sources must be a non-empty list")
        sources_by_name: dict[str, Mapping[str, Any]] = {}
        for source in raw_sources:
            if not isinstance(source, Mapping):
                raise ValueError("feature-cache raw_sources entries must be objects")
            filename = str(source.get("filename", ""))
            if filename in sources_by_name:
                raise ValueError(f"duplicate raw source filename: {filename}")
            sources_by_name[filename] = source
        records_by_name = {record.filename: record for record in dataset.records}
        missing = sorted(set(records_by_name) - set(sources_by_name))
        extra = sorted(set(sources_by_name) - set(records_by_name))
        if missing or extra:
            raise ValueError(
                f"feature-cache/raw inventory differs: missing={missing}, extra={extra}"
            )
        for filename, record in records_by_name.items():
            source = sources_by_name[filename]
            expected_values = {
                "subject_id": record.subject_id,
                "record_id": record.record_id,
                "row_count": record.row_count,
                "sha256": record.sha256,
            }
            mismatches = {
                field: {"manifest": source.get(field), "raw": expected}
                for field, expected in expected_values.items()
                if source.get(field) != expected
            }
            if mismatches:
                raise ValueError(
                    f"feature-cache raw source mismatch for {filename}: {mismatches}"
                )

        expected = sorted(normalize_subject_id(value) for value in expected_subjects)
        raw_subjects = sorted({record.subject_id for record in dataset.records})
        if raw_subjects != expected:
            raise ValueError("raw subject inventory differs from the frozen cohort")
        if manifest.get("subjects") != expected or summary.get("subjects") != expected:
            raise ValueError("feature-cache subjects differ from raw/frozen cohort")
        expected_rows = sum(record.row_count for record in dataset.records)
        if manifest.get("total_rows") != expected_rows or summary.get(
            "total_rows"
        ) != expected_rows:
            raise ValueError("feature-cache total_rows differs from raw inventory")
        for field in ("schema_hash", "extractor_hash", "config_hash", "code_hash"):
            if not _sha256_is_valid(manifest.get(field)):
                raise ValueError(f"feature-cache {field} is not a SHA-256 digest")
        preprocessing_hash = canonical_json_hash(
            {
                "manifest_sha256": manifest_sha256,
                "extractor_hash": manifest["extractor_hash"],
                "extractor_config_hash": manifest["config_hash"],
                "extractor_code_hash": manifest["code_hash"],
            }
        )
        return _FeatureCacheInspection(
            path,
            manifest,
            summary,
            manifest_sha256,
            preprocessing_hash,
            tuple(issues),
        )
    except (ContractError, OSError, ValueError, KeyError, TypeError) as error:
        issues.append(
            _issue(
                "FEATURE_MANIFEST_INVALID",
                path,
                str(error),
                "rebuild the clean feature cache from the exact frozen raw inventory",
                kind="invalid",
            )
        )
        return _FeatureCacheInspection(path, None, None, None, None, tuple(issues))


def _legacy_fusion_diagnostic(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "confirmatory_eligible": False,
            "used_for_fold_selection": False,
            "used_for_readiness": False,
        }
    return {
        "provided": True,
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "confirmatory_eligible": False,
        "used_for_fold_selection": False,
        "used_for_readiness": False,
    }


def _normalize_outer_fold(value: Any) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?:fold_)?([0-9]+)", text)
    if match is None:
        raise ValueError("outer_fold must be fold_XX or a numeric subject ID")
    return normalize_subject_id(match.group(1))


def _validate_run_manifest(
    manifest: Mapping[str, Any],
    *,
    fold_dir: Path,
    selected_subject: str,
    split: SubjectSplit,
    dataset: _DatasetInspection,
    feature_cache: _FeatureCacheInspection,
    repo_root: Path,
    raw_dir: Path,
) -> dict[str, Path]:
    required_fields = set(RUN_CONFIG_FIELDS) | {
        "run_config_hash",
        "primary_checkpoint_frozen",
        "feature_model_frozen",
        "outer_evaluation_count",
        "outer_evaluation_completed",
        "run_config",
        "artifacts",
        "status",
    }
    missing = sorted(required_fields - set(manifest))
    if missing:
        raise ValueError(f"run manifest missing fields: {missing}")
    if manifest["schema_version"] != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("run manifest schema_version is not the frozen v1 schema")
    if manifest["dataset"] != DATASET_ID:
        raise ValueError(f"run manifest dataset must be {DATASET_ID}")
    if _normalize_outer_fold(manifest["outer_fold"]) != selected_subject:
        raise ValueError("run manifest outer_fold does not match selected subject")
    if normalize_subject_id(manifest["outer_subject_id"]) != selected_subject:
        raise ValueError("run manifest outer_subject_id does not match selected subject")
    if isinstance(manifest["seed"], bool) or manifest["seed"] != split.seed:
        raise ValueError("run manifest seed does not match deterministic split")
    if manifest["assignment_rule"] != ASSIGNMENT_RULE:
        raise ValueError("run manifest assignment_rule is not the frozen v1 rule")
    if manifest["subject_split"] != split.as_dict():
        raise ValueError("run manifest subject_split differs from deterministic allocation")
    if not isinstance(manifest["checkpoint_selection_rule"], str) or not str(
        manifest["checkpoint_selection_rule"]
    ).strip():
        raise ValueError("checkpoint_selection_rule must be explicit and non-empty")
    if manifest["primary_checkpoint_frozen"] is not True:
        raise ValueError("primary checkpoint must be declared frozen before outer inference")
    if manifest["feature_model_frozen"] is not True:
        raise ValueError("feature model must be declared frozen before outer inference")
    if (
        type(manifest["outer_evaluation_count"]) is not int
        or manifest["outer_evaluation_count"] != 1
    ):
        raise ValueError("outer_evaluation_count must be exactly one")
    if manifest["outer_evaluation_completed"] is not True:
        raise ValueError("outer evaluation must be explicitly complete")
    if manifest["status"] != "COMPLETE_SINGLE_OUTER_FOLD":
        raise ValueError("run manifest does not declare a complete single outer fold")
    if (
        manifest["tau_h"] != 0.9
        or manifest["kappa"] != 0.15
        or manifest["kl_epsilon"] != 1e-10
    ):
        raise ValueError(
            "confirmatory manifest must freeze tau_h=0.9, kappa=0.15, "
            "and KL epsilon=1e-10"
        )
    if manifest["raw_data_hash"] != dataset.raw_data_hash:
        raise ValueError("run manifest raw_data_hash does not match inspected records")
    if feature_cache.manifest is None or feature_cache.manifest_sha256 is None:
        raise ValueError("validated identity-bearing feature cache is unavailable")
    if manifest["feature_cache_hash"] != feature_cache.manifest_sha256:
        raise ValueError("run manifest feature_cache_hash does not match clean manifest")
    if manifest["feature_preprocessing_hash"] != feature_cache.preprocessing_hash:
        raise ValueError("run manifest feature_preprocessing_hash mismatch")
    if manifest["feature_schema_hash"] != feature_cache.manifest["schema_hash"]:
        raise ValueError("run manifest feature_schema_hash mismatch")
    if manifest["feature_extractor_hash"] != feature_cache.manifest["extractor_hash"]:
        raise ValueError("run manifest feature_extractor_hash mismatch")
    try:
        expected_input_paths = {
            "raw_data_dir": raw_dir.resolve().relative_to(repo_root.resolve()).as_posix(),
            "feature_cache_manifest": feature_cache.path.resolve()
            .relative_to(repo_root.resolve())
            .as_posix(),
        }
    except ValueError as error:
        raise ValueError("confirmatory inputs must reside inside the repository") from error
    if manifest["input_paths"] != expected_input_paths:
        raise ValueError("run manifest input_paths differ from preflight inputs")
    for hash_name in (
        "code_hash",
        "preprocessing_hash",
        "feature_preprocessing_hash",
        "schema_hash",
        "feature_schema_hash",
        "feature_extractor_hash",
        "feature_cache_hash",
        "run_config_hash",
    ):
        if not _sha256_is_valid(manifest[hash_name]):
            raise ValueError(f"run manifest {hash_name} must be a lowercase SHA-256 digest")
    if manifest["schema_hash"] != PREDICTION_SCHEMA_HASH:
        raise ValueError("run manifest schema_hash does not match prediction schema v1")
    expected_config_hash = canonical_json_hash(
        {field: manifest[field] for field in RUN_CONFIG_FIELDS}
    )
    if manifest["run_config"] != {
        field: manifest[field] for field in RUN_CONFIG_FIELDS
    }:
        raise ValueError("nested run_config differs from flattened frozen fields")
    if manifest["run_config_hash"] != expected_config_hash:
        raise ValueError("run_config_hash does not bind the declared frozen run config")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError("run manifest artifacts must be an object")
    required_artifact_names = set(CANONICAL_ARTIFACTS) | {
        "feature_model",
        "feature_model_metadata",
    }
    if set(artifacts) != required_artifact_names:
        raise ValueError(
            "run manifest artifacts must contain exactly the eight canonical artifact keys"
        )
    resolved: dict[str, Path] = {}
    for name in sorted(required_artifact_names):
        entry = artifacts[name]
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"artifact {name} must contain exactly path and sha256")
        relative = PurePosixPath(str(entry["path"]).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"artifact {name} path must stay within the fold directory")
        if name == "feature_model":
            if CONTENT_ADDRESSED_FEATURE_PATTERN.fullmatch(relative.as_posix()) is None:
                raise ValueError(
                    "feature_model path must be feature/cache/<64-hex-cache-key>.joblib"
                )
        elif name == "feature_model_metadata":
            if (
                CONTENT_ADDRESSED_FEATURE_METADATA_PATTERN.fullmatch(
                    relative.as_posix()
                )
                is None
            ):
                raise ValueError(
                    "feature_model_metadata path must be "
                    "feature/cache/<64-hex-cache-key>.json"
                )
        elif relative.as_posix() != CANONICAL_ARTIFACTS[name]:
            raise ValueError(
                f"artifact {name} path must be canonical: {CANONICAL_ARTIFACTS[name]}"
            )
        actual_path = fold_dir.joinpath(*relative.parts)
        if not actual_path.is_file():
            raise ValueError(f"artifact {name} is missing at its declared path")
        actual_hash = sha256_file(actual_path)
        if entry["sha256"] != actual_hash:
            raise ValueError(f"artifact {name} SHA-256 does not match file contents")
        resolved[name] = actual_path
    if resolved["feature_model"].stem != resolved["feature_model_metadata"].stem:
        raise ValueError("feature model and metadata cache keys differ")
    metric_payload = _safe_json_mapping(resolved["metrics_fixed_kappa_015"])
    metric_required = {
        "schema_version",
        "policy",
        "tau_h",
        "kappa",
        "kl_epsilon",
        "min_support",
        "outer_subject",
        "metrics",
    }
    if set(metric_payload) != metric_required:
        raise ValueError("fixed-kappa metric artifact fields differ from frozen schema")
    if (
        metric_payload["schema_version"] != "stage1-clean-metrics-v1"
        or metric_payload["policy"] != "type1_forward_kl"
        or metric_payload["tau_h"] != 0.9
        or metric_payload["kappa"] != 0.15
        or metric_payload["kl_epsilon"] != 1e-10
        or metric_payload["min_support"] != manifest["metric_min_support"]
        or normalize_subject_id(metric_payload["outer_subject"]) != selected_subject
        or not isinstance(metric_payload["metrics"], Mapping)
    ):
        raise ValueError("fixed-kappa metric artifact violates canonical policy provenance")
    return resolved


def _expected_outer_rows(
    records: Sequence[_RecordAsset], selected_subject: str, outer_fold: str
) -> tuple[tuple[RowIdentity, ...], np.ndarray]:
    identities: list[RowIdentity] = []
    labels: list[Any] = []
    for record in sorted(records, key=lambda item: item.filename):
        if record.subject_id != selected_subject:
            continue
        for epoch_index, label in enumerate(record.labels.tolist()):
            identities.append(
                RowIdentity(
                    dataset=DATASET_ID,
                    subject_id=selected_subject,
                    record_id=record.record_id,
                    epoch_index=epoch_index,
                    outer_fold=outer_fold,
                )
            )
            labels.append(label)
    return tuple(identities), np.asarray(labels)


def _validate_feature_cache_metadata(
    model_path: Path,
    metadata_path: Path,
    *,
    split: SubjectSplit,
    run_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    if metadata_path != model_path.with_suffix(".json"):
        raise ValueError("feature model metadata path does not match model cache key")
    metadata = _safe_json_mapping(metadata_path)
    required = {
        "schema_version",
        "cache_key",
        "split_hash",
        "schema_hash",
        "model_hash",
        "data_hash",
        "extractor_hash",
        "config_hash",
        "calibration_hash",
        "seed",
        "training_subjects",
        "calibration_subjects",
        "feature_config",
    }
    if set(metadata) != required:
        raise ValueError(
            f"feature cache sidecar fields differ from contract: "
            f"missing={sorted(required - set(metadata))}, "
            f"extra={sorted(set(metadata) - required)}"
        )
    cache_key = model_path.stem
    if metadata["cache_key"] != cache_key:
        raise ValueError("feature sidecar cache_key differs from content-addressed filename")
    for hash_name in (
        "split_hash",
        "schema_hash",
        "model_hash",
        "data_hash",
        "extractor_hash",
        "config_hash",
        "calibration_hash",
    ):
        if not _sha256_is_valid(metadata[hash_name]):
            raise ValueError(f"feature sidecar {hash_name} is not a SHA-256 digest")
    if metadata["split_hash"] != split.split_hash:
        raise ValueError("feature sidecar split_hash mismatch")
    if metadata["schema_hash"] != run_manifest["feature_schema_hash"]:
        raise ValueError("feature sidecar schema_hash mismatch")
    if metadata["data_hash"] != run_manifest["feature_cache_hash"]:
        raise ValueError("feature sidecar data_hash mismatch")
    if metadata["extractor_hash"] != run_manifest["feature_extractor_hash"]:
        raise ValueError("feature sidecar extractor_hash mismatch")
    if metadata["seed"] != split.seed:
        raise ValueError("feature sidecar seed mismatch")
    if metadata["training_subjects"] != list(split.inner_train):
        raise ValueError("feature sidecar training_subjects differ from inner_train role")
    if metadata["calibration_subjects"] != list(split.calibration):
        raise ValueError("feature sidecar calibration_subjects differ from calibration role")
    expected_model_hash = canonical_json_hash(metadata["feature_config"])
    if metadata["model_hash"] != expected_model_hash:
        raise ValueError("feature sidecar model_hash does not bind feature_config")
    if metadata["config_hash"] != expected_model_hash:
        raise ValueError("feature sidecar config_hash does not bind feature_config")
    expected_calibration_hash = canonical_json_hash(
        {"method": "isotonic", "subjects": list(split.calibration), "smote": False}
    )
    if metadata["calibration_hash"] != expected_calibration_hash:
        raise ValueError("feature sidecar calibration_hash does not bind calibration role")
    expected_cache_key = build_cache_key(
        split_hash=metadata["split_hash"],
        schema_hash=metadata["schema_hash"],
        model_hash=metadata["model_hash"],
        data_hash=metadata["data_hash"],
        config_hash=metadata["config_hash"],
        calibration_hash=metadata["calibration_hash"],
        seed=metadata["seed"],
    )
    if cache_key != expected_cache_key:
        raise ValueError("feature cache filename does not reproduce from provenance fields")
    return metadata


def _validate_bundle(
    path: Path,
    model_path: Path,
    *,
    selected_subject: str,
    split: SubjectSplit,
    dataset: _DatasetInspection,
    expected_identities: tuple[RowIdentity, ...],
    expected_labels: np.ndarray,
    model_role: str,
    run_manifest: Mapping[str, Any],
    feature_metadata: Mapping[str, Any] | None = None,
) -> PredictionBundle:
    bundle = PredictionBundle.from_npz(path)
    outer_fold = selected_subject
    manifest = bundle.manifest
    if manifest.dataset != DATASET_ID or manifest.outer_fold != outer_fold:
        raise ValueError(f"{model_role} bundle dataset/outer_fold provenance mismatch")
    if normalize_subject_id(manifest.outer_subject_id) != selected_subject:
        raise ValueError(f"{model_role} bundle outer_subject_id mismatch")
    if manifest.seed != split.seed or manifest.split_hash != split.split_hash:
        raise ValueError(f"{model_role} bundle seed/split_hash mismatch")
    expected_data_hash = (
        dataset.raw_data_hash
        if model_role == "primary"
        else run_manifest["feature_cache_hash"]
    )
    if manifest.data_hash != expected_data_hash:
        raise ValueError(f"{model_role} bundle data_hash mismatch")
    if manifest.checkpoint_hash != sha256_file(model_path):
        raise ValueError(f"{model_role} bundle checkpoint/model file hash mismatch")
    if selected_subject in {
        normalize_subject_id(subject)
        for subject in manifest.training_subjects + manifest.calibration_subjects
    }:
        raise ValueError(f"outer subject appears in {model_role} training/calibration provenance")
    development = set(split.development_subjects)
    if not set(manifest.training_subjects).issubset(development):
        raise ValueError(f"{model_role} training subjects fall outside development cohort")
    if tuple(manifest.training_subjects) != tuple(sorted(split.inner_train)):
        raise ValueError(f"{model_role} training subjects differ from inner_train role")
    if not set(manifest.calibration_subjects).issubset(set(split.calibration)):
        raise ValueError(f"{model_role} calibration subjects violate the calibration role")
    expected_calibration = tuple(sorted(split.calibration)) if model_role == "feature" else ()
    if tuple(manifest.calibration_subjects) != expected_calibration:
        raise ValueError(f"{model_role} calibration subjects differ from frozen role")
    if any(normalize_subject_id(identity.subject_id) != selected_subject for identity in bundle.identities):
        raise ValueError(f"{model_role} predictions include a non-outer subject")
    if manifest.schema_hash != run_manifest["schema_hash"]:
        raise ValueError(f"{model_role} bundle schema_hash mismatch")
    expected_preprocessing_hash = (
        run_manifest["preprocessing_hash"]
        if model_role == "primary"
        else run_manifest["feature_preprocessing_hash"]
    )
    if manifest.preprocessing_hash != expected_preprocessing_hash:
        raise ValueError(f"{model_role} bundle preprocessing_hash mismatch")
    if manifest.code_hash != run_manifest["code_hash"]:
        raise ValueError(f"{model_role} bundle code_hash mismatch")
    if model_role == "primary":
        expected_primary_hash = canonical_json_hash(run_manifest["primary_config"])
        if (
            manifest.model_hash != expected_primary_hash
            or manifest.config_hash != expected_primary_hash
        ):
            raise ValueError("primary bundle model/config hash mismatch")
    if model_role == "feature":
        if feature_metadata is None:
            raise ValueError("feature cache metadata is required")
        for field_name in ("model_hash", "config_hash", "calibration_hash"):
            if getattr(manifest, field_name) != feature_metadata[field_name]:
                raise ValueError(f"feature bundle {field_name} differs from cache metadata")
    align_one_to_one(
        expected_identities,
        bundle.identities,
        reference_labels=expected_labels,
        candidate_labels=bundle.labels,
    )
    return bundle


def _path_is_legacy(repo_root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    normalized = PurePosixPath(relative.as_posix())
    return any(normalized == root or root in normalized.parents for root in LEGACY_CONFIRMATORY_ROOTS)


def run_preflight(
    *,
    repo_root: str | Path,
    feature_manifest: str | Path,
    data_dir: str | Path | None = None,
    fusion_cache: str | Path | None = None,
    clean_root: str | Path | None = None,
    expected_subjects: Sequence[Any] = tuple(range(20)),
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Audit raw/clean-feature identity and canonical artifacts for one fold."""

    root = Path(repo_root).resolve()
    raw_dir = (
        Path(data_dir).resolve()
        if data_dir is not None
        else root / "data" / "Sleep-EDF-20" / "eeg_fpz_cz"
    )
    feature_manifest_path = Path(feature_manifest).resolve()
    legacy_fusion_path = Path(fusion_cache).resolve() if fusion_cache is not None else None
    output_root = (
        Path(clean_root).resolve()
        if clean_root is not None
        else root / "outputs" / "stage1_clean_nested_loso"
    )
    inspection = _inspect_dataset(raw_dir, expected_subjects)
    feature_inspection = _inspect_feature_manifest(
        feature_manifest_path, inspection, expected_subjects, raw_dir
    )
    issues = list(inspection.issues)
    issues.extend(feature_inspection.issues)

    selected_subject: str | None = None
    split: SubjectSplit | None = None
    if inspection.complete_subjects:
        selected_subject = select_dry_run_subject(inspection.complete_subjects)
        try:
            split = build_subject_split(
                selected_subject,
                expected_subjects,
                seed=seed,
            )
        except SplitProtocolError as error:
            issues.append(
                _issue(
                    "SUBJECT_SPLIT_INVALID",
                    output_root,
                    str(error),
                    "fix the declared cohort/split sizes before any training or inference",
                    kind="invalid",
                )
            )
    else:
        issues.append(
            _issue(
                "NO_ASSET_COMPLETE_SUBJECT",
                raw_dir,
                "no subject passed the raw structural completeness checks",
                "restore canonical raw records before selecting a fold",
                kind="missing",
            )
        )

    required_paths: dict[str, Path] = {}
    if selected_subject is not None:
        fold_dir = output_root / f"fold_{selected_subject}"
        required_paths = {
            "run_manifest": fold_dir / "run_manifest.json",
            **{
                name: fold_dir.joinpath(*PurePosixPath(relative).parts)
                for name, relative in CANONICAL_ARTIFACTS.items()
            },
        }
        manifest: Mapping[str, Any] | None = None
        validated_manifest = False
        feature_metadata: Mapping[str, Any] | None = None
        if split is not None and required_paths["run_manifest"].is_file():
            try:
                manifest = _safe_json_mapping(required_paths["run_manifest"])
                resolved = _validate_run_manifest(
                    manifest,
                    fold_dir=fold_dir,
                    selected_subject=selected_subject,
                    split=split,
                    dataset=inspection,
                    feature_cache=feature_inspection,
                    repo_root=root,
                    raw_dir=raw_dir,
                )
                required_paths.update(resolved)
                validated_manifest = True
            except (ValueError, OSError, ContractError, SplitProtocolError) as error:
                issues.append(
                    _issue(
                        "RUN_MANIFEST_INVALID",
                        required_paths["run_manifest"],
                        str(error),
                        "regenerate the run manifest from actual split and artifact hashes; "
                        "do not patch provenance fields manually",
                        kind="invalid",
                    )
                )

        if "feature_model" not in required_paths:
            candidates = sorted((fold_dir / "feature" / "cache").glob("*.joblib"))
            if len(candidates) == 1:
                required_paths["feature_model"] = candidates[0]
                required_paths["feature_model_metadata"] = candidates[0].with_suffix(
                    ".json"
                )
            else:
                required_paths["feature_model"] = (
                    fold_dir / "feature" / "cache" / "<content-key>.joblib"
                )
                required_paths["feature_model_metadata"] = (
                    fold_dir / "feature" / "cache" / "<content-key>.json"
                )
                if len(candidates) > 1:
                    issues.append(
                        _issue(
                            "FEATURE_MODEL_AMBIGUOUS",
                            fold_dir / "feature" / "cache",
                            "multiple content-addressed models exist without a valid manifest",
                            "use the run manifest to bind exactly one feature cache key",
                            kind="invalid",
                        )
                    )

        action_by_name = {
            "run_manifest": "create the deterministic split/run manifest after artifacts",
            "primary_checkpoint": (
                "train only fold_XX with inner-subject checkpoint selection; freeze it "
                "before evaluating the outer subject"
            ),
            "primary_recovery_checkpoint": (
                "persist the resumable last primary checkpoint bound to run_config_hash"
            ),
            "primary_training_history": (
                "persist the complete inner-train/checkpoint-selection history"
            ),
            "primary_predictions": (
                "infer the frozen primary checkpoint once on the outer subject and persist "
                "stable row IDs plus full provenance"
            ),
            "feature_model": (
                "fit and calibrate the content-addressed feature expert using only its "
                "declared subject-grouped development roles"
            ),
            "feature_model_metadata": (
                "persist the exact content-addressed feature model metadata sidecar"
            ),
            "feature_predictions": (
                "infer the frozen calibrated feature expert on the same stable outer rows"
            ),
            "metrics_fixed_kappa_015": (
                "compute the frozen tau_h=0.9, kappa=0.15 metric artifact without tuning "
                "on outer labels"
            ),
        }
        for name, path in required_paths.items():
            if _path_is_legacy(root, path):
                issues.append(
                    _issue(
                        "LEGACY_ARTIFACT_REJECTED",
                        path,
                        "outputs/sleepedf20 and outputs/mf_models are diagnostic-only",
                        "create the artifact under outputs/stage1_clean_nested_loso/fold_XX",
                        kind="invalid",
                    )
                )
            elif not path.is_file():
                issues.append(
                    _issue(
                        f"{name.upper()}_MISSING",
                        path,
                        f"required clean {name} artifact is absent",
                        action_by_name[name],
                        kind="missing",
                    )
                )

        if validated_manifest and manifest is not None:
            try:
                feature_metadata = _validate_feature_cache_metadata(
                    required_paths["feature_model"],
                    required_paths["feature_model_metadata"],
                    split=split,
                    run_manifest=manifest,
                )
            except (ValueError, OSError) as error:
                issues.append(
                    _issue(
                        "FEATURE_MODEL_PROVENANCE_INVALID",
                        required_paths["feature_model_metadata"],
                        str(error),
                        "regenerate the content-addressed feature model and exact sidecar",
                        kind="invalid",
                    )
                )

        if validated_manifest and manifest is not None:
            outer_fold = selected_subject
            expected_ids, expected_labels = _expected_outer_rows(
                inspection.records, selected_subject, outer_fold
            )
            bundles: dict[str, PredictionBundle] = {}
            for role, prediction_name, model_name in (
                ("primary", "primary_predictions", "primary_checkpoint"),
                ("feature", "feature_predictions", "feature_model"),
            ):
                if not (
                    required_paths[prediction_name].is_file()
                    and required_paths[model_name].is_file()
                ):
                    continue
                if role == "feature" and feature_metadata is None:
                    continue
                try:
                    bundles[role] = _validate_bundle(
                        required_paths[prediction_name],
                        required_paths[model_name],
                        selected_subject=selected_subject,
                        split=split,
                        dataset=inspection,
                        expected_identities=expected_ids,
                        expected_labels=expected_labels,
                        model_role=role,
                        run_manifest=manifest,
                        feature_metadata=feature_metadata if role == "feature" else None,
                    )
                except (ValueError, OSError, ContractError) as error:
                    issues.append(
                        _issue(
                            f"{role.upper()}_PREDICTION_BUNDLE_INVALID",
                            required_paths[prediction_name],
                            str(error),
                            f"regenerate the {role} prediction bundle from its frozen model "
                            "and stable outer epoch IDs",
                            kind="invalid",
                        )
                    )
            if set(bundles) == {"primary", "feature"}:
                try:
                    align_one_to_one(
                        bundles["primary"].identities,
                        bundles["feature"].identities,
                        reference_labels=bundles["primary"].labels,
                        candidate_labels=bundles["feature"].labels,
                    )
                    if (
                        bundles["primary"].manifest.schema_hash
                        != bundles["feature"].manifest.schema_hash
                    ):
                        raise ValueError("primary and feature schema_hash values differ")
                except (ValueError, ContractError) as error:
                    issues.append(
                        _issue(
                            "PREDICTION_BUNDLES_MISALIGNED",
                            fold_dir,
                            str(error),
                            "regenerate both bundles from the same stable outer row manifest",
                            kind="invalid",
                        )
                    )

    invalid_present = any(issue["kind"] == "invalid" for issue in issues)
    if invalid_present:
        status = "BLOCKED_BY_PIPELINE_FAILURE"
    elif issues:
        status = "BLOCKED_BY_MISSING_ASSETS"
    else:
        status = "READY_FOR_SINGLE_FOLD_DRY_RUN_VERIFICATION"
    actions = list(
        dict.fromkeys(issue["minimal_recompute_action"] for issue in issues)
    )
    return {
        "schema_version": "stage1-clean-preflight-v1",
        "status": status,
        "selection": {
            "rule": FOLD_SELECTION_RULE,
            "selected_outer_subject": selected_subject,
            "metric_files_read": [],
        },
        "dataset": {
            "data_dir": str(raw_dir),
            "expected_subjects": [normalize_subject_id(item) for item in expected_subjects],
            "complete_subjects": list(inspection.complete_subjects),
            "record_count": len(inspection.records),
            "epoch_count": sum(record.row_count for record in inspection.records),
            "raw_data_hash": inspection.raw_data_hash,
        },
        "feature_cache": {
            "manifest": str(feature_manifest_path),
            "valid": feature_inspection.manifest is not None,
            "manifest_sha256": feature_inspection.manifest_sha256,
            "preprocessing_hash": feature_inspection.preprocessing_hash,
            "schema_hash": (
                feature_inspection.manifest.get("schema_hash")
                if feature_inspection.manifest is not None
                else None
            ),
            "extractor_hash": (
                feature_inspection.manifest.get("extractor_hash")
                if feature_inspection.manifest is not None
                else None
            ),
            "subjects": (
                list(feature_inspection.summary.get("subjects", []))
                if feature_inspection.summary is not None
                else []
            ),
            "total_rows": (
                feature_inspection.summary.get("total_rows")
                if feature_inspection.summary is not None
                else None
            ),
        },
        "legacy_fusion_diagnostic": _legacy_fusion_diagnostic(legacy_fusion_path),
        "split": split.as_dict() if split is not None else None,
        "required_artifacts": {
            name: str(path.resolve()) for name, path in required_paths.items()
        },
        "issues": issues,
        "minimal_recompute_actions": actions,
        "legacy_artifacts_confirmatory": False,
    }


def _parse_subjects(value: str) -> tuple[str, ...]:
    subjects = tuple(part.strip() for part in value.split(",") if part.strip())
    if not subjects:
        raise argparse.ArgumentTypeError("--subjects must contain at least one ID")
    try:
        return tuple(normalize_subject_id(subject) for subject in subjects)
    except SplitProtocolError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        required=True,
        help="identity-bearing clean feature-cache manifest",
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--fusion-cache",
        type=Path,
        help="optional legacy diagnostic only; never used for readiness",
    )
    parser.add_argument("--clean-root", type=Path)
    parser.add_argument(
        "--subjects",
        type=_parse_subjects,
        default=tuple(normalize_subject_id(index) for index in range(20)),
        help="comma-separated frozen cohort (default: 00 through 19)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--output", type=Path, help="also write the JSON report here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_preflight(
        repo_root=args.repo_root,
        feature_manifest=args.feature_manifest,
        data_dir=args.data_dir,
        fusion_cache=args.fusion_cache,
        clean_root=args.clean_root,
        expected_subjects=args.subjects,
        seed=args.seed,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if result["status"] == "READY_FOR_SINGLE_FOLD_DRY_RUN_VERIFICATION" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANONICAL_ARTIFACTS",
    "DATASET_ID",
    "FOLD_SELECTION_RULE",
    "PREDICTION_SCHEMA_HASH",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "build_parser",
    "main",
    "run_preflight",
    "select_dry_run_subject",
]
