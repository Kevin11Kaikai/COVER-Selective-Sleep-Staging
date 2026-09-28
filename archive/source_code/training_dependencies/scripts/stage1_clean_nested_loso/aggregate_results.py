"""Aggregate exactly 20 clean Stage-1 nested-LOSO folds without reinterpretation."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:  # Make direct path execution usable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.stage1_clean_nested_loso.contracts import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_hash,
    sha256_file,
)
from scripts.stage1_clean_nested_loso.preflight import (  # noqa: E402
    CANONICAL_ARTIFACTS,
    RUN_CONFIG_FIELDS,
)
from scripts.stage1_clean_nested_loso.splits import (  # noqa: E402
    build_subject_split,
    normalize_subject_id,
)


AGGREGATE_SCHEMA_VERSION = "stage1-clean-20fold-aggregate-v1"
RUN_SCHEMA_VERSION = "stage1-clean-run-manifest-v1"
METRIC_SCHEMA_VERSION = "stage1-clean-metrics-v1"
EXPECTED_SUBJECTS = tuple(f"{index:02d}" for index in range(20))
FIXED_TAU_H = 0.9
FIXED_KAPPA = 0.15
FIXED_KL_EPSILON = 1e-10
FIXED_MIN_SUPPORT = 10
BOOTSTRAP_SEED = 20260816
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_PERCENTILES = (2.5, 97.5)
BOOTSTRAP_PERCENTILE_METHOD = "linear"
FROZEN_COMMON_RUN_CONFIG: dict[str, Any] = {
    "schema_version": RUN_SCHEMA_VERSION,
    "dataset": "sleepedf20",
    "seed": BOOTSTRAP_SEED,
    "assignment_rule": (
        "stage1-clean-subject-split-v1: sort development subjects by "
        "(sha256(utf8(f'{schema_version}:{seed}:{outer_subject}:{subject_id}')), "
        "subject_id); then take contiguous roles in inner_train, "
        "checkpoint_selection, calibration, routing_selection order"
    ),
    "checkpoint_selection_rule": (
        "maximum checkpoint-selection accuracy; ties keep earliest epoch"
    ),
    "tau_h": FIXED_TAU_H,
    "kappa": FIXED_KAPPA,
    "kl_epsilon": FIXED_KL_EPSILON,
    "raw_data_hash": (
        "6553de7de5c29572b4b99532ad648d95e54b90ff8d6c32a6fbd40e6aa435130a"
    ),
    "feature_cache_hash": (
        "4de20c8ca0bfccc9337ae8a01ee1bb5b819d960ecab33a646e0914f700214960"
    ),
    "preprocessing_hash": (
        "386d6d9b7b8b8c78c3ae977702ae95f5d0311b2195aea7c4d3e7cdeaa1284094"
    ),
    "feature_preprocessing_hash": (
        "1a5352e2727791ab6a51cefc0fa1dbed9911a55f9484998182599867747f4c6d"
    ),
    "schema_hash": (
        "c52f5b319ce55828fe4da0af19a04cba01280a1742a45f496eb236ecefd3dd54"
    ),
    "feature_schema_hash": (
        "476f98d94a36c2c71cd413ac38d6fd7fc5f810a274320e35f23b82468be0c73c"
    ),
    "feature_extractor_hash": (
        "46ca5bf6c3eee209b719863999363e26a9d68078aa7c27088b227f59a7fca832"
    ),
    "input_paths": {
        "raw_data_dir": "data/Sleep-EDF-20/eeg_fpz_cz",
        "feature_cache_manifest": (
            "outputs/stage1_clean_nested_loso/feature_cache/"
            "manifest-9aa61e01e862e0cc8994611663efd3716a6caff74c4c349c6d19f54e7546e02b.json"
        ),
    },
    "code_hash": (
        "049e2174121f3a22c4840c1cdbb18c5a106bbeaa51c8b5d231c7a12e4c9da000"
    ),
    "primary_config": {
        "architecture": "Conformer",
        "batch_size": 64,
        "class_weights": True,
        "classes": 5,
        "depth": 6,
        "early_stop_patience": 20,
        "electrodes": 1,
        "emb_size": 40,
        "epochs": 100,
        "learning_rate": 0.001,
        "seed": BOOTSTRAP_SEED,
    },
    "feature_smote_ratio": 0.3,
    "metric_min_support": FIXED_MIN_SUPPORT,
    "environment": {
        "cublas_workspace_config": ":4096:8",
        "cuda_available": True,
        "cuda_runtime": "12.1",
        "cudnn": 90100,
        "deterministic_algorithms": True,
        "gpu_name": "NVIDIA GeForce RTX 4080 SUPER",
        "imbalanced_learn": "0.14.1",
        "loky_max_cpu_count": "20",
        "numpy": "2.2.6",
        "platform": "Windows-10-10.0.19045-SP0",
        "python": "3.10.19",
        "scikit_learn": "1.7.2",
        "selected_device": "cuda",
        "tf32": False,
        "torch": "2.5.1+cu121",
    },
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FEATURE_MODEL_RE = re.compile(r"^feature/cache/(?P<key>[0-9a-f]{64})[.]joblib$")
_FEATURE_METADATA_RE = re.compile(r"^feature/cache/(?P<key>[0-9a-f]{64})[.]json$")
_EXACT_ARTIFACT_KEYS = frozenset(
    {*CANONICAL_ARTIFACTS, "feature_model", "feature_model_metadata"}
)
_FINAL_MANIFEST_KEYS = frozenset(
    {
        *RUN_CONFIG_FIELDS,
        "run_config",
        "run_config_hash",
        "primary_checkpoint_frozen",
        "feature_model_frozen",
        "outer_evaluation_count",
        "outer_evaluation_completed",
        "selected_epoch",
        "selection_accuracy",
        "selection_loss",
        "primary_training_epochs",
        "runtime_seconds",
        "peak_gpu_memory_bytes",
        "artifacts",
        "status",
    }
)
RATIO_METRIC_NAMES = (
    "hc_risk",
    "accepted_hce_mass",
    "captured_hce_mass",
    "hce_capture_recall",
    "hc_escalation_mass",
    "hc_escalation_rate",
    "overall_escalation_rate",
    "ordinary_selective_risk",
    "repository_accept_hce_rate",
    "q1",
    "q2",
    "q3",
    "eta1",
    "eta2",
    "eta3",
)
_METRIC_KEYS = frozenset({"n_rows", *RATIO_METRIC_NAMES})
_RATIO_KEYS = frozenset({"numerator", "denominator", "value", "status"})
_UNCONDITIONAL_MASS_METRICS = frozenset(
    {
        "accepted_hce_mass",
        "captured_hce_mass",
        "hc_escalation_mass",
        "overall_escalation_rate",
        "q1",
        "q2",
        "q3",
    }
)
_Q_MASS_METRICS = frozenset({"q1", "q2", "q3"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _load_json(path: Path, description: str) -> Any:
    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"{description} has duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {description} {path}: {exc}") from exc


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    value = _load_json(path, description)
    _require(isinstance(value, dict), f"{description} must contain one JSON object")
    return value


def _expect_exact_keys(actual: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual_keys = set(actual)
    missing = sorted(expected - actual_keys)
    extra = sorted(actual_keys - expected)
    if missing or extra:
        raise ContractError(f"{name} keys differ: missing={missing}, extra={extra}")


def _discover_fold_directories(root: Path) -> dict[str, Path]:
    _require(root.is_dir(), f"aggregate root is not a directory: {root}")
    candidates = [item for item in root.iterdir() if item.name.startswith("fold_")]
    malformed = sorted(item.name for item in candidates if not item.is_dir())
    _require(not malformed, f"fold-prefixed entries must be directories: {malformed}")
    names = {item.name: item.resolve() for item in candidates}
    expected_names = {f"fold_{subject}" for subject in EXPECTED_SUBJECTS}
    missing = sorted(expected_names - set(names))
    extra = sorted(set(names) - expected_names)
    _require(not missing and not extra, f"fold set differs: missing={missing}, extra={extra}")
    return {subject: names[f"fold_{subject}"] for subject in EXPECTED_SUBJECTS}


def _validate_artifact_schema(manifest: Mapping[str, Any], subject: str) -> PurePosixPath:
    artifacts = manifest["artifacts"]
    _require(isinstance(artifacts, dict), f"fold {subject} artifacts must be a mapping")
    _expect_exact_keys(artifacts, _EXACT_ARTIFACT_KEYS, f"fold {subject} artifacts")
    for name, entry in artifacts.items():
        _require(isinstance(entry, dict), f"fold {subject} artifact {name} must be a mapping")
        _expect_exact_keys(entry, frozenset({"path", "sha256"}), f"fold {subject} artifact {name}")
        _require(
            isinstance(entry["sha256"], str) and _SHA256_RE.fullmatch(entry["sha256"]) is not None,
            f"fold {subject} artifact {name} has an invalid SHA-256",
        )
    for name, expected_path in CANONICAL_ARTIFACTS.items():
        _require(
            artifacts[name]["path"] == expected_path,
            f"fold {subject} artifact {name} path is not canonical",
        )
    feature_model_match = _FEATURE_MODEL_RE.fullmatch(str(artifacts["feature_model"]["path"]))
    feature_metadata_match = _FEATURE_METADATA_RE.fullmatch(
        str(artifacts["feature_model_metadata"]["path"])
    )
    _require(feature_model_match is not None, f"fold {subject} feature model path is not content-addressed")
    _require(feature_metadata_match is not None, f"fold {subject} feature metadata path is not content-addressed")
    _require(
        feature_model_match.group("key") == feature_metadata_match.group("key"),
        f"fold {subject} feature model and metadata keys differ",
    )
    return PurePosixPath(artifacts["metrics_fixed_kappa_015"]["path"])


def _validate_run_manifest(
    path: Path,
    *,
    fold_dir: Path,
    subject: str,
) -> tuple[dict[str, Any], Path]:
    manifest = _load_mapping(path, f"fold {subject} run manifest")
    _expect_exact_keys(manifest, _FINAL_MANIFEST_KEYS, f"fold {subject} run manifest")
    _require(manifest["schema_version"] == RUN_SCHEMA_VERSION, f"fold {subject} run schema mismatch")
    _require(manifest["dataset"] == "sleepedf20", f"fold {subject} dataset mismatch")
    _require(manifest["outer_fold"] == subject, f"fold {subject} outer_fold mismatch")
    _require(
        normalize_subject_id(manifest["outer_subject_id"]) == subject
        and manifest["outer_subject_id"] == subject,
        f"fold {subject} outer_subject_id mismatch",
    )
    _require(manifest["status"] == "COMPLETE_SINGLE_OUTER_FOLD", f"fold {subject} is incomplete or contaminated")
    _require(
        type(manifest["outer_evaluation_count"]) is int
        and manifest["outer_evaluation_count"] == 1,
        f"fold {subject} outer evaluation count is not one",
    )
    _require(manifest["outer_evaluation_completed"] is True, f"fold {subject} outer evaluation is incomplete")
    _require(manifest["primary_checkpoint_frozen"] is True, f"fold {subject} primary was not frozen")
    _require(manifest["feature_model_frozen"] is True, f"fold {subject} feature model was not frozen")
    _require(manifest["tau_h"] == FIXED_TAU_H, f"fold {subject} tau_h is not frozen")
    _require(manifest["kappa"] == FIXED_KAPPA, f"fold {subject} kappa is not frozen")
    _require(manifest["kl_epsilon"] == FIXED_KL_EPSILON, f"fold {subject} KL epsilon is not frozen")
    _require(
        manifest["metric_min_support"] == FIXED_MIN_SUPPORT,
        f"fold {subject} metric min_support is not frozen at 10",
    )
    for field, expected in FROZEN_COMMON_RUN_CONFIG.items():
        _require(
            canonical_json_bytes(manifest[field]) == canonical_json_bytes(expected),
            f"fold {subject} frozen common run field {field} drifted",
        )
    _require(type(manifest["seed"]) is int, f"fold {subject} seed must be an integer")

    split = build_subject_split(subject, EXPECTED_SUBJECTS, seed=manifest["seed"])
    _require(manifest["subject_split"] == split.as_dict(), f"fold {subject} split is contaminated")
    _require(manifest["assignment_rule"] == split.assignment_rule, f"fold {subject} assignment rule mismatch")
    run_config = {field: manifest[field] for field in RUN_CONFIG_FIELDS}
    _require(isinstance(manifest["run_config"], dict), f"fold {subject} nested run_config missing")
    _expect_exact_keys(
        manifest["run_config"],
        frozenset(RUN_CONFIG_FIELDS),
        f"fold {subject} nested run_config",
    )
    _require(manifest["run_config"] == run_config, f"fold {subject} nested run_config mismatch")
    _require(
        manifest["run_config_hash"] == canonical_json_hash(run_config),
        f"fold {subject} run_config_hash mismatch",
    )
    _require(
        isinstance(manifest["run_config_hash"], str)
        and _SHA256_RE.fullmatch(manifest["run_config_hash"]) is not None,
        f"fold {subject} run_config_hash is invalid",
    )

    relative_metric = _validate_artifact_schema(manifest, subject)
    resolved_artifacts: dict[str, Path] = {}
    for name, entry in manifest["artifacts"].items():
        relative = PurePosixPath(str(entry["path"]))
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"fold {subject} artifact {name} path is not fold-relative",
        )
        artifact_path = (fold_dir / Path(*relative.parts)).resolve()
        _require(
            artifact_path.is_relative_to(fold_dir),
            f"fold {subject} artifact {name} path escapes fold",
        )
        _require(
            artifact_path.is_file(),
            f"fold {subject} artifact {name} is missing",
        )
        _require(
            sha256_file(artifact_path) == entry["sha256"],
            f"fold {subject} artifact {name} SHA-256 mismatch",
        )
        resolved_artifacts[name] = artifact_path
    metric_path = resolved_artifacts["metrics_fixed_kappa_015"]
    _require(
        PurePosixPath(manifest["artifacts"]["metrics_fixed_kappa_015"]["path"])
        == relative_metric,
        f"fold {subject} metric artifact path mismatch",
    )
    return manifest, metric_path


def _validate_ratio(
    value: Any,
    *,
    metric_name: str,
    subject: str,
    n_rows: int,
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"fold {subject} metric {metric_name} must be a mapping")
    _expect_exact_keys(value, _RATIO_KEYS, f"fold {subject} metric {metric_name}")
    numerator = value["numerator"]
    denominator = value["denominator"]
    _require(
        type(numerator) is int and type(denominator) is int,
        f"fold {subject} metric {metric_name} counts must be integers",
    )
    _require(
        0 <= numerator <= denominator,
        f"fold {subject} metric {metric_name} has invalid counts",
    )
    if metric_name in _UNCONDITIONAL_MASS_METRICS:
        _require(
            denominator == n_rows,
            f"fold {subject} metric {metric_name} denominator must equal n_rows",
        )

    status = value["status"]
    _require(
        status in {"DEFINED", "LOW_SUPPORT", "UNDEFINED"},
        f"fold {subject} metric {metric_name} status is invalid",
    )
    if denominator == 0:
        _require(
            value["value"] is None and status == "UNDEFINED",
            f"fold {subject} metric {metric_name} zero denominator must be undefined",
        )
    else:
        _require(
            isinstance(value["value"], (int, float)) and not isinstance(value["value"], bool),
            f"fold {subject} metric {metric_name} value must be numeric",
        )
        expected_value = numerator / denominator
        _require(
            math.isfinite(float(value["value"]))
            and math.isclose(float(value["value"]), expected_value, rel_tol=0.0, abs_tol=1e-15),
            f"fold {subject} metric {metric_name} value differs from counts",
        )
        support = numerator if metric_name in _Q_MASS_METRICS else denominator
        expected_status = "LOW_SUPPORT" if support < FIXED_MIN_SUPPORT else "DEFINED"
        _require(
            status == expected_status,
            f"fold {subject} metric {metric_name} support status mismatch",
        )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": value["value"],
        "status": status,
    }


def _validate_metric_payload(path: Path, *, subject: str) -> dict[str, Any]:
    payload = _load_mapping(path, f"fold {subject} metric artifact")
    _expect_exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "policy",
                "tau_h",
                "kappa",
                "kl_epsilon",
                "min_support",
                "outer_subject",
                "metrics",
            }
        ),
        f"fold {subject} metric artifact",
    )
    _require(payload["schema_version"] == METRIC_SCHEMA_VERSION, f"fold {subject} metric schema mismatch")
    _require(payload["policy"] == "type1_forward_kl", f"fold {subject} policy mismatch")
    _require(payload["tau_h"] == FIXED_TAU_H, f"fold {subject} metric tau_h mismatch")
    _require(payload["kappa"] == FIXED_KAPPA, f"fold {subject} metric kappa mismatch")
    _require(payload["kl_epsilon"] == FIXED_KL_EPSILON, f"fold {subject} metric epsilon mismatch")
    _require(payload["min_support"] == FIXED_MIN_SUPPORT, f"fold {subject} metric min_support mismatch")
    _require(payload["outer_subject"] == subject, f"fold {subject} metric outer subject mismatch")
    metrics = payload["metrics"]
    _require(isinstance(metrics, dict), f"fold {subject} metrics must be a mapping")
    _expect_exact_keys(metrics, _METRIC_KEYS, f"fold {subject} metrics")
    n_rows = metrics["n_rows"]
    _require(type(n_rows) is int and n_rows > 0, f"fold {subject} n_rows must be positive")
    validated = {"n_rows": n_rows}
    for metric_name in RATIO_METRIC_NAMES:
        validated[metric_name] = _validate_ratio(
            metrics[metric_name],
            metric_name=metric_name,
            subject=subject,
            n_rows=n_rows,
        )
    return validated


def _percentile_interval(values: np.ndarray) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    lower, upper = np.percentile(
        values,
        BOOTSTRAP_PERCENTILES,
        method=BOOTSTRAP_PERCENTILE_METHOD,
    )
    return float(lower), float(upper)


def _bootstrap_summary(
    numerators: np.ndarray,
    denominators: np.ndarray,
    fold_values: np.ndarray,
    indices: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sampled_numerators = numerators[indices].sum(axis=1, dtype=np.int64)
    sampled_denominators = denominators[indices].sum(axis=1, dtype=np.int64)
    valid_micro = sampled_denominators > 0
    micro_values = sampled_numerators[valid_micro] / sampled_denominators[valid_micro]
    micro_lower, micro_upper = _percentile_interval(micro_values)
    micro = {
        "lower": micro_lower,
        "upper": micro_upper,
        "valid_resamples": int(valid_micro.sum()),
        "undefined_resamples": int((~valid_micro).sum()),
    }

    sampled_values = fold_values[indices]
    sampled_defined = np.isfinite(sampled_values)
    sampled_counts = sampled_defined.sum(axis=1)
    valid_macro = sampled_counts > 0
    sampled_sums = np.where(sampled_defined, sampled_values, 0.0).sum(axis=1)
    macro_values = sampled_sums[valid_macro] / sampled_counts[valid_macro]
    macro_lower, macro_upper = _percentile_interval(macro_values)
    macro = {
        "lower": macro_lower,
        "upper": macro_upper,
        "valid_resamples": int(valid_macro.sum()),
        "undefined_resamples": int((~valid_macro).sum()),
    }
    return micro, macro


def _aggregate_metric(
    fold_metrics: Sequence[dict[str, Any]],
    metric_name: str,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    rows = [metrics[metric_name] for metrics in fold_metrics]
    numerators = np.asarray([row["numerator"] for row in rows], dtype=np.int64)
    denominators = np.asarray([row["denominator"] for row in rows], dtype=np.int64)
    fold_values = np.asarray(
        [float(row["value"]) if row["value"] is not None else np.nan for row in rows],
        dtype=np.float64,
    )
    numerator = int(numerators.sum())
    denominator = int(denominators.sum())
    micro_value = float(numerator / denominator) if denominator > 0 else None
    numeric_mask = np.isfinite(fold_values)
    macro_value = float(fold_values[numeric_mask].mean()) if numeric_mask.any() else None
    defined_count = sum(row["status"] == "DEFINED" for row in rows)
    low_support_count = sum(row["status"] == "LOW_SUPPORT" for row in rows)
    undefined_count = sum(row["status"] == "UNDEFINED" for row in rows)
    micro_bootstrap, macro_bootstrap = _bootstrap_summary(
        numerators,
        denominators,
        fold_values,
        bootstrap_indices,
    )
    return {
        "micro": {
            "numerator": numerator,
            "denominator": denominator,
            "value": micro_value,
            "status": "DEFINED" if denominator > 0 else "NA",
            "bootstrap_percentile_ci": micro_bootstrap,
        },
        "subject_macro": {
            "value": macro_value,
            "status": "DEFINED" if macro_value is not None else "NA",
            "numeric_fold_count": int(numeric_mask.sum()),
            "defined_fold_count": defined_count,
            "low_support_fold_count": low_support_count,
            "undefined_fold_count": undefined_count,
            "bootstrap_percentile_ci": macro_bootstrap,
        },
    }


def aggregate_results(root: Path) -> dict[str, Any]:
    """Validate and aggregate the exact frozen 20-fold result set."""

    root = Path(root).resolve()
    fold_directories = _discover_fold_directories(root)
    fold_metrics: list[dict[str, Any]] = []
    input_hashes: list[dict[str, str]] = []
    seen_outer_subjects: set[str] = set()
    for subject in EXPECTED_SUBJECTS:
        fold_dir = fold_directories[subject]
        manifest_path = fold_dir / "run_manifest.json"
        _require(manifest_path.is_file(), f"fold {subject} run_manifest.json is missing")
        manifest, metric_path = _validate_run_manifest(
            manifest_path,
            fold_dir=fold_dir,
            subject=subject,
        )
        outer_subject = manifest["outer_subject_id"]
        _require(outer_subject not in seen_outer_subjects, f"duplicate logical outer subject: {outer_subject}")
        seen_outer_subjects.add(outer_subject)
        fold_metrics.append(_validate_metric_payload(metric_path, subject=subject))
        input_hashes.append(
            {
                "outer_subject": subject,
                "run_manifest_sha256": sha256_file(manifest_path),
                "run_config_hash": manifest["run_config_hash"],
                "metric_artifact_sha256": sha256_file(metric_path),
            }
        )
    _require(tuple(sorted(seen_outer_subjects)) == EXPECTED_SUBJECTS, "logical outer subject set mismatch")

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    bootstrap_indices = rng.integers(
        0,
        len(EXPECTED_SUBJECTS),
        size=(BOOTSTRAP_REPLICATES, len(EXPECTED_SUBJECTS)),
        endpoint=False,
    )
    aggregates = {
        metric_name: _aggregate_metric(fold_metrics, metric_name, bootstrap_indices)
        for metric_name in RATIO_METRIC_NAMES
    }
    content: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "dataset": "sleepedf20",
        "subjects": list(EXPECTED_SUBJECTS),
        "fold_count": len(EXPECTED_SUBJECTS),
        "total_rows": int(sum(metrics["n_rows"] for metrics in fold_metrics)),
        "frozen_policy": {
            "policy": "type1_forward_kl",
            "tau_h": FIXED_TAU_H,
            "kappa": FIXED_KAPPA,
            "kl_epsilon": FIXED_KL_EPSILON,
            "min_support": FIXED_MIN_SUPPORT,
        },
        "frozen_common_run_config": FROZEN_COMMON_RUN_CONFIG,
        "frozen_common_run_config_sha256": canonical_json_hash(
            FROZEN_COMMON_RUN_CONFIG
        ),
        "bootstrap": {
            "unit": "outer_subject_fold",
            "rng": "numpy.PCG64",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "resample_size": len(EXPECTED_SUBJECTS),
            "with_replacement": True,
            "percentiles": list(BOOTSTRAP_PERCENTILES),
            "percentile_method": BOOTSTRAP_PERCENTILE_METHOD,
            "undefined_resamples": "excluded_per_metric_and_reported",
        },
        "reporting_rule": "REPORT_ALL_METRICS_WITHOUT_SUCCESS_OR_FAVORABILITY_CRITERION",
        "input_hashes": input_hashes,
        "input_set_sha256": canonical_json_hash(input_hashes),
        "metrics": aggregates,
    }
    return {
        **content,
        "content_address": {
            "algorithm": "sha256-canonical-json-excluding-content_address",
            "sha256": canonical_json_hash(content),
        },
    }


def _validate_output_path(output: Path, fold_directories: Mapping[str, Path]) -> Path:
    resolved = Path(output).resolve()
    for subject, fold_dir in fold_directories.items():
        _require(
            resolved != fold_dir and not resolved.is_relative_to(fold_dir),
            f"aggregate output must not be inside fold_{subject}",
        )
    return resolved


def write_aggregate(root: Path, output: Path) -> dict[str, Any]:
    """Aggregate and atomically persist one idempotent content-addressed report."""

    root = Path(root).resolve()
    fold_directories = _discover_fold_directories(root)
    output_path = _validate_output_path(output, fold_directories)
    payload = aggregate_results(root)
    expected_bytes = canonical_json_bytes(payload) + b"\n"
    if output_path.exists():
        try:
            existing_bytes = output_path.read_bytes()
        except OSError as exc:
            raise ContractError(f"cannot read existing aggregate output: {exc}") from exc
        _require(
            existing_bytes == expected_bytes,
            "existing aggregate output differs from deterministic content-addressed result",
        )
    else:
        atomic_write_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = write_aggregate(args.root, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
