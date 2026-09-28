"""Fail-closed verification for one Stage-1 clean nested-LOSO outer fold.

The verifier independently reopens the frozen inputs and every declared output,
checks their provenance graph, and exactly recomputes the fixed-policy metrics.
It does not replay either fitted model. Consequently, a successful result is an
artifact-contract pass, not a claim of confirmatory eligibility.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__:  # pragma: no branch - module execution path.
    from .contracts import (
        ContractError,
        PredictionBundle,
        RowIdentity,
        align_one_to_one,
        atomic_write_json,
        build_cache_key,
        canonical_json_hash,
        sha256_file,
    )
    from .feature_cache import (
        EXTRACTOR_HASH,
        FEATURE_CACHE_SCHEMA_HASH,
        load_feature_rows,
        validate_feature_cache,
        validate_feature_cache_against_raw,
    )
    from .metrics import evaluate_policy_metrics
    from .policy import route_type1_forward_kl
    from .preflight import CANONICAL_ARTIFACTS, PREDICTION_SCHEMA_HASH, RUN_CONFIG_FIELDS
    from .splits import SubjectSplit, build_subject_split, normalize_subject_id
else:  # pragma: no cover - exercised by the documented direct CLI.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.stage1_clean_nested_loso.contracts import (
        ContractError,
        PredictionBundle,
        RowIdentity,
        align_one_to_one,
        atomic_write_json,
        build_cache_key,
        canonical_json_hash,
        sha256_file,
    )
    from scripts.stage1_clean_nested_loso.feature_cache import (
        EXTRACTOR_HASH,
        FEATURE_CACHE_SCHEMA_HASH,
        load_feature_rows,
        validate_feature_cache,
        validate_feature_cache_against_raw,
    )
    from scripts.stage1_clean_nested_loso.metrics import evaluate_policy_metrics
    from scripts.stage1_clean_nested_loso.policy import route_type1_forward_kl
    from scripts.stage1_clean_nested_loso.preflight import (
        CANONICAL_ARTIFACTS,
        PREDICTION_SCHEMA_HASH,
        RUN_CONFIG_FIELDS,
    )
    from scripts.stage1_clean_nested_loso.splits import (
        SubjectSplit,
        build_subject_split,
        normalize_subject_id,
    )


RUN_SCHEMA_VERSION = "stage1-clean-run-manifest-v1"
PREDICTION_SCHEMA_VERSION = "stage1-clean-prediction-bundle-v1"
METRIC_SCHEMA_VERSION = "stage1-clean-metrics-v1"
DATASET_ID = "sleepedf20"
FIXED_TAU_H = 0.9
FIXED_KAPPA = 0.15
FIXED_KL_EPSILON = 1e-10
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FEATURE_MODEL_RE = re.compile(r"^feature/cache/(?P<cache_key>[0-9a-f]{64})[.]joblib$")
_RECORD_RE = re.compile(r"^SC4(?P<subject>[0-9]{2})[0-9]+E0$")

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
_FEATURE_METADATA_KEYS = frozenset(
    {
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
)
_BEST_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "run_config_hash",
        "epoch",
        "model_state_dict",
        "selection_accuracy",
        "selection_loss",
        "selection_rule",
    }
)
_RECOVERY_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "run_config_hash",
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_accuracy",
        "best_loss",
        "best_epoch",
        "without_improvement",
        "history",
        "rng_state",
    }
)
_HISTORY_ENTRY_KEYS = frozenset(
    {
        "epoch",
        "train_loss",
        "train_accuracy",
        "selection_loss",
        "selection_accuracy",
        "improved",
        "seconds",
    }
)
_ENVIRONMENT_KEYS = frozenset(
    {
        "python",
        "platform",
        "numpy",
        "scikit_learn",
        "imbalanced_learn",
        "torch",
        "cuda_available",
        "selected_device",
        "cuda_runtime",
        "cudnn",
        "gpu_name",
        "cublas_workspace_config",
        "loky_max_cpu_count",
        "deterministic_algorithms",
        "tf32",
    }
)
_PRIMARY_CONFIG_KEYS = frozenset(
    {
        "architecture",
        "emb_size",
        "depth",
        "classes",
        "electrodes",
        "epochs",
        "batch_size",
        "learning_rate",
        "early_stop_patience",
        "class_weights",
        "seed",
    }
)


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


def _load_json_mapping(path: Path, description: str) -> dict[str, Any]:
    value = _load_json(path, description)
    _require(isinstance(value, dict), f"{description} must contain one JSON object")
    return value


def _expect_exact_keys(actual: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual_keys = set(actual)
    missing = sorted(expected - actual_keys)
    extra = sorted(actual_keys - expected)
    if missing or extra:
        raise ContractError(f"{name} keys differ: missing={missing}, extra={extra}")


def _canonical_relative_path(value: Any, description: str) -> PurePosixPath:
    _require(isinstance(value, str) and bool(value), f"{description} path must be a string")
    _require("\\" not in value, f"{description} path must use canonical POSIX separators")
    path = PurePosixPath(value)
    _require(not path.is_absolute(), f"{description} path must be relative")
    _require(
        bool(path.parts) and all(part not in {"", ".", ".."} for part in path.parts),
        f"{description} path is not canonical: {value}",
    )
    _require(path.as_posix() == value, f"{description} path is not canonical: {value}")
    return path


def _resolve_repo_relative(value: Any, description: str) -> Path:
    relative = _canonical_relative_path(value, description)
    root = PROJECT_ROOT.resolve()
    resolved = (root / Path(*relative.parts)).resolve()
    _require(resolved.is_relative_to(root), f"{description} escapes the repository")
    return resolved


def _resolve_explicit_path(value: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _resolve_declared_inputs(
    manifest: Mapping[str, Any],
    *,
    data_dir: Path | None,
    feature_manifest: Path | None,
) -> tuple[Path, Path]:
    inputs = manifest["input_paths"]
    _require(isinstance(inputs, dict), "input_paths must be a mapping")
    _expect_exact_keys(
        inputs,
        frozenset({"raw_data_dir", "feature_cache_manifest"}),
        "input_paths",
    )
    declared_raw = _resolve_repo_relative(inputs["raw_data_dir"], "raw_data_dir")
    declared_feature = _resolve_repo_relative(
        inputs["feature_cache_manifest"], "feature_cache_manifest"
    )
    if data_dir is not None:
        _require(
            _resolve_explicit_path(data_dir) == declared_raw,
            "explicit data_dir differs from frozen input_paths",
        )
    if feature_manifest is not None:
        _require(
            _resolve_explicit_path(feature_manifest) == declared_feature,
            "explicit feature_manifest differs from frozen input_paths",
        )
    _require(declared_raw.is_dir(), f"raw data directory missing: {declared_raw}")
    _require(declared_feature.is_file(), f"feature-cache manifest missing: {declared_feature}")
    return declared_raw, declared_feature


def _resolve_artifacts(fold_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = manifest["artifacts"]
    _require(isinstance(artifacts, dict), "run manifest artifacts must be a mapping")
    _expect_exact_keys(artifacts, _EXACT_ARTIFACT_KEYS, "artifact mapping")

    for name, expected_path in CANONICAL_ARTIFACTS.items():
        entry = artifacts[name]
        _require(isinstance(entry, dict), f"artifact {name} entry must be a mapping")
        _expect_exact_keys(entry, frozenset({"path", "sha256"}), f"artifact {name}")
        _require(entry["path"] == expected_path, f"artifact {name} must use {expected_path}")

    model_entry = artifacts["feature_model"]
    metadata_entry = artifacts["feature_model_metadata"]
    for name, entry in (
        ("feature_model", model_entry),
        ("feature_model_metadata", metadata_entry),
    ):
        _require(isinstance(entry, dict), f"artifact {name} entry must be a mapping")
        _expect_exact_keys(entry, frozenset({"path", "sha256"}), f"artifact {name}")

    match = _FEATURE_MODEL_RE.fullmatch(str(model_entry["path"]))
    _require(match is not None, "feature_model path must be content-addressed")
    expected_metadata_path = str(model_entry["path"])[: -len(".joblib")] + ".json"
    _require(
        metadata_entry["path"] == expected_metadata_path,
        "feature_model_metadata sidecar path must share the feature-model cache key",
    )

    root = fold_dir.resolve()
    resolved: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for name in sorted(_EXACT_ARTIFACT_KEYS):
        entry = artifacts[name]
        relative = _canonical_relative_path(entry["path"], f"artifact {name}")
        digest = entry["sha256"]
        _require(
            isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
            f"artifact {name} sha256 must be a lowercase 64-hex digest",
        )
        path = (root / Path(*relative.parts)).resolve()
        _require(path.is_relative_to(root), f"artifact {name} escapes the fold directory")
        _require(path not in seen_paths, f"artifact {name} aliases another artifact path")
        _require(path.is_file(), f"artifact {name} missing: {path}")
        _require(sha256_file(path) == digest, f"artifact {name} sha256 mismatch")
        resolved[name] = path
        seen_paths.add(path)
    return resolved


def _subject_from_record(path: Path) -> str:
    match = _RECORD_RE.fullmatch(path.stem)
    if match is None:
        raise ContractError(f"unexpected Sleep-EDF record filename: {path.name}")
    return normalize_subject_id(match.group("subject"))


def _hash_named_files(paths: Sequence[Path]) -> str:
    unique = sorted({path.resolve() for path in paths}, key=lambda item: item.name)
    _require(bool(unique), "no files supplied for aggregate hashing")
    names = [path.name for path in unique]
    _require(len(names) == len(set(names)), "aggregate hash contains duplicate basenames")
    return canonical_json_hash(
        [{"name": path.name, "sha256": sha256_file(path)} for path in unique]
    )


def _read_raw_contract(
    data_dir: Path,
    outer_subject: str,
) -> tuple[tuple[str, ...], str, tuple[RowIdentity, ...], np.ndarray]:
    raw_paths = sorted(data_dir.glob("SC*.npz"), key=lambda item: item.name)
    _require(bool(raw_paths), f"no SC*.npz files found in raw data directory: {data_dir}")
    subjects_by_file = {path: _subject_from_record(path) for path in raw_paths}
    subjects = tuple(sorted(set(subjects_by_file.values())))
    outer_paths = [path for path in raw_paths if subjects_by_file[path] == outer_subject]
    _require(bool(outer_paths), f"no raw records found for outer subject {outer_subject}")

    identities: list[RowIdentity] = []
    labels: list[np.ndarray] = []
    for path in outer_paths:
        try:
            with np.load(path, allow_pickle=False) as payload:
                _require("x" in payload.files and "y" in payload.files, f"raw file lacks x/y: {path}")
                x = np.asarray(payload["x"])
                y = np.asarray(payload["y"])
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"cannot safely read raw file {path}: {exc}") from exc
        _require(x.ndim >= 1, f"raw x must have a row dimension: {path}")
        _require(y.ndim == 1, f"raw y must be one-dimensional: {path}")
        _require(x.shape[0] == y.shape[0], f"raw x/y row count differs: {path}")
        _require(np.issubdtype(y.dtype, np.integer), f"raw y must use an integer dtype: {path}")
        identities.extend(
            RowIdentity(
                dataset=DATASET_ID,
                subject_id=outer_subject,
                record_id=path.stem,
                epoch_index=epoch_index,
                outer_fold=outer_subject,
            )
            for epoch_index in range(y.shape[0])
        )
        labels.append(y.astype(np.int64, copy=False))

    expected_ids = tuple(identities)
    _require(len(set(expected_ids)) == len(expected_ids), "raw expected epoch identities are duplicate")
    return subjects, _hash_named_files(raw_paths), expected_ids, np.concatenate(labels)


def _validate_manifest_shape_and_hash(manifest: Mapping[str, Any]) -> str:
    _expect_exact_keys(manifest, _FINAL_MANIFEST_KEYS, "final run manifest")
    _require(manifest["schema_version"] == RUN_SCHEMA_VERSION, "run schema_version mismatch")
    _require(manifest["dataset"] == DATASET_ID, "run dataset mismatch")
    outer_subject = normalize_subject_id(str(manifest["outer_subject_id"]))
    _require(manifest["outer_subject_id"] == outer_subject, "outer_subject_id is not normalized")
    _require(manifest["outer_fold"] == outer_subject, "outer_fold must equal outer_subject_id")
    _require(manifest["tau_h"] == FIXED_TAU_H, "tau_h is not the frozen value 0.9")
    _require(manifest["kappa"] == FIXED_KAPPA, "kappa is not the frozen value 0.15")
    _require(
        manifest["kl_epsilon"] == FIXED_KL_EPSILON,
        "kl_epsilon is not the frozen value 1e-10",
    )
    _require(manifest["primary_checkpoint_frozen"] is True, "primary checkpoint is not frozen")
    _require(manifest["feature_model_frozen"] is True, "feature model is not frozen")
    _require(
        type(manifest["outer_evaluation_count"]) is int
        and manifest["outer_evaluation_count"] == 1,
        "outer subject was not evaluated exactly once",
    )
    _require(manifest["outer_evaluation_completed"] is True, "outer evaluation is incomplete")
    _require(
        manifest["status"] == "COMPLETE_SINGLE_OUTER_FOLD",
        "run status is not COMPLETE_SINGLE_OUTER_FOLD",
    )
    _require(type(manifest["seed"]) is int, "seed must be an integer")
    _require(
        type(manifest["metric_min_support"]) is int and manifest["metric_min_support"] > 0,
        "metric_min_support must be a positive integer",
    )
    _require(
        type(manifest["selected_epoch"]) is int and manifest["selected_epoch"] > 0,
        "selected_epoch must be a positive integer",
    )
    _require(
        type(manifest["primary_training_epochs"]) is int
        and manifest["primary_training_epochs"] > 0,
        "primary_training_epochs must be a positive integer",
    )
    _require(
        isinstance(manifest["runtime_seconds"], (int, float))
        and not isinstance(manifest["runtime_seconds"], bool)
        and math.isfinite(float(manifest["runtime_seconds"]))
        and float(manifest["runtime_seconds"]) >= 0.0,
        "runtime_seconds must be finite and non-negative",
    )
    _require(
        type(manifest["peak_gpu_memory_bytes"]) is int
        and manifest["peak_gpu_memory_bytes"] >= 0,
        "peak_gpu_memory_bytes must be a non-negative integer",
    )
    for name in (
        "raw_data_hash",
        "feature_cache_hash",
        "preprocessing_hash",
        "feature_preprocessing_hash",
        "schema_hash",
        "feature_schema_hash",
        "feature_extractor_hash",
        "code_hash",
        "run_config_hash",
    ):
        _require(
            isinstance(manifest[name], str) and _SHA256_RE.fullmatch(manifest[name]) is not None,
            f"{name} must be a lowercase SHA-256 digest",
        )

    run_config = {field: manifest[field] for field in RUN_CONFIG_FIELDS}
    _require(isinstance(manifest["run_config"], dict), "nested run_config must be a mapping")
    _expect_exact_keys(manifest["run_config"], frozenset(RUN_CONFIG_FIELDS), "nested run_config")
    _require(
        manifest["run_config"] == run_config,
        "nested run_config differs from flattened frozen fields",
    )
    _require(
        manifest["run_config_hash"] == canonical_json_hash(run_config),
        "run_config_hash does not match the canonical nested run_config",
    )

    primary_config = manifest["primary_config"]
    _require(isinstance(primary_config, dict), "primary_config must be a mapping")
    _expect_exact_keys(primary_config, _PRIMARY_CONFIG_KEYS, "primary_config")
    _require(
        primary_config["architecture"] == "Conformer"
        and primary_config["emb_size"] == 40
        and primary_config["depth"] == 6
        and primary_config["classes"] == 5
        and primary_config["electrodes"] == 1
        and primary_config["class_weights"] is True
        and primary_config["seed"] == manifest["seed"],
        "primary_config differs from the frozen Stage-1 architecture",
    )
    for name in ("epochs", "batch_size", "early_stop_patience"):
        _require(type(primary_config[name]) is int and primary_config[name] > 0, f"{name} is invalid")
    _require(
        isinstance(primary_config["learning_rate"], (int, float))
        and not isinstance(primary_config["learning_rate"], bool)
        and math.isfinite(float(primary_config["learning_rate"]))
        and float(primary_config["learning_rate"]) > 0.0,
        "learning_rate is invalid",
    )
    _require(
        isinstance(manifest["feature_smote_ratio"], (int, float))
        and not isinstance(manifest["feature_smote_ratio"], bool)
        and 0.0 <= float(manifest["feature_smote_ratio"]) <= 1.0,
        "feature_smote_ratio must lie in [0, 1]",
    )

    environment = manifest["environment"]
    _require(isinstance(environment, dict), "environment must be a mapping")
    _expect_exact_keys(environment, _ENVIRONMENT_KEYS, "environment")
    _require(
        environment["selected_device"] in {"cpu", "cuda"},
        "environment selected_device must be cpu or cuda",
    )
    _require(environment["deterministic_algorithms"] is True, "deterministic algorithms not enabled")
    _require(environment["tf32"] is False, "TF32 must be disabled")
    return outer_subject


def _primary_preprocessing_hash() -> str:
    return canonical_json_hash(
        {
            "input": "frozen_preprocessed_sleepedf_npz",
            "runtime_normalization": "none",
            "sample_shape": [1, 1, 3000],
            "data_loader_hash": sha256_file(PROJECT_ROOT / "sleepedf" / "data_loader.py"),
            "upstream_preprocessing_provenance": "unresolved_legacy_input",
        }
    )


def _campaign_code_hash() -> str:
    return _hash_named_files(
        [
            PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "run_outer_fold.py",
            PROJECT_ROOT / "conformer.py",
            PROJECT_ROOT / "sleepedf" / "data_loader.py",
            PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "contracts.py",
            PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "policy.py",
            PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "metrics.py",
            PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "splits.py",
            PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "feature_cache.py",
        ]
    )


def _validate_inputs_and_split(
    manifest: Mapping[str, Any],
    *,
    outer_subject: str,
    raw_dir: Path,
    feature_manifest_path: Path,
) -> tuple[SubjectSplit, tuple[RowIdentity, ...], np.ndarray, Mapping[str, Any]]:
    subjects, raw_hash, raw_ids, raw_labels = _read_raw_contract(raw_dir, outer_subject)
    _require(manifest["raw_data_hash"] == raw_hash, "raw_data_hash differs from raw files")

    try:
        cache_summary = validate_feature_cache(feature_manifest_path)
        raw_binding = validate_feature_cache_against_raw(
            feature_manifest_path,
            raw_dir,
        )
        _, feature_labels, feature_ids, _ = load_feature_rows(
            feature_manifest_path,
            (outer_subject,),
            outer_fold=outer_subject,
        )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"clean feature-cache validation failed: {exc}") from exc
    feature_payload = _load_json_mapping(feature_manifest_path, "clean feature-cache manifest")
    _require(cache_summary.get("identity_and_cache_contract") == "PASS", "feature cache did not pass")
    _require(raw_binding.get("raw_binding") == "PASS", "feature cache/raw binding did not pass")
    _require(feature_payload.get("dataset") == DATASET_ID, "feature-cache dataset mismatch")
    _require(tuple(feature_payload.get("subjects", [])) == subjects, "feature-cache cohort mismatch")
    _require(manifest["feature_cache_hash"] == sha256_file(feature_manifest_path), "feature_cache_hash mismatch")
    _require(
        manifest["feature_schema_hash"] == FEATURE_CACHE_SCHEMA_HASH
        and feature_payload.get("schema_hash") == FEATURE_CACHE_SCHEMA_HASH,
        "feature_schema_hash mismatch",
    )
    _require(
        manifest["feature_extractor_hash"] == EXTRACTOR_HASH
        and feature_payload.get("extractor_hash") == EXTRACTOR_HASH,
        "feature_extractor_hash mismatch",
    )
    expected_feature_preprocessing = canonical_json_hash(
        {
            "manifest_sha256": sha256_file(feature_manifest_path),
            "extractor_hash": feature_payload["extractor_hash"],
            "extractor_config_hash": feature_payload["config_hash"],
            "extractor_code_hash": feature_payload["code_hash"],
        }
    )
    _require(
        manifest["feature_preprocessing_hash"] == expected_feature_preprocessing,
        "feature_preprocessing_hash mismatch",
    )
    _require(manifest["preprocessing_hash"] == _primary_preprocessing_hash(), "preprocessing_hash mismatch")
    _require(manifest["schema_hash"] == PREDICTION_SCHEMA_HASH, "prediction schema_hash mismatch")
    _require(manifest["code_hash"] == _campaign_code_hash(), "campaign code_hash mismatch")

    try:
        align_one_to_one(
            feature_ids,
            raw_ids,
            reference_labels=feature_labels,
            candidate_labels=raw_labels,
        )
    except ContractError as exc:
        raise ContractError(f"clean feature rows do not join one-to-one to raw epochs: {exc}") from exc

    split_payload = manifest["subject_split"]
    _require(isinstance(split_payload, dict), "subject_split must be a mapping")
    declared_subjects = tuple(str(item) for item in split_payload.get("all_subjects", []))
    _require(declared_subjects == subjects, "subject_split cohort differs from raw files")
    split = build_subject_split(outer_subject, subjects, seed=manifest["seed"])
    _require(split.as_dict() == split_payload, "subject_split does not match deterministic assignment")
    _require(manifest["assignment_rule"] == split.assignment_rule, "assignment_rule mismatch")
    return split, feature_ids, feature_labels, feature_payload


def _load_torch_mapping(path: Path, description: str) -> Mapping[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ContractError(f"cannot load {description} {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{description} payload must be a mapping")
    return payload


def _finite_number(value: Any, description: str, *, nonnegative: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{description} must be numeric",
    )
    converted = float(value)
    _require(math.isfinite(converted), f"{description} must be finite")
    if nonnegative:
        _require(converted >= 0.0, f"{description} must be non-negative")
    return converted


def _validate_primary_training(
    best_path: Path,
    recovery_path: Path,
    history_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    best = _load_torch_mapping(best_path, "selected primary checkpoint")
    _expect_exact_keys(best, _BEST_CHECKPOINT_KEYS, "selected primary checkpoint")
    _require(best["schema_version"] == RUN_SCHEMA_VERSION, "checkpoint schema_version mismatch")
    _require(best["run_config_hash"] == manifest["run_config_hash"], "checkpoint run_config_hash mismatch")
    _require(best["epoch"] == manifest["selected_epoch"], "checkpoint epoch differs from selected_epoch")
    _require(best["selection_rule"] == manifest["checkpoint_selection_rule"], "checkpoint selection_rule mismatch")
    _require(isinstance(best["model_state_dict"], Mapping), "checkpoint model_state_dict is invalid")
    for checkpoint_key, manifest_key in (
        ("selection_accuracy", "selection_accuracy"),
        ("selection_loss", "selection_loss"),
    ):
        _require(
            math.isclose(
                _finite_number(best[checkpoint_key], f"checkpoint {checkpoint_key}"),
                _finite_number(manifest[manifest_key], f"manifest {manifest_key}"),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"checkpoint {checkpoint_key} mismatch",
        )

    history = _load_json(history_path, "primary training history")
    _require(isinstance(history, list) and bool(history), "primary training history must be a non-empty list")
    _require(len(history) == manifest["primary_training_epochs"], "primary_training_epochs mismatch")
    best_accuracy = -math.inf
    best_loss = math.inf
    best_epoch = 0
    for index, entry in enumerate(history, start=1):
        _require(isinstance(entry, dict), f"history entry {index} must be a mapping")
        _expect_exact_keys(entry, _HISTORY_ENTRY_KEYS, f"history entry {index}")
        _require(type(entry["epoch"]) is int and entry["epoch"] == index, "history epochs are not contiguous")
        train_loss = _finite_number(entry["train_loss"], "train_loss", nonnegative=True)
        train_accuracy = _finite_number(entry["train_accuracy"], "train_accuracy")
        selection_loss = _finite_number(entry["selection_loss"], "selection_loss", nonnegative=True)
        selection_accuracy = _finite_number(entry["selection_accuracy"], "selection_accuracy")
        _finite_number(entry["seconds"], "history seconds", nonnegative=True)
        _require(0.0 <= train_accuracy <= 1.0, "train_accuracy must lie in [0, 1]")
        _require(0.0 <= selection_accuracy <= 1.0, "selection_accuracy must lie in [0, 1]")
        expected_improved = selection_accuracy > best_accuracy
        _require(type(entry["improved"]) is bool, "history improved must be boolean")
        _require(entry["improved"] is expected_improved, "history improved flags violate earliest-tie rule")
        if expected_improved:
            best_accuracy = selection_accuracy
            best_loss = selection_loss
            best_epoch = index
        _ = train_loss
    _require(best_epoch == manifest["selected_epoch"], "history selected epoch mismatch")
    _require(math.isclose(best_accuracy, float(manifest["selection_accuracy"]), rel_tol=0.0, abs_tol=1e-12), "history best accuracy mismatch")
    _require(math.isclose(best_loss, float(manifest["selection_loss"]), rel_tol=0.0, abs_tol=1e-12), "history best loss mismatch")
    _require(len(history) <= manifest["primary_config"]["epochs"], "history exceeds configured epochs")

    recovery = _load_torch_mapping(recovery_path, "primary recovery checkpoint")
    _expect_exact_keys(recovery, _RECOVERY_CHECKPOINT_KEYS, "primary recovery checkpoint")
    _require(recovery["schema_version"] == RUN_SCHEMA_VERSION, "recovery schema_version mismatch")
    _require(recovery["run_config_hash"] == manifest["run_config_hash"], "recovery run_config_hash mismatch")
    _require(recovery["epoch"] == len(history), "recovery epoch differs from history")
    _require(recovery["history"] == history, "recovery embedded history differs from history artifact")
    _require(recovery["best_epoch"] == best_epoch, "recovery best_epoch mismatch")
    _require(math.isclose(float(recovery["best_accuracy"]), best_accuracy, rel_tol=0.0, abs_tol=1e-12), "recovery best_accuracy mismatch")
    _require(math.isclose(float(recovery["best_loss"]), best_loss, rel_tol=0.0, abs_tol=1e-12), "recovery best_loss mismatch")
    _require(recovery["without_improvement"] == len(history) - best_epoch, "recovery patience state mismatch")
    for name in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state"):
        _require(isinstance(recovery[name], Mapping), f"recovery {name} is invalid")
    rng_keys = set(recovery["rng_state"])
    _require(
        {"python", "numpy", "torch", "loader_generator"} <= rng_keys
        and rng_keys <= {"python", "numpy", "torch", "loader_generator", "cuda"},
        "recovery rng_state fields are invalid",
    )


def _normalized_subject_tuple(value: Any, description: str) -> tuple[str, ...]:
    _require(isinstance(value, list), f"{description} must be a JSON list")
    subjects = tuple(normalize_subject_id(str(item)) for item in value)
    _require(len(subjects) == len(set(subjects)), f"{description} contains duplicates")
    return tuple(sorted(subjects))


def _validate_feature_model(
    model_path: Path,
    sidecar_path: Path,
    manifest: Mapping[str, Any],
    split: SubjectSplit,
    feature_payload: Mapping[str, Any],
) -> dict[str, Any]:
    sidecar = _load_json_mapping(sidecar_path, "feature-model metadata sidecar")
    _expect_exact_keys(sidecar, _FEATURE_METADATA_KEYS, "feature-model metadata")
    try:
        payload = joblib.load(model_path)
    except Exception as exc:
        raise ContractError(f"cannot load feature model {model_path}: {exc}") from exc
    _require(isinstance(payload, dict), "feature-model joblib payload must be a mapping")
    _expect_exact_keys(payload, frozenset({"model", "metadata"}), "feature-model joblib payload")
    _require(isinstance(payload["metadata"], dict), "embedded feature metadata must be a mapping")
    _require(payload["metadata"] == sidecar, "embedded feature metadata differs from sidecar")

    expected_feature_config = {
        "expert": "random_forest_isotonic",
        "feature_names": feature_payload["feature_names"],
        "standard_scaler": True,
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": manifest["seed"],
            "n_jobs": -1,
        },
        "smote_ratio": manifest["feature_smote_ratio"],
        "smote_scope": "inner_train_only",
        "calibration": "isotonic_on_disjoint_subjects",
        "feature_cache_schema_hash": manifest["feature_schema_hash"],
        "feature_extractor_hash": manifest["feature_extractor_hash"],
    }
    _require(sidecar["feature_config"] == expected_feature_config, "feature_config mismatch")
    cache_key = model_path.stem
    _require(_SHA256_RE.fullmatch(cache_key) is not None, "feature-model filename lacks a cache key")
    _require(sidecar["schema_version"] == RUN_SCHEMA_VERSION, "feature metadata schema_version mismatch")
    _require(sidecar["cache_key"] == cache_key, "feature metadata cache_key differs from filename")
    _require(sidecar["split_hash"] == split.split_hash, "feature metadata split_hash mismatch")
    _require(sidecar["schema_hash"] == manifest["feature_schema_hash"], "feature metadata schema_hash mismatch")
    _require(sidecar["data_hash"] == manifest["feature_cache_hash"], "feature metadata data_hash mismatch")
    _require(sidecar["extractor_hash"] == manifest["feature_extractor_hash"], "feature metadata extractor_hash mismatch")
    _require(sidecar["seed"] == manifest["seed"], "feature metadata seed mismatch")
    _require(
        sidecar["model_hash"] == canonical_json_hash(expected_feature_config),
        "feature metadata model_hash mismatch",
    )
    _require(sidecar["config_hash"] == sidecar["model_hash"], "feature metadata config_hash mismatch")
    expected_calibration_hash = canonical_json_hash(
        {"method": "isotonic", "subjects": list(split.calibration), "smote": False}
    )
    _require(sidecar["calibration_hash"] == expected_calibration_hash, "feature metadata calibration_hash mismatch")
    _require(
        _normalized_subject_tuple(sidecar["training_subjects"], "feature training_subjects")
        == tuple(sorted(split.inner_train)),
        "feature metadata training subjects differ from inner_train",
    )
    _require(
        _normalized_subject_tuple(sidecar["calibration_subjects"], "feature calibration_subjects")
        == tuple(sorted(split.calibration)),
        "feature metadata calibration subjects differ from calibration split",
    )
    expected_cache_key = build_cache_key(
        split_hash=sidecar["split_hash"],
        schema_hash=sidecar["schema_hash"],
        model_hash=sidecar["model_hash"],
        data_hash=sidecar["data_hash"],
        config_hash=sidecar["config_hash"],
        calibration_hash=sidecar["calibration_hash"],
        seed=sidecar["seed"],
    )
    _require(expected_cache_key == cache_key, "feature model cache key does not bind its metadata")
    return sidecar


def _validate_bundle_role(
    bundle: PredictionBundle,
    *,
    role: str,
    manifest: Mapping[str, Any],
    split: SubjectSplit,
    checkpoint_hash: str,
    feature_metadata: Mapping[str, Any] | None = None,
) -> None:
    bundle_manifest = bundle.manifest
    common_expected = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "schema_hash": manifest["schema_hash"],
        "dataset": DATASET_ID,
        "outer_fold": split.outer_subject,
        "outer_subject_id": split.outer_subject,
        "split_hash": split.split_hash,
        "code_hash": manifest["code_hash"],
        "seed": manifest["seed"],
        "checkpoint_hash": checkpoint_hash,
    }
    for field, expected in common_expected.items():
        _require(getattr(bundle_manifest, field) == expected, f"{role} prediction manifest {field} mismatch")
    _require(
        bundle_manifest.training_subjects == tuple(sorted(split.inner_train)),
        f"{role} training subjects differ from inner_train",
    )

    if role == "primary":
        expected_model_hash = canonical_json_hash(manifest["primary_config"])
        role_expected = {
            "model_id": "eeg_conformer_primary",
            "model_hash": expected_model_hash,
            "config_hash": expected_model_hash,
            "calibration_hash": canonical_json_hash({"method": "none"}),
            "data_hash": manifest["raw_data_hash"],
            "preprocessing_hash": manifest["preprocessing_hash"],
        }
        _require(bundle_manifest.calibration_subjects == (), "primary calibration_subjects must be empty")
    elif role == "feature":
        _require(feature_metadata is not None, "feature metadata is required")
        role_expected = {
            "model_id": "rf_isotonic_feature_expert",
            "model_hash": feature_metadata["model_hash"],
            "config_hash": feature_metadata["config_hash"],
            "calibration_hash": feature_metadata["calibration_hash"],
            "data_hash": manifest["feature_cache_hash"],
            "preprocessing_hash": manifest["feature_preprocessing_hash"],
        }
        _require(
            bundle_manifest.calibration_subjects == tuple(sorted(split.calibration)),
            "feature calibration subjects differ from calibration split",
        )
    else:  # pragma: no cover - internal programming error.
        raise ContractError(f"unknown prediction role: {role}")
    for field, expected in role_expected.items():
        _require(getattr(bundle_manifest, field) == expected, f"{role} prediction manifest {field} mismatch")

    for row_number, identity in enumerate(bundle.identities):
        _require(identity.dataset == DATASET_ID, f"{role} row {row_number} dataset mismatch")
        _require(identity.outer_fold == split.outer_subject, f"{role} row {row_number} fold mismatch")
        _require(
            normalize_subject_id(identity.subject_id) == split.outer_subject,
            f"{role} row {row_number} is not from the outer subject",
        )


def _align_bundle_to_expected(
    bundle: PredictionBundle,
    expected_ids: tuple[RowIdentity, ...],
    expected_labels: np.ndarray,
    role: str,
) -> np.ndarray:
    try:
        return align_one_to_one(
            expected_ids,
            bundle.identities,
            reference_labels=expected_labels,
            candidate_labels=bundle.labels,
        )
    except ContractError as exc:
        raise ContractError(f"{role} predictions do not join one-to-one to expected epochs: {exc}") from exc


def json_value(value: Any) -> Any:
    """Convert metric structures to the exact persisted JSON representation."""

    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {name: json_value(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def verify_fold(
    fold_dir: Path,
    *,
    min_support: int | None = None,
    data_dir: Path | None = None,
    feature_manifest: Path | None = None,
) -> dict[str, Any]:
    """Verify one completed fold without mutating it or replaying either model."""

    fold_dir = Path(fold_dir).resolve()
    run_manifest_path = fold_dir / "run_manifest.json"
    _require(run_manifest_path.is_file(), f"run manifest missing: {run_manifest_path}")
    manifest = _load_json_mapping(run_manifest_path, "run manifest")
    outer_subject = _validate_manifest_shape_and_hash(manifest)
    raw_dir, feature_manifest_path = _resolve_declared_inputs(
        manifest,
        data_dir=data_dir,
        feature_manifest=feature_manifest,
    )

    frozen_min_support = manifest["metric_min_support"]
    if min_support is not None:
        _require(type(min_support) is int and min_support > 0, "min_support must be positive")
        _require(min_support == frozen_min_support, "min_support override differs from frozen run_config")

    split, expected_ids, expected_labels, feature_payload = _validate_inputs_and_split(
        manifest,
        outer_subject=outer_subject,
        raw_dir=raw_dir,
        feature_manifest_path=feature_manifest_path,
    )
    artifacts = _resolve_artifacts(fold_dir, manifest)
    _validate_primary_training(
        artifacts["primary_checkpoint"],
        artifacts["primary_recovery_checkpoint"],
        artifacts["primary_training_history"],
        manifest,
    )
    feature_metadata = _validate_feature_model(
        artifacts["feature_model"],
        artifacts["feature_model_metadata"],
        manifest,
        split,
        feature_payload,
    )

    primary = PredictionBundle.from_npz(artifacts["primary_predictions"])
    feature = PredictionBundle.from_npz(artifacts["feature_predictions"])
    _validate_bundle_role(
        primary,
        role="primary",
        manifest=manifest,
        split=split,
        checkpoint_hash=sha256_file(artifacts["primary_checkpoint"]),
    )
    _validate_bundle_role(
        feature,
        role="feature",
        manifest=manifest,
        split=split,
        checkpoint_hash=sha256_file(artifacts["feature_model"]),
        feature_metadata=feature_metadata,
    )
    primary_order = _align_bundle_to_expected(primary, expected_ids, expected_labels, "primary")
    feature_order = _align_bundle_to_expected(feature, expected_ids, expected_labels, "feature")

    primary_probs = primary.probabilities[primary_order]
    primary_pred = primary.predictions[primary_order]
    primary_confidences = primary.confidences[primary_order]
    feature_probs = feature.probabilities[feature_order]
    feature_pred = feature.predictions[feature_order]
    routing = route_type1_forward_kl(
        probabilities_t=primary_probs,
        probabilities_f=feature_probs,
        predictions_t=primary_pred,
        predictions_f=feature_pred,
        confidence_t=primary_confidences,
        tau_h=FIXED_TAU_H,
        kappa=FIXED_KAPPA,
        epsilon=FIXED_KL_EPSILON,
    )
    recomputed = evaluate_policy_metrics(expected_labels, routing, min_support=frozen_min_support)
    recomputed_payload = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "policy": routing.policy_name,
        "tau_h": FIXED_TAU_H,
        "kappa": FIXED_KAPPA,
        "kl_epsilon": FIXED_KL_EPSILON,
        "min_support": frozen_min_support,
        "outer_subject": split.outer_subject,
        "metrics": json_value(recomputed),
    }
    stored_payload = _load_json_mapping(
        artifacts["metrics_fixed_kappa_015"], "stored metric payload"
    )
    _expect_exact_keys(
        stored_payload,
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
        "stored metric payload",
    )
    _require(stored_payload["min_support"] == frozen_min_support, "stored metric min_support mismatch")
    _require(stored_payload == recomputed_payload, "stored metrics differ from strict recomputation")

    return {
        "status": "PASS_ARTIFACT_CONTRACTS_ONLY",
        "fold_dir": str(fold_dir),
        "outer_subject": split.outer_subject,
        "run_config_hash": manifest["run_config_hash"],
        "row_count": len(expected_ids),
        "raw_to_clean_feature_epoch_join": "PASS",
        "clean_feature_cache_validation": "PASS",
        "artifact_contracts": "PASS",
        "primary_recovery_state": "PASS",
        "metric_recomputation": "EXACT_MATCH",
        "metric_artifact_sha256": sha256_file(artifacts["metrics_fixed_kappa_015"]),
        "source_binding_replay": "NOT_RUN",
        "confirmatory_eligible_at_artifact_level": False,
        "limitations": [
            "Prediction arrays were not deterministically replayed from the primary checkpoint.",
            "Prediction arrays were not deterministically replayed from the feature-model joblib.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--feature-manifest", type=Path, default=None)
    parser.add_argument(
        "--min-support",
        type=int,
        default=None,
        help="Optional assertion; it must equal the value frozen in run_config_hash.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_fold(
        args.fold_dir,
        min_support=args.min_support,
        data_dir=args.data_dir,
        feature_manifest=args.feature_manifest,
    )
    if args.output_json is not None:
        atomic_write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
