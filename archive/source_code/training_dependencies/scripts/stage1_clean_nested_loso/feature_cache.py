"""Identity-bearing, pickle-free cache for the frozen historical 35-D features.

The extractor in this module is a line-for-line semantic port of
``Paper1_feature_expert_scripts/feature_extraction_fusion.py`` at commit
``4e6b9b4``.  Raw Sleep-EDF record NPZs are the only data source.  In
particular, the legacy object/pickle based ``fusion_features_cache.npz`` is
never opened by this module.

The cache is deliberately independent of an outer fold.  A caller supplies the
outer-fold identifier only when loading rows, at which point complete
``RowIdentity`` values are constructed without changing the persisted
dataset/subject/record/epoch identity.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import pywt
import scipy
from scipy.signal import welch
from scipy.stats import kurtosis, skew

try:  # Historical behavior: entropy values are zero when antropy is absent.
    import antropy as ant

    HAS_ANTROPY = True
except ImportError:  # pragma: no cover - availability is environment-specific.
    ant = None
    HAS_ANTROPY = False

try:  # Support both module and direct-script import modes.
    from .contracts import (
        ContractError,
        RowIdentity,
        atomic_write_json,
        atomic_write_npz,
        canonical_json_bytes,
        canonical_json_hash,
        sha256_file,
    )
except ImportError:  # pragma: no cover - direct script imports are not test mode.
    from contracts import (  # type: ignore[no-redef]
        ContractError,
        RowIdentity,
        atomic_write_json,
        atomic_write_npz,
        canonical_json_bytes,
        canonical_json_hash,
        sha256_file,
    )


FEATURE_CACHE_SCHEMA_VERSION = "stage1-clean-feature-cache-v1"
HISTORICAL_SOURCE_COMMIT = "4e6b9b4b40b7c747db18c4792244e6289a100e93"
HISTORICAL_SOURCE_PATH = (
    "Paper1_feature_expert_scripts/feature_extraction_fusion.py"
)
HISTORICAL_SOURCE_GIT_BLOB = "08fe7c17ae57b690d7a0c0eb93daa7ff5c28f292"
HISTORICAL_SOURCE_SHA256 = (
    "2b1e444efc5e8ec6a507b24986b1f98c0d20d622eb98c1da7ce00c0907005112"
)

RAW_RECORD_PATTERN = re.compile(
    r"^SC4(?P<subject>[0-9]{2})(?P<night>[0-9]+)E0[.]npz$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

YASA_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sdelta", 0.4, 1.0),
    ("fdelta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),
    ("beta", 16.0, 30.0),
)

FEATURE_NAMES: tuple[str, ...] = (
    "std",
    "iqr",
    "skew",
    "kurt",
    "nzc",
    "hmob",
    "hcomp",
    "mean",
    "rms",
    "sample_entropy",
    "perm_entropy",
    "sdelta",
    "fdelta",
    "theta",
    "alpha",
    "sigma",
    "beta",
    "abspow",
    "sdelta_rel",
    "fdelta_rel",
    "theta_rel",
    "alpha_rel",
    "sigma_rel",
    "beta_rel",
    "dt",
    "ds",
    "db",
    "at",
    "ts",
    "spectral_centroid",
    "spectral_entropy",
    "spectral_edge_95",
    "spectral_slope",
    "dwt_D4_energy",
    "dwt_D5_energy",
)

EXTRACTOR_SPEC: dict[str, Any] = {
    "name": "fusion_features_35d_at_4e6b9b4",
    "source_commit": HISTORICAL_SOURCE_COMMIT,
    "source_path": HISTORICAL_SOURCE_PATH,
    "source_git_blob": HISTORICAL_SOURCE_GIT_BLOB,
    "source_sha256": HISTORICAL_SOURCE_SHA256,
    "input_cast": "squeeze_then_float64",
    "output_dtype": "float32",
    "welch": {"window_seconds": 4.0, "overlap_fraction": 0.5},
    "bands_hz_inclusive": [
        {"name": name, "low": low, "high": high}
        for name, low, high in YASA_BANDS
    ],
    "relative_and_ratio_epsilon": 1e-10,
    "spectral_range_hz_inclusive": [0.4, 30.0],
    "spectral_edge_quantile": 0.95,
    "dwt": {"wavelet": "db4", "level": 5, "energies": ["D4", "D5"]},
    "entropy": "antropy_if_importable_else_zero; per-feature exceptions become zero",
    "feature_names": list(FEATURE_NAMES),
}
EXTRACTOR_HASH = canonical_json_hash(EXTRACTOR_SPEC)

FEATURE_CACHE_SCHEMA: dict[str, Any] = {
    "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
    "identity_fields": ["dataset", "subject_id", "record_id", "epoch_index"],
    "label_is_join_validation_not_identity_key": True,
    "feature_count": len(FEATURE_NAMES),
    "feature_dtype": "float32",
    "label_dtype": "int64",
    "sharding": "one_npz_per_subject",
    "serialization": "numpy_npz_allow_pickle_false",
    "manifest": "canonical_json_with_payload_hash",
}
FEATURE_CACHE_SCHEMA_HASH = canonical_json_hash(FEATURE_CACHE_SCHEMA)

SHARD_KEYS = frozenset(
    {
        "features",
        "labels",
        "datasets",
        "subject_ids",
        "record_ids",
        "epoch_indices",
        "feature_names",
        "schema_hash",
        "extractor_hash",
        "config_hash",
        "code_hash",
        "subject_data_hash",
        "raw_sources_json",
        "row_hash",
    }
)
MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "schema",
        "schema_hash",
        "dataset",
        "extractor",
        "extractor_hash",
        "config",
        "config_hash",
        "code_hash",
        "data_hash",
        "raw_sources",
        "feature_names",
        "subjects",
        "total_rows",
        "shards",
        "cache_key",
        "manifest_hash",
    }
)
RAW_SOURCE_KEYS = frozenset(
    {
        "filename",
        "subject_id",
        "record_id",
        "row_count",
        "sampling_frequency",
        "label_hash",
        "sha256",
    }
)
SHARD_MANIFEST_KEYS = frozenset(
    {
        "subject_id",
        "path",
        "sha256",
        "row_count",
        "row_hash",
        "subject_data_hash",
        "raw_files",
    }
)


def _required_text(value: Any, field: str) -> str:
    if value is None or isinstance(value, bool):
        raise ContractError(f"{field} must be a non-empty string")
    result = str(value).strip()
    if not result:
        raise ContractError(f"{field} must be a non-empty string")
    return result


def _require_sha256(value: Any, field: str) -> str:
    result = _required_text(value, field).lower()
    if SHA256_PATTERN.fullmatch(result) is None:
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _normalize_subject(value: Any) -> str:
    result = _required_text(value, "subject_id")
    if result.isdecimal():
        result = f"{int(result):02d}"
    if re.fullmatch(r"[0-9]{2}", result) is None:
        raise ContractError(f"invalid Sleep-EDF subject_id: {result!r}")
    return result


def _normalize_outer_fold(value: Any) -> str:
    result = _required_text(value, "outer_fold")
    if result.isdecimal():
        result = f"{int(result):02d}"
    return result


def _package_version(package: str, module: Any | None = None) -> str:
    if module is not None:
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def _runtime_config() -> dict[str, Any]:
    return {
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
        "pywavelets_version": _package_version("PyWavelets", pywt),
        "antropy_available": HAS_ANTROPY,
        "antropy_version": _package_version("antropy", ant) if HAS_ANTROPY else "absent",
        "trapezoid_implementation": (
            "numpy.trapezoid" if hasattr(np, "trapezoid") else "numpy.trapz_compat"
        ),
        "source_algorithm": EXTRACTOR_HASH,
    }


def _trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    # np.trapezoid was introduced after the historical code was written.  np.trapz
    # is its compatibility name and implements the same trapezoidal calculation.
    implementation = getattr(np, "trapezoid", np.trapz)
    return float(implementation(values, coordinates))


def extract_fusion_features_4e6b9b4(
    epoch: np.ndarray, sf: float = 100.0
) -> dict[str, float]:
    """Return the frozen 35-D fusion feature mapping for one EEG epoch."""

    signal = np.squeeze(epoch).astype(np.float64)
    if signal.ndim != 1 or signal.size == 0:
        raise ContractError("each raw epoch must squeeze to a non-empty 1-D signal")
    if not np.all(np.isfinite(signal)):
        raise ContractError("raw epoch contains NaN or infinite samples")
    if not math.isfinite(float(sf)) or float(sf) <= 0:
        raise ContractError("sampling frequency must be finite and positive")

    features: dict[str, float] = {}
    features["std"] = float(np.std(signal))
    features["iqr"] = float(np.percentile(signal, 75) - np.percentile(signal, 25))
    features["skew"] = float(skew(signal))
    features["kurt"] = float(kurtosis(signal))
    features["nzc"] = float(np.sum(np.diff(np.sign(signal)) != 0) / len(signal))

    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    activity = np.var(signal)
    mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
    complexity = (
        np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
    )
    features["hmob"] = float(mobility)
    features["hcomp"] = float(complexity)

    features["mean"] = float(np.mean(signal))
    features["rms"] = float(np.sqrt(np.mean(signal**2)))
    if HAS_ANTROPY:
        try:
            features["sample_entropy"] = float(ant.sample_entropy(signal))
        except Exception:
            features["sample_entropy"] = 0.0
        try:
            features["perm_entropy"] = float(
                ant.perm_entropy(signal, normalize=True)
            )
        except Exception:
            features["perm_entropy"] = 0.0
    else:
        features["sample_entropy"] = 0.0
        features["perm_entropy"] = 0.0

    nperseg = int(4 * float(sf))
    frequencies, psd = welch(
        signal,
        float(sf),
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )
    total_power = 0.0
    for band_name, low, high in YASA_BANDS:
        index = np.logical_and(frequencies >= low, frequencies <= high)
        band_power = _trapezoid(psd[index], frequencies[index]) if np.any(index) else 0.0
        features[band_name] = band_power
        total_power += band_power
    features["abspow"] = total_power

    for band_name, _, _ in YASA_BANDS:
        features[f"{band_name}_rel"] = features[band_name] / (total_power + 1e-10)

    delta = features["sdelta"] + features["fdelta"]
    features["dt"] = delta / (features["theta"] + 1e-10)
    features["ds"] = delta / (features["sigma"] + 1e-10)
    features["db"] = delta / (features["beta"] + 1e-10)
    features["at"] = features["alpha"] / (features["theta"] + 1e-10)
    features["ts"] = features["theta"] / (features["sigma"] + 1e-10)

    valid = (frequencies >= 0.4) & (frequencies <= 30.0)
    valid_frequencies = frequencies[valid]
    valid_psd = psd[valid]
    if valid_frequencies.size == 0:
        raise ContractError("Welch output has no bins in the frozen 0.4-30 Hz range")
    psd_sum = np.sum(valid_psd) + 1e-10
    features["spectral_centroid"] = float(
        np.sum(valid_frequencies * valid_psd) / psd_sum
    )
    normalized_psd = np.clip(valid_psd / psd_sum, 1e-10, None)
    features["spectral_entropy"] = float(
        -np.sum(normalized_psd * np.log(normalized_psd))
    )
    cumulative = np.cumsum(valid_psd) / psd_sum
    edge_index = min(int(np.searchsorted(cumulative, 0.95)), len(valid_frequencies) - 1)
    features["spectral_edge_95"] = float(valid_frequencies[edge_index])
    log_psd = np.log(valid_psd + 1e-10)
    features["spectral_slope"] = (
        float(np.polyfit(valid_frequencies, log_psd, 1)[0])
        if len(valid_frequencies) > 1
        else 0.0
    )

    try:
        coefficients = pywt.wavedec(signal, "db4", level=5)
        features["dwt_D4_energy"] = float(
            np.sum(coefficients[2] ** 2) / len(coefficients[2])
        )
        features["dwt_D5_energy"] = float(
            np.sum(coefficients[1] ** 2) / len(coefficients[1])
        )
    except Exception:
        features["dwt_D4_energy"] = 0.0
        features["dwt_D5_energy"] = 0.0

    if tuple(features) != FEATURE_NAMES:
        raise ContractError("internal extractor feature order differs from frozen schema")
    return features


def extract_features_batch_4e6b9b4(
    epochs: np.ndarray, sf: float
) -> np.ndarray:
    """Extract the frozen features and return an exact ``float32`` matrix."""

    values = np.asarray(epochs)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ContractError("raw x must have shape (n_epochs, n_samples[, 1])")
    rows = [extract_fusion_features_4e6b9b4(row, sf) for row in values]
    matrix = np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32
    )
    if matrix.shape != (len(values), len(FEATURE_NAMES)):
        raise ContractError("feature extractor returned an unexpected matrix shape")
    if not np.all(np.isfinite(matrix)):
        bad = np.argwhere(~np.isfinite(matrix))[:5].tolist()
        raise ContractError(f"feature matrix contains NaN or infinite values: {bad}")
    return matrix


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"could not read feature-cache manifest: {error}") from error
    if not isinstance(payload, dict):
        raise ContractError("feature-cache manifest must contain one JSON object")
    return payload


def _load_text_scalar(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise ContractError(f"{field} must be a scalar string")
    result = array.item()
    if isinstance(result, bytes):
        try:
            result = result.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContractError(f"{field} is not UTF-8") from error
    return _required_text(result, field)


def _parse_record_name(name: str) -> tuple[str, str]:
    match = RAW_RECORD_PATTERN.fullmatch(name)
    if match is None:
        raise ContractError(f"raw record has no stable Sleep-EDF identity: {name}")
    return match.group("subject"), Path(name).stem


def _label_hash(labels: np.ndarray) -> str:
    return canonical_json_hash([int(value) for value in np.asarray(labels).tolist()])


def _feature_row_hash(
    datasets: Sequence[Any],
    subjects: Sequence[Any],
    records: Sequence[Any],
    epochs: Sequence[Any],
    labels: Sequence[Any],
) -> str:
    lengths = {len(datasets), len(subjects), len(records), len(epochs), len(labels)}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ContractError("feature row hash inputs must be non-empty and equal length")
    rows = []
    seen: set[tuple[str, str, str, int]] = set()
    for dataset, subject, record, epoch, label in zip(
        datasets, subjects, records, epochs, labels
    ):
        key = (
            _required_text(dataset, "dataset"),
            _normalize_subject(subject),
            _required_text(record, "record_id"),
            int(epoch),
        )
        if key[3] < 0:
            raise ContractError("epoch_index must be non-negative")
        if key in seen:
            raise ContractError(f"duplicate feature row identity: {key}")
        seen.add(key)
        rows.append(
            {
                "dataset": key[0],
                "subject_id": key[1],
                "record_id": key[2],
                "epoch_index": key[3],
                "label": int(label),
            }
        )
    return canonical_json_hash(rows)


def _read_raw_record(path: Path) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    before_hash = sha256_file(path)
    subject, record = _parse_record_name(path.name)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ContractError(f"raw NPZ has duplicate member names: {path.name}")
            missing = {"x", "y", "fs"} - set(archive.files)
            if missing:
                raise ContractError(f"raw NPZ {path.name} is missing {sorted(missing)}")
            epochs = np.array(archive["x"], copy=True)
            labels = np.array(archive["y"], copy=True)
            sampling_array = np.asarray(archive["fs"])
    except ContractError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise ContractError(f"could not safely read raw record {path.name}: {error}") from error

    if epochs.ndim == 3 and epochs.shape[-1] == 1:
        epochs = epochs.squeeze(-1)
    if epochs.ndim != 2 or epochs.shape[0] == 0 or epochs.shape[1] == 0:
        raise ContractError(f"raw x has invalid shape in {path.name}: {epochs.shape}")
    if labels.ndim != 1 or len(labels) != len(epochs):
        raise ContractError(
            f"raw x/y row mismatch in {path.name}: x={len(epochs)}, y={labels.shape}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ContractError(f"raw labels must use an integer dtype in {path.name}")
    labels = labels.astype(np.int64, copy=False)
    if sampling_array.size != 1:
        raise ContractError(f"raw fs must be scalar in {path.name}")
    sampling_frequency = float(sampling_array.reshape(-1)[0])
    if not math.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ContractError(f"raw fs must be finite and positive in {path.name}")
    after_hash = sha256_file(path)
    if after_hash != before_hash:
        raise ContractError(f"raw record changed while being read: {path.name}")
    source = {
        "filename": path.name,
        "subject_id": subject,
        "record_id": record,
        "row_count": int(len(labels)),
        "sampling_frequency": sampling_frequency,
        "label_hash": _label_hash(labels),
        "sha256": before_hash,
    }
    return epochs, labels, sampling_frequency, source


def _module_code_hash() -> str:
    return sha256_file(Path(__file__).resolve())


def _safe_shard_path(manifest_path: Path, relative: Any) -> Path:
    relative_text = _required_text(relative, "shard.path")
    pure = PurePosixPath(relative_text)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "." in pure.parts:
        raise ContractError(f"unsafe shard path: {relative_text!r}")
    if pure.as_posix() != relative_text or pure.parts[0] != "shards":
        raise ContractError(f"shard path is not canonical: {relative_text!r}")
    base = manifest_path.parent.resolve()
    resolved = (base / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ContractError(f"shard escapes manifest directory: {relative_text!r}") from error
    return resolved


def _validate_exact_keys(payload: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ContractError(f"{name} keys differ: missing={missing}, extra={extra}")


def _validate_raw_sources(raw_sources: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ContractError("manifest raw_sources must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, source in enumerate(raw_sources):
        if not isinstance(source, dict):
            raise ContractError(f"raw_sources[{index}] must be an object")
        _validate_exact_keys(source, RAW_SOURCE_KEYS, f"raw_sources[{index}]")
        filename = _required_text(source["filename"], "raw filename")
        parsed_subject, parsed_record = _parse_record_name(filename)
        subject = _normalize_subject(source["subject_id"])
        record = _required_text(source["record_id"], "record_id")
        if subject != parsed_subject or record != parsed_record:
            raise ContractError(f"raw source filename identity mismatch: {filename}")
        if filename in seen_names:
            raise ContractError(f"duplicate raw source: {filename}")
        seen_names.add(filename)
        row_count = source["row_count"]
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
            raise ContractError(f"invalid row_count for {filename}")
        sampling_frequency = source["sampling_frequency"]
        if isinstance(sampling_frequency, bool) or not isinstance(
            sampling_frequency, (int, float)
        ):
            raise ContractError(f"invalid sampling_frequency for {filename}")
        sampling_frequency = float(sampling_frequency)
        if not math.isfinite(sampling_frequency) or sampling_frequency <= 0:
            raise ContractError(f"invalid sampling_frequency for {filename}")
        normalized.append(
            {
                "filename": filename,
                "subject_id": subject,
                "record_id": record,
                "row_count": row_count,
                "sampling_frequency": sampling_frequency,
                "label_hash": _require_sha256(source["label_hash"], "label_hash"),
                "sha256": _require_sha256(source["sha256"], "raw sha256"),
            }
        )
    if normalized != sorted(normalized, key=lambda item: item["filename"]):
        raise ContractError("manifest raw_sources are not in canonical filename order")
    return normalized


def _manifest_payload_hash(payload: Mapping[str, Any]) -> str:
    without_hash = dict(payload)
    without_hash.pop("manifest_hash", None)
    return canonical_json_hash(without_hash)


def _cache_key(
    *,
    dataset: str,
    schema_hash: str,
    extractor_hash: str,
    config_hash: str,
    code_hash: str,
    data_hash: str,
) -> str:
    return canonical_json_hash(
        {
            "namespace": FEATURE_CACHE_SCHEMA_VERSION,
            "dataset": dataset,
            "schema_hash": schema_hash,
            "extractor_hash": extractor_hash,
            "config_hash": config_hash,
            "code_hash": code_hash,
            "data_hash": data_hash,
        }
    )


def _load_manifest(manifest_path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise ContractError(f"feature-cache manifest not found: {path}")
    payload = _load_json_file(path)
    _validate_exact_keys(payload, MANIFEST_KEYS, "feature-cache manifest")
    if payload["schema_version"] != FEATURE_CACHE_SCHEMA_VERSION:
        raise ContractError("feature-cache schema_version is not supported")
    if payload["schema"] != FEATURE_CACHE_SCHEMA:
        raise ContractError("feature-cache schema descriptor mismatch")
    if payload["schema_hash"] != FEATURE_CACHE_SCHEMA_HASH:
        raise ContractError("feature-cache schema_hash mismatch")
    dataset = _required_text(payload["dataset"], "dataset")
    if payload["extractor"] != EXTRACTOR_SPEC or payload["extractor_hash"] != EXTRACTOR_HASH:
        raise ContractError("feature-cache extractor provenance mismatch")
    if not isinstance(payload["config"], dict):
        raise ContractError("feature-cache config must be an object")
    config_hash = _require_sha256(payload["config_hash"], "config_hash")
    if config_hash != canonical_json_hash(payload["config"]):
        raise ContractError("feature-cache config_hash mismatch")
    code_hash = _require_sha256(payload["code_hash"], "code_hash")
    raw_sources = _validate_raw_sources(payload["raw_sources"])
    data_hash = _require_sha256(payload["data_hash"], "data_hash")
    if data_hash != canonical_json_hash(raw_sources):
        raise ContractError("feature-cache data_hash mismatch")
    if payload["feature_names"] != list(FEATURE_NAMES):
        raise ContractError("feature-cache feature_names mismatch")

    source_subjects = sorted({source["subject_id"] for source in raw_sources})
    if payload["subjects"] != source_subjects:
        raise ContractError("manifest subjects do not match raw_sources")
    expected_total = sum(source["row_count"] for source in raw_sources)
    if payload["total_rows"] != expected_total:
        raise ContractError("manifest total_rows does not match raw_sources")

    expected_cache_key = _cache_key(
        dataset=dataset,
        schema_hash=FEATURE_CACHE_SCHEMA_HASH,
        extractor_hash=EXTRACTOR_HASH,
        config_hash=config_hash,
        code_hash=code_hash,
        data_hash=data_hash,
    )
    if payload["cache_key"] != expected_cache_key:
        raise ContractError("feature-cache cache_key mismatch")
    if payload["manifest_hash"] != _manifest_payload_hash(payload):
        raise ContractError("feature-cache manifest_hash mismatch")

    shards = payload["shards"]
    if not isinstance(shards, list) or not shards:
        raise ContractError("manifest shards must be a non-empty list")
    seen_subjects: set[str] = set()
    seen_paths: set[str] = set()
    by_subject_sources = {
        subject: [source for source in raw_sources if source["subject_id"] == subject]
        for subject in source_subjects
    }
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ContractError(f"shards[{index}] must be an object")
        _validate_exact_keys(shard, SHARD_MANIFEST_KEYS, f"shards[{index}]")
        subject = _normalize_subject(shard["subject_id"])
        if subject in seen_subjects:
            raise ContractError(f"duplicate subject shard: {subject}")
        seen_subjects.add(subject)
        relative_path = _required_text(shard["path"], "shard.path")
        if relative_path in seen_paths:
            raise ContractError(f"duplicate shard path: {relative_path}")
        seen_paths.add(relative_path)
        sources = by_subject_sources.get(subject)
        if not sources:
            raise ContractError(f"shard subject absent from raw_sources: {subject}")
        source_names = [source["filename"] for source in sources]
        if shard["raw_files"] != source_names:
            raise ContractError(f"shard raw_files mismatch for subject {subject}")
        expected_rows = sum(source["row_count"] for source in sources)
        if shard["row_count"] != expected_rows:
            raise ContractError(f"shard row_count mismatch for subject {subject}")
        expected_subject_hash = canonical_json_hash(sources)
        if shard["subject_data_hash"] != expected_subject_hash:
            raise ContractError(f"subject_data_hash mismatch for subject {subject}")
        _require_sha256(shard["row_hash"], "shard row_hash")
        expected_shard_sha = _require_sha256(shard["sha256"], "shard sha256")
        shard_path = _safe_shard_path(path, relative_path)
        if not shard_path.is_file():
            raise ContractError(f"feature shard is missing: {shard_path}")
        if sha256_file(shard_path) != expected_shard_sha:
            raise ContractError(f"feature shard SHA-256 mismatch: {shard_path.name}")
    if sorted(seen_subjects) != source_subjects:
        raise ContractError("manifest has missing or extra subject shards")
    if [shard["subject_id"] for shard in shards] != source_subjects:
        raise ContractError("manifest shards are not in canonical subject order")
    return path, payload


def _load_shard(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[RowIdentity, ...]]:
    subject = _normalize_subject(shard["subject_id"])
    shard_path = _safe_shard_path(manifest_path, shard["path"])
    try:
        with np.load(shard_path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ContractError(f"shard has duplicate NPZ keys: {shard_path.name}")
            actual_keys = set(archive.files)
            missing = sorted(SHARD_KEYS - actual_keys)
            extra = sorted(actual_keys - SHARD_KEYS)
            if missing or extra:
                raise ContractError(
                    f"shard keys differ for {shard_path.name}: "
                    f"missing={missing}, extra={extra}"
                )
            arrays = {name: np.array(archive[name], copy=True) for name in SHARD_KEYS}
    except ContractError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise ContractError(f"could not safely read shard {shard_path.name}: {error}") from error

    for name, array in arrays.items():
        if np.asarray(array).dtype.kind == "O":
            raise ContractError(f"object dtype is forbidden in shard field {name}")
    for field in (
        "schema_hash",
        "extractor_hash",
        "config_hash",
        "code_hash",
        "subject_data_hash",
        "row_hash",
    ):
        actual = _load_text_scalar(arrays[field], field)
        expected = (
            shard[field]
            if field in {"subject_data_hash", "row_hash"}
            else manifest[field]
        )
        if actual != expected:
            raise ContractError(f"shard {field} mismatch for subject {subject}")

    names = arrays["feature_names"]
    if names.ndim != 1 or names.dtype.kind not in {"U", "S"}:
        raise ContractError("shard feature_names must be a one-dimensional string array")
    decoded_names = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in names.tolist()
    ]
    if decoded_names != list(FEATURE_NAMES):
        raise ContractError(f"shard feature_names mismatch for subject {subject}")

    features = arrays["features"]
    labels = arrays["labels"]
    datasets = arrays["datasets"]
    subjects = arrays["subject_ids"]
    records = arrays["record_ids"]
    epochs = arrays["epoch_indices"]
    row_count = shard["row_count"]
    if features.dtype != np.dtype(np.float32):
        raise ContractError("shard features must use exact float32 dtype")
    if labels.dtype != np.dtype(np.int64) or epochs.dtype != np.dtype(np.int64):
        raise ContractError("shard labels and epoch_indices must use exact int64 dtype")
    if features.shape != (row_count, len(FEATURE_NAMES)):
        raise ContractError(f"shard feature shape mismatch for subject {subject}")
    for field, array in {
        "labels": labels,
        "datasets": datasets,
        "subject_ids": subjects,
        "record_ids": records,
        "epoch_indices": epochs,
    }.items():
        if array.shape != (row_count,):
            raise ContractError(f"shard {field} shape mismatch for subject {subject}")
    if not np.all(np.isfinite(features)):
        raise ContractError(f"shard features contain non-finite values for subject {subject}")
    if datasets.dtype.kind not in {"U", "S"} or subjects.dtype.kind not in {"U", "S"}:
        raise ContractError("shard dataset and subject identity fields must be strings")
    if records.dtype.kind not in {"U", "S"}:
        raise ContractError("shard record identity field must be strings")

    datasets_text = [str(value) for value in datasets.tolist()]
    subjects_text = [str(value) for value in subjects.tolist()]
    records_text = [str(value) for value in records.tolist()]
    if any(value != manifest["dataset"] for value in datasets_text):
        raise ContractError(f"row dataset mismatch for subject {subject}")
    if any(value != subject for value in subjects_text):
        raise ContractError(f"row subject mismatch for subject {subject}")

    raw_sources = json.loads(
        _load_text_scalar(arrays["raw_sources_json"], "raw_sources_json"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    expected_sources = [
        source
        for source in manifest["raw_sources"]
        if source["subject_id"] == subject
    ]
    if raw_sources != expected_sources:
        raise ContractError(f"shard raw-source provenance mismatch for subject {subject}")

    expected_records: list[str] = []
    expected_epochs: list[int] = []
    start = 0
    for source in expected_sources:
        count = source["row_count"]
        expected_records.extend([source["record_id"]] * count)
        expected_epochs.extend(range(count))
        stop = start + count
        if _label_hash(labels[start:stop]) != source["label_hash"]:
            raise ContractError(f"label provenance mismatch for {source['filename']}")
        start = stop
    if records_text != expected_records or epochs.tolist() != expected_epochs:
        raise ContractError(
            f"duplicate, missing, extra, or reordered row identity in subject {subject}"
        )
    actual_row_hash = _feature_row_hash(
        datasets_text, subjects_text, records_text, epochs.tolist(), labels.tolist()
    )
    if actual_row_hash != shard["row_hash"]:
        raise ContractError(f"feature row_hash mismatch for subject {subject}")

    identities = tuple(
        RowIdentity(
            dataset=manifest["dataset"],
            subject_id=subject,
            record_id=record,
            epoch_index=int(epoch),
            outer_fold="__cache_validation__",
        )
        for record, epoch in zip(records_text, epochs.tolist())
    )
    return features, labels, identities


def _arrays_equal(existing: Path, arrays: Mapping[str, np.ndarray]) -> bool:
    try:
        with np.load(existing, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(arrays):
                return False
            for key, expected in arrays.items():
                actual = np.asarray(archive[key])
                if actual.dtype != np.asarray(expected).dtype or not np.array_equal(
                    actual, np.asarray(expected), equal_nan=False
                ):
                    return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def build_feature_cache(
    raw_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    dataset_id: str = "sleepedf20",
) -> Path:
    """Recompute the exact frozen 35-D cache directly from raw record NPZs.

    The returned manifest is content addressed.  Existing content-addressed
    artifacts are reused only when their logical arrays/bytes exactly match; a
    collision or partial artifact fails closed rather than being overwritten.
    """

    raw_root = Path(raw_dir).resolve()
    destination = Path(output_dir).resolve()
    dataset = _required_text(dataset_id, "dataset_id")
    if not raw_root.is_dir():
        raise ContractError(f"raw record directory not found: {raw_root}")
    raw_paths = sorted(raw_root.glob("SC*.npz"), key=lambda path: path.name)
    if not raw_paths:
        raise ContractError(f"no SC*.npz raw records found under {raw_root}")
    if len({path.name for path in raw_paths}) != len(raw_paths):
        raise ContractError("duplicate raw record filenames")
    for path in raw_paths:
        _parse_record_name(path.name)

    code_hash = _module_code_hash()
    config = _runtime_config()
    config_hash = canonical_json_hash(config)
    loaded: list[tuple[Path, np.ndarray, np.ndarray, float, dict[str, Any]]] = []
    for path in raw_paths:
        epochs, labels, sampling_frequency, source = _read_raw_record(path)
        loaded.append((path, epochs, labels, sampling_frequency, source))
    raw_sources = [item[4] for item in loaded]
    data_hash = canonical_json_hash(raw_sources)
    subjects = sorted({source["subject_id"] for source in raw_sources})
    cache_key = _cache_key(
        dataset=dataset,
        schema_hash=FEATURE_CACHE_SCHEMA_HASH,
        extractor_hash=EXTRACTOR_HASH,
        config_hash=config_hash,
        code_hash=code_hash,
        data_hash=data_hash,
    )

    shard_directory = destination / "shards"
    shard_entries: list[dict[str, Any]] = []
    for subject in subjects:
        subject_items = [item for item in loaded if item[4]["subject_id"] == subject]
        subject_sources = [item[4] for item in subject_items]
        subject_data_hash = canonical_json_hash(subject_sources)
        matrices: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        record_values: list[str] = []
        epoch_values: list[int] = []
        for path, epochs, labels, sampling_frequency, source in subject_items:
            matrix = extract_features_batch_4e6b9b4(epochs, sampling_frequency)
            if sha256_file(path) != source["sha256"]:
                raise ContractError(f"raw record changed during extraction: {path.name}")
            matrices.append(matrix)
            label_parts.append(labels.astype(np.int64, copy=False))
            record_values.extend([source["record_id"]] * len(labels))
            epoch_values.extend(range(len(labels)))
        features = np.vstack(matrices).astype(np.float32, copy=False)
        labels = np.concatenate(label_parts).astype(np.int64, copy=False)
        datasets = np.asarray([dataset] * len(labels), dtype=np.str_)
        subject_ids = np.asarray([subject] * len(labels), dtype=np.str_)
        record_ids = np.asarray(record_values, dtype=np.str_)
        epoch_indices = np.asarray(epoch_values, dtype=np.int64)
        row_hash = _feature_row_hash(
            datasets.tolist(),
            subject_ids.tolist(),
            record_ids.tolist(),
            epoch_indices.tolist(),
            labels.tolist(),
        )
        shard_key = canonical_json_hash(
            {
                "dataset": dataset,
                "subject_id": subject,
                "subject_data_hash": subject_data_hash,
                "extractor_hash": EXTRACTOR_HASH,
                "config_hash": config_hash,
                "schema_hash": FEATURE_CACHE_SCHEMA_HASH,
                "code_hash": code_hash,
                "row_hash": row_hash,
            }
        )
        relative_path = PurePosixPath("shards", f"subject-{subject}-{shard_key}.npz")
        shard_path = destination / Path(*relative_path.parts)
        arrays = {
            "features": features,
            "labels": labels,
            "datasets": datasets,
            "subject_ids": subject_ids,
            "record_ids": record_ids,
            "epoch_indices": epoch_indices,
            "feature_names": np.asarray(FEATURE_NAMES, dtype=np.str_),
            "schema_hash": np.asarray(FEATURE_CACHE_SCHEMA_HASH, dtype=np.str_),
            "extractor_hash": np.asarray(EXTRACTOR_HASH, dtype=np.str_),
            "config_hash": np.asarray(config_hash, dtype=np.str_),
            "code_hash": np.asarray(code_hash, dtype=np.str_),
            "subject_data_hash": np.asarray(subject_data_hash, dtype=np.str_),
            "raw_sources_json": np.asarray(
                canonical_json_bytes(subject_sources).decode("utf-8"), dtype=np.str_
            ),
            "row_hash": np.asarray(row_hash, dtype=np.str_),
        }
        if shard_path.exists():
            if not _arrays_equal(shard_path, arrays):
                raise ContractError(
                    f"existing content-addressed shard does not match: {shard_path}"
                )
        else:
            atomic_write_npz(shard_path, arrays, compressed=True)
        shard_sha = sha256_file(shard_path)
        shard_entries.append(
            {
                "subject_id": subject,
                "path": relative_path.as_posix(),
                "sha256": shard_sha,
                "row_count": int(len(labels)),
                "row_hash": row_hash,
                "subject_data_hash": subject_data_hash,
                "raw_files": [source["filename"] for source in subject_sources],
            }
        )

    payload: dict[str, Any] = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "schema": FEATURE_CACHE_SCHEMA,
        "schema_hash": FEATURE_CACHE_SCHEMA_HASH,
        "dataset": dataset,
        "extractor": EXTRACTOR_SPEC,
        "extractor_hash": EXTRACTOR_HASH,
        "config": config,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "data_hash": data_hash,
        "raw_sources": raw_sources,
        "feature_names": list(FEATURE_NAMES),
        "subjects": subjects,
        "total_rows": sum(source["row_count"] for source in raw_sources),
        "shards": shard_entries,
        "cache_key": cache_key,
    }
    payload["manifest_hash"] = _manifest_payload_hash(payload)
    manifest_path = destination / f"manifest-{cache_key}.json"
    expected_bytes = canonical_json_bytes(payload) + b"\n"
    if manifest_path.exists():
        try:
            existing_bytes = manifest_path.read_bytes()
        except OSError as error:
            raise ContractError(f"could not read existing manifest: {error}") from error
        if existing_bytes != expected_bytes:
            raise ContractError(
                f"existing content-addressed manifest does not match: {manifest_path}"
            )
    else:
        atomic_write_json(manifest_path, payload)
    _load_manifest(manifest_path)
    return manifest_path


def validate_feature_cache(manifest_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate manifest provenance and every shard, returning a small summary."""

    path, manifest = _load_manifest(manifest_path)
    for shard in manifest["shards"]:
        _load_shard(path, manifest, shard)
    return {
        "valid": True,
        "manifest": str(path),
        "cache_key": manifest["cache_key"],
        "subjects": list(manifest["subjects"]),
        "total_rows": manifest["total_rows"],
        "data_hash": manifest["data_hash"],
        "extractor_hash": manifest["extractor_hash"],
        "identity_and_cache_contract": "PASS",
        "confirmatory_eligibility": "NOT_DETERMINED_BY_THIS_MODULE",
    }


def validate_feature_cache_against_raw(
    manifest_path: str | os.PathLike[str],
    raw_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Bind every cached source record to the current raw directory.

    Cache self-validation alone cannot prove that its recorded source hashes
    describe the raw cohort supplied to a later fold run.  This check rebuilds
    the canonical source inventory from the current ``SC*.npz`` files and
    requires an exact field-for-field match with the signed manifest.  Signal
    changes with unchanged labels/epoch identities therefore fail closed.
    """

    path, manifest = _load_manifest(manifest_path)
    raw_root = Path(raw_dir).resolve()
    if not raw_root.is_dir():
        raise ContractError(f"raw record directory not found: {raw_root}")
    raw_paths = sorted(raw_root.glob("SC*.npz"), key=lambda item: item.name)
    if not raw_paths:
        raise ContractError(f"no SC*.npz raw records found under {raw_root}")
    if len({item.name for item in raw_paths}) != len(raw_paths):
        raise ContractError("duplicate raw record filenames")

    actual_sources: list[dict[str, Any]] = []
    for raw_path in raw_paths:
        _, _, _, source = _read_raw_record(raw_path)
        actual_sources.append(source)
    expected_sources = manifest["raw_sources"]
    if actual_sources != expected_sources:
        expected_by_name = {item["filename"]: item for item in expected_sources}
        actual_by_name = {item["filename"]: item for item in actual_sources}
        missing = sorted(set(expected_by_name) - set(actual_by_name))
        extra = sorted(set(actual_by_name) - set(expected_by_name))
        changed = sorted(
            name
            for name in set(expected_by_name) & set(actual_by_name)
            if expected_by_name[name] != actual_by_name[name]
        )
        raise ContractError(
            "feature-cache/raw source provenance mismatch: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    data_hash = canonical_json_hash(actual_sources)
    if data_hash != manifest["data_hash"]:
        raise ContractError("feature-cache/raw aggregate data_hash mismatch")
    return {
        "raw_binding": "PASS",
        "manifest": str(path),
        "raw_dir": str(raw_root),
        "record_count": len(actual_sources),
        "row_count": sum(item["row_count"] for item in actual_sources),
        "data_hash": data_hash,
    }


def load_feature_rows(
    manifest_path: str | os.PathLike[str],
    subjects: Sequence[Any],
    outer_fold: Any,
) -> tuple[np.ndarray, np.ndarray, tuple[RowIdentity, ...], tuple[str, ...]]:
    """Load authorized subjects and attach a caller-supplied outer-fold identity.

    Requested subject order is preserved.  Within a subject, rows must be in
    canonical record-name/epoch order; a duplicate, missing, extra, or reordered
    row causes an explicit ``ContractError``.
    """

    path, manifest = _load_manifest(manifest_path)
    requested = tuple(_normalize_subject(subject) for subject in subjects)
    if not requested:
        raise ContractError("subjects must not be empty")
    if len(set(requested)) != len(requested):
        raise ContractError("subjects contains duplicate subject IDs")
    unknown = sorted(set(requested) - set(manifest["subjects"]))
    if unknown:
        raise ContractError(f"requested subjects are missing from cache: {unknown}")
    fold = _normalize_outer_fold(outer_fold)
    shards_by_subject = {shard["subject_id"]: shard for shard in manifest["shards"]}
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    identities: list[RowIdentity] = []
    seen: set[RowIdentity] = set()
    for subject in requested:
        features, labels, cache_identities = _load_shard(
            path, manifest, shards_by_subject[subject]
        )
        feature_parts.append(features)
        label_parts.append(labels)
        for identity in cache_identities:
            attached = RowIdentity(
                dataset=identity.dataset,
                subject_id=identity.subject_id,
                record_id=identity.record_id,
                epoch_index=identity.epoch_index,
                outer_fold=fold,
            )
            if attached in seen:
                raise ContractError(f"duplicate loaded row identity: {attached.as_dict()}")
            seen.add(attached)
            identities.append(attached)
    return (
        np.vstack(feature_parts).astype(np.float32, copy=False),
        np.concatenate(label_parts).astype(np.int64, copy=False),
        tuple(identities),
        FEATURE_NAMES,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Recompute and validate the cache without reading any legacy cache."""

    parser = argparse.ArgumentParser(
        description=(
            "Recompute the frozen 4e6b9b4 35-D feature cache directly from "
            "identity-bearing raw Sleep-EDF record NPZs."
        )
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="sleepedf20")
    arguments = parser.parse_args(argv)
    manifest = build_feature_cache(
        arguments.raw_dir,
        arguments.output_dir,
        dataset_id=arguments.dataset,
    )
    result = {
        "manifest": str(manifest.resolve()),
        "validation": validate_feature_cache(manifest),
    }
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


__all__ = [
    "EXTRACTOR_HASH",
    "EXTRACTOR_SPEC",
    "FEATURE_CACHE_SCHEMA_HASH",
    "FEATURE_CACHE_SCHEMA_VERSION",
    "FEATURE_NAMES",
    "build_feature_cache",
    "extract_features_batch_4e6b9b4",
    "extract_fusion_features_4e6b9b4",
    "load_feature_rows",
    "validate_feature_cache",
    "validate_feature_cache_against_raw",
]


if __name__ == "__main__":  # pragma: no cover - exercised by the real recompute CLI.
    raise SystemExit(main())
