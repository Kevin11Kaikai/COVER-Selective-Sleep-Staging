"""Fail-closed data contracts for the clean Stage-1 nested-LOSO pipeline.

This module deliberately contains no model or dataset-specific training code.  It
defines the immutable row identity, prediction/manifest invariants, content hashes,
and atomic persistence helpers used at artifact boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np


SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
DEFAULT_CONFIDENCE_ATOL = 1e-7
DEFAULT_CONFIDENCE_RTOL = 1e-6
DEFAULT_PROBABILITY_ATOL = 1e-6


class ContractError(ValueError):
    """Raised when an artifact cannot satisfy the confirmatory data contract."""


def _required_text(value: Any, field_name: str) -> str:
    if value is None or isinstance(value, bool):
        raise ContractError(f"{field_name} must be a non-empty identifier")
    normalized = str(value).strip()
    if not normalized:
        raise ContractError(f"{field_name} must be a non-empty identifier")
    return normalized


def _sha256(value: Any, field_name: str) -> str:
    normalized = _required_text(value, field_name).lower()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise ContractError(f"{field_name} must be a bare 64-character SHA-256 hex digest")
    return normalized


def _jsonable(value: Any) -> Any:
    """Convert supported values to an unambiguous JSON-compatible representation."""

    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON mappings require string keys")
            converted[key] = _jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize *value* deterministically for hashing and manifests."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    """Return the SHA-256 digest of the canonical JSON representation."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_text_scalar(value: np.ndarray, field_name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise ContractError(f"{field_name} must be a zero-dimensional string array")
    scalar = array.item()
    if isinstance(scalar, bytes):
        try:
            scalar = scalar.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractError(f"{field_name} is not valid UTF-8") from exc
    return _required_text(scalar, field_name)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_scalar(value: np.ndarray, field_name: str) -> Any:
    text = _load_text_scalar(value, field_name)
    return json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA-256 digest for bytes."""

    if not isinstance(data, bytes):
        raise TypeError("sha256_bytes requires bytes")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it entirely into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, order=True)
class RowIdentity:
    """Stable prediction-row key; ground-truth label is intentionally excluded.

    Labels are checked only after a one-to-one identity join.  Excluding labels from
    the key prevents a wrong row with the same label from masquerading as a match.
    """

    dataset: str
    subject_id: str
    record_id: str
    epoch_index: int
    outer_fold: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _required_text(self.dataset, "dataset"))
        object.__setattr__(self, "subject_id", _required_text(self.subject_id, "subject_id"))
        object.__setattr__(self, "record_id", _required_text(self.record_id, "record_id"))
        object.__setattr__(self, "outer_fold", _required_text(self.outer_fold, "outer_fold"))
        if isinstance(self.epoch_index, bool) or not isinstance(
            self.epoch_index, (int, np.integer)
        ):
            raise ContractError("epoch_index must be a non-negative integer")
        epoch_index = int(self.epoch_index)
        if epoch_index < 0:
            raise ContractError("epoch_index must be a non-negative integer")
        object.__setattr__(self, "epoch_index", epoch_index)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subject_id": self.subject_id,
            "record_id": self.record_id,
            "epoch_index": self.epoch_index,
            "outer_fold": self.outer_fold,
        }


def _identity_preview(identities: Sequence[RowIdentity], limit: int = 3) -> str:
    preview = [identity.as_dict() for identity in identities[:limit]]
    suffix = " ..." if len(identities) > limit else ""
    return f"{preview}{suffix}"


def _validate_unique(identities: Sequence[RowIdentity], side: str) -> None:
    duplicates = [identity for identity, count in Counter(identities).items() if count > 1]
    if duplicates:
        duplicates.sort()
        raise ContractError(
            f"duplicate row identity on {side}: {_identity_preview(duplicates)}"
        )


def align_one_to_one(
    reference_ids: Sequence[RowIdentity],
    candidate_ids: Sequence[RowIdentity],
    *,
    reference_labels: Sequence[Any] | np.ndarray,
    candidate_labels: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Return indices that reorder candidate rows into reference identity order.

    Duplicate, missing, and extra identities fail explicitly.  Ground-truth labels
    are not part of the join key and are compared only after the candidate rows have
    been reordered by identity.
    """

    reference = tuple(reference_ids)
    candidate = tuple(candidate_ids)
    if not all(isinstance(identity, RowIdentity) for identity in reference + candidate):
        raise ContractError("all row identities must be RowIdentity instances")
    if not reference or not candidate:
        raise ContractError("row alignment requires non-empty reference and candidate rows")
    _validate_unique(reference, "reference")
    _validate_unique(candidate, "candidate")

    reference_set = set(reference)
    candidate_set = set(candidate)
    missing = sorted(reference_set - candidate_set)
    extra = sorted(candidate_set - reference_set)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={_identity_preview(missing)}")
        if extra:
            details.append(f"extra={_identity_preview(extra)}")
        raise ContractError("row identity sets differ: " + "; ".join(details))

    reference_label_array = np.asarray(reference_labels)
    candidate_label_array = np.asarray(candidate_labels)
    if reference_label_array.ndim != 1 or len(reference_label_array) != len(reference):
        raise ContractError("reference_labels must be one-dimensional and match reference_ids")
    if candidate_label_array.ndim != 1 or len(candidate_label_array) != len(candidate):
        raise ContractError("candidate_labels must be one-dimensional and match candidate_ids")

    candidate_position = {identity: index for index, identity in enumerate(candidate)}
    reorder = np.asarray([candidate_position[identity] for identity in reference], dtype=np.int64)
    aligned_labels = candidate_label_array[reorder]
    label_matches = np.asarray(reference_label_array == aligned_labels)
    if label_matches.shape != reference_label_array.shape:
        raise ContractError("ground-truth labels could not be compared elementwise")
    mismatches = np.flatnonzero(~label_matches)
    if mismatches.size:
        examples = [
            {
                "identity": reference[index].as_dict(),
                "reference_label": _jsonable(reference_label_array[index]),
                "candidate_label": _jsonable(aligned_labels[index]),
            }
            for index in mismatches[:3]
        ]
        raise ContractError(
            f"ground-truth label mismatch after identity join: {examples}"
        )
    return reorder


def align_rows(
    reference_ids: Sequence[RowIdentity],
    candidate_ids: Sequence[RowIdentity],
    *,
    reference_labels: Sequence[Any] | np.ndarray,
    candidate_labels: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Named alias for :func:`align_one_to_one`."""

    return align_one_to_one(
        reference_ids,
        candidate_ids,
        reference_labels=reference_labels,
        candidate_labels=candidate_labels,
    )


def hash_rows(identities: Sequence[RowIdentity], labels: Sequence[Any] | np.ndarray) -> str:
    """Hash the ordered row identities and labels bound to a prediction artifact."""

    identity_tuple = tuple(identities)
    if not all(isinstance(identity, RowIdentity) for identity in identity_tuple):
        raise ContractError("all row hash identities must be RowIdentity instances")
    if not identity_tuple:
        raise ContractError("row hash input must not be empty")
    _validate_unique(identity_tuple, "row hash input")
    label_array = np.asarray(labels)
    if label_array.ndim != 1 or len(label_array) != len(identity_tuple):
        raise ContractError("labels must be one-dimensional and match identities")
    rows = [
        {"identity": identity.as_dict(), "label": _jsonable(label_array[index])}
        for index, identity in enumerate(identity_tuple)
    ]
    return canonical_json_hash(rows)


def _subject_tuple(values: Sequence[Any], field_name: str, *, allow_empty: bool) -> tuple[str, ...]:
    normalized = tuple(_required_text(value, field_name) for value in values)
    if not allow_empty and not normalized:
        raise ContractError(f"{field_name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ContractError(f"{field_name} contains duplicate subject IDs")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class PredictionManifest:
    """Immutable provenance required for a confirmatory prediction bundle."""

    schema_version: str
    schema_hash: str
    dataset: str
    outer_fold: str
    outer_subject_id: str
    model_id: str
    split_hash: str
    model_hash: str
    checkpoint_hash: str
    data_hash: str
    preprocessing_hash: str
    config_hash: str
    calibration_hash: str
    code_hash: str
    row_hash: str
    seed: int
    training_subjects: tuple[str, ...]
    calibration_subjects: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "schema_version",
            "dataset",
            "outer_fold",
            "outer_subject_id",
            "model_id",
        ):
            normalized = _required_text(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)
        for field_name in (
            "schema_hash",
            "split_hash",
            "model_hash",
            "checkpoint_hash",
            "data_hash",
            "preprocessing_hash",
            "config_hash",
            "calibration_hash",
            "code_hash",
            "row_hash",
        ):
            object.__setattr__(self, field_name, _sha256(getattr(self, field_name), field_name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ContractError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(
            self,
            "training_subjects",
            _subject_tuple(self.training_subjects, "training_subjects", allow_empty=False),
        )
        object.__setattr__(
            self,
            "calibration_subjects",
            _subject_tuple(self.calibration_subjects, "calibration_subjects", allow_empty=True),
        )

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @property
    def manifest_hash(self) -> str:
        return canonical_json_hash(self.as_dict())


@dataclass(frozen=True)
class PredictionBundle:
    """Coherent probabilities, decisions, labels, and immutable provenance."""

    NPZ_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "identities_json",
            "labels",
            "probabilities",
            "predictions",
            "confidences",
            "checkpoint_hash",
            "manifest_json",
        }
    )

    identities: tuple[RowIdentity, ...]
    labels: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    confidences: np.ndarray
    checkpoint_hash: str
    manifest: PredictionManifest

    def __post_init__(self) -> None:
        identities = tuple(self.identities)
        arrays = {
            "labels": np.array(self.labels, copy=True),
            "probabilities": np.array(self.probabilities, copy=True),
            "predictions": np.array(self.predictions, copy=True),
            "confidences": np.array(self.confidences, copy=True),
        }
        for array in arrays.values():
            array.setflags(write=False)
        object.__setattr__(self, "identities", identities)
        for field_name, array in arrays.items():
            object.__setattr__(self, field_name, array)
        checkpoint_hash = _sha256(self.checkpoint_hash, "checkpoint_hash")
        object.__setattr__(self, "checkpoint_hash", checkpoint_hash)
        self.validate()

    def validate(
        self,
        *,
        confidence_atol: float = DEFAULT_CONFIDENCE_ATOL,
        confidence_rtol: float = DEFAULT_CONFIDENCE_RTOL,
        probability_atol: float = DEFAULT_PROBABILITY_ATOL,
    ) -> None:
        """Validate all row, probability, prediction, and provenance invariants."""

        if not isinstance(self.manifest, PredictionManifest):
            raise ContractError("manifest must be a PredictionManifest")
        if not self.identities:
            raise ContractError("prediction bundle must contain at least one row")
        if not all(isinstance(identity, RowIdentity) for identity in self.identities):
            raise ContractError("all bundle identities must be RowIdentity instances")
        _validate_unique(self.identities, "prediction bundle")

        n_rows = len(self.identities)
        if self.labels.ndim != 1 or self.labels.shape != (n_rows,):
            raise ContractError("labels must have shape (n_rows,)")
        if self.predictions.ndim != 1 or self.predictions.shape != (n_rows,):
            raise ContractError("predictions must have shape (n_rows,)")
        if self.confidences.ndim != 1 or self.confidences.shape != (n_rows,):
            raise ContractError("confidences must have shape (n_rows,)")
        if self.probabilities.ndim != 2 or self.probabilities.shape[0] != n_rows:
            raise ContractError("probabilities must have shape (n_rows, n_classes)")
        if self.probabilities.shape[1] < 2:
            raise ContractError("probabilities must contain at least two classes")
        if not np.issubdtype(self.predictions.dtype, np.integer):
            raise ContractError("predictions must use an integer dtype")
        if not np.all(np.isfinite(self.probabilities)):
            raise ContractError("probabilities contain NaN or infinite values")
        if not np.all(np.isfinite(self.confidences)):
            raise ContractError("confidences contain NaN or infinite values")
        if np.any(self.probabilities < 0.0) or np.any(self.probabilities > 1.0):
            raise ContractError("probabilities fall outside [0, 1]")
        if not np.allclose(
            self.probabilities.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=probability_atol,
        ):
            raise ContractError("probability rows do not sum to one")

        expected_predictions = self.probabilities.argmax(axis=1)
        prediction_mismatch = np.flatnonzero(self.predictions != expected_predictions)
        if prediction_mismatch.size:
            raise ContractError(
                "prediction/probability mismatch: prediction must equal argmax(probabilities); "
                f"rows={prediction_mismatch[:10].tolist()}"
            )
        expected_confidences = self.probabilities.max(axis=1)
        confidence_mismatch = np.flatnonzero(
            ~np.isclose(
                self.confidences,
                expected_confidences,
                rtol=confidence_rtol,
                atol=confidence_atol,
            )
        )
        if confidence_mismatch.size:
            raise ContractError(
                "confidence/probability mismatch: confidence must equal max(probabilities); "
                f"rows={confidence_mismatch[:10].tolist()}"
            )

        if self.checkpoint_hash != self.manifest.checkpoint_hash:
            raise ContractError(
                "checkpoint hash mismatch between prediction bundle and manifest"
            )
        if any(identity.dataset != self.manifest.dataset for identity in self.identities):
            raise ContractError("row dataset does not match prediction manifest")
        if any(identity.outer_fold != self.manifest.outer_fold for identity in self.identities):
            raise ContractError("row outer_fold does not match prediction manifest")
        actual_row_hash = hash_rows(self.identities, self.labels)
        if actual_row_hash != self.manifest.row_hash:
            raise ContractError("row hash mismatch between prediction bundle and manifest")

    def to_npz_arrays(self) -> dict[str, np.ndarray]:
        """Return the versioned, pickle-free NPZ layout for this bundle."""

        identities_json = canonical_json_bytes(
            [identity.as_dict() for identity in self.identities]
        ).decode("utf-8")
        manifest_json = canonical_json_bytes(self.manifest.as_dict()).decode("utf-8")
        return {
            "identities_json": np.asarray(identities_json, dtype=np.str_),
            "labels": np.asarray(self.labels),
            "probabilities": np.asarray(self.probabilities),
            "predictions": np.asarray(self.predictions),
            "confidences": np.asarray(self.confidences),
            "checkpoint_hash": np.asarray(self.checkpoint_hash, dtype=np.str_),
            "manifest_json": np.asarray(manifest_json, dtype=np.str_),
        }

    def write_npz(self, path: str | os.PathLike[str]) -> Path:
        """Atomically persist the strict, pickle-free prediction layout."""

        return atomic_write_npz(path, self.to_npz_arrays(), compressed=True)

    @classmethod
    def from_npz(cls, path: str | os.PathLike[str]) -> "PredictionBundle":
        """Load a strict bundle and re-run every constructor invariant."""

        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise ContractError(f"prediction bundle not found: {artifact_path}")
        try:
            with np.load(artifact_path, allow_pickle=False) as archive:
                actual_keys = set(archive.files)
                missing = sorted(cls.NPZ_KEYS - actual_keys)
                extra = sorted(actual_keys - cls.NPZ_KEYS)
                if missing or extra:
                    details: list[str] = []
                    if missing:
                        details.append(f"missing={missing}")
                    if extra:
                        details.append(f"extra={extra}")
                    raise ContractError(
                        "prediction NPZ key contract failed: " + "; ".join(details)
                    )
                identities_payload = _load_json_scalar(
                    archive["identities_json"], "identities_json"
                )
                manifest_payload = _load_json_scalar(
                    archive["manifest_json"], "manifest_json"
                )
                checkpoint_hash = _load_text_scalar(
                    archive["checkpoint_hash"], "checkpoint_hash"
                )
                labels = np.array(archive["labels"], copy=True)
                probabilities = np.array(archive["probabilities"], copy=True)
                predictions = np.array(archive["predictions"], copy=True)
                confidences = np.array(archive["confidences"], copy=True)
        except ContractError:
            raise
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ContractError(f"could not read prediction bundle safely: {exc}") from exc

        if not isinstance(identities_payload, list):
            raise ContractError("identities_json must contain a JSON array")
        if not isinstance(manifest_payload, dict):
            raise ContractError("manifest_json must contain a JSON object")
        try:
            identities = tuple(RowIdentity(**item) for item in identities_payload)
            manifest = PredictionManifest(**manifest_payload)
        except (TypeError, ContractError) as exc:
            raise ContractError(f"invalid identity or manifest payload: {exc}") from exc
        return cls(
            identities=identities,
            labels=labels,
            probabilities=probabilities,
            predictions=predictions,
            confidences=confidences,
            checkpoint_hash=checkpoint_hash,
            manifest=manifest,
        )


def build_cache_key(
    *,
    split_hash: str,
    schema_hash: str,
    model_hash: str,
    data_hash: str,
    config_hash: str,
    calibration_hash: str,
    seed: int,
    namespace: str = "stage1-clean-nested-loso-v1",
) -> str:
    """Build a content-addressed cache key from every provenance dependency."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ContractError("seed must be an integer")
    payload = {
        "namespace": _required_text(namespace, "namespace"),
        "split_hash": _sha256(split_hash, "split_hash"),
        "schema_hash": _sha256(schema_hash, "schema_hash"),
        "model_hash": _sha256(model_hash, "model_hash"),
        "data_hash": _sha256(data_hash, "data_hash"),
        "config_hash": _sha256(config_hash, "config_hash"),
        "calibration_hash": _sha256(calibration_hash, "calibration_hash"),
        "seed": int(seed),
    }
    return canonical_json_hash(payload)


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """Atomically replace *path* with *data* using a temporary sibling file."""

    if not isinstance(data, bytes):
        raise TypeError("atomic_write_bytes requires bytes")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> Path:
    """Atomically write canonical UTF-8 JSON followed by one newline."""

    return atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def atomic_write_npz(
    path: str | os.PathLike[str],
    arrays: Mapping[str, Any],
    *,
    compressed: bool = True,
) -> Path:
    """Atomically write an NPZ archive without NumPy suffix rewriting."""

    if not arrays:
        raise ContractError("cannot write an empty NPZ archive")
    if not all(isinstance(key, str) and key for key in arrays):
        raise ContractError("NPZ array names must be non-empty strings")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = np.savez_compressed if compressed else np.savez
            writer(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


__all__ = [
    "ContractError",
    "PredictionBundle",
    "PredictionManifest",
    "RowIdentity",
    "align_one_to_one",
    "align_rows",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_npz",
    "build_cache_key",
    "canonical_json_bytes",
    "canonical_json_hash",
    "hash_rows",
    "sha256_bytes",
    "sha256_file",
]
