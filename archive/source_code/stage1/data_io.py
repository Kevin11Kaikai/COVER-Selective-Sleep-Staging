"""Fail-closed data access and atomic persistence for the Stage-2 campaign."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sleepedf.data_loader import parse_sleepedf_record

from .firewall import RoleLedger
from .splits import normalize_subject


DATASET_ID = "sleepedf20"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json_bytes(value))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_npz(path: str | Path, arrays: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **arrays)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def discover_raw_records(data_dir: str | Path) -> dict[str, list[Path]]:
    records: dict[str, list[Path]] = {}
    for path in sorted(Path(data_dir).glob("SC*.npz"), key=lambda item: item.name):
        subject, _ = parse_sleepedf_record(path)
        subject_id = normalize_subject(subject)
        records.setdefault(subject_id, []).append(path.resolve())
    if tuple(sorted(records)) != tuple(f"{index:02d}" for index in range(20)):
        raise RuntimeError("raw inventory is not the frozen 20-subject cohort")
    return records


def files_for_subjects(
    records: Mapping[str, Sequence[Path]], subjects: Iterable[object]
) -> list[Path]:
    result: list[Path] = []
    for value in subjects:
        subject = normalize_subject(value)
        if subject not in records:
            raise FileNotFoundError(f"no raw files for subject {subject}")
        result.extend(records[subject])
    return sorted(result, key=lambda item: item.name)


def natural_identity(subject: object, record_id: str, epoch_index: int) -> dict[str, Any]:
    return {
        "dataset": DATASET_ID,
        "subject_id": normalize_subject(subject),
        "record_id": str(record_id),
        "epoch_index": int(epoch_index),
    }


def identity_tuple(identity: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(identity["dataset"]),
        normalize_subject(identity["subject_id"]),
        str(identity["record_id"]),
        int(identity["epoch_index"]),
    )


def load_raw_x_blind(
    records: Mapping[str, Sequence[Path]], subjects: Iterable[object]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Read only ``x`` and filename-derived identities; never index ``y``."""
    rows: list[np.ndarray] = []
    identities: list[dict[str, Any]] = []
    for path in files_for_subjects(records, subjects):
        subject, record_id = parse_sleepedf_record(path)
        with np.load(path, allow_pickle=False) as archive:
            if "x" not in archive.files:
                raise RuntimeError(f"raw record has no x array: {path}")
            x = np.array(archive["x"], copy=True)
        if x.ndim == 3:
            if x.shape[-1] != 1:
                raise RuntimeError(f"ambiguous raw shape: {path} {x.shape}")
            x = x.squeeze(-1)
        if x.ndim != 2 or x.shape[1] != 3000:
            raise RuntimeError(f"unexpected raw x shape: {path} {x.shape}")
        rows.append(x[:, None, None, :].astype(np.float32, copy=False))
        identities.extend(
            natural_identity(subject, record_id, epoch_index)
            for epoch_index in range(len(x))
        )
    if not rows:
        raise RuntimeError("blind raw request returned no rows")
    return np.concatenate(rows, axis=0), identities


def _load_feature_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("shards"), list):
        raise RuntimeError("feature manifest is malformed")
    return manifest_path, payload


def load_feature_x_blind(
    manifest_path: str | Path, subjects: Iterable[object]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Read deployable feature columns/IDs without indexing the label array."""
    path, manifest = _load_feature_manifest(manifest_path)
    requested = {normalize_subject(value) for value in subjects}
    entries = {
        normalize_subject(item["subject_id"]): item
        for item in manifest["shards"]
        if normalize_subject(item["subject_id"]) in requested
    }
    if set(entries) != requested:
        raise RuntimeError("feature manifest lacks a requested blind subject")
    feature_rows: list[np.ndarray] = []
    identities: list[dict[str, Any]] = []
    for subject in sorted(requested):
        entry = entries[subject]
        shard = (path.parent / entry["path"]).resolve()
        if sha256_file(shard) != entry["sha256"]:
            raise RuntimeError(f"feature shard hash mismatch: {shard}")
        with np.load(shard, allow_pickle=False) as archive:
            required = {
                "features",
                "datasets",
                "subject_ids",
                "record_ids",
                "epoch_indices",
            }
            if not required <= set(archive.files):
                raise RuntimeError(f"feature shard blind schema incomplete: {shard}")
            features = np.array(archive["features"], copy=True)
            datasets = np.array(archive["datasets"], copy=True)
            subject_ids = np.array(archive["subject_ids"], copy=True)
            record_ids = np.array(archive["record_ids"], copy=True)
            epoch_indices = np.array(archive["epoch_indices"], copy=True)
        n_rows = len(features)
        if features.shape != (n_rows, 35):
            raise RuntimeError("blind feature matrix is not N x 35")
        if any(len(values) != n_rows for values in (datasets, subject_ids, record_ids, epoch_indices)):
            raise RuntimeError("feature identity columns do not align")
        for dataset, item_subject, record, epoch in zip(
            datasets, subject_ids, record_ids, epoch_indices
        ):
            identity = natural_identity(item_subject, str(record), int(epoch))
            if str(dataset) != DATASET_ID or identity["subject_id"] != subject:
                raise RuntimeError("feature identity contradicts its manifest shard")
            identities.append(identity)
        feature_rows.append(features.astype(np.float32, copy=False))
    return np.concatenate(feature_rows, axis=0), identities


def load_labels_authorized(
    records: Mapping[str, Sequence[Path]],
    subjects: Iterable[object],
    *,
    ledger: RoleLedger,
    phase: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    subject_tuple = ledger.authorize_label_load(subjects, phase)
    labels: list[np.ndarray] = []
    identities: list[dict[str, Any]] = []
    for path in files_for_subjects(records, subject_tuple):
        subject, record_id = parse_sleepedf_record(path)
        with np.load(path, allow_pickle=False) as archive:
            if "y" not in archive.files:
                raise RuntimeError(f"raw record has no label array: {path}")
            y = np.array(archive["y"], copy=True).astype(np.int64)
        if y.ndim != 1 or np.any((y < 0) | (y > 4)):
            raise RuntimeError(f"invalid label array: {path}")
        labels.append(y)
        identities.extend(
            natural_identity(subject, record_id, epoch_index)
            for epoch_index in range(len(y))
        )
    return np.concatenate(labels), identities


def align_values(
    source_ids: Sequence[Mapping[str, Any]],
    values: np.ndarray,
    target_ids: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    array = np.asarray(values)
    if len(source_ids) != len(array):
        raise RuntimeError("source identities and values do not align")
    keys = [identity_tuple(identity) for identity in source_ids]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate source identity")
    index = {key: row for row, key in enumerate(keys)}
    target_keys = [identity_tuple(identity) for identity in target_ids]
    if len(set(target_keys)) != len(target_keys):
        raise RuntimeError("duplicate target identity")
    missing = set(target_keys) - set(index)
    extra = set(index) - set(target_keys)
    if missing or extra:
        raise RuntimeError(
            f"identity join is not one-to-one: missing={len(missing)}, extra={len(extra)}"
        )
    return array[[index[key] for key in target_keys]]


__all__ = [
    "DATASET_ID",
    "align_values",
    "atomic_write_json",
    "atomic_write_npz",
    "canonical_json_hash",
    "discover_raw_records",
    "files_for_subjects",
    "identity_tuple",
    "load_feature_x_blind",
    "load_labels_authorized",
    "load_raw_x_blind",
    "natural_identity",
    "sha256_file",
]
