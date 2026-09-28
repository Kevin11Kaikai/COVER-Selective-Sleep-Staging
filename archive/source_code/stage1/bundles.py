"""Label-free component prediction bundles and coherence checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class BlindBundleError(ValueError):
    """Raised when a supposedly sealed bundle is ambiguous or inconsistent."""


BLIND_KEYS = frozenset(
    {
        "identities_json",
        "logits",
        "primary_probabilities",
        "primary_predictions",
        "primary_confidences",
        "feature_probabilities",
        "feature_predictions",
        "feature_confidences",
        "primary_checkpoint_hash",
        "feature_model_hash",
        "manifest_json",
    }
)
FORBIDDEN_LABEL_KEYS = frozenset({"labels", "y", "y_true", "errors", "correct"})


def identity_key(identity: Mapping[str, Any]) -> tuple[str, str, str, int]:
    required = ("dataset", "subject_id", "record_id", "epoch_index")
    if set(identity) != set(required):
        raise BlindBundleError("identity keys differ from the blind schema")
    epoch = identity["epoch_index"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise BlindBundleError("epoch_index must be a nonnegative integer")
    values = tuple(str(identity[name]).strip() for name in required[:3])
    if any(not value for value in values):
        raise BlindBundleError("identity text fields must be nonempty")
    return values[0], values[1], values[2], epoch


def _parse_scalar_json(value: np.ndarray, name: str) -> Any:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise BlindBundleError(f"{name} must be a scalar string")
    raw = array.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _hash_scalar(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise BlindBundleError(f"{name} must be scalar")
    text = str(array.item()).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise BlindBundleError(f"{name} is not SHA-256")
    return text


def validate_blind_bundle(
    bundle: Mapping[str, Any],
    expected_ids: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    keys = set(bundle)
    if keys & FORBIDDEN_LABEL_KEYS:
        raise BlindBundleError("blind bundle contains a label/outcome key")
    missing = BLIND_KEYS - keys
    extra = keys - BLIND_KEYS
    if missing or extra:
        raise BlindBundleError(f"blind bundle keys differ: missing={sorted(missing)}, extra={sorted(extra)}")

    identities = _parse_scalar_json(np.asarray(bundle["identities_json"]), "identities_json")
    manifest = _parse_scalar_json(np.asarray(bundle["manifest_json"]), "manifest_json")
    if not isinstance(identities, list) or not identities:
        raise BlindBundleError("identities_json must be a nonempty list")
    keys_seen = [identity_key(item) for item in identities]
    if len(set(keys_seen)) != len(keys_seen):
        raise BlindBundleError("duplicate stable identity")
    if expected_ids is not None:
        expected_keys = [identity_key(item) for item in expected_ids]
        if keys_seen != expected_keys:
            raise BlindBundleError(
                "stable identities are missing, extra, duplicated, or reordered"
            )
    n_rows = len(keys_seen)
    logits = np.asarray(bundle["logits"])
    primary = np.asarray(bundle["primary_probabilities"])
    feature = np.asarray(bundle["feature_probabilities"])
    primary_prediction = np.asarray(bundle["primary_predictions"])
    feature_prediction = np.asarray(bundle["feature_predictions"])
    primary_confidence = np.asarray(bundle["primary_confidences"])
    feature_confidence = np.asarray(bundle["feature_confidences"])
    if logits.shape != (n_rows, 5) or primary.shape != (n_rows, 5) or feature.shape != (n_rows, 5):
        raise BlindBundleError("logit/probability shapes are not N x 5")
    for name, array in (
        ("primary_predictions", primary_prediction),
        ("feature_predictions", feature_prediction),
        ("primary_confidences", primary_confidence),
        ("feature_confidences", feature_confidence),
    ):
        if array.shape != (n_rows,):
            raise BlindBundleError(f"{name} shape is not N")
    for name, probability in (("primary", primary), ("feature", feature)):
        if not np.all(np.isfinite(probability)) or np.any(probability < 0) or np.any(probability > 1):
            raise BlindBundleError(f"{name} probabilities invalid")
        if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0):
            raise BlindBundleError(f"{name} probabilities do not sum to one")
    if not np.all(np.isfinite(logits)):
        raise BlindBundleError("logits contain nonfinite values")
    shifted = logits - logits.max(axis=1, keepdims=True)
    softmax = np.exp(shifted)
    softmax /= softmax.sum(axis=1, keepdims=True)
    if not np.allclose(primary, softmax, atol=1e-6, rtol=1e-6):
        raise BlindBundleError("primary logits/probability mismatch")
    if not np.array_equal(primary_prediction, np.argmax(primary, axis=1)):
        raise BlindBundleError("primary prediction/probability mismatch")
    if not np.array_equal(feature_prediction, np.argmax(feature, axis=1)):
        raise BlindBundleError("feature prediction/probability mismatch")
    if not np.allclose(primary_confidence, primary.max(axis=1), atol=1e-7, rtol=1e-6):
        raise BlindBundleError("primary confidence/probability mismatch")
    if not np.allclose(feature_confidence, feature.max(axis=1), atol=1e-7, rtol=1e-6):
        raise BlindBundleError("feature confidence/probability mismatch")
    primary_hash = _hash_scalar(
        np.asarray(bundle["primary_checkpoint_hash"]), "primary_checkpoint_hash"
    )
    feature_hash = _hash_scalar(
        np.asarray(bundle["feature_model_hash"]), "feature_model_hash"
    )
    if not isinstance(manifest, dict) or manifest.get("contains_labels") is not False:
        raise BlindBundleError("blind manifest does not explicitly forbid labels")
    roles = manifest.get("generating_roles")
    if not isinstance(roles, dict) or not roles:
        raise BlindBundleError("blind manifest has no generating-role provenance")
    hashes = manifest.get("hashes")
    required_hashes = {
        "primary_checkpoint",
        "feature_model",
        "data",
        "code",
        "config",
    }
    if not isinstance(hashes, dict) or not required_hashes <= set(hashes):
        raise BlindBundleError("blind manifest has incomplete provenance hashes")
    checked_hashes = {
        name: _hash_scalar(np.asarray(value), f"manifest.hashes.{name}")
        for name, value in hashes.items()
        if name in required_hashes
    }
    if checked_hashes["primary_checkpoint"] != primary_hash:
        raise BlindBundleError("primary checkpoint hash disagrees with manifest")
    if checked_hashes["feature_model"] != feature_hash:
        raise BlindBundleError("feature model hash disagrees with manifest")
    return {"n_rows": n_rows, "identities": identities, "manifest": manifest}


def load_blind_bundle(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    with np.load(file_path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    validate_blind_bundle(payload)
    return payload


__all__ = [
    "BLIND_KEYS",
    "BlindBundleError",
    "identity_key",
    "load_blind_bundle",
    "validate_blind_bundle",
]
