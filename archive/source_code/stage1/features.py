"""Frozen deploy-time risk feature transformations."""

from __future__ import annotations

import numpy as np


EPSILON = 1e-10


def _validate_probability(values: np.ndarray, name: str) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != 5 or not len(probability):
        raise ValueError(f"{name} must be nonempty N x 5")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0):
        raise ValueError(f"{name} contains invalid values")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError(f"{name} rows do not sum to one")
    return probability


def _summary(probability: np.ndarray) -> tuple[np.ndarray, ...]:
    prediction = np.argmax(probability, axis=1)
    order = np.argsort(-probability, axis=1, kind="stable")
    top = probability[np.arange(len(probability)), order[:, 0]]
    runner = probability[np.arange(len(probability)), order[:, 1]]
    margin = top - runner
    clipped = np.clip(probability, EPSILON, 1.0)
    entropy = -np.sum(probability * np.log(clipped), axis=1) / np.log(5.0)
    one_hot = np.eye(5, dtype=np.float64)[prediction]
    return prediction, one_hot, top, runner, margin, entropy


def forward_kl(primary: np.ndarray, feature: np.ndarray) -> np.ndarray:
    p = _validate_probability(primary, "primary")
    q = _validate_probability(feature, "feature")
    if p.shape != q.shape:
        raise ValueError("KL probability shapes differ")
    return np.sum(p * (np.log(np.clip(p, EPSILON, 1.0)) - np.log(np.clip(q, EPSILON, 1.0))), axis=1)


def primary_only_features(primary: np.ndarray) -> tuple[np.ndarray, list[str]]:
    probability = _validate_probability(primary, "primary")
    _, one_hot, msp, runner, margin, entropy = _summary(probability)
    matrix = np.column_stack((probability, one_hot, msp, runner, margin, entropy))
    names = [f"primary_p_{index}" for index in range(5)]
    names += [f"primary_pred_is_{index}" for index in range(5)]
    names += ["primary_msp", "primary_runner_up", "primary_margin", "primary_entropy_normalized"]
    if matrix.shape[1] != 14:
        raise AssertionError("primary feature count changed")
    return matrix, names


def heterogeneous_features(
    primary: np.ndarray, feature: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    p = _validate_probability(primary, "primary")
    q = _validate_probability(feature, "feature")
    if p.shape != q.shape:
        raise ValueError("expert probability shapes differ")
    base, names = primary_only_features(p)
    p_pred, _, _, _, _, _ = _summary(p)
    q_pred, q_one_hot, q_msp, q_runner, q_margin, q_entropy = _summary(q)
    fwd = forward_kl(p, q)
    rev = forward_kl(q, p)
    midpoint = 0.5 * (p + q)
    js = 0.5 * forward_kl(p, midpoint) + 0.5 * forward_kl(q, midpoint)
    extra = np.column_stack(
        (
            q,
            q_one_hot,
            q_msp,
            q_runner,
            q_margin,
            q_entropy,
            fwd,
            rev,
            0.5 * (fwd + rev),
            js,
            np.sum(np.abs(p - q), axis=1),
            np.sqrt(np.sum((p - q) ** 2, axis=1)),
            (p_pred != q_pred).astype(np.float64),
            p[np.arange(len(p)), q_pred],
            q[np.arange(len(q)), p_pred],
        )
    )
    extra_names = [f"feature_p_{index}" for index in range(5)]
    extra_names += [f"feature_pred_is_{index}" for index in range(5)]
    extra_names += [
        "feature_msp", "feature_runner_up", "feature_margin", "feature_entropy_normalized",
        "forward_kl", "reverse_kl", "symmetric_kl", "jensen_shannon",
        "probability_l1", "probability_l2", "label_disagreement",
        "primary_probability_at_feature_prediction", "feature_probability_at_primary_prediction",
    ]
    matrix = np.column_stack((base, extra))
    if matrix.shape[1] != 37 or len(names + extra_names) != 37:
        raise AssertionError("heterogeneous feature count changed")
    return matrix, names + extra_names


__all__ = [
    "EPSILON",
    "forward_kl",
    "heterogeneous_features",
    "primary_only_features",
]

