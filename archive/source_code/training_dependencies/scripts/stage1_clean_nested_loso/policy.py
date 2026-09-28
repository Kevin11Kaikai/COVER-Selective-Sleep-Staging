"""Fail-closed implementations of the two Stage-1 routing policies.

The thesis policy is Type-1 disagreement plus forward KL(P_T || P_F).  The
historical Type-1 plus signed confidence-gap rule is intentionally exposed
under a different function and result type so the two policies cannot be
silently interchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


DEFAULT_TAU_H = 0.9
DEFAULT_KAPPA = 0.15
DEFAULT_CONFIDENCE_GAP = 0.2
DEFAULT_KL_EPSILON = 1e-10
DEFAULT_COHERENCE_ATOL = 1e-8
DEFAULT_PROBABILITY_ATOL = 1e-6

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class PolicyValidationError(ValueError):
    """Raised when routing inputs violate the frozen prediction contract."""


@dataclass(frozen=True)
class Type1ForwardKLRouting:
    """Masks and scores produced by the canonical Type-1 + forward-KL gate."""

    policy_name: str
    tau_h: float
    kappa: float
    y_pred_t: IntArray
    y_pred_f: IntArray
    confidence_t: FloatArray
    kl_t_to_f: FloatArray
    high_confidence: BoolArray
    b1: BoolArray
    b2: BoolArray
    b3: BoolArray
    escalate: BoolArray
    accept: BoolArray


@dataclass(frozen=True)
class Type1ConfidenceGapRouting:
    """Masks from the distinct Type-1 + signed confidence-gap policy."""

    policy_name: str
    tau_h: float
    gap_threshold: float
    y_pred_t: IntArray
    y_pred_f: IntArray
    confidence_t: FloatArray
    confidence_f: FloatArray
    signed_confidence_gap: FloatArray
    high_confidence: BoolArray
    type1_disagreement: BoolArray
    agreement_gap_trigger: BoolArray
    agreement_below_gap: BoolArray
    escalate: BoolArray
    accept: BoolArray


def _validated_probabilities(values: object, *, name: str) -> FloatArray:
    try:
        probabilities = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PolicyValidationError(f"{name} must be a numeric probability matrix") from exc

    if probabilities.ndim != 2:
        raise PolicyValidationError(
            f"{name} must have shape (n_rows, n_classes); got {probabilities.shape}"
        )
    if probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
        raise PolicyValidationError(
            f"{name} requires at least one row and two classes; got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise PolicyValidationError(f"{name} contains NaN or infinite values")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise PolicyValidationError(f"{name} contains values outside [0, 1]")

    row_sums = probabilities.sum(axis=1)
    if not np.isfinite(row_sums).all() or np.any(row_sums <= 0):
        raise PolicyValidationError(f"{name} contains a row with non-positive total mass")
    if not np.allclose(
        row_sums,
        1.0,
        rtol=0.0,
        atol=DEFAULT_PROBABILITY_ATOL,
    ):
        raise PolicyValidationError(f"{name} rows must sum to one")
    return probabilities.copy()


def _validated_predictions(
    values: object, *, n_rows: int, n_classes: int, name: str
) -> IntArray:
    predictions = np.asarray(values)
    if predictions.ndim != 1 or predictions.shape[0] != n_rows:
        raise PolicyValidationError(
            f"{name} must have shape ({n_rows},); got {predictions.shape}"
        )
    if not np.issubdtype(predictions.dtype, np.integer):
        raise PolicyValidationError(f"{name} must contain integer class indices")

    predictions = predictions.astype(np.int64, copy=True)
    if np.any(predictions < 0) or np.any(predictions >= n_classes):
        raise PolicyValidationError(
            f"{name} contains a class outside [0, {n_classes - 1}]"
        )
    return predictions


def _validated_confidence(values: object, *, n_rows: int, name: str) -> FloatArray:
    try:
        confidence = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PolicyValidationError(f"{name} must be a numeric vector") from exc
    if confidence.ndim != 1 or confidence.shape[0] != n_rows:
        raise PolicyValidationError(
            f"{name} must have shape ({n_rows},); got {confidence.shape}"
        )
    if not np.isfinite(confidence).all():
        raise PolicyValidationError(f"{name} contains NaN or infinite values")
    return confidence.copy()


def _validated_unit_threshold(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise PolicyValidationError(f"{name} must be finite and in [0, 1]")
    return value


def _validated_nonnegative_threshold(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise PolicyValidationError(f"{name} must be finite and non-negative")
    return value


def validate_prediction_coherence(
    probabilities: object,
    predictions: object,
    confidence: object,
    *,
    name: str = "primary",
    confidence_atol: float = DEFAULT_COHERENCE_ATOL,
) -> tuple[FloatArray, IntArray, FloatArray]:
    """Validate argmax prediction and max-probability confidence coherence.

    The comparison is against the stored probability matrix itself, before the
    epsilon smoothing used only by the KL calculation.
    """

    confidence_atol = float(confidence_atol)
    if not np.isfinite(confidence_atol) or confidence_atol < 0:
        raise PolicyValidationError("confidence_atol must be finite and non-negative")

    probs = _validated_probabilities(probabilities, name=f"{name}_probabilities")
    pred = _validated_predictions(
        predictions,
        n_rows=probs.shape[0],
        n_classes=probs.shape[1],
        name=f"{name}_predictions",
    )
    conf = _validated_confidence(
        confidence, n_rows=probs.shape[0], name=f"{name}_confidence"
    )

    expected_predictions = np.argmax(probs, axis=1)
    prediction_mismatch = pred != expected_predictions
    if np.any(prediction_mismatch):
        rows = np.flatnonzero(prediction_mismatch)[:5].tolist()
        raise PolicyValidationError(
            f"{name}_predictions != argmax({name}_probabilities) at rows {rows}"
        )

    expected_confidence = np.max(probs, axis=1)
    confidence_mismatch = ~np.isclose(
        conf, expected_confidence, rtol=0.0, atol=confidence_atol
    )
    if np.any(confidence_mismatch):
        rows = np.flatnonzero(confidence_mismatch)[:5].tolist()
        max_gap = float(np.max(np.abs(conf - expected_confidence)))
        raise PolicyValidationError(
            f"{name}_confidence != max({name}_probabilities) at rows {rows}; "
            f"max_abs_gap={max_gap:.12g}"
        )
    return probs, pred, conf


def _validated_feature_predictions(
    probabilities: object, predictions: object, *, n_rows: int, n_classes: int
) -> tuple[FloatArray, IntArray]:
    probs = _validated_probabilities(probabilities, name="feature_probabilities")
    if probs.shape != (n_rows, n_classes):
        raise PolicyValidationError(
            "primary_probabilities and feature_probabilities must have identical shape; "
            f"got {(n_rows, n_classes)} and {probs.shape}"
        )
    pred = _validated_predictions(
        predictions,
        n_rows=n_rows,
        n_classes=n_classes,
        name="feature_predictions",
    )
    expected = np.argmax(probs, axis=1)
    mismatch = pred != expected
    if np.any(mismatch):
        rows = np.flatnonzero(mismatch)[:5].tolist()
        raise PolicyValidationError(
            f"feature_predictions != argmax(feature_probabilities) at rows {rows}"
        )
    return probs, pred


def _forward_kl_validated(
    probabilities_t: FloatArray,
    probabilities_f: FloatArray,
    *,
    epsilon: float,
) -> FloatArray:
    p = probabilities_t + epsilon
    q = probabilities_f + epsilon
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.sum(p * np.log(p / q), axis=1)


def forward_kl(
    probabilities_t: object,
    probabilities_f: object,
    *,
    epsilon: float = DEFAULT_KL_EPSILON,
) -> FloatArray:
    """Compute the frozen forward direction KL(P_T || P_F)."""

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise PolicyValidationError("epsilon must be finite and strictly positive")
    p_t = _validated_probabilities(probabilities_t, name="primary_probabilities")
    p_f = _validated_probabilities(probabilities_f, name="feature_probabilities")
    if p_t.shape != p_f.shape:
        raise PolicyValidationError(
            "primary_probabilities and feature_probabilities must have identical shape; "
            f"got {p_t.shape} and {p_f.shape}"
        )
    return _forward_kl_validated(p_t, p_f, epsilon=epsilon)


def route_type1_forward_kl(
    *,
    probabilities_t: object,
    predictions_t: object,
    confidence_t: object,
    probabilities_f: object,
    predictions_f: object,
    tau_h: float = DEFAULT_TAU_H,
    kappa: float = DEFAULT_KAPPA,
    epsilon: float = DEFAULT_KL_EPSILON,
    confidence_atol: float = DEFAULT_COHERENCE_ATOL,
) -> Type1ForwardKLRouting:
    """Apply the canonical Type-1 + forward-KL policy.

    Equality belongs to H and B2.  Every low-confidence row is accepted.
    """

    tau_h = _validated_unit_threshold(tau_h, name="tau_h")
    kappa = _validated_nonnegative_threshold(kappa, name="kappa")
    epsilon = _validated_nonnegative_threshold(epsilon, name="epsilon")
    if epsilon == 0:
        raise PolicyValidationError("epsilon must be strictly positive")

    p_t, y_t, c_t = validate_prediction_coherence(
        probabilities_t,
        predictions_t,
        confidence_t,
        name="primary",
        confidence_atol=confidence_atol,
    )
    p_f, y_f = _validated_feature_predictions(
        probabilities_f,
        predictions_f,
        n_rows=p_t.shape[0],
        n_classes=p_t.shape[1],
    )

    kl_t_to_f = _forward_kl_validated(p_t, p_f, epsilon=epsilon)
    high_confidence = c_t >= tau_h
    b1 = y_t != y_f
    agreement = ~b1
    b2 = agreement & (kl_t_to_f >= kappa)
    b3 = agreement & (kl_t_to_f < kappa)
    escalate = high_confidence & (b1 | b2)
    accept = ~escalate

    return Type1ForwardKLRouting(
        policy_name="type1_forward_kl",
        tau_h=tau_h,
        kappa=kappa,
        y_pred_t=y_t,
        y_pred_f=y_f,
        confidence_t=c_t,
        kl_t_to_f=kl_t_to_f,
        high_confidence=high_confidence,
        b1=b1,
        b2=b2,
        b3=b3,
        escalate=escalate,
        accept=accept,
    )


def route_type1_confidence_gap(
    *,
    probabilities_t: object,
    predictions_t: object,
    confidence_t: object,
    probabilities_f: object,
    predictions_f: object,
    confidence_f: object,
    tau_h: float = DEFAULT_TAU_H,
    gap_threshold: float = DEFAULT_CONFIDENCE_GAP,
    confidence_atol: float = DEFAULT_COHERENCE_ATOL,
) -> Type1ConfidenceGapRouting:
    """Apply the separately named Type-1 + signed confidence-gap policy."""

    tau_h = _validated_unit_threshold(tau_h, name="tau_h")
    gap_threshold = _validated_nonnegative_threshold(
        gap_threshold, name="gap_threshold"
    )
    p_t, y_t, c_t = validate_prediction_coherence(
        probabilities_t,
        predictions_t,
        confidence_t,
        name="primary",
        confidence_atol=confidence_atol,
    )
    p_f, y_f, c_f = validate_prediction_coherence(
        probabilities_f,
        predictions_f,
        confidence_f,
        name="feature",
        confidence_atol=confidence_atol,
    )
    if p_t.shape != p_f.shape:
        raise PolicyValidationError(
            "primary_probabilities and feature_probabilities must have identical shape; "
            f"got {p_t.shape} and {p_f.shape}"
        )

    high_confidence = c_t >= tau_h
    type1_disagreement = y_t != y_f
    agreement = ~type1_disagreement
    signed_gap = c_t - c_f
    agreement_gap_trigger = agreement & (signed_gap >= gap_threshold)
    agreement_below_gap = agreement & (signed_gap < gap_threshold)
    escalate = high_confidence & (type1_disagreement | agreement_gap_trigger)
    accept = ~escalate

    return Type1ConfidenceGapRouting(
        policy_name="type1_signed_confidence_gap",
        tau_h=tau_h,
        gap_threshold=gap_threshold,
        y_pred_t=y_t,
        y_pred_f=y_f,
        confidence_t=c_t,
        confidence_f=c_f,
        signed_confidence_gap=signed_gap,
        high_confidence=high_confidence,
        type1_disagreement=type1_disagreement,
        agreement_gap_trigger=agreement_gap_trigger,
        agreement_below_gap=agreement_below_gap,
        escalate=escalate,
        accept=accept,
    )
