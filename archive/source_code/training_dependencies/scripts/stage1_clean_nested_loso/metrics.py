"""Single-source Stage-1 metrics and TDR-2 fixed-budget arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .policy import Type1ForwardKLRouting


class RatioStatus(str, Enum):
    """Validity/support state for every reported ratio."""

    DEFINED = "DEFINED"
    LOW_SUPPORT = "LOW_SUPPORT"
    UNDEFINED = "UNDEFINED"


@dataclass(frozen=True)
class RatioMetric:
    numerator: int
    denominator: int
    value: float | None
    status: RatioStatus


@dataclass(frozen=True)
class GateMetrics:
    n_rows: int
    hc_risk: RatioMetric
    accepted_hce_mass: RatioMetric
    captured_hce_mass: RatioMetric
    hce_capture_recall: RatioMetric
    hc_escalation_mass: RatioMetric
    hc_escalation_rate: RatioMetric
    overall_escalation_rate: RatioMetric
    ordinary_selective_risk: RatioMetric
    repository_accept_hce_rate: RatioMetric
    q1: RatioMetric
    q2: RatioMetric
    q3: RatioMetric
    eta1: RatioMetric
    eta2: RatioMetric
    eta3: RatioMetric


@dataclass(frozen=True)
class TDR2GapIdentity:
    current_budget: float
    comparator_budget: float
    current_capture: float
    comparator_capture: float
    observed_gap: float
    identity_rhs: float
    residual: float


@dataclass(frozen=True)
class FractionalBoundaryResult:
    budget: float
    expected_capture: float
    escalation_probability: tuple[float, ...]
    acceptance_probability: tuple[float, ...]
    fractional_boundary: bool
    tie_at_boundary: bool


def _validated_min_support(min_support: int) -> int:
    if isinstance(min_support, bool) or not isinstance(min_support, (int, np.integer)):
        raise ValueError("min_support must be a positive integer")
    min_support = int(min_support)
    if min_support <= 0:
        raise ValueError("min_support must be a positive integer")
    return min_support


def ratio_metric(
    numerator: int,
    denominator: int,
    *,
    min_support: int = 1,
    support: int | None = None,
) -> RatioMetric:
    """Create a ratio without silently converting undefined values to zero.

    ``support`` defaults to the denominator.  Unconditional regional masses may
    instead pass the regional count so a near-zero cell is marked LOW_SUPPORT
    while its mass remains mathematically defined.
    """

    min_support = _validated_min_support(min_support)
    if isinstance(numerator, bool) or isinstance(denominator, bool):
        raise ValueError("ratio counts must be integers, not booleans")
    numerator = int(numerator)
    denominator = int(denominator)
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("ratio counts require 0 <= numerator <= denominator")
    if denominator == 0:
        return RatioMetric(
            numerator=numerator,
            denominator=denominator,
            value=None,
            status=RatioStatus.UNDEFINED,
        )

    if support is None:
        support = denominator
    if isinstance(support, bool):
        raise ValueError("support must be a non-negative integer")
    support = int(support)
    if support < 0:
        raise ValueError("support must be a non-negative integer")
    status = (
        RatioStatus.LOW_SUPPORT if support < min_support else RatioStatus.DEFINED
    )
    return RatioMetric(
        numerator=numerator,
        denominator=denominator,
        value=float(numerator / denominator),
        status=status,
    )


def _validated_targets(values: object, *, n_rows: int) -> NDArray[np.int64]:
    targets = np.asarray(values)
    if targets.ndim != 1 or targets.shape[0] != n_rows:
        raise ValueError(f"y_true must have shape ({n_rows},); got {targets.shape}")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("y_true must contain integer class indices")
    return targets.astype(np.int64, copy=True)


def evaluate_policy_metrics(
    y_true: object,
    routing: Type1ForwardKLRouting,
    *,
    min_support: int = 1,
) -> GateMetrics:
    """Evaluate all frozen metrics with explicit, non-interchangeable denominators."""

    if not isinstance(routing, Type1ForwardKLRouting):
        raise TypeError("routing must be a Type1ForwardKLRouting result")
    min_support = _validated_min_support(min_support)
    n_rows = int(routing.y_pred_t.shape[0])
    targets = _validated_targets(y_true, n_rows=n_rows)

    error = routing.y_pred_t != targets
    high = routing.high_confidence
    accept = routing.accept
    escalate = routing.escalate
    hce = high & error
    accepted_hc = accept & high
    accepted_hce = accept & hce
    captured_hce = escalate & hce
    escalated_hc = escalate & high

    q_masks = (high & routing.b1, high & routing.b2, high & routing.b3)
    q_counts = tuple(int(mask.sum()) for mask in q_masks)
    eta_errors = tuple(int((mask & error).sum()) for mask in q_masks)

    q_metrics = tuple(
        ratio_metric(
            count,
            n_rows,
            min_support=min_support,
            support=count,
        )
        for count in q_counts
    )
    eta_metrics = tuple(
        ratio_metric(errors, count, min_support=min_support)
        for errors, count in zip(eta_errors, q_counts)
    )

    return GateMetrics(
        n_rows=n_rows,
        hc_risk=ratio_metric(
            int(accepted_hce.sum()),
            int(accepted_hc.sum()),
            min_support=min_support,
        ),
        accepted_hce_mass=ratio_metric(
            int(accepted_hce.sum()), n_rows, min_support=min_support
        ),
        captured_hce_mass=ratio_metric(
            int(captured_hce.sum()), n_rows, min_support=min_support
        ),
        hce_capture_recall=ratio_metric(
            int(captured_hce.sum()), int(hce.sum()), min_support=min_support
        ),
        hc_escalation_mass=ratio_metric(
            int(escalated_hc.sum()), n_rows, min_support=min_support
        ),
        hc_escalation_rate=ratio_metric(
            int(escalated_hc.sum()), int(high.sum()), min_support=min_support
        ),
        overall_escalation_rate=ratio_metric(
            int(escalate.sum()), n_rows, min_support=min_support
        ),
        ordinary_selective_risk=ratio_metric(
            int((accept & error).sum()), int(accept.sum()), min_support=min_support
        ),
        repository_accept_hce_rate=ratio_metric(
            int(accepted_hce.sum()), int(accept.sum()), min_support=min_support
        ),
        q1=q_metrics[0],
        q2=q_metrics[1],
        q3=q_metrics[2],
        eta1=eta_metrics[0],
        eta2=eta_metrics[1],
        eta3=eta_metrics[2],
    )


def _validated_regional_vectors(
    masses: object, risks: object
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    q = np.asarray(masses, dtype=np.float64)
    eta = np.asarray(risks, dtype=np.float64)
    if q.ndim != 1 or eta.ndim != 1 or q.shape != eta.shape or q.size == 0:
        raise ValueError("masses and risks must be non-empty vectors of equal shape")
    if not np.isfinite(q).all() or not np.isfinite(eta).all():
        raise ValueError("masses and risks must be finite")
    if np.any(q < 0) or float(q.sum()) > 1.0 + 1e-12:
        raise ValueError("masses must be non-negative and sum to at most one")
    if np.any(eta < 0) or np.any(eta > 1):
        raise ValueError("risks must lie in [0, 1]")
    return q, eta


def tdr2_same_budget_identity(
    masses: object,
    risks: object,
    acceptance_probability: object,
    *,
    atol: float = 1e-12,
) -> TDR2GapIdentity:
    """Validate the frozen TDR-2 same-budget gap identity for q3 > 0."""

    q, eta = _validated_regional_vectors(masses, risks)
    if q.shape != (3,):
        raise ValueError("TDR-2 requires exactly the B1, B2, and B3 regions")
    if q[2] <= 0:
        raise ValueError("the positive-B3 TDR-2 identity requires q3 > 0")
    acceptance = np.asarray(acceptance_probability, dtype=np.float64)
    if acceptance.shape != (3,) or not np.isfinite(acceptance).all():
        raise ValueError("acceptance_probability must be a finite length-three vector")
    if np.any(acceptance < 0) or np.any(acceptance > 1):
        raise ValueError("acceptance probabilities must lie in [0, 1]")
    atol = float(atol)
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")

    current = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    current_budget = float(np.sum(q * (1.0 - current)))
    comparator_budget = float(np.sum(q * (1.0 - acceptance)))
    if not np.isclose(current_budget, comparator_budget, rtol=0.0, atol=atol):
        raise ValueError(
            "comparator does not match the current high-confidence budget: "
            f"{comparator_budget:.12g} != {current_budget:.12g}"
        )

    current_capture = float(np.sum(q * (1.0 - current) * eta))
    comparator_capture = float(np.sum(q * (1.0 - acceptance) * eta))
    observed_gap = current_capture - comparator_capture
    identity_rhs = float(
        q[0] * acceptance[0] * (eta[0] - eta[2])
        + q[1] * acceptance[1] * (eta[1] - eta[2])
    )
    residual = observed_gap - identity_rhs
    if not np.isclose(observed_gap, identity_rhs, rtol=0.0, atol=atol):
        raise AssertionError(
            f"TDR-2 identity residual {residual:.12g} exceeds atol={atol:.12g}"
        )
    return TDR2GapIdentity(
        current_budget=current_budget,
        comparator_budget=comparator_budget,
        current_capture=current_capture,
        comparator_capture=comparator_capture,
        observed_gap=observed_gap,
        identity_rhs=identity_rhs,
        residual=residual,
    )


def expected_fractional_boundary_policy(
    masses: object,
    risks: object,
    budget: float,
    *,
    atol: float = 1e-12,
) -> FractionalBoundaryResult:
    """Compute expected fixed-budget capture with a fractional boundary cell.

    This returns regional probabilities only.  It does not sample rows and
    therefore cannot introduce label-dependent boundary randomness.
    """

    q, eta = _validated_regional_vectors(masses, risks)
    budget = float(budget)
    atol = float(atol)
    if not np.isfinite(budget) or budget < -atol or budget > float(q.sum()) + atol:
        raise ValueError("budget must lie between zero and total regional mass")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")
    budget = min(max(budget, 0.0), float(q.sum()))

    order = sorted(range(q.size), key=lambda index: (-eta[index], index))
    escalation = np.zeros(q.size, dtype=np.float64)
    remaining = budget
    for index in order:
        if remaining <= atol:
            break
        if q[index] <= 0:
            continue
        take = min(remaining, float(q[index]))
        escalation[index] = take / float(q[index])
        remaining -= take
    if remaining > atol:
        raise AssertionError("fractional allocation failed to exhaust the budget")

    expected_capture = float(np.sum(q * escalation * eta))
    fractional_boundary = bool(np.any((escalation > atol) & (escalation < 1.0 - atol)))
    tie_at_boundary = False
    for selected in range(q.size):
        if q[selected] <= 0 or escalation[selected] <= atol:
            continue
        for unselected in range(q.size):
            if (
                selected != unselected
                and q[unselected] > 0
                and escalation[unselected] < 1.0 - atol
                and np.isclose(eta[selected], eta[unselected], rtol=0.0, atol=atol)
            ):
                tie_at_boundary = True
                break
        if tie_at_boundary:
            break

    return FractionalBoundaryResult(
        budget=budget,
        expected_capture=expected_capture,
        escalation_probability=tuple(float(value) for value in escalation),
        acceptance_probability=tuple(float(1.0 - value) for value in escalation),
        fractional_boundary=fractional_boundary,
        tie_at_boundary=tie_at_boundary,
    )
