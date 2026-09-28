"""Public deterministic Stage-1 routing and metric primitives.

The package also contains explicitly invoked audit, cache, and single-fold run
modules. They are deliberately not imported here, so importing this public API
does not discover data, load artifacts, or start training.
"""

from .metrics import (
    FractionalBoundaryResult,
    GateMetrics,
    RatioMetric,
    RatioStatus,
    TDR2GapIdentity,
    evaluate_policy_metrics,
    expected_fractional_boundary_policy,
    ratio_metric,
    tdr2_same_budget_identity,
)
from .policy import (
    DEFAULT_CONFIDENCE_GAP,
    DEFAULT_KAPPA,
    DEFAULT_TAU_H,
    PolicyValidationError,
    Type1ConfidenceGapRouting,
    Type1ForwardKLRouting,
    forward_kl,
    route_type1_confidence_gap,
    route_type1_forward_kl,
    validate_prediction_coherence,
)

__all__ = [
    "DEFAULT_CONFIDENCE_GAP",
    "DEFAULT_KAPPA",
    "DEFAULT_TAU_H",
    "FractionalBoundaryResult",
    "GateMetrics",
    "PolicyValidationError",
    "RatioMetric",
    "RatioStatus",
    "TDR2GapIdentity",
    "Type1ConfidenceGapRouting",
    "Type1ForwardKLRouting",
    "evaluate_policy_metrics",
    "expected_fractional_boundary_policy",
    "forward_kl",
    "ratio_metric",
    "route_type1_confidence_gap",
    "route_type1_forward_kl",
    "tdr2_same_budget_identity",
    "validate_prediction_coherence",
]
