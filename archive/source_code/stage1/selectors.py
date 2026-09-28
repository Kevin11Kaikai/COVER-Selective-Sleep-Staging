"""Exact nested candidate policies and subject-aware risk calculations."""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np


Q_GRID = tuple(Fraction(index, 20) for index in range(1, 21))


def subject_equal_weights(subject_ids: Sequence[object]) -> np.ndarray:
    subjects = np.asarray([str(value) for value in subject_ids])
    if subjects.ndim != 1 or not len(subjects):
        raise ValueError("subject IDs must be nonempty one-dimensional")
    unique, counts = np.unique(subjects, return_counts=True)
    by_subject = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / by_subject[value] for value in subjects], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return weights


def _identity_text(identity: Mapping[str, Any] | str) -> str:
    if isinstance(identity, str):
        if not identity:
            raise ValueError("identity string is empty")
        return identity
    required = ("dataset", "subject_id", "record_id", "epoch_index")
    if any(name not in identity for name in required):
        raise ValueError("identity mapping is incomplete")
    return "|".join(str(identity[name]) for name in required)


def tie_uniform(
    identity: Mapping[str, Any] | str,
    *,
    method_id: str,
    protocol_hash: str,
    outer: str,
) -> Fraction:
    identity_text = _identity_text(identity)
    parts = identity_text.split("|")
    if len(parts) == 4:
        dataset, subject, record, epoch = parts
    else:
        dataset, subject, record, epoch = "identity", "unknown", identity_text, "0"
    text = (
        f"stage2-tie-v2|protocol={protocol_hash}|outer={outer}|dataset={dataset}|"
        f"subject={subject}|record={record}|epoch={epoch}|method={method_id}"
    )
    integer = int.from_bytes(hashlib.sha256(text.encode()).digest(), "big")
    return Fraction(integer, 2**256)


def build_acceptance_mask(
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    tie_fraction: float | Fraction,
    identities: Sequence[Mapping[str, Any] | str],
    method_id: str,
    protocol_hash: str,
    outer: str,
) -> np.ndarray:
    risk = np.asarray(scores, dtype=np.float64)
    if risk.ndim != 1 or len(risk) != len(identities) or not np.all(np.isfinite(risk)):
        raise ValueError("scores/identities invalid")
    fraction = (
        tie_fraction
        if isinstance(tie_fraction, Fraction)
        else Fraction(str(float(tie_fraction)))
    )
    if not 0 <= fraction <= 1:
        raise ValueError("tie fraction outside [0,1]")
    accepted = risk < threshold
    tied = np.flatnonzero(risk == threshold)
    for index in tied:
        accepted[index] = tie_uniform(
            identities[index],
            method_id=method_id,
            protocol_hash=protocol_hash,
            outer=outer,
        ) < fraction
    return accepted


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def build_candidate_policies(
    scores: Sequence[float] | np.ndarray,
    subject_ids: Sequence[object],
) -> list[dict[str, object]]:
    risk = np.asarray(scores, dtype=np.float64)
    subjects = np.asarray([str(value) for value in subject_ids])
    if risk.ndim != 1 or len(risk) != len(subjects) or not len(risk):
        raise ValueError("FIT scores and subjects must align")
    if not np.all(np.isfinite(risk)) or len(set(subjects)) != 15:
        raise ValueError("candidate construction requires 15 FIT subjects")
    counts = {subject: int(np.sum(subjects == subject)) for subject in set(subjects)}
    row_mass = [Fraction(1, 15 * counts[subject]) for subject in subjects]
    unique_scores = np.unique(risk)
    policies: list[dict[str, object]] = []
    for q in Q_GRID:
        below = Fraction(0, 1)
        selected_threshold: float | None = None
        selected_fraction = Fraction(0, 1)
        for threshold in unique_scores:
            equal = sum(
                (row_mass[index] for index in np.flatnonzero(risk == threshold)),
                Fraction(0, 1),
            )
            if below <= q <= below + equal:
                selected_threshold = float(threshold)
                selected_fraction = (q - below) / equal
                break
            below += equal
        if selected_threshold is None:
            raise AssertionError("could not construct a q policy")
        policies.append(
            {
                "q": _fraction_payload(q),
                "q_float": float(q),
                "threshold": selected_threshold,
                "tie_fraction": _fraction_payload(selected_fraction),
                "tie_fraction_float": float(selected_fraction),
            }
        )
    return policies


def subject_equal_risk(
    accept: Sequence[bool] | np.ndarray,
    errors: Sequence[int] | np.ndarray,
    subject_ids: Sequence[object],
) -> dict[str, object]:
    accepted = np.asarray(accept, dtype=bool)
    outcome = np.asarray(errors, dtype=np.int64)
    subjects = np.asarray([str(value) for value in subject_ids])
    if accepted.shape != outcome.shape or accepted.shape != subjects.shape or accepted.ndim != 1:
        raise ValueError("selector rows do not align")
    if np.any((outcome != 0) & (outcome != 1)):
        raise ValueError("errors must be binary")
    c_values: list[Fraction] = []
    q_values: list[Fraction] = []
    per_subject: dict[str, dict[str, object]] = {}
    for subject in sorted(set(subjects)):
        mask = subjects == subject
        n = int(mask.sum())
        a = int(np.sum(accepted & mask))
        k = int(np.sum(accepted & mask & (outcome == 1)))
        c = Fraction(a, n)
        q = Fraction(k, n)
        c_values.append(c)
        q_values.append(q)
        per_subject[subject] = {
            "eligible": n,
            "accepted": a,
            "accepted_errors": k,
            "coverage": _fraction_payload(c),
            "risk": None if a == 0 else _fraction_payload(Fraction(k, a)),
        }
    c_bar = sum(c_values, Fraction(0, 1)) / len(c_values)
    q_bar = sum(q_values, Fraction(0, 1)) / len(q_values)
    risk = None if c_bar == 0 else q_bar / c_bar
    return {
        "coverage": _fraction_payload(c_bar),
        "error_mass": _fraction_payload(q_bar),
        "risk": None if risk is None else _fraction_payload(risk),
        "coverage_float": float(c_bar),
        "risk_float": None if risk is None else float(risk),
        "per_subject": per_subject,
    }


def hoeffding_ratio_ucb(
    coverage_mean: float,
    error_mass_mean: float,
    *,
    m: int = 4,
    candidates: int = 20,
    delta: float = 0.05,
) -> float | None:
    if not 0 <= coverage_mean <= 1 or not 0 <= error_mass_mean <= coverage_mean:
        raise ValueError("invalid subject means")
    if m <= 0 or candidates <= 0 or not 0 < delta < 1:
        raise ValueError("invalid UCB parameters")
    h = math.sqrt(math.log(2 * candidates / delta) / (2 * m))
    if coverage_mean <= h:
        return None
    return min(1.0, (error_mass_mean + h) / (coverage_mean - h))


def select_empirical_policy(
    policy_results: Sequence[Mapping[str, object]],
    *,
    alpha: float,
) -> Mapping[str, object] | None:
    def exact(value: object) -> Fraction:
        if isinstance(value, Fraction):
            return value
        if isinstance(value, Mapping):
            return Fraction(int(value["numerator"]), int(value["denominator"]))
        if isinstance(value, (float, np.floating)):
            return Fraction(str(float(value)))
        return Fraction(value)  # type: ignore[arg-type]

    alpha_exact = exact(alpha)
    if not 0 < alpha_exact < 1:
        raise ValueError("alpha outside (0,1)")
    feasible = [
        row
        for row in policy_results
        if bool(row.get("supported"))
        and row.get("risk") is not None
        and exact(row["risk"]) <= alpha_exact
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda row: (
            -exact(row["coverage"]),
            exact(row["risk"]),
            exact(row["q"]),
        ),
    )


__all__ = [
    "Q_GRID",
    "build_acceptance_mask",
    "build_candidate_policies",
    "hoeffding_ratio_ucb",
    "select_empirical_policy",
    "subject_equal_risk",
    "subject_equal_weights",
    "tie_uniform",
]
