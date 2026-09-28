"""Acceptance-rule grids at arbitrary resolution.

`stage1_analysis.acceptance.build_candidate_policies` rescans every unique score for
every q, which costs O(n * |Pi|) Fraction additions: 4.5 s per candidate at |Pi|=20.
`dense_candidate_policies` walks the sorted score groups once with a shared cursor and
emits an identical payload, which is what makes a 128-candidate per-fold search and a
200-replicate permutation null affordable.

Grids are nested by construction: Fraction(i, 20) == Fraction(10 * i, 200), so the
policies for every report size are row subsets of the |Pi|=200 construction.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np


def grid_fractions(denominator: int) -> tuple[Fraction, ...]:
    """The q grid {1/d, ..., d/d}. d=20 reproduces acceptance.Q_GRID exactly."""
    if denominator < 1:
        raise ValueError("grid denominator must be positive")
    return tuple(Fraction(index, denominator) for index in range(1, denominator + 1))


def grid_is_nested(coarse: int, fine: int) -> bool:
    return fine % coarse == 0


def subset_policies(policies: Sequence[dict], size: int) -> list[dict]:
    """Rows of a finer grid that coincide with the |Pi|=size grid."""
    wanted = set(grid_fractions(size))
    rows = [p for p in policies if Fraction(p["q"]["numerator"], p["q"]["denominator"]) in wanted]
    if len(rows) != size:
        raise ValueError(f"grid of size {size} is not nested in the supplied policies")
    return rows


def _payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def dense_candidate_policies(scores: Sequence[float] | np.ndarray,
                             subject_ids: Sequence[object],
                             q_grid: Iterable[Fraction],
                             *, n_subjects: int = 15) -> list[dict]:
    """Subject-equal exact quantile policies, one forward pass over sorted scores.

    Each row carries mass 1/(n_subjects * count[subject]) so every subject contributes
    1/n_subjects in total. For each q the bracketing score group becomes the threshold
    and the partial fill becomes the tie fraction, matching acceptance.py:95-134.
    """
    risk = np.asarray(scores, dtype=np.float64)
    subjects = np.asarray([str(value) for value in subject_ids])
    if risk.ndim != 1 or len(risk) != len(subjects) or not len(risk):
        raise ValueError("scores and subjects must align")
    if not np.all(np.isfinite(risk)):
        raise ValueError("scores must be finite")
    if len(set(subjects.tolist())) != n_subjects:
        raise ValueError(f"expected {n_subjects} distinct subjects")

    counts = Counter(subjects.tolist())
    order = np.argsort(risk, kind="stable")
    ordered = risk[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    group_subjects = np.split(subjects[order], starts[1:])
    group_mass = [sum((Fraction(1, n_subjects * counts[s]) for s in block), Fraction(0, 1))
                  for block in group_subjects]
    group_value = ordered[starts]

    targets = sorted(q_grid)
    if not targets or targets[0] <= 0:
        raise ValueError("q grid must be positive")
    policies: list[dict] = []
    index = 0
    below = Fraction(0, 1)
    for q in targets:
        # Advance only while q lies strictly beyond this group's upper edge, so the
        # first group satisfying below <= q <= below + equal wins, as in the original.
        while index < len(group_mass) - 1 and below + group_mass[index] < q:
            below += group_mass[index]
            index += 1
        equal = group_mass[index]
        if not (below <= q <= below + equal):
            raise AssertionError("could not construct a q policy")
        fraction = (q - below) / equal
        policies.append({"q": _payload(q), "q_float": float(q),
                         "threshold": float(group_value[index]),
                         "tie_fraction": _payload(fraction),
                         "tie_fraction_float": float(fraction)})
    return policies
