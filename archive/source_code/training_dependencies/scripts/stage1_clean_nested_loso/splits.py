"""Deterministic, subject-grouped splits for clean Stage-1 nested LOSO.

This module is deliberately independent of labels, predictions, and metrics.  A
split is a pure function of the declared subject IDs, outer subject, and seed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SPLIT_SCHEMA_VERSION = "stage1-clean-subject-split-v1"
DEFAULT_SPLIT_SEED = 20260816
ASSIGNMENT_RULE = (
    f"{SPLIT_SCHEMA_VERSION}: sort development subjects by "
    "(sha256(utf8(f'{schema_version}:{seed}:{outer_subject}:{subject_id}')), "
    "subject_id); "
    "then take contiguous roles in inner_train, checkpoint_selection, "
    "calibration, routing_selection order"
)
SELECTION_USE_KEYS = (
    "train",
    "early_stop",
    "checkpoint_selection",
    "calibration",
    "feature_expert_selection",
    "kappa_selection",
    "epsilon_selection",
    "metric_selection",
)


class SplitProtocolError(ValueError):
    """Raised when a subject split violates the frozen leakage protocol."""


def normalize_subject_id(value: Any) -> str:
    """Return the canonical subject identifier used by this campaign.

    Numeric Sleep-EDF identifiers are zero-padded to two digits; non-numeric
    identifiers are preserved after whitespace stripping.
    """

    if value is None or isinstance(value, bool):
        raise SplitProtocolError("subject IDs must be non-empty identifiers")
    if isinstance(value, int):
        if value < 0:
            raise SplitProtocolError("numeric subject IDs must be non-negative")
        return f"{value:02d}"
    subject_id = str(value).strip()
    if not subject_id:
        raise SplitProtocolError("subject IDs must be non-empty identifiers")
    if re.fullmatch(r"[0-9]+", subject_id):
        return f"{int(subject_id):02d}"
    return subject_id


def _normalize_subjects(
    values: Sequence[Any], field_name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    normalized = tuple(normalize_subject_id(value) for value in values)
    if not allow_empty and not normalized:
        raise SplitProtocolError(f"{field_name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise SplitProtocolError(f"{field_name} contains duplicate subject IDs")
    return normalized


def _validate_size(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SplitProtocolError(f"{field_name} must be a non-negative integer")
    return value


def _ordering_digest(seed: int, outer_subject: str, subject_id: str) -> str:
    payload = (
        f"{SPLIT_SCHEMA_VERSION}:{seed}:{outer_subject}:{subject_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_no_outer_leakage(
    outer_subject: Any,
    subject_sets: Mapping[str, Sequence[Any]],
) -> None:
    """Fail if the outer subject occurs in any train or selection use.

    All protocol uses are required.  Requiring the complete declaration prevents
    a caller from silently omitting a selection path from the leakage audit.
    """

    outer = normalize_subject_id(outer_subject)
    missing_uses = [key for key in SELECTION_USE_KEYS if key not in subject_sets]
    if missing_uses:
        raise SplitProtocolError(
            "leakage declaration is incomplete; missing uses: "
            + ", ".join(missing_uses)
        )
    for use in SELECTION_USE_KEYS:
        subjects = _normalize_subjects(tuple(subject_sets[use]), use, allow_empty=True)
        if outer in subjects:
            raise SplitProtocolError(
                f"outer subject {outer} appears in forbidden {use} subjects"
            )


@dataclass(frozen=True)
class SubjectSplit:
    """Frozen subject roles for one outer fold."""

    all_subjects: tuple[str, ...]
    outer_subject: str
    seed: int
    assignment_rule: str
    assignment_order: tuple[str, ...]
    inner_train: tuple[str, ...]
    checkpoint_selection: tuple[str, ...]
    calibration: tuple[str, ...]
    routing_selection: tuple[str, ...]

    @property
    def development_subjects(self) -> tuple[str, ...]:
        return tuple(subject for subject in self.all_subjects if subject != self.outer_subject)

    @property
    def roles(self) -> dict[str, tuple[str, ...]]:
        return {
            "inner_train": self.inner_train,
            "checkpoint_selection": self.checkpoint_selection,
            "calibration": self.calibration,
            "routing_selection": self.routing_selection,
        }

    @property
    def selection_uses(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": self.inner_train,
            "early_stop": self.checkpoint_selection,
            "checkpoint_selection": self.checkpoint_selection,
            "calibration": self.calibration,
            "feature_expert_selection": self.routing_selection,
            "kappa_selection": self.routing_selection,
            "epsilon_selection": self.routing_selection,
            "metric_selection": self.routing_selection,
        }

    @property
    def split_hash(self) -> str:
        payload = self.as_dict(include_hash=False)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "outer_subject": self.outer_subject,
            "seed": self.seed,
            "assignment_rule": self.assignment_rule,
            "all_subjects": list(self.all_subjects),
            "development_subjects": list(self.development_subjects),
            "assignment_order": list(self.assignment_order),
            "roles": {name: list(subjects) for name, subjects in self.roles.items()},
            "selection_uses": {
                name: list(subjects) for name, subjects in self.selection_uses.items()
            },
        }
        if include_hash:
            result["split_hash"] = self.split_hash
        return result


def validate_subject_split(split: SubjectSplit) -> None:
    """Validate exclusion, role disjointness, and full development coverage."""

    if not isinstance(split, SubjectSplit):
        raise SplitProtocolError("split must be a SubjectSplit")
    all_subjects = _normalize_subjects(split.all_subjects, "all_subjects")
    outer = normalize_subject_id(split.outer_subject)
    if outer not in all_subjects:
        raise SplitProtocolError("outer subject is not present in all_subjects")
    if isinstance(split.seed, bool) or not isinstance(split.seed, int):
        raise SplitProtocolError("seed must be an integer")
    if split.assignment_rule != ASSIGNMENT_RULE:
        raise SplitProtocolError("assignment rule does not match the frozen v1 rule")

    development = set(all_subjects) - {outer}
    seen: set[str] = set()
    for role_name, values in split.roles.items():
        subjects = set(_normalize_subjects(values, role_name, allow_empty=True))
        overlap = seen & subjects
        if overlap:
            raise SplitProtocolError(
                f"subject roles are not mutually exclusive; {role_name} overlap={sorted(overlap)}"
            )
        seen.update(subjects)
    missing = sorted(development - seen)
    extra = sorted(seen - development)
    if missing or extra:
        raise SplitProtocolError(
            f"subject roles do not exactly cover development subjects; "
            f"missing={missing}, extra={extra}"
        )

    order = _normalize_subjects(split.assignment_order, "assignment_order")
    if set(order) != development or len(order) != len(development):
        raise SplitProtocolError("assignment_order must contain every development subject once")
    expected_order = tuple(
        sorted(
            development,
            key=lambda subject: (_ordering_digest(split.seed, outer, subject), subject),
        )
    )
    if order != expected_order:
        raise SplitProtocolError("assignment_order does not match seed/outer hash ordering")
    validate_no_outer_leakage(outer, split.selection_uses)


def build_subject_split(
    outer_subject: Any,
    subjects: Sequence[Any] | None = None,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    inner_train_size: int = 13,
    checkpoint_selection_size: int = 2,
    calibration_size: int = 2,
    routing_selection_size: int = 2,
) -> SubjectSplit:
    """Construct the frozen deterministic split without observing any labels."""

    if subjects is None:
        subjects = tuple(range(20))
    all_subjects = _normalize_subjects(tuple(subjects), "subjects")
    outer = normalize_subject_id(outer_subject)
    if outer not in all_subjects:
        raise SplitProtocolError(f"outer subject {outer} is not in declared subjects")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitProtocolError("seed must be an integer")

    sizes = (
        _validate_size(inner_train_size, "inner_train_size"),
        _validate_size(checkpoint_selection_size, "checkpoint_selection_size"),
        _validate_size(calibration_size, "calibration_size"),
        _validate_size(routing_selection_size, "routing_selection_size"),
    )
    development = tuple(subject for subject in all_subjects if subject != outer)
    if sum(sizes) != len(development):
        raise SplitProtocolError(
            "role sizes must exactly cover development subjects; "
            f"sizes={sizes}, development_count={len(development)}"
        )
    order = tuple(
        sorted(
            development,
            key=lambda subject: (_ordering_digest(seed, outer, subject), subject),
        )
    )
    train_end = sizes[0]
    checkpoint_end = train_end + sizes[1]
    calibration_end = checkpoint_end + sizes[2]
    split = SubjectSplit(
        all_subjects=all_subjects,
        outer_subject=outer,
        seed=seed,
        assignment_rule=ASSIGNMENT_RULE,
        assignment_order=order,
        inner_train=order[:train_end],
        checkpoint_selection=order[train_end:checkpoint_end],
        calibration=order[checkpoint_end:calibration_end],
        routing_selection=order[calibration_end:],
    )
    validate_subject_split(split)
    return split


__all__ = [
    "ASSIGNMENT_RULE",
    "DEFAULT_SPLIT_SEED",
    "SELECTION_USE_KEYS",
    "SPLIT_SCHEMA_VERSION",
    "SplitProtocolError",
    "SubjectSplit",
    "build_subject_split",
    "normalize_subject_id",
    "validate_no_outer_leakage",
    "validate_subject_split",
]
