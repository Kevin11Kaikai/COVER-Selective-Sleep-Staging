"""Label-free deterministic subject assignments for the frozen protocol."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Sequence


SUBJECTS = tuple(f"{index:02d}" for index in range(20))
SEED = 20260901
SPLIT_VERSION = "stage2-split-v1"


class SplitError(ValueError):
    """Raised when a Stage-2 subject assignment is not canonical."""


def normalize_subject(value: object) -> str:
    if isinstance(value, bool) or value is None:
        raise SplitError("subject must be a two-digit identifier")
    text = str(value).strip()
    if not text.isdecimal():
        raise SplitError(f"non-numeric subject: {text!r}")
    result = f"{int(text):02d}"
    if result not in SUBJECTS:
        raise SplitError(f"subject outside frozen cohort: {result}")
    return result


def _canonical_subjects(values: Iterable[object]) -> tuple[str, ...]:
    result = tuple(normalize_subject(value) for value in values)
    if len(set(result)) != len(result):
        raise SplitError("duplicate subject")
    return result


def _ordered(values: Sequence[str], prefix: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            values,
            key=lambda subject: (
                hashlib.sha256(f"{prefix}|subject={subject}".encode()).hexdigest(),
                subject,
            ),
        )
    )


def _hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_outer_split(
    outer: object, subjects: Sequence[object] = SUBJECTS
) -> dict[str, object]:
    outer_id = normalize_subject(outer)
    all_subjects = _canonical_subjects(subjects)
    if tuple(sorted(all_subjects)) != SUBJECTS:
        raise SplitError("outer split requires exactly the frozen 20-subject cohort")
    development = tuple(subject for subject in all_subjects if subject != outer_id)
    order = _ordered(
        development,
        f"{SPLIT_VERSION}|{SEED}|outer={outer_id}|fit-cert",
    )
    payload: dict[str, object] = {
        "outer": outer_id,
        "fit": list(order[:15]),
        "cert": list(order[15:]),
        "assignment_order": list(order),
        "assignment_rule": "sha256(stage2-split-v1|20260901|outer|fit-cert|subject)",
    }
    payload["split_hash"] = _hash(payload)
    validate_outer_split(payload)
    return payload


def validate_outer_split(split: dict[str, object]) -> None:
    outer = normalize_subject(split.get("outer"))
    fit = _canonical_subjects(split.get("fit", ()))  # type: ignore[arg-type]
    cert = _canonical_subjects(split.get("cert", ()))  # type: ignore[arg-type]
    if len(fit) != 15 or len(cert) != 4:
        raise SplitError("outer split must be FIT15/CERT4")
    if set(fit) & set(cert) or outer in fit or outer in cert:
        raise SplitError("outer/FIT/CERT roles overlap")
    if set(fit) | set(cert) | {outer} != set(SUBJECTS):
        raise SplitError("outer split does not cover the frozen cohort")


def build_inner_roles(
    outer: object, target: object, fit_subjects: Sequence[object] | None = None
) -> dict[str, object]:
    outer_id = normalize_subject(outer)
    split = build_outer_split(outer_id)
    fit = (
        _canonical_subjects(fit_subjects)
        if fit_subjects is not None
        else tuple(split["fit"])  # type: ignore[arg-type]
    )
    target_id = normalize_subject(target)
    if target_id not in fit:
        raise SplitError("inner target must belong to FIT")
    candidates = tuple(subject for subject in fit if subject != target_id)
    order = _ordered(
        candidates,
        f"{SPLIT_VERSION}|{SEED}|outer={outer_id}|inner-target={target_id}",
    )
    payload: dict[str, object] = {
        "outer": outer_id,
        "target": target_id,
        "component_train": list(order[:10]),
        "checkpoint_selection": list(order[10:12]),
        "feature_calibration": list(order[12:14]),
        "assignment_order": list(order),
        "role": "inner",
    }
    payload["role_hash"] = _hash(payload)
    validate_component_roles(payload, fit=fit, expected_target=target_id)
    return payload


def build_final_roles(
    outer: object, fit_subjects: Sequence[object] | None = None
) -> dict[str, object]:
    outer_id = normalize_subject(outer)
    split = build_outer_split(outer_id)
    fit = (
        _canonical_subjects(fit_subjects)
        if fit_subjects is not None
        else tuple(split["fit"])  # type: ignore[arg-type]
    )
    order = _ordered(
        fit,
        f"{SPLIT_VERSION}|{SEED}|outer={outer_id}|final-component",
    )
    payload: dict[str, object] = {
        "outer": outer_id,
        "target": "final",
        "component_train": list(order[:10]),
        "checkpoint_selection": list(order[10:12]),
        "feature_calibration": list(order[12:14]),
        "unused": list(order[14:15]),
        "assignment_order": list(order),
        "role": "final",
    }
    payload["role_hash"] = _hash(payload)
    validate_component_roles(payload, fit=fit, expected_target=None)
    return payload


def validate_component_roles(
    roles: dict[str, object], *, fit: Sequence[str], expected_target: str | None
) -> None:
    outer = normalize_subject(roles.get("outer"))
    train = _canonical_subjects(roles.get("component_train", ()))  # type: ignore[arg-type]
    checkpoint = _canonical_subjects(roles.get("checkpoint_selection", ()))  # type: ignore[arg-type]
    calibration = _canonical_subjects(roles.get("feature_calibration", ()))  # type: ignore[arg-type]
    unused = _canonical_subjects(roles.get("unused", ()))  # type: ignore[arg-type]
    if (len(train), len(checkpoint), len(calibration)) != (10, 2, 2):
        raise SplitError("component roles must be 10/2/2")
    role_sets = [set(train), set(checkpoint), set(calibration), set(unused)]
    if any(role_sets[i] & role_sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise SplitError("component roles overlap")
    used = set().union(*role_sets)
    expected = set(fit) - ({expected_target} if expected_target is not None else set())
    if used != expected:
        raise SplitError("component roles do not cover their declared FIT pool")
    if outer in used or (expected_target is not None and expected_target in used):
        raise SplitError("outer or predicted target occurs in a component role")


def build_risk_cv(
    outer: object, fit_subjects: Sequence[object] | None = None
) -> list[dict[str, list[str]]]:
    outer_id = normalize_subject(outer)
    split = build_outer_split(outer_id)
    fit = (
        _canonical_subjects(fit_subjects)
        if fit_subjects is not None
        else tuple(split["fit"])  # type: ignore[arg-type]
    )
    order = _ordered(
        fit,
        f"stage2-risk-cv-v1|{SEED}|outer={outer_id}",
    )
    folds: list[dict[str, list[str]]] = []
    for index in range(5):
        # The frozen rule is round-robin after the hash ordering.  Keeping this
        # as an explicit stride (rather than three contiguous chunks) makes the
        # implementation match the protocol text byte-for-byte in meaning.
        validation = list(order[index::5])
        training = [subject for subject in order if subject not in validation]
        folds.append({"train": training, "validation": validation})
    if sorted(subject for fold in folds for subject in fold["validation"]) != sorted(fit):
        raise SplitError("risk CV validation folds do not partition FIT")
    return folds


def derive_fit_seed(outer: object, role: str, target: object | str) -> int:
    outer_id = normalize_subject(outer)
    if role not in {"inner", "final"}:
        raise SplitError("fit seed role must be inner or final")
    target_id = "final" if role == "final" else normalize_subject(target)
    text = (
        f"stage2-fit-seed-v1|{SEED}|outer={outer_id}|"
        f"role={role}|target={target_id}"
    )
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


__all__ = [
    "SEED",
    "SUBJECTS",
    "SplitError",
    "build_final_roles",
    "build_inner_roles",
    "build_outer_split",
    "build_risk_cv",
    "derive_fit_seed",
    "normalize_subject",
    "validate_component_roles",
    "validate_outer_split",
]
