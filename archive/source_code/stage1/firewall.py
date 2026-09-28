"""Fold-scoped label authorization and outer-unblinding guards."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .splits import SUBJECTS, normalize_subject


class FirewallError(RuntimeError):
    """Raised when a label or role request violates the frozen protocol."""


FIT_LABEL_PHASES = {
    "COMPONENT_TRAIN",
    "CHECKPOINT_SELECTION",
    "FEATURE_CALIBRATION",
    "RISK_FIT",
    "RISK_SELECTION",
    "ERROR_RISK_CALIBRATION",
}


@dataclass
class RoleLedger:
    outer: str
    fit: tuple[str, ...]
    cert: tuple[str, ...]
    path: Path | None = None
    events: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.outer = normalize_subject(self.outer)
        self.fit = tuple(normalize_subject(value) for value in self.fit)
        self.cert = tuple(normalize_subject(value) for value in self.cert)
        if len(self.fit) != 15 or len(self.cert) != 4:
            raise FirewallError("ledger requires FIT15/CERT4")
        if set(self.fit) & set(self.cert) or self.outer in self.fit + self.cert:
            raise FirewallError("ledger roles overlap")

    def _record(
        self,
        *,
        phase: str,
        subjects: Sequence[str],
        decision: str,
        reason: str | None = None,
    ) -> None:
        event: dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "outer": self.outer,
            "phase": phase,
            "subjects": list(subjects),
            "decision": decision,
        }
        if reason is not None:
            event["reason"] = reason
        self.events.append(event)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")))
                handle.write("\n")

    def authorize_label_load(
        self, subjects: Iterable[object], phase: str
    ) -> tuple[str, ...]:
        normalized = tuple(normalize_subject(value) for value in subjects)
        reason: str | None = None
        if not normalized or len(set(normalized)) != len(normalized):
            reason = "label request must contain unique subjects"
        elif phase in FIT_LABEL_PHASES:
            if not set(normalized) <= set(self.fit):
                reason = "FIT phase requested outer or CERT labels"
        elif phase == "CERT_SELECTOR":
            if not set(normalized) <= set(self.cert):
                reason = "CERT_SELECTOR may load CERT labels only"
        elif phase == "SINGLE_OUTER_UNBLINDING":
            if normalized != (self.outer,):
                reason = "unblinding may load only its matching outer"
        else:
            reason = f"unknown label phase: {phase}"
        if reason is not None:
            self._record(
                phase=phase,
                subjects=normalized,
                decision="DENIED",
                reason=reason,
            )
            raise FirewallError(reason)
        self._record(
            phase=phase,
            subjects=normalized,
            decision="ALLOWED",
        )
        return normalized


def authorize_label_load(
    ledger: RoleLedger, subjects: Iterable[object], phase: str
) -> tuple[str, ...]:
    return ledger.authorize_label_load(subjects, phase)


def assert_role_exclusion(
    *,
    outer: object,
    predicted_subject: object,
    component_train: Sequence[object],
    checkpoint_selection: Sequence[object],
    feature_calibration: Sequence[object],
    cert: Sequence[object],
) -> None:
    outer_id = normalize_subject(outer)
    predicted = normalize_subject(predicted_subject)
    ancestors = {
        normalize_subject(value)
        for values in (component_train, checkpoint_selection, feature_calibration)
        for value in values
    }
    forbidden = {outer_id, predicted} | {normalize_subject(value) for value in cert}
    overlap = sorted(ancestors & forbidden)
    if overlap:
        raise FirewallError(f"forbidden subject in component ancestry: {overlap}")


def assert_ready_to_unblind(
    fold_manifests: Sequence[Mapping[str, object]],
    *,
    expected_protocol_hash: str | None = None,
    required_outer_subjects: Sequence[object] = SUBJECTS,
) -> None:
    required = {normalize_subject(subject) for subject in required_outer_subjects}
    if len(required) != len(required_outer_subjects):
        raise FirewallError("required outer subjects contain duplicates")
    if len(fold_manifests) != len(required):
        raise FirewallError("unblinding requires every declared frozen fold")
    seen: set[str] = set()
    for manifest in fold_manifests:
        outer = normalize_subject(manifest.get("outer"))
        if outer in seen:
            raise FirewallError("duplicate frozen outer fold")
        seen.add(outer)
        if manifest.get("status") != "OUTER_PREDICTIONS_FROZEN":
            raise FirewallError(f"outer {outer} is not frozen")
        if expected_protocol_hash is not None and manifest.get("protocol_hash") != expected_protocol_hash:
            raise FirewallError(f"outer {outer} protocol hash mismatch")
        if manifest.get("contains_outer_labels") is not False:
            raise FirewallError(f"outer {outer} artifact is not label blind")
        if not manifest.get("artifact_sha256"):
            raise FirewallError(f"outer {outer} has no frozen artifact hash")
    if seen != required:
        raise FirewallError("frozen fold subjects do not match the required cohort")


__all__ = [
    "FirewallError",
    "RoleLedger",
    "assert_ready_to_unblind",
    "assert_role_exclusion",
    "authorize_label_load",
]
