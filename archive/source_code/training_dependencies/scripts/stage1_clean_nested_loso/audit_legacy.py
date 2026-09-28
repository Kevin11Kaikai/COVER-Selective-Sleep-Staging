"""Read-only diagnostic audit for legacy Stage-1 disagreement artifacts.

Legacy archives are never promoted, rewritten, or repaired here.  The audit always
returns ``confirmatory_eligible=false`` and the CLI exits non-zero so that an old
artifact cannot silently enter the clean nested-LOSO result path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:  # Direct script execution and package import both remain supported.
    from .contracts import ContractError, sha256_file
except ImportError:  # pragma: no cover - exercised by the CLI, not package tests.
    from contracts import ContractError, sha256_file


LEGACY_REQUIRED_PROVENANCE_FIELDS = (
    "dataset",
    "subject_id",
    "record_id",
    "epoch_index",
    "outer_fold",
    "schema_hash",
    "split_hash",
    "model_hash",
    "checkpoint_hash",
    "data_hash",
    "preprocessing_hash",
    "config_hash",
    "calibration_hash",
    "code_hash",
)


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _probability_diagnostics(probabilities: np.ndarray, atol: float) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "shape": list(probabilities.shape),
        "finite": bool(np.all(np.isfinite(probabilities))),
    }
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        diagnostics.update(
            {
                "valid_shape": False,
                "out_of_range_count": None,
                "row_sum_mismatch_count": None,
                "max_row_sum_error": None,
            }
        )
        return diagnostics
    diagnostics["valid_shape"] = True
    finite_values = np.isfinite(probabilities)
    out_of_range = finite_values & (
        (probabilities < -atol) | (probabilities > 1.0 + atol)
    )
    row_sums = probabilities.sum(axis=1)
    row_sum_errors = np.abs(row_sums - 1.0)
    finite_row_sum_errors = row_sum_errors[np.isfinite(row_sum_errors)]
    diagnostics.update(
        {
            "out_of_range_count": int(np.count_nonzero(out_of_range)),
            "row_sum_mismatch_count": int(np.count_nonzero(row_sum_errors > atol)),
            "max_row_sum_error": (
                float(finite_row_sum_errors.max())
                if finite_row_sum_errors.size
                else None
            ),
        }
    )
    return diagnostics


def _expert_consistency(
    predictions: np.ndarray,
    confidences: np.ndarray,
    probabilities: np.ndarray,
    *,
    confidence_atol: float,
    confidence_rtol: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prediction_shape": list(predictions.shape),
        "confidence_shape": list(confidences.shape),
        "probability_shape": list(probabilities.shape),
        "shape_compatible": False,
        "prediction_argmax_mismatch_count": None,
        "confidence_max_mismatch_count": None,
        "max_confidence_discrepancy": None,
    }
    if (
        predictions.ndim != 1
        or confidences.ndim != 1
        or probabilities.ndim != 2
        or len(predictions) != len(confidences)
        or len(predictions) != probabilities.shape[0]
        or probabilities.shape[1] < 2
    ):
        return result

    result["shape_compatible"] = True
    expected_predictions = probabilities.argmax(axis=1)
    expected_confidences = probabilities.max(axis=1)
    confidence_discrepancy = np.abs(confidences - expected_confidences)
    finite_discrepancy = confidence_discrepancy[np.isfinite(confidence_discrepancy)]
    confidence_match = np.isclose(
        confidences,
        expected_confidences,
        atol=confidence_atol,
        rtol=confidence_rtol,
    )
    result.update(
        {
            "prediction_argmax_mismatch_count": int(
                np.count_nonzero(predictions != expected_predictions)
            ),
            "confidence_max_mismatch_count": int(np.count_nonzero(~confidence_match)),
            "max_confidence_discrepancy": (
                float(finite_discrepancy.max())
                if finite_discrepancy.size
                else None
            ),
        }
    )
    return result


def _fold_diagnostics(
    fold_ids: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    probabilities: np.ndarray,
    *,
    confidence_atol: float,
    confidence_rtol: float,
) -> list[dict[str, Any]]:
    if fold_ids.ndim != 1 or len(fold_ids) != len(predictions):
        return []
    results: list[dict[str, Any]] = []
    for fold_id in np.unique(fold_ids):
        mask = fold_ids == fold_id
        consistency = _expert_consistency(
            predictions[mask],
            confidences[mask],
            probabilities[mask],
            confidence_atol=confidence_atol,
            confidence_rtol=confidence_rtol,
        )
        results.append(
            {
                "fold_id": _scalar(fold_id),
                "row_count": int(np.count_nonzero(mask)),
                **consistency,
            }
        )
    return results


def audit_legacy_npz(
    artifact_path: str | Path,
    *,
    confidence_atol: float = 1e-7,
    confidence_rtol: float = 1e-6,
    probability_atol: float = 1e-6,
) -> dict[str, Any]:
    """Diagnose a legacy archive without mutating or certifying it."""

    path = Path(artifact_path)
    if not path.is_file():
        raise ContractError(f"legacy artifact not found: {path}")
    if confidence_atol < 0 or confidence_rtol < 0 or probability_atol < 0:
        raise ContractError("numeric tolerances must be non-negative")

    report: dict[str, Any] = {
        "audit_schema": "stage1-legacy-diagnostic-v1",
        "artifact_path": str(path.resolve()),
        "artifact_sha256": sha256_file(path),
        "artifact_disposition": "diagnostic_only",
        "confirmatory_eligible": False,
        "fail_closed": True,
        "repaired": False,
        "reasons": [
            "legacy artifacts are excluded from the clean confirmatory path",
        ],
    }

    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = tuple(archive.files)
            key_set = set(keys)
            report["keys"] = list(keys)
            missing_provenance = sorted(
                set(LEGACY_REQUIRED_PROVENANCE_FIELDS) - key_set
            )
            report["missing_provenance_fields"] = missing_provenance
            if missing_provenance:
                report["reasons"].append(
                    "missing stable row identity and/or immutable provenance fields"
                )

            core_fields = {"y_pred_T", "conf_T", "probs_T"}
            missing_core = sorted(core_fields - key_set)
            report["missing_primary_fields"] = missing_core
            if missing_core:
                report["reasons"].append(
                    "primary prediction consistency cannot be checked: missing "
                    + ", ".join(missing_core)
                )
            else:
                predictions_t = np.asarray(archive["y_pred_T"])
                confidences_t = np.asarray(archive["conf_T"])
                probabilities_t = np.asarray(archive["probs_T"])
                primary = _expert_consistency(
                    predictions_t,
                    confidences_t,
                    probabilities_t,
                    confidence_atol=confidence_atol,
                    confidence_rtol=confidence_rtol,
                )
                primary["probabilities"] = _probability_diagnostics(
                    probabilities_t, probability_atol
                )
                report["primary_expert"] = primary
                if not primary["shape_compatible"]:
                    report["reasons"].append("primary arrays have incompatible shapes")
                elif (
                    primary["prediction_argmax_mismatch_count"]
                    or primary["confidence_max_mismatch_count"]
                ):
                    report["reasons"].append(
                        "primary prediction/probability/confidence arrays are inconsistent"
                    )
                probability_report = primary["probabilities"]
                if (
                    not probability_report["finite"]
                    or probability_report["out_of_range_count"]
                    or probability_report["row_sum_mismatch_count"]
                ):
                    report["reasons"].append("primary probabilities violate simplex checks")

                if "fold_ids" in key_set:
                    report["primary_by_fold"] = _fold_diagnostics(
                        np.asarray(archive["fold_ids"]),
                        predictions_t,
                        confidences_t,
                        probabilities_t,
                        confidence_atol=confidence_atol,
                        confidence_rtol=confidence_rtol,
                    )

            feature_fields = {"y_pred_F", "conf_F", "probs_F"}
            if feature_fields.issubset(key_set):
                predictions_f = np.asarray(archive["y_pred_F"])
                confidences_f = np.asarray(archive["conf_F"])
                probabilities_f = np.asarray(archive["probs_F"])
                feature = _expert_consistency(
                    predictions_f,
                    confidences_f,
                    probabilities_f,
                    confidence_atol=confidence_atol,
                    confidence_rtol=confidence_rtol,
                )
                feature["probabilities"] = _probability_diagnostics(
                    probabilities_f, probability_atol
                )
                report["feature_expert"] = feature
                if not feature["shape_compatible"]:
                    report["reasons"].append("feature-expert arrays have incompatible shapes")
                elif (
                    feature["prediction_argmax_mismatch_count"]
                    or feature["confidence_max_mismatch_count"]
                ):
                    report["reasons"].append(
                        "feature prediction/probability/confidence arrays are inconsistent"
                    )
    except (OSError, ValueError, KeyError) as exc:
        report["audit_error"] = f"{type(exc).__name__}: {exc}"
        report["reasons"].append("legacy archive could not be read safely")

    # De-duplicate while retaining deterministic first-seen order.
    report["reasons"] = list(dict.fromkeys(report["reasons"]))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only, fail-closed diagnostic audit of a legacy Stage-1 NPZ"
    )
    parser.add_argument(
        "artifact",
        nargs="?",
        default="outputs/disagreement_analysis/detailed_data.npz",
        help="legacy NPZ to inspect; it will never be modified",
    )
    parser.add_argument("--confidence-atol", type=float, default=1e-7)
    parser.add_argument("--confidence-rtol", type=float, default=1e-6)
    parser.add_argument("--probability-atol", type=float, default=1e-6)
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_legacy_npz(
            args.artifact,
            confidence_atol=args.confidence_atol,
            confidence_rtol=args.confidence_rtol,
            probability_atol=args.probability_atol,
        )
    except ContractError as exc:
        report = {
            "audit_schema": "stage1-legacy-diagnostic-v1",
            "artifact_path": str(Path(args.artifact).resolve()),
            "artifact_disposition": "diagnostic_only",
            "confirmatory_eligible": False,
            "fail_closed": True,
            "repaired": False,
            "audit_error": str(exc),
        }
    json.dump(
        report,
        sys.stdout,
        ensure_ascii=False,
        sort_keys=True,
        indent=None if args.compact else 2,
    )
    sys.stdout.write("\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
