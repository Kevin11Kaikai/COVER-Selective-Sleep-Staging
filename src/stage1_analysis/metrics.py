"""Count-based metrics. Zero accepted mass has undefined conditional risk."""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def identity_key(row: dict) -> tuple[str, str, str, int]:
    return (str(row["dataset"]), str(row["subject_id"]), str(row["record_id"]),
            int(row["epoch_index"]))


def identity_ties(keys: Iterable[tuple]) -> np.ndarray:
    return np.array([hashlib.sha256("|".join(map(str, key)).encode("utf-8")).hexdigest()
                     for key in keys])


def alignment(source_keys: list[tuple], target_keys: list[tuple]) -> np.ndarray:
    if len(source_keys) != len(set(source_keys)) or len(target_keys) != len(set(target_keys)):
        raise ValueError("Duplicate epoch identity")
    index = {key: i for i, key in enumerate(target_keys)}
    try:
        return np.array([index[key] for key in source_keys], dtype=int)
    except KeyError as exc:
        raise ValueError("Missing target epoch identity") from exc


def count_subjects(mask: np.ndarray, errors: np.ndarray, subjects: np.ndarray,
                   people: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.array([np.count_nonzero(subjects == s) for s in people], dtype=np.int64)
    a = np.array([np.count_nonzero(mask & (subjects == s)) for s in people], dtype=np.int64)
    k = np.array([np.count_nonzero(mask & errors & (subjects == s)) for s in people], dtype=np.int64)
    if (n <= 0).any() or (k > a).any() or (a > n).any():
        raise ValueError("Invalid subject counts")
    return n, a, k


def summary(n: np.ndarray, a: np.ndarray, k: np.ndarray) -> dict:
    total_a, total_k = int(a.sum()), int(k.sum())
    return {"eligible": int(n.sum()), "accepted": total_a, "errors": total_k,
            "macro_coverage": float(np.mean(a / n)),
            "pooled_coverage": float(total_a / n.sum()),
            "subject_equal_ratio_risk": float((k / n).sum() / (a / n).sum()) if total_a else None,
            "pooled_risk": total_k / total_a if total_a else None,
            "passed_subjects": int(((a >= 50) & (20 * k <= a)).sum()),
            "zero_subjects": int((a == 0).sum()),
            "below_min_support_subjects": int((a < 50).sum()),
            "over_risk_subjects": int(((a > 0) & (20 * k > a)).sum())}


def classification(labels: np.ndarray, predictions: np.ndarray) -> tuple[dict, list[dict], np.ndarray]:
    """Five-class, class-macro metrics; undefined class ratios use zero_division=0."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.ndim != 1 or labels.shape != predictions.shape or not len(labels):
        raise ValueError("Nonempty aligned classification vectors are required")
    if not np.isin(labels, np.arange(5)).all() or not np.isin(predictions, np.arange(5)).all():
        raise ValueError("Classification labels must be integers from 0 to 4")
    cm = np.bincount((5 * labels + predictions).astype(int), minlength=25).reshape(5, 5)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    correct = cm.diagonal()
    precision = np.divide(correct, predicted, out=np.zeros(5), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(5), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(5), where=support + predicted > 0)
    overall = {"epochs": len(labels), "accuracy": float(correct.sum() / len(labels)),
               "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
               "macro_f1": float(f1.mean()), "zero_division": 0}
    rows = [{"class": name, "support": int(support[c]), "predicted_support": int(predicted[c]),
             "correct": int(correct[c]), "precision": float(precision[c]), "recall": float(recall[c]),
             "f1": float(f1[c]), "precision_defined": bool(predicted[c]),
             "recall_defined": bool(support[c])}
            for c, name in enumerate(["Wake", "N1", "N2", "N3", "REM"])]
    return overall, rows, cm


def selection_class_counts(labels: np.ndarray, errors: np.ndarray, mask: np.ndarray, c: int) -> dict:
    """Audit a true-class group; undefined selection ratios remain None."""
    group = labels == c
    support = int(group.sum())
    accepted = int((group & mask).sum())
    wrong = int((group & mask & errors).sum())
    base_correct = int((group & ~errors).sum())
    ca = accepted - wrong
    correct_rejected = base_correct - ca
    wrong_rejected = support - base_correct - wrong
    assert accepted + correct_rejected + wrong_rejected == support
    assert ca + correct_rejected == base_correct
    return {"support": support, "accepted": accepted, "errors": wrong,
            "coverage": accepted / support if support else None,
            "risk": wrong / accepted if accepted else None,
            "correct_accepted": ca, "correct_auto_fraction": ca / support if support else None,
            "base_correct": base_correct, "base_incorrect": support - base_correct,
            "correct_rejected": correct_rejected, "incorrect_rejected": wrong_rejected,
            "accepted_accuracy": ca / accepted if accepted else None,
            "correct_retention": ca / base_correct if base_correct else None,
            "primary_class_recall": base_correct / support if support else None}


def ranked_orders(scores: np.ndarray, ties: np.ndarray, subjects: np.ndarray,
                  people: list[str]) -> list[np.ndarray]:
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite score")
    return [idx[np.lexsort((ties[idx], scores[idx]))]
            for s in people for idx in [np.flatnonzero(subjects == s)]]


def quota_mask(orders: list[np.ndarray], quotas: Iterable[int], size: int) -> np.ndarray:
    quotas = list(quotas)
    if len(orders) != len(quotas):
        raise ValueError("One quota is required for each subject")
    result = np.zeros(size, dtype=bool)
    for order, quota in zip(orders, quotas):
        if int(quota) != quota or not 0 <= quota <= len(order):
            raise ValueError("Invalid quota")
        result[order[:int(quota)]] = True
    return result


def overlap(a: np.ndarray, b: np.ndarray, errors: np.ndarray) -> tuple[dict, list[dict]]:
    union = int((a | b).sum())
    shared = int((a & b).sum())
    rows = []
    for name, region in [("shared", a & b), ("added", a & ~b), ("removed", b & ~a)]:
        n, k = int(region.sum()), int((region & errors).sum())
        rows.append({"region": name, "count": n, "errors": k, "risk": k / n if n else None})
    assert int(a.sum()) - int(b.sum()) == rows[1]["count"] - rows[2]["count"]
    assert int((a & errors).sum()) - int((b & errors).sum()) == rows[1]["errors"] - rows[2]["errors"]
    return {"jaccard": shared / union if union else None,
            "symmetric_difference": int((a ^ b).sum()),
            "decision_flip_rate": float(np.mean(a ^ b))}, rows


def bootstrap_metrics(n: np.ndarray, a: np.ndarray, k: np.ndarray,
                      resamples: np.ndarray) -> dict[str, np.ndarray]:
    nn, aa, kk = n[resamples], a[resamples], k[resamples]
    denominator = aa.sum(axis=1)
    weighted_denominator = (aa / nn).sum(axis=1)
    return {
        "macro_coverage": (aa / nn).mean(axis=1),
        "pooled_risk": np.divide(kk.sum(axis=1), denominator,
                                 out=np.full(len(aa), np.nan), where=denominator > 0),
        "subject_equal_ratio_risk": np.divide((kk / nn).sum(axis=1), weighted_denominator,
            out=np.full(len(aa), np.nan), where=weighted_denominator > 0),
        "passed_subjects": ((aa >= 50) & (20 * kk <= aa)).sum(axis=1),
    }


def interval(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    if not finite.any():
        return {"low": None, "high": None, "valid_draws": 0}
    low, high = np.quantile(values[finite], [.025, .975], method="linear")
    return {"low": float(low), "high": float(high), "valid_draws": int(finite.sum())}
