"""Alternate arithmetic replay of scores, exact quantiles, masks and reported counts.

This is a single-agent self-check, not independent scientific replication.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp

from core import ROOT, S2, PEOPLE, PH, METHOD_ID, read_json, write_json, sha256
from run_comparison import fit_fold, verify_frozen


def fraction(x):
    return Fraction(x["numerator"], x["denominator"])


def rebuild_grid(scores, subjects):
    counts = Counter(str(s) for s in subjects)
    denominator = math.lcm(*(15*n for n in counts.values()))
    by_score = {}
    for score, subject in zip(scores, subjects):
        by_score[float(score)] = by_score.get(float(score), 0) + denominator // (15*counts[str(subject)])
    ordered = sorted(by_score)
    cumulative, acc = [], 0
    for score in ordered:
        acc += by_score[score]
        cumulative.append(acc)
    assert acc == denominator
    result = []
    for numerator in range(1, 21):
        target = Fraction(numerator * denominator, 20)
        position = bisect.bisect_left(cumulative, target)
        before = cumulative[position-1] if position else 0
        tie = (target-before) / by_score[ordered[position]]
        result.append((ordered[position], tie))
    return result


def rebuild_mask(scores, ids, choice, outer):
    if choice is None:
        return np.zeros(len(scores), dtype=bool)
    threshold, f = choice["threshold"], fraction(choice["tie_fraction"])
    result = scores < threshold
    for i in np.flatnonzero(scores == threshold):
        item = ids[i]
        text = (f"stage2-tie-v2|protocol={PH}|outer={outer}|dataset={item['dataset']}|"
                f"subject={item['subject_id']}|record={item['record_id']}|epoch={item['epoch_index']}|method={METHOD_ID}")
        u = Fraction(int.from_bytes(hashlib.sha256(text.encode()).digest(), "big"), 2**256)
        result[i] = u < f
    return result


def actual_metrics(mask, errors, subjects):
    people = sorted(set(subjects))
    rows = []
    for subject in people:
        indices = np.flatnonzero(subjects == subject)
        n = len(indices)
        a = sum(bool(mask[i]) for i in indices)
        k = sum(bool(mask[i]) and bool(errors[i]) for i in indices)
        rows.append((subject, n, a, k))
    cov = sum((Fraction(a,n) for s,n,a,k in rows), Fraction(0)) / len(rows)
    mass = sum((Fraction(k,n) for s,n,a,k in rows), Fraction(0))
    accepted_mass = sum((Fraction(a,n) for s,n,a,k in rows), Fraction(0))
    risk = mass / accepted_mass if accepted_mass else None
    return rows, cov, risk


def run():
    start = time.perf_counter()
    verify_frozen()
    source_manifest = read_json(ROOT / "config/input_manifest.json")
    source_manifest.update(read_json(ROOT / "evaluation/input_manifest.json"))
    for path, info in source_manifest.items():
        assert sha256(path) == info["sha256"], path
    max_score_error, max_nll_error = 0., 0.
    fit_seconds, fold_seconds, rss, weight_rows = [], [], [], []
    for outer in PEOPLE:
        directory = ROOT / "folds" / f"outer-{outer}"
        d = joblib.load(S2 / "development/risk" / f"outer-{outer}" / "development.joblib")
        model = read_json(directory / "weights.json")
        policy = read_json(directory / "policies.json")
        complete = read_json(directory / "completion.json")
        wp, wf = model["weights"]
        assert model["success"] and 0 <= wp <= 20 and 0 <= wf <= 20
        subject_counts = Counter(d["fit_data"]["subjects"])
        weights = np.array([1 / subject_counts[s] for s in d["fit_data"]["subjects"]])
        # Cast matches the original fit objective, distinct from the deployment float32 P term.
        x = d["fit_data"]["logits"].astype(float) * wp + np.log(np.clip(d["fit_data"]["feature"], 1e-10, 1)) * wf
        logp = x - logsumexp(x, axis=1)[:, None]
        loss = -float(np.average(logp[np.arange(len(x)), d["fit_data"]["labels"]], weights=weights))
        max_nll_error = max(max_nll_error, abs(loss-model["source_NLL"]))
        assert abs(loss-model["source_NLL"]) < 1e-10
        with np.load(directory / "decisions.npz", allow_pickle=False) as saved:
            for name, part in [("fit", "fit_data"), ("cert", "cert_data_blind"), ("outer", "outer_data_blind")]:
                data = d[part]
                z = wp*data["logits"] + wf*np.log(np.clip(data["feature"], 1e-10, 1))
                ll = z - logsumexp(z, axis=1)[:, None]
                scores = -np.expm1(ll[np.arange(len(z)), data["primary"].argmax(1)])
                diff = float(np.max(np.abs(scores-saved[f"{name}_scores"])))
                max_score_error = max(max_score_error, diff)
                assert diff < 1e-12
            independent_grid = rebuild_grid(saved["fit_scores"], saved["fit_subjects"])
            for actual, expected in zip(independent_grid, policy["candidate_rows"]):
                assert actual == (expected["threshold"], fraction(expected["tie_fraction"]))
            cy = saved["cert_labels"]
            ce = d["cert_data_blind"]["primary"].argmax(1) != cy
            feasible = {"DuoF-0.05": [], "DuoF-0.03": []}
            for candidate in policy["candidate_rows"]:
                mask = rebuild_mask(saved["cert_scores"], d["cert_data_blind"]["identities"], candidate, outer)
                persons, cov, risk = actual_metrics(mask, ce, d["cert_data_blind"]["subjects"])
                assert cov == fraction(candidate["coverage"])
                assert risk == (fraction(candidate["risk"]) if candidate["risk"] else None)
                supported = all(a >= 50 for s,n,a,k in persons)
                assert supported == candidate["supported"]
                for method, alpha in [("DuoF-0.05", Fraction(5,100)), ("DuoF-0.03", Fraction(3,100))]:
                    if supported and risk is not None and risk <= alpha:
                        feasible[method].append((-cov, risk, fraction(candidate["q"]), candidate))
            for method in feasible:
                chosen = min(feasible[method], key=lambda r: r[:3])[3] if feasible[method] else None
                assert chosen == policy["selections"][method]
                mask = rebuild_mask(saved["outer_scores"], d["outer_data_blind"]["identities"], chosen, outer)
                np.testing.assert_array_equal(mask, saved[f"accept_{method}"])
        fit_seconds.append(complete["fit_seconds"])
        fold_seconds.append(complete["seconds"])
        rss.append(complete["RSS_MiB_at_fold_end"])
        weight_rows.append({"subject": outer, "primary_weight": wp, "RF_weight": wf,
                           "iterations": model["iterations"], "fit_rows": model["fit_rows"],
                           "fit_seconds": complete["fit_seconds"], "fold_seconds": complete["seconds"]})
        print(f"REPLAY outer-{outer}: scores, 20 exact candidates, both policies/masks PASS", flush=True)
    results = read_json(ROOT / "evaluation/results.json")
    by_person = {(r["method"], r["subject"]): r for r in results["per_subject"]}
    with np.load(ROOT / "evaluation/analysis_bundle.npz", allow_pickle=False) as saved:
        subjects, errors = saved["subjects"], saved["truth"] != saved["primary_predictions"]
        for col, method in enumerate(saved["method_names"]):
            persons, coverage, risk = actual_metrics(saved["masks"][:, col], errors, subjects)
            for s,n,a,k in persons:
                r = by_person[method, s]
                assert (n,a,k) == (r["eligible"], r["accepted"], r["errors"])
                assert r["passed"] == (a >= 50 and k*20 <= a)
            summary = next(r for r in results["aggregate"] if r["method"] == method)
            assert abs(float(coverage)-summary["macro_coverage"]) < 1e-12
            assert abs(float(risk)-summary["subject_equal_ratio_risk"]) < 1e-12
            assert sum(a for s,n,a,k in persons) == summary["accepted"]
            assert sum(k for s,n,a,k in persons) == summary["errors"]
    # Exercise actual checkpoint reuse; science files and mtimes must remain intact.
    watched = [p for p in (ROOT / "folds").rglob("*") if p.is_file()]
    before = {str(p): (sha256(p), p.stat().st_mtime_ns) for p in watched}
    fit_fold("00")
    after = {str(p): (sha256(p), p.stat().st_mtime_ns) for p in watched}
    assert before == after
    with (ROOT / "tables/weights_and_timing.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(weight_rows[0]))
        writer.writeheader()
        writer.writerows(weight_rows)
    receipt = {"status": "PASS", "verified_utc": now_local(), "attribution": "alternate arithmetic by same agent; not independent replication",
               "folds": 20, "exact_candidate_rows_rebuilt": 400, "policy_masks_rebuilt": 40,
               "metric_rows_recomputed": 160, "maximum_score_difference": max_score_error,
               "maximum_NLL_difference": max_nll_error, "unchanged_source_files": len(source_manifest),
               "actual_resume_test": "PASS; 80 fold files hash and mtime unchanged",
               "verification_seconds": time.perf_counter()-start,
               "total_Duo_fit_seconds": sum(fit_seconds), "total_fold_work_seconds": sum(fold_seconds),
               "maximum_observed_fold_end_RSS_MiB": max(rss), "RSS_note": "sampled at fold end, not a continuous peak measurement",
               "optimizer_iterations_range": [min(r["iterations"] for r in weight_rows), max(r["iterations"] for r in weight_rows)],
               "GPU_training_hours": 0, "external_TEST_access": False,
               "verification_code_sha256": sha256(Path(__file__))}
    write_json(ROOT / "logs/verification.json", receipt)
    print(json.dumps(receipt, indent=2))


def now_local():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    run()
