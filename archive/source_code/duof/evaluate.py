"""Evaluate the locked DuoF extension and exact frozen baseline decisions."""
from __future__ import annotations

import csv
import json
import time

import numpy as np

from core import *
from run_comparison import verify_frozen


def save_csv(name, rows):
    path = ROOT / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    timer = time.perf_counter()
    verify_frozen()
    locked = read_json(ROOT / "config/all_decisions_locked.json")
    assert locked["policy_decisions"] == 40
    for path, expected in locked["files"].items():
        assert sha256(ROOT / path) == expected
    inputs = {}

    def track(path):
        inputs[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    opening = now()
    bundle = S2 / "sealed_outer_results/ALL_OUTER_SEALED.npz"
    track(bundle)
    with np.load(bundle, allow_pickle=False) as z:
        ids = json.loads(str(z["identities_json"].item()))
        keys = [key(x) for x in ids]
        subjects, truth = z["outer_subject_ids"].copy(), z["ground_truth_labels"].copy()
        predictions = z["primary_predictions"].copy()
        errors = predictions != truth
        assert np.array_equal(errors, z["error_indicators"])
        assert np.array_equal(predictions, z["primary_probabilities"].argmax(1))
        assert str(z["protocol_hash"].item()) == PH
        baseline_masks = {str(n): z["acceptance_masks"][:, i].copy()
                          for i, n in enumerate(z["mask_names"])}
    assert len(keys) == len(set(keys)) == 42308
    assert len({k[2] for k in keys}) == 39 and set(subjects) == set(PEOPLE)
    index = {k: i for i, k in enumerate(keys)}
    masks = {}
    for name, (stage, source) in BASELINES.items():
        if stage == "stage2":
            masks[name] = baseline_masks[f"{source}|0.05|empirical"]
        else:
            combined = np.zeros(len(keys), dtype=bool)
            visited = np.zeros(len(keys), dtype=bool)
            for outer in PEOPLE:
                path = S3 / "checkpoints" / f"outer-{outer}" / source / "decisions.npz"
                track(path)
                with np.load(path, allow_pickle=False) as z:
                    order = [index[key(x)] for x in json.loads(str(z["identities_json"].item()))]
                    assert not bool(z["contains_outer_labels"].item())
                    assert np.array_equal(z["primary_predictions"], predictions[order])
                    assert (subjects[order] == outer).all() and not visited[order].any()
                    combined[order] = z["outer_accept"]
                    visited[order] = True
            assert visited.all()
            masks[name] = combined
    for name in METHODS:
        combined, visited = np.zeros(len(keys), dtype=bool), np.zeros(len(keys), dtype=bool)
        for outer in PEOPLE:
            path = ROOT / "folds" / f"outer-{outer}" / "decisions.npz"
            with np.load(path, allow_pickle=False) as z:
                order = [index[key(x)] for x in json.loads(str(z["identities_json"].item()))]
                assert not bool(z["contains_outer_labels"].item())
                assert np.array_equal(z["primary_predictions"], predictions[order])
                assert (subjects[order] == outer).all() and not visited[order].any()
                combined[order] = z[f"accept_{name}"]
                visited[order] = True
        assert visited.all()
        masks[name] = combined
    assert not (masks["DuoF-0.03"] & ~masks["DuoF-0.05"]).any()
    order = ["Baseline-selected-0.05", "MSP-0.05", "Primary-only-0.05", "Learned-0.05", "DuoF-0.05",
             "Primary-only-0.03", "Learned-0.03", "DuoF-0.03"]
    masks = {name: masks[name] for name in order}
    groups = {s: subjects == s for s in PEOPLE}
    per_subject = [individual(m, s, mask[groups[s]].sum(), (mask & errors)[groups[s]].sum(), groups[s].sum())
                   for m, mask in masks.items() for s in PEOPLE]
    summaries = [aggregate(m, [x for x in per_subject if x["method"] == m]) for m in masks]
    by_person = {(r["method"], r["subject"]): r for r in per_subject}
    by_method = {r["method"]: r for r in summaries}
    # Validate each reused baseline row against its frozen table before comparison.
    s2_table, s3_table = S2 / "tables/per_subject_results.csv", S3 / "tables/per_subject_results.csv"
    track(s2_table)
    track(s3_table)
    table2, table3 = read_csv(s2_table), read_csv(s3_table)
    verified = 0
    for name, (stage, source) in BASELINES.items():
        selected = [r for r in (table2 if stage == "stage2" else table3) if r["method"] == source
                    and (stage != "stage2" or r["alpha"] == "0.05" and r["selector"] == "empirical")]
        assert len(selected) == 20
        for row in selected:
            actual = by_person[name, row["subject"]]
            assert (actual["accepted"], actual["errors"], actual["eligible"]) == (
                int(row["accepted"]), int(row["accepted_errors"]), int(row["eligible"]))
            verified += 1

    paired, transitions, exchanges, exchange_subjects = [], [], [], []
    for a_name, b_name in PAIRS:
        a, b = masks[a_name], masks[b_name]
        changes = []
        added_pass, removed_pass = [], []
        for s in PEOPLE:
            aa, bb = by_person[a_name, s], by_person[b_name, s]
            changes.append(aa["coverage"] - bb["coverage"])
            if aa["passed"] and not bb["passed"]:
                added_pass.append(s)
            if bb["passed"] and not aa["passed"]:
                removed_pass.append(s)
            transitions.append({"method_a": a_name, "method_b": b_name, "subject": s,
                "a_accepted": aa["accepted"], "a_errors": aa["errors"], "a_risk": aa["risk"], "a_pass": aa["passed"],
                "b_accepted": bb["accepted"], "b_errors": bb["errors"], "b_risk": bb["risk"], "b_pass": bb["passed"],
                "coverage_delta": changes[-1], "acceptance_turnover": int((a ^ b)[groups[s]].sum())})
        regions = {"shared": a & b, "added": a & ~b, "removed": b & ~a}
        assert int(a.sum()) - int(b.sum()) == int(regions["added"].sum()) - int(regions["removed"].sum())
        assert int((a & errors).sum()) - int((b & errors).sum()) == int((regions["added"] & errors).sum()) - int((regions["removed"] & errors).sum())
        for label, mask in regions.items():
            n, k = int(mask.sum()), int((mask & errors).sum())
            exchanges.append({"method_a": a_name, "method_b": b_name, "region": label,
                              "accepted": n, "errors": k, "pooled_risk": k/n if n else None})
            for s in PEOPLE:
                ns, ks = int(mask[groups[s]].sum()), int((mask & errors)[groups[s]].sum())
                exchange_subjects.append({"method_a": a_name, "method_b": b_name, "region": label,
                                         "subject": s, "accepted": ns, "errors": ks, "risk": ks/ns if ns else None})
        changes = np.array(changes)
        pair = {"method_a": a_name, "method_b": b_name,
                "macro_coverage_delta": float(changes.mean()),
                "passed_subject_delta": by_method[a_name]["passed_subjects"] - by_method[b_name]["passed_subjects"],
                "pass_gains": ";".join(added_pass), "pass_losses": ";".join(removed_pass),
                "coverage_gains": int((changes > 0).sum()), "coverage_losses": int((changes < 0).sum()),
                "coverage_ties": int((changes == 0).sum()), "acceptance_turnover": int((a ^ b).sum()),
                "minimum_leave_one_out_coverage_delta": float(min(np.delete(changes, i).mean() for i in range(20))),
                "delete_largest_gain_coverage_delta": float(np.delete(changes, np.argmax(changes)).mean()),
                "delete_two_largest_gains_coverage_delta": float(np.delete(changes, np.argsort(changes)[-2:]).mean()),
                "inference": "descriptive, overlapping LOSO fits; not an independent-fold significance test"}
        for metric in ["subject_equal_ratio_risk", "pooled_risk"]:
            av, bv = by_method[a_name][metric], by_method[b_name][metric]
            pair[metric + "_delta"] = av - bv if av is not None and bv is not None else None
        paired.append(pair)

    stages = []
    for method, mask in masks.items():
        for stage_kind, labels in [("true", truth), ("predicted", predictions)]:
            for cls, name in enumerate(["W", "N1", "N2", "N3", "REM"]):
                group = labels == cls
                n, a, k = int(group.sum()), int((mask & group).sum()), int((mask & group & errors).sum())
                stages.append({"method": method, "stage_kind": stage_kind, "stage": name, "eligible": n,
                               "accepted": a, "errors": k, "correct_autoaccepted": a-k,
                               "coverage": a/n if n else None, "risk": k/a if a else None,
                               "diagnostic_only": True})
    save_csv("aggregate.csv", summaries)
    save_csv("per_subject.csv", per_subject)
    save_csv("paired_comparisons.csv", paired)
    save_csv("pass_transitions.csv", transitions)
    save_csv("acceptance_exchange.csv", exchanges)
    save_csv("acceptance_exchange_by_subject.csv", exchange_subjects)
    save_csv("stage_support.csv", stages)
    (ROOT / "evaluation").mkdir(exist_ok=True)
    np.savez_compressed(ROOT / "evaluation/analysis_bundle.npz", identities_json=np.array(json.dumps(ids)),
                        subjects=subjects, truth=truth, primary_predictions=predictions,
                        method_names=np.array(list(masks)), masks=np.column_stack(list(masks.values())))
    receipt = {"status": "PASS", "evaluation_started_utc": opening, "evaluation_finished_utc": now(),
               "all_decisions_locked_utc": locked["locked_utc"], "eligible": len(keys), "subjects": 20,
               "decision_pairs_evaluated": 40, "frozen_baseline_rows_verified": verified,
               "primary_labels_unchanged": True, "DuoF_003_subset_of_005": True,
               "seconds": time.perf_counter()-timer, "analysis_code_sha256": sha256(Path(__file__)),
               "attribution": "single-agent self-check; no independent scientific validation",
               "external_TEST_access": False, "certified": False}
    assert locked["locked_utc"] < opening
    write_json(ROOT / "evaluation/input_manifest.json", inputs)
    write_json(ROOT / "evaluation/receipt.json", receipt)
    write_json(ROOT / "evaluation/results.json", {"aggregate": summaries, "paired": paired,
                "stage_support": stages, "per_subject": per_subject, "exchanges": exchanges})
    state = read_json(ROOT / "run_state.json")
    state.update(status="EVALUATED_AWAITING_VERIFICATION_AND_REPORT", updated_utc=now())
    write_json(ROOT / "run_state.json", state)
    print(json.dumps({"receipt": receipt, "aggregate": summaries, "paired": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
