"""Re-anchor step-(i) comparisons on Linear-BCE using saved scores and masks only.

No model is fitted and no label is read beyond the saved primary-error vector.
Step 3 (reproduction) runs first and aborts if any published number differs.
A portable wrapper directs outputs to a separate run directory.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stage1_analysis.metrics import (bootstrap_metrics, count_subjects, identity_key,
                                         identity_ties, interval, quota_mask, ranked_orders,
                                         summary)

OUT = ROOT / "outputs/reproduction/main"
DATA = ROOT / "artifacts/frozen"
FROZEN = DATA / "outputs/analysis_inputs/frozen_analysis_inputs.npz"
NEW = DATA / "New_Objective_Idea/outputs"
PUBLISHED = DATA / "outputs/strengthening/stage1_strengthening_v1/tables/paired_bootstrap.csv"
SEED, REPS = 20260901, 10000


def load_frozen():
    z = np.load(FROZEN, allow_pickle=False)
    ids = json.loads(str(z["identities_json"]))
    keys = [identity_key(r) for r in ids]
    scores = {str(n): z["scores"][:, i] for i, n in enumerate(z["score_names"])}
    masks = {str(n): z["masks"][:, i] for i, n in enumerate(z["mask_names"])}
    return keys, z["subjects"].astype(str), z["errors"].astype(bool), scores, masks


def load_outer_cells(folder: Path, cells: list[str], keys: list[tuple]):
    """Assemble OUTER scores and masks of the requested cells across the 20 outer folds."""
    index = {k: i for i, k in enumerate(keys)}
    size = len(keys)
    score = {c: np.full(size, np.nan) for c in cells}
    mask = {c: np.zeros(size, bool) for c in cells}
    seen = np.zeros(size, bool)
    for outer in range(20):
        z = np.load(folder / f"outer-{outer:02d}.npz", allow_pickle=False)
        names = [str(c) for c in z["cells"]]
        pos = np.array([index[identity_key(r)] for r in json.loads(str(z["identities_json"]))])
        assert not seen[pos].any(), "epoch scored in two outer folds"
        seen[pos] = True
        for c in cells:
            j = names.index(c)
            score[c][pos] = z["scores"][:, j]
            mask[c][pos] = z["masks"][:, j].astype(bool)
    assert seen.all(), "some epochs have no OUTER score"
    return score, mask


def main():
    keys, subjects, errors, fscores, fmasks = load_frozen()
    people = sorted(set(subjects.tolist()))
    n = np.array([np.count_nonzero(subjects == s) for s in people])
    ties = identity_ties(keys)
    lin_s, lin_m = load_outer_cells(NEW / "decisions", ["BCE|20", "SELE|20"], keys)
    anc_s, anc_m = load_outer_cells(NEW / "anchored/decisions", ["FixedScale|20", "BCEAnchor|20"], keys)

    step1_names = ["Learned-Selected", "MSP", "DuoF", "Primary-only", "Forward-KL"]
    own = {m: fmasks[f"{m}|0.05|empirical"] for m in step1_names}
    own.update({"Linear-BCE": lin_m["BCE|20"], "SELE": lin_m["SELE|20"],
                "FixedScale": anc_m["FixedScale|20"], "BCEAnchor": anc_m["BCEAnchor|20"]})
    score = {m: fscores[m] for m in step1_names}
    score.update({"Linear-BCE": lin_s["BCE|20"], "SELE": lin_s["SELE|20"],
                  "FixedScale": anc_s["FixedScale|20"], "BCEAnchor": anc_s["BCEAnchor|20"]})
    orders = {m: ranked_orders(score[m], ties, subjects, people) for m in score}

    def matched(anchor: str):
        _, quotas, _ = count_subjects(own[anchor], errors, subjects, people)
        out = {m: quota_mask(orders[m], quotas, len(errors)) for m in score}
        out[anchor] = own[anchor]           # anchor keeps its own rule, as in the paper
        return out

    rng = np.random.Generator(np.random.PCG64(SEED))
    draws = rng.integers(0, len(people), size=(REPS, len(people)))
    counts = lambda mask: count_subjects(mask, errors, subjects, people)

    def paired(context, masks, a, others, metrics):
        rows = []
        ca = counts(masks[a]); ba = bootstrap_metrics(*ca, draws); sa = summary(*ca)
        for b in others:
            cb = counts(masks[b]); bb = bootstrap_metrics(*cb, draws); sb = summary(*cb)
            for metric in metrics:
                iv = interval(ba[metric] - bb[metric])
                rows.append({"context": context, "method_a": a, "method_b": b, "metric": metric,
                             "a_minus_b": sa[metric] - sb[metric], **iv})
        return rows

    # ---------------- Step 3: reproduce published numbers ----------------
    ref = matched("Learned-Selected")
    got = {m: int((ref[m] & errors).sum()) for m in ref}
    expect = {"Learned-Selected": 817, "MSP": 1369, "Primary-only": 1133, "DuoF": 946,
              "Forward-KL": 2171, "Linear-BCE": 794, "SELE": 798, "FixedScale": 803, "BCEAnchor": 780}
    own_expect = {"Learned-Selected": (19892, 817), "Linear-BCE": (20639, None),
                  "FixedScale": (21455, None), "SELE": (20875, None), "BCEAnchor": (20805, None)}
    repro = {"matched_errors_original_budget": {m: {"recomputed": got[m], "published": expect[m],
                                                     "match": got[m] == expect[m]} for m in expect}}
    repro["own_rule_accepted"] = {m: {"recomputed": int(own[m].sum()), "published": v[0],
                                      "match": int(own[m].sum()) == v[0]} for m, v in own_expect.items()}
    with PUBLISHED.open(encoding="utf-8") as f:
        pub = {(r["context"], r["method_a"], r["method_b"], r["metric"]): r for r in csv.DictReader(f)}
    checks = paired("frozen_cert05", own, "Learned-Selected", ["MSP", "DuoF", "Primary-only"],
                    ["macro_coverage", "pooled_risk", "subject_equal_ratio_risk"])
    checks += paired("matched_original_reference", ref, "Learned-Selected", ["MSP", "DuoF", "Primary-only"],
                     ["pooled_risk"])
    repro["paired_intervals"] = []
    for r in checks:
        p = pub[(r["context"], r["method_a"], r["method_b"], r["metric"])]
        ok = all(abs(float(p[k]) - r[k2]) < 1e-12 for k, k2 in (("a_minus_b", "a_minus_b"), ("low", "low"), ("high", "high")))
        repro["paired_intervals"].append({**{k: r[k] for k in ("context", "method_b", "metric")},
            "recomputed": [r["a_minus_b"], r["low"], r["high"]],
            "published": [float(p["a_minus_b"]), float(p["low"]), float(p["high"])], "match": ok})
    all_ok = (all(v["match"] for v in repro["matched_errors_original_budget"].values())
              and all(v["match"] for v in repro["own_rule_accepted"].values())
              and all(v["match"] for v in repro["paired_intervals"]))
    repro["all_match"] = all_ok
    (OUT / "step3_reproduction.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
    print("STEP 3 reproduction all_match =", all_ok)
    if not all_ok:
        print(json.dumps(repro, indent=2))
        sys.exit(1)

    # ---------------- Step 1: own-rule intervals anchored on Linear-BCE ----------------
    others = ["MSP", "Forward-KL", "Primary-only", "DuoF", "Learned-Selected"]
    step1 = paired("frozen_cert05", own, "Linear-BCE", others,
                   ["macro_coverage", "subject_equal_ratio_risk", "pooled_risk"])
    own_summary = [{"method": m, **summary(*counts(own[m]))} for m in own]

    # ---------------- Step 2: matched budget anchored on Linear-BCE ----------------
    lin = matched("Linear-BCE")
    matched_summary = [{"method": m, "accepted": int(lin[m].sum()), "errors": int((lin[m] & errors).sum()),
                        "pooled_risk": float((lin[m] & errors).sum() / lin[m].sum())} for m in lin]
    step2 = paired("matched_linear_bce", lin, "Linear-BCE",
                   ["MSP", "Forward-KL", "Primary-only", "DuoF", "Learned-Selected",
                    "SELE", "FixedScale", "BCEAnchor"], ["pooled_risk"])
    fewer = {b: int(sum(1 for i, s in enumerate(people)
                        if ((lin["Linear-BCE"] & errors & (subjects == s)).sum()
                            < (lin[b] & errors & (subjects == s)).sum()))) for b in lin if b != "Linear-BCE"}

    for name, rows in (("step1_own_rule_intervals.csv", step1), ("own_rule_summary.csv", own_summary),
                       ("step2_matched_linear_bce.csv", matched_summary),
                       ("step2_matched_intervals.csv", step2)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "step2_subjects_fewer_errors.json").write_text(json.dumps(fewer, indent=2), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
