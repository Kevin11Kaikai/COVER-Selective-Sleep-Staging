"""Focused delivery checks for BCE-anchored learning; preserves all old artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "src"), str(ROOT.parent / "AutoML_Claude")]

import numpy as np
from new_objective.storage import OUT, read, write, digest
from new_objective.anchor_report import AOUT, require_seal


def fraction(x):
    return Fraction(x["numerator"], x["denominator"]) if isinstance(x, dict) else Fraction(str(x))


def standalone_counts(mask, errors, subjects):
    """Plain counting implementation; intentionally does not call analysis metric helpers."""
    rows = []
    for s in sorted(set(subjects.tolist())):
        ids = np.flatnonzero(subjects == s)
        nn = len(ids)
        aa = sum(bool(mask[i]) for i in ids)
        kk = sum(bool(mask[i]) and bool(errors[i]) for i in ids)
        rows.append((nn, aa, kk))
    a, k, n = sum(v[1] for v in rows), sum(v[2] for v in rows), sum(v[0] for v in rows)
    cov = sum(Fraction(v[1], v[0]) for v in rows) / len(rows)
    numerator = sum(Fraction(v[2], v[0]) for v in rows)
    denominator = sum(Fraction(v[1], v[0]) for v in rows)
    return {"eligible": n, "accepted": a, "errors": k, "macro_coverage": float(cov),
        "pooled_coverage": a / n,
        "subject_equal_ratio_risk": float(numerator / denominator) if a else None,
        "pooled_risk": k / a if a else None,
        "passed_subjects": sum(aa >= 50 and 20 * kk <= aa for nn, aa, kk in rows),
        "zero_subjects": sum(aa == 0 for nn, aa, kk in rows),
        "below_min_support_subjects": sum(aa < 50 for nn, aa, kk in rows),
        "over_risk_subjects": sum(aa > 0 and 20 * kk > aa for nn, aa, kk in rows)}


def main():
    seal = require_seal()
    spec = read(ROOT / "anchor_protocol.json")
    checks = []
    def check(name, value):
        checks.append({"name": name, "passed": bool(value)})
        if not value:
            raise AssertionError(name)
    pre = read(AOUT / "preflight.json")
    protected = pre["protected_sha256"]
    for path, expected in protected.items():
        check("preserved " + path, digest(ROOT.parent / path) == expected)
    for path, expected in pre["identity"]["files"].items():
        check("fit code frozen " + path, digest(ROOT.parent / path) == expected)
    selected = read(AOUT / "fit_selections.json")
    final = read(AOUT / "fit_completed.json")
    check("final fits completed before first decision freeze",
        final["completed"] <= min(r["completed"] for r in seal["folds"].values()))
    check("all FIT choices before final fit seal", selected["sealed_at"] <= final["completed"])
    check("all stages share fingerprint", pre["identity"]["sha256"] == seal["fingerprint"])
    with np.load(AOUT / "replay_inputs.npz", allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    ids = json.loads(data["identities_json"].item())
    keys = ["|".join(str(d[k]) for k in ("dataset", "subject_id", "record_id", "epoch_index")) for d in ids]
    check("42308 unique epochs", len(set(keys)) == len(keys) == 42308)
    check("unchanged truth and primary correctness", np.array_equal(data["errors"], data["predictions"] != data["labels"]))
    # Rebuild each new hard decision from explicit threshold and an integer SHA256 ballot.
    for outer, record in seal["folds"].items():
        at = np.flatnonzero(data["subjects"] == outer)
        for cell, meta in record["cells"].items():
            j = data["cells"].tolist().index(cell)
            score = data["scores"][at, j]
            rule = meta["rule"]
            rebuilt = np.zeros(len(at), bool)
            if rule is not None:
                check(f"{outer}/{cell} CERT risk feasible", fraction(rule["risk"]) <= Fraction(1, 20))
                check(f"{outer}/{cell} CERT support feasible", rule["minimum_accepted"] >= 50)
                rebuilt = score < rule["threshold"]
                ratio = fraction(rule["tie_fraction"])
                for local in np.flatnonzero(score == rule["threshold"]):
                    d = ids[at[local]]
                    text = (f"stage2-tie-v2|protocol={spec['protocol_hash']}|outer={outer}|dataset={d['dataset']}|"
                        f"subject={d['subject_id']}|record={d['record_id']}|epoch={d['epoch_index']}|method={spec['tie_method_id']}")
                    integer = int.from_bytes(hashlib.sha256(text.encode()).digest(), "big")
                    rebuilt[local] = integer * ratio.denominator < ratio.numerator * (2**256)
            check(f"{outer}/{cell} saved mask equals reconstructed rule", np.array_equal(rebuilt, data["masks"][at, j]))
    with (OUT / "tables/anchor_summary.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tie_keys = [hashlib.sha256(key.encode()).hexdigest() for key in keys]
    subjects, errors = data["subjects"], data["errors"].astype(bool)
    for j, cell in enumerate(data["cells"].tolist()):
        matched = np.zeros(len(errors), bool)
        for s in sorted(set(subjects.tolist())):
            at = np.flatnonzero(subjects == s)
            quota = int(data["reference_masks"][at, 0].sum())
            ordered = sorted(at.tolist(), key=lambda i: (float(data["scores"][i, j]), tie_keys[i]))
            matched[ordered[:quota]] = True
        for context, mask in [("cert", data["masks"][:, j]), ("matched", matched)]:
            values = standalone_counts(mask, errors, subjects)
            row = next(r for r in rows if r["context"] == context and r["cell"] == cell)
            for key, expected in values.items():
                actual = None if row[key] == "" else float(row[key])
                check(f"count crosscheck {context}/{cell}/{key}",
                    actual is None if expected is None else actual is not None and abs(actual - expected) <= 1e-12)
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_anchored*.py", "-v"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    check("anchored targeted tests", result.returncode == 0)
    before = {p.name: digest(p) for p in (OUT / "tables").glob("anchor_*.csv")}
    rerun = subprocess.run([sys.executable, "-X", "utf8", "-B", "run_anchored.py", "replay"], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8")
    check("second outcome replay", rerun.returncode == 0)
    after = {p.name: digest(p) for p in (OUT / "tables").glob("anchor_*.csv")}
    check("CSV replay byte equality", before == after)
    check("report HTML has embedded images", (ROOT / "docs/ANCHOR_RESULTS_ZH.html").read_text(encoding="utf-8").count('src="data:image/png;base64,') == 2)
    write(AOUT / "validation.json", {"status": "PASS", "checks": checks,
        "passed": len(checks), "preserved_files": len(protected),
        "targeted_tests_output": result.stdout + result.stderr, "replay_output": rerun.stdout,
        "csv_sha256": after, "validator_sha256": digest(Path(__file__)),
        "verification_type": "same-agent separate counting implementation; not independent external review"})
    print(f"Anchored validation PASS: {len(checks)} checks; {len(protected)} old files preserved; {len(after)} CSVs identical", flush=True)


if __name__ == "__main__":
    main()
