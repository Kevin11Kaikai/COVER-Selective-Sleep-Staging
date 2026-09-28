"""Recompute the released COVER results and check them against frozen references."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "2")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "New_Objective_Idea/src"),
                str(ROOT / "AutoML_Claude")]


def check_inputs():
    manifest = json.loads((ROOT / "provenance/import_manifest.json").read_text())
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Imported source/input changed: " + record["path"])
    return len(manifest["files"])


def compare_csv(actual: Path, reference: Path, atol=1e-12):
    def read(path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    left, right = read(actual), read(reference)
    if len(left) != len(right):
        raise ValueError(f"Row mismatch: {actual.name}: {len(left)} != {len(right)}")
    checked = 0
    for i, (a, b) in enumerate(zip(left, right)):
        if a.keys() != b.keys():
            raise ValueError("Column mismatch: " + actual.name)
        for key in a:
            if a[key] == b[key]:
                checked += 1
                continue
            try:
                x, y = float(a[key]), float(b[key])
                same = (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=0, abs_tol=atol)
            except ValueError:
                same = False
            if not same:
                raise ValueError(f"Mismatch: {actual.name} row {i + 1}, {key}: {a[key]} != {b[key]}")
            checked += 1
    return {"file": actual.name, "rows": len(left), "checked_cells": checked, "status": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    # Never overwrite versioned inputs, code, or references.
    if out == ROOT or any(out == p or p in out.parents for p in
            (ROOT / "artifacts", ROOT / "src", ROOT / "New_Objective_Idea", ROOT / "provenance")):
        raise ValueError("Choose a separate output directory")
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "verification.json"
    report = {"status": "RUNNING", "source_inputs": 0, "scientific_refitting": False}
    started = time.monotonic()
    try:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        report["source_inputs"] = check_inputs()
        import recompute_main
        recompute_main.OUT = out / "main"
        recompute_main.OUT.mkdir(exist_ok=True)
        recompute_main.main()
        checks = []
        for expected in sorted((ROOT / "artifacts/expected/main").glob("*.csv")):
            checks.append(compare_csv(out / "main" / expected.name, expected))
        expected_fewer = ROOT / "artifacts/expected/main/step2_subjects_fewer_errors.json"
        if json.loads(expected_fewer.read_text()) != json.loads((out / "main" / expected_fewer.name).read_text()):
            raise ValueError("Per-subject error comparison differs")
        import reproduce_nested
        report["development"] = reproduce_nested.run(ROOT, out / "development")
        frozen = ROOT / "artifacts/frozen/New_Objective_Idea/outputs/bceanchor_upgrade/tables"
        for name in ("subject_means", "summary", "paired_differences", "delete_one_subject"):
            checks.append(compare_csv(out / "development" / (name + ".csv"), frozen / (name + ".csv")))
        report["checks"] = checks
        if not args.no_figure:
            import plot_reproduction
            plot_reproduction.draw(out)
        report["packages"] = {name: importlib.metadata.version(name) for name in
                              ("numpy", "scipy", "scikit-learn", "pandas", "matplotlib")}
        report["status"] = "PASS"
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(out),
                      "checked_tables": len(report["checks"]),
                      "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
