"""Pure selection/aggregation rules and contained study accounting."""
from __future__ import annotations
import hashlib
import json
import os
import time
from fractions import Fraction
import numpy as np
from .storage import ROOT, OUT, PROJECT, read, write, digest, utc, safe
from .policies import exact

UOUT = OUT / "bceanchor_upgrade"


def spec():
    return read(ROOT / "bceanchor_upgrade_protocol.json")


def object_hash(value):
    return hashlib.sha256(json.dumps(safe(value), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def array_hash(*values):
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def fit_identity():
    names = ["bceanchor_upgrade_core.py", "bceanchor_upgrade_solver.py",
             "bceanchor_upgrade.py", "anchor_solver.py", "solvers.py", "losses.py",
             "calibration.py", "policies.py", "experiment.py", "storage.py"]
    paths = [ROOT / "bceanchor_upgrade_protocol.json", ROOT / "run_bceanchor_upgrade.py"]
    paths += [ROOT / "src/new_objective" / n for n in names]
    paths += [PROJECT / p for p in ("src/stage1_analysis/selection_ablation.py",
              "src/stage1_analysis/risk_features.py", "src/stage1_analysis/acceptance.py",
              "AutoML_Claude/automl/grids.py")]
    hashes = {p.relative_to(PROJECT).as_posix(): digest(p) for p in paths}
    return {"sha256": object_hash(hashes), "files": hashes}


def partition_subjects(subjects, context, fold, cfg):
    """Label-free, deterministic balanced four-way partition of a parent set."""
    names = sorted(set(str(s) for s in subjects))
    if len(names) != 12:
        raise ValueError("A parent training fold must have exactly 12 subjects")
    def key(s):
        tag = f"{cfg['partition_namespace']}|{cfg['protocol_hash']}|{context}|{fold}|{s}"
        return hashlib.sha256(tag.encode()).hexdigest()
    ordered = sorted(names, key=key)
    return [{"train": sorted(set(names) - set(ordered[3*j:3*j+3])),
             "validation": sorted(ordered[3*j:3*j+3])} for j in range(4)]


def candidate_key(row, method, nominal=True):
    parameter = float(row["parameter"])
    tie = -parameter if method == "BCEAnchor" else parameter
    picked = row["picked"]
    if picked is None:
        return (1, Fraction(0), Fraction(2), tie)
    first = exact(picked["q"] if nominal else picked["coverage"])
    return (0, -first, exact(picked["risk"]), tie)


def select_candidate(rows, method, nominal=True):
    if not rows or any(r.get("status", "OK") != "OK" for r in rows):
        raise RuntimeError("Incomplete/invalid candidate matrix; no silent exclusions")
    return min(rows, key=lambda r: candidate_key(r, method, nominal))


def equal_norm_control(beta0, full, average, tolerance=1e-12):
    d, m = full-beta0, average-beta0
    denominator = float(np.linalg.norm(d))
    floor = tolerance * max(1., float(np.linalg.norm(beta0)))
    if denominator <= floor:
        return None, {"status": "UNDEFINED_ZERO_FULL_DIRECTION", "floor": floor}
    coefficient = float(np.linalg.norm(m)) / denominator
    return beta0 + coefficient*d, {"status": "OK", "coefficient": coefficient,
                                    "kind": "shrink" if coefficient <= 1 else "expand"}


def average_models(models):
    if len(models) != 4 or any(m["status"] != "OK" for m in models):
        raise RuntimeError("Exactly four valid members required")
    return {"status": "OK", "theta": np.mean([m["theta"] for m in models], axis=0),
            "intercept": float(np.mean([m["intercept"] for m in models]))}


def counts_risk(counts):
    """Exact ratio of subject-normalized errors and acceptances."""
    if not counts:
        raise ValueError("No eligible subjects")
    c = sum((Fraction(int(r["accepted"]), int(r["n"])) for r in counts), Fraction(0))
    e = sum((Fraction(int(r["errors"]), int(r["n"])) for r in counts), Fraction(0))
    return {"coverage": c/len(counts), "risk": e/c if c else None}


class UpgradeBudget:
    """New grant; the historical compute ledger is never rewritten."""
    def __init__(self, mode):
        self.path = UOUT / "compute_budget.json"
        self.mode, self.started = mode, time.monotonic()
        self.last = 0.
        self.fit = mode in ("pilot", "fit")
        self.state = read(self.path) if self.path.exists() else {
            "schema": "incremental-24h-v1", "used_seconds": 0., "fit_seconds": 0.,
            "historical_seconds": read(OUT / "compute_budget.json")["used_seconds"],
            "historical_ledger_sha256": digest(OUT / "compute_budget.json"), "sessions": []}
        if self.state["sessions"] and self.state["sessions"][-1]["status"] == "RUNNING":
            previous = self.state["sessions"][-1]
            # The CLI lock prevents overlapping jobs. An interrupted session retains
            # its last heartbeat charge; no time or prior work is reset on resume.
            previous["status"] = "INTERRUPTED_AT_LAST_HEARTBEAT"
        self.base = self.state["used_seconds"]
        self.fit_base = self.state["fit_seconds"]
        self.index = len(self.state["sessions"])
        self.state["sessions"].append({"mode": mode, "pid": os.getpid(), "started": utc(),
                                       "seconds": 0., "status": "RUNNING"})
        self.check(force=True)

    def check(self, force=False):
        elapsed = time.monotonic()-self.started
        if not force and elapsed-self.last < 1.:
            return
        self.last = elapsed
        self.state["used_seconds"] = self.base+elapsed
        self.state["fit_seconds"] = self.fit_base+(elapsed if self.fit else 0.)
        self.state["sessions"][self.index]["seconds"] = elapsed
        write(self.path, self.state)
        cfg = spec()
        if self.state["used_seconds"] >= cfg["incremental_total_seconds"]:
            raise TimeoutError("New 24-hour grant exhausted; incomplete matrix preserved")
        if self.fit and self.state["fit_seconds"] >= cfg["incremental_fit_seconds"]:
            raise TimeoutError("New 20-hour fitting allocation exhausted")

    def finish(self, status):
        elapsed = time.monotonic()-self.started
        self.state["used_seconds"] = self.base+elapsed
        self.state["fit_seconds"] = self.fit_base+(elapsed if self.fit else 0.)
        self.state["sessions"][self.index].update(status=status, seconds=elapsed, ended=utc())
        write(self.path, self.state)
