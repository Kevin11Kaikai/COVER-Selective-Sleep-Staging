"""Contained atomic writes, provenance, and cumulative compute accounting."""
from __future__ import annotations
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
OUT = ROOT / "outputs"

def utc():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def safe(x):
    from fractions import Fraction
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return safe(x.tolist())
    if isinstance(x, np.generic):
        return safe(x.item())
    if isinstance(x, Fraction):
        return {"numerator": x.numerator, "denominator": x.denominator}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x

def target(path):
    p = Path(path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f"Write outside experiment refused: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write(path, data):
    p = target(path)
    q = p.with_suffix(p.suffix + ".partial")
    q.write_text(json.dumps(safe(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(q, p)
            break
        except PermissionError as error:
            if attempt == 5 or getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            time.sleep(.05 * (attempt + 1))

def text_file(path, text):
    p = target(path)
    p.write_text(text, encoding="utf-8")

def arrays(path, **data):
    p = target(path)
    q = p.with_suffix(p.suffix + ".partial")
    with q.open("wb") as f:
        np.savez_compressed(f, **data)
    os.replace(q, p)

def model_save(path, model):
    import joblib
    p = target(path)
    q = p.with_suffix(p.suffix + ".partial")
    joblib.dump(model, q)
    os.replace(q, p)

def specification():
    return read(ROOT / "protocol.json")

def fitting_identity():
    files = [ROOT / "protocol.json", ROOT / "run.py"]
    files += [ROOT / "src/new_objective" / n for n in ("storage.py", "losses.py", "solvers.py", "calibration.py", "experiment.py", "policies.py")]
    files += [PROJECT / p for p in ("src/stage1_analysis/selection_ablation.py", "src/stage1_analysis/risk_features.py", "src/stage1_analysis/acceptance.py", "AutoML_Claude/automl/grids.py")]
    hashes = {str(p.relative_to(PROJECT)): digest(p) for p in files}
    h = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": h, "files": hashes}

class Budget:
    def __init__(self, mode):
        self.mode, self.started = mode, time.monotonic()
        self.path = OUT / "compute_budget.json"
        self.state = read(self.path) if self.path.exists() else {"used_seconds": 0., "sessions": []}
        self.base = self.state["used_seconds"]
        self.limit = specification()["budget_seconds"]
        if mode in ("fit", "pilot"):
            self.limit = specification()["fit_budget_seconds"]
        self.index = len(self.state["sessions"])
        self.state["sessions"].append({"mode": mode, "started": utc(), "seconds": 0., "status": "RUNNING"})
        self.check()

    def check(self):
        elapsed = time.monotonic() - self.started
        self.state["used_seconds"] = self.base + elapsed
        self.state["sessions"][self.index]["seconds"] = elapsed
        write(self.path, self.state)
        if self.state["used_seconds"] >= self.limit:
            raise TimeoutError("Cumulative experiment compute budget reached")
        if self.mode == "pilot" and elapsed >= specification()["pilot_budget_seconds"]:
            raise TimeoutError("Pilot budget reached")

    def finish(self, status):
        self.state["used_seconds"] = self.base + time.monotonic() - self.started
        self.state["sessions"][self.index].update(seconds=time.monotonic()-self.started, status=status, ended=utc())
        write(self.path, self.state)
