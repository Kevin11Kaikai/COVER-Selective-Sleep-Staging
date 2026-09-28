"""Bounded SC20 DuoF extension: fixed primary predictions, fold-local fitting."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
# Historical external inputs are not part of the public frozen-result replay.
S2 = Path(os.environ.get("COVER_STAGE2_ROOT", "historical_inputs/stage2"))
S3 = Path(os.environ.get("COVER_STAGE3_ROOT", "historical_inputs/stage3"))
S4 = Path(os.environ.get("COVER_STAGE4_ROOT", "historical_inputs/stage4"))
PH = "b264de85d45f7d49763a69743dd27504871a682e6066f0fd1167dee4caf8a73e"
PEOPLE = [f"{i:02d}" for i in range(20)]
METHOD_ID = "Baseline|DuoUQ_F|stage4_definition_v1"
METHODS = {"DuoF-0.05": 0.05, "DuoF-0.03": 0.03}
BASELINES = {
    "Learned-0.05": ("stage2", "Learned-Selected"),
    "Primary-only-0.05": ("stage2", "Primary-only"),
    "MSP-0.05": ("stage2", "MSP"),
    "Baseline-selected-0.05": ("stage2", "Baseline-Selected"),
    "Learned-0.03": ("stage3", "B0.03-Heterogeneous"),
    "Primary-only-0.03": ("stage3", "B0.03-Primary-only"),
}
PAIRS = [("DuoF-0.05", b) for b in list(BASELINES)[:4]] + [
    ("DuoF-0.03", "Learned-0.03"), ("DuoF-0.03", "Primary-only-0.03"),
    ("DuoF-0.03", "DuoF-0.05"),
]


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(S2.parent))
OLD = source_module("duof_original_selectors", S2 / "code/selectors.py")
STAGE3 = source_module("duof_stage3_helpers", S3 / "code/stage3_core.py")
DUO = source_module("duof_original_duo", S4 / "code/duo_baseline.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(safe(value), sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path).resolve()
    assert ROOT in path.parents
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False)
                   + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def key(item):
    return (str(item["dataset"]), str(item["subject_id"]), str(item["record_id"]),
            int(item["epoch_index"]))


def as_duo(data):
    return {"P_logits": data["logits"], "P": data["primary"],
            "F": data["feature"], "subjects": data["subjects"]}


def score_f(data, fitted):
    """Stage4 F branch only; probability is evaluated at the unchanged P argmax."""
    d = as_duo(data)
    wp, wf = fitted["weights"]
    probability = softmax(wp * d["P_logits"] + wf * DUO.auxiliary_logits(d, "F"), axis=1)
    return 1 - probability[np.arange(len(probability)), d["P"].argmax(1)]


def validate_dev(dev):
    outer = dev["outer"]
    split = read_json(S2 / "split_manifest.json")["folds"][outer]["outer_split"]
    assert list(dev["fit_subjects"]) == split["fit"]
    assert list(dev["cert_subjects"]) == split["cert"]
    assert dev["protocol_hash"] == PH
    expected = {"fit_data": split["fit"], "cert_data_blind": split["cert"],
                "outer_data_blind": [outer]}
    for part, people in expected.items():
        data = dev[part]
        if part != "fit_data" and set(data) & {"labels", "errors", "y", "ground_truth_labels"}:
            raise ValueError("Outcome field in blind prediction data")
        assert set(data["subjects"]) == set(people)
        ids = [key(x) for x in data["identities"]]
        assert len(ids) == len(set(ids)) == len(data["subjects"])
        assert all(x[1] == s for x, s in zip(ids, data["subjects"]))
        for field in ("logits", "primary", "feature"):
            assert data[field].shape == (len(ids), 5)
            assert np.isfinite(data[field]).all()
        assert np.allclose(softmax(data["logits"], axis=1), data["primary"], atol=1e-6)
        assert np.allclose(data["feature"].sum(1), 1, atol=1e-6)
        assert (data["feature"] >= 0).all()
    assert len(dev["fit_data"]["labels"]) == len(dev["fit_data"]["subjects"])


def compute_decisions(dev, fitted, cert_labels):
    """No outer ground truth is an argument, or read by this function."""
    outer = dev["outer"]
    fit, cert, outside = dev["fit_data"], dev["cert_data_blind"], dev["outer_data_blind"]
    scores = {"fit": score_f(fit, fitted), "cert": score_f(cert, fitted),
              "outer": score_f(outside, fitted)}
    grid = STAGE3.policies(scores["fit"], fit["subjects"])
    errors = (cert["primary"].argmax(1) != cert_labels).astype(int)
    rows = STAGE3.policy_rows(grid, scores["cert"], cert, errors, METHOD_ID, outer)
    selections, masks = {}, {}
    for name, ceiling in METHODS.items():
        selected = OLD.select_empirical_policy(rows, alpha=ceiling)
        selections[name] = selected
        masks[name] = STAGE3.accept(scores["outer"], outside["identities"],
                                    selected, METHOD_ID, outer)
    assert not (masks["DuoF-0.03"] & ~masks["DuoF-0.05"]).any()
    return scores, grid, rows, selections, masks


def individual(method, subject, accepted, errors, eligible):
    a, k, n = int(accepted), int(errors), int(eligible)
    return {"method": method, "subject": subject, "eligible": n, "accepted": a,
            "errors": k, "coverage": a / n, "risk": k / a if a else None,
            "passed": a >= 50 and 20 * k <= a, "zero_acceptance": a == 0}


def aggregate(method, rows):
    n = np.array([x["eligible"] for x in rows])
    a = np.array([x["accepted"] for x in rows])
    k = np.array([x["errors"] for x in rows])
    assert len(rows) == 20
    return {"method": method, "subjects": 20, "eligible": int(n.sum()),
            "accepted": int(a.sum()), "errors": int(k.sum()),
            "macro_coverage": float((a / n).mean()), "pooled_coverage": float(a.sum() / n.sum()),
            "subject_equal_ratio_risk": float((k / n).sum() / (a / n).sum()) if a.sum() else None,
            "pooled_risk": float(k.sum() / a.sum()) if a.sum() else None,
            "passed_subjects": sum(x["passed"] for x in rows),
            "zero_subjects": ";".join(x["subject"] for x in rows if not x["accepted"]),
            "failed_subjects": ";".join(x["subject"] for x in rows if not x["passed"]),
            "over_risk_subjects": ";".join(x["subject"] for x in rows if x["accepted"] and 20*x["errors"] > x["accepted"])}
