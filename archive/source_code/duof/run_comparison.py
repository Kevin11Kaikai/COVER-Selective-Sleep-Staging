"""Freeze, fit/lock twenty folds, then evaluate; never modifies historical results."""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import psutil
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from core import *


def protocol():
    return {
        "name": "SC20 DuoF strong-baseline extension v1", "date": "2026-09-12",
        "status": "HUMAN_CONFIRMED_AND_EXECUTION_AUTHORIZED",
        "user_confirmation": "Ok,就按照你的思路去做强基线比较吧！",
        "question": "How does fold-local Stage4-defined DuoF compare with the frozen Stage2/3 scorers under the same CERT ceilings?",
        "cohort": {"subjects": 20, "records": 39, "epochs": 42308},
        "split": "Unchanged Stage2 LOSO: 15 FIT, 4 CERT, 1 outer; all upstream models/predictions reused",
        "fit": {"input": "FIT component-OOF primary logits and calibrated RF probabilities",
                "weights_fitted_on": "all 15 current-fold FIT subjects only",
                "objective": "subject-equal multiclass NLL of softmax(wP*P_logits+wF*log(clip(F,1e-10,1)))",
                "parameters": ["wP", "wF"], "bounds": [[0, 20], [0, 20]], "initial": [1.0, 1.0],
                "optimizer": "original Stage4 L-BFGS-B", "maxiter": 500, "ftol": 1e-12, "gtol": 1e-8,
                "duo_weight_crossfit": False,
                "FIT_quantile_scores": "in-sample with respect to Duo weights, as in Stage4; component predictions remain OOF",
                "new_component_training": False, "stage4_learned_weights_reused": False},
        "score": "1 - fused probability at the ORIGINAL primary argmax; never ensemble-max confidence",
        "selection": {"candidates": "20 exact subject-equal FIT quantiles .05,.10,...,1.00",
                      "CERT_ceilings": METHODS, "support": "Every CERT subject accepted>=50",
                      "objective": "maximize CERT macro coverage; then lower ratio-risk; then lower q",
                      "risk": "sum_s(errors_s/eligible_s)/sum_s(accepted_s/eligible_s)",
                      "no_feasible_policy": "zero acceptance", "tie_protocol_hash": PH,
                      "tie_method_id_shared_by_both_ceilings": METHOD_ID},
        "baseline_mapping": BASELINES, "paired_comparisons": PAIRS,
        "primary_comparisons": [PAIRS[0], PAIRS[4]],
        "endpoints": ["20-subject macro coverage", "subject-equal ratio risk", "pooled accepted risk",
                      "pass count: accepted>=50 and errors/accepted<=5%, all20 denominator", "zero acceptance"],
        "diagnostics": ["paired subject transitions", "added/removed/shared acceptance regions",
                        "fixed-policy leave-one-out influence", "true/predicted stage support and errors"],
        "evaluation": "Freeze all forty policies and masks before evaluation entry; stable epoch-ID join",
        "interpretation": "retrospective SC20 extension; historical Stage2 NO-GO retained; no independent replication or certification",
        "resources": {"CPU_thread_limit": 2, "GPU_training_hours": 0, "new_downloads": False,
                      "prior_estimate_wall_minutes": [5, 20], "prior_estimate_total_work_hours": [2, 4],
                      "estimate_is_not_hard_user_budget": True},
        "stop": "Deliver every prespecified comparison including negative results; no adaptive method, threshold or seed search",
        "external_TEST": "NOT_ACCESSED", "existing_MVP": "PRESERVED",
    }


def prepare():
    path = ROOT / "config/protocol.json"
    planned = protocol()
    if path.exists():
        assert read_json(path) == safe(planned), "Protocol drift; do not reset an existing run"
        print("Protocol already frozen; validating existing state", flush=True)
        verify_frozen()
        return
    files = [S2 / "split_manifest.json", S2 / "frozen_protocol.json", S2 / "artifact_manifest.json",
             S2 / "code/selectors.py", S2 / "code/splits.py", S3 / "code/stage3_core.py",
             S3 / "code/run_experiment.py", S3 / "config/frozen_config.json", S4 / "code/duo_baseline.py"]
    expected_bundles = []
    splits = read_json(S2 / "split_manifest.json")["folds"]
    for outer in PEOPLE:
        root = S2 / "development/risk" / f"outer-{outer}"
        dev = root / "development.joblib"
        meta = root / "development_manifest.json"
        assert read_json(meta)["artifact_sha256"] == sha256(dev), outer
        files += [dev, meta]
        expected_bundles += [S2 / "development/components" / f"outer-{outer}" / f"inner-{s}" / "blind_predictions.npz"
                             for s in splits[outer]["outer_split"]["fit"]]
        expected_bundles += [S2 / "development/components" / f"outer-{outer}" / "final/blind_predictions.npz"]
    assert len(expected_bundles) == 320 and all(p.is_file() for p in expected_bundles)
    inputs = {str(p): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}
    write_json(path, planned)
    write_json(ROOT / "config/input_manifest.json", inputs)
    code = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "code").glob("*.py"))}
    write_json(ROOT / "config/freeze_receipt.json", {
        "frozen_utc": now(), "protocol_hash": digest(planned), "input_manifest_hash": digest(inputs),
        "execution_code": code, "duof_outer_outcomes_seen": False,
        "prepared_component_bundles_present": len(expected_bundles),
    })
    write_json(ROOT / "logs/environment.json", {"python": sys.version, "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__, "platform": platform.platform(),
        "CPU_threads": 2, "logical_cpus": psutil.cpu_count(),
        "physical_memory_GiB": psutil.virtual_memory().total / 2**30})
    write_json(ROOT / "run_state.json", {"status": "PREPARED", "started_utc": now(), "completed_folds": []})
    print("FROZEN: 20 folds, 2 CERT ceilings, 320 component bundles present; no new outcomes", flush=True)


def verify_frozen():
    receipt = read_json(ROOT / "config/freeze_receipt.json")
    assert receipt["protocol_hash"] == digest(read_json(ROOT / "config/protocol.json"))
    assert receipt["input_manifest_hash"] == digest(read_json(ROOT / "config/input_manifest.json"))
    for p, expected in receipt["execution_code"].items():
        assert sha256(ROOT / p) == expected, f"Execution code changed: {p}"
    return receipt


def input_check(path):
    expected = read_json(ROOT / "config/input_manifest.json")[str(path)]["sha256"]
    assert sha256(path) == expected, path


def fit_fold(outer):
    freeze = verify_frozen()
    out = ROOT / "folds" / f"outer-{outer}"
    checkpoint = out / "completion.json"
    contract = digest({"protocol": freeze["protocol_hash"], "code": freeze["execution_code"],
                       "outer": outer, "inputs": freeze["input_manifest_hash"]})
    if checkpoint.exists():
        saved = read_json(checkpoint)
        assert saved["contract"] == contract
        assert all(sha256(out / name) == value for name, value in saved["outputs"].items())
        print(f"RESUME_SKIP outer-{outer}: outputs verified", flush=True)
        return saved
    timer = time.perf_counter()
    path = S2 / "development/risk" / f"outer-{outer}" / "development.joblib"
    input_check(path)
    read_start = time.perf_counter()
    dev = joblib.load(path)
    read_seconds = time.perf_counter() - read_start
    validate_dev(dev)
    fit_start = time.perf_counter()
    fitted = DUO.fit_duo(as_duo(dev["fit_data"]), dev["fit_data"]["labels"], "F")
    fit_seconds = time.perf_counter() - fit_start
    cy = STAGE3.labels_for(outer, "cert", dev["cert_data_blind"]["identities"])
    score_start = time.perf_counter()
    scores, grid, rows, selections, masks = compute_decisions(dev, fitted, cy)
    score_seconds = time.perf_counter() - score_start
    write_json(out / "weights.json", {**fitted, "outer": outer, "fit_subjects": dev["fit_subjects"],
                                     "fit_rows": len(dev["fit_data"]["labels"]), "fit_seconds": fit_seconds})
    write_json(out / "policies.json", {"outer": outer, "candidate_rows": rows, "selections": selections,
                                     "method_id": METHOD_ID, "cert_subjects": dev["cert_subjects"],
                                     "cert_labels_sha256": hashlib.sha256(cy.tobytes()).hexdigest(),
                                     "outer_labels_used": False})
    arrays = {"identities_json": np.array(json.dumps(dev["outer_data_blind"]["identities"])),
              "cert_identities_json": np.array(json.dumps(dev["cert_data_blind"]["identities"])),
              "primary_predictions": dev["outer_data_blind"]["primary"].argmax(1),
              "outer_scores": scores["outer"], "cert_scores": scores["cert"], "fit_scores": scores["fit"],
              "fit_subjects": dev["fit_data"]["subjects"], "cert_labels": cy,
              "contains_outer_labels": np.array(False),
              **{f"accept_{method}": mask for method, mask in masks.items()}}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "decisions.npz", **arrays)
    result = {"status": "COMPLETE", "contract": contract, "outer": outer, "completed_utc": now(),
              "seconds": time.perf_counter() - timer, "fit_seconds": fit_seconds,
              "load_seconds": read_seconds, "score_select_seconds": score_seconds,
              "RSS_MiB_at_fold_end": psutil.Process().memory_info().rss / 2**20,
              "outputs": {name: sha256(out / name) for name in ["weights.json", "policies.json", "decisions.npz"]},
              "outer_labels_used": False}
    write_json(checkpoint, result)
    print(f"COMPLETE outer-{outer}: fit {fit_seconds:.3f}s, total {result['seconds']:.2f}s, iterations {fitted['iterations']}; outer labels unread", flush=True)
    return result


def fit_all(fold):
    state = read_json(ROOT / "run_state.json")
    timer = time.perf_counter()
    for outer in PEOPLE if fold == "all" else [fold]:
        fit_fold(outer)
        complete = [s for s in PEOPLE if (ROOT / "folds" / f"outer-{s}" / "completion.json").exists()]
        state.update(status="DECISIONS_IN_PROGRESS", completed_folds=complete, updated_utc=now())
        write_json(ROOT / "run_state.json", state)
    if len(state["completed_folds"]) == 20:
        receipt = ROOT / "config/all_decisions_locked.json"
        if not receipt.exists():
            frozen = {}
            for outer in PEOPLE:
                d = ROOT / "folds" / f"outer-{outer}"
                completion = read_json(d / "completion.json")
                for name, expected in completion["outputs"].items():
                    assert sha256(d / name) == expected
                    frozen[str((d / name).relative_to(ROOT))] = expected
            write_json(receipt, {"locked_utc": now(), "folds": 20, "methods": 2,
                                 "policy_decisions": 40, "outer_outcomes_read_by_evaluation_entry": False,
                                 "files": frozen, "protocol_hash": verify_frozen()["protocol_hash"]})
        state.update(status="ALL_DECISIONS_LOCKED", updated_utc=now())
        write_json(ROOT / "run_state.json", state)
    print(f"Invocation finished in {time.perf_counter()-timer:.2f}s; complete folds {len(state['completed_folds'])}/20", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "fit"])
    parser.add_argument("--fold", choices=PEOPLE + ["all"], default="all")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.phase == "prepare":
            prepare()
        else:
            fit_all(args.fold)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_json(ROOT / "logs/latest_failure.json", {"utc": now(), "traceback": traceback.format_exc()})
        raise
