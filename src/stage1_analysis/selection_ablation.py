"""Fixed-candidate selection ablation; fitting and outcome replay are separate phases."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

from .acceptance import (build_acceptance_mask, select_empirical_policy,
                         subject_equal_risk, subject_equal_weights)
from .inputs import PH, load, read_json, sha256
from .metrics import (alignment, bootstrap_metrics, count_subjects, identity_key,
                      identity_ties, interval, quota_mask, ranked_orders, summary)

VARIANTS = ["COVER", "COVER-NLL", "COVER-AUROC"]
FAMILIES = ["Primary-only", "Heterogeneous-expert"]
EXPERIMENT = "outputs/strengthening/cover_selection_ablation"
SPEC = "configs/analysis/cover_selection_ablation.json"
CORE_FILES = [SPEC, "src/stage1_analysis/selection_ablation.py",
              "src/stage1_analysis/risk_fit.py", "src/stage1_analysis/risk_features.py",
              "src/stage1_analysis/acceptance.py", "Scripts/run_selection_ablation.py"]


def safe(value):
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(safe(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def exact(value) -> Fraction:
    if isinstance(value, dict):
        return Fraction(value["numerator"], value["denominator"])
    return value if isinstance(value, Fraction) else Fraction(str(value))


def nll_auc(scores: np.ndarray, errors: np.ndarray, subjects: np.ndarray) -> tuple[float, float]:
    """No CERT/OUTER arguments: both criteria use calibrated FIT CV scores only."""
    scores, errors = np.asarray(scores), np.asarray(errors)
    if scores.shape != errors.shape or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Invalid FIT scores")
    if (scores < 0).any() or (scores > 1).any() or set(np.unique(errors)) != {0, 1}:
        raise ValueError("Expected probabilities and both binary outcomes")
    weights = subject_equal_weights(subjects)
    p = np.clip(scores, 1e-6, 1 - 1e-6)
    nll = np.average(-errors * np.log(p) - (1 - errors) * np.log1p(-p), weights=weights)
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    positive = np.add.reduceat(weights[order] * errors[order], starts)
    negative = np.add.reduceat(weights[order] * (1 - errors[order]), starts)
    negative_below = np.cumsum(negative) - negative
    auc = np.sum(positive * (negative_below + .5 * negative)) / (positive.sum() * negative.sum())
    return float(nll), float(auc)


def model_priority(config: dict) -> tuple:
    if config["family"] == "logistic":
        return (0, float(config["C"]), config["id"])
    return (1, config["id"])


def selection_rank(row: dict, variant: str) -> tuple:
    base = (-exact(row["coverage"]), Fraction(2) if row["risk"] is None else exact(row["risk"]),
            FAMILIES.index(row["family"]), model_priority(row["config"]))
    if variant == "COVER":
        return base
    if variant == "COVER-NLL":
        return (row["nll"], *base)
    if variant == "COVER-AUROC":
        return (-row["auroc"], *base)
    raise ValueError(variant)


def select_fit_candidates(learned: dict, fit_data: dict) -> tuple[dict, list[dict]]:
    """Only FIT objects are accepted; the learned dict contains FIT candidates only."""
    rows = []
    for family in FAMILIES:
        for candidate in learned[family]["candidates"]:
            nll, auc = nll_auc(candidate["calibrated_oof"], fit_data["errors"], fit_data["subjects"])
            rows.append({"family": family, "config": candidate["config"],
                         "coverage": candidate["coverage"], "risk": candidate["risk"],
                         "q": candidate["q"], "nll": nll, "auroc": auc})
    selected = {name: min(rows, key=lambda r: selection_rank(r, name)) for name in VARIANTS}
    return selected, rows


def apply_rule(scores: np.ndarray, rule: dict | None, data: dict, method_id: str, outer: str):
    if rule is None or rule.get("zero_fallback", False):
        return np.zeros(len(scores), dtype=bool)
    return build_acceptance_mask(scores, threshold=float(rule["threshold"]),
        tie_fraction=exact(rule["tie_fraction"]), identities=data["identities"],
        method_id=method_id, protocol_hash=PH, outer=outer)


def select_cert_rule(policies: list[dict], scores: np.ndarray, errors: np.ndarray,
                     data: dict, method_id: str, outer: str) -> tuple[dict | None, list[dict]]:
    if len(set(data["subjects"])) != 4 or outer in set(data["subjects"]):
        raise ValueError("CERT must have four subjects excluding OUTER")
    rows = []
    for policy in policies:
        accept = apply_rule(scores, policy, data, method_id, outer)
        metric = subject_equal_risk(accept, errors, data["subjects"])
        minimum = min(r["accepted"] for r in metric["per_subject"].values())
        rows.append({"q": exact(policy["q"]), "coverage": exact(metric["coverage"]),
            "risk": None if metric["risk"] is None else exact(metric["risk"]),
            "supported": minimum >= 50, "minimum_accepted": minimum,
            "threshold": policy["threshold"], "tie_fraction": exact(policy["tie_fraction"]),
            "metric": metric})
    return select_empirical_policy(rows, alpha=.05), rows


def code_receipt(root: Path) -> dict:
    files = {name: sha256(root / name) for name in CORE_FILES}
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def load_development(root: Path, outer: str) -> tuple[dict, dict]:
    import joblib

    base = root / "models/stage1/development/risk" / f"outer-{outer}"
    receipt = read_json(base / "development_manifest.json")
    path = base / "development.joblib"
    digest = sha256(path)
    if digest != receipt["artifact_sha256"] or receipt["protocol_hash"] != PH:
        raise RuntimeError(f"Development fingerprint mismatch: {outer}")
    data = joblib.load(path)  # Trusted local, fingerprint-checked historical artifact.
    if data["outer"] != outer or data["protocol_hash"] != PH:
        raise RuntimeError("Wrong development identity")
    fit_people, cert_people = set(data["fit_subjects"]), set(data["cert_subjects"])
    assert len(fit_people) == 15 and len(cert_people) == 4
    assert not fit_people & cert_people and outer not in fit_people | cert_people
    assert fit_people | cert_people | {outer} == {f"{i:02d}" for i in range(20)}
    for partition, people in [("fit_data", fit_people), ("cert_data_blind", cert_people),
                              ("outer_data_blind", {outer})]:
        value = data[partition]
        keys = [identity_key(i) for i in value["identities"]]
        assert len(keys) == len(set(keys)) == len(value["subjects"])
        assert set(value["subjects"]) == people
        assert all(k[1] == s for k, s in zip(keys, value["subjects"]))
        if partition != "fit_data":
            assert not {"labels", "errors"} & value.keys()
    return data, {"path": path.relative_to(root).as_posix(), "sha256": digest,
                  "manifest_sha256": sha256(base / "development_manifest.json")}


def read_cert_labels(root: Path, cert_data: dict, allowed: list[str], outer: str) -> tuple[np.ndarray, dict]:
    """Read y only for the current CERT roles, after all FIT decisions are sealed."""
    if outer in allowed or set(cert_data["subjects"]) != set(allowed):
        raise ValueError("Invalid CERT authorization")
    expected = {r["path"]: r["sha256"] for r in read_json(root / "manifests/asset_manifest.json")["files"]}
    record_labels, fingerprints = {}, {}
    for identity in cert_data["identities"]:
        if identity["subject_id"] not in allowed:
            raise ValueError("Non-CERT identity passed to label reader")
        record = identity["record_id"]
        if record not in record_labels:
            relative = f"data/eeg_fpz_cz/{record}.npz"
            path = root / relative
            digest = sha256(path)
            if expected.get(relative) != digest:
                raise RuntimeError(f"CERT label source fingerprint mismatch: {relative}")
            fingerprints[relative] = digest
            with np.load(path, allow_pickle=False) as z:
                record_labels[record] = z["y"].copy()
    labels = np.asarray([record_labels[i["record_id"]][int(i["epoch_index"])]
                         for i in cert_data["identities"]], dtype=np.int64)
    return labels, fingerprints


def check_resume(path: Path, expected: str) -> dict:
    receipt = read_json(path)
    if receipt.get("run_fingerprint") != expected:
        raise RuntimeError("Resume code/config/input identity mismatch; do not silently reuse")
    return receipt


def fit(root: Path, *, resume: bool, pilot: bool) -> None:
    import joblib
    import sklearn
    from threadpoolctl import threadpool_limits
    from . import risk_fit

    started = time.perf_counter()
    directory = root / EXPERIMENT
    directory.mkdir(parents=True, exist_ok=True)
    spec = read_json(root / SPEC)
    code = code_receipt(root)
    fingerprint = code["sha256"]
    selection_path = directory / "fit_selections.json"
    if selection_path.exists():
        if not resume:
            raise FileExistsError("Use --resume to preserve existing experiment")
        selections = check_resume(selection_path, fingerprint)
        if selections["status"] != "ALL_FIT_SELECTIONS_FROZEN":
            raise RuntimeError("Incomplete FIT selection phase")
    else:
        selections = {"status": "ALL_FIT_SELECTIONS_FROZEN", "created_utc": utc(),
            "run_fingerprint": fingerprint, "code": code, "spec": spec,
            "environment": {"numpy": np.__version__, "sklearn": sklearn.__version__},
            "selection_inputs": "FIT only; cached component OOF ancestry retained", "folds": {}}
        for i in range(20):
            outer = f"{i:02d}"
            data, source = load_development(root, outer)
            for family in FAMILIES:
                candidates = data["learned"][family]["candidates"]
                assert [x["config"]["id"] for x in candidates] == spec["config_ids"]
                for candidate in candidates:
                    for key in ["raw_oof", "calibrated_oof"]:
                        v = candidate[key]
                        assert v.shape == data["fit_data"]["errors"].shape
                        assert np.isfinite(v).all() and ((0 <= v) & (v <= 1)).all()
                    assert len(candidate["policies"]) == 20
            selected, rows = select_fit_candidates(data["learned"], data["fit_data"])
            cover = selected["COVER"]
            assert cover["family"] == data["learned_selected_family"]
            assert cover["config"] == data["learned"][cover["family"]]["selected"]["config"]
            selections["folds"][outer] = {"source": source, "fit_subjects": data["fit_subjects"],
                "cert_subjects": data["cert_subjects"], "selected": selected, "candidates": rows}
            print(f"FIT selection frozen: outer-{outer}", flush=True)
        save_json(selection_path, selections)
        selections = read_json(selection_path)
    if len(selections["folds"]) != 20:
        raise RuntimeError("All FIT decisions must precede CERT/OUTER execution")

    # The fixed outer-00 saved LR/HGB models provide one compatibility check each.
    compatibility_path = directory / "compatibility.json"
    with threadpool_limits(limits=2):
        if compatibility_path.exists():
            check_resume(compatibility_path, fingerprint)
        else:
            data, _ = load_development(root, "00")
            checks = []
            for family in FAMILIES:
                candidate = data["learned"][family]["selected"]
                x = risk_fit.feature_matrix(data["fit_data"], family)
                y, s = data["fit_data"]["errors"], data["fit_data"]["subjects"]
                t = time.perf_counter()
                fresh = {"base": risk_fit.fit_model(candidate["config"], x, y, s),
                         "platt": risk_fit.fit_platt(candidate["raw_oof"], y, s)}
                distances = {}
                for part in ["cert", "outer"]:
                    actual = risk_fit.predict(fresh, data[f"{part}_data_blind"], family)
                    expected = data["learned_scores"][family][part]
                    distances[part] = float(np.max(np.abs(actual - expected)))
                    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
                    original = read_json(root / "outputs/frozen/stage1/calibration/outer-00/selector_calibration.json")
                    rule = original["selectors"][family]["alphas"]["0.05"]["empirical"]
                    full_id = data["method_full_ids"][family]
                    blind = data[f"{part}_data_blind"]
                    assert np.array_equal(apply_rule(actual, rule, blind, full_id, "00"),
                                          apply_rule(expected, rule, blind, full_id, "00"))
                checks.append({"family": family, "config": candidate["config"],
                    "max_absolute_difference": distances, "seconds": time.perf_counter() - t})
            save_json(compatibility_path, {"run_fingerprint": fingerprint, "status": "PASS",
                "checks": checks, "fits": 2, "atol": 1e-8, "rtol": 1e-8,
                "original_cert_outer_decisions_bit_exact": True})

        for i in range(1 if pilot else 20):
            outer = f"{i:02d}"
            receipt_path = directory / f"outer-{outer}.json"
            arrays_path = directory / f"outer-{outer}.npz"
            if receipt_path.exists():
                receipt = check_resume(receipt_path, fingerprint)
                assert receipt["arrays_sha256"] == sha256(arrays_path)
                assert all(sha256(root / p) == h for p, h in receipt["model_files"].items())
                assert receipt["source_sha256"] == sha256(root / selections["folds"][outer]["source"]["path"])
                print(f"Resume verified: outer-{outer}", flush=True)
                continue
            t = time.perf_counter()
            data, source = load_development(root, outer)
            assert source == selections["folds"][outer]["source"]
            unique = {}
            model_files = {}
            for variant in VARIANTS:
                picked = selections["folds"][outer]["selected"][variant]
                family, config = picked["family"], picked["config"]
                full_id = family + "|" + config["id"]
                if full_id in unique:
                    continue
                candidate = next(c for c in data["learned"][family]["candidates"] if c["config"] == config)
                saved = data["final_models"][family]
                if saved["config"] == config:
                    scores = {}
                    for part in ["cert", "outer"]:
                        computed = risk_fit.predict(saved, data[f"{part}_data_blind"], family)
                        expected = data["learned_scores"][family][part]
                        np.testing.assert_allclose(computed, expected, rtol=1e-12, atol=1e-12)
                        # Preserve historical prediction bytes after compatibility validation.
                        scores[part] = expected.copy()
                    new_fit = False
                else:
                    model_path = directory / f"outer-{outer}__{family}__{config['id']}.joblib"
                    model_receipt = model_path.with_suffix(".json")
                    if model_receipt.exists():
                        checkpoint = check_resume(model_receipt, fingerprint)
                        assert checkpoint["sha256"] == sha256(model_path)
                        assert checkpoint["source_sha256"] == source["sha256"]
                        model = joblib.load(model_path)
                    else:
                        x = risk_fit.feature_matrix(data["fit_data"], family)
                        y, s = data["fit_data"]["errors"], data["fit_data"]["subjects"]
                        model = {"config": config, "family": family,
                            "base": risk_fit.fit_model(config, x, y, s),
                            "platt": risk_fit.fit_platt(candidate["raw_oof"], y, s)}
                        temporary = model_path.with_suffix(".partial")
                        joblib.dump(model, temporary, compress=3)
                        os.replace(temporary, model_path)
                        save_json(model_receipt, {"run_fingerprint": fingerprint,
                            "source_sha256": source["sha256"], "sha256": sha256(model_path),
                            "family": family, "config": config})
                    model_files[model_path.relative_to(root).as_posix()] = sha256(model_path)
                    scores = {part: risk_fit.predict(model, data[f"{part}_data_blind"], family)
                              for part in ["cert", "outer"]}
                    new_fit = True
                unique[full_id] = {"scores": scores, "policies": candidate["policies"],
                                   "new_fit": new_fit}
            cert_labels, label_sources = read_cert_labels(root, data["cert_data_blind"], data["cert_subjects"], outer)
            cert_errors = (data["cert_data_blind"]["primary"].argmax(axis=1) != cert_labels).astype(int)
            rules = {}
            for full_id, result in unique.items():
                rule, rows = select_cert_rule(result["policies"], result["scores"]["cert"],
                    cert_errors, data["cert_data_blind"], full_id, outer)
                result["mask"] = apply_rule(result["scores"]["outer"], rule,
                    data["outer_data_blind"], full_id, outer)
                rules[full_id] = {"choice": rule, "cert_candidates": rows, "new_fit": result["new_fit"]}
            picked_cover = selections["folds"][outer]["selected"]["COVER"]
            cover_id = picked_cover["family"] + "|" + picked_cover["config"]["id"]
            assert cover_id == data["method_full_ids"]["Learned-Selected"]
            original = read_json(root / f"outputs/frozen/stage1/calibration/outer-{outer}/selector_calibration.json")
            original_rule = original["selectors"]["Learned-Selected"]["alphas"]["0.05"]["empirical"]
            actual_rule = rules[cover_id]["choice"]
            if original_rule["zero_fallback"]:
                assert actual_rule is None
            else:
                for key in ["q", "coverage", "risk", "tie_fraction"]:
                    assert exact(original_rule[key]) == actual_rule[key]
                assert original_rule["threshold"] == actual_rule["threshold"]
            with np.load(root / f"outputs/frozen/stage1/outer_predictions/outer-{outer}/outer_decisions.npz", allow_pickle=False) as z:
                mask_index = list(z["mask_names"]).index("Learned-Selected|0.05|empirical")
                assert np.array_equal(unique[cover_id]["mask"], z["acceptance_masks"][:, mask_index])
            ids = [selections["folds"][outer]["selected"][v]["family"] + "|" +
                   selections["folds"][outer]["selected"][v]["config"]["id"] for v in VARIANTS]
            blind = data["outer_data_blind"]
            with arrays_path.with_suffix(".partial").open("wb") as stream:
                np.savez_compressed(stream, identities_json=json.dumps(blind["identities"]),
                    subjects=blind["subjects"], predictions=blind["primary"].argmax(axis=1),
                    scores=np.column_stack([unique[k]["scores"]["outer"] for k in ids]),
                    masks=np.column_stack([unique[k]["mask"] for k in ids]),
                    variants=np.asarray(VARIANTS), method_full_ids=np.asarray(ids))
            os.replace(arrays_path.with_suffix(".partial"), arrays_path)
            save_json(receipt_path, {"status": "OUTER_DECISIONS_FROZEN", "outer": outer,
                "run_fingerprint": fingerprint, "source_sha256": source["sha256"],
                "contains_outer_labels": False, "completed_utc": utc(),
                "selection_file_sha256": sha256(selection_path), "cert_label_sources": label_sources,
                "rules": rules, "model_files": model_files,
                "arrays_sha256": sha256(arrays_path), "seconds": time.perf_counter() - t,
                "cover_control_bit_exact": True, "same_config_shared_outputs": True})
            print(f"Completed outer-{outer}: {time.perf_counter() - t:.1f}s; "
                  f"{sum(r['new_fit'] for r in unique.values())} new final fits", flush=True)
    if pilot:
        print("Pilot complete; all 20 FIT selections are already frozen.", flush=True)
        return
    build_portable(root, directory, selections, fingerprint)
    receipts = [read_json(directory / f"outer-{i:02d}.json") for i in range(20)]
    new_fits = sum(r["new_fit"] for item in receipts for r in item["rules"].values())
    assert new_fits <= 40
    save_json(directory / "completion.json", {"status": "COMPLETE_20_FOLDS",
        "run_fingerprint": fingerprint, "completed_utc": utc(), "folds": 20,
        "new_final_risk_fits": new_fits, "compatibility_fits": 2,
        "component_fits": 0, "invocation_seconds": time.perf_counter() - started,
        "total_fold_seconds": sum(r["seconds"] for r in receipts),
        "compatibility_seconds": sum(x["seconds"] for x in read_json(compatibility_path)["checks"])})


def build_portable(root: Path, directory: Path, selections: dict, fingerprint: str) -> None:
    frozen = load(root, compact=True)  # OUTER truth is used only after all decisions exist.
    scores = np.full((len(frozen.errors), 3), np.nan)
    masks = np.zeros(scores.shape, dtype=bool)
    visited = np.zeros(len(frozen.errors), dtype=bool)
    sources = {}
    for i in range(20):
        outer = f"{i:02d}"
        receipt = check_resume(directory / f"outer-{outer}.json", fingerprint)
        path = directory / f"outer-{outer}.npz"
        assert receipt["arrays_sha256"] == sha256(path)
        sources[path.relative_to(root).as_posix()] = sha256(path)
        with np.load(path, allow_pickle=False) as z:
            positions = alignment([identity_key(x) for x in json.loads(z["identities_json"].item())], frozen.keys)
            assert not visited[positions].any() and (frozen.subjects[positions] == outer).all()
            assert np.array_equal(z["predictions"], frozen.predictions[positions])
            scores[positions], masks[positions] = z["scores"], z["masks"]
            visited[positions] = True
    assert visited.all() and np.isfinite(scores).all()
    assert np.array_equal(scores[:, 0], frozen.scores["Learned-Selected"])
    assert np.array_equal(masks[:, 0], frozen.masks["Learned-Selected|0.05|empirical"])
    path = root / "outputs/analysis_inputs/selection_ablation_inputs.npz"
    with path.with_suffix(".partial").open("wb") as stream:
        np.savez_compressed(stream, identities_json=json.dumps(frozen.identities),
            scores=scores, masks=masks, variants=np.asarray(VARIANTS))
    os.replace(path.with_suffix(".partial"), path)
    save_json(path.with_suffix(".json"), {"schema": "cover-selection-ablation-portable-v1",
        "sha256": sha256(path), "run_fingerprint": fingerprint, "source_outputs": sources,
        "fit_selection_sha256": sha256(directory / "fit_selections.json"),
        "base_input_sha256": sha256(root / "outputs/analysis_inputs/frozen_analysis_inputs.npz"),
        "fit_selections": selections, "outer_labels_used_for_selection": False,
        "all_20_decisions_frozen_before_outcome_replay": True})


def replay(root: Path, out: Path) -> dict:
    """No fitting imports; regenerate all ablation tables from portable predictions."""
    from .analysis import write_csv

    frozen = load(root, compact=True)
    path = root / "outputs/analysis_inputs/selection_ablation_inputs.npz"
    receipt = read_json(path.with_suffix(".json"))
    assert sha256(path) == receipt["sha256"]
    assert sha256(root / "outputs/analysis_inputs/frozen_analysis_inputs.npz") == receipt["base_input_sha256"]
    folds = receipt["fit_selections"]["folds"]
    if set(folds) != {f"{i:02d}" for i in range(20)}:
        raise ValueError("Incomplete folds cannot enter 20-fold summary")
    with np.load(path, allow_pickle=False) as z:
        assert list(z["variants"]) == VARIANTS
        positions = alignment(frozen.keys, [identity_key(x) for x in json.loads(z["identities_json"].item())])
        scores, masks = z["scores"][positions], z["masks"][positions]
    assert np.array_equal(scores[:, 0], frozen.scores["Learned-Selected"])
    reference = frozen.masks["Learned-Selected|0.05|empirical"]
    assert np.array_equal(masks[:, 0], reference)
    people, errors, subjects = frozen.people, frozen.errors, frozen.subjects
    _, quotas, _ = count_subjects(reference, errors, subjects, people)
    ties = identity_ties(frozen.keys)
    matched = np.column_stack([quota_mask(ranked_orders(scores[:, i], ties, subjects, people),
                        quotas, len(errors)) for i in range(3)])
    assert int((matched[:, 0] & errors).sum()) == 817
    totals, per_subject, counts = [], [], {}
    for context, matrix in [("cert05", masks), ("matched_common_ties", matched)]:
        for i, name in enumerate(VARIANTS):
            n, a, k = count_subjects(matrix[:, i], errors, subjects, people)
            counts[context, name] = (n, a, k)
            totals.append({"context": context, "method": name, **summary(n, a, k)})
            for s, nn, aa, kk in zip(people, n, a, k):
                per_subject.append({"context": context, "method": name, "subject": s,
                    "eligible": int(nn), "accepted": int(aa), "errors": int(kk),
                    "coverage": float(aa / nn), "risk": float(kk / aa) if aa else None,
                    "passed": bool(aa >= 50 and 20 * kk <= aa)})
    rng = np.random.Generator(np.random.PCG64(20260901))
    resamples = rng.integers(0, 20, size=(10000, 20))
    paired = []
    for context in ["cert05", "matched_common_ties"]:
        a = bootstrap_metrics(*counts[context, "COVER"], resamples)
        point_a = summary(*counts[context, "COVER"])
        for name in VARIANTS[1:]:
            b = bootstrap_metrics(*counts[context, name], resamples)
            point_b = summary(*counts[context, name])
            for metric in a:
                paired.append({"context": context, "method_a": "COVER", "method_b": name,
                    "metric": metric, "a_minus_b": point_a[metric] - point_b[metric],
                    **interval(a[metric] - b[metric])})
    selected_rows, candidate_rows = [], []
    for outer, fold in folds.items():
        cover = fold["selected"]["COVER"]
        for name, row in fold["selected"].items():
            selected_rows.append({"outer": outer, "method": name, "family": row["family"],
                "config": row["config"]["id"], "fit_feasible_coverage": float(exact(row["coverage"])),
                "fit_risk": float(exact(row["risk"])) if row["risk"] else None,
                "fit_nll": row["nll"], "fit_auroc": row["auroc"],
                "same_as_cover": row["family"] == cover["family"] and row["config"] == cover["config"]})
        for row in fold["candidates"]:
            candidate_rows.append({"outer": outer, "family": row["family"],
                "config": row["config"]["id"], "fit_feasible_coverage": float(exact(row["coverage"])),
                "fit_risk": float(exact(row["risk"])) if row["risk"] is not None else None,
                "fit_nll": row["nll"], "fit_auroc": row["auroc"]})
        for a in range(3):
            for b in range(a + 1, 3):
                ra, rb = fold["selected"][VARIANTS[a]], fold["selected"][VARIANTS[b]]
                if ra["family"] == rb["family"] and ra["config"] == rb["config"]:
                    at = subjects == outer
                    assert np.array_equal(scores[at, a], scores[at, b])
                    assert np.array_equal(masks[at, a], masks[at, b])
    for name, rows in [("summary", totals), ("per_subject", per_subject),
                       ("paired_bootstrap", paired), ("selected_configs", selected_rows),
                       ("fit_candidates", candidate_rows)]:
        write_csv(out / "tables" / f"ablation_{name}.csv", rows)
    result = {"status": "PASS", "folds": 20, "epochs": len(errors),
        "cover_scores_masks_bit_exact": True, "same_config_identical_outputs": True,
        "config_agreement": {name: sum(r["same_as_cover"] for r in selected_rows if r["method"] == name)
                             for name in VARIANTS},
        "summary": totals, "bootstrap": "paired subjects; 10000 PCG64 draws; seed 20260901; descriptive",
        "portable_input_sha256": sha256(path), "run_fingerprint": receipt["run_fingerprint"]}
    save_json(out / "metrics/ablation_checks.json", result)
    print(json.dumps(result, allow_nan=False))
    return result
