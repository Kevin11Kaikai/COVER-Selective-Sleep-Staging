"""Frozen three-route FIT study. All fitting APIs receive training-only records."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from .storage import ROOT, OUT, PROJECT, read, write, digest, arrays, model_save, utc, safe
from .experiment import load_fit, require_source, features, raw_predict
from .losses import weights
from .solvers import fit_bce, fit_sele
from .calibration import fit_mapping, apply_mapping
from .policies import choose, mask, subset_policies, exact
from automl.grids import dense_candidate_policies, grid_fractions
from .bceanchor_upgrade_solver import fit_refinement
from .bceanchor_upgrade_core import (
    UOUT, spec, fit_identity, object_hash, array_hash, partition_subjects,
    select_candidate, average_models, equal_norm_control,
)

AVERAGE_CELLS = ["BCEAnchor-average", "FixedScale-average", "SELE-average",
                 "BCEAnchor-equalnorm"]
NESTED_CELLS = ["Nested-BCE", "Nested-FixedScale-1", "Nested-BCEAnchor-1",
                "Nested-BCEAnchor-selected", "Nested-SELE"]


def forbidden(*args, **kwargs):
    raise RuntimeError("FIT-only study: CERT/OUTER evaluation is forbidden")


def install_guards():
    from stage1_analysis import selection_ablation, inputs
    from . import experiment, anchor_experiment, cross_setting, sele_refinement
    selection_ablation.read_cert_labels = forbidden
    inputs.load = forbidden
    experiment.evaluate = anchor_experiment.evaluate = forbidden
    if hasattr(cross_setting, "evaluate"):
        cross_setting.evaluate = forbidden
    sele_refinement.fit = forbidden


def data_only(context):
    development, source = load_fit(context)
    require_source(context, source)
    data = development["fit_data"]
    assert np.array_equal(data["errors"], data["primary"].argmax(1) != data["labels"])
    # The trusted legacy pickle contains blind fields; only FIT leaves this adapter.
    return {"x": features(data), "errors": np.asarray(data["errors"], dtype=int),
            "subjects": np.asarray(data["subjects"], dtype=str),
            "identities": data["identities"]}, development["risk_folds"], source


def subset(data, selection):
    ix = np.flatnonzero(selection)
    return {"x": data["x"][ix], "errors": data["errors"][ix],
            "subjects": data["subjects"][ix],
            "identities": [data["identities"][i] for i in ix]}


def assert_model(model):
    if (model["status"] != "OK" or model.get("theta") is None
            or not np.isfinite(model["theta"]).all()):
        raise RuntimeError("Invalid fitted model: " + str(model.get("status")))


class Store:
    def __init__(self, budget, resume):
        self.budget, self.resume = budget, resume
        self.fp = fit_identity()["sha256"]
        self.history = {}
        self.old = read(OUT / "fit_selections.json")
        self.anchors = read(OUT / "anchored/fit_selections.json")

    def cached(self, path, key, fit, kind="risk"):
        import joblib
        path = Path(path)
        receipt = path.with_suffix(".json")
        key = object_hash(key)
        if receipt.exists():
            old = read(receipt)
            if (not self.resume or old["fingerprint"] != self.fp or old["input_key"] != key
                    or old["sha256"] != digest(path)):
                raise RuntimeError("Checkpoint incompatible: " + str(path))
            return joblib.load(path)
        self.budget.check()
        started = time.monotonic()
        result = fit()
        seconds = time.monotonic()-started
        model_save(path, result)
        write(receipt, {"fingerprint": self.fp, "input_key": key, "kind": kind,
              "seconds": seconds, "sha256": digest(path), "created": utc(),
              "status": result.get("status", "OK"), "pid": __import__('os').getpid()})
        self.budget.check()
        return result

    def legacy(self, context, fold, parameter, source, anchor=False):
        import joblib
        family = "anchored/fits" if anchor else "fits"
        parameter_dir = f"lambda-{parameter:g}" if anchor else f"c-{parameter:g}"
        path = OUT / family / f"outer-{context}" / parameter_dir / f"fold-{fold}.joblib"
        if path not in self.history:
            receipt = read(path.with_suffix(".json"))
            expected_fp = (self.anchors if anchor else self.old)["fingerprint"]
            sha = receipt["sha256"] if anchor else receipt["model_sha256"]
            if (receipt["source"] != source or receipt["fingerprint"] != expected_fp
                    or digest(path) != sha):
                raise RuntimeError("Historical fit provenance mismatch: " + str(path))
            bundle = joblib.load(path)
            for model in bundle["models"].values():
                assert_model(model)
            self.history[path] = bundle
        return self.history[path]

    def calibrate(self, path, raw, y, subjects):
        key = array_hash(raw, y, subjects)
        return self.cached(path, key, lambda: fit_mapping(raw, y, subjects), "platt")

    def cross_calibrate(self, folder, raw, y, subjects, folds):
        calibrated = np.full(len(y), np.nan)
        mappings = []
        for i, fold in enumerate(folds):
            tr, va = np.isin(subjects, fold["train"]), np.isin(subjects, fold["validation"])
            if np.any(tr & va):
                raise ValueError("Calibration train/validation subjects overlap")
            mapping = self.calibrate(folder / f"fold-{i}.joblib", raw[tr], y[tr], subjects[tr])
            calibrated[va] = apply_mapping(mapping, raw[va])
            mappings.append(mapping)
        if not np.isfinite(calibrated).all():
            raise RuntimeError("Incomplete cross-calibrated stream")
        return calibrated, mappings


def protected_files():
    protected = dict(read(OUT / "preflight.json")["critical_hashes"])
    # Only historical research assets, not unrelated whole-repository files.
    for path in ROOT.rglob("*"):
        if (not path.is_file() or path.is_relative_to(UOUT)
                or "__pycache__" in path.parts or path.is_relative_to(OUT / "cache")
                or "bceanchor_upgrade" in path.name.lower()
                or path.name.startswith(("08_", "BCEANCHOR_UPGRADE"))):
            continue
        protected[path.relative_to(PROJECT).as_posix()] = digest(path)
    return protected


def preflight(budget, resume=False, branch="all"):
    path = UOUT / "preflight.json"
    if path.exists():
        require_preflight()
        print("Upgrade preflight reused", flush=True)
        return
    cfg = spec()
    for key in ("bce", "sele", "c_grid", "alpha", "tie_method_id", "protocol_hash"):
        assert cfg[key] == read(ROOT / "protocol.json")[key]
    for key in ("lambda_grid", "temperature", "starts", "attempts", "maxiter", "ftol",
                "risk_tolerance", "minimum_soft_coverage"):
        assert cfg[key] == read(ROOT / "anchor_protocol.json")[key]
    store = Store(budget, resume)
    sources, inputs = {}, {}
    for i in range(20):
        budget.check()
        context = f"{i:02d}"
        data, folds, source = data_only(context)
        assert len(set(data["subjects"])) == 15 and len(folds) == 5
        fold_ids = np.full(len(data["subjects"]), -1, dtype=int)
        for j, fold in enumerate(folds):
            assert len(fold["train"]) == 12 and len(fold["validation"]) == 3
            assert not set(fold["train"]) & set(fold["validation"])
            va = np.isin(data["subjects"], fold["validation"])
            assert np.all(fold_ids[va] == -1)
            fold_ids[va] = j
            for c in cfg["c_grid"]:
                b = store.legacy(context, j, c, source)
                assert set(b["train_subjects"]) == set(fold["train"])
            for lam in cfg["lambda_grid"]:
                b = store.legacy(context, j, lam, source, anchor=True)
                assert set(b["train_subjects"]) == set(fold["train"])
        assert np.all(fold_ids >= 0)
        p = UOUT / "inputs" / f"context-{context}.npz"
        arrays(p, errors=data["errors"], subjects=data["subjects"], fold_ids=fold_ids,
               identities_json=np.asarray(json.dumps(safe(data["identities"]))))
        inputs[context] = {"sha256": digest(p), "source": source, "folds": folds,
                           "training_input_sha256": array_hash(data["x"], data["errors"], data["subjects"])}
        sources[context] = source
        store.history.clear()
    write(path, {"status": "PASS", "created": utc(), "identity": fit_identity(),
          "spec": cfg, "sources": sources, "inputs": inputs, "protected": protected_files(),
          "no_component_training": True, "cert_outer_evaluation": "NOT_EXECUTED"})
    print("PREFLIGHT PASS: 20 FIT contexts; historical models verified", flush=True)


def require_preflight():
    p = read(UOUT / "preflight.json")
    if p["identity"]["sha256"] != fit_identity()["sha256"]:
        raise RuntimeError("Training/selection code or protocol changed: resume refused")
    return p


def rule_statistics(scores, data, context, cfg, counts=False):
    grid = dense_candidate_policies(scores, data["subjects"], grid_fractions(20),
                                    n_subjects=len(set(data["subjects"])))
    picked, rows = choose(grid, scores, data["errors"], data, context, cfg)
    if counts:
        for row in rows:
            accepted = mask(scores, row, data, context, cfg)
            row["counts"] = []
            for s in sorted(set(data["subjects"])):
                ix = data["subjects"] == s
                row["counts"].append({"subject": str(s), "n": int(ix.sum()),
                    "accepted": int(accepted[ix].sum()),
                    "errors": int(data["errors"][ix & accepted].sum())})
    return picked, rows


def _verify_selected(row, historical):
    c = 0 if row["picked"] is None else row["picked"]["coverage"]
    r = None if row["picked"] is None else row["picked"]["risk"]
    assert exact(c) == exact(historical["coverage"])
    assert (r is None) == (historical["risk"] is None)
    if r is not None:
        assert exact(r) == exact(historical["risk"])


def selection_context(context, data, folds, source, store):
    folder = UOUT / "selection" / f"context-{context}"
    marker = folder / "selection.json"
    if marker.exists():
        result = read(marker)
        assert result["fingerprint"] == store.fp
        for name, receipt in result["streams"].items():
            assert digest(folder / (name + ".npz")) == receipt["sha256"]
        return result
    cfg = spec()
    streams, candidates = {}, {"BCEAnchor": [], "SELE": []}
    for method in ("BCEAnchor", "SELE", "BCE", "FixedScale"):
        if method in candidates:
            params = cfg["lambda_grid"] if method == "BCEAnchor" else cfg["c_grid"]
        else:
            params = [1.] if method == "FixedScale" else [store.old["folds"][context]["selected"]["BCE"]["C"]]
        history = (store.anchors if method in ("BCEAnchor", "FixedScale") else store.old)
        selected = history["folds"][context]["selected"][method]
        parameter_key = "lambda" if method in ("BCEAnchor", "FixedScale") else "C"
        for parameter in params:
            name = f"{method}-{parameter:g}"
            raw = np.full(len(data["errors"]), np.nan)
            for j, fold in enumerate(folds):
                b = store.legacy(context, j, parameter, source, anchor=parameter_key == "lambda")
                va = np.isin(data["subjects"], fold["validation"])
                raw[va] = raw_predict(b, method, data["x"][va])
            reuse = float(selected[parameter_key]) == float(parameter)
            if reuse:
                old_path = OUT / ("anchored/selected" if parameter_key == "lambda" else "selected") / f"outer-{context}.npz"
                entry = history["folds"][context]
                assert digest(old_path) == entry["array_sha256" if parameter_key == "lambda" else "arrays_sha256"]
                with np.load(old_path, allow_pickle=False) as z:
                    assert np.array_equal(z[method+"_raw"], raw)
                    calibrated = z[method+"_calibrated"].copy()
            else:
                calibrated, _ = store.cross_calibrate(folder / "calibration" / name,
                    raw, data["errors"], data["subjects"], folds)
            picked, rows = rule_statistics(calibrated, data, context, cfg, method in candidates)
            row = {"method": method, "parameter": parameter, "status": "OK", "picked": picked}
            if method in candidates:
                candidates[method].append(row)
            if reuse:
                _verify_selected(row, selected)
            path = folder / (name+".npz")
            arrays(path, raw=raw, calibrated=calibrated)
            write(folder / (name+"_rules.json"), {"rules": rows, "picked": picked})
            streams[name] = {"sha256": digest(path), "reused_calibrated": reuse,
                             "method": method, "parameter": parameter}
    chosen = {}
    for method, old_name, new_name in (("BCEAnchor", "A0", "A1"), ("SELE", "S0", "S1")):
        chosen[old_name] = select_candidate(candidates[method], method, False)
        chosen[new_name] = select_candidate(candidates[method], method, True)
        old = (store.anchors if method == "BCEAnchor" else store.old)["folds"][context]["selected"][method]
        key = "lambda" if method == "BCEAnchor" else "C"
        assert chosen[old_name]["parameter"] == old[key]
        _verify_selected(chosen[old_name], old)
    chosen["A2"] = next(r for r in candidates["BCEAnchor"] if r["parameter"] == 1.)
    result = {"fingerprint": store.fp, "source": source, "candidates": candidates,
              "chosen": chosen, "streams": streams, "status": "COMPLETE"}
    write(marker, result)
    print(f"A {context}: old choices reproduced; exact-q candidates sealed", flush=True)
    return result


def _model_key(x, y, subjects, **params):
    return {"data": array_hash(x, y, subjects), **params}


def averaging_fold(context, fold_number, data, fold, source, selection, store):
    folder = UOUT / "averaging" / f"context-{context}" / f"fold-{fold_number}"
    marker = folder / "completion.json"
    if marker.exists():
        done = read(marker)
        assert done["fingerprint"] == store.fp
        assert digest(folder / "predictions.npz") == done["scores_sha256"]
        return done
    started, cfg = time.monotonic(), spec()
    tr = np.isin(data["subjects"], fold["train"])
    va = np.isin(data["subjects"], fold["validation"])
    train = subset(data, tr)
    c0 = store.old["folds"][context]["selected"]["BCE"]["C"]
    cs = selection["chosen"]["S1"]["parameter"]
    base = store.legacy(context, fold_number, c0, source)
    sele_base = store.legacy(context, fold_number, cs, source)
    full = store.legacy(context, fold_number, 1., source, anchor=True)
    assert np.array_equal(base["mean"], sele_base["mean"])
    assert np.array_equal(base["scale"], sele_base["scale"])
    xx = (train["x"]-base["mean"])/base["scale"]
    xv = (data["x"][va]-base["mean"])/base["scale"]
    scale = full["models"]["BCEAnchor"]["score_scale"]
    raw0 = xx @ base["models"]["BCE"]["theta"]
    w = weights(train["subjects"])
    expected_scale = max(float(np.sqrt(w@((raw0-w@raw0)**2))), 1e-6)
    assert scale == expected_scale
    groups = partition_subjects(train["subjects"], context, fold_number, cfg)
    members = {"BCEAnchor": [], "FixedScale": [], "SELE": []}
    for j, group in enumerate(groups):
        ix = np.isin(train["subjects"], group["train"])
        x, y, subjects = xx[ix], train["errors"][ix], train["subjects"][ix]
        for method in members:
            key = _model_key(x, y, subjects, method=method, scale=scale,
                  parent_base=array_hash(base["models"]["BCE"]["theta"]),
                  sele_base=array_hash(sele_base["models"]["BCE"]["theta"]),
                  parent_n=len(xx), sele_C=cs)
            if method == "SELE":
                fit = lambda: fit_sele(x, y, weights(subjects), 1./(len(xx)*cs),
                                      sele_base["models"]["BCE"]["theta"], cfg, store.budget.check)
            else:
                fit = lambda: fit_refinement(x, y, subjects, base["models"]["BCE"],
                       1., method, cfg, store.budget.check, fixed_scale=scale)
            model = store.cached(folder / "members" / f"{method}-{j}.joblib", key, fit)
            assert_model(model)
            members[method].append(model)
    models = {method+"-average": average_models(m) for method, m in members.items()}
    beta0 = full["models"]["BCEAnchor"]["beta0"]
    bfull = full["models"]["BCEAnchor"]["beta"]
    bmean = np.mean([m["beta"] for m in members["BCEAnchor"]], axis=0)
    control, info = equal_norm_control(beta0, bfull, bmean, cfg["equal_norm_relative_floor"])
    if control is not None:
        models["BCEAnchor-equalnorm"] = {"status": "OK",
            "theta": base["models"]["BCE"]["theta"]+scale*(control-beta0),
            "intercept": base["models"]["BCE"]["intercept"]}
    scores = np.full((int(va.sum()), len(AVERAGE_CELLS)), np.nan)
    for j, name in enumerate(AVERAGE_CELLS):
        if name in models:
            scores[:, j] = xv @ models[name]["theta"] + models[name]["intercept"]
    member_scores = np.column_stack([xv@m["theta"]+m["intercept"]
                                    for m in members["BCEAnchor"]])
    assert np.allclose(scores[:, 0], member_scores.mean(axis=1), atol=1e-12, rtol=1e-12)
    geometry = {"mean_delta_norm": float(np.linalg.norm(bmean-beta0)),
                "full_delta_norm": float(np.linalg.norm(bfull-beta0)),
                "mean_full_distance": float(np.linalg.norm(bmean-bfull)),
                "member_pair_distances": [float(np.linalg.norm(a["beta"]-b["beta"]))
                    for i, a in enumerate(members["BCEAnchor"]) for b in members["BCEAnchor"][i+1:]],
                "member_score_correlation": np.corrcoef(member_scores.T), "equalnorm": info}
    arrays(folder / "predictions.npz", scores=scores, cells=np.asarray(AVERAGE_CELLS),
           indices=np.flatnonzero(va), member_anchor_scores=member_scores)
    model_save(folder / "aggregate.joblib", {"models": models, "mean": base["mean"],
               "scale": base["scale"], "parent_score_scale": scale})
    done = {"fingerprint": store.fp, "status": "COMPLETE", "groups": groups,
            "parent_BCE_C": c0, "SELE_C": cs, "SELE_reg": 1./(len(xx)*cs),
            "parent_epoch_n": len(xx), "geometry": geometry,
            "seconds": time.monotonic()-started,
            "scores_sha256": digest(folder / "predictions.npz")}
    write(marker, done)
    print(f"B2 {context}/{fold_number}: four members x three methods fixed", flush=True)
    return done


def nested_pipeline(train, context, fold_number, source, store):
    """No heldout records/labels accepted. Produces a fully locked risk-layer pipeline."""
    cfg = spec()
    folder = UOUT / "nested" / f"context-{context}" / f"fold-{fold_number}"
    key = _model_key(train["x"], train["errors"], train["subjects"],
                     identities=train["identities"], context=context, fold=fold_number)

    def build():
        groups = partition_subjects(train["subjects"], context, fold_number, cfg)
        n = len(train["errors"])
        raw, calibrated, rows, rules, inner = {}, {}, {}, {}, {}
        for method in ("BCE", "SELE"):
            for c in cfg["c_grid"]:
                raw[f"{method}-{c:g}"] = np.full(n, np.nan)
        for j, group in enumerate(groups):
            tr, va = np.isin(train["subjects"], group["train"]), np.isin(train["subjects"], group["validation"])
            scaler = StandardScaler().fit(train["x"][tr])
            xx, xv = scaler.transform(train["x"][tr]), scaler.transform(train["x"][va])
            y, subjects = train["errors"][tr], train["subjects"][tr]
            inner[j] = {"tr": tr, "va": va, "x": xx, "xv": xv, "bases": {}}
            for c in cfg["c_grid"]:
                base_key = _model_key(xx, y, subjects, C=c, method="BCE")
                bce = store.cached(folder / "models" / f"BCE-inner{j}-c{c:g}.joblib", base_key,
                    lambda: fit_bce(xx, y, weights(subjects), 1./(len(y)*c), cfg))
                assert_model(bce)
                inner[j]["bases"][c] = bce
                raw[f"BCE-{c:g}"][va] = xv@bce["theta"]+bce["intercept"]
                sele_key = {**base_key, "method": "SELE", "initial": array_hash(bce["theta"])}
                sele = store.cached(folder / "models" / f"SELE-inner{j}-c{c:g}.joblib", sele_key,
                    lambda: fit_sele(xx, y, weights(subjects), 1./(len(y)*c),
                                     bce["theta"], cfg, store.budget.check))
                assert_model(sele)
                raw[f"SELE-{c:g}"][va] = xv@sele["theta"]+sele["intercept"]
        def evaluate_stream(name, parameter):
            values = raw[name]
            if not np.isfinite(values).all():
                raise RuntimeError("Incomplete nested inner raw stream")
            calibrated[name], _ = store.cross_calibrate(folder / "calibration" / name,
                values, train["errors"], train["subjects"], groups)
            picked, rule_rows = rule_statistics(calibrated[name], train, context, cfg)
            rows[name] = {"status": "OK", "parameter": parameter, "picked": picked, "stream": name}
            rules[name] = rule_rows
        for method in ("BCE", "SELE"):
            for c in cfg["c_grid"]:
                evaluate_stream(f"{method}-{c:g}", c)
        selected_bce = select_candidate([rows[f"BCE-{c:g}"] for c in cfg["c_grid"]], "BCE")
        selected_sele = select_candidate([rows[f"SELE-{c:g}"] for c in cfg["c_grid"]], "SELE")
        c0, cs = selected_bce["parameter"], selected_sele["parameter"]
        for method, params in (("BCEAnchor", cfg["lambda_grid"]), ("FixedScale", [1.])):
            for lam in params:
                name = f"{method}-{lam:g}"
                raw[name] = np.full(n, np.nan)
                for j in range(4):
                    item = inner[j]
                    tr, va, xx = item["tr"], item["va"], item["x"]
                    y, subjects = train["errors"][tr], train["subjects"][tr]
                    base = item["bases"][c0]
                    fit_key = _model_key(xx, y, subjects, C=c0, method=method, lam=lam,
                                        base=array_hash(base["theta"]))
                    fitted = store.cached(folder / "models" / f"{method}-inner{j}-lambda{lam:g}.joblib",
                        fit_key, lambda: fit_refinement(xx, y, subjects, base, lam,
                                                       method, cfg, store.budget.check))
                    assert_model(fitted)
                    raw[name][va] = item["xv"]@fitted["theta"]+fitted["intercept"]
                evaluate_stream(name, lam)
        selected_anchor = select_candidate([rows[f"BCEAnchor-{lam:g}"]
                                             for lam in cfg["lambda_grid"]], "BCEAnchor")
        lam = selected_anchor["parameter"]
        assignment = {"Nested-BCE": f"BCE-{c0:g}", "Nested-SELE": f"SELE-{cs:g}",
            "Nested-BCEAnchor-selected": f"BCEAnchor-{lam:g}",
            "Nested-BCEAnchor-1": "BCEAnchor-1", "Nested-FixedScale-1": "FixedScale-1"}
        scaler = StandardScaler().fit(train["x"])
        xx = scaler.transform(train["x"])
        base = store.legacy(context, fold_number, c0, source)
        competitor = store.legacy(context, fold_number, cs, source)
        for bundle in (base, competitor):
            assert set(bundle["train_subjects"]) == set(train["subjects"])
            assert np.array_equal(bundle["mean"], scaler.mean_)
            assert np.array_equal(bundle["scale"], scaler.scale_)
        models = {f"BCE-{c0:g}": base["models"]["BCE"], f"SELE-{cs:g}": competitor["models"]["SELE"]}
        historical_c = store.old["folds"][context]["selected"]["BCE"]["C"]
        reused = [f"BCE-{c0:g}", f"SELE-{cs:g}"]
        for method, ls in (("BCEAnchor", sorted({1., lam})), ("FixedScale", [1.])):
            for strength in ls:
                name = f"{method}-{strength:g}"
                if c0 == historical_c:
                    models[name] = store.legacy(context, fold_number, strength, source,
                                                anchor=True)["models"][method]
                    reused.append(name)
                else:
                    fit_key = _model_key(xx, train["errors"], train["subjects"], C=c0,
                        method=method, lam=strength, base=array_hash(base["models"]["BCE"]["theta"]))
                    models[name] = store.cached(folder / "models" / f"{name}-parent.joblib", fit_key,
                        lambda: fit_refinement(xx, train["errors"], train["subjects"],
                            base["models"]["BCE"], strength, method, cfg, store.budget.check))
                assert_model(models[name])
        maps = {}
        for name in sorted(set(assignment.values())):
            maps[name] = store.calibrate(folder / "final_calibration" / (name+".joblib"),
                raw[name], train["errors"], train["subjects"])
        names = sorted(raw)
        arrays(folder / "inner_scores.npz", cells=np.asarray(names),
               raw=np.column_stack([raw[n] for n in names]),
               calibrated=np.column_stack([calibrated[n] for n in names]))
        write(folder / "inner_selection.json", {"candidates": rows, "rules": rules,
              "selected_BCE_C": c0, "selected_SELE_C": cs, "selected_lambda": lam,
              "assignment": assignment, "groups": groups,
              "reused_parent_models": reused, "training_key": key})
        return {"status": "OK", "models": models, "maps": maps, "assignment": assignment,
                "locked_rules": {cell: rows[name]["picked"] for cell, name in assignment.items()},
                "mean": scaler.mean_, "scale": scaler.scale_, "groups": groups,
                "selected_BCE_C": c0, "selected_SELE_C": cs, "selected_lambda": lam,
                "training_subjects": sorted(set(train["subjects"])), "training_key": key}
    return store.cached(folder / "pipeline.joblib", key, build, "pipeline")


def predict_nested(pipeline, x, subjects, identities, context, cfg):
    """Frozen inference has no error/true-label argument."""
    xx = (x-pipeline["mean"])/pipeline["scale"]
    raw, calibrated, decisions = [], [], []
    data = {"subjects": subjects, "identities": identities}
    for cell in NESTED_CELLS:
        name = pipeline["assignment"][cell]
        model = pipeline["models"][name]
        r = xx@model["theta"]+model["intercept"]
        s = apply_mapping(pipeline["maps"][name], r)
        raw.append(r)
        calibrated.append(s)
        decisions.append(mask(s, pipeline["locked_rules"][cell], data, context, cfg))
    return np.column_stack(raw), np.column_stack(calibrated), np.column_stack(decisions)


def nested_fold(context, fold_number, data, fold, source, store):
    folder = UOUT / "nested" / f"context-{context}" / f"fold-{fold_number}"
    marker = folder / "completion.json"
    if marker.exists():
        done = read(marker)
        assert done["fingerprint"] == store.fp
        assert digest(folder / "predictions.npz") == done["scores_sha256"]
        return done
    started = time.monotonic()
    tr, va = np.isin(data["subjects"], fold["train"]), np.isin(data["subjects"], fold["validation"])
    train = subset(data, tr)
    pipeline = nested_pipeline(train, context, fold_number, source, store)
    indices = np.flatnonzero(va)
    raw, calibrated, decisions = predict_nested(pipeline, data["x"][va], data["subjects"][va],
        [data["identities"][i] for i in indices], context, spec())
    arrays(folder / "predictions.npz", raw=raw, calibrated=calibrated, decisions=decisions,
           indices=indices, cells=np.asarray(NESTED_CELLS))
    done = {"status": "COMPLETE", "fingerprint": store.fp,
            "seconds": time.monotonic()-started, "validation_subjects": fold["validation"],
            "scores_sha256": digest(folder / "predictions.npz"),
            "pipeline_sha256": digest(folder / "pipeline.joblib"),
            "selected_BCE_C": pipeline["selected_BCE_C"],
            "selected_SELE_C": pipeline["selected_SELE_C"],
            "selected_lambda": pipeline["selected_lambda"],
            "locked_rules": pipeline["locked_rules"]}
    write(marker, done)
    print(f"B1 {context}/{fold_number}: inner choices and rules sealed; heldout scores fixed", flush=True)
    return done


def aggregate_averaging(context, data, folds, store):
    folder = UOUT / "averaging" / f"context-{context}"
    path = folder / "scores.npz"
    if (folder / "scores.json").exists():
        r = read(folder / "scores.json")
        assert r["fingerprint"] == store.fp and digest(path) == r["sha256"]
        return
    scores = np.full((len(data["errors"]), len(AVERAGE_CELLS)), np.nan)
    for j in range(5):
        done = read(folder / f"fold-{j}/completion.json")
        p = folder / f"fold-{j}/predictions.npz"
        assert done["fingerprint"] == store.fp and digest(p) == done["scores_sha256"]
        with np.load(p, allow_pickle=False) as z:
            scores[z["indices"]] = z["scores"]
    calibrated = np.full_like(scores, np.nan)
    undefined = []
    for j, name in enumerate(AVERAGE_CELLS):
        if not np.isfinite(scores[:, j]).all():
            if name != "BCEAnchor-equalnorm":
                raise RuntimeError("Incomplete average scores")
            undefined.append(name)
            continue
        calibrated[:, j], _ = store.cross_calibrate(folder / "calibration" / name,
            scores[:, j], data["errors"], data["subjects"], folds)
    arrays(path, raw=scores, calibrated=calibrated, cells=np.asarray(AVERAGE_CELLS))
    write(folder / "scores.json", {"fingerprint": store.fp, "sha256": digest(path),
                                   "undefined_controls": undefined})


def pilot(budget, resume=False, branch="all"):
    preflight(budget, resume, branch)
    require_preflight()
    store = Store(budget, resume)
    timings = {"selection": [], "averaging": [], "nested": []}
    for context in spec()["pilot_contexts"]:
        data, folds, source = data_only(context)
        before = time.monotonic()
        selected = selection_context(context, data, folds, source, store)
        timings["selection"].append(time.monotonic()-before)
        if branch in ("averaging", "all"):
            r = averaging_fold(context, 0, data, folds[0], source, selected, store)
            timings["averaging"].append(r["seconds"])
        if branch in ("nested", "all"):
            r = nested_fold(context, 0, data, folds[0], source, store)
            timings["nested"].append(r["seconds"])
        store.history.clear()
    estimate = spec()["cost_safety_factor"] * sum(
        max(v)*(20 if k == "selection" else 100) for k, v in timings.items() if v)
    remaining = spec()["incremental_fit_seconds"]-budget.state["fit_seconds"]
    status = "PASS" if estimate < remaining else "BUDGET_REVIEW_NEEDED"
    write(UOUT / "pilot.json", {"status": status, "fingerprint": store.fp, "branch": branch,
          "timings": timings, "estimated_full_seconds_with_margin": estimate,
          "remaining_fit_seconds": remaining, "performance_gate": False, "created": utc()})
    print(f"PILOT {status}: conservative full estimate {estimate/3600:.2f}h", flush=True)
    if status != "PASS":
        raise TimeoutError("Pilot projection exceeds fit grant; matrix not pruned")


def fit(budget, resume=False, branch="all"):
    require_preflight()
    pilot_info = read(UOUT / "pilot.json")
    store = Store(budget, resume)
    if (pilot_info["status"] != "PASS" or pilot_info["fingerprint"] != store.fp
            or (branch != "selection" and pilot_info["branch"] not in ("all", branch))):
        raise RuntimeError("A compatible technical pilot is required")
    choices = {}
    for i in range(20):
        context = f"{i:02d}"
        data, folds, source = data_only(context)
        choices[context] = selection_context(context, data, folds, source, store)
        store.history.clear()
    write(UOUT / "selection_complete.json", {"status": "COMPLETE", "fingerprint": store.fp,
                                              "contexts": list(choices), "created": utc()})
    if branch == "selection":
        return
    for i in range(20):
        context = f"{i:02d}"
        data, folds, source = data_only(context)
        for j, fold in enumerate(folds):
            budget.check()
            if branch in ("all", "averaging"):
                averaging_fold(context, j, data, fold, source, choices[context], store)
            if branch in ("all", "nested"):
                nested_fold(context, j, data, fold, source, store)
        if branch in ("all", "averaging"):
            aggregate_averaging(context, data, folds, store)
        store.history.clear()
        write(UOUT / "progress.json", {"fingerprint": store.fp, "branch": branch,
              "last_complete_context": context, "status": "FITTING", "updated": utc()})
    for b in (["averaging", "nested"] if branch == "all" else [branch]):
        receipts = {}
        for i in range(20):
            for j in range(5):
                p = UOUT / b / f"context-{i:02d}" / f"fold-{j}/completion.json"
                r = read(p)
                assert r["status"] == "COMPLETE" and r["fingerprint"] == store.fp
                receipts[p.relative_to(UOUT).as_posix()] = digest(p)
        write(UOUT / (b+"_complete.json"), {"status": "COMPLETE", "fingerprint": store.fp,
              "parent_folds": 100, "receipts": receipts, "created": utc()})
    write(UOUT / "progress.json", {"fingerprint": store.fp, "branch": branch,
          "last_complete_context": "19", "status": "FIT_COMPLETE", "updated": utc()})
    print("ALL REQUESTED FIT BRANCHES COMPLETE", flush=True)
