"""Replay saved FIT scores/decisions; never fits or loads development bundles."""
from __future__ import annotations
import json
from fractions import Fraction
import numpy as np
import pandas as pd
from .storage import ROOT, PROJECT, read, write, digest, text_file, utc
from .policies import mask, exact
from .losses import weights
from .sele_refinement import quota_mask
from .bceanchor_upgrade_core import UOUT, spec, counts_risk, select_candidate
from .bceanchor_upgrade import AVERAGE_CELLS, NESTED_CELLS, require_preflight

COMPARISONS = {
    "selection": [("A1", "A0"), ("A1", "A2"), ("A1", "S1"), ("A2", "S0"), ("S1", "S0")],
    "averaging": [("BCEAnchor-average", "BCEAnchor-full-1"),
        ("BCEAnchor-average", "BCEAnchor-equalnorm"),
        ("BCEAnchor-average", "SELE-full-S1"), ("BCEAnchor-average", "SELE-average"),
        ("FixedScale-average", "FixedScale-full-1"), ("SELE-average", "SELE-full-S1"),
        ("BCEAnchor-average", "FixedScale-average")],
    "nested": [("Nested-BCEAnchor-selected", "Nested-BCEAnchor-1"),
        ("Nested-BCEAnchor-selected", "Nested-SELE"),
        ("Nested-BCEAnchor-selected", "Nested-FixedScale-1"),
        ("Nested-BCEAnchor-1", "Nested-SELE"), ("Nested-BCE", "Nested-SELE")],
}


def tie_keys(identities, context, cfg):
    from stage1_analysis.acceptance import tie_uniform
    uniforms = [tie_uniform(identity, method_id=cfg["tie_method_id"],
        protocol_hash=cfg["protocol_hash"], outer=context) for identity in identities]
    return np.asarray([format(int(value * 2**256), "064x") for value in uniforms])


def hard_oracle(scores, errors, subjects, alpha):
    """Full tied-score groups, same diagnostic as historical FIT notebooks."""
    w = weights(subjects)
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered) != 0), len(ordered)-1]
    c, e = np.cumsum(w[order]), np.cumsum(w[order]*errors[order])
    feasible = ends[e[ends] <= alpha*c[ends]+1e-12]
    return np.zeros(len(scores), dtype=bool) if not len(feasible) else scores <= ordered[feasible[-1]]


def load_inputs(context, pre):
    path = UOUT / "inputs" / f"context-{context}.npz"
    assert digest(path) == pre["inputs"][context]["sha256"]
    with np.load(path, allow_pickle=False) as z:
        return {"errors": z["errors"].copy(), "subjects": z["subjects"].copy(),
                "fold_ids": z["fold_ids"].copy(),
                "identities": json.loads(str(z["identities_json"]))}


def selected_stream(context, chosen, selection):
    name = f'{chosen["method"]}-{chosen["parameter"]:g}'
    path = UOUT / "selection" / f"context-{context}" / (name+".npz")
    assert digest(path) == selection["streams"][name]["sha256"]
    with np.load(path, allow_pickle=False) as z:
        return z["raw"].copy(), z["calibrated"].copy(), chosen["picked"]


def selection_streams(context, selection):
    result = {name: selected_stream(context, r, selection) for name, r in selection["chosen"].items()}
    for name, entry in selection["streams"].items():
        if entry["method"] not in ("BCE", "FixedScale"):
            continue
        picked = read(UOUT / "selection" / f"context-{context}" / (name+"_rules.json"))["picked"]
        cell = "BCE" if entry["method"] == "BCE" else "FixedScale-1"
        result[cell] = selected_stream(context, {**entry, "picked": picked}, selection)
    return result


def require_complete(branch, pre):
    path = UOUT / (branch+"_complete.json")
    status = read(path)
    if status["status"] != "COMPLETE" or status["fingerprint"] != pre["identity"]["sha256"]:
        raise RuntimeError("Incomplete or incompatible branch: " + branch)
    if branch == "selection":
        assert status["contexts"] == [f"{i:02d}" for i in range(20)]
    else:
        assert status["parent_folds"] == 100 and len(status["receipts"]) == 100
        for rel, sha in status["receipts"].items():
            assert digest(UOUT / rel) == sha


def subject_rows(branch, context, cell, evaluation, accepted, data, space):
    rows = []
    for subject in sorted(set(data["subjects"])):
        ix = data["subjects"] == subject
        n, a = int(ix.sum()), int(accepted[ix].sum())
        k = int(data["errors"][ix & accepted].sum())
        if not 0 <= k <= a <= n:
            raise AssertionError("Invalid count identity")
        assert k == sum(int(e) for e, keep in zip(data["errors"][ix], accepted[ix]) if keep)
        folds = set(data["fold_ids"][ix].tolist())
        assert len(folds) == 1
        rows.append({"branch": branch, "context": context, "risk_fold": next(iter(folds)),
            "cell": cell, "evaluation": evaluation, "subject": subject, "n": n,
            "accepted": a, "errors": k, "coverage": a/n, "error_mass": k/n,
            "subject_risk": k/a if a else np.nan, "score_space": space})
    return rows


def evaluate_cells(branch, context, cells, data, cfg):
    rows, exchanges, quota_masks = [], [], {}
    keys = tie_keys(data["identities"], context, cfg)
    for cell, item in cells.items():
        raw, operation = item["raw"], item.get("operation")
        if not np.isfinite(raw).all():
            if cell != "BCEAnchor-equalnorm":
                raise RuntimeError("Invalid primary score stream")
            continue
        if operation is not None:
            label = "locked_rule" if branch == "nested" else "selected_fit_rule"
            rows += subject_rows(branch, context, cell, label, operation, data, "calibrated_logit")
        for q in cfg["quotas"]:
            accepted = quota_mask(raw, data["subjects"], keys, q)
            r = subject_rows(branch, context, cell, f"quota_{q:.1f}", accepted, data, "raw")
            assert all(z["accepted"] == int(np.floor(q*z["n"])) for z in r)
            rows += r
            if q == .5:
                quota_masks[cell] = accepted
        oracle_mask = np.zeros(len(raw), dtype=bool)
        for fold in range(5):
            ix = data["fold_ids"] == fold
            oracle_mask[ix] = hard_oracle(raw[ix], data["errors"][ix], data["subjects"][ix], cfg["alpha"])
        rows += subject_rows(branch, context, cell, "oracle", oracle_mask, data, "raw")
    for cell, reference in COMPARISONS[branch]:
        if cell not in quota_masks or reference not in quota_masks:
            continue
        a, b = quota_masks[cell], quota_masks[reference]
        for s in sorted(set(data["subjects"])):
            ix = data["subjects"] == s
            added = int(data["errors"][ix & a & ~b].sum())
            removed = int(data["errors"][ix & ~a & b].sum())
            difference = int(data["errors"][ix & a].sum()-data["errors"][ix & b].sum())
            assert added-removed == difference and a[ix].sum() == b[ix].sum()
            exchanges.append({"branch": branch, "context": context, "subject": s,
                "cell": cell, "reference": reference, "quota": .5,
                "changed_decisions": int(((a != b) & ix).sum()),
                "added_errors": added, "removed_errors": removed, "net_errors": difference})
    return rows, exchanges


def delete_one_selection(context, selection, cfg):
    from stage1_analysis.acceptance import select_empirical_policy
    result = []
    candidates = {}
    for method in ("BCEAnchor", "SELE"):
        candidates[method] = []
        for candidate in selection["candidates"][method]:
            name = f'{method}-{candidate["parameter"]:g}'
            rule_rows = read(UOUT / "selection" / f"context-{context}" / (name+"_rules.json"))["rules"]
            assert len(rule_rows) == 20 and all(len(r["counts"]) == 15 for r in rule_rows)
            candidates[method].append((candidate, rule_rows))
    subjects = [r["subject"] for r in candidates["BCEAnchor"][0][1][0]["counts"]]
    for excluded in subjects:
        by_method = {}
        for method, sources in candidates.items():
            reranked = []
            for candidate, rules in sources:
                changed = []
                for r in rules:
                    counts = [z for z in r["counts"] if z["subject"] != excluded]
                    metrics = counts_risk(counts)
                    changed.append({k: r[k] for k in ("q", "threshold", "tie_fraction")}
                        | metrics | {"supported": True})
                picked = select_empirical_policy(changed, alpha=cfg["alpha"])
                reranked.append({"parameter": candidate["parameter"], "picked": picked, "status": "OK"})
            by_method[method] = reranked
        for cell, method, nominal in (("A0", "BCEAnchor", False), ("A1", "BCEAnchor", True),
                                     ("S0", "SELE", False), ("S1", "SELE", True), ("A2", "BCEAnchor", True)):
            choice = (next(r for r in by_method[method] if r["parameter"] == 1.) if cell == "A2"
                      else select_candidate(by_method[method], method, nominal))
            picked = choice["picked"]
            result.append({"context": context, "excluded_subject": excluded, "cell": cell,
                "parameter": choice["parameter"], "q": float(exact(picked["q"])) if picked else np.nan,
                "coverage": float(exact(picked["coverage"])) if picked else 0.,
                "R_SE": float(exact(picked["risk"])) if picked else np.nan,
                "interpretation": "fixed_scores_fixed_thresholds_reselect_not_nested"})
    return result


def aggregate_people(per):
    keys = ["branch", "cell", "evaluation", "subject"]
    return per.groupby(keys, sort=True).agg(
        c_bar=("coverage", "mean"), e_bar=("error_mass", "mean"),
        repeated_accepted=("accepted", "sum"), repeated_errors=("errors", "sum"),
        appearances=("context", "nunique"), records=("n", "size"),
        context_signature=("context", lambda values: json.dumps(
            sorted({str(value) for value in values}), separators=(",", ":"))),
        zero_accept_records=("accepted", lambda x: int((x == 0).sum()))).reset_index()


def complete_people(group, expected_contexts):
    """Check actual frozen role memberships; patient frequencies need not equal 15."""
    if len(group) != len(expected_contexts) or set(group.subject) != set(expected_contexts):
        return False
    for row in group.itertuples():
        expected = sorted(expected_contexts[row.subject])
        if (row.appearances != len(expected) or row.records != len(expected)
                or json.loads(row.context_signature) != expected):
            return False
    return True


def summarize(people, expected_contexts):
    rows = []
    for (branch, cell, evaluation), g in people.groupby(["branch", "cell", "evaluation"], sort=True):
        c, e = float(g.c_bar.mean()), float(g.e_bar.mean())
        complete = complete_people(g, expected_contexts)
        rows.append({"branch": branch, "cell": cell, "evaluation": evaluation,
            "coverage": c if complete else np.nan,
            "R_SE": e/c if complete and c else np.nan,
            "repeated_accepted": int(g.repeated_accepted.sum()),
            "repeated_errors": int(g.repeated_errors.sum()), "subjects": len(g),
            "repeat_records": int(g.records.sum()),
            "zero_accept_records": int(g.zero_accept_records.sum()),
            "status": "COMPLETE" if complete else "UNDEFINED_OR_INCOMPLETE_CONTROL"})
    return pd.DataFrame(rows)


def ratio(c, e, axis=-1):
    denominator, numerator = np.mean(c, axis=axis), np.mean(e, axis=axis)
    return np.divide(numerator, denominator, out=np.full_like(denominator, np.nan, dtype=float),
                     where=denominator > 0)


def paired_tables(people, cfg, expected_contexts):
    paired, deletion = [], []
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    draws = rng.integers(0, 20, size=(cfg["bootstrap_replicates"], 20))
    for branch, pairs in COMPARISONS.items():
        for evaluation in sorted(people[people.branch == branch].evaluation.unique()):
            frame = people[(people.branch == branch) & (people.evaluation == evaluation)]
            for cell, reference in pairs:
                a = frame[frame.cell == cell].set_index("subject").sort_index()
                b = frame[frame.cell == reference].set_index("subject").sort_index()
                if (not a.index.equals(b.index)
                        or not complete_people(a.reset_index(), expected_contexts)
                        or not complete_people(b.reset_index(), expected_contexts)):
                    continue
                ac, ae = a.c_bar.to_numpy(), a.e_bar.to_numpy()
                bc, be = b.c_bar.to_numpy(), b.e_bar.to_numpy()
                boot = {"coverage": (ac[draws]-bc[draws]).mean(axis=1),
                        "R_SE": ratio(ac[draws], ae[draws])-ratio(bc[draws], be[draws])}
                points = {"coverage": float((ac-bc).mean()),
                          "R_SE": float(ratio(ac, ae)-ratio(bc, be))}
                for metric in points:
                    values = boot[metric]; valid = np.isfinite(values)
                    for j, subject in enumerate(a.index):
                        keep = np.arange(20) != j
                        delta = ((ac[keep]-bc[keep]).mean() if metric == "coverage" else
                                 ratio(ac[keep], ae[keep])-ratio(bc[keep], be[keep]))
                        deletion.append({"branch": branch, "cell": cell, "reference": reference,
                            "evaluation": evaluation, "metric": metric, "excluded_subject": subject,
                            "difference": float(delta)})
                    # No redraw. Any undefined bootstrap sample makes the ordinary
                    # percentile interval unavailable; its frequency stays visible.
                    ci = np.quantile(values, [.025, .975]) if valid.all() else [np.nan, np.nan]
                    individual = ac-bc if metric == "coverage" else ratio(ac[:, None], ae[:, None])-ratio(bc[:, None], be[:, None])
                    paired.append({"branch": branch, "cell": cell, "reference": reference,
                        "evaluation": evaluation, "metric": metric, "difference": points[metric],
                        "ci_low": ci[0], "ci_high": ci[1], "undefined_bootstrap": int((~valid).sum()),
                        "negative_subjects": int((individual < -1e-12).sum()),
                        "positive_subjects": int((individual > 1e-12).sum()),
                        "tied_subjects": int((np.abs(individual) <= 1e-12).sum()),
                        "undefined_subjects": int((~np.isfinite(individual)).sum())})
    return pd.DataFrame(paired), pd.DataFrame(deletion)


def export_table(name, frame):
    path = UOUT / "tables" / (name+".csv")
    text_file(path, frame.to_csv(index=False, lineterminator="\n", float_format="%.17g"))
    return path.relative_to(UOUT).as_posix(), digest(path)


def replay(budget, branch="all"):
    import scipy.optimize
    from . import bceanchor_upgrade as training, calibration, solvers
    def forbid(*args, **kwargs):
        raise RuntimeError("Replay cannot fit")
    scipy.optimize.minimize = forbid
    training.fit_bce = training.fit_sele = training.fit_refinement = forbid
    training.fit_mapping = calibration.fit_mapping = forbid
    solvers.fit_bce = solvers.fit_sele = forbid
    training.Store.cached = training.Store.legacy = forbid
    training.data_only = training.load_fit = forbid
    cfg, pre = spec(), require_preflight()
    branches = ["selection", "averaging", "nested"] if branch == "all" else [branch]
    for b in branches:
        require_complete(b, pre)
    rows, exchanges, deletions, selected_rows, geometry, undefined = [], [], [], [], [], []
    expected_contexts = {}
    for i in range(20):
        context = f"{i:02d}"
        budget.check()
        data = load_inputs(context, pre)
        for subject in sorted(set(data["subjects"])):
            expected_contexts.setdefault(str(subject), []).append(context)
        selection = read(UOUT / "selection" / f"context-{context}/selection.json")
        reference = selection_streams(context, selection)
        if "selection" in branches:
            cells = {name: {"raw": r, "operation": mask(c, p, data, context, cfg)}
                     for name, (r, c, p) in reference.items()}
            r, e = evaluate_cells("selection", context, cells, data, cfg)
            rows += r; exchanges += e
            deletions += delete_one_selection(context, selection, cfg)
            for name, chosen in selection["chosen"].items():
                p = chosen["picked"]
                selected_rows.append({"branch": "selection", "context": context, "risk_fold": -1,
                    "cell": name, "parameter": chosen["parameter"],
                    "picked_q": float(exact(p["q"])) if p else np.nan})
        if "averaging" in branches:
            folder = UOUT / "averaging" / f"context-{context}"
            receipt = read(folder / "scores.json")
            assert digest(folder / "scores.npz") == receipt["sha256"]
            with np.load(folder / "scores.npz", allow_pickle=False) as z:
                cells = {name: {"raw": z["raw"][:, j].copy()} for j, name in enumerate(z["cells"].tolist())}
            for new, old in (("BCEAnchor-full-1", "A2"), ("FixedScale-full-1", "FixedScale-1"),
                             ("SELE-full-S1", "S1"), ("SELE-historical-S0", "S0"), ("BCE", "BCE")):
                cells[new] = {"raw": reference[old][0]}
            for name in receipt["undefined_controls"]:
                undefined.append({"context": context, "cell": name})
            r, e = evaluate_cells("averaging", context, cells, data, cfg)
            rows += r; exchanges += e
            for j in range(5):
                d = read(folder / f"fold-{j}/completion.json")
                g = d["geometry"]
                geometry.append({"context": context, "risk_fold": j,
                    "parent_BCE_C": d["parent_BCE_C"], "SELE_C": d["SELE_C"],
                    "SELE_reg": d["SELE_reg"], "parent_epoch_n": d["parent_epoch_n"],
                    "mean_delta_norm": g["mean_delta_norm"], "full_delta_norm": g["full_delta_norm"],
                    "mean_full_distance": g["mean_full_distance"],
                    "equalnorm_coefficient": g["equalnorm"].get("coefficient", np.nan),
                    "mean_member_pair_distance": np.mean(g["member_pair_distances"]),
                    "mean_member_score_correlation": np.asarray(g["member_score_correlation"])[np.triu_indices(4, 1)].mean()})
        if "nested" in branches:
            raw = np.full((len(data["errors"]), len(NESTED_CELLS)), np.nan)
            operation = np.zeros_like(raw, dtype=bool)
            for j in range(5):
                folder = UOUT / "nested" / f"context-{context}" / f"fold-{j}"
                receipt = read(folder / "completion.json")
                assert digest(folder / "predictions.npz") == receipt["scores_sha256"]
                with np.load(folder / "predictions.npz", allow_pickle=False) as z:
                    assert z["cells"].tolist() == NESTED_CELLS
                    assert np.all(data["fold_ids"][z["indices"]] == j)
                    raw[z["indices"]] = z["raw"]
                    operation[z["indices"]] = z["decisions"]
                selected_rows.append({"branch": "nested", "context": context, "risk_fold": j,
                    "cell": "Nested-BCEAnchor-selected", "parameter": receipt["selected_lambda"],
                    "BCE_C": receipt["selected_BCE_C"], "SELE_C": receipt["selected_SELE_C"],
                    "picked_q": float(exact(receipt["locked_rules"]["Nested-BCEAnchor-selected"]["q"]))
                        if receipt["locked_rules"]["Nested-BCEAnchor-selected"] else np.nan})
            cells = {name: {"raw": raw[:, j], "operation": operation[:, j]}
                     for j, name in enumerate(NESTED_CELLS)}
            r, e = evaluate_cells("nested", context, cells, data, cfg)
            rows += r; exchanges += e
    per = pd.DataFrame(rows)
    people = aggregate_people(per)
    assert len(expected_contexts) == 20 and sum(map(len, expected_contexts.values())) == 300
    write(UOUT / "evaluation_roles.json", {"source": "saved_frozen_FIT_identity_inputs",
        "subject_contexts": expected_contexts,
        "appearances": {s: len(contexts) for s, contexts in expected_contexts.items()}})
    summary = summarize(people, expected_contexts)
    primary = summary[summary.cell != "BCEAnchor-equalnorm"]
    if not primary.status.eq("COMPLETE").all():
        raise RuntimeError("Incomplete primary comparison; frozen role memberships do not match")
    paired, delete_patient = paired_tables(people, cfg, expected_contexts)
    if paired.empty or delete_patient.empty:
        raise RuntimeError("Complete comparison requires paired and delete-one results")
    costs = []
    for folder in ("selection", "averaging", "nested"):
        for p in sorted((UOUT / folder).rglob("*.json")):
            r = read(p)
            if r.get("kind") not in ("risk", "platt"):
                continue
            costs.append({"branch": folder, "kind": r["kind"], "seconds": r["seconds"],
                          "status": r["status"], "path": p.relative_to(UOUT).as_posix()})
    frames = {"summary": summary, "per_subject_records": per, "subject_means": people,
        "paired_differences": paired, "delete_one_subject": delete_patient,
        "acceptance_changes": pd.DataFrame(exchanges), "selected_configs": pd.DataFrame(selected_rows),
        "selector_delete_one": pd.DataFrame(deletions), "averaging_geometry": pd.DataFrame(geometry),
        "fit_costs": pd.DataFrame(costs)}
    hashes = dict(export_table(name, frame) for name, frame in frames.items())
    changed = [rel for rel, sha in pre["protected"].items() if digest(PROJECT / rel) != sha]
    if changed:
        raise RuntimeError("Historical assets changed: " + repr(changed))
    old = read(UOUT / "replay_checks.json") if (UOUT / "replay_checks.json").exists() else None
    checks = {"status": "PASS", "branches": branches, "contexts": 20,
        "subject_count_checks": len(per), "protected_unchanged": len(pre["protected"]),
        "csv_sha256": hashes, "repeat_replay_byte_identical": bool(old and old["csv_sha256"] == hashes),
        "undefined_controls": undefined, "fit_fingerprint": pre["identity"]["sha256"],
        "report_code_sha256": digest(__file__), "cert_outer_evaluation": "NOT_EXECUTED",
        "no_fitting_or_calibration_in_replay": True}
    write(UOUT / "replay_checks.json", checks)
    print(summary[summary.evaluation.isin(["locked_rule", "quota_0.5"])].to_string(index=False), flush=True)


def report(budget, branch="all"):
    checks = read(UOUT / "replay_checks.json")
    assert checks["status"] == "PASS" and set(checks["branches"]) == {"selection", "averaging", "nested"}
    for rel, sha in checks["csv_sha256"].items():
        assert digest(UOUT / rel) == sha
    from .bceanchor_upgrade_teaching import build_report_and_notebook
    build_report_and_notebook(budget)
