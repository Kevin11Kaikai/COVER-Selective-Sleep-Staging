"""Frozen-point comparisons, controlled ranking diagnostics and composition audits."""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .inputs import load, read_json, save_compact, sha256
from .metrics import (bootstrap_metrics, classification, count_subjects, identity_ties, interval,
                      overlap, quota_mask, ranked_orders, selection_class_counts, summary)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, out: Path, compact: bool = False) -> dict:
    started = time.perf_counter()
    out.resolve().relative_to(root.resolve())
    cfg_path = root / "configs/analysis/strengthening_v1.json"
    cfg = read_json(cfg_path)
    data = load(root, compact=compact)
    if not compact:
        save_compact(data, root)
    people, methods = data.people, cfg["main_methods"]
    n = np.array([np.count_nonzero(data.subjects == s) for s in people], dtype=int)
    errors, labels = data.errors, data.labels
    ties = identity_ties(data.keys)
    orders = {m: ranked_orders(data.scores[m], ties, data.subjects, people) for m in methods}
    selected = {m: data.masks[f"{m}|0.05|empirical"] for m in methods}
    reference = selected["Learned-Selected"]
    _, quotas, _ = count_subjects(reference, errors, data.subjects, people)
    matched = {m: quota_mask(orders[m], quotas, len(errors)) for m in methods}
    legacy = dict(matched)
    legacy["Learned-Selected"] = reference
    assert (int(reference.sum()), int((reference & errors).sum())) == (19892, 817)
    assert int((legacy["MSP"] & errors).sum()) == 1369
    assert int((legacy["Primary-only"] & errors).sum()) == 1133
    main_contexts = {"frozen_cert05": selected, "matched_original_reference": legacy,
                     "matched_common_ties": matched}
    secondary = {"Learned-Selected": data.masks["historical|B0.03-Heterogeneous"],
                 "Primary-only": data.masks["historical|B0.03-Primary-only"],
                 "DuoF": data.masks["DuoF|0.03|empirical"]}
    contexts = {**main_contexts, "frozen_cert03": secondary,
                "historical_policies": {k.split("|", 1)[1]: v for k, v in data.masks.items()
                                        if k.startswith("historical|")},
                "all_frozen_cert05": {k.split("|")[0]: v for k, v in data.masks.items()
                                      if k.endswith("|0.05|empirical")}}
    table, per_subject, class_rows, confusion_rows, count_cache = [], [], [], [], {}
    for context, masks in contexts.items():
        for method, mask in masks.items():
            counts = count_subjects(mask, errors, data.subjects, people)
            nn, aa, kk = counts
            count_cache[context, method] = counts
            table.append({"context": context, "method": method, **summary(*counts)})
            for s, ni, ai, ki in zip(people, nn, aa, kk):
                per_subject.append({"context": context, "method": method, "subject": s,
                    "eligible": int(ni), "accepted": int(ai), "errors": int(ki),
                    "coverage": float(ai / ni), "risk": float(ki / ai) if ai else None,
                    "passed": bool(ai >= 50 and 20 * ki <= ai), "zero": bool(ai == 0)})
            for c, name in enumerate(["Wake", "N1", "N2", "N3", "REM"]):
                cm = mask & (labels == c)
                class_rows.append({"context": context, "method": method, "true_class": name,
                    **selection_class_counts(labels, errors, mask, c)})
                for cp, pname in enumerate(["Wake", "N1", "N2", "N3", "REM"]):
                    confusion_rows.append({"context": context, "method": method, "true_class": name,
                        "predicted_class": pname, "accepted": int((cm & (data.predictions == cp)).sum())})
    rng = np.random.Generator(np.random.PCG64(cfg["bootstrap"]["seed"]))
    draws = rng.integers(0, len(people), size=(cfg["bootstrap"]["replicates"], len(people)))
    boots = {key: bootstrap_metrics(*counts, draws) for key, counts in count_cache.items()
             if key[0] in main_contexts or key[0] == "frozen_cert03"}
    paired, exchanges, regions, transitions, influences = [], [], [], [], []
    comparisons = [(context, "Learned-Selected", b) for context in main_contexts for b in methods[1:]]
    comparisons += [("frozen_cert03", "Learned-Selected", b) for b in ["DuoF", "Primary-only"]]
    comparisons += [("historical_policies", a, b) for a, b in cfg["historical_case_pairs"]]
    for context, a_name, b_name in comparisons:
        a, b = contexts[context][a_name], contexts[context][b_name]
        ca, cb = count_cache[context, a_name], count_cache[context, b_name]
        meta = {"context": context, "method_a": a_name, "method_b": b_name}
        osum, rs = overlap(a, b, errors)
        boot_union = np.array([((a | b) & (data.subjects == s)).sum() for s in people])[draws].sum(axis=1)
        boot_shared = np.array([((a & b) & (data.subjects == s)).sum() for s in people])[draws].sum(axis=1)
        boot_flips = np.array([((a ^ b) & (data.subjects == s)).sum() for s in people])[draws].sum(axis=1)
        boot_j = np.divide(boot_shared, boot_union, out=np.full(len(draws), np.nan), where=boot_union > 0)
        ji, fi = interval(boot_j), interval(boot_flips / n[draws].sum(axis=1))
        exchanges.append({**meta, **osum, "jaccard_low": ji["low"], "jaccard_high": ji["high"],
                          "flip_rate_low": fi["low"], "flip_rate_high": fi["high"]})
        regions.extend([{**meta, **r} for r in rs])
        aa, ka, ab, kb = ca[1], ca[2], cb[1], cb[2]
        dc = (aa - ab) / n
        gains = dc if context.startswith("frozen") or context == "historical_policies" else (kb - ka) / n
        # Descriptive deletion checks are clearly outcome-dependent, never model selection.
        positive = np.flatnonzero(gains > 0)
        gain_order = positive[np.argsort(-gains[positive], kind="stable")]
        for drop in [[], *[[i] for i in range(len(people))], gain_order[:1].tolist(), gain_order[:2].tolist()]:
            keep = np.ones(len(people), bool)
            keep[drop] = False
            sa, sb = summary(n[keep], aa[keep], ka[keep]), summary(n[keep], ab[keep], kb[keep])
            influences.append({**meta, "deleted_subjects": ";".join(people[i] for i in drop),
                "deletion_rank_metric": "coverage_gain" if context.startswith("frozen") or context == "historical_policies" else "error_mass_reduction",
                "remaining_subjects": int(keep.sum()), "macro_coverage_delta": sa["macro_coverage"] - sb["macro_coverage"],
                "pooled_risk_delta": sa["pooled_risk"] - sb["pooled_risk"] if sa["pooled_risk"] is not None and sb["pooled_risk"] is not None else None})
        for i, s in enumerate(people):
            transitions.append({**meta, "subject": s, "coverage_delta": float(dc[i]),
                "a_accepted": int(aa[i]), "b_accepted": int(ab[i]), "a_errors": int(ka[i]), "b_errors": int(kb[i]),
                "a_pass": bool(aa[i] >= 50 and 20 * ka[i] <= aa[i]),
                "b_pass": bool(ab[i] >= 50 and 20 * kb[i] <= ab[i]),
                "flip_count": int(((a ^ b) & (data.subjects == s)).sum())})
        if (context, a_name) not in boots:
            boots[context, a_name] = bootstrap_metrics(*ca, draws)
            boots[context, b_name] = bootstrap_metrics(*cb, draws)
        sa, sb = summary(*ca), summary(*cb)
        for metric in ["macro_coverage", "pooled_risk", "subject_equal_ratio_risk", "passed_subjects"]:
            diffs = boots[context, a_name][metric] - boots[context, b_name][metric]
            paired.append({**meta, "metric": metric, "a_minus_b": sa[metric] - sb[metric],
                           **interval(diffs), "replicates": len(draws), "interpretation": "descriptive_paired_subject_bootstrap"})
    curve_rows = []
    for pct in cfg["quota_percentages"]:
        quota = n * pct // 100
        for method in methods:
            mask = quota_mask(orders[method], quota, len(errors))
            counts = count_subjects(mask, errors, data.subjects, people)
            values = bootstrap_metrics(*counts, draws)
            ci = interval(values["pooled_risk"])
            curve_rows.append({"method": method, "quota_percent": pct, **summary(*counts),
                               "pooled_risk_low": ci["low"], "pooled_risk_high": ci["high"]})
    tie_changes = []
    for s in people:
        sm = data.subjects == s
        tie_changes.append({"subject": s, "quota": int(reference[sm].sum()),
            "flipped": int((reference[sm] ^ matched["Learned-Selected"][sm]).sum()),
            "old_errors": int((reference & errors & sm).sum()),
            "common_tie_errors": int((matched["Learned-Selected"] & errors & sm).sum())})
    checks = {"canonical_rows": len(errors), "subjects": len(people), "records": 39,
        "historical_matched_errors": {m: int((v & errors).sum()) for m, v in legacy.items()},
        "common_tie_matched_errors": {m: int((v & errors).sum()) for m, v in matched.items()},
        "learned_common_tie_flips": int((reference ^ matched["Learned-Selected"]).sum()),
        "matched_accept_count": int(reference.sum()), "matched_subject_quotas_equal": True,
        "main_methods": methods, "new_fitting": False, "certified": False,
        "external_test": "NOT_EXECUTED", "historical_gate": "NO_GO_UNCHANGED",
        "primary_accuracy": float(np.mean(~errors)), "analysis_spec_sha256": sha256(cfg_path)}
    for context in ["matched_original_reference", "matched_common_ties"]:
        for method in methods:
            assert np.array_equal(count_cache[context, method][1], quotas)
    assert not (contexts["historical_policies"]["B0.02-Heterogeneous"] & ~reference).any()
    historical_e, historical_b = (contexts["historical_policies"][m] for m in ["E-Heterogeneous", "B0.03-Heterogeneous"])
    assert (historical_e ^ historical_b).sum() == 2309
    component_overall, component_classes, component_subjects, component_confusion = [], [], [], []
    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    for model, probabilities in [("EEG Conformer", data.primary_probabilities),
                                 ("RF (isotonic)", data.feature_probabilities)]:
        predictions = probabilities.argmax(axis=1)
        aggregate, classes, cm = classification(labels, predictions)
        component_overall.append({"model": model, "aggregation": "pooled_epochs", **aggregate})
        component_classes.extend({"model": model, **r} for r in classes)
        subject_results = []
        for person in people:
            group = data.subjects == person
            stats, cr, _ = classification(labels[group], predictions[group])
            row = {"model": model, "subject": person, **stats,
                   "no_predicted_classes": ";".join(r["class"] for r in cr if not r["precision_defined"])}
            component_subjects.append(row)
            subject_results.append(stats)
        component_overall.append({"model": model, "aggregation": "mean_subject", "epochs": len(labels),
                                  **{m: float(np.mean([s[m] for s in subject_results])) for m in metric_names},
                                  "zero_division": 0})
        for i, cname in enumerate(["Wake", "N1", "N2", "N3", "REM"]):
            for j, pname in enumerate(["Wake", "N1", "N2", "N3", "REM"]):
                component_confusion.append({"model": model, "true_class": cname,
                    "predicted_class": pname, "count": int(cm[i, j]),
                    "true_class_fraction": float(cm[i, j] / cm[i].sum()) if cm[i].sum() else None})
    n1 = next(r for r in class_rows if r["context"] == "frozen_cert05"
              and r["method"] == "Learned-Selected" and r["true_class"] == "N1")
    assert [n1[k] for k in ["support", "base_correct", "correct_accepted", "errors",
                            "correct_rejected", "incorrect_rejected"]] == [2804, 831, 13, 252, 818, 1721]
    n1_predicted = reference & (data.predictions == 1)
    assert (int(n1_predicted.sum()), int((n1_predicted & ~errors).sum())) == (20, 13)
    rem = [next(r for r in class_rows if r["context"] == "matched_common_ties"
                and r["method"] == m and r["true_class"] == "REM") for m in ["Learned-Selected", "DuoF"]]
    rem_delta = {k: rem[0][k] - rem[1][k] for k in ["accepted", "correct_accepted", "errors"]}
    assert list(rem_delta.values()) == [149, 117, 32]
    checks["class_diagnostics"] = {"n1": n1, "n1_standard_accepted_precision": 13 / 20,
                                   "rem_matched_delta": rem_delta,
                                   "class_accounting_verified": True}
    checks["component_probabilities_verified"] = True
    files = {"operating_points": table, "per_subject": per_subject, "class_acceptance": class_rows,
             "accepted_confusion": confusion_rows, "paired_bootstrap": paired, "acceptance_overlap": exchanges,
             "exchange_regions": regions, "subject_transitions": transitions, "influence_deletions": influences,
             "risk_coverage_diagnostic": curve_rows, "common_tie_changes": tie_changes,
             "component_overall": component_overall, "component_per_class": component_classes,
             "component_per_subject": component_subjects, "component_confusion": component_confusion}
    for name, rows in files.items():
        write_csv(out / "tables" / (name + ".csv"), rows)
    write_csv(out / "tables/component_roles.csv", data.audit["splits"]["role_rows"])
    np.savez_compressed(out / "metrics/matched_masks.npz", method_names=methods,
        original_reference_masks=np.column_stack([legacy[m] for m in methods]),
        common_tie_masks=np.column_stack([matched[m] for m in methods]),
        subjects=data.subjects, identities_json=json.dumps(data.identities))
    write_json(out / "metrics/source_audit.json", data.audit)
    write_json(out / "metrics/input_fingerprints.json", data.tracker.files)
    write_json(out / "metrics/checks.json", checks)
    source_files = list((root / "src/stage1_analysis").glob("*.py")) + [root / "Scripts/reproduce.py", cfg_path]
    write_json(out / "metrics/execution_fingerprints.json", {p.relative_to(root).as_posix(): sha256(p) for p in source_files})
    state = {"status": "COMPLETE", "completed_utc": datetime.now(timezone.utc).isoformat(),
             "elapsed_seconds": round(time.perf_counter() - started, 3), "compact_replay": compact,
             "tables": {name: len(rows) for name, rows in files.items()}, **checks}
    write_json(out / "metrics/analysis_status.json", state)
    return state
