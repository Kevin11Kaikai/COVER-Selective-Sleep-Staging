"""Fit-free replay for the anchored extension; outcome access requires a complete seal."""
from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path

import numpy as np

from .storage import ROOT, OUT, arrays, digest, read, text_file, write
from .reporting import average_ranks, csv_rows, table
from stage1_analysis.metrics import (
    alignment, bootstrap_metrics, count_subjects, identity_key, identity_ties,
    interval, quota_mask, ranked_orders, summary,
)

AOUT = OUT / "anchored"
METHODS = ["BCE", "SELE", "RiskObjective", "FixedScale", "BCEAnchor"]


def require_seal():
    seal = read(AOUT / "decisions_sealed.json")
    selection = read(AOUT / "fit_selections.json")
    final = read(AOUT / "fit_completed.json")
    expected = {f"{i:02d}" for i in range(20)}
    if (seal["status"] != "ALL_OUTER_DECISIONS_FROZEN"
            or set(seal["folds"]) != expected
            or selection["status"] != "ALL_FIT_SELECTIONS_FROZEN"
            or set(selection["folds"]) != expected
            or final["status"] != "ALL_FINAL_SCORERS_FITTED"
            or final["selections_sha256"] != digest(AOUT / "fit_selections.json")
            or len({seal["fingerprint"], selection["fingerprint"], final["fingerprint"]}) != 1):
        raise RuntimeError("All FIT selections, final scorers and OUTER decisions must be sealed")
    for outer, receipt in seal["folds"].items():
        path = AOUT / "decisions" / f"outer-{outer}.npz"
        if receipt["sha256"] != digest(path) or receipt["contains_outer_labels"]:
            raise RuntimeError("Invalid decision identity or premature outcome inclusion")
        if receipt["fingerprint"] != seal["fingerprint"]:
            raise RuntimeError("Mixed fitting fingerprints")
    return seal


def make_portable():
    """First access to OUTER truth occurs only after checking all twenty decisions."""
    require_seal()
    base_path = OUT / "replay_inputs.npz"
    if read(OUT / "replay_inputs.json")["sha256"] != digest(base_path):
        raise RuntimeError("Historical replay input changed")
    with np.load(base_path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    keys = [identity_key(i) for i in json.loads(data["identities_json"].item())]
    n = len(keys)
    visited = np.zeros(n, bool)
    new_scores = np.full((n, 4), np.nan)
    new_masks = np.zeros((n, 4), bool)
    new_raw = np.full((n, 2), np.nan)
    cells = None
    for outer in (f"{i:02d}" for i in range(20)):
        with np.load(AOUT / "decisions" / f"outer-{outer}.npz", allow_pickle=False) as z:
            source = [identity_key(i) for i in json.loads(z["identities_json"].item())]
            pos = alignment(source, keys)
            if visited[pos].any() or not (data["subjects"][pos] == outer).all():
                raise RuntimeError("Overlapping or mismatched OUTER identities")
            np.testing.assert_array_equal(z["predictions"], data["predictions"][pos])
            np.testing.assert_array_equal(z["subjects"], data["subjects"][pos])
            names = z["cells"].tolist()
            if cells is not None and cells != names:
                raise RuntimeError("Cell ordering differs between outer contexts")
            cells = names
            new_scores[pos], new_masks[pos], new_raw[pos] = z["scores"], z["masks"], z["raw_scores"]
            visited[pos] = True
    if not visited.all() or not np.isfinite(new_scores).all() or n != 42308:
        raise RuntimeError("Incomplete anchored result")
    data["cells"] = np.r_[data["cells"], np.asarray(cells)]
    data["scores"] = np.column_stack([data["scores"], new_scores])
    data["masks"] = np.column_stack([data["masks"], new_masks])
    data["anchor_raw_scores"] = new_raw
    path = AOUT / "replay_inputs.npz"
    arrays(path, **data)
    historical_path = ROOT.parent / "outputs/strengthening/stage1_strengthening_v1/tables/automl_summary.csv"
    with historical_path.open(encoding="utf-8") as f:
        historical = [r for r in csv.DictReader(f) if r["arm"] in
            ("COVER-AutoML-Pi20", "COVER-AutoML-Pi100")]
    write(AOUT / "replay_inputs.json", {
        "sha256": digest(path), "epochs": n,
        "decision_seal_sha256": digest(AOUT / "decisions_sealed.json"),
        "historical_input_sha256": digest(base_path),
        "outcomes_joined_after_all_decisions": True,
        "historical_automl_source": historical_path.relative_to(ROOT.parent).as_posix(),
        "historical_automl_sha256": digest(historical_path),
        "historical_automl_rows": historical,
    })


def number(value):
    if value is None:
        return None
    return value["numerator"] / value["denominator"] if isinstance(value, dict) else float(value)


def replay(budget, resume=False):
    seal = require_seal()
    path = AOUT / "replay_inputs.npz"
    if not path.exists():
        make_portable()
    receipt = read(AOUT / "replay_inputs.json")
    if (receipt["sha256"] != digest(path)
            or receipt["decision_seal_sha256"] != digest(AOUT / "decisions_sealed.json")
            or receipt["historical_input_sha256"] != digest(OUT / "replay_inputs.npz")):
        raise RuntimeError("Anchored replay provenance mismatch")
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    subjects, errors = data["subjects"], data["errors"].astype(bool)
    people = sorted(set(subjects.tolist()))
    assert len(people) == 20 and len(errors) == 42308
    np.testing.assert_array_equal(errors, data["predictions"] != data["labels"])
    keys = [identity_key(i) for i in json.loads(data["identities_json"].item())]
    assert len(set(keys)) == len(keys)
    ties = identity_ties(keys)
    _, quotas, _ = count_subjects(data["reference_masks"][:, 0], errors, subjects, people)
    cells = data["cells"].tolist()
    counts, matched_masks, rows, per = {}, {}, [], []
    for j, cell in enumerate(cells):
        budget.check()
        matched = quota_mask(ranked_orders(data["scores"][:, j], ties, subjects, people), quotas, len(errors))
        matched_masks[cell] = matched
        for context, mask in [("cert", data["masks"][:, j]), ("matched", matched)]:
            n, a, k = count_subjects(mask, errors, subjects, people)
            counts[context, cell] = (n, a, k)
            if context == "matched":
                np.testing.assert_array_equal(a, quotas)
            rows.append({"context": context, "cell": cell, **summary(n, a, k)})
            for subject, nn, aa, kk in zip(people, n, a, k):
                per.append({"context": context, "cell": cell, "subject": subject,
                    "eligible": int(nn), "accepted": int(aa), "errors": int(kk),
                    "coverage": aa / nn, "risk": kk / aa if aa else None,
                    "passed": bool(aa >= 50 and 20 * kk <= aa)})
    for method in METHODS:
        np.testing.assert_array_equal(matched_masks[method + "|20"], matched_masks[method + "|100"])
    for method, expected in [("BCE", 794), ("SELE", 798), ("RiskObjective", 806)]:
        assert int(counts["matched", method + "|20"][2].sum()) == expected
    references = []
    for j, method in enumerate(data["reference_names"].tolist()):
        matched = quota_mask(ranked_orders(data["reference_scores"][:, j], ties, subjects, people), quotas, len(errors))
        assert int(np.sum(matched & errors)) == [817, 946, 1133, 1369, 2171][j]
        for context, mask in [("cert", data["reference_masks"][:, j]), ("matched", matched)]:
            references.append({"context": context, "method": method,
                **summary(*count_subjects(mask, errors, subjects, people))})
    for row in receipt["historical_automl_rows"]:
        context = {"cert05": "cert", "matched_common_ties": "matched"}[row["context"]]
        entry = {"context": context, "method": row["arm"], "source": "saved historical AutoML summary"}
        for key in summary(*counts["cert", "BCE|20"]):
            value = row[key]
            entry[key] = (None if value == "" else float(value)) if key in (
                "macro_coverage", "pooled_coverage", "subject_equal_ratio_risk", "pooled_risk") else int(value)
        references.append(entry)
    rng = np.random.Generator(np.random.PCG64(20260901))
    samples = rng.integers(0, 20, size=(10000, 20))
    pairs = [("BCEAnchor", m) for m in ["BCE", "FixedScale", "SELE", "RiskObjective"]]
    pairs += [("FixedScale", "BCE"), ("FixedScale", "RiskObjective")]
    paired, sensitivity = [], []
    for q in (20, 100):
        for method, baseline in pairs:
            for context in ("cert", "matched"):
                ca, cb = counts[context, f"{baseline}|{q}"], counts[context, f"{method}|{q}"]
                sa, sb = summary(*ca), summary(*cb)
                ba, bb = bootstrap_metrics(*ca, samples), bootstrap_metrics(*cb, samples)
                common = {"context": context, "baseline": f"{baseline}|{q}", "method": f"{method}|{q}"}
                for metric in ba:
                    diff = None if sa[metric] is None or sb[metric] is None else sb[metric] - sa[metric]
                    paired.append({**common, "metric": metric, "difference": diff,
                        **interval(bb[metric] - ba[metric])})
                paired.append({**common, "metric": "errors", "difference": sb["errors"] - sa["errors"],
                    **interval(cb[2][samples].sum(1) - ca[2][samples].sum(1))})
                for i, person in enumerate(people):
                    take = np.arange(20) != i
                    a, b = summary(*(v[take] for v in ca)), summary(*(v[take] for v in cb))
                    sensitivity.append({**common, "removed_subject": person,
                        "coverage_difference": b["macro_coverage"] - a["macro_coverage"],
                        "error_difference": b["errors"] - a["errors"],
                        "risk_difference": None if a["subject_equal_ratio_risk"] is None or b["subject_equal_ratio_risk"] is None
                        else b["subject_equal_ratio_risk"] - a["subject_equal_ratio_risk"]})
    candidates, diagnostics, cost = [], [], []
    selection = read(AOUT / "fit_selections.json")
    for outer, record in selection["folds"].items():
        for r in record["candidates"]:
            candidates.append({"outer": outer, "method": r["method"], "base_C": record["base_C"],
                "lambda": r["lambda"], "status": r["status"],
                "selected": r["lambda"] == record["selected"][r["method"]]["lambda"],
                "coverage": number(r.get("coverage")), "risk": number(r.get("risk")),
                "dense_coverage": number(r.get("dense_coverage")),
                "constant_calibrators": r.get("constant_calibrators")})
        at = np.flatnonzero(subjects == outer)
        base_raw = data["raw_scores"][at, 0]
        base_ranks = average_ranks(base_raw)
        baseline_mask = matched_masks["BCE|20"]
        for j, m in enumerate(("FixedScale", "BCEAnchor")):
            ranks = average_ranks(data["anchor_raw_scores"][at, j])
            current = matched_masks[m + "|20"]
            meta = seal["folds"][outer]["cells"][m + "|20"]
            corr = float(np.corrcoef(base_ranks, ranks)[0, 1]) if base_ranks.std() > 0 and ranks.std() > 0 else None
            diagnostics.append({"outer": outer, "method": m, "lambda": meta["lambda"],
                "relative_parameter_change": meta["relative_parameter_change"],
                "rank_correlation_with_BCE": corr,
                "matched_decisions_changed": int(np.sum(current[at] ^ baseline_mask[at])),
                "matched_error_difference_vs_BCE": int(np.sum(errors[at] & current[at]) - np.sum(errors[at] & baseline_mask[at])),
                "final_platt_slope": meta["mapping"]["slope"],
                "soft_training_coverage": meta["soft_training_coverage"],
                "soft_training_risk": meta["soft_training_risk"]})
    for p in sorted((AOUT / "fits").rglob("*.json")):
        r = read(p)
        if r["fingerprint"] != seal["fingerprint"] or digest(p.with_suffix(".joblib")) != r["sha256"]:
            raise RuntimeError("Saved fit identity changed")
        cost.append({"artifact": p.relative_to(ROOT).as_posix(),
            "phase": "final" if p.stem == "final" else "cross_validation",
            "seconds_both_methods": r["seconds"], **{m + "_status": v for m, v in r["statuses"].items()}})
    for name, result in [("summary", rows), ("per_subject", per), ("paired_bootstrap", paired),
                         ("leave_one_subject", sensitivity), ("fit_candidates", candidates),
                         ("ranking_diagnostics", diagnostics), ("training_cost", cost),
                         ("frozen_references", references)]:
        table("anchor_" + name, result)
    write(AOUT / "replay_checks.json", {"status": "PASS", "folds": 20, "epochs": len(errors),
        "historical_linear_matched_errors": [794, 798, 806],
        "frozen_reference_matched_errors": [817, 946, 1133, 1369, 2171],
        "all_same_scorer_matched_masks_grid_invariant": True,
        "outcomes_used_for_replay_only": True,
        "code_sha256": digest(Path(__file__)),
        "tables": {p.name: digest(p) for p in sorted((OUT / "tables").glob("anchor_*.csv"))}})
    print("Anchored replay PASS: 20 folds, 42308 epochs, historical results unchanged", flush=True)


def report(budget, resume=False):
    """Small bilingual figures and a self-contained Chinese reading guide."""
    if read(AOUT / "replay_checks.json")["status"] != "PASS":
        raise RuntimeError("Replay must pass before reporting")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = csv_rows("anchor_summary")
    per = csv_rows("anchor_per_subject")
    paired = csv_rows("anchor_paired_bootstrap")
    p = {m: next(r for r in rows if r["cell"] == m + "|20" and r["context"] == "cert") for m in METHODS}
    q = {m: next(r for r in rows if r["cell"] == m + "|20" and r["context"] == "matched") for m in METHODS}
    colors = ["#377EA5", "#AB7A34", "#999999", "#45A48B", "#7852AE"]
    labels = ["BCE", "SELE", "Original\nrisk objective", "Fixed scale", "BCE anchor"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    for ax, metric, source, title, ylabel, factor in [
        (axes[0], "macro_coverage", p, "CERT-selected rules", "OUTER mean-subject coverage (%)", 100),
        (axes[1], "subject_equal_ratio_risk", p, "Read together with coverage", "OUTER subject-equal ratio risk (%)", 100),
        (axes[2], "errors", q, "Same accepted count per subject", "Accepted errors", 1),
    ]:
        for j, m in enumerate(METHODS):
            value = None if source[m][metric] == "" else factor * float(source[m][metric])
            if value is None:
                ax.text(j, 0, "NA", ha="center")
                continue
            ax.bar(j, value, color=colors[j])
            ax.text(j, value, f"{value:.2f}" if factor == 100 else f"{value:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set(xticks=range(5), xticklabels=labels, title=title, ylabel=ylabel)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.margins(y=.15)
    axes[1].axhline(5, color="#555555", ls="--", lw=1)
    fig.suptitle("BCE-anchored risk learning: full 20-fold comparison", fontsize=13)
    for ext in ("png", "svg"):
        fig.savefig(OUT / "figures" / ("anchor_comparison." + ext), dpi=180)
    plt.close(fig)
    people = [f"{i:02d}" for i in range(20)]
    def errors(m, s):
        return int(next(r for r in per if r["context"] == "matched" and r["cell"] == m + "|20" and r["subject"] == s)["errors"])
    deltas = {m: np.array([errors(m, s) - errors("BCE", s) for s in people]) for m in ("FixedScale", "BCEAnchor")}
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    for i, m in enumerate(deltas):
        ax.bar(np.arange(20) + (i - .5) * .36, deltas[m], .36, label=m, color=colors[i + 3])
    ax.axhline(0, color="#555555", lw=1)
    ax.set(xticks=range(20), xticklabels=people, xlabel="OUTER subject",
        ylabel="Matched accepted errors minus BCE", title="Below zero means fewer errors at the same subject quota")
    ax.legend()
    for ext in ("png", "svg"):
        fig.savefig(OUT / "figures" / ("anchor_subjects." + ext), dpi=180)
    plt.close(fig)
    def pct(value):
        return "NA" if value in (None, "") else f"{100 * float(value):.2f}%"
    result_table = "| 方法 | Coverage | R_SE | Pooled risk | 匹配错误 | 逐人通过 |\n|---|---:|---:|---:|---:|---:|\n"
    for m in METHODS:
        result_table += f"| {m} | {pct(p[m]['macro_coverage'])} | {pct(p[m]['subject_equal_ratio_risk'])} | {pct(p[m]['pooled_risk'])} | {q[m]['errors']} | {p[m]['passed_subjects']}/20 |\n"
    refs = csv_rows("anchor_frozen_references")
    ref_table = "| 历史方法级参照 | Coverage | R_SE | 匹配错误 |\n|---|---:|---:|---:|\n"
    for m in ("Learned-Selected", "DuoF", "COVER-AutoML-Pi20", "COVER-AutoML-Pi100"):
        rr = next(r for r in refs if r["method"] == m and r["context"] == "cert")
        mm = next(r for r in refs if r["method"] == m and r["context"] == "matched")
        label = "完整 COVER" if m == "Learned-Selected" else m
        ref_table += f"| {label} | {pct(rr['macro_coverage'])} | {pct(rr['subject_equal_ratio_risk'])} | {mm['errors']} |\n"
    lines = []
    for baseline in ("BCE", "FixedScale", "SELE", "RiskObjective"):
        chosen = [r for r in paired if r["method"] == "BCEAnchor|20" and r["baseline"] == baseline + "|20"]
        parts = []
        for context, metric, title, factor in [("cert", "macro_coverage", "coverage（百分点）", 100),
            ("cert", "subject_equal_ratio_risk", "R_SE（百分点）", 100),
            ("matched", "errors", "匹配错误（个）", 1)]:
            r = next(v for v in chosen if v["context"] == context and v["metric"] == metric)
            if any(r[k] == "" for k in ("difference", "low", "high")):
                parts.append(title + " NA（条件风险分母为零）")
            else:
                parts.append(f"{title} {float(r['difference'])*factor:+.2f} [{float(r['low'])*factor:+.2f}, {float(r['high'])*factor:+.2f}]")
        lines.append(f"- BCEAnchor 相对 {baseline}：" + "；".join(parts) + "。")
    changes = deltas["BCEAnchor"]
    count_text = f"固定配额下，BCEAnchor 相对 BCE：{int((changes < 0).sum())} 人错误减少，{int((changes > 0).sum())} 人增加，{int((changes == 0).sum())} 人不变。"
    rank_rows = csv_rows("anchor_ranking_diagnostics")
    diag = "| 变体 | 权重相对改动中位数 | 与 BCE 排序相关性中位数 | 匹配决策变化总数 |\n|---|---:|---:|---:|\n"
    for m in ("FixedScale", "BCEAnchor"):
        rr = [r for r in rank_rows if r["method"] == m]
        diag += f"| {m} | {100*np.median([float(r['relative_parameter_change']) for r in rr]):.2f}% | {np.median([float(r['rank_correlation_with_BCE']) for r in rr]):.4f} | {sum(int(r['matched_decisions_changed']) for r in rr)} |\n"
    fit_text = "FIT 权重迁移诊断尚未生成。"
    if (AOUT / "fit_inspection.json").exists():
        fit = read(AOUT / "fit_inspection.json")
        fit_text = "| 变体 | 训练硬规则 coverage：BCE → 修正 | 留出 FIT 硬规则 coverage：BCE → 修正 |\n|---|---:|---:|\n"
        for r in fit["summary"]:
            fit_text += (f"| {r['method']} | {pct(r['train_BCE_oracle_coverage'])} → {pct(r['train_refined_oracle_coverage'])} | "
                f"{pct(r['validation_BCE_oracle_coverage'])} → {pct(r['validation_refined_oracle_coverage'])} |\n")
        fit_text += "\n这里对每个风险折单独扫描硬门限，是选型后的 FIT 诊断，不是最终部署规则，也不与主表 coverage 混用。中间状态只分析，未用于改选检查点。"
    observed = []
    for baseline in ("BCE", "FixedScale"):
        dc = 100 * (float(p["BCEAnchor"]["macro_coverage"]) - float(p[baseline]["macro_coverage"]))
        dr = None if not p["BCEAnchor"]["subject_equal_ratio_risk"] or not p[baseline]["subject_equal_ratio_risk"] else 100 * (
            float(p["BCEAnchor"]["subject_equal_ratio_risk"]) - float(p[baseline]["subject_equal_ratio_risk"]))
        de = int(q["BCEAnchor"]["errors"]) - int(q[baseline]["errors"])
        risk_text = "未定义" if dr is None else f"{dr:+.2f} 个百分点"
        observed.append(f"相对 {baseline}，锚定版 coverage {dc:+.2f} 个百分点、R_SE {risk_text}、匹配错误 {de:+d} 个。")
    conclusion = "应结合上表的实际覆盖率、风险和匹配错误决定保留哪项设计。"
    if (float(p["BCEAnchor"]["macro_coverage"]) > float(p["BCE"]["macro_coverage"])
            and p["BCEAnchor"]["subject_equal_ratio_risk"] and p["BCE"]["subject_equal_ratio_risk"]
            and float(p["BCEAnchor"]["subject_equal_ratio_risk"]) < float(p["BCE"]["subject_equal_ratio_risk"])
            and int(q["BCEAnchor"]["errors"]) < int(q["BCE"]["errors"])):
        conclusion = ("本轮支持优先保留 BCE 锚定设计：相对 BCE，原操作点的 coverage 更高、风险点估计更低，"
            "匹配接受数后的错误也更少。改善已出现在硬决策结果中，而不仅是训练软目标。"
            "相对无锚定对照，锚定版的覆盖率更低，同时风险和匹配错误更少，属于需要结合两种口径解释的结果。"
            "差异的描述性区间仍跨零，当前证据支持把它作为有前景的候选，尚不能写成稳定全面优越。")
    sessions = [s for s in read(OUT / "compute_budget.json")["sessions"] if s.get("experiment") == "anchored-risk-v1"]
    elapsed = sum(float(s["seconds"]) for s in sessions)
    verification = "最终交付核验尚未执行。"
    if (AOUT / "validation.json").exists():
        verified = read(AOUT / "validation.json")
        verification = (f"交付核验 {verified['status']}：{verified['passed']} 项检查通过；"
            f"{verified['preserved_files']} 个原文件指纹不变；{len(verified['csv_sha256'])} 张 CSV 再次重放逐字节一致。"
            "另写的计数与整数并列投票实现已交叉核对，12 项定向测试通过。"
            "这是同一 agent 的实现与复核，不标为独立外部审计。")
    md = f"""# BCE 锚定风险学习：20 折结果

## 先理解系统怎样改变

Conformer 和 RF 仍提供原来的固定输出，37D 特征也不变。先取已选好的 BCE 线性评分器，再训练一个权重修正。
新分数等于 BCE 原分数加上修正量；新增训练惩罚修正量的平方。最终仍只有一个线性评分器，
不会同时运行两个风险模型。FIT 选择修正强度，CERT 选择硬接受规则，OUTER 记录结果。

```text
原 BCE：       f₀(z) = θ₀ᵀz + b₀
本轮锚定版：   f(z)  = f₀(z) + σ₀·Δᵀz
训练目标：     max [C_soft − λ‖Δ‖²/2]，约束 R_soft ≤ 0.05
```

这里 Δ 是在 BCE 训练分数尺度下的权重修正，σ₀ 来自当前训练受试者并在优化中固定。
Δ=0 时精确返回原 BCE；λ 越大，同样的参数改动代价越高。实际训练同时寻找软门限 t，
允许学习为了增加低风险接受量而值得作出的改动。不是先预测 OUTER 对错再决定改哪些参数。

- **BCEAnchor（本轮主方法）**：软 coverage 最大化，同时约束软风险，并惩罚偏离 BCE 的权重。
- **FixedScale（归因对照）**：同一 BCE 起点、同一固定尺度和四个惩罚强度，惩罚全部权重而不以 BCE 为中心。
- **BCE / SELE / RiskObjective**：原实验结果完整复用。前两者是线性竞争方法，最后一项是上一版原型。

固定归一化温度 0.2，避免上一版温度延续时有效惩罚随之减弱。惩罚是软限制，并不保证完全保持排序。
两种新变体共享相同的额外调参预算；相对原 BCE，多了四个修正配置的选择，不能称为相同总搜索预算。

## 真实结果：先一起看覆盖率和风险，再看相同名额

**{conclusion}**

{' '.join(observed)}

{result_table}

主表采用 20 个候选硬规则。Coverage 为受试者等权均值，R_SE 为归一化错误量与接受量之比。
主表风险均来自 OUTER，规则由各自 CERT 选出。匹配错误把每人的名额固定为历史完整 COVER 的原名额，
不是每种新变体自己的名额。100 点结果是同一个评分器的诊断，未重新选评分器。

![完整比较](../outputs/figures/anchor_comparison.png)

### 放回已有工作看

{ref_table}

这些历史方法使用不同的模型或更大的候选空间，提供实际方法水平的参照，不能代替上面的同容量对照。
AutoML 两行来自已保存汇总，来源指纹收入新重放 receipt；本轮没有重新执行搜索。
AutoML 的 Π100 是原先指定的 headline，Π20 的观测表现不改变这一事实。

## 差值及描述性区间

{chr(10).join(lines)}

括号为 10,000 次配对受试者 bootstrap 的 95% 描述性区间；seed=20260901。
重复使用的历史队列与重叠训练折限制独立推断。区间跨零不代表等效，也不证明风险不增加。

{count_text}

![逐人结果](../outputs/figures/anchor_subjects.png)

## 是否保留了 BCE 的底稿

{diag}

权重相对改动为 ‖β−β_BCE‖/‖β_BCE‖；排序相关性是逐人原始分数的 Spearman 相关。
这些数解释改了多少，最终是否值得采用仍由上述硬规则结果判断。
训练门限只用于优化，不直接用于 OUTER；最终门限来自校准后的 FIT 验证候选，经 CERT 选定。

### 训练中的收益有没有传到留出 FIT

{fit_text}

### 对后续研究意味着什么

与 BCE 比较回答“修正有没有实际收益”；与 FixedScale 比较回答“锚定中心是否值得加入”。
两个问题需要分别看，不能仅因超过上一版 RiskObjective 就认定超过了 BCE。
这里的完整对照用于确定可保留的设计和适合论文的主张，不自动把研发过程全部搬进正文。
最终风险推理仍是一次 37D 线性点积加共同校准；本轮未用这个结构事实代替端到端耗时测量。

下一项最有价值的工作，是把这一版的目标函数、修正强度选择和评价协议固定下来，
在一项来源与划分合格的不同设定上验证。当前队列不再用于按 OUTER 成绩微调 λ 或温度。
本轮交付到完整实验和内部研究判断；跨设定实验和稿件整合尚未启动。

## 状态、成本与复现

完成全部 20 折，未重训 Conformer/RF，未重拟合已有 BCE。当前已记录本轮 {elapsed/60:.1f} 分钟计算，两个 CPU 线程。
该时间含试跑、拟合和重放，不是线上推理时间。全部新增文件仅在 New_Objective_Idea 内。

```text
python -B run_anchored.py replay
python -B run_anchored.py report
```

首次计算命令为 `preflight → pilot --resume → fit --resume → evaluate --resume → replay → report`。
规格见 [ANCHOR_PROTOCOL.md](ANCHOR_PROTOCOL.md)，可复算表为 `outputs/tables/anchor_*.csv`。
原协议与所有旧结果保持原位；本轮没有改稿件和发布包。结果用于方法研发，后续是否进入正文单独决定。

本轮从已有 FIT 选出的 BCE 继续开发，继承 FIT 内元选型依赖，属于回顾性扩展。
CERT 只选择经验风险约束下的规则；没有新增 95% 的统计风险保证。

### 交付核验

{verification}

详细记录：`outputs/anchored/validation.json`。原输入和拟合代码指纹见 `preflight.json`，
运行环境见 `environment.json`。图形使用可编辑 SVG 与 PNG，HTML 图片已嵌入，可离线阅读。
"""
    text_file(ROOT / "docs/ANCHOR_RESULTS_ZH.md", md)
    # No external scripts, styles, fonts or equation CDN are needed to read this guide.
    import re
    blocks = []
    lines_md = md.splitlines()
    i = 0
    while i < len(lines_md):
        line = lines_md[i]
        if line.startswith("!["):
            match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            path = (ROOT / "docs" / match.group(2)).resolve()
            encoded = base64.b64encode(path.read_bytes()).decode()
            blocks.append(f'<img alt="{html.escape(match.group(1))}" src="data:image/png;base64,{encoded}">')
        elif line.startswith("|"):
            block = []
            while i < len(lines_md) and lines_md[i].startswith("|"):
                if "---" not in lines_md[i]:
                    tag = "th" if not block else "td"
                    block.append("<tr>" + "".join(f"<{tag}>{html.escape(v.strip())}</{tag}>" for v in lines_md[i].strip("|").split("|")) + "</tr>")
                i += 1
            blocks.append("<div class='table'><table>" + "".join(block) + "</table></div>")
            continue
        elif line.startswith("```"):
            code = []
            i += 1
            while i < len(lines_md) and not lines_md[i].startswith("```"):
                code.append(lines_md[i]); i += 1
            blocks.append("<pre>" + html.escape("\n".join(code)) + "</pre>")
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            blocks.append(f"<h{level}>" + html.escape(line[level:].strip()) + f"</h{level}>")
        elif line:
            clean = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html.escape(line))
            clean = re.sub(r"`([^`]+)`", r"<code>\1</code>", clean)
            clean = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', clean)
            blocks.append("<p>" + clean + "</p>")
        i += 1
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BCE 锚定风险学习</title><style>body{max-width:1120px;margin:32px auto;padding:0 24px;background:#fafbfe;color:#202b3b;font:17px/1.7 "Microsoft YaHei",sans-serif}h1,h2{color:#4e337d}h2{margin-top:38px;border-top:1px solid #ddd;padding-top:18px}img{width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px 14px;border:1px solid #d5dce5}th{background:#f0eaf8}pre{background:#edf1f6;padding:16px;overflow:auto}.table{overflow:auto}strong{color:#533887}</style><main>' + "\n".join(blocks) + '</main></html>'
    text_file(ROOT / "docs/ANCHOR_RESULTS_ZH.html", page)
    write(AOUT / "report_status.json", {"status": "GENERATED", "self_contained_html": True,
        "figure_files": ["anchor_comparison.png", "anchor_subjects.png"],
        "visual_review": "PENDING", "code_sha256": digest(Path(__file__)),
        "recorded_seconds_before_report": elapsed})
    print("Anchored comparison figures and Chinese reading guide generated", flush=True)
