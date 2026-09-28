"""Chinese teaching/report delivery from completed, saved FIT tables only."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import ROOT, read, write, digest, text_file, utc
from .bceanchor_upgrade_core import UOUT

NAMES = {
    "A0": "BCEAnchor / historical selector", "A1": "BCEAnchor / exact-q selector",
    "A2": "BCEAnchor / fixed lambda=1", "S0": "SELE / historical selector",
    "S1": "SELE / exact-q selector", "BCE": "BCE", "FixedScale-1": "FixedScale / lambda=1",
}
SHORT = {
    "BCEAnchor-average": "Anchor average", "BCEAnchor-full-1": "Anchor full",
    "BCEAnchor-equalnorm": "Equal-norm control", "FixedScale-average": "FixedScale average",
    "FixedScale-full-1": "FixedScale full", "SELE-full-S1": "SELE full (S1)",
    "SELE-average": "SELE average", "SELE-historical-S0": "SELE historical",
    "Nested-BCE": "BCE", "Nested-FixedScale-1": "FixedScale",
    "Nested-BCEAnchor-1": "Anchor, lambda=1",
    "Nested-BCEAnchor-selected": "Anchor, inner-selected lambda", "Nested-SELE": "SELE",
}


def tables():
    paths = UOUT / "tables"
    return {p.stem: pd.read_csv(p, dtype={"subject": str, "context": str,
             "excluded_subject": str}) for p in sorted(paths.glob("*.csv"))}


def percent(value, digits=3):
    return "NA" if not np.isfinite(value) else f"{100*value:.{digits}f}%"


def pp(value):
    return "NA" if not np.isfinite(value) else f"{100*value:+.4f} pp"


def markdown(frame):
    """No optional tabulate dependency; values are already presentation strings."""
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
    return "\n".join(rows)


def performance_table(frames, branch, evaluation, cells=None):
    data = frames["summary"]
    data = data[(data.branch == branch) & (data.evaluation == evaluation)].copy()
    if cells:
        data = data.set_index("cell").reindex(cells).reset_index()
    return pd.DataFrame({
        "方法": data.cell,
        "Coverage": data.coverage.map(percent),
        "R_SE": data.R_SE.map(percent),
        "重复记录接受数": data.repeated_accepted.map(lambda v: f"{int(v):,}" if pd.notna(v) else "NA"),
        "重复记录错误数": data.repeated_errors.map(lambda v: f"{int(v):,}" if pd.notna(v) else "NA"),
        "零接受记录 / 300": data.zero_accept_records.map(lambda v: int(v) if pd.notna(v) else "NA"),
    })


def contrasts(frames, branch, evaluation, metric):
    data = frames["paired_differences"]
    data = data[(data.branch == branch) & (data.evaluation == evaluation)
                & (data.metric == metric)].copy()
    return pd.DataFrame({
        "候选": data.cell, "参照": data.reference,
        "差值（候选−参照）": data.difference.map(pp),
        "描述性95%区间": [f"[{pp(a)}, {pp(b)}]" for a, b in zip(data.ci_low, data.ci_high)],
        "负 / 零 / 正的患者数": [f"{a} / {b} / {c}" for a, b, c in
            zip(data.negative_subjects, data.tied_subjects, data.positive_subjects)],
    })


def figure_file(name):
    folder = UOUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (name + ".png")


def setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 130, "savefig.dpi": 170})
    return plt


def workflow_figure():
    plt = setup_plot()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12.3, 6.0))
    ax.set(xlim=(0, 12), ylim=(0, 6)); ax.axis("off")
    fig.suptitle("Three routes to improve a BCE-based risk scorer", fontsize=19, weight="bold", y=.98)
    entries = [
        (4.55, "A  SELECTOR", "Same saved models\nSame calibration\nRead exact picked q", "Change which model is selected",
         "15-subject FIT selection\nRetrospective development", "#1565A9"),
        (2.65, "B2  AVERAGING", "12-subject shared BCE\n4 corrections on 9 subjects\nAverage linear parameters", "Change the correction direction",
         "3 held-out subjects\nQuota / oracle diagnosis", "#7852A7"),
        (.75, "B1  INNER TUNING", "Inside 12: four 9/3 splits\nChoose base C and lambda\nLock calibration and rule", "Test the full tuning procedure",
         "3 held-out subjects\nApply the locked rule", "#008C80"),
    ]
    for y, tag, left, question, right, color in entries:
        ax.text(.15, y+.5, tag, color=color, fontsize=12, weight="bold")
        for x, w, text in [(2.3, 3.2, left), (7.0, 4.5, right)]:
            box = FancyBboxPatch((x, y-.45), w, 1.25, boxstyle="round,pad=.14",
                                 edgecolor=color, facecolor=color+"10", linewidth=1.5)
            ax.add_patch(box); ax.text(x+w/2, y+.16, text, ha="center", va="center", fontsize=11)
        ax.annotate("", xy=(6.85, y+.15), xytext=(5.65, y+.15),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
        ax.text(7.05, y+1.04, question, fontsize=10.5, color=color, weight="bold", va="bottom")
    fig.text(.5, .02, "All results use historical FIT patients. No new CERT or OUTER evaluation.",
             ha="center", fontsize=11, color="#555555")
    fig.subplots_adjust(left=.03, right=.99, bottom=.08, top=.91)
    path = figure_file("three_routes"); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def route_figure(frames, branch):
    plt = setup_plot()
    settings = {
        "selection": (["A0", "A1", "A2", "S0", "S1"], "selected_fit_rule",
                      "A: retrospective configuration selection"),
        "averaging": (["BCEAnchor-full-1", "BCEAnchor-average", "BCEAnchor-equalnorm",
                       "FixedScale-full-1", "FixedScale-average", "SELE-full-S1", "SELE-average"],
                      "oracle", "B2: averaged corrections on held-out FIT subjects"),
        "nested": (["Nested-BCE", "Nested-FixedScale-1", "Nested-BCEAnchor-1",
                    "Nested-BCEAnchor-selected", "Nested-SELE"], "locked_rule",
                   "B1: training-only choices, locked rules on held-out FIT subjects"),
    }
    cells, operating, title = settings[branch]
    summary = frames["summary"]
    operating_data = summary[(summary.branch == branch) & (summary.evaluation == operating)].set_index("cell").reindex(cells)
    matched = summary[(summary.branch == branch) & (summary.evaluation == "quota_0.5")].set_index("cell").reindex(cells)
    colors = ["#1565A9" if ("SELE" in c or c.startswith("S")) else
              "#D58217" if "FixedScale" in c else "#7852A7" for c in cells]
    labels = [SHORT.get(c, c) for c in cells]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.12]})
    ax = axes[0]
    for i, (cell, label, color) in enumerate(zip(cells, labels, colors)):
        row = operating_data.loc[cell]
        if not (np.isfinite(row.R_SE) and np.isfinite(row.coverage)):
            continue
        ax.scatter(100*row.R_SE, 100*row.coverage, color=color, s=66, zorder=3)
        offset = ((30, 6), (30, -22), (70, -7))[i] if branch == "averaging" and i < 3 else (6, 5)
        ax.annotate(str(i+1), (100*row.R_SE, 100*row.coverage), xytext=offset,
                    textcoords="offset points", fontsize=10, weight="bold",
                    arrowprops={"arrowstyle": "-", "color": "#777777", "lw": .8}
                    if branch == "averaging" and i < 3 else None)
    ax.axvline(5, color="#888888", ls="--", lw=1)
    ax.set(xlabel="Accepted risk R_SE (%)", ylabel="Mean-subject coverage (%)",
           title={"selected_fit_rule": "Each selected FIT rule", "oracle": "Hard-threshold oracle (diagnostic)",
                  "locked_rule": "Locked-rule transfer (primary)"}[operating])
    ax.grid(alpha=.15)
    ax = axes[1]
    values = [100*matched.loc[c, "R_SE"] for c in cells]
    bars = ax.barh(np.arange(len(cells)), values, color=colors, height=.6)
    ax.set(yticks=np.arange(len(cells)), yticklabels=[f"{i+1}. {s}" for i, s in enumerate(labels)],
           xlabel="Accepted risk R_SE (%) — lower is better", title="Same 50% quota for each subject")
    maximum = np.nanmax(values)
    ax.invert_yaxis(); ax.set_xlim(0, maximum*1.24)
    for b, v, cell in zip(bars, values, cells):
        ax.text((v if np.isfinite(v) else 0)+.015*maximum, b.get_y()+b.get_height()/2,
                f"{v:.3f}%" if np.isfinite(v) else "NA", va="center", fontsize=10)
    ax.grid(axis="x", alpha=.15)
    fig.suptitle(title, fontsize=15, weight="bold")
    footer = {"selection": "Left: selected on these FIT validation scores. Right: identical subject quotas. Both are development evidence.",
              "averaging": "Different panels answer different questions. Oracle coverage is not a deployable operating point.",
              "nested": "Left: rules fixed within the parent 12 subjects. Right: a separate matched-quota ranking diagnostic."}[branch]
    fig.text(.5, .02, footer,
             ha="center", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.075, right=.985, top=.82, bottom=.15, wspace=.8)
    path = figure_file(branch); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def subject_figure(frames):
    plt = setup_plot()
    people = frames["subject_means"]
    settings = [("selection", "A1", "S1", "quota_0.5", "A1 versus S1 — matched 50% quota"),
                ("averaging", "BCEAnchor-average", "SELE-average", "quota_0.5", "B2: Anchor versus SELE average — matched quota"),
                ("nested", "Nested-BCEAnchor-selected", "Nested-SELE", "quota_0.5", "B1: inner-selected Anchor versus SELE — matched quota"),
                ("nested", "Nested-BCEAnchor-selected", "Nested-SELE", "locked_rule", "B1: the same comparison — each locked rule")]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10.4), sharex=True)
    for ax, (branch, cell, ref, evaluation, title) in zip(axes, settings):
        frame = people[(people.branch == branch) & (people.evaluation == evaluation)]
        a = frame[frame.cell == cell].set_index("subject").sort_index()
        b = frame[frame.cell == ref].set_index("subject").sort_index()
        assert a.index.equals(b.index)
        values = 100*(a.e_bar/a.c_bar - b.e_bar/b.c_bar)
        ax.bar(np.arange(len(values)), values, color=["#7852A7" if x <= 0 else "#C65C53" for x in values])
        ax.axhline(0, color="#555555", lw=1); ax.set_title(title, loc="left")
        ax.set_ylabel("Risk difference\n(pp)"); ax.grid(axis="y", alpha=.15)
    axes[-1].set(xticks=np.arange(len(a)), xticklabels=a.index, xlabel="Subject (all 20 retained)")
    fig.suptitle("Does the difference repeat across subjects?", fontsize=17, weight="bold")
    fig.text(.5, .015, "Negative: Anchor lower accepted risk. Positive: branch-matched SELE lower risk. Panel 4 uses different acceptance counts.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .95))
    path = figure_file("subject_differences"); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def comparison_sentence(frames, branch, cell, reference, evaluation, metric):
    rows = frames["paired_differences"]
    row = rows[(rows.branch == branch) & (rows.cell == cell) & (rows.reference == reference)
               & (rows.evaluation == evaluation) & (rows.metric == metric)].iloc[0]
    return (f"{cell} 对 {reference}，{evaluation} 的 {metric} 差为 **{pp(row.difference)}**，"
            f"描述性区间 [{pp(row.ci_low)}, {pp(row.ci_high)}]；"
            f"逐人负／零／正差：{int(row.negative_subjects)}／{int(row.tied_subjects)}／{int(row.positive_subjects)}"
            f"（未定义：{int(row.undefined_subjects)}人）。")


def research_readout(frames):
    """Interpret this completed matrix; all numerical statements read saved tables."""
    summary, paired, deleted = (frames[name] for name in
                               ("summary", "paired_differences", "delete_one_subject"))
    def result(branch, cell, evaluation="quota_0.5"):
        return summary[(summary.branch == branch) & (summary.cell == cell)
                       & (summary.evaluation == evaluation)].iloc[0]
    def difference(branch, cell, reference, evaluation, metric):
        return paired[(paired.branch == branch) & (paired.cell == cell)
                      & (paired.reference == reference) & (paired.evaluation == evaluation)
                      & (paired.metric == metric)].iloc[0]
    selected, fixed, sele, scale = [result("nested", cell, "locked_rule") for cell in
        ("Nested-BCEAnchor-selected", "Nested-BCEAnchor-1", "Nested-SELE", "Nested-FixedScale-1")]
    d = difference("nested", "Nested-BCEAnchor-selected", "Nested-SELE", "locked_rule", "R_SE")
    c = difference("nested", "Nested-BCEAnchor-selected", "Nested-SELE", "locked_rule", "coverage")
    q = difference("nested", "Nested-BCEAnchor-selected", "Nested-SELE", "quota_0.5", "R_SE")
    sens = deleted[(deleted.branch == "nested") & (deleted.cell == "Nested-BCEAnchor-selected")
                   & (deleted.reference == "Nested-SELE") & (deleted.evaluation == "locked_rule")]
    dr = sens[sens.metric == "R_SE"].difference
    dc = sens[sens.metric == "coverage"].difference
    a0, a1 = result("selection", "A0"), result("selection", "A1")
    full, average, equal = [result("averaging", cell) for cell in
                           ("BCEAnchor-full-1", "BCEAnchor-average", "BCEAnchor-equalnorm")]
    lambda_q = difference("nested", "Nested-BCEAnchor-selected", "Nested-BCEAnchor-1", "quota_0.5", "R_SE")
    return "\n\n".join([
        "### 实际结论与下一轮候选",
        "**推荐把 B1 的完整 Nested-BCEAnchor-selected 流程固定为下一轮确认候选，保留固定 λ=1、FixedScale 和原损失 SELE。** "
        "这是看完本轮完整开发比较后的建议；没有据此重跑或拼接新方法。",
        f"**最强的正面证据来自固定规则迁移。** Nested-BCEAnchor-selected 的 coverage / R_SE 为 "
        f"{percent(selected.coverage)} / {percent(selected.R_SE)}，Nested-SELE 为 "
        f"{percent(sele.coverage)} / {percent(sele.R_SE)}。风险差 {pp(d.difference)}，描述性区间 "
        f"[{pp(d.ci_low)}, {pp(d.ci_high)}]；{int(d.negative_subjects)}/20 人的条件风险较低。"
        f"coverage 差 {pp(c.difference)}，区间 [{pp(c.ci_low)}, {pp(c.ci_high)}]。"
        f"全部20次删一后，风险差仍在 [{pp(dr.min())}, {pp(dr.max())}]，coverage 差仍在 "
        f"[{pp(dc.min())}, {pp(dc.max())}]。这支持保留该候选进入确认。"
        "删一只拿掉一人，自举会重复抽取患者；两种扰动不同，所以删一方向一致不等于自举区间必然不跨零。",
        f"**排序改善还没有同样稳定。** 每人固定50%接受量后，风险差为 {pp(q.difference)}，"
        f"区间 [{pp(q.ci_low)}, {pp(q.ci_high)}]；逐人更少错 / 更多错 / 相同为 "
        f"{int(q.negative_subjects)} / {int(q.positive_subjects)} / {int(q.tied_subjects)}。"
        "固定规则的优势不能直接改写成所有患者的排序都更好。",
        f"**A 修复了选择损失；B2 没有增加锚定收益。** A0→A1 的50%配额风险为 "
        f"{percent(a0.R_SE)}→{percent(a1.R_SE)}。BCEAnchor-full→average 为 "
        f"{percent(full.R_SE)}→{percent(average.R_SE)}，等幅控制为 {percent(equal.R_SE)}。"
        f"本配方成员分数相关系数平均为 {frames['averaging_geometry'].mean_member_score_correlation.mean():.4f}。"
        "本轮不推荐为同一四成员平均方案继续增加训练；这不排除其他平均方法在不同条件下有效。",
        f"**内部选 λ 的作用主要是 coverage–risk 取舍。** 相对同流程固定 λ=1，它增加 "
        f"{pp(selected.coverage-fixed.coverage)} coverage，同时增加 {pp(selected.R_SE-fixed.R_SE)} 风险；"
        f"匹配50%数量的风险差为 {pp(lambda_q.difference)}。因此，不能把整个 nested 流程对 SELE 的差异全部归功于选 λ。",
        f"**FixedScale 必须继续保留。** Nested-FixedScale-1 的 coverage / R_SE 为 "
        f"{percent(scale.coverage)} / {percent(scale.R_SE)}，聚合点估计优于本轮两种 nested Anchor。"
        "这说明参数锚定并非所有收益的唯一解释。下一轮应同时确认强度选择、参数锚定与固定尺度三个作用，"
        "沿用本轮已有配方，不新增结果驱动的网格。",
        "下一步先核验未参与开发的患者资格并冻结确认协议，再评价上述完整流程。"
        "本轮结果是给定冻结组件特征的风险层开发证据；没有新的 CERT／OUTER 或独立人群结果。",
    ])


def report_text(frames):
    checks = read(UOUT / "replay_checks.json")
    ledger = read(UOUT / "compute_budget.json")
    numerical = read(UOUT / "solver_attempt_checks.json") if (UOUT / "solver_attempt_checks.json").exists() else None
    selected = frames["selected_configs"]
    a = selected[(selected.branch == "selection") & (selected.cell == "A0")].set_index("context")
    b = selected[(selected.branch == "selection") & (selected.cell == "A1")].set_index("context")
    changed = int((a.parameter != b.parameter).sum())
    nested = selected[selected.branch == "nested"]
    costs = frames["fit_costs"].groupby(["branch", "kind"]).agg(
        count=("seconds", "size"), seconds=("seconds", "sum"), max_seconds=("seconds", "max")).reset_index()
    sentences = [
        comparison_sentence(frames, "selection", "A1", "A0", "quota_0.5", "R_SE"),
        comparison_sentence(frames, "averaging", "BCEAnchor-average", "SELE-average", "quota_0.5", "R_SE"),
        comparison_sentence(frames, "nested", "Nested-BCEAnchor-selected", "Nested-SELE", "locked_rule", "coverage"),
        comparison_sentence(frames, "nested", "Nested-BCEAnchor-selected", "Nested-SELE", "locked_rule", "R_SE"),
        comparison_sentence(frames, "nested", "Nested-BCEAnchor-selected", "Nested-SELE", "quota_0.5", "R_SE"),
    ]
    text = ["# BCEAnchor 三路线升级：完整 FIT 开发结果", "",
        "## 结论先读", "",
        "**20 个上下文、100 个父风险折的预定三路线已完成。以下报告实际差异，不把本轮最优者重新命名成预先指定主方法。**", "",
        *["- " + s for s in sentences], "",
        "本轮只使用历史 FIT 开发患者。SELE 保留原损失；没有新的 CERT／OUTER 评价，也没有重新训练 EEG 组件。", "",
        research_readout(frames), "",
        "## 先理解输入怎样成为输出", "",
        "冻结组件输出 → 37D 风险输入 → BCE 底稿／原损失 SELE → 三条开发路线。A 对同一批已有模型换选择器；"
        "B2 对共同 BCE 底稿上的四个修正取平均；B1 把底稿、强度、校准和规则的选择都放进父层12人，最后只对外面3人应用固定流程。", "",
        "![三条路线](../outputs/bceanchor_upgrade/figures/three_routes.png)", "",
        "## A：配置选择是否少损失已有收益？", "",
        f"A1 相对 A0 更换了 **{changed}/20** 个配置。候选内仍按实际 coverage 选规则；"
        "候选间读取该规则的精确 q，随后比较风险。没有用四舍五入 q，没有同时修改候选内规则。", "",
        markdown(performance_table(frames, "selection", "selected_fit_rule")), "",
        "下面固定每人50%接受数，比较被选出的评分器排序。相同分支内模型原分数、校准和候选都相同，差值来自选型。", "",
        markdown(performance_table(frames, "selection", "quota_0.5")), "",
        markdown(contrasts(frames, "selection", "quota_0.5", "R_SE")), "",
        "## B2：四个修正的平均是否更可迁移？", "",
        "共同的父层 Scaler、BCE 参数和分数尺度保持固定；四个9人修正平均。"
        "等幅控制只沿完整12人修正方向缩放，且允许放大；SELE 接受相同四成员平均操作。", "",
        markdown(performance_table(frames, "averaging", "quota_0.5")), "",
        markdown(contrasts(frames, "averaging", "quota_0.5", "R_SE")), "",
        "硬阈值 oracle 是在每个留出3人折查看标签后扫描得到的排序潜力，不能作为部署规则或认证结果。", "",
        markdown(performance_table(frames, "averaging", "oracle")), "",
        "## B1：在训练人群内选强度，固定规则能否迁移？", "",
        "该表的3名受试者既不选底稿，也不选修正强度、最终 Platt 或阈值。"
        "父12人的内层交叉校准分数产生候选阈值；最终 Platt 拟合对应 raw OOF；固定规则直接应用到外3人。", "",
        markdown(performance_table(frames, "nested", "locked_rule")), "",
        markdown(contrasts(frames, "nested", "locked_rule", "coverage")), "",
        markdown(contrasts(frames, "nested", "locked_rule", "R_SE")), "",
        "匹配数量是辅助问题：在每人同样50%的接受量下比较排序。它不保证任一方法此时低于5%风险。", "",
        markdown(performance_table(frames, "nested", "quota_0.5")), "",
        markdown(contrasts(frames, "nested", "quota_0.5", "R_SE")), "",
        "内层所选 λ 次数：" + "; ".join(f"{p:g}: {int(n)}" for p, n in nested.parameter.value_counts().sort_index().items()) + "。", "",
        "## 增量应归因给什么？", "",
        "| 比较 | 可归因的变化 | 不能替代的证据 |", "|---|---|---|",
        "| A1 − A0 | 配置间排序键 | 不代表训练学出新排序 |",
        "| Anchor-average − Anchor-full | 四个修正的平均 | 同时查看等幅控制、SELE-average |",
        "| Anchor-average − equalnorm | 平均方向相对单方向缩放 | 分数改动均方不是估计方差 |",
        "| Nested-selected − Nested-lambda1 | 同一内层流程内选择强度 | 不能把新旧整条流程差都归给 λ |",
        "| 各分支对本分支 SELE | 实际竞争表现 | 历史 FIT 开发结果不是独立患者确认 |", "",
        "## 患者汇总、删一和区间", "",
        "每人先在其实际进入 FIT 的上下文记录上分别平均 a/n 与 k/n，再对20人等权计算 coverage 以及错误量/接受量。"
        "每个上下文15人、20个上下文合计300条患者记录，但各人出现次数不必相同；完整性按保存的角色清单逐人核对。"
        "不平均各次条件风险。表中错误数是**重复记录计数**，不是独立 epoch 数；每个方法/指标有300条患者记录。", "",
        "主结果保留全部20人。`delete_one_subject.csv` 给出全部20个删一结果；"
        "`selector_delete_one.csv` 另从全部固定候选、固定阈值重新选规则和配置，不重新校准，因此不称严格 nested CV。"
        "按患者组重抽样10000次、seed 20260901；保持重叠训练/组件复用背景下的描述性解释。", "",
        "![逐人差异](../outputs/bceanchor_upgrade/figures/subject_differences.png)", "",
        "## 成本与失败记录", "", markdown(costs), "",
        f"本次报告生成时，新账本累计 **{ledger['used_seconds']/3600:.3f} 小时**，其中试跑/拟合"
        f" **{ledger['fit_seconds']/3600:.3f} 小时**。历史账本 {ledger['historical_seconds']/3600:.3f} 小时保持不变。"
        "报告与 Notebook 执行的剩余时间由入口退出时记入新账本；精确终值见 compute_budget.json。", "",
        "技术试跑曾发现规则构造器默认15人，不能用于B1的12人。已改成显式传入人数，并用12/15人回归测试验证。"
        "修复前尚未产生B1选规产物；既有新实验缓存经逐文件哈希核对后迁移元数据，模型字节未变。"
        "详见 compatibility_migration.json。报告阶段还修正了把每人平均15次误作每人恰好15次的完整性检查，"
        "现逐人核对实际上下文清单；模型、分数和逐条计数未因此变化。技术失败记录保留在计算账本和 reporting_corrections.json 中。", "",
        (f"全部新增模型最终状态为OK；Anchor/FixedScale求解器共记录 {numerical['anchor_solver_attempts']} 次尝试，"
         f"其中 {len(numerical['unsuccessful_or_infeasible_attempts'])} 次未成功或未达可行条件，"
         f"执行了 {numerical['anchor_solver_retries']} 次预定重试。逐项记录见 solver_attempt_checks.json；"
         "没有通过丢弃失败成员改变平均权重。" if numerical else "数值尝试的独立保存检查尚未生成，详见模型内 diagnostics。"), "",
        f"重放检查：{checks['status']}；历史资产 **{checks['protected_unchanged']}** 个哈希未变；"
        f"连续重放CSV逐字节相同：**{checks['repeat_replay_byte_identical']}**。", "",
        "## 下一轮确认需要什么？", "",
        "本轮开发比较可提出候选，但确认前必须冻结完整配方、主比较、风险与coverage口径和运行预算。"
        "本轮20人都已参与方法开发，RF互换仍是同一批人；它们不能再当新患者确认集。"
        "本轮定向读取了现有身份/用途清单：SC78共78人均被原记录标为已暴露（含原20人之外的58人）；"
        "ST 22人、CAP 16人只是历史资格待确认的候选，并未据此建立新的合格确认集。"
        "清单指纹、候选身份和文件存在性见 confirmation_source_inventory.json；本轮未打开候选EEG或标签，"
        "也没有重做一次全项目历史审计。下一轮应先逐一核对 patient/record 身份、来源哈希、历史用途及排除交集，再确定患者数量；"
        "本轮不下载、不训练、不评价新的数据。", "",
        "## 可复现入口", "", "```text",
        "python run_bceanchor_upgrade.py --mode replay",
        "python run_bceanchor_upgrade.py --mode report",
        "python run_bceanchor_upgrade.py --mode fit --resume",
        "```", "",
        "replay 只读已保存分数与决策；report 只读结果表。fit 需要原开发缓存，默认入口不是 fit。"
        "本实验全部新增文件在 New_Objective_Idea 内，训练指纹与报告指纹分开。", "",
        "本机 fit／replay／validate 使用 `anaconda3/envs/eeg-conformer/python.exe`；"
        "report 使用已有的 `anaconda3/python.exe`，其中包含 nbformat、nbclient、nbconvert 和 ipykernel。"
        "报告内核显式绑定后者；两种 Python 的用途分开，未修改拟合环境。", "",
        "完整计数、规则、模型来源、运行状态见 outputs/bceanchor_upgrade；教学入口见 notebooks/08_selection_and_correction_stability.html。", "",
    ]
    return "\n".join(text)


def notebook():
    import nbformat as nb
    n = nb.v4.new_notebook()
    n.metadata.update(language_info={"name": "python", "version": sys.version.split()[0]},
        kernelspec={"name": "python3", "display_name": "Python 3", "language": "python"},
        offline_static=True, study="bceanchor-three-route-v1")
    md = lambda value: n.cells.append(nb.v4.new_markdown_cell(value))
    code = lambda value: n.cells.append(nb.v4.new_code_cell(value.strip()))
    md("""# 08｜从选型稳定化到修正稳定性

**先回答：这次究竟改变了什么？** 我们没有换主分类器、没有把 SELE 改成自己的方法。
这次完整比较三种办法：**更合理地点名同一批模型；平均几个修正；只在训练患者内部选修正强度。**

阅读顺序：1 看信息流 → 2 看三份成绩单 → 3–5 看各路线 → 6 看是否人人受益 → 7 看研究结论。
本页只读已完成实验的表格，执行本 notebook **不会启动任何拟合**。""")
    code("""
from pathlib import Path
import sys
import pandas as pd
from IPython.display import display, Image, Markdown
root = Path.cwd().resolve()
if root.name == 'notebooks':
    root = root.parent
assert (root / 'bceanchor_upgrade_protocol.json').exists(), '请从 New_Objective_Idea 或 notebooks 执行'
sys.dont_write_bytecode = True
sys.path[:0] = [str(root / 'src'), str(root.parent / 'src'), str(root.parent / 'AutoML_Claude')]
from new_objective.bceanchor_upgrade_teaching import (
    tables, performance_table, contrasts, workflow_figure, route_figure, subject_figure, research_readout)
frames = tables()
primary = frames['summary'][frames['summary'].cell != 'BCEAnchor-equalnorm']
assert primary.status.eq('COMPLETE').all(), '不完整比较不能当成全量结果'
pd.set_option('display.max_colwidth', 90)
pd.set_option('display.max_rows', 100)
""")
    md("""## 1. 冻结组件不变，三条路线解决三个问题

每段 EEG 已经有冻结 Conformer 的预测。37D 特征来自主模型/辅助模型的输出；标签 e 表示主模型这次是否出错。
BCE 学出一条风险排序；BCEAnchor 围绕这条排序做面向风险目标的有限修正。
**风险分数越低，越倾向接受；最终输出标签始终不变。**""")
    code("display(Image(filename=str(workflow_figure())))")
    md("""## 2. 先认清三份不同的成绩单

| 成绩单 | 怎么做 | 能回答什么 |
|---|---|---|
| A 的 FIT 选中规则 | 在同批 FIT 验证分数上选配置和阈值 | 新选择器有没有少损失现有收益？仍是开发诊断 |
| 每人接受 50% / oracle | 固定每人名额；或看标签后扫描最大可行硬阈值 | 排序好不好？oracle 不是部署成绩 |
| B1 的固定规则 | 父12人选完一切，再对外3人应用 | 完整风险层选型流程能否迁移？ |

**R_SE = 平均错误量 / 平均接受量。** 同一患者多次出现，先对他的 a/n 和 k/n 分别取平均。
不要把各患者条件风险的平均当成 R_SE；不要把重复错误计数当成独立样本数。
50% 是用于排序比较的固定配额，不意味着此处风险必定≤5%。""")
    md("""## 3. A：不是重新训练，而是换一个选模型的规则

历史选择器先比精确实际 coverage。极小的并列抽签差可能压过较明显的风险差。
新选择器保留每个候选内部的选规方式，只在候选间比较：**选中规则的精确 q → 风险 → 固定并列优先级。**

| 编号 | 含义 |
|---|---|
| A0 / A1 / A2 | 原 Anchor 选择器 / 精确 q 选择器 / 固定 λ=1 |
| S0 / S1 | 原 SELE 选择器 / 同样的精确 q 选择器 |

q 来自真正被选中的规则，不能从实际 coverage 四舍五入，也不偷偷改成最大的可行 q。""")
    md("""**先看一个教学例子（不是实验结果）：** 两个候选都选中 q=0.50。
模型甲的实际 coverage=50.0001%、风险=4.9%；模型乙为50.0000%、风险=4.5%。
历史选择器会为0.0001个百分点的接受量优先选甲；精确q选择器认为二者名义配额相同，再按风险选乙。
**它没有重新训练乙，也没有证明乙在新患者上更好；它改变的是该选谁。**

下面开始才是真实的20上下文结果。""")
    code("display(performance_table(frames, 'selection', 'selected_fit_rule'))\ndisplay(Image(filename=str(route_figure(frames, 'selection'))))")
    md("左图是各自选中的 FIT 操作点；右图锁定每人一半名额。**只有右图能排除接受数量不同的影响；两图都仍是FIT开发结果。**")
    code("display(contrasts(frames, 'selection', 'quota_0.5', 'R_SE'))")
    md("""## 4. B2：平均四个修正，究竟是换方向，还是只缩小步子？

12人先给出同一套 Scaler、BCE 底稿和分数尺度。分成四组三人，每次留出一组、用9人学修正。
四个成员等权平均，最后对外面3人打分。

**β_avg = β₀ + (Δ₁ + Δ₂ + Δ₃ + Δ₄)/4。** 不平均阈值，不平均 Platt 概率，不挑表现最好的成员。

等幅控制沿完整12人修正 Δ_full 的方向，长度调到与平均修正一样。
若二者同样好，可能只需要少改一点；若平均更好，才值得继续研究方向的作用。
系数大于1时就是放大，不能强行截断成缩小。SELE 也做四成员平均，防止把通用平均的收益说成锚定独有。""")
    code("display(performance_table(frames, 'averaging', 'quota_0.5'))\ndisplay(Image(filename=str(route_figure(frames, 'averaging'))))\ndisplay(contrasts(frames, 'averaging', 'quota_0.5', 'R_SE'))")
    md("B2左图使用 oracle，回答排序潜力；它和下面 B1 左图的固定规则不能横向混排。参数平均是否有效，由这些实测对照回答，不能只靠方差比喻预判。")
    md("""## 5. B1：把“选哪个底稿、改多少、在哪里画线”都关在训练12人内

```text
父层12人 → 四个内层9/3折 → 原始OOF分数 → 交叉Platt → 选C、λ和规则
          → 完整12人最终风险模型 + raw OOF拟合最终Platt
          → 固定这套评分器、映射、阈值 → 外面的3人只评价
```

外3人的标签不参与配置、校准或阈值选择；训练求解器里的 t_dev 不直接拿来部署。
每个9人内层重新拟合自己的 Scaler/BCE；这与 B2 固定父层底稿的做法不同。

**主要归因：Nested-BCEAnchor-selected 对 Nested-BCEAnchor-1。** 两者共享整个嵌套流程，才可把差别归给内部选择λ。
**竞争比较：Nested-BCEAnchor-selected 对 Nested-SELE。** SELE 同样只能在父12人内选C。""")
    code("display(performance_table(frames, 'nested', 'locked_rule'))\ndisplay(Image(filename=str(route_figure(frames, 'nested'))))")
    md("先看固定规则 coverage 与风险一起怎样变化，再看右图匹配数量的排序；接受更多本身不能代表全面更好。")
    code("display(contrasts(frames, 'nested', 'locked_rule', 'coverage'))\ndisplay(contrasts(frames, 'nested', 'locked_rule', 'R_SE'))\ndisplay(contrasts(frames, 'nested', 'quota_0.5', 'R_SE'))")
    md("""## 6. 平均改善，是不是大多数患者也受益？

下面每个柱子是一名患者的重复记录先汇总后计算的条件风险差。前三行固定50%名额；第四行使用各自已固定的规则，接受数量会不同。
紫色负值表示 Anchor 的接受风险更低，红色正值表示该分支 SELE 更低。固定规则风险下降与相同名额下排序改善，是两个不同问题。
**20人全部保留，包括S11。** 删除每个人的敏感性结果都在CSV中；删一不是新的一次独立实验。""")
    code("display(Image(filename=str(subject_figure(frames))))")
    code("""
d = frames['delete_one_subject']
d = d[(d.evaluation == 'quota_0.5') & (d.metric == 'R_SE')]
display(d.groupby(['branch', 'cell', 'reference']).difference.agg(['min', 'max']).mul(100)
        .rename(columns={'min': '删一最小差(pp)', 'max': '删一最大差(pp)'}))
""")
    md("""## 7. 这轮能把什么带进下一次确认？

1. **A** 若改善：减少了选择损失；不能写成新的训练排序算法。
2. **B2** 若改善：先对照等幅控制和 SELE-average，判断是否有锚定特有的增量。
3. **B1** 若改善：说明完整风险层选择流程在留出患者上的开发表现更好。

三条路线都完整报告，不能把事后最好的几部分拼成新方法再继续在同一评价数据上搜索。
组件 OOF 及其祖先训练仍复用，20人也已参与历史开发；**这不是全新人群确认，不是95%风险认证。**
下一轮先冻结一个值得确认的完整配方，再核对未参与开发的患者身份、记录和历史用途。
Sleep-EDF-78这个名称、换RF作主模型，都不能自动等价于“新患者”。""")
    code("display(Markdown(research_readout(frames)))")
    code("""
display(frames['fit_costs'].groupby(['branch', 'kind']).seconds.agg(['count', 'sum', 'max'])
        .rename(columns={'sum': '累计拟合秒', 'max': '最长单任务秒'}))
""")
    md("""## 8. 文件在哪里？怎样重现？

- 数字：`outputs/bceanchor_upgrade/tables/`；完整报告：`docs/BCEANCHOR_UPGRADE_RESULTS.md`。
- 入口：`python run_bceanchor_upgrade.py --mode replay`，再 `--mode report`。
- 默认只 replay，只有显式 `--mode fit --resume` 才会拟合。
- 本页所有结果来自保存表格，图是这些表的视图；代码、缓存、日志和新账本全部在 New_Objective_Idea 内。
- 定向来源盘点见 `outputs/bceanchor_upgrade/confirmation_source_inventory.json`：SC78已被历史清单列为暴露；ST/CAP仅为资格待确认候选。本轮未打开候选EEG或标签。

参考：原 [SELE](https://proceedings.mlr.press/v97/franc19a.html)、
[配置选择偏差研究](https://www.jmlr.org/papers/v11/cawley10a.html)、
[DiWA](https://proceedings.neurips.cc/paper_files/paper/2022/file/46108d807b50ad4144eb353b5d0e8851-Paper-Conference.pdf)。
这些工作提供问题依据，不能替代本轮的效果验证。""")
    return n


def build_report_and_notebook(budget):
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from nbconvert.preprocessors import TagRemovePreprocessor
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    cache = UOUT / "cache"
    for key, rel in (("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter_runtime"),
                     ("JUPYTER_CONFIG_DIR", "jupyter_config"), ("JUPYTER_DATA_DIR", "jupyter_data"),
                     ("MPLCONFIGDIR", "matplotlib")):
        path = cache / rel; path.mkdir(parents=True, exist_ok=True); os.environ[key] = str(path)
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg", "PYDEVD_DISABLE_FILE_VALIDATION": "1",
                "TEMP": str(cache/"tmp"), "TMP": str(cache/"tmp")})
    kernel_dir = cache / "kernels" / "bceanchor-upgrade"
    write(kernel_dir / "kernel.json", {"argv": [sys.executable, "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "BCEAnchor upgrade (current Python)", "language": "python"})
    hashes = read(UOUT / "replay_checks.json")["csv_sha256"]
    frames = tables()
    path = ROOT / "notebooks/08_selection_and_correction_stability.ipynb"
    n = notebook()
    n.metadata.teaching_sources = {"outputs/bceanchor_upgrade/"+p: h for p, h in hashes.items()}
    manager = AsyncKernelManager(kernel_name="bceanchor-upgrade",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
    client = NotebookClient(n, km=manager, timeout=180,
        resources={"metadata": {"path": str(ROOT)}})
    executed = client.execute(env=env, cwd=str(ROOT), cleanup_kc=True)
    nbformat.validate(executed)
    assert not any(o.output_type == "error" for c in executed.cells if c.cell_type == "code" for o in c.outputs)
    assert all(digest(UOUT / p) == h for p, h in hashes.items())
    nbformat.write(executed, path)
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input_prompt = exporter.exclude_output_prompt = True
    exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={"hide-input"}), enabled=True)
    body, _ = exporter.from_notebook_node(executed)
    body = re.sub(r"<script\b[^>]*>.*?</script\s*>", "", body, flags=re.I | re.S)
    style = """<style>body{font-family:Arial,'Microsoft YaHei',sans-serif}
    .jp-Notebook{max-width:1220px;margin:auto}.jp-RenderedHTMLCommon table{font-size:14px}
    .jp-RenderedHTMLCommon p,.jp-RenderedHTMLCommon li{line-height:1.75}
    h1,h2{color:#203851}img{max-width:100%;height:auto}</style>"""
    body = body.replace("</head>", style+"</head>")
    text_file(path.with_suffix(".html"), body)
    text_file(ROOT / "docs/BCEANCHOR_UPGRADE_RESULTS.md", report_text(frames))
    budget.check()
    write(UOUT / "teaching_status.json", {"status": "NOTEBOOK_EXECUTED_HTML_EXPORTED",
        "time": utc(), "notebook": str(path.relative_to(ROOT)), "notebook_sha256": digest(path),
        "html_sha256": digest(path.with_suffix(".html")), "csv_sha256": hashes,
        "code_cells": sum(c.cell_type == "code" for c in n.cells), "fitting_performed": False,
        "render_python": sys.executable,
        "offline_static": True, "teaching_code_sha256": digest(__file__)})
    print(path.with_suffix(".html"), flush=True)
