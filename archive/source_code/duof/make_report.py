"""Create the verified comparison figures and a manuscript-ready supplement."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results = json.loads((ROOT / "evaluation/results.json").read_text(encoding="utf-8"))
verification = json.loads((ROOT / "logs/verification.json").read_text(encoding="utf-8"))
assert verification["status"] == "PASS"
rows = {r["method"]: r for r in results["aggregate"]}
figures = ROOT / "figures"
figures.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.3), layout="constrained")
colors = {"Learned": "#596775", "DuoF": "#2676AF"}
for ax, metric, ylabel in zip(axes, ["macro_coverage", "pooled_risk", "passed_subjects"],
                            ["Macro coverage (%)", "Pooled accepted risk (%)", "Passing subjects (out of 20)"]):
    for offset, family in [(-.18, "Learned"), (.18, "DuoF")]:
        values = [rows[f"{family}-{alpha}"][metric] * (1 if metric == "passed_subjects" else 100)
                  for alpha in ("0.05", "0.03")]
        bars = ax.bar(np.arange(2)+offset, values, width=.34, color=colors[family], label=family)
        ax.bar_label(bars, labels=[f"{x:.0f}" if metric == "passed_subjects" else f"{x:.2f}" for x in values],
                     padding=3, fontsize=8)
    ax.set_xticks([0, 1], ["5%", "3%"])
    ax.set_xlabel("CERT risk ceiling")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, {"macro_coverage": 55, "pooled_risk": 5.5, "passed_subjects": 20.5}[metric])
    if metric == "passed_subjects":
        ax.axhline(16, color="#9C4141", linestyle="--", linewidth=.8)
        ax.set_yticks([0, 5, 10, 15, 20])
axes[0].legend(loc="upper right", fontsize=8, frameon=False)
fig.suptitle("SC20: identical primary predictions, two fixed CERT ceilings", fontsize=11)
for suffix in ("pdf", "png"):
    fig.savefig(figures / f"duof_matched_comparison.{suffix}", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.1), layout="constrained")
for ax, alpha in zip(axes, ["0.05", "0.03"]):
    exchange = {r["region"]: r for r in results["exchanges"]
                if r["method_a"] == f"DuoF-{alpha}" and r["method_b"] == f"Learned-{alpha}"}
    parts = ["shared", "added", "removed"]
    bars = ax.barh(["Shared", "Added by DuoF", "Removed by DuoF"],
                   [100*exchange[p]["pooled_risk"] for p in parts],
                   color=["#9AA3AA", "#2676AF", "#596775"])
    ax.bar_label(bars, labels=[f"{100*exchange[p]['pooled_risk']:.2f}% (n={exchange[p]['accepted']:,})" for p in parts],
                 padding=3, fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.invert_yaxis()
    ax.set_xlim(0, 23)
    ax.axvline(5, color="#9C4141", linestyle="--", linewidth=.8)
    ax.set_xlabel("Pooled error rate within\nacceptance region (%)")
    ax.set_ylabel("Acceptance region relative to Learned")
    ax.set_title(f"CERT ceiling {100*float(alpha):.0f}%")
for suffix in ("pdf", "png"):
    fig.savefig(figures / f"duof_acceptance_exchange.{suffix}", dpi=180)
plt.close(fig)

project = ROOT / "overleaf_project"
project.mkdir(exist_ok=True)
(project / "figures").mkdir(exist_ok=True)
(project / "tables").mkdir(exist_ok=True)
source = ROOT.parent / "sc20_mvp_complete_20260912/overleaf_project"
for name in ["iclr2027_conference.sty", "iclr2027_conference.bst", "natbib.sty", "fancyhdr.sty"]:
    shutil.copy2(source / name, project / name)
for p in figures.glob("*.pdf"):
    shutil.copy2(p, project / "figures" / p.name)
for p in (ROOT / "tables").glob("*.csv"):
    shutil.copy2(p, project / "tables" / p.name)

display = {
    "Baseline-selected-0.05": "Baseline-Selected", "MSP-0.05": "MSP",
    "Primary-only-0.05": "Primary-only", "Learned-0.05": "Learned-Selected",
    "DuoF-0.05": "DuoF", "Primary-only-0.03": "Primary-only",
    "Learned-0.03": "Learned (B0.03)", "DuoF-0.03": "DuoF",
}
table = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Scorer & CERT (\%) & Cov. (\%) & $R_{\rm SE}$ (\%) & $R_{\rm pool}$ (\%) & Pass & Zero\\", r"\midrule"]
for name, row in rows.items():
    table.append(f"{display[name]} & {100*float(name.rsplit('-',1)[1]):.0f} & {100*row['macro_coverage']:.2f} & "
                 f"{100*row['subject_equal_ratio_risk']:.2f} & {100*row['pooled_risk']:.2f} & "
                 f"{row['passed_subjects']}/20 & {len(row['zero_subjects'].split(';')) if row['zero_subjects'] else 0}"+r"\\")
table += [r"\bottomrule", r"\end{tabular}"]
(project / "comparison_table.tex").write_text("\n".join(table)+"\n", encoding="utf-8")

tex = r"""\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{iclr2027_conference,times}
\usepackage{amsmath,booktabs,graphicx,hyperref,url}
\iclrfinalcopy
\title{A Fixed-Component DuoF Comparison on SC20}
\author{Empirical baseline supplement}
\begin{document}
\maketitle
\lhead{SC20 DuoF strong-baseline comparison -- 12 September 2026}
\begin{abstract}
We add a fixed-primary DuoF uncertainty score to the frozen 20-subject SC20 pipeline,
reusing component predictions and fitting two weights separately in each outer fold.
At a 3\% CERT ceiling, the original learned scorer has higher macro coverage
(38.03\% versus 35.00\%), lower pooled accepted risk (2.30\% versus 2.49\%), and
more individual passes (17/20 versus 15/20). At 5\%, DuoF passes one additional
subject but has lower coverage and higher risk. These aggregate comparisons do not
extend uniformly to stages: DuoF improves the observed true-REM accepted risk and
correct automatic acceptance at 3\%. This is a retrospective comparison on the
same cohort, not independent confirmation or statistical certification.
\end{abstract}

\section{System and Frozen Comparison}
The data, sample identities and primary labels are unchanged: 20 subjects, 39 records,
and 42,308 eligible EEG epochs. Each outer fold retains its 15 FIT subjects, four CERT
subjects and one outer subject. The Conformer and random forest are not retrained.
For primary logits $z_P$ and calibrated RF class probabilities $p_F$, we fit
\begin{equation}
 p_{\mathrm{Duo}}=\operatorname{softmax}\{w_Pz_P+w_F\log(\operatorname{clip}(p_F,10^{-10},1))\},
 \quad r(x)=1-p_{\mathrm{Duo}}(\hat y_P\mid x).
\end{equation}
The selected label remains $\hat y_P=\arg\max p_P$, even when the fused distribution
has a different argmax. The frozen Stage 4 fitting function optimizes subject-equal
multiclass NLL on the current fold's FIT labels, with two weights bounded in $[0,20]$,
initial value $(1,1)$, and L-BFGS-B. No Stage 4 fitted weights are imported.
The component FIT predictions are out of fold. Duo weights are fitted on all 15 FIT
subjects, and their in-sample scores define the FIT quantiles, matching the Stage 4
adaptation; there is no additional cross-fitting of the two weights.

Twenty exact subject-equal FIT quantiles define candidates. CERT maximizes macro
coverage subject to subject-equal ratio risk at most 5\% or 3\%, with at least 50
acceptances for every CERT subject. Both ceilings share the same weights, scores and
identity-based tie rule. An infeasible selector returns zero acceptance. All 40 outer
masks were locked before the evaluation entry read outer outcomes. The primary
comparisons are DuoF versus the original learned scorer at each matched CERT ceiling.
Primary-only, MSP and the frozen FIT-selected baseline are retained as additional controls.

\section{Matched-Ceiling Results}
\begin{table}[t]
\centering\resizebox{\linewidth}{!}{\input{comparison_table}}
\caption{All prespecified scorers. Coverage is the 20-subject macro mean.
$R_{\rm SE}=\sum_s K_s/n_s\,/\,\sum_s A_s/n_s$;
$R_{\rm pool}=\sum_sK_s/\sum_sA_s$. A pass requires $A_s\ge50$ and
$K_s/A_s\le5\%$. Zero acceptance has undefined risk and is a failed pass.
The outer criterion is always 5\%, regardless of the CERT ceiling.}
\label{tab:all}
\end{table}

At a 5\% CERT ceiling, DuoF accepts 18,598 epochs with 810 errors, compared with
19,892 and 817 for Learned-Selected. DuoF loses 3.44 percentage points (pp) of macro
coverage, increases pooled risk by 0.25pp and changes passes from 12/20 to 13/20.
Subjects 02, 12 and 16 gain a pass, while 03 and 08 lose one. Subject 03 becomes
zero acceptance, and 07 remains zero. The extra pass therefore does not constitute
a joint improvement over the original scorer.

At a 3\% ceiling, DuoF accepts 14,909 epochs with 371 errors; the original learned
scorer accepts 15,992 with 368 errors. The original scorer has a 3.04pp macro-coverage
advantage, lower pooled and subject-equal risks, and 17 rather than 15 passes.
DuoF gains subject 16 but loses subjects 03, 09 and 17. Its zero-acceptance subjects
are 03, 07 and 09. This is an observed aggregate advantage for the learned scorer at
the tested operating points, not population-level dominance or a risk-matched frontier.

\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{figures/duof_matched_comparison.pdf}
\caption{The two prespecified comparisons, using the same underlying primary predictions.
The dashed pass-count line is the inherited 16/20 descriptive target. No threshold was
selected on outer labels.}
\end{figure}

DuoF remains a useful strong baseline. At 5\%, it exceeds MSP macro coverage by
12.17pp and passes 13 rather than 11 subjects, with slightly higher pooled risk.
At 3\%, it exceeds Primary-only coverage by 8.30pp, with lower pooled risk and one
additional pass. Both DuoF variants remain below the inherited 16/20 individual-pass
target. The original Stage 2 NO-GO is unchanged.

\section{Acceptance Turnover and Stage Support}
At 5\%, DuoF adds 1,564 predictions relative to Learned-Selected with 258 errors
(16.50\%) and removes 2,858 with 265 errors (9.27\%). At 3\%, it adds 1,781 with
151 errors (8.48\%) and removes 2,864 with 148 errors (5.17\%). The corresponding
symmetric differences are 4,422 and 4,645 epochs. These comparisons again show that
small changes in error totals conceal substantial changes in accepted membership.

Lowering DuoF's own CERT ceiling from 5\% to 3\% is a strict contraction: it removes
3,689 acceptances with 439 errors (11.90\%) and adds none. Subjects 01, 08 and 18 gain
a pass, while 09 loses one through zero acceptance. Macro coverage falls by 8.63pp
and passes increase from 13/20 to 15/20.

\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{figures/duof_acceptance_exchange.pdf}
\caption{Retrospective regions for DuoF relative to the learned scorer at the same CERT
ceiling. Each bar reports the error rate and size of that region; the dashed 5\% line
is a reference, not a separate region-level policy constraint.}
\end{figure}

The aggregate result has an important stage-specific qualification. At a 3\% ceiling,
DuoF accepts 728 true-REM epochs, makes 53 errors (7.28\%), and correctly autoaccepts
675 of 7,717 eligible REM epochs. The learned scorer accepts 663, makes 106 errors
(15.99\%), and correctly autoaccepts 557. Thus the learned scorer's aggregate advantage
does not imply better REM handling. Under DuoF, 91 true-N1 epochs are accepted with
86 errors (94.51\%); five of 2,804 eligible N1 epochs are correctly autoaccepted,
compared with zero under the learned scorer. This small recovery does not establish
reliable N1 automation. True-stage labels are diagnostic only.

\section{Verification, Cost and Interpretation}
Six interface and selector tests passed. Alternate arithmetic rebuilt all 400 candidate
rows and 40 selected masks; all 160 subject--method metric rows were recomputed,
and 120 reused baseline rows matched their historical tables. The maximum score
difference was below $8\times10^{-15}$. A checkpoint resume check preserved the
hashes and modification times of all 80 fold files. These are checks by the same
agent, not independent scientific validation.

Twenty two-parameter fits took 2.44 seconds in total. Summed fold work, including
loading, scoring, selection and saving, took 24.80 seconds, excluding process startup
and implementation work. The CPU thread limit was two; no GPU training was performed.
The maximum process memory observed at fold completion was 520.2 MiB, which is a
sampled measurement rather than a continuously measured peak.

This addition strengthens the empirical comparison: a competitive auxiliary baseline
does not consistently replace the original scorer on SC20, and aggregate improvements
remain distinct from individual and stage-specific benefits. Historical cohort exposure,
overlapping LOSO training sets and limited CERT support remain. We do not infer
independent transfer, statistical significance, clinical utility or a general superiority
claim. The completed MVP and the historical results are preserved; this supplement
records the newly authorized comparison.
\end{document}
"""
(project / "main.tex").write_text(tex, encoding="utf-8")

report = """# SC20 DuoF 强基线比较：已完成

同一套 Stage 2 预测缓存下，DuoF 没有带来一致的整体提升。CERT 上限为 3% 时，
原学习型方法的覆盖率、接受风险和逐人通过数都更好；上限为 5% 时，DuoF 多通过一人，
但覆盖率下降、风险略升。阶段条件分析则显示 DuoF 的 REM 表现更好，不能把总体结果
扩张为“每个受试者、每个睡眠阶段都更好”。

## 系统怎样运行

复用每折的 Conformer logits 和已校准 RF 概率，在该折 15 个 FIT 受试者上拟合两个
非负融合权重。得到的不确定性分数用于接受/拒绝，输出标签始终是原 Conformer 的标签。
每折的四个 CERT 受试者分别按 5% 和 3% 风险上限选择阈值；20 折的 40 组决策全部锁定后，
评价原来的 20 人、39 条记录、42,308 个 epochs。

“Learned”是原 Stage 2 的 Learned-Selected（每折都选中异构信息），不是另训的新模型；
3% 的对应项是已有 Stage 3 B0.03-Heterogeneous。Primary-only、MSP 和原 Baseline-Selected
均保留为对照。两个 DuoF 版本共享权重，差异仅来自 CERT 上限。

## 两个主要比较

| CERT 上限 | 方法 | 宏平均覆盖率 | 合并接受风险 | 通过人数 | 零接受人数 |
|---|---|---:|---:|---:|---:|
| 5% | 原 Learned | 47.07% | 4.11% | 12/20 | 1 |
| 5% | DuoF | 43.62% | 4.36% | 13/20 | 2 |
| 3% | 原 Learned | 38.03% | 2.30% | 17/20 | 1 |
| 3% | DuoF | 35.00% | 2.49% | 15/20 | 3 |

通过标准始终是该受试者接受量至少 50、接受错误率不超过 5%；分母始终为全部 20 人。
这里的 3%/5% 是 CERT 选阈值的限制，不是改变外层评价标准。完整八种方法在 tables/aggregate.csv。

5% 比较中，DuoF 让 02、12、16 从失败变为通过，同时让 03、08 失去通过资格；03 变成
零接受，07 仍然零接受。因此是三人改善、两人退步，净增一人。

3% 比较中，DuoF 改善了 16，却让 03、09、17 从通过变成失败；03 和 09 变成零接受，
07 仍然零接受。原方法在这两个风险指标和宏覆盖率、通过数上具有观察到的总体优势。
这是当前工作点的比较，没有用外层标签匹配风险或重新挑阈值，不是普遍支配或显著性结论。

## 它仍然是一个有价值的强基线

DuoF-5% 相比 MSP 多 12.17 个百分点覆盖率，通过人数从 11 变为 13，但风险略高。
DuoF-3% 相比 Primary-only 多 8.30 个百分点覆盖率，风险略低，通过人数从 14 变为 15。
两版 DuoF 都没有达到原来的 16/20 逐人目标，历史 Stage 2 NO-GO 不变。

在 3% 上限下，DuoF 正确自动接受 675/7,717 个真实 REM epochs，原方法为 557/7,717；
真实 REM 接受集错误率为 53/728=7.28%，原方法为 106/663=15.99%。DuoF 的 REM 表现更好。
真实 N1 中，DuoF 正确自动接受 5/2,804 个，原方法为 0；但 DuoF 接受的 91 个真实 N1
中有 86 个错误，错误率仍达 94.51%，不能称为解决了 N1。

## 接受集合解释

DuoF-5% 相比 Learned-5% 新增 1,564 个接受（258 错误，16.50%），移除 2,858 个
接受（265 错误，9.27%），共交换 4,422 个决策。DuoF-3% 相比 Learned-3% 新增
1,781 个（151 错误，8.48%），移除 2,864 个（148 错误，5.17%），共交换 4,645 个。

把 DuoF 自己从 5% 收紧到 3%，只移除 3,689 个接受、其中 439 个错误（11.90%），
不新增接受。01、08、18 变为通过，09 因零接受失去通过，净通过数增加两人。
这继续支持 MVP 的主线：接受集合组成、个体变化和阶段表现不能只靠汇总 coverage 判断。

## 实际成本与验证

20 次权重拟合合计 2.44 秒；20 折加载、评分、选阈值、保存合计 24.80 秒。
它们不包含代码开发、进程启动与报告整理时间。使用 CPU、线程上限为 2，新增 GPU 训练为 0。
每折结束时观测到的进程内存最大为 520.2 MiB，这不是持续采样峰值。

六项相关测试通过。另一套算术实现复核了 400 行候选策略、40 个最终掩码、160 行逐人结果，
并核对 120 行既有基线结果。分数复算最大误差 7.11e-15；92 个源文件哈希未变。
断点复用实测通过，80 个折产物的哈希和修改时间均未变。
上述验证均由同一 agent 完成，不称为独立复现。

当前结论限于已历史暴露的 SC20；没有新增受试者、重训上游模型、访问外部 TEST 或调参搜索。
原 MVP 文件保持不变，新结果作为论文强基线补充交付。
"""
(ROOT / "RESULTS_zh.md").write_text(report, encoding="utf-8")

readme = """# SC20 DuoF baseline supplement

This directory contains the actual completed, predeclared 20-fold comparison.
Read RESULTS_zh.md, then SC20_DuoF_Comparison.pdf. The Overleaf ZIP is the editable
supplement and includes its figures and all CSV tables; main.tex is its main document.
Use XeLaTeX. Local compilation uses the existing Tectonic XeTeX/LaTeX engine.

The original completed MVP was preserved. The supplement can be incorporated into a
later manuscript revision. It is not a remote submission or standalone raw-data release.

## Reproduce in the existing Windows workspace

Use the existing eeg-conformer Python environment. No packages, datasets or GPU training
are needed beyond the existing local inputs. Commands from this output directory:

```powershell
$duoPython = 'python'
& $duoPython -X utf8 -B -m unittest discover -s '.\\code' -p 'test_contract.py' -v
& $duoPython -X utf8 -B '.\\code\\run_comparison.py' prepare
& $duoPython -X utf8 -B '.\\code\\run_comparison.py' fit --fold all
& $duoPython -X utf8 -B '.\\code\\evaluate.py'
& $duoPython -X utf8 -B '.\\code\\verify_results.py'
& $duoPython -X utf8 -B '.\\code\\make_report.py'
```

Valid fold outputs are verified and skipped, not refitted. prepare refuses protocol
drift; the frozen execution-code hashes are enforced. Editing a scientific rule requires
a separate experiment scope, not deletion of the current checkpoint.

code/core.py records the source roots. config/input_manifest.json records the source
fingerprints. config/protocol.json records the actual authorized protocol and the
distinction between component-OOF inputs and in-sample two-weight quantile scores.
config/all_decisions_locked.json predates evaluation/receipt.json. Label-free means
no current-fold outer labels: permitted FIT and CERT labels are used as documented.
Historical source files named sealed_outer_results contain the already-evaluated SC20
records; no ST/CAP TEST files are accessed by this extension.

tables/ contains full aggregate, subject, transition, exchange, stage-support and timing
results. logs/verification.json contains the alternate arithmetic replay and resume check.
The analysis bundle is an internal labeled diagnostic artifact; public release is a
separate step. No claim of independent scientific validation is made.
"""
(ROOT / "README.md").write_text(readme, encoding="utf-8")
(project / "README.md").write_text("# SC20 DuoF empirical supplement\n\nUpload this ZIP to Overleaf; main.tex is the main document and XeLaTeX the compiler.\nIt includes two figures and eight CSV tables. Local compilation and visual QA are recorded\nin BUILD_QA.json. Online Overleaf compilation was not executed.\n\nThis is the new strong-baseline supplement; the previously completed MVP was preserved.\nCode, frozen inputs and full receipts are in the companion local experiment directory.\n", encoding="utf-8")
print(json.dumps({"figures": str(figures), "supplement": str(project / "main.tex"),
                  "report": str(ROOT / "RESULTS_zh.md")}, ensure_ascii=False))
