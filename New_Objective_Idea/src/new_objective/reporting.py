"""Fit-free outcome replay and teaching artifacts. No sklearn/joblib/torch imports."""
from __future__ import annotations
import base64
import csv
import html
import io
import json
from pathlib import Path
import numpy as np
from .storage import ROOT,PROJECT,OUT,read,write,digest,arrays,text_file,target,specification,utc
from stage1_analysis.metrics import (identity_key,alignment,identity_ties,count_subjects,summary,ranked_orders,quota_mask,bootstrap_metrics,interval)

def table(name,rows):
    path=target(OUT/"tables"/(name+".csv"))
    columns=list(dict.fromkeys(k for r in rows for k in r))
    s=io.StringIO(newline="")
    writer=csv.DictWriter(s,fieldnames=columns,lineterminator="\n")
    writer.writeheader();writer.writerows(rows)
    text_file(path,s.getvalue())

def csv_rows(name):
    with (OUT/"tables"/(name+".csv")).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))

def require_seal():
    seal=read(OUT/"decisions_sealed.json")
    if seal["status"]!="ALL_OUTER_DECISIONS_FROZEN" or set(seal["folds"])!={f"{i:02d}" for i in range(20)}:
        raise RuntimeError("OUTER truth cannot be read before all 20 decisions are sealed")
    for outer,receipt in seal["folds"].items():
        if receipt["sha256"]!=digest(OUT/"decisions"/f"outer-{outer}.npz"):
            raise RuntimeError("Decision artifact fingerprint mismatch")
    return seal

def make_portable():
    seal=require_seal()
    from stage1_analysis.inputs import load
    frozen=load(PROJECT,compact=True)
    n=len(frozen.errors);scores=np.full((n,6),np.nan);masks=np.zeros((n,6),bool)
    raw=np.full((n,3),np.nan);warm=np.full((n,3),np.nan);visited=np.zeros(n,bool)
    cells=None
    for outer,receipt in seal["folds"].items():
        with np.load(OUT/"decisions"/f"outer-{outer}.npz",allow_pickle=False) as z:
            keys=[identity_key(i) for i in json.loads(z["identities_json"].item())]
            pos=alignment(keys,frozen.keys)
            if visited[pos].any() or not (frozen.subjects[pos]==outer).all():
                raise RuntimeError("OUTER overlap or role mismatch")
            np.testing.assert_array_equal(z["predictions"],frozen.predictions[pos])
            names=z["cells"].tolist()
            if cells is not None and cells!=names:raise RuntimeError("Cell order changed")
            cells=names;scores[pos]=z["scores"];masks[pos]=z["masks"];raw[pos]=z["raw_scores"]
            warm[pos]=z["warm_raw_scores"]
            visited[pos]=True
    if not visited.all() or not np.isfinite(scores).all():raise RuntimeError("Incomplete OUTER result")
    ref_names=["Learned-Selected","DuoF","Primary-only","MSP","Forward-KL"]
    ref_scores=np.column_stack([frozen.scores[m] for m in ref_names])
    ref_masks=np.column_stack([frozen.masks[m+"|0.05|empirical"] for m in ref_names])
    path=OUT/"replay_inputs.npz"
    arrays(path,cells=np.asarray(cells),scores=scores,masks=masks,raw_scores=raw,warm_raw_scores=warm,subjects=frozen.subjects,errors=frozen.errors,labels=frozen.labels,predictions=frozen.predictions,identities_json=json.dumps(frozen.identities),reference_names=np.asarray(ref_names),reference_scores=ref_scores,reference_masks=ref_masks)
    write(OUT/"replay_inputs.json",{"sha256":digest(path),"decision_seal_sha256":digest(OUT/"decisions_sealed.json"),"source_base_sha256":digest(PROJECT/"outputs/analysis_inputs/frozen_analysis_inputs.npz"),"outcomes_joined_after_all_decisions":True,"epochs":n})

def average_ranks(x):
    order=np.argsort(x,kind="stable");v=x[order]
    starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
    ends=np.r_[starts[1:],len(x)]
    result=np.empty(len(x),float)
    for a,b in zip(starts,ends):result[order[a:b]]=(a+b-1)/2
    return result

def replay(budget):
    decision_seal=require_seal()
    path=OUT/"replay_inputs.npz"
    if not path.exists():make_portable()
    receipt=read(OUT/"replay_inputs.json")
    if receipt["sha256"]!=digest(path) or receipt["decision_seal_sha256"]!=digest(OUT/"decisions_sealed.json"):
        raise RuntimeError("Replay identity changed")
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    subjects=data["subjects"];errors=data["errors"].astype(bool)
    people=sorted(set(subjects.tolist()))
    keys=[identity_key(i) for i in json.loads(data["identities_json"].item())]
    ties=identity_ties(keys)
    _,quotas,_=count_subjects(data["reference_masks"][:,0],errors,subjects,people)
    if len(people)!=20 or len(errors)!=42308:raise RuntimeError("Wrong cohort")
    cells=data["cells"].tolist()
    counts={};rows=[];per=[];matched_all=[]
    for j,cell in enumerate(cells):
        budget.check()
        matched=quota_mask(ranked_orders(data["scores"][:,j],ties,subjects,people),quotas,len(errors))
        matched_all.append(matched)
        for context,mask in [("cert",data["masks"][:,j]),("matched",matched)]:
            n,a,k=count_subjects(mask,errors,subjects,people)
            counts[context,cell]=(n,a,k)
            rows.append({"context":context,"cell":cell,**summary(n,a,k)})
            for person,nn,aa,kk in zip(people,n,a,k):
                per.append({"context":context,"cell":cell,"subject":person,"eligible":int(nn),"accepted":int(aa),"errors":int(kk),"coverage":float(aa/nn),"risk":float(kk/aa) if aa else None,"passed":bool(aa>=50 and 20*kk<=aa)})
    reference=[]
    for j,m in enumerate(data["reference_names"].tolist()):
        matched=quota_mask(ranked_orders(data["reference_scores"][:,j],ties,subjects,people),quotas,len(errors))
        expected=[817,946,1133,1369,2171][j]
        if int(np.sum(matched & errors))!=expected:raise RuntimeError("Frozen reference changed: "+m)
        for context,mask in [("cert",data["reference_masks"][:,j]),("matched",matched)]:
            reference.append({"context":context,"method":m,**summary(*count_subjects(mask,errors,subjects,people))})
    rng=np.random.Generator(np.random.PCG64(20260901));samples=rng.integers(0,20,size=(10000,20))
    paired=[];sensitivity=[]
    for size in (20,100):
        new=f"RiskObjective|{size}"
        for baseline in (f"BCE|{size}",f"SELE|{size}"):
            for context in ("cert","matched"):
                ac=counts[context,baseline];bc=counts[context,new]
                aa=bootstrap_metrics(*ac,samples);bb=bootstrap_metrics(*bc,samples)
                ap=summary(*ac);bp=summary(*bc)
                for metric in aa:
                    delta=None if ap[metric] is None or bp[metric] is None else bp[metric]-ap[metric]
                    paired.append({"context":context,"baseline":baseline,"method":new,"metric":metric,"difference":delta,**interval(bb[metric]-aa[metric])})
                paired.append({"context":context,"baseline":baseline,"method":new,"metric":"errors","difference":int(bc[2].sum()-ac[2].sum()),**interval(bc[2][samples].sum(1)-ac[2][samples].sum(1))})
                for ix,person in enumerate(people):
                    keep=np.arange(20)!=ix
                    one=summary(*(v[keep] for v in ac));two=summary(*(v[keep] for v in bc))
                    sensitivity.append({"context":context,"baseline":baseline,"method":new,"removed_subject":person,"coverage_difference":two["macro_coverage"]-one["macro_coverage"],"error_difference":two["errors"]-one["errors"],"risk_difference":None if one["subject_equal_ratio_risk"] is None or two["subject_equal_ratio_risk"] is None else two["subject_equal_ratio_risk"]-one["subject_equal_ratio_risk"]})
    diag=[]
    for person,quota in zip(people,quotas):
        at=np.flatnonzero(subjects==person)
        for j,m in enumerate(specification()["methods"]):
            raw=data["raw_scores"][at,j];warm=data["warm_raw_scores"][at,j]
            ar,br=average_ranks(raw),average_ranks(warm)
            corr=float(np.corrcoef(ar,br)[0,1]) if ar.std()>0 and br.std()>0 else None
            new_order=at[np.lexsort((ties[at],raw))];old_order=at[np.lexsort((ties[at],warm))]
            na=np.zeros(len(errors),bool);oa=na.copy();na[new_order[:quota]]=True;oa[old_order[:quota]]=True
            slope=decision_seal["folds"][person]["cells"][m+"|20"]["mapping"]["slope"]
            diag.append({"subject":person,"method":m,"rank_correlation_with_own_bce_initialization":corr,"matched_decisions_changed":int(np.sum(na^oa)),"matched_error_difference_vs_initialization":int(np.sum(na&errors)-np.sum(oa&errors)),"quota":int(quota),"final_platt_slope":slope,"calibration_collapsed_ranking":slope==0})
    fit_rows=[];solver_rows=[];training_cost=[]
    selection=read(OUT/"fit_selections.json")
    for outer,record in selection["folds"].items():
        for candidate in record["candidates"]:
            def number(v):
                if v is None:return None
                return v["numerator"]/v["denominator"] if isinstance(v,dict) else v
            fit_rows.append({"outer":outer,"method":candidate["method"],"C":candidate["C"],"status":candidate["status"],"selected":candidate["C"]==record["selected"][candidate["method"]]["C"],"coverage":number(candidate.get("coverage")),"risk":number(candidate.get("risk")),"dense_coverage":number(candidate.get("dense_coverage")),"constant_calibrators":candidate.get("constant_calibrators")})
    for p in sorted((OUT/"fits").rglob("*.json")):
        r=read(p)
        for m,model in r["models"].items():
            d=model["diagnostics"]
            solver_rows.append({"artifact":p.relative_to(ROOT).as_posix(),"method":m,"C":r["C"],"status":model["status"],"seconds":d.get("seconds"),"iterations":d.get("iterations"),"gap":d.get("gap"),"soft_coverage":d.get("soft_coverage"),"soft_risk":d.get("soft_risk")})
            own=float(d["seconds"]);warm=0. if m=="BCE" else float(r["models"]["BCE"]["diagnostics"]["seconds"])
            training_cost.append({"artifact":p.relative_to(ROOT).as_posix(),"method":m,"C":r["C"],"phase":"final" if p.stem=="final" else "cross_validation","status":model["status"],"incremental_solver_seconds":own,"including_bce_initialization_seconds":own+warm,"shared_group_seconds":r["seconds"]})
    for name,value in [("summary",rows),("per_subject",per),("paired_bootstrap",paired),("leave_one_subject",sensitivity),("ranking_diagnostics",diag),("fit_candidates",fit_rows),("solver_status",solver_rows),("training_cost",training_cost),("frozen_references",reference)]:table(name,value)
    # Matching is grid-independent because Pi100 uses the very same final scorer.
    for j in (0,2,4):np.testing.assert_array_equal(matched_all[j],matched_all[j+1])
    write(OUT/"replay_checks.json",{"status":"PASS","folds":20,"epochs":len(errors),"reference_errors":[817,946,1133,1369,2171],"same_scorer_matched_masks_grid_invariant":True,"analysis_code_sha256":digest(Path(__file__)),"tables":{p.name:digest(p) for p in sorted((OUT/"tables").glob("*.csv"))}})
    print("Replay PASS: 20 folds, 42308 epochs; frozen references unchanged",flush=True)

def report(budget):
    if not (OUT/"replay_checks.json").exists():raise RuntimeError("Run outcome replay first")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=csv_rows("summary");per=csv_rows("per_subject");paired=csv_rows("paired_bootstrap")
    methods=specification()["methods"];colors=["#2572A5","#B97517","#7753B5"]
    primary={m:next(r for r in rows if r["context"]=="cert" and r["cell"]==m+"|20") for m in methods}
    matched={m:next(r for r in rows if r["context"]=="matched" and r["cell"]==m+"|20") for m in methods}
    figdir=OUT/"figures";figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for j,m in enumerate(methods):
        if primary[m]["subject_equal_ratio_risk"]:
            ax[0].scatter(100*float(primary[m]["subject_equal_ratio_risk"]),100*float(primary[m]["macro_coverage"]),s=100,color=colors[j],label=m)
        else:
            ax[0].text(.02,.08+.07*j,m+": all abstain; risk undefined",transform=ax[0].transAxes,color=colors[j],fontsize=9)
        ax[1].bar(j,int(matched[m]["errors"]),color=colors[j]);ax[1].text(j,int(matched[m]["errors"]),matched[m]["errors"],ha="center",va="bottom")
    ax[0].axvline(5,color="#666666",ls="--",lw=1);ax[0].set(xlabel="OUTER subject-equal accepted risk (%)",ylabel="Mean-subject coverage (%)",title="Frozen CERT policies (20 candidates)");ax[0].legend(fontsize=9)
    ax[1].set(xticks=range(3),xticklabels=methods,ylabel="Accepted errors",title="Same accepted count per subject")
    ax[1].set_ylim(0,max(int(matched[m]["errors"]) for m in methods)*1.15)
    for ext in ("png","svg"):fig.savefig(figdir/("main_comparison."+ext),dpi=170)
    plt.close(fig)
    people=[f"{i:02d}" for i in range(20)]
    def count(m,s):return int(next(r for r in per if r["context"]=="matched" and r["cell"]==m+"|20" and r["subject"]==s)["errors"])
    fig,ax=plt.subplots(figsize=(11,3.5),layout="constrained")
    delta=np.array([count("RiskObjective",s)-count("BCE",s) for s in people])
    ax.bar(people,delta,color=np.where(delta<0,"#2572A5","#B97517"));ax.axhline(0,color="#444444",lw=1)
    ax.set(xlabel="OUTER subject",ylabel="New objective minus BCE errors",title="Matched counts: below zero means fewer errors")
    for ext in ("png","svg"):fig.savefig(figdir/("per_subject."+ext),dpi=170)
    plt.close(fig)
    # Predetermined illustrative fold; chosen on FIT, never chosen by OUTER performance.
    selected=read(OUT/"fit_selections.json")["folds"]["00"]["selected"]["RiskObjective"]
    fitted=read(OUT/"fits"/"outer-00"/f"c-{selected['C']:g}"/"final.json")
    trace=fitted["models"]["RiskObjective"]["diagnostics"]["trace"]
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout="constrained")
    for start,color in zip(specification()["risk"]["starts"],("#2572A5","#7753B5")):
        points=[r for r in trace if r["start_scale"]==start]
        axes[0].plot(range(1,len(points)+1),[100*r["coverage"] for r in points],"o-",label=f"BCE initialization x{start:g}",color=color)
        axes[1].plot(range(1,len(points)+1),[100*r["risk"] if r["risk"] is not None else np.nan for r in points],"o-",color=color)
    for ax in axes:
        ax.set_xticks([1,2,3],["0.5", "0.2", "0.1"]);ax.set_xlabel("Temperature / initial score SD")
    axes[0].set(ylabel="Training soft coverage (%)",title="Fixed illustrative outer fold 00");axes[0].legend(fontsize=9)
    axes[1].axhline(5,color="#666666",ls="--",lw=1);axes[1].set(ylabel="Training soft accepted risk (%)",title="Training feasibility, not OUTER reliability")
    axes[1].set_ylim(0,6)
    axes[1].ticklabel_format(axis="y",style="plain",useOffset=False)
    for ext in ("png","svg"):fig.savefig(figdir/("training_path."+ext),dpi=170)
    plt.close(fig)
    def percent(x):return "NA" if x in (None,"") else f"{float(x)*100:.2f}%"
    body="| 方法 | Coverage | R_SE | Pooled risk | 匹配错误 | 逐人通过 |\n|---|---:|---:|---:|---:|---:|\n"
    for m in methods:
        r=primary[m];body+=f"| {m} | {percent(r['macro_coverage'])} | {percent(r['subject_equal_ratio_risk'])} | {percent(r['pooled_risk'])} | {matched[m]['errors']} | {r['passed_subjects']}/20 |\n"
    references=csv_rows("frozen_references")
    reference_text="| 历史方法级参照 | Coverage | R_SE | 匹配错误 |\n|---|---:|---:|---:|\n"
    for name in ("Learned-Selected","DuoF","Primary-only","MSP","Forward-KL"):
        r=next(q for q in references if q["context"]=="cert" and q["method"]==name)
        q=next(q for q in references if q["context"]=="matched" and q["method"]==name)
        label="完整 COVER" if name=="Learned-Selected" else name
        reference_text+=f"| {label} | {percent(r['macro_coverage'])} | {percent(r['subject_equal_ratio_risk'])} | {q['errors']} |\n"
    reference_text+="\n这些方法共享原分类答案，但使用不同的输入／模型／候选空间。它们用于评估实际方法水平，不能代替上面的同容量损失对照。"
    d=int(matched["RiskObjective"]["errors"])-int(matched["BCE"]["errors"])
    interpretation=f"新目标在原 COVER 逐人接受配额下，相比受控 BCE 的错误变化为 {d:+d} 个；{int((delta<0).sum())} 人减少、{int((delta>0).sum())} 人增加、{int((delta==0).sum())} 人不变。"
    sele_delta=int(matched["RiskObjective"]["errors"])-int(matched["SELE"]["errors"])
    interpretation+=f" 相比受控 SELE，匹配错误变化为 {sele_delta:+d} 个。"
    costs="推理测量尚未执行；不据训练时间推断线上成本。"
    if (OUT/"tables/inference_cost.csv").exists():
        timings=csv_rows("inference_cost")
        costs="| 方法 | 已有 37D 输入：单段中位耗时 | 组件概率到分数：单段中位耗时 |\n|---|---:|---:|\n"
        for m in methods:
            vals=[]
            for scope in ("cached_37d_to_score","component_probabilities_to_score"):
                values=[float(r["median_epoch_us"]) for r in timings if r["method"]==m and r["scope"]==scope and r["batch_size"]=="1"]
                vals.append(float(np.median(values)))
            costs+=f"| {m} | {vals[0]:.2f} μs | {vals[1]:.2f} μs |\n"
        costs+="\n以上为 20 折的折内测量中位数再取中位数，两个 CPU 线程。完整表另含 batch=512；不含 Conformer、RF 特征提取／推理和决策并列处理。"
    training=csv_rows("training_cost")
    training_text="| 方法 | CV 拟合次数 | 每次拟合中位耗时（含 BCE 初始化） |\n|---|---:|---:|\n"
    for m in methods:
        values=[float(r["including_bce_initialization_seconds"]) for r in training if r["method"]==m and r["phase"]=="cross_validation"]
        training_text+=f"| {m} | {len(values)} | {np.median(values):.3f} 秒 |\n"
    training_text+="\n这是模型求解时间；不含交叉校准、规则枚举和数据加载。BCE 初始化在实际执行中共享，表中为了说明单独训练成本而计入 SELE 和新目标，不可把这三行简单相加当作本轮总耗时。"
    status=read(OUT/"replay_checks.json");runtime=read(OUT/"compute_budget.json")
    ci=next(r for r in paired if r["context"]=="matched" and r["baseline"]=="BCE|20" and r["metric"]=="errors")
    sele_ci=next(r for r in paired if r["context"]=="matched" and r["baseline"]=="SELE|20" and r["metric"]=="errors")
    comparisons=[]
    for other in ("BCE","SELE"):
        dc=100*(float(primary["RiskObjective"]["macro_coverage"])-float(primary[other]["macro_coverage"]))
        dr=None if not primary["RiskObjective"]["subject_equal_ratio_risk"] or not primary[other]["subject_equal_ratio_risk"] else 100*(float(primary["RiskObjective"]["subject_equal_ratio_risk"])-float(primary[other]["subject_equal_ratio_risk"]))
        drtext="未定义" if dr is None else f"{dr:+.2f} 个百分点"
        comparisons.append(f"- 相比 {other}：coverage {dc:+.2f} 个百分点，R_SE {drtext}。")
    if d<0 and sele_delta<0:
        conclusion="在本轮固定配额的聚合计数上，新目标优于两种同容量对照。是否形成稳定收益，还要结合以下描述性区间与逐人分布；这项结果不等于已经确立算法新颖性。"
    elif d>=0 and sele_delta>=0:
        conclusion="本轮固定配额结果不支持新目标优于两种同容量对照。应保留这个结果，检查训练代理目标向硬接受规则迁移的差距；不能据此把该原型升级为主方法。"
    else:
        conclusion="本轮新目标相对两种同容量对照的结果不一致，尚不支持同时优于一般错误学习和已有排序学习。应据实际比较限定方法主张。"
    outcome_text=conclusion+"\n\n"+"\n".join(comparisons)+f"\n\n相对 SELE 的匹配错误差描述性区间：[{float(sele_ci['low']):.1f}, {float(sele_ci['high']):.1f}]。"
    candidates=csv_rows("fit_candidates")
    selected_fit={m:[r for r in candidates if r["method"]==m and r["selected"]=="True"] for m in methods}
    fit_comparison="| 目标 | FIT 验证：20 点可行 coverage 均值 | 同配置 100 点诊断 |\n|---|---:|---:|\n"
    for m in methods:
        fit_comparison+=f"| {m} | {100*np.mean([float(r['coverage']) for r in selected_fit[m]]):.2f}% | {100*np.mean([float(r['dense_coverage']) for r in selected_fit[m]]):.2f}% |\n"
    ranks=[r for r in csv_rows("ranking_diagnostics") if r["method"]=="RiskObjective"]
    median_rank=float(np.median([float(r["rank_correlation_with_own_bce_initialization"]) for r in ranks if r["rank_correlation_with_own_bce_initialization"]]))
    flips=sum(int(r["matched_decisions_changed"]) for r in ranks)
    collapsed=sum(r["calibration_collapsed_ranking"]=="True" for r in ranks)
    diagnostics_text=fit_comparison+f"\n相对同配置 BCE 初始化，新目标的逐人排序相关性中位数为 {median_rank:.4f}，固定配额下改变了 {flips:,} 个决策；Platt 将排序压成常数的折数为 {collapsed}/20。\n\n这些诊断用于区分三件事：参数确实学出了不同排序；软约束数值达标；验证和最终硬规则是否更好。前两项成立不能代替第三项。FIT 验证和 100 点诊断也应一起读，不能把结果直接归因于 CERT 或某一种校准机制。"
    validation_text="最终交付验证尚未执行。"
    if (OUT/"validation.json").exists():
        validation=read(OUT/"validation.json")
        validation_text=f"{validation.get('unit_tests_run','已记录的')} 项单元测试通过；结果、门禁、模型和文件核验 {validation['checks_passed']}/{validation['checks_total']} 项通过。"
        if validation.get("external_activity"):
            validation_text+=" 唯一未满足项为全目录字节／时间状态完全不变：两个既有 TinySleepNet guardian 日志在外部刷新，状态仍为 PAUSED 且没有工作进程。原输入、结果表、论文和发布包的专门指纹检查通过。本实验没有修改这些 guardian 文件。"
        repeat=read(OUT/"replay_repeatability.json") if (OUT/"replay_repeatability.json").exists() else None
        if repeat:validation_text+=f" 再次重放的 {repeat['table_count']} 张 CSV 逐字节一致。"
    narrative=f"""# 新训练目标：实际结果与研究判断

## 先理解输入和输出

Conformer 和 RF 保持固定。每段 EEG 的两组预测输出被转换为 37D 特征，三种线性评分器学习如何给主模型原答案排序。输出仍是原分类标签或弃权。

本轮改变的是评分器训练目标。BCE 预测对错；SELE 优化错误排序；RiskObjective 用软接受量与错误预算约束更新参数。最终接受阈值仍由 CERT 选择。

## 完整 20 折结果

{body}
{interpretation}

匹配错误差的配对受试者描述性 95% 区间为 [{float(ci['low']):.1f}, {float(ci['high']):.1f}]。区间涉及重叠训练的历史队列，不作为独立确认性推断。

## 本轮回答了什么

{outcome_text}

## 与完整历史方法的关系

{reference_text}

## 如何判断贡献

这张表比较的是共同输入、线性函数类、候选预算及校准流程下的三个训练算法，不是原 COVER 的重新命名。原 COVER/AutoML 属于额外参照。
必须同时阅读 coverage 与接受风险：更高 coverage 若伴随更高风险，不自动构成改进。匹配配额结果检验选择质量；逐人图说明收益是否集中。
软目标变好还可能只是改变分数尺度。`ranking_diagnostics.csv` 将各模型与其对应 BCE 初始化比较；`fit_candidates.csv` 对比 FIT 硬操作点与细网格诊断。
初始化使用与该目标相同的正则化配置，不一定是主表中单独选出的 BCE 配置；这两个参照不能混为一谈。
排序诊断采用原始分数；表中另标注最终 Platt 斜率，若斜率为零，部署分数的排序会退化。

`training_path.png` 固定展示 outer-00，说明温度延续过程中软接受量和约束的变化。它是训练过程说明，不作为跨受试者收益证据。

## 用已有产物定位下一步

{diagnostics_text}

## 推理成本

{costs}

## 离线训练成本

{training_text}

## 边界与状态

- 已完成 {status['folds']} 折、{status['epochs']:,} 段评价；原五方法匹配错误仍为 817/946/1133/1369/2171。
- 本轮不重训组件，全部 FIT 选择先固定，随后 CERT 选规，全部 OUTER 决策固定后才合并真值。
- CERT 约束为经验约束，不提供 95% 新受试者风险认证；FIT 沿用组件 OOF 的祖先复用。
- 各目标的优化器不同；相同容量和候选预算不等于相同计算时间。
- 截至生成本说明，累计受监控计算时间约 {runtime['used_seconds']/60:.1f} 分钟。完整计时见 compute_budget.json。

## 当前入口

表：`outputs/tables/`；图：`outputs/figures/`；教学 notebook：`notebooks/03_new_objective_results.ipynb`。
复算：`python -B run.py replay`；重建报告：`python -B run.py report`。
本轮未修改稿件、旧 notebook 或发布包。

## 验证记录

{validation_text}
"""
    text_file(ROOT/"docs/RESULTS_ZH.md",narrative)
    make_notebook(body,interpretation,costs+"\n\n"+training_text,outcome_text,reference_text,diagnostics_text)
    write(OUT/"report_status.json",{"status":"GENERATED","created":utc(),"figures":["main_comparison","per_subject","training_path"],"result_md":"docs/RESULTS_ZH.md","notebook":"notebooks/03_new_objective_results.ipynb"})
    print(body,flush=True);print(interpretation,flush=True)

def make_notebook(body,interpretation,costs,outcome_text,reference_text,diagnostics_text):
    def md(s):return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}
    def code(s):return {"cell_type":"code","metadata":{},"source":s.splitlines(keepends=True),"execution_count":None,"outputs":[]}
    cells=[md("# 03 新目标究竟改变了什么？\n\n01 解释现有 COVER/AutoML 怎样运行；02 解释为什么值得研究训练目标；**03 展示现在真正实现的原型和完整实验结果。**\n\n阅读顺序：训练反馈 → 公平比较 → 主结果 → 个体差异 → 研究判断。代码只读结果，不启动拟合。"),
      md("## 1. 同一份答案，三种训练方式\n\nConformer 的答案和 RF 输出固定。BCE 问‘错误概率预测准了吗’；SELE 问‘错误是否排在后面’；新目标问‘在错误预算约束下，软接受区域能否扩大’。最终 CERT 阈值独立选择，训练阈值不会直接部署。"),
      code("from pathlib import Path\nimport csv\nfrom html import escape\nimport numpy as np\nfrom IPython.display import display, Image, HTML\nHERE=Path.cwd()\nROOT=HERE if (HERE/'protocol.json').exists() else HERE.parent\nassert (ROOT/'protocol.json').exists()\ndef read_table(name):\n    with (ROOT/'outputs'/'tables'/f'{name}.csv').open(encoding='utf-8') as f:\n        return list(csv.DictReader(f))\ndef show_table(rows, columns):\n    head='<tr>'+''.join('<th>'+escape(label)+'</th>' for key,label in columns)+'</tr>'\n    body=''.join('<tr>'+''.join('<td>'+escape(str(row.get(key,'')))+'</td>' for key,label in columns)+'</tr>' for row in rows)\n    display(HTML('<table>'+head+body+'</table>'))\nrows=read_table('summary')\nprint('已保存的汇总行数:',len(rows))"),
      md("## 2. 为什么不能只看软目标？\n\n下面固定排序和硬阈值，仅把分数乘以正数。硬接受集合完全相同，但 sigmoid 软风险变化。这是数学教学例子，不是真实数据结果。"),
      code("scores=np.arange(1.,101.)\nerrors=scores>80\nfor k in (.02,.1,.5):\n    gate=1/(1+np.exp(-k*(70.5-scores)))\n    print(k, 'hard coverage', np.mean(k*scores<=k*70.5), 'soft coverage',gate.mean(), 'soft risk', (gate*errors).sum()/gate.sum())"),
      md("## 3. 真正实现的优化是什么？\n\n令 $a_i=\\sigma((t-\\theta^Tz_i)/T)$、$w_i=1/(S n_{s_i})$。本次实际求解\n\n$$\\max_{\\theta,t}\\;\\sum_i w_i a_i-\\frac{\\rho}{2}\\|\\theta\\|^2\\quad\\mathrm{s.t.}\\quad\\sum_i w_i a_i e_i\\le0.05\\sum_iw_ia_i,\\;\\sum_iw_ia_i\\ge10^{-4}.$$\n\n**它改的是训练反馈。** 第一项鼓励留下更多，约束限制留下来的错误。BCE 则最小化全体样本的二元预测损失。$t$ 只在训练时使用，部署时仍由 CERT 单独选择阈值。SLSQP 给出经检查的局部解，不宣称全局最优。\n\n三个目标共同使用 37D 输入、线性排序、四个正则化、FIT/CERT/OUTER 与单调标准化 Platt。SELE 检查凸优化最优值差；新目标检查软约束残差。不同求解器的耗时单独报告。"),
      code("display(Image(filename=str(ROOT/'outputs/figures/training_path.png')))"),
      md("上图固定展示 outer-00。温度降低让软边界接近硬边界。训练可行不保证最终 CERT/OUTER 可行，下面继续检查真正的硬决策。"),
      md("## 4. 实际主结果\n\n"+body+"\n"+interpretation),
      code("display(Image(filename=str(ROOT/'outputs/figures/main_comparison.png')))"),
      md("左图回答各自 CERT 规则最终接受多少、错多少；右图锁定每人接受数量，只比较错误。Π100 使用同一 scorer，因此匹配配额结果应与 Π20 相同。"),
      md("### 与原来的 COVER 相比呢？\n\n"+reference_text),
      md("## 5. 是不是每个人都受益？\n\n负柱表示新目标比 BCE 少错，正柱表示多错。总体净收益不能代替逐人检查。"),
      code("display(Image(filename=str(ROOT/'outputs/figures/per_subject.png')))\npaired=read_table('paired_bootstrap')\nselected=[r for r in paired if r['context']=='matched' and r['baseline']=='BCE|20' and r['metric']=='errors']\nshow_table(selected,[('difference','新目标 − BCE 错误'),('low','描述性区间下界'),('high','描述性区间上界'),('valid_draws','有效重抽样次数')])"),
      md("## 6. 排序改变，还是只换了尺度？\n\n下表将选中的新模型与**同配置的 BCE 初始化**比较。这个初值不一定是主表中按 FIT 选出的 BCE 模型，因此此表的错误变化不一定等于主表差值。\n\n排序相关性接近 1 和几乎不变的匹配集合，提示变化可能主要来自尺度／操作点。相关性变化本身也不等于性能提升，需结合错误数。完整 CSV 另记录 Platt 是否将分数压成常数。"),
      code("diagnostics=read_table('ranking_diagnostics')\nselected=[dict(r) for r in diagnostics if r['method']=='RiskObjective']\nfor r in selected:\n    value=r['rank_correlation_with_own_bce_initialization']\n    r['correlation_display']=f'{float(value):.4f}' if value else 'NA'\nshow_table(selected,[('subject','受试者'),('correlation_display','与对应 BCE 初始排序的相关性'),('matched_decisions_changed','同配额下改变的决策数'),('matched_error_difference_vs_initialization','错误变化')])"),
      md("### 差距是否已经出现在 FIT 验证中？\n\n"+diagnostics_text),
      md("## 7. 额外计算是多少？\n\n"+costs+"\n\n搜索／拟合成本和线上推理成本是不同的账。相同线性函数类可能有相近推理成本，却需要不同的离线求解时间。"),
      md("## 8. 这次能支持什么贡献？\n\n"+outcome_text+"\n\n软约束成功但硬策略没有收益，说明代理目标和部署规则仍有距离；单凭训练目标达到可行，不足以替代本表中的比较。本文实验属于历史队列回顾性扩展，未提供新受试者风险保证。\n\n完整说明与失败记录在 ../docs/RESULTS_ZH.md 和 ../outputs/。\n\n参考：[SELE 原文](https://proceedings.mlr.press/v97/franc19a.html)。这里实现的是受试者加权的线性 hinge 适配。" )]
    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"cells":cells}
    for i,c in enumerate(cells):c["id"]=f"new-objective-{i:02d}"
    write(ROOT/"notebooks/03_new_objective_results.ipynb",notebook)
