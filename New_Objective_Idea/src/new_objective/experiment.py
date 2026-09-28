"""Strict FIT -> CERT -> decision seal orchestration; writes only to this experiment."""
from __future__ import annotations
import json
import time
import platform
from pathlib import Path
import numpy as np
from .storage import ROOT,PROJECT,OUT,read,write,digest,model_save,arrays,specification,fitting_identity,utc
from .losses import weights
from .solvers import fit_bce,fit_sele,fit_risk
from .calibration import fit_mapping,apply_mapping,cross_calibrate
from .policies import policies,choose,mask,subset_policies,fit_rank

def load_fit(outer):
    from stage1_analysis.selection_ablation import load_development
    return load_development(PROJECT,outer)

def features(data):
    from stage1_analysis.risk_features import heterogeneous_features
    x,names=heterogeneous_features(data["primary"],data["feature"])
    if x.shape[1]!=37 or not np.isfinite(x).all():
        raise ValueError("Expected finite 37D risk features")
    return x

def inventory():
    result={}
    for p in PROJECT.rglob("*"):
        if p.is_file() and not p.is_relative_to(ROOT):
            s=p.stat();result[p.relative_to(PROJECT).as_posix()]=[s.st_size,s.st_mtime_ns]
    return result

def preflight(budget,resume):
    import scipy,sklearn
    marker=OUT/"preflight.json"
    if marker.exists():
        old=read(marker)
        print("Preflight baseline preserved:",old["created"],flush=True)
        return
    before=inventory()
    write(OUT/"outside_inventory.json",before)
    sources={}
    for i in range(20):
        budget.check();outer=f"{i:02d}"
        d,source=load_fit(outer)
        sources[outer]=source
        assert len(d["risk_folds"])==5
        assert features(d["fit_data"]).shape[0]==len(d["fit_data"]["errors"])
    critical=["paper/main.tex","paper/appendix.tex","outputs/analysis_inputs/frozen_analysis_inputs.npz","outputs/analysis_inputs/automl_inputs.npz"]
    critical += [str(p.relative_to(PROJECT)) for p in (PROJECT/"outputs/strengthening/stage1_strengthening_v1/tables").glob("*.csv")]
    critical += [str(p.relative_to(PROJECT)) for p in (PROJECT/"releases").glob("*") if p.is_file()]
    hashes={p:digest(PROJECT/p) for p in critical if (PROJECT/p).exists()}
    write(marker,{"created":utc(),"environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},"development_sources":sources,"critical_hashes":hashes,"outside_files":len(before),"status":"PASS"})
    print("Preflight PASS: 20 verified development caches; baseline recorded",flush=True)

def require_source(outer,source):
    if source!=read(OUT/"preflight.json")["development_sources"][outer]:
        raise RuntimeError("Development source changed")

def fit_three(x,y,subjects,c,spec,budget):
    from sklearn.preprocessing import StandardScaler
    started=time.monotonic()
    scaler=StandardScaler().fit(x)
    xx=np.ascontiguousarray(scaler.transform(x));w=weights(subjects);reg=1./(len(y)*c)
    if set(np.unique(y))!={0,1}:
        raise ValueError("Both error classes required")
    bce=fit_bce(xx,y,w,reg,spec)
    if bce["status"]!="OK":
        raise RuntimeError("BCE reference did not converge")
    budget.check()
    sele=fit_sele(xx,y,w,reg,bce["theta"],spec,budget.check)
    budget.check()
    risk=fit_risk(xx,y,w,reg,bce["theta"],spec,budget.check)
    return {"mean":scaler.mean_,"scale":scaler.scale_,"models":{"BCE":bce,"SELE":sele,"RiskObjective":risk},"C":c,"train_subjects":sorted(set(subjects.tolist())),"seconds":time.monotonic()-started}

def raw_predict(bundle,method,x):
    m=bundle["models"][method]
    if m["status"]!="OK":
        raise RuntimeError("Invalid fitted scorer: "+m["status"])
    return ((x-bundle["mean"])/bundle["scale"])@m["theta"]+m["intercept"]

def cache_fit(path,x,y,subjects,c,spec,budget,fp,source,resume):
    import joblib
    receipt=path.with_suffix(".json")
    if receipt.exists():
        old=read(receipt)
        if not resume or old["fingerprint"]!=fp or old["source"]!=source or old["model_sha256"]!=digest(path):
            raise RuntimeError("Incompatible existing fit checkpoint: "+str(path))
        return joblib.load(path)
    result=fit_three(x,y,subjects,c,spec,budget)
    model_save(path,result)
    write(receipt,{"fingerprint":fp,"source":source,"model_sha256":digest(path),"C":c,"seconds":result["seconds"],"models":{m:{"status":v["status"],"diagnostics":v["diagnostics"]} for m,v in result["models"].items()}})
    return result

def pilot(budget,resume):
    spec=specification();identity=fitting_identity();fp=identity["sha256"]
    if not (OUT/"preflight.json").exists():
        preflight(budget,resume)
    rows=[]
    for outer in spec["pilot_outers"]:
        d,source=load_fit(outer);require_source(outer,source)
        fit=d["fit_data"];x=features(fit);fold=d["risk_folds"][0]
        tr=np.isin(fit["subjects"],fold["train"])
        for c in spec["pilot_c"]:
            path=OUT/"fits"/f"outer-{outer}"/f"c-{c:g}"/"fold-0.joblib"
            result=cache_fit(path,x[tr],fit["errors"][tr],fit["subjects"][tr],c,spec,budget,fp,source,resume)
            statuses={m:v["status"] for m,v in result["models"].items()}
            rows.append({"outer":outer,"C":c,"n":int(tr.sum()),"seconds":result["seconds"],"statuses":statuses})
            print("Pilot",outer,c,round(result["seconds"],2),statuses,flush=True)
    worst=max(r["seconds"] for r in rows)
    estimate=worst*(20*5*4+20*3)*1.25
    ok=all(s=="OK" for row in rows for s in row["statuses"].values())
    write(OUT/"pilot.json",{"status":"PASS" if ok else "NEEDS_NUMERICAL_REPAIR","fingerprint":fp,"rows":rows,"conservative_fit_seconds":estimate,"within_fit_budget":estimate<=spec["fit_budget_seconds"]-budget.state["used_seconds"],"interpretation":"technical timing and solver quality only; no CERT/OUTER labels used"})
    if not ok:
        raise RuntimeError("Pilot identified unconverged candidates; fix numerical implementation before the full matrix")
    if estimate>spec["fit_budget_seconds"]-budget.state["used_seconds"]:
        raise TimeoutError("Pilot estimate exceeds approved fitting budget")

def fit(budget,resume):
    spec=specification();identity=fitting_identity();fp=identity["sha256"]
    pilot_result=read(OUT/"pilot.json")
    if pilot_result["status"]!="PASS" or pilot_result["fingerprint"]!=fp:
        raise RuntimeError("Current fitting implementation needs a passing pilot")
    selections={"status":"FITTING","fingerprint":fp,"identity":identity,"folds":{},"spec":spec}
    if resume and (OUT/"fit_progress.json").exists():
        previous=read(OUT/"fit_progress.json")
        if previous["fingerprint"]!=fp:
            raise RuntimeError("FIT selection resume fingerprint mismatch")
        selections["folds"]=previous["folds"]
    for index in range(20):
        budget.check();outer=f"{index:02d}";d,source=load_fit(outer);require_source(outer,source)
        if outer in selections["folds"]:
            old=selections["folds"][outer]
            if old["source"]!=source or old["arrays_sha256"]!=digest(OUT/"selected"/f"outer-{outer}.npz"):
                raise RuntimeError("Completed FIT selection source changed")
            for receipt_path in (OUT/"fits"/f"outer-{outer}").rglob("fold-*.json"):
                checkpoint=read(receipt_path)
                if checkpoint["fingerprint"]!=fp or checkpoint["source"]!=source or checkpoint["model_sha256"]!=digest(receipt_path.with_suffix(".joblib")):
                    raise RuntimeError("Completed FIT model checkpoint changed")
            print("Reusing verified completed FIT selection: outer-"+outer,flush=True)
            continue
        fitdata=d["fit_data"];x=features(fitdata);y=fitdata["errors"];sub=fitdata["subjects"]
        rows=[];candidate_outputs={}
        for c in spec["c_grid"]:
            raw={m:np.full(len(y),np.nan) for m in spec["methods"]}
            errors={m:[] for m in spec["methods"]}
            for foldno,fold in enumerate(d["risk_folds"]):
                tr=np.isin(sub,fold["train"]);va=np.isin(sub,fold["validation"])
                path=OUT/"fits"/f"outer-{outer}"/f"c-{c:g}"/f"fold-{foldno}.joblib"
                bundle=cache_fit(path,x[tr],y[tr],sub[tr],c,spec,budget,fp,source,resume)
                for m in spec["methods"]:
                    if bundle["models"][m]["status"]=="OK":
                        raw[m][va]=raw_predict(bundle,m,x[va])
                    else:
                        errors[m].append({"fold":foldno,"status":bundle["models"][m]["status"]})
            for m in spec["methods"]:
                if errors[m]:
                    rows.append({"method":m,"C":c,"status":"INVALID_FIT","failures":errors[m]});continue
                calibrated,mappings=cross_calibrate(raw[m],y,sub,d["risk_folds"])
                grid=policies(calibrated,sub)
                picked,_=choose(subset_policies(grid,20),calibrated,y,fitdata,outer,spec)
                dense,_=choose(grid,calibrated,y,fitdata,outer,spec)
                row={"method":m,"C":c,"status":"OK","coverage":0 if picked is None else picked["coverage"],"risk":None if picked is None else picked["risk"],"dense_coverage":0 if dense is None else dense["coverage"],"constant_calibrators":sum(v["constant"] for v in mappings)}
                rows.append(row);candidate_outputs[m,c]={"raw":raw[m],"calibrated":calibrated,"policies":grid}
            print(f"FIT outer-{outer} C={c:g} complete",flush=True)
        chosen={}
        for m in spec["methods"]:
            valid=[r for r in rows if r["method"]==m and r["status"]=="OK"]
            if not valid:
                write(OUT/"fit_failure.json",{"outer":outer,"method":m,"candidates":rows,"status":"NO_VALID_FIT"})
                raise RuntimeError(f"No valid FIT candidate for {m}, outer-{outer}; cannot drop this fold")
            chosen[m]=min(valid,key=fit_rank)
        array_payload={}
        selected_metadata={}
        for m,row in chosen.items():
            value=candidate_outputs[m,row["C"]]
            array_payload[m+"_raw"]=value["raw"]
            array_payload[m+"_calibrated"]=value["calibrated"]
            selected_metadata[m]={**row,"policies":value["policies"]}
        array_path=OUT/"selected"/f"outer-{outer}.npz"
        arrays(array_path,**array_payload)
        selections["folds"][outer]={"source":source,"selected":selected_metadata,"candidates":rows,"arrays_sha256":digest(array_path),"fit_subjects":d["fit_subjects"],"cert_subjects":d["cert_subjects"]}
        write(OUT/"fit_progress.json",selections)
        print("All FIT choices fixed for outer-"+outer,flush=True)
    selections["status"]="ALL_FIT_SELECTIONS_FROZEN";selections["sealed_at"]=utc()
    write(OUT/"fit_selections.json",selections)
    # Refit all final scorers before any CERT-label read.
    for outer,record in selections["folds"].items():
        d,source=load_fit(outer);fitdata=d["fit_data"];x=features(fitdata)
        for c in sorted({r["C"] for r in record["selected"].values()}):
            path=OUT/"fits"/f"outer-{outer}"/f"c-{c:g}"/"final.joblib"
            cache_fit(path,x,fitdata["errors"],fitdata["subjects"],c,spec,budget,fp,source,resume)
        print("Final risk scorers fitted: outer-"+outer,flush=True)
    write(OUT/"fit_completed.json",{"status":"ALL_FINAL_SCORERS_FITTED","fingerprint":fp,"selection_sha256":digest(OUT/"fit_selections.json"),"completed":utc()})

def evaluate(budget,resume):
    import joblib
    from stage1_analysis.selection_ablation import read_cert_labels
    spec=specification();fp=fitting_identity()["sha256"]
    seal=read(OUT/"fit_selections.json");done=read(OUT/"fit_completed.json")
    if seal["status"]!="ALL_FIT_SELECTIONS_FROZEN" or set(seal["folds"])!={f"{i:02d}" for i in range(20)} or done["fingerprint"]!=fp or done["selection_sha256"]!=digest(OUT/"fit_selections.json"):
        raise RuntimeError("All FIT choices/final models must be fixed before CERT")
    cells=[f"{m}|{size}" for m in spec["methods"] for size in (20,100)]
    receipts={}
    for outer,record in seal["folds"].items():
        budget.check();path=OUT/"decisions"/f"outer-{outer}.npz";meta=path.with_suffix(".json")
        if meta.exists():
            old=read(meta)
            if not resume or old["fingerprint"]!=fp or old["sha256"]!=digest(path):
                raise RuntimeError("Decision resume identity mismatch")
            receipts[outer]=old;continue
        d,source=load_fit(outer);require_source(outer,source)
        cert,outerdata=d["cert_data_blind"],d["outer_data_blind"]
        labels,label_sources=read_cert_labels(PROJECT,cert,d["cert_subjects"],outer)
        ey=(cert["primary"].argmax(1)!=labels).astype(int)
        selected_path=OUT/"selected"/f"outer-{outer}.npz"
        if digest(selected_path)!=record["arrays_sha256"]:
            raise RuntimeError("Selected FIT arrays changed")
        raw_arrays=np.load(selected_path,allow_pickle=False)
        scores=[];masks=[];info={};raw_outer=[];warm_outer=[]
        for m in spec["methods"]:
            chosen=record["selected"][m];c=chosen["C"]
            model_path=OUT/"fits"/f"outer-{outer}"/f"c-{c:g}"/"final.joblib"
            model_receipt=read(model_path.with_suffix(".json"))
            if model_receipt["fingerprint"]!=fp or digest(model_path)!=model_receipt["model_sha256"]:
                raise RuntimeError("Final model identity changed")
            bundle=joblib.load(model_path)
            mapping=fit_mapping(raw_arrays[m+"_raw"],d["fit_data"]["errors"],d["fit_data"]["subjects"])
            rc=raw_predict(bundle,m,features(cert));ro=raw_predict(bundle,m,features(outerdata))
            sc=apply_mapping(mapping,rc);so=apply_mapping(mapping,ro)
            raw_outer.append(ro)
            warm_outer.append(raw_predict(bundle,"BCE",features(outerdata)))
            for size in (20,100):
                grid=subset_policies(chosen["policies"],size)
                rule,rows=choose(grid,sc,ey,cert,outer,spec,minimum=50)
                masks.append(mask(so,rule,outerdata,outer,spec));scores.append(so)
                info[f"{m}|{size}"]={"rule":rule,"feasible_rules":sum(z["supported"] and z["risk"] is not None and float(z["risk"]["numerator"])/z["risk"]["denominator"]<=.05 for z in rows),"mapping":mapping,"C":c,"solver":bundle["models"][m]["diagnostics"],"zero_fallback":rule is None}
        raw_arrays.close()
        arrays(path,cells=np.asarray(cells),scores=np.column_stack(scores),raw_scores=np.column_stack(raw_outer),warm_raw_scores=np.column_stack(warm_outer),masks=np.column_stack(masks),identities_json=json.dumps(outerdata["identities"]),predictions=outerdata["primary"].argmax(1),subjects=outerdata["subjects"])
        receipts[outer]={"status":"OUTER_DECISIONS_FROZEN","fingerprint":fp,"sha256":digest(path),"source":source,"cert_label_sources":label_sources,"contains_outer_labels":False,"cells":info,"completed":utc()}
        write(meta,receipts[outer]);print("CERT rule and OUTER decisions fixed:",outer,flush=True)
    write(OUT/"decisions_sealed.json",{"status":"ALL_OUTER_DECISIONS_FROZEN","fingerprint":fp,"folds":receipts,"completed":utc()})
    print("All OUTER decisions sealed. Outcome replay is now permitted.",flush=True)
