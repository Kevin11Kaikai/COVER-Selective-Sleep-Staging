"""Frozen-base anchored experiment; all mutations stay under New_Objective_Idea."""
from __future__ import annotations
import hashlib
import json
import time
import numpy as np
from .storage import ROOT,PROJECT,OUT,read,write,digest,arrays,model_save,utc
from .experiment import load_fit,require_source,features,raw_predict
from .calibration import cross_calibrate,fit_mapping,apply_mapping
from .policies import policies,choose,subset_policies,mask,exact
from .anchor_solver import fit_refinement

AOUT=OUT/'anchored'


def spec():return read(ROOT/'anchor_protocol.json')


def identity():
    files=[ROOT/'anchor_protocol.json',ROOT/'run_anchored.py']
    files += [ROOT/'src/new_objective'/n for n in ('anchor_solver.py','anchor_experiment.py','storage.py','losses.py','calibration.py','policies.py','experiment.py')]
    files += [PROJECT/n for n in ('src/stage1_analysis/risk_features.py','src/stage1_analysis/selection_ablation.py','src/stage1_analysis/acceptance.py','AutoML_Claude/automl/grids.py')]
    hashes={str(p.relative_to(PROJECT)):digest(p) for p in files}
    return {'sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),'files':hashes}


def old_selection():
    return read(OUT/'fit_selections.json')


def load_base(outer,fold,source):
    import joblib
    old=old_selection();c=old['folds'][outer]['selected']['BCE']['C']
    name='final' if fold=='final' else f'fold-{fold}'
    path=OUT/'fits'/f'outer-{outer}'/f'c-{c:g}'/(name+'.joblib')
    receipt=read(path.with_suffix('.json'))
    assert receipt['fingerprint']==old['fingerprint'] and receipt['source']==source
    assert digest(path)==receipt['model_sha256']
    return joblib.load(path),{'path':str(path.relative_to(ROOT)),'sha256':receipt['model_sha256'],'C':c}


def preflight(budget,resume):
    path=AOUT/'preflight.json'
    if path.exists():
        require_preflight();print('Anchored baseline preserved',flush=True);return
    current=identity();old=old_selection();sources={}
    for i in range(20):
        budget.check();outer=f'{i:02d}';d,source=load_fit(outer);require_source(outer,source)
        for fold in list(range(5))+['final']:load_base(outer,fold,source)
        sources[outer]=source
    old_paths=[ROOT/'protocol.json',OUT/'fit_selections.json',OUT/'decisions_sealed.json',OUT/'replay_inputs.npz']
    old_paths+=list((OUT/'fits').rglob('*.joblib'))+list((OUT/'selected').glob('*.npz'))+list((OUT/'decisions').glob('*.npz'))
    old_paths+=list((OUT/'tables').glob('*.csv'))
    old_paths += [PROJECT/p for p in read(OUT/'preflight.json')['critical_hashes']]
    protected={str(p.relative_to(PROJECT)):digest(p) for p in old_paths if not p.name.startswith('anchor_')}
    write(path,{'status':'PASS','created':utc(),'identity':current,'base_selection_sha256':digest(OUT/'fit_selections.json'),
        'sources':sources,'protected_sha256':protected,'base_final_models_exist':20,'new_base_fits_needed':0})
    print('Preflight PASS: all 120 selected-BCE CV/final bases available; no base refits',flush=True)


def require_preflight():
    p=read(AOUT/'preflight.json')
    if p['identity']['sha256']!=identity()['sha256'] or p['base_selection_sha256']!=digest(OUT/'fit_selections.json'):
        raise RuntimeError('Anchored protocol/code/base selection changed; explicit compatibility review required')
    return p


def path_for(outer,lam,fold):
    return AOUT/'fits'/f'outer-{outer}'/f'lambda-{lam:g}'/('final.joblib' if fold=='final' else f'fold-{fold}.joblib')


def fit_group(outer,fold,lam,x,y,subjects,source,budget,resume):
    import joblib
    path=path_for(outer,lam,fold);receipt=path.with_suffix('.json')
    base,base_info=load_base(outer,fold,source);fp=identity()['sha256']
    if receipt.exists():
        old=read(receipt)
        if not resume or old['fingerprint']!=fp or old['base']!=base_info or old['sha256']!=digest(path):
            raise RuntimeError('Incompatible anchored checkpoint: '+str(path))
        return joblib.load(path)
    assert set(base['train_subjects'])==set(subjects.tolist())
    xx=(x-base['mean'])/base['scale'];cfg=spec();models={};started=time.monotonic()
    for method in cfg['methods']:
        budget.check()
        models[method]=fit_refinement(xx,y,subjects,base['models']['BCE'],lam,method,cfg,budget.check)
    bundle={'mean':base['mean'],'scale':base['scale'],'models':models,'lambda':lam,'base':base_info,
            'train_subjects':sorted(set(subjects.tolist())),'seconds':time.monotonic()-started}
    model_save(path,bundle)
    write(receipt,{'fingerprint':fp,'source':source,'base':base_info,'sha256':digest(path),'seconds':bundle['seconds'],
        'statuses':{m:z['status'] for m,z in models.items()},'created':utc()})
    return bundle


def pilot(budget,resume):
    preflight(budget,resume);cfg=spec();rows=[]
    for outer in cfg['pilot_outers']:
        d,source=load_fit(outer);require_source(outer,source)
        data=d['fit_data'];x=features(data);tr=np.isin(data['subjects'],d['risk_folds'][0]['train'])
        for lam in cfg['pilot_lambdas']:
            b=fit_group(outer,0,lam,x[tr],data['errors'][tr],data['subjects'][tr],source,budget,resume)
            statuses={m:v['status'] for m,v in b['models'].items()}
            rows.append({'outer':outer,'lambda':lam,'seconds':b['seconds'],'statuses':statuses})
            print('Anchored pilot',outer,lam,round(b['seconds'],2),statuses,flush=True)
    estimate=max(r['seconds'] for r in rows)*440*1.25
    ok=all(s=='OK' for r in rows for s in r['statuses'].values())
    remaining=read(ROOT/'protocol.json')['fit_budget_seconds']-budget.state['used_seconds']
    write(AOUT/'pilot.json',{'status':'PASS' if ok and estimate<remaining else 'NEEDS_TECHNICAL_REVIEW',
        'fingerprint':identity()['sha256'],'rows':rows,'estimated_seconds':estimate,'remaining_fit_budget_seconds':remaining,
        'no_cert_or_outer_outcomes':True})
    if not ok:raise RuntimeError('Pilot solver failed; review numerical behavior before full run')
    if estimate>=remaining:raise TimeoutError('Pilot estimate exceeds remaining original budget')


def rank_candidate(row):
    return (-exact(row['coverage']),exact(row['risk']) if row['risk'] is not None else 2,-row['lambda'])


def fit(budget,resume):
    baseline=require_preflight();cfg=spec();fp=identity()['sha256'];pilot_info=read(AOUT/'pilot.json')
    assert pilot_info['status']=='PASS' and pilot_info['fingerprint']==fp
    progress={'status':'FITTING','fingerprint':fp,'spec':cfg,'folds':{}}
    if resume and (AOUT/'fit_progress.json').exists():
        progress=read(AOUT/'fit_progress.json')
        assert progress['fingerprint']==fp
    for i in range(20):
        budget.check();outer=f'{i:02d}'
        if outer in progress['folds']:
            record=progress['folds'][outer]
            assert digest(AOUT/'selected'/f'outer-{outer}.npz')==record['array_sha256']
            assert record['source']==baseline['sources'][outer]
            print('Reuse anchored completed FIT',outer,flush=True);continue
        d,source=load_fit(outer);require_source(outer,source)
        data=d['fit_data'];x=features(data);y=data['errors'];subjects=data['subjects'];rows=[];outputs={}
        for lam in cfg['lambda_grid']:
            raw={m:np.full(len(y),np.nan) for m in cfg['methods']}
            failures={m:[] for m in cfg['methods']}
            for j,fold in enumerate(d['risk_folds']):
                tr=np.isin(subjects,fold['train']);va=np.isin(subjects,fold['validation'])
                assert not np.any(tr&va)
                b=fit_group(outer,j,lam,x[tr],y[tr],subjects[tr],source,budget,resume)
                for m in cfg['methods']:
                    if b['models'][m]['status']=='OK':raw[m][va]=raw_predict(b,m,x[va])
                    else:failures[m].append({'fold':j,'status':b['models'][m]['status']})
            for m in cfg['methods']:
                if failures[m]:rows.append({'method':m,'lambda':lam,'status':'INVALID','failures':failures[m]});continue
                calibrated,mappings=cross_calibrate(raw[m],y,subjects,d['risk_folds'])
                grid=policies(calibrated,subjects)
                picked,_=choose(subset_policies(grid,20),calibrated,y,data,outer,cfg)
                dense,_=choose(grid,calibrated,y,data,outer,cfg)
                row={'method':m,'lambda':lam,'status':'OK','coverage':0 if picked is None else picked['coverage'],
                    'risk':None if picked is None else picked['risk'],'dense_coverage':0 if dense is None else dense['coverage'],
                    'constant_calibrators':sum(z['constant'] for z in mappings)}
                rows.append(row);outputs[m,lam]={'raw':raw[m],'calibrated':calibrated,'policies':grid}
            print(f'Anchored FIT {outer}: lambda={lam:g} complete',flush=True)
        selected={};payload={}
        for m in cfg['methods']:
            valid=[r for r in rows if r['method']==m and r['status']=='OK']
            if not valid:raise RuntimeError(f'No valid anchored FIT candidate {outer}/{m}; no incomplete aggregate allowed')
            row=min(valid,key=rank_candidate);selected[m]={**row,'policies':outputs[m,row['lambda']]['policies']}
            payload[m+'_raw']=outputs[m,row['lambda']]['raw']
            payload[m+'_calibrated']=outputs[m,row['lambda']]['calibrated']
        path=AOUT/'selected'/f'outer-{outer}.npz';arrays(path,**payload)
        progress['folds'][outer]={'source':source,'base_C':old_selection()['folds'][outer]['selected']['BCE']['C'],
            'selected':selected,'candidates':rows,'array_sha256':digest(path)}
        write(AOUT/'fit_progress.json',progress)
        print('Anchored FIT choices fixed',outer,flush=True)
    progress.update(status='ALL_FIT_SELECTIONS_FROZEN',sealed_at=utc())
    write(AOUT/'fit_selections.json',progress)
    for outer,r in progress['folds'].items():
        d,source=load_fit(outer);require_source(outer,source);data=d['fit_data'];x=features(data)
        for lam in sorted({z['lambda'] for z in r['selected'].values()}):
            b=fit_group(outer,'final',lam,x,data['errors'],data['subjects'],source,budget,resume)
            needed=[m for m in cfg['methods'] if r['selected'][m]['lambda']==lam]
            if any(b['models'][m]['status']!='OK' for m in needed):raise RuntimeError('Selected final refinement failed')
        print('Anchored final scorers fixed',outer,flush=True)
    write(AOUT/'fit_completed.json',{'status':'ALL_FINAL_SCORERS_FITTED','fingerprint':fp,
        'selections_sha256':digest(AOUT/'fit_selections.json'),'completed':utc()})


def evaluate(budget,resume):
    import joblib
    from stage1_analysis.selection_ablation import read_cert_labels
    require_preflight();cfg=spec();fp=identity()['sha256']
    selected=read(AOUT/'fit_selections.json');done=read(AOUT/'fit_completed.json')
    assert selected['status']=='ALL_FIT_SELECTIONS_FROZEN' and set(selected['folds'])=={f'{i:02d}' for i in range(20)}
    assert done['fingerprint']==fp and done['selections_sha256']==digest(AOUT/'fit_selections.json')
    receipts={};cells=[f'{m}|{q}' for m in cfg['methods'] for q in (20,100)]
    for outer,r in selected['folds'].items():
        budget.check();path=AOUT/'decisions'/f'outer-{outer}.npz';meta=path.with_suffix('.json')
        if meta.exists():
            old=read(meta)
            if not resume or old['fingerprint']!=fp or old['sha256']!=digest(path):raise RuntimeError('Incompatible saved anchored decision')
            receipts[outer]=old;continue
        d,source=load_fit(outer);require_source(outer,source)
        cert,outerdata=d['cert_data_blind'],d['outer_data_blind']
        labels,label_sources=read_cert_labels(PROJECT,cert,d['cert_subjects'],outer)
        errors=(cert['primary'].argmax(1)!=labels).astype(int)
        spath=AOUT/'selected'/f'outer-{outer}.npz'
        assert digest(spath)==r['array_sha256']
        with np.load(spath,allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
        scores=[];masks=[];raw_scores=[];info={}
        for m in cfg['methods']:
            chosen=r['selected'][m];mpath=path_for(outer,chosen['lambda'],'final')
            mr=read(mpath.with_suffix('.json'));assert mr['fingerprint']==fp and digest(mpath)==mr['sha256']
            b=joblib.load(mpath)
            mapping=fit_mapping(raw[m+'_raw'],d['fit_data']['errors'],d['fit_data']['subjects'])
            sc=apply_mapping(mapping,raw_predict(b,m,features(cert)))
            ro=raw_predict(b,m,features(outerdata));so=apply_mapping(mapping,ro);raw_scores.append(ro)
            for q in (20,100):
                rule,rule_rows=choose(subset_policies(chosen['policies'],q),sc,errors,cert,outer,cfg,minimum=50)
                masks.append(mask(so,rule,outerdata,outer,cfg));scores.append(so)
                info[f'{m}|{q}']={'rule':rule,'mapping':mapping,'lambda':chosen['lambda'],
                    'relative_parameter_change':b['models'][m]['diagnostics']['relative_parameter_change'],
                    'soft_training_coverage':b['models'][m]['diagnostics']['soft_coverage'],
                    'soft_training_risk':b['models'][m]['diagnostics']['soft_risk'],
                    'zero_fallback':rule is None}
        arrays(path,cells=np.asarray(cells),scores=np.column_stack(scores),masks=np.column_stack(masks),
            raw_scores=np.column_stack(raw_scores),subjects=outerdata['subjects'],predictions=outerdata['primary'].argmax(1),
            identities_json=json.dumps(outerdata['identities']))
        receipts[outer]={'status':'OUTER_DECISIONS_FROZEN','fingerprint':fp,'sha256':digest(path),
            'source':source,'cert_label_sources':label_sources,'contains_outer_labels':False,'cells':info,'completed':utc()}
        write(meta,receipts[outer]);print('Anchored CERT rule and OUTER decision frozen',outer,flush=True)
    write(AOUT/'decisions_sealed.json',{'status':'ALL_OUTER_DECISIONS_FROZEN','fingerprint':fp,
        'folds':receipts,'completed':utc()})
