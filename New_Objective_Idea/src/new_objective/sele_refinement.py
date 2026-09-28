"""Conditional FIT diagnostic of SELE-centered selective-risk refinement.

The reference C is inherited from an earlier FIT-wide choice. This deliberately
does not claim nested CV, independent evidence, or a CERT-deployable policy.
"""
from __future__ import annotations
import hashlib
import json
import time
import numpy as np
from .storage import ROOT, PROJECT, OUT, read, write, digest, arrays, model_save, utc
from .experiment import load_fit, require_source, features, raw_predict
from .anchor_solver import fit_refinement

SOUT = OUT / 'sele_refinement_fit'


def spec():
    return read(ROOT / 'sele_refinement_protocol.json')


def identity():
    paths = [ROOT / 'sele_refinement_protocol.json', ROOT / 'run_sele_refinement.py']
    paths += [ROOT / 'src/new_objective' / name for name in
              ('sele_refinement.py', 'anchor_solver.py', 'losses.py', 'storage.py',
               'experiment.py')]
    paths += [PROJECT / 'src/stage1_analysis' / name for name in
              ('risk_features.py', 'selection_ablation.py', 'acceptance.py')]
    hashes = {p.relative_to(PROJECT).as_posix(): digest(p) for p in paths}
    return {'sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            'files': hashes}


def load_reference(outer, fold, source):
    import joblib
    selection = read(OUT / 'fit_selections.json')
    c = selection['folds'][outer]['selected']['SELE']['C']
    path = OUT / 'fits' / f'outer-{outer}' / f'c-{c:g}' / f'fold-{fold}.joblib'
    receipt = read(path.with_suffix('.json'))
    if (receipt['fingerprint'] != selection['fingerprint'] or receipt['source'] != source
            or receipt['model_sha256'] != digest(path)):
        raise RuntimeError('SELE reference provenance mismatch')
    bundle = joblib.load(path)
    if bundle['models']['SELE']['status'] != 'OK':
        raise RuntimeError('SELE reference is not a converged model')
    return bundle, {'path': path.relative_to(ROOT).as_posix(), 'C': c,
                    'sha256': receipt['model_sha256']}


def fit_data_only(outer):
    """Only FIT records and its risk folds leave this adapter."""
    development, source = load_fit(outer)
    require_source(outer, source)
    return development['fit_data'], development['risk_folds'], source


def preflight(budget, resume=False):
    if (SOUT / 'preflight.json').exists():
        require_preflight()
        print('SELE refinement preflight reused', flush=True)
        return
    protected = {}
    sources = {}
    for i in range(20):
        budget.check()
        outer = f'{i:02d}'
        data, folds, source = fit_data_only(outer)
        assert len(set(data['subjects'])) == 15 and len(folds) == 5
        assert np.array_equal(data['errors'], data['primary'].argmax(1) != data['labels'])
        seen = []
        for j, fold in enumerate(folds):
            base, info = load_reference(outer, j, source)
            assert set(base['train_subjects']) == set(fold['train'])
            assert not set(fold['train']) & set(fold['validation'])
            assert len(fold['train']) == 12 and len(fold['validation']) == 3
            seen.extend(fold['validation'])
            protected[(ROOT / info['path']).relative_to(PROJECT).as_posix()] = info['sha256']
        assert sorted(seen) == sorted(set(data['subjects']))
        sources[outer] = source
    for path in [OUT / 'fit_selections.json', ROOT / 'anchor_protocol.json',
                 ROOT / 'cross_setting_protocol.json', ROOT / 'src/new_objective/anchor_solver.py']:
        protected[path.relative_to(PROJECT).as_posix()] = digest(path)
    for path in (OUT / 'tables').glob('*.csv'):
        if not path.name.startswith('sele_fit_'):
            protected[path.relative_to(PROJECT).as_posix()] = digest(path)
    protected.update(read(OUT / 'preflight.json')['critical_hashes'])
    write(SOUT / 'preflight.json', {'status': 'PASS', 'created': utc(), 'identity': identity(),
          'spec': spec(), 'sources': sources, 'protected': protected,
          'base_selection_sha256': digest(OUT / 'fit_selections.json'),
          'reference_cv_models': 100, 'no_new_cert_or_outer_outcomes': True})
    print('Preflight PASS: 100 saved SELE CV models, 20 FIT contexts', flush=True)


def require_preflight():
    old = read(SOUT / 'preflight.json')
    if old['identity']['sha256'] != identity()['sha256']:
        raise RuntimeError('Changed code/spec; resume refused')
    if old['base_selection_sha256'] != digest(OUT / 'fit_selections.json'):
        raise RuntimeError('Reference selection changed')
    return old


def fit_group(outer, fold, lam, x, y, subjects, source, budget, resume):
    import joblib
    path = SOUT / 'fits' / f'outer-{outer}' / f'lambda-{lam:g}' / f'fold-{fold}.joblib'
    base, info = load_reference(outer, fold, source)
    fingerprint = identity()['sha256']
    if path.with_suffix('.json').exists():
        receipt = read(path.with_suffix('.json'))
        if (not resume or receipt['fingerprint'] != fingerprint or receipt['base'] != info
                or receipt['sha256'] != digest(path)):
            raise RuntimeError('Incompatible SELE refinement checkpoint')
        return joblib.load(path)
    assert set(subjects) == set(base['train_subjects'])
    xx = (x - base['mean']) / base['scale']
    models = {}
    started = time.monotonic()
    for method in spec()['methods']:
        # The frozen solver names the reference penalty BCEAnchor; its equations
        # accept any linear reference. Pass the saved SELE parameters explicitly.
        internal = 'BCEAnchor' if method == 'SELEAnchor' else 'FixedScale'
        models[method] = fit_refinement(xx, y, subjects, base['models']['SELE'],
                                       lam, internal, spec(), budget.check)
    result = {'mean': base['mean'], 'scale': base['scale'], 'models': models,
              'train_subjects': sorted(set(subjects)), 'base': info, 'lambda': lam,
              'seconds': time.monotonic() - started}
    model_save(path, result)
    write(path.with_suffix('.json'), {'fingerprint': fingerprint, 'base': info,
          'sha256': digest(path), 'source': source, 'seconds': result['seconds'],
          'statuses': {m: r['status'] for m, r in models.items()}})
    if any(r['status'] != 'OK' for r in models.values()):
        raise RuntimeError('Refinement numerical failure; do not drop a failed cell')
    return result


def pilot(budget, resume=False):
    preflight(budget, resume)
    cfg = spec()
    outer, j, lam = (cfg['pilot'][key] for key in ('outer', 'fold', 'lambda'))
    data, folds, source = fit_data_only(outer)
    tr = np.isin(data['subjects'], folds[j]['train'])
    result = fit_group(outer, j, lam, features(data)[tr], data['errors'][tr],
                       data['subjects'][tr], source, budget, resume)
    estimate = result['seconds'] * 400 * 1.5
    available = read(ROOT / 'protocol.json')['fit_budget_seconds'] - budget.state['used_seconds']
    status = 'PASS' if estimate < available else 'BUDGET_ESTIMATE_EXCEEDED'
    write(SOUT / 'pilot.json', {'status': status, 'fingerprint': identity()['sha256'],
          'seconds': result['seconds'], 'estimated_full_fit_seconds': estimate,
          'remaining_fit_budget_seconds': available,
          'statuses': {m: r['status'] for m, r in result['models'].items()}})
    print('Pilot', status, round(result['seconds'], 2), 's; estimated',
          round(estimate / 60, 1), 'min', flush=True)
    if status != 'PASS':
        raise TimeoutError('Pilot estimate exceeds remaining cumulative budget')


def quota_mask(scores, subjects, tie_keys, quota):
    """No labels: accept exactly floor(q*n_s), with common deterministic ties."""
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or not 0 <= quota <= 1:
        raise ValueError('Invalid scores/quota')
    result = np.zeros(len(scores), dtype=bool)
    for subject in np.unique(subjects):
        ix = np.flatnonzero(subjects == subject)
        order = np.lexsort((tie_keys[ix], scores[ix]))
        result[ix[order[:int(np.floor(quota * len(ix)))]]] = True
    return result


def tie_keys(identities, outer):
    from stage1_analysis.acceptance import tie_uniform
    cfg = spec()
    result = []
    for item in identities:
        value = tie_uniform(item, method_id=cfg['tie_method_id'],
                            protocol_hash=cfg['protocol_hash'], outer=outer)
        result.append(f'{int(value * 2**256):064x}')
    return np.asarray(result)


def hard_oracle(scores, errors, subjects, alpha):
    """Diagnostic upper envelope using all full tie groups, not a usable policy.

    Weight each subject equally; do not stop at the first infeasible threshold,
    since accepted risk need not be monotone in the threshold.
    """
    from .losses import weights
    w = weights(subjects)
    order = np.argsort(scores, kind='stable')
    s = np.asarray(scores)[order]
    c = np.cumsum(w[order])
    k = np.cumsum(w[order] * np.asarray(errors)[order])
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    valid = ends[k[ends] <= alpha * c[ends] + 1e-12]
    if not len(valid):
        return {'coverage': 0., 'risk': None}
    index = valid[-1]
    return {'coverage': float(c[index]), 'risk': float(k[index] / c[index])}


def fit(budget, resume=False):
    baseline = require_preflight()
    cfg, fp = spec(), identity()['sha256']
    pil = read(SOUT / 'pilot.json')
    assert pil['status'] == 'PASS' and pil['fingerprint'] == fp
    progress = {'status': 'FITTING', 'fingerprint': fp, 'contexts': {}, 'spec': cfg}
    if resume and (SOUT / 'progress.json').exists():
        progress = read(SOUT / 'progress.json')
        assert progress['fingerprint'] == fp
    cells = [('SELE', None)] + [(m, lam) for lam in cfg['lambda_grid'] for m in cfg['methods']]
    for i in range(20):
        outer = f'{i:02d}'
        budget.check()
        if outer in progress['contexts']:
            old = progress['contexts'][outer]
            assert digest(SOUT / 'scores' / f'outer-{outer}.npz') == old['sha256']
            print('Reuse complete FIT context', outer, flush=True)
            continue
        data, folds, source = fit_data_only(outer)
        assert source == baseline['sources'][outer]
        x, y, subjects = features(data), data['errors'], np.asarray(data['subjects'])
        scores = np.full((len(y), len(cells)), np.nan)
        fold_ids = np.full(len(y), -1, dtype=int)
        diagnostics = []
        for j, fold in enumerate(folds):
            tr, va = (np.isin(subjects, fold[key]) for key in ('train', 'validation'))
            base, info = load_reference(outer, j, source)
            training = {'SELE': raw_predict(base, 'SELE', x[tr])}
            scores[va, 0] = raw_predict(base, 'SELE', x[va])
            fold_ids[va] = j
            oracle = hard_oracle(training['SELE'], y[tr], subjects[tr], cfg['alpha'])
            diagnostics.append({'outer': outer, 'fold': j, 'method': 'SELE', 'lambda': None,
                                'train_hard_oracle': oracle, 'relative_parameter_change': 0.})
            for lam in cfg['lambda_grid']:
                bundle = fit_group(outer, j, lam, x[tr], y[tr], subjects[tr], source, budget, resume)
                for method in cfg['methods']:
                    index = cells.index((method, lam))
                    scores[va, index] = raw_predict(bundle, method, x[va])
                    trained = raw_predict(bundle, method, x[tr])
                    diag = bundle['models'][method]['diagnostics']
                    diagnostics.append({'outer': outer, 'fold': j, 'method': method,
                        'lambda': lam, 'train_hard_oracle': hard_oracle(trained, y[tr], subjects[tr], cfg['alpha']),
                        'relative_parameter_change': diag['relative_parameter_change'],
                        'soft_training_coverage': diag['soft_coverage'], 'soft_training_risk': diag['soft_risk']})
            print(f'SELE refinement FIT context {outer}, risk fold {j+1}/5 finished', flush=True)
        assert np.isfinite(scores).all() and np.all(fold_ids >= 0)
        with np.load(OUT / 'selected' / f'outer-{outer}.npz', allow_pickle=False) as old:
            assert np.array_equal(scores[:, 0], old['SELE_raw']), 'Reference OOF scores changed'
        path = SOUT / 'scores' / f'outer-{outer}.npz'
        arrays(path, scores=scores, subjects=subjects, errors=y, fold_ids=fold_ids,
               ties=tie_keys(data['identities'], outer),
               cell_names=np.array([m + ('' if lam is None else f'|{lam:g}') for m, lam in cells]))
        meta = {'source': source, 'sha256': digest(path), 'reference_C': info['C'],
                'diagnostics': diagnostics}
        write(path.with_suffix('.json'), meta)
        progress['contexts'][outer] = meta
        write(SOUT / 'progress.json', progress)
    progress.update(status='FIT_DIAGNOSTIC_COMPLETE', completed=utc(),
                    no_cert_or_outer_outcomes=True, no_final_model_fit=True)
    write(SOUT / 'completion.json', progress)
    print('All 20 FIT contexts complete; no CERT/OUTER evaluation', flush=True)


def replay(budget, resume=False):
    import pandas as pd
    from stage1_analysis.acceptance import subject_equal_risk
    pre = require_preflight()
    done = read(SOUT / 'completion.json')
    assert done['status'] == 'FIT_DIAGNOSTIC_COMPLETE'
    assert set(done['contexts']) == {f'{i:02d}' for i in range(20)}
    assert done['fingerprint'] == identity()['sha256']
    subject_rows, context_rows, diagnostics = [], [], []
    cfg = spec()
    for outer, saved in done['contexts'].items():
        budget.check()
        path = SOUT / 'scores' / f'outer-{outer}.npz'
        assert digest(path) == saved['sha256']
        with np.load(path, allow_pickle=False) as z:
            scores, y, subjects, ties = (z[k].copy() for k in ('scores', 'errors', 'subjects', 'ties'))
            fold_ids, names = z['fold_ids'].copy(), z['cell_names'].tolist()
        for index, name in enumerate(names):
            method, lam = (name.split('|') + [None])[:2] if '|' in name else (name, None)
            lam = None if lam is None else float(lam)
            for quota in cfg['diagnostic_quotas']:
                accepted = quota_mask(scores[:, index], subjects, ties, quota)
                metric = subject_equal_risk(accepted, y, subjects)
                context_rows.append({'outer': outer, 'method': method, 'lambda': lam,
                    'quota': quota, 'coverage': metric['coverage_float'], 'R_SE': metric['risk_float'],
                    'accepted': int(accepted.sum()), 'errors': int(y[accepted].sum())})
                for subject, r in metric['per_subject'].items():
                    subject_rows.append({'outer': outer, 'subject': subject, 'method': method,
                        'lambda': lam, 'quota': quota, 'n': r['eligible'], 'accepted': r['accepted'],
                        'errors': r['accepted_errors'], 'risk': r['accepted_errors']/r['accepted']})
            for j in range(5):
                va = fold_ids == j
                oracle = hard_oracle(scores[va, index], y[va], subjects[va], cfg['alpha'])
                training = next(r for r in saved['diagnostics'] if r['fold'] == j
                                and r['method'] == method and r['lambda'] == lam)
                diagnostics.append({'outer': outer, 'fold': j, 'method': method, 'lambda': lam,
                    'train_oracle_coverage': training['train_hard_oracle']['coverage'],
                    'validation_oracle_coverage': oracle['coverage'],
                    'relative_parameter_change': training['relative_parameter_change'],
                    'soft_training_coverage': training.get('soft_training_coverage')})
    context = pd.DataFrame(context_rows)
    people = pd.DataFrame(subject_rows)
    diag = pd.DataFrame(diagnostics)
    summary = context.groupby(['method', 'lambda', 'quota'], dropna=False).agg(
        mean_R_SE=('R_SE', 'mean'), mean_coverage=('coverage', 'mean'), contexts=('outer', 'nunique')).reset_index()
    baseline = summary.loc[summary.method == 'SELE'].set_index('quota').mean_R_SE.to_dict()
    summary['delta_R_SE_pp'] = 100*(summary.mean_R_SE - summary.quota.map(baseline))
    diagnostic_summary = diag.groupby(['method', 'lambda'], dropna=False).agg(
        mean_train_oracle_coverage=('train_oracle_coverage', 'mean'),
        mean_validation_oracle_coverage=('validation_oracle_coverage', 'mean'),
        median_relative_parameter_change=('relative_parameter_change', 'median')).reset_index()
    summary = summary.merge(diagnostic_summary, on=['method', 'lambda'])
    by_person = people.groupby(['subject', 'method', 'lambda', 'quota'], dropna=False).risk.mean().reset_index()
    base_person = by_person[by_person.method == 'SELE'][['subject', 'quota', 'risk']].rename(columns={'risk':'SELE_risk'})
    by_person = by_person.merge(base_person, on=['subject', 'quota'])
    by_person['delta_risk_pp'] = 100*(by_person.risk-by_person.SELE_risk)
    frames = {'summary': summary, 'contexts': context, 'per_subject': people,
              'subject_means': by_person, 'fit_transfer': diag}
    hashes = {}
    for name, frame in frames.items():
        path = OUT / 'tables' / f'sele_fit_{name}.csv'
        frame.to_csv(path, index=False, lineterminator='\n', float_format='%.17g')
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    changed = [p for p, h in pre['protected'].items() if digest(PROJECT / p) != h]
    if changed:
        raise RuntimeError('Protected inputs/results changed: '+repr(changed))
    write(SOUT / 'replay_checks.json', {'status': 'PASS', 'contexts': 20, 'risk_folds': 100,
          'models': 800, 'csv_sha256': hashes, 'protected_files_unchanged': len(pre['protected']),
          'no_cert_or_outer_outcomes': True, 'no_new_hyperparameter_selection': True})
    print(summary[(summary.quota == .5) & ((summary.method == 'SELE') | (summary['lambda'] == 1))].to_string(index=False), flush=True)
