import os
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'src'), str(ROOT.parent / 'AutoML_Claude')]
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np
from new_objective.bceanchor_upgrade_core import (
    spec, candidate_key, select_candidate, equal_norm_control, average_models,
    partition_subjects, counts_risk, array_hash,
)
from new_objective.bceanchor_upgrade_solver import fit_refinement
from new_objective.anchor_solver import fit_refinement as old_refinement
from new_objective.storage import target, read, write, digest
from new_objective.policies import exact
from stage1_analysis.acceptance import select_empirical_policy
from new_objective import bceanchor_upgrade as experiment


def candidate(parameter, q=.5, c=.5, risk=.04):
    return {'parameter': parameter, 'status': 'OK', 'picked': {
        'q': Fraction(str(q)), 'coverage': Fraction(str(c)),
        'risk': Fraction(str(risk)), 'supported': True}}


class UpgradeTests(unittest.TestCase):
    def test_exact_q_not_realized_rounding(self):
        a, b = candidate(.1, c=.5000001, risk=.045), candidate(1, c=.5, risk=.041)
        self.assertIs(select_candidate([a, b], 'BCEAnchor', False), a)
        self.assertIs(select_candidate([a, b], 'BCEAnchor', True), b)

    def test_anchor_and_sele_ties(self):
        rows = [candidate(p) for p in [.01, .1, 1, 10]]
        self.assertEqual(select_candidate(rows, 'BCEAnchor')['parameter'], 10)
        self.assertEqual(select_candidate(rows, 'SELE')['parameter'], .01)
        self.assertEqual(select_candidate(rows, 'BCE')['parameter'], .01)

    def test_picked_q_differs_from_max_feasible_q(self):
        rows = [candidate(1, q=q)['picked'] for q in [.5, .55]]
        picked = select_empirical_policy(rows, alpha=.05)
        self.assertEqual(exact(picked['q']), Fraction(1, 2))
        self.assertNotEqual(exact(picked['q']), max(exact(r['q']) for r in rows))

    def test_all_infeasible_and_invalid_are_distinct(self):
        rows = [{'parameter': p, 'picked': None, 'status': 'OK'} for p in [.01, .1, 1, 10]]
        self.assertEqual(select_candidate(rows, 'BCEAnchor')['parameter'], 10)
        self.assertEqual(select_candidate(rows, 'SELE')['parameter'], .01)
        rows[0]['status'] = 'INVALID'
        with self.assertRaises(RuntimeError):
            select_candidate(rows, 'SELE')

    def test_any_feasible_precedes_empty(self):
        rows = [{'parameter': 10, 'picked': None}, candidate(.01, q=.05, c=.001)]
        self.assertIs(select_candidate(rows, 'BCEAnchor'), rows[1])

    def test_partition_is_balanced_and_label_free(self):
        subjects = [f'{i:02d}' for i in range(12)]
        a = partition_subjects(subjects, '00', 0, spec())
        b = partition_subjects(list(reversed(subjects)), '00', 0, spec())
        self.assertEqual(a, b)
        self.assertEqual(sorted(s for g in a for s in g['validation']), subjects)
        self.assertTrue(all(len(g['train']) == 9 and not set(g['train']) & set(g['validation']) for g in a))
        self.assertTrue(all(sum(s in g['train'] for g in a) == 3 for s in subjects))

    def test_equal_norm_is_not_projection_or_clipped(self):
        base = np.array([1., 1.]); full = base + np.array([.1, 0.])
        mean = base + np.array([0., .3])
        control, info = equal_norm_control(base, full, mean)
        self.assertAlmostEqual(info['coefficient'], 3.)
        self.assertEqual(info['kind'], 'expand')
        self.assertAlmostEqual(np.linalg.norm(control-base), np.linalg.norm(mean-base))

    def test_zero_full_direction_is_na(self):
        base = np.ones(3)
        control, info = equal_norm_control(base, base, base+.1)
        self.assertIsNone(control)
        self.assertEqual(info['status'], 'UNDEFINED_ZERO_FULL_DIRECTION')

    def test_linear_parameter_and_score_averages_identical(self):
        rng = np.random.default_rng(42); x = rng.normal(size=(15, 4))
        members = [{'theta': rng.normal(size=4), 'intercept': float(i), 'status': 'OK'} for i in range(4)]
        mean = average_models(members)
        np.testing.assert_allclose(x@mean['theta']+mean['intercept'],
            np.mean([x@m['theta']+m['intercept'] for m in members], axis=0), atol=1e-14)

    def test_failed_member_never_dropped(self):
        good = {'status': 'OK', 'theta': np.ones(2), 'intercept': 0.}
        with self.assertRaises(RuntimeError):
            average_models([good, good, good])
        with self.assertRaises(RuntimeError):
            average_models([good, good, good, {**good, 'status': 'NOT_CONVERGED'}])

    def test_new_scale_adapter_preserves_old_default_exactly(self):
        x = np.linspace(-2, 2, 80).reshape(-1, 1)
        y = (x[:, 0] > 1).astype(int)
        subjects = np.asarray([f's{i//10}' for i in range(80)])
        base = {'theta': np.array([2.]), 'intercept': -.3}
        old = old_refinement(x, y, subjects, base, 1., 'BCEAnchor', spec())
        new = fit_refinement(x, y, subjects, base, 1., 'BCEAnchor', spec())
        self.assertEqual(old['status'], 'OK')
        np.testing.assert_array_equal(old['theta'], new['theta'])
        self.assertEqual(old['training_threshold'], new['training_threshold'])
        frozen = fit_refinement(x, y, subjects, base, 1., 'BCEAnchor', spec(), fixed_scale=3.)
        self.assertEqual(frozen['score_scale'], 3.)
        np.testing.assert_array_equal(frozen['beta0'], base['theta']/3.)

    def test_bad_fixed_scale_rejected(self):
        with self.assertRaises(ValueError):
            fit_refinement(np.ones((4, 1)), np.array([0, 0, 1, 1]), np.array(['a']*4),
                {'theta': np.ones(1), 'intercept': 0}, 1., 'BCEAnchor', spec(), fixed_scale=0)

    def test_counts_are_ratio_not_average_risk(self):
        stat = counts_risk([{'n': 100, 'accepted': 10, 'errors': 1},
                            {'n': 200, 'accepted': 100, 'errors': 1}])
        self.assertEqual(stat['risk'], Fraction(1, 40))
        self.assertEqual(stat['coverage'], Fraction(3, 10))

    def test_zero_acceptance_remains_in_coverage(self):
        s = counts_risk([{'n': 100, 'accepted': 0, 'errors': 0},
                         {'n': 100, 'accepted': 20, 'errors': 1}])
        self.assertEqual(s['coverage'], Fraction(1, 10))
        self.assertEqual(s['risk'], Fraction(1, 20))
        self.assertIsNone(counts_risk([{'n': 2, 'accepted': 0, 'errors': 0}])['risk'])

    def test_no_true_labels_in_frozen_inference_api(self):
        import inspect
        self.assertFalse({'y', 'errors', 'labels'} & set(inspect.signature(experiment.predict_nested).parameters))
        self.assertEqual(set(inspect.signature(experiment.nested_pipeline).parameters),
                         {'train', 'context', 'fold_number', 'source', 'store'})

    def test_fit_data_subsetting_excludes_validation(self):
        data = {'x': np.arange(24).reshape(12, 2), 'errors': np.zeros(12),
                'subjects': np.array(['a']*6+['b']*6), 'identities': list(range(12))}
        a = experiment.subset(data, data['subjects'] == 'a')
        data['errors'][data['subjects'] == 'b'] = 1
        b = experiment.subset(data, data['subjects'] == 'a')
        self.assertEqual(array_hash(a['x'], a['errors'], a['subjects']),
                         array_hash(b['x'], b['errors'], b['subjects']))

    def test_checkpoint_reuse_and_rejection(self):
        from new_objective.bceanchor_upgrade import Store
        cache = ROOT/'outputs/bceanchor_upgrade/cache/tests'
        cache.mkdir(parents=True, exist_ok=True)
        class Budget:
            def check(self): pass
        # The checkpoint test does not inspect historical model selections.
        # Keep that unrelated, unavailable training state synthetic in a public checkout.
        with patch.object(experiment, 'read', return_value={}):
            store = Store(Budget(), True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            p = Path(tmp)/'model.joblib'; calls=[]
            def fit():
                calls.append(1)
                return {'status': 'OK', 'theta': np.ones(2)}
            a = store.cached(p, {'key': 1}, fit)
            b = store.cached(p, {'key': 1}, fit)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(a['theta'], b['theta'])
            with self.assertRaises(RuntimeError):
                store.cached(p, {'key': 2}, fit)
            meta = read(p.with_suffix('.json')); meta['fingerprint'] = 'changed'
            write(p.with_suffix('.json'), meta)
            with self.assertRaises(RuntimeError):
                store.cached(p, {'key': 1}, fit)

    def test_output_containment(self):
        with self.assertRaises(ValueError):
            target(ROOT.parent/'upgrade_must_not_exist.json')

    def test_sele_reg_uses_parent_epochs_not_subject_count(self):
        parent_n, member_n, c = 12000, 9000, .1
        self.assertNotEqual(1/(parent_n*c), 1/(member_n*c))
        self.assertNotEqual(1/(parent_n*c), 1/(12*c))

    def test_rule_builder_explicitly_supports_twelve_subjects(self):
        from automl.grids import dense_candidate_policies, grid_fractions
        for n_subjects in (12, 15):
            subjects = np.repeat(np.arange(n_subjects).astype(str), 4)
            scores = np.linspace(-1., 1., len(subjects))
            data = {'subjects': subjects, 'errors': np.zeros(len(subjects))}
            with patch.object(experiment, 'choose', return_value=(None, [])):
                with patch.object(experiment, 'dense_candidate_policies',
                                  wraps=dense_candidate_policies) as build:
                    experiment.rule_statistics(scores, data, '00', spec())
                    self.assertEqual(build.call_args.kwargs['n_subjects'], n_subjects)
            if n_subjects == 15:
                self.assertEqual(dense_candidate_policies(scores, subjects, grid_fractions(20)),
                    dense_candidate_policies(scores, subjects, grid_fractions(20), n_subjects=15))

    def test_nested_orchestration_is_invariant_to_heldout_labels(self):
        """Exercise the full selection/refit/calibration/prediction call graph.

        Deterministic inexpensive toy estimators replace numerical fitting, while
        the production partition, scaler, cross-calibration, policy construction,
        selection, parent reuse, final mapping and frozen inference remain active.
        """
        from sklearn.preprocessing import StandardScaler
        from new_objective.bceanchor_upgrade_core import object_hash
        rng = np.random.default_rng(890)
        subjects = np.repeat([f's{i:02d}' for i in range(15)], 8)
        data = {'x': rng.normal(size=(120, 37)), 'subjects': subjects,
                'errors': (np.arange(120) % 13 == 0).astype(int),
                'identities': [f'toy|{s}|r|{i}' for i, s in enumerate(subjects)]}
        fold = {'train': sorted(set(subjects))[:12], 'validation': sorted(set(subjects))[12:]}
        tr = np.isin(subjects, fold['train'])
        parent = experiment.subset(data, tr)
        scaler = StandardScaler().fit(parent['x'])
        xx = scaler.transform(parent['x'])

        def bce(x, y, w, reg, cfg):
            return {'theta': x.T@(w*(y-.1))/(1+reg), 'intercept': float(w@y), 'status': 'OK'}

        def sele(x, y, w, reg, initial, cfg, check=None):
            return {'theta': initial+.1*(x.T@(w*y))/(1+reg), 'intercept': 0., 'status': 'OK'}

        def refinement(x, y, ss, base, lam, method, cfg, check=None):
            from new_objective.losses import weights
            return {'theta': base['theta']+.02*(x.T@(weights(ss)*y))/(1+lam),
                    'intercept': base['intercept'], 'status': 'OK'}

        class FakeStore:
            fp = 'toy-fingerprint'
            old = {'folds': {'00': {'selected': {'BCE': {'C': .1}}}}}
            budget = type('Budget', (), {'check': lambda self: None})()
            cross_calibrate = experiment.Store.cross_calibrate
            def __init__(self):
                self.pipelines = []
                self.calibration_subjects = []
            def cached(self, path, key, function, kind='risk'):
                value = function()
                if kind == 'pipeline':
                    self.pipelines.append(value)
                    experiment.model_save(path, value)
                return value
            def calibrate(self, path, raw, y, ss):
                self.calibration_subjects.append(set(ss))
                return {'mean': float(np.mean(raw)), 'scale': max(float(np.std(raw)), .01),
                        'slope': 1., 'intercept': float(np.mean(y)), 'success': True}
            def legacy(self, context, fold_number, parameter, source, anchor=False):
                from new_objective.losses import weights
                c = .1 if anchor else parameter
                w = weights(parent['subjects'])
                base = bce(xx, parent['errors'], w, 1/(len(xx)*c), spec())
                models = {'BCE': base, 'SELE': sele(xx, parent['errors'], w,
                          1/(len(xx)*c), base['theta'], spec())}
                if anchor:
                    for method in ('BCEAnchor', 'FixedScale'):
                        models[method] = refinement(xx, parent['errors'], parent['subjects'],
                            base, parameter, method, spec())
                return {'mean': scaler.mean_, 'scale': scaler.scale_,
                        'train_subjects': fold['train'], 'models': models}

        cache = ROOT/'outputs/bceanchor_upgrade/cache/tests'
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            results = []
            for iteration in range(2):
                store = FakeStore()
                if iteration:
                    data['errors'][~tr] = 1-data['errors'][~tr]
                out = Path(temporary)/str(iteration)
                with patch.object(experiment, 'UOUT', out), patch.object(experiment, 'fit_bce', bce), \
                     patch.object(experiment, 'fit_sele', sele), patch.object(experiment, 'fit_refinement', refinement):
                    experiment.nested_fold('00', 0, data, fold, {}, store)
                with np.load(out/'nested/context-00/fold-0/predictions.npz') as z:
                    results.append((object_hash(store.pipelines[0]), z['raw'].copy(),
                                    z['calibrated'].copy(), z['decisions'].copy()))
                self.assertTrue(all(ss <= set(fold['train']) for ss in store.calibration_subjects))
                self.assertEqual(set(store.pipelines[0]['training_subjects']), set(fold['train']))
                self.assertGreater(len(store.calibration_subjects), 50)
            self.assertEqual(results[0][0], results[1][0])
            for first, second in zip(results[0][1:], results[1][1:]):
                np.testing.assert_array_equal(first, second)

    def test_patient_repetitions_are_averaged_before_ratio(self):
        import pandas as pd
        from new_objective.bceanchor_upgrade_report import aggregate_people, ratio
        rows = []
        for subject, measurements in [('a', [(100, 10, 1), (200, 100, 1)]),
                                      ('b', [(100, 50, 0), (200, 100, 2)])]:
            for j, (n, a, k) in enumerate(measurements):
                rows.append({'branch': 'toy', 'cell': 'x', 'evaluation': 'quota', 'subject': subject,
                    'context': str(j), 'n': n, 'accepted': a, 'errors': k,
                    'coverage': a/n, 'error_mass': k/n})
        people = aggregate_people(pd.DataFrame(rows))
        np.testing.assert_allclose(people.c_bar, [.3, .5])
        np.testing.assert_allclose(people.e_bar, [.0075, .005])
        self.assertAlmostEqual(float(ratio(people.c_bar.to_numpy(), people.e_bar.to_numpy())), .015625)
        self.assertNotAlmostEqual(.015625, np.mean([.1, .01, 0., .02]))

    def test_incomplete_patient_matrix_does_not_report_complete_summary(self):
        import pandas as pd
        from new_objective.bceanchor_upgrade_report import summarize
        people = pd.DataFrame([{'branch': 'toy', 'cell': 'x', 'evaluation': 'locked_rule',
            'subject': 'one', 'c_bar': .5, 'e_bar': .02, 'appearances': 15,
            'repeated_accepted': 50, 'repeated_errors': 2, 'records': 15, 'zero_accept_records': 0}])
        expected = {f's{i:02d}': [f'{j:02d}' for j in range(15)] for i in range(20)}
        summary = summarize(people, expected)
        self.assertTrue(np.isnan(summary.iloc[0].coverage))
        self.assertNotEqual(summary.iloc[0].status, 'COMPLETE')

    def test_complete_patient_matrix_allows_unequal_frozen_role_frequencies(self):
        import pandas as pd
        from new_objective.bceanchor_upgrade_report import aggregate_people, summarize
        expected = {'a': ['00', '01'], 'b': ['00']}
        rows = [{'branch': 'toy', 'cell': 'x', 'evaluation': 'quota',
            'subject': subject, 'context': context, 'n': 100, 'accepted': 50,
            'errors': 2, 'coverage': .5, 'error_mass': .02}
            for subject, contexts in expected.items() for context in contexts]
        people = aggregate_people(pd.DataFrame(rows))
        summary = summarize(people, expected)
        self.assertEqual(summary.iloc[0].status, 'COMPLETE')
        self.assertAlmostEqual(summary.iloc[0].R_SE, .04)
        altered = people.copy()
        altered.loc[altered.subject == 'a', 'context_signature'] = '["00","02"]'
        self.assertNotEqual(summarize(altered, expected).iloc[0].status, 'COMPLETE')

    def test_duplicate_patient_records_do_not_pass_role_completeness(self):
        import pandas as pd
        from new_objective.bceanchor_upgrade_report import aggregate_people, summarize
        row = {'branch': 'toy', 'cell': 'x', 'evaluation': 'quota',
            'subject': 'a', 'context': '00', 'n': 100, 'accepted': 50,
            'errors': 2, 'coverage': .5, 'error_mass': .02}
        people = aggregate_people(pd.DataFrame([row, row]))
        self.assertNotEqual(summarize(people, {'a': ['00']}).iloc[0].status, 'COMPLETE')

    def test_oracle_keeps_entire_tied_groups(self):
        from new_objective.bceanchor_upgrade_report import hard_oracle
        score = np.array([0., 0., 1., 1.])
        errors = np.array([0, 0, 1, 0])
        accepted = hard_oracle(score, errors, np.array(['a']*4), .05)
        np.testing.assert_array_equal(accepted, [True, True, False, False])
        np.testing.assert_array_equal(hard_oracle(score, np.ones(4), np.array(['a']*4), .05),
                                      np.zeros(4, dtype=bool))


if __name__ == '__main__':
    unittest.main()
