import sys
import unittest
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'src'), str(ROOT.parent / 'AutoML_Claude')]
from new_objective.sele_refinement import hard_oracle, quota_mask, tie_keys, spec
from new_objective.anchor_solver import anchored_oracle
from new_objective.losses import weights


class SeleRefinementTests(unittest.TestCase):
    def test_exact_subject_quota_and_common_ties(self):
        subjects = np.array(['a'] * 5 + ['b'] * 4)
        risk = np.zeros(9)
        keys = np.array([f'{i:064x}' for i in range(8, -1, -1)])
        mask = quota_mask(risk, subjects, keys, .5)
        self.assertEqual(np.flatnonzero(mask).tolist(), [3, 4, 7, 8])
        np.testing.assert_array_equal(mask, quota_mask(risk + 7, subjects, keys, .5))

    def test_ties_match_frozen_identity_hash(self):
        from stage1_analysis.acceptance import tie_uniform
        items = ['d|s|r|0', 'd|s|r|1']
        cfg = spec()
        expected = [int(tie_uniform(i, method_id=cfg['tie_method_id'],
                                   protocol_hash=cfg['protocol_hash'], outer='00')*2**256)
                    for i in items]
        self.assertEqual([int(k, 16) for k in tie_keys(items, '00')], expected)

    def test_monotone_score_mapping_preserves_quota(self):
        s = np.array(['a']*6)
        x = np.array([.7, .1, -.3, .2, .2, 1.])
        ties = np.array(list('abcdef'))
        np.testing.assert_array_equal(quota_mask(x, s, ties, .5),
                                      quota_mask(3*x+4, s, ties, .5))

    def test_oracle_uses_subject_weights_and_full_ties(self):
        scores = np.array([0., 0., 1., 2., 3., 4.])
        errors = np.array([1, 0, 0, 0, 0, 0])
        subjects = np.array(['a', 'a', 'b', 'b', 'b', 'b'])
        # First tied group has mass 1/2 and error mass 1/4. The final risk is 1/4.
        r = hard_oracle(scores, errors, subjects, .26)
        self.assertAlmostEqual(r['coverage'], 1.)
        self.assertAlmostEqual(r['risk'], .25)

    def test_oracle_does_not_stop_at_first_risk_violation(self):
        r = hard_oracle(np.arange(10), np.array([1]+[0]*9), np.array(['a']*10), .11)
        self.assertAlmostEqual(r['coverage'], 1.)

    def test_oracle_empty_and_all_correct(self):
        for y, c in [(np.ones(6), 0.), (np.zeros(6), 1.)]:
            r = hard_oracle(np.arange(6), y, np.array(['a']*6), .05)
            self.assertAlmostEqual(r['coverage'], c)
        self.assertIsNone(hard_oracle(np.arange(6), np.ones(6), np.array(['a']*6), .05)['risk'])

    def test_reference_center_accepts_non_bce_parameters(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(20, 3))
        theta = np.array([.7, -.2, .9])
        v = np.r_[theta, -.5]
        w = weights(np.array(['a']*10+['b']*10))
        y = np.array([0]*17+[1]*3)
        anchored = anchored_oracle(v, x, y, w, .2, 1., theta, .05)
        free = anchored_oracle(v, x, y, w, .2, 0., np.zeros(3), .05)
        self.assertEqual(anchored['loss'], free['loss'])
        np.testing.assert_array_equal(anchored['gradient'], free['gradient'])

    def test_no_cert_outer_evaluation_in_entrypoint(self):
        entry = (ROOT / 'run_sele_refinement.py').read_text(encoding='utf-8')
        code = (ROOT / 'src/new_objective/sele_refinement.py').read_text(encoding='utf-8')
        self.assertNotIn("'evaluate'", entry)
        for forbidden in ('read_cert_labels', 'outer_data_blind', 'cert_data_blind'):
            self.assertNotIn(forbidden, code)
        self.assertEqual(spec()['primary_lambda'], 1.)
        self.assertEqual(spec()['primary_quota'], .5)

    def test_adapter_passes_sele_reference_to_both_refinements(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from new_objective import sele_refinement as module
        sele = {'status': 'OK', 'theta': np.ones(3), 'intercept': .3}
        base = {'mean': np.zeros(3), 'scale': np.ones(3), 'train_subjects': ['a'],
                'models': {'SELE': sele, 'BCE': {'theta': np.zeros(3)}}}
        fitted = {'status': 'OK', 'theta': np.ones(3), 'intercept': .3}
        with patch.object(module, 'load_reference', return_value=(base, {'C': 1})), \
             patch.object(module, 'identity', return_value={'sha256': 'test'}), \
             patch.object(module, 'fit_refinement', return_value=fitted) as solver, \
             patch.object(module, 'model_save'), patch.object(module, 'write'), \
             patch.object(module, 'digest', return_value='test'):
            module.fit_group('__unit__', 0, 1., np.ones((4, 3)), np.array([0, 0, 0, 1]),
                             np.array(['a']*4), {}, SimpleNamespace(check=lambda: None), False)
        self.assertEqual(solver.call_count, 2)
        for call in solver.call_args_list:
            self.assertIs(call.args[3], sele)
        self.assertEqual([c.args[5] for c in solver.call_args_list], ['FixedScale', 'BCEAnchor'])


if __name__ == '__main__':
    unittest.main()
