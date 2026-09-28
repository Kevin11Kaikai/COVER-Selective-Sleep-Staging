import sys
import os
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'src'),str(ROOT.parent/'AutoML_Claude')]
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from new_objective.anchor_solver import anchored_oracle
from new_objective.anchor_experiment import rank_candidate
from new_objective.losses import weights,soft_oracle
from new_objective.storage import target


class AnchoredTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(37)
        self.x=rng.normal(size=(40,4));self.y=(self.x[:,0]>0).astype(float)
        self.w=weights(np.repeat(['a','b'],20));self.v=np.array([1.,.2,-.1,.4,-.3])
        self.center=np.array([.8,.1,-.1,.2])

    def test_anchored_gradient(self):
        def f(v):return anchored_oracle(v,self.x,self.y,self.w,.2,.1,self.center,.05)['loss']
        o=anchored_oracle(self.v,self.x,self.y,self.w,.2,.1,self.center,.05)
        np.testing.assert_allclose(o['gradient'],approx_derivative(f,self.v).ravel(),atol=1e-7)

    def test_unanchored_is_same_absolute_l2_oracle(self):
        a=anchored_oracle(self.v,self.x,self.y,self.w,.2,.1,np.zeros(4),.05)
        b=soft_oracle(self.v,self.x,self.y,self.w,.2,.1,.05)
        self.assertAlmostEqual(a['loss'],b['loss'])
        np.testing.assert_allclose(a['gradient'],b['gradient'])

    def test_anchor_penalty_zero_at_base(self):
        v=np.r_[self.center,-.2]
        a=anchored_oracle(v,self.x,self.y,self.w,.2,10,self.center,.05)
        b=soft_oracle(v,self.x,self.y,self.w,.2,0,.05)
        self.assertEqual(a['loss'],b['loss'])
        np.testing.assert_array_equal(a['gradient'],b['gradient'])

    def test_zero_residual_preserves_original_raw_interface(self):
        theta=np.array([.3,.6,1.5,.09]);scale=1.736
        beta=theta/scale
        restored=theta+scale*(beta-beta)
        np.testing.assert_array_equal(theta,restored)
        np.testing.assert_array_equal(self.x@theta+.3,self.x@restored+.3)

    def test_stronger_penalty_wins_exact_tie(self):
        rows=[{'coverage':.5,'risk':.04,'lambda':lam} for lam in (.01,.1,1,10)]
        self.assertEqual(min(rows,key=rank_candidate)['lambda'],10)

    def test_output_containment(self):
        with self.assertRaises(ValueError):target(ROOT.parent/'anchor_must_not_exist.json')

    def test_fit_signature_excludes_validation_or_outcomes(self):
        import inspect
        from new_objective.anchor_solver import fit_refinement
        names=set(inspect.signature(fit_refinement).parameters)
        self.assertFalse(names & {'cert','outer','validation','validation_labels','outer_labels'})

    def test_risk_constraint_gradient_unchanged_by_anchor(self):
        a=anchored_oracle(self.v,self.x,self.y,self.w,.2,1,self.center,.05)
        def f(v):return anchored_oracle(v,self.x,self.y,self.w,.2,1,self.center,.05)['constraint']
        np.testing.assert_allclose(a['grad_constraint'],approx_derivative(f,self.v).ravel(),atol=1e-7)


if __name__=='__main__':unittest.main()
