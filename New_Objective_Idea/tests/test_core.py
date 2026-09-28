import os
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/"src"),str(ROOT.parent/"src"),str(ROOT.parent/"AutoML_Claude")]
for n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[n]="2"
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from new_objective.losses import weights,bce_oracle,sele_oracle,soft_oracle
from new_objective.solvers import fit_bce,fit_sele,fit_risk
from new_objective.calibration import fit_mapping,apply_mapping,cross_calibrate
from new_objective.storage import specification,target,OUT

class CoreTests(unittest.TestCase):
    def setUp(self):
        g=np.random.default_rng(17)
        self.x=g.normal(size=(30,4));self.y=(self.x[:,0]+g.normal(size=30)*.4>0).astype(int)
        self.w=np.full(30,1/30);self.theta=g.normal(size=4);self.spec=specification()

    def test_bce_gradient(self):
        v=np.r_[self.theta,.2]
        _,actual=bce_oracle(v,self.x,self.y,self.w,.03)
        expected=approx_derivative(lambda t:bce_oracle(t,self.x,self.y,self.w,.03)[0],v).ravel()
        np.testing.assert_allclose(actual,expected,atol=1e-7)

    def test_sele_exact_and_gradient(self):
        s=self.x@self.theta
        naive=np.sum(self.w[:,None]*self.w[None,:]*self.y[:,None]*np.maximum(0,1+s[None,:]-s[:,None]))
        loss,grad=sele_oracle(self.theta,self.x,self.y,self.w)
        self.assertAlmostEqual(loss,naive,12)
        numeric=approx_derivative(lambda t:sele_oracle(t,self.x,self.y,self.w)[0],self.theta).ravel()
        np.testing.assert_allclose(grad,numeric,atol=1e-7)

    def test_sele_ties(self):
        x=np.zeros((5,2));y=np.array([0,1,0,1,0]);w=np.full(5,.2)
        value,gradient=sele_oracle(np.zeros(2),x,y,w)
        self.assertAlmostEqual(value,.4);np.testing.assert_array_equal(gradient,np.zeros(2))

    def test_sele_unequal_subject_weights(self):
        subject=np.asarray(["a"]*7+["b"]*23)
        w=weights(subject);s=self.x@self.theta
        expected=sum(w[i]*w[j]*self.y[i]*max(0.,1+s[j]-s[i])
                     for i in range(30) for j in range(30))
        value,_=sele_oracle(self.theta,self.x,self.y,w)
        self.assertAlmostEqual(value,expected,12)

    def test_soft_gradients(self):
        v=np.r_[self.theta,.3]
        o=soft_oracle(v,self.x,self.y,self.w,.7,.01,.05)
        for key,grad in [("loss","gradient"),("constraint","grad_constraint"),("coverage","grad_coverage")]:
            numeric=approx_derivative(lambda t:soft_oracle(t,self.x,self.y,self.w,.7,.01,.05)[key],v).ravel()
            np.testing.assert_allclose(o[grad],numeric,atol=1e-7)

    def test_saturated_soft_no_nan(self):
        for t in (-1e6,1e6):
            o=soft_oracle(np.r_[np.zeros(4),t],self.x,self.y,self.w,.1,.1,.05)
            self.assertTrue(np.isfinite(o["gradient"]).all())
            if t<0:self.assertIsNone(o["risk"])

    def test_subject_weight_mass(self):
        for subjects in (np.array(["a"]*3+["b"]*6),np.array(["a"]*9+["b"]*6)):
            w=weights(subjects)
            self.assertAlmostEqual(w[subjects=="a"].sum(),.5)

    def test_bce_converges(self):
        r=fit_bce(self.x,self.y,self.w,.01,self.spec)
        self.assertEqual(r["status"],"OK")

    def test_sele_bound_certificate(self):
        r=fit_sele(self.x,self.y,self.w,.1,np.zeros(4),self.spec)
        self.assertEqual(r["status"],"OK")
        self.assertGreaterEqual(r["diagnostics"]["upper"]+1e-8,r["diagnostics"]["lower"])
        self.assertLessEqual(r["diagnostics"]["gap"],1e-4)

    def test_risk_separable_feasible(self):
        x=np.array([[-2.],[-1.],[1.],[2.]])
        y=np.array([0,0,1,1]);w=np.full(4,.25)
        r=fit_risk(x,y,w,.001,np.array([2.]),self.spec)
        self.assertEqual(r["status"],"OK")
        self.assertLessEqual(r["diagnostics"]["soft_risk"],.050001)

    def test_calibration_affine_invariance(self):
        raw=self.x@self.theta;s=np.array(["a"]*10+["b"]*20)
        a=fit_mapping(raw,self.y,s);b=fit_mapping(3.7*raw-8.1,self.y,s)
        np.testing.assert_allclose(apply_mapping(a,raw),apply_mapping(b,3.7*raw-8.1),atol=1e-7)
        self.assertGreaterEqual(a["slope"],0)

    def test_calibration_abnormal_retry_keeps_objective(self):
        from unittest.mock import patch
        from scipy.optimize import minimize as real_minimize
        import new_objective.calibration as calibration
        raw=self.x@self.theta;s=np.array(["a"]*10+["b"]*20)
        original=fit_mapping(raw,self.y,s)
        calls=[]
        def one_abnormal(*args,**kwargs):
            result=real_minimize(*args,**kwargs)
            calls.append(kwargs["options"])
            if len(calls)==1:result.success=False
            return result
        with patch.object(calibration,"minimize",side_effect=one_abnormal):
            repaired=fit_mapping(raw,self.y,s)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0]["ftol"],calls[1]["ftol"])
        self.assertEqual(calls[0]["gtol"],calls[1]["gtol"])
        self.assertEqual(calls[1]["maxls"],100)
        np.testing.assert_allclose(apply_mapping(original,raw),apply_mapping(repaired,raw),atol=1e-7)

    def test_constant_calibration(self):
        m=fit_mapping(np.ones(30),self.y,np.array(["a"]*30))
        self.assertTrue(m["constant"])
        self.assertEqual(np.unique(apply_mapping(m,np.ones(30))).size,1)

    def test_calibrator_excludes_its_validation_labels(self):
        sub=np.asarray(["a"]*10+["b"]*10+["c"]*10)
        folds=[{"train":["b","c"],"validation":["a"]},
               {"train":["a","c"],"validation":["b"]},
               {"train":["a","b"],"validation":["c"]}]
        raw=self.x@self.theta
        changed=self.y.copy();changed[sub=="a"]=1-changed[sub=="a"]
        original,_=cross_calibrate(raw,self.y,sub,folds)
        counterfactual,_=cross_calibrate(raw,changed,sub,folds)
        np.testing.assert_array_equal(original[sub=="a"],counterfactual[sub=="a"])

    def test_write_escape_rejected(self):
        with self.assertRaises(ValueError):target(ROOT.parent/"outside.txt")

    def test_fit_function_has_no_cert_argument(self):
        import inspect
        from new_objective.experiment import fit_three
        self.assertEqual(list(inspect.signature(fit_three).parameters),["x","y","subjects","c","spec","budget"])

    def test_scale_changes_soft_not_ranking(self):
        from scipy.special import expit
        s=np.arange(1.,101.);e=(s>80).astype(int)
        a=expit(.02*(70.5-s));b=expit(.5*(70.5-s))
        self.assertGreater(abs(a.mean()-b.mean()),.1)
        np.testing.assert_array_equal(.02*s<=.02*70.5,.5*s<=.5*70.5)

if __name__=="__main__":unittest.main()
