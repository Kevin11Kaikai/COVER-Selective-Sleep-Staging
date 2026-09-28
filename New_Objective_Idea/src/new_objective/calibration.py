"""Shared monotone Platt calibration on standardized, unclipped raw scores."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .losses import weights

def fit_mapping(raw,y,subjects):
    w=weights(subjects)
    mean=float(w@raw)
    scale=float(np.sqrt(w@((raw-mean)**2)))
    constant=scale<=np.finfo(float).eps*max(1.,np.max(np.abs(raw)))
    scale=1. if constant else scale
    z=(raw-mean)/scale
    p=np.clip(w@y,1e-8,1-1e-8)
    if constant:
        return {"mean":mean,"scale":scale,"slope":0.,"intercept":float(np.log(p/(1-p))),"constant":True,"success":True}
    def loss(v):
        s=v[0]*z+v[1];d=w*(expit(s)-y)
        return float(w@(np.logaddexp(0.,s)-y*s)+.5*v[0]**2/len(y)),np.array([d@z+v[0]/len(y),d.sum()])
    r=minimize(loss,[1.,np.log(p/(1-p))],jac=True,method="L-BFGS-B",bounds=[(0.,None),(None,None)],options={"maxiter":2000,"ftol":1e-13,"gtol":1e-9})
    if not r.success and np.isfinite(r.x).all():
        # The same convex objective and stopping tolerances; only retry the line search.
        r=minimize(loss,r.x,jac=True,method="L-BFGS-B",bounds=[(0.,None),(None,None)],options={"maxiter":2000,"ftol":1e-13,"gtol":1e-9,"maxls":100})
    if not r.success or not np.isfinite(r.x).all():
        raise RuntimeError("Platt optimization failed: "+str(r.message))
    return {"mean":mean,"scale":scale,"slope":float(r.x[0]),"intercept":float(r.x[1]),"constant":bool(r.x[0]==0),"success":True}

def apply_mapping(mapping,raw):
    return mapping["slope"]*(raw-mapping["mean"])/mapping["scale"]+mapping["intercept"]

def cross_calibrate(raw,y,subjects,folds):
    out=np.full(len(y),np.nan)
    mappings=[]
    for fold in folds:
        tr=np.isin(subjects,fold["train"]);va=np.isin(subjects,fold["validation"])
        m=fit_mapping(raw[tr],y[tr],subjects[tr])
        out[va]=apply_mapping(m,raw[va]);mappings.append(m)
    if not np.isfinite(out).all():
        raise RuntimeError("Incomplete calibrated FIT OOF scores")
    return out,mappings
