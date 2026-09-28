"""Study-local fixed-parent-scale adapter; historical anchor_solver.py is immutable."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize
from .losses import soft_oracle, weights


def anchored_oracle(v, x, y, w, temperature, lam, center, alpha):
    """Center 0 gives absolute L2; center beta_BCE gives a residual penalty."""
    o = soft_oracle(v, x, y, w, temperature, 0., alpha)
    delta = v[:-1]-center
    o['loss'] += .5*lam*float(delta@delta)
    o['gradient'] = o['gradient']+np.r_[lam*delta, 0.]
    return o


def fit_refinement(x, y, subjects, base, lam, method, spec, heartbeat=None, fixed_scale=None):
    """All arguments are training data or fixed protocol; no validation/CERT/OUTER inputs."""
    started=time.monotonic()
    w=weights(subjects)
    raw=x@base['theta']
    scale=(max(float(np.sqrt(w@((raw-w@raw)**2))),1e-6) if fixed_scale is None else float(fixed_scale))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError('Invalid frozen parent score scale')
    beta0=base['theta']/scale
    center=beta0 if method=='BCEAnchor' else np.zeros_like(beta0)
    if method not in ('FixedScale','BCEAnchor'):raise ValueError(method)
    T, alpha = spec['temperature'], spec['alpha']
    minimum=spec['minimum_soft_coverage']
    traces, solutions = [], []

    def feasible(o):
        return (o['coverage']>=minimum-1e-10 and o['risk'] is not None and
                o['risk']<=alpha+spec['risk_tolerance'])

    for start_scale in spec['starts']:
        beta=start_scale*beta0
        s=x@beta
        order=np.argsort(s,kind='stable')
        ts=np.interp(np.linspace(.001,1.,101),np.cumsum(w[order]),s[order])
        # Extend below the lowest observed score to include low-coverage soft starts.
        ts=np.r_[s.min()-T*np.array([20.,10.,5.]),ts]
        choices=[]
        for t in ts:
            v=np.r_[beta,t]
            o=anchored_oracle(v,x,y,w,T,lam,center,alpha)
            choices.append((v,o))
        good=[a for a in choices if feasible(a[1])]
        v=min(good,key=lambda a:a[1]['loss'])[0] if good else min(choices,key=lambda a:max(0.,-a[1]['constraint'])+max(0.,minimum-a[1]['coverage']))[0]
        cached_v,cached_o=None,None
        def oracle(vv):
            nonlocal cached_v,cached_o
            if cached_v is None or not np.array_equal(cached_v,vv):
                cached_v=vv.copy();cached_o=anchored_oracle(vv,x,y,w,T,lam,center,alpha)
            return cached_o
        steps=[0];trace=[]
        def checkpoint(vv,kind):
            o=oracle(vv)
            trace.append({'iteration':steps[0],'kind':kind,'beta':vv[:-1].copy(),'threshold':float(vv[-1]),
                'loss':o['loss'],'soft_coverage':o['coverage'],'soft_risk':o['risk'],
                'penalty':.5*lam*float((vv[:-1]-center)@(vv[:-1]-center)),
                'valid':feasible(o)})
        def callback(vv):
            steps[0]+=1
            if steps[0]%spec['record_every']==0:
                checkpoint(vv,'iterate')
                if heartbeat:heartbeat()
        checkpoint(v,'initial')
        constraints=[{'type':'ineq','fun':lambda vv:oracle(vv)['constraint'],'jac':lambda vv:oracle(vv)['grad_constraint']},
                     {'type':'ineq','fun':lambda vv:oracle(vv)['coverage']-minimum,'jac':lambda vv:oracle(vv)['grad_coverage']}]
        attempts=[]
        for attempt in range(spec['attempts']):
            if heartbeat:heartbeat()
            r=minimize(lambda vv:(oracle(vv)['loss'],oracle(vv)['gradient']),v,
                jac=True,method='SLSQP',constraints=constraints,callback=callback,
                options={'maxiter':spec['maxiter'],'ftol':spec['ftol']})
            v=r.x.copy();o=oracle(v)
            ok=bool(r.success and np.isfinite(v).all() and feasible(o))
            attempts.append({'success':bool(r.success),'valid':feasible(o),'iterations':int(r.nit),'message':str(r.message)})
            if ok:break
        checkpoint(v,'final')
        traces.append({'start_scale':start_scale,'attempts':attempts,'states':trace})
        if ok:solutions.append((o['loss'],v.copy(),o.copy(),start_scale))
    if not solutions:
        return {'status':'NO_FEASIBLE_SOLUTION_FOUND','theta':None,'intercept':base['intercept'],
                'diagnostics':{'seconds':time.monotonic()-started,'traces':traces}}
    _,v,o,selected_start=min(solutions,key=lambda a:a[0])
    return {'status':'OK','theta':base['theta']+scale*(v[:-1]-beta0),'intercept':base['intercept'],
        'beta':v[:-1], 'beta0':beta0, 'training_threshold':float(v[-1]), 'score_scale':scale,
        'diagnostics':{'seconds':time.monotonic()-started,'traces':traces,'selected_start':selected_start,
            'soft_coverage':o['coverage'],'soft_risk':o['risk'],'loss':o['loss'],
            'relative_parameter_change':float(np.linalg.norm(v[:-1]-beta0)/max(np.linalg.norm(beta0),1e-12)),
            'temperature':T,'lambda':lam}}


