"""Deterministic linear BCE, convex SELE bundle, and constrained risk refinement."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize
from .losses import bce_oracle, sele_oracle, soft_oracle

def fit_bce(x, y, w, reg, spec):
    start = time.monotonic()
    p = np.clip(w @ y, 1e-6, 1-1e-6)
    initial = np.r_[np.zeros(x.shape[1]), np.log(p/(1-p))]
    cfg = spec["bce"]
    r = minimize(bce_oracle, initial, args=(x,y,w,reg), jac=True, method="L-BFGS-B", options={**cfg,"maxls":50})
    f, grad = bce_oracle(r.x,x,y,w,reg)
    return {"theta": r.x[:-1], "intercept": float(r.x[-1]),
            "status": "OK" if r.success and np.isfinite(r.x).all() else "NOT_CONVERGED",
            "diagnostics": {"seconds":time.monotonic()-start,"loss":f,"gradient_max":float(np.max(np.abs(grad))),"iterations":int(r.nit),"message":str(r.message)}}

def fit_sele(x, y, w, reg, initial, spec, heartbeat=None):
    started = time.monotonic()
    cfg = spec["sele"]
    theta = initial.copy()
    def full_oracle(v):
        loss,grad=sele_oracle(v,x,y,w)
        return loss+.5*reg*(v@v),grad+reg*v
    def warm_callback(v):
        if heartbeat:
            heartbeat()
    warm=minimize(full_oracle,theta,jac=True,method="L-BFGS-B",callback=warm_callback,
                  options={"maxiter":cfg["warm_lbfgs_maxiter"],"ftol":cfg["warm_lbfgs_ftol"],"gtol":cfg["warm_lbfgs_gtol"],"maxls":50})
    theta=warm.x.copy()
    value,total_gradient=full_oracle(theta)
    # Strong convexity: F* >= F(v) - ||g(v)||^2/(2*reg), for any subgradient g.
    certificate=float(total_gradient@total_gradient/(2*reg))
    if np.isfinite(theta).all() and certificate<=cfg["relative_gap"]*max(1.,abs(value)):
        return {"theta":theta,"intercept":0.,"status":"OK",
                "diagnostics":{"seconds":time.monotonic()-started,"upper":float(value),"lower":float(value-certificate),"gap":certificate,"iterations":int(warm.nit),"backend":"exact_loss_lbfgs_with_strong_convexity_certificate","trace":[]}}
    gs, bs, u = [], [], np.empty(0)
    upper, lower = value, value-certificate
    best = theta.copy()
    trace = []
    for k in range(cfg["max_oracle"]):
        if heartbeat and k % 25 == 0:
            heartbeat()
        loss, grad = sele_oracle(theta,x,y,w)
        full = loss+.5*reg*np.dot(theta,theta)
        if full < upper:
            upper, best = full, theta.copy()
        gs.append(grad)
        bs.append(loss-grad@theta)
        g, b = np.asarray(gs), np.asarray(bs)
        u = np.r_[u, 0.] if len(u) else np.ones(1)
        gram = g@g.T/reg
        def dual(v):
            return float(.5*v@gram@v-b@v), gram@v-b
        master = minimize(dual,u,jac=True,method="SLSQP",bounds=[(0.,1.)]*len(u),
            constraints=[{"type":"eq","fun":lambda v:v.sum()-1.,"jac":lambda v:np.ones_like(v)}],
            options={"maxiter":cfg["master_maxiter"],"ftol":cfg["master_ftol"]})
        # Any simplex-feasible dual point is a valid lower bound, even if the QP hits its cap.
        u = np.maximum(master.x,0.)
        u = u/u.sum() if np.isfinite(u).all() and u.sum()>0 else np.full(len(b),1./len(b))
        lower = max(lower, float(b@u-.5*u@gram@u))
        gap = max(0.,upper-lower)
        if k % 10 == 0 or gap <= cfg["relative_gap"]*max(1.,abs(upper)):
            trace.append({"iteration":k+1,"upper":float(upper),"lower":float(lower),"gap":float(gap),"master_success":bool(master.success)})
        if lower > upper + 1e-6:
            raise ArithmeticError("SELE lower bound exceeds upper bound")
        if gap <= cfg["relative_gap"]*max(1.,abs(upper)):
            break
        theta = -(u@g)/reg
        if len(gs) >= cfg["active_cuts"]:
            # The aggregate cut retains the present feasible dual certificate.
            tail = cfg["active_cuts"]-2
            split = len(gs)-tail
            mass = float(u[:split].sum())
            if mass > 1e-15:
                gs = [u[:split]@g[:split]/mass]+gs[split:]
                bs = [float(u[:split]@b[:split]/mass)]+bs[split:]
                u = np.r_[mass,u[split:]]
            else:
                gs,bs,u = gs[split:],bs[split:],u[split:]
                u /= u.sum()
    converged = gap <= cfg["relative_gap"]*max(1.,abs(upper))
    return {"theta":best,"intercept":0.,"status":"OK" if converged else "NOT_CONVERGED",
            "diagnostics":{"seconds":time.monotonic()-started,"upper":float(upper),"lower":float(lower),"gap":float(gap),"iterations":k+1,"backend":"bundle_after_exact_loss_lbfgs","trace":trace}}

def fit_risk(x,y,w,reg,initial,spec,heartbeat=None):
    started = time.monotonic()
    cfg,alpha = spec["risk"],spec["alpha"]
    minimum = spec["minimum_soft_coverage"]
    base_scores = x@initial
    scale = max(float(np.sqrt(w@((base_scores-w@base_scores)**2))),cfg["temperature_scale_floor"])
    traces, solutions = [], []
    def valid(o):
        return o["coverage"] >= minimum-1e-10 and o["risk"] is not None and o["risk"] <= alpha+cfg["risk_tolerance"]
    for start_scale in cfg["starts"]:
        theta = initial*start_scale
        scores = x@theta
        order = np.argsort(scores)
        thresholds = np.interp(np.linspace(.001,1.,101),np.cumsum(w[order]),scores[order])
        temperature = cfg["temperatures"][0]*scale
        trials = [(np.r_[theta,t],soft_oracle(np.r_[theta,t],x,y,w,temperature,reg,alpha)) for t in thresholds]
        feasible = [(v,o) for v,o in trials if valid(o)]
        v = min(feasible,key=lambda z:z[1]["loss"])[0] if feasible else min(trials,key=lambda z:max(0.,-z[1]["constraint"])+max(0.,minimum-z[1]["coverage"]))[0]
        for stage,tau in enumerate(cfg["temperatures"]):
            if heartbeat:
                heartbeat()
            temperature = tau*scale
            cache_v, cache_o = None,None
            def oracle(vv):
                nonlocal cache_v,cache_o
                if cache_v is None or not np.array_equal(vv,cache_v):
                    cache_v=vv.copy();cache_o=soft_oracle(vv,x,y,w,temperature,reg,alpha)
                return cache_o
            steps=[0]
            def callback(vv):
                steps[0]+=1
                if heartbeat and steps[0]%25==0:
                    heartbeat()
            cons=[{"type":"ineq","fun":lambda vv:oracle(vv)["constraint"],"jac":lambda vv:oracle(vv)["grad_constraint"]},
                  {"type":"ineq","fun":lambda vv:oracle(vv)["coverage"]-minimum,"jac":lambda vv:oracle(vv)["grad_coverage"]}]
            attempts=[]
            for attempt in range(2):
                r=minimize(lambda vv:(oracle(vv)["loss"],oracle(vv)["gradient"]),v,jac=True,
                    method="SLSQP",constraints=cons,callback=callback,options={"maxiter":cfg["maxiter"],"ftol":cfg["ftol"]})
                v=r.x.copy();o=oracle(v)
                ok=bool(r.success and np.isfinite(v).all() and valid(o))
                attempts.append({"success":bool(r.success),"valid":valid(o),"iterations":int(r.nit),"message":str(r.message)})
                if ok:
                    break
            traces.append({"start_scale":start_scale,"tau":tau,"loss":o["loss"],"coverage":o["coverage"],"risk":o["risk"],"attempts":attempts})
            if stage==len(cfg["temperatures"])-1 and ok:
                solutions.append((o["loss"],v.copy(),o.copy()))
    if not solutions:
        return {"status":"NO_FEASIBLE_SOLUTION_FOUND","theta":None,"intercept":0.,"diagnostics":{"seconds":time.monotonic()-started,"trace":traces}}
    _,v,o=min(solutions,key=lambda q:q[0])
    return {"theta":v[:-1],"intercept":0.,"training_threshold":float(v[-1]),"status":"OK",
            "diagnostics":{"seconds":time.monotonic()-started,"trace":traces,"soft_coverage":o["coverage"],"soft_risk":o["risk"],"theta_norm":float(np.linalg.norm(v[:-1])),"temperature_scale":scale}}
