"""Analytic weighted loss/constraint oracles; no data loading or model selection."""
from __future__ import annotations
import numpy as np
from scipy.special import expit

def weights(subjects):
    _, inv, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[inv])

def bce_oracle(v, x, y, w, reg):
    s = x @ v[:-1] + v[-1]
    p = expit(s)
    loss = np.dot(w, np.logaddexp(0., s) - y*s) + .5*reg*np.dot(v[:-1], v[:-1])
    d = w * (p-y)
    return float(loss), np.r_[x.T @ d + reg*v[:-1], d.sum()]

def sele_oracle(theta, x, y, w):
    s = x @ theta
    order = np.argsort(s, kind="stable")
    ss, ww, ee = s[order], w[order], y[order]
    pw = np.r_[0., np.cumsum(ww)]
    ps = np.r_[0., np.cumsum(ww*ss)]
    pe = np.r_[0., np.cumsum(ww*ee)]
    # Only strict hinge-positive terms enter this valid subgradient.
    left = np.searchsorted(ss, ss-1., side="right")
    right = np.searchsorted(ss, ss+1., side="left")
    mass = pw[-1]-pw[left]
    value = np.sum(ww*ee*((1.-ss)*mass + ps[-1]-ps[left]))
    grad_s = ww*pe[right] - ww*ee*mass
    derivative = np.empty_like(grad_s)
    derivative[order] = grad_s
    return float(value), x.T @ derivative

def soft_oracle(v, x, y, w, temperature, reg, alpha):
    theta, threshold = v[:-1], v[-1]
    g = expit((threshold - x @ theta) / temperature)
    d = w*g*(1.-g)/temperature
    coverage, error = float(w @ g), float((w*y) @ g)
    grad_c = np.r_[-x.T @ d, d.sum()]
    de = d*y
    grad_e = np.r_[-x.T @ de, de.sum()]
    loss = -coverage + .5*reg*np.dot(theta, theta)
    grad_loss = -grad_c + np.r_[reg*theta, 0.]
    # Positive constraint means feasible: C - E/alpha >= 0.
    return {"loss": float(loss), "gradient": grad_loss,
            "coverage": coverage, "error": error,
            "risk": error/coverage if coverage > 0 else None,
            "constraint": coverage-error/alpha,
            "grad_constraint": grad_c-grad_e/alpha, "grad_coverage": grad_c}
