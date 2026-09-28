"""Read-only reuse of frozen exact policy/identity conventions."""
from fractions import Fraction
import numpy as np
from stage1_analysis.acceptance import build_acceptance_mask,subject_equal_risk,select_empirical_policy
from automl.grids import dense_candidate_policies,grid_fractions,subset_policies

def policies(scores,subjects):
    return dense_candidate_policies(scores,subjects,grid_fractions(100))

def exact(x):
    return Fraction(x["numerator"],x["denominator"]) if isinstance(x,dict) else Fraction(str(x))

def mask(scores,rule,data,outer,spec):
    if rule is None:
        return np.zeros(len(scores),dtype=bool)
    return build_acceptance_mask(scores,threshold=rule["threshold"],tie_fraction=exact(rule["tie_fraction"]),identities=data["identities"],method_id=spec["tie_method_id"],protocol_hash=spec["protocol_hash"],outer=outer)

def choose(grid,scores,errors,data,outer,spec,minimum=None):
    rows=[]
    for rule in grid:
        m=mask(scores,rule,data,outer,spec)
        metric=subject_equal_risk(m,errors,data["subjects"])
        support=min(z["accepted"] for z in metric["per_subject"].values())
        rows.append({**rule,"coverage":metric["coverage"],"risk":metric["risk"],"minimum_accepted":support,"supported":True if minimum is None else support>=minimum})
    best=select_empirical_policy(rows,alpha=spec["alpha"])
    return best,rows

def fit_rank(row):
    return (-exact(row["coverage"]),Fraction(2) if row["risk"] is None else exact(row["risk"]),row["C"])
