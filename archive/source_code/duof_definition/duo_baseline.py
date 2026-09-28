"""Paper-definition fixed-primary UQ-Only Duo, explicitly adapted to source subject weights."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
from STAGE2_ICLR_CAMPAIGN_20260902T025610Z.code.selectors import subject_equal_weights


def auxiliary_logits(data, info):
    # Native H logits; calibrated RF probabilities have no native logits, use log-probability.
    return data["H_logits"].astype(float) if info == "H" else np.log(np.clip(data["F"], 1e-10, 1))


def fit_duo(data, y, info):
    primary, auxiliary = data["P_logits"].astype(float), auxiliary_logits(data, info)
    weights = subject_equal_weights(data["subjects"])
    weights /= weights.sum()
    target = np.eye(5)[y]
    def objective(parameters):
        z = parameters[0]*primary + parameters[1]*auxiliary
        loss = np.dot(weights, logsumexp(z, axis=1)-z[np.arange(len(y)), y])
        residual = (softmax(z, axis=1)-target)*weights[:, None]
        gradient = np.array([(residual*primary).sum(), (residual*auxiliary).sum()])
        return loss, gradient
    fitted = minimize(objective, [1.0, 1.0], jac=True, method="L-BFGS-B", bounds=[(0, 20), (0, 20)],
                      options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    if not fitted.success:
        raise RuntimeError(f"Duo baseline optimization failed: {fitted.message}")
    return {"information": info, "weights": fitted.x.tolist(), "source_NLL": float(fitted.fun),
            "success": bool(fitted.success), "iterations": int(fitted.nit), "message": str(fitted.message),
            "adaptation": "two nonnegative bounded logit weights; source RISK_FIT subject-equal multiclass NLL; primary classification unchanged"}


def duo_scores(data, fitted):
    result = {}
    for info in ("H", "F"):
        primary_weight, aux_weight = fitted[info]["weights"]
        probability = softmax(primary_weight*data["P_logits"] + aux_weight*auxiliary_logits(data, info), axis=1)
        # Not ensemble max: evaluate the probability assigned to the original primary label.
        result[f"DuoUQ_{info}"] = 1-probability[np.arange(len(probability)), data["P"].argmax(1)]
    return result
