"""Frozen risk fitting recipe extracted from archive/source_code/stage1/risk_models.py.

Only imported for explicit --mode fit. No component training is imported.
"""
from __future__ import annotations

import numpy as np

from .acceptance import subject_equal_weights
from .risk_features import heterogeneous_features, primary_only_features


def feature_matrix(data: dict, family: str) -> np.ndarray:
    if family == "Primary-only":
        return primary_only_features(data["primary"])[0]
    if family == "Heterogeneous-expert":
        return heterogeneous_features(data["primary"], data["feature"])[0]
    raise ValueError(family)


def fit_model(config: dict, x: np.ndarray, y: np.ndarray, subjects: np.ndarray):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Risk fitting requires both error classes")
    if config["family"] == "logistic":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=float(config["C"]), solver="lbfgs",
                penalty="l2", max_iter=2000, tol=1e-8, class_weight=None,
                random_state=20260901)),
        ])
        model.fit(x, y, model__sample_weight=subject_equal_weights(subjects))
    elif config["family"] == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(loss="log_loss", learning_rate=.05,
            max_iter=200, max_leaf_nodes=int(config["max_leaf_nodes"]),
            min_samples_leaf=100, l2_regularization=float(config["l2_regularization"]),
            max_bins=64, early_stopping=False, random_state=20260901)
        model.fit(x, y, sample_weight=subject_equal_weights(subjects))
    else:
        raise ValueError(config["family"])
    return model


def positive_probability(model, x: np.ndarray) -> np.ndarray:
    position = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(position) != 1:
        raise ValueError("Missing primary-error class")
    return np.asarray(model.predict_proba(x)[:, position[0]], dtype=np.float64)


def score_logit(raw: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(raw, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[:, None]


def fit_platt(raw: np.ndarray, y: np.ndarray, subjects: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1., solver="lbfgs", penalty="l2", max_iter=2000,
        class_weight=None, random_state=20260901)
    model.fit(score_logit(raw), y, sample_weight=subject_equal_weights(subjects))
    return model


def predict(bundle: dict, data: dict, family: str) -> np.ndarray:
    raw = positive_probability(bundle["base"], feature_matrix(data, family))
    return positive_probability(bundle["platt"], score_logit(raw))
