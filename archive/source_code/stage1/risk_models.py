"""Frozen learned-risk development, calibration, and sealed selector logic."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.stage1_clean_nested_loso.run_outer_fold import atomic_joblib_dump

from . import PROTOCOL_SHA256
from .bundles import load_blind_bundle
from .data_io import (
    align_values,
    atomic_write_json,
    atomic_write_npz,
    canonical_json_hash,
    load_labels_authorized,
    sha256_file,
)
from .features import forward_kl, heterogeneous_features, primary_only_features
from .firewall import RoleLedger
from .provenance import (
    audited_environment,
    component_code_provenance,
    feature_cache_provenance,
    phase_provenance_hash,
    raw_inventory_provenance,
    risk_code_provenance,
)
from .selectors import (
    build_acceptance_mask,
    build_candidate_policies,
    hoeffding_ratio_ucb,
    select_empirical_policy,
    subject_equal_risk,
    subject_equal_weights,
)
from .splits import (
    build_final_roles,
    build_inner_roles,
    build_risk_cv,
    normalize_subject,
)


RISK_SCHEMA = "stage2-risk-development-v1"
ALPHAS = (0.02, 0.05, 0.10)
PRIMARY_ALPHA = 0.05
BASELINE_ORDER = (
    "MSP",
    "Margin",
    "Entropy",
    "Temperature-MSP",
    "Forward-KL",
    "Disagreement",
)


def _fraction(payload: Mapping[str, Any] | Fraction | int) -> Fraction:
    if isinstance(payload, Fraction):
        return payload
    if isinstance(payload, Mapping):
        return Fraction(int(payload["numerator"]), int(payload["denominator"]))
    return Fraction(payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _bundle_payload(path: Path) -> dict[str, Any]:
    payload = load_blind_bundle(path)
    identities_raw = np.asarray(payload["identities_json"]).item()
    if isinstance(identities_raw, bytes):
        identities_raw = identities_raw.decode("utf-8")
    identities = json.loads(str(identities_raw))
    manifest_raw = np.asarray(payload["manifest_json"]).item()
    if isinstance(manifest_raw, bytes):
        manifest_raw = manifest_raw.decode("utf-8")
    manifest = json.loads(str(manifest_raw))
    return {
        "identities": identities,
        "logits": np.asarray(payload["logits"], dtype=np.float64),
        "primary": np.asarray(payload["primary_probabilities"], dtype=np.float64),
        "feature": np.asarray(payload["feature_probabilities"], dtype=np.float64),
        "manifest": manifest,
    }


def _validated_component_payload(
    *,
    bundle_path: Path,
    expected_roles: Mapping[str, Any],
    expected_predicted_subjects: Sequence[str],
) -> dict[str, Any]:
    directory = bundle_path.parent
    completion_path = directory / "component_manifest.json"
    run_config_path = directory / "run_config.json"
    if not completion_path.exists() or not run_config_path.exists():
        raise RuntimeError(f"component provenance sidecar missing: {directory}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if completion.get("bundle_sha256") != sha256_file(bundle_path):
        raise RuntimeError(f"component bundle content hash mismatch: {bundle_path}")
    if completion.get("run_config_hash") != canonical_json_hash(run_config):
        raise RuntimeError(f"component run-config hash mismatch: {directory}")
    execution = run_config.get("execution_provenance")
    if not isinstance(execution, dict):
        raise RuntimeError(f"component has no execution provenance: {directory}")
    execution_hash = canonical_json_hash(execution)
    if (
        run_config.get("execution_provenance_hash") != execution_hash
        or completion.get("execution_provenance_hash") != execution_hash
    ):
        raise RuntimeError(f"component execution-provenance hash mismatch: {directory}")
    current_code = component_code_provenance()
    current_environment = audited_environment()
    embedded_raw = execution.get("inputs", {}).get("raw", {})
    embedded_feature = execution.get("inputs", {}).get("feature_cache", {})
    current_raw = raw_inventory_provenance(str(embedded_raw.get("directory", "")))
    current_feature = feature_cache_provenance(
        str(embedded_feature.get("manifest_path", ""))
    )
    if (
        execution.get("code") != current_code
        or execution.get("environment") != current_environment
        or embedded_raw != current_raw
        or embedded_feature != current_feature
    ):
        raise RuntimeError(f"component execution inputs/code/environment changed: {directory}")
    expected_roles_dict = dict(expected_roles)
    expected_predicted = list(expected_predicted_subjects)
    if (
        run_config.get("protocol_hash") != PROTOCOL_SHA256
        or run_config.get("roles") != expected_roles_dict
        or run_config.get("predicted_subjects") != expected_predicted
        or completion.get("protocol_hash") != PROTOCOL_SHA256
        or completion.get("role_hash") != expected_roles["role_hash"]
        or completion.get("predicted_subjects") != expected_predicted
    ):
        raise RuntimeError(f"component declared roles differ from frozen split: {directory}")
    payload = _bundle_payload(bundle_path)
    manifest = payload["manifest"]
    if (
        manifest.get("protocol_hash") != PROTOCOL_SHA256
        or manifest.get("outer") != expected_roles["outer"]
        or manifest.get("role") != expected_roles["role"]
        or str(manifest.get("target")) != str(expected_roles["target"])
        or manifest.get("predicted_subjects") != expected_predicted
        or manifest.get("generating_roles") != expected_roles_dict
        or manifest.get("provenance_hash") != execution_hash
        or manifest.get("hashes", {}).get("code") != current_code["hash"]
        or manifest.get("hashes", {}).get("data") != current_raw["file_set_hash"]
        or manifest.get("hashes", {}).get("config") != completion["run_config_hash"]
    ):
        raise RuntimeError(f"component blind manifest differs from frozen ancestry: {directory}")
    checkpoint_path = directory / "primary" / "best_model.pth"
    feature_cache_key = completion.get("feature_model_metadata", {}).get("cache_key")
    feature_path = directory / "feature" / "cache" / f"{feature_cache_key}.joblib"
    if (
        not checkpoint_path.exists()
        or sha256_file(checkpoint_path) != manifest["hashes"]["primary_checkpoint"]
        or not feature_path.exists()
        or sha256_file(feature_path) != manifest["hashes"]["feature_model"]
    ):
        raise RuntimeError(f"component model/checkpoint hash mismatch: {directory}")
    return payload


def _concatenate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("cannot concatenate an empty prediction collection")
    identities = [identity for row in rows for identity in row["identities"]]
    return {
        "identities": identities,
        "subjects": np.asarray([identity["subject_id"] for identity in identities]),
        "logits": np.vstack([row["logits"] for row in rows]),
        "primary": np.vstack([row["primary"] for row in rows]),
        "feature": np.vstack([row["feature"] for row in rows]),
    }


def load_fit_predictions(
    *,
    outer: str,
    fit_subjects: Sequence[str],
    components_root: Path,
    records: Mapping[str, Sequence[Path]],
    ledger: RoleLedger,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for subject in fit_subjects:
        path = components_root / f"outer-{outer}" / f"inner-{subject}" / "blind_predictions.npz"
        expected_roles = build_inner_roles(outer, subject, fit_subjects)
        row = _validated_component_payload(
            bundle_path=path,
            expected_roles=expected_roles,
            expected_predicted_subjects=[subject],
        )
        if set(identity["subject_id"] for identity in row["identities"]) != {subject}:
            raise RuntimeError(f"inner component bundle predicts the wrong target: {path}")
        rows.append(row)
    result = _concatenate(rows)
    labels, label_ids = load_labels_authorized(
        records,
        fit_subjects,
        ledger=ledger,
        phase="RISK_FIT",
    )
    labels = align_values(label_ids, labels, result["identities"]).astype(np.int64)
    result["labels"] = labels
    result["errors"] = (
        np.argmax(result["primary"], axis=1).astype(np.int64) != labels
    ).astype(np.int64)
    return result


def load_final_blind_predictions(
    *,
    outer: str,
    fit_subjects: Sequence[str],
    cert_subjects: Sequence[str],
    components_root: Path,
) -> dict[str, dict[str, Any]]:
    expected_roles = build_final_roles(outer, fit_subjects)
    payload = _validated_component_payload(
        bundle_path=(
            components_root / f"outer-{outer}" / "final" / "blind_predictions.npz"
        ),
        expected_roles=expected_roles,
        expected_predicted_subjects=list(cert_subjects) + [outer],
    )
    subjects = np.asarray([identity["subject_id"] for identity in payload["identities"]])
    result: dict[str, dict[str, Any]] = {}
    for subject in sorted(set(subjects)):
        mask = subjects == subject
        result[subject] = {
            "identities": [
                identity for identity, keep in zip(payload["identities"], mask) if keep
            ],
            "subjects": subjects[mask],
            "logits": payload["logits"][mask],
            "primary": payload["primary"][mask],
            "feature": payload["feature"][mask],
        }
    return result


def risk_registry() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        id_by_c = {0.01: "lr_c001", 0.1: "lr_c01", 1.0: "lr_c1", 10.0: "lr_c10"}
        configs.append(
            {
                "id": id_by_c[c_value],
                "family": "logistic",
                "C": c_value,
            }
        )
    for leaves in (7, 15):
        for l2 in (0.0, 1.0):
            configs.append(
                {
                    "id": f"hgb_l{leaves}_r{int(l2)}",
                    "family": "hist_gradient_boosting",
                    "max_leaf_nodes": leaves,
                    "l2_regularization": l2,
                }
            )
    return configs


def _new_model(config: Mapping[str, Any]) -> Any:
    if config["family"] == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(config["C"]),
                        solver="lbfgs",
                        penalty="l2",
                        max_iter=2000,
                        tol=1e-8,
                        class_weight=None,
                        random_state=20260901,
                    ),
                ),
            ]
        )
    if config["family"] == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            min_samples_leaf=100,
            l2_regularization=float(config["l2_regularization"]),
            max_bins=64,
            early_stopping=False,
            random_state=20260901,
        )
    raise ValueError(f"unknown risk model family: {config['family']}")


def _fit_model(
    config: Mapping[str, Any], x: np.ndarray, y: np.ndarray, subjects: np.ndarray
) -> Any:
    if set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError("risk-model training lacks one error class")
    model = _new_model(config)
    model.fit(x, y, **({"model__sample_weight": subject_equal_weights(subjects)} if isinstance(model, Pipeline) else {"sample_weight": subject_equal_weights(subjects)}))
    return model


def _positive_probability(model: Any, x: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_)
    index = np.flatnonzero(classes == 1)
    if len(index) != 1:
        raise RuntimeError("risk model has no unique error class")
    return np.asarray(model.predict_proba(x)[:, index[0]], dtype=np.float64)


def _score_logit(score: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(score, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))[:, None]


def _fit_platt(raw: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> LogisticRegression:
    if set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError("Platt calibration lacks one error class")
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        penalty="l2",
        max_iter=2000,
        class_weight=None,
        random_state=20260901,
    )
    model.fit(
        _score_logit(raw),
        y,
        sample_weight=subject_equal_weights(subjects),
    )
    return model


def _apply_platt(model: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_)
    return np.asarray(
        model.predict_proba(_score_logit(raw))[:, int(np.flatnonzero(classes == 1)[0])],
        dtype=np.float64,
    )


def _feature_matrix(data: Mapping[str, Any], family: str) -> np.ndarray:
    if family == "Primary-only":
        return primary_only_features(data["primary"])[0]
    if family == "Heterogeneous-expert":
        return heterogeneous_features(data["primary"], data["feature"])[0]
    raise ValueError(f"unknown feature family: {family}")


def crossfit_configuration(
    *,
    config: Mapping[str, Any],
    family: str,
    fit_data: Mapping[str, Any],
    risk_folds: Sequence[Mapping[str, Sequence[str]]],
) -> tuple[np.ndarray, np.ndarray]:
    x = _feature_matrix(fit_data, family)
    y = np.asarray(fit_data["errors"], dtype=np.int64)
    subjects = np.asarray(fit_data["subjects"])
    raw = np.full(len(y), np.nan, dtype=np.float64)
    for fold in risk_folds:
        train = np.isin(subjects, fold["train"])
        validation = np.isin(subjects, fold["validation"])
        model = _fit_model(config, x[train], y[train], subjects[train])
        raw[validation] = _positive_probability(model, x[validation])
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("base risk cross-fitting did not cover every FIT row")
    calibrated = np.full(len(y), np.nan, dtype=np.float64)
    for fold in risk_folds:
        calibration = np.isin(subjects, fold["train"])
        validation = np.isin(subjects, fold["validation"])
        platt = _fit_platt(raw[calibration], y[calibration], subjects[calibration])
        calibrated[validation] = _apply_platt(platt, raw[validation])
    if not np.all(np.isfinite(calibrated)):
        raise RuntimeError("held-subject Platt calibration did not cover FIT")
    return raw, calibrated


def _policy_rows(
    *,
    scores: np.ndarray,
    errors: np.ndarray,
    subjects: np.ndarray,
    identities: Sequence[Mapping[str, Any]],
    outer: str,
    method_id: str,
    require_minimum: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policies = build_candidate_policies(scores, subjects)
    rows: list[dict[str, Any]] = []
    for policy in policies:
        accept = build_acceptance_mask(
            scores,
            threshold=float(policy["threshold"]),
            tie_fraction=Fraction(
                int(policy["tie_fraction"]["numerator"]),
                int(policy["tie_fraction"]["denominator"]),
            ),
            identities=identities,
            method_id=method_id,
            protocol_hash=PROTOCOL_SHA256,
            outer=outer,
        )
        metric = subject_equal_risk(accept, errors, subjects)
        minimum = min(
            int(item["accepted"]) for item in metric["per_subject"].values()
        )
        rows.append(
            {
                "q": Fraction(int(policy["q"]["numerator"]), int(policy["q"]["denominator"])),
                "coverage": _fraction(metric["coverage"]),
                "risk": None if metric["risk"] is None else _fraction(metric["risk"]),
                "supported": require_minimum is None or minimum >= require_minimum,
                "minimum_accepted": minimum,
                "threshold": policy["threshold"],
                "tie_fraction": Fraction(
                    int(policy["tie_fraction"]["numerator"]),
                    int(policy["tie_fraction"]["denominator"]),
                ),
                "metric": metric,
            }
        )
    return policies, rows


def _choice_summary(choice: Mapping[str, Any] | None) -> dict[str, Any]:
    if choice is None:
        return {"coverage": Fraction(0), "risk": None, "q": Fraction(0)}
    return {
        "coverage": choice["coverage"],
        "risk": choice["risk"],
        "q": choice["q"],
    }


def _configuration_priority(config: Mapping[str, Any]) -> tuple[Any, ...]:
    if config["family"] == "logistic":
        return (0, float(config["C"]), str(config["id"]))
    return (1, str(config["id"]))


def _candidate_rank(
    coverage: Fraction,
    risk: Fraction | None,
    priority: tuple[Any, ...],
) -> tuple[Any, ...]:
    return (-coverage, Fraction(2) if risk is None else risk, priority)


def select_family_configuration(
    *,
    outer: str,
    family: str,
    fit_data: Mapping[str, Any],
    risk_folds: Sequence[Mapping[str, Sequence[str]]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for config in risk_registry():
        raw, calibrated = crossfit_configuration(
            config=config,
            family=family,
            fit_data=fit_data,
            risk_folds=risk_folds,
        )
        policies, rows = _policy_rows(
            scores=calibrated,
            errors=fit_data["errors"],
            subjects=fit_data["subjects"],
            identities=fit_data["identities"],
            outer=outer,
            method_id=f"{family}|{config['id']}",
            require_minimum=None,
        )
        choice = select_empirical_policy(rows, alpha=PRIMARY_ALPHA)
        summary = _choice_summary(choice)
        candidates.append(
            {
                "config": dict(config),
                "raw_oof": raw,
                "calibrated_oof": calibrated,
                "policies": policies,
                "policy_rows": rows,
                "choice": choice,
                **summary,
            }
        )
    selected = min(
        candidates,
        key=lambda item: _candidate_rank(
            item["coverage"], item["risk"], _configuration_priority(item["config"])
        ),
    )
    return {"family": family, "selected": selected, "candidates": candidates}


def _weighted_temperature_nll(
    log_temperature: float,
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    temperature = math.exp(log_temperature)
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    log_denominator = np.log(np.exp(scaled).sum(axis=1))
    loss = -scaled[np.arange(len(labels)), labels] + log_denominator
    return float(np.average(loss, weights=weights))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, subjects: np.ndarray) -> float:
    result = minimize_scalar(
        _weighted_temperature_nll,
        bounds=(math.log(0.05), math.log(10.0)),
        args=(logits, labels, subject_equal_weights(subjects)),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    return float(np.clip(math.exp(float(result.x)), 0.05, 10.0))


def temperature_risk(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probability = np.exp(scaled)
    probability /= probability.sum(axis=1, keepdims=True)
    return 1.0 - probability.max(axis=1)


def crossfit_temperature(
    fit_data: Mapping[str, Any], risk_folds: Sequence[Mapping[str, Sequence[str]]]
) -> tuple[np.ndarray, list[float]]:
    subjects = np.asarray(fit_data["subjects"])
    scores = np.full(len(subjects), np.nan, dtype=np.float64)
    temperatures: list[float] = []
    for fold in risk_folds:
        train = np.isin(subjects, fold["train"])
        validation = np.isin(subjects, fold["validation"])
        temperature = fit_temperature(
            fit_data["logits"][train], fit_data["labels"][train], subjects[train]
        )
        temperatures.append(temperature)
        scores[validation] = temperature_risk(fit_data["logits"][validation], temperature)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("temperature cross-fitting did not cover FIT")
    return scores, temperatures


def baseline_scores(data: Mapping[str, Any], temperature: float | None) -> dict[str, np.ndarray]:
    primary = np.asarray(data["primary"], dtype=np.float64)
    feature = np.asarray(data["feature"], dtype=np.float64)
    ordered = np.sort(primary, axis=1)
    entropy = -np.sum(primary * np.log(np.clip(primary, 1e-10, 1)), axis=1) / np.log(5)
    scores = {
        "MSP": 1.0 - primary.max(axis=1),
        "Margin": 1.0 - (ordered[:, -1] - ordered[:, -2]),
        "Entropy": entropy,
        "Forward-KL": forward_kl(primary, feature),
        "Disagreement": (
            np.argmax(primary, axis=1) != np.argmax(feature, axis=1)
        ).astype(np.float64),
    }
    if temperature is not None:
        scores["Temperature-MSP"] = temperature_risk(data["logits"], temperature)
    return scores


def _select_baseline(
    *, outer: str, fit_data: Mapping[str, Any], fit_scores: Mapping[str, np.ndarray]
) -> tuple[str, dict[str, Any]]:
    rows_by_method: dict[str, dict[str, Any]] = {}
    for method in BASELINE_ORDER:
        _, rows = _policy_rows(
            scores=fit_scores[method],
            errors=fit_data["errors"],
            subjects=fit_data["subjects"],
            identities=fit_data["identities"],
            outer=outer,
            method_id=f"Baseline|{method}",
            require_minimum=None,
        )
        choice = select_empirical_policy(rows, alpha=PRIMARY_ALPHA)
        rows_by_method[method] = {
            "choice": choice,
            **_choice_summary(choice),
            "policy_rows": rows,
        }
    order = {method: index for index, method in enumerate(BASELINE_ORDER)}
    selected = min(
        BASELINE_ORDER,
        key=lambda method: _candidate_rank(
            rows_by_method[method]["coverage"],
            rows_by_method[method]["risk"],
            (order[method],),
        ),
    )
    return selected, rows_by_method


def _development_execution_provenance(
    *,
    outer: str,
    fit_subjects: Sequence[str],
    cert_subjects: Sequence[str],
    components_root: Path,
    records: Mapping[str, Sequence[Path]],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    component_entries: list[dict[str, Any]] = []
    for subject in fit_subjects:
        directory = components_root / f"outer-{outer}" / f"inner-{subject}"
        _validated_component_payload(
            bundle_path=directory / "blind_predictions.npz",
            expected_roles=build_inner_roles(outer, subject, fit_subjects),
            expected_predicted_subjects=[subject],
        )
        component_entries.append(
            {
                "role": f"inner-{subject}",
                "component_manifest_sha256": sha256_file(
                    directory / "component_manifest.json"
                ),
                "run_config_sha256": sha256_file(directory / "run_config.json"),
                "bundle_sha256": sha256_file(directory / "blind_predictions.npz"),
            }
        )
    final_directory = components_root / f"outer-{outer}" / "final"
    _validated_component_payload(
        bundle_path=final_directory / "blind_predictions.npz",
        expected_roles=build_final_roles(outer, fit_subjects),
        expected_predicted_subjects=list(cert_subjects) + [outer],
    )
    component_entries.append(
        {
            "role": "final",
            "component_manifest_sha256": sha256_file(
                final_directory / "component_manifest.json"
            ),
            "run_config_sha256": sha256_file(final_directory / "run_config.json"),
            "bundle_sha256": sha256_file(final_directory / "blind_predictions.npz"),
        }
    )
    data_directories = {
        path.resolve().parent
        for paths in records.values()
        for path in paths
    }
    if len(data_directories) != 1:
        raise RuntimeError("risk development raw records have ambiguous roots")
    final_run_config = json.loads(
        (final_directory / "run_config.json").read_text(encoding="utf-8")
    )
    feature_manifest_path = final_run_config["execution_provenance"]["inputs"][
        "feature_cache"
    ]["manifest_path"]
    scientific_config = {
        "schema_version": RISK_SCHEMA,
        "protocol_hash": PROTOCOL_SHA256,
        "outer": outer,
        "fit_subjects": list(fit_subjects),
        "cert_subjects": list(cert_subjects),
        "risk_folds": build_risk_cv(outer, fit_subjects),
        "risk_registry": risk_registry(),
        "feature_schema_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "feature_schema.json"
        ),
        "model_registry_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "model_registry.json"
        ),
        "selection_alpha": PRIMARY_ALPHA,
        "alphas": list(ALPHAS),
    }
    provenance, provenance_hash = phase_provenance_hash(
        code=risk_code_provenance(),
        environment=audited_environment(),
        inputs={
            "components": component_entries,
            "component_set_hash": canonical_json_hash(component_entries),
            "raw": raw_inventory_provenance(str(next(iter(data_directories)))),
            "feature_cache": feature_cache_provenance(feature_manifest_path),
        },
        config=scientific_config,
    )
    return provenance, provenance_hash, scientific_config


def develop_outer_fold(
    *,
    outer: str,
    fit_subjects: Sequence[str],
    cert_subjects: Sequence[str],
    components_root: Path,
    records: Mapping[str, Sequence[Path]],
    ledger: RoleLedger,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    completion_path = output_dir / "development_manifest.json"
    artifact_path = output_dir / "development.joblib"
    execution_provenance, execution_provenance_hash, scientific_config = (
        _development_execution_provenance(
            outer=outer,
            fit_subjects=fit_subjects,
            cert_subjects=cert_subjects,
            components_root=components_root,
            records=records,
        )
    )
    if completion_path.exists():
        if not resume or not artifact_path.exists():
            raise FileExistsError(f"development output exists or is incomplete: {output_dir}")
        manifest = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            manifest.get("protocol_hash") != PROTOCOL_SHA256
            or manifest.get("execution_provenance_hash")
            != execution_provenance_hash
            or manifest.get("artifact_sha256") != sha256_file(artifact_path)
        ):
            raise RuntimeError("development resume manifest is incompatible")
        return manifest

    fit_data = load_fit_predictions(
        outer=outer,
        fit_subjects=fit_subjects,
        components_root=components_root,
        records=records,
        ledger=ledger,
    )
    final_by_subject = load_final_blind_predictions(
        outer=outer,
        fit_subjects=fit_subjects,
        cert_subjects=cert_subjects,
        components_root=components_root,
    )
    if set(final_by_subject) != set(cert_subjects) | {outer}:
        raise RuntimeError("final component bundle does not contain CERT4 plus outer")
    cert_data = _concatenate([final_by_subject[subject] for subject in cert_subjects])
    outer_data = final_by_subject[outer]
    risk_folds = build_risk_cv(outer, fit_subjects)
    learned = {
        family: select_family_configuration(
            outer=outer,
            family=family,
            fit_data=fit_data,
            risk_folds=risk_folds,
        )
        for family in ("Primary-only", "Heterogeneous-expert")
    }
    family_order = {"Primary-only": 0, "Heterogeneous-expert": 1}
    learned_selected = min(
        learned,
        key=lambda family: _candidate_rank(
            learned[family]["selected"]["coverage"],
            learned[family]["selected"]["risk"],
            (family_order[family],),
        ),
    )

    learned_scores: dict[str, dict[str, np.ndarray]] = {}
    final_models: dict[str, Any] = {}
    learned_fit_scores: dict[str, np.ndarray] = {}
    for family, family_result in learned.items():
        selected = family_result["selected"]
        config = selected["config"]
        x_fit = _feature_matrix(fit_data, family)
        base = _fit_model(config, x_fit, fit_data["errors"], fit_data["subjects"])
        platt = _fit_platt(
            selected["raw_oof"], fit_data["errors"], fit_data["subjects"]
        )
        learned_fit_scores[family] = selected["calibrated_oof"]
        learned_scores[family] = {}
        for partition, data in (("cert", cert_data), ("outer", outer_data)):
            raw = _positive_probability(base, _feature_matrix(data, family))
            learned_scores[family][partition] = _apply_platt(platt, raw)
        final_models[family] = {"config": config, "base": base, "platt": platt}
    learned_scores["Learned-Selected"] = learned_scores[learned_selected]
    learned_fit_scores["Learned-Selected"] = learned_fit_scores[learned_selected]
    method_full_ids = {
        family: f"{family}|{learned[family]['selected']['config']['id']}"
        for family in ("Primary-only", "Heterogeneous-expert")
    }
    method_full_ids["Learned-Selected"] = method_full_ids[learned_selected]

    fit_temp_score, crossfit_temperatures = crossfit_temperature(fit_data, risk_folds)
    deployment_temperature = fit_temperature(
        fit_data["logits"], fit_data["labels"], fit_data["subjects"]
    )
    fit_baselines = baseline_scores(fit_data, temperature=None)
    fit_baselines["Temperature-MSP"] = fit_temp_score
    fit_baselines = {method: fit_baselines[method] for method in BASELINE_ORDER}
    baseline_selected, baseline_development = _select_baseline(
        outer=outer, fit_data=fit_data, fit_scores=fit_baselines
    )
    cert_baselines = baseline_scores(cert_data, deployment_temperature)
    outer_baselines = baseline_scores(outer_data, deployment_temperature)
    cert_baselines["Baseline-Selected"] = cert_baselines[baseline_selected]
    outer_baselines["Baseline-Selected"] = outer_baselines[baseline_selected]
    fit_baselines["Baseline-Selected"] = fit_baselines[baseline_selected]
    method_full_ids.update(
        {method: f"Baseline|{method}" for method in BASELINE_ORDER}
    )
    method_full_ids["Baseline-Selected"] = method_full_ids[baseline_selected]

    artifact = {
        "schema_version": RISK_SCHEMA,
        "protocol_hash": PROTOCOL_SHA256,
        "execution_provenance": execution_provenance,
        "execution_provenance_hash": execution_provenance_hash,
        "scientific_config": scientific_config,
        "outer": outer,
        "fit_subjects": list(fit_subjects),
        "cert_subjects": list(cert_subjects),
        "risk_folds": risk_folds,
        "fit_data": fit_data,
        "cert_data_blind": cert_data,
        "outer_data_blind": outer_data,
        "learned": learned,
        "learned_selected_family": learned_selected,
        "learned_fit_scores": learned_fit_scores,
        "learned_scores": learned_scores,
        "final_models": final_models,
        "fit_baseline_scores": fit_baselines,
        "cert_baseline_scores": cert_baselines,
        "outer_baseline_scores": outer_baselines,
        "baseline_selected_method": baseline_selected,
        "baseline_development": baseline_development,
        "method_full_ids": method_full_ids,
        "temperature": {
            "crossfit": crossfit_temperatures,
            "deployment": deployment_temperature,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_joblib_dump(artifact_path, artifact)
    summary = {
        "schema_version": RISK_SCHEMA,
        "protocol_hash": PROTOCOL_SHA256,
        "execution_provenance_hash": execution_provenance_hash,
        "status": "DEVELOPMENT_COMPLETE",
        "outer": outer,
        "learned_selected_family": learned_selected,
        "baseline_selected_method": baseline_selected,
        "primary_selected_config": learned["Primary-only"]["selected"]["config"],
        "heterogeneous_selected_config": learned["Heterogeneous-expert"]["selected"]["config"],
        "deployment_temperature": deployment_temperature,
        "artifact_path": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
    }
    atomic_write_json(completion_path, summary)
    return summary


def construct_and_freeze_outer_selectors(
    *,
    outer: str,
    cert_subjects: Sequence[str],
    records: Mapping[str, Sequence[Path]],
    ledger: RoleLedger,
    development_dir: Path,
    calibration_dir: Path,
    outer_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    """Use CERT labels once, then freeze a label-free outer decision artifact."""
    manifest_path = outer_dir / "outer_prediction_manifest.json"
    artifact_path = outer_dir / "outer_decisions.npz"
    development_path = development_dir / "development.joblib"
    development_manifest_path = development_dir / "development_manifest.json"
    development_manifest = json.loads(
        development_manifest_path.read_text(encoding="utf-8")
    )
    if development_manifest.get("artifact_sha256") != sha256_file(development_path):
        raise RuntimeError("development artifact content changed before selector construction")
    artifact = joblib.load(development_path)
    if (
        artifact.get("protocol_hash") != PROTOCOL_SHA256
        or artifact.get("outer") != outer
        or artifact.get("execution_provenance_hash")
        != development_manifest.get("execution_provenance_hash")
    ):
        raise RuntimeError("development artifact is incompatible")
    learned_methods = ("Primary-only", "Heterogeneous-expert", "Learned-Selected")
    baseline_methods = BASELINE_ORDER + ("Baseline-Selected",)
    methods = learned_methods + baseline_methods
    method_full_ids = artifact["method_full_ids"]
    selector_config = {
        "schema_version": "stage2-selector-calibration-v1",
        "protocol_hash": PROTOCOL_SHA256,
        "outer": outer,
        "cert_subjects": list(cert_subjects),
        "methods": list(methods),
        "method_full_ids": method_full_ids,
        "alphas": list(ALPHAS),
        "candidate_q": [index / 20 for index in range(1, 21)],
        "minimum_accepted_per_cert_subject": 50,
    }
    selector_provenance, selector_provenance_hash = phase_provenance_hash(
        code=risk_code_provenance(),
        environment=audited_environment(),
        inputs={
            "development_sha256": sha256_file(development_path),
            "development_manifest_sha256": sha256_file(development_manifest_path),
            "development_provenance_hash": artifact["execution_provenance_hash"],
        },
        config=selector_config,
    )
    if manifest_path.exists():
        if not resume or not artifact_path.exists():
            raise FileExistsError(f"outer selector exists or is incomplete: {outer_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("protocol_hash") != PROTOCOL_SHA256
            or manifest.get("selector_provenance_hash")
            != selector_provenance_hash
            or manifest.get("artifact_sha256") != sha256_file(artifact_path)
            or manifest.get("contains_outer_labels") is not False
            or manifest.get("development_sha256") != sha256_file(development_path)
            or manifest.get("calibration_sha256")
            != sha256_file(calibration_dir / "selector_calibration.json")
        ):
            raise RuntimeError("frozen outer selector manifest is incompatible")
        return manifest

    cert_data = artifact["cert_data_blind"]
    outer_data = artifact["outer_data_blind"]
    cert_labels, label_ids = load_labels_authorized(
        records,
        cert_subjects,
        ledger=ledger,
        phase="CERT_SELECTOR",
    )
    cert_labels = align_values(label_ids, cert_labels, cert_data["identities"])
    cert_errors = (
        np.argmax(cert_data["primary"], axis=1).astype(np.int64) != cert_labels
    ).astype(np.int64)
    cert_subject_array = np.asarray(cert_data["subjects"])

    fit_scores = {
        **artifact["learned_fit_scores"],
        **artifact["fit_baseline_scores"],
    }
    cert_scores = {
        **{method: artifact["learned_scores"][method]["cert"] for method in learned_methods},
        **artifact["cert_baseline_scores"],
    }
    outer_scores = {
        **{method: artifact["learned_scores"][method]["outer"] for method in learned_methods},
        **artifact["outer_baseline_scores"],
    }
    fit_subjects = np.asarray(artifact["fit_data"]["subjects"])
    selectors: dict[str, dict[str, Any]] = {}
    outer_masks: dict[str, np.ndarray] = {}
    for method in methods:
        policies = build_candidate_policies(fit_scores[method], fit_subjects)
        cert_rows: list[dict[str, Any]] = []
        for policy in policies:
            tie_fraction = Fraction(
                int(policy["tie_fraction"]["numerator"]),
                int(policy["tie_fraction"]["denominator"]),
            )
            accepted = build_acceptance_mask(
                cert_scores[method],
                threshold=float(policy["threshold"]),
                tie_fraction=tie_fraction,
                identities=cert_data["identities"],
                method_id=method_full_ids[method],
                protocol_hash=PROTOCOL_SHA256,
                outer=outer,
            )
            metric = subject_equal_risk(accepted, cert_errors, cert_subject_array)
            minimum = min(
                int(item["accepted"]) for item in metric["per_subject"].values()
            )
            cert_rows.append(
                {
                    "q": Fraction(
                        int(policy["q"]["numerator"]),
                        int(policy["q"]["denominator"]),
                    ),
                    "coverage": _fraction(metric["coverage"]),
                    "risk": None if metric["risk"] is None else _fraction(metric["risk"]),
                    "supported": minimum >= 50,
                    "minimum_accepted": minimum,
                    "threshold": float(policy["threshold"]),
                    "tie_fraction": tie_fraction,
                    "metric": metric,
                }
            )
        selectors[method] = {
            "full_method_id": method_full_ids[method],
            "candidate_rows": cert_rows,
            "alphas": {},
        }
        for alpha in ALPHAS:
            choice = select_empirical_policy(cert_rows, alpha=alpha)
            alpha_key = f"{alpha:.2f}"
            if choice is None:
                outer_accept = np.zeros(len(outer_scores[method]), dtype=bool)
                empirical = {
                    "q": Fraction(0),
                    "coverage": Fraction(0),
                    "risk": None,
                    "threshold": None,
                    "tie_fraction": Fraction(0),
                    "zero_fallback": True,
                    "supported": False,
                    "minimum_accepted_per_cert_subject": 0,
                    "cert_metric": None,
                    "cert_risk_ucb": None,
                }
            else:
                outer_accept = build_acceptance_mask(
                    outer_scores[method],
                    threshold=float(choice["threshold"]),
                    tie_fraction=choice["tie_fraction"],
                    identities=outer_data["identities"],
                    method_id=method_full_ids[method],
                    protocol_hash=PROTOCOL_SHA256,
                    outer=outer,
                )
                empirical = {
                    "q": choice["q"],
                    "coverage": choice["coverage"],
                    "risk": choice["risk"],
                    "threshold": choice["threshold"],
                    "tie_fraction": choice["tie_fraction"],
                    "zero_fallback": False,
                    "supported": bool(choice["supported"]),
                    "minimum_accepted_per_cert_subject": int(
                        choice["minimum_accepted"]
                    ),
                    "cert_metric": choice["metric"],
                    "cert_risk_ucb": 1.0,
                }
            outer_masks[f"{method}|{alpha_key}|empirical"] = outer_accept
            outer_masks[f"{method}|{alpha_key}|conservative"] = np.zeros(
                len(outer_scores[method]), dtype=bool
            )
            selectors[method]["alphas"][alpha_key] = {
                "empirical": empirical,
                "conservative": {
                    "coverage": Fraction(0),
                    "risk_ucb": None,
                    "zero_fallback": True,
                    "certificate_supported": False,
                    "reason": "m=4,M=20 Hoeffding ratio UCB is provably vacuous",
                },
            }

    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_payload = {
        "schema_version": "stage2-selector-calibration-v1",
        "protocol_hash": PROTOCOL_SHA256,
        "selector_provenance": selector_provenance,
        "selector_provenance_hash": selector_provenance_hash,
        "outer": outer,
        "cert_subjects": list(cert_subjects),
        "selectors": _json_safe(selectors),
        "learned_selected_family": artifact["learned_selected_family"],
        "baseline_selected_method": artifact["baseline_selected_method"],
        "method_full_ids": method_full_ids,
    }
    atomic_write_json(calibration_dir / "selector_calibration.json", calibration_payload)
    method_list = list(methods)
    mask_names = sorted(outer_masks)
    arrays = {
        "identities_json": np.asarray(
            json.dumps(outer_data["identities"], sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
        "primary_logits": np.asarray(outer_data["logits"], dtype=np.float32),
        "primary_probabilities": np.asarray(outer_data["primary"], dtype=np.float64),
        "primary_predictions": np.argmax(outer_data["primary"], axis=1).astype(np.int64),
        "feature_probabilities": np.asarray(outer_data["feature"], dtype=np.float64),
        "method_names": np.asarray(method_list, dtype=np.str_),
        "risk_scores": np.column_stack([outer_scores[method] for method in method_list]),
        "mask_names": np.asarray(mask_names, dtype=np.str_),
        "acceptance_masks": np.column_stack([outer_masks[name] for name in mask_names]),
        "selector_metadata_json": np.asarray(
            json.dumps(_json_safe(selectors), sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
        "selector_provenance_hash": np.asarray(
            selector_provenance_hash, dtype=np.str_
        ),
    }
    forbidden = {"labels", "y", "y_true", "errors", "correct"}
    if forbidden & set(arrays):
        raise RuntimeError("outer decision artifact contains an outcome key")
    atomic_write_npz(artifact_path, arrays)
    manifest = {
        "schema_version": "stage2-frozen-outer-predictions-v1",
        "status": "OUTER_PREDICTIONS_FROZEN",
        "protocol_hash": PROTOCOL_SHA256,
        "selector_provenance_hash": selector_provenance_hash,
        "outer": outer,
        "contains_outer_labels": False,
        "artifact_path": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "development_sha256": sha256_file(development_path),
        "calibration_sha256": sha256_file(calibration_dir / "selector_calibration.json"),
        "learned_selected_family": artifact["learned_selected_family"],
        "baseline_selected_method": artifact["baseline_selected_method"],
        "methods": method_list,
        "method_full_ids": method_full_ids,
        "alphas": list(ALPHAS),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


__all__ = [
    "ALPHAS",
    "BASELINE_ORDER",
    "PRIMARY_ALPHA",
    "construct_and_freeze_outer_selectors",
    "develop_outer_fold",
    "risk_registry",
]
