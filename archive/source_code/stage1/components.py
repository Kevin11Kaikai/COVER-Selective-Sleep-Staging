"""Leakage-safe component fitting and label-blind prediction generation."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from scripts.stage1_clean_nested_loso.feature_cache import FEATURE_CACHE_SCHEMA_HASH
from scripts.stage1_clean_nested_loso.run_outer_fold import (
    _restore_rng_state,
    _rng_state,
    atomic_torch_save,
    build_primary_model,
    class_weights,
    evaluate_selection_accuracy,
    fit_feature_expert,
    set_seed,
    train_epoch,
)
from sleepedf.data_loader import SleepEDFDataset

from . import PROTOCOL_SHA256
from .bundles import load_blind_bundle, validate_blind_bundle
from .data_io import (
    align_values,
    atomic_write_json,
    atomic_write_npz,
    canonical_json_hash,
    files_for_subjects,
    load_feature_x_blind,
    load_raw_x_blind,
    sha256_file,
)
from .firewall import RoleLedger, assert_role_exclusion
from .provenance import (
    audited_environment,
    component_code_provenance,
    feature_cache_provenance,
    phase_provenance_hash,
    raw_inventory_provenance,
)
from .splits import derive_fit_seed, normalize_subject


COMPONENT_SCHEMA = "stage2-component-v1"
RAW_DATA_HASH = "6553de7de5c29572b4b99532ad648d95e54b90ff8d6c32a6fbd40e6aa435130a"
MAX_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 20


def _selection_macro(
    model: torch.nn.Module,
    loaders: Sequence[DataLoader],
    device: torch.device,
) -> tuple[float, float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for loader in loaders:
        loss, accuracy = evaluate_selection_accuracy(model, loader, device)
        rows.append({"loss": float(loss), "accuracy": float(accuracy)})
    return (
        float(np.mean([row["loss"] for row in rows])),
        float(np.mean([row["accuracy"] for row in rows])),
        rows,
    )


def train_primary_component(
    *,
    train_files: Sequence[Path],
    selection_files_by_subject: Sequence[Sequence[Path]],
    output_dir: Path,
    run_config_hash: str,
    seed: int,
    resume: bool,
    device: torch.device,
) -> tuple[Path, dict[str, Any]]:
    """Fit a Conformer and select by the mean accuracy of two subjects."""
    primary_dir = output_dir / "primary"
    best_path = primary_dir / "best_model.pth"
    recovery_path = primary_dir / "last_checkpoint.pth"
    history_path = primary_dir / "history.json"
    generator = torch.Generator().manual_seed(seed)
    set_seed(seed)
    train_dataset = SleepEDFDataset(train_files)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    selection_datasets = [SleepEDFDataset(files) for files in selection_files_by_subject]
    selection_loaders = [
        DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        for dataset in selection_datasets
    ]
    if len(selection_loaders) != 2:
        raise RuntimeError("checkpoint selection requires exactly two subjects")

    model = build_primary_model(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_files).to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=0, last_epoch=-1
    )
    start_epoch = 1
    best_accuracy = -1.0
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, Any]] = []

    if resume and recovery_path.exists():
        recovery = torch.load(recovery_path, map_location=device, weights_only=False)
        if recovery.get("run_config_hash") != run_config_hash:
            raise RuntimeError("component recovery belongs to a different config")
        model.load_state_dict(recovery["model_state_dict"])
        optimizer.load_state_dict(recovery["optimizer_state_dict"])
        scheduler.load_state_dict(recovery["scheduler_state_dict"])
        start_epoch = int(recovery["epoch"]) + 1
        best_accuracy = float(recovery["best_accuracy"])
        best_epoch = int(recovery["best_epoch"])
        without_improvement = int(recovery["without_improvement"])
        history = list(recovery["history"])
        _restore_rng_state(recovery["rng_state"], generator)
    elif recovery_path.exists() or best_path.exists():
        raise FileExistsError(f"component output exists; use resume: {primary_dir}")

    if without_improvement < PATIENCE:
        for epoch in range(start_epoch, MAX_EPOCHS + 1):
            started = time.perf_counter()
            train_loss, train_accuracy = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            selection_loss, selection_accuracy, subject_rows = _selection_macro(
                model, selection_loaders, device
            )
            scheduler.step()
            improved = selection_accuracy > best_accuracy
            if improved:
                best_accuracy = selection_accuracy
                best_epoch = epoch
                without_improvement = 0
                atomic_torch_save(
                    best_path,
                    {
                        "schema_version": COMPONENT_SCHEMA,
                        "protocol_hash": PROTOCOL_SHA256,
                        "run_config_hash": run_config_hash,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "selection_macro_accuracy": selection_accuracy,
                        "selection_macro_loss": selection_loss,
                        "selection_subject_metrics": subject_rows,
                        "selection_rule": (
                            "maximum mean of two per-subject accuracies; "
                            "earliest epoch on exact ties"
                        ),
                    },
                )
            else:
                without_improvement += 1
            epoch_row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "selection_macro_loss": selection_loss,
                "selection_macro_accuracy": selection_accuracy,
                "selection_subject_metrics": subject_rows,
                "improved": improved,
                "seconds": time.perf_counter() - started,
            }
            history.append(epoch_row)
            atomic_torch_save(
                recovery_path,
                {
                    "schema_version": COMPONENT_SCHEMA,
                    "protocol_hash": PROTOCOL_SHA256,
                    "run_config_hash": run_config_hash,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_accuracy": best_accuracy,
                    "best_epoch": best_epoch,
                    "without_improvement": without_improvement,
                    "history": history,
                    "rng_state": _rng_state(generator),
                },
            )
            print(
                f"component={output_dir.name} epoch={epoch:03d} "
                f"train_acc={train_accuracy:.6f} "
                f"selection_macro_acc={selection_accuracy:.6f} "
                f"seconds={epoch_row['seconds']:.2f}",
                flush=True,
            )
            if PATIENCE > 0 and without_improvement >= PATIENCE:
                break
    if not best_path.exists() or best_epoch <= 0:
        raise RuntimeError("no component checkpoint was selected")
    atomic_write_json(history_path, history)
    summary = {
        "best_epoch": best_epoch,
        "best_selection_macro_accuracy": best_accuracy,
        "epochs_completed": len(history),
        "history_sha256": sha256_file(history_path),
    }
    del model, optimizer, scheduler, train_loader, train_dataset
    del selection_loaders, selection_datasets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_path, summary


def primary_inference_blind(
    *,
    checkpoint_path: Path,
    expected_run_config_hash: str,
    x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != COMPONENT_SCHEMA:
        raise RuntimeError("component checkpoint schema mismatch")
    if checkpoint.get("protocol_hash") != PROTOCOL_SHA256:
        raise RuntimeError("component checkpoint protocol mismatch")
    if checkpoint.get("run_config_hash") != expected_run_config_hash:
        raise RuntimeError("component checkpoint config mismatch")
    model = build_primary_model(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logits_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            batch = torch.from_numpy(x[start : start + BATCH_SIZE]).to(device)
            _, logits = model(batch)
            logits_rows.append(logits.cpu().numpy())
    logits_array = np.vstack(logits_rows).astype(np.float32)
    shifted = logits_array.astype(np.float64) - logits_array.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits_array, probability.astype(np.float64)


def _feature_split(
    roles: Mapping[str, Any], provenance_hash: str
) -> SimpleNamespace:
    return SimpleNamespace(
        inner_train=tuple(roles["component_train"]),
        calibration=tuple(roles["feature_calibration"]),
        outer_subject=roles["outer"],
        split_hash=canonical_json_hash(
            {
                "role_hash": roles["role_hash"],
                "component_provenance_hash": provenance_hash,
            }
        ),
    )


def _bundle_arrays(
    *,
    identities: list[dict[str, Any]],
    logits: np.ndarray,
    primary_probability: np.ndarray,
    feature_probability: np.ndarray,
    primary_hash: str,
    feature_hash: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "identities_json": np.asarray(
            json.dumps(identities, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
        "logits": logits,
        "primary_probabilities": primary_probability,
        "primary_predictions": np.argmax(primary_probability, axis=1).astype(np.int64),
        "primary_confidences": primary_probability.max(axis=1),
        "feature_probabilities": feature_probability,
        "feature_predictions": np.argmax(feature_probability, axis=1).astype(np.int64),
        "feature_confidences": feature_probability.max(axis=1),
        "primary_checkpoint_hash": np.asarray(primary_hash, dtype=np.str_),
        "feature_model_hash": np.asarray(feature_hash, dtype=np.str_),
        "manifest_json": np.asarray(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
    }


def run_component_fit(
    *,
    roles: Mapping[str, Any],
    predicted_subjects: Sequence[object],
    records: Mapping[str, Sequence[Path]],
    feature_manifest: Path,
    output_dir: Path,
    ledger: RoleLedger,
    resume: bool,
    device: torch.device,
) -> dict[str, Any]:
    """Fit both components and emit a strict, outcome-blind prediction bundle."""
    outer = normalize_subject(roles["outer"])
    role = str(roles["role"])
    target = str(roles["target"])
    predicted = tuple(normalize_subject(value) for value in predicted_subjects)
    completion_path = output_dir / "component_manifest.json"
    bundle_path = output_dir / "blind_predictions.npz"
    if not records:
        raise RuntimeError("component fit received no raw-record inventory")
    data_directories = {
        path.resolve().parent
        for paths in records.values()
        for path in paths
    }
    if len(data_directories) != 1:
        raise RuntimeError("raw records do not share one authoritative directory")
    seed = derive_fit_seed(outer, role, target)
    scientific_config = {
        "schema_version": COMPONENT_SCHEMA,
        "protocol_hash": PROTOCOL_SHA256,
        "roles": dict(roles),
        "predicted_subjects": list(predicted),
        "seed": seed,
        "primary": {
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "patience": PATIENCE,
            "scheduler": "CosineAnnealingLR(T_max=100,eta_min=0,last_epoch=-1)",
        },
        "feature": {
            "random_forest_estimators": 100,
            "max_depth": 10,
            "smote_ratio": 0.3,
            "calibration": "isotonic",
        },
    }
    raw_provenance = raw_inventory_provenance(str(next(iter(data_directories))))
    feature_provenance = feature_cache_provenance(str(feature_manifest.resolve()))
    code_provenance = component_code_provenance()
    environment_provenance = audited_environment()
    execution_provenance, execution_provenance_hash = phase_provenance_hash(
        code=code_provenance,
        environment=environment_provenance,
        inputs={
            "raw": raw_provenance,
            "feature_cache": feature_provenance,
        },
        config=scientific_config,
    )
    run_config = {
        **scientific_config,
        "execution_provenance": execution_provenance,
        "execution_provenance_hash": execution_provenance_hash,
    }
    run_config_hash = canonical_json_hash(run_config)
    if completion_path.exists():
        if not resume or not bundle_path.exists():
            raise FileExistsError(f"component fit exists or is incomplete: {output_dir}")
        manifest = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            manifest.get("protocol_hash") != PROTOCOL_SHA256
            or manifest.get("role_hash") != roles["role_hash"]
            or manifest.get("run_config_hash") != run_config_hash
            or manifest.get("execution_provenance_hash")
            != execution_provenance_hash
            or manifest.get("bundle_sha256") != sha256_file(bundle_path)
        ):
            raise RuntimeError("completed component manifest is incompatible")
        bundle = load_blind_bundle(bundle_path)
        bundle_manifest_raw = np.asarray(bundle["manifest_json"]).item()
        if isinstance(bundle_manifest_raw, bytes):
            bundle_manifest_raw = bundle_manifest_raw.decode("utf-8")
        bundle_manifest = json.loads(str(bundle_manifest_raw))
        if (
            bundle_manifest.get("generating_roles") != dict(roles)
            or bundle_manifest.get("provenance_hash")
            != execution_provenance_hash
            or bundle_manifest.get("hashes", {}).get("code")
            != code_provenance["hash"]
            or bundle_manifest.get("hashes", {}).get("data")
            != raw_provenance["file_set_hash"]
            or bundle_manifest.get("hashes", {}).get("config")
            != run_config_hash
        ):
            raise RuntimeError("completed blind bundle provenance is incompatible")
        return manifest

    for subject in predicted:
        assert_role_exclusion(
            outer=outer,
            predicted_subject=subject,
            component_train=roles["component_train"],
            checkpoint_selection=roles["checkpoint_selection"],
            feature_calibration=roles["feature_calibration"],
            cert=ledger.cert,
        )
    ledger.authorize_label_load(roles["component_train"], "COMPONENT_TRAIN")
    ledger.authorize_label_load(roles["checkpoint_selection"], "CHECKPOINT_SELECTION")
    ledger.authorize_label_load(roles["feature_calibration"], "FEATURE_CALIBRATION")

    atomic_write_json(output_dir / "run_config.json", run_config)
    started = time.time()
    train_files = files_for_subjects(records, roles["component_train"])
    selection_files = [
        files_for_subjects(records, [subject])
        for subject in roles["checkpoint_selection"]
    ]
    checkpoint_path, primary_summary = train_primary_component(
        train_files=train_files,
        selection_files_by_subject=selection_files,
        output_dir=output_dir,
        run_config_hash=run_config_hash,
        seed=seed,
        resume=resume,
        device=device,
    )
    feature_data_hash = sha256_file(feature_manifest)
    feature_model, feature_model_path, _, feature_metadata = fit_feature_expert(
        feature_manifest_path=feature_manifest,
        split=_feature_split(roles, execution_provenance_hash),
        output_dir=output_dir,
        feature_data_hash=feature_data_hash,
        feature_schema_hash=FEATURE_CACHE_SCHEMA_HASH,
        seed=seed,
        smote_ratio=0.3,
        resume=resume,
    )

    raw_x, identities = load_raw_x_blind(records, predicted)
    logits, primary_probability = primary_inference_blind(
        checkpoint_path=checkpoint_path,
        expected_run_config_hash=run_config_hash,
        x=raw_x,
        device=device,
    )
    feature_x, feature_identities = load_feature_x_blind(feature_manifest, predicted)
    feature_probability_unaligned = np.asarray(
        feature_model.predict_proba(feature_x), dtype=np.float64
    )
    feature_probability = align_values(
        feature_identities, feature_probability_unaligned, identities
    )
    if feature_probability.shape != primary_probability.shape:
        raise RuntimeError("component prediction matrices do not align")
    primary_hash = sha256_file(checkpoint_path)
    feature_hash = sha256_file(feature_model_path)
    bundle_manifest = {
        "schema_version": COMPONENT_SCHEMA,
        "protocol_hash": PROTOCOL_SHA256,
        "contains_labels": False,
        "outer": outer,
        "role": role,
        "target": target,
        "predicted_subjects": list(predicted),
        "generating_roles": dict(roles),
        "provenance_hash": execution_provenance_hash,
        "hashes": {
            "primary_checkpoint": primary_hash,
            "feature_model": feature_hash,
            "data": raw_provenance["file_set_hash"],
            "code": code_provenance["hash"],
            "config": run_config_hash,
        },
    }
    arrays = _bundle_arrays(
        identities=identities,
        logits=logits,
        primary_probability=primary_probability,
        feature_probability=feature_probability,
        primary_hash=primary_hash,
        feature_hash=feature_hash,
        manifest=bundle_manifest,
    )
    validate_blind_bundle(arrays, expected_ids=identities)
    atomic_write_npz(bundle_path, arrays)
    load_blind_bundle(bundle_path)
    completion = {
        "schema_version": COMPONENT_SCHEMA,
        "status": "COMPLETE",
        "protocol_hash": PROTOCOL_SHA256,
        "outer": outer,
        "role": role,
        "target": target,
        "role_hash": roles["role_hash"],
        "predicted_subjects": list(predicted),
        "seed": seed,
        "run_config_hash": run_config_hash,
        "execution_provenance_hash": execution_provenance_hash,
        "primary": primary_summary,
        "feature_model_metadata": feature_metadata,
        "bundle_path": bundle_path.name,
        "bundle_sha256": sha256_file(bundle_path),
        "wall_seconds": time.time() - started,
    }
    atomic_write_json(completion_path, completion)
    del feature_model, raw_x, feature_x
    gc.collect()
    return completion


__all__ = [
    "BATCH_SIZE",
    "COMPONENT_SCHEMA",
    "run_component_fit",
    "train_primary_component",
]
