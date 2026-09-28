#!/usr/bin/env python3
"""Run one provenance-safe Stage-1 outer fold.

This command is intentionally single-fold only.  It never loops over outer
subjects, and it never loads a legacy LOSO checkpoint or RF cache.  The outer
dataset is constructed only after primary checkpoint selection and feature
calibration are frozen.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

# CUDA reads this before the first cuBLAS handle is created.  ``setdefault``
# preserves an explicitly frozen caller value while making the campaign default
# deterministic and visible in the run manifest.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import imblearn
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from conformer import Conformer
from scripts.stage1_clean_nested_loso.contracts import (
    PredictionBundle,
    PredictionManifest,
    RowIdentity,
    align_one_to_one,
    atomic_write_json,
    build_cache_key,
    canonical_json_hash,
    hash_rows,
    sha256_file,
)
from scripts.stage1_clean_nested_loso.feature_cache import (
    EXTRACTOR_HASH,
    FEATURE_CACHE_SCHEMA_HASH,
    load_feature_rows,
    validate_feature_cache,
    validate_feature_cache_against_raw,
)
from scripts.stage1_clean_nested_loso.metrics import evaluate_policy_metrics
from scripts.stage1_clean_nested_loso.policy import (
    DEFAULT_KL_EPSILON,
    route_type1_forward_kl,
)
from scripts.stage1_clean_nested_loso.splits import (
    ASSIGNMENT_RULE,
    build_subject_split,
    normalize_subject_id,
)
from sleepedf.data_loader import SleepEDFDataset, parse_sleepedf_record


RUN_SCHEMA_VERSION = "stage1-clean-run-manifest-v1"
PREDICTION_SCHEMA_VERSION = "stage1-clean-prediction-bundle-v1"
DEFAULT_SEED = 20260816
DEFAULT_TAU_H = 0.9
DEFAULT_KAPPA = 0.15
DATASET_ID = "sleepedf20"


def validate_resumable_progress(
    progress: Any,
    *,
    expected_run_config_hash: str,
) -> None:
    """Reject any ambiguous or post-outer in-progress state."""

    if not isinstance(progress, dict):
        raise RuntimeError("in-progress manifest must be a JSON mapping")
    if progress.get("run_config_hash") != expected_run_config_hash:
        raise RuntimeError("in-progress manifest belongs to a different run config")
    if "outer_evaluation_started" not in progress:
        raise RuntimeError("in-progress manifest lacks the outer-evaluation marker")
    marker = progress["outer_evaluation_started"]
    if type(marker) is not bool or marker is not False:
        raise RuntimeError(
            "prior execution may already have accessed outer rows; refusing to "
            "repeat inference or claim outer_evaluation_count=1"
        )
    if "outer_evaluation_count" not in progress:
        raise RuntimeError("in-progress manifest lacks outer_evaluation_count")
    evaluation_count = progress["outer_evaluation_count"]
    if type(evaluation_count) is not int or evaluation_count != 0:
        raise RuntimeError(
            "in-progress outer_evaluation_count must be the integer zero"
        )


def set_seed(seed: int) -> None:
    """Configure deterministic Python, NumPy, and Torch state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def atomic_torch_save(path: Path, value: Any) -> None:
    """Atomically persist a Torch checkpoint beside its destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(value, temporary)
        # Windows' ``os.fsync`` maps to a writable-handle commit operation;
        # reopening read-only raises ``OSError: [Errno 9] Bad file descriptor``.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def atomic_joblib_dump(path: Path, value: Any) -> None:
    """Atomically persist a feature model and its embedded cache metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        joblib.dump(value, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def hash_named_files(paths: Iterable[Path]) -> str:
    """Hash file names and content hashes in stable order."""
    rows = [
        {"name": path.name, "sha256": sha256_file(path)}
        for path in sorted(set(map(Path, paths)), key=lambda item: item.name)
    ]
    if not rows:
        raise ValueError("cannot hash an empty file set")
    return canonical_json_hash(rows)


def discover_raw_records(data_dir: Path) -> dict[str, list[Path]]:
    """Discover raw records without reading labels or performance values."""
    records: dict[str, list[Path]] = {}
    for path in sorted(data_dir.glob("SC*.npz"), key=lambda item: item.name):
        subject_id, _ = parse_sleepedf_record(path)
        subject = normalize_subject_id(subject_id)
        records.setdefault(subject, []).append(path)
    if not records:
        raise FileNotFoundError(f"no Sleep-EDF record NPZs under {data_dir}")
    return records


def files_for_subjects(
    records: dict[str, list[Path]], subjects: Sequence[str]
) -> list[Path]:
    """Return sorted files for an explicit, already validated subject role."""
    result: list[Path] = []
    for subject in subjects:
        if subject not in records:
            raise FileNotFoundError(f"no raw records for subject {subject}")
        result.extend(records[subject])
    return sorted(result, key=lambda item: item.name)


def class_weights(files: Sequence[Path], n_classes: int = 5) -> torch.Tensor:
    """Compute class weights from inner-training subjects only."""
    labels = [np.load(path, allow_pickle=False)["y"] for path in files]
    counts = np.bincount(np.concatenate(labels).astype(int), minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError(f"inner training is missing a class: counts={counts.tolist()}")
    weights = counts.sum() / (n_classes * counts.astype(np.float64))
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def build_primary_model(device: torch.device) -> Conformer:
    """Construct the frozen EEG-Conformer architecture."""
    model = Conformer(
        emb_size=40,
        depth=6,
        n_classes=5,
        n_electrodes=1,
    ).to(device)
    with torch.no_grad():
        model(torch.zeros(1, 1, 1, 3000, device=device))
    return model


def train_epoch(
    model: Conformer,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train one epoch and return loss and accuracy."""
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for batch, labels in loader:
        batch = batch.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * len(labels)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        count += len(labels)
    if count == 0:
        raise RuntimeError("empty inner-training loader")
    return loss_sum / count, correct / count


def evaluate_selection_accuracy(
    model: Conformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate checkpoint-selection subjects without retaining predictions."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device)
            labels = labels.to(device)
            _, logits = model(batch)
            loss_sum += float(criterion(logits, labels).item()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += len(labels)
    if count == 0:
        raise RuntimeError("empty checkpoint-selection loader")
    return loss_sum / count, correct / count


def _rng_state(generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def train_primary(
    *,
    train_files: Sequence[Path],
    selection_files: Sequence[Path],
    output_dir: Path,
    run_config_hash: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
    resume: bool,
    device: torch.device,
) -> tuple[Path, list[dict[str, Any]], int, float, float]:
    """Train/select one primary checkpoint without constructing outer rows."""
    primary_dir = output_dir / "primary"
    best_path = primary_dir / "best_model.pth"
    recovery_path = primary_dir / "last_checkpoint.pth"
    generator = torch.Generator().manual_seed(seed)
    train_dataset = SleepEDFDataset(train_files)
    selection_dataset = SleepEDFDataset(selection_files)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_primary_model(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_files).to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start_epoch = 1
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, Any]] = []

    if resume and recovery_path.exists():
        recovery = torch.load(
            recovery_path,
            map_location=device,
            weights_only=False,
        )
        if recovery.get("run_config_hash") != run_config_hash:
            raise RuntimeError("recovery checkpoint belongs to a different run config")
        model.load_state_dict(recovery["model_state_dict"])
        optimizer.load_state_dict(recovery["optimizer_state_dict"])
        scheduler.load_state_dict(recovery["scheduler_state_dict"])
        start_epoch = int(recovery["epoch"]) + 1
        best_accuracy = float(recovery["best_accuracy"])
        best_loss = float(recovery["best_loss"])
        best_epoch = int(recovery["best_epoch"])
        without_improvement = int(recovery["without_improvement"])
        history = list(recovery["history"])
        _restore_rng_state(recovery["rng_state"], generator)
    elif recovery_path.exists() or best_path.exists():
        raise FileExistsError(
            f"primary output already exists under {primary_dir}; use --resume"
        )

    if without_improvement < patience:
        for epoch in range(start_epoch, epochs + 1):
            started = time.perf_counter()
            train_loss, train_accuracy = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )
            selection_loss, selection_accuracy = evaluate_selection_accuracy(
                model,
                selection_loader,
                device,
            )
            scheduler.step()
            epoch_seconds = time.perf_counter() - started
            improved = selection_accuracy > best_accuracy
            if improved:
                best_accuracy = selection_accuracy
                best_loss = selection_loss
                best_epoch = epoch
                without_improvement = 0
                atomic_torch_save(
                    best_path,
                    {
                        "schema_version": RUN_SCHEMA_VERSION,
                        "run_config_hash": run_config_hash,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "selection_accuracy": best_accuracy,
                        "selection_loss": best_loss,
                        "selection_rule": (
                            "maximum checkpoint-selection accuracy; "
                            "ties keep earliest epoch"
                        ),
                    },
                )
            else:
                without_improvement += 1

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "selection_loss": selection_loss,
                    "selection_accuracy": selection_accuracy,
                    "improved": improved,
                    "seconds": epoch_seconds,
                }
            )
            atomic_torch_save(
                recovery_path,
                {
                    "schema_version": RUN_SCHEMA_VERSION,
                    "run_config_hash": run_config_hash,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_accuracy": best_accuracy,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "without_improvement": without_improvement,
                    "history": history,
                    "rng_state": _rng_state(generator),
                },
            )
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"train_acc={train_accuracy:.6f} "
                f"selection_loss={selection_loss:.6f} "
                f"selection_acc={selection_accuracy:.6f} "
                f"seconds={epoch_seconds:.2f}"
            )
            if patience > 0 and without_improvement >= patience:
                break

    if not best_path.exists() or best_epoch <= 0:
        raise RuntimeError("no primary checkpoint was selected")
    atomic_write_json(primary_dir / "history.json", history)
    del model, optimizer, scheduler, train_loader, selection_loader
    del train_dataset, selection_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_path, history, best_epoch, best_accuracy, best_loss


def primary_outer_inference(
    *,
    checkpoint_path: Path,
    expected_run_config_hash: str,
    outer_files: Sequence[Path],
    outer_fold: str,
    batch_size: int,
    device: torch.device,
) -> tuple[
    tuple[RowIdentity, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate the frozen primary model once on the outer subject."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != RUN_SCHEMA_VERSION:
        raise RuntimeError("primary checkpoint schema does not match the clean run")
    if checkpoint.get("run_config_hash") != expected_run_config_hash:
        raise RuntimeError("primary checkpoint belongs to a different run config")
    if not isinstance(checkpoint.get("epoch"), int) or checkpoint["epoch"] <= 0:
        raise RuntimeError("primary checkpoint has no valid selected epoch")
    model = build_primary_model(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = SleepEDFDataset(
        outer_files,
        return_identity=True,
        dataset_id=DATASET_ID,
        outer_fold=int(outer_fold),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    probability_rows: list[np.ndarray] = []
    prediction_rows: list[np.ndarray] = []
    confidence_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    identities: list[RowIdentity] = []

    def metadata_item(values: Any, index: int) -> Any:
        if isinstance(values, torch.Tensor):
            return values[index].item()
        if isinstance(values, np.ndarray):
            return values[index].item() if values[index].ndim == 0 else values[index]
        return values[index]

    with torch.no_grad():
        for batch, labels, metadata in loader:
            batch = batch.to(device)
            _, logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)
            probability_rows.append(probabilities.cpu().numpy())
            prediction_rows.append(prediction.cpu().numpy())
            confidence_rows.append(confidence.cpu().numpy())
            label_rows.append(labels.numpy())
            for index in range(len(labels)):
                metadata_label = int(
                    metadata_item(metadata["ground_truth_label"], index)
                )
                if metadata_label != int(labels[index].item()):
                    raise RuntimeError(
                        "outer batch metadata label differs from the inference label"
                    )
                identities.append(
                    RowIdentity(
                        dataset=str(metadata_item(metadata["dataset"], index)),
                        subject_id=normalize_subject_id(
                            metadata_item(metadata["subject_id"], index)
                        ),
                        record_id=str(metadata_item(metadata["record_id"], index)),
                        epoch_index=int(
                            metadata_item(metadata["epoch_index"], index)
                        ),
                        outer_fold=normalize_subject_id(
                            metadata_item(metadata["outer_fold"], index)
                        ),
                    )
                )

    identity_tuple = tuple(identities)
    labels = np.concatenate(label_rows).astype(np.int64)
    probabilities = np.vstack(probability_rows)
    predictions = np.concatenate(prediction_rows).astype(np.int64)
    confidences = np.concatenate(confidence_rows)
    del model, loader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return identity_tuple, labels, probabilities, predictions, confidences


def fit_feature_expert(
    *,
    feature_manifest_path: Path,
    split: Any,
    output_dir: Path,
    feature_data_hash: str,
    feature_schema_hash: str,
    seed: int,
    smote_ratio: float,
    resume: bool,
) -> tuple[Any, Path, Path, dict[str, Any]]:
    """Fit RF on inner-train and isotonic calibration on disjoint subjects."""
    x_train, y_train, _, feature_names = load_feature_rows(
        feature_manifest_path,
        split.inner_train,
        outer_fold=split.outer_subject,
    )
    x_calibration, y_calibration, _, calibration_names = load_feature_rows(
        feature_manifest_path,
        split.calibration,
        outer_fold=split.outer_subject,
    )
    if tuple(feature_names) != tuple(calibration_names):
        raise RuntimeError("feature schema differs between train and calibration shards")
    feature_config = {
        "expert": "random_forest_isotonic",
        "feature_names": list(feature_names),
        "standard_scaler": True,
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": -1,
        },
        "smote_ratio": smote_ratio,
        "smote_scope": "inner_train_only",
        "calibration": "isotonic_on_disjoint_subjects",
        "feature_cache_schema_hash": feature_schema_hash,
        "feature_extractor_hash": EXTRACTOR_HASH,
    }
    model_hash = canonical_json_hash(feature_config)
    calibration_hash = canonical_json_hash(
        {
            "method": "isotonic",
            "subjects": list(split.calibration),
            "smote": False,
        }
    )
    cache_key = build_cache_key(
        split_hash=split.split_hash,
        schema_hash=feature_schema_hash,
        model_hash=model_hash,
        data_hash=feature_data_hash,
        config_hash=model_hash,
        calibration_hash=calibration_hash,
        seed=seed,
    )
    model_path = output_dir / "feature" / "cache" / f"{cache_key}.joblib"
    sidecar_path = model_path.with_suffix(".json")
    expected_metadata = {
        "schema_version": RUN_SCHEMA_VERSION,
        "cache_key": cache_key,
        "split_hash": split.split_hash,
        "schema_hash": feature_schema_hash,
        "model_hash": model_hash,
        "data_hash": feature_data_hash,
        "extractor_hash": EXTRACTOR_HASH,
        "config_hash": model_hash,
        "calibration_hash": calibration_hash,
        "seed": seed,
        "training_subjects": list(split.inner_train),
        "calibration_subjects": list(split.calibration),
        "feature_config": feature_config,
    }
    if model_path.exists() or sidecar_path.exists():
        if not resume or not model_path.exists():
            raise FileExistsError(
                f"feature cache exists or is incomplete: {model_path}"
            )
        payload = joblib.load(model_path)
        if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
            raise RuntimeError("embedded feature cache metadata mismatch")
        if sidecar_path.exists():
            existing_metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if existing_metadata != expected_metadata:
                raise RuntimeError("feature cache metadata mismatch")
        else:
            # Embedded metadata is authoritative, so an interrupted sidecar
            # write is safely recoverable without refitting the model.
            atomic_write_json(sidecar_path, expected_metadata)
        return payload["model"], model_path, sidecar_path, expected_metadata

    counts = np.bincount(y_train, minlength=5)
    if np.any(counts == 0):
        raise RuntimeError(f"feature inner train lacks a class: {counts.tolist()}")
    target_count = int(counts.max() * smote_ratio)
    strategy = {
        class_index: max(int(count), target_count)
        for class_index, count in enumerate(counts)
    }
    if smote_ratio > 0:
        x_train, y_train = SMOTE(
            sampling_strategy=strategy,
            random_state=seed,
        ).fit_resample(x_train, y_train)

    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    base.fit(x_train, y_train)
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(base),
            method="isotonic",
        )
    except ImportError:
        calibrated = CalibratedClassifierCV(
            estimator=base,
            method="isotonic",
            cv="prefit",
        )
    calibrated.fit(x_calibration, y_calibration)
    if not np.array_equal(calibrated.classes_, np.arange(5)):
        raise RuntimeError(
            f"feature expert classes are incomplete: {calibrated.classes_}"
        )
    atomic_joblib_dump(
        model_path,
        {"model": calibrated, "metadata": expected_metadata},
    )
    atomic_write_json(sidecar_path, expected_metadata)
    return calibrated, model_path, sidecar_path, expected_metadata


def metric_to_dict(value: Any) -> Any:
    """Convert metric dataclasses, enums, and NumPy scalars to JSON values."""
    if hasattr(value, "value") and type(value).__module__ == "enum":
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: metric_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): metric_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metric_to_dict(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def artifact_entry(path: Path, fold_dir: Path) -> dict[str, str]:
    """Return manifest path/hash for an already written artifact."""
    return {
        "path": path.relative_to(fold_dir).as_posix(),
        "sha256": sha256_file(path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-subject", required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/Sleep-EDF-20/eeg_fpz_cz"),
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        required=True,
        help="identity-bearing manifest produced by feature_cache.py; legacy pickle caches are rejected",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/stage1_clean_nested_loso"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smote-ratio", type=float, default=0.3)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="frozen execution device (auto chooses CUDA when available)",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outer_subject = normalize_subject_id(args.outer_subject)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if args.early_stop_patience <= 0:
        raise ValueError("early-stop patience must be positive")
    if not 0 <= args.smote_ratio <= 1:
        raise ValueError("smote ratio must lie in [0, 1]")
    if args.min_support <= 0:
        raise ValueError("min-support must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")
    selected_device = (
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    set_seed(args.seed)
    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    feature_manifest_path = (PROJECT_ROOT / args.feature_manifest).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    records = discover_raw_records(data_dir)
    subjects = tuple(sorted(records))
    split = build_subject_split(
        outer_subject,
        subjects,
        seed=args.seed,
    )
    fold_dir = output_root / f"fold_{outer_subject}"
    final_manifest_path = fold_dir / "run_manifest.json"
    if final_manifest_path.exists():
        raise FileExistsError(
            f"completed fold already exists: {final_manifest_path}"
        )
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_files = files_for_subjects(records, split.inner_train)
    selection_files = files_for_subjects(records, split.checkpoint_selection)
    outer_files = files_for_subjects(records, (split.outer_subject,))
    all_raw_files = [path for paths in records.values() for path in paths]
    raw_data_hash = hash_named_files(all_raw_files)
    feature_cache_summary = validate_feature_cache(feature_manifest_path)
    feature_raw_binding = validate_feature_cache_against_raw(
        feature_manifest_path,
        data_dir,
    )
    if feature_raw_binding["raw_binding"] != "PASS":
        raise RuntimeError("feature cache is not bound to the current raw cohort")
    if set(feature_cache_summary["subjects"]) != set(subjects):
        raise RuntimeError(
            "identity-bearing feature cache subjects differ from the raw cohort"
        )
    feature_data_hash = sha256_file(feature_manifest_path)
    feature_cache_payload = json.loads(
        feature_manifest_path.read_text(encoding="utf-8")
    )
    feature_preprocessing_hash = canonical_json_hash(
        {
            "manifest_sha256": feature_data_hash,
            "extractor_hash": feature_cache_payload["extractor_hash"],
            "extractor_config_hash": feature_cache_payload["config_hash"],
            "extractor_code_hash": feature_cache_payload["code_hash"],
        }
    )
    source_paths = [
        Path(__file__),
        PROJECT_ROOT / "conformer.py",
        PROJECT_ROOT / "sleepedf" / "data_loader.py",
        PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "contracts.py",
        PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "policy.py",
        PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "metrics.py",
        PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "splits.py",
        PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "feature_cache.py",
    ]
    code_hash = hash_named_files(source_paths)

    def assert_frozen_inputs() -> None:
        current_raw_hash = hash_named_files(all_raw_files)
        validate_feature_cache(feature_manifest_path)
        validate_feature_cache_against_raw(feature_manifest_path, data_dir)
        current_feature_hash = sha256_file(feature_manifest_path)
        current_code_hash = hash_named_files(source_paths)
        if current_raw_hash != raw_data_hash:
            raise RuntimeError("raw input files changed during the fold run")
        if current_feature_hash != feature_data_hash:
            raise RuntimeError("feature cache changed during the fold run")
        if current_code_hash != code_hash:
            raise RuntimeError("campaign code changed during the fold run")
    prediction_schema = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "identity": [
            "dataset",
            "subject_id",
            "record_id",
            "epoch_index",
            "outer_fold",
        ],
        "arrays": [
            "labels",
            "probabilities",
            "predictions",
            "confidences",
        ],
    }
    prediction_schema_hash = canonical_json_hash(prediction_schema)
    primary_preprocessing_hash = canonical_json_hash(
        {
            "input": "frozen_preprocessed_sleepedf_npz",
            "runtime_normalization": "none",
            "sample_shape": [1, 1, 3000],
            "data_loader_hash": sha256_file(
                PROJECT_ROOT / "sleepedf" / "data_loader.py"
            ),
            "upstream_preprocessing_provenance": "unresolved_legacy_input",
        }
    )
    primary_config = {
        "architecture": "Conformer",
        "emb_size": 40,
        "depth": 6,
        "classes": 5,
        "electrodes": 1,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "early_stop_patience": args.early_stop_patience,
        "class_weights": True,
        "seed": args.seed,
    }
    primary_model_hash = canonical_json_hash(primary_config)
    try:
        input_paths = {
            "raw_data_dir": data_dir.relative_to(PROJECT_ROOT).as_posix(),
            "feature_cache_manifest": feature_manifest_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
        }
    except ValueError as error:
        raise RuntimeError("all run inputs must reside inside the repository") from error
    run_config = {
        "schema_version": RUN_SCHEMA_VERSION,
        "dataset": DATASET_ID,
        "outer_fold": outer_subject,
        "outer_subject_id": outer_subject,
        "seed": args.seed,
        "assignment_rule": ASSIGNMENT_RULE,
        "subject_split": split.as_dict(),
        "checkpoint_selection_rule": (
            "maximum checkpoint-selection accuracy; ties keep earliest epoch"
        ),
        "tau_h": DEFAULT_TAU_H,
        "kappa": DEFAULT_KAPPA,
        "kl_epsilon": DEFAULT_KL_EPSILON,
        "raw_data_hash": raw_data_hash,
        "feature_cache_hash": feature_data_hash,
        "preprocessing_hash": primary_preprocessing_hash,
        "feature_preprocessing_hash": feature_preprocessing_hash,
        "schema_hash": prediction_schema_hash,
        "feature_schema_hash": FEATURE_CACHE_SCHEMA_HASH,
        "feature_extractor_hash": EXTRACTOR_HASH,
        "input_paths": input_paths,
        "code_hash": code_hash,
        "primary_config": primary_config,
        "feature_smote_ratio": args.smote_ratio,
        "metric_min_support": args.min_support,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "imbalanced_learn": imblearn.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "selected_device": selected_device,
            "cuda_runtime": torch.version.cuda,
            "cudnn": (
                torch.backends.cudnn.version() if torch.cuda.is_available() else None
            ),
            "gpu_name": (
                torch.cuda.get_device_name(0) if selected_device == "cuda" else None
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "loky_max_cpu_count": os.environ.get("LOKY_MAX_CPU_COUNT"),
            "deterministic_algorithms": True,
            "tf32": False,
        },
    }
    run_config_hash = canonical_json_hash(run_config)
    progress_path = fold_dir / "run_manifest.inprogress.json"
    existing_progress: dict[str, Any] | None = None
    if progress_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"incomplete fold state exists: {progress_path}; use --resume"
            )
        existing_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        validate_resumable_progress(
            existing_progress,
            expected_run_config_hash=run_config_hash,
        )
    elif args.resume:
        raise FileNotFoundError("--resume requested without an in-progress manifest")

    prior_runtime_seconds = float(
        (existing_progress or {}).get("accumulated_runtime_seconds", 0.0)
    )
    started = time.perf_counter()
    in_progress_manifest = {
        **run_config,
        "run_config": run_config,
        "run_config_hash": run_config_hash,
        "primary_checkpoint_frozen": False,
        "feature_model_frozen": False,
        "outer_evaluation_started": False,
        "outer_evaluation_count": 0,
        "accumulated_runtime_seconds": prior_runtime_seconds,
        "phase": "PRIMARY_TRAINING",
    }
    atomic_write_json(progress_path, in_progress_manifest)

    device = torch.device(selected_device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_path, history, best_epoch, best_accuracy, best_loss = train_primary(
        train_files=train_files,
        selection_files=selection_files,
        output_dir=fold_dir,
        run_config_hash=run_config_hash,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.early_stop_patience,
        seed=args.seed,
        resume=args.resume,
        device=device,
    )
    selected_checkpoint = torch.load(
        best_path, map_location="cpu", weights_only=False
    )
    if selected_checkpoint.get("schema_version") != RUN_SCHEMA_VERSION:
        raise RuntimeError("selected checkpoint has the wrong schema")
    if selected_checkpoint.get("run_config_hash") != run_config_hash:
        raise RuntimeError("selected checkpoint has a stale run_config_hash")
    if int(selected_checkpoint.get("epoch", -1)) != best_epoch:
        raise RuntimeError("selected checkpoint epoch differs from recovery state")
    if not np.isclose(
        float(selected_checkpoint.get("selection_accuracy", np.nan)),
        best_accuracy,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("selected checkpoint accuracy differs from recovery state")
    del selected_checkpoint
    primary_checkpoint_hash = sha256_file(best_path)
    recovery_checkpoint_path = fold_dir / "primary" / "last_checkpoint.pth"
    training_history_path = fold_dir / "primary" / "history.json"
    if not recovery_checkpoint_path.is_file() or not training_history_path.is_file():
        raise RuntimeError("primary recovery checkpoint/history is incomplete")
    in_progress_manifest.update(
        {
            "primary_checkpoint_frozen": True,
            "primary_checkpoint_hash": primary_checkpoint_hash,
            "selected_epoch": best_epoch,
            "selection_accuracy": best_accuracy,
            "selection_loss": best_loss,
            "accumulated_runtime_seconds": prior_runtime_seconds
            + time.perf_counter()
            - started,
            "phase": "FEATURE_TRAINING",
        }
    )
    atomic_write_json(progress_path, in_progress_manifest)

    (
        feature_model,
        feature_model_path,
        feature_model_sidecar_path,
        feature_metadata,
    ) = fit_feature_expert(
        feature_manifest_path=feature_manifest_path,
        split=split,
        output_dir=fold_dir,
        feature_data_hash=feature_data_hash,
        feature_schema_hash=FEATURE_CACHE_SCHEMA_HASH,
        seed=args.seed,
        smote_ratio=args.smote_ratio,
        resume=args.resume,
    )
    feature_model_hash = sha256_file(feature_model_path)
    in_progress_manifest.update(
        {
            "feature_model_frozen": True,
            "feature_model_hash": feature_model_hash,
            "accumulated_runtime_seconds": prior_runtime_seconds
            + time.perf_counter()
            - started,
            "phase": "READY_FOR_OUTER_EVALUATION",
        }
    )
    atomic_write_json(progress_path, in_progress_manifest)

    # Commit an irreversible audit marker before any outer model inference.
    # Prior raw/cache reads are provenance-only validation and never feed model
    # selection.  A crash after this point makes the fold non-resumable: fail
    # closed instead of repeating inference and claiming it happened once.
    assert_frozen_inputs()
    in_progress_manifest.update(
        {
            "outer_evaluation_started": True,
            "outer_evaluation_count": 1,
            "phase": "OUTER_EVALUATION",
            "accumulated_runtime_seconds": prior_runtime_seconds
            + time.perf_counter()
            - started,
        }
    )
    atomic_write_json(progress_path, in_progress_manifest)

    # The first construction of outer rows occurs only after both models freeze.
    (
        primary_identities,
        primary_labels,
        primary_probabilities,
        primary_predictions,
        primary_confidences,
    ) = primary_outer_inference(
        checkpoint_path=best_path,
        expected_run_config_hash=run_config_hash,
        outer_files=outer_files,
        outer_fold=outer_subject,
        batch_size=args.batch_size,
        device=device,
    )
    primary_row_hash = hash_rows(primary_identities, primary_labels)
    primary_manifest = PredictionManifest(
        schema_version=PREDICTION_SCHEMA_VERSION,
        schema_hash=prediction_schema_hash,
        dataset=DATASET_ID,
        outer_fold=outer_subject,
        outer_subject_id=outer_subject,
        model_id="eeg_conformer_primary",
        split_hash=split.split_hash,
        model_hash=primary_model_hash,
        checkpoint_hash=primary_checkpoint_hash,
        data_hash=raw_data_hash,
        preprocessing_hash=primary_preprocessing_hash,
        config_hash=primary_model_hash,
        calibration_hash=canonical_json_hash({"method": "none"}),
        code_hash=code_hash,
        row_hash=primary_row_hash,
        seed=args.seed,
        training_subjects=split.inner_train,
        calibration_subjects=(),
    )
    primary_bundle = PredictionBundle(
        identities=primary_identities,
        labels=primary_labels,
        probabilities=primary_probabilities,
        predictions=primary_predictions,
        confidences=primary_confidences,
        checkpoint_hash=primary_checkpoint_hash,
        manifest=primary_manifest,
    )

    (
        x_outer,
        feature_labels,
        feature_identities,
        outer_feature_names,
    ) = load_feature_rows(
        feature_manifest_path,
        (outer_subject,),
        outer_fold=outer_subject,
    )
    if tuple(outer_feature_names) != tuple(feature_cache_payload["feature_names"]):
        raise RuntimeError("outer feature schema differs from the frozen manifest")
    feature_probabilities = np.asarray(feature_model.predict_proba(x_outer))
    if feature_probabilities.shape != (len(x_outer), 5):
        raise RuntimeError(
            f"feature probability shape mismatch: {feature_probabilities.shape}"
        )
    feature_predictions = feature_probabilities.argmax(axis=1).astype(np.int64)
    feature_confidences = feature_probabilities.max(axis=1)
    reorder = align_one_to_one(
        primary_identities,
        feature_identities,
        reference_labels=primary_labels,
        candidate_labels=feature_labels,
    )
    feature_identities = tuple(feature_identities[index] for index in reorder)
    feature_labels = feature_labels[reorder]
    feature_probabilities = feature_probabilities[reorder]
    feature_predictions = feature_predictions[reorder]
    feature_confidences = feature_confidences[reorder]
    feature_row_hash = hash_rows(feature_identities, feature_labels)
    feature_manifest = PredictionManifest(
        schema_version=PREDICTION_SCHEMA_VERSION,
        schema_hash=prediction_schema_hash,
        dataset=DATASET_ID,
        outer_fold=outer_subject,
        outer_subject_id=outer_subject,
        model_id="rf_isotonic_feature_expert",
        split_hash=split.split_hash,
        model_hash=feature_metadata["model_hash"],
        checkpoint_hash=feature_model_hash,
        data_hash=feature_data_hash,
        preprocessing_hash=feature_preprocessing_hash,
        config_hash=feature_metadata["config_hash"],
        calibration_hash=feature_metadata["calibration_hash"],
        code_hash=code_hash,
        row_hash=feature_row_hash,
        seed=args.seed,
        training_subjects=split.inner_train,
        calibration_subjects=split.calibration,
    )
    feature_bundle = PredictionBundle(
        identities=feature_identities,
        labels=feature_labels,
        probabilities=feature_probabilities,
        predictions=feature_predictions,
        confidences=feature_confidences,
        checkpoint_hash=feature_model_hash,
        manifest=feature_manifest,
    )

    routing = route_type1_forward_kl(
        probabilities_t=primary_bundle.probabilities,
        predictions_t=primary_bundle.predictions,
        confidence_t=primary_bundle.confidences,
        probabilities_f=feature_bundle.probabilities,
        predictions_f=feature_bundle.predictions,
        tau_h=DEFAULT_TAU_H,
        kappa=DEFAULT_KAPPA,
        epsilon=DEFAULT_KL_EPSILON,
    )
    metrics = evaluate_policy_metrics(
        primary_bundle.labels,
        routing,
        min_support=args.min_support,
    )
    primary_predictions_path = fold_dir / "primary" / "predictions.npz"
    feature_predictions_path = fold_dir / "feature" / "predictions.npz"
    metrics_path = fold_dir / "metrics_fixed_kappa_015.json"
    primary_bundle.write_npz(primary_predictions_path)
    feature_bundle.write_npz(feature_predictions_path)
    metric_payload = {
        "schema_version": "stage1-clean-metrics-v1",
        "policy": "type1_forward_kl",
        "tau_h": DEFAULT_TAU_H,
        "kappa": DEFAULT_KAPPA,
        "kl_epsilon": DEFAULT_KL_EPSILON,
        "min_support": args.min_support,
        "outer_subject": outer_subject,
        "metrics": metric_to_dict(metrics),
    }
    atomic_write_json(metrics_path, metric_payload)

    assert_frozen_inputs()
    runtime_seconds = prior_runtime_seconds + time.perf_counter() - started
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    final_manifest = {
        **run_config,
        "run_config": run_config,
        "run_config_hash": run_config_hash,
        "primary_checkpoint_frozen": True,
        "feature_model_frozen": True,
        "outer_evaluation_count": 1,
        "outer_evaluation_completed": True,
        "selected_epoch": best_epoch,
        "selection_accuracy": best_accuracy,
        "selection_loss": best_loss,
        "primary_training_epochs": len(history),
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "artifacts": {
            "primary_checkpoint": artifact_entry(best_path, fold_dir),
            "primary_recovery_checkpoint": artifact_entry(
                recovery_checkpoint_path, fold_dir
            ),
            "primary_training_history": artifact_entry(
                training_history_path, fold_dir
            ),
            "primary_predictions": artifact_entry(
                primary_predictions_path, fold_dir
            ),
            "feature_model": artifact_entry(feature_model_path, fold_dir),
            "feature_model_metadata": artifact_entry(
                feature_model_sidecar_path, fold_dir
            ),
            "feature_predictions": artifact_entry(
                feature_predictions_path, fold_dir
            ),
            "metrics_fixed_kappa_015": artifact_entry(metrics_path, fold_dir),
        },
        "status": "COMPLETE_SINGLE_OUTER_FOLD",
    }
    atomic_write_json(final_manifest_path, final_manifest)
    print(json.dumps(final_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
