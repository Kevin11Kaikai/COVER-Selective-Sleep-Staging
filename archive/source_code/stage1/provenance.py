"""Content-addressed execution provenance for every resumable Stage-2 phase."""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .data_io import canonical_json_hash, sha256_file


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CAMPAIGN_ROOT.parent


def _distribution_version(name: str) -> str:
    return metadata.version(name)


def runtime_snapshot() -> dict[str, Any]:
    return {
        "python_executable": Path(sys.executable).resolve().as_posix(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _distribution_version("numpy"),
        "scipy": _distribution_version("scipy"),
        "scikit_learn": _distribution_version("scikit-learn"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "joblib": _distribution_version("joblib"),
        "imbalanced_learn": _distribution_version("imbalanced-learn"),
        "matplotlib": _distribution_version("matplotlib"),
        "pandas": _distribution_version("pandas"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def audited_environment() -> dict[str, Any]:
    manifest_path = CAMPAIGN_ROOT / "environment_manifest.json"
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = runtime_snapshot()
    compared = (
        "python_executable",
        "python",
        "numpy",
        "scipy",
        "scikit_learn",
        "torch",
        "cuda_runtime",
        "cudnn",
        "joblib",
        "imbalanced_learn",
        "matplotlib",
        "pandas",
        "gpu",
    )
    mismatches: dict[str, dict[str, Any]] = {}
    for key in compared:
        actual_value = actual[key]
        expected_value = expected[key]
        if key == "python_executable":
            equal = str(actual_value).lower() == str(expected_value).lower()
        else:
            equal = actual_value == expected_value
        if not equal:
            mismatches[key] = {"expected": expected_value, "actual": actual_value}
    if mismatches:
        raise RuntimeError(f"audited runtime mismatch: {mismatches}")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": expected,
        "runtime": actual,
        "runtime_hash": canonical_json_hash(actual),
    }


def hash_local_files(paths: Iterable[Path]) -> dict[str, Any]:
    entries = [
        {
            "path": path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted({Path(path).resolve() for path in paths}, key=lambda item: item.as_posix())
    ]
    if not entries:
        raise RuntimeError("executed-code provenance cannot be empty")
    return {"files": entries, "hash": canonical_json_hash(entries)}


def component_code_provenance() -> dict[str, Any]:
    code_dir = Path(__file__).resolve().parent
    paths = [
        PROJECT_ROOT / "conformer.py",
        PROJECT_ROOT / "sleepedf" / "data_loader.py",
        PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "run_outer_fold.py",
        PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "feature_cache.py",
        PROJECT_ROOT / "scripts" / "stage1_clean_nested_loso" / "contracts.py",
        code_dir / "__init__.py",
        code_dir / "bundles.py",
        code_dir / "components.py",
        code_dir / "data_io.py",
        code_dir / "firewall.py",
        code_dir / "provenance.py",
        code_dir / "splits.py",
    ]
    return hash_local_files(paths)


def risk_code_provenance() -> dict[str, Any]:
    code_dir = Path(__file__).resolve().parent
    paths = [
        code_dir / "__init__.py",
        code_dir / "bundles.py",
        code_dir / "data_io.py",
        code_dir / "features.py",
        code_dir / "firewall.py",
        code_dir / "provenance.py",
        code_dir / "risk_models.py",
        code_dir / "selectors.py",
        code_dir / "splits.py",
    ]
    return hash_local_files(paths)


def evaluation_code_provenance() -> dict[str, Any]:
    """Hash every local module that can affect label joining or reported outcomes."""
    code_dir = Path(__file__).resolve().parent
    paths = [
        PROJECT_ROOT / "sleepedf" / "data_loader.py",
        code_dir / "__init__.py",
        code_dir / "bundles.py",
        code_dir / "campaign.py",
        code_dir / "data_io.py",
        code_dir / "evaluation.py",
        code_dir / "firewall.py",
        code_dir / "provenance.py",
        code_dir / "risk_models.py",
        code_dir / "selectors.py",
        code_dir / "splits.py",
    ]
    return hash_local_files(paths)


def raw_inventory_provenance(data_dir_text: str) -> dict[str, Any]:
    from .data_io import discover_raw_records

    data_dir = Path(data_dir_text).resolve()
    records = discover_raw_records(data_dir)
    entries = [
        {"name": path.name, "sha256": sha256_file(path)}
        for subject in sorted(records)
        for path in records[subject]
    ]
    return {
        "directory": data_dir.as_posix(),
        "files": entries,
        "file_set_hash": canonical_json_hash(entries),
    }


def feature_cache_provenance(manifest_text: str) -> dict[str, Any]:
    manifest_path = Path(manifest_text).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards: list[dict[str, Any]] = []
    for item in manifest["shards"]:
        shard_path = (manifest_path.parent / item["path"]).resolve()
        actual = sha256_file(shard_path)
        if actual != item["sha256"]:
            raise RuntimeError(f"feature shard hash mismatch: {shard_path}")
        shards.append(
            {
                "subject_id": item["subject_id"],
                "path": item["path"],
                "sha256": actual,
                "row_hash": item["row_hash"],
                "subject_data_hash": item["subject_data_hash"],
            }
        )
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "schema_hash": manifest["schema_hash"],
        "extractor_hash": manifest["extractor_hash"],
        "config_hash": manifest["config_hash"],
        "code_hash": manifest["code_hash"],
        "data_hash": manifest["data_hash"],
        "shards": shards,
        "shard_set_hash": canonical_json_hash(shards),
    }


def phase_provenance_hash(
    *,
    code: Mapping[str, Any],
    environment: Mapping[str, Any],
    inputs: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    payload = {
        "code": dict(code),
        "environment": dict(environment),
        "inputs": dict(inputs),
        "config": dict(config),
    }
    return payload, canonical_json_hash(payload)


__all__ = [
    "audited_environment",
    "component_code_provenance",
    "evaluation_code_provenance",
    "feature_cache_provenance",
    "hash_local_files",
    "phase_provenance_hash",
    "raw_inventory_provenance",
    "risk_code_provenance",
    "runtime_snapshot",
]
