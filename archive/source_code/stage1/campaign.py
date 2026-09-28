"""Resumable command-line harness for the frozen Stage-2 campaign."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from . import PROTOCOL_SHA256
from .components import run_component_fit
from .data_io import (
    atomic_write_json,
    discover_raw_records,
    sha256_file,
)
from .firewall import RoleLedger, assert_ready_to_unblind
from .provenance import (
    audited_environment,
    feature_cache_provenance,
    raw_inventory_provenance,
)
from .risk_models import construct_and_freeze_outer_selectors, develop_outer_fold, risk_registry
from .splits import (
    SUBJECTS,
    build_final_roles,
    build_inner_roles,
    build_outer_split,
    build_risk_cv,
)


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CAMPAIGN_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data" / "Sleep-EDF-20" / "eeg_fpz_cz"
FEATURE_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "stage1_clean_nested_loso"
    / "feature_cache"
    / "manifest-9aa61e01e862e0cc8994611663efd3716a6caff74c4c349c6d19f54e7546e02b.json"
)
RAW_FILE_SET_HASH = "6553de7de5c29572b4b99532ad648d95e54b90ff8d6c32a6fbd40e6aa435130a"
FEATURE_MANIFEST_HASH = "4de20c8ca0bfccc9337ae8a01ee1bb5b819d960ecab33a646e0914f700214960"


class _TeeStream:
    def __init__(self, original: Any, log_path: Path) -> None:
        self.original = original
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = log_path.open("a", encoding="utf-8", buffering=1)

    def write(self, text: str) -> int:
        self.original.write(text)
        self.handle.write(text)
        return len(text)

    def flush(self) -> None:
        self.original.flush()
        self.handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON mapping: {path}")
    return value


def _update_state(
    state: str,
    *,
    completed: str | None = None,
    progress: dict[str, Any] | None = None,
    outer_unblinded: bool | None = None,
) -> None:
    path = CAMPAIGN_ROOT / "run_state.json"
    payload = _read_json(path)
    payload["state"] = state
    completed_states = list(payload.get("completed_states", []))
    if completed is not None and completed not in completed_states:
        completed_states.append(completed)
    payload["completed_states"] = completed_states
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    if progress is not None:
        payload["progress"] = progress
    if outer_unblinded is not None:
        payload["outer_unblinded"] = outer_unblinded
    atomic_write_json(path, payload)


def _raw_inventory_hash(records: dict[str, list[Path]]) -> str:
    from .data_io import canonical_json_hash

    return canonical_json_hash(
        [
            {
                "name": path.name,
                "sha256": sha256_file(path),
            }
            for subject in sorted(records)
            for path in records[subject]
        ]
    )


def preflight() -> dict[str, Any]:
    if sha256_file(CAMPAIGN_ROOT / "frozen_protocol.json") != PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol byte hash changed")
    records = discover_raw_records(DATA_DIR)
    raw_provenance = raw_inventory_provenance(str(DATA_DIR.resolve()))
    if raw_provenance["file_set_hash"] != RAW_FILE_SET_HASH:
        raise RuntimeError("raw input inventory/content hash changed")
    feature_provenance = feature_cache_provenance(str(FEATURE_MANIFEST.resolve()))
    if feature_provenance["manifest_sha256"] != FEATURE_MANIFEST_HASH:
        raise RuntimeError("feature manifest byte hash changed")
    registry = _read_json(CAMPAIGN_ROOT / "model_registry.json")
    if registry.get("risk_configs") != risk_registry():
        raise RuntimeError("implemented risk registry differs from frozen registry")
    environment = audited_environment()
    usage = shutil.disk_usage(CAMPAIGN_ROOT)
    if usage.free < 10 * 1024**3:
        raise RuntimeError("less than 10 GiB free on the campaign volume")
    return {
        "protocol_hash": PROTOCOL_SHA256,
        "raw_hash": RAW_FILE_SET_HASH,
        "feature_manifest_hash": FEATURE_MANIFEST_HASH,
        "feature_shard_set_hash": feature_provenance["shard_set_hash"],
        "environment_runtime_hash": environment["runtime_hash"],
        "subjects": list(SUBJECTS),
        "records": sum(len(paths) for paths in records.values()),
        "cuda_device": torch.cuda.get_device_name(0),
        "free_bytes": usage.free,
    }


def write_split_manifest() -> Path:
    folds: dict[str, Any] = {}
    for outer in SUBJECTS:
        split = build_outer_split(outer)
        fit = split["fit"]
        folds[outer] = {
            "outer_split": split,
            "inner_components": {
                target: build_inner_roles(outer, target, fit) for target in fit
            },
            "final_component": build_final_roles(outer, fit),
            "risk_cv": build_risk_cv(outer, fit),
        }
    payload = {
        "schema_version": "stage2-split-manifest-v1",
        "protocol_hash": PROTOCOL_SHA256,
        "assignment_is_label_free": True,
        "folds": folds,
    }
    return atomic_write_json(CAMPAIGN_ROOT / "split_manifest.json", payload)


def run_contract_tests() -> None:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(CAMPAIGN_ROOT / "tests"),
        "-v",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError("Stage-2 unit/leakage tests failed")
    _update_state(
        "UNIT_AND_LEAKAGE_TESTS",
        completed="UNIT_AND_LEAKAGE_TESTS",
        progress={"tests": "29/29 PASS"},
    )


def _ledger(outer: str, split: dict[str, Any]) -> RoleLedger:
    return RoleLedger(
        outer=outer,
        fit=tuple(split["fit"]),
        cert=tuple(split["cert"]),
        path=CAMPAIGN_ROOT / "logs" / f"label_access_outer-{outer}.jsonl",
    )


def run_components(*, only_outer: str | None = None) -> None:
    records = discover_raw_records(DATA_DIR)
    targets = (only_outer,) if only_outer is not None else SUBJECTS
    completed_count = 0
    total = len(targets) * 16
    _update_state("DEVELOPMENT", progress={"component_fits_completed": 0, "component_fits_total": total})
    for outer in targets:
        split = build_outer_split(outer)
        ledger = _ledger(outer, split)
        fit_subjects = list(split["fit"])
        for target in fit_subjects:
            roles = build_inner_roles(outer, target, fit_subjects)
            output = CAMPAIGN_ROOT / "development" / "components" / f"outer-{outer}" / f"inner-{target}"
            run_component_fit(
                roles=roles,
                predicted_subjects=[target],
                records=records,
                feature_manifest=FEATURE_MANIFEST,
                output_dir=output,
                ledger=ledger,
                resume=True,
                device=torch.device("cuda"),
            )
            completed_count += 1
            _update_state(
                "DEVELOPMENT",
                progress={
                    "component_fits_completed": completed_count,
                    "component_fits_total": total,
                    "last_outer": outer,
                    "last_role": f"inner-{target}",
                },
            )
        final_roles = build_final_roles(outer, fit_subjects)
        output = CAMPAIGN_ROOT / "development" / "components" / f"outer-{outer}" / "final"
        run_component_fit(
            roles=final_roles,
            predicted_subjects=list(split["cert"]) + [outer],
            records=records,
            feature_manifest=FEATURE_MANIFEST,
            output_dir=output,
            ledger=ledger,
            resume=True,
            device=torch.device("cuda"),
        )
        completed_count += 1
        _update_state(
            "DEVELOPMENT",
            progress={
                "component_fits_completed": completed_count,
                "component_fits_total": total,
                "last_outer": outer,
                "last_role": "final",
            },
        )
    if only_outer is None:
        _update_state(
            "DEVELOPMENT",
            completed="DEVELOPMENT",
            progress={"component_fits_completed": 320, "component_fits_total": 320},
        )


def develop_and_freeze(*, only_outer: str | None = None) -> None:
    records = discover_raw_records(DATA_DIR)
    targets = (only_outer,) if only_outer is not None else SUBJECTS
    manifests: list[dict[str, Any]] = []
    for index, outer in enumerate(targets, start=1):
        split = build_outer_split(outer)
        ledger = _ledger(outer, split)
        development_dir = CAMPAIGN_ROOT / "development" / "risk" / f"outer-{outer}"
        develop_outer_fold(
            outer=outer,
            fit_subjects=split["fit"],
            cert_subjects=split["cert"],
            components_root=CAMPAIGN_ROOT / "development" / "components",
            records=records,
            ledger=ledger,
            output_dir=development_dir,
            resume=True,
        )
        _update_state(
            "CALIBRATION",
            progress={"risk_folds_completed": index, "risk_folds_total": len(targets)},
        )
        manifest = construct_and_freeze_outer_selectors(
            outer=outer,
            cert_subjects=split["cert"],
            records=records,
            ledger=ledger,
            development_dir=development_dir,
            calibration_dir=CAMPAIGN_ROOT / "calibration" / f"outer-{outer}",
            outer_dir=CAMPAIGN_ROOT / "outer_predictions" / f"outer-{outer}",
            resume=True,
        )
        manifests.append(manifest)
        _update_state(
            "CALIBRATION",
            progress={"outer_folds_frozen": index, "outer_folds_total": len(targets)},
        )
    if only_outer is None:
        assert_ready_to_unblind(
            manifests,
            expected_protocol_hash=PROTOCOL_SHA256,
            required_outer_subjects=SUBJECTS,
        )
        _update_state("OUTER_PREDICTIONS_FROZEN", completed="CALIBRATION")
        _update_state(
            "OUTER_PREDICTIONS_FROZEN", completed="OUTER_PREDICTIONS_FROZEN"
        )


def run_all_pre_unblind() -> None:
    preflight_result = preflight()
    atomic_write_json(CAMPAIGN_ROOT / "logs" / "execution_preflight.json", preflight_result)
    write_split_manifest()
    run_contract_tests()
    run_components()
    develop_and_freeze()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "test", "components", "develop-freeze", "pre-unblind", "all"),
    )
    parser.add_argument("--outer", choices=SUBJECTS, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command in {"components", "develop-freeze", "pre-unblind", "all"}:
        tee = _TeeStream(sys.stdout, CAMPAIGN_ROOT / "logs" / "full_execution.log")
        sys.stdout = tee
        sys.stderr = tee
        print(
            f"stage2 command={args.command} outer={args.outer} "
            f"started_utc={datetime.now(timezone.utc).isoformat()}",
            flush=True,
        )
    if args.command == "preflight":
        result = preflight()
        write_split_manifest()
        atomic_write_json(CAMPAIGN_ROOT / "logs" / "execution_preflight.json", result)
        print(json.dumps(result, indent=2))
    elif args.command == "test":
        run_contract_tests()
    elif args.command == "components":
        preflight()
        run_components(only_outer=args.outer)
    elif args.command == "develop-freeze":
        preflight()
        develop_and_freeze(only_outer=args.outer)
    elif args.command in {"pre-unblind", "all"}:
        run_all_pre_unblind()
        if args.command == "all":
            from .evaluation import closeout_campaign

            closeout_campaign()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
