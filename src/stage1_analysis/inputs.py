"""Identity-aligned readers for copied frozen outputs; no model fitting imports."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import alignment, identity_key

PH = "b264de85d45f7d49763a69743dd27504871a682e6066f0fd1167dee4caf8a73e"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Tracker:
    def __init__(self, root: Path):
        self.root = root
        self.files = {}
        manifest = root / "manifests/asset_manifest.json"
        self.expected = {r["path"]: r["sha256"] for r in read_json(manifest)["files"]} if manifest.exists() else {}

    def path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        path.relative_to(self.root)
        if relative not in self.files:
            digest = sha256(path)
            if relative in self.expected and digest != self.expected[relative]:
                raise RuntimeError(f"Copied source fingerprint mismatch: {relative}")
            self.files[relative] = {"sha256": digest, "bytes": path.stat().st_size}
        return path


@dataclass
class FrozenData:
    identities: list[dict]
    subjects: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    errors: np.ndarray
    scores: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    audit: dict
    tracker: Tracker
    primary_probabilities: np.ndarray
    feature_probabilities: np.ndarray

    @property
    def keys(self):
        return [identity_key(x) for x in self.identities]

    @property
    def people(self):
        return sorted(set(self.subjects.tolist()))


def audit_splits(tracker: Tracker) -> dict:
    split = read_json(tracker.path("configs/frozen/stage1/split_manifest.json"))
    assert split["protocol_hash"] == PH
    assert set(split["folds"]) == {f"{i:02d}" for i in range(20)}
    role_rows, fits = [], 0
    for outer, fold in split["folds"].items():
        fit, cert = set(fold["outer_split"]["fit"]), set(fold["outer_split"]["cert"])
        assert len(fit) == 15 and len(cert) == 4 and not fit & cert
        assert fit | cert | {outer} == set(split["folds"])
        assert outer not in fit | cert
        assert set(fold["inner_components"]) == fit
        for target, component in list(fold["inner_components"].items()) + [("final", fold["final_component"])]:
            roles = {name: set(component[name]) for name in
                     ["component_train", "checkpoint_selection", "feature_calibration"]}
            assert [len(x) for x in roles.values()] == [10, 2, 2]
            united = set.union(*roles.values())
            assert len(united) == 14 and united <= fit
            assert not united & (cert | {outer})
            if target != "final":
                assert united == fit - {target}
            else:
                assert set(component["unused"]) == fit - united
            for role, subjects in roles.items():
                role_rows.append({"outer": outer, "fit_target": target, "role": role,
                                  "subjects": ";".join(sorted(subjects))})
            fits += 1
        held = []
        for cv in fold["risk_cv"]:
            tr, va = set(cv["train"]), set(cv["validation"])
            assert len(tr) == 12 and len(va) == 3 and not tr & va and tr | va == fit
            held.extend(va)
        assert len(held) == len(set(held)) == 15
    return {"component_fits": fits, "role_rows": role_rows,
            "roles_disjoint": True, "fit_meta_cv_fully_nested": False,
            "note": "Component OOF excludes each target; risk-model CV reuses OOF components. No claim of fully nested FIT model selection."}


def load(root: Path, compact: bool = False) -> FrozenData:
    tracker = Tracker(root)
    compact_path = root / "outputs/analysis_inputs/frozen_analysis_inputs.npz"
    if compact:
        expected = read_json(root / "outputs/analysis_inputs/receipt.json")
        assert sha256(compact_path) == expected["sha256"]
        with np.load(tracker.path("outputs/analysis_inputs/frozen_analysis_inputs.npz"), allow_pickle=False) as z:
            required = {"primary_probabilities", "feature_probabilities"}
            if not required.issubset(z.files):
                raise ValueError("Compact inputs lack component probabilities; regenerate from frozen sources")
            data = FrozenData(json.loads(z["identities_json"].item()), z["subjects"], z["labels"],
                z["predictions"], z["errors"],
                {str(m): z["scores"][:, i] for i, m in enumerate(z["score_names"])},
                {str(m): z["masks"][:, i] for i, m in enumerate(z["mask_names"])},
                {"mode": "compact_replay", "source_audit": expected["source_audit"],
                 "splits": audit_splits(tracker)}, tracker,
                z["primary_probabilities"], z["feature_probabilities"])
        assert data.audit["splits"]["role_rows"] == expected["source_audit"]["splits"]["role_rows"]
        validate_components(data)
        return data
    base = "outputs/frozen/stage1/"
    with np.load(tracker.path(base + "sealed_outer_results/ALL_OUTER_SEALED.npz"), allow_pickle=False) as z:
        assert z["protocol_hash"].item() == PH
        ids = json.loads(z["identities_json"].item())
        subjects, labels, pred = z["outer_subject_ids"], z["ground_truth_labels"], z["primary_predictions"]
        errors = z["error_indicators"].astype(bool)
        primary_probabilities = z["primary_probabilities"].copy()
        feature_probabilities = z["feature_probabilities"].copy()
        assert np.array_equal(pred, z["primary_probabilities"].argmax(axis=1))
        assert np.array_equal(pred, z["primary_logits"].argmax(axis=1))
        scores = {str(m): z["risk_scores"][:, i].copy() for i, m in enumerate(z["method_names"])}
        masks = {str(m): z["acceptance_masks"][:, i].copy() for i, m in enumerate(z["mask_names"])}
    keys = [identity_key(x) for x in ids]
    assert len(keys) == len(set(keys)) == 42308
    assert len({x[2] for x in keys}) == 39 and len(set(subjects)) == 20
    assert np.array_equal(errors, pred != labels) and set(labels) == set(range(5))
    assert all(x[1] == s for x, s in zip(keys, subjects))
    assert all(np.isfinite(v).all() for v in scores.values())
    assert np.array_equal(masks["Learned-Selected|0.05|empirical"], masks["Heterogeneous-expert|0.05|empirical"])
    audit = {"mode": "full_source_replay", "identity_rows": 42308, "records": 39,
             "subjects": 20, "primary_labels_fixed": True, "protocol_hash": PH,
             "splits": audit_splits(tracker)}
    # Check all historical primary campaign table rows, not just headline methods.
    table = read_csv(tracker.path(base + "tables/per_subject_results.csv"))
    for row in table:
        mask = masks[f"{row['method']}|{row['alpha']}|{row['selector']}"] & (subjects == row["subject"])
        assert (int(mask.sum()), int((mask & errors).sum())) == (int(row["accepted"]), int(row["accepted_errors"]))
    audit["original_stage1_rows_recomputed"] = len(table)
    # Read all saved historical policy masks; no calibrator or joblib is executed.
    table = read_csv(tracker.path("outputs/frozen/policies/tables/per_subject_results.csv"))
    visited = {}
    for row in table:
        name, person = row["method"], row["subject"]
        target = "historical|" + name
        if target not in masks:
            masks[target] = np.zeros(len(keys), dtype=bool)
            visited[target] = np.zeros(len(keys), dtype=bool)
        path = tracker.path(f"outputs/frozen/policies/checkpoints/outer-{person}/{name}/decisions.npz")
        with np.load(path, allow_pickle=False) as z:
            positions = alignment([identity_key(x) for x in json.loads(z["identities_json"].item())], keys)
            assert not z["contains_outer_labels"].item()
            assert np.array_equal(z["primary_predictions"], pred[positions])
            assert (subjects[positions] == person).all() and not visited[target][positions].any()
            masks[target][positions] = z["outer_accept"]
            visited[target][positions] = True
        selected = masks[target] & (subjects == person)
        assert (int(selected.sum()), int((selected & errors).sum())) == (int(row["accepted"]), int(row["accepted_errors"]))
    assert all(v.all() for v in visited.values())
    assert np.array_equal(masks["historical|A-Heterogeneous"], masks["Learned-Selected|0.05|empirical"])
    audit["historical_policy_rows_recomputed"] = len(table)
    # Existing DuoF weights and policies were already frozen; only load scores and masks.
    scores["DuoF"] = np.full(len(keys), np.nan)
    for alpha in ["0.05", "0.03"]:
        masks[f"DuoF|{alpha}|empirical"] = np.zeros(len(keys), dtype=bool)
    visited = np.zeros(len(keys), dtype=bool)
    for person in sorted(set(subjects)):
        prefix = f"outputs/frozen/duof/folds/outer-{person}/"
        receipt = read_json(tracker.path(prefix + "completion.json"))
        assert receipt["status"] == "COMPLETE" and receipt["outer_labels_used"] is False
        for file, expected in receipt["outputs"].items():
            assert sha256(tracker.path(prefix + file)) == expected
        with np.load(tracker.path(prefix + "decisions.npz"), allow_pickle=False) as z:
            pos = alignment([identity_key(x) for x in json.loads(z["identities_json"].item())], keys)
            assert not z["contains_outer_labels"].item() and not visited[pos].any()
            assert np.array_equal(z["primary_predictions"], pred[pos]) and (subjects[pos] == person).all()
            scores["DuoF"][pos] = z["outer_scores"]
            for alpha in ["0.05", "0.03"]:
                masks[f"DuoF|{alpha}|empirical"][pos] = z[f"accept_DuoF-{alpha}"]
            visited[pos] = True
    assert visited.all() and np.isfinite(scores["DuoF"]).all()
    audit["duof_folds_verified"] = 20
    # Align original processed EEG labels by record and stored epoch index.
    raw_rows = 0
    for record in sorted({key[2] for key in keys}):
        matches = sorted((root / "data/eeg_fpz_cz").glob(record + "*.npz"))
        assert len(matches) == 1, (record, matches)
        path = tracker.path(matches[0].relative_to(root).as_posix())
        pos = np.array([i for i, key in enumerate(keys) if key[2] == record])
        epoch = np.array([keys[i][3] for i in pos])
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z["y"][epoch], labels[pos])
            assert float(z["fs"].item()) == 100.0
            assert str(z["ch_label"].item()) == "EEG Fpz-Cz"
            x = z["x"]
            assert x.shape == (len(z["y"]), 3000, 1), (record, x.shape)
        raw_rows += len(pos)
    audit["processed_eeg_labels_verified"] = raw_rows
    audit["all_39_records_channel"] = "EEG Fpz-Cz"
    audit["sampling_rate_hz"] = 100
    audit["samples_per_epoch"] = 3000
    data = FrozenData(ids, subjects, labels, pred, errors, scores, masks, audit, tracker,
                      primary_probabilities, feature_probabilities)
    validate_components(data)
    return data


def validate_components(data: FrozenData) -> None:
    """Require the actual five-class outputs in the canonical identity order."""
    for probabilities in [data.primary_probabilities, data.feature_probabilities]:
        if probabilities.shape != (len(data.labels), 5):
            raise ValueError("Component probability shape does not match canonical epochs")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("Invalid component probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-7):
            raise ValueError("Component probabilities must sum to one")
    if not np.array_equal(data.primary_probabilities.argmax(axis=1), data.predictions):
        raise ValueError("Primary labels changed")


def save_compact(data: FrozenData, root: Path) -> None:
    dest = root / "outputs/analysis_inputs"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "frozen_analysis_inputs.npz"
    np.savez_compressed(path, identities_json=json.dumps(data.identities), subjects=data.subjects,
        labels=data.labels, predictions=data.predictions, errors=data.errors,
        primary_probabilities=data.primary_probabilities,
        feature_probabilities=data.feature_probabilities,
        score_names=list(data.scores), scores=np.column_stack(list(data.scores.values())),
        mask_names=list(data.masks), masks=np.column_stack(list(data.masks.values())))
    (dest / "receipt.json").write_text(json.dumps({"sha256": sha256(path),
        "description": "Derived lossless identity-aligned component probabilities, frozen scores and masks for analysis-only replay",
        "new_fitting": False, "source_audit": data.audit,
        "source_files": data.tracker.files}, indent=2) + "\n", encoding="utf-8")
