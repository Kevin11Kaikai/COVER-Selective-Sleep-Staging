"""Boundary tests using synthetic data; never read evaluation outcomes."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "src"),
                str(ROOT.parent / "AutoML_Claude")]
import numpy as np
from new_objective import reporting
from new_objective.storage import OUT, specification, write, digest
from new_objective.policies import policies, subset_policies, choose, mask
from new_objective.experiment import cache_fit
from new_objective.calibration import cross_calibrate
from stage1_analysis.metrics import summary, quota_mask, ranked_orders, identity_ties


class ProtocolTests(unittest.TestCase):
    def data(self, n=100):
        people = [f"s{i:02d}" for i in range(15)]
        return {
            "subjects": np.repeat(people, n),
            "identities": [{"dataset": "synthetic", "subject_id": s,
                            "record_id": s, "epoch_index": i}
                           for s in people for i in range(n)]}

    def test_empty_feasible_set_abstains(self):
        d = self.data(); scores = np.tile(np.arange(100.), 15)
        picked, _ = choose(subset_policies(policies(scores, d["subjects"]), 20),
                           scores, np.ones(len(scores), int), d, "00", specification(), 50)
        self.assertIsNone(picked)
        self.assertFalse(mask(scores, picked, d, "00", specification()).any())

    def test_support_requirement_is_cert_only(self):
        d = self.data(n=20); scores = np.tile(np.arange(20.), 15)
        grid = subset_policies(policies(scores, d["subjects"]), 20)
        fit, _ = choose(grid, scores, np.zeros(len(scores), int), d, "00", specification())
        cert, _ = choose(grid, scores, np.zeros(len(scores), int), d, "00", specification(), 50)
        self.assertIsNotNone(fit)
        self.assertIsNone(cert)

    def test_nested_grids_and_common_tie_ballot(self):
        d = self.data(); scores = np.repeat(np.arange(10.), 150)
        dense = policies(scores, d["subjects"])
        coarse = subset_policies(dense, 20)
        last = np.zeros(len(scores), bool)
        for rule in dense:
            current = mask(scores, rule, d, "00", specification())
            self.assertFalse(np.any(last & ~current))
            last = current
        for rule in coarse:
            self.assertIn(rule, dense)
        rule = coarse[3]
        np.testing.assert_array_equal(mask(scores, rule, d, "00", specification()),
                                      mask(scores.copy(), rule, d, "00", specification()))

    def test_zero_acceptance_is_not_zero_risk(self):
        r = summary(np.array([10, 20]), np.array([0, 0]), np.array([0, 0]))
        self.assertEqual(r["macro_coverage"], 0)
        self.assertIsNone(r["subject_equal_ratio_risk"])
        self.assertIsNone(r["pooled_risk"])

    def test_matched_quotas_are_exact_with_ties(self):
        subjects = np.array(["a"] * 4 + ["b"] * 6)
        scores = np.ones(10)
        ties = identity_ties([("synthetic", s, s, i) for i, s in enumerate(subjects)])
        accepted = quota_mask(ranked_orders(scores, ties, subjects, ["a", "b"]),
                              [1, 3], len(scores))
        self.assertEqual(int(accepted[subjects == "a"].sum()), 1)
        self.assertEqual(int(accepted[subjects == "b"].sum()), 3)

    def test_partial_decision_seal_rejected(self):
        with patch.object(reporting, "read", return_value={
                "status": "ALL_OUTER_DECISIONS_FROZEN", "folds": {"00": {}}}):
            with self.assertRaises(RuntimeError):
                reporting.require_seal()

    def test_corrupted_decisions_rejected(self):
        seal = {"status": "ALL_OUTER_DECISIONS_FROZEN",
                "folds": {f"{i:02d}": {"sha256": "expected"} for i in range(20)}}
        with patch.object(reporting, "read", return_value=seal), \
             patch.object(reporting, "digest", return_value="changed"):
            with self.assertRaises(RuntimeError):
                reporting.require_seal()

    def test_truth_reader_cannot_run_before_seal(self):
        with patch.object(reporting, "require_seal", side_effect=RuntimeError("blocked")), \
             patch("stage1_analysis.inputs.load") as reader:
            with self.assertRaises(RuntimeError):
                reporting.make_portable()
            reader.assert_not_called()

    def test_resume_rejects_wrong_fingerprint_before_loading_model(self):
        parent = OUT / "tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            path = Path(folder) / "model.joblib"
            path.write_bytes(b"not a model")
            write(path.with_suffix(".json"), {"fingerprint": "old", "source": "s",
                                             "model_sha256": digest(path)})
            with self.assertRaises(RuntimeError):
                cache_fit(path, None, None, None, 1, {}, None, "new", "s", True)

    def test_windows_atomic_json_replace_retries_only_known_transient_errors(self):
        import new_objective.storage as storage
        parent = OUT / "tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            path = Path(folder) / "atomic.json"
            real_replace = os.replace
            calls = []
            error = PermissionError("synthetic Windows sharing denial")
            error.winerror = 5
            def one_denial(source, dest):
                calls.append(1)
                if len(calls) == 1:
                    raise error
                return real_replace(source, dest)
            with patch.object(storage.os, "replace", side_effect=one_denial), \
                 patch.object(storage.time, "sleep"):
                write(path, {"value": 7})
            self.assertEqual(json.loads(path.read_text())["value"], 7)
            self.assertEqual(len(calls), 2)

    def test_report_module_does_not_import_training_libraries(self):
        code = ("import sys;sys.path[:0]=" + repr(sys.path[:3]) + ";"
                "import new_objective.reporting;"
                "assert not any(k in sys.modules for k in "
                "('sklearn','joblib','torch','new_objective.experiment'))")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
