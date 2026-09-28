import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT.parent / "src")]
from new_objective import anchor_report
from validate_anchored import standalone_counts


class AnchoredReportingTests(unittest.TestCase):
    def test_replay_module_does_not_import_fitting_libraries(self):
        code = ("import sys; sys.path[:0]=" + repr([str(ROOT / "src"), str(ROOT.parent / "src")])
            + "; import new_objective.anchor_report; "
              "assert not any(x in sys.modules for x in ('sklearn','joblib','torch')); print('PASS')")
        proc = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_incomplete_seal_prevents_outcome_access(self):
        def fake_read(path):
            if path.name == "decisions_sealed.json":
                return {"status": "INCOMPLETE", "folds": {}}
            return {}
        with patch.object(anchor_report, "read", side_effect=fake_read):
            with self.assertRaises(RuntimeError):
                anchor_report.require_seal()

    def test_zero_acceptance_risk_is_undefined(self):
        x = standalone_counts(np.zeros(4, bool), np.ones(4, bool), np.array(["a", "a", "b", "b"]))
        self.assertIsNone(x["subject_equal_ratio_risk"])
        self.assertIsNone(x["pooled_risk"])
        self.assertEqual(x["macro_coverage"], 0)

    def test_subject_equal_ratio_not_arithmetic_mean(self):
        subjects = np.array(["a"] * 2 + ["b"] * 4)
        mask = np.array([1, 0, 1, 1, 1, 1], bool)
        errors = np.array([1, 0, 1, 0, 0, 0], bool)
        x = standalone_counts(mask, errors, subjects)
        self.assertEqual(x["macro_coverage"], .75)
        self.assertEqual(x["subject_equal_ratio_risk"], .5)
        self.assertEqual(x["pooled_risk"], .4)


if __name__ == "__main__":
    unittest.main()
