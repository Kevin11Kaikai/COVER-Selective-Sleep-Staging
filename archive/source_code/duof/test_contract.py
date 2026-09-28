"""Small tests for the specific score/selection risks in the DuoF adapter."""
import unittest
from fractions import Fraction

import numpy as np
from scipy.special import softmax

from core import DUO, OLD, STAGE3, as_duo, score_f, individual


class DuoContract(unittest.TestCase):
    def test_score_matches_original_stage4_F_branch(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(90, 5))
        data = {"logits": logits, "primary": softmax(logits, axis=1),
                "feature": rng.dirichlet(np.ones(5), size=90),
                "subjects": np.repeat([f"{i:02d}" for i in range(15)], 6)}
        fitted = {"weights": [.18, .61]}
        native = as_duo(data)
        native["H_logits"] = logits
        expected = DUO.duo_scores(native, {"F": fitted, "H": fitted})["DuoUQ_F"]
        np.testing.assert_array_equal(score_f(data, fitted), expected)

    def test_primary_label_is_preserved_when_fused_argmax_differs(self):
        data = {"logits": np.array([[4., 0., 0., 0., 0.]]),
                "primary": softmax([[4., 0., 0., 0., 0.]], axis=1),
                "feature": np.array([[.001, .996, .001, .001, .001]]),
                "subjects": np.array(["00"])}
        score = score_f(data, {"weights": [.1, 2.]})
        self.assertGreater(score[0], .99)
        self.assertEqual(data["primary"].argmax(1)[0], 0)

    def test_fast_exact_grid_equals_frozen_stage2_with_ties(self):
        subjects = np.repeat([f"{i:02d}" for i in range(15)], np.arange(1, 16))
        scores = np.random.default_rng(5).integers(0, 8, len(subjects)) / 10
        expected = OLD.build_candidate_policies(scores, subjects)
        self.assertEqual(STAGE3.policies(scores, subjects), expected)

    def test_ties_are_identity_stable_and_nested(self):
        ids = [f"SC|00|R0|{i}" for i in range(200)]
        scores = np.ones(200)
        kwargs = {"threshold": 1., "identities": ids, "method_id": "frozen", "protocol_hash": "h", "outer": "00"}
        small = OLD.build_acceptance_mask(scores, tie_fraction=Fraction(1, 4), **kwargs)
        large = OLD.build_acceptance_mask(scores, tie_fraction=Fraction(3, 4), **kwargs)
        self.assertFalse((small & ~large).any())
        order = np.random.default_rng(1).permutation(200)
        changed = dict(kwargs, identities=[ids[i] for i in order])
        permuted = OLD.build_acceptance_mask(scores[order], tie_fraction=Fraction(1, 4), **changed)
        np.testing.assert_array_equal(small[order], permuted)

    def test_ceiling_and_support_are_separate_requirements(self):
        rows = [dict(q=Fraction(1, 2), coverage=Fraction(1, 2), risk=Fraction(4, 100), supported=True),
                dict(q=Fraction(1, 4), coverage=Fraction(1, 4), risk=Fraction(2, 100), supported=True),
                dict(q=Fraction(3, 4), coverage=Fraction(3, 4), risk=Fraction(1, 100), supported=False)]
        self.assertEqual(OLD.select_empirical_policy(rows, alpha=.05), rows[0])
        self.assertEqual(OLD.select_empirical_policy(rows, alpha=.03), rows[1])
        self.assertIsNone(OLD.select_empirical_policy(rows, alpha=.01))

    def test_zero_and_insufficient_acceptance_are_not_passes(self):
        self.assertIsNone(individual("m", "s", 0, 0, 100)["risk"])
        self.assertFalse(individual("m", "s", 0, 0, 100)["passed"])
        self.assertFalse(individual("m", "s", 49, 0, 100)["passed"])
        self.assertTrue(individual("m", "s", 100, 5, 200)["passed"])
        self.assertFalse(individual("m", "s", 100, 6, 200)["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
