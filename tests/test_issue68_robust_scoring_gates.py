import math
import unittest

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency for notebook env
    pd = None

from src.walk_forward_analyzer import (
    _normalize_additional_buy_priority,
    apply_robust_gates,
    compute_robust_score,
)


class TestIssue68RobustScoringAndGates(unittest.TestCase):
    def test_normalize_additional_buy_priority(self):
        self.assertEqual(_normalize_additional_buy_priority(None), "lowest_order")
        self.assertEqual(_normalize_additional_buy_priority(0), "lowest_order")
        self.assertEqual(_normalize_additional_buy_priority(1), "highest_drop")
        self.assertEqual(_normalize_additional_buy_priority("lowest_order"), "lowest_order")
        self.assertEqual(_normalize_additional_buy_priority("highest_drop"), "highest_drop")
        self.assertEqual(_normalize_additional_buy_priority("biggest_drop"), "highest_drop")

    @unittest.skipIf(pd is None, "pandas is required for robust scoring/gate tests")
    def test_compute_robust_score_formula(self):
        df = pd.DataFrame(
            {
                "calmar_ratio_mean": [2.0, 1.5],
                "calmar_ratio_std": [0.5, 0.1],
                "size": [9, 99],
            },
            index=[0, 1],
        )

        scores = compute_robust_score(df, metric="calmar_ratio", k=1.0, size_col="size")
        expected0 = (2.0 - 0.5) * math.log1p(9)
        self.assertAlmostEqual(float(scores.loc[0]), expected0, places=9)

    @unittest.skipIf(pd is None, "pandas is required for robust scoring/gate tests")
    def test_apply_robust_gates_passes(self):
        df = pd.DataFrame(
            {
                "fold": list(range(1, 11)),
                "is_calmar_ratio": [1.0] * 10,
                "oos_calmar_ratio": [0.7] * 10,
                "oos_mdd": [-0.20] * 10,
            }
        )

        report_df, summary = apply_robust_gates(df, metric="calmar_ratio")
        self.assertTrue(summary["gate_passed"])
        self.assertAlmostEqual(summary["median_oos_is_ratio"], 0.7, places=12)
        self.assertAlmostEqual(summary["fold_pass_rate"], 1.0, places=12)
        self.assertAlmostEqual(summary["oos_mdd_p95"], 0.20, places=12)
        self.assertTrue(report_df["fold_pass"].all())

    @unittest.skipIf(pd is None, "pandas is required for robust scoring/gate tests")
    def test_apply_robust_gates_inclusive_thresholds(self):
        # 10 folds: 7 pass (0.7), 3 fail (0.2) -> pass_rate == 0.70, median == 0.7
        df = pd.DataFrame(
            {
                "fold": list(range(1, 11)),
                "is_calmar_ratio": [1.0] * 10,
                "oos_calmar_ratio": ([0.7] * 7) + ([0.2] * 3),
                "oos_mdd": [-0.20] * 10,
            }
        )

        _, summary = apply_robust_gates(
            df,
            metric="calmar_ratio",
            min_oos_is_ratio=0.60,
            min_fold_pass_rate=0.70,
            max_oos_mdd_p95=0.25,
        )
        self.assertTrue(summary["gate_passed"])
        self.assertAlmostEqual(summary["fold_pass_rate"], 0.7, places=12)

    @unittest.skipIf(pd is None, "pandas is required for robust scoring/gate tests")
    def test_apply_robust_gates_fails_on_oos_mdd_p95(self):
        df = pd.DataFrame(
            {
                "fold": list(range(1, 101)),
                "is_calmar_ratio": [1.0] * 100,
                "oos_calmar_ratio": [0.7] * 100,
                # Top 5% have 30% drawdown -> p95 should be 0.30 (fail at 0.25)
                "oos_mdd": ([-0.20] * 95) + ([-0.30] * 5),
            }
        )

        _, summary = apply_robust_gates(df, metric="calmar_ratio", max_oos_mdd_p95=0.25)
        self.assertFalse(summary["gate_passed"])
        self.assertGreater(summary["oos_mdd_p95"], 0.25)


if __name__ == "__main__":
    unittest.main()
