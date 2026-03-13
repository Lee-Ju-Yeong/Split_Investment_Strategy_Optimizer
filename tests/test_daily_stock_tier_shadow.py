import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.daily_stock_tier_shadow import build_shadow_diff_frame, summarize_shadow_diff


class TestDailyStockTierShadow(unittest.TestCase):
    def test_build_shadow_diff_flags_sbv_and_tier_changes(self):
        base = pd.DataFrame(
            [
                {"date": "2024-01-01", "stock_code": "A0001", "tier": 1, "reason": "prime", "sbv_ratio": None},
                {"date": "2024-01-02", "stock_code": "A0001", "tier": 1, "reason": "prime", "sbv_ratio": 0.03},
            ]
        )
        lagged = pd.DataFrame(
            [
                {"date": "2024-01-01", "stock_code": "A0001", "tier": 1, "reason": "prime", "sbv_ratio": None},
                {"date": "2024-01-02", "stock_code": "A0001", "tier": 3, "reason": "prime+sbv_ratio_extreme", "sbv_ratio": None},
                {"date": "2024-01-03", "stock_code": "A0001", "tier": 3, "reason": "prime+sbv_ratio_extreme", "sbv_ratio": 0.03},
            ]
        )

        diff = build_shadow_diff_frame(base, lagged)
        day2 = diff.loc[diff["date"] == "2024-01-02"].iloc[0]
        day3 = diff.loc[diff["date"] == "2024-01-03"].iloc[0]

        self.assertTrue(day2["sbv_disappeared"])
        self.assertTrue(day2["tier_changed"])
        self.assertTrue(day2["reason_changed"])
        self.assertTrue(day2["affected"])

        self.assertTrue(day3["row_presence_changed"])
        self.assertTrue(day3["sbv_appeared"])
        self.assertTrue(day3["affected"])

    def test_summarize_shadow_diff_counts_affected_dates(self):
        diff = pd.DataFrame(
            [
                {"date": "2024-01-01", "stock_code": "A", "sbv_ratio_base": None, "sbv_ratio_lagged": None, "sbv_appeared": False, "sbv_disappeared": False, "sbv_changed_value": False, "tier_changed": False, "reason_changed": False, "affected": False},
                {"date": "2024-01-02", "stock_code": "A", "sbv_ratio_base": 0.02, "sbv_ratio_lagged": None, "sbv_appeared": False, "sbv_disappeared": True, "sbv_changed_value": False, "tier_changed": True, "reason_changed": True, "affected": True},
                {"date": "2024-01-03", "stock_code": "A", "sbv_ratio_base": None, "sbv_ratio_lagged": 0.02, "sbv_appeared": True, "sbv_disappeared": False, "sbv_changed_value": False, "tier_changed": True, "reason_changed": True, "affected": True},
            ]
        )

        summary = summarize_shadow_diff(diff, base_lag_days=0, lag_days=3)

        self.assertEqual(summary["base_lag_days"], 0)
        self.assertEqual(summary["lag_days"], 3)
        self.assertEqual(summary["rows_compared"], 3)
        self.assertEqual(summary["affected_rows"], 2)
        self.assertEqual(summary["affected_dates"], 2)
        self.assertEqual(summary["first_affected_date"], "2024-01-02")
        self.assertEqual(summary["last_affected_date"], "2024-01-03")
        self.assertEqual(summary["sbv_appeared_rows"], 1)
        self.assertEqual(summary["sbv_disappeared_rows"], 1)
        self.assertEqual(summary["tier_changed_rows"], 2)
        self.assertEqual(summary["reason_changed_rows"], 2)


if __name__ == "__main__":
    unittest.main()
