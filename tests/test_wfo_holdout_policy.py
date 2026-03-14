import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analysis import walk_forward_analyzer as wfo


class TestWfoHoldoutPolicy(unittest.TestCase):
    def test_short_holdout_is_internal_provisional_even_with_adequacy_fields(self):
        policy = wfo.evaluate_holdout_policy(
            holdout_start="2025-01-01",
            holdout_end="2025-11-30",
            adequacy_metrics={
                "trade_count": 120,
                "closed_trade_count": 80,
                "avg_hold_days": 42.5,
                "distinct_entry_months": 10,
            },
        )

        self.assertEqual(policy["holdout_class"], "internal_provisional")
        self.assertFalse(policy["approval_eligible"])
        self.assertIn("holdout_too_short=334<730", policy["reasons"])

    def test_contaminated_holdout_is_internal_provisional(self):
        policy = wfo.evaluate_holdout_policy(
            holdout_start="2024-01-01",
            holdout_end="2025-12-31",
            contaminated_ranges=[("2025-12-01", "2026-01-31")],
            adequacy_metrics={
                "trade_count": 200,
                "closed_trade_count": 140,
                "avg_hold_days": 55.0,
                "distinct_entry_months": 20,
            },
        )

        self.assertEqual(policy["holdout_class"], "internal_provisional")
        self.assertFalse(policy["approval_eligible"])
        self.assertTrue(policy["contaminated_overlap"])
        self.assertIn("holdout_range_contaminated", policy["reasons"])

    def test_long_clean_holdout_with_required_adequacy_fields_is_approval_grade(self):
        policy = wfo.evaluate_holdout_policy(
            holdout_start="2022-01-01",
            holdout_end="2023-12-31",
            adequacy_metrics={
                "trade_count": 220,
                "closed_trade_count": 170,
                "avg_hold_days": 48.0,
                "distinct_entry_months": 22,
                "avg_invested_capital_ratio": 0.78,
            },
        )

        self.assertEqual(policy["holdout_length_days"], 730)
        self.assertEqual(policy["holdout_class"], "approval_grade")
        self.assertTrue(policy["approval_eligible"])
        self.assertEqual(policy["missing_adequacy_fields"], [])

    def test_build_holdout_manifest_carries_policy_and_metrics(self):
        manifest = wfo.build_holdout_manifest(
            holdout_start="2022-01-01",
            holdout_end="2023-12-31",
            wfo_end="2021-12-31",
            adequacy_metrics={
                "trade_count": 220,
                "closed_trade_count": 170,
                "avg_hold_days": 48.0,
                "distinct_entry_months": 22,
                "peak_slot_utilization": 0.9,
                "realized_split_depth": 4.0,
                "avg_invested_capital_ratio": 0.78,
                "cash_drag_ratio": 0.22,
            },
        )

        self.assertEqual(manifest["holdout_class"], "approval_grade")
        self.assertTrue(manifest["approval_eligible"])
        self.assertEqual(manifest["holdout_length_days"], 730)
        self.assertEqual(manifest["wfo_end"], "2021-12-31")
        self.assertEqual(manifest["trade_count"], 220)
        self.assertEqual(manifest["avg_invested_capital_ratio"], 0.78)
        self.assertEqual(manifest["cash_drag_ratio"], 0.22)
        self.assertTrue(manifest["holdout_date_reuse_forbidden"])

    def test_build_lane_manifest_defaults_to_internal_provisional_when_not_eligible(self):
        manifest = wfo.build_lane_manifest(
            lane_type="legacy_wfo",
            approval_eligible=False,
            decision_date="2026-03-14",
            research_data_cutoff="2021-12-31",
            promotion_data_cutoff="2021-12-31",
            composite_curve_allowed=True,
            cpu_audit_outcome="disabled",
            reasons=["lane_mode_not_separated"],
        )

        self.assertEqual(manifest["lane_type"], "legacy_wfo")
        self.assertEqual(manifest["evidence_tier"], "internal_provisional")
        self.assertFalse(manifest["approval_eligible"])
        self.assertTrue(manifest["composite_curve_allowed"])
        self.assertEqual(manifest["cpu_audit_outcome"], "disabled")
        self.assertEqual(manifest["reasons"], ["lane_mode_not_separated"])

    def test_write_wfo_manifests_persists_lane_and_holdout_json(self):
        lane_manifest = wfo.build_lane_manifest(
            lane_type="legacy_wfo",
            approval_eligible=False,
            decision_date="2026-03-14",
            promotion_data_cutoff="2021-12-31",
            composite_curve_allowed=True,
            cpu_audit_outcome="disabled",
            reasons=["lane_mode_not_separated"],
        )
        holdout_manifest = wfo.build_holdout_manifest(
            holdout_start="2025-01-01",
            holdout_end="2025-11-30",
            wfo_end="2021-12-31",
            adequacy_metrics={},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = wfo.write_wfo_manifests(
                results_dir=tmp_dir,
                lane_manifest=lane_manifest,
                holdout_manifest=holdout_manifest,
            )

            lane_payload = Path(paths["lane_manifest_path"]).read_text(encoding="utf-8")
            holdout_payload = Path(paths["holdout_manifest_path"]).read_text(encoding="utf-8")

        self.assertIn('"lane_type": "legacy_wfo"', lane_payload)
        self.assertIn('"cpu_audit_outcome": "disabled"', lane_payload)
        self.assertIn('"holdout_start": "2025-01-01"', holdout_payload)
        self.assertIn('"holdout_class": "internal_provisional"', holdout_payload)


if __name__ == "__main__":
    unittest.main()
