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

    def test_holdout_is_internal_provisional_when_it_starts_before_wfo_end(self):
        policy = wfo.evaluate_holdout_policy(
            holdout_start="2021-12-31",
            holdout_end="2023-12-31",
            wfo_end="2021-12-31",
            adequacy_metrics={
                "trade_count": 220,
                "closed_trade_count": 170,
                "avg_hold_days": 48.0,
                "distinct_entry_months": 22,
            },
        )

        self.assertEqual(policy["holdout_class"], "internal_provisional")
        self.assertFalse(policy["approval_eligible"])
        self.assertFalse(policy["promotion_wfo_end_before_holdout"])
        self.assertIn("holdout_starts_on_or_before_wfo_end", policy["reasons"])

    def test_long_clean_holdout_with_required_adequacy_fields_is_approval_grade(self):
        policy = wfo.evaluate_holdout_policy(
            holdout_start="2022-01-01",
            holdout_end="2023-12-31",
            wfo_end="2021-12-31",
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
            holdout_backtest_executed=True,
        )

        self.assertEqual(manifest["holdout_class"], "approval_grade")
        self.assertTrue(manifest["approval_eligible"])
        self.assertEqual(manifest["holdout_length_days"], 730)
        self.assertEqual(manifest["wfo_end"], "2021-12-31")
        self.assertTrue(manifest["holdout_backtest_executed"])
        self.assertTrue(manifest["promotion_wfo_end_before_holdout"])
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

    def test_build_lane_manifest_can_record_cpu_audit_pass(self):
        manifest = wfo.build_lane_manifest(
            lane_type="legacy_wfo",
            approval_eligible=False,
            decision_date="2026-03-14",
            promotion_data_cutoff="2021-12-31",
            composite_curve_allowed=True,
            cpu_audit_outcome="pass",
            reasons=["lane_mode_not_separated"],
        )

        self.assertEqual(manifest["cpu_audit_outcome"], "pass")
        self.assertIn("lane_mode_not_separated", manifest["reasons"])

    def test_resolve_lane_type_accepts_promotion_evaluation(self):
        lane_type = wfo._resolve_lane_type({"lane_type": "promotion_evaluation"})

        self.assertEqual(lane_type, "promotion_evaluation")

    def test_resolve_lane_type_accepts_research_start_date_robustness(self):
        lane_type = wfo._resolve_lane_type({"lane_type": "research_start_date_robustness"})

        self.assertEqual(lane_type, "research_start_date_robustness")

    def test_research_lane_requires_frozen_shortlist_path_and_anchor_dates(self):
        with self.assertRaisesRegex(ValueError, "research_shortlist_path"):
            wfo._resolve_research_runtime_settings({"research_mode": "frozen_shortlist_multi_anchor_eval"})

    def test_build_anchor_manifest_records_anchor_contract(self):
        manifest = wfo.build_anchor_manifest(
            anchor_set_id="anchor_set_v1",
            anchor_dates=["2014-01-01", "2015-01-01"],
            anchor_spacing_rule="manual_explicit",
            minimum_is_length_days=1826,
            minimum_oos_length_days=365,
            coverage_normalized=True,
        )

        self.assertEqual(manifest["anchor_set_id"], "anchor_set_v1")
        self.assertEqual(manifest["anchor_dates"], ["2014-01-01", "2015-01-01"])
        self.assertEqual(manifest["shortlist_freeze_mode"], "frozen_shortlist_multi_anchor_eval")

    def test_build_promotion_fold_periods_creates_single_anchor_non_overlap_schedule(self):
        fold_periods, overlap_days = wfo._build_promotion_fold_periods(
            "2020-01-01",
            "2023-12-31",
            total_folds=2,
            period_length_days=365,
        )

        self.assertEqual(overlap_days, 0)
        self.assertEqual(fold_periods[0]["IS_Start"].isoformat(), "2020-01-01")
        self.assertEqual(fold_periods[1]["IS_Start"].isoformat(), "2020-01-01")
        self.assertEqual(fold_periods[0]["OOS_Start"].isoformat(), "2022-01-01")
        self.assertEqual(fold_periods[0]["OOS_End"].isoformat(), "2022-12-31")
        self.assertEqual(fold_periods[1]["OOS_Start"].isoformat(), "2023-01-01")

    def test_aggregate_oos_curves_stitches_non_overlap_promotion_curve(self):
        import pandas as pd

        first_curve = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        second_curve = pd.Series(
            [102.0, 103.0],
            index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
        )

        aggregated = wfo._aggregate_oos_curves(
            [first_curve, second_curve],
            mode="stitch_non_overlap",
        )

        self.assertEqual(len(aggregated), 4)
        self.assertEqual(float(aggregated.iloc[-1]), 103.0)

    def test_aggregate_oos_curves_rejects_duplicate_dates_in_promotion_mode(self):
        import pandas as pd

        duplicated = [
            pd.Series([100.0], index=pd.to_datetime(["2024-01-01"])),
            pd.Series([101.0], index=pd.to_datetime(["2024-01-01"])),
        ]

        with self.assertRaisesRegex(ValueError, "duplicated OOS dates"):
            wfo._aggregate_oos_curves(duplicated, mode="stitch_non_overlap")

    def test_build_current_lane_reasons_is_empty_for_clean_promotion_lane(self):
        reasons = wfo._build_current_lane_reasons(
            lane_type="promotion_evaluation",
            total_folds=4,
            overlap_days=0,
            cpu_audit_outcome="pass",
        )

        self.assertEqual(reasons, [])

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
        self.assertIn('"holdout_backtest_executed": false', holdout_payload)


if __name__ == "__main__":
    unittest.main()
