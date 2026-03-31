import os
import sys
import tempfile
import unittest
from pathlib import Path
import time as time_module
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analysis import walk_forward_analyzer as wfo


class TestWfoHoldoutPolicy(unittest.TestCase):
    def test_resolve_holdout_runtime_settings_reads_min_length_override(self):
        settings = wfo._resolve_holdout_runtime_settings(
            {
                "holdout_start": "2024-01-01",
                "holdout_end": "2025-12-31",
                "holdout_min_length_days": "486",
                "holdout_contaminated_ranges": [
                    {"start": "2026-01-01", "end": "2026-01-31"}
                ],
            }
        )

        self.assertEqual(settings["min_length_days"], 486)

    def test_resolve_holdout_runtime_settings_rejects_non_positive_min_length_override(self):
        with self.assertRaisesRegex(
            ValueError,
            "holdout_min_length_days must be a positive integer",
        ):
            wfo._resolve_holdout_runtime_settings(
                {
                    "holdout_min_length_days": 0,
                }
            )

    @patch("src.data_handler.DataHandler")
    def test_load_runtime_trading_dates_requires_daily_stock_price_when_trading_days(self, mock_handler_cls):
        mock_handler = mock_handler_cls.return_value
        mock_handler.get_trading_dates.side_effect = RuntimeError("db unavailable")

        with self.assertRaisesRegex(
            RuntimeError,
            "period_length_basis=trading_days requires runtime trading dates",
        ):
            wfo._load_runtime_trading_dates(
                config={"database": {"host": "127.0.0.1"}},
                backtest_settings={
                    "start_date": "2020-01-01",
                    "end_date": "2024-12-31",
                },
                wfo_settings={
                    "period_length_basis": "trading_days",
                    "lane_type": "promotion_evaluation",
                },
                holdout_settings={},
            )

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

        self.assertEqual(policy["internal_holdout_class"], "internal_provisional")
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

        self.assertEqual(policy["internal_holdout_class"], "internal_provisional")
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

        self.assertEqual(policy["internal_holdout_class"], "internal_provisional")
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
        self.assertEqual(policy["internal_holdout_class"], "internal_approval_ready")
        self.assertTrue(policy["approval_eligible"])
        self.assertEqual(policy["missing_adequacy_fields"], [])

    def test_evaluate_holdout_policy_uses_trading_day_count_when_requested(self):
        policy = wfo.evaluate_holdout_policy(
            holdout_start="2025-01-01",
            holdout_end="2025-01-06",
            wfo_end="2024-12-31",
            adequacy_metrics={
                "trade_count": 10,
                "closed_trade_count": 8,
                "avg_hold_days": 5.0,
                "distinct_entry_months": 1,
            },
            min_length_days=4,
            length_basis="trading_days",
            trading_dates=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
        )

        self.assertEqual(policy["holdout_length_days"], 3)
        self.assertEqual(policy["holdout_length_basis"], "trading_days")
        self.assertEqual(policy["holdout_min_length_days"], 4)
        self.assertIn("holdout_too_short=3<4", policy["reasons"])

    def test_holdout_is_internal_provisional_when_adequacy_thresholds_fail(self):
        policy = wfo.evaluate_holdout_policy(
            holdout_start="2022-01-01",
            holdout_end="2023-12-31",
            wfo_end="2021-12-31",
            adequacy_metrics={
                "trade_count": 12,
                "closed_trade_count": 8,
                "avg_hold_days": 48.0,
                "distinct_entry_months": 4,
                "avg_invested_capital_ratio": 0.12,
                "cash_drag_ratio": 0.88,
            },
            adequacy_thresholds={
                "min_trade_count": 20,
                "min_distinct_entry_months": 6,
                "min_avg_invested_capital_ratio": 0.20,
                "max_cash_drag_ratio": 0.80,
            },
        )

        self.assertEqual(policy["internal_holdout_class"], "internal_provisional")
        self.assertFalse(policy["approval_eligible"])
        self.assertIn("trade_count_below_min=12.0000<20.0000", policy["reasons"])
        self.assertIn("distinct_entry_months_below_min=4.0000<6.0000", policy["reasons"])
        self.assertIn(
            "avg_invested_capital_ratio_below_min=0.1200<0.2000",
            policy["reasons"],
        )
        self.assertIn("cash_drag_ratio_above_max=0.8800>0.8000", policy["reasons"])

    def test_holdout_waiver_can_preserve_approval_for_short_or_adequacy_failures(self):
        policy = wfo.evaluate_holdout_policy(
            holdout_start="2025-01-01",
            holdout_end="2025-11-30",
            wfo_end="2024-12-31",
            adequacy_metrics={
                "trade_count": 12,
                "closed_trade_count": 8,
                "avg_hold_days": 48.0,
                "distinct_entry_months": 4,
            },
            adequacy_thresholds={
                "min_trade_count": 20,
                "min_distinct_entry_months": 6,
            },
            waiver_reason="committee_override_due_to_limited_fresh_window",
        )

        self.assertTrue(policy["approval_eligible"])
        self.assertEqual(policy["internal_holdout_class"], "internal_approval_ready")
        self.assertFalse(policy["external_claim_eligible"])
        self.assertTrue(policy["waiver_applied"])
        self.assertEqual(
            policy["waiver_reason"],
            "committee_override_due_to_limited_fresh_window",
        )
        self.assertEqual(policy["reasons"], [])
        self.assertIn("holdout_too_short=334<730", policy["waived_reasons"])
        self.assertIn("trade_count_below_min=12.0000<20.0000", policy["waived_reasons"])

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
            holdout_backtest_attempted=True,
            holdout_backtest_success=True,
            holdout_backtest_blocked=False,
        )

        self.assertEqual(manifest["internal_holdout_class"], "internal_approval_ready")
        self.assertTrue(manifest["approval_eligible"])
        self.assertTrue(manifest["external_claim_eligible"])
        self.assertEqual(manifest["holdout_length_days"], 730)
        self.assertEqual(manifest["wfo_end"], "2021-12-31")
        self.assertTrue(manifest["holdout_backtest_executed"])
        self.assertTrue(manifest["holdout_backtest_attempted"])
        self.assertTrue(manifest["holdout_backtest_success"])
        self.assertFalse(manifest["holdout_backtest_blocked"])
        self.assertTrue(manifest["promotion_wfo_end_before_holdout"])
        self.assertEqual(manifest["trade_count"], 220)
        self.assertEqual(manifest["avg_invested_capital_ratio"], 0.78)
        self.assertEqual(manifest["cash_drag_ratio"], 0.22)
        self.assertTrue(manifest["holdout_date_reuse_forbidden"])

    def test_build_holdout_manifest_records_thresholds_and_waiver(self):
        manifest = wfo.build_holdout_manifest(
            holdout_start="2025-01-01",
            holdout_end="2025-11-30",
            wfo_end="2024-12-31",
            adequacy_metrics={
                "trade_count": 12,
                "closed_trade_count": 8,
                "avg_hold_days": 48.0,
                "distinct_entry_months": 4,
            },
            adequacy_thresholds={
                "min_trade_count": 20,
                "min_distinct_entry_months": 6,
            },
            waiver_reason="committee_override_due_to_limited_fresh_window",
            holdout_backtest_attempted=True,
            holdout_backtest_success=True,
            holdout_backtest_blocked=False,
        )

        self.assertTrue(manifest["approval_eligible"])
        self.assertFalse(manifest["external_claim_eligible"])
        self.assertTrue(manifest["waiver_applied"])
        self.assertEqual(
            manifest["adequacy_thresholds"]["min_trade_count"],
            20.0,
        )
        self.assertIn("holdout_too_short=334<730", manifest["waived_reasons"])

    def test_build_holdout_manifest_distinguishes_attempted_from_success(self):
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
            holdout_backtest_attempted=True,
            holdout_backtest_success=False,
            holdout_backtest_blocked=True,
        )

        self.assertTrue(manifest["holdout_backtest_executed"])
        self.assertTrue(manifest["holdout_backtest_attempted"])
        self.assertFalse(manifest["holdout_backtest_success"])
        self.assertTrue(manifest["holdout_backtest_blocked"])
        self.assertFalse(manifest["approval_eligible"])
        self.assertIn("holdout_backtest_not_successful", manifest["reasons"])
        self.assertIn("holdout_backtest_blocked", manifest["reasons"])

    def test_build_lane_manifest_defaults_to_internal_provisional_when_not_eligible(self):
        manifest = wfo.build_lane_manifest(
            lane_type="legacy_wfo",
            approval_eligible=False,
            external_claim_eligible=False,
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
        self.assertFalse(manifest["external_claim_eligible"])
        self.assertTrue(manifest["composite_curve_allowed"])
        self.assertEqual(manifest["cpu_audit_outcome"], "disabled")
        self.assertEqual(manifest["reasons"], ["lane_mode_not_separated"])

    def test_build_lane_manifest_can_record_cpu_audit_pass(self):
        manifest = wfo.build_lane_manifest(
            lane_type="legacy_wfo",
            approval_eligible=False,
            external_claim_eligible=False,
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

    def test_promotion_lane_requires_frozen_shortlist_path(self):
        with self.assertRaisesRegex(ValueError, "promotion_shortlist_path"):
            wfo._resolve_promotion_runtime_settings({})

    def test_resolve_promotion_runtime_settings_uses_explicit_shortlist_and_metric(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            shortlist_path = Path(tmp_dir, "promotion_shortlist.csv")
            shortlist_path.write_text(
                "max_stocks,order_investment_ratio,additional_buy_drop_rate,sell_profit_rate,"
                "additional_buy_priority,stop_loss_rate,max_splits_limit,max_inactivity_period\n"
                "20,0.02,0.04,0.05,0,-0.15,10,90\n",
                encoding="utf-8",
            )

            settings = wfo._resolve_promotion_runtime_settings(
                {
                    "promotion_mode": "frozen_shortlist_single_anchor_eval",
                    "promotion_shortlist_path": shortlist_path.as_posix(),
                    "promotion_selection_metric": "cagr",
                    "decision_date": "2026-03-14",
                }
            )

        self.assertEqual(
            settings["promotion_shortlist_path"],
            shortlist_path.as_posix(),
        )
        self.assertEqual(settings["promotion_selection_metric"], "cagr")
        self.assertEqual(
            settings["promotion_mode"],
            "frozen_shortlist_single_anchor_eval",
        )
        self.assertTrue(settings["shortlist_hash"])
        self.assertEqual(settings["decision_date"], "2026-03-14")

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

    def test_build_promotion_fold_periods_uses_trading_day_windows_when_requested(self):
        trading_dates = pd.to_datetime(
            [
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ]
        )

        fold_periods, overlap_days = wfo._build_promotion_fold_periods(
            "2020-01-01",
            "2020-01-10",
            total_folds=2,
            period_length_days=2,
            length_basis="trading_days",
            trading_dates=trading_dates,
        )

        self.assertEqual(overlap_days, 0)
        self.assertEqual(fold_periods[0]["IS_Start"].isoformat(), "2020-01-02")
        self.assertEqual(fold_periods[0]["IS_End"].isoformat(), "2020-01-06")
        self.assertEqual(fold_periods[0]["OOS_Start"].isoformat(), "2020-01-07")
        self.assertEqual(fold_periods[0]["OOS_End"].isoformat(), "2020-01-08")
        self.assertEqual(fold_periods[1]["IS_End"].isoformat(), "2020-01-08")
        self.assertEqual(fold_periods[1]["OOS_Start"].isoformat(), "2020-01-09")
        self.assertEqual(fold_periods[1]["OOS_End"].isoformat(), "2020-01-10")

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

    def test_final_candidate_manifest_uses_single_champion_and_gate_passing_reserves_only(self):
        summary_df = pd.DataFrame(
            [
                {
                    "selection_rank": 1,
                    "shortlist_candidate_id": 2,
                    "candidate_signature": "max_stocks=30",
                    "hard_gate_pass": True,
                    "hard_gate_fail_reasons": "",
                    "robust_score": 0.42,
                    "promotion_fold_pass_rate": 1.0,
                    "promotion_oos_calmar_median": 0.20,
                    "promotion_oos_mdd_depth_worst": 0.00,
                    "promotion_oos_cagr_median": 0.20,
                    "max_stocks": 30,
                    "order_investment_ratio": 0.03,
                    "additional_buy_drop_rate": 0.05,
                    "sell_profit_rate": 0.06,
                    "additional_buy_priority": "highest_drop",
                    "stop_loss_rate": -0.10,
                    "max_splits_limit": 15,
                    "max_inactivity_period": 60,
                },
                {
                    "selection_rank": 2,
                    "shortlist_candidate_id": 3,
                    "candidate_signature": "max_stocks=25",
                    "hard_gate_pass": True,
                    "hard_gate_fail_reasons": "",
                    "robust_score": 0.31,
                    "promotion_fold_pass_rate": 0.8,
                    "promotion_oos_calmar_median": 0.18,
                    "promotion_oos_mdd_depth_worst": 0.05,
                    "promotion_oos_cagr_median": 0.18,
                    "max_stocks": 25,
                    "order_investment_ratio": 0.025,
                    "additional_buy_drop_rate": 0.05,
                    "sell_profit_rate": 0.055,
                    "additional_buy_priority": "highest_drop",
                    "stop_loss_rate": -0.10,
                    "max_splits_limit": 12,
                    "max_inactivity_period": 70,
                },
                {
                    "selection_rank": 3,
                    "shortlist_candidate_id": 1,
                    "candidate_signature": "max_stocks=20",
                    "hard_gate_pass": False,
                    "hard_gate_fail_reasons": "promotion_fold_pass_rate_below_min",
                    "robust_score": 0.08,
                    "promotion_fold_pass_rate": 0.5,
                    "promotion_oos_calmar_median": 0.10,
                    "promotion_oos_mdd_depth_worst": 0.03,
                    "promotion_oos_cagr_median": 0.10,
                    "max_stocks": 20,
                    "order_investment_ratio": 0.02,
                    "additional_buy_drop_rate": 0.04,
                    "sell_profit_rate": 0.05,
                    "additional_buy_priority": "lowest_order",
                    "stop_loss_rate": -0.15,
                    "max_splits_limit": 10,
                    "max_inactivity_period": 90,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            shortlist_path = Path(tmp_dir, "promotion_shortlist.csv")
            shortlist_path.write_text(
                "max_stocks,order_investment_ratio,additional_buy_drop_rate,sell_profit_rate,additional_buy_priority,stop_loss_rate,max_splits_limit,max_inactivity_period\n"
                "20,0.02,0.04,0.05,0,-0.15,10,90\n"
                "30,0.03,0.05,0.06,1,-0.10,15,60\n"
                "25,0.025,0.05,0.055,1,-0.10,12,70\n",
                encoding="utf-8",
            )
            past_ts = 1735603200
            os.utime(shortlist_path, (past_ts, past_ts))
            manifest = wfo._build_final_candidate_manifest(
                summary_df,
                selection_settings=wfo._resolve_selection_contract_settings(
                    {"selection_contract": {"reserve_count": 2}}
                ),
                shortlist_path=shortlist_path.as_posix(),
                shortlist_hash=wfo._hash_file_sha256(shortlist_path.as_posix()),
                decision_date="2026-03-14",
                research_data_cutoff="2024-12-31",
                promotion_data_cutoff="2024-12-31",
                holdout_settings={
                    "holdout_start": "2025-01-01",
                    "holdout_end": "2025-11-30",
                    "canonical_promotion_wfo_end": "2024-12-31",
                    "canonical_holdout_start": "2025-01-01",
                    "canonical_holdout_end": "2025-11-30",
                },
                engine_version_hash="engine123",
                cpu_audit_required=True,
            )

        self.assertEqual(manifest["selection_mode"], "single_champion_only")
        self.assertEqual(manifest["champion_candidate_id"], 2)
        self.assertEqual(manifest["reserve_candidate_ids"], [3])
        self.assertEqual(manifest["champion_params"]["max_stocks"], 30)
        self.assertEqual(manifest["reserve_candidates"][0]["candidate_id"], 3)
        self.assertTrue(manifest["holdout_candidate_pack_forbidden"])
        self.assertEqual(
            manifest["reserve_succession_rule"],
            "prelocked_non_performance_only",
        )
        self.assertEqual(manifest["robust_score_version"], "promotion_robust_score_v1")
        self.assertEqual(
            manifest["robust_score_thresholds"]["robust_score_std_penalty"],
            0.50,
        )
        self.assertEqual(
            manifest["freeze_contract_version"],
            "promotion_freeze_contract_v1",
        )
        self.assertTrue(manifest["freeze_contract_verified"])
        self.assertTrue(manifest["promotion_shortlist_hash_verified"])
        self.assertFalse(manifest["promotion_shortlist_modified_after_decision_date"])
        self.assertTrue(manifest["canonical_holdout_contract_verified"])
        self.assertEqual(len(manifest["hard_gate_results_by_candidate"]), 3)

    def test_compute_holdout_adequacy_metrics_uses_daily_snapshots_and_trade_history(self):
        cpu_result = {
            "daily_snapshots": [
                {"date": "2025-01-02", "total_value": 100.0, "cash": 40.0, "stock_count": 1},
                {"date": "2025-01-03", "total_value": 120.0, "cash": 20.0, "stock_count": 2},
            ],
            "trade_history": [
                {"date": "2025-01-02", "code": "005930", "trade_type": "BUY", "order": 1},
                {"date": "2025-01-15", "code": "005930", "trade_type": "SELL", "order": 1},
                {"date": "2025-02-01", "code": "000660", "trade_type": "BUY", "order": 2},
            ],
        }

        metrics = wfo._compute_holdout_adequacy_metrics(cpu_result, max_stocks=4)

        self.assertEqual(metrics["trade_count"], 3)
        self.assertEqual(metrics["closed_trade_count"], 1)
        self.assertEqual(metrics["distinct_entry_months"], 2)
        self.assertEqual(metrics["realized_split_depth"], 2.0)
        self.assertAlmostEqual(metrics["avg_hold_days"], 13.0)
        self.assertAlmostEqual(metrics["peak_slot_utilization"], 0.5)
        self.assertAlmostEqual(metrics["avg_invested_capital_ratio"], ((0.6 + (100.0 / 120.0)) / 2.0))
        self.assertAlmostEqual(metrics["cash_drag_ratio"], ((0.4 + (20.0 / 120.0)) / 2.0))

    def test_holdout_auto_execute_block_reasons_require_dates_and_hard_gate_pass(self):
        reasons = wfo._build_holdout_auto_execute_block_reasons(
            {
                "champion_hard_gate_pass": False,
                "champion_params": {},
                "freeze_contract_verified": False,
                "freeze_contract_hash": "",
                "freeze_contract_reasons": [
                    "promotion_shortlist_modified_after_decision_date"
                ],
            },
            holdout_start=None,
            holdout_end="2025-11-30",
        )

        self.assertIn("holdout_window_missing_for_auto_execute", reasons)
        self.assertIn("no_candidate_passed_hard_gate", reasons)
        self.assertIn("champion_params_missing", reasons)
        self.assertIn("freeze_contract_not_verified", reasons)
        self.assertIn("freeze_contract_hash_mismatch", reasons)
        self.assertIn("promotion_shortlist_modified_after_decision_date", reasons)

    def test_build_final_candidate_manifest_flags_shortlist_modified_after_decision_date(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            shortlist_path = Path(tmp_dir, "promotion_shortlist.csv")
            shortlist_path.write_text(
                "max_stocks,order_investment_ratio,additional_buy_drop_rate,sell_profit_rate,additional_buy_priority,stop_loss_rate,max_splits_limit,max_inactivity_period\n"
                "30,0.03,0.05,0.06,1,-0.10,15,60\n",
                encoding="utf-8",
            )
            future_ts = time_module.time() + 86400
            os.utime(shortlist_path, (future_ts, future_ts))
            summary_df = pd.DataFrame(
                [
                    {
                        "selection_rank": 1,
                        "shortlist_candidate_id": 1,
                        "candidate_signature": "max_stocks=30",
                        "hard_gate_pass": True,
                        "hard_gate_fail_reasons": "",
                        "robust_score": 0.42,
                        "promotion_fold_pass_rate": 1.0,
                        "promotion_oos_calmar_median": 0.8,
                        "promotion_oos_mdd_depth_worst": 0.1,
                        "promotion_oos_cagr_median": 0.2,
                        "max_stocks": 30,
                        "order_investment_ratio": 0.03,
                        "additional_buy_drop_rate": 0.05,
                        "sell_profit_rate": 0.06,
                        "additional_buy_priority": "highest_drop",
                        "stop_loss_rate": -0.10,
                        "max_splits_limit": 15,
                        "max_inactivity_period": 60,
                    }
                ]
            )

            manifest = wfo._build_final_candidate_manifest(
                summary_df,
                selection_settings=wfo._resolve_selection_contract_settings({}),
                shortlist_path=shortlist_path.as_posix(),
                shortlist_hash=wfo._hash_file_sha256(shortlist_path.as_posix()),
                decision_date="2026-03-14",
                research_data_cutoff="2024-12-31",
                promotion_data_cutoff="2024-12-31",
                holdout_settings={
                    "holdout_start": "2025-01-01",
                    "holdout_end": "2025-11-30",
                    "canonical_promotion_wfo_end": "2024-12-31",
                    "canonical_holdout_start": "2025-01-01",
                    "canonical_holdout_end": "2025-11-30",
                },
                engine_version_hash="engine123",
                cpu_audit_required=True,
            )

        self.assertFalse(manifest["freeze_contract_verified"])
        self.assertTrue(manifest["promotion_shortlist_modified_after_decision_date"])
        self.assertIn(
            "promotion_shortlist_modified_after_decision_date",
            manifest["freeze_contract_reasons"],
        )

    def test_build_final_candidate_manifest_allows_same_day_shortlist_when_decision_date_is_date_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            shortlist_path = Path(tmp_dir, "promotion_shortlist.csv")
            shortlist_path.write_text(
                "max_stocks,order_investment_ratio,additional_buy_drop_rate,sell_profit_rate,additional_buy_priority,stop_loss_rate,max_splits_limit,max_inactivity_period\n"
                "30,0.03,0.05,0.06,1,-0.10,15,60\n",
                encoding="utf-8",
            )
            same_day_ts = pd.Timestamp("2026-03-14T15:30:00Z").timestamp()
            os.utime(shortlist_path, (same_day_ts, same_day_ts))
            summary_df = pd.DataFrame(
                [
                    {
                        "selection_rank": 1,
                        "shortlist_candidate_id": 1,
                        "candidate_signature": "max_stocks=30",
                        "hard_gate_pass": True,
                        "hard_gate_fail_reasons": "",
                        "robust_score": 0.42,
                        "promotion_fold_pass_rate": 1.0,
                        "promotion_oos_calmar_median": 0.8,
                        "promotion_oos_mdd_depth_worst": 0.1,
                        "promotion_oos_cagr_median": 0.2,
                        "max_stocks": 30,
                        "order_investment_ratio": 0.03,
                        "additional_buy_drop_rate": 0.05,
                        "sell_profit_rate": 0.06,
                        "additional_buy_priority": "highest_drop",
                        "stop_loss_rate": -0.10,
                        "max_splits_limit": 15,
                        "max_inactivity_period": 60,
                    }
                ]
            )

            manifest = wfo._build_final_candidate_manifest(
                summary_df,
                selection_settings=wfo._resolve_selection_contract_settings({}),
                shortlist_path=shortlist_path.as_posix(),
                shortlist_hash=wfo._hash_file_sha256(shortlist_path.as_posix()),
                decision_date="2026-03-14",
                research_data_cutoff="2024-12-31",
                promotion_data_cutoff="2024-12-31",
                holdout_settings={
                    "holdout_start": "2025-01-01",
                    "holdout_end": "2025-11-30",
                    "canonical_promotion_wfo_end": "2024-12-31",
                    "canonical_holdout_start": "2025-01-01",
                    "canonical_holdout_end": "2025-11-30",
                },
                engine_version_hash="engine123",
                cpu_audit_required=True,
            )

        self.assertTrue(manifest["freeze_contract_verified"])
        self.assertFalse(manifest["promotion_shortlist_modified_after_decision_date"])

    def test_require_decision_date_preserves_timestamp_precision(self):
        self.assertEqual(
            wfo._require_decision_date(
                "2026-03-14T09:30:00+09:00",
                context_label="promotion freeze contract",
            ),
            "2026-03-14T09:30:00+09:00",
        )

    def test_compute_robust_score_penalizes_variance(self):
        stable_score, stable_mean, stable_std = wfo._compute_robust_score(
            pd.Series([0.20, 0.20, 0.20]),
            std_penalty=0.50,
        )
        volatile_score, volatile_mean, volatile_std = wfo._compute_robust_score(
            pd.Series([0.05, 0.20, 0.35]),
            std_penalty=0.50,
        )

        self.assertAlmostEqual(stable_mean, volatile_mean)
        self.assertLess(stable_std, volatile_std)
        self.assertGreater(stable_score, volatile_score)

    def test_build_current_lane_reasons_no_longer_uses_legacy_cpu_selection_for_promotion_gate(self):
        reasons = wfo._build_current_lane_reasons(
            lane_type="promotion_evaluation",
            total_folds=4,
            overlap_days=0,
            cpu_audit_outcome="disabled",
        )

        self.assertEqual(reasons, [])

    def test_write_wfo_manifests_persists_lane_and_holdout_json(self):
        lane_manifest = wfo.build_lane_manifest(
            lane_type="legacy_wfo",
            approval_eligible=False,
            external_claim_eligible=False,
            decision_date="2026-03-14",
            promotion_data_cutoff="2021-12-31",
            composite_curve_allowed=True,
            cpu_audit_outcome="disabled",
            selection_cpu_check_outcome="disabled",
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
        self.assertIn('"selection_cpu_check_outcome": "disabled"', lane_payload)
        self.assertIn('"holdout_start": "2025-01-01"', holdout_payload)
        self.assertIn('"internal_holdout_class": "internal_provisional"', holdout_payload)
        self.assertIn('"holdout_backtest_executed": false', holdout_payload)


if __name__ == "__main__":
    unittest.main()
