import glob
import json
import os
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.analysis import walk_forward_analyzer as wfo


class _FakePerformanceAnalyzer:
    def __init__(self, history_df):
        self.daily_values = history_df["total_value"]

    def get_metrics(self, formatted=False):
        start_value = float(self.daily_values.iloc[0])
        end_value = float(self.daily_values.iloc[-1])
        min_value = float(self.daily_values.min())
        cagr = (end_value / start_value) - 1.0 if start_value else 0.0
        mdd = (min_value / start_value) - 1.0 if start_value else 0.0
        calmar = cagr / abs(mdd) if mdd not in (0.0, -0.0) else cagr
        if formatted:
            return {
                "CAGR": f"{cagr:.2%}",
                "MDD": f"{mdd:.2%}",
                "Calmar": f"{calmar:.2f}",
            }
        return {
            "cagr": cagr,
            "mdd": mdd,
            "calmar_ratio": calmar,
        }


class TestWfoLaneExecution(unittest.TestCase):
    def _promotion_config(self):
        return {
            "database": {"host": "127.0.0.1", "user": "root", "password": "pw", "database": "stocks"},
            "backtest_settings": {
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
                "initial_cash": 10_000_000,
            },
            "strategy_params": {
                "price_basis": "adjusted",
                "adjusted_price_gate_start_date": "2013-11-20",
                "universe_mode": "strict_pit",
                "candidate_source_mode": "tier",
                "max_stocks": 20,
                "order_investment_ratio": 0.02,
                "additional_buy_drop_rate": 0.04,
                "sell_profit_rate": 0.05,
                "additional_buy_priority": "lowest_order",
                "stop_loss_rate": -0.15,
                "max_splits_limit": 10,
                "max_inactivity_period": 90,
            },
            "execution_params": {
                "buy_commission_rate": 0.00015,
                "sell_commission_rate": 0.00015,
                "sell_tax_rate": 0.0018,
            },
            "paths": {"results_dir": "results"},
            "walk_forward_settings": {
                "lane_type": "promotion_evaluation",
                "promotion_mode": "frozen_shortlist_single_anchor_eval",
                "total_folds": 2,
                "period_length_days": 365,
                "cpu_certification_enabled": False,
                "decision_date": "2099-12-31",
                "promotion_data_cutoff": "2023-12-31",
                "holdout_start": "2025-01-01",
                "holdout_end": "2025-11-30",
            },
        }

    def _write_shortlist(self, path: Path) -> None:
        pd.DataFrame(
            [
                {
                    "max_stocks": 20,
                    "order_investment_ratio": 0.02,
                    "additional_buy_drop_rate": 0.04,
                    "sell_profit_rate": 0.05,
                    "additional_buy_priority": 0.0,
                    "stop_loss_rate": -0.15,
                    "max_splits_limit": 10,
                    "max_inactivity_period": 90,
                },
                {
                    "max_stocks": 30,
                    "order_investment_ratio": 0.03,
                    "additional_buy_drop_rate": 0.05,
                    "sell_profit_rate": 0.06,
                    "additional_buy_priority": 1.0,
                    "stop_loss_rate": -0.10,
                    "max_splits_limit": 15,
                    "max_inactivity_period": 60,
                },
            ]
        ).to_csv(path, index=False)

    def test_promotion_lane_writes_non_overlap_manifest_and_stitched_curve(self):
        config = self._promotion_config()
        gpu_calls = []

        def _fake_find_optimal_parameters(*args, **kwargs):
            raise AssertionError("promotion lane should not call find_optimal_parameters")

        def _fake_run_single_backtest(*, start_date, end_date, params_dict, initial_cash):
            gpu_calls.append((start_date, end_date, float(initial_cash)))
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            duration_days = int((end - start).days + 1)
            if duration_days > 365:
                self.assertEqual(float(initial_cash), 10_000_000.0)
                if int(params_dict["max_stocks"]) == 30:
                    return pd.Series(
                        [initial_cash, initial_cash * 1.30],
                        index=pd.to_datetime([start_date, end_date]),
                    )
                return pd.Series(
                    [initial_cash, initial_cash * 1.10],
                    index=pd.to_datetime([start_date, end_date]),
                )
            if start_date == "2022-01-01":
                factor = 1.20 if int(params_dict["max_stocks"]) == 30 else 1.05
                return pd.Series(
                    [initial_cash, initial_cash * factor],
                    index=pd.to_datetime([start_date, end_date]),
                )
            if start_date == "2023-01-01":
                factor = 1.20 if int(params_dict["max_stocks"]) == 30 else 1.05
                return pd.Series(
                    [initial_cash, initial_cash * factor],
                    index=pd.to_datetime([start_date, end_date]),
                )
            raise AssertionError(f"unexpected OOS window: {start_date}~{end_date}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            shortlist_path = Path(tmp_dir, "promotion_shortlist.csv")
            pd.DataFrame(
                [
                    {
                        "max_stocks": 20,
                        "order_investment_ratio": 0.02,
                        "additional_buy_drop_rate": 0.04,
                        "sell_profit_rate": 0.05,
                        "additional_buy_priority": 0.0,
                        "stop_loss_rate": -0.15,
                        "max_splits_limit": 10,
                        "max_inactivity_period": 90,
                    },
                    {
                        "max_stocks": 30,
                        "order_investment_ratio": 0.03,
                        "additional_buy_drop_rate": 0.05,
                        "sell_profit_rate": 0.06,
                        "additional_buy_priority": 1.0,
                        "stop_loss_rate": -0.10,
                        "max_splits_limit": 15,
                        "max_inactivity_period": 60,
                    },
                ]
            ).to_csv(shortlist_path, index=False)
            config["walk_forward_settings"]["promotion_shortlist_path"] = shortlist_path.as_posix()
            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                fake_sim_module = types.ModuleType("src.parameter_simulation_gpu")
                fake_sim_module.find_optimal_parameters = _fake_find_optimal_parameters
                fake_gpu_module = types.ModuleType("src.debug_gpu_single_run")
                fake_gpu_module.run_single_backtest = _fake_run_single_backtest
                fake_perf_module = types.ModuleType("src.performance_analyzer")
                fake_perf_module.PerformanceAnalyzer = _FakePerformanceAnalyzer

                with patch("src.analysis.walk_forward_analyzer.load_config", return_value=config), \
                     patch("src.analysis.walk_forward_analyzer.plot_wfo_results", return_value=None), \
                     patch.dict(
                         sys.modules,
                         {
                             "src.parameter_simulation_gpu": fake_sim_module,
                             "src.debug_gpu_single_run": fake_gpu_module,
                             "src.performance_analyzer": fake_perf_module,
                         },
                     ):
                    wfo.run_walk_forward_analysis()

                result_dirs = sorted(glob.glob(os.path.join("results", "wfo_run_*")))
                self.assertEqual(len(result_dirs), 1)
                result_dir = result_dirs[0]
                lane_manifest = json.loads(
                    Path(result_dir, "lane_manifest.json").read_text(encoding="utf-8")
                )
                selection_audit_df = pd.read_csv(
                    Path(result_dir, "wfo_selection_audit.csv")
                )
                candidate_fold_df = pd.read_csv(
                    Path(result_dir, "promotion_candidate_fold_metrics.csv")
                )
                candidate_summary_df = pd.read_csv(
                    Path(result_dir, "promotion_candidate_summary.csv")
                )
                ablation_df = pd.read_csv(
                    Path(result_dir, "promotion_ablation_summary.csv")
                )
                explanation_report = json.loads(
                    Path(result_dir, "promotion_explanation_report.json").read_text(
                        encoding="utf-8"
                    )
                )
                explanation_summary_md = Path(
                    result_dir, "promotion_explanation_summary.md"
                ).read_text(encoding="utf-8")
                final_candidate_manifest = json.loads(
                    Path(result_dir, "final_candidate_manifest.json").read_text(encoding="utf-8")
                )
                holdout_manifest = json.loads(
                    Path(result_dir, "holdout_manifest.json").read_text(encoding="utf-8")
                )
                curve = pd.read_csv(
                    os.path.join(result_dir, "wfo_equity_curve_data.csv"),
                    index_col=0,
                    parse_dates=True,
                ).iloc[:, 0]
            finally:
                os.chdir(previous_cwd)

        call_counts = Counter((start_date, end_date) for start_date, end_date, _ in gpu_calls)
        self.assertEqual(call_counts[("2020-01-01", "2021-12-31")], 2)
        self.assertEqual(call_counts[("2020-01-01", "2022-12-31")], 2)
        self.assertEqual(call_counts[("2022-01-01", "2022-12-31")], 3)
        self.assertEqual(call_counts[("2023-01-01", "2023-12-31")], 3)
        self.assertIn(("2023-01-01", "2023-12-31", 12_000_000.0), gpu_calls)
        self.assertEqual(lane_manifest["lane_type"], "promotion_evaluation")
        self.assertIsNotNone(lane_manifest["shortlist_hash"])
        self.assertEqual(lane_manifest["cpu_audit_outcome"], "pending_final_candidate_audit")
        self.assertEqual(lane_manifest["selection_cpu_check_outcome"], "disabled")
        self.assertIn(
            "final_candidate_cpu_audit_outcome=pending_final_candidate_audit",
            lane_manifest["reasons"],
        )
        self.assertTrue(lane_manifest["composite_curve_allowed"])
        self.assertEqual(
            selection_audit_df["selected_shortlist_candidate_id"].tolist(),
            [2, 2],
        )
        self.assertEqual(len(candidate_fold_df), 4)
        self.assertEqual(candidate_summary_df["shortlist_candidate_id"].tolist(), [2, 1])
        self.assertEqual(
            candidate_summary_df["selection_role"].tolist(),
            ["champion", "ranked_only"],
        )
        self.assertEqual(
            candidate_summary_df["hard_gate_pass"].tolist(),
            [True, False],
        )
        self.assertEqual(final_candidate_manifest["champion_candidate_id"], 2)
        self.assertEqual(final_candidate_manifest["selection_mode"], "single_champion_only")
        self.assertEqual(final_candidate_manifest["reserve_candidate_ids"], [])
        self.assertEqual(final_candidate_manifest["reserve_candidate_signatures"], [])
        self.assertFalse(final_candidate_manifest["reserve_auto_succession_implemented"])
        self.assertTrue(final_candidate_manifest["reserve_auto_succession_deferred"])
        self.assertTrue(final_candidate_manifest["holdout_candidate_pack_forbidden"])
        self.assertEqual(
            final_candidate_manifest["reserve_succession_rule"],
            "prelocked_non_performance_only",
        )
        self.assertFalse(final_candidate_manifest["holdout_ready"])
        self.assertIn(
            "final_candidate_cpu_audit_not_executed",
            final_candidate_manifest["holdout_readiness_reasons"],
        )
        self.assertFalse(final_candidate_manifest.get("holdout_attempted", False))
        self.assertFalse(final_candidate_manifest.get("holdout_success", False))
        self.assertFalse(final_candidate_manifest.get("holdout_blocked", False))
        self.assertFalse(holdout_manifest["approval_eligible"])
        self.assertTrue(holdout_manifest["promotion_wfo_end_before_holdout"])
        self.assertEqual(
            ablation_df["axis"].tolist(),
            [
                "Legacy-Calmar",
                "Robust-Score",
                "Robust+Gate",
                "Robust+Gate+Behavior",
            ],
        )
        self.assertEqual(
            ablation_df["matches_final_champion"].tolist(),
            [True, True, True, True],
        )
        self.assertEqual(
            ablation_df["selection_interpretation"].tolist(),
            ["same_as_final_champion"] * 4,
        )
        self.assertEqual(explanation_report["report_version"], "promotion_explanation_report_v2")
        self.assertTrue(explanation_report["reserve_policy"]["reserve_auto_succession_deferred"])
        self.assertEqual(
            explanation_report["executive_summary"]["champion_selection_reason"],
            "hard gate passed and deterministic tie-break won",
        )
        self.assertTrue(explanation_report["runner_up_comparison"]["runner_up_present"])
        self.assertIn(
            "threshold_checks",
            explanation_report["behavior_evidence"],
        )
        self.assertEqual(
            explanation_report["behavior_evidence"]["behavior_gate_status"],
            "not_attempted",
        )
        self.assertIn("# Promotion Explanation Summary", explanation_summary_md)
        self.assertIn("## Runner-up Comparison", explanation_summary_md)
        self.assertEqual(curve.index.strftime("%Y-%m-%d").tolist(), [
            "2022-01-01",
            "2022-12-31",
            "2023-01-01",
            "2023-12-31",
        ])
        self.assertAlmostEqual(float(curve.iloc[-1]), 14_400_000.0)

    def test_promotion_lane_can_auto_execute_holdout_from_final_candidate_manifest(self):
        config = self._promotion_config()
        config["walk_forward_settings"]["holdout_auto_execute"] = True

        def _fake_find_optimal_parameters(*args, **kwargs):
            raise AssertionError("promotion lane should not call find_optimal_parameters")

        def _fake_run_single_backtest(*, start_date, end_date, params_dict, initial_cash):
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            duration_days = int((end - start).days + 1)
            if duration_days > 365:
                factor = 1.30 if int(params_dict["max_stocks"]) == 30 else 1.10
                return pd.Series(
                    [initial_cash, initial_cash * factor],
                    index=pd.to_datetime([start_date, end_date]),
                )
            factor = 1.20 if int(params_dict["max_stocks"]) == 30 else 1.05
            return pd.Series(
                [initial_cash, initial_cash * factor],
                index=pd.to_datetime([start_date, end_date]),
            )

        def _fake_cpu_run(_config, *, start_date, end_date, params_dict, initial_cash):
            if (start_date, end_date) == ("2020-01-01", "2023-12-31"):
                return pd.Series(
                    [initial_cash, initial_cash * 1.40],
                    index=pd.to_datetime([start_date, end_date]),
                ), {
                    "success": True,
                    "promotion_blocked": False,
                    "metrics": {"calmar_ratio": 1.4, "cagr": 0.20, "mdd": -0.10},
                    "daily_snapshots": [
                        {"date": "2020-01-01", "total_value": initial_cash, "cash": initial_cash * 0.6, "stock_count": 1},
                        {"date": "2023-12-31", "total_value": initial_cash * 1.4, "cash": initial_cash * 0.3, "stock_count": 2},
                    ],
                    "trade_history": [],
                    "final_positions": [],
                }
            if (start_date, end_date) == ("2025-01-01", "2025-11-30"):
                return pd.Series(
                    [initial_cash, initial_cash * 1.12],
                    index=pd.to_datetime([start_date, end_date]),
                ), {
                    "success": True,
                    "promotion_blocked": False,
                    "metrics": {"calmar_ratio": 1.1, "cagr": 0.12, "mdd": -0.08},
                    "daily_snapshots": [
                        {"date": "2025-01-02", "total_value": initial_cash, "cash": initial_cash * 0.7, "stock_count": 1},
                        {"date": "2025-06-30", "total_value": initial_cash * 1.05, "cash": initial_cash * 0.2, "stock_count": 3},
                        {"date": "2025-11-30", "total_value": initial_cash * 1.12, "cash": initial_cash * 0.4, "stock_count": 2},
                    ],
                    "trade_history": [
                        {"date": "2025-01-02", "code": "005930", "trade_type": "BUY", "order": 1},
                        {"date": "2025-03-15", "code": "005930", "trade_type": "SELL", "order": 1},
                        {"date": "2025-05-10", "code": "000660", "trade_type": "BUY", "order": 2},
                    ],
                    "final_positions": [],
                }
            raise AssertionError(f"unexpected cpu window: {start_date}~{end_date}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            shortlist_path = Path(tmp_dir, "promotion_shortlist.csv")
            pd.DataFrame(
                [
                    {
                        "max_stocks": 20,
                        "order_investment_ratio": 0.02,
                        "additional_buy_drop_rate": 0.04,
                        "sell_profit_rate": 0.05,
                        "additional_buy_priority": 0.0,
                        "stop_loss_rate": -0.15,
                        "max_splits_limit": 10,
                        "max_inactivity_period": 90,
                    },
                    {
                        "max_stocks": 30,
                        "order_investment_ratio": 0.03,
                        "additional_buy_drop_rate": 0.05,
                        "sell_profit_rate": 0.06,
                        "additional_buy_priority": 1.0,
                        "stop_loss_rate": -0.10,
                        "max_splits_limit": 15,
                        "max_inactivity_period": 60,
                    },
                ]
            ).to_csv(shortlist_path, index=False)
            config["walk_forward_settings"]["promotion_shortlist_path"] = shortlist_path.as_posix()
            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                fake_sim_module = types.ModuleType("src.parameter_simulation_gpu")
                fake_sim_module.find_optimal_parameters = _fake_find_optimal_parameters
                fake_gpu_module = types.ModuleType("src.debug_gpu_single_run")
                fake_gpu_module.run_single_backtest = _fake_run_single_backtest
                fake_perf_module = types.ModuleType("src.performance_analyzer")
                fake_perf_module.PerformanceAnalyzer = _FakePerformanceAnalyzer

                with patch("src.analysis.walk_forward_analyzer.load_config", return_value=config), \
                     patch("src.analysis.walk_forward_analyzer.plot_wfo_results", return_value=None), \
                     patch("src.analysis.walk_forward_analyzer.run_cpu_single_backtest", side_effect=_fake_cpu_run), \
                     patch.dict(
                         sys.modules,
                         {
                             "src.parameter_simulation_gpu": fake_sim_module,
                             "src.debug_gpu_single_run": fake_gpu_module,
                             "src.performance_analyzer": fake_perf_module,
                         },
                     ):
                    wfo.run_walk_forward_analysis()

                result_dirs = sorted(glob.glob(os.path.join("results", "wfo_run_*")))
                self.assertEqual(len(result_dirs), 1)
                result_dir = result_dirs[0]
                final_candidate_manifest = json.loads(
                    Path(result_dir, "final_candidate_manifest.json").read_text(encoding="utf-8")
                )
                holdout_manifest = json.loads(
                    Path(result_dir, "holdout_manifest.json").read_text(encoding="utf-8")
                )
                explanation_report = json.loads(
                    Path(result_dir, "promotion_explanation_report.json").read_text(
                        encoding="utf-8"
                    )
                )
                explanation_summary_md = Path(
                    result_dir, "promotion_explanation_summary.md"
                ).read_text(encoding="utf-8")
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(final_candidate_manifest["cpu_audit_outcome"], "pass")
        self.assertTrue(final_candidate_manifest["holdout_ready"])
        self.assertEqual(final_candidate_manifest["holdout_execution_status"], "executed")
        self.assertTrue(final_candidate_manifest["holdout_attempted"])
        self.assertTrue(final_candidate_manifest["holdout_success"])
        self.assertFalse(final_candidate_manifest["holdout_blocked"])
        self.assertTrue(holdout_manifest["holdout_backtest_executed"])
        self.assertTrue(holdout_manifest["holdout_backtest_attempted"])
        self.assertTrue(holdout_manifest["holdout_backtest_success"])
        self.assertFalse(holdout_manifest["holdout_backtest_blocked"])
        self.assertEqual(holdout_manifest["trade_count"], 3)
        self.assertEqual(holdout_manifest["closed_trade_count"], 1)
        self.assertEqual(holdout_manifest["distinct_entry_months"], 2)
        self.assertTrue(holdout_manifest["holdout_auto_execute"])
        self.assertTrue(holdout_manifest["holdout_candidate_hash_verified"])
        self.assertIn("holdout_too_short=334<730", holdout_manifest["reasons"])
        self.assertEqual(
            explanation_report["behavior_evidence"]["behavior_gate_status"],
            "passed",
        )
        self.assertFalse(explanation_report["behavior_evidence"]["approval_eligible"])
        self.assertGreaterEqual(
            len(explanation_report["behavior_evidence"]["threshold_checks"]),
            0,
        )
        self.assertIn("## Behavior Evidence", explanation_summary_md)

    def test_promotion_lane_auto_execute_blocks_when_no_candidate_passes_hard_gate(self):
        config = self._promotion_config()
        config["walk_forward_settings"]["holdout_auto_execute"] = True
        config["walk_forward_settings"]["selection_contract"] = {
            "min_promotion_fold_pass_rate": 1.1,
        }

        def _fake_find_optimal_parameters(*args, **kwargs):
            raise AssertionError("promotion lane should not call find_optimal_parameters")

        def _fake_run_single_backtest(*, start_date, end_date, params_dict, initial_cash):
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            duration_days = int((end - start).days + 1)
            if duration_days > 365:
                factor = 1.30 if int(params_dict["max_stocks"]) == 30 else 1.10
                return pd.Series(
                    [initial_cash, initial_cash * factor],
                    index=pd.to_datetime([start_date, end_date]),
                )
            factor = 1.20 if int(params_dict["max_stocks"]) == 30 else 1.05
            return pd.Series(
                [initial_cash, initial_cash * factor],
                index=pd.to_datetime([start_date, end_date]),
            )

        def _unexpected_cpu_run(*args, **kwargs):
            raise AssertionError("auto_execute should be blocked before CPU audit or holdout")

        with tempfile.TemporaryDirectory() as tmp_dir:
            shortlist_path = Path(tmp_dir, "promotion_shortlist.csv")
            pd.DataFrame(
                [
                    {
                        "max_stocks": 20,
                        "order_investment_ratio": 0.02,
                        "additional_buy_drop_rate": 0.04,
                        "sell_profit_rate": 0.05,
                        "additional_buy_priority": 0.0,
                        "stop_loss_rate": -0.15,
                        "max_splits_limit": 10,
                        "max_inactivity_period": 90,
                    },
                    {
                        "max_stocks": 30,
                        "order_investment_ratio": 0.03,
                        "additional_buy_drop_rate": 0.05,
                        "sell_profit_rate": 0.06,
                        "additional_buy_priority": 1.0,
                        "stop_loss_rate": -0.10,
                        "max_splits_limit": 15,
                        "max_inactivity_period": 60,
                    },
                ]
            ).to_csv(shortlist_path, index=False)
            config["walk_forward_settings"]["promotion_shortlist_path"] = shortlist_path.as_posix()
            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                fake_sim_module = types.ModuleType("src.parameter_simulation_gpu")
                fake_sim_module.find_optimal_parameters = _fake_find_optimal_parameters
                fake_gpu_module = types.ModuleType("src.debug_gpu_single_run")
                fake_gpu_module.run_single_backtest = _fake_run_single_backtest
                fake_perf_module = types.ModuleType("src.performance_analyzer")
                fake_perf_module.PerformanceAnalyzer = _FakePerformanceAnalyzer

                with patch("src.analysis.walk_forward_analyzer.load_config", return_value=config), \
                     patch("src.analysis.walk_forward_analyzer.plot_wfo_results", return_value=None), \
                     patch("src.analysis.walk_forward_analyzer.run_cpu_single_backtest", side_effect=_unexpected_cpu_run), \
                     patch.dict(
                         sys.modules,
                         {
                             "src.parameter_simulation_gpu": fake_sim_module,
                             "src.debug_gpu_single_run": fake_gpu_module,
                             "src.performance_analyzer": fake_perf_module,
                         },
                     ):
                    wfo.run_walk_forward_analysis()

                result_dirs = sorted(glob.glob(os.path.join("results", "wfo_run_*")))
                self.assertEqual(len(result_dirs), 1)
                result_dir = result_dirs[0]
                final_candidate_manifest = json.loads(
                    Path(result_dir, "final_candidate_manifest.json").read_text(encoding="utf-8")
                )
                holdout_manifest = json.loads(
                    Path(result_dir, "holdout_manifest.json").read_text(encoding="utf-8")
                )
                lane_manifest = json.loads(
                    Path(result_dir, "lane_manifest.json").read_text(encoding="utf-8")
                )
            finally:
                os.chdir(previous_cwd)

        self.assertFalse(final_candidate_manifest["champion_hard_gate_pass"])
        self.assertEqual(final_candidate_manifest["cpu_audit_outcome"], "blocked_preconditions")
        self.assertEqual(final_candidate_manifest["holdout_execution_status"], "blocked")
        self.assertFalse(final_candidate_manifest["holdout_attempted"])
        self.assertFalse(final_candidate_manifest["holdout_success"])
        self.assertTrue(final_candidate_manifest["holdout_blocked"])
        self.assertIn("no_candidate_passed_hard_gate", final_candidate_manifest["holdout_execution_reasons"])
        self.assertFalse(holdout_manifest["holdout_backtest_attempted"])
        self.assertFalse(holdout_manifest["holdout_backtest_success"])
        self.assertTrue(holdout_manifest["holdout_backtest_blocked"])
        self.assertIsNone(holdout_manifest["holdout_candidate_hash_verified"])
        self.assertIn("final_candidate_no_hard_gate_pass", lane_manifest["reasons"])

    def test_promotion_lane_requires_decision_date_for_freeze_contract(self):
        config = self._promotion_config()
        config["walk_forward_settings"]["decision_date"] = ""

        with tempfile.TemporaryDirectory() as tmp_dir:
            shortlist_path = Path(tmp_dir, "promotion_shortlist.csv")
            self._write_shortlist(shortlist_path)
            config["walk_forward_settings"]["promotion_shortlist_path"] = shortlist_path.as_posix()

            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                fake_sim_module = types.ModuleType("src.parameter_simulation_gpu")
                fake_sim_module.find_optimal_parameters = lambda *args, **kwargs: None
                fake_gpu_module = types.ModuleType("src.debug_gpu_single_run")
                fake_gpu_module.run_single_backtest = lambda *args, **kwargs: None
                fake_perf_module = types.ModuleType("src.performance_analyzer")
                fake_perf_module.PerformanceAnalyzer = _FakePerformanceAnalyzer

                with patch("src.analysis.walk_forward_analyzer.load_config", return_value=config), \
                     patch.dict(
                         sys.modules,
                         {
                             "src.parameter_simulation_gpu": fake_sim_module,
                             "src.debug_gpu_single_run": fake_gpu_module,
                             "src.performance_analyzer": fake_perf_module,
                         },
                     ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "promotion_evaluation requires walk_forward_settings.decision_date",
                    ):
                        wfo.run_walk_forward_analysis()
            finally:
                os.chdir(previous_cwd)

    def test_promotion_lane_auto_execute_blocks_when_shortlist_changes_after_decision_date(self):
        config = self._promotion_config()
        config["walk_forward_settings"]["holdout_auto_execute"] = True
        config["walk_forward_settings"]["decision_date"] = "2026-03-14"

        def _fake_find_optimal_parameters(*args, **kwargs):
            raise AssertionError("promotion lane should not call find_optimal_parameters")

        def _fake_run_single_backtest(*, start_date, end_date, params_dict, initial_cash):
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            duration_days = int((end - start).days + 1)
            factor = 1.30 if int(params_dict["max_stocks"]) == 30 else 1.10
            if duration_days <= 365:
                factor = 1.20 if int(params_dict["max_stocks"]) == 30 else 1.05
            return pd.Series(
                [initial_cash, initial_cash * factor],
                index=pd.to_datetime([start_date, end_date]),
            )

        def _unexpected_cpu_run(*args, **kwargs):
            raise AssertionError("freeze-contract mismatch should block CPU audit and holdout")

        with tempfile.TemporaryDirectory() as tmp_dir:
            shortlist_path = Path(tmp_dir, "promotion_shortlist.csv")
            self._write_shortlist(shortlist_path)
            future_ts = pd.Timestamp("2026-03-15T00:00:00Z").timestamp()
            os.utime(shortlist_path, (future_ts, future_ts))
            config["walk_forward_settings"]["promotion_shortlist_path"] = shortlist_path.as_posix()

            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                fake_sim_module = types.ModuleType("src.parameter_simulation_gpu")
                fake_sim_module.find_optimal_parameters = _fake_find_optimal_parameters
                fake_gpu_module = types.ModuleType("src.debug_gpu_single_run")
                fake_gpu_module.run_single_backtest = _fake_run_single_backtest
                fake_perf_module = types.ModuleType("src.performance_analyzer")
                fake_perf_module.PerformanceAnalyzer = _FakePerformanceAnalyzer

                with patch("src.analysis.walk_forward_analyzer.load_config", return_value=config), \
                     patch("src.analysis.walk_forward_analyzer.plot_wfo_results", return_value=None), \
                     patch("src.analysis.walk_forward_analyzer.run_cpu_single_backtest", side_effect=_unexpected_cpu_run), \
                     patch.dict(
                         sys.modules,
                         {
                             "src.parameter_simulation_gpu": fake_sim_module,
                             "src.debug_gpu_single_run": fake_gpu_module,
                             "src.performance_analyzer": fake_perf_module,
                         },
                     ):
                    wfo.run_walk_forward_analysis()

                result_dir = sorted(glob.glob(os.path.join("results", "wfo_run_*")))[0]
                final_candidate_manifest = json.loads(
                    Path(result_dir, "final_candidate_manifest.json").read_text(encoding="utf-8")
                )
                holdout_manifest = json.loads(
                    Path(result_dir, "holdout_manifest.json").read_text(encoding="utf-8")
                )
            finally:
                os.chdir(previous_cwd)

        self.assertFalse(final_candidate_manifest["freeze_contract_verified"])
        self.assertTrue(final_candidate_manifest["promotion_shortlist_modified_after_decision_date"])
        self.assertEqual(final_candidate_manifest["cpu_audit_outcome"], "blocked_preconditions")
        self.assertEqual(final_candidate_manifest["holdout_execution_status"], "blocked")
        self.assertFalse(final_candidate_manifest["holdout_attempted"])
        self.assertTrue(final_candidate_manifest["holdout_blocked"])
        self.assertIn(
            "promotion_shortlist_modified_after_decision_date",
            final_candidate_manifest["holdout_execution_reasons"],
        )
        self.assertFalse(holdout_manifest["holdout_backtest_attempted"])
        self.assertFalse(holdout_manifest["holdout_backtest_success"])
        self.assertTrue(holdout_manifest["holdout_backtest_blocked"])

    def test_promotion_lane_auto_execute_blocks_when_canonical_holdout_contract_mismatches(self):
        config = self._promotion_config()
        config["walk_forward_settings"]["holdout_auto_execute"] = True
        config["walk_forward_settings"]["canonical_holdout_start"] = "2025-02-01"

        def _fake_find_optimal_parameters(*args, **kwargs):
            raise AssertionError("promotion lane should not call find_optimal_parameters")

        def _fake_run_single_backtest(*, start_date, end_date, params_dict, initial_cash):
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            duration_days = int((end - start).days + 1)
            factor = 1.30 if int(params_dict["max_stocks"]) == 30 else 1.10
            if duration_days <= 365:
                factor = 1.20 if int(params_dict["max_stocks"]) == 30 else 1.05
            return pd.Series(
                [initial_cash, initial_cash * factor],
                index=pd.to_datetime([start_date, end_date]),
            )

        def _unexpected_cpu_run(*args, **kwargs):
            raise AssertionError("canonical holdout mismatch should block CPU audit and holdout")

        with tempfile.TemporaryDirectory() as tmp_dir:
            shortlist_path = Path(tmp_dir, "promotion_shortlist.csv")
            self._write_shortlist(shortlist_path)
            config["walk_forward_settings"]["promotion_shortlist_path"] = shortlist_path.as_posix()

            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                fake_sim_module = types.ModuleType("src.parameter_simulation_gpu")
                fake_sim_module.find_optimal_parameters = _fake_find_optimal_parameters
                fake_gpu_module = types.ModuleType("src.debug_gpu_single_run")
                fake_gpu_module.run_single_backtest = _fake_run_single_backtest
                fake_perf_module = types.ModuleType("src.performance_analyzer")
                fake_perf_module.PerformanceAnalyzer = _FakePerformanceAnalyzer

                with patch("src.analysis.walk_forward_analyzer.load_config", return_value=config), \
                     patch("src.analysis.walk_forward_analyzer.plot_wfo_results", return_value=None), \
                     patch("src.analysis.walk_forward_analyzer.run_cpu_single_backtest", side_effect=_unexpected_cpu_run), \
                     patch.dict(
                         sys.modules,
                         {
                             "src.parameter_simulation_gpu": fake_sim_module,
                             "src.debug_gpu_single_run": fake_gpu_module,
                             "src.performance_analyzer": fake_perf_module,
                         },
                     ):
                    wfo.run_walk_forward_analysis()

                result_dir = sorted(glob.glob(os.path.join("results", "wfo_run_*")))[0]
                final_candidate_manifest = json.loads(
                    Path(result_dir, "final_candidate_manifest.json").read_text(encoding="utf-8")
                )
                holdout_manifest = json.loads(
                    Path(result_dir, "holdout_manifest.json").read_text(encoding="utf-8")
                )
            finally:
                os.chdir(previous_cwd)

        self.assertFalse(final_candidate_manifest["canonical_holdout_contract_verified"])
        self.assertFalse(final_candidate_manifest["freeze_contract_verified"])
        self.assertEqual(final_candidate_manifest["cpu_audit_outcome"], "blocked_preconditions")
        self.assertEqual(final_candidate_manifest["holdout_execution_status"], "blocked")
        self.assertIn(
            "holdout_start_mismatch_canonical_contract",
            final_candidate_manifest["freeze_contract_reasons"],
        )
        self.assertIn(
            "holdout_start_mismatch_canonical_contract",
            final_candidate_manifest["holdout_execution_reasons"],
        )
        self.assertFalse(holdout_manifest["holdout_backtest_attempted"])
        self.assertTrue(holdout_manifest["holdout_backtest_blocked"])

    def test_research_lane_uses_frozen_shortlist_multi_anchor_without_composite_curve(self):
        config = self._promotion_config()
        config["backtest_settings"]["end_date"] = "2024-12-31"
        config["walk_forward_settings"].update(
            {
                "lane_type": "research_start_date_robustness",
                "research_mode": "frozen_shortlist_multi_anchor_eval",
                "research_anchor_start_dates": ["2020-01-01", "2021-01-01"],
                "anchor_set_id": "anchor_set_v1",
                "anchor_spacing_rule": "manual_explicit",
                "coverage_normalized": True,
                "cpu_certification_enabled": False,
            }
        )
        backtest_calls = []

        def _fake_find_optimal_parameters(*args, **kwargs):
            raise AssertionError("research lane should not call find_optimal_parameters")

        def _fake_run_single_backtest(*, start_date, end_date, params_dict, initial_cash):
            backtest_calls.append((start_date, end_date, int(params_dict["max_stocks"]), float(initial_cash)))
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            duration_days = int((end - start).days + 1)
            if duration_days > 365:
                if int(params_dict["max_stocks"]) == 30:
                    return pd.Series(
                        [initial_cash, initial_cash * 1.30],
                        index=pd.to_datetime([start_date, end_date]),
                    )
                return pd.Series(
                    [initial_cash, initial_cash * 1.10],
                    index=pd.to_datetime([start_date, end_date]),
                )
            if int(params_dict["max_stocks"]) == 30:
                return pd.Series(
                    [initial_cash, initial_cash * 1.08],
                    index=pd.to_datetime([start_date, end_date]),
                )
            return pd.Series(
                [initial_cash, initial_cash * 1.02],
                index=pd.to_datetime([start_date, end_date]),
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            shortlist_path = Path(tmp_dir, "research_shortlist.csv")
            pd.DataFrame(
                [
                    {
                        "max_stocks": 20,
                        "order_investment_ratio": 0.02,
                        "additional_buy_drop_rate": 0.04,
                        "sell_profit_rate": 0.05,
                        "additional_buy_priority": 0.0,
                        "stop_loss_rate": -0.15,
                        "max_splits_limit": 10,
                        "max_inactivity_period": 90,
                    },
                    {
                        "max_stocks": 30,
                        "order_investment_ratio": 0.03,
                        "additional_buy_drop_rate": 0.05,
                        "sell_profit_rate": 0.06,
                        "additional_buy_priority": 1.0,
                        "stop_loss_rate": -0.10,
                        "max_splits_limit": 15,
                        "max_inactivity_period": 60,
                    },
                ]
            ).to_csv(shortlist_path, index=False)
            config["walk_forward_settings"]["research_shortlist_path"] = shortlist_path.as_posix()

            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                fake_sim_module = types.ModuleType("src.parameter_simulation_gpu")
                fake_sim_module.find_optimal_parameters = _fake_find_optimal_parameters
                fake_gpu_module = types.ModuleType("src.debug_gpu_single_run")
                fake_gpu_module.run_single_backtest = _fake_run_single_backtest
                fake_perf_module = types.ModuleType("src.performance_analyzer")
                fake_perf_module.PerformanceAnalyzer = _FakePerformanceAnalyzer

                with patch("src.analysis.walk_forward_analyzer.load_config", return_value=config), \
                     patch("src.analysis.walk_forward_analyzer.plot_wfo_results", return_value=None), \
                     patch.dict(
                         sys.modules,
                         {
                             "src.parameter_simulation_gpu": fake_sim_module,
                             "src.debug_gpu_single_run": fake_gpu_module,
                             "src.performance_analyzer": fake_perf_module,
                         },
                     ):
                    wfo.run_walk_forward_analysis()

                result_dirs = sorted(glob.glob(os.path.join("results", "wfo_run_*")))
                self.assertEqual(len(result_dirs), 1)
                result_dir = result_dirs[0]
                lane_manifest = json.loads(
                    Path(result_dir, "lane_manifest.json").read_text(encoding="utf-8")
                )
                anchor_manifest = json.loads(
                    Path(result_dir, "anchor_manifest.json").read_text(encoding="utf-8")
                )
                metrics_df = pd.read_csv(Path(result_dir, "research_anchor_fold_metrics.csv"))
                equity_curve_exists = Path(result_dir, "wfo_equity_curve_data.csv").exists()
            finally:
                os.chdir(previous_cwd)

        self.assertFalse(equity_curve_exists)
        self.assertTrue(all(call[3] == 10_000_000.0 for call in backtest_calls))
        self.assertEqual(lane_manifest["lane_type"], "research_start_date_robustness")
        self.assertFalse(lane_manifest["approval_eligible"])
        self.assertFalse(lane_manifest["composite_curve_allowed"])
        self.assertIn("research_lane_distribution_only", lane_manifest["reasons"])
        self.assertEqual(anchor_manifest["anchor_set_id"], "anchor_set_v1")
        self.assertEqual(anchor_manifest["anchor_dates"], ["2020-01-01", "2021-01-01"])
        self.assertEqual(len(metrics_df), 4)
        self.assertEqual(sorted(metrics_df["selected_shortlist_candidate_id"].unique().tolist()), [2])


if __name__ == "__main__":
    unittest.main()
