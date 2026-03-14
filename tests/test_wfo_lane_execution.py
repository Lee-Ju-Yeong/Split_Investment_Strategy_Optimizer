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
                "decision_date": "2026-03-14",
                "promotion_data_cutoff": "2023-12-31",
                "holdout_start": "2025-01-01",
                "holdout_end": "2025-11-30",
            },
        }

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
        self.assertEqual(lane_manifest["cpu_audit_outcome"], "disabled")
        self.assertIn("cpu_audit_required_for_promotion", lane_manifest["reasons"])
        self.assertTrue(lane_manifest["composite_curve_allowed"])
        self.assertEqual(
            selection_audit_df["selected_shortlist_candidate_id"].tolist(),
            [2, 2],
        )
        self.assertEqual(len(candidate_fold_df), 4)
        self.assertEqual(candidate_summary_df["shortlist_candidate_id"].tolist(), [2, 1])
        self.assertEqual(final_candidate_manifest["champion_candidate_id"], 2)
        self.assertEqual(final_candidate_manifest["reserve_candidate_ids"], [1])
        self.assertFalse(final_candidate_manifest["holdout_ready"])
        self.assertIn(
            "final_candidate_cpu_audit_not_executed",
            final_candidate_manifest["holdout_readiness_reasons"],
        )
        self.assertFalse(holdout_manifest["approval_eligible"])
        self.assertTrue(holdout_manifest["promotion_wfo_end_before_holdout"])
        self.assertEqual(curve.index.strftime("%Y-%m-%d").tolist(), [
            "2022-01-01",
            "2022-12-31",
            "2023-01-01",
            "2023-12-31",
        ])
        self.assertAlmostEqual(float(curve.iloc[-1]), 14_400_000.0)

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
