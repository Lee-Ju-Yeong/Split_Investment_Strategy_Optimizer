import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analysis import walk_forward_analyzer as wfo


class TestWfoCpuCertification(unittest.TestCase):
    def setUp(self):
        self.base_config = {
            "database": {"host": "127.0.0.1", "user": "root", "password": "pw", "database": "stocks"},
            "backtest_settings": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_cash": 10_000_000,
                "save_full_trade_history": False,
            },
            "strategy_params": {
                "price_basis": "adjusted",
                "adjusted_price_gate_start_date": "2013-11-20",
                "universe_mode": "strict_pit",
                "candidate_source_mode": "tier",
                "max_stocks": 20,
                "order_investment_ratio": 0.02,
                "additional_buy_drop_rate": 0.04,
                "sell_profit_rate": 0.04,
                "additional_buy_priority": "lowest_order",
                "stop_loss_rate": -0.15,
                "max_splits_limit": 10,
                "max_inactivity_period": 90,
                "candidate_lookup_error_policy": "raise",
            },
            "execution_params": {
                "buy_commission_rate": 0.00015,
                "sell_commission_rate": 0.00015,
                "sell_tax_rate": 0.0018,
            },
            "paths": {"results_dir": "results"},
        }

    def test_cpu_certification_settings_default_to_disabled_for_backwards_compat(self):
        settings = wfo._get_cpu_certification_settings({})

        self.assertFalse(settings["enabled"])
        self.assertEqual(settings["top_n"], 5)
        self.assertEqual(settings["metric"], "calmar_ratio")

    def test_build_gpu_finalist_shortlist_includes_robust_cluster_row(self):
        simulation_results = pd.DataFrame(
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
                    "cagr": 0.30,
                    "mdd": -0.20,
                    "calmar_ratio": 1.50,
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
                    "cagr": 0.25,
                    "mdd": -0.10,
                    "calmar_ratio": 1.10,
                },
            ]
        )

        shortlist = wfo.build_gpu_finalist_shortlist(
            simulation_results,
            robust_params_dict=simulation_results.iloc[1].to_dict(),
            top_n=1,
            metric="calmar_ratio",
        )

        self.assertEqual(len(shortlist), 2)
        self.assertEqual(shortlist.iloc[0]["selection_reason"], "gpu_top_n")
        self.assertEqual(shortlist.iloc[1]["selection_reason"], "robust_cluster")
        self.assertEqual(shortlist.iloc[1]["max_stocks"], 30)

    def test_build_gpu_finalist_shortlist_marks_robust_candidate_already_in_top_n(self):
        simulation_results = pd.DataFrame(
            [
                {
                    "max_stocks": 20,
                    "order_investment_ratio": 0.02,
                    "additional_buy_drop_rate": 0.04,
                    "sell_profit_rate": 0.05,
                    "additional_buy_priority": 1.0,
                    "stop_loss_rate": -0.15,
                    "max_splits_limit": 10,
                    "max_inactivity_period": 90,
                    "cagr": 0.30,
                    "mdd": -0.20,
                    "calmar_ratio": 1.50,
                }
            ]
        )

        shortlist = wfo.build_gpu_finalist_shortlist(
            simulation_results,
            robust_params_dict=simulation_results.iloc[0].to_dict(),
            top_n=1,
            metric="calmar_ratio",
        )

        self.assertEqual(len(shortlist), 1)
        self.assertEqual(shortlist.iloc[0]["selection_reason"], "gpu_top_n+robust_cluster")

    @patch("src.analysis.walk_forward_analyzer.run_cpu_single_backtest")
    def test_certify_gpu_finalists_with_cpu_promotes_best_cpu_metric(self, mock_run_cpu_single_backtest):
        finalists = pd.DataFrame(
            [
                {
                    "gpu_rank": 1,
                    "gpu_result_index": 10,
                    "selection_reason": "gpu_top_n",
                    "max_stocks": 20,
                    "order_investment_ratio": 0.02,
                    "additional_buy_drop_rate": 0.04,
                    "sell_profit_rate": 0.05,
                    "additional_buy_priority": 0.0,
                    "stop_loss_rate": -0.15,
                    "max_splits_limit": 10,
                    "max_inactivity_period": 90,
                    "cagr": 0.30,
                    "mdd": -0.20,
                    "calmar_ratio": 1.50,
                },
                {
                    "gpu_rank": 2,
                    "gpu_result_index": 11,
                    "selection_reason": "robust_cluster",
                    "max_stocks": 30,
                    "order_investment_ratio": 0.03,
                    "additional_buy_drop_rate": 0.05,
                    "sell_profit_rate": 0.06,
                    "additional_buy_priority": 1.0,
                    "stop_loss_rate": -0.10,
                    "max_splits_limit": 15,
                    "max_inactivity_period": 60,
                    "cagr": 0.28,
                    "mdd": -0.15,
                    "calmar_ratio": 1.20,
                },
            ]
        )

        def _fake_cpu_run(_config, *, start_date, end_date, params_dict, initial_cash):
            self.assertEqual(start_date, "2024-01-01")
            self.assertEqual(end_date, "2024-01-31")
            self.assertEqual(initial_cash, 10_000_000)
            max_stocks = params_dict["max_stocks"]
            cpu_metrics = {
                20: {"calmar_ratio": 1.40, "cagr": 0.22, "mdd": -0.18},
                30: {"calmar_ratio": 1.80, "cagr": 0.26, "mdd": -0.12},
            }[max_stocks]
            result = {
                "success": True,
                "metrics": cpu_metrics,
                "degraded_run": False,
                "promotion_blocked": False,
            }
            return pd.Series([10_000_000, 10_500_000]), result

        mock_run_cpu_single_backtest.side_effect = _fake_cpu_run

        selected_params, certification_df = wfo.certify_gpu_finalists_with_cpu(
            self.base_config,
            finalists,
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_cash=10_000_000,
            metric="calmar_ratio",
            top_n_requested=5,
        )

        self.assertEqual(selected_params["max_stocks"], 30)
        self.assertEqual(selected_params["additional_buy_priority"], "highest_drop")
        self.assertEqual(selected_params["selection_source"], "cpu_certified_finalist")
        self.assertEqual(selected_params["cpu_certification_top_n"], 5)
        self.assertEqual(selected_params["cpu_certification_shortlist_size"], 2)
        self.assertEqual(selected_params["cpu_certification_gpu_rank"], 2)
        self.assertAlmostEqual(selected_params["cpu_calmar_ratio"], 1.80)
        self.assertTrue(certification_df["cpu_certified"].all())

    @patch("src.analysis.walk_forward_analyzer.run_cpu_single_backtest")
    def test_certify_gpu_finalists_with_cpu_fails_when_any_candidate_cpu_run_fails(self, mock_run_cpu_single_backtest):
        finalists = pd.DataFrame(
            [
                {
                    "gpu_rank": 1,
                    "gpu_result_index": 10,
                    "selection_reason": "gpu_top_n",
                    "max_stocks": 20,
                    "order_investment_ratio": 0.02,
                    "additional_buy_drop_rate": 0.04,
                    "sell_profit_rate": 0.05,
                    "additional_buy_priority": 0.0,
                    "stop_loss_rate": -0.15,
                    "max_splits_limit": 10,
                    "max_inactivity_period": 90,
                    "cagr": 0.30,
                    "mdd": -0.20,
                    "calmar_ratio": 1.50,
                },
                {
                    "gpu_rank": 2,
                    "gpu_result_index": 11,
                    "selection_reason": "robust_cluster",
                    "max_stocks": 30,
                    "order_investment_ratio": 0.03,
                    "additional_buy_drop_rate": 0.05,
                    "sell_profit_rate": 0.06,
                    "additional_buy_priority": 1.0,
                    "stop_loss_rate": -0.10,
                    "max_splits_limit": 15,
                    "max_inactivity_period": 60,
                    "cagr": 0.28,
                    "mdd": -0.15,
                    "calmar_ratio": 1.20,
                },
            ]
        )

        def _fake_cpu_run(_config, *, start_date, end_date, params_dict, initial_cash):
            if params_dict["max_stocks"] == 20:
                raise ValueError("cpu runner failed")
            return pd.Series([10_000_000, 10_500_000]), {
                "success": True,
                "metrics": {"calmar_ratio": 1.80, "cagr": 0.26, "mdd": -0.12},
                "degraded_run": False,
                "promotion_blocked": False,
            }

        mock_run_cpu_single_backtest.side_effect = _fake_cpu_run

        with self.assertRaisesRegex(RuntimeError, "failed finalists"):
            wfo.certify_gpu_finalists_with_cpu(
                self.base_config,
                finalists,
                start_date="2024-01-01",
                end_date="2024-01-31",
                initial_cash=10_000_000,
                metric="calmar_ratio",
                top_n_requested=5,
            )


if __name__ == "__main__":
    unittest.main()
