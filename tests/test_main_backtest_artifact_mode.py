import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.main_backtest import run_backtest_from_config
from src.data_handler import PitRuntimeError


class _FakePortfolio:
    def __init__(self):
        self.daily_snapshot_history = [
            {"date": "2024-01-02", "total_value": 10_000_000},
            {"date": "2024-01-03", "total_value": 10_500_000},
        ]
        self.trade_history = []
        self.positions = {}
        self.run_metrics = {}


class _FakeAnalyzer:
    def __init__(self, history_df):
        self.history_df = history_df

    def get_metrics(self, formatted=False):
        metrics = {
            "period_start": "2024-01-02",
            "period_end": "2024-01-03",
            "initial_value": 10_000_000,
            "final_value": 10_500_000,
            "final_cumulative_returns": 0.05,
            "cagr": 0.10,
            "annualized_volatility": 0.01,
            "mdd": -0.02,
            "sharpe_ratio": 1.0,
            "sortino_ratio": 1.0,
            "calmar_ratio": 5.0,
        }
        return metrics if not formatted else {}

    def plot_equity_curve(self, *args, **kwargs):
        raise AssertionError("plot_equity_curve should not run when persist_artifacts=False")


class TestMainBacktestArtifactMode(unittest.TestCase):
    @patch("src.main_backtest._write_run_manifest")
    @patch("src.performance_analyzer.PerformanceAnalyzer", new=_FakeAnalyzer)
    @patch("src.main_backtest.BacktestEngine")
    @patch("src.main_backtest.BasicExecutionHandler")
    @patch("src.main_backtest.Portfolio")
    @patch("src.main_backtest.MagicSplitStrategy")
    @patch("src.main_backtest.DataHandler")
    def test_run_backtest_from_config_skips_artifacts_when_disabled(
        self,
        mock_data_handler,
        mock_strategy,
        mock_portfolio,
        mock_execution_handler,
        mock_backtest_engine,
        mock_write_manifest,
    ):
        fake_portfolio = _FakePortfolio()
        mock_backtest_engine.return_value.run.return_value = fake_portfolio
        mock_strategy.return_value.get_candidate_lookup_error_summary = MagicMock(return_value={})
        mock_portfolio.return_value = fake_portfolio

        config = {
            "database": {"host": "127.0.0.1", "user": "root", "password": "pw", "database": "stocks"},
            "backtest_settings": {
                "start_date": "2024-01-02",
                "end_date": "2024-01-03",
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

        result = run_backtest_from_config(config, persist_artifacts=False)

        self.assertTrue(result["success"])
        self.assertIsNone(result["run_manifest_path"])
        self.assertIsNone(result["plot_file_path"])
        self.assertIsNone(result["trade_file_path"])
        self.assertEqual(len(result["daily_values"]), 2)
        mock_write_manifest.assert_not_called()
        mock_data_handler.assert_called_once()
        mock_strategy.assert_called_once()
        mock_execution_handler.assert_called_once()
        mock_backtest_engine.assert_called_once()

    @patch("src.performance_analyzer.PerformanceAnalyzer", new=_FakeAnalyzer)
    @patch("src.main_backtest.BacktestEngine")
    @patch("src.main_backtest.BasicExecutionHandler")
    @patch("src.main_backtest.Portfolio")
    @patch("src.main_backtest.MagicSplitStrategy")
    @patch("src.main_backtest.DataHandler")
    def test_run_backtest_from_config_returns_structured_pit_failure(
        self,
        mock_data_handler,
        mock_strategy,
        mock_portfolio,
        mock_execution_handler,
        mock_backtest_engine,
    ):
        mock_backtest_engine.return_value.run.side_effect = PitRuntimeError(
            "tier12_coverage_gate_failed",
            "Tier coverage gate failed on 2024-01-02",
            stage="tier12_coverage_gate",
            details={"date": "2024-01-02", "tier12_ratio": 0.44},
        )
        mock_strategy.return_value.get_candidate_lookup_error_summary = MagicMock(return_value={})
        mock_portfolio.return_value = _FakePortfolio()

        config = {
            "database": {"host": "127.0.0.1", "user": "root", "password": "pw", "database": "stocks"},
            "backtest_settings": {
                "start_date": "2024-01-02",
                "end_date": "2024-01-03",
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

        result = run_backtest_from_config(config, persist_artifacts=False)

        self.assertIn("error", result)
        self.assertEqual(result["error_type"], "PitRuntimeError")
        self.assertEqual(result["pit_failure"]["code"], "tier12_coverage_gate_failed")
        self.assertEqual(result["pit_failure"]["stage"], "tier12_coverage_gate")

    @patch("src.main_backtest._write_run_manifest", return_value="results/run_x/run_manifest.json")
    @patch("src.performance_analyzer.PerformanceAnalyzer", new=_FakeAnalyzer)
    @patch("src.main_backtest.BacktestEngine")
    @patch("src.main_backtest.BasicExecutionHandler")
    @patch("src.main_backtest.Portfolio")
    @patch("src.main_backtest.MagicSplitStrategy")
    @patch("src.main_backtest.DataHandler")
    def test_failure_persists_run_manifest_when_artifacts_enabled(
        self,
        mock_data_handler,
        mock_strategy,
        mock_portfolio,
        mock_execution_handler,
        mock_backtest_engine,
        mock_write_manifest,
    ):
        mock_backtest_engine.return_value.run.side_effect = PitRuntimeError(
            "tier12_coverage_gate_failed",
            "Tier coverage gate failed on 2024-01-02",
            stage="tier12_coverage_gate",
            details={"date": "2024-01-02", "tier12_ratio": 0.44},
        )
        mock_strategy.return_value.get_candidate_lookup_error_summary = MagicMock(return_value={})
        mock_portfolio.return_value = _FakePortfolio()

        config = {
            "database": {"host": "127.0.0.1", "user": "root", "password": "pw", "database": "stocks"},
            "backtest_settings": {
                "start_date": "2024-01-02",
                "end_date": "2024-01-03",
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

        result = run_backtest_from_config(config, persist_artifacts=True)

        self.assertEqual(result["run_manifest_path"], "results/run_x/run_manifest.json")
        mock_write_manifest.assert_called_once()
        self.assertEqual(mock_write_manifest.call_args.kwargs["status"], "failed")


if __name__ == "__main__":
    unittest.main()
