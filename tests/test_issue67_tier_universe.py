import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_handler import DataHandler
from strategy import MagicSplitStrategy
from portfolio import Portfolio
from backtest_strategy_gpu import (
    _collect_candidate_atr_asof,
    _resolve_signal_date_for_gpu,
    _sort_candidates_by_atr_then_ticker,
)

class TestIssue67TierUniverse(unittest.TestCase):
    def setUp(self):
        self.db_config = {
            "host": "fake_host",
            "user": "fake_user",
            "password": "fake_password",
            "database": "fake_db",
        }
        self.pool_patcher = patch("mysql.connector.pooling.MySQLConnectionPool")
        self.mock_pool = self.pool_patcher.start()
        self.mock_conn = MagicMock()
        self.mock_pool.return_value.get_connection.return_value = self.mock_conn
        self.data_handler = DataHandler(self.db_config)
        self.data_handler.load_stock_data.cache_clear()

    def tearDown(self):
        self.pool_patcher.stop()

    @patch("pandas.read_sql")
    def test_get_candidates_with_tier_fallback_tier1_exists(self, mock_read_sql):
        # Tier 1 query returns results
        mock_read_sql.side_effect = [
            pd.DataFrame({"stock_code": ["000001", "000002"]}), # Tier 1
        ]
        
        candidates, source = self.data_handler.get_candidates_with_tier_fallback("2024-01-01")
        
        self.assertEqual(candidates, ["000001", "000002"])
        self.assertEqual(source, "TIER_1")
        self.assertEqual(mock_read_sql.call_count, 1) # Should only call Tier 1 query

    @patch("pandas.read_sql")
    def test_get_candidates_with_tier_fallback_tier1_empty_tier2_exists(self, mock_read_sql):
        # Tier 1 empty, Tier 2 returns results
        mock_read_sql.side_effect = [
            pd.DataFrame({"stock_code": []}), # Tier 1
            pd.DataFrame({"stock_code": ["000003", "000004"]}), # Tier 2
        ]
        
        candidates, source = self.data_handler.get_candidates_with_tier_fallback("2024-01-01")
        
        self.assertEqual(candidates, ["000003", "000004"])
        self.assertEqual(source, "TIER_2_FALLBACK")
        self.assertEqual(mock_read_sql.call_count, 2)

    @patch("pandas.read_sql")
    def test_get_candidates_with_tier_fallback_both_empty(self, mock_read_sql):
        # Both empty
        mock_read_sql.side_effect = [
            pd.DataFrame({"stock_code": []}), # Tier 1
            pd.DataFrame({"stock_code": []}), # Tier 2
        ]
        
        candidates, source = self.data_handler.get_candidates_with_tier_fallback("2024-01-01")
        
        self.assertEqual(candidates, [])
        self.assertEqual(source, "NO_CANDIDATES")
        self.assertEqual(mock_read_sql.call_count, 2)

class TestIssue67StrategyModes(unittest.TestCase):
    def setUp(self):
        self.data_handler = MagicMock()
        self.data_handler.get_previous_trading_date.return_value = pd.Timestamp("2024-01-01")
        self.portfolio = MagicMock()
        self.portfolio.positions = {}
        self.portfolio.initial_cash = 1000000
        self.portfolio.get_total_value.return_value = 1000000
        self.trading_dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
        
        # Base config
        self.base_config = {
            "max_stocks": 10,
            "order_investment_ratio": 0.1,
            "additional_buy_drop_rate": 0.05,
            "sell_profit_rate": 0.05,
            "backtest_start_date": "2024-01-01",
            "backtest_end_date": "2024-01-31",
        }

    def test_strategy_mode_weekly_is_forced_to_tier_path(self):
        strategy = MagicSplitStrategy(**self.base_config, candidate_source_mode="weekly")
        self.data_handler.get_candidates_with_tier_fallback.return_value = (["A", "B"], "TIER_1")
        self.data_handler.get_stock_row_as_of.return_value = None

        strategy.generate_new_entry_signals(
            pd.Timestamp("2024-01-02"),
            self.portfolio,
            self.data_handler,
            self.trading_dates,
            1,
        )

        self.data_handler.get_candidates_with_tier_fallback.assert_called_once_with(pd.Timestamp("2024-01-01"))
        self.data_handler.get_filtered_stock_codes.assert_not_called()

    def test_strategy_mode_tier(self):
        strategy = MagicSplitStrategy(**self.base_config, candidate_source_mode="tier")
        self.data_handler.get_candidates_with_tier_fallback.return_value = (["C", "D"], "TIER_1")
        
        # Mock get_stock_row_as_of to return None so we don't need to mock signal creation details
        self.data_handler.get_stock_row_as_of.return_value = None
        
        strategy.generate_new_entry_signals(pd.Timestamp("2024-01-02"), self.portfolio, self.data_handler, self.trading_dates, 1)
        
        self.data_handler.get_candidates_with_tier_fallback.assert_called_once_with(pd.Timestamp("2024-01-01"))
        self.data_handler.get_filtered_stock_codes.assert_not_called()

    def test_strategy_mode_tier_prefers_pit_candidate_api(self):
        strategy = MagicSplitStrategy(**self.base_config, candidate_source_mode="tier")
        self.data_handler.get_candidates_with_tier_fallback_pit.return_value = (
            ["P1", "P2"],
            "TIER_1_SNAPSHOT_ASOF",
        )
        self.data_handler.get_stock_row_as_of.return_value = None

        strategy.generate_new_entry_signals(
            pd.Timestamp("2024-01-02"),
            self.portfolio,
            self.data_handler,
            self.trading_dates,
            1,
        )

        self.data_handler.get_candidates_with_tier_fallback_pit.assert_called_once_with(
            pd.Timestamp("2024-01-01")
        )
        self.data_handler.get_candidates_with_tier_fallback.assert_not_called()

    def test_strategy_mode_hybrid_is_forced_to_tier_path(self):
        strategy = MagicSplitStrategy(
            **self.base_config,
            candidate_source_mode="hybrid_transition",
            use_weekly_alpha_gate=True,
        )
        self.data_handler.get_candidates_with_tier_fallback.return_value = (["A", "B"], "TIER_1")
        self.data_handler.get_stock_row_as_of.return_value = None

        strategy.generate_new_entry_signals(
            pd.Timestamp("2024-01-02"),
            self.portfolio,
            self.data_handler,
            self.trading_dates,
            1,
        )

        self.data_handler.get_candidates_with_tier_fallback.assert_called_once_with(pd.Timestamp("2024-01-01"))
        self.data_handler.get_filtered_stock_codes.assert_not_called()

    def test_strategy_mode_tier_returns_empty_on_tier_exception(self):
        strategy = MagicSplitStrategy(**self.base_config, candidate_source_mode="tier")
        self.data_handler.get_candidates_with_tier_fallback.side_effect = RuntimeError("tier query error")
        self.data_handler.get_stock_row_as_of.return_value = None

        signals = strategy.generate_new_entry_signals(
            pd.Timestamp("2024-01-02"),
            self.portfolio,
            self.data_handler,
            self.trading_dates,
            1,
        )

        self.data_handler.get_candidates_with_tier_fallback.assert_called_once_with(pd.Timestamp("2024-01-01"))
        self.assertEqual(signals, [])

class TestIssue67GpuParityHelpers(unittest.TestCase):
    def test_resolve_signal_date_for_gpu_uses_previous_trading_day(self):
        trading_dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        signal_date, signal_day_idx = _resolve_signal_date_for_gpu(2, trading_dates)
        self.assertEqual(signal_day_idx, 1)
        self.assertEqual(signal_date, pd.Timestamp("2024-01-02"))

    def test_resolve_signal_date_for_gpu_first_day_has_no_signal(self):
        trading_dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
        signal_date, signal_day_idx = _resolve_signal_date_for_gpu(0, trading_dates)
        self.assertIsNone(signal_date)
        self.assertEqual(signal_day_idx, -1)

    def test_sort_candidates_by_atr_then_ticker_is_deterministic(self):
        candidates = [
            ("005930", 0.10),
            ("000660", 0.20),
            ("373220", 0.20),
            ("035420", 0.05),
        ]
        ranked = _sort_candidates_by_atr_then_ticker(candidates)
        self.assertEqual(
            ranked,
            [
                ("000660", 0.20),
                ("373220", 0.20),
                ("005930", 0.10),
                ("035420", 0.05),
            ],
        )

    def test_collect_candidate_atr_asof_uses_previous_row_when_same_day_missing(self):
        try:
            import cudf
        except ImportError:
            self.skipTest("cudf is required for GPU helper test")

        all_data = pd.DataFrame(
            {
                "ticker": ["A", "A", "B"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-01"]),
                "atr_14_ratio": [0.11, 0.33, 0.22],
            }
        )
        all_data_reset_idx = cudf.from_pandas(all_data)
        result = _collect_candidate_atr_asof(
            all_data_reset_idx=all_data_reset_idx,
            final_candidate_tickers=["A", "B", "C"],
            signal_date=pd.Timestamp("2024-01-02"),
        )

        self.assertIsNotNone(result)
        result_map = result.to_pandas().to_dict()
        self.assertAlmostEqual(float(result_map["A"]), 0.11, places=6)
        self.assertAlmostEqual(float(result_map["B"]), 0.22, places=6)
        self.assertNotIn("C", result_map)

if __name__ == "__main__":
    unittest.main()
