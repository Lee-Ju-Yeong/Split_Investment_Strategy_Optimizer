import unittest
from unittest.mock import MagicMock

import pandas as pd

from src.backtest.cpu.strategy import MagicSplitStrategy


class _DummyPortfolio:
    def __init__(self):
        self.positions = {}
        self.initial_cash = 1_000_000.0
        self.cash = 1_000_000.0

    def get_total_value(self, *_args, **_kwargs):
        return self.initial_cash


class _HandlerWithoutTierCandidateApi:
    def __init__(self):
        self._signal_date = pd.Timestamp("2024-01-01")

    def get_previous_trading_date(self, *_args, **_kwargs):
        return self._signal_date


class TestCpuStrategyEntryContext(unittest.TestCase):
    def setUp(self):
        self.base_config = {
            "max_stocks": 10,
            "order_investment_ratio": 0.1,
            "additional_buy_drop_rate": 0.05,
            "sell_profit_rate": 0.05,
            "backtest_start_date": "2024-01-01",
            "backtest_end_date": "2024-01-31",
        }
        self.strategy = MagicSplitStrategy(**self.base_config, candidate_source_mode="tier")
        self.portfolio = _DummyPortfolio()
        self.handler = MagicMock()
        self.trading_dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]

    def test_sets_no_signal_date_context(self):
        self.handler.get_previous_trading_date.return_value = None

        signals = self.strategy.generate_new_entry_signals(
            pd.Timestamp("2024-01-01"),
            self.portfolio,
            self.handler,
            self.trading_dates,
            0,
        )

        self.assertEqual(signals, [])
        self.assertEqual(self.strategy.last_entry_context["tier_source"], "NO_SIGNAL_DATE")

    def test_sets_lookup_error_context(self):
        strategy = MagicSplitStrategy(
            **self.base_config,
            candidate_source_mode="tier",
            candidate_lookup_error_policy="skip",
        )
        self.handler.get_previous_trading_date.return_value = pd.Timestamp("2024-01-01")
        self.handler.get_candidates_with_tier_fallback_pit_gated.side_effect = RuntimeError("query failed")

        signals = strategy.generate_new_entry_signals(
            pd.Timestamp("2024-01-02"),
            self.portfolio,
            self.handler,
            self.trading_dates,
            1,
        )

        self.assertEqual(signals, [])
        self.assertEqual(strategy.last_entry_context["tier_source"], "CANDIDATE_LOOKUP_ERROR")
        self.assertEqual(strategy.last_entry_context["raw_candidate_count"], 0)

    def test_raises_on_lookup_error_when_policy_raise(self):
        self.handler.get_previous_trading_date.return_value = pd.Timestamp("2024-01-01")
        self.handler.get_candidates_with_tier_fallback_pit_gated.side_effect = RuntimeError("query failed")

        with self.assertRaises(RuntimeError):
            self.strategy.generate_new_entry_signals(
                pd.Timestamp("2024-01-02"),
                self.portfolio,
                self.handler,
                self.trading_dates,
                1,
            )

    def test_sets_source_missing_context(self):
        handler = _HandlerWithoutTierCandidateApi()

        signals = self.strategy.generate_new_entry_signals(
            pd.Timestamp("2024-01-02"),
            self.portfolio,
            handler,
            self.trading_dates,
            1,
        )

        self.assertEqual(signals, [])
        self.assertEqual(self.strategy.last_entry_context["tier_source"], "CANDIDATE_SOURCE_MISSING")
        self.assertEqual(self.strategy.last_entry_context["raw_candidate_count"], 0)

    def test_tracks_candidate_counts(self):
        self.handler.get_previous_trading_date.return_value = pd.Timestamp("2024-01-01")
        self.handler.get_candidates_with_tier_fallback_pit_gated.return_value = (
            ["005930"],
            "TIER_1_SNAPSHOT_ASOF",
        )
        self.handler.get_stock_row_as_of.return_value = pd.Series(
            {
                "atr_14_ratio": 0.12,
                "close_price": 70000.0,
                "market_cap": 400_000_000_000.0,
                "cheap_score": 0.8,
                "cheap_score_confidence": 0.9,
            }
        )
        self.handler.get_ohlc_data_on_date.return_value = pd.Series({"open_price": 70000.0})

        signals = self.strategy.generate_new_entry_signals(
            pd.Timestamp("2024-01-02"),
            self.portfolio,
            self.handler,
            self.trading_dates,
            1,
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(self.strategy.last_entry_context["tier_source"], "TIER_1_SNAPSHOT_ASOF")
        self.assertEqual(self.strategy.last_entry_context["raw_candidate_count"], 1)
        self.assertEqual(self.strategy.last_entry_context["active_candidate_count"], 1)
        self.assertEqual(self.strategy.last_entry_context["ranked_candidate_count"], 1)
        self.assertEqual(self.strategy.last_entry_context["selected_count"], 1)


if __name__ == "__main__":
    unittest.main()
