import unittest
from unittest.mock import MagicMock

import pandas as pd

from src.backtest.cpu.strategy import MagicSplitStrategy, Position


class _SimplePortfolio:
    def __init__(self, initial_cash=1_000_000):
        self.initial_cash = initial_cash
        self.cash = float(initial_cash)
        self.positions = {}
        self.last_trade_day_indices = {}

    def get_total_value(self, *_args, **_kwargs):
        return self.initial_cash


class TestCpuStrategyCache(unittest.TestCase):
    def setUp(self):
        self.strategy = MagicSplitStrategy(
            max_stocks=5,
            order_investment_ratio=0.1,
            additional_buy_drop_rate=0.05,
            sell_profit_rate=0.05,
            backtest_start_date="2024-01-01",
            backtest_end_date="2024-01-31",
            candidate_source_mode="tier",
            tier_hysteresis_mode="strict_hysteresis_v1",
        )
        self.data_handler = MagicMock()
        self.data_handler.get_previous_trading_date.return_value = pd.Timestamp("2024-01-01")
        self.trading_dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
        self.current_date = self.trading_dates[1]

    def test_strict_hysteresis_reuses_tier_lookup_for_same_signal_date(self):
        portfolio = _SimplePortfolio()
        pos = Position(
            buy_price=100.0,
            quantity=10,
            order=1,
            additional_buy_drop_rate=self.strategy.additional_buy_drop_rate,
            sell_profit_rate=self.strategy.sell_profit_rate,
        )
        pos.open_date = self.trading_dates[0]
        portfolio.positions["TICK"] = [pos]
        portfolio.last_trade_day_indices["TICK"] = 0
        self.data_handler.get_tiers_as_of.return_value = {"TICK": {"tier": 1}}
        self.data_handler.get_stock_row_as_of.return_value = pd.Series(
            {"close_price": 95.0, "low_price": 94.0, "high_price": 96.0}
        )

        self.strategy.generate_additional_buy_signals(
            self.current_date,
            portfolio,
            self.data_handler,
            self.trading_dates,
            1,
        )
        self.strategy.generate_sell_signals(
            self.current_date,
            portfolio,
            self.data_handler,
            self.trading_dates,
            1,
        )

        self.data_handler.get_tiers_as_of.assert_called_once_with(
            as_of_date=pd.Timestamp("2024-01-01"),
            tickers=["TICK"],
        )

    def test_new_entry_signal_contains_cached_ohlc(self):
        portfolio = _SimplePortfolio(initial_cash=10_000_000)
        self.data_handler.get_candidates_with_tier_fallback_pit_gated.return_value = (["A"], "TIER_1")
        self.data_handler.get_stock_row_as_of.return_value = pd.Series(
            {"close_price": 100.0, "atr_14_ratio": 0.25, "market_cap": 5_000_000_000.0}
        )
        self.data_handler.get_ohlc_data_on_date.return_value = pd.Series(
            {"open_price": 101.0, "high_price": 102.0, "low_price": 99.0, "close_price": 101.5}
        )

        signals = self.strategy.generate_new_entry_signals(
            self.current_date,
            portfolio,
            self.data_handler,
            self.trading_dates,
            1,
        )

        self.assertEqual(len(signals), 1)
        self.assertIn("_cached_ohlc", signals[0])
        self.assertEqual(float(signals[0]["_cached_ohlc"]["open_price"]), 101.0)
        self.data_handler.get_ohlc_data_on_date.assert_called_once()


if __name__ == "__main__":
    unittest.main()
