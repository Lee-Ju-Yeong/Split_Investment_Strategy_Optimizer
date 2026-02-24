import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.cpu.strategy import MagicSplitStrategy


class _DummyPortfolio:
    def __init__(self):
        self.initial_cash = 10_000_000
        self.cash = float(self.initial_cash)
        self.positions = {}

    def get_total_value(self, _date, _data_handler):
        return float(self.initial_cash)


class _DummyDataHandler:
    def __init__(self, *, candidate_codes, rows_by_ticker):
        self._candidate_codes = list(candidate_codes)
        self._rows_by_ticker = dict(rows_by_ticker)

    def get_previous_trading_date(self, trading_dates, current_day_idx):
        if current_day_idx <= 0:
            return None
        return pd.to_datetime(trading_dates[current_day_idx - 1])

    def get_candidates_with_tier_fallback(self, _signal_date):
        return self._candidate_codes, "TIER_1"

    def get_stock_row_as_of(self, ticker, *_args, **_kwargs):
        return self._rows_by_ticker.get(ticker)

    def get_ohlc_data_on_date(self, _current_date, _ticker, *_args, **_kwargs):
        return pd.Series({"open_price": 1000.0})


class TestCpuCandidatePriority(unittest.TestCase):
    def _build_strategy(self):
        return MagicSplitStrategy(
            max_stocks=3,
            order_investment_ratio=0.1,
            additional_buy_drop_rate=0.05,
            sell_profit_rate=0.10,
            backtest_start_date="2024-01-01",
            backtest_end_date="2024-12-31",
            candidate_source_mode="tier",
            cooldown_period_days=5,
            buy_commission_rate=0.00015,
        )

    def test_new_entry_ranks_by_cheap_then_atr_then_market_cap(self):
        strategy = self._build_strategy()
        portfolio = _DummyPortfolio()
        handler = _DummyDataHandler(
            candidate_codes=["A", "B", "C"],
            rows_by_ticker={
                "A": pd.Series(
                    {
                        "atr_14_ratio": 0.30,
                        "close_price": 1000.0,
                        "market_cap": 800_000_000,
                        "cheap_score": 0.20,
                        "cheap_score_confidence": 1.0,
                    }
                ),
                "B": pd.Series(
                    {
                        "atr_14_ratio": 0.10,
                        "close_price": 1000.0,
                        "market_cap": 200_000_000,
                        "cheap_score": 0.90,
                        "cheap_score_confidence": 1.0,
                    }
                ),
                "C": pd.Series(
                    {
                        "atr_14_ratio": 0.20,
                        "close_price": 1000.0,
                        "market_cap": 600_000_000,
                        "cheap_score": 0.50,
                        "cheap_score_confidence": 1.0,
                    }
                ),
            },
        )

        signals = strategy.generate_new_entry_signals(
            current_date=pd.Timestamp("2024-01-02"),
            portfolio=portfolio,
            data_handler=handler,
            trading_dates=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
            current_day_idx=1,
        )

        self.assertEqual([signal["ticker"] for signal in signals], ["B", "C", "A"])

    def test_new_entry_uses_market_cap_then_ticker_when_cheap_and_atr_tie(self):
        strategy = self._build_strategy()
        portfolio = _DummyPortfolio()
        handler = _DummyDataHandler(
            candidate_codes=["A", "C", "B"],
            rows_by_ticker={
                "A": pd.Series(
                    {
                        "atr_14_ratio": 0.20,
                        "close_price": 1000.0,
                        "market_cap": 100_000_000,
                        "cheap_score": 0.70,
                        "cheap_score_confidence": 1.0,
                    }
                ),
                "B": pd.Series(
                    {
                        "atr_14_ratio": 0.20,
                        "close_price": 1000.0,
                        "market_cap": 300_000_000,
                        "cheap_score": 0.70,
                        "cheap_score_confidence": 1.0,
                    }
                ),
                "C": pd.Series(
                    {
                        "atr_14_ratio": 0.20,
                        "close_price": 1000.0,
                        "market_cap": 300_000_000,
                        "cheap_score": 0.70,
                        "cheap_score_confidence": 1.0,
                    }
                ),
            },
        )

        signals = strategy.generate_new_entry_signals(
            current_date=pd.Timestamp("2024-01-02"),
            portfolio=portfolio,
            data_handler=handler,
            trading_dates=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
            current_day_idx=1,
        )

        self.assertEqual([signal["ticker"] for signal in signals], ["B", "C", "A"])


if __name__ == "__main__":
    unittest.main()
