import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.cpu.strategy import MagicSplitStrategy
from src.backtest.cpu.portfolio import Position


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
        self.get_stock_row_as_of_call_count = 0
        self.get_stock_rows_as_of_call_count = 0
        self.get_tiers_as_of_call_count = 0
        self.last_stock_rows_as_of_tickers = None
        self.last_tiers_as_of_tickers = None
        self.last_tiers_as_of_allowed_tiers = None

    def get_previous_trading_date(self, trading_dates, current_day_idx):
        if current_day_idx <= 0:
            return None
        return pd.to_datetime(trading_dates[current_day_idx - 1])

    def get_candidates_with_tier_fallback_pit_gated(self, _signal_date, **_kwargs):
        return self._candidate_codes, "TIER_1"

    def get_stock_row_as_of(self, ticker, *_args, **_kwargs):
        self.get_stock_row_as_of_call_count += 1
        return self._rows_by_ticker.get(ticker)

    def get_stock_rows_as_of(self, tickers, *_args, **_kwargs):
        self.get_stock_rows_as_of_call_count += 1
        self.last_stock_rows_as_of_tickers = tuple(tickers)
        return {ticker: self._rows_by_ticker.get(ticker) for ticker in tickers}

    def get_stock_tier_as_of(self, _ticker, _signal_date):
        return {"tier": 2}

    def get_tiers_as_of(self, _signal_date, tickers=None, allowed_tiers=None):
        self.get_tiers_as_of_call_count += 1
        self.last_tiers_as_of_tickers = tuple(tickers or [])
        self.last_tiers_as_of_allowed_tiers = list(allowed_tiers) if allowed_tiers is not None else None
        if allowed_tiers == [1, 2]:
            return {ticker: {"tier": 2} for ticker in (tickers or [])}
        return {}

    def get_ohlc_data_on_date(self, _current_date, _ticker, *_args, **_kwargs):
        return pd.Series({"open_price": 1000.0})


class TestCpuCandidatePriority(unittest.TestCase):
    def _build_strategy(self, *, additional_buy_priority="lowest_order"):
        return MagicSplitStrategy(
            max_stocks=3,
            order_investment_ratio=0.1,
            additional_buy_drop_rate=0.05,
            sell_profit_rate=0.10,
            backtest_start_date="2024-01-01",
            backtest_end_date="2024-12-31",
            additional_buy_priority=additional_buy_priority,
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

    def test_invalid_additional_buy_priority_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported additional_buy_priority"):
            self._build_strategy(additional_buy_priority="momentum")

    def test_additional_buy_skips_nonpositive_signal_low(self):
        strategy = self._build_strategy(additional_buy_priority="highest_drop")
        portfolio = _DummyPortfolio()
        first_pos = Position(100.0, 10, 1, strategy.additional_buy_drop_rate, strategy.sell_profit_rate)
        first_pos.open_date = pd.Timestamp("2024-01-01")
        portfolio.positions["A"] = [first_pos]

        handler = _DummyDataHandler(
            candidate_codes=[],
            rows_by_ticker={"A": pd.Series({"close_price": 96.0, "low_price": 0.0})},
        )

        signals = strategy.generate_additional_buy_signals(
            current_date=pd.Timestamp("2024-01-02"),
            portfolio=portfolio,
            data_handler=handler,
            trading_dates=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
            current_day_idx=1,
        )

        self.assertEqual(signals, [])

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
        self.assertEqual(handler.get_stock_rows_as_of_call_count, 1)

    def test_additional_buy_batches_tier_and_signal_row_lookup(self):
        strategy = self._build_strategy(additional_buy_priority="highest_drop")
        portfolio = _DummyPortfolio()
        first_pos = Position(100.0, 10, 1, strategy.additional_buy_drop_rate, strategy.sell_profit_rate)
        first_pos.open_date = pd.Timestamp("2024-01-01")
        second_pos = Position(120.0, 5, 1, strategy.additional_buy_drop_rate, strategy.sell_profit_rate)
        second_pos.open_date = pd.Timestamp("2024-01-01")
        portfolio.positions["A"] = [first_pos]
        portfolio.positions["B"] = [second_pos]

        handler = _DummyDataHandler(
            candidate_codes=[],
            rows_by_ticker={
                "A": pd.Series({"close_price": 94.0, "low_price": 94.0}),
                "B": pd.Series({"close_price": 113.0, "low_price": 113.0}),
            },
        )

        signals = strategy.generate_additional_buy_signals(
            current_date=pd.Timestamp("2024-01-02"),
            portfolio=portfolio,
            data_handler=handler,
            trading_dates=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
            current_day_idx=1,
        )

        self.assertEqual([signal["ticker"] for signal in signals], ["A", "B"])
        self.assertEqual(handler.get_tiers_as_of_call_count, 1)
        self.assertEqual(handler.get_stock_rows_as_of_call_count, 1)
        self.assertEqual(handler.last_tiers_as_of_allowed_tiers, [1, 2])
        self.assertEqual(handler.last_stock_rows_as_of_tickers, ("A", "B"))
        self.assertEqual(handler.get_stock_row_as_of_call_count, 0)

    def test_additional_buy_batches_rows_only_after_cheap_filters(self):
        strategy = self._build_strategy(additional_buy_priority="highest_drop")
        strategy.max_splits_limit = 2
        portfolio = _DummyPortfolio()

        eligible = Position(100.0, 10, 1, strategy.additional_buy_drop_rate, strategy.sell_profit_rate)
        eligible.open_date = pd.Timestamp("2024-01-01")
        cooldown = Position(100.0, 10, 1, strategy.additional_buy_drop_rate, strategy.sell_profit_rate)
        cooldown.open_date = pd.Timestamp("2024-01-01")
        new_today = Position(100.0, 10, 1, strategy.additional_buy_drop_rate, strategy.sell_profit_rate)
        new_today.open_date = pd.Timestamp("2024-01-02")
        maxed1 = Position(100.0, 10, 1, strategy.additional_buy_drop_rate, strategy.sell_profit_rate)
        maxed1.open_date = pd.Timestamp("2024-01-01")
        maxed2 = Position(95.0, 10, 2, strategy.additional_buy_drop_rate, strategy.sell_profit_rate)
        maxed2.open_date = pd.Timestamp("2024-01-01")

        portfolio.positions["A"] = [eligible]
        portfolio.positions["B"] = [cooldown]
        portfolio.positions["C"] = [new_today]
        portfolio.positions["D"] = [maxed1, maxed2]
        strategy.cooldown_tracker["B"] = 1

        handler = _DummyDataHandler(
            candidate_codes=[],
            rows_by_ticker={
                "A": pd.Series({"close_price": 94.0, "low_price": 94.0}),
                "B": pd.Series({"close_price": 94.0, "low_price": 94.0}),
                "C": pd.Series({"close_price": 94.0, "low_price": 94.0}),
                "D": pd.Series({"close_price": 94.0, "low_price": 94.0}),
            },
        )

        signals = strategy.generate_additional_buy_signals(
            current_date=pd.Timestamp("2024-01-02"),
            portfolio=portfolio,
            data_handler=handler,
            trading_dates=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
            current_day_idx=1,
        )

        self.assertEqual([signal["ticker"] for signal in signals], ["A"])
        self.assertEqual(handler.get_stock_rows_as_of_call_count, 1)
        self.assertEqual(handler.last_stock_rows_as_of_tickers, ("A",))
        self.assertEqual(handler.last_tiers_as_of_allowed_tiers, [1, 2])
        self.assertEqual(handler.get_stock_row_as_of_call_count, 0)


if __name__ == "__main__":
    unittest.main()
