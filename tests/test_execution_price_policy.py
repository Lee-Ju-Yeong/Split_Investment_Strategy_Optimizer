import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# CPU 고정 검사(requirements-review.txt)에서 돌므로 yaml이 필요한 main_backtest는 가져오지 않는다.
# 진입점 허용 목록·실행 기록 검사는 tests/test_main_backtest_execution_policy.py에 있다.
from src.backtest.cpu.execution import BasicExecutionHandler
from src.backtest.cpu.portfolio import Portfolio, Position
from src.backtest.cpu.strategy import MagicSplitStrategy
from src.execution_price_policy import (
    DEFAULT_SELL_PRICE_TICK_ROUNDING,
    adjust_price_down,
    adjust_price_up,
    adjust_sell_price,
    get_tick_size,
    normalize_sell_price_tick_rounding,
)

TICKER = "000001"
BUY_DATE = pd.Timestamp("2024-01-02")
SELL_DATE = pd.Timestamp("2024-01-03")
INITIAL_CASH = 2_000_000


def _as_f32(price):
    # 체결 함수는 시가를 float32로 바꾼 뒤 호가를 맞춘다.
    return float(np.float32(price))


class TestSellPriceTickRoundingKey(unittest.TestCase):
    def test_missing_key_keeps_legacy_up(self):
        self.assertEqual(DEFAULT_SELL_PRICE_TICK_ROUNDING, "up")
        self.assertEqual(normalize_sell_price_tick_rounding(None), "up")
        self.assertEqual(BasicExecutionHandler().sell_price_tick_rounding, "up")

    def test_accepts_supported_values_case_insensitively(self):
        self.assertEqual(normalize_sell_price_tick_rounding(" Down "), "down")
        self.assertEqual(normalize_sell_price_tick_rounding("UP"), "up")

    def test_rejects_unknown_values(self):
        for value in ("nearest", "", "none", 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_sell_price_tick_rounding(value)
        with self.assertRaises(ValueError):
            BasicExecutionHandler(sell_price_tick_rounding="floor")


class TestTickPriceFunctions(unittest.TestCase):
    def test_tick_band_boundaries(self):
        cases = [(1999.99, 1), (2000, 5), (4999.99, 5), (5000, 10), (19999, 10), (20000, 50),
                 (49999, 50), (50000, 100), (199999, 100), (200000, 500), (499999, 500), (500000, 1000)]
        for price, tick in cases:
            with self.subTest(price=price):
                self.assertEqual(get_tick_size(price), tick)

    def test_on_tick_prices_are_unchanged_by_both_directions(self):
        for price in (1, 1999, 2000, 4995, 5000, 19990, 20000, 49950, 50000, 199900, 200000, 499500, 500000, 1234000):
            with self.subTest(price=price):
                self.assertEqual(adjust_price_up(_as_f32(price)), price)
                self.assertEqual(adjust_price_down(_as_f32(price)), price)

    def test_off_tick_prices(self):
        # (기준가, 올림, 내림): 수정주가처럼 호가에 맞지 않는 시가
        cases = [
            (10000 * 0.98765, 9880, 9870),
            (52300 * 0.97531, 51100, 51000),
            (100.1, 101, 100),
            (1999.95, 2000, 1999),
            (2000.1, 2005, 2000),
            (4999.99, 5000, 4995),
            (199999.5, 200000, 199900),
            (500000.5, 501000, 500000),
        ]
        for price, up, down in cases:
            p = _as_f32(price)
            with self.subTest(price=p):
                self.assertEqual(adjust_price_up(p), up)
                self.assertEqual(adjust_price_down(p), down)
                self.assertEqual(adjust_sell_price(p, "up"), up)
                self.assertEqual(adjust_sell_price(p, "down"), down)

    def test_float_noise_does_not_move_a_whole_tick(self):
        self.assertEqual(adjust_price_up(9879.99999), 9880)
        self.assertEqual(adjust_price_down(9879.99999), 9880)
        self.assertEqual(adjust_price_up(9880.00001), 9880)
        self.assertEqual(adjust_price_down(9880.00001), 9880)

    def test_down_never_settles_a_positive_price_at_zero(self):
        for price in (0.0001, 0.4, 0.99):
            with self.subTest(price=price):
                self.assertEqual(adjust_price_down(price), 1)
        self.assertEqual(adjust_price_down(1.7), 1)

    def test_strategy_buy_estimate_uses_same_rounding(self):
        for price in (100.1, 1999.95, 9876.5, 51008.7, 500000.5):
            with self.subTest(price=price):
                self.assertEqual(MagicSplitStrategy._adjust_price_up(price), adjust_price_up(price))
                self.assertEqual(MagicSplitStrategy._get_tick_size(price), get_tick_size(price))


class _OneTickerData:
    def __init__(self, opens_by_date):
        self.opens_by_date = opens_by_date

    def get_ohlc_data_on_date(self, date, ticker, start, end):
        price = self.opens_by_date[pd.Timestamp(date)]
        return {"open_price": price, "high_price": price, "low_price": price, "close_price": price}

    def get_name_from_ticker(self, ticker):
        return "합성종목"


class TestExecutionHandlerRounding(unittest.TestCase):
    """매수 9,876.5 → 매도 기준가 10,371.3 한 번 왕복 (투자 1,000,000원, 현금 2,000,000원)."""

    BUY_OPEN = 9876.5
    SELL_OPEN = 10371.3

    def _run_round_trip(self, mode, reason):
        handler = BasicExecutionHandler(sell_price_tick_rounding=mode)
        data = _OneTickerData({BUY_DATE: self.BUY_OPEN, SELL_DATE: self.SELL_OPEN})
        portfolio = Portfolio(initial_cash=INITIAL_CASH, start_date="2024-01-02", end_date="2024-01-03")
        position = Position(self.BUY_OPEN, 0, 1, 0.05, 0.04)
        common = {"ticker": TICKER, "start_date": "2024-01-02", "end_date": "2024-01-03"}
        handler.execute_order(
            {**common, "type": "BUY", "date": BUY_DATE, "investment_amount": 1_000_000,
             "position": position, "reason_for_trade": "신규 진입", "trigger_price": self.BUY_OPEN},
            portfolio, data, current_day_idx=0,
        )
        handler.execute_order(
            {**common, "type": "SELL", "date": SELL_DATE, "quantity": position.quantity,
             "position": position, "reason_for_trade": reason, "trigger_price": self.SELL_OPEN},
            portfolio, data, current_day_idx=1,
        )
        buy_trade, sell_trade = portfolio.trade_history
        return portfolio, buy_trade, sell_trade

    def test_buy_is_rounded_up_in_both_modes(self):
        for mode in ("up", "down"):
            with self.subTest(mode=mode):
                _, buy_trade, _ = self._run_round_trip(mode, "수익 실현")
                self.assertEqual(buy_trade.buy_price, 9880)
                self.assertEqual(int(buy_trade.quantity), 101)

    def test_sell_price_and_cash_follow_the_mode(self):
        expected = {"up": (10380, 1046335), "down": (10370, 1045327)}
        for reason in ("수익 실현", "손절매 (평균가 기준)", "매매 미발생 기간 초과"):
            for mode, (sell_price, net_revenue) in expected.items():
                with self.subTest(reason=reason, mode=mode):
                    portfolio, buy_trade, sell_trade = self._run_round_trip(mode, reason)
                    self.assertEqual(sell_trade.sell_price, sell_price)
                    self.assertEqual(float(sell_trade.trade_value), net_revenue)
                    self.assertEqual(portfolio.positions.get(TICKER, []), [])
                    cash_after_buy = INITIAL_CASH - float(buy_trade.trade_value)
                    self.assertAlmostEqual(float(portfolio.cash), cash_after_buy + net_revenue, places=0)

    def test_modes_match_when_open_is_already_on_tick(self):
        self.BUY_OPEN, self.SELL_OPEN = 10000.0, 10400.0
        results = {mode: self._run_round_trip(mode, "수익 실현") for mode in ("up", "down")}
        self.assertEqual(results["up"][2].sell_price, 10400)
        self.assertEqual(results["down"][2].sell_price, 10400)
        self.assertEqual(float(results["up"][0].cash), float(results["down"][0].cash))


if __name__ == "__main__":
    unittest.main()
