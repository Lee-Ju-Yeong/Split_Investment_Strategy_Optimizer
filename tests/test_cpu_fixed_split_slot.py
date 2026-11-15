"""CPU 분할 차수 번호 규칙(`legacy` 고정, `fixed_slot_v1` 명세)을 실제 엔진으로 검사한다.

명세: todos/2026_11_15-fixed-split-slot-v1.md. 신호는 전일(T-1) 저가·고가, 체결은
당일(T0) 시가의 호가 올림이다. 주문 금액은 1월 첫날 자산 1,000,000원의 10%다.
기본값은 legacy이고, fixed_slot_v1은 설정으로 켠다(현재 CPU 전용).
"""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest

import pandas as pd

from examples.cpu_split_demo import (
    EXECUTION_SETTINGS, INDICATORS, PRICE_FIELDS, STRATEGY_SETTINGS, TICKER,
    TRADING_DATES, InMemoryDataHandler,
)
from src.backtest.cpu.backtester import BacktestEngine
from src.backtest.cpu.execution import BasicExecutionHandler
from src.backtest.cpu.portfolio import Portfolio
from src.backtest.cpu.strategy import MagicSplitStrategy
from src.gpu_execution_policy import build_gpu_execution_params
from src.split_slot_policy import require_gpu_supported_split_slot_policy


D = TRADING_DATES
MAX_SPLITS = 3
# 1·2·3차 보유 → 2차 익절 → 다시 추가 매수. legacy는 [1, 3, 3]이 된다.
INTERIOR_GAP = ((10000, 10000, 10000, 10000), (10000, 10000, 8500, 9000),
                (9000, 9000, 8500, 9000), (9500, 10000, 9500, 10000),
                (10000, 10000, 8000, 8000), (8000, 8100, 7900, 8000),
                (8000, 8100, 7900, 8000))
# 1·2차 보유 → 1차만 먼저 익절. legacy는 [2]에서 추가 매수가 멈춘다.
ORPHANED_FIRST = ((10000, 10000, 10000, 10000), (10000, 10000, 9400, 10000),
                  (10400, 11000, 10400, 11000), (11000, 11000, 9000, 9000),
                  (9000, 9000, 8800, 8900), (8900, 8900, 8700, 8800),
                  (8800, 9000, 8500, 8800))
# 중간 빈칸 기준가 확인용: 5일째(D[4]) 저가만 9400으로 바꾼다.
INTERIOR_GAP_ANCHOR = INTERIOR_GAP[:4] + ((10000, 10000, 9400, 9500),) + INTERIOR_GAP[5:]
# 1·2차를 함께 익절하고 3차만 남긴다. 3차는 D[3] 시가 12000에 비싸게 산다.
MISSING_ANCHOR = ((10000, 10000, 10000, 10000), (10000, 10000, 9400, 9500),
                  (9000, 9000, 8500, 8600), (12000, 12000, 12000, 12000),
                  (11000, 11000, 8000, 8000), (8000, 8100, 7900, 8000),
                  (8000, 8100, 7900, 8000))
ALL_TIER1 = (1,) * 7
# 신규 진입(1차 재진입 포함)은 신호일 등급 1만, 추가 매수는 등급 1~2를 허용한다.
TIER_BLOCK_REENTRY_FROM_D2 = (1, 1, 2, 2, 2, 2, 2)
TIER_REENTRY_ON_D3 = (1, 1, 2, 1, 1, 1, 1)
TIER_BLOCK_REENTRY_FROM_D3 = (1, 1, 1, 2, 2, 2, 2)


class SlotRecordingPortfolio(Portfolio):
    """일별 평가 시점마다 종목별 보유 차수 번호를 함께 남긴다."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daily_orders = []

    def record_daily_snapshot(self, date, total_value):
        super().record_daily_snapshot(date, total_value)
        self.daily_orders.append({ticker: sorted(int(p.order) for p in positions)
                                  for ticker, positions in self.positions.items()})


def build_strategy(policy=None, **overrides):
    """policy를 줄 때만 split_slot_policy를 넘긴다. 없으면 현재 기본값을 쓴다."""
    settings = dict(STRATEGY_SETTINGS, max_splits_limit=MAX_SPLITS, stop_loss_rate=-0.99,
                    cooldown_period_days=0, **overrides)
    if policy is not None:
        settings["split_slot_policy"] = policy
    return MagicSplitStrategy(**settings)


def run_engine(ohlc, strategy, tiers=ALL_TIER1):
    prices = [dict(date=date, **dict(zip(PRICE_FIELDS, values)), **INDICATORS)
              for date, values in zip(D, ohlc)]
    tier_rows = [dict(date=date, tier=tier) for date, tier in zip(D, tiers)]
    portfolio = SlotRecordingPortfolio(1_000_000, D[0], D[-1])
    engine = BacktestEngine(D[0], D[-1], portfolio, strategy,
                            InMemoryDataHandler(prices, tier_rows),
                            BasicExecutionHandler(**EXECUTION_SETTINGS))
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        engine.run()
    return portfolio


class MultiTickerHandler:
    """여러 가상 종목의 가격·등급만 조회한다. 신규 후보 = 신호일 등급 1인 종목(코드 순)."""

    def __init__(self, ohlc_by_ticker, tiers_by_ticker):
        index = pd.DatetimeIndex(pd.to_datetime(D))
        self.prices = {
            ticker: pd.DataFrame([dict(zip(PRICE_FIELDS, values), **INDICATORS) for values in ohlc],
                                 index=index)
            for ticker, ohlc in ohlc_by_ticker.items()
        }
        self.tiers = {ticker: pd.Series(tiers, index=index) for ticker, tiers in tiers_by_ticker.items()}

    def get_trading_dates(self, start, end):
        return next(iter(self.prices.values())).loc[pd.Timestamp(start):pd.Timestamp(end)].index

    @staticmethod
    def get_previous_trading_date(trading_dates, current_day_idx):
        return None if current_day_idx <= 0 else pd.Timestamp(trading_dates[current_day_idx - 1])

    def get_candidates_with_tier_fallback_pit_gated(self, signal_date, **_):
        codes = sorted(t for t in self.tiers if self.get_stock_tier_as_of(t, signal_date)["tier"] == 1)
        return codes, "TIER_1_SYNTHETIC_ASOF"

    def get_stock_tier_as_of(self, ticker, signal_date):
        series = self.tiers[ticker].loc[:pd.Timestamp(signal_date)]
        return None if series.empty else {"tier": int(series.iloc[-1])}

    def get_stock_row_as_of(self, ticker, date, start, end):
        frame = self.prices[ticker].loc[:pd.Timestamp(date)]
        return None if frame.empty else frame.iloc[-1].copy()

    def get_ohlc_data_on_date(self, date, ticker, start, end):
        return self.prices[ticker].loc[pd.Timestamp(date)].copy()

    def get_latest_price(self, date, ticker, start, end):
        return self.get_stock_row_as_of(ticker, date, start, end)["close_price"]

    def get_name_from_ticker(self, ticker):
        return ticker


def run_multi(ohlc_by_ticker, tiers_by_ticker, strategy):
    portfolio = SlotRecordingPortfolio(1_000_000, D[0], D[-1])
    engine = BacktestEngine(D[0], D[-1], portfolio, strategy,
                            MultiTickerHandler(ohlc_by_ticker, tiers_by_ticker),
                            BasicExecutionHandler(**EXECUTION_SETTINGS))
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        engine.run()
    return portfolio


def trade_rows(portfolio):
    return [(t.date.strftime("%Y-%m-%d"), t.trade_type, int(t.order),
             int(t.buy_price if t.trade_type == "BUY" else t.sell_price))
            for t in portfolio.trade_history]


def final_orders(portfolio):
    return portfolio.daily_orders[-1].get(TICKER, [])


class LegacySplitSlotTests(unittest.TestCase):
    """현행 동작 고정. 기본값과 명시적 legacy가 같아야 한다(구현 전에는 명시적 쪽 건너뜀)."""

    def _explicit_legacy(self):
        try:
            return build_strategy("legacy")
        except TypeError as exc:
            self.skipTest(f"split_slot_policy 미구현: {exc}")

    def _assert_interior_gap(self, strategy):
        portfolio = run_engine(INTERIOR_GAP, strategy)
        # 마지막 BUY는 len(positions)+1=3으로 기록되고 기준가는 3차(9500)다.
        self.assertEqual(trade_rows(portfolio), [
            (D[1], "BUY", 1, 10000), (D[2], "BUY", 2, 9000), (D[3], "BUY", 3, 9500),
            (D[4], "SELL", 2, 10000), (D[5], "BUY", 3, 8000),
        ])
        self.assertEqual(final_orders(portfolio), [1, 3, 3])

    def _assert_orphaned_first(self, strategy):
        portfolio = run_engine(ORPHANED_FIRST, strategy)
        # 1차가 사라진 뒤 D[3]~D[5] 저가가 2차 기준 9880 이하여도 추가 매수하지 않는다.
        self.assertEqual(trade_rows(portfolio), [
            (D[1], "BUY", 1, 10000), (D[2], "BUY", 2, 10400), (D[3], "SELL", 1, 11000),
        ])
        self.assertEqual(final_orders(portfolio), [2])

    def test_interior_gap_duplicates_order_by_default(self):
        self._assert_interior_gap(build_strategy())

    def test_interior_gap_duplicates_order_with_explicit_legacy(self):
        self._assert_interior_gap(self._explicit_legacy())

    def test_orphaned_first_level_stops_adds_by_default(self):
        self._assert_orphaned_first(build_strategy())

    def test_orphaned_first_level_stops_adds_with_explicit_legacy(self):
        self._assert_orphaned_first(self._explicit_legacy())


class FixedSlotV1Tests(unittest.TestCase):
    """fixed_slot_v1 명세 규칙. 기대값은 명세와 현행 가격·체결 규칙으로 손으로 계산했다."""

    def run_fixed(self, ohlc, tiers=ALL_TIER1):
        portfolio = run_engine(ohlc, build_strategy("fixed_slot_v1"), tiers)
        for day, held in zip(D, portfolio.daily_orders):
            orders = held.get(TICKER, [])
            # 규칙 1: 모든 일별 시점에서 번호는 1..N이고 겹치지 않는다.
            self.assertEqual(len(orders), len(set(orders)), f"{day} 중복 차수 {orders}")
            self.assertTrue(all(1 <= o <= MAX_SPLITS for o in orders), f"{day} 범위 밖 {orders}")
        return portfolio

    def buy_trigger(self, portfolio, day, order):
        trade, = [t for t in portfolio.trade_history if t.trade_type == "BUY"
                  and t.date.strftime("%Y-%m-%d") == day and int(t.order) == order]
        return float(trade.trigger_price)

    def test_interior_gap_refills_order_two_and_respects_same_day_sell(self):
        portfolio = self.run_fixed(INTERIOR_GAP)
        # D1 신규 1차 시가 10000. D2 기준 1차 10000×0.95=9500 ≥ D1 저가 8500 → 시가 9000.
        # D3 기준 2차 9000×0.95=8550 ≥ D2 저가 8500 → 시가 9500.
        # D4 D3 고가 10000 ≥ 2차 9900 → 2차 익절(시가 10000). 빈칸 2의 트리거(1차 10000×0.95)
        #    9500 ≥ D3 저가 9500이지만 같은 날 매도가 있어 추가 매수 없음(규칙 7).
        # D5 빈칸 2, 트리거 9500 ≥ D4 저가 8000 → 2차 시가 8000.
        # D6 1..3 모두 보유 → D5 저가 7900이어도 4차 없음(최대 차수).
        self.assertEqual(trade_rows(portfolio), [
            (D[1], "BUY", 1, 10000), (D[2], "BUY", 2, 9000), (D[3], "BUY", 3, 9500),
            (D[4], "SELL", 2, 10000), (D[5], "BUY", 2, 8000),
        ])
        self.assertEqual(final_orders(portfolio), [1, 2, 3])

    def test_interior_gap_anchor_is_previous_order_price(self):
        portfolio = self.run_fixed(INTERIOR_GAP_ANCHOR)
        # D4까지 위와 같음. D5 빈칸 2의 기준은 1차 10000 → 트리거 9500 ≥ D4 저가 9400 → 매수.
        # (3차 9500 기준이면 9025 < 9400이라 사지 않았을 것.) D6 최대 차수라 없음.
        self.assertEqual(trade_rows(portfolio), [
            (D[1], "BUY", 1, 10000), (D[2], "BUY", 2, 9000), (D[3], "BUY", 3, 9500),
            (D[4], "SELL", 2, 10000), (D[5], "BUY", 2, 8000),
        ])
        self.assertAlmostEqual(self.buy_trigger(portfolio, D[5], 2), 9500, delta=0.5)
        self.assertEqual(final_orders(portfolio), [1, 2, 3])

    def test_missing_previous_order_blocks_add(self):
        portfolio = self.run_fixed(MISSING_ANCHOR, TIER_BLOCK_REENTRY_FROM_D3)
        # D1 1차 10000. D2 1차 기준 9500 ≥ D1 저가 9400 → 2차 9000.
        # D3 2차 기준 8550 ≥ D2 저가 8500 → 3차 시가 12000.
        # D4 D3 고가 12000 ≥ 2차 9900·1차 11000 → 2차·1차 익절(시가 11000), 3차 13200은 미달.
        # D5·D6 빈칸 k=2인데 1차가 없어 매수 없음(규칙 3). 3차 기준이면 11400 ≥ 저가 8000이었다.
        # D3부터 등급 2라 1차 재진입은 막는다.
        self.assertEqual(trade_rows(portfolio), [
            (D[1], "BUY", 1, 10000), (D[2], "BUY", 2, 9000), (D[3], "BUY", 3, 12000),
            (D[4], "SELL", 2, 11000), (D[4], "SELL", 1, 11000),
        ])
        self.assertEqual(final_orders(portfolio), [3])

    def test_adds_continue_after_first_order_sold(self):
        portfolio = self.run_fixed(ORPHANED_FIRST, TIER_BLOCK_REENTRY_FROM_D2)
        # D1 1차 10000. D2 1차 기준 9500 ≥ D1 저가 9400 → 2차 시가 10400.
        # D3 D2 고가 11000 ≥ 1차 11000 → 1차 익절(시가 11000). D2부터 등급 2라 재진입 없음.
        # D4 빈칸 2..3 중 가장 낮은 3, 기준 2차 10400×0.95=9880 ≥ D3 저가 9000 → 3차 시가 9000.
        # D5·D6 2..3이 차 있어 추가 매수 없음(1차는 추가 매수로 채우지 않음).
        self.assertEqual(trade_rows(portfolio), [
            (D[1], "BUY", 1, 10000), (D[2], "BUY", 2, 10400), (D[3], "SELL", 1, 11000),
            (D[4], "BUY", 3, 9000),
        ])
        self.assertAlmostEqual(self.buy_trigger(portfolio, D[4], 3), 9880, delta=0.5)
        self.assertEqual(final_orders(portfolio), [2, 3])

    def test_first_order_reentry_without_using_max_stocks_slot(self):
        portfolio = self.run_fixed(ORPHANED_FIRST, TIER_REENTRY_ON_D3)
        self.assertEqual(STRATEGY_SETTINGS["max_stocks"], 1)  # 이 종목이 이미 자리 1개를 쓴다.
        # D1 1차 10000, D2 2차 10400, D3 1차 익절 11000(D2 등급 2라 같은 날 재진입 없음).
        # D4 D3 등급 1 → 1차 재진입 시가 9000(규칙 6). 같은 날 매수라 3차 추가 매수 없음
        #    (규칙 7. 없었다면 기준 2차 9880 ≥ D3 저가 9000).
        # D5 빈칸 3, 기준 2차 9880 ≥ D4 저가 8800 → 3차 시가 8900.
        #    (최근 매수 1차 9000 기준이면 8550 < 8800이라 사지 않았을 것.)
        # D6 1..3 모두 보유 → 없음.
        self.assertEqual(trade_rows(portfolio), [
            (D[1], "BUY", 1, 10000), (D[2], "BUY", 2, 10400), (D[3], "SELL", 1, 11000),
            (D[4], "BUY", 1, 9000), (D[5], "BUY", 3, 8900),
        ])
        self.assertAlmostEqual(self.buy_trigger(portfolio, D[5], 3), 9880, delta=0.5)
        self.assertEqual(final_orders(portfolio), [1, 2, 3])

    def test_no_reentry_on_the_day_first_order_sold(self):
        portfolio = self.run_fixed(ORPHANED_FIRST, ALL_TIER1)
        # D3 1차 익절 뒤 D2 등급 1이어도 같은 날 재진입 없음(규칙 7, 대기 기간 0이어도).
        # 이후는 재진입 사례와 같다: D4 1차 재진입 9000, D5 3차 8900(기준 2차 9880).
        self.assertEqual(trade_rows(portfolio), [
            (D[1], "BUY", 1, 10000), (D[2], "BUY", 2, 10400), (D[3], "SELL", 1, 11000),
            (D[4], "BUY", 1, 9000), (D[5], "BUY", 3, 8900),
        ])
        self.assertEqual(final_orders(portfolio), [1, 2, 3])

    def test_no_duplicate_orders_at_any_daily_snapshot(self):
        for name, ohlc, tiers in (
            ("interior_gap", INTERIOR_GAP, ALL_TIER1),
            ("orphaned_first_all_tier1", ORPHANED_FIRST, ALL_TIER1),
            ("missing_anchor", MISSING_ANCHOR, TIER_BLOCK_REENTRY_FROM_D3),
            ("reentry", ORPHANED_FIRST, TIER_REENTRY_ON_D3),
        ):
            with self.subTest(name=name):
                portfolio = self.run_fixed(ohlc, tiers)
                self.assertEqual(len(portfolio.daily_orders), len(D))

    def test_reentry_does_not_open_a_slot_for_new_tickers(self):
        flat = ((10000, 10000, 10000, 10000),) * len(D)
        portfolio = run_multi({"A001": ORPHANED_FIRST, "B001": flat},
                              {"A001": TIER_REENTRY_ON_D3, "B001": ALL_TIER1},
                              build_strategy("fixed_slot_v1"))
        # 지표가 같아 순위는 코드 순 → D1에 A001이 하나뿐인 종목 자리(max_stocks=1)를 쓴다.
        # D4 A001 1차 재진입은 자리를 새로 쓰지 않으므로, 같은 날 B001은 여전히 들어오지 못한다.
        rows = [(t.code, t.date.strftime("%Y-%m-%d"), t.trade_type, int(t.order)) for t in portfolio.trade_history]
        self.assertEqual(rows, [
            ("A001", D[1], "BUY", 1), ("A001", D[2], "BUY", 2), ("A001", D[3], "SELL", 1),
            ("A001", D[4], "BUY", 1), ("A001", D[5], "BUY", 3),
        ])
        self.assertEqual(sorted(portfolio.daily_orders[-1]), ["A001"])

    def test_invalid_policy_value_is_rejected(self):
        with self.assertRaises(ValueError):
            build_strategy("bogus")

    def test_gpu_paths_reject_cpu_only_policy(self):
        # GPU 커널은 legacy만 구현하므로 fixed_slot_v1 설정으로 GPU 탐색·대조를 시작하면 멈춘다.
        self.assertEqual(require_gpu_supported_split_slot_policy({}), "legacy")
        self.assertEqual(require_gpu_supported_split_slot_policy({"split_slot_policy": "legacy"}), "legacy")
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            require_gpu_supported_split_slot_policy({"split_slot_policy": "fixed_slot_v1"})

    def test_shared_gpu_execution_params_reject_cpu_only_policy(self):
        # WFO 후보 평가·GPU 단일 실행이 함께 쓰는 실행 설정 구성 단계. 정책이 사라지기 전에 멈춰야
        # legacy 커널 결과가 fixed_slot_v1 이름으로 기록되지 않는다.
        legacy = build_gpu_execution_params({}, {"split_slot_policy": "legacy"}, "strict_pit")
        self.assertEqual(legacy["split_slot_policy"], "legacy")
        self.assertEqual(build_gpu_execution_params({}, {}, "strict_pit")["split_slot_policy"], "legacy")
        for base, params in (({}, {"split_slot_policy": "fixed_slot_v1"}),
                             ({"split_slot_policy": "fixed_slot_v1"}, {})):
            with self.subTest(base=base, params=params), self.assertRaisesRegex(ValueError, "CPU-only"):
                build_gpu_execution_params(base, params, "strict_pit")


if __name__ == "__main__":
    unittest.main()
