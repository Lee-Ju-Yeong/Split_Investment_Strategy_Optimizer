"""`fixed_slot_v1` 추가 매수 차수·기준 차수를 실제 제품(split-trader V2)과 사례로 대조한다.

명세: todos/2026_11_15-fixed-split-slot-v1.md 검증 계획 5번.
제품 근거: split-trader 커밋 1bd2d32,
`back/src/split_trader/kiwoom_integration/product_trading_policy.py`의 `decide`(93-125행)와
`price_drop_met`(64-66행). 이 저장소는 split-trader를 import하지 않는다. 같은 보유 상태를
제품 코드에 넣었을 때의 결과를 손으로 따라가 고정 기대값으로 적었다.

제품 `decide`의 추가 매수 부분 요약:
- 107행 `held`: 수량이 0보다 큰 차수만 보유로 본다.
- 108행 `target`: 2..len(rules) 중 보유하지 않은 가장 낮은 차수. 1차는 대상이 아니다.
- 109-110행: 대상이 없으면 `maximum_level`로 매수하지 않는다.
- 111-113행: 기준은 `target - 1` 차수. 없으면 `buy_anchor_missing`으로 매수하지 않는다.
- 116행 `price_drop_met`: 가격 × 수량 ≤ 원가(basis_q8) × (1 − 하락률)이면 조건 충족.

이 저장소는 `max_splits_limit`이 len(rules), 신호일(T−1) 저가가 제품의 판단 가격에 해당한다.
제품은 매도 후보가 있으면 매수보다 먼저 매도 주문을 낸다(102-104행). 이 대조는 매도 후보가
없는 상태만 다룬다.
"""

import unittest

import numpy as np
import pandas as pd

from examples.cpu_split_demo import STRATEGY_SETTINGS
from src.backtest.cpu.strategy import MagicSplitStrategy, Position


DAYS = (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"))
TICKER = "SYNTH001"
DROP = 0.05  # additional_buy_drop_rate. 제품 rules[k-1]['drop_pct']를 모든 차수 5%로 둔 경우.


def held_price(order):
    """차수마다 다른 매입가(모두 호가 단위에 맞음). 트리거 가격으로 기준 차수를 구별한다."""
    return 10000 + 1000 * order


class _Portfolio:
    def __init__(self, positions):
        self.positions = {TICKER: positions} if positions else {}
        self.initial_cash = 1_000_000
        self.cash = 1_000_000

    def get_total_value(self, date, data_handler):
        return 1_000_000


class _Handler:
    """신호일 등급 1, 신호일 종가·저가만 돌려주는 최소 조회기."""

    def __init__(self, low, close):
        self.row = pd.Series({"close_price": float(close), "low_price": float(low)})

    @staticmethod
    def get_previous_trading_date(trading_dates, current_day_idx):
        return trading_dates[current_day_idx - 1]

    def get_stock_tier_as_of(self, ticker, signal_date):
        return {"tier": 1}

    def get_stock_row_as_of(self, ticker, date, start, end):
        return self.row.copy()


def next_add(held_orders, max_splits, low=1000, prices=None):
    """보유 차수 상태로 fixed_slot_v1 추가 매수 신호를 만든다. 없으면 None, 있으면 (대상 차수, 트리거)."""
    prices = prices or {}
    positions = []
    for order in sorted(held_orders):
        position = Position(prices.get(order, held_price(order)), 10, order, DROP, 0.1)
        position.open_date = DAYS[0]
        positions.append(position)
    strategy = MagicSplitStrategy(**dict(
        STRATEGY_SETTINGS, max_splits_limit=max_splits, additional_buy_drop_rate=DROP,
        cooldown_period_days=0, split_slot_policy="fixed_slot_v1",
        backtest_start_date=DAYS[0], backtest_end_date=DAYS[1],
    ))
    signals = strategy.generate_additional_buy_signals(
        DAYS[1], _Portfolio(positions), _Handler(low, low), list(DAYS), 1)
    if not signals:
        return None
    signal, = signals
    return int(signal["position"].order), float(signal["trigger_price"])


def trigger_of(anchor_order):
    return float(np.float32(held_price(anchor_order)) * (np.float32(1.0) - np.float32(DROP)))


class FixedSlotProductParityTests(unittest.TestCase):
    """대상 차수와 기준 차수(트리거 = 기준 차수 매입가 × 0.95)를 제품 결과와 맞춘다."""

    def assert_add(self, held, max_splits, expected_target, expected_anchor):
        result = next_add(held, max_splits)
        self.assertIsNotNone(result, f"보유 {held}: 매수 신호가 없음")
        target, trigger = result
        self.assertEqual(target, expected_target)
        self.assertAlmostEqual(trigger, trigger_of(expected_anchor), delta=1e-6)

    def assert_no_add(self, held, max_splits):
        self.assertIsNone(next_add(held, max_splits), f"보유 {held}: 매수 신호가 생김")

    # 제품 108행: range(2, 4)에서 2가 빔 → target 2. 111행 previous = held[1] → buy_condition.
    def test_interior_gap_one_and_three_held(self):
        self.assert_add({1, 3}, 3, expected_target=2, expected_anchor=1)

    # 제품 108행: 2는 보유, 3이 빔 → target 3. 111행 previous = held[2]. 1차 빈칸은 보지 않는다.
    def test_first_order_empty_only_two_held(self):
        self.assert_add({2}, 3, expected_target=3, expected_anchor=2)

    # 제품 108행: target 2. 111-113행 held[1] 없음 → buy_anchor_missing. 3차로 대체하지 않는다.
    def test_previous_order_missing_only_three_held(self):
        self.assert_no_add({3}, 3)

    # 제품 108-110행: 2..3 모두 보유 → target None → maximum_level.
    def test_maximum_level_all_held(self):
        self.assert_no_add({1, 2, 3}, 3)

    # 제품 108-110행: 1차가 비어도 108행은 2부터 보므로 2..3 보유면 maximum_level. 1차는 추가 매수로
    # 채우지 않는다(이 저장소는 신규 진입 경로의 1차 재진입으로 채운다. 아래 보고 참고).
    def test_first_order_empty_and_two_to_max_held(self):
        self.assert_no_add({2, 3}, 3)

    # 제품 108행: range(2, 6)에서 2·3·5가 빔 → 가장 낮은 2. 111행 previous = held[1].
    def test_multiple_gaps_lowest_wins(self):
        self.assert_add({1, 4}, 5, expected_target=2, expected_anchor=1)

    # 제품 108행: 3·5가 빔 → 3. 111행 previous = held[2](가장 최근·가장 높은 4차가 아님).
    def test_multiple_gaps_anchor_is_previous_not_highest(self):
        self.assert_add({1, 2, 4}, 5, expected_target=3, expected_anchor=2)

    # 제품 108행: 1·3·5가 빔 → 3(1은 대상 아님). 111행 previous = held[2].
    def test_multiple_gaps_with_first_empty(self):
        self.assert_add({2, 4}, 5, expected_target=3, expected_anchor=2)

    # 제품 108행: 2가 빔 → 2. 111-113행 held[1] 없음 → buy_anchor_missing(4차 기준으로 대체하지 않음).
    def test_multiple_gaps_missing_anchor(self):
        self.assert_no_add({3, 4}, 5)

    # 제품 64-66행 price_drop_met: 한 번에 체결된 차수(원가 = 매입가 × 수량)라면
    # 가격 × 수량 × Q8 × 10^9 ≤ 매입가 × 수량 × Q8 × (10^9 − 5×10^7) ⇔ 가격 ≤ 매입가 × 0.95.
    # 기준 1차 10000 → 경계 9500은 충족(<=), 한 호가 위 9510은 미충족. 이 저장소는 신호일 저가로 같다.
    def test_price_boundary_matches_product_inequality(self):
        prices = {1: 10000}
        self.assertEqual(next_add({1}, 3, low=9500, prices=prices), (2, 9500.0))
        self.assertIsNone(next_add({1}, 3, low=9510, prices=prices))


if __name__ == "__main__":
    unittest.main()
