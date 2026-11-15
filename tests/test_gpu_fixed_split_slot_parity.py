"""CPU와 GPU 엔진에 같은 가상 시나리오를 넣어 차수 규칙별 일별 자산이 같은지 본다.

CPU 쪽은 tests/test_cpu_fixed_split_slot.py의 시나리오·실행 도구를 그대로 쓴다. GPU 쪽은 실제
`backtest.gpu.engine.run_magic_split_strategy_on_gpu`를 쓴다. CuPy·cuDF가 없으면 건너뛴다.
명세: todos/2026_11_15-fixed-split-slot-v1.md
"""

import unittest

import numpy as np
import pandas as pd

from examples.cpu_split_demo import EXECUTION_SETTINGS, INDICATORS, PRICE_FIELDS, STRATEGY_SETTINGS
from tests import test_cpu_fixed_split_slot as cpu_case

_GPU_IMPORT_ERROR = None
try:
    import cudf
    import cupy as cp
    from src.backtest.gpu import engine as gpu_engine
except Exception as exc:  # pragma: no cover - 환경에 따라 다름
    cp = cudf = gpu_engine = None
    _GPU_IMPORT_ERROR = exc

D = cpu_case.D
FLAT = ((10000, 10000, 10000, 10000),) * len(D)
# (이름, 종목별 OHLC, 종목별 등급)
SCENARIOS = (
    ("interior_gap", {"A001": cpu_case.INTERIOR_GAP}, {"A001": cpu_case.ALL_TIER1}),
    ("interior_gap_anchor", {"A001": cpu_case.INTERIOR_GAP_ANCHOR}, {"A001": cpu_case.ALL_TIER1}),
    ("missing_anchor", {"A001": cpu_case.MISSING_ANCHOR}, {"A001": cpu_case.TIER_BLOCK_REENTRY_FROM_D3}),
    ("orphaned_first", {"A001": cpu_case.ORPHANED_FIRST}, {"A001": cpu_case.TIER_BLOCK_REENTRY_FROM_D2}),
    ("reentry", {"A001": cpu_case.ORPHANED_FIRST}, {"A001": cpu_case.TIER_REENTRY_ON_D3}),
    ("reentry_same_day_blocked", {"A001": cpu_case.ORPHANED_FIRST}, {"A001": cpu_case.ALL_TIER1}),
    ("reentry_with_new_ticker", {"A001": cpu_case.ORPHANED_FIRST, "B001": FLAT},
     {"A001": cpu_case.TIER_REENTRY_ON_D3, "B001": cpu_case.ALL_TIER1}),
)


def _gpu_param_row():
    # [max_stocks, order_ratio, add_drop, sell_profit, priority(0=lowest_order), stop_loss, max_splits, inactivity]
    s = STRATEGY_SETTINGS
    return cp.asarray([[s["max_stocks"], s["order_investment_ratio"], s["additional_buy_drop_rate"],
                        s["sell_profit_rate"], 0, -0.99, cpu_case.MAX_SPLITS, s["max_inactivity_period"]]],
                      dtype=cp.float32)


def run_gpu(ohlc_by_ticker, tiers_by_ticker, policy):
    tickers = sorted(ohlc_by_ticker)
    dates = pd.DatetimeIndex(pd.to_datetime(D))
    rows = []
    for ticker in tickers:
        for date, values in zip(dates, ohlc_by_ticker[ticker]):
            rows.append(dict(ticker=ticker, date=date, **dict(zip(PRICE_FIELDS, map(float, values))), **INDICATORS))
    frame = pd.DataFrame(rows).set_index(["ticker", "date"])
    tier_tensor = cp.asarray(np.array([tiers_by_ticker[t] for t in tickers], dtype=np.int8).T)
    execution_params = dict(EXECUTION_SETTINGS, cooldown_period_days=0, candidate_source_mode="tier",
                            use_weekly_alpha_gate=False, parity_mode="strict",
                            tier_hysteresis_mode="strict_hysteresis_v1", progress_log_enabled=False,
                            split_slot_policy=policy)
    values = gpu_engine.run_magic_split_strategy_on_gpu(
        initial_cash=1_000_000.0,
        param_combinations=_gpu_param_row(),
        all_data_gpu=cudf.from_pandas(frame),
        trading_date_indices=cp.arange(len(dates), dtype=cp.int32),
        trading_dates_pd_cpu=dates,
        all_tickers=tickers,
        execution_params=execution_params,
        max_splits_limit=cpu_case.MAX_SPLITS,
        tier_tensor=tier_tensor,
    )
    return cp.asnumpy(values[0])


def run_cpu(ohlc_by_ticker, tiers_by_ticker, policy):
    portfolio = cpu_case.run_multi(ohlc_by_ticker, tiers_by_ticker, cpu_case.build_strategy(policy))
    return np.array([row["total_value"] for row in portfolio.daily_snapshot_history], dtype=np.float64)


class GpuFixedSplitSlotParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _GPU_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"GPU 의존성 없음: {type(_GPU_IMPORT_ERROR).__name__}: {_GPU_IMPORT_ERROR}")

    def test_daily_values_match_cpu_for_each_policy(self):
        for policy in ("legacy", "fixed_slot_v1"):
            for name, ohlc, tiers in SCENARIOS:
                with self.subTest(policy=policy, scenario=name):
                    cpu_values = run_cpu(ohlc, tiers, policy)
                    gpu_values = run_gpu(ohlc, tiers, policy)
                    # 원 단위 정수 계산이므로 float32 표현 오차(0.5원 미만) 안에서 같아야 한다.
                    np.testing.assert_allclose(gpu_values, cpu_values, rtol=0, atol=0.5)

    def test_policies_differ_where_the_rule_differs(self):
        # 같은 입력이라도 규칙이 바뀌면 결과가 달라지는 사례가 있어야, 위 비교가 규칙을 실제로 검사한다.
        _, ohlc, tiers = SCENARIOS[4]
        self.assertFalse(np.allclose(run_gpu(ohlc, tiers, "legacy"), run_gpu(ohlc, tiers, "fixed_slot_v1")))


if __name__ == "__main__":
    unittest.main()
