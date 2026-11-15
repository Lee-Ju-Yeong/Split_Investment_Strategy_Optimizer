import os
import sys
import unittest
from unittest.mock import patch

import cupy as cp
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.gpu import logic as gpu_logic
from src.backtest.gpu import utils as gpu_utils
from src.execution_price_policy import adjust_price_down, adjust_price_up


def _tick_size_cpu(prices: np.ndarray) -> np.ndarray:
    result = np.full(prices.shape, 1000, dtype=np.int32)
    result = np.where(prices < 500000, 500, result)
    result = np.where(prices < 200000, 100, result)
    result = np.where(prices < 50000, 50, result)
    result = np.where(prices < 20000, 10, result)
    result = np.where(prices < 5000, 5, result)
    result = np.where(prices < 2000, 1, result)
    return result


def _adjust_up_reference(prices: np.ndarray) -> np.ndarray:
    ticks = _tick_size_cpu(prices)
    divided = prices.astype(np.float64) / ticks.astype(np.float64)
    rounded = np.round(divided, 5)
    adjusted = np.ceil(rounded) * ticks
    return adjusted.astype(np.float32)


class TestGpuPriceAdjustment(unittest.TestCase):
    def setUp(self):
        self.prev_force_chunked = gpu_utils._ADJUST_PRICE_FORCE_CHUNKED
        gpu_utils._ADJUST_PRICE_FORCE_CHUNKED = False

    def tearDown(self):
        gpu_utils._ADJUST_PRICE_FORCE_CHUNKED = self.prev_force_chunked

    def test_adjust_price_up_gpu_matches_reference(self):
        prices_np = np.array(
            [1999.9, 2000.01, 4999.5, 5000.2, 19999.9, 20000.1, 49999.9, 50000.2],
            dtype=np.float32,
        )
        prices_cp = cp.asarray(prices_np)

        out_gpu = gpu_utils.adjust_price_up_gpu(prices_cp).get()
        out_ref = _adjust_up_reference(prices_np)

        np.testing.assert_array_equal(out_gpu, out_ref)

    def test_logic_and_utils_use_same_adjust_path(self):
        prices_cp = cp.asarray([1234.1, 7777.7, 43210.5], dtype=cp.float32)
        out_logic = gpu_logic.adjust_price_up_gpu(prices_cp).get()
        out_utils = gpu_utils.adjust_price_up_gpu(prices_cp).get()
        np.testing.assert_array_equal(out_logic, out_utils)

    def test_adjust_price_up_gpu_switches_to_chunked_after_oom(self):
        input_cp = cp.asarray([1000.0, 2000.0], dtype=cp.float32)
        sentinel = cp.asarray([1000.0, 2000.0], dtype=cp.float32)

        with patch(
            "src.backtest.gpu.utils._adjust_price_up_gpu_float64_inplace",
            side_effect=MemoryError("oom"),
        ), patch(
            "src.backtest.gpu.utils._adjust_price_up_gpu_chunked",
            return_value=sentinel,
        ) as mock_chunked:
            out = gpu_utils.adjust_price_up_gpu(input_cp)

        self.assertTrue(gpu_utils._ADJUST_PRICE_FORCE_CHUNKED)
        mock_chunked.assert_called_once()
        np.testing.assert_array_equal(out.get(), sentinel.get())

    def test_adjust_price_up_gpu_switches_to_chunked_on_runtime_oom_message(self):
        input_cp = cp.asarray([1000.0, 2000.0], dtype=cp.float32)
        sentinel = cp.asarray([1000.0, 2000.0], dtype=cp.float32)

        with patch(
            "src.backtest.gpu.utils._adjust_price_up_gpu_float64_inplace",
            side_effect=RuntimeError("std::bad_alloc: out_of_memory"),
        ), patch(
            "src.backtest.gpu.utils._adjust_price_up_gpu_chunked",
            return_value=sentinel,
        ) as mock_chunked:
            out = gpu_utils.adjust_price_up_gpu(input_cp)

        self.assertTrue(gpu_utils._ADJUST_PRICE_FORCE_CHUNKED)
        mock_chunked.assert_called_once()
        np.testing.assert_array_equal(out.get(), sentinel.get())


# CPU(기준) 함수와 같은 결과가 나와야 하는 가격: 호가 정렬·비정렬·구간 경계·1원 미만·0원.
_PARITY_PRICES = [
    0.0, 0.4, 0.99, 1.0, 100.1, 145.965, 1999.0, 1999.95, 2000.0, 2000.1, 4999.99, 5000.2,
    9876.5, 9879.99999, 9880.00001, 19999.9, 20000.1, 49999.9, 50000.2, 51008.7148,
    199999.5, 200000.0, 499999.0, 500000.5, 1234567.0,
]


def _cpu_reference(prices_np, fn):
    return np.asarray([fn(float(p)) for p in prices_np], dtype=np.float32)


class TestGpuSellPriceRounding(unittest.TestCase):
    def setUp(self):
        self.prev_force_chunked = gpu_utils._ADJUST_PRICE_FORCE_CHUNKED
        gpu_utils._ADJUST_PRICE_FORCE_CHUNKED = False

    def tearDown(self):
        gpu_utils._ADJUST_PRICE_FORCE_CHUNKED = self.prev_force_chunked

    def test_up_and_down_match_cpu_policy_functions(self):
        prices_np = np.asarray(_PARITY_PRICES, dtype=np.float32)
        prices_cp = cp.asarray(prices_np)
        np.testing.assert_array_equal(
            gpu_utils.adjust_price_up_gpu(prices_cp).get(), _cpu_reference(prices_np, adjust_price_up)
        )
        np.testing.assert_array_equal(
            gpu_utils.adjust_price_down_gpu(prices_cp).get(), _cpu_reference(prices_np, adjust_price_down)
        )

    def test_down_keeps_positive_prices_at_least_one_tick(self):
        out = gpu_utils.adjust_price_down_gpu(cp.asarray([0.0, 0.4, 0.99, 1.7], dtype=cp.float32)).get()
        np.testing.assert_array_equal(out, np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float32))

    def test_sell_dispatch_and_scalar_input(self):
        price = cp.asarray(9876.5, dtype=cp.float32)
        self.assertEqual(gpu_logic.adjust_sell_price_gpu(price, "up").item(), 9880.0)
        self.assertEqual(gpu_logic.adjust_sell_price_gpu(price, "down").item(), 9870.0)

    def test_down_chunked_path_matches_full_path(self):
        prices_np = np.tile(np.asarray(_PARITY_PRICES, dtype=np.float32), (3, 1))
        prices_cp = cp.asarray(prices_np)
        full = gpu_utils._adjust_price_down_gpu_float64_inplace(prices_cp).get()
        chunked = gpu_utils._adjust_price_down_gpu_chunked(prices_cp, chunk_size=7).get()
        np.testing.assert_array_equal(full, chunked)
        np.testing.assert_array_equal(full[0], _cpu_reference(prices_np[0], adjust_price_down))

    def test_down_switches_to_chunked_after_oom(self):
        input_cp = cp.asarray([1000.5, 2000.5], dtype=cp.float32)
        sentinel = cp.asarray([1000.0, 2000.0], dtype=cp.float32)
        with patch(
            "src.backtest.gpu.utils._adjust_price_down_gpu_float64_inplace",
            side_effect=MemoryError("oom"),
        ), patch(
            "src.backtest.gpu.utils._adjust_price_down_gpu_chunked",
            return_value=sentinel,
        ) as mock_chunked:
            out = gpu_utils.adjust_price_down_gpu(input_cp)
        self.assertTrue(gpu_utils._ADJUST_PRICE_FORCE_CHUNKED)
        mock_chunked.assert_called_once()
        np.testing.assert_array_equal(out.get(), sentinel.get())


_SELL_COMMISSION = 0.00015
_SELL_TAX = 0.0018


def _cpu_net_revenue(exec_price, quantity):
    # src/backtest/cpu/execution.py `_execute_sell`과 같은 float32 정산식(포지션 1건).
    revenue = np.float32(exec_price) * np.float32(quantity)
    cost_factor = np.float32(1.0 - _SELL_COMMISSION - _SELL_TAX)
    return float(np.floor(revenue * cost_factor))


class TestGpuSellSignalsTickRounding(unittest.TestCase):
    """CPU 왕복 예(9,880원 x 101주 보유, 시가 10,371.3원)와 같은 정산 현금인지 확인한다."""

    OPEN = 10371.3
    INITIAL_CASH = 1_000_000.0

    def _run(self, *, mode, strict, quantities, signal_close, signal_high):
        num_splits = len(quantities)
        portfolio_state = cp.array([[self.INITIAL_CASH, 100_000.0]], dtype=cp.float32)
        positions_state = cp.zeros((1, 1, num_splits, 3), dtype=cp.float32)
        for split, qty in enumerate(quantities):
            positions_state[0, 0, split, 0] = qty
            positions_state[0, 0, split, 1] = 9880.0
            positions_state[0, 0, split, 2] = 0.0
        params = cp.array([[10, 0.02, 0.05, 0.04, 1, -0.50, num_splits, 999]], dtype=cp.float32)
        kwargs = dict(
            portfolio_state=portfolio_state,
            positions_state=positions_state,
            cooldown_state=cp.full((1, 1), -1, dtype=cp.int32),
            last_trade_day_idx_state=cp.array([[0]], dtype=cp.int32),
            current_day_idx=1,
            param_combinations=params,
            current_open_prices=cp.array([self.OPEN], dtype=cp.float32),
            current_close_prices=cp.array([self.OPEN], dtype=cp.float32),
            current_high_prices=cp.array([self.OPEN], dtype=cp.float32),
            signal_close_prices=cp.array([signal_close], dtype=cp.float32),
            signal_high_prices=cp.array([signal_high], dtype=cp.float32),
            signal_day_idx=0,
            sell_commission_rate=_SELL_COMMISSION,
            sell_tax_rate=_SELL_TAX,
            strict_cash_rounding=strict,
        )
        if mode is not None:
            kwargs["sell_price_tick_rounding"] = mode
        portfolio_after, positions_after, _, _, sell_mask = gpu_logic._process_sell_signals_gpu(**kwargs)
        self.assertTrue(bool(sell_mask[0, 0].item()))
        self.assertEqual(float(cp.sum(positions_after[0, 0, :, 0]).item()), 0.0)
        return float(portfolio_after[0, 0].item()) - self.INITIAL_CASH

    def test_profit_taking_cash_matches_cpu_for_each_mode(self):
        expected = {None: (10380, 1046335.0), "up": (10380, 1046335.0), "down": (10370, 1045327.0)}
        for mode, (exec_price, net) in expected.items():
            for strict in (False, True):
                with self.subTest(mode=mode, strict=strict):
                    self.assertEqual(_cpu_net_revenue(exec_price, 101), net)
                    cash_delta = self._run(
                        mode=mode, strict=strict, quantities=[101.0], signal_close=10300.0, signal_high=10400.0
                    )
                    self.assertEqual(cash_delta, net)

    def test_stop_loss_liquidation_cash_follows_mode(self):
        # 평균 매수가 9,880 x (1 - 0.50) 이하의 T-1 종가 → 전체 청산(수익 실현 조건은 미충족)
        for mode, exec_price in (("up", 10380), ("down", 10370)):
            with self.subTest(mode=mode):
                cash_delta = self._run(
                    mode=mode, strict=True, quantities=[101.0], signal_close=4000.0, signal_high=4000.0
                )
                self.assertEqual(cash_delta, _cpu_net_revenue(exec_price, 101))

    def test_strict_mode_settles_each_split_like_cpu(self):
        quantities = [101.0, 37.0]
        for mode, exec_price in (("up", 10380), ("down", 10370)):
            with self.subTest(mode=mode):
                cash_delta = self._run(
                    mode=mode, strict=True, quantities=quantities, signal_close=10300.0, signal_high=10400.0
                )
                self.assertEqual(cash_delta, sum(_cpu_net_revenue(exec_price, q) for q in quantities))

    def test_fast_mode_floors_the_summed_revenue(self):
        quantities = [101.0, 37.0]
        cost_factor = np.float32(1.0) - np.float32(_SELL_COMMISSION) - np.float32(_SELL_TAX)
        for mode, exec_price in (("up", 10380), ("down", 10370)):
            with self.subTest(mode=mode):
                cash_delta = self._run(
                    mode=mode, strict=False, quantities=quantities, signal_close=10300.0, signal_high=10400.0
                )
                total = np.float32(exec_price) * np.float32(sum(quantities))
                self.assertEqual(cash_delta, float(np.floor(total * cost_factor)))

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            self._run(mode="nearest", strict=True, quantities=[101.0], signal_close=10300.0, signal_high=10400.0)


if __name__ == "__main__":
    unittest.main()
