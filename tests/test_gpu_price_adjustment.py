import os
import sys
import unittest
from unittest.mock import patch

import cupy as cp
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.gpu import logic as gpu_logic
from src.backtest.gpu import utils as gpu_utils


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


if __name__ == "__main__":
    unittest.main()
