import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import cupy as cp
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.gpu import engine as gpu_engine


class TestGpuEnginePrepPath(unittest.TestCase):
    @staticmethod
    def _empty_tensors():
        return {
            "open": cp.zeros((0, 1), dtype=cp.float32),
            "close": cp.zeros((0, 1), dtype=cp.float32),
            "high": cp.zeros((0, 1), dtype=cp.float32),
            "low": cp.zeros((0, 1), dtype=cp.float32),
        }

    @staticmethod
    def _execution_params(mode, use_weekly_alpha_gate=False):
        return {
            "buy_commission_rate": 0.00015,
            "sell_commission_rate": 0.00015,
            "sell_tax_rate": 0.0018,
            "candidate_source_mode": mode,
            "use_weekly_alpha_gate": use_weekly_alpha_gate,
            "parity_mode": "strict",
        }

    @staticmethod
    def _param_combinations():
        return cp.asarray(
            [[20.0, 0.02, 0.04, 0.04, 0.0, -0.15, 10.0, 90.0]],
            dtype=cp.float32,
        )

    @staticmethod
    def _run_with_mode(mode, all_data_gpu, weekly_filtered_gpu):
        tier_tensor = cp.zeros((0, 1), dtype=cp.int8) if mode in {"tier", "hybrid_transition"} else None
        return gpu_engine.run_magic_split_strategy_on_gpu(
            initial_cash=10_000_000.0,
            param_combinations=TestGpuEnginePrepPath._param_combinations(),
            all_data_gpu=all_data_gpu,
            weekly_filtered_gpu=weekly_filtered_gpu,
            trading_date_indices=cp.asarray([], dtype=cp.int32),
            trading_dates_pd_cpu=pd.DatetimeIndex([]),
            all_tickers=["005930"],
            execution_params=TestGpuEnginePrepPath._execution_params(mode=mode),
            tier_tensor=tier_tensor,
            debug_mode=False,
        )

    @patch("src.backtest.gpu.engine.create_gpu_data_tensors")
    def test_reuses_single_all_data_reset_index_for_tensor_build(self, mock_create_tensors):
        all_data_gpu = MagicMock()
        reset_view = MagicMock()
        all_data_gpu.reset_index.return_value = reset_view
        weekly_filtered_gpu = MagicMock()

        mock_create_tensors.return_value = self._empty_tensors()

        result = self._run_with_mode("tier", all_data_gpu, weekly_filtered_gpu)

        self.assertEqual(result.shape, (1, 0))
        all_data_gpu.reset_index.assert_called_once_with()
        self.assertIs(mock_create_tensors.call_args.args[0], reset_view)

    @patch("src.backtest.gpu.engine.create_gpu_data_tensors")
    def test_skips_weekly_reset_in_tier_mode(self, mock_create_tensors):
        all_data_gpu = MagicMock()
        all_data_gpu.reset_index.return_value = MagicMock()
        weekly_filtered_gpu = MagicMock()

        mock_create_tensors.return_value = self._empty_tensors()

        self._run_with_mode("tier", all_data_gpu, weekly_filtered_gpu)

        weekly_filtered_gpu.reset_index.assert_not_called()

    @patch("src.backtest.gpu.engine.create_gpu_data_tensors")
    def test_keeps_weekly_reset_for_weekly_mode(self, mock_create_tensors):
        all_data_gpu = MagicMock()
        all_data_gpu.reset_index.return_value = MagicMock()
        weekly_filtered_gpu = MagicMock()
        weekly_filtered_gpu.reset_index.return_value = MagicMock()

        mock_create_tensors.return_value = self._empty_tensors()

        self._run_with_mode("weekly", all_data_gpu, weekly_filtered_gpu)

        weekly_filtered_gpu.reset_index.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
