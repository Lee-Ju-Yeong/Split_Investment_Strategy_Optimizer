import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import cupy as cp
import cudf
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
    def _execution_params(mode, use_weekly_alpha_gate=False, tier_hysteresis_mode="legacy"):
        return {
            "buy_commission_rate": 0.00015,
            "sell_commission_rate": 0.00015,
            "sell_tax_rate": 0.0018,
            "candidate_source_mode": mode,
            "use_weekly_alpha_gate": use_weekly_alpha_gate,
            "parity_mode": "strict",
            "tier_hysteresis_mode": tier_hysteresis_mode,
        }

    @staticmethod
    def _mock_all_data_reset_view():
        return cudf.DataFrame(
            {
                "ticker": ["005930"],
                "date": pd.to_datetime(["2026-01-05"]),
                "open_price": [100000.0],
                "close_price": [100000.0],
                "high_price": [100000.0],
                "low_price": [100000.0],
                "atr_14_ratio": [0.1],
                "market_cap": [1_000_000_000.0],
            }
        )

    @staticmethod
    def _param_combinations():
        return cp.asarray(
            [[20.0, 0.02, 0.04, 0.04, 0.0, -0.15, 10.0, 90.0]],
            dtype=cp.float32,
        )

    @staticmethod
    def _run_with_mode(mode, all_data_gpu, weekly_filtered_gpu):
        # candidate_source_mode는 강제로 tier 경로를 사용하므로 tier_tensor는 항상 제공한다.
        tier_tensor = cp.zeros((0, 1), dtype=cp.int8)
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
        reset_view = self._mock_all_data_reset_view()
        all_data_gpu.reset_index.return_value = reset_view
        weekly_filtered_gpu = MagicMock()

        mock_create_tensors.return_value = self._empty_tensors()

        result = self._run_with_mode("tier", all_data_gpu, weekly_filtered_gpu)

        self.assertEqual(result.shape, (1, 0))
        all_data_gpu.reset_index.assert_called_once_with()
        prepared_view = mock_create_tensors.call_args.args[0]
        self.assertIn("ticker_idx", prepared_view.columns)
        self.assertEqual(int(len(prepared_view)), int(len(reset_view)))

    @patch("src.backtest.gpu.engine.create_gpu_data_tensors")
    def test_skips_weekly_reset_in_tier_mode(self, mock_create_tensors):
        all_data_gpu = MagicMock()
        all_data_gpu.reset_index.return_value = self._mock_all_data_reset_view()
        weekly_filtered_gpu = MagicMock()

        mock_create_tensors.return_value = self._empty_tensors()

        self._run_with_mode("tier", all_data_gpu, weekly_filtered_gpu)

        weekly_filtered_gpu.reset_index.assert_not_called()

    @patch("src.backtest.gpu.engine.create_gpu_data_tensors")
    def test_forces_tier_path_for_weekly_mode(self, mock_create_tensors):
        all_data_gpu = MagicMock()
        all_data_gpu.reset_index.return_value = self._mock_all_data_reset_view()
        weekly_filtered_gpu = MagicMock()
        weekly_filtered_gpu.reset_index.return_value = MagicMock()

        mock_create_tensors.return_value = self._empty_tensors()

        self._run_with_mode("weekly", all_data_gpu, weekly_filtered_gpu)

        weekly_filtered_gpu.reset_index.assert_not_called()

    @patch("src.backtest.gpu.engine._process_additional_buy_signals_gpu")
    @patch("src.backtest.gpu.engine._process_new_entry_signals_gpu")
    @patch("src.backtest.gpu.engine._process_sell_signals_gpu")
    @patch("src.backtest.gpu.engine.create_gpu_data_tensors")
    def test_additional_buy_receives_signal_tiers_in_legacy_mode(
        self,
        mock_create_tensors,
        mock_process_sell,
        mock_process_new_entry,
        mock_process_additional_buy,
    ):
        all_data_gpu = MagicMock()
        all_data_gpu.reset_index.return_value = self._mock_all_data_reset_view()
        weekly_filtered_gpu = MagicMock()

        mock_create_tensors.return_value = {
            "open": cp.array([[100000.0]], dtype=cp.float32),
            "close": cp.array([[100000.0]], dtype=cp.float32),
            "high": cp.array([[100000.0]], dtype=cp.float32),
            "low": cp.array([[100000.0]], dtype=cp.float32),
        }

        def _sell_stub(*args, **_kwargs):
            return (
                args[0],
                args[1],
                args[2],
                args[3],
                cp.zeros((1, 1), dtype=cp.bool_),
            )

        def _entry_stub(*args, **_kwargs):
            return (
                args[0],
                args[1],
                args[2],
            )

        def _add_stub(*args, **_kwargs):
            return (
                args[0],
                args[1],
                args[2],
            )

        mock_process_sell.side_effect = _sell_stub
        mock_process_new_entry.side_effect = _entry_stub
        mock_process_additional_buy.side_effect = _add_stub

        gpu_engine.run_magic_split_strategy_on_gpu(
            initial_cash=10_000_000.0,
            param_combinations=self._param_combinations(),
            all_data_gpu=all_data_gpu,
            weekly_filtered_gpu=weekly_filtered_gpu,
            trading_date_indices=cp.asarray([0], dtype=cp.int32),
            trading_dates_pd_cpu=pd.DatetimeIndex(["2026-01-06"]),
            all_tickers=["005930"],
            execution_params=self._execution_params(
                mode="tier",
                tier_hysteresis_mode="legacy",
            ),
            tier_tensor=cp.zeros((1, 1), dtype=cp.int8),
            debug_mode=False,
        )

        self.assertTrue(mock_process_additional_buy.called)
        signal_tiers_arg = mock_process_additional_buy.call_args.kwargs["signal_tiers"]
        self.assertIsNotNone(signal_tiers_arg)
        self.assertEqual(signal_tiers_arg.shape, (1,))


if __name__ == "__main__":
    unittest.main()
