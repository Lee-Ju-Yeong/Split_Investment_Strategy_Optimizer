import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.candidate_runtime_policy import normalize_runtime_candidate_policy

_GPU_IMPORT_ERROR = None
try:
    import cupy as cp
    import cudf
    from src.backtest.gpu import engine as gpu_engine
except Exception as exc:  # pragma: no cover - environment dependent
    cp = None
    cudf = None
    gpu_engine = None
    _GPU_IMPORT_ERROR = exc


class TestGpuEnginePrepPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _GPU_IMPORT_ERROR is not None:
            raise unittest.SkipTest(
                f"GPU engine test dependencies unavailable: {type(_GPU_IMPORT_ERROR).__name__}: {_GPU_IMPORT_ERROR}"
            )

    @staticmethod
    def _empty_tensors():
        return {
            "open": cp.zeros((0, 1), dtype=cp.float32),
            "close": cp.zeros((0, 1), dtype=cp.float32),
            "high": cp.zeros((0, 1), dtype=cp.float32),
            "low": cp.zeros((0, 1), dtype=cp.float32),
        }

    @staticmethod
    def _execution_params(
        mode,
        use_weekly_alpha_gate=False,
        tier_hysteresis_mode="strict_hysteresis_v1",
    ):
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
    def _prepared_market_data(num_tickers=1):
        return {
            "all_data_reset_idx": cudf.DataFrame(
                {
                    "ticker": [],
                    "date": [],
                    "ticker_idx": [],
                }
            ),
            "month_start_indices": [],
            "open_prices_tensor": cp.zeros((0, num_tickers), dtype=cp.float32),
            "close_prices_tensor": cp.zeros((0, num_tickers), dtype=cp.float32),
            "high_prices_tensor": cp.zeros((0, num_tickers), dtype=cp.float32),
            "low_prices_tensor": cp.zeros((0, num_tickers), dtype=cp.float32),
            "candidate_rank_tensors": {
                "atr_14_ratio": cp.zeros((0, num_tickers), dtype=cp.float32),
                "flow5_mcap": cp.zeros((0, num_tickers), dtype=cp.float32),
                "cheap_score_effective": cp.zeros((0, num_tickers), dtype=cp.float32),
                "market_cap_q": cp.zeros((0, num_tickers), dtype=cp.int64),
            },
            "zero_signal_prices_gpu": cp.zeros((num_tickers,), dtype=cp.float32),
            "zero_signal_tiers_gpu": cp.zeros((num_tickers,), dtype=cp.int8),
        }

    def test_single_sim_available_slots_counts_held_stocks(self):
        positions_state = cp.zeros((1, 3, 2, 1), dtype=cp.float32)
        positions_state[0, 0, 0, 0] = 10.0
        positions_state[0, 1, 1, 0] = 5.0

        available_slots = gpu_engine._get_single_sim_available_slots(
            positions_state=positions_state,
            max_stocks=2.0,
        )

        self.assertEqual(available_slots, 0)

    @staticmethod
    def _run_with_mode(mode, all_data_gpu):
        # strict-only runtime에서는 valid tier policy만 허용한다.
        tier_tensor = cp.zeros((0, 1), dtype=cp.int8)
        return gpu_engine.run_magic_split_strategy_on_gpu(
            initial_cash=10_000_000.0,
            param_combinations=TestGpuEnginePrepPath._param_combinations(),
            all_data_gpu=all_data_gpu,
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

        mock_create_tensors.return_value = self._empty_tensors()

        result = self._run_with_mode("tier", all_data_gpu)

        self.assertEqual(result.shape, (1, 0))
        all_data_gpu.reset_index.assert_called_once_with()
        prepared_view = mock_create_tensors.call_args.args[0]
        self.assertIn("ticker_idx", prepared_view.columns)
        self.assertEqual(int(len(prepared_view)), int(len(reset_view)))

    def test_engine_signature_no_longer_exposes_weekly_filtered_frame(self):
        self.assertNotIn(
            "weekly_filtered_gpu",
            gpu_engine.run_magic_split_strategy_on_gpu.__code__.co_varnames,
        )

    def test_runtime_candidate_policy_rejects_weekly_mode(self):
        with self.assertRaisesRegex(ValueError, "Unsupported runtime candidate policy"):
            normalize_runtime_candidate_policy("weekly", False)

    def test_runtime_candidate_policy_rejects_weekly_alpha_gate(self):
        with self.assertRaisesRegex(ValueError, "Unsupported runtime candidate policy"):
            normalize_runtime_candidate_policy("tier", True)

    @patch("src.backtest.gpu.engine.create_gpu_data_tensors")
    def test_prepared_market_data_bypasses_tensor_rebuild(self, mock_create_tensors):
        all_data_gpu = MagicMock()
        result = gpu_engine.run_magic_split_strategy_on_gpu(
            initial_cash=10_000_000.0,
            param_combinations=self._param_combinations(),
            all_data_gpu=all_data_gpu,
            trading_date_indices=cp.asarray([], dtype=cp.int32),
            trading_dates_pd_cpu=pd.DatetimeIndex([]),
            all_tickers=["005930"],
            execution_params=self._execution_params(mode="tier"),
            tier_tensor=cp.zeros((0, 1), dtype=cp.int8),
            prepared_market_data=self._prepared_market_data(),
            debug_mode=False,
        )

        self.assertEqual(result.shape, (1, 0))
        all_data_gpu.reset_index.assert_not_called()
        mock_create_tensors.assert_not_called()

    @patch("src.backtest.gpu.engine._process_additional_buy_signals_gpu")
    @patch("src.backtest.gpu.engine._process_new_entry_signals_gpu")
    @patch("src.backtest.gpu.engine._process_sell_signals_gpu")
    @patch("src.backtest.gpu.engine.create_gpu_data_tensors")
    def test_additional_buy_receives_signal_tiers_in_strict_mode(
        self,
        mock_create_tensors,
        mock_process_sell,
        mock_process_new_entry,
        mock_process_additional_buy,
    ):
        all_data_gpu = MagicMock()
        all_data_gpu.reset_index.return_value = self._mock_all_data_reset_view()

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
            trading_date_indices=cp.asarray([0], dtype=cp.int32),
            trading_dates_pd_cpu=pd.DatetimeIndex(["2026-01-06"]),
            all_tickers=["005930"],
            execution_params=self._execution_params(
                mode="tier",
                tier_hysteresis_mode="strict_hysteresis_v1",
            ),
            tier_tensor=cp.zeros((1, 1), dtype=cp.int8),
            debug_mode=False,
        )

        self.assertTrue(mock_process_additional_buy.called)
        signal_tiers_arg = mock_process_additional_buy.call_args.kwargs["signal_tiers"]
        self.assertIsNotNone(signal_tiers_arg)
        self.assertEqual(signal_tiers_arg.shape, (1,))

    @patch("src.backtest.gpu.engine._process_additional_buy_signals_gpu")
    @patch("src.backtest.gpu.engine._process_new_entry_signals_gpu")
    @patch("src.backtest.gpu.engine._process_sell_signals_gpu")
    @patch("src.backtest.gpu.engine.create_gpu_data_tensors")
    def test_additional_buy_probe_kwargs_forwarded_when_kernel_breakdown_enabled(
        self,
        mock_create_tensors,
        mock_process_sell,
        mock_process_new_entry,
        mock_process_additional_buy,
    ):
        all_data_gpu = MagicMock()
        all_data_gpu.reset_index.return_value = self._mock_all_data_reset_view()
        mock_create_tensors.return_value = {
            "open": cp.array([[100000.0]], dtype=cp.float32),
            "close": cp.array([[100000.0]], dtype=cp.float32),
            "high": cp.array([[100000.0]], dtype=cp.float32),
            "low": cp.array([[100000.0]], dtype=cp.float32),
        }

        def _sell_stub(*args, **_kwargs):
            return args[0], args[1], args[2], args[3], cp.zeros((1, 1), dtype=cp.bool_)

        def _entry_stub(*args, **_kwargs):
            return args[0], args[1], args[2]

        def _add_stub(*args, **_kwargs):
            return args[0], args[1], args[2]

        mock_process_sell.side_effect = _sell_stub
        mock_process_new_entry.side_effect = _entry_stub
        mock_process_additional_buy.side_effect = _add_stub

        execution_params = self._execution_params(mode="tier")
        execution_params["kernel_stage_timing_enabled"] = True

        gpu_engine.run_magic_split_strategy_on_gpu(
            initial_cash=10_000_000.0,
            param_combinations=self._param_combinations(),
            all_data_gpu=all_data_gpu,
            trading_date_indices=cp.asarray([0], dtype=cp.int32),
            trading_dates_pd_cpu=pd.DatetimeIndex(["2026-01-06"]),
            all_tickers=["005930"],
            execution_params=execution_params,
            tier_tensor=cp.zeros((1, 1), dtype=cp.int8),
            debug_mode=False,
        )

        kwargs = mock_process_additional_buy.call_args.kwargs
        self.assertTrue(kwargs["kernel_stage_timing_enabled"])
        self.assertIn("kernel_stage_totals", kwargs)
        self.assertIn("additional_buy_mask_gen_s", kwargs["kernel_stage_totals"])
        self.assertIn("additional_buy_state_update_s", kwargs["kernel_stage_totals"])
        self.assertIn("additional_buy_state_final_compact_s", kwargs["kernel_stage_totals"])
        self.assertIn("additional_buy_state_last_trade_update_s", kwargs["kernel_stage_totals"])

    def test_rejects_legacy_tier_hysteresis_mode(self):
        with self.assertRaisesRegex(ValueError, "Unsupported tier_hysteresis_mode"):
            self._execution_params(mode="tier", tier_hysteresis_mode="legacy")
            gpu_engine.run_magic_split_strategy_on_gpu(
                initial_cash=10_000_000.0,
                param_combinations=self._param_combinations(),
                all_data_gpu=MagicMock(),
                trading_date_indices=cp.asarray([], dtype=cp.int32),
                trading_dates_pd_cpu=pd.DatetimeIndex([]),
                all_tickers=["005930"],
                execution_params=self._execution_params(
                    mode="tier",
                    tier_hysteresis_mode="legacy",
                ),
                tier_tensor=cp.zeros((0, 1), dtype=cp.int8),
                debug_mode=False,
            )

    @patch("src.backtest.gpu.engine._process_additional_buy_signals_gpu")
    @patch("src.backtest.gpu.engine._process_new_entry_signals_gpu")
    @patch("src.backtest.gpu.engine._process_sell_signals_gpu")
    @patch("src.backtest.gpu.engine.create_gpu_data_tensors")
    def test_no_signal_day_reuses_shared_zero_buffers(
        self,
        mock_create_tensors,
        mock_process_sell,
        mock_process_new_entry,
        mock_process_additional_buy,
    ):
        all_data_gpu = MagicMock()
        all_data_gpu.reset_index.return_value = self._mock_all_data_reset_view()
        mock_create_tensors.return_value = {
            "open": cp.array([[100000.0]], dtype=cp.float32),
            "close": cp.array([[100000.0]], dtype=cp.float32),
            "high": cp.array([[100000.0]], dtype=cp.float32),
            "low": cp.array([[100000.0]], dtype=cp.float32),
        }

        captured = {}

        def _sell_stub(*args, **_kwargs):
            captured["sell_signal_close"] = args[9]
            captured["sell_signal_high"] = args[10]
            captured["sell_signal_day_idx"] = args[11]
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
            captured["add_signal_close"] = args[7]
            captured["add_signal_low"] = args[8]
            captured["add_signal_day_idx"] = args[9]
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
            trading_date_indices=cp.asarray([0], dtype=cp.int32),
            trading_dates_pd_cpu=pd.DatetimeIndex(["2026-01-06"]),
            all_tickers=["005930"],
            execution_params=self._execution_params(mode="tier"),
            tier_tensor=cp.zeros((1, 1), dtype=cp.int8),
            debug_mode=False,
        )

        self.assertEqual(captured["sell_signal_day_idx"], -1)
        self.assertEqual(captured["add_signal_day_idx"], -1)
        self.assertIs(captured["sell_signal_close"], captured["sell_signal_high"])
        self.assertIs(captured["add_signal_close"], captured["add_signal_low"])
        self.assertEqual(int(cp.count_nonzero(captured["sell_signal_close"]).item()), 0)
        self.assertEqual(int(cp.count_nonzero(captured["add_signal_close"]).item()), 0)

    @patch("src.backtest.gpu.engine._process_additional_buy_signals_gpu")
    @patch("src.backtest.gpu.engine._process_new_entry_signals_gpu")
    @patch("src.backtest.gpu.engine._process_sell_signals_gpu")
    @patch("src.backtest.gpu.engine._collect_candidate_rank_metrics_asof")
    @patch("src.backtest.gpu.engine.collect_candidate_rank_metrics_from_tensors")
    def test_prepared_rank_tensors_bypass_legacy_asof_lookup(
        self,
        mock_collect_from_tensors,
        mock_collect_legacy,
        mock_process_sell,
        mock_process_new_entry,
        mock_process_additional_buy,
    ):
        prepared_market_data = {
            **self._prepared_market_data(num_tickers=1),
            "month_start_indices": [0],
            "open_prices_tensor": cp.array([[100000.0], [100000.0]], dtype=cp.float32),
            "close_prices_tensor": cp.array([[100000.0], [100000.0]], dtype=cp.float32),
            "high_prices_tensor": cp.array([[100000.0], [100000.0]], dtype=cp.float32),
            "low_prices_tensor": cp.array([[100000.0], [100000.0]], dtype=cp.float32),
            "candidate_rank_tensors": {
                "atr_14_ratio": cp.array([[0.1], [0.1]], dtype=cp.float32),
                "flow5_mcap": cp.array([[100.0], [100.0]], dtype=cp.float32),
                "cheap_score_effective": cp.array([[0.6], [0.6]], dtype=cp.float32),
                "market_cap_q": cp.array([[100], [100]], dtype=cp.int64),
            },
        }
        mock_collect_from_tensors.return_value = cudf.DataFrame(
            {
                "ticker_idx": [0],
                "ticker": ["005930"],
                "atr_14_ratio": [0.1],
                "flow5_mcap": [100.0],
                "cheap_score_effective": [0.6],
                "market_cap_q": [100],
            }
        )
        mock_collect_legacy.side_effect = AssertionError("legacy as-of lookup should not run")

        def _sell_stub(*args, **_kwargs):
            return args[0], args[1], args[2], args[3], cp.zeros((1, 1), dtype=cp.bool_)

        def _entry_stub(*args, **_kwargs):
            return args[0], args[1], args[2]

        def _add_stub(*args, **_kwargs):
            return args[0], args[1], args[2]

        mock_process_sell.side_effect = _sell_stub
        mock_process_new_entry.side_effect = _entry_stub
        mock_process_additional_buy.side_effect = _add_stub

        result = gpu_engine.run_magic_split_strategy_on_gpu(
            initial_cash=10_000_000.0,
            param_combinations=self._param_combinations(),
            all_data_gpu=MagicMock(),
            trading_date_indices=cp.asarray([0, 1], dtype=cp.int32),
            trading_dates_pd_cpu=pd.DatetimeIndex(["2026-01-06", "2026-01-07"]),
            all_tickers=["005930"],
            execution_params=self._execution_params(mode="tier"),
            tier_tensor=cp.asarray([[1], [1]], dtype=cp.int8),
            prepared_market_data=prepared_market_data,
            debug_mode=False,
        )

        self.assertEqual(result.shape, (1, 2))
        mock_collect_from_tensors.assert_called()
        mock_collect_legacy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
