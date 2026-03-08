import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.candidate_runtime_policy import normalize_runtime_candidate_policy
from src.optimization.gpu import parameter_simulation as sim


class _FakeUniqueValues:
    def __init__(self, values):
        self._values = list(values)

    def to_pandas(self):
        return pd.Series(self._values)


class _FakeIndexLevel:
    def __init__(self, values):
        self._values = list(values)

    def isin(self, other):
        other_set = set(other)
        return np.array([value in other_set for value in self._values], dtype=bool)

    def unique(self):
        return _FakeUniqueValues(sorted(set(self._values)))


class _FakeMultiIndex:
    def __init__(self, dates, tickers):
        self._dates = list(dates)
        self._tickers = list(tickers)

    def get_level_values(self, name):
        if name == "date":
            return _FakeIndexLevel(self._dates)
        if name == "ticker":
            return _FakeIndexLevel(self._tickers)
        raise KeyError(name)


class _FakeGpuFrame:
    def __init__(self, dates, tickers, mem_bytes=1024):
        self._dates = list(dates)
        self._tickers = list(tickers)
        self._mem_bytes = int(mem_bytes)
        self.index = _FakeMultiIndex(self._dates, self._tickers)

    def __getitem__(self, mask):
        filtered_dates = [d for d, keep in zip(self._dates, mask) if bool(keep)]
        filtered_tickers = [t for t, keep in zip(self._tickers, mask) if bool(keep)]
        return _FakeGpuFrame(filtered_dates, filtered_tickers, self._mem_bytes)

    def memory_usage(self, deep=True):
        return pd.Series([self._mem_bytes], dtype="int64")


class TestGpuParameterSimulationOrchestration(unittest.TestCase):
    def _build_context(self, candidate_source_mode, use_weekly_alpha_gate):
        param_combinations = np.array(
            [[20, 0.02, 0.04, 0.04, 0, -0.15, 10, 90]],
            dtype=np.float32,
        )
        return types.SimpleNamespace(
            config={},
            backtest_settings={},
            strategy_params={
                "candidate_source_mode": candidate_source_mode,
                "use_weekly_alpha_gate": use_weekly_alpha_gate,
                "tier_hysteresis_mode": "strict_hysteresis_v1",
                "cooldown_period_days": 5,
                "min_liquidity_20d_avg_value": 123,
                "min_tier12_coverage_ratio": 0.45,
            },
            execution_params_base={},
            db_connection_str="mysql+pymysql://user:pw@host/db",
            param_combinations=param_combinations,
            num_combinations=param_combinations.shape[0],
        )

    def _fake_core_deps(self):
        trading_dates = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            name="date",
        )
        fake_pd = types.SimpleNamespace(
            read_sql=MagicMock(return_value=pd.DataFrame(index=trading_dates)),
            DataFrame=pd.DataFrame,
        )
        return np, fake_pd

    def _fake_gpu_deps(self):
        fake_cp = types.SimpleNamespace(
            int32=np.int32,
            arange=np.arange,
            vstack=np.vstack,
        )
        return fake_cp, None, MagicMock(return_value=object()), None

    def _fake_all_data_gpu(self):
        dates = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-03"),
        ]
        tickers = ["005930", "000660", "005930", "000660"]
        return _FakeGpuFrame(dates, tickers, mem_bytes=2048)

    @patch("src.optimization.gpu.parameter_simulation._ensure_core_deps")
    @patch("src.optimization.gpu.parameter_simulation._ensure_gpu_deps")
    @patch("src.optimization.gpu.parameter_simulation._get_context")
    def test_find_optimal_parameters_rejects_legacy_hysteresis_mode(
        self,
        mock_get_context,
        mock_gpu_deps,
        mock_core_deps,
    ):
        ctx = self._build_context("tier", False)
        ctx.strategy_params["tier_hysteresis_mode"] = "legacy"
        mock_get_context.return_value = ctx
        mock_gpu_deps.return_value = self._fake_gpu_deps()
        mock_core_deps.return_value = self._fake_core_deps()

        with self.assertRaisesRegex(ValueError, "Unsupported tier_hysteresis_mode"):
            sim.find_optimal_parameters("2024-01-01", "2024-01-03", 10_000_000.0)

    @patch("src.optimization.gpu.parameter_simulation.analyze_and_save_results")
    @patch("src.optimization.gpu.parameter_simulation.run_gpu_optimization")
    @patch("src.optimization.gpu.parameter_simulation.prepare_market_data_bundle")
    @patch("src.optimization.gpu.parameter_simulation.get_optimal_batch_size", return_value=1)
    @patch("src.optimization.gpu.parameter_simulation.preload_tier_data_to_tensor")
    @patch("src.optimization.gpu.parameter_simulation.preload_pit_universe_mask_to_tensor")
    @patch("src.optimization.gpu.parameter_simulation.preload_all_data_to_gpu")
    @patch("src.optimization.gpu.parameter_simulation._ensure_core_deps")
    @patch("src.optimization.gpu.parameter_simulation._ensure_gpu_deps")
    @patch("src.optimization.gpu.parameter_simulation._get_context")
    def test_find_optimal_parameters_uses_tier_only_runtime_inputs(
        self,
        mock_get_context,
        mock_gpu_deps,
        mock_core_deps,
        mock_preload_all,
        mock_preload_pit_mask,
        mock_preload_tier,
        mock_batch_size,
        mock_prepare_market_data,
        mock_run_gpu,
        mock_analyze,
    ):
        mock_get_context.return_value = self._build_context("tier", False)
        mock_gpu_deps.return_value = self._fake_gpu_deps()
        mock_core_deps.return_value = self._fake_core_deps()
        mock_preload_all.return_value = self._fake_all_data_gpu()
        mock_preload_tier.return_value = np.zeros((2, 2), dtype=np.int8)
        mock_preload_pit_mask.return_value = np.zeros((2, 2), dtype=np.int8)
        prepared_market_data = {"prepared": True}
        mock_prepare_market_data.return_value = prepared_market_data
        mock_run_gpu.side_effect = lambda param_batch, *_args, **_kwargs: np.zeros(
            (param_batch.shape[0], 2), dtype=np.float32
        )
        mock_analyze.return_value = (
            {"additional_buy_priority": 1.0},
            pd.DataFrame({"calmar_ratio": [1.0]}),
        )

        best_params, _ = sim.find_optimal_parameters("2024-01-01", "2024-01-03", 10_000_000.0)

        mock_preload_all.assert_called_once()
        _, kwargs = mock_preload_all.call_args
        self.assertTrue(kwargs["use_adjusted_prices"])
        self.assertEqual(kwargs["adjusted_price_gate_start_date"], "2013-11-20")
        self.assertEqual(kwargs["universe_mode"], "optimistic_survivor")
        _, tier_kwargs = mock_preload_tier.call_args
        self.assertEqual(tier_kwargs["universe_mode"], "optimistic_survivor")
        self.assertEqual(tier_kwargs["min_liquidity_20d_avg_value"], 123)
        self.assertAlmostEqual(float(tier_kwargs["min_tier12_coverage_ratio"]), 0.45, places=6)
        mock_prepare_market_data.assert_called_once()
        _, batch_kwargs = mock_batch_size.call_args
        self.assertEqual(batch_kwargs["fixed_data_memory_bytes"], 2120)
        run_call = mock_run_gpu.call_args
        self.assertEqual(run_call.args[6]["candidate_source_mode"], "tier")
        self.assertFalse(run_call.args[6]["use_weekly_alpha_gate"])
        self.assertEqual(run_call.args[6]["universe_mode"], "optimistic_survivor")
        self.assertIs(run_call.kwargs["prepared_market_data"], prepared_market_data)
        self.assertEqual(best_params["additional_buy_priority"], "highest_drop")

    def test_runtime_candidate_policy_rejects_weekly_mode(self):
        with self.assertRaisesRegex(ValueError, "Unsupported runtime candidate policy"):
            normalize_runtime_candidate_policy("weekly", False)

    def test_runtime_candidate_policy_rejects_weekly_alpha_gate(self):
        with self.assertRaisesRegex(ValueError, "Unsupported runtime candidate policy"):
            normalize_runtime_candidate_policy("tier", True)

    @patch("src.optimization.gpu.parameter_simulation._ensure_core_deps")
    @patch("src.optimization.gpu.parameter_simulation._ensure_gpu_deps")
    @patch("src.optimization.gpu.parameter_simulation._get_context")
    def test_find_optimal_parameters_rejects_pre_gate_start_in_adjusted_mode(
        self,
        mock_get_context,
        mock_gpu_deps,
        mock_core_deps,
    ):
        mock_get_context.return_value = self._build_context("tier", False)
        mock_gpu_deps.return_value = self._fake_gpu_deps()
        mock_core_deps.return_value = self._fake_core_deps()

        with self.assertRaises(ValueError):
            sim.find_optimal_parameters("2013-01-01", "2013-12-31", 10_000_000.0)

if __name__ == "__main__":
    unittest.main()
