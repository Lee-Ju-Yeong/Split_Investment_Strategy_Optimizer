import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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


class _FakeWeeklyFrame:
    def __init__(self, mem_bytes=64):
        self._mem_bytes = int(mem_bytes)

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
                "tier_hysteresis_mode": "legacy",
                "cooldown_period_days": 5,
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

    @patch("src.optimization.gpu.parameter_simulation.analyze_and_save_results")
    @patch("src.optimization.gpu.parameter_simulation.run_gpu_optimization")
    @patch("src.optimization.gpu.parameter_simulation.get_optimal_batch_size", return_value=1)
    @patch("src.optimization.gpu.parameter_simulation.preload_tier_data_to_tensor")
    @patch("src.optimization.gpu.parameter_simulation.build_empty_weekly_filtered_gpu")
    @patch("src.optimization.gpu.parameter_simulation.preload_weekly_filtered_stocks_to_gpu")
    @patch("src.optimization.gpu.parameter_simulation.preload_all_data_to_gpu")
    @patch("src.optimization.gpu.parameter_simulation._ensure_core_deps")
    @patch("src.optimization.gpu.parameter_simulation._ensure_gpu_deps")
    @patch("src.optimization.gpu.parameter_simulation._get_context")
    def test_find_optimal_parameters_skips_weekly_preload_in_tier_mode(
        self,
        mock_get_context,
        mock_gpu_deps,
        mock_core_deps,
        mock_preload_all,
        mock_preload_weekly,
        mock_build_empty_weekly,
        mock_preload_tier,
        _mock_batch_size,
        mock_run_gpu,
        mock_analyze,
    ):
        mock_get_context.return_value = self._build_context("tier", False)
        mock_gpu_deps.return_value = self._fake_gpu_deps()
        mock_core_deps.return_value = self._fake_core_deps()
        mock_preload_all.return_value = self._fake_all_data_gpu()
        mock_build_empty_weekly.return_value = _FakeWeeklyFrame(mem_bytes=0)
        mock_preload_tier.return_value = np.zeros((2, 2), dtype=np.int8)
        mock_run_gpu.side_effect = lambda param_batch, *_args, **_kwargs: np.zeros(
            (param_batch.shape[0], 2), dtype=np.float32
        )
        mock_analyze.return_value = (
            {"additional_buy_priority": 1.0},
            pd.DataFrame({"calmar_ratio": [1.0]}),
        )

        best_params, _ = sim.find_optimal_parameters("2024-01-01", "2024-01-03", 10_000_000.0)

        mock_preload_weekly.assert_not_called()
        mock_build_empty_weekly.assert_called_once()
        self.assertEqual(best_params["additional_buy_priority"], "highest_drop")

    @patch("src.optimization.gpu.parameter_simulation.analyze_and_save_results")
    @patch("src.optimization.gpu.parameter_simulation.run_gpu_optimization")
    @patch("src.optimization.gpu.parameter_simulation.get_optimal_batch_size", return_value=1)
    @patch("src.optimization.gpu.parameter_simulation.preload_tier_data_to_tensor")
    @patch("src.optimization.gpu.parameter_simulation.build_empty_weekly_filtered_gpu")
    @patch("src.optimization.gpu.parameter_simulation.preload_weekly_filtered_stocks_to_gpu")
    @patch("src.optimization.gpu.parameter_simulation.preload_all_data_to_gpu")
    @patch("src.optimization.gpu.parameter_simulation._ensure_core_deps")
    @patch("src.optimization.gpu.parameter_simulation._ensure_gpu_deps")
    @patch("src.optimization.gpu.parameter_simulation._get_context")
    def test_find_optimal_parameters_preloads_weekly_in_weekly_mode(
        self,
        mock_get_context,
        mock_gpu_deps,
        mock_core_deps,
        mock_preload_all,
        mock_preload_weekly,
        mock_build_empty_weekly,
        mock_preload_tier,
        _mock_batch_size,
        mock_run_gpu,
        mock_analyze,
    ):
        mock_get_context.return_value = self._build_context("weekly", False)
        mock_gpu_deps.return_value = self._fake_gpu_deps()
        mock_core_deps.return_value = self._fake_core_deps()
        mock_preload_all.return_value = self._fake_all_data_gpu()
        mock_preload_weekly.return_value = _FakeWeeklyFrame(mem_bytes=128)
        mock_preload_tier.return_value = np.zeros((2, 2), dtype=np.int8)
        mock_run_gpu.side_effect = lambda param_batch, *_args, **_kwargs: np.zeros(
            (param_batch.shape[0], 2), dtype=np.float32
        )
        mock_analyze.return_value = ({}, pd.DataFrame({"calmar_ratio": [1.0]}))

        sim.find_optimal_parameters("2024-01-01", "2024-01-03", 10_000_000.0)

        mock_preload_weekly.assert_called_once()
        mock_build_empty_weekly.assert_not_called()


if __name__ == "__main__":
    unittest.main()
