import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.optimization.gpu.parameter_simulation import (
    _resolve_adaptive_fallback_batch_size,
    _resolve_batch_size,
)


class TestGpuParameterBatchFallback(unittest.TestCase):
    def test_resolve_batch_size_prefers_auto_when_available(self):
        batch_size, source = _resolve_batch_size(
            optimal_batch_size=512,
            backtest_settings={},
            num_combinations=1000,
        )
        self.assertEqual(batch_size, 512)
        self.assertEqual(source, "auto")

    def test_resolve_batch_size_uses_configured_fallback_when_auto_missing(self):
        batch_size, source = _resolve_batch_size(
            optimal_batch_size=None,
            backtest_settings={"simulation_batch_size": 300},
            num_combinations=1000,
        )
        self.assertEqual(batch_size, 300)
        self.assertEqual(source, "config.simulation_batch_size")

    def test_resolve_batch_size_uses_safe_adaptive_default_when_config_invalid(self):
        batch_size, source = _resolve_batch_size(
            optimal_batch_size=None,
            backtest_settings={"simulation_batch_size": 0},
            num_combinations=50000,
        )
        self.assertEqual(batch_size, 2048)
        self.assertEqual(source, "adaptive-safe-default")

    def test_resolve_batch_size_caps_to_num_combinations(self):
        batch_size, source = _resolve_batch_size(
            optimal_batch_size=None,
            backtest_settings={},
            num_combinations=120,
        )
        self.assertEqual(batch_size, 120)
        self.assertEqual(source, "adaptive-safe-default")

    def test_adaptive_fallback_batch_size_minimum_is_one(self):
        self.assertEqual(_resolve_adaptive_fallback_batch_size(0), 1)


if __name__ == "__main__":
    unittest.main()
