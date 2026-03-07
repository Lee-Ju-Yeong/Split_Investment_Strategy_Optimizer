import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.candidate_runtime_policy import normalize_runtime_candidate_policy
from src.optimization.gpu.parameter_simulation import _resolve_adaptive_fallback_batch_size, _resolve_batch_size


class TestGpuParameterBatchFallback(unittest.TestCase):
    def test_resolve_batch_size_prefers_auto_when_available(self):
        batch_size, source = _resolve_batch_size(
            optimal_batch_size=512,
            backtest_settings={},
            num_combinations=1000,
        )
        self.assertEqual(batch_size, 512)
        self.assertEqual(source, "auto")

    def test_resolve_batch_size_caps_auto_by_config_when_both_available(self):
        batch_size, source = _resolve_batch_size(
            optimal_batch_size=512,
            backtest_settings={"simulation_batch_size": 339},
            num_combinations=1000,
        )
        self.assertEqual(batch_size, 339)
        self.assertEqual(source, "auto-capped-by-config")

    def test_resolve_batch_size_keeps_auto_when_config_cap_is_higher(self):
        batch_size, source = _resolve_batch_size(
            optimal_batch_size=512,
            backtest_settings={"simulation_batch_size": 1024},
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

    def test_runtime_candidate_policy_keeps_tier(self):
        mode, weekly_gate = normalize_runtime_candidate_policy("tier", False)
        self.assertEqual(mode, "tier")
        self.assertFalse(weekly_gate)

    def test_runtime_candidate_policy_rejects_weekly(self):
        with self.assertRaisesRegex(ValueError, "Unsupported runtime candidate policy"):
            normalize_runtime_candidate_policy("weekly", False)

    def test_runtime_candidate_policy_rejects_hybrid_with_gate(self):
        with self.assertRaisesRegex(ValueError, "Unsupported runtime candidate policy"):
            normalize_runtime_candidate_policy("hybrid_transition", True)

    def test_runtime_candidate_policy_rejects_truthy_string_gate(self):
        with self.assertRaisesRegex(ValueError, "Unsupported runtime candidate policy"):
            normalize_runtime_candidate_policy("tier", "true")


if __name__ == "__main__":
    unittest.main()
