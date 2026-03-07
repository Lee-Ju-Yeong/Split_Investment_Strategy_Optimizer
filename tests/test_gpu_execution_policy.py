import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gpu_execution_policy import build_gpu_execution_params


class TestGpuExecutionPolicy(unittest.TestCase):
    def test_build_gpu_execution_params_keeps_strict_only_policy(self):
        params = build_gpu_execution_params(
            {},
            {
                "candidate_source_mode": "tier",
                "use_weekly_alpha_gate": False,
                "tier_hysteresis_mode": "strict_hysteresis_v1",
            },
            universe_mode="strict_pit",
        )
        self.assertEqual(params["candidate_source_mode"], "tier")
        self.assertFalse(params["use_weekly_alpha_gate"])
        self.assertEqual(params["tier_hysteresis_mode"], "strict_hysteresis_v1")
        self.assertEqual(params["universe_mode"], "strict_pit")

    def test_build_gpu_execution_params_rejects_legacy_hysteresis(self):
        with self.assertRaisesRegex(ValueError, "Unsupported tier_hysteresis_mode"):
            build_gpu_execution_params(
                {},
                {
                    "candidate_source_mode": "tier",
                    "use_weekly_alpha_gate": False,
                    "tier_hysteresis_mode": "legacy",
                },
                universe_mode="strict_pit",
            )


if __name__ == "__main__":
    unittest.main()
