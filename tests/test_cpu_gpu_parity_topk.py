import unittest

import pandas as pd

from src import cpu_gpu_parity_topk as parity


class TestIssue56ParityTopK(unittest.TestCase):
    def _build_base_params(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "param_id": [0, 1, 2, 3],
                "max_stocks": [10, 10, 10, 10],
                "order_investment_ratio": [0.1, 0.1, 0.1, 0.1],
                "additional_buy_drop_rate": [0.05, 0.05, 0.05, 0.05],
                "sell_profit_rate": [0.07, 0.07, 0.07, 0.07],
                "additional_buy_priority": [1, 1, 1, 1],
                "stop_loss_rate": [-0.15, -0.15, -0.15, -0.15],
                "max_splits_limit": [6, 6, 6, 6],
                "max_inactivity_period": [60, 60, 60, 60],
            }
        )

    def test_build_scenarios_all_contains_expected_entries(self):
        scenarios = parity._build_scenarios(
            base_params=self._build_base_params(),
            scenario="all",
            seeded_stress_count=2,
            jackknife_max_drop=2,
        )

        self.assertEqual(len(scenarios), 5)
        scenario_keys = {(item["scenario_type"], item["seed_id"], item["drop_top_n"]) for item in scenarios}
        self.assertIn(("baseline_deterministic", None, 0), scenario_keys)
        self.assertIn(("seeded_stress", 0, 0), scenario_keys)
        self.assertIn(("seeded_stress", 1, 0), scenario_keys)
        self.assertIn(("jackknife_drop_topn", None, 1), scenario_keys)
        self.assertIn(("jackknife_drop_topn", None, 2), scenario_keys)

    def test_compare_curves_reports_mismatch_index_and_state_dump(self):
        idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        cpu_curve = pd.Series([100.0, 105.0, 110.0], index=idx)
        gpu_curve = pd.Series([100.0, 100.0, 110.0], index=idx)
        cpu_snapshots = pd.DataFrame(
            {
                "total_value": [100.0, 105.0, 110.0],
                "cash": [20.0, 25.0, 30.0],
                "stock_count": [3, 4, 5],
            },
            index=idx,
        )

        result = parity._compare_curves(
            cpu_curve=cpu_curve,
            gpu_curve=gpu_curve,
            cpu_snapshots=cpu_snapshots,
            tolerance=1e-3,
        )

        self.assertFalse(result["matched"])
        self.assertEqual(result["first_mismatch_index"], 1)
        self.assertEqual(result["first_mismatch"]["date"], "2024-01-02")
        self.assertAlmostEqual(result["first_mismatch"]["abs_diff"], 5.0, places=6)
        self.assertEqual(result["cpu_state_dump"]["stock_count"], 4)
        self.assertEqual(result["positions_dump"]["cpu_stock_count"], 4)
        self.assertGreaterEqual(len(result["value_dump"]), 1)

    def test_compare_curves_match(self):
        idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
        cpu_curve = pd.Series([100.0, 101.0], index=idx)
        gpu_curve = pd.Series([100.0, 101.0], index=idx)
        cpu_snapshots = pd.DataFrame(
            {"total_value": [100.0, 101.0], "cash": [50.0, 49.0], "stock_count": [1, 1]},
            index=idx,
        )

        result = parity._compare_curves(
            cpu_curve=cpu_curve,
            gpu_curve=gpu_curve,
            cpu_snapshots=cpu_snapshots,
            tolerance=1e-6,
        )
        self.assertTrue(result["matched"])
        self.assertIsNone(result["first_mismatch"])
        self.assertEqual(result["value_dump"], [])

    def test_parser_supports_all_candidate_mode_and_no_fail_flag(self):
        parser = parity._build_parser(
            {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_cash": 1000000,
                "candidate_source_mode": "tier",
                "use_weekly_alpha_gate": False,
            }
        )
        args = parser.parse_args(["--candidate-source-mode", "all", "--no-fail-on-mismatch"])
        self.assertEqual(args.candidate_source_mode, "all")
        self.assertFalse(args.fail_on_mismatch)


if __name__ == "__main__":
    unittest.main()

