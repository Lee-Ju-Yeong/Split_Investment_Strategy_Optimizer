import unittest

import pandas as pd

from src.cpu_gpu_parity_topk import ParityParamRow, _compare_curves


class TestCpuGpuParityTopk(unittest.TestCase):
    def test_param_row_priority_mapping(self):
        row0 = ParityParamRow.from_mapping(
            {
                "max_stocks": 20,
                "order_investment_ratio": 0.02,
                "additional_buy_drop_rate": 0.04,
                "sell_profit_rate": 0.04,
                "additional_buy_priority": 0,
                "stop_loss_rate": -0.15,
                "max_splits_limit": 10,
                "max_inactivity_period": 90,
            }
        )
        self.assertEqual(row0.additional_buy_priority, "lowest_order")
        self.assertEqual(int(row0.to_gpu_row()[4]), 0)

        row1 = ParityParamRow.from_mapping(
            {
                "max_stocks": 20,
                "order_investment_ratio": 0.02,
                "additional_buy_drop_rate": 0.04,
                "sell_profit_rate": 0.04,
                "additional_buy_priority": 1,
                "stop_loss_rate": -0.15,
                "max_splits_limit": 10,
                "max_inactivity_period": 90,
            }
        )
        self.assertEqual(row1.additional_buy_priority, "highest_drop")
        self.assertEqual(int(row1.to_gpu_row()[4]), 1)

    def test_compare_curves(self):
        idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
        cpu = pd.Series([100.0, 101.0, 102.0], index=idx)
        gpu_ok = pd.Series([100.0, 101.0001, 102.0], index=idx)
        gpu_bad = pd.Series([100.0, 105.0, 102.0], index=idx)

        ok = _compare_curves(cpu, gpu_ok, tolerance=1e-3)
        self.assertEqual(ok["mismatch_count"], 0)
        self.assertIsNone(ok["first_mismatch"])

        bad = _compare_curves(cpu, gpu_bad, tolerance=1e-3)
        self.assertGreater(bad["mismatch_count"], 0)
        self.assertIsNotNone(bad["first_mismatch"])


if __name__ == "__main__":
    unittest.main()

