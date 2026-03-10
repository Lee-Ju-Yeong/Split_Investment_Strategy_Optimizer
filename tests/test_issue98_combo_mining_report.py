import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.issue98_combo_mining_report import build_report, write_report_files  # noqa: E402


HEADER = [
    "max_stocks",
    "order_investment_ratio",
    "additional_buy_drop_rate",
    "sell_profit_rate",
    "additional_buy_priority",
    "stop_loss_rate",
    "max_splits_limit",
    "max_inactivity_period",
    "period_start",
    "period_end",
    "initial_value",
    "final_value",
    "final_cumulative_returns",
    "cagr",
    "annualized_volatility",
    "mdd",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "universe_mode",
    "is_experimental",
]


class TestIssue98ComboMiningReport(unittest.TestCase):
    def _rows(self):
        return [
            {
                "max_stocks": "20.0000",
                "order_investment_ratio": "0.0200",
                "additional_buy_drop_rate": "0.0500",
                "sell_profit_rate": "0.1000",
                "additional_buy_priority": "0.0000",
                "stop_loss_rate": "-0.9000",
                "max_splits_limit": "10.0000",
                "max_inactivity_period": "126.0000",
                "period_start": "2017-01-02",
                "period_end": "2021-12-30",
                "initial_value": "10000000.0000",
                "final_value": "15000000.0000",
                "final_cumulative_returns": "0.5000",
                "cagr": "0.1200",
                "annualized_volatility": "0.2000",
                "mdd": "-0.3000",
                "sharpe_ratio": "0.6000",
                "sortino_ratio": "0.7000",
                "calmar_ratio": "0.4000",
                "universe_mode": "optimistic_survivor",
                "is_experimental": "True",
            },
            {
                "max_stocks": "20.0000",
                "order_investment_ratio": "0.0300",
                "additional_buy_drop_rate": "0.0500",
                "sell_profit_rate": "0.2500",
                "additional_buy_priority": "0.0000",
                "stop_loss_rate": "-0.9000",
                "max_splits_limit": "10.0000",
                "max_inactivity_period": "126.0000",
                "period_start": "2017-01-02",
                "period_end": "2021-12-30",
                "initial_value": "10000000.0000",
                "final_value": "18000000.0000",
                "final_cumulative_returns": "0.8000",
                "cagr": "0.1800",
                "annualized_volatility": "0.2200",
                "mdd": "-0.3500",
                "sharpe_ratio": "0.8000",
                "sortino_ratio": "0.9500",
                "calmar_ratio": "0.5100",
                "universe_mode": "optimistic_survivor",
                "is_experimental": "True",
            },
            {
                "max_stocks": "20.0000",
                "order_investment_ratio": "0.0150",
                "additional_buy_drop_rate": "0.0900",
                "sell_profit_rate": "0.1600",
                "additional_buy_priority": "1.0000",
                "stop_loss_rate": "-0.6000",
                "max_splits_limit": "15.0000",
                "max_inactivity_period": "504.0000",
                "period_start": "2017-01-02",
                "period_end": "2021-12-30",
                "initial_value": "10000000.0000",
                "final_value": "12000000.0000",
                "final_cumulative_returns": "0.2000",
                "cagr": "0.0500",
                "annualized_volatility": "0.2500",
                "mdd": "-0.5000",
                "sharpe_ratio": "0.2000",
                "sortino_ratio": "0.2500",
                "calmar_ratio": "0.1000",
                "universe_mode": "optimistic_survivor",
                "is_experimental": "True",
            },
        ]

    def test_build_report_ranks_main_effects(self):
        report = build_report(self._rows(), top_percent=34.0, metric="calmar_ratio", shortlist_size=2)
        self.assertEqual(report["row_count"], 3)
        self.assertEqual(report["top_rows"][0]["calmar_ratio"], "0.5100")
        self.assertEqual(report["parameter_importance"][0]["parameter"], "order_investment_ratio")
        self.assertEqual(
            report["top_subset_summary"]["parameter_value_frequency"]["additional_buy_priority"][0]["value"],
            "0.0000",
        )

    def test_write_report_files_outputs_json_and_markdown(self):
        rows = self._rows()
        report = build_report(rows, top_percent=34.0, metric="calmar_ratio", shortlist_size=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=HEADER)
                writer.writeheader()
                writer.writerows(rows)

            json_path, md_path = write_report_files(
                csv_path=csv_path,
                report_dir=Path(tmpdir) / "report",
                report=report,
                metric="calmar_ratio",
            )

            self.assertTrue(json_path.is_file())
            self.assertTrue(md_path.is_file())
            self.assertIn("Issue #98 Combo Mining Report", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
