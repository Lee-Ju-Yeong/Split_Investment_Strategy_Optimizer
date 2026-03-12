import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.tier_coverage_report import build_tier_coverage_rows, main, summarize_tier_coverage_rows


class FakeCoverageHandler:
    def __init__(self, daily_maps):
        self.daily_maps = daily_maps

    def get_pit_universe_codes_as_of(self, as_of_date):
        payload = self.daily_maps[as_of_date.strftime("%Y-%m-%d")]
        return payload["pit_codes"], payload.get("pit_source", "SNAPSHOT_ASOF")

    def get_tiers_as_of(self, *, as_of_date, tickers, allowed_tiers):
        payload = self.daily_maps[as_of_date.strftime("%Y-%m-%d")]
        allowed = set(allowed_tiers)
        result = {}
        for code in tickers:
            info = payload["tiers"].get(code)
            if info is None:
                continue
            if int(info.get("tier", 0)) in allowed:
                result[code] = info
        return result


class TestTierCoverageReport(unittest.TestCase):
    def test_build_tier_coverage_rows_uses_default_threshold_when_none(self):
        handler = FakeCoverageHandler(
            {
                "2024-01-01": {
                    "pit_codes": ["A", "B", "C", "D"],
                    "tiers": {
                        "A": {"tier": 1, "liquidity_20d_avg_value": 100},
                        "B": {"tier": 2, "liquidity_20d_avg_value": 110},
                    },
                }
            }
        )

        rows = build_tier_coverage_rows(
            handler=handler,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            step_days=1,
            min_liquidity_20d_avg_value=0,
            min_tier12_coverage_ratio=None,
        )

        self.assertEqual(rows[0]["required_tier12_ratio"], 0.45)
        self.assertTrue(rows[0]["coverage_gate_pass"])

    def test_build_tier_coverage_rows_applies_ratio_gate_per_day(self):
        handler = FakeCoverageHandler(
            {
                "2024-01-01": {
                    "pit_codes": ["A", "B", "C", "D"],
                    "tiers": {
                        "A": {"tier": 1, "liquidity_20d_avg_value": 100},
                        "B": {"tier": 2, "liquidity_20d_avg_value": 110},
                    },
                },
                "2024-01-02": {
                    "pit_codes": ["A", "B"],
                    "tiers": {
                        "A": {"tier": 1, "liquidity_20d_avg_value": 100},
                        "B": {"tier": 2, "liquidity_20d_avg_value": 110},
                    },
                },
            }
        )

        rows = build_tier_coverage_rows(
            handler=handler,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            step_days=1,
            min_liquidity_20d_avg_value=0,
            min_tier12_coverage_ratio=0.75,
        )

        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["tier12_ratio"], 0.5, places=6)
        self.assertFalse(rows[0]["coverage_gate_pass"])
        self.assertAlmostEqual(rows[1]["tier12_ratio"], 1.0, places=6)
        self.assertTrue(rows[1]["coverage_gate_pass"])

    def test_summarize_tier_coverage_rows_reports_first_failed_day(self):
        rows = [
            {"date": "2024-01-01", "tier12_ratio": 0.5, "coverage_gate_pass": False},
            {"date": "2024-01-02", "tier12_ratio": 1.0, "coverage_gate_pass": True},
        ]

        summary = summarize_tier_coverage_rows(rows, min_tier12_coverage_ratio=0.75)

        self.assertEqual(summary["sampled_days"], 2)
        self.assertFalse(summary["coverage_gate_pass"])
        self.assertEqual(summary["failed_days"], 1)
        self.assertEqual(summary["first_failed_date"], "2024-01-01")
        self.assertAlmostEqual(summary["min_observed_tier12_ratio"], 0.5, places=6)
        self.assertAlmostEqual(summary["avg_observed_tier12_ratio"], 0.75, places=6)

    @patch("src.tier_coverage_report.DataHandler")
    @patch("src.tier_coverage_report.load_config")
    def test_main_fail_on_gate_raises_and_writes_summary(
        self,
        mock_load_config,
        mock_data_handler,
    ):
        mock_load_config.return_value = {"database": {"host": "h", "user": "u", "password": "p", "database": "d"}}
        mock_data_handler.return_value = FakeCoverageHandler(
            {
                "2024-01-01": {
                    "pit_codes": ["A", "B", "C", "D"],
                    "tiers": {
                        "A": {"tier": 1, "liquidity_20d_avg_value": 100},
                        "B": {"tier": 2, "liquidity_20d_avg_value": 100},
                    },
                }
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            argv = [
                "tier_coverage_report",
                "--start-date",
                "20240101",
                "--end-date",
                "20240101",
                "--step-days",
                "1",
                "--min-tier12-coverage-ratio",
                "0.75",
                "--summary-out",
                str(summary_path),
                "--fail-on-gate",
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(ValueError):
                    main()

            self.assertTrue(summary_path.exists())
            self.assertIn("first_failed_date", summary_path.read_text(encoding="utf-8"))

    @patch("src.tier_coverage_report.DataHandler")
    @patch("src.tier_coverage_report.load_config")
    def test_main_uses_config_threshold_when_cli_override_is_missing(
        self,
        mock_load_config,
        mock_data_handler,
    ):
        mock_load_config.return_value = {
            "database": {"host": "h", "user": "u", "password": "p", "database": "d"},
            "strategy_params": {"min_tier12_coverage_ratio": 0.45},
        }
        mock_data_handler.return_value = FakeCoverageHandler(
            {
                "2024-01-01": {
                    "pit_codes": ["A", "B", "C", "D"],
                    "tiers": {
                        "A": {"tier": 1, "liquidity_20d_avg_value": 100},
                    },
                }
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            argv = [
                "tier_coverage_report",
                "--start-date",
                "20240101",
                "--end-date",
                "20240101",
                "--step-days",
                "1",
                "--summary-out",
                str(summary_path),
                "--fail-on-gate",
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(ValueError):
                    main()

            self.assertTrue(summary_path.exists())
            self.assertIn('"min_tier12_coverage_ratio": 0.45', summary_path.read_text(encoding="utf-8"))

    def test_fail_on_gate_requires_daily_step(self):
        argv = [
            "tier_coverage_report",
            "--start-date",
            "20240101",
            "--end-date",
            "20240131",
            "--step-days",
            "7",
            "--fail-on-gate",
        ]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "--step-days 1"):
                main()


if __name__ == "__main__":
    unittest.main()
