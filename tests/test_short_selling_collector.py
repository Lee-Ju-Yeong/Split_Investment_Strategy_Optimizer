import os
import sys
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import short_selling_collector


class TestShortSellingCollector(unittest.TestCase):
    @patch("src.short_selling_collector.get_done_empty_coverage_windows")
    @patch("src.short_selling_collector.get_ticker_listed_dates")
    @patch("src.short_selling_collector.get_short_selling_date_bounds")
    def test_resolve_effective_windows_applies_listing_and_done_empty_skip(
        self,
        mock_date_bounds,
        mock_listed_dates,
        mock_done_empty,
    ):
        mock_date_bounds.return_value = (
            {
                "A": date(2026, 1, 20),
                "B": date(2013, 11, 20),
            },
            {},
        )
        mock_listed_dates.return_value = {
            "A": date(2020, 1, 1),
            "B": date(2010, 1, 1),
            "C": date(2025, 1, 1),
        }
        mock_done_empty.return_value = {
            ("A", date(2020, 1, 1), date(2026, 1, 19)),
        }

        windows, skipped = short_selling_collector._resolve_effective_windows(
            conn=MagicMock(),
            ticker_codes=["A", "B", "C"],
            mode="backfill",
            start_date_str="20131120",
            end_date=date(2026, 2, 3),
        )

        self.assertEqual(skipped, 2)
        self.assertEqual(
            windows,
            {
                "C": (date(2025, 1, 1), date(2026, 2, 3)),
            },
        )

    def test_split_fetch_windows_builds_latest_first_windows(self):
        windows = short_selling_collector._split_fetch_windows(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 10),
            chunk_days=4,
        )
        self.assertEqual(
            windows,
            [
                (date(2026, 1, 7), date(2026, 1, 10)),
                (date(2026, 1, 3), date(2026, 1, 6)),
                (date(2026, 1, 1), date(2026, 1, 2)),
            ],
        )

    def test_fetch_stops_after_two_consecutive_empty_chunks_once_data_seen(self):
        windows = [
            (date(2026, 1, 7), date(2026, 1, 10)),
            (date(2026, 1, 3), date(2026, 1, 6)),
            (date(2026, 1, 1), date(2026, 1, 2)),
        ]
        calls = []

        def fake_get_shorting_status_by_date(start_str, end_str, ticker_code):
            calls.append((start_str, end_str, ticker_code))
            if len(calls) == 1:
                return pd.DataFrame({"날짜": ["2026-01-10"], "공매도": [12345]})
            if len(calls) in (2, 3):
                return pd.DataFrame()
            raise AssertionError("must stop after second consecutive empty chunk following data")

        fake_pykrx = SimpleNamespace(
            stock=SimpleNamespace(get_shorting_status_by_date=fake_get_shorting_status_by_date)
        )

        with patch.dict(sys.modules, {"pykrx": fake_pykrx}):
            with patch(
                "src.short_selling_collector._split_fetch_windows",
                return_value=windows,
            ):
                result = short_selling_collector._fetch_and_normalize_short_selling(
                    ticker_code="005930",
                    effective_start=date(2026, 1, 1),
                    effective_end=date(2026, 1, 10),
                    wait_slot=lambda: None,
                )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result.get("stopped_on_empty_after_data"))
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(len(calls), 3)

    def test_fetch_returns_partial_error_with_rows_when_later_chunk_fails(self):
        windows = [
            (date(2026, 1, 7), date(2026, 1, 10)),
            (date(2026, 1, 3), date(2026, 1, 6)),
        ]
        calls = []

        def fake_get_shorting_status_by_date(start_str, end_str, ticker_code):
            calls.append((start_str, end_str, ticker_code))
            if len(calls) == 1:
                return pd.DataFrame({"날짜": ["2026-01-10"], "공매도": [99]})
            raise RuntimeError("simulated chunk failure")

        fake_pykrx = SimpleNamespace(
            stock=SimpleNamespace(get_shorting_status_by_date=fake_get_shorting_status_by_date)
        )

        with patch.dict(sys.modules, {"pykrx": fake_pykrx}):
            with patch(
                "src.short_selling_collector._split_fetch_windows",
                return_value=windows,
            ):
                result = short_selling_collector._fetch_and_normalize_short_selling(
                    ticker_code="005930",
                    effective_start=date(2026, 1, 1),
                    effective_end=date(2026, 1, 10),
                    wait_slot=lambda: None,
                )

        self.assertEqual(result["status"], "partial_error")
        self.assertEqual(result["error_type"], "unexpected_error")
        self.assertEqual(len(result["rows"]), 1)

    @patch("src.short_selling_collector.upsert_short_selling_rows", return_value=1)
    @patch("src.short_selling_collector._build_rate_limiter", return_value=lambda: None)
    @patch("src.short_selling_collector._run_short_selling_preflight")
    @patch("src.short_selling_collector._resolve_effective_windows")
    @patch("src.short_selling_collector._cap_end_date_by_short_selling_lag")
    @patch("src.short_selling_collector._cap_end_date_by_latest_trading_date")
    @patch("src.short_selling_collector._fetch_and_normalize_short_selling")
    def test_run_batch_counts_partial_errors_even_when_rows_saved(
        self,
        mock_fetch,
        mock_cap_latest,
        mock_cap_lag,
        mock_resolve_windows,
        _mock_preflight,
        _mock_rate_limiter,
        _mock_upsert,
    ):
        mock_cap_latest.return_value = date(2026, 2, 3)
        mock_cap_lag.return_value = date(2026, 2, 3)
        mock_resolve_windows.return_value = (
            {"005930": (date(2026, 2, 1), date(2026, 2, 3))},
            0,
        )
        mock_fetch.return_value = {
            "status": "partial_error",
            "error_type": "http_error",
            "rows": [
                ("005930", "2026-02-03", 1, 1, 1, 1, "pykrx"),
            ],
        }

        conn = MagicMock()
        summary = short_selling_collector.run_short_selling_batch(
            conn=conn,
            mode="backfill",
            start_date_str="20260201",
            end_date_str="20260203",
            ticker_codes=["005930"],
            workers=1,
            write_batch_size=10,
            log_interval=0,
        )

        self.assertEqual(summary["rows_saved"], 1)
        self.assertEqual(summary["partial_errors"], 1)
        self.assertEqual(summary["http_errors"], 1)
        self.assertEqual(summary["errors"], 1)

    @patch("src.short_selling_collector.upsert_short_selling_backfill_coverage", return_value=1)
    @patch("src.short_selling_collector.upsert_short_selling_rows", return_value=0)
    @patch("src.short_selling_collector._build_rate_limiter", return_value=lambda: None)
    @patch("src.short_selling_collector._run_short_selling_preflight")
    @patch("src.short_selling_collector._resolve_effective_windows")
    @patch("src.short_selling_collector._cap_end_date_by_short_selling_lag")
    @patch("src.short_selling_collector._cap_end_date_by_latest_trading_date")
    @patch("src.short_selling_collector._fetch_and_normalize_short_selling")
    def test_run_batch_persists_done_empty_coverage_on_empty_result(
        self,
        mock_fetch,
        mock_cap_latest,
        mock_cap_lag,
        mock_resolve_windows,
        _mock_preflight,
        _mock_rate_limiter,
        _mock_upsert_rows,
        mock_upsert_coverage,
    ):
        window_start = date(2013, 11, 20)
        window_end = date(2026, 1, 19)
        mock_cap_latest.return_value = date(2026, 2, 3)
        mock_cap_lag.return_value = date(2026, 2, 3)
        mock_resolve_windows.return_value = ({"005930": (window_start, window_end)}, 0)
        mock_fetch.return_value = {"status": "empty", "rows": []}

        conn = MagicMock()
        summary = short_selling_collector.run_short_selling_batch(
            conn=conn,
            mode="backfill",
            start_date_str="20131120",
            end_date_str="20260203",
            ticker_codes=["005930"],
            workers=1,
            write_batch_size=10,
            log_interval=0,
        )

        self.assertEqual(summary["empty_results"], 1)
        self.assertEqual(summary["coverage_empty_written"], 1)
        mock_upsert_coverage.assert_called_once()
        payload = mock_upsert_coverage.call_args.args[1]
        self.assertEqual(
            payload[0],
            (
                "005930",
                window_start,
                window_end,
                short_selling_collector.SHORT_SELLING_COVERAGE_DONE_EMPTY,
                0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
