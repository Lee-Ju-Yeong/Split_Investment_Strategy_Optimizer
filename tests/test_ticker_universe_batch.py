import os
import sys
import unittest
from datetime import date
from unittest.mock import MagicMock, patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline.ticker_universe_batch import (
    build_snapshot_dates,
    rebuild_ticker_universe_history,
    run_snapshot_batch,
)


class TestTickerUniverseBatch(unittest.TestCase):
    def test_build_snapshot_dates_backfill_includes_end_date(self):
        dates = build_snapshot_dates(
            mode="backfill",
            start_date_str="20240101",
            end_date_str="20240110",
            step_days=7,
        )
        self.assertEqual(
            dates,
            [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 10)],
        )

    def test_build_snapshot_dates_daily_returns_single_end_date(self):
        dates = build_snapshot_dates(
            mode="daily",
            start_date_str=None,
            end_date_str="20260207",
            step_days=7,
        )
        self.assertEqual(dates, [date(2026, 2, 7)])

    @patch("src.pipeline.ticker_universe_batch.upsert_snapshot_rows", return_value=1)
    @patch(
        "src.pipeline.ticker_universe_batch.collect_snapshot_rows",
        return_value=[("2024-01-08", "005930", "KOSPI", None, "pykrx")],
    )
    @patch(
        "src.pipeline.ticker_universe_batch.get_existing_snapshot_dates",
        return_value={date(2024, 1, 1)},
    )
    def test_run_snapshot_batch_resume_skips_existing_dates(
        self,
        _mock_existing_dates,
        _mock_collect_rows,
        _mock_upsert_rows,
    ):
        conn = MagicMock()
        summary = run_snapshot_batch(
            conn=conn,
            mode="backfill",
            start_date_str="20240101",
            end_date_str="20240108",
            markets=["KOSPI"],
            step_days=7,
            workers=1,
            resume=True,
            include_names=False,
            api_call_delay=0.0,
            log_interval=0,
        )
        self.assertEqual(summary["dates_total"], 2)
        self.assertEqual(summary["dates_processed"], 1)
        self.assertEqual(summary["dates_skipped"], 1)
        self.assertEqual(summary["rows_saved"], 1)
        self.assertEqual(summary["errors"], 0)

    def test_rebuild_ticker_universe_history_handles_empty_snapshot(self):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = (0, None, None)
        conn.cursor.return_value.__enter__.return_value = cursor

        summary = rebuild_ticker_universe_history(conn)

        self.assertEqual(summary["snapshot_rows"], 0)
        self.assertEqual(summary["history_rows"], 0)
        self.assertEqual(summary["upserted"], 0)
        self.assertEqual(summary["deleted_stale"], 0)


if __name__ == "__main__":
    unittest.main()
