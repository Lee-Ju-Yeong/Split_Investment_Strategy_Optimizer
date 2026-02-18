import os
import sys
import unittest
from datetime import date
from unittest.mock import MagicMock, patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.collectors.financial_collector import (
    get_financial_ticker_universe,
    run_financial_batch,
)


class TestFinancialCollectorUniverse(unittest.TestCase):
    def _mock_conn_with_fetchall_side_effect(self, side_effect):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.side_effect = side_effect
        conn.cursor.return_value.__enter__.return_value = cursor
        return conn, cursor

    def test_get_financial_ticker_universe_raises_without_legacy_fallback(self):
        conn, cursor = self._mock_conn_with_fetchall_side_effect([[], []])

        with self.assertRaises(RuntimeError):
            get_financial_ticker_universe(
                conn=conn,
                end_date=date(2026, 2, 7),
                mode="daily",
                allow_legacy_fallback=False,
            )
        self.assertEqual(cursor.fetchall.call_count, 2)

    def test_get_financial_ticker_universe_uses_legacy_when_allowed(self):
        conn, _ = self._mock_conn_with_fetchall_side_effect(
            [
                [],  # snapshot
                [],  # active history
                [("005930",)],  # weekly fallback
            ]
        )

        tickers, source = get_financial_ticker_universe(
            conn=conn,
            end_date=date(2026, 2, 7),
            mode="daily",
            allow_legacy_fallback=True,
            return_source=True,
        )

        self.assertEqual(tickers, ["005930"])
        self.assertEqual(source, "legacy_weekly")

    @patch("src.data.collectors.financial_collector.get_financial_ticker_universe")
    def test_run_financial_batch_tracks_legacy_summary_fields(self, mock_get_universe):
        mock_get_universe.return_value = ([], "legacy_weekly")
        conn = MagicMock()

        summary = run_financial_batch(
            conn=conn,
            mode="daily",
            end_date_str="20260207",
            allow_legacy_fallback=True,
            log_interval=0,
        )

        self.assertEqual(summary["universe_source"], "legacy_weekly")
        self.assertTrue(summary["allow_legacy_fallback"])
        self.assertTrue(summary["legacy_fallback_used"])
        self.assertEqual(summary["legacy_fallback_tickers"], 0)
        self.assertEqual(summary["legacy_fallback_runs"], 1)
        mock_get_universe.assert_called_once_with(
            conn,
            end_date=date(2026, 2, 7),
            mode="daily",
            allow_legacy_fallback=True,
            return_source=True,
        )


if __name__ == "__main__":
    unittest.main()
