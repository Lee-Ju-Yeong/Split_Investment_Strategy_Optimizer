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

    def test_get_financial_ticker_universe_raises_when_snapshot_and_history_are_empty(self):
        conn, cursor = self._mock_conn_with_fetchall_side_effect([[], []])

        with self.assertRaises(RuntimeError):
            get_financial_ticker_universe(
                conn=conn,
                end_date=date(2026, 2, 7),
                mode="daily",
            )
        self.assertEqual(cursor.fetchall.call_count, 2)

    def test_get_financial_ticker_universe_returns_history_active_source(self):
        conn, _ = self._mock_conn_with_fetchall_side_effect(
            [
                [],  # snapshot
                [("005930",)],  # active history
            ]
        )

        tickers, source = get_financial_ticker_universe(
            conn=conn,
            end_date=date(2026, 2, 7),
            mode="daily",
            return_source=True,
        )

        self.assertEqual(tickers, ["005930"])
        self.assertEqual(source, "history_active")

    @patch("src.data.collectors.financial_collector.get_financial_ticker_universe")
    def test_run_financial_batch_uses_strict_universe_summary(self, mock_get_universe):
        mock_get_universe.return_value = ([], "history_active")
        conn = MagicMock()

        summary = run_financial_batch(
            conn=conn,
            mode="daily",
            end_date_str="20260207",
            log_interval=0,
        )

        self.assertEqual(summary["universe_source"], "history_active")
        self.assertNotIn("allow_legacy_fallback", summary)
        self.assertNotIn("legacy_fallback_used", summary)
        self.assertNotIn("legacy_fallback_tickers", summary)
        self.assertNotIn("legacy_fallback_runs", summary)
        mock_get_universe.assert_called_once_with(
            conn,
            end_date=date(2026, 2, 7),
            mode="daily",
            return_source=True,
        )


if __name__ == "__main__":
    unittest.main()
