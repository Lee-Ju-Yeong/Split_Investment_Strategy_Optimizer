import os
import sys
import unittest
from datetime import date
from unittest.mock import MagicMock


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import market_cap_collector


class TestMarketCapCollector(unittest.TestCase):
    def _mock_conn_with_fetchall_side_effect(self, side_effect):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.side_effect = side_effect
        conn.cursor.return_value.__enter__.return_value = cursor
        return conn, cursor

    def test_get_market_cap_ticker_universe_raises_without_snapshot_or_history(self):
        conn, cursor = self._mock_conn_with_fetchall_side_effect([[], []])

        with self.assertRaises(RuntimeError):
            market_cap_collector.get_market_cap_ticker_universe(
                conn=conn,
                end_date=date(2026, 2, 7),
                mode="daily",
            )

        self.assertEqual(cursor.fetchall.call_count, 2)


if __name__ == "__main__":
    unittest.main()
