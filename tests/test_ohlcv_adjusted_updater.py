import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import ohlcv_adjusted_updater as updater


class TestOhlcvAdjustedUpdater(unittest.TestCase):
    def _mock_conn_cursor(self):
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        return conn, cur

    def test_update_adjusted_prices_filters_hard_invalid(self):
        conn, cur = self._mock_conn_cursor()
        df = pd.DataFrame(
            {"종가": [9_999_999, 1_000_000, 1234]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        )
        df.index.name = "날짜"

        stats = updater.update_adjusted_prices(conn, "000001", df)

        self.assertEqual(stats["updated_rows"], 2)
        self.assertEqual(stats["skipped_hard_invalid"], 1)
        self.assertEqual(stats["observed_soft_sentinel"], 1)
        cur.executemany.assert_called_once()
        sql, rows = cur.executemany.call_args[0]
        self.assertIn("UPDATE DailyStockPrice SET adj_close", sql)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(float(r[0]) != 9_999_999 for r in rows))
        conn.commit.assert_called_once()

    def test_cleanup_known_adj_anomalies_counts(self):
        conn, cur = self._mock_conn_cursor()

        def _execute_side_effect(sql, params=None):
            if "WHERE adj_close = %s" in sql and "close_price" not in sql:
                cur.rowcount = 11
            elif "AND close_price > 0" in sql:
                cur.rowcount = 7
            else:
                cur.rowcount = 0

        cur.execute.side_effect = _execute_side_effect

        with patch("src.ohlcv_adjusted_updater._has_stored_adj_ohlc_columns", return_value=False):
            stats = updater.cleanup_known_adj_anomalies(conn, soft_ratio_threshold=500.0)

        self.assertEqual(stats["hard_invalid_rows_nullified"], 11)
        self.assertEqual(stats["soft_sentinel_rows_nullified"], 7)
        self.assertEqual(cur.execute.call_count, 2)
        conn.commit.assert_called_once()

    def test_cleanup_known_adj_anomalies_clears_adj_ohlc_when_columns_present(self):
        conn, cur = self._mock_conn_cursor()
        cur.rowcount = 0

        with patch("src.ohlcv_adjusted_updater._has_stored_adj_ohlc_columns", return_value=True):
            updater.cleanup_known_adj_anomalies(conn, soft_ratio_threshold=500.0)

        first_sql = cur.execute.call_args_list[0].args[0]
        second_sql = cur.execute.call_args_list[1].args[0]
        self.assertIn("adj_open = NULL, adj_high = NULL, adj_low = NULL", first_sql)
        self.assertIn("adj_open = NULL, adj_high = NULL, adj_low = NULL", second_sql)

    def test_calculate_all_adj_ratios_counts(self):
        conn, cur = self._mock_conn_cursor()

        def _execute_side_effect(sql, params=None):
            if "SET adj_ratio = NULL" in sql:
                cur.rowcount = 13
            else:
                cur.rowcount = 29

        cur.execute.side_effect = _execute_side_effect

        with patch("src.ohlcv_adjusted_updater._has_stored_adj_ohlc_columns", return_value=False):
            stats = updater.calculate_all_adj_ratios(conn)

        self.assertEqual(stats["rows_nullified"], 13)
        self.assertEqual(stats["rows_updated"], 29)
        self.assertEqual(cur.execute.call_count, 2)
        conn.commit.assert_called_once()

    @patch("src.ohlcv_adjusted_updater._has_stored_adj_ohlc_columns", return_value=True)
    def test_calculate_all_adj_ratios_clears_adj_ohlc_when_columns_present(self, _mock_has_columns):
        conn, cur = self._mock_conn_cursor()
        cur.rowcount = 0

        updater.calculate_all_adj_ratios(conn)

        first_sql = cur.execute.call_args_list[0].args[0]
        self.assertIn("SET adj_ratio = NULL, adj_open = NULL, adj_high = NULL, adj_low = NULL", first_sql)

    @patch("src.ohlcv_adjusted_updater._has_stored_adj_ohlc_columns", return_value=True)
    def test_calculate_all_adj_ohlc_counts(self, _mock_has_columns):
        conn, cur = self._mock_conn_cursor()

        def _execute_side_effect(sql, params=None):
            if "SET adj_open = NULL" in sql:
                cur.rowcount = 5
            else:
                cur.rowcount = 17

        cur.execute.side_effect = _execute_side_effect

        stats = updater.calculate_all_adj_ohlc(conn)

        self.assertFalse(stats["skipped"])
        self.assertEqual(stats["rows_nullified"], 5)
        self.assertEqual(stats["rows_updated"], 17)
        self.assertEqual(cur.execute.call_count, 2)
        conn.commit.assert_called_once()

    @patch("src.ohlcv_adjusted_updater._has_stored_adj_ohlc_columns", return_value=False)
    def test_calculate_all_adj_ohlc_skips_when_columns_missing(self, _mock_has_columns):
        conn, cur = self._mock_conn_cursor()

        stats = updater.calculate_all_adj_ohlc(conn)

        self.assertTrue(stats["skipped"])
        cur.execute.assert_not_called()
        conn.commit.assert_not_called()

    def test_collect_adj_quality_summary_maps_values(self):
        conn, cur = self._mock_conn_cursor()
        cur.fetchone.side_effect = [
            (3,),
            (100, 90, 80, 70, 69, 68, 7, 3, 10, 2, 1),
        ]

        summary = updater.collect_adj_quality_summary(conn)

        self.assertEqual(
            summary,
            {
                "total_rows": 100,
                "adj_close_rows": 90,
                "adj_ratio_rows": 80,
                "adj_open_rows": 70,
                "adj_high_rows": 69,
                "adj_low_rows": 68,
                "sentinel_1m_rows": 7,
                "sentinel_9999999_rows": 3,
                "ratio_gt_100_rows": 10,
                "ratio_gt_1000_rows": 2,
                "ratio_lt_0_1_rows": 1,
            },
        )
        self.assertEqual(cur.execute.call_count, 2)

    def test_has_stored_adj_ohlc_columns_handles_exception(self):
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        cur.execute.side_effect = RuntimeError("metadata lock timeout")

        result = updater._has_stored_adj_ohlc_columns(conn)

        self.assertFalse(result)

    @patch("src.ohlcv_adjusted_updater.collect_adj_quality_summary", return_value={})
    @patch("src.ohlcv_adjusted_updater.calculate_all_adj_ohlc", side_effect=RuntimeError("adj ohlc failed"))
    @patch("src.ohlcv_adjusted_updater.calculate_all_adj_ratios", return_value={"rows_updated": 1, "rows_nullified": 0})
    @patch(
        "src.ohlcv_adjusted_updater.cleanup_known_adj_anomalies",
        return_value={"hard_invalid_rows_nullified": 0, "soft_sentinel_rows_nullified": 0},
    )
    @patch(
        "src.ohlcv_adjusted_updater.update_adjusted_prices",
        return_value={"updated_rows": 1, "skipped_hard_invalid": 0, "observed_soft_sentinel": 0},
    )
    @patch("src.ohlcv_adjusted_updater.fetch_adjusted_ohlcv")
    def test_run_adjusted_update_batch_rolls_back_on_adj_ohlc_failure(
        self,
        mock_fetch,
        _mock_update,
        _mock_cleanup,
        _mock_ratio,
        _mock_adj_ohlc,
        _mock_quality,
    ):
        conn, _ = self._mock_conn_cursor()
        df = pd.DataFrame({"종가": [1000]}, index=pd.to_datetime(["2020-01-02"]))
        df.index.name = "날짜"
        mock_fetch.return_value = (df, None)

        summary = updater.run_adjusted_update_batch(
            conn,
            "20200101",
            "20200131",
            ticker_codes=["000001"],
            workers=1,
            api_call_delay=0.0,
            log_interval=1,
        )

        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["rows_updated"], 0)
        conn.rollback.assert_called_once()

    @patch("src.ohlcv_adjusted_updater.stock.get_market_ohlcv")
    def test_fetch_adjusted_ohlcv_returns_error_tuple(self, mock_get):
        mock_get.side_effect = RuntimeError("network error")

        df, err = updater.fetch_adjusted_ohlcv(
            "000001",
            "20200101",
            "20200131",
            wait_slot=lambda: None,
        )

        self.assertTrue(df.empty)
        self.assertIn("network error", err)


if __name__ == "__main__":
    unittest.main()
