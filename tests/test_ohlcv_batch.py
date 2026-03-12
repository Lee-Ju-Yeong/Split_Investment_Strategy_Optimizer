import os
import sys
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline.ohlcv_batch import (
    _build_universe_ranges_from_history_rows,
    _resolve_effective_collection_window,
    get_ohlcv_ticker_universe,
    normalize_ohlcv_df,
    run_ohlcv_batch,
)


class TestOhlcvBatch(unittest.TestCase):
    def test_build_universe_ranges_from_history_rows_applies_intersection(self):
        rows = [
            ("A0001", date(2010, 1, 1), date(2026, 2, 7), None),
            ("A0002", date(2025, 1, 1), date(2025, 12, 31), date(2025, 12, 31)),
            ("A0003", date(2026, 1, 1), date(2026, 2, 7), None),
        ]

        ranges = _build_universe_ranges_from_history_rows(
            rows=rows,
            requested_start_date=date(2024, 1, 1),
            requested_end_date=date(2025, 12, 31),
        )
        self.assertEqual(
            ranges,
            [
                ("A0001", date(2024, 1, 1), date(2025, 12, 31)),
                ("A0002", date(2025, 1, 1), date(2025, 12, 31)),
            ],
        )

    def test_resolve_effective_collection_window_with_resume(self):
        conn = MagicMock()
        universe_start = date(2010, 1, 1)
        universe_end = date(2026, 2, 7)

        with patch(
            "src.pipeline.ohlcv_batch.ohlcv_collector.get_latest_ohlcv_date_for_ticker",
            return_value=date(2024, 1, 31),
        ):
            effective_start, effective_end = _resolve_effective_collection_window(
                conn=conn,
                ticker_code="005930",
                universe_start_date=universe_start,
                universe_end_date=universe_end,
                resume=True,
            )
        self.assertEqual(effective_start, date(2024, 2, 1))
        self.assertEqual(effective_end, date(2026, 2, 7))

    def test_resolve_effective_collection_window_skips_when_latest_is_end(self):
        conn = MagicMock()
        universe_start = date(2010, 1, 1)
        universe_end = date(2026, 2, 7)

        with patch(
            "src.pipeline.ohlcv_batch.ohlcv_collector.get_latest_ohlcv_date_for_ticker",
            return_value=universe_end,
        ):
            effective_start, effective_end = _resolve_effective_collection_window(
                conn=conn,
                ticker_code="005930",
                universe_start_date=universe_start,
                universe_end_date=universe_end,
                resume=True,
            )
        self.assertIsNone(effective_start)
        self.assertIsNone(effective_end)

    @patch("src.pipeline.ohlcv_batch._fetch_history_universe_ranges")
    def test_get_ohlcv_ticker_universe_prefers_history(
        self,
        mock_fetch_history,
    ):
        mock_fetch_history.return_value = [
            ("005930", date(2010, 1, 1), date(2026, 2, 7))
        ]
        conn = MagicMock()

        ranges, source = get_ohlcv_ticker_universe(
            conn=conn,
            requested_start_date=date(2010, 1, 1),
            requested_end_date=date(2026, 2, 7),
        )
        self.assertEqual(source, "history")
        self.assertEqual(ranges, [("005930", date(2010, 1, 1), date(2026, 2, 7))])

    @patch("src.pipeline.ohlcv_batch._fetch_history_universe_ranges")
    def test_get_ohlcv_ticker_universe_raises_when_history_empty(
        self,
        mock_fetch_history,
    ):
        mock_fetch_history.return_value = []
        conn = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "TickerUniverseHistory returned no rows"):
            get_ohlcv_ticker_universe(
                conn=conn,
                requested_start_date=date(2010, 1, 1),
                requested_end_date=date(2026, 2, 7),
            )

    @patch("src.pipeline.ohlcv_batch.upsert_ohlcv_rows", return_value=2)
    @patch("src.pipeline.ohlcv_batch._resolve_effective_collection_window")
    @patch("src.pipeline.ohlcv_batch.get_ohlcv_ticker_universe")
    @patch("src.pipeline.ohlcv_batch.ohlcv_collector.get_market_ohlcv_with_fallback")
    def test_run_ohlcv_batch_reports_history_universe_source_only(
        self,
        mock_get_ohlcv,
        mock_get_universe,
        mock_resolve_window,
        _mock_upsert,
    ):
        conn = MagicMock()
        mock_get_universe.return_value = (
            [("005930", date(2024, 1, 1), date(2024, 1, 31))],
            "history",
        )
        mock_resolve_window.return_value = (date(2024, 1, 2), date(2024, 1, 31))
        idx = pd.to_datetime(["2024-01-02"])
        df = pd.DataFrame(
            {
                "시가": [100],
                "고가": [110],
                "저가": [90],
                "종가": [105],
                "거래량": [1000],
            },
            index=idx,
        )
        df.index.name = "날짜"
        mock_get_ohlcv.return_value = df

        summary = run_ohlcv_batch(
            conn=conn,
            start_date_str="20240101",
            end_date_str="20240131",
            log_interval=0,
            api_call_delay=0.0,
        )
        self.assertEqual(summary["universe_source"], "history")
        self.assertNotIn("allow_legacy_fallback", summary)
        self.assertNotIn("legacy_fallback_used", summary)
        self.assertNotIn("legacy_fallback_tickers", summary)
        self.assertNotIn("legacy_fallback_runs", summary)

    def test_normalize_ohlcv_df_maps_expected_columns(self):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        df = pd.DataFrame(
            {
                "시가": [100, 110],
                "고가": [120, 130],
                "저가": [90, 100],
                "종가": [115, 125],
                "거래량": [1000, 2000],
            },
            index=idx,
        )
        df.index.name = "날짜"

        normalized = normalize_ohlcv_df(df, "005930")
        self.assertEqual(
            list(normalized.columns),
            [
                "stock_code",
                "date",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
            ],
        )
        self.assertEqual(normalized["stock_code"].tolist(), ["005930", "005930"])
        self.assertEqual(normalized["date"].tolist(), ["2024-01-02", "2024-01-03"])
        self.assertEqual(normalized["close_price"].tolist(), [115, 125])


if __name__ == "__main__":
    unittest.main()
