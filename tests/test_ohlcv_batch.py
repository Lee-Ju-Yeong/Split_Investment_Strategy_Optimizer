import os
import sys
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ohlcv_batch import (
    _build_universe_ranges_from_history_rows,
    _resolve_effective_collection_window,
    get_ohlcv_ticker_universe,
    normalize_ohlcv_df,
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
            "src.ohlcv_batch.ohlcv_collector.get_latest_ohlcv_date_for_ticker",
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
            "src.ohlcv_batch.ohlcv_collector.get_latest_ohlcv_date_for_ticker",
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

    @patch("src.ohlcv_batch._fetch_legacy_universe_ranges")
    @patch("src.ohlcv_batch._fetch_history_universe_ranges")
    def test_get_ohlcv_ticker_universe_prefers_history(
        self,
        mock_fetch_history,
        mock_fetch_legacy,
    ):
        mock_fetch_history.return_value = [
            ("005930", date(2010, 1, 1), date(2026, 2, 7))
        ]
        mock_fetch_legacy.return_value = [
            ("000660", date(2010, 1, 1), date(2026, 2, 7))
        ]
        conn = MagicMock()

        ranges, source = get_ohlcv_ticker_universe(
            conn=conn,
            requested_start_date=date(2010, 1, 1),
            requested_end_date=date(2026, 2, 7),
        )
        self.assertEqual(source, "history")
        self.assertEqual(ranges, [("005930", date(2010, 1, 1), date(2026, 2, 7))])
        mock_fetch_legacy.assert_not_called()

    @patch("src.ohlcv_batch._fetch_legacy_universe_ranges")
    @patch("src.ohlcv_batch._fetch_history_universe_ranges")
    def test_get_ohlcv_ticker_universe_falls_back_to_legacy(
        self,
        mock_fetch_history,
        mock_fetch_legacy,
    ):
        mock_fetch_history.return_value = []
        mock_fetch_legacy.return_value = [
            ("005930", date(2010, 1, 1), date(2026, 2, 7))
        ]
        conn = MagicMock()

        ranges, source = get_ohlcv_ticker_universe(
            conn=conn,
            requested_start_date=date(2010, 1, 1),
            requested_end_date=date(2026, 2, 7),
        )
        self.assertEqual(source, "legacy")
        self.assertEqual(ranges, [("005930", date(2010, 1, 1), date(2026, 2, 7))])
        mock_fetch_legacy.assert_called_once()

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
