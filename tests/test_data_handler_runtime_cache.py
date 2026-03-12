import unittest
from unittest.mock import MagicMock

import pandas as pd

from src.data_handler import DataHandler


class TestDataHandlerRuntimeCache(unittest.TestCase):
    def _build_handler(self):
        handler = DataHandler.__new__(DataHandler)
        handler.clear_runtime_lookup_cache()
        frame = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-02"),
                    "open_price": 100.0,
                    "high_price": 110.0,
                    "low_price": 95.0,
                    "close_price": 105.0,
                    "atr_14_ratio": 0.1,
                },
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "open_price": 106.0,
                    "high_price": 112.0,
                    "low_price": 101.0,
                    "close_price": 111.0,
                    "atr_14_ratio": 0.2,
                },
            ]
        ).set_index("date")
        handler.load_stock_data = MagicMock(return_value=frame)
        return handler

    def test_get_stock_row_as_of_reuses_runtime_cache_until_cleared(self):
        handler = self._build_handler()

        row1 = handler.get_stock_row_as_of("005930", "2024-01-03", "2024-01-01", "2024-01-31")
        row2 = handler.get_stock_row_as_of("005930", "2024-01-03", "2024-01-01", "2024-01-31")

        self.assertEqual(handler.load_stock_data.call_count, 1)
        self.assertEqual(float(row1["close_price"]), 111.0)
        self.assertEqual(float(row2["close_price"]), 111.0)

        handler.clear_runtime_lookup_cache()
        handler.get_stock_row_as_of("005930", "2024-01-03", "2024-01-01", "2024-01-31")
        self.assertEqual(handler.load_stock_data.call_count, 2)

    def test_get_ohlc_data_populates_latest_price_cache_for_same_day(self):
        handler = self._build_handler()

        ohlc_row = handler.get_ohlc_data_on_date("2024-01-03", "005930", "2024-01-01", "2024-01-31")
        latest_price = handler.get_latest_price("2024-01-03", "005930", "2024-01-01", "2024-01-31")

        self.assertEqual(handler.load_stock_data.call_count, 1)
        self.assertEqual(float(ohlc_row["close_price"]), 111.0)
        self.assertEqual(float(latest_price), 111.0)


if __name__ == "__main__":
    unittest.main()
