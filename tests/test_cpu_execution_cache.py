import unittest
from unittest.mock import MagicMock

import pandas as pd

from src.backtest.cpu.execution import BasicExecutionHandler


class TestCpuExecutionCache(unittest.TestCase):
    def setUp(self):
        self.execution = BasicExecutionHandler()
        self.portfolio = MagicMock()
        self.portfolio.cash = 1_000_000
        self.portfolio.positions = {}
        self.data_handler = MagicMock()
        self.current_date = pd.Timestamp("2024-01-02")
        self.start_date = pd.Timestamp("2024-01-01")
        self.end_date = pd.Timestamp("2024-01-31")

    def test_execute_order_uses_cached_ohlc_when_present(self):
        order_event = {
            "ticker": "A",
            "type": "BUY",
            "date": self.current_date,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "_cached_ohlc": {"open_price": 100.0, "high_price": 101.0, "low_price": 99.0, "close_price": 100.5},
        }
        self.execution._execute_buy = MagicMock()

        self.execution.execute_order(order_event, self.portfolio, self.data_handler, current_day_idx=0)

        self.data_handler.get_ohlc_data_on_date.assert_not_called()
        self.execution._execute_buy.assert_called_once()
        passed_ohlc = self.execution._execute_buy.call_args[0][3]
        self.assertEqual(float(passed_ohlc["open_price"]), 100.0)

    def test_execute_order_fetches_ohlc_when_cache_missing(self):
        order_event = {
            "ticker": "A",
            "type": "BUY",
            "date": self.current_date,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }
        self.data_handler.get_ohlc_data_on_date.return_value = {
            "open_price": 100.0,
            "high_price": 101.0,
            "low_price": 99.0,
            "close_price": 100.5,
        }
        self.execution._execute_buy = MagicMock()

        self.execution.execute_order(order_event, self.portfolio, self.data_handler, current_day_idx=0)

        self.data_handler.get_ohlc_data_on_date.assert_called_once_with(
            self.current_date,
            "A",
            self.start_date,
            self.end_date,
        )
        self.execution._execute_buy.assert_called_once()

    def test_get_cached_name_avoids_repeated_lookup(self):
        self.data_handler.get_name_from_ticker.return_value = "Alpha"

        name_first = self.execution._get_cached_name(self.data_handler, "A")
        name_second = self.execution._get_cached_name(self.data_handler, "A")

        self.assertEqual(name_first, "Alpha")
        self.assertEqual(name_second, "Alpha")
        self.data_handler.get_name_from_ticker.assert_called_once_with("A")


if __name__ == "__main__":
    unittest.main()
