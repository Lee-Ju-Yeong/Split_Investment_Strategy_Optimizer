import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.backtest.cpu.execution import BasicExecutionHandler


class TestExecutionHaltGuard(unittest.TestCase):
    def setUp(self):
        self.handler = BasicExecutionHandler()
        self.portfolio = MagicMock()
        self.portfolio.cash = 1_000_000
        self.portfolio.positions = {"013570": []}
        self.data_handler = MagicMock()
        self.base_order = {
            "ticker": "013570",
            "date": pd.Timestamp("2014-12-24"),
            "start_date": "2014-01-01",
            "end_date": "2014-12-31",
        }

    def test_execute_order_skips_sell_when_open_price_is_zero(self):
        order_event = {**self.base_order, "type": "SELL"}
        self.data_handler.get_ohlc_data_on_date.return_value = {
            "open_price": 0.0,
            "high_price": 0.0,
            "low_price": 0.0,
            "close_price": 6580.0,
        }

        with patch.object(self.handler, "_execute_sell") as mock_sell:
            self.handler.execute_order(order_event, self.portfolio, self.data_handler, current_day_idx=100)

        mock_sell.assert_not_called()

    def test_execute_order_calls_sell_when_open_price_is_positive(self):
        order_event = {**self.base_order, "type": "SELL"}
        self.data_handler.get_ohlc_data_on_date.return_value = {
            "open_price": 6580.0,
            "high_price": 6700.0,
            "low_price": 6500.0,
            "close_price": 6580.0,
        }

        with patch.object(self.handler, "_execute_sell") as mock_sell:
            self.handler.execute_order(order_event, self.portfolio, self.data_handler, current_day_idx=100)

        mock_sell.assert_called_once()

    def test_execute_order_skips_buy_when_open_price_is_zero(self):
        order_event = {**self.base_order, "type": "BUY"}
        self.data_handler.get_ohlc_data_on_date.return_value = {
            "open_price": 0.0,
            "high_price": 0.0,
            "low_price": 0.0,
            "close_price": 6580.0,
        }

        with patch.object(self.handler, "_execute_buy") as mock_buy:
            self.handler.execute_order(order_event, self.portfolio, self.data_handler, current_day_idx=100)

        mock_buy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
