import unittest
from unittest.mock import MagicMock, patch
from datetime import date
import pandas as pd
import numpy as np
import sys
import os

# src 폴더를 sys.path에 추가하여 모듈 임포트
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_handler import DataHandler, PointInTimeViolation
from src.backtest.cpu.strategy import MagicSplitStrategy


class DummyPortfolio:
    def __init__(self, initial_cash=1_000_000):
        self.initial_cash = initial_cash
        self.positions = {}

    def get_total_value(self, *_args, **_kwargs):
        return self.initial_cash


class TestPointInTimeDataHandler(unittest.TestCase):
    def setUp(self):
        self.db_config = {
            'host': 'fake_host',
            'user': 'fake_user',
            'password': 'fake_password',
            'database': 'fake_db',
        }
        self.pool_patcher = patch('mysql.connector.pooling.MySQLConnectionPool')
        self.mock_pool = self.pool_patcher.start()
        self.mock_conn = MagicMock()
        self.mock_pool.return_value.get_connection.return_value = self.mock_conn

        self.data_handler = DataHandler(self.db_config)
        self.data_handler.load_stock_data.cache_clear()

    def tearDown(self):
        self.pool_patcher.stop()

    def test_get_stock_row_as_of_returns_latest_past_row(self):
        dates = pd.to_datetime(['2022-01-03', '2022-01-05'])
        mock_df = pd.DataFrame(
            {
                'close_price': [100, 105],
                'high_price': [101, 106],
                'low_price': [99, 103],
            },
            index=dates,
        )

        with patch.object(self.data_handler, 'load_stock_data', return_value=mock_df):
            row = self.data_handler.get_stock_row_as_of(
                '005930', date(2022, 1, 4), '2022-01-01', '2022-01-31'
            )

        self.assertIsNotNone(row)
        self.assertEqual(row.name, pd.Timestamp('2022-01-03'))
        self.assertEqual(row['close_price'], 100)

    def test_assert_point_in_time_raises_on_future_row(self):
        with self.assertRaises(PointInTimeViolation):
            DataHandler.assert_point_in_time(
                pd.Timestamp('2022-01-05'),
                pd.Timestamp('2022-01-04'),
            )


class TestPointInTimeStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MagicSplitStrategy(
            max_stocks=1,
            order_investment_ratio=0.1,
            additional_buy_drop_rate=0.05,
            sell_profit_rate=0.1,
            backtest_start_date='2022-01-01',
            backtest_end_date='2022-01-31',
        )
        self.portfolio = DummyPortfolio()
        self.trading_dates = [
            pd.Timestamp('2022-01-03'),
            pd.Timestamp('2022-01-04'),
        ]
        self.current_date = self.trading_dates[1]

    @staticmethod
    def _build_stock_df(previous_atr, current_atr):
        index = pd.to_datetime(['2022-01-03', '2022-01-04'])
        return pd.DataFrame(
            {
                'close_price': [100.0, 101.0],
                'high_price': [101.0, 102.0],
                'low_price': [99.0, 100.0],
                'atr_14_ratio': [previous_atr, current_atr],
            },
            index=index,
        )

    @staticmethod
    def _build_data_handler_mock(stock_df):
        handler = MagicMock()
        handler.get_filtered_stock_codes.return_value = ['005930']
        handler.get_previous_trading_date.side_effect = (
            lambda trading_dates, idx: trading_dates[idx - 1] if idx and idx > 0 else None
        )
        handler.get_stock_row_as_of.side_effect = (
            lambda ticker, as_of_date, _start, _end: stock_df.asof(pd.to_datetime(as_of_date))
        )
        return handler

    def test_new_entry_signal_ignores_current_day_only_atr(self):
        stock_df = self._build_stock_df(np.nan, 0.2)
        data_handler = self._build_data_handler_mock(stock_df)

        signals = self.strategy.generate_new_entry_signals(
            self.current_date,
            self.portfolio,
            data_handler,
            self.trading_dates,
            1,
        )

        self.assertEqual(signals, [])
        self.assertGreaterEqual(data_handler.get_stock_row_as_of.call_count, 1)
        for call in data_handler.get_stock_row_as_of.call_args_list:
            self.assertEqual(pd.to_datetime(call.args[1]), pd.Timestamp('2022-01-03'))

    def test_new_entry_signal_uses_previous_day_atr_when_available(self):
        stock_df = self._build_stock_df(0.2, 0.1)
        data_handler = self._build_data_handler_mock(stock_df)

        signals = self.strategy.generate_new_entry_signals(
            self.current_date,
            self.portfolio,
            data_handler,
            self.trading_dates,
            1,
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]['ticker'], '005930')
        self.assertEqual(pd.to_datetime(signals[0]['date']), self.current_date)


if __name__ == '__main__':
    unittest.main()
