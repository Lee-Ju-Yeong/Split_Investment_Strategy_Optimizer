import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_handler import DataHandler


class TestDataHandlerTierApis(unittest.TestCase):
    def setUp(self):
        self.db_config = {
            "host": "fake_host",
            "user": "fake_user",
            "password": "fake_password",
            "database": "fake_db",
        }
        self.pool_patcher = patch("mysql.connector.pooling.MySQLConnectionPool")
        self.mock_pool = self.pool_patcher.start()
        self.mock_conn = MagicMock()
        self.mock_pool.return_value.get_connection.return_value = self.mock_conn
        self.handler = DataHandler(self.db_config)

    def tearDown(self):
        self.pool_patcher.stop()

    @patch("pandas.read_sql")
    def test_get_stock_tier_as_of(self, mock_read_sql):
        mock_read_sql.return_value = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "stock_code": "005930",
                    "tier": 1,
                    "reason": "prime_liquidity",
                    "liquidity_20d_avg_value": 123,
                }
            ]
        )
        result = self.handler.get_stock_tier_as_of("005930", "2024-01-04")
        self.assertIsNotNone(result)
        self.assertEqual(result["stock_code"], "005930")
        self.assertEqual(result["tier"], 1)

    @patch("pandas.read_sql")
    def test_get_tiers_as_of(self, mock_read_sql):
        mock_read_sql.return_value = pd.DataFrame(
            [
                {
                    "stock_code": "005930",
                    "date": pd.Timestamp("2024-01-03"),
                    "tier": 1,
                    "reason": "prime_liquidity",
                    "liquidity_20d_avg_value": 100,
                },
                {
                    "stock_code": "000660",
                    "date": pd.Timestamp("2024-01-03"),
                    "tier": 2,
                    "reason": "normal_liquidity",
                    "liquidity_20d_avg_value": 80,
                },
            ]
        )
        result = self.handler.get_tiers_as_of(
            as_of_date="2024-01-04",
            tickers=["005930", "000660"],
            allowed_tiers=[1, 2],
        )
        self.assertEqual(set(result.keys()), {"005930", "000660"})
        self.assertEqual(result["005930"]["tier"], 1)
        self.assertEqual(result["000660"]["tier"], 2)

    def test_get_filtered_stock_codes_with_tier(self):
        with patch.object(
            self.handler, "get_filtered_stock_codes", return_value=["A", "B", "C"]
        ), patch.object(
            self.handler,
            "get_tiers_as_of",
            return_value={
                "A": {"tier": 1},
                "C": {"tier": 2},
            },
        ):
            result = self.handler.get_filtered_stock_codes_with_tier(
                date="2024-01-04",
                allowed_tiers=(1, 2),
            )
        self.assertEqual(result, ["A", "C"])


if __name__ == "__main__":
    unittest.main()

