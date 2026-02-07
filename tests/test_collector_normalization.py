import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.financial_collector import normalize_fundamental_df
from src.investor_trading_collector import normalize_investor_df


class TestCollectorNormalization(unittest.TestCase):
    def test_normalize_fundamental_df_keeps_stock_code(self):
        raw = pd.DataFrame(
            {
                "날짜": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
                "PER": [10.0, 11.0],
                "PBR": [1.1, 1.2],
                "EPS": [1000, 1100],
                "BPS": [10000, 10000],
                "DPS": [100, 100],
                "DIV": [1.0, 1.0],
            }
        ).set_index("날짜")

        normalized = normalize_fundamental_df(raw, "005930")
        self.assertEqual(len(normalized), 2)
        self.assertTrue((normalized["stock_code"] == "005930").all())
        self.assertFalse(normalized["stock_code"].isna().any())

    def test_normalize_investor_df_keeps_stock_code(self):
        raw = pd.DataFrame(
            {
                "날짜": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
                "개인": [100, 50],
                "외국인": [-70, -20],
                "기관합계": [-30, -30],
            }
        ).set_index("날짜")

        normalized = normalize_investor_df(raw, "005930")
        self.assertEqual(len(normalized), 2)
        self.assertTrue((normalized["stock_code"] == "005930").all())
        self.assertFalse(normalized["stock_code"].isna().any())
        self.assertEqual(int(normalized["total_net_buy"].iloc[0]), 0)


if __name__ == "__main__":
    unittest.main()
