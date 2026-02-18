import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.financial_collector import normalize_fundamental_df
from src.investor_trading_collector import normalize_investor_df
from src.market_cap_collector import normalize_market_cap_df
from src.short_selling_collector import normalize_short_selling_df


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

    def test_normalize_fundamental_df_keeps_non_positive_per_pbr(self):
        raw = pd.DataFrame(
            {
                "날짜": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
                "PER": [-3.5, 12.0],
                "PBR": [0.0, 1.4],
                "EPS": [150.0, 1200.0],
                "BPS": [10000.0, 10000.0],
                "DPS": [0.0, 100.0],
                "DIV": [0.0, 1.0],
            }
        ).set_index("날짜")

        normalized = normalize_fundamental_df(raw, "005930")
        first_row = normalized.iloc[0]
        second_row = normalized.iloc[1]

        self.assertEqual(float(first_row["per"]), -3.5)
        self.assertEqual(float(first_row["pbr"]), 0.0)
        self.assertEqual(float(second_row["per"]), 12.0)
        self.assertEqual(float(second_row["pbr"]), 1.4)

    def test_normalize_fundamental_df_drops_all_zero_rows(self):
        raw = pd.DataFrame(
            {
                "날짜": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
                "PER": [0.0, 10.0],
                "PBR": [0.0, 1.1],
                "EPS": [0.0, 100.0],
                "BPS": [0.0, 10000.0],
                "DPS": [0.0, 100.0],
                "DIV": [0.0, 1.0],
            }
        ).set_index("날짜")

        normalized = normalize_fundamental_df(raw, "005930")
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized.iloc[0]["date"], "2024-01-03")

    def test_normalize_investor_df_drops_uninformative_rows(self):
        raw = pd.DataFrame(
            {
                "날짜": [
                    pd.Timestamp("2024-01-02"),
                    pd.Timestamp("2024-01-03"),
                    pd.Timestamp("2024-01-04"),
                ],
                "개인": [0, None, 120],
                "외국인": [0, None, -20],
                "기관합계": [0, None, -100],
            }
        ).set_index("날짜")

        normalized = normalize_investor_df(raw, "005930")
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized["date"].iloc[0], "2024-01-04")
        self.assertEqual(int(normalized["total_net_buy"].iloc[0]), 0)

    def test_normalize_investor_df_returns_empty_when_columns_missing(self):
        raw = pd.DataFrame(
            {
                "날짜": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
                "기타": [1, 2],
            }
        ).set_index("날짜")

        normalized = normalize_investor_df(raw, "005930")
        self.assertTrue(normalized.empty)

    def test_normalize_market_cap_df_keeps_stock_code(self):
        raw = pd.DataFrame(
            {
                "날짜": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
                "시가총액": [100_000_000_000, 110_000_000_000],
                "상장주식수": [5_000_000_000, 5_000_000_000],
            }
        ).set_index("날짜")

        normalized = normalize_market_cap_df(raw, "005930")
        self.assertEqual(len(normalized), 2)
        self.assertTrue((normalized["stock_code"] == "005930").all())
        self.assertFalse(normalized["stock_code"].isna().any())

    def test_normalize_market_cap_df_drops_all_zero_rows(self):
        raw = pd.DataFrame(
            {
                "날짜": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
                "시가총액": [0, 110_000_000_000],
                "상장주식수": [0, 5_000_000_000],
            }
        ).set_index("날짜")

        normalized = normalize_market_cap_df(raw, "005930")
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized.iloc[0]["date"], "2024-01-03")

    def test_normalize_short_selling_df_keeps_stock_code(self):
        raw = pd.DataFrame(
            {
                "날짜": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
                "공매도": [1000, 0],
                "잔고": [2000, 2100],
                "공매도금액": [10_000_000, 0],
                "잔고금액": [20_000_000, 21_000_000],
            }
        ).set_index("날짜")

        normalized = normalize_short_selling_df(raw, "005930")
        self.assertEqual(len(normalized), 2)
        self.assertTrue((normalized["stock_code"] == "005930").all())
        self.assertFalse(normalized["stock_code"].isna().any())


if __name__ == "__main__":
    unittest.main()
