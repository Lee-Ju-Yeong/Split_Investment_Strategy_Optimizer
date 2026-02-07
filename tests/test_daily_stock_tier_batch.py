import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.daily_stock_tier_batch import build_daily_stock_tier_frame


class TestDailyStockTierBatch(unittest.TestCase):
    def test_build_daily_stock_tier_frame_applies_liquidity_and_financial_risk(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rows = []
        for day in dates:
            rows.append(
                {
                    "stock_code": "A",
                    "date": day,
                    "close_price": 100,
                    "volume": 20_000_000,  # 2,000,000,000
                }
            )
            rows.append(
                {
                    "stock_code": "B",
                    "date": day,
                    "close_price": 100,
                    "volume": 1_000_000,  # 100,000,000
                }
            )
            rows.append(
                {
                    "stock_code": "C",
                    "date": day,
                    "close_price": 100,
                    "volume": 5_000_000,  # 500,000,000
                }
            )
        price_df = pd.DataFrame(rows)
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A",
                    "date": pd.Timestamp("2024-01-01"),
                    "roe": -1.0,
                    "bps": 100.0,
                }
            ]
        )

        result = build_daily_stock_tier_frame(
            price_df=price_df,
            financial_df=financial_df,
            lookback_days=3,
            financial_lag_days=0,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
        )

        latest_date = result["date"].max()
        latest = result[result["date"] == latest_date].set_index("stock_code")
        self.assertEqual(int(latest.loc["A", "tier"]), 3)
        self.assertIn("financial_risk", str(latest.loc["A", "reason"]))
        self.assertEqual(int(latest.loc["B", "tier"]), 3)
        self.assertEqual(int(latest.loc["C", "tier"]), 2)


if __name__ == "__main__":
    unittest.main()

