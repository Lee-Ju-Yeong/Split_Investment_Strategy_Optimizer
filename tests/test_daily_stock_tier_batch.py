import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.daily_stock_tier_batch import build_daily_stock_tier_frame


class TestDailyStockTierBatch(unittest.TestCase):
    def _build_price_df(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rows = []
        for date_value in dates:
            rows.append(
                {
                    "stock_code": "A0001",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 5_000_000,
                }
            )
            rows.append(
                {
                    "stock_code": "A0002",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                }
            )
        return pd.DataFrame(rows)

    def _build_investor_df(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rows = []
        for date_value in dates:
            rows.append(
                {
                    "stock_code": "A0001",
                    "date": date_value,
                    "foreigner_net_buy": -300_000_000,
                    "institution_net_buy": -300_000_000,
                }
            )
            rows.append(
                {
                    "stock_code": "A0002",
                    "date": date_value,
                    "foreigner_net_buy": -300_000_000,
                    "institution_net_buy": -300_000_000,
                }
            )
        return pd.DataFrame(rows)

    def test_investor_overlay_disabled_keeps_tier2(self):
        output = build_daily_stock_tier_frame(
            price_df=self._build_price_df(),
            financial_df=pd.DataFrame(),
            investor_df=self._build_investor_df(),
            lookback_days=2,
            financial_lag_days=45,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
            enable_investor_v1_write=False,
            investor_flow5_threshold=-500_000_000,
        )
        latest = output[
            (output["stock_code"] == "A0001") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 2)
        self.assertNotIn("investor_flow5", str(latest["reason"]))

    def test_investor_overlay_applies_only_to_tier2(self):
        output = build_daily_stock_tier_frame(
            price_df=self._build_price_df(),
            financial_df=pd.DataFrame(),
            investor_df=self._build_investor_df(),
            lookback_days=2,
            financial_lag_days=45,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
            enable_investor_v1_write=True,
            investor_flow5_threshold=-500_000_000,
        )
        tier2_latest = output[
            (output["stock_code"] == "A0001") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(tier2_latest["tier"], 3)
        self.assertIn("investor_flow5", str(tier2_latest["reason"]))

        tier1_latest = output[
            (output["stock_code"] == "A0002") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(tier1_latest["tier"], 1)
        self.assertNotIn("investor_flow5", str(tier1_latest["reason"]))

    def test_financial_risk_override_still_applies(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0003",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                }
                for date_value in dates
            ]
        )
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0003",
                    "date": pd.Timestamp("2024-01-01"),
                    "roe": -1.0,
                    "bps": 100.0,
                }
            ]
        )
        output = build_daily_stock_tier_frame(
            price_df=price_df,
            financial_df=financial_df,
            investor_df=pd.DataFrame(),
            lookback_days=2,
            financial_lag_days=0,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
            enable_investor_v1_write=False,
            investor_flow5_threshold=-500_000_000,
        )
        latest = output[
            (output["stock_code"] == "A0003") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 3)
        self.assertIn("financial_risk", str(latest["reason"]))


if __name__ == "__main__":
    unittest.main()
