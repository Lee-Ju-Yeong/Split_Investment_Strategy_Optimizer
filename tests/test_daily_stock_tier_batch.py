import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.daily_stock_tier_batch import build_daily_stock_tier_frame
from src.daily_stock_tier_batch import upsert_daily_stock_tier


class TestDailyStockTierBatch(unittest.TestCase):
    class _FakeCursor:
        def __init__(self):
            self.executed_chunks = []
            self.rowcount = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def executemany(self, _sql, chunk):
            self.executed_chunks.append(len(chunk))
            self.rowcount = len(chunk)

    class _FakeConn:
        def __init__(self):
            self.cursor_obj = TestDailyStockTierBatch._FakeCursor()
            self.committed = False

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.committed = True

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

    def test_upsert_daily_stock_tier_uses_chunked_batches(self):
        conn = self._FakeConn()
        rows = pd.DataFrame(
            [
                ("2024-01-01", f"A{i:05d}", 2, "normal_liquidity", 1000)
                for i in range(25_001)
            ],
            columns=[
                "date",
                "stock_code",
                "tier",
                "reason",
                "liquidity_20d_avg_value",
            ],
        )

        affected = upsert_daily_stock_tier(conn, rows, batch_size=10_000)

        self.assertEqual(affected, 25_001)
        self.assertEqual(conn.cursor_obj.executed_chunks, [10_000, 10_000, 5_001])
        self.assertTrue(conn.committed)


if __name__ == "__main__":
    unittest.main()
