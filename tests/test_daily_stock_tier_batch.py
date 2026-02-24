import os
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline.daily_stock_tier_batch import build_daily_stock_tier_frame
from src.pipeline.daily_stock_tier_batch import upsert_daily_stock_tier


class TestDailyStockTierBatch(unittest.TestCase):
    class _FakeCursor:
        def __init__(self):
            self.executed_chunks = []
            self.chunk_rows = []
            self.rowcount = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def executemany(self, _sql, chunk):
            self.executed_chunks.append(len(chunk))
            self.chunk_rows.append(chunk)
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

    def test_dividend_non_positive_demotes_tier1_to_tier2(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0010",
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
                    "stock_code": "A0010",
                    "date": date_value,
                    "roe": 5.0,
                    "bps": 1000.0,
                    "per": 10.0 + idx,
                    "pbr": 1.0 + (idx * 0.1),
                    "div_yield": 0.0,
                }
                for idx, date_value in enumerate(dates)
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
            cheap_score_min_obs_days=2,
        )
        latest = output[
            (output["stock_code"] == "A0010") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 2)
        self.assertIn("div_zero_or_negative", str(latest["reason"]))
        self.assertIn("cheap_score", output.columns)
        self.assertIn("cheap_score_confidence", output.columns)

    def test_sbv_ratio_elevated_demotes_tier1_to_tier2(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0020",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                }
                for date_value in dates
            ]
        )
        short_balance_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0020",
                    "date": date_value,
                    "short_balance_value": 20_000_000,
                    "market_cap": 1_000_000_000,
                }
                for date_value in dates
            ]
        )

        output = build_daily_stock_tier_frame(
            price_df=price_df,
            financial_df=pd.DataFrame(),
            investor_df=pd.DataFrame(),
            short_balance_df=short_balance_df,
            lookback_days=2,
            financial_lag_days=0,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
        )
        latest = output[
            (output["stock_code"] == "A0020") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 2)
        self.assertIn("sbv_ratio_elevated", str(latest["reason"]))
        self.assertGreater(float(latest["sbv_ratio"]), 0.0139)

    def test_sbv_ratio_extreme_demotes_to_tier3(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0030",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                }
                for date_value in dates
            ]
        )
        short_balance_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0030",
                    "date": date_value,
                    "short_balance_value": 30_000_000,
                    "market_cap": 1_000_000_000,
                }
                for date_value in dates
            ]
        )

        output = build_daily_stock_tier_frame(
            price_df=price_df,
            financial_df=pd.DataFrame(),
            investor_df=pd.DataFrame(),
            short_balance_df=short_balance_df,
            lookback_days=2,
            financial_lag_days=0,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
        )
        latest = output[
            (output["stock_code"] == "A0030") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 3)
        self.assertIn("sbv_ratio_extreme", str(latest["reason"]))
        self.assertGreater(float(latest["sbv_ratio"]), 0.0272)

    def test_upsert_daily_stock_tier_uses_chunked_batches(self):
        conn = self._FakeConn()
        rows = pd.DataFrame(
            [
                (
                    "2024-01-01",
                    f"A{i:05d}",
                    2,
                    "normal_liquidity",
                    1000,
                    0.002,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    "cheap_v1",
                    1.0,
                )
                for i in range(25_001)
            ],
            columns=[
                "date",
                "stock_code",
                "tier",
                "reason",
                "liquidity_20d_avg_value",
                "sbv_ratio",
                "pbr_discount",
                "per_discount",
                "div_premium",
                "cheap_score",
                "cheap_score_version",
                "cheap_score_confidence",
            ],
        )

        affected = upsert_daily_stock_tier(conn, rows, batch_size=10_000)

        self.assertEqual(affected, 25_001)
        self.assertEqual(conn.cursor_obj.executed_chunks, [10_000, 10_000, 5_001])
        self.assertTrue(conn.committed)

    def test_upsert_daily_stock_tier_normalizes_nan_inf_to_none(self):
        conn = self._FakeConn()
        rows = pd.DataFrame(
            [
                (
                    "2024-01-01",
                    "A99999",
                    2,
                    "normal_liquidity",
                    1000,
                    np.nan,
                    np.inf,
                    -np.inf,
                    np.nan,
                    np.nan,
                    "cheap_v1",
                    np.nan,
                )
            ],
            columns=[
                "date",
                "stock_code",
                "tier",
                "reason",
                "liquidity_20d_avg_value",
                "sbv_ratio",
                "pbr_discount",
                "per_discount",
                "div_premium",
                "cheap_score",
                "cheap_score_version",
                "cheap_score_confidence",
            ],
        )

        affected = upsert_daily_stock_tier(conn, rows, batch_size=10_000)

        self.assertEqual(affected, 1)
        written = conn.cursor_obj.chunk_rows[0][0]
        self.assertIsNone(written[5])   # sbv_ratio
        self.assertIsNone(written[6])   # pbr_discount
        self.assertIsNone(written[7])   # per_discount
        self.assertIsNone(written[8])   # div_premium
        self.assertIsNone(written[9])   # cheap_score
        self.assertEqual(written[10], "cheap_v1")  # cheap_score_version
        self.assertIsNone(written[11])  # cheap_score_confidence


if __name__ == "__main__":
    unittest.main()
