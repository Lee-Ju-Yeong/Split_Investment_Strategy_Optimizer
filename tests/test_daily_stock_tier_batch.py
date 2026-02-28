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
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.20,
                }
            )
            rows.append(
                {
                    "stock_code": "A0002",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.20,
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

    def _build_quality_financial_df(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rows = []
        for date_value in dates:
            rows.append(
                {
                    "stock_code": "A0001",
                    "date": date_value,
                    "roe": 12.0,
                    "bps": 1000.0,
                    "per": 10.0,
                    "pbr": 1.0,
                    "div_yield": 1.0,
                }
            )
            rows.append(
                {
                    "stock_code": "A0002",
                    "date": date_value,
                    "roe": 12.0,
                    "bps": 1000.0,
                    "per": 10.0,
                    "pbr": 1.0,
                    "div_yield": 1.0,
                }
            )
        return pd.DataFrame(rows)

    def test_investor_overlay_disabled_keeps_tier2(self):
        output = build_daily_stock_tier_frame(
            price_df=self._build_price_df(),
            financial_df=self._build_quality_financial_df(),
            investor_df=self._build_investor_df(),
            lookback_days=2,
            financial_lag_days=0,
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
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
            financial_df=self._build_quality_financial_df(),
            investor_df=self._build_investor_df(),
            lookback_days=2,
            financial_lag_days=0,
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
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
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.20,
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
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
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

    def test_dividend_negative_demotes_tier1_to_tier2(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0010",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.20,
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
                    "div_yield": -0.1,
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
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
            cheap_score_min_obs_days=2,
        )
        latest = output[
            (output["stock_code"] == "A0010") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 2)
        self.assertIn("tier1_quality_gate_failed", str(latest["reason"]))
        self.assertIn("cheap_score", output.columns)
        self.assertIn("cheap_score_confidence", output.columns)

    def test_quality_gate_requires_minimum_roe(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0011",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.20,
                }
                for date_value in dates
            ]
        )
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0011",
                    "date": date_value,
                    "roe": 4.0,
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
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
            cheap_score_min_obs_days=2,
        )
        latest = output[
            (output["stock_code"] == "A0011") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 2)
        self.assertIn("tier1_quality_gate_failed", str(latest["reason"]))

    def test_quality_gate_requires_roe_continuity(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0012",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.20,
                }
                for date_value in dates
            ]
        )
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0012",
                    "date": date_value,
                    "roe": 6.0 if idx < 4 else 4.0,
                    "bps": 1000.0,
                    "div_yield": 0.1,
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
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=3,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
        )
        latest = output[
            (output["stock_code"] == "A0012") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 2)
        self.assertIn("tier1_quality_gate_failed", str(latest["reason"]))

    def test_quality_gate_requires_5y_position_under_threshold(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0013",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 108,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.40,
                    "price_vs_10y_low_pct": 0.20,
                }
                for date_value in dates
            ]
        )
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0013",
                    "date": date_value,
                    "roe": 6.0,
                    "bps": 1000.0,
                    "div_yield": 0.1,
                }
                for date_value in dates
            ]
        )
        output = build_daily_stock_tier_frame(
            price_df=price_df,
            financial_df=financial_df,
            investor_df=pd.DataFrame(),
            lookback_days=2,
            financial_lag_days=0,
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
        )
        latest = output[
            (output["stock_code"] == "A0013") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 2)
        self.assertIn("tier1_quality_gate_failed", str(latest["reason"]))

    def test_quality_gate_ignores_10y_position_threshold(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0014",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.90,
                }
                for date_value in dates
            ]
        )
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0014",
                    "date": date_value,
                    "roe": 6.0,
                    "bps": 1000.0,
                    "div_yield": 0.1,
                }
                for date_value in dates
            ]
        )
        output = build_daily_stock_tier_frame(
            price_df=price_df,
            financial_df=financial_df,
            investor_df=pd.DataFrame(),
            lookback_days=2,
            financial_lag_days=0,
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
        )
        latest = output[
            (output["stock_code"] == "A0014") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 1)
        self.assertNotIn("tier1_quality_gate_failed", str(latest["reason"]))

    def test_quality_gate_before_switch_uses_raw_position(self):
        dates = pd.date_range("2014-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0015",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.90,
                    "price_vs_10y_low_pct": 0.90,
                }
                for date_value in dates
            ]
        )
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0015",
                    "date": date_value,
                    "roe": 6.0,
                    "bps": 1000.0,
                    "div_yield": 0.1,
                }
                for date_value in dates
            ]
        )
        output = build_daily_stock_tier_frame(
            price_df=price_df,
            financial_df=financial_df,
            investor_df=pd.DataFrame(),
            lookback_days=2,
            financial_lag_days=0,
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
            tier1_position_gate_start_date="2014-11-20",
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
        )
        latest = output[
            (output["stock_code"] == "A0015") & (output["date"] == "2014-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 2)
        self.assertIn("tier1_quality_gate_failed", str(latest["reason"]))

    def test_quality_gate_after_switch_uses_adjusted_position(self):
        dates = pd.date_range("2014-11-20", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0016",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.90,
                    "price_vs_10y_low_pct": 0.90,
                }
                for date_value in dates
            ]
        )
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0016",
                    "date": date_value,
                    "roe": 6.0,
                    "bps": 1000.0,
                    "div_yield": 0.1,
                }
                for date_value in dates
            ]
        )
        output = build_daily_stock_tier_frame(
            price_df=price_df,
            financial_df=financial_df,
            investor_df=pd.DataFrame(),
            lookback_days=2,
            financial_lag_days=0,
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
            tier1_position_gate_start_date="2014-11-20",
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
        )
        latest = output[
            (output["stock_code"] == "A0016") & (output["date"] == "2014-11-24")
        ].iloc[0]
        self.assertEqual(latest["tier"], 1)
        self.assertNotIn("tier1_quality_gate_failed", str(latest["reason"]))

    def test_financial_distress_2y_demotes_to_tier3(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0017",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.20,
                }
                for date_value in dates
            ]
        )
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0017",
                    "date": date_value,
                    "roe": 0.0,
                    "bps": 1000.0,
                    "div_yield": 0.1,
                }
                for date_value in dates
            ]
        )
        output = build_daily_stock_tier_frame(
            price_df=price_df,
            financial_df=financial_df,
            investor_df=pd.DataFrame(),
            lookback_days=2,
            financial_lag_days=0,
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
        )
        latest = output[
            (output["stock_code"] == "A0017") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 3)
        self.assertIn("financial_distress_2y", str(latest["reason"]))

    def test_financial_distress_2y_uses_daily_or_then_continuity(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0017B",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.20,
                }
                for date_value in dates
            ]
        )
        # Daily OR는 항상 True지만, roe/bps 각각은 2일 연속 음수 조건을 만족하지 않도록 교차 구성
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0017B",
                    "date": dates[0],
                    "roe": 0.0,
                    "bps": 1000.0,
                    "div_yield": 0.1,
                },
                {
                    "stock_code": "A0017B",
                    "date": dates[1],
                    "roe": 6.0,
                    "bps": 0.0,
                    "div_yield": 0.1,
                },
                {
                    "stock_code": "A0017B",
                    "date": dates[2],
                    "roe": 0.0,
                    "bps": 1000.0,
                    "div_yield": 0.1,
                },
                {
                    "stock_code": "A0017B",
                    "date": dates[3],
                    "roe": 6.0,
                    "bps": 0.0,
                    "div_yield": 0.1,
                },
                {
                    "stock_code": "A0017B",
                    "date": dates[4],
                    "roe": 0.0,
                    "bps": 1000.0,
                    "div_yield": 0.1,
                },
            ]
        )
        output = build_daily_stock_tier_frame(
            price_df=price_df,
            financial_df=financial_df,
            investor_df=pd.DataFrame(),
            lookback_days=2,
            financial_lag_days=0,
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
        )
        latest = output[
            (output["stock_code"] == "A0017B") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 3)
        self.assertIn("financial_distress_2y", str(latest["reason"]))

    def test_flow20_mcap_tail_demotes_to_tier3(self):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        price_rows = []
        fin_rows = []
        inv_rows = []
        mcap_rows = []
        for date_value in dates:
            for code in ["A0018", "A0019"]:
                price_rows.append(
                    {
                        "stock_code": code,
                        "date": date_value,
                        "close_price": 100,
                        "volume": 20_000_000,
                        "high_price": 110,
                        "low_price": 90,
                        "adj_close": 92,
                        "adj_high": 110,
                        "adj_low": 90,
                        "price_vs_5y_low_pct": 0.20,
                        "price_vs_10y_low_pct": 0.20,
                    }
                )
                fin_rows.append(
                    {
                        "stock_code": code,
                        "date": date_value,
                        "roe": 6.0,
                        "bps": 1000.0,
                        "div_yield": 0.1,
                    }
                )
                mcap_rows.append(
                    {
                        "stock_code": code,
                        "date": date_value,
                        "market_cap": 1_000_000_000,
                    }
                )
            inv_rows.append(
                {
                    "stock_code": "A0018",
                    "date": date_value,
                    "foreigner_net_buy": -500_000_000,
                    "institution_net_buy": -500_000_000,
                }
            )
            inv_rows.append(
                {
                    "stock_code": "A0019",
                    "date": date_value,
                    "foreigner_net_buy": 500_000_000,
                    "institution_net_buy": 500_000_000,
                }
            )
        output = build_daily_stock_tier_frame(
            price_df=pd.DataFrame(price_rows),
            financial_df=pd.DataFrame(fin_rows),
            investor_df=pd.DataFrame(inv_rows),
            market_cap_df=pd.DataFrame(mcap_rows),
            lookback_days=2,
            financial_lag_days=0,
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
            tier3_flow20_quantile=0.5,
            tier3_flow20_valid_coverage_threshold=1.0,
            tier3_flow20_consecutive_days=3,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
        )
        latest_low_flow = output[
            (output["stock_code"] == "A0018") & (output["date"] == "2024-01-25")
        ].iloc[0]
        latest_high_flow = output[
            (output["stock_code"] == "A0019") & (output["date"] == "2024-01-25")
        ].iloc[0]
        self.assertEqual(latest_low_flow["tier"], 3)
        self.assertIn("flow20_mcap_tail", str(latest_low_flow["reason"]))
        self.assertIn(int(latest_high_flow["tier"]), (1, 2))

    def test_sbv_ratio_elevated_demotes_tier1_to_tier2(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0020",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.20,
                }
                for date_value in dates
            ]
        )
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0020",
                    "date": date_value,
                    "roe": 12.0,
                    "bps": 1000.0,
                    "div_yield": 1.0,
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
            financial_df=financial_df,
            investor_df=pd.DataFrame(),
            short_balance_df=short_balance_df,
            lookback_days=2,
            financial_lag_days=0,
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
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
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.20,
                }
                for date_value in dates
            ]
        )
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0030",
                    "date": date_value,
                    "roe": 12.0,
                    "bps": 1000.0,
                    "div_yield": 1.0,
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
            financial_df=financial_df,
            investor_df=pd.DataFrame(),
            short_balance_df=short_balance_df,
            lookback_days=2,
            financial_lag_days=0,
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
        )
        latest = output[
            (output["stock_code"] == "A0030") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 3)
        self.assertIn("sbv_ratio_extreme", str(latest["reason"]))
        self.assertGreater(float(latest["sbv_ratio"]), 0.0272)

    def test_sbv_overlay_skips_when_daily_coverage_is_low(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        price_rows = []
        for date_value in dates:
            price_rows.append(
                {
                    "stock_code": "A0040",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.20,
                }
            )
            price_rows.append(
                {
                    "stock_code": "A0041",
                    "date": date_value,
                    "close_price": 100,
                    "volume": 20_000_000,
                    "high_price": 110,
                    "low_price": 90,
                    "adj_close": 92,
                    "adj_high": 110,
                    "adj_low": 90,
                    "price_vs_5y_low_pct": 0.20,
                    "price_vs_10y_low_pct": 0.20,
                }
            )
        price_df = pd.DataFrame(price_rows)
        financial_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0040",
                    "date": date_value,
                    "roe": 12.0,
                    "bps": 1000.0,
                    "div_yield": 1.0,
                }
                for date_value in dates
            ]
            + [
                {
                    "stock_code": "A0041",
                    "date": date_value,
                    "roe": 12.0,
                    "bps": 1000.0,
                    "div_yield": 1.0,
                }
                for date_value in dates
            ]
        )
        short_balance_df = pd.DataFrame(
            [
                {
                    "stock_code": "A0040",
                    "date": date_value,
                    "short_balance_value": 30_000_000,
                    "market_cap": 1_000_000_000,
                }
                for date_value in dates
            ]
        )
        output = build_daily_stock_tier_frame(
            price_df=price_df,
            financial_df=financial_df,
            investor_df=pd.DataFrame(),
            short_balance_df=short_balance_df,
            lookback_days=2,
            financial_lag_days=0,
            tier1_position_lookback_days=5,
            tier1_position_min_periods_days=2,
            tier1_quality_lookback_days=2,
            danger_liquidity=300_000_000,
            prime_liquidity=1_000_000_000,
            sbv_valid_coverage_threshold=0.9,
        )
        latest = output[
            (output["stock_code"] == "A0040") & (output["date"] == "2024-01-05")
        ].iloc[0]
        self.assertEqual(latest["tier"], 1)
        self.assertNotIn("sbv_ratio_extreme", str(latest["reason"]))

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
                    0.000123,
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
                "flow5_mcap",
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
                "flow5_mcap",
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
        self.assertIsNone(written[6])   # flow5_mcap
        self.assertIsNone(written[7])   # pbr_discount
        self.assertIsNone(written[8])   # per_discount
        self.assertIsNone(written[9])   # div_premium
        self.assertIsNone(written[10])  # cheap_score
        self.assertEqual(written[11], "cheap_v1")  # cheap_score_version
        self.assertIsNone(written[12])  # cheap_score_confidence


if __name__ == "__main__":
    unittest.main()
