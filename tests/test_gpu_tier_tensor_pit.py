import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.optimization.gpu.data_loading import _build_tier_frame


class TestGpuTierTensorPit(unittest.TestCase):
    def test_build_tier_frame_keeps_prestart_asof_value(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-12-28", "2026-01-07"]),
                "ticker": ["196170", "005490"],
                "tier": [1, 1],
            }
        )
        trading_dates = pd.DatetimeIndex(pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]))
        frame = _build_tier_frame(df, trading_dates, all_tickers=["196170", "005490"])

        self.assertEqual(int(frame.loc[pd.Timestamp("2026-01-05"), "196170"]), 1)
        self.assertEqual(int(frame.loc[pd.Timestamp("2026-01-06"), "196170"]), 1)
        self.assertEqual(int(frame.loc[pd.Timestamp("2026-01-07"), "196170"]), 1)

    def test_build_tier_frame_normalizes_ticker_columns_to_string(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-12-28"]),
                "ticker": ["196170"],
                "tier": [1],
            }
        )
        trading_dates = pd.DatetimeIndex(pd.to_datetime(["2026-01-05"]))
        frame = _build_tier_frame(df, trading_dates, all_tickers=[196170, "005490"])

        self.assertEqual(int(frame.loc[pd.Timestamp("2026-01-05"), "196170"]), 1)
        self.assertEqual(int(frame.loc[pd.Timestamp("2026-01-05"), "005490"]), 0)


if __name__ == "__main__":
    unittest.main()
