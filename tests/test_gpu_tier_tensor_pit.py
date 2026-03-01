import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.optimization.gpu.data_loading import _build_tier_frame, preload_tier_data_to_tensor


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

    @patch("src.optimization.gpu.data_loading._get_sql_engine", return_value="engine")
    @patch("src.optimization.gpu.data_loading._build_universe_mask_frame")
    @patch("src.optimization.gpu.data_loading._ensure_core_deps")
    @patch("src.optimization.gpu.data_loading._ensure_gpu_deps")
    @patch("pandas.read_sql")
    def test_preload_tier_tensor_applies_universe_mask(
        self,
        mock_read_sql,
        mock_gpu_deps,
        mock_core_deps,
        mock_build_mask,
        _mock_sql_engine,
    ):
        fake_cp = type(
            "_FakeCp",
            (),
            {
                "int8": np.int8,
                "zeros": staticmethod(lambda shape, dtype=None: np.zeros(shape, dtype=dtype)),
                "asarray": staticmethod(lambda arr, dtype=None: np.asarray(arr, dtype=dtype)),
            },
        )()
        mock_gpu_deps.return_value = (fake_cp, None, None, None)
        mock_core_deps.return_value = (np, pd)

        mock_read_sql.return_value = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
                "ticker": ["A", "B"],
                "tier": [1, 1],
            }
        )
        trading_dates = pd.DatetimeIndex(pd.to_datetime(["2026-01-05"]))
        mask = pd.DataFrame(
            [[True, False]],
            index=trading_dates,
            columns=["A", "B"],
            dtype=bool,
        )
        mock_build_mask.return_value = mask

        tensor = preload_tier_data_to_tensor(
            engine="mysql://dummy",
            start_date="2026-01-05",
            end_date="2026-01-05",
            all_tickers=["A", "B"],
            trading_dates_pd=trading_dates,
            universe_mode="strict_pit",
        )

        self.assertEqual(int(tensor[0, 0]), 1)
        self.assertEqual(int(tensor[0, 1]), 0)


if __name__ == "__main__":
    unittest.main()
