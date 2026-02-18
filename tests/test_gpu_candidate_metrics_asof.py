import os
import sys
import unittest

import cudf
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.gpu.data import _collect_candidate_rank_metrics_asof


class TestGpuCandidateMetricsAsOf(unittest.TestCase):
    def test_collect_candidate_rank_metrics_asof_picks_latest_row_per_ticker(self):
        all_data = cudf.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-07",  # A old
                        "2026-01-08",  # A signal day
                        "2026-01-06",  # B as-of fallback
                        "2026-01-09",  # B future row (must be excluded)
                    ]
                ),
                "ticker": ["A", "A", "B", "B"],
                "atr_14_ratio": [0.10, 0.20, 0.30, 0.40],
                "market_cap": [10_000_000, 11_000_000, 20_000_000, 21_000_000],
            }
        )

        metrics_df = _collect_candidate_rank_metrics_asof(
            all_data_reset_idx=all_data,
            final_candidate_tickers=["A", "B", "C"],
            signal_date=pd.Timestamp("2026-01-08"),
        )

        self.assertIsNotNone(metrics_df)
        self.assertEqual(set(metrics_df.index.to_arrow().to_pylist()), {"A", "B"})
        self.assertAlmostEqual(float(metrics_df.loc["A", "atr_14_ratio"]), 0.20, places=6)
        self.assertAlmostEqual(float(metrics_df.loc["B", "atr_14_ratio"]), 0.30, places=6)

    def test_collect_candidate_rank_metrics_asof_returns_none_when_no_rows(self):
        all_data = cudf.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-10"]),
                "ticker": ["A"],
                "atr_14_ratio": [0.5],
                "market_cap": [10_000_000],
            }
        )

        metrics_df = _collect_candidate_rank_metrics_asof(
            all_data_reset_idx=all_data,
            final_candidate_tickers=["A"],
            signal_date=pd.Timestamp("2026-01-08"),
        )

        self.assertIsNone(metrics_df)


if __name__ == "__main__":
    unittest.main()
