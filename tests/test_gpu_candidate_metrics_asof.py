import os
import sys
import unittest

import cupy as cp
import cudf
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.gpu.data import _collect_candidate_rank_metrics_asof, ensure_cheap_score_columns


class TestGpuCandidateMetricsAsOf(unittest.TestCase):
    def test_ensure_cheap_score_columns_adds_defaults_once(self):
        metrics_df = cudf.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-08"]),
                "ticker": ["A"],
                "ticker_idx": [0],
                "atr_14_ratio": [0.2],
                "market_cap": [10_000_000],
            }
        )

        missing_first = ensure_cheap_score_columns(metrics_df)
        missing_second = ensure_cheap_score_columns(metrics_df)

        self.assertEqual(set(missing_first), {"cheap_score", "cheap_score_confidence"})
        self.assertEqual(missing_second, [])
        self.assertIn("cheap_score", metrics_df.columns)
        self.assertIn("cheap_score_confidence", metrics_df.columns)

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
                "ticker_idx": [0, 0, 1, 1],
                "atr_14_ratio": [0.10, 0.20, 0.30, 0.40],
                "market_cap": [10_000_000, 11_000_000, 20_000_000, 21_000_000],
            }
        )

        metrics_df = _collect_candidate_rank_metrics_asof(
            all_data_reset_idx=all_data,
            final_candidate_indices=cp.asarray([0, 1, 2], dtype=cp.int32),
            signal_date=pd.Timestamp("2026-01-08"),
        )

        self.assertIsNotNone(metrics_df)
        self.assertEqual(set(metrics_df["ticker"].to_arrow().to_pylist()), {"A", "B"})
        self.assertIn("cheap_score", metrics_df.columns)
        self.assertIn("cheap_score_confidence", metrics_df.columns)
        metrics_by_ticker = metrics_df.set_index("ticker")
        self.assertAlmostEqual(float(metrics_by_ticker.loc["A", "atr_14_ratio"]), 0.20, places=6)
        self.assertAlmostEqual(float(metrics_by_ticker.loc["B", "atr_14_ratio"]), 0.30, places=6)

    def test_collect_candidate_rank_metrics_asof_returns_none_when_no_rows(self):
        all_data = cudf.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-10"]),
                "ticker": ["A"],
                "ticker_idx": [0],
                "atr_14_ratio": [0.5],
                "market_cap": [10_000_000],
            }
        )

        metrics_df = _collect_candidate_rank_metrics_asof(
            all_data_reset_idx=all_data,
            final_candidate_indices=cp.asarray([0], dtype=cp.int32),
            signal_date=pd.Timestamp("2026-01-08"),
        )

        self.assertIsNone(metrics_df)

    def test_collect_candidate_rank_metrics_asof_returns_none_when_indices_do_not_match(self):
        all_data = cudf.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
                "ticker": ["A", "A"],
                "ticker_idx": [0, 0],
                "atr_14_ratio": [0.1, 0.2],
                "market_cap": [10_000_000, 10_000_000],
            }
        )

        metrics_df = _collect_candidate_rank_metrics_asof(
            all_data_reset_idx=all_data,
            final_candidate_indices=cp.asarray([99], dtype=cp.int32),
            signal_date=pd.Timestamp("2026-01-06"),
        )

        self.assertIsNone(metrics_df)


if __name__ == "__main__":
    unittest.main()
