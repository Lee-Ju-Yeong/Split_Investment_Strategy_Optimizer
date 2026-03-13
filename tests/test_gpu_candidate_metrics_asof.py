import os
import sys
import unittest

import cupy as cp
import cudf
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.gpu.data import (
    _collect_candidate_rank_metrics_asof,
    build_ranked_candidate_payload,
    collect_candidate_rank_metrics_from_tensors,
    create_candidate_rank_tensors,
    ensure_cheap_score_columns,
)
from src.backtest.gpu.engine import _collect_candidate_rank_metrics


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

        self.assertEqual(
            set(missing_first),
            {"cheap_score", "cheap_score_confidence", "flow5_mcap"},
        )
        self.assertEqual(missing_second, [])
        self.assertIn("cheap_score", metrics_df.columns)
        self.assertIn("cheap_score_confidence", metrics_df.columns)
        self.assertIn("flow5_mcap", metrics_df.columns)

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
        self.assertIn("flow5_mcap", metrics_df.columns)
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

    def test_tensor_gather_matches_legacy_asof_ranking_payload(self):
        trading_dates = pd.DatetimeIndex(
            ["2026-01-06", "2026-01-07", "2026-01-08"],
            name="date",
        )
        all_data = cudf.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-06",
                        "2026-01-07",
                        "2026-01-06",
                        "2026-01-08",
                    ]
                ),
                "ticker": ["A", "A", "B", "B"],
                "ticker_idx": [0, 0, 1, 1],
                "atr_14_ratio": [0.10, 0.20, 0.30, 0.40],
                "market_cap": [10_000_000, 11_000_000, 20_000_000, 21_000_000],
                "cheap_score": [0.40, 0.60, 0.20, 0.90],
                "cheap_score_confidence": [1.0, 1.0, 0.5, 1.0],
                "flow5_mcap": [100.0, 110.0, 200.0, 210.0],
            }
        )
        final_candidate_indices = cp.asarray([0, 1], dtype=cp.int32)

        legacy_df = _collect_candidate_rank_metrics_asof(
            all_data_reset_idx=all_data.copy(deep=True),
            final_candidate_indices=final_candidate_indices,
            signal_date=pd.Timestamp("2026-01-07"),
        )
        tensor_map = create_candidate_rank_tensors(
            all_data.copy(deep=True),
            all_tickers=["A", "B"],
            trading_dates_pd=trading_dates,
        )
        tensor_df = collect_candidate_rank_metrics_from_tensors(
            rank_metric_tensors=tensor_map,
            final_candidate_indices=final_candidate_indices,
            signal_day_idx=1,
            all_tickers=["A", "B"],
        )

        legacy_indices, legacy_atrs, legacy_records = build_ranked_candidate_payload(
            legacy_df,
            return_ranked_records=True,
        )
        tensor_indices, tensor_atrs, tensor_records = build_ranked_candidate_payload(
            tensor_df,
            return_ranked_records=True,
        )

        self.assertEqual(legacy_indices.get().tolist(), tensor_indices.get().tolist())
        self.assertEqual(
            [round(float(value), 6) for value in legacy_atrs.get().tolist()],
            [round(float(value), 6) for value in tensor_atrs.get().tolist()],
        )
        self.assertEqual(legacy_records, tensor_records)

    def test_tensor_gather_without_ticker_strings_matches_legacy_live_payload(self):
        trading_dates = pd.DatetimeIndex(
            ["2026-01-06", "2026-01-07", "2026-01-08"],
            name="date",
        )
        all_data = cudf.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-06",
                        "2026-01-07",
                        "2026-01-06",
                        "2026-01-08",
                    ]
                ),
                "ticker": ["A", "A", "B", "B"],
                "ticker_idx": [0, 0, 1, 1],
                "atr_14_ratio": [0.10, 0.20, 0.30, 0.40],
                "market_cap": [10_000_000, 11_000_000, 20_000_000, 21_000_000],
                "cheap_score": [0.40, 0.60, 0.20, 0.90],
                "cheap_score_confidence": [1.0, 1.0, 0.5, 1.0],
                "flow5_mcap": [100.0, 110.0, 200.0, 210.0],
            }
        )
        final_candidate_indices = cp.asarray([0, 1], dtype=cp.int32)

        legacy_df = _collect_candidate_rank_metrics_asof(
            all_data_reset_idx=all_data.copy(deep=True),
            final_candidate_indices=final_candidate_indices,
            signal_date=pd.Timestamp("2026-01-07"),
        )
        tensor_map = create_candidate_rank_tensors(
            all_data.copy(deep=True),
            all_tickers=["A", "B"],
            trading_dates_pd=trading_dates,
        )
        tensor_df = collect_candidate_rank_metrics_from_tensors(
            rank_metric_tensors=tensor_map,
            final_candidate_indices=final_candidate_indices,
            signal_day_idx=1,
            all_tickers=["A", "B"],
            include_ticker_strings=False,
        )

        self.assertNotIn("ticker", tensor_df.columns)
        self.assertIn("ticker_rank", tensor_df.columns)

        legacy_indices, legacy_atrs, _ = build_ranked_candidate_payload(
            legacy_df,
            return_ranked_records=False,
        )
        tensor_indices, tensor_atrs, _ = build_ranked_candidate_payload(
            tensor_df,
            return_ranked_records=False,
        )

        self.assertEqual(legacy_indices.get().tolist(), tensor_indices.get().tolist())
        self.assertEqual(
            [round(float(value), 6) for value in legacy_atrs.get().tolist()],
            [round(float(value), 6) for value in tensor_atrs.get().tolist()],
        )

    def test_tensor_gather_preserves_nan_flow_ranking_semantics(self):
        trading_dates = pd.DatetimeIndex(
            ["2026-01-06", "2026-01-07"],
            name="date",
        )
        all_data = cudf.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-07", "2026-01-07", "2026-01-07"]),
                "ticker": ["A", "B", "C"],
                "ticker_idx": [0, 1, 2],
                "atr_14_ratio": [0.10, 0.20, 0.30],
                "market_cap": [10_000_000, 11_000_000, 12_000_000],
                "cheap_score": [0.50, 0.50, 0.50],
                "cheap_score_confidence": [1.0, 1.0, 1.0],
                "flow5_mcap": [100.0, None, 300.0],
            }
        )
        final_candidate_indices = cp.asarray([0, 1, 2], dtype=cp.int32)

        legacy_df = _collect_candidate_rank_metrics_asof(
            all_data_reset_idx=all_data.copy(deep=True),
            final_candidate_indices=final_candidate_indices,
            signal_date=pd.Timestamp("2026-01-07"),
        )
        tensor_map = create_candidate_rank_tensors(
            all_data.copy(deep=True),
            all_tickers=["A", "B", "C"],
            trading_dates_pd=trading_dates,
        )
        tensor_df = collect_candidate_rank_metrics_from_tensors(
            rank_metric_tensors=tensor_map,
            final_candidate_indices=final_candidate_indices,
            signal_day_idx=1,
            all_tickers=["A", "B", "C"],
        )

        _, _, legacy_records = build_ranked_candidate_payload(
            legacy_df,
            return_ranked_records=True,
        )
        _, _, tensor_records = build_ranked_candidate_payload(
            tensor_df,
            return_ranked_records=True,
        )

        self.assertEqual(legacy_records, tensor_records)

    def test_engine_collect_candidate_rank_metrics_skips_ticker_strings_when_debug_disabled(self):
        trading_dates = pd.DatetimeIndex(["2026-01-06", "2026-01-07"], name="date")
        all_data = cudf.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-07", "2026-01-07"]),
                "ticker": ["A", "B"],
                "ticker_idx": [0, 1],
                "atr_14_ratio": [0.10, 0.20],
                "market_cap": [10_000_000, 20_000_000],
                "cheap_score": [0.50, 0.60],
                "cheap_score_confidence": [1.0, 1.0],
                "flow5_mcap": [100.0, 200.0],
            }
        )
        tensor_map = create_candidate_rank_tensors(
            all_data.copy(deep=True),
            all_tickers=["A", "B"],
            trading_dates_pd=trading_dates,
        )

        metrics_df = _collect_candidate_rank_metrics(
            all_data_reset_idx=all_data.copy(deep=True),
            candidate_rank_tensors=tensor_map,
            final_candidate_indices=cp.asarray([0, 1], dtype=cp.int32),
            signal_date=pd.Timestamp("2026-01-07"),
            signal_day_idx=1,
            all_tickers=["A", "B"],
            include_ticker_strings=False,
        )

        self.assertIsNotNone(metrics_df)
        self.assertNotIn("ticker", metrics_df.columns)
        self.assertIn("ticker_rank", metrics_df.columns)


if __name__ == "__main__":
    unittest.main()
