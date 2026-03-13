import os
import sys
import unittest

import cudf
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.gpu.data import build_ranked_candidate_payload
from src.backtest.cpu.strategy import MagicSplitStrategy
from src.parity_sell_event_dump import (
    CandidateRankRow,
    _build_candidate_order_artifact,
    _pair_candidate_rank_rows,
)


class _RankFixturePortfolio:
    def __init__(self):
        self.initial_cash = 1_000_000.0
        self.cash = 1_000_000.0
        self.positions = {}

    def get_total_value(self, *_args, **_kwargs):
        return self.initial_cash


class _RankFixtureHandler:
    def __init__(self, *, candidate_codes, rows_by_ticker):
        self._candidate_codes = list(candidate_codes)
        self._rows_by_ticker = dict(rows_by_ticker)

    def get_previous_trading_date(self, trading_dates, current_day_idx):
        if current_day_idx <= 0:
            return None
        return pd.to_datetime(trading_dates[current_day_idx - 1])

    def get_candidates_with_tier_fallback_pit_gated(self, _signal_date, **_kwargs):
        return self._candidate_codes, "TIER_1"

    def get_stock_rows_as_of(self, tickers, *_args, **_kwargs):
        return {ticker: self._rows_by_ticker.get(ticker) for ticker in tickers}

    def get_stock_row_as_of(self, ticker, *_args, **_kwargs):
        return self._rows_by_ticker.get(ticker)

    def get_ohlc_data_on_date(self, _current_date, _ticker, *_args, **_kwargs):
        return pd.Series({"open_price": 1000.0})


class TestGpuCandidatePayloadBuilder(unittest.TestCase):
    def test_build_ranked_candidate_payload_filters_and_ranks(self):
        metrics_df = cudf.DataFrame(
            {
                "ticker_idx": [1, 0, 2, 3, 9],
                "ticker": ["B", "A", "C", "D", "X"],
                "atr_14_ratio": [0.30, 0.30, 0.10, None, 0.50],
                "market_cap": [10_000_000, 20_000_000, -1, 5_000_000, 1_000_000],
            }
        )

        candidate_indices, atrs, ranked_records = build_ranked_candidate_payload(
            valid_candidate_metrics_df=metrics_df,
            return_ranked_records=True,
        )

        self.assertEqual(candidate_indices.get().tolist(), [9, 0, 1, 2])
        self.assertEqual([round(v, 4) for v in atrs.get().tolist()], [0.5, 0.3, 0.3, 0.1])
        self.assertEqual([row[0] for row in ranked_records], ["X", "A", "B", "C"])

    def test_build_ranked_candidate_payload_uses_ticker_as_last_tiebreaker(self):
        metrics_df = cudf.DataFrame(
            {
                "ticker_idx": [1, 0],
                "ticker": ["B", "A"],
                "atr_14_ratio": [0.20, 0.20],
                "market_cap": [7_000_000, 7_000_000],
            }
        )

        candidate_indices, _, ranked_records = build_ranked_candidate_payload(
            valid_candidate_metrics_df=metrics_df,
            return_ranked_records=True,
        )

        self.assertEqual(candidate_indices.get().tolist(), [0, 1])
        self.assertEqual([row[0] for row in ranked_records], ["A", "B"])

    def test_build_ranked_candidate_payload_prioritizes_effective_cheap_score(self):
        metrics_df = cudf.DataFrame(
            {
                "ticker_idx": [2, 0, 1],
                "ticker": ["C", "A", "B"],
                "atr_14_ratio": [0.20, 0.20, 0.20],
                "market_cap": [10_000_000, 10_000_000, 10_000_000],
                "cheap_score": [0.7, 0.9, 0.8],
                "cheap_score_confidence": [1.0, 0.5, 1.0],  # A=0.45, B=0.8, C=0.7
            }
        )

        candidate_indices, _, ranked_records = build_ranked_candidate_payload(
            valid_candidate_metrics_df=metrics_df,
            return_ranked_records=True,
        )

        self.assertEqual(candidate_indices.get().tolist(), [1, 2, 0])
        self.assertEqual([row[0] for row in ranked_records], ["B", "C", "A"])

    def test_direct_composite_rank_parity_fixture_matches_cpu_history(self):
        metrics_rows = {
            "A": pd.Series(
                {
                    "atr_14_ratio": 0.40,
                    "close_price": 1000.0,
                    "market_cap": 50_000_000,
                    "cheap_score": 0.60,
                    "cheap_score_confidence": 0.50,
                    "flow5_mcap": 20.0,
                }
            ),
            "B": pd.Series(
                {
                    "atr_14_ratio": 0.10,
                    "close_price": 1000.0,
                    "market_cap": 20_000_000,
                    "cheap_score": 0.80,
                    "cheap_score_confidence": 1.0,
                    "flow5_mcap": 10.0,
                }
            ),
            "C": pd.Series(
                {
                    "atr_14_ratio": 0.20,
                    "close_price": 1000.0,
                    "market_cap": 10_000_000,
                    "cheap_score": 0.80,
                    "cheap_score_confidence": 1.0,
                    "flow5_mcap": 30.0,
                }
            ),
            "D": pd.Series(
                {
                    "atr_14_ratio": 0.20,
                    "close_price": 1000.0,
                    "market_cap": 30_000_000,
                    "cheap_score": 0.80,
                    "cheap_score_confidence": 1.0,
                    "flow5_mcap": 30.0,
                }
            ),
        }
        strategy = MagicSplitStrategy(
            max_stocks=4,
            order_investment_ratio=0.1,
            additional_buy_drop_rate=0.05,
            sell_profit_rate=0.10,
            backtest_start_date="2024-01-01",
            backtest_end_date="2024-01-31",
            candidate_source_mode="tier",
            enable_candidate_rank_trace=True,
        )
        portfolio = _RankFixturePortfolio()
        handler = _RankFixtureHandler(
            candidate_codes=["B", "A", "D", "C"],
            rows_by_ticker=metrics_rows,
        )

        strategy.generate_new_entry_signals(
            current_date=pd.Timestamp("2024-01-02"),
            portfolio=portfolio,
            data_handler=handler,
            trading_dates=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
            current_day_idx=1,
        )

        cpu_rows = [
            CandidateRankRow(source="cpu", **row)
            for row in strategy.candidate_rank_history
        ]
        metrics_df = cudf.DataFrame(
            {
                "ticker_idx": [0, 1, 2, 3],
                "ticker": ["A", "B", "C", "D"],
                "atr_14_ratio": [0.40, 0.10, 0.20, 0.20],
                "market_cap": [50_000_000, 20_000_000, 10_000_000, 30_000_000],
                "cheap_score": [0.60, 0.80, 0.80, 0.80],
                "cheap_score_confidence": [0.50, 1.0, 1.0, 1.0],
                "flow5_mcap": [20.0, 10.0, 30.0, 30.0],
            }
        )
        _, _, ranked_records = build_ranked_candidate_payload(
            valid_candidate_metrics_df=metrics_df,
            return_ranked_records=True,
        )
        gpu_rows = [
            CandidateRankRow(
                source="gpu",
                trade_date="2024-01-02",
                signal_date="2024-01-01",
                rank=rank,
                ticker=str(row[0]),
                entry_composite_score_q=int(row[1]),
                flow_score_q=int(row[2]),
                atr_score_q=int(row[3]),
                market_cap_q=int(row[4]),
                atr_14_ratio=float(row[5]),
            )
            for rank, row in enumerate(ranked_records, start=1)
        ]

        pairs = _pair_candidate_rank_rows(cpu_rows, gpu_rows)
        artifact = _build_candidate_order_artifact(pairs)

        self.assertEqual([row.ticker for row in cpu_rows], ["D", "C", "B", "A"])
        self.assertEqual([row.ticker for row in gpu_rows], ["D", "C", "B", "A"])
        self.assertEqual(artifact["paired_count"], 4)
        self.assertEqual(artifact["mismatched_pairs"], 0)
        self.assertTrue(artifact["zero_mismatch"])


if __name__ == "__main__":
    unittest.main()
