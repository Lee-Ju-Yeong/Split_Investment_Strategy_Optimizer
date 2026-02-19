import os
import sys
import unittest

import cudf


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.gpu.data import build_ranked_candidate_payload


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

        self.assertEqual(candidate_indices.get().tolist(), [0, 1, 9, 2])
        self.assertEqual([round(v, 4) for v in atrs.get().tolist()], [0.3, 0.3, 0.5, 0.1])
        self.assertEqual([row[0] for row in ranked_records], ["A", "B", "X", "C"])

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


if __name__ == "__main__":
    unittest.main()
