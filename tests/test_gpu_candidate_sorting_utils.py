import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.gpu.utils import (
    _sort_candidates_by_atr_then_ticker,
    _sort_candidates_by_market_cap_then_atr_then_ticker,
)


class TestGpuCandidateSortingUtils(unittest.TestCase):
    def test_sort_market_cap_then_atr_then_ticker_is_deterministic(self):
        candidate_records = [
            ("B", 10, 20, 0.20),
            ("A", 10, 20, 0.20),
            ("C", 20, 5, 0.50),
            ("D", 20, 5, 0.50),
            ("E", 20, 7, 0.70),
        ]

        ranked = _sort_candidates_by_market_cap_then_atr_then_ticker(candidate_records)
        self.assertEqual([item[0] for item in ranked], ["E", "C", "D", "A", "B"])

    def test_sort_atr_then_ticker_is_deterministic(self):
        candidate_pairs = [
            ("B", 0.20),
            ("A", 0.20),
            ("C", 0.30),
            ("D", 0.10),
        ]

        ranked = _sort_candidates_by_atr_then_ticker(candidate_pairs)
        self.assertEqual([item[0] for item in ranked], ["C", "A", "B", "D"])

    def test_sort_helpers_handle_empty_inputs(self):
        self.assertEqual(_sort_candidates_by_market_cap_then_atr_then_ticker([]), [])
        self.assertEqual(_sort_candidates_by_atr_then_ticker([]), [])


if __name__ == "__main__":
    unittest.main()
