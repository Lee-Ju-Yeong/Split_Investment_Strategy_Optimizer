import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.tier_backfill_window import _build_arg_parser


class TestTierBackfillWindow(unittest.TestCase):
    def test_build_arg_parser_default_financial_lag_days_is_1(self):
        parser = _build_arg_parser()
        args = parser.parse_args(["--start-date", "20240101", "--end-date", "20240131"])
        self.assertEqual(args.financial_lag_days, 1)


if __name__ == "__main__":
    unittest.main()
