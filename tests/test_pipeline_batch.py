import os
import sys
import unittest
from unittest.mock import MagicMock, patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import pipeline_batch


class TestPipelineBatch(unittest.TestCase):
    def test_backfill_requires_start_date(self):
        conn = MagicMock()
        with self.assertRaises(ValueError):
            pipeline_batch.run_pipeline_batch(
                conn=conn,
                mode="backfill",
                start_date_str=None,
                end_date_str="20260207",
            )

    @patch("src.pipeline_batch.run_daily_stock_tier_batch")
    @patch("src.pipeline_batch.run_investor_trading_batch")
    @patch("src.pipeline_batch.run_financial_batch")
    def test_run_pipeline_batch_calls_collectors(
        self,
        mock_run_financial,
        mock_run_investor,
        mock_run_tier,
    ):
        mock_run_financial.return_value = {"rows_saved": 10}
        mock_run_investor.return_value = {"rows_saved": 20}
        mock_run_tier.return_value = {"rows_saved": 30}
        conn = MagicMock()

        summary = pipeline_batch.run_pipeline_batch(
            conn=conn,
            mode="daily",
            start_date_str=None,
            end_date_str="20260207",
            run_financial=True,
            run_investor=True,
            run_tier=True,
            lookback_days=30,
            financial_lag_days=45,
        )

        self.assertEqual(summary["financial"]["rows_saved"], 10)
        self.assertEqual(summary["investor"]["rows_saved"], 20)
        self.assertEqual(summary["tier"]["rows_saved"], 30)
        mock_run_financial.assert_called_once_with(
            conn=conn,
            mode="daily",
            start_date_str=None,
            end_date_str="20260207",
        )
        mock_run_investor.assert_called_once_with(
            conn=conn,
            mode="daily",
            start_date_str=None,
            end_date_str="20260207",
        )
        mock_run_tier.assert_called_once_with(
            conn=conn,
            mode="daily",
            start_date_str=None,
            end_date_str="20260207",
            lookback_days=30,
            financial_lag_days=45,
        )


if __name__ == "__main__":
    unittest.main()

