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
    @patch("src.pipeline_batch.run_ticker_universe_batch")
    def test_run_pipeline_batch_calls_collectors(
        self,
        mock_run_universe,
        mock_run_financial,
        mock_run_investor,
        mock_run_tier,
    ):
        mock_run_universe.return_value = {"snapshot": {"rows_saved": 100}}
        mock_run_financial.return_value = {"rows_saved": 10}
        mock_run_investor.return_value = {"rows_saved": 20}
        mock_run_tier.return_value = {"rows_saved": 30}
        conn = MagicMock()

        summary = pipeline_batch.run_pipeline_batch(
            conn=conn,
            mode="daily",
            start_date_str=None,
            end_date_str="20260207",
            run_universe=True,
            run_financial=True,
            run_investor=True,
            run_tier=True,
            universe_markets=["KOSPI", "KOSDAQ"],
            universe_step_days=7,
            universe_workers=2,
            universe_resume=True,
            universe_with_names=False,
            universe_api_call_delay=0.1,
            lookback_days=30,
            financial_lag_days=45,
            log_interval=25,
        )

        self.assertEqual(summary["universe"]["snapshot"]["rows_saved"], 100)
        self.assertEqual(summary["financial"]["rows_saved"], 10)
        self.assertEqual(summary["investor"]["rows_saved"], 20)
        self.assertEqual(summary["tier"]["rows_saved"], 30)
        mock_run_universe.assert_called_once_with(
            conn=conn,
            mode="daily",
            start_date_str=None,
            end_date_str="20260207",
            markets=["KOSPI", "KOSDAQ"],
            step_days=7,
            workers=2,
            resume=True,
            include_names=False,
            api_call_delay=0.1,
            log_interval=25,
        )
        mock_run_financial.assert_called_once_with(
            conn=conn,
            mode="daily",
            start_date_str=None,
            end_date_str="20260207",
            workers=4,
            write_batch_size=20000,
            log_interval=25,
        )
        mock_run_investor.assert_called_once_with(
            conn=conn,
            mode="daily",
            start_date_str=None,
            end_date_str="20260207",
            workers=4,
            write_batch_size=20000,
            log_interval=25,
        )
        mock_run_tier.assert_called_once_with(
            conn=conn,
            mode="daily",
            start_date_str=None,
            end_date_str="20260207",
            lookback_days=30,
            financial_lag_days=45,
            enable_investor_v1_write=False,
            investor_flow5_threshold=-500_000_000,
        )


if __name__ == "__main__":
    unittest.main()
