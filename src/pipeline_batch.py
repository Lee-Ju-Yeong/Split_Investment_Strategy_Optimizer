"""
pipeline_batch.py

Batch orchestrator for:
- FinancialData collection
- InvestorTradingTrend collection
- DailyStockTier pre-calculation
"""

import argparse
from datetime import datetime

from .db_setup import create_tables, get_db_connection
from .daily_stock_tier_batch import run_daily_stock_tier_batch
from .financial_collector import run_financial_batch
from .investor_trading_collector import run_investor_trading_batch


def run_pipeline_batch(
    conn,
    mode="daily",
    start_date_str=None,
    end_date_str=None,
    run_financial=True,
    run_investor=True,
    run_tier=True,
    lookback_days=20,
    financial_lag_days=45,
):
    if mode not in {"daily", "backfill"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == "backfill" and not start_date_str:
        raise ValueError("`start_date_str` is required in backfill mode (YYYYMMDD).")

    if end_date_str is None:
        end_date_str = datetime.today().strftime("%Y%m%d")

    summary = {}
    if run_financial:
        summary["financial"] = run_financial_batch(
            conn=conn,
            mode=mode,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
        )

    if run_investor:
        summary["investor"] = run_investor_trading_batch(
            conn=conn,
            mode=mode,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
        )

    if run_tier:
        summary["tier"] = run_daily_stock_tier_batch(
            conn=conn,
            mode=mode,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            lookback_days=lookback_days,
            financial_lag_days=financial_lag_days,
        )

    return summary


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Run batch pipeline for financial/investor/tier data.")
    parser.add_argument(
        "--mode",
        choices=["daily", "backfill"],
        default="daily",
        help="Batch mode: daily incremental or historical backfill.",
    )
    parser.add_argument(
        "--start-date",
        dest="start_date",
        default=None,
        help="Start date in YYYYMMDD (required for backfill).",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        default=datetime.today().strftime("%Y%m%d"),
        help="End date in YYYYMMDD.",
    )
    parser.add_argument(
        "--skip-financial",
        action="store_true",
        help="Skip FinancialData collection.",
    )
    parser.add_argument(
        "--skip-investor",
        action="store_true",
        help="Skip InvestorTradingTrend collection.",
    )
    parser.add_argument(
        "--skip-tier",
        action="store_true",
        help="Skip DailyStockTier pre-calculation.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=20,
        help="Rolling lookback days for tier liquidity calculation.",
    )
    parser.add_argument(
        "--financial-lag-days",
        type=int,
        default=45,
        help="Lag days for financial data alignment in tier calculation.",
    )
    return parser


def main():
    args = _build_arg_parser().parse_args()

    conn = None
    try:
        conn = get_db_connection()
        create_tables(conn)

        summary = run_pipeline_batch(
            conn=conn,
            mode=args.mode,
            start_date_str=args.start_date,
            end_date_str=args.end_date,
            run_financial=not args.skip_financial,
            run_investor=not args.skip_investor,
            run_tier=not args.skip_tier,
            lookback_days=args.lookback_days,
            financial_lag_days=args.financial_lag_days,
        )
        print("[pipeline_batch] completed")
        for key, value in summary.items():
            print(f"  - {key}: {value}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()

