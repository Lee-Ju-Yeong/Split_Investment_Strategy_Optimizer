"""
pipeline_batch.py

Batch orchestrator for:
- TickerUniverseSnapshot/History build
- FinancialData collection
- InvestorTradingTrend collection
- DailyStockTier pre-calculation
"""

import argparse
from datetime import datetime
import time

from .db_setup import create_tables, get_db_connection
from .daily_stock_tier_batch import run_daily_stock_tier_batch
from .financial_collector import run_financial_batch
from .investor_trading_collector import run_investor_trading_batch
from .market_cap_collector import run_market_cap_batch
from .short_selling_collector import run_short_selling_batch
from .ticker_universe_batch import run_ticker_universe_batch


def run_pipeline_batch(
    conn,
    mode="daily",
    start_date_str=None,
    end_date_str=None,
    run_universe=False,
    run_financial=True,
    run_investor=True,
    run_market_cap=False,
    run_short_selling=False,
    run_tier=True,
    financial_workers=4,
    financial_write_batch_size=20000,
    investor_workers=4,
    investor_write_batch_size=20000,
    market_cap_workers=4,
    market_cap_write_batch_size=20000,
    short_selling_workers=4,
    short_selling_write_batch_size=20000,
    short_selling_lag_trading_days=3,
    universe_markets=None,
    universe_step_days=7,
    universe_workers=1,
    universe_resume=True,
    universe_with_names=False,
    universe_api_call_delay=0.2,
    lookback_days=20,
    financial_lag_days=45,
    tier_v1_write_enabled=False,
    tier_v1_flow5_threshold=-500_000_000,
    log_interval=50,
):
    if mode not in {"daily", "backfill"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == "backfill" and not start_date_str:
        raise ValueError("`start_date_str` is required in backfill mode (YYYYMMDD).")

    if end_date_str is None:
        end_date_str = datetime.today().strftime("%Y%m%d")

    summary = {}
    if run_universe:
        summary["universe"] = run_ticker_universe_batch(
            conn=conn,
            mode=mode,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            markets=universe_markets,
            step_days=universe_step_days,
            workers=universe_workers,
            resume=universe_resume,
            include_names=universe_with_names,
            api_call_delay=universe_api_call_delay,
            log_interval=log_interval,
        )

    if run_financial:
        summary["financial"] = run_financial_batch(
            conn=conn,
            mode=mode,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            workers=financial_workers,
            write_batch_size=financial_write_batch_size,
            log_interval=log_interval,
        )

    if run_investor:
        summary["investor"] = run_investor_trading_batch(
            conn=conn,
            mode=mode,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            workers=investor_workers,
            write_batch_size=investor_write_batch_size,
            log_interval=log_interval,
        )

    if run_market_cap:
        summary["market_cap"] = run_market_cap_batch(
            conn=conn,
            mode=mode,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            workers=market_cap_workers,
            write_batch_size=market_cap_write_batch_size,
            log_interval=log_interval,
        )

    if run_short_selling:
        summary["short_selling"] = run_short_selling_batch(
            conn=conn,
            mode=mode,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            workers=short_selling_workers,
            write_batch_size=short_selling_write_batch_size,
            log_interval=log_interval,
            lag_trading_days=short_selling_lag_trading_days,
        )

    if run_tier:
        summary["tier"] = run_daily_stock_tier_batch(
            conn=conn,
            mode=mode,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            lookback_days=lookback_days,
            financial_lag_days=financial_lag_days,
            enable_investor_v1_write=tier_v1_write_enabled,
            investor_flow5_threshold=tier_v1_flow5_threshold,
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
        "--run-universe",
        action="store_true",
        help="Run TickerUniverseSnapshot/History batch before other collectors.",
    )
    parser.add_argument(
        "--universe-markets",
        default="KOSPI,KOSDAQ",
        help="Comma-separated market list for ticker universe batch.",
    )
    parser.add_argument(
        "--universe-step-days",
        type=int,
        default=7,
        help="Snapshot interval in days for universe backfill.",
    )
    parser.add_argument(
        "--universe-workers",
        type=int,
        default=1,
        help="Worker count for parallel universe snapshot fetch.",
    )
    parser.add_argument(
        "--universe-api-call-delay",
        type=float,
        default=0.2,
        help="Sleep seconds between universe API calls.",
    )
    parser.add_argument(
        "--universe-with-names",
        action="store_true",
        help="Fetch company names in universe snapshots.",
    )
    parser.add_argument(
        "--universe-no-resume",
        action="store_true",
        help="Force recollect universe snapshot dates even when already stored.",
    )
    parser.add_argument(
        "--financial-workers",
        type=int,
        default=4,
        help="Worker count for FinancialData API fetch/normalize pipeline.",
    )
    parser.add_argument(
        "--financial-write-batch-size",
        type=int,
        default=20000,
        help="Row batch size for FinancialData upsert commits.",
    )
    parser.add_argument(
        "--investor-workers",
        type=int,
        default=4,
        help="Worker count for InvestorTradingTrend API fetch/normalize pipeline.",
    )
    parser.add_argument(
        "--investor-write-batch-size",
        type=int,
        default=20000,
        help="Row batch size for InvestorTradingTrend upsert commits.",
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
        "--run-marketcap",
        action="store_true",
        help="Run MarketCapDaily collection (pykrx get_market_cap by ticker/date-range).",
    )
    parser.add_argument(
        "--marketcap-workers",
        type=int,
        default=4,
        help="Worker count for MarketCapDaily API fetch/normalize pipeline.",
    )
    parser.add_argument(
        "--marketcap-write-batch-size",
        type=int,
        default=20000,
        help="Row batch size for MarketCapDaily upsert commits.",
    )
    parser.add_argument(
        "--run-shortsell",
        action="store_true",
        help="Run ShortSellingDaily collection (pykrx get_shorting_status_by_date by ticker).",
    )
    parser.add_argument(
        "--shortsell-workers",
        type=int,
        default=4,
        help="Worker count for ShortSellingDaily API fetch/normalize pipeline.",
    )
    parser.add_argument(
        "--shortsell-write-batch-size",
        type=int,
        default=20000,
        help="Row batch size for ShortSellingDaily upsert commits.",
    )
    parser.add_argument(
        "--shortsell-lag-trading-days",
        type=int,
        default=3,
        help="Clamp end_date by N trading days to account for short-selling publication lag (default: 3).",
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
    parser.add_argument(
        "--enable-tier-v1-write",
        action="store_true",
        help=(
            "Enable Tier v1 investor overlay write "
            "(tier2 + flow5<threshold => tier3). Default is OFF."
        ),
    )
    parser.add_argument(
        "--tier-v1-flow5-threshold",
        type=int,
        default=-500_000_000,
        help="Investor flow5 threshold used when --enable-tier-v1-write is enabled.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Progress log interval by ticker for financial/investor collectors. Set 0 to disable.",
    )
    return parser


def main():
    args = _build_arg_parser().parse_args()

    conn = None
    try:
        conn = get_db_connection()
        create_tables(conn)

        started_at = time.time()
        print(
            "[pipeline_batch] start "
            f"mode={args.mode}, start_date={args.start_date}, end_date={args.end_date}, "
            f"run_universe={args.run_universe}, "
            f"run_financial={not args.skip_financial}, "
            f"run_investor={not args.skip_investor}, "
            f"run_market_cap={args.run_marketcap}, "
            f"run_short_selling={args.run_shortsell}, "
            f"run_tier={not args.skip_tier}, "
            f"financial_workers={args.financial_workers}, "
            f"financial_write_batch_size={args.financial_write_batch_size}, "
            f"investor_workers={args.investor_workers}, "
            f"investor_write_batch_size={args.investor_write_batch_size}, "
            f"market_cap_workers={args.marketcap_workers}, "
            f"market_cap_write_batch_size={args.marketcap_write_batch_size}, "
            f"short_selling_workers={args.shortsell_workers}, "
            f"short_selling_write_batch_size={args.shortsell_write_batch_size}, "
            f"short_selling_lag_trading_days={args.shortsell_lag_trading_days}, "
            f"tier_v1_write_enabled={args.enable_tier_v1_write}, "
            f"log_interval={args.log_interval}"
        )

        summary = run_pipeline_batch(
            conn=conn,
            mode=args.mode,
            start_date_str=args.start_date,
            end_date_str=args.end_date,
            run_universe=args.run_universe,
            run_financial=not args.skip_financial,
            run_investor=not args.skip_investor,
            run_market_cap=args.run_marketcap,
            run_short_selling=args.run_shortsell,
            run_tier=not args.skip_tier,
            financial_workers=max(int(args.financial_workers), 1),
            financial_write_batch_size=max(int(args.financial_write_batch_size), 1),
            investor_workers=max(int(args.investor_workers), 1),
            investor_write_batch_size=max(int(args.investor_write_batch_size), 1),
            market_cap_workers=max(int(args.marketcap_workers), 1),
            market_cap_write_batch_size=max(int(args.marketcap_write_batch_size), 1),
            short_selling_workers=max(int(args.shortsell_workers), 1),
            short_selling_write_batch_size=max(int(args.shortsell_write_batch_size), 1),
            short_selling_lag_trading_days=max(int(args.shortsell_lag_trading_days), 0),
            universe_markets=[
                market.strip().upper()
                for market in args.universe_markets.split(",")
                if market.strip()
            ],
            universe_step_days=args.universe_step_days,
            universe_workers=max(int(args.universe_workers), 1),
            universe_resume=not args.universe_no_resume,
            universe_with_names=args.universe_with_names,
            universe_api_call_delay=max(float(args.universe_api_call_delay), 0.0),
            lookback_days=args.lookback_days,
            financial_lag_days=args.financial_lag_days,
            tier_v1_write_enabled=args.enable_tier_v1_write,
            tier_v1_flow5_threshold=args.tier_v1_flow5_threshold,
            log_interval=args.log_interval,
        )
        elapsed_seconds = int(time.time() - started_at)
        print(f"[pipeline_batch] completed elapsed={elapsed_seconds}s")
        for key, value in summary.items():
            print(f"  - {key}: {value}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
