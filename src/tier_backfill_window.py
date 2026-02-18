"""
tier_backfill_window.py

Windowed backfill runner for DailyStockTier.

Why:
- `run_daily_stock_tier_batch` computes tiers from DailyStockPrice + FinancialData overlays.
- Long ranges can be heavy, so this script runs in date windows (chunked) and upserts.

Example:
  PYTHONPATH=$PWD conda run --no-capture-output -n rapids-env \
    python -m src.tier_backfill_window --start-date 20131120 --end-date 20231231 --chunk-days 90
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import time

import pymysql

from .config_loader import load_config
from .daily_stock_tier_batch import run_daily_stock_tier_batch


def _parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _iter_windows(start: date, end: date, chunk_days: int):
    days = max(int(chunk_days), 1)
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _connect_from_yaml(cfg: dict):
    db = cfg["database"]
    return pymysql.connect(
        host=db["host"],
        user=db["user"],
        password=db["password"],
        database=db["database"],
        charset="utf8mb4",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Windowed backfill for DailyStockTier.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD.")
    parser.add_argument("--chunk-days", type=int, default=90)
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--financial-lag-days", type=int, default=45)
    parser.add_argument("--danger-liquidity", type=int, default=300_000_000)
    parser.add_argument("--prime-liquidity", type=int, default=1_000_000_000)
    parser.add_argument("--enable-tier-v1-write", action="store_true")
    parser.add_argument("--tier-v1-flow5-threshold", type=int, default=-500_000_000)
    args = parser.parse_args()

    start_date = _parse_yyyymmdd(args.start_date)
    end_date = _parse_yyyymmdd(args.end_date)
    if start_date > end_date:
        raise ValueError(f"Invalid range: start_date({start_date}) > end_date({end_date})")

    cfg = load_config()
    conn = _connect_from_yaml(cfg)
    try:
        total_saved = 0
        total_calculated = 0
        started_at = time.time()

        for chunk_start, chunk_end in _iter_windows(start_date, end_date, chunk_days=args.chunk_days):
            chunk_started_at = time.time()
            summary = run_daily_stock_tier_batch(
                conn=conn,
                mode="backfill",
                start_date_str=chunk_start.strftime("%Y%m%d"),
                end_date_str=chunk_end.strftime("%Y%m%d"),
                lookback_days=int(args.lookback_days),
                financial_lag_days=int(args.financial_lag_days),
                danger_liquidity=int(args.danger_liquidity),
                prime_liquidity=int(args.prime_liquidity),
                enable_investor_v1_write=bool(args.enable_tier_v1_write),
                investor_flow5_threshold=int(args.tier_v1_flow5_threshold),
            )
            total_saved += int(summary.get("rows_saved") or 0)
            total_calculated += int(summary.get("rows_calculated") or 0)
            chunk_elapsed_s = int(time.time() - chunk_started_at)
            print(
                "[tier_backfill_window] "
                f"range={summary.get('start_date')}..{summary.get('end_date')} "
                f"saved={summary.get('rows_saved')} calculated={summary.get('rows_calculated')} "
                f"tier_v1_write_enabled={summary.get('tier_v1_write_enabled')} "
                f"elapsed_s={chunk_elapsed_s}"
            )

        total_elapsed_s = int(time.time() - started_at)
        print(
            "[tier_backfill_window] completed "
            f"total_saved={total_saved} total_calculated={total_calculated} "
            f"start={start_date} end={end_date} elapsed_s={total_elapsed_s}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
