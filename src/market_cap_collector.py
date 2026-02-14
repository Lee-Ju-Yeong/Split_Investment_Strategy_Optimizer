"""
market_cap_collector.py

Collects daily market cap snapshots into MarketCapDaily table.
Supports both backfill and daily incremental modes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import threading
import time

import numpy as np
import pandas as pd


# Suppress FutureWarning: Downcasting behavior in `replace` is deprecated
pd.set_option("future.no_silent_downcasting", True)

API_CALL_DELAY = 0.3
DEFAULT_START_DATE_STR = "19950101"
DEFAULT_LOG_INTERVAL = 50
DEFAULT_WORKERS = 4
DEFAULT_WRITE_BATCH_SIZE = 20000


def get_market_cap_ticker_universe(conn, end_date=None, mode="daily"):
    """
    Resolves ticker universe consistently by mode.
    - backfill: TickerUniverseHistory
    - daily: TickerUniverseSnapshot(as-of end_date) -> active history(as-of) fallback
    - legacy fallback: WeeklyFilteredStocks -> CompanyInfo
    """
    with conn.cursor() as cur:
        if mode == "backfill":
            if end_date is None:
                cur.execute(
                    """
                    SELECT stock_code
                    FROM TickerUniverseHistory
                    ORDER BY listed_date, stock_code
                    """
                )
            else:
                cur.execute(
                    """
                    SELECT stock_code
                    FROM TickerUniverseHistory
                    WHERE listed_date <= %s
                    ORDER BY listed_date, stock_code
                    """,
                    (end_date,),
                )
            rows = cur.fetchall()
            if rows:
                return [row[0] for row in rows]

        if end_date is None:
            cur.execute(
                """
                SELECT stock_code
                FROM TickerUniverseSnapshot
                WHERE snapshot_date = (
                    SELECT MAX(snapshot_date) FROM TickerUniverseSnapshot
                )
                ORDER BY stock_code
                """
            )
        else:
            cur.execute(
                """
                SELECT stock_code
                FROM TickerUniverseSnapshot
                WHERE snapshot_date = (
                    SELECT MAX(snapshot_date)
                    FROM TickerUniverseSnapshot
                    WHERE snapshot_date <= %s
                )
                ORDER BY stock_code
                """,
                (end_date,),
            )
        rows = cur.fetchall()
        if rows:
            return [row[0] for row in rows]

        if end_date is not None:
            cur.execute(
                """
                SELECT stock_code
                FROM TickerUniverseHistory
                WHERE listed_date <= %s
                  AND COALESCE(delisted_date, last_seen_date) >= %s
                ORDER BY stock_code
                """,
                (end_date, end_date),
            )
            rows = cur.fetchall()
            if rows:
                return [row[0] for row in rows]

        if end_date is None:
            cur.execute("SELECT DISTINCT stock_code FROM WeeklyFilteredStocks")
        else:
            cur.execute(
                """
                SELECT DISTINCT stock_code
                FROM WeeklyFilteredStocks
                WHERE filter_date <= %s
                """,
                (end_date,),
            )
        rows = cur.fetchall()
        if rows:
            return [row[0] for row in rows]

        cur.execute("SELECT stock_code FROM CompanyInfo")
        return [row[0] for row in cur.fetchall()]


def _get_latest_trading_date(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM DailyStockPrice")
            row = cur.fetchone()
        if not row or not row[0]:
            return None
        value = row[0]
        if hasattr(value, "date"):
            return value.date() if isinstance(value, datetime) else value
        return value
    except Exception:
        return None


def _cap_end_date_by_latest_trading_date(conn, end_date):
    latest_trading = _get_latest_trading_date(conn)
    if latest_trading is None:
        return end_date
    return min(end_date, latest_trading)


def get_market_cap_date_bounds(conn, ticker_codes, chunk_size=1000):
    if not ticker_codes:
        return {}, {}

    min_dates = {}
    max_dates = {}
    with conn.cursor() as cur:
        for start_index in range(0, len(ticker_codes), chunk_size):
            chunk = ticker_codes[start_index : start_index + chunk_size]
            placeholders = ", ".join(["%s"] * len(chunk))
            cur.execute(
                f"""
                SELECT stock_code, MIN(date), MAX(date)
                FROM MarketCapDaily
                WHERE stock_code IN ({placeholders})
                GROUP BY stock_code
                """,
                chunk,
            )
            for stock_code, min_date, max_date in cur.fetchall():
                if min_date:
                    min_dates[stock_code] = pd.to_datetime(min_date).date()
                if max_date:
                    max_dates[stock_code] = pd.to_datetime(max_date).date()
    return min_dates, max_dates


def _resolve_column(df, candidates):
    normalized = {str(col).strip().upper(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().upper()
        if key in normalized:
            return normalized[key]
    return None


def normalize_market_cap_df(df_market_cap, ticker_code):
    if df_market_cap is None or df_market_cap.empty:
        return pd.DataFrame()

    df = df_market_cap.copy().reset_index()
    date_col = _resolve_column(df, ["날짜", "DATE"])
    if date_col is None:
        return pd.DataFrame()

    market_cap_col = _resolve_column(df, ["시가총액", "MARKET_CAP", "MKTCAP"])
    shares_col = _resolve_column(df, ["상장주식수", "SHARES", "SHARES_OUTSTANDING"])
    if market_cap_col is None and shares_col is None:
        return pd.DataFrame()

    output = pd.DataFrame()
    output["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    output["stock_code"] = ticker_code
    output["market_cap"] = df[market_cap_col] if market_cap_col else np.nan
    output["shares_outstanding"] = df[shares_col] if shares_col else np.nan

    for col in ["market_cap", "shares_outstanding"]:
        output[col] = pd.to_numeric(output[col], errors="coerce")

    all_zero_mask = output[["market_cap", "shares_outstanding"]].fillna(0).eq(0).all(axis=1)
    output = output.loc[~all_zero_mask].copy()
    if output.empty:
        return pd.DataFrame()

    for col in ["market_cap", "shares_outstanding"]:
        output[col] = output[col].round().astype("Int64")

    output["source"] = "pykrx"
    output.replace({np.nan: None}, inplace=True)
    return output[["stock_code", "date", "market_cap", "shares_outstanding", "source"]]


def upsert_market_cap_rows(conn, rows):
    if not rows:
        return 0

    sql = """
        INSERT INTO MarketCapDaily (
            stock_code, date, market_cap, shares_outstanding, source
        )
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            market_cap = VALUES(market_cap),
            shares_outstanding = VALUES(shares_outstanding),
            source = VALUES(source),
            updated_at = CURRENT_TIMESTAMP
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
        affected = cur.rowcount
    conn.commit()
    return max(affected, 0)


def _format_duration(seconds):
    safe_seconds = max(int(seconds), 0)
    return str(timedelta(seconds=safe_seconds))


def _estimate_eta(elapsed_seconds, done_count, total_count):
    if done_count <= 0 or total_count <= done_count:
        return "00:00:00"
    per_item = elapsed_seconds / done_count
    remaining = total_count - done_count
    return _format_duration(per_item * remaining)


def _resolve_effective_windows(conn, ticker_codes, mode, start_date_str, end_date):
    windows = {}
    skipped = 0

    min_start_date = datetime.strptime(DEFAULT_START_DATE_STR, "%Y%m%d").date()
    if start_date_str:
        base_start = datetime.strptime(start_date_str, "%Y%m%d").date()
    elif mode == "backfill":
        base_start = min_start_date
    else:
        base_start = None

    min_dates, max_dates = get_market_cap_date_bounds(conn, ticker_codes)
    for ticker_code in ticker_codes:
        if mode == "daily":
            latest = max_dates.get(ticker_code)
            effective_start = min_start_date if latest is None else latest + timedelta(days=1)
            effective_end = end_date
        else:
            effective_start = base_start
            effective_end = end_date
            min_existing = min_dates.get(ticker_code)
            if min_existing is not None and min_existing > effective_start:
                effective_end = min(end_date, min_existing - timedelta(days=1))

        if effective_start is None or effective_start > effective_end:
            skipped += 1
            continue
        windows[ticker_code] = (effective_start, effective_end)

    return windows, skipped


def _build_rate_limiter(api_call_delay):
    if api_call_delay <= 0:
        return lambda: None

    lock = threading.Lock()
    next_allowed = [0.0]

    def _wait_slot():
        wait_seconds = 0.0
        with lock:
            now = time.monotonic()
            if now < next_allowed[0]:
                wait_seconds = next_allowed[0] - now
            next_allowed[0] = max(now, next_allowed[0]) + api_call_delay
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    return _wait_slot


def _fetch_and_normalize_market_cap(ticker_code, effective_start, effective_end, wait_slot):
    from pykrx import stock

    wait_slot()
    df_cap = stock.get_market_cap(
        effective_start.strftime("%Y%m%d"),
        effective_end.strftime("%Y%m%d"),
        ticker_code,
    )
    normalized = normalize_market_cap_df(df_cap, ticker_code)
    if normalized.empty:
        return []

    rows = normalized.where(pd.notna(normalized), None)
    return [tuple(row) for row in rows.itertuples(index=False, name=None)]


def run_market_cap_batch(
    conn,
    mode="daily",
    start_date_str=None,
    end_date_str=None,
    ticker_codes=None,
    api_call_delay=API_CALL_DELAY,
    workers=DEFAULT_WORKERS,
    write_batch_size=DEFAULT_WRITE_BATCH_SIZE,
    log_interval=DEFAULT_LOG_INTERVAL,
):
    if mode not in {"daily", "backfill"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if end_date_str is None:
        end_date_str = datetime.today().strftime("%Y%m%d")
    requested_end = datetime.strptime(end_date_str, "%Y%m%d").date()
    end_date = _cap_end_date_by_latest_trading_date(conn, requested_end)
    end_date_str = end_date.strftime("%Y%m%d")

    if ticker_codes is None:
        ticker_codes = get_market_cap_ticker_universe(conn, end_date=end_date, mode=mode)

    summary = {
        "tickers_total": len(ticker_codes),
        "tickers_processed": 0,
        "tickers_skipped": 0,
        "rows_saved": 0,
        "errors": 0,
        "end_date": end_date_str,
    }
    if not ticker_codes:
        return summary

    effective_windows, skipped_count = _resolve_effective_windows(
        conn=conn,
        ticker_codes=ticker_codes,
        mode=mode,
        start_date_str=start_date_str,
        end_date=end_date,
    )
    summary["tickers_skipped"] = skipped_count
    target_items = list(effective_windows.items())
    if not target_items:
        return summary

    started_at = time.time()
    total_tickers = len(ticker_codes)
    print(
        f"[market_cap_collector] start mode={mode}, "
        f"tickers={total_tickers}, target_tickers={len(target_items)}, "
        f"workers={max(int(workers), 1)}, write_batch_size={max(int(write_batch_size), 1)}, "
        f"end_date={end_date_str}"
    )

    row_buffer = []
    completed_count = summary["tickers_skipped"]
    wait_slot = _build_rate_limiter(max(float(api_call_delay), 0.0))
    worker_count = max(int(workers), 1)
    batch_size = max(int(write_batch_size), 1)

    def _flush_buffer():
        nonlocal row_buffer
        if not row_buffer:
            return
        saved = upsert_market_cap_rows(conn, row_buffer)
        summary["rows_saved"] += max(saved, 0)
        row_buffer = []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                _fetch_and_normalize_market_cap,
                ticker_code,
                window[0],
                window[1],
                wait_slot,
            ): ticker_code
            for ticker_code, window in target_items
        }

        for future in as_completed(future_map):
            ticker_code = future_map[future]
            try:
                rows = future.result()
                if rows:
                    row_buffer.extend(rows)
                    if len(row_buffer) >= batch_size:
                        _flush_buffer()
                summary["tickers_processed"] += 1
            except Exception as exc:
                print(f"[market_cap_collector] Error processing {ticker_code}: {exc}")
                summary["errors"] += 1
            finally:
                completed_count += 1
                if (
                    log_interval
                    and log_interval > 0
                    and (completed_count % log_interval == 0 or completed_count == total_tickers)
                ):
                    elapsed = time.time() - started_at
                    print(
                        f"[market_cap_collector] progress {completed_count}/{total_tickers} "
                        f"({completed_count / total_tickers:.1%}) "
                        f"processed={summary['tickers_processed']} "
                        f"skipped={summary['tickers_skipped']} "
                        f"rows_saved={summary['rows_saved']} "
                        f"errors={summary['errors']} "
                        f"elapsed={_format_duration(elapsed)} "
                        f"eta={_estimate_eta(elapsed, completed_count, total_tickers)}"
                    )

    try:
        _flush_buffer()
    except Exception:
        summary["errors"] += 1
        conn.rollback()

    total_elapsed = time.time() - started_at
    print(
        f"[market_cap_collector] completed "
        f"processed={summary['tickers_processed']} "
        f"skipped={summary['tickers_skipped']} "
        f"rows_saved={summary['rows_saved']} "
        f"errors={summary['errors']} "
        f"elapsed={_format_duration(total_elapsed)}"
    )
    return summary


def main():
    import argparse

    from .db_setup import create_tables, get_db_connection

    parser = argparse.ArgumentParser(description="Collect MarketCapDaily from pykrx.")
    parser.add_argument("--mode", choices=["daily", "backfill"], default="daily")
    parser.add_argument("--start-date", dest="start_date", default=None, help="YYYYMMDD")
    parser.add_argument("--end-date", dest="end_date", default=None, help="YYYYMMDD")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--delay", type=float, default=API_CALL_DELAY)
    parser.add_argument("--write-batch-size", type=int, default=DEFAULT_WRITE_BATCH_SIZE)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--ticker-limit", type=int, default=None)
    args = parser.parse_args()

    conn = None
    try:
        conn = get_db_connection()
        create_tables(conn)

        tickers = None
        if args.ticker_limit:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT stock_code FROM TickerUniverseHistory ORDER BY stock_code LIMIT %s",
                    (int(args.ticker_limit),),
                )
                tickers = [row[0] for row in cur.fetchall()]

        run_market_cap_batch(
            conn=conn,
            mode=args.mode,
            start_date_str=args.start_date,
            end_date_str=args.end_date,
            ticker_codes=tickers,
            api_call_delay=max(float(args.delay), 0.0),
            workers=max(int(args.workers), 1),
            write_batch_size=max(int(args.write_batch_size), 1),
            log_interval=args.log_interval,
        )
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()

