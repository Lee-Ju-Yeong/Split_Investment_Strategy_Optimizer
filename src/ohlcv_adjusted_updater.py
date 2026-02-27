"""
ohlcv_adjusted_updater.py

Updates adj_close and adj_ratio in DailyStockPrice using pykrx adjusted data.
Includes anomaly guards for known sentinel-like values.
"""

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/ohlcv_adjusted_updater.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

import argparse
import time
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pykrx import stock

from .db_setup import get_db_connection

# Suppress FutureWarning: Downcasting behavior in `replace` is deprecated
pd.set_option("future.no_silent_downcasting", True)

# Default settings
API_CALL_DELAY = 0.2
DEFAULT_WORKERS = 4
DEFAULT_LOG_INTERVAL = 50
DEFAULT_WRITE_BATCH_SIZE = 10000
DEFAULT_SOFT_SENTINEL_RATIO_THRESHOLD = 1000.0

# Known problematic adjusted close values from source data.
# 9,999,999 is treated as hard-invalid.
HARD_INVALID_ADJ_CLOSE_VALUES = {9_999_999.0}
# 1,000,000 can be real for some dates/tickers, so handled as soft-sentinel.
SOFT_SENTINEL_ADJ_CLOSE_VALUE = 1_000_000.0


def _build_scope_filter(ticker_codes=None, start_date_str=None, end_date_str=None):
    clauses = []
    params = []

    if ticker_codes:
        placeholders = ", ".join(["%s"] * len(ticker_codes))
        clauses.append(f"stock_code IN ({placeholders})")
        params.extend(list(ticker_codes))

    if start_date_str and end_date_str:
        clauses.append("date BETWEEN %s AND %s")
        params.extend([start_date_str, end_date_str])
    elif start_date_str:
        clauses.append("date >= %s")
        params.append(start_date_str)
    elif end_date_str:
        clauses.append("date <= %s")
        params.append(end_date_str)

    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def _has_stored_adj_ohlc_columns(conn):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'DailyStockPrice'
                  AND COLUMN_NAME IN ('adj_open', 'adj_high', 'adj_low')
                """
            )
            row = cur.fetchone()
        return int((row[0] if row else 0) or 0) == 3
    except Exception as exc:
        print(
            "[adjusted_updater] warning: failed to detect stored adjusted OHLC columns "
            f"({type(exc).__name__}). Fallback to formula mode."
        )
        return False


def fetch_adjusted_ohlcv(ticker, start_date, end_date, wait_slot):
    """
    Fetches adjusted OHLCV from pykrx with rate limiting.
    Returns (dataframe, error_message_or_none).
    """
    wait_slot()
    try:
        df = stock.get_market_ohlcv(start_date, end_date, ticker, adjusted=True)
        return df, None
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return pd.DataFrame(), str(exc)


def is_hard_invalid_adj_close(value):
    return float(value) in HARD_INVALID_ADJ_CLOSE_VALUES


def is_soft_sentinel_adj_close(value):
    return float(value) == SOFT_SENTINEL_ADJ_CLOSE_VALUE


def update_adjusted_prices(conn, ticker, df, *, commit=True):
    if df.empty:
        return {
            "updated_rows": 0,
            "skipped_hard_invalid": 0,
            "observed_soft_sentinel": 0,
        }

    # We only care about adjusted close.
    df = df.reset_index()
    if "종가" not in df.columns or "날짜" not in df.columns:
        return {
            "updated_rows": 0,
            "skipped_hard_invalid": 0,
            "observed_soft_sentinel": 0,
        }

    df.rename(columns={"날짜": "date", "종가": "adj_close"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    skipped_hard_invalid = 0
    observed_soft_sentinel = 0
    rows = []
    for _, row in df.iterrows():
        try:
            val = float(row["adj_close"])
            if is_hard_invalid_adj_close(val):
                skipped_hard_invalid += 1
                continue
            if is_soft_sentinel_adj_close(val):
                observed_soft_sentinel += 1
            rows.append((val, ticker, row["date"]))
        except (ValueError, TypeError):
            continue

    if not rows:
        return {
            "updated_rows": 0,
            "skipped_hard_invalid": skipped_hard_invalid,
            "observed_soft_sentinel": observed_soft_sentinel,
        }

    sql = "UPDATE DailyStockPrice SET adj_close = %s WHERE stock_code = %s AND date = %s"
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    if commit:
        conn.commit()
    return {
        "updated_rows": len(rows),
        "skipped_hard_invalid": skipped_hard_invalid,
        "observed_soft_sentinel": observed_soft_sentinel,
    }


def cleanup_known_adj_anomalies(
    conn,
    soft_ratio_threshold=DEFAULT_SOFT_SENTINEL_RATIO_THRESHOLD,
    ticker_codes=None,
    start_date_str=None,
    end_date_str=None,
    *,
    commit=True,
):
    """
    Nullifies known anomaly patterns before ratio recomputation:
    - hard invalid adj_close sentinel values (9,999,999)
    - soft sentinel 1,000,000 when implied ratio is implausibly large
    """
    where_scope, scope_params = _build_scope_filter(
        ticker_codes=ticker_codes,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
    )
    has_stored_adj_ohlc = _has_stored_adj_ohlc_columns(conn)
    clear_adj_ohlc_sql = ", adj_open = NULL, adj_high = NULL, adj_low = NULL" if has_stored_adj_ohlc else ""

    hard_invalid_rows = 0
    soft_sentinel_rows = 0
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE DailyStockPrice
            SET adj_close = NULL, adj_ratio = NULL{clear_adj_ohlc_sql}
            WHERE adj_close = %s
            {where_scope}
            """,
            [9_999_999] + scope_params,
        )
        hard_invalid_rows = cur.rowcount

        cur.execute(
            f"""
            UPDATE DailyStockPrice
            SET adj_close = NULL, adj_ratio = NULL{clear_adj_ohlc_sql}
            WHERE adj_close = %s
              AND close_price > 0
              AND (adj_close / close_price) > %s
            {where_scope}
            """,
            [1_000_000, float(soft_ratio_threshold)] + scope_params,
        )
        soft_sentinel_rows = cur.rowcount

    if commit:
        conn.commit()
    return {
        "hard_invalid_rows_nullified": int(hard_invalid_rows),
        "soft_sentinel_rows_nullified": int(soft_sentinel_rows),
    }


def calculate_all_adj_ratios(
    conn,
    ticker_codes=None,
    start_date_str=None,
    end_date_str=None,
    *,
    commit=True,
    verbose=True,
):
    """
    Recomputes adj_ratio and clears invalid ratio rows.
    """
    if verbose:
        print("[adjusted_updater] calculating adj_ratios for all rows...")
    where_scope, scope_params = _build_scope_filter(
        ticker_codes=ticker_codes,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
    )
    has_stored_adj_ohlc = _has_stored_adj_ohlc_columns(conn)
    clear_adj_ohlc_sql = ", adj_open = NULL, adj_high = NULL, adj_low = NULL" if has_stored_adj_ohlc else ""

    rows_nullified = 0
    rows_updated = 0
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE DailyStockPrice
            SET adj_ratio = NULL{clear_adj_ohlc_sql}
            WHERE adj_ratio IS NOT NULL
              AND (adj_close IS NULL OR close_price <= 0)
              {where_scope}
            """,
            scope_params,
        )
        rows_nullified = cur.rowcount

        cur.execute(
            f"""
            UPDATE DailyStockPrice
            SET adj_ratio = adj_close / close_price
            WHERE adj_close IS NOT NULL
              AND close_price > 0
              AND (
                    adj_ratio IS NULL
                    OR ABS(adj_ratio - (adj_close / close_price)) > 1e-10
                  )
              {where_scope}
            """,
            scope_params,
        )
        rows_updated = cur.rowcount

    if commit:
        conn.commit()
    return {
        "rows_nullified": int(rows_nullified),
        "rows_updated": int(rows_updated),
    }


def calculate_all_adj_ohlc(
    conn,
    ticker_codes=None,
    start_date_str=None,
    end_date_str=None,
    *,
    commit=True,
    verbose=True,
):
    """
    Recomputes adj_open/adj_high/adj_low from raw OHLC * adj_ratio.
    """
    if not _has_stored_adj_ohlc_columns(conn):
        return {"skipped": True, "rows_nullified": 0, "rows_updated": 0}

    if verbose:
        print("[adjusted_updater] calculating adj_open/adj_high/adj_low...")
    where_scope, scope_params = _build_scope_filter(
        ticker_codes=ticker_codes,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
    )
    rows_nullified = 0
    rows_updated = 0
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE DailyStockPrice
            SET adj_open = NULL, adj_high = NULL, adj_low = NULL
            WHERE (adj_ratio IS NULL OR adj_close IS NULL OR close_price <= 0)
              AND (adj_open IS NOT NULL OR adj_high IS NOT NULL OR adj_low IS NOT NULL)
              {where_scope}
            """,
            scope_params,
        )
        rows_nullified = cur.rowcount

        cur.execute(
            f"""
            UPDATE DailyStockPrice
            SET
                adj_open = open_price * adj_ratio,
                adj_high = high_price * adj_ratio,
                adj_low = low_price * adj_ratio
            WHERE adj_ratio IS NOT NULL
              AND adj_close IS NOT NULL
              AND close_price > 0
              AND (
                    adj_open IS NULL
                    OR adj_high IS NULL
                    OR adj_low IS NULL
                    OR ABS(adj_open - (open_price * adj_ratio)) > 1e-5
                    OR ABS(adj_high - (high_price * adj_ratio)) > 1e-5
                    OR ABS(adj_low - (low_price * adj_ratio)) > 1e-5
                  )
              {where_scope}
            """,
            scope_params,
        )
        rows_updated = cur.rowcount

    if commit:
        conn.commit()
    return {
        "skipped": False,
        "rows_nullified": int(rows_nullified),
        "rows_updated": int(rows_updated),
    }


def collect_adj_quality_summary(conn, ticker_codes=None, start_date_str=None, end_date_str=None):
    has_adj_ohlc_columns = _has_stored_adj_ohlc_columns(conn)
    adj_ohlc_select = ""
    adj_ohlc_keys = []
    if has_adj_ohlc_columns:
        adj_ohlc_select = """
                SUM(CASE WHEN adj_open IS NOT NULL THEN 1 ELSE 0 END) AS adj_open_rows,
                SUM(CASE WHEN adj_high IS NOT NULL THEN 1 ELSE 0 END) AS adj_high_rows,
                SUM(CASE WHEN adj_low IS NOT NULL THEN 1 ELSE 0 END) AS adj_low_rows,
        """
        adj_ohlc_keys = ["adj_open_rows", "adj_high_rows", "adj_low_rows"]

    where_scope, params = _build_scope_filter(
        ticker_codes=ticker_codes,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
    )
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN adj_close IS NOT NULL THEN 1 ELSE 0 END) AS adj_close_rows,
                SUM(CASE WHEN adj_ratio IS NOT NULL THEN 1 ELSE 0 END) AS adj_ratio_rows,
                {adj_ohlc_select}
                SUM(CASE WHEN adj_close = 1000000 THEN 1 ELSE 0 END) AS sentinel_1m_rows,
                SUM(CASE WHEN adj_close = 9999999 THEN 1 ELSE 0 END) AS sentinel_9999999_rows,
                SUM(CASE WHEN adj_ratio > 100 THEN 1 ELSE 0 END) AS ratio_gt_100_rows,
                SUM(CASE WHEN adj_ratio > 1000 THEN 1 ELSE 0 END) AS ratio_gt_1000_rows,
                SUM(CASE WHEN adj_ratio < 0.1 THEN 1 ELSE 0 END) AS ratio_lt_0_1_rows
            FROM DailyStockPrice
            WHERE 1=1 {where_scope}
            """,
            params,
        )
        row = cur.fetchone()

    keys = [
        "total_rows",
        "adj_close_rows",
        "adj_ratio_rows",
        *adj_ohlc_keys,
        "sentinel_1m_rows",
        "sentinel_9999999_rows",
        "ratio_gt_100_rows",
        "ratio_gt_1000_rows",
        "ratio_lt_0_1_rows",
    ]
    return {k: int(v or 0) for k, v in zip(keys, row)}


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


def _format_duration(seconds):
    return str(timedelta(seconds=max(int(seconds), 0)))


def _estimate_eta(elapsed_seconds, done_count, total_count):
    if done_count <= 0 or total_count <= done_count:
        return "00:00:00"
    per_item = elapsed_seconds / done_count
    remaining = total_count - done_count
    return _format_duration(per_item * remaining)


def run_adjusted_update_batch(
    conn,
    start_date_str,
    end_date_str,
    ticker_codes=None,
    workers=DEFAULT_WORKERS,
    api_call_delay=API_CALL_DELAY,
    log_interval=DEFAULT_LOG_INTERVAL,
    soft_sentinel_ratio_threshold=DEFAULT_SOFT_SENTINEL_RATIO_THRESHOLD,
    enable_anomaly_cleanup=True,
):
    start_date_sql = pd.to_datetime(start_date_str).strftime("%Y-%m-%d")
    end_date_sql = pd.to_datetime(end_date_str).strftime("%Y-%m-%d")

    if ticker_codes is None:
        with conn.cursor() as cur:
            cur.execute("SELECT stock_code FROM TickerUniverseHistory ORDER BY stock_code")
            ticker_codes = [row[0] for row in cur.fetchall()]

    if not ticker_codes:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT stock_code FROM DailyStockPrice ORDER BY stock_code")
            ticker_codes = [row[0] for row in cur.fetchall()]

    total = len(ticker_codes)
    started_at = time.time()
    wait_slot = _build_rate_limiter(max(float(api_call_delay), 0.0))
    worker_count = max(int(workers), 1)

    summary = {
        "tickers_total": total,
        "tickers_processed": 0,
        "rows_updated": 0,
        "ratio_rows_updated": 0,
        "ratio_rows_nullified": 0,
        "adj_ohlc_rows_updated": 0,
        "adj_ohlc_rows_nullified": 0,
        "cleanup_hard_invalid_rows_nullified": 0,
        "cleanup_soft_sentinel_rows_nullified": 0,
        "errors": 0,
        "fetch_errors": 0,
        "fetch_error_tickers": [],
        "skipped_hard_invalid": 0,
        "observed_soft_sentinel": 0,
    }

    print(f"[adjusted_updater] starting update for {total} tickers with {worker_count} workers...")

    db_lock = threading.Lock()

    def _update_db(ticker, df):
        with db_lock:
            try:
                affected = update_adjusted_prices(conn, ticker, df, commit=False)
                cleanup_stats = {
                    "hard_invalid_rows_nullified": 0,
                    "soft_sentinel_rows_nullified": 0,
                }
                if affected["updated_rows"] > 0:
                    ticker_scope = [ticker]
                    if enable_anomaly_cleanup:
                        cleanup_stats = cleanup_known_adj_anomalies(
                            conn,
                            soft_ratio_threshold=soft_sentinel_ratio_threshold,
                            ticker_codes=ticker_scope,
                            start_date_str=start_date_sql,
                            end_date_str=end_date_sql,
                            commit=False,
                        )
                    ratio_stats = calculate_all_adj_ratios(
                        conn,
                        ticker_codes=ticker_scope,
                        start_date_str=start_date_sql,
                        end_date_str=end_date_sql,
                        commit=False,
                        verbose=False,
                    )
                    adj_ohlc_stats = calculate_all_adj_ohlc(
                        conn,
                        ticker_codes=ticker_scope,
                        start_date_str=start_date_sql,
                        end_date_str=end_date_sql,
                        commit=False,
                        verbose=False,
                    )
                else:
                    ratio_stats = {"rows_updated": 0, "rows_nullified": 0}
                    adj_ohlc_stats = {"skipped": False, "rows_updated": 0, "rows_nullified": 0}

                conn.commit()
                return {
                    **affected,
                    "cleanup_stats": cleanup_stats,
                    "ratio_stats": ratio_stats,
                    "adj_ohlc_stats": adj_ohlc_stats,
                }
            except Exception:
                conn.rollback()
                raise

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                fetch_adjusted_ohlcv,
                ticker,
                start_date_str,
                end_date_str,
                wait_slot,
            ): ticker
            for ticker in ticker_codes
        }

        for i, future in enumerate(as_completed(future_map), start=1):
            ticker = future_map[future]
            try:
                df, fetch_error = future.result()
                if fetch_error is not None:
                    summary["fetch_errors"] += 1
                    if len(summary["fetch_error_tickers"]) < 20:
                        summary["fetch_error_tickers"].append((ticker, fetch_error))
                if not df.empty:
                    affected = _update_db(ticker, df)
                    summary["rows_updated"] += affected["updated_rows"]
                    summary["skipped_hard_invalid"] += affected["skipped_hard_invalid"]
                    summary["observed_soft_sentinel"] += affected["observed_soft_sentinel"]
                    summary["ratio_rows_updated"] += affected["ratio_stats"]["rows_updated"]
                    summary["ratio_rows_nullified"] += affected["ratio_stats"]["rows_nullified"]
                    summary["adj_ohlc_rows_updated"] += affected["adj_ohlc_stats"]["rows_updated"]
                    summary["adj_ohlc_rows_nullified"] += affected["adj_ohlc_stats"]["rows_nullified"]
                    summary["cleanup_hard_invalid_rows_nullified"] += affected["cleanup_stats"]["hard_invalid_rows_nullified"]
                    summary["cleanup_soft_sentinel_rows_nullified"] += affected["cleanup_stats"]["soft_sentinel_rows_nullified"]
                summary["tickers_processed"] += 1
            except Exception as exc:
                print(f"[adjusted_updater] error ticker={ticker}: {exc}")
                summary["errors"] += 1

            if log_interval and (i % log_interval == 0 or i == total):
                elapsed = time.time() - started_at
                print(
                    f"[adjusted_updater] progress {i}/{total} ({i/total:.1%}) "
                    f"processed={summary['tickers_processed']} updated_rows={summary['rows_updated']} "
                    f"elapsed={_format_duration(elapsed)} eta={_estimate_eta(elapsed, i, total)}"
                )

    quality_summary = collect_adj_quality_summary(
        conn,
        ticker_codes=ticker_codes,
        start_date_str=start_date_sql,
        end_date_str=end_date_sql,
    )
    print(f"[adjusted_updater] quality summary: {quality_summary}")
    if summary["fetch_error_tickers"]:
        print(
            "[adjusted_updater] fetch error samples: "
            f"{summary['fetch_error_tickers']}"
        )

    total_elapsed = time.time() - started_at
    print(
        f"[adjusted_updater] completed in {_format_duration(total_elapsed)}. "
        f"processed={summary['tickers_processed']} rows_updated={summary['rows_updated']} "
        f"ratio_rows_updated={summary['ratio_rows_updated']} "
        f"adj_ohlc_rows_updated={summary['adj_ohlc_rows_updated']} "
        f"errors={summary['errors']} fetch_errors={summary['fetch_errors']} "
        f"skipped_hard_invalid={summary['skipped_hard_invalid']} "
        f"observed_soft_sentinel={summary['observed_soft_sentinel']}"
    )
    summary["cleanup_stats"] = {
        "hard_invalid_rows_nullified": int(summary["cleanup_hard_invalid_rows_nullified"]),
        "soft_sentinel_rows_nullified": int(summary["cleanup_soft_sentinel_rows_nullified"]),
    }
    summary["ratio_stats"] = {
        "rows_updated": int(summary["ratio_rows_updated"]),
        "rows_nullified": int(summary["ratio_rows_nullified"]),
    }
    summary["adj_ohlc_stats"] = {
        "rows_updated": int(summary["adj_ohlc_rows_updated"]),
        "rows_nullified": int(summary["adj_ohlc_rows_nullified"]),
    }
    summary["quality_summary"] = quality_summary
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update adjusted prices in DailyStockPrice.")
    parser.add_argument("--start-date", default="19950101", help="Start date in YYYYMMDD.")
    parser.add_argument(
        "--end-date",
        default=datetime.today().strftime("%Y%m%d"),
        help="End date in YYYYMMDD.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--delay", type=float, default=API_CALL_DELAY)
    parser.add_argument("--ticker-limit", type=int)
    parser.add_argument(
        "--soft-sentinel-ratio-threshold",
        type=float,
        default=DEFAULT_SOFT_SENTINEL_RATIO_THRESHOLD,
        help=(
            "When adj_close=1,000,000 and implied ratio exceeds this threshold, "
            "nullify as anomaly."
        ),
    )
    parser.add_argument(
        "--skip-anomaly-cleanup",
        action="store_true",
        help="Skip anomaly cleanup before adj_ratio recomputation.",
    )

    args = parser.parse_args()

    conn = get_db_connection()
    try:
        tickers = None
        if args.ticker_limit:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT stock_code FROM TickerUniverseHistory LIMIT %s",
                    (int(args.ticker_limit),),
                )
                tickers = [row[0] for row in cur.fetchall()]

        run_adjusted_update_batch(
            conn,
            args.start_date,
            args.end_date,
            ticker_codes=tickers,
            workers=args.workers,
            api_call_delay=args.delay,
            soft_sentinel_ratio_threshold=args.soft_sentinel_ratio_threshold,
            enable_anomaly_cleanup=not args.skip_anomaly_cleanup,
        )
    finally:
        conn.close()
