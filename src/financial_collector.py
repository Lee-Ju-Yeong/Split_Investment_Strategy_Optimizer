"""
financial_collector.py

Collects financial factor snapshots into FinancialData table.
Supports both backfill and daily incremental modes.
"""

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

import numpy as np
import pandas as pd

# Suppress FutureWarning: Downcasting behavior in `replace` is deprecated
pd.set_option('future.no_silent_downcasting', True)

API_CALL_DELAY = 0.3
DEFAULT_START_DATE_STR = "19800101"
DEFAULT_LOG_INTERVAL = 50
DEFAULT_WORKERS = 4
DEFAULT_WRITE_BATCH_SIZE = 20000


def get_financial_ticker_universe(conn, end_date=None, mode="daily"):
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
        else:
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


def get_latest_financial_dates(conn, ticker_codes, chunk_size=1000):
    if not ticker_codes:
        return {}

    latest_dates = {}
    with conn.cursor() as cur:
        for start_index in range(0, len(ticker_codes), chunk_size):
            chunk = ticker_codes[start_index : start_index + chunk_size]
            placeholders = ", ".join(["%s"] * len(chunk))
            cur.execute(
                f"""
                SELECT stock_code, MAX(date)
                FROM FinancialData
                WHERE stock_code IN ({placeholders})
                GROUP BY stock_code
                """,
                chunk,
            )
            for stock_code, latest_date in cur.fetchall():
                if latest_date:
                    latest_dates[stock_code] = pd.to_datetime(latest_date).date()
    return latest_dates


def _resolve_column(df, candidates):
    normalized = {str(col).strip().upper(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().upper()
        if key in normalized:
            return normalized[key]
    return None


def normalize_fundamental_df(df_fundamental, ticker_code):
    if df_fundamental is None or df_fundamental.empty:
        return pd.DataFrame()

    df = df_fundamental.copy()
    df = df.reset_index()
    date_column = _resolve_column(df, ["날짜", "DATE"])
    if date_column is None:
        return pd.DataFrame()

    per_col = _resolve_column(df, ["PER"])
    pbr_col = _resolve_column(df, ["PBR"])
    eps_col = _resolve_column(df, ["EPS"])
    bps_col = _resolve_column(df, ["BPS"])
    dps_col = _resolve_column(df, ["DPS"])
    div_col = _resolve_column(df, ["DIV", "DIV_YIELD", "DIVIDEND_YIELD"])

    output = pd.DataFrame()
    output["date"] = pd.to_datetime(df[date_column]).dt.strftime("%Y-%m-%d")
    output["stock_code"] = ticker_code
    output["per"] = df[per_col] if per_col else np.nan
    output["pbr"] = df[pbr_col] if pbr_col else np.nan
    output["eps"] = df[eps_col] if eps_col else np.nan
    output["bps"] = df[bps_col] if bps_col else np.nan
    output["dps"] = df[dps_col] if dps_col else np.nan
    output["div_yield"] = df[div_col] if div_col else np.nan

    numeric_cols = ["per", "pbr", "eps", "bps", "dps", "div_yield"]
    for col in numeric_cols:
        output[col] = pd.to_numeric(output[col], errors="coerce")

    eps_numeric = pd.to_numeric(output["eps"], errors="coerce")
    bps_numeric = pd.to_numeric(output["bps"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        roe = np.where(
            bps_numeric > 0,
            (eps_numeric / bps_numeric) * 100.0,
            np.nan,
        )
    output["roe"] = roe
    output.replace([np.inf, -np.inf], np.nan, inplace=True)
    value_cols = ["per", "pbr", "eps", "bps", "dps", "div_yield", "roe"]
    all_zero_mask = output[value_cols].fillna(0).eq(0).all(axis=1)
    output.loc[all_zero_mask, value_cols] = np.nan
    output = output.loc[output[value_cols].notna().any(axis=1)].copy()
    if output.empty:
        return pd.DataFrame()
    output["source"] = "pykrx"
    output.replace({np.nan: None}, inplace=True)
    return output


def upsert_financial_rows(conn, rows):
    if not rows:
        return 0

    sql = """
        INSERT INTO FinancialData (
            stock_code, date, per, pbr, eps, bps, dps, div_yield, roe, source
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            per = VALUES(per),
            pbr = VALUES(pbr),
            eps = VALUES(eps),
            bps = VALUES(bps),
            dps = VALUES(dps),
            div_yield = VALUES(div_yield),
            roe = VALUES(roe),
            source = VALUES(source),
            updated_at = CURRENT_TIMESTAMP
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
        affected = cur.rowcount
    conn.commit()
    return affected


def _format_duration(seconds):
    safe_seconds = max(int(seconds), 0)
    return str(timedelta(seconds=safe_seconds))


def _estimate_eta(elapsed_seconds, done_count, total_count):
    if done_count <= 0 or total_count <= done_count:
        return "00:00:00"
    per_item = elapsed_seconds / done_count
    remaining = total_count - done_count
    return _format_duration(per_item * remaining)


def _resolve_effective_starts(conn, ticker_codes, mode, start_date_str, end_date):
    effective_starts = {}
    skipped = 0

    if start_date_str:
        start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
        for ticker_code in ticker_codes:
            if start_date > end_date:
                skipped += 1
                continue
            effective_starts[ticker_code] = start_date
        return effective_starts, skipped

    if mode == "backfill":
        start_date = datetime.strptime(DEFAULT_START_DATE_STR, "%Y%m%d").date()
        for ticker_code in ticker_codes:
            if start_date > end_date:
                skipped += 1
                continue
            effective_starts[ticker_code] = start_date
        return effective_starts, skipped

    latest_by_ticker = get_latest_financial_dates(conn, ticker_codes)
    min_start_date = datetime.strptime(DEFAULT_START_DATE_STR, "%Y%m%d").date()
    for ticker_code in ticker_codes:
        latest = latest_by_ticker.get(ticker_code)
        if latest is None:
            effective_starts[ticker_code] = min_start_date
            continue
        if latest >= end_date:
            skipped += 1
            continue
        effective_starts[ticker_code] = latest + timedelta(days=1)
    return effective_starts, skipped


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


def _fetch_and_normalize_fundamental(ticker_code, effective_start, end_date_str, wait_slot):
    from pykrx import stock

    wait_slot()
    df_fundamental = stock.get_market_fundamental(
        effective_start.strftime("%Y%m%d"),
        end_date_str,
        ticker_code,
    )
    normalized = normalize_fundamental_df(df_fundamental, ticker_code)
    if normalized.empty:
        return []
    return normalized[
        [
            "stock_code",
            "date",
            "per",
            "pbr",
            "eps",
            "bps",
            "dps",
            "div_yield",
            "roe",
            "source",
        ]
    ].to_records(index=False).tolist()


def run_financial_batch(
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
    """
    Executes financial data collection for the given mode.
    Returns summary dict.
    """
    if mode not in {"daily", "backfill"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if end_date_str is None:
        end_date_str = datetime.today().strftime("%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d").date()

    if ticker_codes is None:
        ticker_codes = get_financial_ticker_universe(
            conn,
            end_date=end_date,
            mode=mode,
        )

    summary = {
        "tickers_total": len(ticker_codes),
        "tickers_processed": 0,
        "tickers_skipped": 0,
        "rows_saved": 0,
        "errors": 0,
    }

    if not ticker_codes:
        return summary

    effective_starts, skipped_count = _resolve_effective_starts(
        conn=conn,
        ticker_codes=ticker_codes,
        mode=mode,
        start_date_str=start_date_str,
        end_date=end_date,
    )
    summary["tickers_skipped"] = skipped_count
    target_items = list(effective_starts.items())
    if not target_items:
        return summary

    started_at = time.time()
    total_tickers = len(ticker_codes)
    print(
        f"[financial_collector] start mode={mode}, "
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
        saved = upsert_financial_rows(conn, row_buffer)
        summary["rows_saved"] += max(saved, 0)
        row_buffer = []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                _fetch_and_normalize_fundamental,
                ticker_code,
                effective_start,
                end_date_str,
                wait_slot,
            ): ticker_code
            for ticker_code, effective_start in target_items
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
            except Exception as e:
                print(f"[financial_collector] Error processing {ticker_code}: {e}")
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
                        f"[financial_collector] progress {completed_count}/{total_tickers} "
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
        f"[financial_collector] completed "
        f"processed={summary['tickers_processed']} "
        f"skipped={summary['tickers_skipped']} "
        f"rows_saved={summary['rows_saved']} "
        f"errors={summary['errors']} "
        f"elapsed={_format_duration(total_elapsed)}"
    )
    return summary
