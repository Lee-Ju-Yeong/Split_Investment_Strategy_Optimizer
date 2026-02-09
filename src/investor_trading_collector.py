"""
investor_trading_collector.py

Collects investor net-buy trend snapshots into InvestorTradingTrend table.
Supports both backfill and daily incremental modes.
"""

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

import pandas as pd


API_CALL_DELAY = 0.3
DEFAULT_START_DATE_STR = "19800101"
DEFAULT_LOG_INTERVAL = 50
DEFAULT_WORKERS = 4
DEFAULT_WRITE_BATCH_SIZE = 20000


def get_investor_ticker_universe(conn, end_date=None, mode="daily"):
    with conn.cursor() as cur:
        if mode == "backfill":
            cur.execute(
                """
                SELECT stock_code
                FROM TickerUniverseHistory
                ORDER BY listed_date, stock_code
                """
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


def get_investor_date_bounds(conn, ticker_codes, chunk_size=1000):
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
                FROM InvestorTradingTrend
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


def _resolve_column(df, contains_candidates):
    for col in df.columns:
        col_name = str(col).strip().lower()
        for key in contains_candidates:
            if key in col_name:
                return col
    return None


def _numeric_series_or_missing(df, column_name):
    if column_name is None:
        return pd.Series([pd.NA] * len(df), dtype="Float64")
    return pd.to_numeric(df[column_name], errors="coerce").astype("Float64")


def normalize_investor_df(df_trading, ticker_code):
    if df_trading is None or df_trading.empty:
        return pd.DataFrame()

    df = df_trading.copy().reset_index()
    date_col = _resolve_column(df, ["날짜", "date"])
    if date_col is None:
        return pd.DataFrame()

    individual_col = _resolve_column(df, ["개인", "individual"])
    foreigner_col = _resolve_column(df, ["외국인", "foreigner", "foreign"])
    institution_col = _resolve_column(df, ["기관", "institution"])
    if not any([individual_col, foreigner_col, institution_col]):
        return pd.DataFrame()

    output = pd.DataFrame()
    output["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    output["stock_code"] = ticker_code
    output["individual_net_buy"] = _numeric_series_or_missing(df, individual_col)
    output["foreigner_net_buy"] = _numeric_series_or_missing(df, foreigner_col)
    output["institution_net_buy"] = _numeric_series_or_missing(df, institution_col)

    value_cols = [
        "individual_net_buy",
        "foreigner_net_buy",
        "institution_net_buy",
    ]
    has_observed_value = output[value_cols].notna().any(axis=1)
    output = output.loc[has_observed_value].copy()
    if output.empty:
        return pd.DataFrame()

    all_zero_observed = output[value_cols].fillna(0).abs().sum(axis=1) == 0
    output = output.loc[~all_zero_observed].copy()
    if output.empty:
        return pd.DataFrame()

    output["total_net_buy"] = output[value_cols].sum(axis=1, min_count=1)
    for col in value_cols + ["total_net_buy"]:
        output[col] = pd.to_numeric(output[col], errors="coerce").round().astype("Int64")
    return output


def upsert_investor_rows(conn, rows):
    if not rows:
        return 0

    sql = """
        INSERT INTO InvestorTradingTrend (
            stock_code, date, individual_net_buy, foreigner_net_buy, institution_net_buy, total_net_buy
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            individual_net_buy = VALUES(individual_net_buy),
            foreigner_net_buy = VALUES(foreigner_net_buy),
            institution_net_buy = VALUES(institution_net_buy),
            total_net_buy = VALUES(total_net_buy),
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

    min_dates, max_dates = get_investor_date_bounds(conn, ticker_codes)

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


def _fetch_and_normalize_investor(
    ticker_code,
    effective_start,
    effective_end,
    wait_slot,
):
    from pykrx import stock

    wait_slot()
    df_trading = stock.get_market_trading_value_by_date(
        effective_start.strftime("%Y%m%d"),
        effective_end.strftime("%Y%m%d"),
        ticker_code,
        on="순매수",
    )
    normalized = normalize_investor_df(df_trading, ticker_code)
    if normalized.empty:
        return []
    rows = normalized[
        [
            "stock_code",
            "date",
            "individual_net_buy",
            "foreigner_net_buy",
            "institution_net_buy",
            "total_net_buy",
        ]
    ].where(pd.notna(normalized), None)
    return [tuple(row) for row in rows.itertuples(index=False, name=None)]


def run_investor_trading_batch(
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
    end_date = datetime.strptime(end_date_str, "%Y%m%d").date()

    if ticker_codes is None:
        ticker_codes = get_investor_ticker_universe(
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
        f"[investor_trading_collector] start mode={mode}, "
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
        saved = upsert_investor_rows(conn, row_buffer)
        summary["rows_saved"] += max(saved, 0)
        row_buffer = []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                _fetch_and_normalize_investor,
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
            except Exception as e:
                print(f"[investor_trading_collector] Error processing {ticker_code}: {e}")
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
                        f"[investor_trading_collector] progress {completed_count}/{total_tickers} "
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
        f"[investor_trading_collector] completed "
        f"processed={summary['tickers_processed']} "
        f"skipped={summary['tickers_skipped']} "
        f"rows_saved={summary['rows_saved']} "
        f"errors={summary['errors']} "
        f"elapsed={_format_duration(total_elapsed)}"
    )
    return summary
