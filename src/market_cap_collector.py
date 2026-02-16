"""
market_cap_collector.py

Collects daily market cap snapshots into MarketCapDaily table.
Supports both backfill and daily incremental modes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import random
import threading
import time

import numpy as np
import pandas as pd
import requests
import json


# Suppress FutureWarning: Downcasting behavior in `replace` is deprecated
pd.set_option("future.no_silent_downcasting", True)

API_CALL_DELAY = 3.5
DEFAULT_START_DATE_STR = "19950101"
DEFAULT_LOG_INTERVAL = 50
DEFAULT_WORKERS = 1
DEFAULT_WRITE_BATCH_SIZE = 20000
DEFAULT_API_JITTER_MAX_SECONDS = 3.0
MAX_API_JITTER_MAX_SECONDS = 5.0
DEFAULT_MACRO_PAUSE_EVERY = 50
DEFAULT_MACRO_PAUSE_MIN_SECONDS = 40.0
DEFAULT_MACRO_PAUSE_MAX_SECONDS = 60.0
DEFAULT_ERROR_COOLDOWN_SECONDS = 600.0
DEFAULT_PREFLIGHT_RETRY_COUNT = 1
DEFAULT_PREFLIGHT_TICKER = "005930"
DEFAULT_PREFLIGHT_ISIN = "KR7005930003"
DEFAULT_PREFLIGHT_LOOKBACK_DAYS = 60
FAIL_FAST_MIN_PROGRESS_RATIO = 0.1
FAIL_FAST_MIN_SAMPLES = 200
MIN_API_CALL_DELAY = 1.5
KRX_JSON_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
KRX_REFERER = "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd"
KRX_USER_AGENT = "Mozilla/5.0"
KRX_PROBE_TIMEOUT_SECONDS = 20


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


def _build_rate_limiter(
    api_call_delay,
    api_jitter_max_seconds=0.0,
    macro_pause_every=0,
    macro_pause_min_seconds=0.0,
    macro_pause_max_seconds=0.0,
):
    jitter_max = min(max(float(api_jitter_max_seconds), 0.0), MAX_API_JITTER_MAX_SECONDS)
    pause_every = max(int(macro_pause_every), 0)
    pause_min = max(float(macro_pause_min_seconds), 0.0)
    pause_max = max(float(macro_pause_max_seconds), pause_min)
    if api_call_delay <= 0 and jitter_max <= 0:
        return lambda: None

    lock = threading.Lock()
    next_allowed = [0.0]
    call_count = [0]

    def _wait_slot():
        wait_seconds = 0.0
        macro_pause = 0.0
        with lock:
            now = time.monotonic()
            if now < next_allowed[0]:
                wait_seconds = next_allowed[0] - now
            call_count[0] += 1
            jitter = random.uniform(0.0, jitter_max) if jitter_max > 0 else 0.0
            next_allowed[0] = max(now, next_allowed[0]) + api_call_delay + jitter
            if pause_every > 0 and call_count[0] % pause_every == 0 and pause_max > 0:
                macro_pause = random.uniform(pause_min, pause_max)
                next_allowed[0] += macro_pause
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        if macro_pause > 0:
            print(
                "[market_cap_collector] human-like pause "
                f"after {call_count[0]} calls: {macro_pause:.1f}s"
            )

    return _wait_slot


def _should_cooldown_retry(reason):
    text = str(reason).lower()
    return (
        "http_status=403" in text
        or "non-json response" in text
        or "invalid json" in text
    )


def _fetch_and_normalize_market_cap(ticker_code, effective_start, effective_end, wait_slot):
    from pykrx import stock

    wait_slot()
    try:
        df_cap = stock.get_market_cap(
            effective_start.strftime("%Y%m%d"),
            effective_end.strftime("%Y%m%d"),
            ticker_code,
        )
    except requests.exceptions.HTTPError as e:
        print(f"[market_cap_collector] HTTPError for {ticker_code} on {effective_start}-{effective_end}: {e}")
        return {"status": "http_error", "error": str(e), "rows": []}
    except json.JSONDecodeError as e:
        print(f"[market_cap_collector] JSONDecodeError for {ticker_code} on {effective_start}-{effective_end}: {e}")
        return {"status": "json_decode_error", "error": str(e), "rows": []}
    except Exception as e:
        print(f"[market_cap_collector] Unexpected error fetching market cap for {ticker_code} on {effective_start}-{effective_end}: {e}")
        return {"status": "unexpected_error", "error": str(e), "rows": []}

    if df_cap is None or df_cap.empty:
        return {"status": "empty", "rows": []}

    normalized = normalize_market_cap_df(df_cap, ticker_code)
    if normalized.empty:
        return {"status": "schema_mismatch", "rows": []}

    rows = normalized.where(pd.notna(normalized), None)
    return {
        "status": "ok",
        "rows": [tuple(row) for row in rows.itertuples(index=False, name=None)],
    }


def _preflight_window(end_date):
    min_start_date = datetime.strptime(DEFAULT_START_DATE_STR, "%Y%m%d").date()
    start = end_date - timedelta(days=int(DEFAULT_PREFLIGHT_LOOKBACK_DAYS))
    return max(start, min_start_date), end_date


def _probe_krx_market_cap_endpoint(preflight_start, preflight_end):
    isin = DEFAULT_PREFLIGHT_ISIN

    payload = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT01701",
        "isuCd": isin,
        "strtDd": preflight_start.strftime("%Y%m%d"),
        "endDd": preflight_end.strftime("%Y%m%d"),
        "adjStkPrc": 2,
    }
    headers = {"User-Agent": KRX_USER_AGENT, "Referer": KRX_REFERER}
    response = requests.post(
        KRX_JSON_URL,
        headers=headers,
        data=payload,
        timeout=KRX_PROBE_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise RuntimeError(
            "[market_cap_collector] KRX endpoint blocked "
            f"(http_status={response.status_code}, bld={payload['bld']}). "
            "This host cannot access KRX JSON endpoint."
        )

    content_type = (response.headers.get("content-type") or "").lower()
    if "json" not in content_type:
        snippet = " ".join(response.text.splitlines()[:3]).strip()
        raise RuntimeError(
            "[market_cap_collector] KRX endpoint returned non-JSON response "
            f"(content_type={content_type}). head={snippet[:200]}"
        )

    try:
        response.json()
    except ValueError as exc:
        snippet = " ".join(response.text.splitlines()[:3]).strip()
        raise RuntimeError(
            "[market_cap_collector] KRX endpoint returned invalid JSON. "
            f"head={snippet[:200]}"
        ) from exc


def _run_market_cap_preflight(end_date):
    preflight_start, preflight_end = _preflight_window(end_date)
    _probe_krx_market_cap_endpoint(preflight_start, preflight_end)
    try:
        result = _fetch_and_normalize_market_cap(
            DEFAULT_PREFLIGHT_TICKER,
            preflight_start,
            preflight_end,
            lambda: None,
        )
    except Exception as exc:
        raise RuntimeError(
            "[market_cap_collector] preflight failed with API error "
            f"(ticker={DEFAULT_PREFLIGHT_TICKER}, start={preflight_start}, end={preflight_end}): {exc}"
        ) from exc

    if result["status"] != "ok":
        raise RuntimeError(
            "[market_cap_collector] preflight failed "
            f"(ticker={DEFAULT_PREFLIGHT_TICKER}, start={preflight_start}, end={preflight_end}, "
            f"status={result['status']}). Check pykrx endpoint/schema availability first."
        )


def run_market_cap_batch(
    conn,
    mode="daily",
    start_date_str=None,
    end_date_str=None,
    ticker_codes=None,
    api_call_delay=API_CALL_DELAY,
    api_jitter_max_seconds=DEFAULT_API_JITTER_MAX_SECONDS,
    macro_pause_every=DEFAULT_MACRO_PAUSE_EVERY,
    macro_pause_min_seconds=DEFAULT_MACRO_PAUSE_MIN_SECONDS,
    macro_pause_max_seconds=DEFAULT_MACRO_PAUSE_MAX_SECONDS,
    error_cooldown_seconds=DEFAULT_ERROR_COOLDOWN_SECONDS,
    preflight_retry_count=DEFAULT_PREFLIGHT_RETRY_COUNT,
    workers=DEFAULT_WORKERS,
    write_batch_size=DEFAULT_WRITE_BATCH_SIZE,
    log_interval=DEFAULT_LOG_INTERVAL,
    fail_on_krx_unavailable=True,
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
        "http_errors": 0,
        "json_decode_errors": 0,
        "unexpected_errors": 0,
        "empty_results": 0,
        "schema_mismatch": 0,
        "skipped_due_to_unavailable": False,
        "unavailable_reason": None,
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

    preflight_error = None
    max_retries = max(int(preflight_retry_count), 0)
    for attempt in range(max_retries + 1):
        try:
            _run_market_cap_preflight(end_date)
            preflight_error = None
            break
        except RuntimeError as exc:
            preflight_error = exc
            can_retry = (
                attempt < max_retries
                and _should_cooldown_retry(exc)
                and float(error_cooldown_seconds) > 0
            )
            if can_retry:
                cooldown = max(float(error_cooldown_seconds), 0.0)
                print(
                    "[market_cap_collector] preflight blocked; cooldown "
                    f"{cooldown:.0f}s before retry {attempt + 1}/{max_retries + 1}"
                )
                time.sleep(cooldown)
                continue
            break

    if preflight_error is not None:
        if fail_on_krx_unavailable:
            raise preflight_error
        summary["skipped_due_to_unavailable"] = True
        summary["unavailable_reason"] = str(preflight_error)
        print(f"[market_cap_collector] unavailable -> skipped: {preflight_error}")
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
    wait_slot = _build_rate_limiter(
        max(float(api_call_delay), MIN_API_CALL_DELAY),
        api_jitter_max_seconds=min(
            max(float(api_jitter_max_seconds), 0.0),
            MAX_API_JITTER_MAX_SECONDS,
        ),
        macro_pause_every=max(int(macro_pause_every), 0),
        macro_pause_min_seconds=max(float(macro_pause_min_seconds), 0.0),
        macro_pause_max_seconds=max(
            float(macro_pause_max_seconds),
            max(float(macro_pause_min_seconds), 0.0),
        ),
    )
    worker_count = max(int(workers), 1)
    batch_size = max(int(write_batch_size), 1)
    target_total = len(target_items)
    abort_reason = None

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
                result = future.result()
                rows = result.get("rows", [])
                status = result.get("status", "ok")
                if rows:
                    row_buffer.extend(rows)
                    if len(row_buffer) >= batch_size:
                        _flush_buffer()
                elif status == "empty":
                    summary["empty_results"] += 1
                elif status == "schema_mismatch":
                    summary["schema_mismatch"] += 1
                elif status == "http_error":
                    summary["http_errors"] += 1
                    summary["errors"] += 1
                elif status == "json_decode_error":
                    summary["json_decode_errors"] += 1
                    summary["errors"] += 1
                elif status == "unexpected_error":
                    summary["unexpected_errors"] += 1
                    summary["errors"] += 1
                summary["tickers_processed"] += 1
            except Exception as exc:
                print(f"[market_cap_collector] Error processing {ticker_code}: {exc}")
                summary["errors"] += 1
                summary["unexpected_errors"] += 1
            finally:
                completed_count += 1
                progress_ratio = (completed_count / total_tickers) if total_tickers else 1.0
                enough_progress = progress_ratio >= FAIL_FAST_MIN_PROGRESS_RATIO
                enough_samples = summary["tickers_processed"] >= FAIL_FAST_MIN_SAMPLES
                all_empty_or_schema = (
                    summary["empty_results"] + summary["schema_mismatch"]
                    >= summary["tickers_processed"]
                )
                has_no_data_yet = summary["rows_saved"] == 0 and not row_buffer
                if enough_progress and enough_samples and has_no_data_yet and all_empty_or_schema:
                    abort_reason = (
                        "[market_cap_collector] fail-fast triggered: "
                        f"no rows after {summary['tickers_processed']}/{target_total} processed "
                        f"(empty={summary['empty_results']}, schema_mismatch={summary['schema_mismatch']}, "
                        f"http_errors={summary['http_errors']}, "
                        f"json_decode_errors={summary['json_decode_errors']}, "
                        f"unexpected_errors={summary['unexpected_errors']})."
                    )
                    for pending in future_map:
                        if not pending.done():
                            pending.cancel()
                    break
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
                        f"empty={summary['empty_results']} "
                        f"schema_mismatch={summary['schema_mismatch']} "
                        f"elapsed={_format_duration(elapsed)} "
                        f"eta={_estimate_eta(elapsed, completed_count, total_tickers)}"
                    )

    if abort_reason:
        raise RuntimeError(abort_reason)

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
        f"empty={summary['empty_results']} "
        f"schema_mismatch={summary['schema_mismatch']} "
        f"http_errors={summary['http_errors']} "
        f"json_decode_errors={summary['json_decode_errors']} "
        f"unexpected_errors={summary['unexpected_errors']} "
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
