"""
short_selling_collector.py

Collects daily short-selling status snapshots into ShortSellingDaily table.
Supports both backfill and daily incremental modes.
"""

from __future__ import annotations

from collections import Counter
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
DEFAULT_LAG_TRADING_DAYS = 3  # KRX short-selling data is typically delayed (T+2).
DEFAULT_PREFLIGHT_TICKER = "005930"
DEFAULT_PREFLIGHT_ISIN = "KR7005930003"
DEFAULT_PREFLIGHT_LOOKBACK_DAYS = 60
DEFAULT_PREFILTER_MARKETS = ("KOSPI", "KOSDAQ")
DEFAULT_PREFILTER_MIN_HITS = 1
DEFAULT_PREFILTER_PROBE_DELAY = 2.0
DEFAULT_PREFILTER_PROBE_JITTER_MAX_SECONDS = 1.0
DEFAULT_PREFILTER_PROBE_MACRO_PAUSE_EVERY = 10
DEFAULT_PREFILTER_PROBE_MACRO_PAUSE_MIN_SECONDS = 5.0
DEFAULT_PREFILTER_PROBE_MACRO_PAUSE_MAX_SECONDS = 8.0
# KRX short-selling endpoint is sensitive to long ranges; keep each call strictly under 2 years.
# 729 inclusive days means "2 years minus at least 1 day" in all calendar cases.
DEFAULT_FETCH_CHUNK_MAX_DAYS = 729
EMPTY_STOP_THRESHOLD_AFTER_DATA = 2
SHORT_SELLING_COVERAGE_DONE_EMPTY = "DONE_EMPTY"
FAIL_FAST_MIN_PROGRESS_RATIO = 0.1
FAIL_FAST_MIN_SAMPLES = 200
MIN_API_CALL_DELAY = 1.5
KRX_JSON_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
KRX_REFERER = "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd"
KRX_USER_AGENT = "Mozilla/5.0"
KRX_PROBE_TIMEOUT_SECONDS = 20


def get_short_selling_ticker_universe(conn, end_date=None, mode="daily"):
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


def _get_nth_previous_trading_date(conn, base_date, n):
    if base_date is None or n <= 0:
        return base_date

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT date
                FROM DailyStockPrice
                WHERE date <= %s
                ORDER BY date DESC
                LIMIT %s
                """,
                (base_date, int(n) + 1),
            )
            rows = cur.fetchall()
        dates = [row[0] for row in rows if row and row[0]]
        if not dates:
            return None
        if len(dates) <= n:
            return dates[-1]
        return dates[n]
    except Exception:
        return base_date - timedelta(days=int(n))


def _cap_end_date_by_latest_trading_date(conn, end_date):
    latest_trading = _get_latest_trading_date(conn)
    if latest_trading is None:
        return end_date
    return min(end_date, latest_trading)


def _cap_end_date_by_short_selling_lag(conn, end_date, lag_trading_days):
    latest_trading = _get_latest_trading_date(conn)
    if latest_trading is None:
        return end_date
    safe_end = _get_nth_previous_trading_date(conn, latest_trading, lag_trading_days)
    if safe_end is None:
        return end_date
    return min(end_date, safe_end)


def get_short_selling_date_bounds(conn, ticker_codes, chunk_size=1000):
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
                FROM ShortSellingDaily
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


def get_ticker_listed_dates(conn, ticker_codes, chunk_size=1000):
    if not ticker_codes:
        return {}

    listed_dates = {}
    try:
        with conn.cursor() as cur:
            for start_index in range(0, len(ticker_codes), chunk_size):
                chunk = ticker_codes[start_index : start_index + chunk_size]
                placeholders = ", ".join(["%s"] * len(chunk))
                cur.execute(
                    f"""
                    SELECT stock_code, listed_date
                    FROM TickerUniverseHistory
                    WHERE stock_code IN ({placeholders})
                      AND listed_date IS NOT NULL
                    """,
                    chunk,
                )
                for stock_code, listed_date in cur.fetchall():
                    if listed_date:
                        listed_dates[stock_code] = (
                            listed_date.date() if isinstance(listed_date, datetime) else listed_date
                        )
    except Exception as exc:
        print(
            "[short_selling_collector] listed_date lookup skipped; "
            f"fallback to default start logic: {exc}"
        )
    return listed_dates


def get_done_empty_coverage_windows(conn, ticker_codes, chunk_size=1000):
    if not ticker_codes:
        return set()

    covered_windows = set()
    try:
        with conn.cursor() as cur:
            for start_index in range(0, len(ticker_codes), chunk_size):
                chunk = ticker_codes[start_index : start_index + chunk_size]
                placeholders = ", ".join(["%s"] * len(chunk))
                cur.execute(
                    f"""
                    SELECT stock_code, window_start, window_end
                    FROM ShortSellingBackfillCoverage
                    WHERE stock_code IN ({placeholders})
                      AND status = %s
                    """,
                    chunk + [SHORT_SELLING_COVERAGE_DONE_EMPTY],
                )
                for stock_code, window_start, window_end in cur.fetchall():
                    if not (window_start and window_end):
                        continue
                    start_date = window_start.date() if isinstance(window_start, datetime) else window_start
                    end_date = window_end.date() if isinstance(window_end, datetime) else window_end
                    covered_windows.add((stock_code, start_date, end_date))
    except Exception as exc:
        print(
            "[short_selling_collector] empty-coverage lookup skipped; "
            f"fallback to raw backfill windows: {exc}"
        )
    return covered_windows


def _resolve_column(df, candidates):
    normalized = {str(col).strip().upper(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().upper()
        if key in normalized:
            return normalized[key]
    return None


def _normalize_ticker_code(value):
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return None
    if raw.isdigit():
        return raw.zfill(6)
    return raw


def _extract_ticker_set_from_df(df):
    if df is None or df.empty:
        return set()

    ticker_col = _resolve_column(df, ["종목코드", "티커", "TICKER", "ISU_SRT_CD", "ISU_CD"])
    if ticker_col is not None:
        source_values = df[ticker_col].tolist()
    else:
        source_values = list(df.index)

    tickers = set()
    for value in source_values:
        ticker_code = _normalize_ticker_code(value)
        if ticker_code:
            tickers.add(ticker_code)
    return tickers


def _resolve_prefilter_anchor_dates(conn, start_date, end_date):
    safe_start = start_date or end_date
    if safe_start > end_date:
        safe_start = end_date

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MAX(date) AS anchor_date
                FROM DailyStockPrice
                WHERE date BETWEEN %s AND %s
                GROUP BY YEAR(date)
                ORDER BY anchor_date
                """,
                (safe_start, end_date),
            )
            rows = cur.fetchall()
    except Exception as exc:
        print(
            "[short_selling_collector] prefilter yearly-anchor query failed; "
            f"fallback to end_date anchor: {exc}"
        )
        return [end_date]

    anchors = []
    seen = set()
    for row in rows:
        anchor = row[0] if row else None
        if not anchor:
            continue
        if hasattr(anchor, "date") and isinstance(anchor, datetime):
            anchor = anchor.date()
        if anchor in seen:
            continue
        seen.add(anchor)
        anchors.append(anchor)

    if anchors:
        return anchors
    return [end_date]


def _build_short_selling_prefilter_set(
    conn,
    start_date,
    end_date,
    markets,
    min_hits,
    include_stock_only=True,
    probe_delay=DEFAULT_PREFILTER_PROBE_DELAY,
    probe_jitter_max_seconds=DEFAULT_PREFILTER_PROBE_JITTER_MAX_SECONDS,
    probe_macro_pause_every=DEFAULT_PREFILTER_PROBE_MACRO_PAUSE_EVERY,
    probe_macro_pause_min_seconds=DEFAULT_PREFILTER_PROBE_MACRO_PAUSE_MIN_SECONDS,
    probe_macro_pause_max_seconds=DEFAULT_PREFILTER_PROBE_MACRO_PAUSE_MAX_SECONDS,
):
    from pykrx import stock

    normalized_markets = [str(market).strip().upper() for market in markets if str(market).strip()]
    anchor_dates = _resolve_prefilter_anchor_dates(conn, start_date, end_date)
    wait_slot = _build_rate_limiter(
        max(float(probe_delay), MIN_API_CALL_DELAY),
        api_jitter_max_seconds=min(max(float(probe_jitter_max_seconds), 0.0), MAX_API_JITTER_MAX_SECONDS),
        macro_pause_every=max(int(probe_macro_pause_every), 0),
        macro_pause_min_seconds=max(float(probe_macro_pause_min_seconds), 0.0),
        macro_pause_max_seconds=max(
            float(probe_macro_pause_max_seconds),
            max(float(probe_macro_pause_min_seconds), 0.0),
        ),
    )
    hit_counter = Counter()
    probe_calls = 0
    probe_errors = 0

    for anchor_date in anchor_dates:
        date_str = anchor_date.strftime("%Y%m%d")
        for market in normalized_markets:
            probe_calls += 1
            try:
                wait_slot()
                kwargs = {
                    "date": date_str,
                    "market": market,
                    "alternative": True,
                }
                if include_stock_only:
                    kwargs["include"] = ["주식"]
                df_probe = stock.get_shorting_volume_by_ticker(**kwargs)
                for ticker_code in _extract_ticker_set_from_df(df_probe):
                    hit_counter[ticker_code] += 1
            except Exception as exc:
                probe_errors += 1
                print(
                    "[short_selling_collector] prefilter probe error "
                    f"(date={date_str}, market={market}): {exc}"
                )

    required_hits = max(int(min_hits), 1)
    eligible = {ticker for ticker, count in hit_counter.items() if count >= required_hits}
    safe_start = start_date or end_date
    if safe_start > end_date:
        safe_start = end_date
    return eligible, {
        "anchor_strategy": "yearly_last_trading_day",
        "anchor_start": safe_start.strftime("%Y%m%d"),
        "anchor_end": end_date.strftime("%Y%m%d"),
        "anchor_dates": [anchor.strftime("%Y%m%d") for anchor in anchor_dates],
        "markets": normalized_markets,
        "min_hits": required_hits,
        "probe_calls": probe_calls,
        "probe_errors": probe_errors,
        "eligible_count": len(eligible),
    }


def normalize_short_selling_df(df_short, ticker_code):
    if df_short is None or df_short.empty:
        return pd.DataFrame()

    df = df_short.copy().reset_index()
    date_col = _resolve_column(df, ["날짜", "DATE"])
    if date_col is None:
        return pd.DataFrame()

    volume_col = _resolve_column(df, ["공매도", "SHORT_VOLUME", "SHORTING_VOLUME"])
    value_col = _resolve_column(df, ["공매도금액", "SHORT_VALUE", "SHORTING_VALUE"])
    balance_col = _resolve_column(df, ["잔고", "SHORT_BALANCE", "BALANCE"])
    balance_value_col = _resolve_column(df, ["잔고금액", "SHORT_BALANCE_VALUE", "BALANCE_VALUE"])
    if not any([volume_col, value_col, balance_col, balance_value_col]):
        return pd.DataFrame()

    output = pd.DataFrame()
    output["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    output["stock_code"] = ticker_code
    output["short_volume"] = df[volume_col] if volume_col else np.nan
    output["short_value"] = df[value_col] if value_col else np.nan
    output["short_balance"] = df[balance_col] if balance_col else np.nan
    output["short_balance_value"] = df[balance_value_col] if balance_value_col else np.nan

    numeric_cols = [
        "short_volume",
        "short_value",
        "short_balance",
        "short_balance_value",
    ]
    for col in numeric_cols:
        output[col] = pd.to_numeric(output[col], errors="coerce")

    # Drop placeholder rows when *all* metrics are missing (pykrx returns empty df instead for most cases).
    has_any_value = output[numeric_cols].notna().any(axis=1)
    output = output.loc[has_any_value].copy()
    if output.empty:
        return pd.DataFrame()

    for col in numeric_cols:
        output[col] = output[col].round().astype("Int64")

    output["source"] = "pykrx"
    output.replace({np.nan: None}, inplace=True)
    return output[
        [
            "stock_code",
            "date",
            "short_volume",
            "short_value",
            "short_balance",
            "short_balance_value",
            "source",
        ]
    ]


def upsert_short_selling_rows(conn, rows):
    if not rows:
        return 0

    sql = """
        INSERT INTO ShortSellingDaily (
            stock_code, date, short_volume, short_value, short_balance, short_balance_value, source
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            short_volume = VALUES(short_volume),
            short_value = VALUES(short_value),
            short_balance = VALUES(short_balance),
            short_balance_value = VALUES(short_balance_value),
            source = VALUES(source),
            updated_at = CURRENT_TIMESTAMP
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
        affected = cur.rowcount
    conn.commit()
    return max(affected, 0)


def upsert_short_selling_backfill_coverage(conn, rows):
    if not rows:
        return 0

    sql = """
        INSERT INTO ShortSellingBackfillCoverage (
            stock_code, window_start, window_end, status, rows_saved
        )
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            status = VALUES(status),
            rows_saved = VALUES(rows_saved),
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

    min_dates, max_dates = get_short_selling_date_bounds(conn, ticker_codes)
    listed_dates = get_ticker_listed_dates(conn, ticker_codes)
    done_empty_coverage = (
        get_done_empty_coverage_windows(conn, ticker_codes)
        if mode == "backfill"
        else set()
    )
    for ticker_code in ticker_codes:
        listed_date = listed_dates.get(ticker_code)
        if mode == "daily":
            latest = max_dates.get(ticker_code)
            effective_start = min_start_date if latest is None else latest + timedelta(days=1)
            if listed_date is not None:
                effective_start = max(effective_start, listed_date)
            effective_end = end_date
        else:
            effective_start = base_start
            if listed_date is not None and effective_start is not None:
                effective_start = max(effective_start, listed_date)
            effective_end = end_date
            min_existing = min_dates.get(ticker_code)
            if (
                min_existing is not None
                and effective_start is not None
                and min_existing >= effective_start
            ):
                effective_end = min(end_date, min_existing - timedelta(days=1))

        if effective_start is None or effective_start > effective_end:
            skipped += 1
            continue
        if mode == "backfill" and (ticker_code, effective_start, effective_end) in done_empty_coverage:
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
                "[short_selling_collector] human-like pause "
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


def _split_fetch_windows(start_date, end_date, chunk_days=DEFAULT_FETCH_CHUNK_MAX_DAYS):
    if start_date > end_date:
        return []
    step_days = max(int(chunk_days), 1)
    windows = []
    # Newest-first windows: better resume ergonomics for interrupted backfills.
    current_end = end_date
    while current_end >= start_date:
        current_start = max(start_date, current_end - timedelta(days=step_days - 1))
        windows.append((current_start, current_end))
        current_end = current_start - timedelta(days=1)
    return windows


def _fetch_and_normalize_short_selling(ticker_code, effective_start, effective_end, wait_slot):
    from pykrx import stock

    chunk_windows = _split_fetch_windows(
        effective_start,
        effective_end,
        chunk_days=DEFAULT_FETCH_CHUNK_MAX_DAYS,
    )
    collected_rows = []
    saw_schema_mismatch = False
    schema_mismatch_chunks = 0
    stopped_on_empty_after_data = False
    consecutive_empty_after_data = 0

    for chunk_start, chunk_end in chunk_windows:
        wait_slot()
        try:
            df_short = stock.get_shorting_status_by_date(
                chunk_start.strftime("%Y%m%d"),
                chunk_end.strftime("%Y%m%d"),
                ticker_code,
            )
        except requests.exceptions.HTTPError as e:
            print(
                "[short_selling_collector] HTTPError for "
                f"{ticker_code} on {chunk_start}-{chunk_end}: {e}"
            )
            if collected_rows:
                return {
                    "status": "partial_error",
                    "error_type": "http_error",
                    "error": str(e),
                    "rows": collected_rows,
                }
            return {"status": "http_error", "error": str(e), "rows": []}
        except json.JSONDecodeError as e:
            print(
                "[short_selling_collector] JSONDecodeError for "
                f"{ticker_code} on {chunk_start}-{chunk_end}: {e}"
            )
            if collected_rows:
                return {
                    "status": "partial_error",
                    "error_type": "json_decode_error",
                    "error": str(e),
                    "rows": collected_rows,
                }
            return {"status": "json_decode_error", "error": str(e), "rows": []}
        except Exception as e:
            print(
                "[short_selling_collector] Unexpected error fetching short selling for "
                f"{ticker_code} on {chunk_start}-{chunk_end}: {e}"
            )
            if collected_rows:
                return {
                    "status": "partial_error",
                    "error_type": "unexpected_error",
                    "error": str(e),
                    "rows": collected_rows,
                }
            return {"status": "unexpected_error", "error": str(e), "rows": []}

        if df_short is None or df_short.empty:
            if collected_rows:
                consecutive_empty_after_data += 1
                if consecutive_empty_after_data >= EMPTY_STOP_THRESHOLD_AFTER_DATA:
                    stopped_on_empty_after_data = True
                    break
            continue
        consecutive_empty_after_data = 0

        normalized = normalize_short_selling_df(df_short, ticker_code)
        if normalized.empty:
            saw_schema_mismatch = True
            schema_mismatch_chunks += 1
            continue

        rows = normalized.where(pd.notna(normalized), None)
        collected_rows.extend(
            tuple(row) for row in rows.itertuples(index=False, name=None)
        )

    if collected_rows:
        if saw_schema_mismatch:
            return {
                "status": "partial_error",
                "error_type": "schema_mismatch",
                "schema_mismatch_chunks": schema_mismatch_chunks,
                "rows": collected_rows,
                "stopped_on_empty_after_data": stopped_on_empty_after_data,
            }
        return {
            "status": "ok",
            "rows": collected_rows,
            "stopped_on_empty_after_data": stopped_on_empty_after_data,
        }
    if saw_schema_mismatch:
        return {"status": "schema_mismatch", "rows": []}
    return {"status": "empty", "rows": []}


def _preflight_window(end_date):
    min_start_date = datetime.strptime(DEFAULT_START_DATE_STR, "%Y%m%d").date()
    start = end_date - timedelta(days=int(DEFAULT_PREFLIGHT_LOOKBACK_DAYS))
    return max(start, min_start_date), end_date


def _probe_krx_short_selling_endpoint(preflight_start, preflight_end):
    isin = DEFAULT_PREFLIGHT_ISIN

    payload = {
        "bld": "dbms/MDC/STAT/srt/MDCSTAT30001",
        "isuCd": isin,
        "strtDd": preflight_start.strftime("%Y%m%d"),
        "endDd": preflight_end.strftime("%Y%m%d"),
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
            "[short_selling_collector] KRX endpoint blocked "
            f"(http_status={response.status_code}, bld={payload['bld']}). "
            "This host cannot access KRX JSON endpoint."
        )

    content_type = (response.headers.get("content-type") or "").lower()
    try:
        response.json()
    except ValueError as exc:
        snippet = " ".join(response.text.splitlines()[:3]).strip()
        raise RuntimeError(
            "[short_selling_collector] KRX endpoint returned invalid JSON. "
            f"head={snippet[:200]}"
        ) from exc

    if "json" not in content_type:
        # KRX occasionally returns JSON payload with text/html content-type.
        # We accept parseable JSON and keep this warning for observability.
        print(
            "[short_selling_collector] KRX endpoint returned non-standard "
            f"content_type={content_type}, but JSON payload was valid."
        )


def _run_short_selling_preflight(end_date):
    preflight_start, preflight_end = _preflight_window(end_date)
    _probe_krx_short_selling_endpoint(preflight_start, preflight_end)
    try:
        result = _fetch_and_normalize_short_selling(
            DEFAULT_PREFLIGHT_TICKER,
            preflight_start,
            preflight_end,
            lambda: None,
        )
    except Exception as exc:
        raise RuntimeError(
            "[short_selling_collector] preflight failed with API error "
            f"(ticker={DEFAULT_PREFLIGHT_TICKER}, start={preflight_start}, end={preflight_end}): {exc}"
        ) from exc

    if result["status"] != "ok":
        raise RuntimeError(
            "[short_selling_collector] preflight failed "
            f"(ticker={DEFAULT_PREFLIGHT_TICKER}, start={preflight_start}, end={preflight_end}, "
            f"status={result['status']}). Check pykrx endpoint/schema availability first."
        )


def run_short_selling_batch(
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
    lag_trading_days=DEFAULT_LAG_TRADING_DAYS,
    fail_on_krx_unavailable=True,
    prefilter_enabled=False,
    prefilter_markets=DEFAULT_PREFILTER_MARKETS,
    prefilter_min_hits=DEFAULT_PREFILTER_MIN_HITS,
    prefilter_include_stock_only=True,
    prefilter_probe_delay=DEFAULT_PREFILTER_PROBE_DELAY,
    prefilter_probe_jitter_max_seconds=DEFAULT_PREFILTER_PROBE_JITTER_MAX_SECONDS,
    prefilter_probe_macro_pause_every=DEFAULT_PREFILTER_PROBE_MACRO_PAUSE_EVERY,
    prefilter_probe_macro_pause_min_seconds=DEFAULT_PREFILTER_PROBE_MACRO_PAUSE_MIN_SECONDS,
    prefilter_probe_macro_pause_max_seconds=DEFAULT_PREFILTER_PROBE_MACRO_PAUSE_MAX_SECONDS,
):
    if mode not in {"daily", "backfill"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if end_date_str is None:
        end_date_str = datetime.today().strftime("%Y%m%d")
    requested_end = datetime.strptime(end_date_str, "%Y%m%d").date()
    end_date = _cap_end_date_by_latest_trading_date(conn, requested_end)
    end_date = _cap_end_date_by_short_selling_lag(conn, end_date, max(int(lag_trading_days), 0))
    end_date_str = end_date.strftime("%Y%m%d")

    if ticker_codes is None:
        ticker_codes = get_short_selling_ticker_universe(conn, end_date=end_date, mode=mode)

    summary = {
        "tickers_total": len(ticker_codes),
        "tickers_processed": 0,
        "tickers_skipped": 0,
        "rows_saved": 0,
        "coverage_empty_written": 0,
        "errors": 0,
        "http_errors": 0,
        "json_decode_errors": 0,
        "unexpected_errors": 0,
        "partial_errors": 0,
        "empty_results": 0,
        "schema_mismatch": 0,
        "early_stop_on_empty_after_data": 0,
        "skipped_due_to_unavailable": False,
        "unavailable_reason": None,
        "end_date": end_date_str,
        "lag_trading_days": int(lag_trading_days),
        "prefilter_enabled": bool(prefilter_enabled),
        "prefilter_applied": False,
        "prefilter_before_count": len(ticker_codes),
        "prefilter_after_count": len(ticker_codes),
        "prefilter_dropped_count": 0,
    }
    if not ticker_codes:
        return summary

    if prefilter_enabled:
        prefilter_start_date = (
            datetime.strptime(start_date_str, "%Y%m%d").date()
            if start_date_str
            else (
                datetime.strptime(DEFAULT_START_DATE_STR, "%Y%m%d").date()
                if mode == "backfill"
                else end_date
            )
        )
        eligible_tickers, prefilter_meta = _build_short_selling_prefilter_set(
            conn=conn,
            start_date=prefilter_start_date,
            end_date=end_date,
            markets=prefilter_markets,
            min_hits=prefilter_min_hits,
            include_stock_only=bool(prefilter_include_stock_only),
            probe_delay=prefilter_probe_delay,
            probe_jitter_max_seconds=prefilter_probe_jitter_max_seconds,
            probe_macro_pause_every=prefilter_probe_macro_pause_every,
            probe_macro_pause_min_seconds=prefilter_probe_macro_pause_min_seconds,
            probe_macro_pause_max_seconds=prefilter_probe_macro_pause_max_seconds,
        )
        summary["prefilter"] = prefilter_meta
        before_count = len(ticker_codes)
        if eligible_tickers:
            ticker_codes = [ticker for ticker in ticker_codes if ticker in eligible_tickers]
            after_count = len(ticker_codes)
            summary["prefilter_applied"] = True
            summary["prefilter_after_count"] = after_count
            summary["prefilter_dropped_count"] = before_count - after_count
            print(
                "[short_selling_collector] prefilter applied "
                f"before={before_count}, after={after_count}, "
                f"dropped={before_count - after_count}, "
                f"anchors={prefilter_meta['anchor_dates']}, "
                f"markets={prefilter_meta['markets']}, "
                f"min_hits={prefilter_meta['min_hits']}, "
                f"probe_calls={prefilter_meta['probe_calls']}, "
                f"probe_errors={prefilter_meta['probe_errors']}"
            )
        else:
            print(
                "[short_selling_collector] prefilter skipped "
                "(eligible_count=0). Falling back to full ticker universe."
            )

    summary["tickers_total"] = len(ticker_codes)
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
            _run_short_selling_preflight(end_date)
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
                    "[short_selling_collector] preflight blocked; cooldown "
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
        print(f"[short_selling_collector] unavailable -> skipped: {preflight_error}")
        return summary

    started_at = time.time()
    total_tickers = len(ticker_codes)
    print(
        f"[short_selling_collector] start mode={mode}, "
        f"tickers={total_tickers}, target_tickers={len(target_items)}, "
        f"workers={max(int(workers), 1)}, write_batch_size={max(int(write_batch_size), 1)}, "
        f"end_date={end_date_str}, lag_trading_days={int(lag_trading_days)}, "
        f"fetch_chunk_max_days={DEFAULT_FETCH_CHUNK_MAX_DAYS}, "
        f"chunk_order=latest_to_oldest, stop_on_consecutive_empty_after_data={EMPTY_STOP_THRESHOLD_AFTER_DATA}"
    )

    row_buffer = []
    coverage_buffer = []
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
        saved = upsert_short_selling_rows(conn, row_buffer)
        summary["rows_saved"] += max(saved, 0)
        row_buffer = []

    def _flush_coverage_buffer():
        nonlocal coverage_buffer
        if not coverage_buffer:
            return
        saved = upsert_short_selling_backfill_coverage(conn, coverage_buffer)
        summary["coverage_empty_written"] += max(saved, 0)
        coverage_buffer = []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                _fetch_and_normalize_short_selling,
                ticker_code,
                window[0],
                window[1],
                wait_slot,
            ): (ticker_code, window[0], window[1])
            for ticker_code, window in target_items
        }

        for future in as_completed(future_map):
            ticker_code, window_start, window_end = future_map[future]
            try:
                result = future.result()
                rows = result.get("rows", [])
                status = result.get("status", "ok")
                if rows:
                    row_buffer.extend(rows)
                    if len(row_buffer) >= batch_size:
                        _flush_buffer()
                if result.get("stopped_on_empty_after_data"):
                    summary["early_stop_on_empty_after_data"] += 1

                if status == "empty":
                    summary["empty_results"] += 1
                    if mode == "backfill":
                        coverage_buffer.append(
                            (
                                ticker_code,
                                window_start,
                                window_end,
                                SHORT_SELLING_COVERAGE_DONE_EMPTY,
                                0,
                            )
                        )
                        if len(coverage_buffer) >= batch_size:
                            _flush_coverage_buffer()
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
                elif status == "partial_error":
                    summary["partial_errors"] += 1
                    summary["errors"] += 1
                    error_type = result.get("error_type")
                    if error_type == "http_error":
                        summary["http_errors"] += 1
                    elif error_type == "json_decode_error":
                        summary["json_decode_errors"] += 1
                    elif error_type == "schema_mismatch":
                        summary["schema_mismatch"] += 1
                    else:
                        summary["unexpected_errors"] += 1
                summary["tickers_processed"] += 1
            except Exception as exc:
                print(f"[short_selling_collector] Error processing {ticker_code}: {exc}")
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
                        "[short_selling_collector] fail-fast triggered: "
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
                        f"[short_selling_collector] progress {completed_count}/{total_tickers} "
                        f"({completed_count / total_tickers:.1%}) "
                        f"processed={summary['tickers_processed']} "
                        f"skipped={summary['tickers_skipped']} "
                        f"rows_saved={summary['rows_saved']} "
                        f"coverage_empty_written={summary['coverage_empty_written']} "
                        f"errors={summary['errors']} "
                        f"partial_errors={summary['partial_errors']} "
                        f"empty={summary['empty_results']} "
                        f"schema_mismatch={summary['schema_mismatch']} "
                        f"early_stop_on_empty_after_data={summary['early_stop_on_empty_after_data']} "
                        f"elapsed={_format_duration(elapsed)} "
                        f"eta={_estimate_eta(elapsed, completed_count, total_tickers)}"
                    )

    if abort_reason:
        try:
            _flush_buffer()
            _flush_coverage_buffer()
        except Exception:
            conn.rollback()
        raise RuntimeError(abort_reason)

    try:
        _flush_buffer()
        _flush_coverage_buffer()
    except Exception:
        summary["errors"] += 1
        conn.rollback()

    total_elapsed = time.time() - started_at
    print(
        f"[short_selling_collector] completed "
        f"processed={summary['tickers_processed']} "
        f"skipped={summary['tickers_skipped']} "
        f"rows_saved={summary['rows_saved']} "
        f"coverage_empty_written={summary['coverage_empty_written']} "
        f"errors={summary['errors']} "
        f"partial_errors={summary['partial_errors']} "
        f"empty={summary['empty_results']} "
        f"schema_mismatch={summary['schema_mismatch']} "
        f"http_errors={summary['http_errors']} "
        f"json_decode_errors={summary['json_decode_errors']} "
        f"unexpected_errors={summary['unexpected_errors']} "
        f"early_stop_on_empty_after_data={summary['early_stop_on_empty_after_data']} "
        f"elapsed={_format_duration(total_elapsed)}"
    )
    return summary


def main():
    import argparse

    from .db_setup import create_tables, get_db_connection

    parser = argparse.ArgumentParser(description="Collect ShortSellingDaily from pykrx.")
    parser.add_argument("--mode", choices=["daily", "backfill"], default="daily")
    parser.add_argument("--start-date", dest="start_date", default=None, help="YYYYMMDD")
    parser.add_argument("--end-date", dest="end_date", default=None, help="YYYYMMDD")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--delay", type=float, default=API_CALL_DELAY)
    parser.add_argument("--write-batch-size", type=int, default=DEFAULT_WRITE_BATCH_SIZE)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--lag-trading-days", type=int, default=DEFAULT_LAG_TRADING_DAYS)
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

        run_short_selling_batch(
            conn=conn,
            mode=args.mode,
            start_date_str=args.start_date,
            end_date_str=args.end_date,
            ticker_codes=tickers,
            api_call_delay=max(float(args.delay), 0.0),
            workers=max(int(args.workers), 1),
            write_batch_size=max(int(args.write_batch_size), 1),
            log_interval=args.log_interval,
            lag_trading_days=max(int(args.lag_trading_days), 0),
        )
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
