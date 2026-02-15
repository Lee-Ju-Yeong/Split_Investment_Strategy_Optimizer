"""
ohlcv_batch.py

Resume-capable OHLCV batch collector for DailyStockPrice.
"""

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/ohlcv_batch.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

import argparse
from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd

from . import ohlcv_collector
from .db_setup import create_tables, get_db_connection


DEFAULT_LOG_INTERVAL = 50


def _as_date(value):
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date()
    return value


def _build_universe_ranges_from_history_rows(
    rows,
    requested_start_date,
    requested_end_date,
):
    ranges = []
    for row in rows:
        stock_code = row[0]
        listed_date = _as_date(row[1])
        last_seen_date = _as_date(row[2])
        delisted_date = _as_date(row[3])

        observed_end_date = delisted_date or last_seen_date
        if not listed_date or not observed_end_date:
            continue

        effective_start = max(requested_start_date, listed_date)
        effective_end = min(requested_end_date, observed_end_date)
        if effective_start <= effective_end:
            ranges.append((stock_code, effective_start, effective_end))
    return ranges


def _fetch_history_universe_ranges(conn, requested_start_date, requested_end_date):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stock_code, listed_date, last_seen_date, delisted_date
                FROM TickerUniverseHistory
                WHERE listed_date <= %s
                  AND COALESCE(delisted_date, last_seen_date) >= %s
                ORDER BY stock_code
                """,
                (requested_end_date, requested_start_date),
            )
            rows = cur.fetchall()
    except Exception as exc:
        print(f"[ohlcv_batch] history universe lookup failed: {exc}")
        return []

    return _build_universe_ranges_from_history_rows(
        rows=rows,
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
    )


def _fetch_legacy_universe_ranges(conn, requested_start_date, requested_end_date):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT stock_code
            FROM WeeklyFilteredStocks
                WHERE filter_date <= %s
                ORDER BY stock_code
                """,
            (requested_end_date,),
        )
        rows = cur.fetchall()
        if rows:
            return [
                (row[0], requested_start_date, requested_end_date)
                for row in rows
            ]

        cur.execute("SELECT stock_code FROM CompanyInfo ORDER BY stock_code")
        rows = cur.fetchall()
        return [
            (row[0], requested_start_date, requested_end_date)
            for row in rows
        ]


def _build_history_unavailable_message(start_date, end_date):
    return (
        "TickerUniverseHistory returned no rows "
        f"for requested range [{start_date}, {end_date}]. "
        "Run `python -m src.ticker_universe_batch --mode backfill` first "
        "or pass `--allow-legacy-fallback` explicitly."
    )


def get_ohlcv_ticker_universe(
    conn,
    requested_start_date,
    requested_end_date,
    allow_legacy_fallback=False,
):
    history_ranges = _fetch_history_universe_ranges(
        conn=conn,
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
    )
    if history_ranges:
        return history_ranges, "history"

    if not allow_legacy_fallback:
        raise RuntimeError(
            _build_history_unavailable_message(
                start_date=requested_start_date,
                end_date=requested_end_date,
            )
        )

    print(
        "[ohlcv_batch] warning using legacy fallback "
        f"requested_range=[{requested_start_date}, {requested_end_date}]"
    )
    legacy_ranges = _fetch_legacy_universe_ranges(
        conn=conn,
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
    )
    return legacy_ranges, "legacy"


def normalize_ohlcv_df(df_ohlcv, ticker_code):
    if df_ohlcv is None or df_ohlcv.empty:
        return pd.DataFrame()

    output = df_ohlcv.copy().reset_index()
    output.rename(
        columns={
            "날짜": "date",
            "시가": "open_price",
            "고가": "high_price",
            "저가": "low_price",
            "종가": "close_price",
            "거래량": "volume",
        },
        inplace=True,
    )
    required_columns = {
        "date",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
    }
    if not required_columns.issubset(output.columns):
        return pd.DataFrame()

    output["stock_code"] = ticker_code
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output.replace({np.nan: None}, inplace=True)
    return output[
        [
            "stock_code",
            "date",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
        ]
    ]


def upsert_ohlcv_rows(conn, rows):
    if not rows:
        return 0

    sql = """
        INSERT INTO DailyStockPrice (
            stock_code, date, open_price, high_price, low_price, close_price, volume
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open_price = VALUES(open_price),
            high_price = VALUES(high_price),
            low_price = VALUES(low_price),
            close_price = VALUES(close_price),
            volume = VALUES(volume)
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
        affected = cur.rowcount
    conn.commit()
    return affected


def _resolve_effective_collection_window(
    conn,
    ticker_code,
    universe_start_date,
    universe_end_date,
    resume,
):
    if universe_start_date > universe_end_date:
        return None, None

    effective_start = universe_start_date
    effective_end = universe_end_date
    if not resume:
        return effective_start, effective_end

    latest_saved_date = ohlcv_collector.get_latest_ohlcv_date_for_ticker(conn, ticker_code)
    if latest_saved_date is not None:
        if latest_saved_date >= effective_end:
            return None, None
        next_date = latest_saved_date + timedelta(days=1)
        effective_start = max(effective_start, next_date)

    if effective_start > effective_end:
        return None, None
    return effective_start, effective_end


def _format_duration(seconds):
    safe_seconds = max(int(seconds), 0)
    return str(timedelta(seconds=safe_seconds))


def _estimate_eta(elapsed_seconds, done_count, total_count):
    if done_count <= 0 or total_count <= done_count:
        return "00:00:00"
    per_item = elapsed_seconds / done_count
    remaining = total_count - done_count
    return _format_duration(per_item * remaining)


def run_ohlcv_batch(
    conn,
    start_date_str=None,
    end_date_str=None,
    ticker_codes=None,
    allow_legacy_fallback=False,
    resume=True,
    api_call_delay=ohlcv_collector.API_CALL_DELAY,
    log_interval=DEFAULT_LOG_INTERVAL,
    ticker_limit=None,
):
    if end_date_str is None:
        end_date_str = datetime.today().strftime("%Y%m%d")
    if start_date_str is None:
        start_date_str = ohlcv_collector.DEFAULT_PYKRX_START_DATE_STR

    requested_start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
    end_date = datetime.strptime(end_date_str, "%Y%m%d").date()
    if requested_start_date > end_date:
        raise ValueError(
            f"Invalid date range: start_date({start_date_str}) > end_date({end_date_str})"
        )

    universe_source = "explicit"
    if ticker_codes is None:
        ticker_universe, universe_source = get_ohlcv_ticker_universe(
            conn=conn,
            requested_start_date=requested_start_date,
            requested_end_date=end_date,
            allow_legacy_fallback=allow_legacy_fallback,
        )
    else:
        ticker_universe = [
            (ticker_code, requested_start_date, end_date)
            for ticker_code in ticker_codes
        ]
    if ticker_limit is not None and ticker_limit > 0:
        ticker_universe = ticker_universe[:ticker_limit]

    summary = {
        "tickers_total": len(ticker_universe),
        "tickers_processed": 0,
        "tickers_skipped": 0,
        "rows_saved": 0,
        "errors": 0,
        "start_date": requested_start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "universe_source": universe_source,
        "allow_legacy_fallback": allow_legacy_fallback,
        "legacy_fallback_used": universe_source == "legacy",
        "legacy_fallback_tickers": len(ticker_universe) if universe_source == "legacy" else 0,
        "legacy_fallback_runs": 1 if universe_source == "legacy" else 0,
        "resume": resume,
    }

    total_tickers = len(ticker_universe)
    if total_tickers == 0:
        return summary

    ohlcv_collector.API_CALL_DELAY = max(float(api_call_delay), 0.0)
    started_at = time.time()
    print(
        f"[ohlcv_batch] start start_date={start_date_str}, end_date={end_date_str}, "
        f"tickers={total_tickers}, adjusted={ohlcv_collector.USE_ADJUSTED_OHLCV}, "
        f"resume={resume}, universe_source={universe_source}, "
        f"allow_legacy_fallback={allow_legacy_fallback}"
    )

    for index, ticker_entry in enumerate(ticker_universe, start=1):
        ticker_code, universe_start_date, universe_end_date = ticker_entry
        try:
            effective_start, effective_end = _resolve_effective_collection_window(
                conn=conn,
                ticker_code=ticker_code,
                universe_start_date=universe_start_date,
                universe_end_date=universe_end_date,
                resume=resume,
            )
            if effective_start is None or effective_end is None:
                summary["tickers_skipped"] += 1
                continue

            df_ohlcv = ohlcv_collector.get_market_ohlcv_with_fallback(
                effective_start.strftime("%Y%m%d"),
                effective_end.strftime("%Y%m%d"),
                ticker_code,
            )
            normalized = normalize_ohlcv_df(df_ohlcv, ticker_code)
            if normalized.empty:
                summary["tickers_processed"] += 1
                continue

            rows = normalized.to_records(index=False).tolist()
            saved = upsert_ohlcv_rows(conn, rows)
            summary["rows_saved"] += max(saved, 0)
            summary["tickers_processed"] += 1
        except Exception as exc:
            summary["errors"] += 1
            conn.rollback()
            print(f"[ohlcv_batch] ticker={ticker_code} error={exc}")
        finally:
            if (
                log_interval
                and log_interval > 0
                and (index % log_interval == 0 or index == total_tickers)
            ):
                elapsed = time.time() - started_at
                print(
                    f"[ohlcv_batch] progress {index}/{total_tickers} "
                    f"({index / total_tickers:.1%}) "
                    f"processed={summary['tickers_processed']} "
                    f"skipped={summary['tickers_skipped']} "
                    f"rows_saved={summary['rows_saved']} "
                    f"errors={summary['errors']} "
                    f"elapsed={_format_duration(elapsed)} "
                    f"eta={_estimate_eta(elapsed, index, total_tickers)}"
                )

    total_elapsed = time.time() - started_at
    rows_per_sec = 0.0
    tickers_per_sec = 0.0
    if total_elapsed > 0:
        rows_per_sec = summary["rows_saved"] / total_elapsed
        tickers_per_sec = summary["tickers_processed"] / total_elapsed
    summary["elapsed_seconds"] = int(total_elapsed)
    summary["rows_per_sec"] = round(rows_per_sec, 2)
    summary["tickers_per_sec"] = round(tickers_per_sec, 4)
    print(
        f"[ohlcv_batch] completed "
        f"processed={summary['tickers_processed']} "
        f"skipped={summary['tickers_skipped']} "
        f"rows_saved={summary['rows_saved']} "
        f"errors={summary['errors']} "
        f"elapsed={_format_duration(total_elapsed)} "
        f"throughput_rows_per_sec={summary['rows_per_sec']} "
        f"throughput_tickers_per_sec={summary['tickers_per_sec']}"
    )
    print(
        "[ohlcv_batch] fallback_report "
        f"legacy_used={1 if summary['legacy_fallback_used'] else 0} "
        f"legacy_tickers={summary['legacy_fallback_tickers']} "
        f"legacy_runs={summary['legacy_fallback_runs']}"
    )
    return summary


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run resume-capable OHLCV batch for DailyStockPrice."
    )
    parser.add_argument(
        "--start-date",
        dest="start_date",
        default=ohlcv_collector.DEFAULT_PYKRX_START_DATE_STR,
        help="Start date in YYYYMMDD.",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        default=datetime.today().strftime("%Y%m%d"),
        help="End date in YYYYMMDD.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Enable resume by ticker's latest saved date (default).",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and force upsert from start-date to end-date.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL,
        help="Progress log interval by ticker. Set 0 to disable.",
    )
    parser.add_argument(
        "--api-call-delay",
        type=float,
        default=ohlcv_collector.API_CALL_DELAY,
        help="Sleep seconds between API calls.",
    )
    parser.add_argument(
        "--ticker-limit",
        type=int,
        default=None,
        help="Optional limit for number of tickers (for smoke test).",
    )
    parser.add_argument(
        "--allow-legacy-fallback",
        action="store_true",
        help="Use legacy universe sources when TickerUniverseHistory has no rows.",
    )
    return parser


def main():
    args = _build_arg_parser().parse_args()
    conn = None
    try:
        conn = get_db_connection()
        create_tables(conn)
        summary = run_ohlcv_batch(
            conn=conn,
            start_date_str=args.start_date,
            end_date_str=args.end_date,
            allow_legacy_fallback=args.allow_legacy_fallback,
            resume=args.resume,
            api_call_delay=args.api_call_delay,
            log_interval=args.log_interval,
            ticker_limit=args.ticker_limit,
        )
        print(f"[ohlcv_batch] summary {summary}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
