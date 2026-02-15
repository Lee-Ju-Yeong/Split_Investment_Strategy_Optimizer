"""
ticker_universe_batch.py

Builds point-in-time ticker universe snapshot/history tables.
"""

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/ticker_universe_batch.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import time

from .db_setup import create_tables, get_db_connection


DEFAULT_START_DATE_STR = "19950102"
DEFAULT_STEP_DAYS = 7
DEFAULT_LOG_INTERVAL = 10
DEFAULT_API_CALL_DELAY = 0.2
DEFAULT_MARKETS = ("KOSPI", "KOSDAQ")


def _parse_date(date_str):
    return datetime.strptime(date_str, "%Y%m%d").date()


def _format_duration(seconds):
    safe_seconds = max(int(seconds), 0)
    return str(timedelta(seconds=safe_seconds))


def _estimate_eta(elapsed_seconds, done_count, total_count):
    if done_count <= 0 or total_count <= done_count:
        return "00:00:00"
    per_item = elapsed_seconds / done_count
    remaining = total_count - done_count
    return _format_duration(per_item * remaining)


def _parse_markets(markets_arg):
    if not markets_arg:
        return list(DEFAULT_MARKETS)
    values = [market.strip().upper() for market in markets_arg.split(",")]
    return [market for market in values if market]


def build_snapshot_dates(mode, start_date_str, end_date_str, step_days):
    if mode not in {"daily", "backfill"}:
        raise ValueError(f"Unsupported mode: {mode}")

    end_date = _parse_date(end_date_str) if end_date_str else date.today()
    if mode == "daily":
        return [end_date]

    if not start_date_str:
        raise ValueError("`start_date_str` is required in backfill mode (YYYYMMDD).")

    start_date = _parse_date(start_date_str)
    if start_date > end_date:
        raise ValueError(
            f"Invalid date range: start_date({start_date_str}) > end_date({end_date_str})"
        )

    dates = []
    cursor_date = start_date
    interval = max(int(step_days), 1)
    while cursor_date <= end_date:
        dates.append(cursor_date)
        cursor_date += timedelta(days=interval)
    if dates[-1] != end_date:
        dates.append(end_date)
    return dates


def get_existing_snapshot_dates(conn, start_date, end_date):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT snapshot_date
            FROM TickerUniverseSnapshot
            WHERE snapshot_date BETWEEN %s AND %s
            """,
            (start_date, end_date),
        )
        rows = cur.fetchall()
    return {row[0] for row in rows if row and row[0]}


def collect_snapshot_rows(snapshot_date, markets, include_names=False, api_call_delay=0.0):
    from pykrx import stock

    date_str = snapshot_date.strftime("%Y%m%d")
    merged = {}

    for market_type in markets:
        ticker_codes = stock.get_market_ticker_list(date=date_str, market=market_type)
        for ticker_code in ticker_codes:
            merged[ticker_code] = {
                "market_type": market_type,
                "company_name": None,
            }
        if api_call_delay > 0:
            time.sleep(api_call_delay)

    if include_names:
        for ticker_code in merged:
            try:
                merged[ticker_code]["company_name"] = stock.get_market_ticker_name(ticker_code)
            except Exception:
                merged[ticker_code]["company_name"] = None
            if api_call_delay > 0:
                time.sleep(api_call_delay)

    rows = []
    snapshot_date_str = snapshot_date.strftime("%Y-%m-%d")
    for ticker_code in sorted(merged.keys()):
        rows.append(
            (
                snapshot_date_str,
                ticker_code,
                merged[ticker_code]["market_type"],
                merged[ticker_code]["company_name"],
                "pykrx",
            )
        )
    return rows


def upsert_snapshot_rows(conn, rows):
    if not rows:
        return 0
    sql = """
        INSERT INTO TickerUniverseSnapshot (
            snapshot_date, stock_code, market_type, company_name, source
        )
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            market_type = VALUES(market_type),
            company_name = VALUES(company_name),
            source = VALUES(source),
            updated_at = CURRENT_TIMESTAMP
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
        affected = cur.rowcount
    conn.commit()
    return max(affected, 0)


def run_snapshot_batch(
    conn,
    mode,
    start_date_str,
    end_date_str,
    markets,
    step_days=DEFAULT_STEP_DAYS,
    workers=1,
    resume=True,
    include_names=False,
    api_call_delay=DEFAULT_API_CALL_DELAY,
    log_interval=DEFAULT_LOG_INTERVAL,
):
    snapshot_dates = build_snapshot_dates(mode, start_date_str, end_date_str, step_days)
    summary = {
        "dates_total": len(snapshot_dates),
        "dates_processed": 0,
        "dates_skipped": 0,
        "rows_saved": 0,
        "errors": 0,
        "failed_dates": [],
    }
    if not snapshot_dates:
        return summary

    if resume:
        existing_dates = get_existing_snapshot_dates(conn, snapshot_dates[0], snapshot_dates[-1])
    else:
        existing_dates = set()
    target_dates = [snapshot_date for snapshot_date in snapshot_dates if snapshot_date not in existing_dates]
    summary["dates_skipped"] = len(snapshot_dates) - len(target_dates)

    if not target_dates:
        return summary

    started_at = time.time()
    total_target = len(target_dates)
    print(
        f"[ticker_universe_batch] snapshot start mode={mode}, dates={total_target}, "
        f"markets={','.join(markets)}, workers={workers}, resume={resume}"
    )

    def handle_result(index, total_count, snapshot_date, rows, error_message):
        if error_message:
            summary["errors"] += 1
            summary["failed_dates"].append(snapshot_date.strftime("%Y-%m-%d"))
            conn.rollback()
            print(f"[ticker_universe_batch] snapshot_date={snapshot_date} error={error_message}")
        else:
            saved = upsert_snapshot_rows(conn, rows)
            summary["rows_saved"] += saved
            summary["dates_processed"] += 1
        if log_interval and log_interval > 0 and (
            index % log_interval == 0 or index == total_count
        ):
            elapsed = time.time() - started_at
            print(
                f"[ticker_universe_batch] snapshot progress {index}/{total_count} "
                f"({index / total_count:.1%}) "
                f"processed={summary['dates_processed']} "
                f"skipped={summary['dates_skipped']} "
                f"rows_saved={summary['rows_saved']} "
                f"errors={summary['errors']} "
                f"elapsed={_format_duration(elapsed)} "
                f"eta={_estimate_eta(elapsed, index, total_count)}"
            )

    if workers <= 1:
        for index, snapshot_date in enumerate(target_dates, start=1):
            rows = None
            error_message = None
            try:
                rows = collect_snapshot_rows(
                    snapshot_date=snapshot_date,
                    markets=markets,
                    include_names=include_names,
                    api_call_delay=api_call_delay,
                )
            except Exception as exc:
                error_message = str(exc)
            handle_result(index, total_target, snapshot_date, rows, error_message)
    else:
        max_workers = max(int(workers), 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {
                executor.submit(
                    collect_snapshot_rows,
                    snapshot_date,
                    markets,
                    include_names,
                    api_call_delay,
                ): snapshot_date
                for snapshot_date in target_dates
            }
            for index, future in enumerate(as_completed(future_to_date), start=1):
                snapshot_date = future_to_date[future]
                rows = None
                error_message = None
                try:
                    rows = future.result()
                except Exception as exc:
                    error_message = str(exc)
                handle_result(index, total_target, snapshot_date, rows, error_message)

    total_elapsed = time.time() - started_at
    print(
        f"[ticker_universe_batch] snapshot completed "
        f"processed={summary['dates_processed']} "
        f"skipped={summary['dates_skipped']} "
        f"rows_saved={summary['rows_saved']} "
        f"errors={summary['errors']} "
        f"elapsed={_format_duration(total_elapsed)}"
    )
    return summary


def rebuild_ticker_universe_history(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*), MIN(snapshot_date), MAX(snapshot_date)
            FROM TickerUniverseSnapshot
            """
        )
        snapshot_count, min_snapshot_date, max_snapshot_date = cur.fetchone()
        if not snapshot_count or max_snapshot_date is None:
            return {
                "snapshot_rows": 0,
                "history_rows": 0,
                "upserted": 0,
                "deleted_stale": 0,
                "first_snapshot_date": None,
                "last_snapshot_date": None,
            }

        cur.execute(
            """
            REPLACE INTO TickerUniverseHistory (
                stock_code,
                listed_date,
                last_seen_date,
                delisted_date,
                latest_market_type,
                latest_company_name,
                source
            )
            SELECT
                aggregated.stock_code,
                aggregated.listed_date,
                aggregated.last_seen_date,
                CASE
                    WHEN aggregated.last_seen_date < max_snapshot.max_snapshot_date
                    THEN aggregated.last_seen_date
                    ELSE NULL
                END AS delisted_date,
                latest.market_type,
                latest.company_name,
                'snapshot_agg'
            FROM (
                SELECT
                    stock_code,
                    MIN(snapshot_date) AS listed_date,
                    MAX(snapshot_date) AS last_seen_date
                FROM TickerUniverseSnapshot
                GROUP BY stock_code
            ) aggregated
            CROSS JOIN (
                SELECT MAX(snapshot_date) AS max_snapshot_date
                FROM TickerUniverseSnapshot
            ) max_snapshot
            LEFT JOIN TickerUniverseSnapshot latest
                ON latest.stock_code = aggregated.stock_code
               AND latest.snapshot_date = aggregated.last_seen_date
            """
        )
        upserted = max(cur.rowcount, 0)

        cur.execute(
            """
            DELETE history
            FROM TickerUniverseHistory history
            LEFT JOIN (
                SELECT DISTINCT stock_code
                FROM TickerUniverseSnapshot
            ) snapshot
              ON snapshot.stock_code = history.stock_code
            WHERE snapshot.stock_code IS NULL
            """
        )
        deleted_stale = max(cur.rowcount, 0)
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM TickerUniverseHistory")
        history_rows = cur.fetchone()[0]

    return {
        "snapshot_rows": snapshot_count,
        "history_rows": history_rows,
        "upserted": upserted,
        "deleted_stale": deleted_stale,
        "first_snapshot_date": str(min_snapshot_date) if min_snapshot_date else None,
        "last_snapshot_date": str(max_snapshot_date) if max_snapshot_date else None,
    }


def run_ticker_universe_batch(
    conn,
    mode="daily",
    start_date_str=None,
    end_date_str=None,
    markets=None,
    step_days=DEFAULT_STEP_DAYS,
    workers=1,
    resume=True,
    include_names=False,
    run_snapshot=True,
    run_history=True,
    api_call_delay=DEFAULT_API_CALL_DELAY,
    log_interval=DEFAULT_LOG_INTERVAL,
):
    if end_date_str is None:
        end_date_str = datetime.today().strftime("%Y%m%d")
    if mode == "backfill" and not start_date_str:
        raise ValueError("`start_date_str` is required in backfill mode (YYYYMMDD).")
    markets = markets or list(DEFAULT_MARKETS)

    summary = {}
    if run_snapshot:
        summary["snapshot"] = run_snapshot_batch(
            conn=conn,
            mode=mode,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            markets=markets,
            step_days=step_days,
            workers=workers,
            resume=resume,
            include_names=include_names,
            api_call_delay=api_call_delay,
            log_interval=log_interval,
        )
    if run_history:
        history_started_at = time.time()
        print("[ticker_universe_batch] history aggregation start")
        summary["history"] = rebuild_ticker_universe_history(conn)
        elapsed = _format_duration(time.time() - history_started_at)
        print(
            f"[ticker_universe_batch] history aggregation completed "
            f"history_rows={summary['history']['history_rows']} elapsed={elapsed}"
        )
    return summary


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build ticker universe snapshot/history tables."
    )
    parser.add_argument(
        "--mode",
        choices=["daily", "backfill"],
        default="daily",
        help="Batch mode: daily snapshot or historical backfill.",
    )
    parser.add_argument(
        "--start-date",
        dest="start_date",
        default=None,
        help="Start date in YYYYMMDD (required for backfill mode).",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        default=datetime.today().strftime("%Y%m%d"),
        help="End date in YYYYMMDD.",
    )
    parser.add_argument(
        "--markets",
        default="KOSPI,KOSDAQ",
        help="Comma-separated market list. Example: KOSPI,KOSDAQ",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=DEFAULT_STEP_DAYS,
        help="Snapshot interval in days for backfill mode.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker count for parallel snapshot fetch. Default: 1 (safe).",
    )
    parser.add_argument(
        "--with-names",
        action="store_true",
        help="Collect company_name from pykrx for each ticker snapshot.",
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="Skip snapshot collection.",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Skip history aggregation.",
    )
    parser.add_argument(
        "--api-call-delay",
        type=float,
        default=DEFAULT_API_CALL_DELAY,
        help="Sleep seconds between API calls.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL,
        help="Progress log interval by snapshot date.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Skip snapshot dates already stored (default).",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Force recollect snapshots even when date already exists.",
    )
    parser.set_defaults(resume=True)
    return parser


def main():
    args = _build_arg_parser().parse_args()
    conn = None
    try:
        conn = get_db_connection()
        create_tables(conn)
        markets = _parse_markets(args.markets)
        started_at = time.time()
        print(
            "[ticker_universe_batch] start "
            f"mode={args.mode}, start_date={args.start_date}, end_date={args.end_date}, "
            f"markets={','.join(markets)}, workers={args.workers}, "
            f"run_snapshot={not args.skip_snapshot}, run_history={not args.skip_history}, "
            f"resume={args.resume}, with_names={args.with_names}"
        )
        summary = run_ticker_universe_batch(
            conn=conn,
            mode=args.mode,
            start_date_str=args.start_date,
            end_date_str=args.end_date,
            markets=markets,
            step_days=args.step_days,
            workers=args.workers,
            resume=args.resume,
            include_names=args.with_names,
            run_snapshot=not args.skip_snapshot,
            run_history=not args.skip_history,
            api_call_delay=max(float(args.api_call_delay), 0.0),
            log_interval=args.log_interval,
        )
        elapsed = _format_duration(time.time() - started_at)
        print(f"[ticker_universe_batch] completed elapsed={elapsed}")
        for key, value in summary.items():
            print(f"  - {key}: {value}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
