"""
corporate_event_collector.py

Fetches and stores corporate major changes (name, sector, par value, CEO) from pykrx.
Optimized with parallel workers and rate limiting.
"""

import time
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from pykrx import stock
from .db_setup import get_db_connection

# Suppress FutureWarning: Downcasting behavior in `replace` is deprecated
pd.set_option('future.no_silent_downcasting', True)

# Default Settings
API_CALL_DELAY = 0.5
DEFAULT_WORKERS = 4
DEFAULT_LOG_INTERVAL = 50
DEFAULT_WRITE_BATCH_SIZE = 5000

def get_corporate_major_changes(ticker, wait_slot):
    """
    Fetches major changes for a given ticker from pykrx with rate limiting.
    Returns:
        tuple[pd.DataFrame, Optional[str]]
        - dataframe: fetched result (can be empty)
        - error message: None if success, otherwise exception message
    """
    wait_slot()
    try:
        df = stock.get_stock_major_changes(ticker)
        return df, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)

def normalize_major_changes(df, ticker):
    if df is None or df.empty:
        return pd.DataFrame()
    
    output = df.copy().reset_index()
    column_map = {
        '날짜': 'change_date',
        '상호변경전': 'prev_company_name',
        '상호변경후': 'new_company_name',
        '업종변경전': 'prev_sector',
        '업종변경후': 'new_sector',
        '액면변경전': 'prev_par_value',
        '액면변경후': 'new_par_value',
        '대표이사변경전': 'prev_ceo',
        '대표이사변경후': 'new_ceo'
    }
    output.rename(columns=column_map, inplace=True)
    
    if 'change_date' not in output.columns:
        return pd.DataFrame()

    output['stock_code'] = ticker
    output['change_date'] = pd.to_datetime(output['change_date']).dt.strftime('%Y-%m-%d')
    
    # Replace '-' or NaN with None for MySQL NULL
    output = output.replace('-', None)
    output = output.replace({np.nan: None})
    
    # Ensure all required columns are present
    required_cols = [
        'stock_code', 'change_date', 'prev_company_name', 'new_company_name',
        'prev_sector', 'new_sector', 'prev_par_value', 'new_par_value',
        'prev_ceo', 'new_ceo'
    ]
    for col in required_cols:
        if col not in output.columns:
            output[col] = None
            
    return output[required_cols]

def upsert_major_changes(conn, rows):
    if not rows:
        return 0
    
    sql = """
        INSERT INTO CorporateMajorChanges (
            stock_code, change_date, prev_company_name, new_company_name,
            prev_sector, new_sector, prev_par_value, new_par_value,
            prev_ceo, new_ceo
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            prev_company_name = VALUES(prev_company_name),
            new_company_name = VALUES(new_company_name),
            prev_sector = VALUES(prev_sector),
            new_sector = VALUES(new_sector),
            prev_par_value = VALUES(prev_par_value),
            new_par_value = VALUES(new_par_value),
            prev_ceo = VALUES(prev_ceo),
            new_ceo = VALUES(new_ceo),
            updated_at = CURRENT_TIMESTAMP
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)

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


def _clip_error_message(message, max_len=200):
    if message is None:
        return ""
    text = str(message).strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."

def run_corporate_major_changes_batch(
    conn, 
    ticker_codes=None, 
    workers=DEFAULT_WORKERS, 
    api_call_delay=API_CALL_DELAY,
    log_interval=DEFAULT_LOG_INTERVAL,
    write_batch_size=DEFAULT_WRITE_BATCH_SIZE
):
    if ticker_codes is None:
        with conn.cursor() as cur:
            cur.execute("SELECT stock_code FROM TickerUniverseHistory ORDER BY stock_code")
            ticker_codes = [row[0] for row in cur.fetchall()]
            
    if not ticker_codes:
        with conn.cursor() as cur:
            cur.execute("SELECT stock_code FROM CompanyInfo ORDER BY stock_code")
            ticker_codes = [row[0] for row in cur.fetchall()]

    if not ticker_codes:
        print("[corporate_event_collector] no tickers to process.")
        return 0

    total = len(ticker_codes)
    started_at = time.time()
    wait_slot = _build_rate_limiter(max(float(api_call_delay), 0.0))
    worker_count = max(int(workers), 1)
    
    summary = {
        "tickers_total": total,
        "tickers_processed": 0,
        "rows_saved": 0,
        "errors": 0,
        "fetch_errors": 0,
        "empty_results": 0,
        "nonempty_results": 0,
        "normalize_empty": 0,
        "fetch_error_tickers": [],
        "empty_result_tickers": [],
        "nonempty_result_tickers": [],
    }
    
    row_buffer = []
    buffer_lock = threading.Lock()

    def _flush_buffer():
        nonlocal row_buffer
        rows_to_write = []
        with buffer_lock:
            if not row_buffer:
                return 0
            rows_to_write = row_buffer
            row_buffer = []
        count = upsert_major_changes(conn, rows_to_write)
        summary["rows_saved"] += count
        return count

    print(f"[corporate_event_collector] starting batch for {total} tickers with {worker_count} workers...")
    
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(get_corporate_major_changes, ticker, wait_slot): ticker 
            for ticker in ticker_codes
        }
        
        for i, future in enumerate(as_completed(future_map), start=1):
            ticker = future_map[future]
            try:
                df, fetch_error = future.result()

                if fetch_error is not None:
                    summary["fetch_errors"] += 1
                    if len(summary["fetch_error_tickers"]) < 20:
                        summary["fetch_error_tickers"].append(
                            (ticker, _clip_error_message(fetch_error))
                        )
                elif df is None or df.empty:
                    summary["empty_results"] += 1
                    if len(summary["empty_result_tickers"]) < 20:
                        summary["empty_result_tickers"].append(ticker)
                else:
                    normalized = normalize_major_changes(df, ticker)
                    if normalized.empty:
                        summary["normalize_empty"] += 1
                    else:
                        summary["nonempty_results"] += 1
                        if len(summary["nonempty_result_tickers"]) < 20:
                            summary["nonempty_result_tickers"].append(ticker)
                        rows = [tuple(row) for row in normalized.to_numpy()]
                        should_flush = False
                        with buffer_lock:
                            row_buffer.extend(rows)
                            should_flush = len(row_buffer) >= write_batch_size
                        if should_flush:
                            _flush_buffer()
                summary["tickers_processed"] += 1
            except Exception as e:
                print(f"[corporate_event_collector] error ticker={ticker}: {e}")
                summary["errors"] += 1
            
            if log_interval and (i % log_interval == 0 or i == total):
                elapsed = time.time() - started_at
                print(f"[corporate_event_collector] progress {i}/{total} ({i/total:.1%}) "
                      f"processed={summary['tickers_processed']} saved_rows={summary['rows_saved']} "
                      f"fetch_errors={summary['fetch_errors']} empty={summary['empty_results']} "
                      f"nonempty={summary['nonempty_results']} normalize_empty={summary['normalize_empty']} "
                      f"elapsed={_format_duration(elapsed)} eta={_estimate_eta(elapsed, i, total)}")
    
    _flush_buffer()
    total_elapsed = time.time() - started_at
    print(f"[corporate_event_collector] completed in {_format_duration(total_elapsed)}. "
          f"processed={summary['tickers_processed']} saved_rows={summary['rows_saved']} errors={summary['errors']} "
          f"fetch_errors={summary['fetch_errors']} empty={summary['empty_results']} "
          f"nonempty={summary['nonempty_results']} normalize_empty={summary['normalize_empty']}")
    if summary["fetch_error_tickers"]:
        print(
            "[corporate_event_collector] fetch error samples: "
            f"{summary['fetch_error_tickers']}"
        )
    if summary["empty_result_tickers"]:
        print(
            "[corporate_event_collector] empty result samples: "
            f"{summary['empty_result_tickers']}"
        )
    if summary["nonempty_result_tickers"]:
        print(
            "[corporate_event_collector] non-empty result samples: "
            f"{summary['nonempty_result_tickers']}"
        )
            
    return summary["rows_saved"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect corporate major changes.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--delay", type=float, default=API_CALL_DELAY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_WRITE_BATCH_SIZE)
    parser.add_argument("--ticker-limit", type=int)
    
    args = parser.parse_args()
    
    conn = get_db_connection()
    try:
        tickers = None
        if args.ticker_limit:
            with conn.cursor() as cur:
                cur.execute(f"SELECT stock_code FROM TickerUniverseHistory LIMIT {args.ticker_limit}")
                tickers = [row[0] for row in cur.fetchall()]
        
        run_corporate_major_changes_batch(
            conn, 
            ticker_codes=tickers, 
            workers=args.workers, 
            api_call_delay=args.delay,
            write_batch_size=args.batch_size
        )
    finally:
        conn.close()
