"""
financial_collector.py

Collects financial factor snapshots into FinancialData table.
Supports both backfill and daily incremental modes.
"""

from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd


API_CALL_DELAY = 0.3
DEFAULT_START_DATE_STR = "19800101"


def get_financial_ticker_universe(conn, end_date=None):
    """
    Returns distinct stock codes from WeeklyFilteredStocks.
    Falls back to CompanyInfo when WeeklyFilteredStocks is empty.
    """
    with conn.cursor() as cur:
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


def get_latest_financial_date_for_ticker(conn, ticker_code):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(date) FROM FinancialData WHERE stock_code = %s",
            (ticker_code,),
        )
        result = cur.fetchone()
    if result and result[0]:
        return pd.to_datetime(result[0]).date()
    return None


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

    eps_numeric = pd.to_numeric(output["eps"], errors="coerce")
    bps_numeric = pd.to_numeric(output["bps"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        roe = np.where(
            bps_numeric > 0,
            (eps_numeric / bps_numeric) * 100.0,
            np.nan,
        )
    output["roe"] = roe
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


def _resolve_start_date(conn, ticker_code, mode, start_date_str, end_date):
    if start_date_str:
        return datetime.strptime(start_date_str, "%Y%m%d").date()

    if mode == "backfill":
        return datetime.strptime(DEFAULT_START_DATE_STR, "%Y%m%d").date()

    latest = get_latest_financial_date_for_ticker(conn, ticker_code)
    if latest is None:
        return datetime.strptime(DEFAULT_START_DATE_STR, "%Y%m%d").date()
    if latest >= end_date:
        return None
    return latest + timedelta(days=1)


def run_financial_batch(
    conn,
    mode="daily",
    start_date_str=None,
    end_date_str=None,
    ticker_codes=None,
    api_call_delay=API_CALL_DELAY,
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
        ticker_codes = get_financial_ticker_universe(conn, end_date=end_date)

    summary = {
        "tickers_total": len(ticker_codes),
        "tickers_processed": 0,
        "rows_saved": 0,
        "errors": 0,
    }

    if not ticker_codes:
        return summary

    from pykrx import stock

    for ticker_code in ticker_codes:
        try:
            effective_start = _resolve_start_date(
                conn=conn,
                ticker_code=ticker_code,
                mode=mode,
                start_date_str=start_date_str,
                end_date=end_date,
            )
            if effective_start is None or effective_start > end_date:
                continue

            time.sleep(api_call_delay)
            df_fundamental = stock.get_market_fundamental(
                effective_start.strftime("%Y%m%d"),
                end_date_str,
                ticker_code,
            )
            normalized = normalize_fundamental_df(df_fundamental, ticker_code)
            if normalized.empty:
                summary["tickers_processed"] += 1
                continue

            rows = normalized[
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
            saved = upsert_financial_rows(conn, rows)
            summary["rows_saved"] += max(saved, 0)
            summary["tickers_processed"] += 1
        except Exception:
            summary["errors"] += 1
            conn.rollback()
            continue

    return summary
