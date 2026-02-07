"""
investor_trading_collector.py

Collects investor net-buy trend snapshots into InvestorTradingTrend table.
Supports both backfill and daily incremental modes.
"""

from datetime import datetime, timedelta
import time

import pandas as pd


API_CALL_DELAY = 0.3
DEFAULT_START_DATE_STR = "19800101"


def get_investor_ticker_universe(conn, end_date=None):
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


def get_latest_investor_date_for_ticker(conn, ticker_code):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(date) FROM InvestorTradingTrend WHERE stock_code = %s",
            (ticker_code,),
        )
        result = cur.fetchone()
    if result and result[0]:
        return pd.to_datetime(result[0]).date()
    return None


def _resolve_column(df, contains_candidates):
    for col in df.columns:
        col_name = str(col).strip().lower()
        for key in contains_candidates:
            if key in col_name:
                return col
    return None


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

    output = pd.DataFrame()
    output["stock_code"] = ticker_code
    output["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    output["individual_net_buy"] = pd.to_numeric(
        df[individual_col], errors="coerce"
    ).fillna(0) if individual_col else 0
    output["foreigner_net_buy"] = pd.to_numeric(
        df[foreigner_col], errors="coerce"
    ).fillna(0) if foreigner_col else 0
    output["institution_net_buy"] = pd.to_numeric(
        df[institution_col], errors="coerce"
    ).fillna(0) if institution_col else 0
    output["total_net_buy"] = (
        output["individual_net_buy"]
        + output["foreigner_net_buy"]
        + output["institution_net_buy"]
    )
    output = output.astype(
        {
            "individual_net_buy": "int64",
            "foreigner_net_buy": "int64",
            "institution_net_buy": "int64",
            "total_net_buy": "int64",
        }
    )
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


def _resolve_start_date(conn, ticker_code, mode, start_date_str, end_date):
    if start_date_str:
        return datetime.strptime(start_date_str, "%Y%m%d").date()

    if mode == "backfill":
        return datetime.strptime(DEFAULT_START_DATE_STR, "%Y%m%d").date()

    latest = get_latest_investor_date_for_ticker(conn, ticker_code)
    if latest is None:
        return datetime.strptime(DEFAULT_START_DATE_STR, "%Y%m%d").date()
    if latest >= end_date:
        return None
    return latest + timedelta(days=1)


def run_investor_trading_batch(
    conn,
    mode="daily",
    start_date_str=None,
    end_date_str=None,
    ticker_codes=None,
    api_call_delay=API_CALL_DELAY,
):
    if mode not in {"daily", "backfill"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if end_date_str is None:
        end_date_str = datetime.today().strftime("%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d").date()

    if ticker_codes is None:
        ticker_codes = get_investor_ticker_universe(conn, end_date=end_date)

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
            df_trading = stock.get_market_trading_value_by_date(
                effective_start.strftime("%Y%m%d"),
                end_date_str,
                ticker_code,
                on="순매수",
            )
            normalized = normalize_investor_df(df_trading, ticker_code)
            if normalized.empty:
                summary["tickers_processed"] += 1
                continue

            rows = normalized[
                [
                    "stock_code",
                    "date",
                    "individual_net_buy",
                    "foreigner_net_buy",
                    "institution_net_buy",
                    "total_net_buy",
                ]
            ].to_records(index=False).tolist()
            saved = upsert_investor_rows(conn, rows)
            summary["rows_saved"] += max(saved, 0)
            summary["tickers_processed"] += 1
        except Exception:
            summary["errors"] += 1
            conn.rollback()
            continue

    return summary

