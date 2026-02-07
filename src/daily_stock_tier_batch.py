"""
daily_stock_tier_batch.py

Pre-calculates DailyStockTier using liquidity and financial-risk proxies.
Supports backfill and daily incremental modes.
"""

from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd


DEFAULT_LOOKBACK_DAYS = 20
DEFAULT_FINANCIAL_LAG_DAYS = 45
DEFAULT_DANGER_LIQUIDITY = 300_000_000
DEFAULT_PRIME_LIQUIDITY = 1_000_000_000


def get_tier_ticker_universe(conn, end_date=None):
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

        cur.execute("SELECT DISTINCT stock_code FROM DailyStockPrice")
        return [row[0] for row in cur.fetchall()]


def get_latest_tier_date(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM DailyStockTier")
        row = cur.fetchone()
    if row and row[0]:
        return pd.to_datetime(row[0]).date()
    return None


def get_min_price_date(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT MIN(date) FROM DailyStockPrice")
        row = cur.fetchone()
    if row and row[0]:
        return pd.to_datetime(row[0]).date()
    return None


def _build_in_clause_params(base_sql, values):
    placeholders = ", ".join(["%s"] * len(values))
    return f"{base_sql} ({placeholders})", list(values)


def fetch_price_history(conn, start_date, end_date, ticker_codes=None):
    params = [start_date, end_date]
    query = """
        SELECT stock_code, date, close_price, volume
        FROM DailyStockPrice
        WHERE date BETWEEN %s AND %s
    """
    if ticker_codes:
        in_sql, in_params = _build_in_clause_params(" AND stock_code IN", ticker_codes)
        query += in_sql
        params.extend(in_params)
    query += " ORDER BY stock_code, date"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.read_sql(query, conn, params=params)


def fetch_financial_history(conn, end_date, ticker_codes=None):
    params = [end_date]
    query = """
        SELECT stock_code, date, roe, bps
        FROM FinancialData
        WHERE date <= %s
    """
    if ticker_codes:
        in_sql, in_params = _build_in_clause_params(" AND stock_code IN", ticker_codes)
        query += in_sql
        params.extend(in_params)
    query += " ORDER BY stock_code, date"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.read_sql(query, conn, params=params)


def _apply_financial_risk(price_df, financial_df, financial_lag_days):
    if financial_df is None or financial_df.empty:
        output = price_df.copy()
        output["financial_risk"] = False
        return output

    output_groups = []
    for stock_code, price_group in price_df.groupby("stock_code", sort=False):
        stock_prices = price_group.copy().sort_values("date")
        stock_financials = financial_df[financial_df["stock_code"] == stock_code].copy()
        if stock_financials.empty:
            stock_prices["financial_risk"] = False
            output_groups.append(stock_prices)
            continue

        stock_financials["date"] = pd.to_datetime(stock_financials["date"])
        stock_financials = stock_financials.sort_values("date")

        stock_prices["financial_as_of"] = stock_prices["date"] - pd.to_timedelta(
            financial_lag_days, unit="D"
        )
        merged = pd.merge_asof(
            stock_prices.sort_values("financial_as_of"),
            stock_financials[["date", "roe", "bps"]].sort_values("date"),
            left_on="financial_as_of",
            right_on="date",
            direction="backward",
            suffixes=("", "_financial"),
        )
        financial_risk = (
            (pd.to_numeric(merged["bps"], errors="coerce") <= 0)
            | (pd.to_numeric(merged["roe"], errors="coerce") < 0)
        ).fillna(False)
        merged["financial_risk"] = financial_risk
        merged = merged.sort_values("date").drop(
            columns=["financial_as_of", "date_financial"], errors="ignore"
        )
        output_groups.append(merged)

    return pd.concat(output_groups, ignore_index=True)


def build_daily_stock_tier_frame(
    price_df,
    financial_df=None,
    lookback_days=DEFAULT_LOOKBACK_DAYS,
    financial_lag_days=DEFAULT_FINANCIAL_LAG_DAYS,
    danger_liquidity=DEFAULT_DANGER_LIQUIDITY,
    prime_liquidity=DEFAULT_PRIME_LIQUIDITY,
):
    if price_df is None or price_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "stock_code",
                "tier",
                "reason",
                "liquidity_20d_avg_value",
            ]
        )

    base = price_df.copy()
    base["date"] = pd.to_datetime(base["date"])
    base["close_price"] = pd.to_numeric(base["close_price"], errors="coerce").fillna(0)
    base["volume"] = pd.to_numeric(base["volume"], errors="coerce").fillna(0)
    base.sort_values(["stock_code", "date"], inplace=True)
    base["trading_value"] = base["close_price"] * base["volume"]
    base["liquidity_20d_avg_value"] = (
        base.groupby("stock_code")["trading_value"]
        .rolling(window=lookback_days, min_periods=lookback_days)
        .mean()
        .reset_index(level=0, drop=True)
    )

    with_financial = _apply_financial_risk(base, financial_df, financial_lag_days)
    avg_value = pd.to_numeric(with_financial["liquidity_20d_avg_value"], errors="coerce")

    with_financial["tier"] = np.where(
        avg_value >= prime_liquidity,
        1,
        np.where(avg_value < danger_liquidity, 3, 2),
    )
    with_financial["reason"] = np.where(
        avg_value >= prime_liquidity,
        "prime_liquidity",
        np.where(avg_value < danger_liquidity, "low_liquidity", "normal_liquidity"),
    )

    risk_mask = with_financial["financial_risk"].fillna(False)
    with_financial.loc[risk_mask, "tier"] = 3
    with_financial.loc[risk_mask, "reason"] = with_financial.loc[risk_mask, "reason"].apply(
        lambda reason: f"{reason}+financial_risk"
        if "financial_risk" not in str(reason)
        else reason
    )

    output = with_financial[
        ["date", "stock_code", "tier", "reason", "liquidity_20d_avg_value"]
    ].copy()
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output["tier"] = output["tier"].astype(int)
    output["liquidity_20d_avg_value"] = (
        pd.to_numeric(output["liquidity_20d_avg_value"], errors="coerce")
        .fillna(0)
        .astype(np.int64)
    )
    return output


def upsert_daily_stock_tier(conn, rows):
    if not rows:
        return 0

    sql = """
        INSERT INTO DailyStockTier (
            date, stock_code, tier, reason, liquidity_20d_avg_value
        )
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            tier = VALUES(tier),
            reason = VALUES(reason),
            liquidity_20d_avg_value = VALUES(liquidity_20d_avg_value),
            computed_at = CURRENT_TIMESTAMP
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
        affected = cur.rowcount
    conn.commit()
    return affected


def _resolve_start_date(conn, mode, start_date_str, lookback_days):
    if start_date_str:
        return datetime.strptime(start_date_str, "%Y%m%d").date()

    if mode == "backfill":
        return get_min_price_date(conn)

    latest_tier_date = get_latest_tier_date(conn)
    if latest_tier_date is None:
        return get_min_price_date(conn)

    return latest_tier_date - timedelta(days=max(lookback_days - 1, 0))


def run_daily_stock_tier_batch(
    conn,
    mode="daily",
    start_date_str=None,
    end_date_str=None,
    ticker_codes=None,
    lookback_days=DEFAULT_LOOKBACK_DAYS,
    financial_lag_days=DEFAULT_FINANCIAL_LAG_DAYS,
    danger_liquidity=DEFAULT_DANGER_LIQUIDITY,
    prime_liquidity=DEFAULT_PRIME_LIQUIDITY,
):
    if mode not in {"daily", "backfill"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if end_date_str is None:
        end_date_str = datetime.today().strftime("%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d").date()

    if ticker_codes is None:
        ticker_codes = get_tier_ticker_universe(conn, end_date=end_date)

    start_date = _resolve_start_date(conn, mode, start_date_str, lookback_days)
    if start_date is None or start_date > end_date:
        return {
            "rows_saved": 0,
            "rows_calculated": 0,
            "start_date": None,
            "end_date": end_date.strftime("%Y-%m-%d"),
        }

    query_start = start_date - timedelta(days=lookback_days + financial_lag_days + 5)
    price_df = fetch_price_history(
        conn=conn,
        start_date=query_start.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        ticker_codes=ticker_codes,
    )
    if price_df.empty:
        return {
            "rows_saved": 0,
            "rows_calculated": 0,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }

    financial_df = fetch_financial_history(
        conn=conn,
        end_date=end_date.strftime("%Y-%m-%d"),
        ticker_codes=ticker_codes,
    )
    calculated = build_daily_stock_tier_frame(
        price_df=price_df,
        financial_df=financial_df,
        lookback_days=lookback_days,
        financial_lag_days=financial_lag_days,
        danger_liquidity=danger_liquidity,
        prime_liquidity=prime_liquidity,
    )
    if calculated.empty:
        return {
            "rows_saved": 0,
            "rows_calculated": 0,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }

    in_range = calculated[
        (pd.to_datetime(calculated["date"]).dt.date >= start_date)
        & (pd.to_datetime(calculated["date"]).dt.date <= end_date)
    ]
    rows = in_range[
        ["date", "stock_code", "tier", "reason", "liquidity_20d_avg_value"]
    ].to_records(index=False).tolist()
    saved = upsert_daily_stock_tier(conn, rows)

    return {
        "rows_saved": max(saved, 0),
        "rows_calculated": len(in_range),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }

