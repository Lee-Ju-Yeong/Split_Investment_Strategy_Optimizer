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
DEFAULT_ENABLE_INVESTOR_V1_WRITE = False
DEFAULT_INVESTOR_FLOW5_THRESHOLD = -500_000_000


def get_tier_ticker_universe(conn, end_date=None, mode="daily"):
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


def fetch_financial_history(conn, end_date, ticker_codes=None, start_date=None):
    """
    Fetch financial history up to `end_date`.

    When `start_date` is provided, it returns:
      1) rows in [start_date, end_date]
      2) plus the latest row per stock_code before start_date (PIT-friendly as-of seed)

    This prevents loading the entire FinancialData table for long backfills while
    preserving merge_asof correctness within the requested window.
    """
    end_date = str(end_date)

    def _read_sql(query, params):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.read_sql(query, conn, params=params)

    if start_date is None:
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
        return _read_sql(query, params=params)

    start_date = str(start_date)

    params = [start_date, end_date]
    range_query = """
        SELECT stock_code, date, roe, bps
        FROM FinancialData
        WHERE date BETWEEN %s AND %s
    """
    if ticker_codes:
        in_sql, in_params = _build_in_clause_params(" AND stock_code IN", ticker_codes)
        range_query += in_sql
        params.extend(in_params)
    range_query += " ORDER BY stock_code, date"
    in_range = _read_sql(range_query, params=params)

    prev_params = [start_date]
    prev_query = """
        SELECT f.stock_code, f.date, f.roe, f.bps
        FROM FinancialData f
        JOIN (
            SELECT stock_code, MAX(date) AS max_date
            FROM FinancialData
            WHERE date < %s
    """
    if ticker_codes:
        in_sql, in_params = _build_in_clause_params(" AND stock_code IN", ticker_codes)
        prev_query += in_sql
        prev_params.extend(in_params)
    prev_query += """
            GROUP BY stock_code
        ) latest
        ON f.stock_code = latest.stock_code AND f.date = latest.max_date
        ORDER BY f.stock_code, f.date
    """
    prev_rows = _read_sql(prev_query, params=prev_params)

    if in_range.empty and prev_rows.empty:
        return in_range

    combined = pd.concat([prev_rows, in_range], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined.sort_values(["stock_code", "date"], inplace=True)
    return combined


def fetch_investor_history(conn, start_date, end_date, ticker_codes=None):
    params = [start_date, end_date]
    query = """
        SELECT stock_code, date, foreigner_net_buy, institution_net_buy
        FROM InvestorTradingTrend
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


def _append_reason(base_reason, suffix):
    if suffix in str(base_reason):
        return str(base_reason)
    return f"{base_reason}+{suffix}"


def _apply_investor_flow_overlay(with_financial, investor_df, investor_flow5_threshold):
    if investor_df is None or investor_df.empty:
        return with_financial

    investor = investor_df.copy()
    investor["date"] = pd.to_datetime(investor["date"])
    investor["foreigner_net_buy"] = pd.to_numeric(
        investor["foreigner_net_buy"], errors="coerce"
    ).fillna(0)
    investor["institution_net_buy"] = pd.to_numeric(
        investor["institution_net_buy"], errors="coerce"
    ).fillna(0)
    investor.sort_values(["stock_code", "date"], inplace=True)
    investor["flow"] = investor["foreigner_net_buy"] + investor["institution_net_buy"]
    investor["flow5"] = (
        investor.groupby("stock_code")["flow"]
        .rolling(window=5, min_periods=5)
        .sum()
        .reset_index(level=0, drop=True)
    )

    merged = with_financial.merge(
        investor[["stock_code", "date", "flow5"]],
        on=["stock_code", "date"],
        how="left",
    )
    flow5 = pd.to_numeric(merged["flow5"], errors="coerce")
    flow_mask = (
        (merged["tier"] == 2)
        & flow5.notna()
        & (flow5 < int(investor_flow5_threshold))
    )
    merged.loc[flow_mask, "tier"] = 3
    merged.loc[flow_mask, "reason"] = merged.loc[flow_mask, "reason"].apply(
        lambda reason: _append_reason(reason, "investor_flow5")
    )
    return merged.drop(columns=["flow5"], errors="ignore")


def build_daily_stock_tier_frame(
    price_df,
    financial_df=None,
    investor_df=None,
    lookback_days=DEFAULT_LOOKBACK_DAYS,
    financial_lag_days=DEFAULT_FINANCIAL_LAG_DAYS,
    danger_liquidity=DEFAULT_DANGER_LIQUIDITY,
    prime_liquidity=DEFAULT_PRIME_LIQUIDITY,
    enable_investor_v1_write=DEFAULT_ENABLE_INVESTOR_V1_WRITE,
    investor_flow5_threshold=DEFAULT_INVESTOR_FLOW5_THRESHOLD,
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
        lambda reason: _append_reason(reason, "financial_risk")
    )
    if enable_investor_v1_write:
        with_financial = _apply_investor_flow_overlay(
            with_financial=with_financial,
            investor_df=investor_df,
            investor_flow5_threshold=investor_flow5_threshold,
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


def upsert_daily_stock_tier(conn, rows_df, batch_size=10000):
    if rows_df.empty:
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
    total_affected = 0
    with conn.cursor() as cur:
        for offset in range(0, len(rows_df), batch_size):
            chunk_df = rows_df.iloc[offset : offset + batch_size]
            chunk_rows = list(chunk_df.itertuples(index=False, name=None))
            cur.executemany(sql, chunk_rows)
            chunk_affected = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            total_affected += chunk_affected
        conn.commit()
    return total_affected


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
    enable_investor_v1_write=DEFAULT_ENABLE_INVESTOR_V1_WRITE,
    investor_flow5_threshold=DEFAULT_INVESTOR_FLOW5_THRESHOLD,
):
    if mode not in {"daily", "backfill"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if end_date_str is None:
        end_date_str = datetime.today().strftime("%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d").date()

    if ticker_codes is None:
        ticker_codes = get_tier_ticker_universe(conn, end_date=end_date, mode=mode)

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
        start_date=query_start.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        ticker_codes=ticker_codes,
    )
    investor_df = None
    if enable_investor_v1_write:
        investor_df = fetch_investor_history(
            conn=conn,
            start_date=query_start.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            ticker_codes=ticker_codes,
        )
    calculated = build_daily_stock_tier_frame(
        price_df=price_df,
        financial_df=financial_df,
        investor_df=investor_df,
        lookback_days=lookback_days,
        financial_lag_days=financial_lag_days,
        danger_liquidity=danger_liquidity,
        prime_liquidity=prime_liquidity,
        enable_investor_v1_write=enable_investor_v1_write,
        investor_flow5_threshold=investor_flow5_threshold,
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
    rows_df = in_range[
        ["date", "stock_code", "tier", "reason", "liquidity_20d_avg_value"]
    ]
    saved = upsert_daily_stock_tier(conn, rows_df)
    investor_overlay_rows = 0
    if enable_investor_v1_write:
        investor_overlay_rows = int(
            in_range["reason"].astype(str).str.contains("investor_flow5", na=False).sum()
        )

    return {
        "rows_saved": max(saved, 0),
        "rows_calculated": len(in_range),
        "tier_v1_write_enabled": bool(enable_investor_v1_write),
        "investor_overlay_rows": investor_overlay_rows,
        "investor_flow5_threshold": (
            int(investor_flow5_threshold) if enable_investor_v1_write else None
        ),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
