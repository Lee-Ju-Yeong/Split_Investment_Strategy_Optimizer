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
DEFAULT_FINANCIAL_LAG_DAYS = 1
DEFAULT_DANGER_LIQUIDITY = 300_000_000
DEFAULT_PRIME_LIQUIDITY = 1_000_000_000
DEFAULT_ENABLE_INVESTOR_V1_WRITE = False
DEFAULT_INVESTOR_FLOW5_THRESHOLD = -500_000_000
DEFAULT_CHEAP_SCORE_PBR_LOOKBACK_YEARS = 5
DEFAULT_CHEAP_SCORE_PER_LOOKBACK_YEARS = 3
DEFAULT_CHEAP_SCORE_DIV_LOOKBACK_YEARS = 7
DEFAULT_CHEAP_SCORE_WEIGHT_PBR = 0.45
DEFAULT_CHEAP_SCORE_WEIGHT_PER = 0.35
DEFAULT_CHEAP_SCORE_WEIGHT_DIV = 0.20
DEFAULT_CHEAP_SCORE_MIN_OBS_DAYS = 126
DEFAULT_ENABLE_SBV_TIER_OVERLAY = True
DEFAULT_SBV_TIER3_THRESHOLD = 0.0272
DEFAULT_SBV_TIER1_DEMOTE_THRESHOLD = 0.0139


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
            SELECT stock_code, date, roe, bps, per, pbr, div_yield
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
        SELECT stock_code, date, roe, bps, per, pbr, div_yield
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
        SELECT f.stock_code, f.date, f.roe, f.bps, f.per, f.pbr, f.div_yield
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


def fetch_short_balance_ratio_inputs(conn, start_date, end_date, ticker_codes=None):
    params = [start_date, end_date]
    query = """
        SELECT
            s.stock_code,
            s.date,
            s.short_balance_value,
            m.market_cap
        FROM ShortSellingDaily s
        LEFT JOIN MarketCapDaily m
          ON m.stock_code = s.stock_code
         AND m.date = s.date
        WHERE s.date BETWEEN %s AND %s
    """
    if ticker_codes:
        in_sql, in_params = _build_in_clause_params(" AND s.stock_code IN", ticker_codes)
        query += in_sql
        params.extend(in_params)
    query += " ORDER BY s.stock_code, s.date"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.read_sql(query, conn, params=params)


def _to_trading_window_days(lookback_years):
    return max(int(lookback_years) * 252, 1)


def _to_calendar_lookback_days(lookback_years):
    return max(int(lookback_years) * 366, 1)


def _grouped_rolling_pct_rank(frame, values, window_days, min_obs_days):
    return (
        values.groupby(frame["stock_code"], sort=False)
        .rolling(window=window_days, min_periods=min_obs_days)
        .rank(pct=True)
        .reset_index(level=0, drop=True)
    )


def _build_cheap_score_version(
    pbr_lookback_years,
    per_lookback_years,
    div_lookback_years,
    weight_pbr,
    weight_per,
    weight_div,
):
    pbr_token = int(round(float(weight_pbr) * 100))
    per_token = int(round(float(weight_per) * 100))
    div_token = int(round(float(weight_div) * 100))
    return (
        f"cheap_v1_pbr{int(pbr_lookback_years)}"
        f"_per{int(per_lookback_years)}"
        f"_div{int(div_lookback_years)}"
        f"_w{pbr_token:02d}{per_token:02d}{div_token:02d}"
    )


def _augment_financial_factors(
    financial_df,
    pbr_lookback_years,
    per_lookback_years,
    div_lookback_years,
    weight_pbr,
    weight_per,
    weight_div,
    min_obs_days,
):
    if financial_df is None or financial_df.empty:
        return financial_df

    output = financial_df.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output.sort_values(["stock_code", "date"], inplace=True)
    for col in ["roe", "bps", "per", "pbr", "div_yield"]:
        if col not in output.columns:
            output[col] = np.nan

    output["per"] = pd.to_numeric(output["per"], errors="coerce")
    output["pbr"] = pd.to_numeric(output["pbr"], errors="coerce")
    output["div_yield"] = pd.to_numeric(output["div_yield"], errors="coerce")

    pbr_values = output["pbr"].where(output["pbr"] > 0)
    per_values = output["per"].where(output["per"] > 0)
    div_values = output["div_yield"].where(output["div_yield"] > 0)

    pbr_rank = _grouped_rolling_pct_rank(
        output,
        pbr_values,
        window_days=_to_trading_window_days(pbr_lookback_years),
        min_obs_days=min_obs_days,
    )
    per_rank = _grouped_rolling_pct_rank(
        output,
        per_values,
        window_days=_to_trading_window_days(per_lookback_years),
        min_obs_days=min_obs_days,
    )
    div_rank = _grouped_rolling_pct_rank(
        output,
        div_values,
        window_days=_to_trading_window_days(div_lookback_years),
        min_obs_days=min_obs_days,
    )

    output["pbr_discount"] = 1.0 - pbr_rank
    output["per_discount"] = 1.0 - per_rank
    output["div_premium"] = div_rank

    total_weight = float(weight_pbr + weight_per + weight_div)
    if total_weight <= 0:
        raise ValueError("cheap score weights must sum to a positive value")

    pbr_available = output["pbr_discount"].notna().astype(float)
    per_available = output["per_discount"].notna().astype(float)
    div_available = output["div_premium"].notna().astype(float)

    weighted_numerator = (
        float(weight_pbr) * output["pbr_discount"].fillna(0.0)
        + float(weight_per) * output["per_discount"].fillna(0.0)
        + float(weight_div) * output["div_premium"].fillna(0.0)
    )
    weighted_available = (
        float(weight_pbr) * pbr_available
        + float(weight_per) * per_available
        + float(weight_div) * div_available
    )
    output["cheap_score"] = np.where(
        weighted_available > 0,
        weighted_numerator / weighted_available,
        np.nan,
    )
    output["cheap_score_confidence"] = weighted_available / total_weight
    output["cheap_score_version"] = _build_cheap_score_version(
        pbr_lookback_years=pbr_lookback_years,
        per_lookback_years=per_lookback_years,
        div_lookback_years=div_lookback_years,
        weight_pbr=weight_pbr,
        weight_per=weight_per,
        weight_div=weight_div,
    )
    return output


def _apply_financial_risk(price_df, financial_df, financial_lag_days):
    if financial_df is None or financial_df.empty:
        output = price_df.copy()
        output["financial_risk"] = False
        output["per"] = np.nan
        output["pbr"] = np.nan
        output["div_yield"] = np.nan
        output["pbr_discount"] = np.nan
        output["per_discount"] = np.nan
        output["div_premium"] = np.nan
        output["cheap_score"] = np.nan
        output["cheap_score_confidence"] = 0.0
        output["cheap_score_version"] = None
        return output

    financial_base = financial_df.copy()
    for col in [
        "per",
        "pbr",
        "div_yield",
        "pbr_discount",
        "per_discount",
        "div_premium",
        "cheap_score",
        "cheap_score_confidence",
        "cheap_score_version",
    ]:
        if col not in financial_base.columns:
            financial_base[col] = np.nan if col != "cheap_score_version" else None

    output_groups = []
    for stock_code, price_group in price_df.groupby("stock_code", sort=False):
        stock_prices = price_group.copy().sort_values("date")
        stock_financials = financial_base[
            financial_base["stock_code"] == stock_code
        ].copy()
        if stock_financials.empty:
            stock_prices["financial_risk"] = False
            stock_prices["per"] = np.nan
            stock_prices["pbr"] = np.nan
            stock_prices["div_yield"] = np.nan
            stock_prices["pbr_discount"] = np.nan
            stock_prices["per_discount"] = np.nan
            stock_prices["div_premium"] = np.nan
            stock_prices["cheap_score"] = np.nan
            stock_prices["cheap_score_confidence"] = 0.0
            stock_prices["cheap_score_version"] = None
            output_groups.append(stock_prices)
            continue

        stock_financials["date"] = pd.to_datetime(stock_financials["date"])
        stock_financials = stock_financials.sort_values("date")

        stock_prices["financial_as_of"] = stock_prices["date"] - pd.to_timedelta(
            financial_lag_days, unit="D"
        )
        merged = pd.merge_asof(
            stock_prices.sort_values("financial_as_of"),
            stock_financials[
                [
                    "date",
                    "roe",
                    "bps",
                    "per",
                    "pbr",
                    "div_yield",
                    "pbr_discount",
                    "per_discount",
                    "div_premium",
                    "cheap_score",
                    "cheap_score_confidence",
                    "cheap_score_version",
                ]
            ].sort_values("date"),
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
        merged["cheap_score_confidence"] = (
            pd.to_numeric(merged["cheap_score_confidence"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0, upper=1.0)
        )
        merged = merged.sort_values("date").drop(
            columns=["financial_as_of", "date_financial"], errors="ignore"
        )
        output_groups.append(merged)

    return pd.concat(output_groups, ignore_index=True)


def _apply_short_balance_ratio(base_df, short_balance_df):
    output = base_df.copy()
    output["sbv_ratio"] = np.nan
    if short_balance_df is None or short_balance_df.empty:
        return output

    short_data = short_balance_df.copy()
    short_data["date"] = pd.to_datetime(short_data["date"], errors="coerce")
    short_data["short_balance_value"] = pd.to_numeric(
        short_data["short_balance_value"], errors="coerce"
    )
    short_data["market_cap"] = pd.to_numeric(short_data["market_cap"], errors="coerce")
    short_data = short_data.dropna(subset=["date"])

    merged = output.merge(
        short_data[["stock_code", "date", "short_balance_value", "market_cap"]],
        on=["stock_code", "date"],
        how="left",
    )
    valid_mask = merged["short_balance_value"].notna() & (merged["market_cap"] > 0)
    merged["sbv_ratio"] = np.where(
        valid_mask,
        merged["short_balance_value"] / merged["market_cap"],
        np.nan,
    )
    merged["sbv_ratio"] = pd.to_numeric(merged["sbv_ratio"], errors="coerce")
    return merged


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
    short_balance_df=None,
    lookback_days=DEFAULT_LOOKBACK_DAYS,
    financial_lag_days=DEFAULT_FINANCIAL_LAG_DAYS,
    danger_liquidity=DEFAULT_DANGER_LIQUIDITY,
    prime_liquidity=DEFAULT_PRIME_LIQUIDITY,
    enable_investor_v1_write=DEFAULT_ENABLE_INVESTOR_V1_WRITE,
    investor_flow5_threshold=DEFAULT_INVESTOR_FLOW5_THRESHOLD,
    cheap_score_pbr_lookback_years=DEFAULT_CHEAP_SCORE_PBR_LOOKBACK_YEARS,
    cheap_score_per_lookback_years=DEFAULT_CHEAP_SCORE_PER_LOOKBACK_YEARS,
    cheap_score_div_lookback_years=DEFAULT_CHEAP_SCORE_DIV_LOOKBACK_YEARS,
    cheap_score_weight_pbr=DEFAULT_CHEAP_SCORE_WEIGHT_PBR,
    cheap_score_weight_per=DEFAULT_CHEAP_SCORE_WEIGHT_PER,
    cheap_score_weight_div=DEFAULT_CHEAP_SCORE_WEIGHT_DIV,
    cheap_score_min_obs_days=DEFAULT_CHEAP_SCORE_MIN_OBS_DAYS,
    enable_sbv_tier_overlay=DEFAULT_ENABLE_SBV_TIER_OVERLAY,
    sbv_tier3_threshold=DEFAULT_SBV_TIER3_THRESHOLD,
    sbv_tier1_demote_threshold=DEFAULT_SBV_TIER1_DEMOTE_THRESHOLD,
):
    if price_df is None or price_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "stock_code",
                "tier",
                "reason",
                "liquidity_20d_avg_value",
                "sbv_ratio",
                "pbr_discount",
                "per_discount",
                "div_premium",
                "cheap_score",
                "cheap_score_version",
                "cheap_score_confidence",
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
    base = _apply_short_balance_ratio(base, short_balance_df)

    financial_with_factors = _augment_financial_factors(
        financial_df=financial_df,
        pbr_lookback_years=cheap_score_pbr_lookback_years,
        per_lookback_years=cheap_score_per_lookback_years,
        div_lookback_years=cheap_score_div_lookback_years,
        weight_pbr=cheap_score_weight_pbr,
        weight_per=cheap_score_weight_per,
        weight_div=cheap_score_weight_div,
        min_obs_days=cheap_score_min_obs_days,
    )
    with_financial = _apply_financial_risk(
        base,
        financial_with_factors,
        financial_lag_days,
    )
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
    div_yield = pd.to_numeric(with_financial["div_yield"], errors="coerce")
    div_non_positive_mask = (
        (with_financial["tier"] == 1)
        & div_yield.notna()
        & (div_yield <= 0)
    )
    with_financial.loc[div_non_positive_mask, "tier"] = 2
    with_financial.loc[div_non_positive_mask, "reason"] = with_financial.loc[
        div_non_positive_mask, "reason"
    ].apply(lambda reason: _append_reason(reason, "div_zero_or_negative"))

    if enable_investor_v1_write:
        with_financial = _apply_investor_flow_overlay(
            with_financial=with_financial,
            investor_df=investor_df,
            investor_flow5_threshold=investor_flow5_threshold,
        )
    if enable_sbv_tier_overlay:
        sbv_ratio = pd.to_numeric(with_financial["sbv_ratio"], errors="coerce")
        sbv_tier3_mask = sbv_ratio.notna() & (sbv_ratio >= float(sbv_tier3_threshold))
        with_financial.loc[sbv_tier3_mask, "tier"] = 3
        with_financial.loc[sbv_tier3_mask, "reason"] = with_financial.loc[
            sbv_tier3_mask, "reason"
        ].apply(lambda reason: _append_reason(reason, "sbv_ratio_extreme"))

        sbv_tier1_demote_mask = (
            (with_financial["tier"] == 1)
            & sbv_ratio.notna()
            & (sbv_ratio >= float(sbv_tier1_demote_threshold))
        )
        with_financial.loc[sbv_tier1_demote_mask, "tier"] = 2
        with_financial.loc[sbv_tier1_demote_mask, "reason"] = with_financial.loc[
            sbv_tier1_demote_mask, "reason"
        ].apply(lambda reason: _append_reason(reason, "sbv_ratio_elevated"))

    output = with_financial[
        [
            "date",
            "stock_code",
            "tier",
            "reason",
            "liquidity_20d_avg_value",
            "sbv_ratio",
            "pbr_discount",
            "per_discount",
            "div_premium",
            "cheap_score",
            "cheap_score_version",
            "cheap_score_confidence",
        ]
    ].copy()
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output["tier"] = output["tier"].astype(int)
    output["liquidity_20d_avg_value"] = (
        pd.to_numeric(output["liquidity_20d_avg_value"], errors="coerce")
        .fillna(0)
        .astype(np.int64)
    )
    output["sbv_ratio"] = (
        pd.to_numeric(output["sbv_ratio"], errors="coerce")
        .clip(lower=0.0)
    )
    for col in ["pbr_discount", "per_discount", "div_premium", "cheap_score"]:
        output[col] = (
            pd.to_numeric(output[col], errors="coerce")
            .clip(lower=0.0, upper=1.0)
        )
    output["cheap_score_confidence"] = (
        pd.to_numeric(output["cheap_score_confidence"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    output["cheap_score_version"] = output["cheap_score_version"].astype(object)
    return output


def upsert_daily_stock_tier(conn, rows_df, batch_size=10000):
    if rows_df.empty:
        return 0

    sql = """
        INSERT INTO DailyStockTier (
            date,
            stock_code,
            tier,
            reason,
            liquidity_20d_avg_value,
            sbv_ratio,
            pbr_discount,
            per_discount,
            div_premium,
            cheap_score,
            cheap_score_version,
            cheap_score_confidence
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            tier = VALUES(tier),
            reason = VALUES(reason),
            liquidity_20d_avg_value = VALUES(liquidity_20d_avg_value),
            sbv_ratio = VALUES(sbv_ratio),
            pbr_discount = VALUES(pbr_discount),
            per_discount = VALUES(per_discount),
            div_premium = VALUES(div_premium),
            cheap_score = VALUES(cheap_score),
            cheap_score_version = VALUES(cheap_score_version),
            cheap_score_confidence = VALUES(cheap_score_confidence),
            computed_at = CURRENT_TIMESTAMP
    """
    total_affected = 0

    def _to_mysql_safe(value):
        if value is None:
            return None
        if pd.isna(value):
            return None
        if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
            return None
        return value

    def _to_mysql_safe_rows(frame):
        return [
            tuple(_to_mysql_safe(value) for value in row)
            for row in frame.itertuples(index=False, name=None)
        ]

    with conn.cursor() as cur:
        for offset in range(0, len(rows_df), batch_size):
            chunk_df = rows_df.iloc[offset : offset + batch_size].copy()
            chunk_rows = _to_mysql_safe_rows(chunk_df)
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
    cheap_score_pbr_lookback_years=DEFAULT_CHEAP_SCORE_PBR_LOOKBACK_YEARS,
    cheap_score_per_lookback_years=DEFAULT_CHEAP_SCORE_PER_LOOKBACK_YEARS,
    cheap_score_div_lookback_years=DEFAULT_CHEAP_SCORE_DIV_LOOKBACK_YEARS,
    cheap_score_weight_pbr=DEFAULT_CHEAP_SCORE_WEIGHT_PBR,
    cheap_score_weight_per=DEFAULT_CHEAP_SCORE_WEIGHT_PER,
    cheap_score_weight_div=DEFAULT_CHEAP_SCORE_WEIGHT_DIV,
    cheap_score_min_obs_days=DEFAULT_CHEAP_SCORE_MIN_OBS_DAYS,
    enable_sbv_tier_overlay=DEFAULT_ENABLE_SBV_TIER_OVERLAY,
    sbv_tier3_threshold=DEFAULT_SBV_TIER3_THRESHOLD,
    sbv_tier1_demote_threshold=DEFAULT_SBV_TIER1_DEMOTE_THRESHOLD,
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
    financial_lookback_days = _to_calendar_lookback_days(
        max(
            cheap_score_pbr_lookback_years,
            cheap_score_per_lookback_years,
            cheap_score_div_lookback_years,
        )
    )
    financial_query_start = query_start - timedelta(days=financial_lookback_days + 30)
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
        start_date=financial_query_start.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        ticker_codes=ticker_codes,
    )
    short_balance_df = fetch_short_balance_ratio_inputs(
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
        short_balance_df=short_balance_df,
        lookback_days=lookback_days,
        financial_lag_days=financial_lag_days,
        danger_liquidity=danger_liquidity,
        prime_liquidity=prime_liquidity,
        enable_investor_v1_write=enable_investor_v1_write,
        investor_flow5_threshold=investor_flow5_threshold,
        cheap_score_pbr_lookback_years=cheap_score_pbr_lookback_years,
        cheap_score_per_lookback_years=cheap_score_per_lookback_years,
        cheap_score_div_lookback_years=cheap_score_div_lookback_years,
        cheap_score_weight_pbr=cheap_score_weight_pbr,
        cheap_score_weight_per=cheap_score_weight_per,
        cheap_score_weight_div=cheap_score_weight_div,
        cheap_score_min_obs_days=cheap_score_min_obs_days,
        enable_sbv_tier_overlay=enable_sbv_tier_overlay,
        sbv_tier3_threshold=sbv_tier3_threshold,
        sbv_tier1_demote_threshold=sbv_tier1_demote_threshold,
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
        [
            "date",
            "stock_code",
            "tier",
            "reason",
            "liquidity_20d_avg_value",
            "sbv_ratio",
            "pbr_discount",
            "per_discount",
            "div_premium",
            "cheap_score",
            "cheap_score_version",
            "cheap_score_confidence",
        ]
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
