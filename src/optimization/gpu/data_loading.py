"""
GPU data preload/tensorization helpers for parameter simulation.
"""

from __future__ import annotations

import time
from datetime import timedelta
from functools import lru_cache

try:
    from ...universe_policy import (
        is_survivor_optimistic_mode,
        normalize_universe_mode,
    )
except ImportError:  # pragma: no cover
    from universe_policy import (  # type: ignore
        is_survivor_optimistic_mode,
        normalize_universe_mode,
    )

from .context import _ensure_core_deps, _ensure_gpu_deps


@lru_cache(maxsize=4)
def _get_sql_engine(engine_url: str):
    _, _, create_engine, _ = _ensure_gpu_deps()
    return create_engine(engine_url)


def _read_sql_to_cudf(query, sql_engine, parse_dates=None):
    """
    Prefer cuDF SQL read path to reduce pandas->cuDF round-trip overhead.
    Falls back to pandas.read_sql + cudf.from_pandas for compatibility.
    """
    _, cudf, _, _ = _ensure_gpu_deps()
    parse_dates = parse_dates or []

    reader = getattr(cudf, "read_sql_query", None) or getattr(cudf, "read_sql", None)
    if reader is not None:
        gdf = reader(query, sql_engine)
        for col in parse_dates:
            if col not in gdf.columns:
                continue
            dtype_name = str(getattr(gdf[col], "dtype", ""))
            if "datetime64" in dtype_name:
                continue
            gdf[col] = cudf.to_datetime(gdf[col])
        return gdf

    _, pd = _ensure_core_deps()
    df_pd = pd.read_sql(query, sql_engine, parse_dates=parse_dates)
    return cudf.from_pandas(df_pd)


# -----------------------------------------------------------------------------
# GPU Data Pre-loader
# -----------------------------------------------------------------------------
_FLOAT32_COLUMNS = (
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "atr_14_ratio",
    "cheap_score",
    "cheap_score_confidence",
    "flow5_mcap",
)


def _normalize_loaded_types(gdf):
    """Cast only columns needed for GPU simulation to compact/consistent dtypes."""
    for col in _FLOAT32_COLUMNS:
        if col not in gdf.columns:
            continue
        if str(gdf[col].dtype) != "float32":
            gdf[col] = gdf[col].astype("float32")
    # Keep market_cap as float64 to preserve nullable behavior and large integer safety.
    if "market_cap" in gdf.columns and str(gdf["market_cap"].dtype) != "float64":
        gdf["market_cap"] = gdf["market_cap"].astype("float64")
    return gdf


def _validate_adjusted_ohlc_not_null(gdf, adjusted_gate_start_date: str):
    null_counts = {}
    for col in ("open_price", "high_price", "low_price", "close_price"):
        if col not in gdf.columns:
            continue
        null_counts[col] = int(gdf[col].isna().sum())
    if any(count > 0 for count in null_counts.values()):
        first_ticker = "unknown"
        first_date = "unknown"
        try:
            _, pd = _ensure_core_deps()
            missing_mask = gdf[["open_price", "high_price", "low_price", "close_price"]].isna().any(axis=1)
            missing_rows = gdf[missing_mask].head(1)
            if hasattr(missing_rows, "to_pandas"):
                first_row_df = missing_rows.reset_index().to_pandas()
            else:
                first_row_df = missing_rows.reset_index()
            if not first_row_df.empty:
                first_ticker = str(first_row_df.iloc[0].get("ticker", "unknown"))
                date_value = first_row_df.iloc[0].get("date")
                if pd.notna(date_value):
                    first_date = str(pd.to_datetime(date_value).date())
        except Exception:
            pass
        raise ValueError(
            "Adjusted price mode found NULL adjusted OHLC values in GPU preload. "
            f"ticker={first_ticker}, date={first_date}. "
            f"Backtest window must satisfy date >= {adjusted_gate_start_date}. "
            f"null_counts={null_counts}"
        )


@lru_cache(maxsize=4)
def _has_stored_adj_ohlc_columns(engine_url: str) -> bool:
    _, pd = _ensure_core_deps()
    sql_engine = _get_sql_engine(engine_url)
    try:
        df = pd.read_sql(
            """
            SELECT COUNT(*) AS cnt
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'DailyStockPrice'
              AND COLUMN_NAME IN ('adj_open', 'adj_high', 'adj_low')
            """,
            sql_engine,
        )
        if df.empty:
            return False
        return int(df.iloc[0]["cnt"]) == 3
    except Exception:
        return False


@lru_cache(maxsize=4)
def _has_tier_flow5_mcap_column(engine_url: str) -> bool:
    _, pd = _ensure_core_deps()
    sql_engine = _get_sql_engine(engine_url)
    try:
        df = pd.read_sql(
            """
            SELECT COUNT(*) AS cnt
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'DailyStockTier'
              AND COLUMN_NAME = 'flow5_mcap'
            """,
            sql_engine,
        )
        if df.empty:
            return False
        return int(df.iloc[0]["cnt"]) == 1
    except Exception:
        return False


def _build_price_select_sql(use_adjusted_prices: bool, *, use_stored_adj_ohlc: bool = False) -> str:
    if use_adjusted_prices:
        if use_stored_adj_ohlc:
            return """
                CAST(
                    CASE
                        WHEN dsp.adj_ratio IS NULL THEN NULL
                        WHEN dsp.adj_open IS NULL THEN dsp.open_price * dsp.adj_ratio
                        WHEN ABS(dsp.adj_open - (dsp.open_price * dsp.adj_ratio)) > 1e-5
                            THEN dsp.open_price * dsp.adj_ratio
                        ELSE dsp.adj_open
                    END AS FLOAT
                ) AS open_price,
                CAST(
                    CASE
                        WHEN dsp.adj_ratio IS NULL THEN NULL
                        WHEN dsp.adj_high IS NULL THEN dsp.high_price * dsp.adj_ratio
                        WHEN ABS(dsp.adj_high - (dsp.high_price * dsp.adj_ratio)) > 1e-5
                            THEN dsp.high_price * dsp.adj_ratio
                        ELSE dsp.adj_high
                    END AS FLOAT
                ) AS high_price,
                CAST(
                    CASE
                        WHEN dsp.adj_ratio IS NULL THEN NULL
                        WHEN dsp.adj_low IS NULL THEN dsp.low_price * dsp.adj_ratio
                        WHEN ABS(dsp.adj_low - (dsp.low_price * dsp.adj_ratio)) > 1e-5
                            THEN dsp.low_price * dsp.adj_ratio
                        ELSE dsp.adj_low
                    END AS FLOAT
                ) AS low_price,
                CAST(dsp.adj_close AS FLOAT) AS close_price
            """
        return """
            CAST(
                CASE WHEN dsp.adj_ratio IS NOT NULL THEN dsp.open_price * dsp.adj_ratio ELSE NULL END
                AS FLOAT
            ) AS open_price,
            CAST(
                CASE WHEN dsp.adj_ratio IS NOT NULL THEN dsp.high_price * dsp.adj_ratio ELSE NULL END
                AS FLOAT
            ) AS high_price,
            CAST(
                CASE WHEN dsp.adj_ratio IS NOT NULL THEN dsp.low_price * dsp.adj_ratio ELSE NULL END
                AS FLOAT
            ) AS low_price,
            CAST(dsp.adj_close AS FLOAT) AS close_price
        """
    return """
        CAST(dsp.open_price AS FLOAT) AS open_price,
        CAST(dsp.high_price AS FLOAT) AS high_price,
        CAST(dsp.low_price AS FLOAT) AS low_price,
        CAST(dsp.close_price AS FLOAT) AS close_price
    """


def _build_universe_filter_clause(universe_mode: str, table_alias: str) -> str:
    normalized = normalize_universe_mode(universe_mode)
    if not is_survivor_optimistic_mode(normalized):
        return ""
    return (
        " AND NOT EXISTS ("
        "SELECT 1 FROM TickerUniverseHistory tuh "
        f"WHERE tuh.stock_code = {table_alias}.stock_code "
        "AND tuh.delisted_date IS NOT NULL)"
    )


def _build_universe_mask_frame(
    sql_engine,
    start_date_sql: str,
    end_date_sql: str,
    all_tickers,
    trading_dates_pd,
    universe_mode: str,
):
    """
    Build date x ticker PIT universe mask at data-loading stage.
    strict_pit:
      - baseline: History active interval (listed<=d<delisted)
      - override: latest Snapshot(as-of d) when snapshot exists
    optimistic_survivor:
      - all True (actual delisted exclusion is already pushed down in SQL filters)
    """
    _, pd = _ensure_core_deps()
    mode = normalize_universe_mode(universe_mode)

    trading_index = pd.DatetimeIndex(trading_dates_pd)
    ticker_columns = [str(t) for t in all_tickers]
    mask = pd.DataFrame(False, index=trading_index, columns=ticker_columns, dtype=bool)
    if mask.empty:
        return mask

    if is_survivor_optimistic_mode(mode):
        mask.loc[:, :] = True
        return mask

    history_query = """
        SELECT stock_code AS ticker, listed_date, delisted_date
        FROM TickerUniverseHistory
        WHERE listed_date <= %s
          AND (delisted_date IS NULL OR delisted_date > %s)
    """
    history_df = pd.read_sql(
        history_query,
        sql_engine,
        params=(end_date_sql, start_date_sql),
        parse_dates=["listed_date", "delisted_date"],
    )
    if not history_df.empty:
        history_df["ticker"] = history_df["ticker"].astype(str)
        history_df = history_df[history_df["ticker"].isin(mask.columns)]
        if not history_df.empty:
            first_day = trading_index[0]
            last_day = trading_index[-1]
            for row in history_df.itertuples(index=False):
                listed_date = pd.to_datetime(row.listed_date)
                delisted_date = (
                    pd.to_datetime(row.delisted_date)
                    if pd.notna(row.delisted_date)
                    else last_day + pd.Timedelta(days=1)
                )
                start = max(first_day, listed_date)
                end = min(last_day, delisted_date - pd.Timedelta(days=1))
                if start > end:
                    continue
                day_mask = (mask.index >= start) & (mask.index <= end)
                mask.loc[day_mask, row.ticker] = True

    snapshot_query = """
        SELECT snapshot_date AS date, stock_code AS ticker
        FROM TickerUniverseSnapshot
        WHERE snapshot_date <= %s
          AND snapshot_date >= (
              SELECT COALESCE(MAX(snapshot_date), '1900-01-01')
              FROM TickerUniverseSnapshot
              WHERE snapshot_date <= %s
          )
        ORDER BY snapshot_date, stock_code
    """
    snapshot_df = pd.read_sql(
        snapshot_query,
        sql_engine,
        params=(end_date_sql, start_date_sql),
        parse_dates=["date"],
    )
    if snapshot_df.empty:
        return mask

    snapshot_df["ticker"] = snapshot_df["ticker"].astype(str)
    snapshot_df = snapshot_df[snapshot_df["ticker"].isin(mask.columns)]
    if snapshot_df.empty:
        return mask

    snapshot_dates = pd.DataFrame(
        {"snapshot_date": pd.DatetimeIndex(snapshot_df["date"].drop_duplicates().sort_values())}
    )
    trade_dates = pd.DataFrame({"trade_date": trading_index})
    trade_to_snapshot = pd.merge_asof(
        trade_dates.sort_values("trade_date"),
        snapshot_dates.sort_values("snapshot_date"),
        left_on="trade_date",
        right_on="snapshot_date",
        direction="backward",
    )
    grouped_snapshot = (
        snapshot_df.groupby("date")["ticker"]
        .apply(list)
        .to_dict()
    )
    for row in trade_to_snapshot.itertuples(index=False):
        if pd.isna(row.snapshot_date):
            continue
        allowed = grouped_snapshot.get(row.snapshot_date, [])
        mask.loc[row.trade_date, :] = False
        if allowed:
            mask.loc[row.trade_date, allowed] = True
    return mask


def preload_all_data_to_gpu(
    engine,
    start_date,
    end_date,
    *,
    use_adjusted_prices=False,
    adjusted_price_gate_start_date="2013-11-20",
    universe_mode="optimistic_survivor",
):
    _ensure_gpu_deps()
    _, pd = _ensure_core_deps()

    print("⏳ Loading all stock data into GPU memory...")
    start_time = time.time()
    start_date_sql = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date_sql = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    adjusted_gate_sql = pd.to_datetime(adjusted_price_gate_start_date).strftime("%Y-%m-%d")

    has_stored_adj_ohlc = (
        _has_stored_adj_ohlc_columns(str(engine)) if use_adjusted_prices else False
    )
    has_tier_flow5_mcap = _has_tier_flow5_mcap_column(str(engine))
    price_select_sql = _build_price_select_sql(
        bool(use_adjusted_prices),
        use_stored_adj_ohlc=has_stored_adj_ohlc,
    )
    flow5_mcap_sql = (
        "CAST(dst.flow5_mcap AS FLOAT) AS flow5_mcap"
        if has_tier_flow5_mcap
        else "CAST(NULL AS FLOAT) AS flow5_mcap"
    )
    universe_filter_clause = _build_universe_filter_clause(universe_mode, "dsp")
    adjusted_gate_clause = (
        f" AND dsp.date >= '{adjusted_gate_sql}'"
        if use_adjusted_prices
        else ""
    )
    query = f"""
    SELECT
        dsp.stock_code AS ticker,
        dsp.date,
        {price_select_sql},
        CAST(ci.atr_14_ratio AS FLOAT) AS atr_14_ratio,
        mcd.market_cap AS market_cap,
        CAST(dst.cheap_score AS FLOAT) AS cheap_score,
        CAST(dst.cheap_score_confidence AS FLOAT) AS cheap_score_confidence,
        {flow5_mcap_sql}
    FROM
        DailyStockPrice AS dsp
    LEFT JOIN
        CalculatedIndicators AS ci ON dsp.stock_code = ci.stock_code AND dsp.date = ci.date
    LEFT JOIN
        MarketCapDaily AS mcd ON dsp.stock_code = mcd.stock_code AND dsp.date = mcd.date
    LEFT JOIN
        DailyStockTier AS dst ON dsp.stock_code = dst.stock_code AND dsp.date = dst.date
    WHERE
        dsp.date BETWEEN '{start_date_sql}' AND '{end_date_sql}'
        {adjusted_gate_clause}
        {universe_filter_clause}
    """
    sql_engine = _get_sql_engine(str(engine))
    gdf = _read_sql_to_cudf(query, sql_engine, parse_dates=["date"]).set_index(["ticker", "date"])
    gdf = _normalize_loaded_types(gdf)
    if use_adjusted_prices:
        _validate_adjusted_ohlc_not_null(gdf, adjusted_gate_sql)
    print(f"✅ Data loaded to GPU. Shape: {gdf.shape}. Time: {time.time() - start_time:.2f}s")
    return gdf


def preload_weekly_filtered_stocks_to_gpu(engine, start_date, end_date):
    _ensure_gpu_deps()
    _, pd = _ensure_core_deps()

    print("⏳ Loading weekly filtered stocks data to GPU memory...")
    start_time = time.time()
    extended_start_date = pd.to_datetime(start_date) - timedelta(days=14)
    query = (
        "SELECT `filter_date` as date, `stock_code` as ticker "
        "FROM WeeklyFilteredStocks "
        f"WHERE `filter_date` BETWEEN '{extended_start_date.strftime('%Y-%m-%d')}' AND '{end_date}'"
    )
    sql_engine = _get_sql_engine(str(engine))
    gdf = _read_sql_to_cudf(query, sql_engine, parse_dates=["date"]).set_index("date")
    print(f"✅ Weekly filtered stocks loaded to GPU. Shape: {gdf.shape}. Time: {time.time() - start_time:.2f}s")
    return gdf


def build_empty_weekly_filtered_gpu():
    _, cudf, _, _ = _ensure_gpu_deps()
    empty_df = cudf.DataFrame(
        {
            "date": cudf.Series([], dtype="datetime64[ns]"),
            "ticker": cudf.Series([], dtype="str"),
        }
    )
    return empty_df.set_index("date")


def preload_tier_data_to_tensor(
    engine,
    start_date,
    end_date,
    all_tickers,
    trading_dates_pd,
    *,
    universe_mode="optimistic_survivor",
):
    """
    Loads DailyStockTier data and converts it to a dense (num_days, num_tickers) int8 tensor.
    Performs forward-fill to ensure PIT compliance (latest <= date).
    """
    cp, _, _, _ = _ensure_gpu_deps()
    _, pd = _ensure_core_deps()

    print("⏳ Loading DailyStockTier data to GPU tensor...")
    start_time = time.time()

    start_date_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    universe_filter_t = _build_universe_filter_clause(universe_mode, "t")
    query = f"""
        SELECT t.date, t.stock_code as ticker, t.tier
        FROM DailyStockTier t
        WHERE t.date BETWEEN '{start_date_str}' AND '{end_date_str}'
          {universe_filter_t}
        UNION ALL
        SELECT t.date, t.stock_code as ticker, t.tier
        FROM DailyStockTier t
        JOIN (
            SELECT stock_code, MAX(date) AS max_date
            FROM DailyStockTier
            WHERE date < '{start_date_str}'
            GROUP BY stock_code
        ) latest ON t.stock_code = latest.stock_code AND t.date = latest.max_date
        WHERE 1=1
          {universe_filter_t}
    """
    sql_engine = _get_sql_engine(str(engine))
    df_pd = pd.read_sql(query, sql_engine, parse_dates=["date"])

    if df_pd.empty:
        print("⚠️ No Tier data found. Returning empty tensor.")
        return cp.zeros((len(trading_dates_pd), len(all_tickers)), dtype=cp.int8)

    df_reindexed = _build_tier_frame(df_pd, trading_dates_pd, all_tickers)
    universe_mask = _build_universe_mask_frame(
        sql_engine=sql_engine,
        start_date_sql=start_date_str,
        end_date_sql=end_date_str,
        all_tickers=all_tickers,
        trading_dates_pd=trading_dates_pd,
        universe_mode=universe_mode,
    )
    if not universe_mask.empty:
        df_reindexed = df_reindexed.where(universe_mask, 0)
    tier_tensor = cp.asarray(df_reindexed.values, dtype=cp.int8)

    print(f"✅ Tier data loaded and tensorized. Shape: {tier_tensor.shape}. Time: {time.time() - start_time:.2f}s")
    return tier_tensor


def _build_tier_frame(df_pd, trading_dates_pd, all_tickers):
    df_wide = (
        df_pd.assign(ticker=df_pd["ticker"].astype(str))
        .pivot_table(index="date", columns="ticker", values="tier")
        .sort_index()
    )
    # Reindexing directly to trading_dates drops all pre-start history rows.
    # Build a union index first, then forward-fill, so latest tier <= date is kept.
    union_index = df_wide.index.union(trading_dates_pd).sort_values()
    df_ffilled = df_wide.reindex(index=union_index).ffill()
    return (
        df_ffilled.reindex(index=trading_dates_pd, columns=[str(t) for t in all_tickers])
        .fillna(0)
        .astype(int)
    )


__all__ = [
    "_get_sql_engine",
    "_read_sql_to_cudf",
    "build_empty_weekly_filtered_gpu",
    "preload_all_data_to_gpu",
    "preload_weekly_filtered_stocks_to_gpu",
    "preload_tier_data_to_tensor",
]
