"""
GPU data preload/tensorization helpers for parameter simulation.
"""

from __future__ import annotations

import time
from datetime import timedelta
from functools import lru_cache

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
def _build_price_select_sql(use_adjusted_prices: bool) -> str:
    if use_adjusted_prices:
        return """
            CASE WHEN dsp.adj_ratio IS NOT NULL THEN dsp.open_price * dsp.adj_ratio ELSE NULL END AS open_price,
            CASE WHEN dsp.adj_ratio IS NOT NULL THEN dsp.high_price * dsp.adj_ratio ELSE NULL END AS high_price,
            CASE WHEN dsp.adj_ratio IS NOT NULL THEN dsp.low_price * dsp.adj_ratio ELSE NULL END AS low_price,
            dsp.adj_close AS close_price
        """
    return """
        dsp.open_price,
        dsp.high_price,
        dsp.low_price,
        dsp.close_price
    """


def preload_all_data_to_gpu(
    engine,
    start_date,
    end_date,
    *,
    use_adjusted_prices=False,
    adjusted_price_gate_start_date="2013-11-20",
):
    _ensure_gpu_deps()

    print("⏳ Loading all stock data into GPU memory...")
    start_time = time.time()
    price_select_sql = _build_price_select_sql(bool(use_adjusted_prices))
    adjusted_gate_clause = (
        f" AND dsp.date >= '{adjusted_price_gate_start_date}'"
        if use_adjusted_prices
        else ""
    )
    query = f"""
    SELECT
        dsp.stock_code AS ticker,
        dsp.date,
        {price_select_sql},
        dsp.volume,
        ci.atr_14_ratio,
        mcd.market_cap,
        dst.cheap_score,
        dst.cheap_score_confidence
    FROM
        DailyStockPrice AS dsp
    LEFT JOIN
        CalculatedIndicators AS ci ON dsp.stock_code = ci.stock_code AND dsp.date = ci.date
    LEFT JOIN
        MarketCapDaily AS mcd ON dsp.stock_code = mcd.stock_code AND dsp.date = mcd.date
    LEFT JOIN
        DailyStockTier AS dst ON dsp.stock_code = dst.stock_code AND dsp.date = dst.date
    WHERE
        dsp.date BETWEEN '{start_date}' AND '{end_date}'
        {adjusted_gate_clause}
    """
    sql_engine = _get_sql_engine(str(engine))
    gdf = _read_sql_to_cudf(query, sql_engine, parse_dates=["date"]).set_index(["ticker", "date"])
    for col in ("cheap_score", "cheap_score_confidence"):
        if col in gdf.columns:
            gdf[col] = gdf[col].astype("float32")
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


def preload_tier_data_to_tensor(engine, start_date, end_date, all_tickers, trading_dates_pd):
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
    query = f"""
        SELECT date, stock_code as ticker, tier
        FROM DailyStockTier
        WHERE date BETWEEN '{start_date_str}' AND '{end_date_str}'
        UNION ALL
        SELECT t.date, t.stock_code as ticker, t.tier
        FROM DailyStockTier t
        JOIN (
            SELECT stock_code, MAX(date) AS max_date
            FROM DailyStockTier
            WHERE date < '{start_date_str}'
            GROUP BY stock_code
        ) latest ON t.stock_code = latest.stock_code AND t.date = latest.max_date
    """
    sql_engine = _get_sql_engine(str(engine))
    df_pd = pd.read_sql(query, sql_engine, parse_dates=["date"])

    if df_pd.empty:
        print("⚠️ No Tier data found. Returning empty tensor.")
        return cp.zeros((len(trading_dates_pd), len(all_tickers)), dtype=cp.int8)

    df_reindexed = _build_tier_frame(df_pd, trading_dates_pd, all_tickers)
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
