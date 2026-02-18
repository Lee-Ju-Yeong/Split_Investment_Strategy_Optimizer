"""
GPU data preload/tensorization helpers for parameter simulation.
"""

from __future__ import annotations

import time
from datetime import timedelta

from .context import _ensure_core_deps, _ensure_gpu_deps


# -----------------------------------------------------------------------------
# GPU Data Pre-loader
# -----------------------------------------------------------------------------
def preload_all_data_to_gpu(engine, start_date, end_date):
    _, cudf, create_engine, _ = _ensure_gpu_deps()
    _, pd = _ensure_core_deps()

    print("⏳ Loading all stock data into GPU memory...")
    start_time = time.time()
    query = f"""
    SELECT
        dsp.stock_code AS ticker,
        dsp.date,
        dsp.open_price,
        dsp.high_price,
        dsp.low_price,
        dsp.close_price,
        dsp.volume,
        ci.atr_14_ratio,
        mcd.market_cap
    FROM
        DailyStockPrice AS dsp
    LEFT JOIN
        CalculatedIndicators AS ci ON dsp.stock_code = ci.stock_code AND dsp.date = ci.date
    LEFT JOIN
        MarketCapDaily AS mcd ON dsp.stock_code = mcd.stock_code AND dsp.date = mcd.date
    WHERE
        dsp.date BETWEEN '{start_date}' AND '{end_date}'
    """
    sql_engine = create_engine(engine)
    df_pd = pd.read_sql(query, sql_engine, parse_dates=["date"])
    gdf = cudf.from_pandas(df_pd).set_index(["ticker", "date"])
    print(f"✅ Data loaded to GPU. Shape: {gdf.shape}. Time: {time.time() - start_time:.2f}s")
    return gdf


def preload_weekly_filtered_stocks_to_gpu(engine, start_date, end_date):
    _, cudf, create_engine, _ = _ensure_gpu_deps()
    _, pd = _ensure_core_deps()

    print("⏳ Loading weekly filtered stocks data to GPU memory...")
    start_time = time.time()
    extended_start_date = pd.to_datetime(start_date) - timedelta(days=14)
    query = (
        "SELECT `filter_date` as date, `stock_code` as ticker "
        "FROM WeeklyFilteredStocks "
        f"WHERE `filter_date` BETWEEN '{extended_start_date.strftime('%Y-%m-%d')}' AND '{end_date}'"
    )
    sql_engine = create_engine(engine)
    df_pd = pd.read_sql(query, sql_engine, parse_dates=["date"])
    gdf = cudf.from_pandas(df_pd).set_index("date")
    print(f"✅ Weekly filtered stocks loaded to GPU. Shape: {gdf.shape}. Time: {time.time() - start_time:.2f}s")
    return gdf


def preload_tier_data_to_tensor(engine, start_date, end_date, all_tickers, trading_dates_pd):
    """
    Loads DailyStockTier data and converts it to a dense (num_days, num_tickers) int8 tensor.
    Performs forward-fill to ensure PIT compliance (latest <= date).
    """
    cp, _, create_engine, _ = _ensure_gpu_deps()
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
    sql_engine = create_engine(engine)
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
    "preload_all_data_to_gpu",
    "preload_weekly_filtered_stocks_to_gpu",
    "preload_tier_data_to_tensor",
]
