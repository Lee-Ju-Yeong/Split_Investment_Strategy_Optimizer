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


def _set_index_lean(gdf, keys):
    """
    Prefer in-place index set to avoid an extra deep-copy allocation peak on large loads.
    """
    try:
        gdf.set_index(keys, inplace=True)
        return gdf
    except (TypeError, NotImplementedError):
        # Compatibility fallback for APIs that do not expose inplace.
        return gdf.set_index(keys)


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
    gdf = _read_sql_to_cudf(query, sql_engine, parse_dates=["date"])
    gdf = _set_index_lean(gdf, ["ticker", "date"])
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
    gdf = _read_sql_to_cudf(query, sql_engine, parse_dates=["date"])
    gdf = _set_index_lean(gdf, "date")
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


def _load_snapshot_rows_for_window(sql_engine, start_date_str, end_date_str):
    _, pd = _ensure_core_deps()

    between_df = pd.read_sql(
        """
        SELECT DISTINCT snapshot_date
        FROM TickerUniverseSnapshot
        WHERE snapshot_date BETWEEN %s AND %s
        ORDER BY snapshot_date
        """,
        sql_engine,
        params=(start_date_str, end_date_str),
        parse_dates=["snapshot_date"],
    )
    prev_df = pd.read_sql(
        """
        SELECT MAX(snapshot_date) AS snapshot_date
        FROM TickerUniverseSnapshot
        WHERE snapshot_date < %s
        """,
        sql_engine,
        params=(start_date_str,),
        parse_dates=["snapshot_date"],
    )

    snapshot_dates = []
    if not between_df.empty:
        snapshot_dates.extend(pd.to_datetime(between_df["snapshot_date"]).dropna().tolist())
    if not prev_df.empty:
        prev_date = pd.to_datetime(prev_df.iloc[0]["snapshot_date"])
        if pd.notna(prev_date):
            snapshot_dates.append(prev_date)

    if not snapshot_dates:
        return pd.DataFrame(columns=["snapshot_date", "ticker"])

    unique_dates = sorted({pd.Timestamp(d) for d in snapshot_dates})
    placeholders = ", ".join(["%s"] * len(unique_dates))
    rows_query = f"""
        SELECT snapshot_date, stock_code AS ticker
        FROM TickerUniverseSnapshot
        WHERE snapshot_date IN ({placeholders})
        ORDER BY snapshot_date, stock_code
    """
    params = tuple(d.strftime("%Y-%m-%d") for d in unique_dates)
    return pd.read_sql(rows_query, sql_engine, params=params, parse_dates=["snapshot_date"])


def _fill_snapshot_asof_mask_inplace(mask, has_snapshot, snapshot_df, trading_dates_pd, ticker_to_idx):
    np, pd = _ensure_core_deps()

    if snapshot_df.empty:
        return

    snapshot_df = snapshot_df.assign(
        snapshot_date=pd.to_datetime(snapshot_df["snapshot_date"]),
        ticker=snapshot_df["ticker"].astype(str),
    )
    grouped = snapshot_df.groupby("snapshot_date")["ticker"].apply(list).sort_index()
    if grouped.empty:
        return

    snapshot_dates = pd.DatetimeIndex(grouped.index)
    snapshot_arr = snapshot_dates.to_numpy(dtype="datetime64[ns]")
    trading_arr = pd.DatetimeIndex(trading_dates_pd).to_numpy(dtype="datetime64[ns]")

    asof_pos = np.searchsorted(snapshot_arr, trading_arr, side="right") - 1
    valid_day_mask = asof_pos >= 0
    if not valid_day_mask.any():
        return

    has_snapshot[valid_day_mask] = True
    ticker_index_by_snapshot = []
    for snapshot_date in snapshot_dates:
        tickers = grouped.loc[snapshot_date]
        ticker_indices = [ticker_to_idx[t] for t in tickers if t in ticker_to_idx]
        ticker_index_by_snapshot.append(np.asarray(ticker_indices, dtype=np.int32))

    for pos in np.unique(asof_pos[valid_day_mask]):
        day_rows = np.where(asof_pos == pos)[0]
        ticker_cols = ticker_index_by_snapshot[int(pos)]
        if day_rows.size == 0 or ticker_cols.size == 0:
            continue
        mask[np.ix_(day_rows, ticker_cols)] = True


def _fill_history_fallback_mask_inplace(mask, has_snapshot, sql_engine, trading_dates_pd, ticker_to_idx):
    np, pd = _ensure_core_deps()

    missing_rows = np.flatnonzero(~has_snapshot)
    if missing_rows.size == 0:
        return

    missing_dates = pd.DatetimeIndex(trading_dates_pd)[missing_rows]
    missing_start_str = missing_dates[0].strftime("%Y-%m-%d")
    missing_end_str = missing_dates[-1].strftime("%Y-%m-%d")

    history_df = pd.read_sql(
        """
        SELECT stock_code AS ticker, listed_date, delisted_date
        FROM TickerUniverseHistory
        WHERE listed_date <= %s
          AND (delisted_date IS NULL OR delisted_date > %s)
        """,
        sql_engine,
        params=(missing_end_str, missing_start_str),
        parse_dates=["listed_date", "delisted_date"],
    )
    if history_df.empty:
        return

    history_df = history_df.assign(ticker=history_df["ticker"].astype(str))
    missing_arr = missing_dates.to_numpy(dtype="datetime64[ns]")
    for row in history_df.itertuples(index=False):
        ticker_idx = ticker_to_idx.get(str(row.ticker))
        if ticker_idx is None:
            continue
        listed_date = pd.Timestamp(row.listed_date).to_datetime64()
        if pd.isna(row.delisted_date):
            active_mask = missing_arr >= listed_date
        else:
            delisted_date = pd.Timestamp(row.delisted_date).to_datetime64()
            active_mask = (missing_arr >= listed_date) & (missing_arr < delisted_date)
        if not active_mask.any():
            continue
        mask[missing_rows[active_mask], ticker_idx] = True


def preload_pit_universe_mask_to_tensor(engine, start_date, end_date, all_tickers, trading_dates_pd):
    """
    Build PIT universe mask tensor with CPU parity semantics:
    - Prefer latest TickerUniverseSnapshot(as-of <= date)
    - Fallback to TickerUniverseHistory(active as-of) when snapshot is unavailable for a day
    """
    cp, _, _, _ = _ensure_gpu_deps()
    np, pd = _ensure_core_deps()

    print("⏳ Loading PIT universe mask to GPU tensor...")
    start_time = time.time()

    num_days = len(trading_dates_pd)
    num_tickers = len(all_tickers)
    if num_days == 0 or num_tickers == 0:
        return cp.zeros((num_days, num_tickers), dtype=cp.int8)

    start_date_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    sql_engine = _get_sql_engine(str(engine))

    ticker_to_idx = {str(ticker): idx for idx, ticker in enumerate(all_tickers)}
    mask = np.zeros((num_days, num_tickers), dtype=np.bool_)
    has_snapshot = np.zeros(num_days, dtype=np.bool_)

    snapshot_df = _load_snapshot_rows_for_window(
        sql_engine=sql_engine,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
    )
    _fill_snapshot_asof_mask_inplace(
        mask=mask,
        has_snapshot=has_snapshot,
        snapshot_df=snapshot_df,
        trading_dates_pd=trading_dates_pd,
        ticker_to_idx=ticker_to_idx,
    )
    _fill_history_fallback_mask_inplace(
        mask=mask,
        has_snapshot=has_snapshot,
        sql_engine=sql_engine,
        trading_dates_pd=trading_dates_pd,
        ticker_to_idx=ticker_to_idx,
    )

    pit_mask_tensor = cp.asarray(mask, dtype=cp.int8)
    print(f"✅ PIT mask loaded and tensorized. Shape: {pit_mask_tensor.shape}. Time: {time.time() - start_time:.2f}s")
    return pit_mask_tensor


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
    "preload_pit_universe_mask_to_tensor",
    "preload_weekly_filtered_stocks_to_gpu",
    "preload_tier_data_to_tensor",
]
