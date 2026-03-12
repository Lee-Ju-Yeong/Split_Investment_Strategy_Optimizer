"""
GPU data tensor builders and ATR candidate helpers.
"""

import time

import cudf
import cupy as cp
import pandas as pd


RANKING_OPTIONAL_COLUMNS = {
    "cheap_score": 0.0,
    "cheap_score_confidence": 0.0,
    "flow5_mcap": 0.0,
}


def ensure_cheap_score_columns(metrics_df):
    """
    Ensure ranking columns exist exactly once on the input frame.
    Returns list of missing columns that were created with zero defaults.
    """
    missing_columns = []
    for col, default_value in RANKING_OPTIONAL_COLUMNS.items():
        if col not in metrics_df.columns:
            metrics_df[col] = default_value
            missing_columns.append(col)
    return missing_columns


def _build_tensor_indices(data_valid: cudf.DataFrame) -> tuple[cp.ndarray, cp.ndarray]:
    day_indices = cp.asarray(data_valid["day_idx"].astype(cp.int32))
    ticker_indices = cp.asarray(data_valid["ticker_idx"].astype(cp.int32))
    return day_indices, ticker_indices


def _forward_fill_tensor_with_presence(
    tensor: cp.ndarray,
    presence_mask: cp.ndarray,
) -> cp.ndarray:
    if tensor.size == 0:
        return tensor

    day_indices = cp.arange(tensor.shape[0], dtype=cp.int32).reshape(-1, 1)
    ticker_indices = cp.arange(tensor.shape[1], dtype=cp.int32).reshape(1, -1)
    last_valid_day_idx = cp.where(presence_mask, day_indices, 0)

    try:
        last_valid_day_idx = cp.maximum.accumulate(last_valid_day_idx, axis=0)
        seen_mask = cp.maximum.accumulate(presence_mask.astype(cp.int8), axis=0) > 0
        filled = tensor[last_valid_day_idx, ticker_indices]
        return cp.where(seen_mask, filled, cp.zeros_like(filled))
    except NotImplementedError:
        filled = tensor.copy()
        seen_mask = presence_mask.copy()
        for day_idx in range(1, filled.shape[0]):
            filled[day_idx] = cp.where(seen_mask[day_idx], filled[day_idx], filled[day_idx - 1])
            seen_mask[day_idx] = seen_mask[day_idx] | seen_mask[day_idx - 1]
        return cp.where(seen_mask, filled, cp.zeros_like(filled))


def create_gpu_data_tensors(all_data_gpu: cudf.DataFrame, all_tickers: list, trading_dates_pd: pd.Index) -> dict:
    """
    [수정] 인덱스 매핑을 사용하여 Long-format cuDF를 Wide-format CuPy 텐서로 직접 변환합니다.
    이 방식은 pivot/join보다 명시적이고 데이터 정렬 오류에 강건합니다.
    """
    print("⏳ Creating wide-format GPU data tensors using direct index mapping...")
    start_time = time.time()

    num_days = len(trading_dates_pd)
    num_tickers = len(all_tickers)

    # 1. 날짜와 티커를 정수 인덱스로 매핑하는 딕셔너리 생성
    #    trading_dates_pd는 DatetimeIndex, all_tickers는 list 여야 함
    date_map = {date.to_datetime64(): i for i, date in enumerate(trading_dates_pd)}
    ticker_map = {ticker: i for i, ticker in enumerate(all_tickers)}
    
    # cuDF의 map 함수를 사용하기 위해 매핑 딕셔너리를 cudf.Series로 변환
    date_map_gdf = cudf.Series(date_map)
    ticker_map_gdf = cudf.Series(ticker_map)
    
    # 2. 원본 데이터에 정수 인덱스 컬럼 추가
    #    .astype('datetime64[ns]')로 타입을 맞춰줘야 map이 잘 동작함
    all_data_gpu['day_idx'] = all_data_gpu['date'].astype('datetime64[ns]').map(date_map_gdf)
    all_data_gpu['ticker_idx'] = all_data_gpu['ticker'].map(ticker_map_gdf)
    
    # 유효한 인덱스만 필터링
    data_valid = all_data_gpu.dropna(subset=['day_idx', 'ticker_idx'])
    day_indices, ticker_indices = _build_tensor_indices(data_valid)
    
    # 3. 필요한 각 컬럼에 대해 (num_days, num_tickers) 텐서 생성하고 값 채우기
    tensors = {}
    for col_name in ['open_price', 'close_price', 'high_price', 'low_price']:
        # 0으로 채워진 빈 텐서 생성
        tensor = cp.zeros((num_days, num_tickers), dtype=cp.float32)
        
        # 값을 채워넣을 위치(row, col)와 값(value)을 CuPy 배열로 추출
        values = cp.asarray(data_valid[col_name].astype(cp.float32))
        
        # CuPy의 고급 인덱싱(fancy indexing)을 사용하여 값을 한 번에 할당
        tensor[day_indices, ticker_indices] = values
        tensors[col_name.replace('_price', '')] = tensor # "close", "high", "low" 키로 저장

    print(f"✅ GPU Tensors created successfully in {time.time() - start_time:.2f}s.")
    return tensors


def create_candidate_rank_tensors(
    all_data_gpu: cudf.DataFrame,
    all_tickers: list,
    trading_dates_pd: pd.Index,
) -> dict:
    num_days = len(trading_dates_pd)
    num_tickers = len(all_tickers)
    if num_days == 0 or num_tickers == 0:
        return {
            "atr_14_ratio": cp.zeros((num_days, num_tickers), dtype=cp.float64),
            "flow5_mcap": cp.zeros((num_days, num_tickers), dtype=cp.float64),
            "cheap_score_effective": cp.zeros((num_days, num_tickers), dtype=cp.float64),
            "market_cap_q": cp.zeros((num_days, num_tickers), dtype=cp.int64),
        }

    if "day_idx" not in all_data_gpu.columns or "ticker_idx" not in all_data_gpu.columns:
        date_map = {date.to_datetime64(): i for i, date in enumerate(trading_dates_pd)}
        ticker_map = {ticker: i for i, ticker in enumerate(all_tickers)}
        date_map_gdf = cudf.Series(date_map)
        ticker_map_gdf = cudf.Series(ticker_map)
        all_data_gpu["day_idx"] = all_data_gpu["date"].astype("datetime64[ns]").map(date_map_gdf)
        all_data_gpu["ticker_idx"] = all_data_gpu["ticker"].map(ticker_map_gdf)

    ensure_cheap_score_columns(all_data_gpu)
    data_valid = all_data_gpu.dropna(subset=["day_idx", "ticker_idx"]).copy(deep=True)
    if data_valid.empty:
        return {
            "atr_14_ratio": cp.zeros((num_days, num_tickers), dtype=cp.float64),
            "flow5_mcap": cp.zeros((num_days, num_tickers), dtype=cp.float64),
            "cheap_score_effective": cp.zeros((num_days, num_tickers), dtype=cp.float64),
            "market_cap_q": cp.zeros((num_days, num_tickers), dtype=cp.int64),
        }

    cheap_score_series = data_valid["cheap_score"].fillna(0).astype("float64")
    cheap_conf_series = data_valid["cheap_score_confidence"].fillna(0).astype("float64")
    data_valid["cheap_score_effective"] = (
        cheap_score_series.clip(lower=0.0, upper=1.0)
        * cheap_conf_series.clip(lower=0.0, upper=1.0)
    ).astype("float64")
    market_cap_series = data_valid["market_cap"].fillna(0).astype("float64")
    data_valid["market_cap_q"] = (market_cap_series // 1_000_000).clip(lower=0).astype("int64")
    data_valid["atr_14_ratio"] = data_valid["atr_14_ratio"].fillna(float("nan")).astype("float64")
    data_valid["flow5_mcap"] = data_valid["flow5_mcap"].fillna(float("nan")).astype("float64")

    day_indices, ticker_indices = _build_tensor_indices(data_valid)
    presence_mask = cp.zeros((num_days, num_tickers), dtype=cp.bool_)
    presence_mask[day_indices, ticker_indices] = True

    tensor_specs = {
        "atr_14_ratio": cp.float64,
        "flow5_mcap": cp.float64,
        "cheap_score_effective": cp.float64,
        "market_cap_q": cp.int64,
    }
    tensors = {}
    for column_name, dtype in tensor_specs.items():
        tensor = cp.zeros((num_days, num_tickers), dtype=dtype)
        values = cp.asarray(data_valid[column_name].astype(dtype))
        tensor[day_indices, ticker_indices] = values
        tensors[column_name] = _forward_fill_tensor_with_presence(tensor, presence_mask)

    return tensors



def _collect_candidate_rank_metrics_asof(all_data_reset_idx, final_candidate_indices, signal_date):
    if signal_date is None:
        return None
    if final_candidate_indices is None:
        return None
    if int(final_candidate_indices.size) == 0:
        return None
    candidate_index_series = cudf.Series(final_candidate_indices.astype(cp.int32, copy=False))
    if candidate_index_series.empty:
        return None

    ensure_cheap_score_columns(all_data_reset_idx)
    metrics_source = all_data_reset_idx

    # PIT(as-of <= date) 규칙에 맞춰 signal_date 이전/당일 전체에서 ticker별 최신 1건을 선택한다.
    candidate_rows = metrics_source[
        (metrics_source["date"] <= signal_date)
        & (metrics_source["ticker_idx"].isin(candidate_index_series))
    ][
        [
            "ticker_idx",
            "ticker",
            "date",
            "atr_14_ratio",
            "market_cap",
            "cheap_score",
            "cheap_score_confidence",
            "flow5_mcap",
        ]
    ]
    if candidate_rows.empty:
        return None

    latest_rows = candidate_rows.sort_values("date").drop_duplicates(subset=["ticker_idx"], keep="last")
    return latest_rows[
        [
            "ticker_idx",
            "ticker",
            "atr_14_ratio",
            "market_cap",
            "cheap_score",
            "cheap_score_confidence",
            "flow5_mcap",
        ]
    ]


def collect_candidate_rank_metrics_from_tensors(
    *,
    rank_metric_tensors: dict,
    final_candidate_indices,
    signal_day_idx: int,
    all_tickers: list[str],
):
    if signal_day_idx < 0:
        return None
    if final_candidate_indices is None or int(final_candidate_indices.size) == 0:
        return None

    candidate_indices = final_candidate_indices.astype(cp.int32, copy=False)
    atr_values = rank_metric_tensors["atr_14_ratio"][signal_day_idx, candidate_indices]
    valid_mask = atr_values > 0
    if not bool(cp.any(valid_mask)):
        return None

    filtered_indices = candidate_indices[valid_mask]
    filtered_tickers = [all_tickers[int(idx)] for idx in cp.asnumpy(filtered_indices).tolist()]

    return cudf.DataFrame(
        {
            "ticker_idx": filtered_indices,
            "ticker": filtered_tickers,
            "atr_14_ratio": atr_values[valid_mask],
            "flow5_mcap": rank_metric_tensors["flow5_mcap"][signal_day_idx, filtered_indices],
            "cheap_score_effective": rank_metric_tensors["cheap_score_effective"][
                signal_day_idx, filtered_indices
            ],
            "market_cap_q": rank_metric_tensors["market_cap_q"][signal_day_idx, filtered_indices],
        }
    )


def build_ranked_candidate_payload(valid_candidate_metrics_df, *, return_ranked_records=False):
    if valid_candidate_metrics_df is None or valid_candidate_metrics_df.empty:
        return cp.array([], dtype=cp.int32), cp.array([], dtype=cp.float32), []

    base_columns = ["ticker_idx", "ticker", "atr_14_ratio"]
    if "market_cap_q" in valid_candidate_metrics_df.columns:
        base_columns.append("market_cap_q")
    else:
        base_columns.append("market_cap")

    if "cheap_score_effective" in valid_candidate_metrics_df.columns:
        base_columns.append("cheap_score_effective")
    else:
        ensure_cheap_score_columns(valid_candidate_metrics_df)
        base_columns.extend(["cheap_score", "cheap_score_confidence"])

    if "flow5_mcap" not in valid_candidate_metrics_df.columns:
        ensure_cheap_score_columns(valid_candidate_metrics_df)
    base_columns.append("flow5_mcap")

    metrics_rows = valid_candidate_metrics_df[base_columns]
    metrics_rows = metrics_rows.dropna(subset=["ticker_idx", "atr_14_ratio"])
    if metrics_rows.empty:
        return cp.array([], dtype=cp.int32), cp.array([], dtype=cp.float32), []

    metrics_rows = metrics_rows[metrics_rows["atr_14_ratio"] > 0]
    if metrics_rows.empty:
        return cp.array([], dtype=cp.int32), cp.array([], dtype=cp.float32), []

    metrics_rows = metrics_rows.copy(deep=True)
    metrics_rows["ticker_idx"] = metrics_rows["ticker_idx"].astype("int32")
    if "market_cap_q" in metrics_rows.columns:
        metrics_rows["market_cap_q"] = metrics_rows["market_cap_q"].fillna(0).astype("int64")
    else:
        market_cap_series = metrics_rows["market_cap"].fillna(0).astype("float64")
        metrics_rows["market_cap_q"] = (market_cap_series // 1_000_000).clip(lower=0).astype("int64")
    if "cheap_score_effective" in metrics_rows.columns:
        metrics_rows["cheap_score_effective"] = (
            metrics_rows["cheap_score_effective"].fillna(0).astype("float64").clip(lower=0.0, upper=1.0)
        )
    else:
        cheap_score_series = metrics_rows["cheap_score"].fillna(0).astype("float64")
        cheap_conf_series = metrics_rows["cheap_score_confidence"].fillna(0).astype("float64")
        metrics_rows["cheap_score_effective"] = (
            cheap_score_series.clip(lower=0.0, upper=1.0)
            * cheap_conf_series.clip(lower=0.0, upper=1.0)
        )
    flow_rank = (
        metrics_rows["flow5_mcap"]
        .astype("float64")
        .rank(method="average", pct=True, ascending=True)
        .fillna(0.0)
    )
    atr_rank = (
        metrics_rows["atr_14_ratio"]
        .astype("float64")
        .rank(method="average", pct=True, ascending=True)
        .fillna(0.0)
    )
    metrics_rows["flow_score_q"] = (flow_rank * 10000.0).round().astype("int64")
    metrics_rows["atr_score_q"] = (atr_rank * 10000.0).round().astype("int64")
    metrics_rows["entry_composite_score"] = (
        (0.50 * metrics_rows["cheap_score_effective"])
        + (0.30 * flow_rank)
        + (0.20 * atr_rank)
    )
    metrics_rows["entry_composite_score_q"] = (
        metrics_rows["entry_composite_score"].fillna(0.0) * 10000.0
    ).round().astype("int64")

    ranked_rows = metrics_rows.sort_values(
        by=["entry_composite_score_q", "market_cap_q", "ticker"],
        ascending=[False, False, True],
    )
    candidate_indices_final = cp.asarray(ranked_rows["ticker_idx"].astype("int32"))
    valid_atrs_final = cp.asarray(ranked_rows["atr_14_ratio"].astype("float32"))

    ranked_records = []
    if return_ranked_records:
        ranked_records = list(
            zip(
                ranked_rows["ticker"].to_arrow().to_pylist(),
                ranked_rows["entry_composite_score_q"].to_arrow().to_pylist(),
                ranked_rows["flow_score_q"].to_arrow().to_pylist(),
                ranked_rows["atr_score_q"].to_arrow().to_pylist(),
                ranked_rows["market_cap_q"].to_arrow().to_pylist(),
                ranked_rows["atr_14_ratio"].astype("float32").to_arrow().to_pylist(),
            )
        )
    return candidate_indices_final, valid_atrs_final, ranked_records


def _collect_candidate_atr_asof(all_data_reset_idx, final_candidate_tickers, signal_date):
    """
    Backward-compatible helper kept for legacy tests/callers.
    Returns:
      cudf.Series(index=ticker, values=atr_14_ratio) or None
    """
    if signal_date is None or not final_candidate_tickers:
        return None

    same_day_rows = all_data_reset_idx[
        (all_data_reset_idx["date"] == signal_date)
        & (all_data_reset_idx["ticker"].isin(final_candidate_tickers))
    ][["ticker", "atr_14_ratio"]]

    available_tickers = set(same_day_rows["ticker"].to_arrow().to_pylist()) if not same_day_rows.empty else set()
    missing_tickers = [ticker for ticker in final_candidate_tickers if ticker not in available_tickers]

    if missing_tickers:
        historical_rows = all_data_reset_idx[
            (all_data_reset_idx["date"] < signal_date)
            & (all_data_reset_idx["ticker"].isin(missing_tickers))
        ][["ticker", "date", "atr_14_ratio"]]
        if not historical_rows.empty:
            latest_history_rows = historical_rows.sort_values("date").drop_duplicates(
                subset=["ticker"], keep="last"
            )
            same_day_rows = cudf.concat(
                [same_day_rows, latest_history_rows[["ticker", "atr_14_ratio"]]],
                ignore_index=True,
            )

    if same_day_rows.empty:
        return None
    return same_day_rows.drop_duplicates(subset=["ticker"], keep="first").set_index("ticker")[
        "atr_14_ratio"
    ].dropna()
