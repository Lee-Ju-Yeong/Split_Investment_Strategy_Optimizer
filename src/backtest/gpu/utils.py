"""
GPU pricing and small helper utilities.
"""

import math

import cupy as cp
import pandas as pd

_ADJUST_PRICE_FORCE_CHUNKED = False
_ADJUST_PRICE_CHUNK_SIZE = 1_000_000
_ADJUST_PRICE_LARGE_INPUT_ELEMS = 5_000_000


def _is_gpu_oom_error(error):
    if isinstance(error, MemoryError):
        return True
    oom_cls = getattr(getattr(cp.cuda, "memory", None), "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(error, oom_cls):
        return True

    message = str(error).lower()
    markers = (
        "out_of_memory",
        "out of memory",
        "std::bad_alloc",
        "failed to allocate",
        "cudaerroroutofmemory",
    )
    return any(marker in message for marker in markers)


def get_tick_size_gpu(price_array):
    """ Vectorized tick size calculation on GPU. """
    # cp.select는 내부적으로 큰 임시 배열들을 생성하여 메모리 사용량이 많습니다.
    # cp.where를 연쇄적으로 사용하여 단일 결과 배열을 점진적으로 채워나가 메모리 사용량을 최소화합니다.
    # 기본값(1000원)으로 결과 배열을 초기화합니다.
    result = cp.full_like(price_array, 1000, dtype=cp.int32)
    
    # 가격이 낮은 조건부터 순서대로 값을 덮어씁니다.
    result = cp.where(price_array < 500000, 500, result)
    result = cp.where(price_array < 200000, 100, result)
    result = cp.where(price_array < 50000, 50, result)
    result = cp.where(price_array < 20000, 10, result)
    result = cp.where(price_array < 5000, 5, result)
    result = cp.where(price_array < 2000, 1, result)
    
    return result


def _adjust_price_up_gpu_float64_inplace(price_array):
    tick_size = get_tick_size_gpu(price_array)
    adjusted = price_array.astype(cp.float64, copy=True)
    cp.divide(adjusted, tick_size, out=adjusted)
    cp.round(adjusted, 5, out=adjusted)
    cp.ceil(adjusted, out=adjusted)
    cp.multiply(adjusted, tick_size, out=adjusted)
    return adjusted.astype(cp.float32)


def _adjust_price_up_gpu_chunked(price_array, chunk_size=_ADJUST_PRICE_CHUNK_SIZE):
    safe_chunk_size = max(int(chunk_size), 1)
    if price_array.ndim == 0:
        return _adjust_price_up_gpu_float64_inplace(price_array)

    if price_array.ndim == 1:
        output = cp.empty(price_array.shape[0], dtype=cp.float32)
        for start in range(0, price_array.shape[0], safe_chunk_size):
            end = min(start + safe_chunk_size, price_array.shape[0])
            output[start:end] = _adjust_price_up_gpu_float64_inplace(price_array[start:end])
        return output

    # Avoid reshape(-1) because non-contiguous/broadcast arrays trigger a full copy allocation.
    tail_elems = max(math.prod(price_array.shape[1:]), 1)
    row_chunk = max(safe_chunk_size // tail_elems, 1)
    output = cp.empty(price_array.shape, dtype=cp.float32)
    for start in range(0, price_array.shape[0], row_chunk):
        end = min(start + row_chunk, price_array.shape[0])
        output[start:end] = _adjust_price_up_gpu_float64_inplace(price_array[start:end])
    return output


def adjust_price_up_gpu(price_array):
    """ Vectorized price adjustment on GPU. """
    global _ADJUST_PRICE_FORCE_CHUNKED
    if (
        price_array.size >= _ADJUST_PRICE_LARGE_INPUT_ELEMS
        and (price_array.ndim > 1 or not price_array.flags.c_contiguous)
    ):
        return _adjust_price_up_gpu_chunked(price_array)
    if _ADJUST_PRICE_FORCE_CHUNKED:
        return _adjust_price_up_gpu_chunked(price_array)

    try:
        return _adjust_price_up_gpu_float64_inplace(price_array)
    except Exception as err:
        if not _is_gpu_oom_error(err):
            raise
        _ADJUST_PRICE_FORCE_CHUNKED = True
        print(
            "[GPU_WARNING] adjust_price_up_gpu OOM on full vector path; "
            "switching to chunked fallback."
        )
        return _adjust_price_up_gpu_chunked(price_array)


def _resolve_signal_date_for_gpu(day_idx: int, trading_dates_pd_cpu: pd.DatetimeIndex):
    if day_idx <= 0:
        return None, -1
    signal_day_idx = day_idx - 1
    return trading_dates_pd_cpu[signal_day_idx], signal_day_idx


def _build_ticker_rank_codes(tickers: list[str]) -> cp.ndarray:
    if not tickers:
        return cp.array([], dtype=cp.int32)
    unique_sorted_tickers = sorted(set(tickers))
    ticker_rank_map = {ticker: idx for idx, ticker in enumerate(unique_sorted_tickers)}
    return cp.asarray([ticker_rank_map[ticker] for ticker in tickers], dtype=cp.int32)


def _sort_candidates_by_atr_then_market_cap_then_ticker(candidate_records):
    """
    Backward-compatible helper kept for legacy callers/tests.
    candidate_records item format:
      (ticker, atr_q, market_cap_q, ...)
    """
    if not candidate_records:
        return []

    tickers = [item[0] for item in candidate_records]
    atr_scores = cp.asarray([int(item[1]) for item in candidate_records], dtype=cp.int64)
    market_caps = cp.asarray([int(item[2]) for item in candidate_records], dtype=cp.int64)
    ticker_ranks = _build_ticker_rank_codes(tickers)

    # Primary: atr desc, Secondary: market_cap desc, Tertiary: ticker asc
    sort_keys = cp.stack([ticker_ranks, -market_caps, -atr_scores], axis=0)
    sort_indices = cp.lexsort(sort_keys)
    return [candidate_records[idx] for idx in sort_indices.get().tolist()]


def _sort_candidates_by_market_cap_then_atr_then_ticker(candidate_records):
    """
    Deterministic candidate ranking for parity:
    1) market_cap_q desc
    2) atr_q desc
    3) ticker asc
    candidate_records item format:
      (ticker, market_cap_q, atr_q, ...)
    """
    if not candidate_records:
        return []

    tickers = [item[0] for item in candidate_records]
    market_caps = cp.asarray([int(item[1]) for item in candidate_records], dtype=cp.int64)
    atr_scores = cp.asarray([int(item[2]) for item in candidate_records], dtype=cp.int64)
    ticker_ranks = _build_ticker_rank_codes(tickers)

    # cp.lexsort는 마지막 키가 1순위다.
    sort_keys = cp.stack([ticker_ranks, -atr_scores, -market_caps], axis=0)
    sort_indices = cp.lexsort(sort_keys)
    return [candidate_records[idx] for idx in sort_indices.get().tolist()]


def _sort_candidates_by_atr_then_ticker(candidate_pairs):
    if not candidate_pairs:
        return []

    tickers = [item[0] for item in candidate_pairs]
    atr_values = cp.asarray([float(item[1]) for item in candidate_pairs], dtype=cp.float64)
    ticker_ranks = _build_ticker_rank_codes(tickers)

    sort_keys = cp.stack([ticker_ranks, -atr_values], axis=0)
    sort_indices = cp.lexsort(sort_keys)
    return [candidate_pairs[idx] for idx in sort_indices.get().tolist()]
