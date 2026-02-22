"""
GPU pricing and small helper utilities.
"""

import cupy as cp
import pandas as pd

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

def adjust_price_up_gpu(price_array):
    """ Vectorized price adjustment on GPU. """
    # CPU 경로(_adjust_price_up)의 python float round/ceil 동작과 정합을 맞추기 위해
    # 연산 구간을 float64로 승격 후 반올림/올림을 수행합니다.
    tick_size = get_tick_size_gpu(price_array).astype(cp.float64)
    divided = price_array.astype(cp.float64) / tick_size
    rounded = cp.round(divided, 5) 
    adjusted = cp.ceil(rounded) * tick_size
    return adjusted.astype(cp.float32)


def _resolve_signal_date_for_gpu(day_idx: int, trading_dates_pd_cpu: pd.DatetimeIndex):
    if day_idx <= 0:
        return None, -1
    signal_day_idx = day_idx - 1
    return trading_dates_pd_cpu[signal_day_idx], signal_day_idx

def _sort_candidates_by_atr_then_market_cap_then_ticker(candidate_records):
    """
    Deterministic candidate ranking for parity:
    1) atr_q desc
    2) market_cap_q desc
    3) ticker asc
    candidate_records item format:
      (ticker, atr_q, market_cap_q, ...)
    """
    return sorted(candidate_records, key=lambda item: (-item[1], -item[2], item[0]))


def _sort_candidates_by_market_cap_then_atr_then_ticker(candidate_records):
    """
    Backward-compatible helper kept for legacy callers.
    candidate_records item format:
      (ticker, market_cap_q, atr_q, ...)
    """
    return sorted(candidate_records, key=lambda item: (-item[1], -item[2], item[0]))


def _sort_candidates_by_atr_then_ticker(candidate_pairs):
    pairs_sorted_by_ticker = sorted(candidate_pairs, key=lambda pair: pair[0])
    return sorted(pairs_sorted_by_ticker, key=lambda pair: pair[1], reverse=True)
