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
    tick_size = get_tick_size_gpu(price_array)
    # [수정] float32 나눗셈에서 발생할 수 있는 미세한 오차를 보정하기 위해
    # 소수점 5자리에서 반올림(round)한 후 올림(ceil)을 적용합니다.
    # 예: 18430 / 10 = 1843.0000001 -> round -> 1843.0 -> ceil -> 1843.0
    divided = price_array / tick_size
    rounded = cp.round(divided, 5) 
    return cp.ceil(rounded) * tick_size


def _resolve_signal_date_for_gpu(day_idx: int, trading_dates_pd_cpu: pd.DatetimeIndex):
    if day_idx <= 0:
        return None, -1
    signal_day_idx = day_idx - 1
    return trading_dates_pd_cpu[signal_day_idx], signal_day_idx

def _sort_candidates_by_atr_then_ticker(candidate_pairs):
    pairs_sorted_by_ticker = sorted(candidate_pairs, key=lambda pair: pair[0])
    return sorted(pairs_sorted_by_ticker, key=lambda pair: pair[1], reverse=True)

