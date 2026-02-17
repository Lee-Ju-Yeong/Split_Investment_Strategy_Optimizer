"""
GPU data tensor builders and ATR candidate helpers.
"""

import time

import cudf
import cupy as cp
import pandas as pd

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
    
    # 3. 필요한 각 컬럼에 대해 (num_days, num_tickers) 텐서 생성하고 값 채우기
    tensors = {}
    for col_name in ['open_price', 'close_price', 'high_price', 'low_price']:
        # 0으로 채워진 빈 텐서 생성
        tensor = cp.zeros((num_days, num_tickers), dtype=cp.float32)
        
        # 값을 채워넣을 위치(row, col)와 값(value)을 CuPy 배열로 추출
        day_indices = cp.asarray(data_valid['day_idx'].astype(cp.int32))
        ticker_indices = cp.asarray(data_valid['ticker_idx'].astype(cp.int32))
        values = cp.asarray(data_valid[col_name].astype(cp.float32))
        
        # CuPy의 고급 인덱싱(fancy indexing)을 사용하여 값을 한 번에 할당
        tensor[day_indices, ticker_indices] = values
        tensors[col_name.replace('_price', '')] = tensor # "close", "high", "low" 키로 저장

    print(f"✅ GPU Tensors created successfully in {time.time() - start_time:.2f}s.")
    return tensors



def _collect_candidate_rank_metrics_asof(all_data_reset_idx, final_candidate_tickers, signal_date):
    if signal_date is None or not final_candidate_tickers:
        return None

    # CPU get_stock_row_as_of(ticker, signal_date)의 PIT(as-of <= date) 동작을 맞추기 위해
    # 우선 signal_date 당일 값을 사용하고, 결측 티커만 직전 최신 행으로 보완한다.
    # Ranking metrics: atr_14_ratio, market_cap
    same_day_rows = all_data_reset_idx[
        (all_data_reset_idx['date'] == signal_date) &
        (all_data_reset_idx['ticker'].isin(final_candidate_tickers))
    ][['ticker', 'atr_14_ratio', 'market_cap']]

    available_tickers = set(same_day_rows['ticker'].to_arrow().to_pylist()) if not same_day_rows.empty else set()
    missing_tickers = [ticker for ticker in final_candidate_tickers if ticker not in available_tickers]

    if missing_tickers:
        historical_rows = all_data_reset_idx[
            (all_data_reset_idx['date'] < signal_date) &
            (all_data_reset_idx['ticker'].isin(missing_tickers))
        ][['ticker', 'date', 'atr_14_ratio', 'market_cap']]
        if not historical_rows.empty:
            latest_history_rows = historical_rows.sort_values('date').drop_duplicates(subset=['ticker'], keep='last')
            same_day_rows = cudf.concat(
                [same_day_rows, latest_history_rows[['ticker', 'atr_14_ratio', 'market_cap']]],
                ignore_index=True
            )

    if same_day_rows.empty:
        return None
    dedup_rows = same_day_rows.drop_duplicates(subset=['ticker'], keep='first')
    return dedup_rows.set_index('ticker')[['atr_14_ratio', 'market_cap']]
