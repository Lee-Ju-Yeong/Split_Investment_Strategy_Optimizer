# src/indicator_calculator_gpu.py
import pandas as pd
import cudf
import cupy as cp

def calculate_indicators_gpu(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    주어진 OHLCV 데이터프레임(pandas)으로 각종 기술적 지표를 GPU(cuDF)를 사용하여 계산합니다.
    """
    if df_ohlcv.empty:
        return pd.DataFrame()

    # 1. Pandas DataFrame을 cuDF DataFrame으로 변환 (CPU -> GPU 데이터 전송)
    gdf = cudf.from_pandas(df_ohlcv)

    indicators = cudf.DataFrame(index=gdf.index)

    # 2. cuDF를 사용하여 GPU에서 모든 계산 수행 (코드는 pandas와 거의 동일)
    # 이동평균
    indicators['ma_5'] = gdf['close_price'].rolling(window=5, min_periods=5).mean()
    indicators['ma_20'] = gdf['close_price'].rolling(window=20, min_periods=20).mean()

    # ATR (Average True Range)
    tr_df = cudf.concat([
        gdf['high_price'] - gdf['low_price'],
        (gdf['high_price'] - gdf['close_price'].shift()).abs(),
        (gdf['low_price'] - gdf['close_price'].shift()).abs()
    ], axis=1)
    tr = tr_df.max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    indicators['atr_14_ratio'] = atr_14 / gdf['close_price']

    # 주가 위치 비율
    years_5 = 252 * 5
    years_10 = 252 * 10
    
    # 5년/10년 롤링 최고가/최저가
    low_5y = gdf['low_price'].rolling(window=years_5, min_periods=252).min()
    high_5y = gdf['high_price'].rolling(window=years_5, min_periods=252).max()
    low_10y = gdf['low_price'].rolling(window=years_10, min_periods=252).min()
    high_10y = gdf['high_price'].rolling(window=years_10, min_periods=252).max()

    # 위치 비율 계산 (분모가 0이 되는 경우 방지)
    indicators['price_vs_5y_low_pct'] = ((gdf['close_price'] - low_5y) / (high_5y - low_5y)).fillna(0)
    indicators['price_vs_10y_low_pct'] = ((gdf['close_price'] - low_10y) / (high_10y - low_10y)).fillna(0)
    
    # 0~1 사이 값으로 클리핑
    indicators['price_vs_5y_low_pct'] = indicators['price_vs_5y_low_pct'].clip(0, 1)
    indicators['price_vs_10y_low_pct'] = indicators['price_vs_10y_low_pct'].clip(0, 1)

    # 3. 계산 완료된 cuDF DataFrame을 다시 Pandas DataFrame으로 변환 (GPU -> CPU 데이터 전송)
    return indicators.to_pandas()

# 참고: 이 파일은 단독으로 실행되지 않으며, 다른 모듈에서 임포트하여 사용합니다.
# `calculate_and_store_indicators_for_all` 같은 상위 로직에서
# 이 `calculate_indicators_gpu` 함수를 호출하도록 수정해야 합니다.
