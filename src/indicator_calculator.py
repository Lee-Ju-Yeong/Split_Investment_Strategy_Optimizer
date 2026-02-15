# indicator_calculator.py
import pandas as pd
import numpy as np
import time
import warnings
from datetime import datetime, timedelta

# pandas SQLAlchemy 경고 메시지 억제
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

def get_ohlcv_from_db(conn, ticker_code):
    """
    DailyStockPrice 테이블에서 특정 종목의 전체 OHLCV 데이터를 조회합니다.
    """
    try:
        sql = "SELECT date, open_price, high_price, low_price, close_price, volume FROM DailyStockPrice WHERE stock_code = %s ORDER BY date ASC"
        df = pd.read_sql(sql, conn, params=(ticker_code,), index_col='date')
        df.index = pd.to_datetime(df.index)
        # DB에서 DECIMAL로 가져온 컬럼을 float으로 변환
        for col in ['open_price', 'high_price', 'low_price', 'close_price']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"    [오류] {ticker_code} OHLCV 데이터 로드 중: {e}")
        return pd.DataFrame()

def calculate_indicators(df_ohlcv):
    """
    주어진 OHLCV 데이터프레임으로 각종 기술적 지표를 계산합니다.
    """
    if df_ohlcv.empty:
        return pd.DataFrame()

    indicators = pd.DataFrame(index=df_ohlcv.index)

    # 이동평균
    indicators['ma_5'] = df_ohlcv['close_price'].rolling(window=5, min_periods=5).mean()
    indicators['ma_20'] = df_ohlcv['close_price'].rolling(window=20, min_periods=20).mean()

    # ATR (Average True Range)
    tr_df = pd.concat([
        df_ohlcv['high_price'] - df_ohlcv['low_price'],
        np.abs(df_ohlcv['high_price'] - df_ohlcv['close_price'].shift()),
        np.abs(df_ohlcv['low_price'] - df_ohlcv['close_price'].shift())
    ], axis=1)
    tr = tr_df.max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    indicators['atr_14_ratio'] = atr_14 / df_ohlcv['close_price']

    # 주가 위치 비율
    years_5 = 252 * 5
    years_10 = 252 * 10
    
    # 5년/10년 롤링 최고가/최저가
    low_5y = df_ohlcv['low_price'].rolling(window=years_5, min_periods=252).min()
    high_5y = df_ohlcv['high_price'].rolling(window=years_5, min_periods=252).max()
    low_10y = df_ohlcv['low_price'].rolling(window=years_10, min_periods=252).min()
    high_10y = df_ohlcv['high_price'].rolling(window=years_10, min_periods=252).max()

    # 위치 비율 계산 (분모가 0이 되는 경우 방지)
    indicators['price_vs_5y_low_pct'] = ((df_ohlcv['close_price'] - low_5y) / (high_5y - low_5y)).fillna(0)
    indicators['price_vs_10y_low_pct'] = ((df_ohlcv['close_price'] - low_10y) / (high_10y - low_10y)).fillna(0)
    
    # 0~1 사이 값으로 클리핑
    indicators['price_vs_5y_low_pct'] = indicators['price_vs_5y_low_pct'].clip(0, 1)
    indicators['price_vs_10y_low_pct'] = indicators['price_vs_10y_low_pct'].clip(0, 1)

    return indicators # NaN 값을 가진 행 제거

def get_last_indicator_date(conn, ticker_code):
    """
    CalculatedIndicators 테이블에서 특정 종목의 가장 최근 데이터 날짜를 조회합니다.
    """
    try:
        with conn.cursor() as cur:
            sql = "SELECT MAX(date) FROM CalculatedIndicators WHERE stock_code = %s"
            cur.execute(sql, (ticker_code,))
            result = cur.fetchone()
        if result and result[0]:
            return pd.to_datetime(result[0]).date()
        return None
    except Exception as e:
        print(f"    [오류] {ticker_code}의 마지막 지표 날짜 조회 중: {e}")
        return None

def save_indicators_to_db(conn, ticker_code, df_indicators):
    """
    계산된 지표 데이터프레임을 CalculatedIndicators 테이블에 저장합니다.
    """
    if df_indicators.empty:
        return


    last_saved_date = get_last_indicator_date(conn, ticker_code)

    if last_saved_date:
        df_to_save = df_indicators[df_indicators.index.date > last_saved_date]
    else:
        df_to_save = df_indicators

    if df_to_save.empty:
        # print(f"    - {ticker_code}: 신규 저장할 지표 데이터 없음.")
        return

    df_to_save = df_to_save.copy()
    # NaN 값을 None으로 변환
    df_to_save.replace({np.nan: None}, inplace=True)
    df_to_save.reset_index(inplace=True)
    df_to_save.rename(columns={'index': 'date'}, inplace=True)
    df_to_save['stock_code'] = ticker_code
    
    df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
    
    cols = ['stock_code', 'date', 'ma_5', 'ma_20', 'atr_14_ratio', 'price_vs_5y_low_pct', 'price_vs_10y_low_pct']
    df_to_save = df_to_save[cols]

    rows_to_insert = df_to_save.to_records(index=False).tolist()


    try:
        with conn.cursor() as cur:
            sql_insert = """
                INSERT IGNORE INTO CalculatedIndicators (stock_code, date, ma_5, ma_20, atr_14_ratio, price_vs_5y_low_pct, price_vs_10y_low_pct)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cur.executemany(sql_insert, rows_to_insert)
            saved_count = cur.rowcount
            conn.commit()
            if saved_count > 0:
                print(f"    - {ticker_code}: 지표 데이터 {saved_count} 건 DB 저장 완료.")
    except Exception as e:
        print(f"    [오류] {ticker_code} 지표 데이터 저장 중: {e}")
        conn.rollback()


def calculate_and_store_indicators_for_all(conn, use_gpu=False):
    """
    모든 종목에 대해 지표를 계산하고 저장하는 메인 함수
    """
    print("\n" + "="*50)
    print(f"STEP 4: 기술적/변동성 지표 계산 및 저장 시작 (Mode: {'GPU' if use_gpu else 'CPU'})")
    print("="*50)

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT stock_code FROM DailyStockPrice")
            all_tickers = [row[0] for row in cur.fetchall()]
        print(f"  총 {len(all_tickers)}개의 종목에 대해 지표 계산을 시작합니다.")
    except Exception as e:
        print(f"  [오류] DailyStockPrice에서 종목코드 로드 중: {e}")
        return

    total_start_time = time.time()
    for i, ticker_code in enumerate(all_tickers):
        print(f"\n  ({i+1}/{len(all_tickers)}) {ticker_code} 지표 계산 중...")
        
        df_ohlcv = get_ohlcv_from_db(conn, ticker_code)
        if df_ohlcv.empty:
            continue
        
        calc_start_time = time.time()
        if use_gpu:
            try:
                from .indicator_calculator_gpu import calculate_indicators_gpu
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "GPU 지표 계산을 사용하려면 `cudf`, `cupy`가 필요합니다. "
                    "rapids-env에서 실행하거나 config.yaml의 data_pipeline.flags.use_gpu를 false로 설정하세요."
                ) from e

            df_indicators = calculate_indicators_gpu(df_ohlcv)
        else:
            df_indicators = calculate_indicators(df_ohlcv)
        calc_end_time = time.time()
        print(f"    - {ticker_code}: 지표 계산 완료 (소요 시간: {calc_end_time - calc_start_time:.4f}초)")
            
        save_indicators_to_db(conn, ticker_code, df_indicators)
    
    total_end_time = time.time()
    print(f"\n--- 모든 종목의 지표 계산 및 저장 완료 (총 소요 시간: {total_end_time - total_start_time:.2f}초) ---")
