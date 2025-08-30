# debug_gpu_single_run.py (수정된 최종본)

"""
This script is used to debug the GPU single run.
It is used to test the GPU single run with the parameters from the config.yaml file.
"""
import time
import cudf
import cupy as cp
import pandas as pd
from sqlalchemy import create_engine
from datetime import timedelta, datetime # datetime 추가
import os
import urllib.parse

# --- 필요한 모듈 추가 임포트 ---
from src.config_loader import load_config
from src.backtest_strategy_gpu import run_magic_split_strategy_on_gpu
### 이슈 #3 동기화를 위한 모듈 임포트 ###
from src.performance_analyzer import PerformanceAnalyzer

# -----------------------------------------------------------------------------
# 1. Configuration and Parameter Setup
# -----------------------------------------------------------------------------

# --- 설정 파일 로드 (YAML 로더로 통일) ---
config = load_config()
db_config = config['database']
backtest_settings = config['backtest_settings']
strategy_params = config['strategy_params']
execution_params = config['execution_params']

# GPU 커널에 쿨다운 기간 전달을 위해 execution_params에 추가
execution_params['cooldown_period_days'] = strategy_params.get('cooldown_period_days', 5)

# URL 인코딩을 포함하여 DB 연결 문자열 생성
db_pass_encoded = urllib.parse.quote_plus(db_config['password'])
db_connection_str = (
    f"mysql+pymysql://{db_config['user']}:{db_pass_encoded}"
    f"@{db_config['host']}/{db_config['database']}"
)

# --- Debug를 위한 단일 파라미터 조합 정의 ---
cpu_test_params = config['strategy_params']
max_stocks_options = cp.array([cpu_test_params['max_stocks']], dtype=cp.int32)
order_investment_ratio_options = cp.array([cpu_test_params['order_investment_ratio']], dtype=cp.float32)
additional_buy_drop_rate_options = cp.array([cpu_test_params['additional_buy_drop_rate']], dtype=cp.float32)
sell_profit_rate_options = cp.array([cpu_test_params['sell_profit_rate']], dtype=cp.float32)

priority_map = {'lowest_order': 0, 'highest_drop': 1}
priority_val = priority_map.get(cpu_test_params['additional_buy_priority'], 0)
additional_buy_priority_options = cp.array([priority_val], dtype=cp.int32)

# --- [New] Load Advanced Risk Management Parameters ---
stop_loss_rate_options = cp.array([cpu_test_params.get('stop_loss_rate', -0.15)], dtype=cp.float32)
max_splits_limit_options = cp.array([cpu_test_params.get('max_splits_limit', 10)], dtype=cp.int32)
max_inactivity_period_options = cp.array([cpu_test_params.get('max_inactivity_period', 90)], dtype=cp.int32)


grid = cp.meshgrid(
    max_stocks_options,
    order_investment_ratio_options,
    additional_buy_drop_rate_options,
    sell_profit_rate_options,
    additional_buy_priority_options,
    stop_loss_rate_options,          
    max_splits_limit_options,        
    max_inactivity_period_options       
)
param_combinations = cp.vstack([item.flatten() for item in grid]).T
num_combinations = param_combinations.shape[0]

print("✅ [DEBUG MODE] Single parameter combination for GPU test:")
print(param_combinations.get())
print(f"✅ Total parameter combinations generated for GPU: {num_combinations}")


# -----------------------------------------------------------------------------
# 2. GPU Data Pre-loader
# -----------------------------------------------------------------------------

def preload_all_data_to_gpu(engine, start_date, end_date):
    """
    Loads all necessary stock data for the entire backtest period into a
    single cuDF DataFrame.
    """
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
        ci.atr_14_ratio
    FROM 
        DailyStockPrice AS dsp
    LEFT JOIN 
        CalculatedIndicators AS ci ON dsp.stock_code = ci.stock_code AND dsp.date = ci.date
    WHERE 
        dsp.date BETWEEN '{start_date}' AND '{end_date}'
    """
    sql_engine = create_engine(engine)
    df_pd = pd.read_sql(query, sql_engine, parse_dates=['date'])
    gdf = cudf.from_pandas(df_pd)
    gdf = gdf.set_index(['ticker', 'date'])
    
    end_time = time.time()
    print(f"✅ Data loaded to GPU. Shape: {gdf.shape}. Time: {end_time - start_time:.2f}s")
    return gdf

def preload_weekly_filtered_stocks_to_gpu(engine, start_date, end_date):
    """
    Loads all weekly filtered stock codes for the backtest period.
    """
    print("⏳ Loading weekly filtered stocks data to GPU memory...")
    start_time = time.time()
    
    extended_start_date = pd.to_datetime(start_date) - timedelta(days=14)
    extended_start_date_str = extended_start_date.strftime('%Y-%m-%d')
    
    query =  f"""
    SELECT `filter_date` as date, `stock_code` as ticker
    FROM WeeklyFilteredStocks
    WHERE `filter_date` BETWEEN '{extended_start_date_str}' AND '{end_date}'
    """
    sql_engine = create_engine(engine)
    df_pd = pd.read_sql(query, sql_engine, parse_dates=['date'])
    gdf = cudf.from_pandas(df_pd)
    gdf = gdf.set_index('date')
    
    end_time = time.time()
    print(f"✅ Weekly filtered stocks loaded to GPU. Shape: {gdf.shape}. Time: {end_time - start_time:.2f}s")
    return gdf

# -----------------------------------------------------------------------------
# 3. GPU Backtesting Kernel
# -----------------------------------------------------------------------------

### ### 로직 동기화: 이제 daily_portfolio_values 만 반환합니다. ### ###
def run_gpu_optimization(params_gpu, data_gpu,
                         weekly_filtered_gpu, all_tickers,
                         trading_date_indices_gpu,
                         trading_dates_pd,
                         initial_cash_value,
                         exec_params: dict,
                         debug_mode: bool = False,
                         ):
    """
    GPU-accelerated backtesting을 오케스트레이션합니다.
    """
    print("🚀 Starting GPU backtesting kernel...")
    
    daily_portfolio_values = run_magic_split_strategy_on_gpu(
        initial_cash=initial_cash_value,
        param_combinations=params_gpu,
        all_data_gpu=data_gpu,
        weekly_filtered_gpu=weekly_filtered_gpu,
        trading_date_indices=trading_date_indices_gpu,
        trading_dates_pd_cpu=trading_dates_pd,
        all_tickers=all_tickers,
        max_splits_limit=20,
        execution_params=exec_params,
        debug_mode=debug_mode,
    )
    
    print("🎉 GPU backtesting kernel finished.")
    
    return daily_portfolio_values


# -----------------------------------------------------------------------------
# 4. Main Execution Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    backtest_start_date = backtest_settings['start_date']
    backtest_end_date = backtest_settings['end_date']
    initial_cash = backtest_settings['initial_cash']
    
    print(f"📅 테스트 기간: {backtest_start_date} ~ {backtest_end_date}")
    
    # 1. 데이터 로드 및 준비
    all_data_gpu = preload_all_data_to_gpu(db_connection_str, backtest_start_date, backtest_end_date)
    weekly_filtered_gpu = preload_weekly_filtered_stocks_to_gpu(db_connection_str, backtest_start_date, backtest_end_date)
    
    # [수정] CPU와 동일하게 DB에서 실제 거래일만 가져오도록 변경합니다.
    print("Fetching actual trading dates from DB...")
    sql_engine = create_engine(db_connection_str)
    trading_dates_query = f"""
        SELECT DISTINCT date 
        FROM DailyStockPrice 
        WHERE date BETWEEN '{backtest_start_date}' AND '{backtest_end_date}'
        ORDER BY date
    """
    trading_dates_pd = pd.read_sql(trading_dates_query, sql_engine, parse_dates=['date'])['date']
    trading_date_indices_gpu = cp.arange(len(trading_dates_pd), dtype=cp.int32)
    all_data_gpu = all_data_gpu[all_data_gpu.index.get_level_values('date').isin(trading_dates_pd)]
    all_tickers = all_data_gpu.index.get_level_values('ticker').unique().to_pandas().tolist()
    # [추가] <<<<<<< 단일화된 시스템 사전 검증 블록 >>>>>>>
    print("\n" + "="*50)
    print("🔬 GPU KERNEL PRE-FLIGHT CHECK")
    print("="*50)
    try:
        # 1. Ticker-Index 매핑 순서의 비결정성(Non-determinism) 검증
        print("\n[1] Ticker-Index Mapping Order Verification")
        print("  - Purpose: Check if the order of `all_tickers` is consistent.")
        print("  - Method: Displaying first 5 and last 5 tickers.")
        print("\n  [First 5 Tickers in list]")
        for i in range(min(5, len(all_tickers))):
            print(f"    Index {i:<3} -> {all_tickers[i]}")
        print("\n  [Last 5 Tickers in list]")
        if len(all_tickers) > 5:
            for i in range(len(all_tickers) - 5, len(all_tickers)):
                print(f"    Index {i:<3} -> {all_tickers[i]}")

        # 2. 핵심 종목 인덱스 추적
        print("\n[2] Key Ticker Index Tracking")
        print("  - Purpose: Track the indices of specific tickers involved in debugging.")
        tickers_to_watch = ['020000', '192440', '014570', '045060', '006650', '043370']
        ticker_to_idx_map = {ticker: i for i, ticker in enumerate(all_tickers)}
        
        for ticker in tickers_to_watch:
            print(f"    - Ticker {ticker} -> Index: {ticker_to_idx_map.get(ticker, 'Not Found')}")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during pre-flight check: {e}")
    print("="*50 + "\n")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # [핵심 수정] Ticker-Index 매핑의 일관성을 보장하기 위해 리스트를 정렬합니다.
    all_tickers = sorted(all_tickers)
    print("✅ Ticker list has been sorted to ensure deterministic mapping.")

    print(f"📊 로드된 종목 수: {len(all_tickers)}")
    
    # 2. 백테스팅 커널 실행
    print(f"\n🚀 {num_combinations}개 파라미터 조합으로 GPU 백테스팅 시작...")
    start_time = time.time()
    
    ### ### 로직 동기화: 이제 반환값은 하나입니다. ### ###
    daily_values_result = run_gpu_optimization(
        param_combinations, 
        all_data_gpu, 
        weekly_filtered_gpu,
        all_tickers, 
        trading_date_indices_gpu,
        trading_dates_pd,
        initial_cash,
        execution_params,
        debug_mode=True, # 디버깅 스크립트이므로 항상 True
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n--- 🎉 GPU 백테스팅 완료 ---")
    print(f"⏱️  총 소요 시간: {elapsed_time:.2f}초")
    
    ### ### 로직 동기화: 기존 결과 요약 로직을 삭제하고, 상세 분석 로직으로 대체 ### ###
    
    # 3. 상세 성과 분석
    print("\n" + "="*60)
    print("📈 GPU 백테스팅 성과 요약 (단일 실행)")
    print("="*60)

    # GPU에서 CPU로 데이터 이동 (결과는 (1, num_days) 형태)
    daily_values_cpu = daily_values_result.get()
    
    # 첫 번째 (그리고 유일한) 시뮬레이션 결과에 대해 분석
    if daily_values_cpu.shape[0] > 0:
        daily_series = pd.Series(daily_values_cpu[0], index=trading_dates_pd)
        
        # PerformanceAnalyzer가 요구하는 DataFrame 형식으로 변환
        history_df = pd.DataFrame(daily_series, columns=['total_value'])

        # PerformanceAnalyzer를 사용하여 지표 계산
        analyzer = PerformanceAnalyzer(history_df)
        # CPU 백테스터(main_backtest)와 동일한 포맷으로 출력
        metrics = analyzer.get_metrics(formatted=True)

        # 결과 출력
        for key, value in metrics.items():
            print(f"{key:<25}: {value}")
    else:
        print("오류: 분석할 백테스팅 결과 데이터가 없습니다.")

    print("="*60)
    print(f"\n✅ GPU 디버깅 및 분석 완료!")