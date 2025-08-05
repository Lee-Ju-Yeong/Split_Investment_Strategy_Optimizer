"""
debug_gpu_single_run.py

This script is used to debug the GPU single run.
It is used to test the GPU single run with the parameters from the config.yaml file.



"""
import time
import cudf
import cupy as cp
import pandas as pd
from sqlalchemy import create_engine
from datetime import timedelta # timedelta 임포트 추가
# --- 필요한 모듈 추가 임포트 ---
from src.config_loader import load_config
from src.backtest_strategy_gpu import run_magic_split_strategy_on_gpu

# -----------------------------------------------------------------------------
# 1. Configuration and Parameter Setup
# -----------------------------------------------------------------------------

import urllib.parse

# --- 설정 파일 로드 (YAML 로더로 통일) ---
config = load_config()
db_config = config['database']
backtest_settings = config['backtest_settings']
execution_params = config['execution_params']

# URL 인코딩을 포함하여 DB 연결 문자열 생성
db_pass_encoded = urllib.parse.quote_plus(db_config['password'])
db_connection_str = (
    f"mysql+pymysql://{db_config['user']}:{db_pass_encoded}"
    f"@{db_config['host']}/{db_config['database']}"
)

# --- Debug를 위한 단일 파라미터 조합 정의 ---
# CPU 단일 테스트(config.yaml)와 동일한 값으로 설정
cpu_test_params = config['strategy_params']

# 변경 후 (config에서 읽어온 값을 사용):
max_stocks_options = cp.array([cpu_test_params['max_stocks']], dtype=cp.int32)
order_investment_ratio_options = cp.array([cpu_test_params['order_investment_ratio']], dtype=cp.float32)
additional_buy_drop_rate_options = cp.array([cpu_test_params['additional_buy_drop_rate']], dtype=cp.float32)
sell_profit_rate_options = cp.array([cpu_test_params['sell_profit_rate']], dtype=cp.float32)

# additional_buy_priority는 문자열이므로 숫자로 변환
priority_map = {'lowest_order': 0, 'highest_drop': 1}
priority_val = priority_map.get(cpu_test_params['additional_buy_priority'], 0)
additional_buy_priority_options = cp.array([priority_val], dtype=cp.int32)

# Create all combinations using CuPy's broadcasting capabilities
grid = cp.meshgrid(
    max_stocks_options,
    order_investment_ratio_options,
    additional_buy_drop_rate_options,
    sell_profit_rate_options,
    additional_buy_priority_options
)
# Flatten the grid to get a list of all combinations
param_combinations = cp.vstack([item.flatten() for item in grid]).T
num_combinations = param_combinations.shape[0]

print("✅ [DEBUG MODE] Single parameter combination for GPU test:")
print(param_combinations)
print(f"✅ Total parameter combinations generated for GPU: {num_combinations}")


# -----------------------------------------------------------------------------
# 2. GPU Data Pre-loader
# -----------------------------------------------------------------------------

def preload_all_data_to_gpu(engine, start_date, end_date):
    """
    Loads all necessary stock data for the entire backtest period into a
    single cuDF DataFrame, minimizing I/O during simulation.

    Returns:
        cudf.DataFrame: A DataFrame containing all OHLCV and indicator data,
                        indexed by (ticker, date).
    """
    print("⏳ Loading all stock data into GPU memory...")
    start_time = time.time()
    
    query = f"""
    SELECT 
        stock_code AS ticker, 
        date, 
        open_price, 
        high_price, 
        low_price, 
        close_price, 
        volume,
        atr_14_ratio
    FROM DailyStockPrice
    JOIN CalculatedIndicators USING (stock_code, date)
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """
    
    # Use cuDF to read directly from the database into a GPU DataFrame
    # Note: This requires a compatible database connector and setup.
    # For now, we load via pandas as an intermediate step.
    
    # Create a SQLAlchemy engine
    sql_engine = create_engine(engine)
    
    # Load data using pandas first
    df_pd = pd.read_sql(query, sql_engine, parse_dates=['date'])
    
    # Move the DataFrame to the GPU
    gdf = cudf.from_pandas(df_pd)
    
    # Set a multi-index for efficient lookups
    gdf = gdf.set_index(['ticker', 'date'])
    
    end_time = time.time()
    print(f"✅ Data loaded to GPU. Shape: {gdf.shape}. Time: {end_time - start_time:.2f}s")
    
    return gdf

def preload_weekly_filtered_stocks_to_gpu(engine, start_date, end_date):
    """
    Loads all weekly filtered stock codes for the backtest period into a
    cuDF DataFrame.
    """
    print("⏳ Loading weekly filtered stocks data to GPU memory...")
    start_time = time.time()
    
    # ★★★ 핵심 수정 ★★★
    # 백테스팅 시작일보다 넉넉하게 2주 전 데이터부터 로드하여,
    # 연초에 이전 년도 데이터를 참조할 수 있도록 함
    extended_start_date = pd.to_datetime(start_date) - timedelta(days=14)
    extended_start_date_str = extended_start_date.strftime('%Y-%m-%d')
    
    # WeeklyFilteredStocks 테이블에서 해당 기간의 모든 데이터를 가져옴
    query =  f"""
    SELECT `filter_date` as date, `stock_code` as ticker
    FROM WeeklyFilteredStocks
    WHERE `filter_date` BETWEEN '{extended_start_date_str}' AND '{end_date}'
    """
    sql_engine = create_engine(engine)
    df_pd = pd.read_sql(query, sql_engine, parse_dates=['date'])
    
    # cuDF로 변환
    gdf = cudf.from_pandas(df_pd)
    gdf = gdf.set_index('date')
    
    end_time = time.time()
    print(f"✅ Weekly filtered stocks loaded to GPU. Shape: {gdf.shape}. Time: {end_time - start_time:.2f}s")
    return gdf

# -----------------------------------------------------------------------------
# 3. GPU Backtesting Kernel (to be implemented)
# -----------------------------------------------------------------------------

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
    
    # GPU 백테스팅 함수 직접 호출
    daily_portfolio_values = run_magic_split_strategy_on_gpu(
        initial_cash=initial_cash_value, # <<< 수정된 인자 이름 및 외부에서 받은 값 사용
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
    
    # Calculate final results for each parameter combination
    final_values = daily_portfolio_values[:, -1]  # Last day values
    initial_values = cp.full(len(params_gpu), initial_cash_value, dtype=cp.float32)
    
    # Calculate total returns
    total_returns = (final_values / initial_values) - 1
    
    return total_returns, daily_portfolio_values


# -----------------------------------------------------------------------------
# 4. Main Execution Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    backtest_start_date = backtest_settings['start_date']
    backtest_end_date = backtest_settings['end_date']
    initial_cash = backtest_settings['initial_cash'] # <<< config에서 초기 자본 가져오기
    
    print(f"📅 테스트 기간: {backtest_start_date} ~ {backtest_end_date}")
    
    # 1. Pre-load all data to GPU
    all_data_gpu = preload_all_data_to_gpu(db_connection_str, backtest_start_date, backtest_end_date)
    # 💡 새로운 데이터 로더 호출: 주간 필터링된 종목 데이터 로드
    weekly_filtered_gpu = preload_weekly_filtered_stocks_to_gpu(db_connection_str, backtest_start_date, backtest_end_date)
    
    
    # --- 💡 수정된 부분 시작 💡 ---

    # 2. Generate trading dates and convert them to integer indices for GPU
    # Pandas의 bdate_range를 사용하여 실제 거래일만 가져옴
    trading_dates_pd = pd.bdate_range(start=backtest_start_date, end=backtest_end_date)
    
    # GPU 커널에서는 0, 1, 2... 와 같은 정수 인덱스로 날짜를 순회
    trading_date_indices_gpu = cp.arange(len(trading_dates_pd), dtype=cp.int32)

    # 3. Filter the main GPU DataFrame to include only actual trading dates
    #    This ensures the GPU data aligns with our trading date indices.
    #    cuDF는 datetime 객체를 인덱스로 직접 사용할 수 있음
    all_data_gpu = all_data_gpu[all_data_gpu.index.get_level_values('date').isin(trading_dates_pd)]
    
    # --- 💡 수정된 부분 끝 💡 ---

    # 4. Get all tickers from the loaded data
    #    (all_data_gpu가 필터링되었으므로, 여기서 티커를 가져오는 것이 정확함)
    all_tickers = all_data_gpu.index.get_level_values('ticker').unique().to_pandas().tolist()
    print(f"📊 로드된 종목 수: {len(all_tickers)}")
    print(f"📊 실제 거래일 수: {len(trading_date_indices_gpu)}")
    
    # 5. Run the backtesting kernel
    print(f"\n🚀 {num_combinations}개 파라미터 조합으로 GPU 백테스팅 시작...")
    start_time = time.time()
    
    # 함수 이름 변경 및 initial_cash 전달
    total_returns, daily_values = run_gpu_optimization(
        param_combinations, 
        all_data_gpu, 
        weekly_filtered_gpu,
        all_tickers, 
        trading_date_indices_gpu,
        trading_dates_pd,
        initial_cash,
        execution_params,
        debug_mode=True,
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 5. Results summary
    print(f"\n--- 🎉 GPU 백테스팅 완료 ---")
    print(f"⏱️  총 소요 시간: {elapsed_time:.2f}초")
    print(f"📈 조합 당 평균 시간: {elapsed_time/num_combinations*1000:.2f}ms")
    print(f"🔥 CPU 대비 예상 가속도: {8 * elapsed_time / (num_combinations * 0.1):.1f}x")
    
    # 6. Top performing parameters
    returns_cpu = total_returns.get()  # Move to CPU for analysis
    best_idx = cp.argmax(total_returns).get()
    worst_idx = cp.argmin(total_returns).get()
    
    print(f"\n📊 성과 요약:")
    print(f"   최고 수익률: {returns_cpu[best_idx]*100:.2f}%")
    print(f"   최저 수익률: {returns_cpu[worst_idx]*100:.2f}%")
    print(f"   평균 수익률: {cp.mean(total_returns).get()*100:.2f}%")
    
    best_params = param_combinations[best_idx].get()
    print(f"\n🏆 최고 성과 파라미터 조합:")
    print(f"   Max Stocks: {best_params[0]}")
    print(f"   Order Investment Ratio: {best_params[1]:.3f}")
    print(f"   Additional Buy Drop Rate: {best_params[2]:.3f}")
    print(f"   Sell Profit Rate: {best_params[3]:.3f}")
    print(f"   Additional Buy Priority: {best_params[4]}")
    
    print(f"\n✅ GPU 백테스팅 시스템 테스트 완료!")
