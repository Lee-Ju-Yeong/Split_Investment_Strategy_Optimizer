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

def preload_tier_data_to_tensor(engine, start_date, end_date, all_tickers, trading_dates_pd):
    """
    Loads DailyStockTier data and converts it to a dense (num_days, num_tickers) int8 tensor.
    Performs forward-fill to ensure PIT compliance (latest <= date).
    """
    print("⏳ Loading DailyStockTier data to GPU tensor...")
    start_time = time.time()
    
    start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    query = f"""
        SELECT date, stock_code as ticker, tier
        FROM DailyStockTier
        WHERE date BETWEEN '{start_date_str}' AND '{end_date_str}'
        UNION ALL
        SELECT t.date, t.stock_code as ticker, t.tier
        FROM DailyStockTier t
        JOIN (
            SELECT stock_code, MAX(date) AS max_date
            FROM DailyStockTier
            WHERE date < '{start_date_str}'
            GROUP BY stock_code
        ) latest ON t.stock_code = latest.stock_code AND t.date = latest.max_date
    """
    sql_engine = create_engine(engine)
    df_pd = pd.read_sql(query, sql_engine, parse_dates=['date'])
    
    if df_pd.empty:
        print("⚠️ No Tier data found. Returning empty tensor.")
        return cp.zeros((len(trading_dates_pd), len(all_tickers)), dtype=cp.int8)

    # Pivot to (date, ticker) -> tier
    # index=date, columns=ticker, values=tier
    df_wide = df_pd.pivot_table(index='date', columns='ticker', values='tier')
    
    # Reindex to full trading dates and all tickers
    # Forward fill to propagate the latest tier
    df_reindexed = df_wide.reindex(index=trading_dates_pd, columns=all_tickers).ffill().fillna(0).astype(int)
    
    # Convert to CuPy
    tier_tensor = cp.asarray(df_reindexed.values, dtype=cp.int8)
    
    print(f"✅ Tier data loaded and tensorized. Shape: {tier_tensor.shape}. Time: {time.time() - start_time:.2f}s")
    return tier_tensor

# -----------------------------------------------------------------------------
# 3. GPU Backtesting Kernel
# -----------------------------------------------------------------------------

def run_gpu_backtest_kernel(params_gpu, data_gpu,
                         weekly_filtered_gpu, all_tickers,
                         trading_date_indices_gpu,
                         trading_dates_pd,
                         initial_cash_value,
                         exec_params: dict,
                         debug_mode: bool = False,
                         tier_tensor: cp.ndarray = None # [Issue #67]
                         ):
    """
    GPU-accelerated 백테스팅 커널을 직접 실행합니다.
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
        tier_tensor=tier_tensor
    )
    
    print("🎉 GPU backtesting kernel finished.")
    
    return daily_portfolio_values

# 4. [신규] 워커 함수: run_single_backtest
def run_single_backtest(start_date: str, end_date: str, params_dict: dict, initial_cash: float, debug_mode: bool = False):
    """
    주어진 기간과 파라미터로 단일 GPU 백테스트를 수행하고 결과를 반환합니다.
    WFO 오케스트레이터에 의해 호출되는 '워커' 함수입니다.
    """
    print(f"\n" + "="*80)
    print(f"WORKER: Running Single Backtest for {start_date} to {end_date}")
    print(f"Params: {params_dict}")
    print("="*80)

    # 1. 파라미터 딕셔너리를 GPU가 사용할 수 있는 cp.ndarray로 변환
    priority_map = {'lowest_order': 0, 'highest_drop': 1}
    priority_val = priority_map.get(params_dict.get('additional_buy_priority', 'lowest_order'), 0)
    
    param_combinations = cp.array([[
        params_dict['max_stocks'],
        params_dict['order_investment_ratio'],
        params_dict['additional_buy_drop_rate'],
        params_dict['sell_profit_rate'],
        priority_val,
        params_dict['stop_loss_rate'],
        params_dict['max_splits_limit'],
        params_dict['max_inactivity_period'],
    ]], dtype=cp.float32)

    # 2. 데이터 로드 및 준비
    all_data_gpu = preload_all_data_to_gpu(db_connection_str, start_date, end_date)
    weekly_filtered_gpu = preload_weekly_filtered_stocks_to_gpu(db_connection_str, start_date, end_date)
    
    sql_engine = create_engine(db_connection_str)
    trading_dates_query = f"""
        SELECT DISTINCT date 
        FROM DailyStockPrice 
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    trading_dates_pd_df = pd.read_sql(trading_dates_query, sql_engine, parse_dates=['date'], index_col='date')
    trading_dates_pd = trading_dates_pd_df.index # 이제 DatetimeIndex 객체
    trading_date_indices_gpu = cp.arange(len(trading_dates_pd), dtype=cp.int32)
    
    all_data_gpu = all_data_gpu[all_data_gpu.index.get_level_values('date').isin(trading_dates_pd)]
    all_tickers = sorted(all_data_gpu.index.get_level_values('ticker').unique().to_pandas().tolist())
    print(f"  - Tickers for period: {len(all_tickers)}")
    print(f"  - Trading days for period: {len(trading_dates_pd)}")

    # [Issue #67] Preload Tier Tensor
    tier_tensor = preload_tier_data_to_tensor(db_connection_str, start_date, end_date, all_tickers, trading_dates_pd)

    # 3. 백테스팅 커널 실행
    start_time_kernel = time.time()
    
    # exec_params에 모드 정보 추가
    execution_params = execution_params.copy()
    execution_params['candidate_source_mode'] = params_dict.get('candidate_source_mode', 'weekly')
    execution_params['use_weekly_alpha_gate'] = params_dict.get('use_weekly_alpha_gate', False)
    
    daily_values_result = run_gpu_backtest_kernel(
        param_combinations, 
        all_data_gpu, 
        weekly_filtered_gpu,
        all_tickers, 
        trading_date_indices_gpu,
        trading_dates_pd,
        initial_cash,
        execution_params,
        debug_mode=debug_mode,
        tier_tensor=tier_tensor
    )
    end_time_kernel = time.time()
    print(f"  - GPU Kernel Execution Time: {end_time_kernel - start_time_kernel:.2f}s")
    
    # 4. 결과 처리 및 반환
    if daily_values_result is None or daily_values_result.shape[0] == 0:
        print("  - [Warning] Backtest returned no data. Returning empty series.")
        return pd.Series(dtype=float)

    daily_values_cpu = daily_values_result.get()[0] # 첫 번째 (유일한) 시뮬레이션 결과
    equity_curve_series = pd.Series(daily_values_cpu, index=trading_dates_pd)
    
    return equity_curve_series
# -----------------------------------------------------------------------------
# 5. Main Execution Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 이 파일이 단독으로 실행될 때, config.yaml의 설정으로 CPU-GPU 비교 검증을 수행합니다.
    backtest_start_date = backtest_settings['start_date']
    backtest_end_date = backtest_settings['end_date']
    initial_cash = backtest_settings['initial_cash']
    
    print(f"📅 Running Standalone GPU Debug/Verification Run")
    print(f"📅 Period: {backtest_start_date} ~ {backtest_end_date}")
    # config.yaml에서 직접 파라미터 로드
    params_for_debug = config['strategy_params']
    
    # 리팩토링된 워커 함수 호출
    equity_curve = run_single_backtest(
        start_date=backtest_start_date,
        end_date=backtest_end_date,
        params_dict=params_for_debug,
        initial_cash=initial_cash,
        debug_mode=True  # 단독 실행 시에는 항상 상세 로그 출력
    )
    
    # 기존과 동일한 성과 분석 및 출력 로직
    print("\n" + "="*60)
    print("📈 GPU Standalone Run - Performance Summary")
    print("="*60)
    

    if not equity_curve.empty:
        history_df = pd.DataFrame(equity_curve, columns=['total_value'])
        analyzer = PerformanceAnalyzer(history_df)
        metrics = analyzer.get_metrics(formatted=True)

        for key, value in metrics.items():
            print(f"{key:<25}: {value}")
    else:
        print("Error: No backtesting result data to analyze.")

    print("="*60)
    print(f"\n✅ GPU standalone run and analysis complete!")
