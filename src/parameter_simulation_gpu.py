"""
parameter_simulation_gpu.py

GPU-accelerated parameter simulation for the MagicSplitStrategy.
This script orchestrates the backtesting of thousands of parameter combinations
by leveraging CuPy and CuDF for massive parallelization on the GPU.
"""

import time
import cudf
import cupy as cp
import pandas as pd
from sqlalchemy import create_engine
from datetime import timedelta, datetime
import os
import urllib.parse
from src.config_loader import load_config
from src.backtest_strategy_gpu import run_magic_split_strategy_on_gpu
from src.performance_analyzer import PerformanceAnalyzer
import numpy as np


# 1. Configuration and Parameter Setup
config = load_config()
execution_params = config["execution_params"]
db_config = config["database"]
backtest_settings = config["backtest_settings"]
strategy_params = config["strategy_params"]
execution_params['cooldown_period_days'] = strategy_params.get('cooldown_period_days', 5)
strategy_params = config["strategy_params"]
db_pass_encoded = urllib.parse.quote_plus(db_config["password"])
db_connection_str = f"mysql+pymysql://{db_config['user']}:{db_pass_encoded}@{db_config['host']}/{db_config['database']}"

# [추가] config.yaml의 parameter_space를 읽어 동적으로 파라미터 조합 생성
param_space_config = config['parameter_space']
param_order = [ # meshgrid 순서를 보장하기 위한 리스트
    'max_stocks', 'order_investment_ratio', 'additional_buy_drop_rate', 
    'sell_profit_rate', 'additional_buy_priority', 'stop_loss_rate', 
    'max_splits_limit', 'max_inactivity_period'
]
param_options_list = []
for key in param_order:
    spec = param_space_config[key]
    dtype = np.int32 if key in ['max_stocks', 'additional_buy_priority', 'max_splits_limit', 'max_inactivity_period'] else np.float32
    
    if spec['type'] == 'linspace':
        options = np.linspace(spec['start'], spec['stop'], spec['num'], dtype=dtype)
    elif spec['type'] == 'list':
        options = np.array(spec['values'], dtype=dtype)
    elif spec['type'] == 'range':
        options = np.arange(spec['start'], spec['stop'], spec['step'], dtype=dtype)
    else:
        raise ValueError(f"Unsupported parameter type '{spec['type']}' for '{key}'")
    param_options_list.append(cp.asarray(options))

grid = cp.meshgrid(*param_options_list) # 리스트를 언패킹하여 전달
param_combinations = cp.vstack([item.flatten() for item in grid]).T
num_combinations = param_combinations.shape[0]

print(f"✅ Dynamically generated {num_combinations} parameter combinations from config.yaml.")
# [추가] 변경사항 확인을 위한 검증용 print문
print(f"✅ [VERIFICATION] Newly compiled code is running. Num combinations: {num_combinations}")


# 2. GPU Data Pre-loader
def preload_all_data_to_gpu(engine, start_date, end_date):
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
    gdf = cudf.from_pandas(df_pd).set_index(['ticker', 'date'])
    print(f"✅ Data loaded to GPU. Shape: {gdf.shape}. Time: {time.time() - start_time:.2f}s")
    return gdf

def preload_weekly_filtered_stocks_to_gpu(engine, start_date, end_date):
    print("⏳ Loading weekly filtered stocks data to GPU memory...")
    start_time = time.time()
    extended_start_date = pd.to_datetime(start_date) - timedelta(days=14)
    query = f"SELECT `filter_date` as date, `stock_code` as ticker FROM WeeklyFilteredStocks WHERE `filter_date` BETWEEN '{extended_start_date.strftime('%Y-%m-%d')}' AND '{end_date}'"
    sql_engine = create_engine(engine)
    df_pd = pd.read_sql(query, sql_engine, parse_dates=['date'])
    gdf = cudf.from_pandas(df_pd).set_index('date')
    print(f"✅ Weekly filtered stocks loaded to GPU. Shape: {gdf.shape}. Time: {time.time() - start_time:.2f}s")
    return gdf

# 3. GPU Backtesting Kernel Orchestrator
def run_gpu_optimization(params_gpu, data_gpu, weekly_filtered_gpu, all_tickers, trading_date_indices_gpu, trading_dates_pd, initial_cash_value, exec_params):
    print("🚀 Starting GPU backtesting kernel...")
    max_splits_from_params = int(cp.max(params_gpu[:, 6]).get())
    daily_portfolio_values = run_magic_split_strategy_on_gpu(
        initial_cash=initial_cash_value,
        param_combinations=params_gpu,
        all_data_gpu=data_gpu,
        weekly_filtered_gpu=weekly_filtered_gpu,
        trading_date_indices=trading_date_indices_gpu,
        trading_dates_pd_cpu=trading_dates_pd,
        all_tickers=all_tickers,
        execution_params=exec_params,
        max_splits_limit=max_splits_from_params
    )
    print("🎉 GPU backtesting kernel finished.")
    return daily_portfolio_values

# 4. Analysis and Result Saving
def analyze_and_save_results(param_combinations_gpu, daily_values_gpu, trading_dates_pd, save_to_file=True):
    print("\n--- 🔬 Analyzing detailed performance metrics ---")
    start_time = time.time()
    param_combinations_cpu = param_combinations_gpu.get()
    daily_values_cpu = daily_values_gpu.get()
    
    results_list = []
    for i in range(daily_values_cpu.shape[0]):
        history_df_mock = pd.DataFrame(pd.Series(daily_values_cpu[i], index=trading_dates_pd), columns=['total_value'])
        analyzer = PerformanceAnalyzer(history_df_mock)
        results_list.append(analyzer.get_metrics(formatted=False))

    param_names = [
        'max_stocks', 'order_investment_ratio', 'additional_buy_drop_rate', 'sell_profit_rate', 
        'additional_buy_priority', 'stop_loss_rate', 'max_splits_limit', 'max_inactivity_period' # [수정] 변수명 동기화
    ]
    params_df = pd.DataFrame(param_combinations_cpu, columns=param_names)
    metrics_df = pd.DataFrame(results_list)
    full_results_df = pd.concat([params_df, metrics_df], axis=1)

    full_results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sorted_df = full_results_df.sort_values(by='calmar_ratio', ascending=False).dropna(subset=['calmar_ratio'])
    
    # [추가] 최적 파라미터를 딕셔너리로 반환하는 기능 추가
    # [수정] 엣지 케이스: 유효한 결과가 없을 경우 IndexError 방지
    if not sorted_df.empty:
        best_params_series = sorted_df.iloc[0]
        best_params_dict = best_params_series.to_dict()
    else:
        best_params_dict = {} # 빈 딕셔너리 반환

    print("\n🏆 Top 10 Performing Parameter Combinations (by Calmar Ratio):")
    # [수정] 터미널 출력에 max_inactivity_period 포함
    display_columns = [
        'calmar_ratio', 'cagr', 'mdd', 'sharpe_ratio', 'stop_loss_rate', 
        'max_splits_limit', 'max_inactivity_period', 'sell_profit_rate', 'additional_buy_drop_rate'
    ]
    display_df = sorted_df.head(10).get(display_columns, pd.DataFrame())
    if not display_df.empty:
        for col in ['cagr', 'mdd', 'annualized_volatility', 'stop_loss_rate']:
            if col in display_df.columns: display_df[col] = display_df[col].map('{:.2%}'.format)
        for col in ['calmar_ratio', 'sharpe_ratio', 'sortino_ratio']:
            if col in display_df.columns: display_df[col] = display_df[col].map('{:.2f}'.format)
        print(display_df.to_string(index=False))


    #  파일 저장 로직을 조건부로 실행
    if save_to_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'gpu_simulation_results_{timestamp}.csv')
        sorted_df.to_csv(filepath, index=False, float_format='%.4f')
        print(f"\n✅ Full analysis saved to: {filepath}")
    print(f"⏱️  Analysis took: {time.time() - start_time:.2f} seconds.") 
    
    return best_params_dict, sorted_df # [추가] 전체 결과 DF도 반환
# 5. [신규] 워커 함수: find_optimal_parameters
def find_optimal_parameters(start_date: str, end_date: str, initial_cash: float):
    """
   [역할 변경] 주어진 기간 동안 GPU를 사용하여 파라미터 최적화를 실행하고,
  '전체 시뮬레이션 결과'를 DataFrame으로 반환합니다.
   (WFO 오케스트레이터가 이 결과를 받아 클러스터링 분석을 수행합니다.
    """
    print(f"\n" + "="*80)
    print(f"WORKER: Running GPU Simulations for {start_date} to {end_date}")
    print("="*80)
    
    # 데이터 로드
    all_data_gpu = preload_all_data_to_gpu(db_connection_str, start_date, end_date)
    weekly_filtered_gpu = preload_weekly_filtered_stocks_to_gpu(db_connection_str, start_date, end_date)
    
    # 거래일 및 티커 리스트 준비
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
    all_tickers = sorted(all_data_gpu.index.get_level_values("ticker").unique().to_pandas().tolist())
    
    print(f"  - Tickers for period: {len(all_tickers)}")
    print(f"  - Trading days for period: {len(trading_dates_pd)}")
    #  배치 처리 로직 
    batch_size = backtest_settings.get('simulation_batch_size')
    if batch_size is None or batch_size <= 0:
        batch_size = num_combinations
    num_batches = (num_combinations + batch_size - 1) // batch_size
    print(f"  - Total Simulations: {num_combinations} | Batch Size: {batch_size} | Batches: {num_batches}")    
   
   
   
   
    all_daily_values_list = []
    total_kernel_time = 0
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_combinations)
        param_batch = param_combinations[start_idx:end_idx]
        
        print(f"\n  --- Running Batch {i+1}/{num_batches} (Sims {start_idx}-{end_idx-1}) ---")
        
        start_time_kernel = time.time()
        daily_values_batch = run_gpu_optimization(
            param_batch, all_data_gpu, weekly_filtered_gpu, all_tickers, 
            trading_date_indices_gpu, trading_dates_pd, initial_cash, execution_params
        )
        end_time_kernel = time.time()
        
        batch_time = end_time_kernel - start_time_kernel
        total_kernel_time += batch_time
        print(f"  - Batch {i+1} Kernel Execution Time: {batch_time:.2f}s")

        all_daily_values_list.append(daily_values_batch)

    print(f"\n  - Total GPU Kernel Execution Time: {total_kernel_time:.2f}s")
    # 모든 배치의 결과를 하나로 합침
    if not all_daily_values_list:
        print("[Error] No simulation results were generated.")
        return {}, pd.DataFrame()
    
    daily_values_result = cp.vstack(all_daily_values_list)
    
    # 결과 분석 및 최적 파라미터 반환
    # 이 함수는 분석만 수행하고, 결과 DF를 그대로 반환합니다.
    # 파일 저장은 단독 실행 시에만 이루어집니다.
    best_params_for_log, all_results_df = analyze_and_save_results(
        param_combinations, daily_values_result, trading_dates_pd, save_to_file=False
    )
    priority_map_rev = {0: 'lowest_order', 1: 'highest_drop'}
    if 'additional_buy_priority' in best_params_for_log:
        best_params_for_log['additional_buy_priority'] = priority_map_rev.get(int(best_params_for_log.get('additional_buy_priority', -1)), 'unknown')
    
    # 반환값은 (단순 최적 파라미터, 전체 시뮬레이션 결과 DF) 튜플을 유지합니다.
    # 오케스트레이터는 이 중 두 번째 값(all_results_df)을 사용합니다.
    return best_params_for_log, all_results_df
    
# 6. Main Execution Block
if __name__ == "__main__":
    # 이 파일이 단독으로 실행될 때, config.yaml의 전체 기간으로 최적화를 수행합니다.
    backtest_start_date = backtest_settings["start_date"]
    backtest_end_date = backtest_settings["end_date"]
    initial_cash = backtest_settings["initial_cash"]
    
    # 실행 모드를 명확히 알리고 사용자에게 가이드를 제공
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print(" 실행 모드: 단독 파라미터 최적화 (STANDALONE OPTIMIZATION MODE)")
    print("="*80)
    print(f" 이 스크립트는 아래 명시된 '단일 고정 기간'에 대해서만 1회 최적화를 수행합니다.")
    print(f"  - 최적화 대상 기간: {backtest_start_date} ~ {backtest_end_date}")
    
    wfo_settings = config.get('walk_forward_settings')
    if wfo_settings and wfo_settings.get('total_folds'):
        total_folds = wfo_settings.get('total_folds')
        print("\n [정보] 전체 Walk-Forward 분석을 실행하시려면 아래 명령어를 사용하십시오.")
        print(f"  - 명령어: python -m src.walk_forward_analyzer")
        print(f"  - 예상 Fold 수: {total_folds} folds")
    print("="*80 + "\n")
    # -------------------------------------------------------------------------
    # ----------------------------------------------------
    # 리팩토링된 워커 함수 호출
    best_parameters_found, all_results_df = find_optimal_parameters(
        start_date=backtest_start_date,
        end_date=backtest_end_date,
        initial_cash=initial_cash
    )
    
    print("\n" + "="*80)
    print("🏆 STANDALONE RUN - BEST PARAMETERS (by Calmar Ratio) 🏆")
    print("="*80)
    if best_parameters_found:
        for key, value in best_parameters_found.items():
            if isinstance(value, float):
                print(f"  - {key:<25}: {value:.4f}")
            else:
                print(f"  - {key:<25}: {value}")
    else:
        print("  No valid parameters found.")
    print("="*80)

    # 단독 실행 시에는 전체 결과 CSV 파일을 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'standalone_simulation_results_{timestamp}.csv')
    all_results_df.to_csv(filepath, index=False, float_format='%.4f')
    print(f"\n✅ Full simulation analysis saved to: {filepath}")

