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

db_pass_encoded = urllib.parse.quote_plus(db_config["password"])
db_connection_str = f"mysql+pymysql://{db_config['user']}:{db_pass_encoded}@{db_config['host']}/{db_config['database']}"

# Define the parameter space to be tested
max_stocks_options = cp.array([12,24], dtype=cp.int32)
order_investment_ratio_options = cp.array([0.015,0.02,0.025], dtype=cp.float32) # [0.02,0.15,0.25]
additional_buy_drop_rate_options = cp.array([0.06,0.08], dtype=cp.float32) # [0.06,0.07,0.08 ]
sell_profit_rate_options = cp.array([0.16,0.12], dtype=cp.float32) # [0.16,0.15,0.14,0.13]
additional_buy_priority_options = cp.array([0,1], dtype=cp.int32) # 0: lowest_order, 1: highest_drop

# --- [New] Define search space for advanced risk parameters ---
stop_loss_rate_options = cp.array([-0.40,-0.50,-0.6], dtype=cp.float32) # [-0.40,-0.50, -0.55,-0.6]
max_splits_limit_options = cp.array([10,20], dtype=cp.int32) # [10,15,20]
max_inactivity_period_options = cp.array([50,60], dtype=cp.int32) # [30,45,60]

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
# [추가] 변경사항 확인을 위한 검증용 print문
print(f"✅ [VERIFICATION] Newly compiled code is running. Num combinations: {num_combinations}")
print(f"✅ [VERIFICATION] max_stocks_options shape: {max_stocks_options.shape}")

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
    best_params_series = sorted_df.iloc[0]
    best_params_dict = best_params_series.to_dict()

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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'gpu_simulation_results_{timestamp}.csv')
    sorted_df.to_csv(filepath, index=False, float_format='%.4f')
    print(f"\n✅ Full analysis saved to: {filepath}")
    print(f"⏱️  Analysis and saving took: {time.time() - start_time:.2f} seconds.")
    # [수정] 파일 저장 로직을 조건부로 실행
    if save_to_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'gpu_simulation_results_{timestamp}.csv')
        sorted_df.to_csv(filepath, index=False, float_format='%.4f')
        print(f"\n✅ Full analysis saved to: {filepath}")
    print(f"⏱️  Analysis took: {time.time() - start_time:.2f} seconds.") # [수정] 문구 변경
    
    return best_params_dict, sorted_df # [추가] 전체 결과 DF도 반환
# 5. [신규] 워커 함수: find_optimal_parameters
def find_optimal_parameters(start_date: str, end_date: str, initial_cash: float):
    """
    주어진 기간 동안 GPU를 사용하여 최적의 전략 파라미터를 찾습니다.
    WFO 오케스트레이터에 의해 호출되는 '워커' 함수입니다.
    """
    print(f"\n" + "="*80)
    print(f"WORKER: Finding Optimal Parameters for {start_date} to {end_date}")
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
    trading_dates_pd = pd.read_sql(trading_dates_query, sql_engine, parse_dates=['date'])['date']
    trading_date_indices_gpu = cp.arange(len(trading_dates_pd), dtype=cp.int32)
    
    all_data_gpu = all_data_gpu[all_data_gpu.index.get_level_values('date').isin(trading_dates_pd)]
    all_tickers = sorted(all_data_gpu.index.get_level_values("ticker").unique().to_pandas().tolist())
    
    print(f"  - Tickers for period: {len(all_tickers)}")
    print(f"  - Trading days for period: {len(trading_dates_pd)}")

    # 백테스팅 커널 실행
    start_time_kernel = time.time()
    daily_values_result = run_gpu_optimization(
        param_combinations, all_data_gpu, weekly_filtered_gpu, all_tickers, 
        trading_date_indices_gpu, trading_dates_pd, initial_cash, execution_params
    )
    end_time_kernel = time.time()
    elapsed_time = end_time_kernel - start_time_kernel
    print(f"  - GPU Kernel Execution Time: {elapsed_time:.2f}s")
    
    # 결과 분석 및 최적 파라미터 반환
    # 이 함수가 워커로 호출될 때는 전체 시뮬레이션 결과를 파일로 저장할 필요가 없음 (save_to_file=False)
    best_params, all_results_df = analyze_and_save_results(
        param_combinations, daily_values_result, trading_dates_pd, save_to_file=False
    )

    # additional_buy_priority 값을 문자열로 변환하여 가독성 증진
    priority_map_rev = {0: 'lowest_order', 1: 'highest_drop'}
    if 'additional_buy_priority' in best_params:
        best_params['additional_buy_priority'] = priority_map_rev.get(int(best_params['additional_buy_priority']), 'unknown')
    
    strategy_param_keys = [
        'max_stocks', 'order_investment_ratio', 'additional_buy_drop_rate', 
        'sell_profit_rate', 'additional_buy_priority', 'stop_loss_rate', 
        'max_splits_limit', 'max_inactivity_period'
    ]
    cleaned_params = {key: best_params[key] for key in strategy_param_keys if key in best_params}    
    
    # [핵심 수정] 2개의 값을 튜플로 묶어 반환
    return cleaned_params, all_results_df
    
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
    print("🏆 FINAL OPTIMAL PARAMETERS FOUND 🏆")
    print("="*80)
    # 딕셔너리를 예쁘게 출력
    for key, value in best_parameters_found.items():
        if isinstance(value, float):
            print(f"  - {key:<25}: {value:.4f}")
        else:
            print(f"  - {key:<25}: {value}")
    print("="*80)

    # 단독 실행 시에는 전체 결과 CSV 파일을 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'standalone_simulation_results_{timestamp}.csv')
    all_results_df.to_csv(filepath, index=False, float_format='%.4f')
    print(f"\n✅ Full simulation analysis saved to: {filepath}")

