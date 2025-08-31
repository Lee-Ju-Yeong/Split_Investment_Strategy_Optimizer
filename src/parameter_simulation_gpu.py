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
max_stocks_options = cp.array([10], dtype=cp.int32)
order_investment_ratio_options = cp.array([0.03], dtype=cp.float32)
additional_buy_drop_rate_options = cp.array([0.03, ], dtype=cp.float32)
sell_profit_rate_options = cp.array([0.03], dtype=cp.float32)
additional_buy_priority_options = cp.array([0, 1], dtype=cp.int32) # 0: lowest_order, 1: highest_drop

# --- [New] Define search space for advanced risk parameters ---
stop_loss_rate_options = cp.array([-0.40,-0.50, -0.60], dtype=cp.float32)
max_splits_limit_options = cp.array([10], dtype=cp.int32)
max_inactivity_period_options = cp.array([60, 120], dtype=cp.int32)

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
print(f"✅ Total parameter combinations generated for GPU: {num_combinations}")

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
def analyze_and_save_results(param_combinations_gpu, daily_values_gpu, trading_dates_pd):
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

# 5. Main Execution Block
if __name__ == "__main__":
    backtest_start_date = backtest_settings["start_date"]
    backtest_end_date = backtest_settings["end_date"]
    initial_cash = backtest_settings["initial_cash"]
    print(f"📅 테스트 기간: {backtest_start_date} ~ {backtest_end_date}")
    # [이동] all_data_gpu를 먼저 로드하여 NameError를 방지
    all_data_gpu = preload_all_data_to_gpu(db_connection_str, backtest_start_date, backtest_end_date)
    weekly_filtered_gpu = preload_weekly_filtered_stocks_to_gpu(db_connection_str, backtest_start_date, backtest_end_date)
    
    
    # CPU/Debug 스크립트와 동일하게 DB에서 실제 거래일만 가져오도록 변경
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
    # [추가] Ticker-Index 매핑의 일관성을 보장하기 위해 리스트를 정렬합니다.
    all_tickers = sorted(all_data_gpu.index.get_level_values("ticker").unique().to_pandas().tolist())
    print("✅ Ticker list has been sorted to ensure deterministic mapping.")

    print(f"📊 로드된 종목 수: {len(all_tickers)}")
    print(f"📊 실제 거래일 수: {len(trading_date_indices_gpu)}")

    print(f"\n🚀 {num_combinations}개 파라미터 조합으로 GPU 백테스팅 시작...")
    start_time = time.time()
    daily_values_result = run_gpu_optimization(
        param_combinations, all_data_gpu, weekly_filtered_gpu, all_tickers, 
        trading_date_indices_gpu, trading_dates_pd, initial_cash, execution_params
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n--- 🎉 GPU 백테스팅 완료 ---")
    print(f"⏱️  총 소요 시간: {elapsed_time:.2f}초")
    print(f"📈 조합 당 평균 시간: {elapsed_time/num_combinations*1000:.2f}ms")

    analyze_and_save_results(param_combinations, daily_values_result, trading_dates_pd)
    print(f"\n✅ GPU 파라미터 최적화 및 분석 완료!")
