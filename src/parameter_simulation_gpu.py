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
from datetime import timedelta, datetime  # timedelta 임포트 추가
import os # os 추가
import urllib.parse
# --- 필요한 모듈 추가 임포트 ---
from src.config_loader import load_config
from src.backtest_strategy_gpu import run_magic_split_strategy_on_gpu
from src.performance_analyzer import PerformanceAnalyzer

# -----------------------------------------------------------------------------
# 1. Configuration and Parameter Setup
# -----------------------------------------------------------------------------

import urllib.parse

# --- 설정 파일 로드 (YAML 로더로 통일) ---
config = load_config()
execution_params = config["execution_params"]  # config에서 파라미터 로드
db_config = config["database"]
backtest_settings = config["backtest_settings"]

# URL 인코딩을 포함하여 DB 연결 문자열 생성
db_pass_encoded = urllib.parse.quote_plus(db_config["password"])
db_connection_str = (
    f"mysql+pymysql://{db_config['user']}:{db_pass_encoded}"
    f"@{db_config['host']}/{db_config['database']}"
)

# Define the parameter space to be tested
max_stocks_options = cp.array([24], dtype=cp.int32)
order_investment_ratio_options = cp.array([0.02], dtype=cp.float32)
additional_buy_drop_rate_options = cp.array([0.04], dtype=cp.float32)
sell_profit_rate_options = cp.array([0.04], dtype=cp.float32)
additional_buy_priority_options = cp.array(
    [0,1], dtype=cp.int32
)  # 0: lowest_order, 1: highest_drop

# Create all combinations using CuPy's broadcasting capabilities
grid = cp.meshgrid(
    max_stocks_options,
    order_investment_ratio_options,
    additional_buy_drop_rate_options,
    sell_profit_rate_options,
    additional_buy_priority_options,
)
# Flatten the grid to get a list of all combinations
param_combinations = cp.vstack([item.flatten() for item in grid]).T
num_combinations = param_combinations.shape[0]

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
    df_pd = pd.read_sql(query, sql_engine, parse_dates=["date"])

    # Move the DataFrame to the GPU
    gdf = cudf.from_pandas(df_pd)

    # Set a multi-index for efficient lookups
    gdf = gdf.set_index(["ticker", "date"])

    end_time = time.time()
    print(
        f"✅ Data loaded to GPU. Shape: {gdf.shape}. Time: {end_time - start_time:.2f}s"
    )

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


def run_gpu_optimization(
    params_gpu,
    data_gpu,
    weekly_filtered_gpu,
    all_tickers,
    trading_date_indices_gpu,
    trading_dates_pd,
    initial_cash_value,
    execution_params,
    debug_mode: bool = False,
):
    """
    GPU-accelerated backtesting을 오케스트레이션합니다.
    """
    print("🚀 Starting GPU backtesting kernel...")

    # GPU 백테스팅 함수 직접 호출
    daily_portfolio_values = run_magic_split_strategy_on_gpu(
        initial_cash=initial_cash_value,  # <<< 수정된 인자 이름 및 외부에서 받은 값 사용
        param_combinations=params_gpu,
        all_data_gpu=data_gpu,
        weekly_filtered_gpu=weekly_filtered_gpu,
        trading_date_indices=trading_date_indices_gpu,
        trading_dates_pd_cpu=trading_dates_pd,
        all_tickers=all_tickers,
        execution_params=execution_params,  # ★★★ 추가된 인자 전달
        max_splits_limit=20,
        debug_mode=debug_mode,
    )

    print("🎉 GPU backtesting kernel finished.")

    return daily_portfolio_values

# parameter_simulation_gpu.py 파일에 있는 이 함수를 아래 코드로 교체하세요.

def analyze_and_save_results(param_combinations_gpu, daily_values_gpu, trading_dates_pd, initial_cash):
    """
    Analyzes the results from the GPU backtest, calculates detailed metrics,
    prints the top performers, and saves the full results to a CSV file.

    Args:
        param_combinations_gpu (cp.ndarray): The parameter combinations tested.
        daily_values_gpu (cp.ndarray): The daily portfolio values for each combination.
        trading_dates_pd (pd.DatetimeIndex): The trading dates for the backtest period.
        initial_cash (float): The initial capital for the backtest.
    """
    print("\n--- 🔬 Analyzing detailed performance metrics ---")
    start_time = time.time()
    
    # 1. GPU에서 CPU로 데이터 이동
    param_combinations_cpu = param_combinations_gpu.get()
    daily_values_cpu = daily_values_gpu.get()
    
    results_list = []
    num_combinations = daily_values_cpu.shape[0]

    # 2. 각 파라미터 조합에 대해 상세 지표 계산
    for i in range(num_combinations):
        # 일별 가치 데이터를 Pandas Series로 변환
        daily_series = pd.Series(daily_values_cpu[i], index=trading_dates_pd)
        
         ### ### [핵심 수정] Series를 DataFrame으로 변환 ### ###
        # PerformanceAnalyzer가 요구하는 'total_value' 컬럼을 가진 DataFrame 생성
        history_df_mock = pd.DataFrame(daily_series, columns=['total_value'])
        
        # PerformanceAnalyzer를 사용하여 지표 계산
        analyzer = PerformanceAnalyzer(history_df_mock)
        metrics = analyzer.get_metrics(formatted=False) # 원본 숫자 데이터로 받기
        results_list.append(metrics)

    # 3. 파라미터와 성과 지표를 하나의 DataFrame으로 결합
    param_names = ['max_stocks', 'order_investment_ratio', 'additional_buy_drop_rate', 'sell_profit_rate', 'additional_buy_priority']
    params_df = pd.DataFrame(param_combinations_cpu, columns=param_names)
    metrics_df = pd.DataFrame(results_list)
    
    full_results_df = pd.concat([params_df, metrics_df], axis=1)

    # 4. Calmar Ratio 기준으로 정렬하여 상위 결과 출력
    # Calmar Ratio가 무한대(inf)나 NaN인 경우를 대비하여 정렬 전에 처리
    full_results_df.replace([cp.inf, -cp.inf], cp.nan, inplace=True)
    sorted_df = full_results_df.sort_values(by='calmar_ratio', ascending=False).dropna(subset=['calmar_ratio'])

    print("\n🏆 Top 10 Performing Parameter Combinations (by Calmar Ratio):")
    
    # 출력할 컬럼 선택 및 포맷팅
    display_columns = [
        'calmar_ratio', 'cagr', 'mdd', 'sharpe_ratio', 'sortino_ratio', 'annualized_volatility',
        'max_stocks', 'order_investment_ratio', 'additional_buy_drop_rate', 'sell_profit_rate'
    ]
    # 'additional_buy_priority'가 있는지 확인하고 추가
    if 'additional_buy_priority' in sorted_df.columns:
        display_columns.append('additional_buy_priority')
        
    display_df = sorted_df.head(10).get(display_columns, pd.DataFrame())

    if not display_df.empty:
        # 숫자 포맷팅
        display_df['calmar_ratio'] = display_df['calmar_ratio'].map('{:.2f}'.format)
        display_df['cagr'] = display_df['cagr'].map('{:.2%}'.format)
        display_df['mdd'] = display_df['mdd'].map('{:.2%}'.format)
        display_df['sharpe_ratio'] = display_df['sharpe_ratio'].map('{:.2f}'.format)
        display_df['sortino_ratio'] = display_df['sortino_ratio'].map('{:.2f}'.format)
        display_df['annualized_volatility'] = display_df['annualized_volatility'].map('{:.2%}'.format)
        
        print(display_df.to_string(index=False))
    else:
        print("결과를 표시할 데이터가 없습니다.")

    # 5. 전체 결과를 CSV 파일로 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'gpu_simulation_results_{timestamp}.csv')
    
    sorted_df.to_csv(filepath, index=False, float_format='%.4f')
    
    end_time = time.time()
    print(f"\n✅ Full analysis saved to: {filepath}")
    print(f"⏱️  Analysis and saving took: {end_time - start_time:.2f} seconds.")

# -----------------------------------------------------------------------------
# 4. Main Execution Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    backtest_start_date = backtest_settings["start_date"]
    backtest_end_date = backtest_settings["end_date"]
    initial_cash = backtest_settings[
        "initial_cash"
    ]  # <<< config에서 초기 자본 가져오기

    print(f"📅 테스트 기간: {backtest_start_date} ~ {backtest_end_date}")

    # 1. Pre-load all data to GPU
    all_data_gpu = preload_all_data_to_gpu(
        db_connection_str, backtest_start_date, backtest_end_date
    )
    # 💡 새로운 데이터 로더 호출: 주간 필터링된 종목 데이터 로드
    weekly_filtered_gpu = preload_weekly_filtered_stocks_to_gpu(
        db_connection_str, backtest_start_date, backtest_end_date
    )

    # --- 💡 수정된 부분 시작 💡 ---

    # 2. Generate trading dates and convert them to integer indices for GPU
    # Pandas의 bdate_range를 사용하여 실제 거래일만 가져옴
    trading_dates_pd = pd.bdate_range(start=backtest_start_date, end=backtest_end_date)

    # GPU 커널에서는 0, 1, 2... 와 같은 정수 인덱스로 날짜를 순회
    trading_date_indices_gpu = cp.arange(len(trading_dates_pd), dtype=cp.int32)

    # 3. Filter the main GPU DataFrame to include only actual trading dates
    #    This ensures the GPU data aligns with our trading date indices.
    #    cuDF는 datetime 객체를 인덱스로 직접 사용할 수 있음
    all_data_gpu = all_data_gpu[
        all_data_gpu.index.get_level_values("date").isin(trading_dates_pd)
    ]

    # --- 💡 수정된 부분 끝 💡 ---

    # 4. Get all tickers from the loaded data
    #    (all_data_gpu가 필터링되었으므로, 여기서 티커를 가져오는 것이 정확함)
    all_tickers = (
        all_data_gpu.index.get_level_values("ticker").unique().to_pandas().tolist()
    )
    print(f"📊 로드된 종목 수: {len(all_tickers)}")
    print(f"📊 실제 거래일 수: {len(trading_date_indices_gpu)}")

    # 5. Run the backtesting kernel
    print(f"\n🚀 {num_combinations}개 파라미터 조합으로 GPU 백테스팅 시작...")
    start_time = time.time()

    # 함수 이름 변경 및 initial_cash 전달
    daily_values_result = run_gpu_optimization(
        param_combinations,
        all_data_gpu,
        weekly_filtered_gpu,
        all_tickers,
        trading_date_indices_gpu,
        trading_dates_pd,
        initial_cash,
        execution_params,
        debug_mode=False, # 대규모 실행 시에는 디버그 모드 비활성화
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 5. Results summary
    print(f"\n--- 🎉 GPU 백테스팅 완료 ---")
    print(f"⏱️  총 소요 시간: {elapsed_time:.2f}초")
    print(f"📈 조합 당 평균 시간: {elapsed_time/num_combinations*1000:.2f}ms")
    print(
        f"🔥 CPU 대비 예상 가속도: {8 * elapsed_time / (num_combinations * 0.1):.1f}x"
    )


    # ### 이슈 #3 구현: 새로운 분석 및 저장 함수 호출 ###
    analyze_and_save_results(param_combinations, daily_values_result, trading_dates_pd, initial_cash)
    
    print(f"\n✅ GPU 파라미터 최적화 및 분석 완료!")