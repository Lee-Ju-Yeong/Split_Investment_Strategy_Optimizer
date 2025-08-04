# parameter_simulation_gpu.py (수정 후)

"""
GPU-accelerated parameter simulation for the MagicSplitStrategy.
This script orchestrates the backtesting of thousands of parameter combinations
by leveraging CuPy and CuDF for massive parallelization on the GPU.
"""
import time
import cudf
import cupy as cp
import pandas as pd

# from sqlalchemy import create_engine -> SQLAlchemy 의존성 제거
# import configparser -> configparser 의존성 제거
# import urllib.parse -> urllib 의존성 제거

# --- 필요한 모듈 추가 임포트 ---
from src.config_loader import load_config
from src.db_setup import get_db_connection  # 중앙화된 DB 연결 함수 임포트
from src.backtest_strategy_gpu import run_magic_split_strategy_on_gpu

# -----------------------------------------------------------------------------
# 1. Configuration and Parameter Setup
# -----------------------------------------------------------------------------

# --- 설정 파일 로드 (YAML 로더로 통일) ---
config = load_config()
db_config = config["database"]
backtest_settings = config["backtest_settings"]

# URL 인코딩을 포함하여 DB 연결 문자열 생성 -> 더 이상 필요 없음

# Define the parameter space to be tested
max_stocks_options = cp.array([15, 30], dtype=cp.int32)
order_investment_ratio_options = cp.array([0.015, 0.022, 0.03], dtype=cp.float32)
additional_buy_drop_rate_options = cp.array([0.03, 0.04, 0.05], dtype=cp.float32)
sell_profit_rate_options = cp.array([0.03, 0.04, 0.05], dtype=cp.float32)
additional_buy_priority_options = cp.array(
    [0, 1], dtype=cp.int32
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


def preload_all_data_to_gpu(
    conn, start_date, end_date
):  # engine 대신 conn 객체를 받도록 수정
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

    # Load data using pandas first, with the provided pymysql connection
    df_pd = pd.read_sql(query, conn, parse_dates=["date"])

    # Move the DataFrame to the GPU
    gdf = cudf.from_pandas(df_pd)

    # Set a multi-index for efficient lookups
    gdf = gdf.set_index(["ticker", "date"])

    end_time = time.time()
    print(
        f"✅ Data loaded to GPU. Shape: {gdf.shape}. Time: {end_time - start_time:.2f}s"
    )

    return gdf


def preload_weekly_filtered_stocks_to_gpu(
    conn, start_date, end_date
):  # engine 대신 conn 객체를 받도록 수정
    """
    Loads all weekly filtered stock codes for the backtest period into a
    cuDF DataFrame.
    """
    print("⏳ Loading weekly filtered stocks data to GPU memory...")
    start_time = time.time()

    # WeeklyFilteredStocks 테이블에서 해당 기간의 모든 데이터를 가져옴
    query = f"""
    SELECT `filter_date` as date, `stock_code` as ticker
    FROM WeeklyFilteredStocks
    WHERE `filter_date` BETWEEN '{start_date}' AND '{end_date}'
    """
    # Load data using pandas first, with the provided pymysql connection
    df_pd = pd.read_sql(query, conn, parse_dates=["date"])

    # cuDF로 변환
    gdf = cudf.from_pandas(df_pd)
    gdf = gdf.set_index("date")

    end_time = time.time()
    print(
        f"✅ Weekly filtered stocks loaded to GPU. Shape: {gdf.shape}. Time: {end_time - start_time:.2f}s"
    )
    return gdf


# -----------------------------------------------------------------------------
# 3. GPU Backtesting Kernel (to be implemented)
# -----------------------------------------------------------------------------


# 이 함수는 DB 접근 로직이 없으므로 수정사항 없음
def run_gpu_optimization(
    params_gpu,
    data_gpu,
    weekly_filtered_gpu,
    all_tickers,
    trading_date_indices_gpu,
    trading_dates_pd,
    initial_cash_value,
):
    """
    GPU-accelerated backtesting을 오케스트레이션합니다.
    """
    # ... (기존 코드와 동일) ...
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
    )

    print("🎉 GPU backtesting kernel finished.")

    final_values = daily_portfolio_values[:, -1]
    initial_values = cp.full(len(params_gpu), initial_cash_value, dtype=cp.float32)

    total_returns = (final_values / initial_values) - 1

    return total_returns, daily_portfolio_values


# -----------------------------------------------------------------------------
# 4. Main Execution Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    backtest_start_date = backtest_settings["start_date"]
    backtest_end_date = backtest_settings["end_date"]
    initial_cash = backtest_settings["initial_cash"]

    print(f"📅 테스트 기간: {backtest_start_date} ~ {backtest_end_date}")

    # DB 연결을 main 블록 시작 시 한 번만 수행
    db_conn = get_db_connection()
    if not db_conn:
        print("DB 연결 실패. 프로그램을 종료합니다.")
        exit()

    try:
        # 1. Pre-load all data to GPU, passing the connection object
        all_data_gpu = preload_all_data_to_gpu(
            db_conn, backtest_start_date, backtest_end_date
        )
        weekly_filtered_gpu = preload_weekly_filtered_stocks_to_gpu(
            db_conn, backtest_start_date, backtest_end_date
        )

        # 2. Generate trading dates and convert them to integer indices for GPU
        # Pandas의 bdate_range를 사용하여 실제 거래일만 가져옴
        trading_dates_pd = pd.bdate_range(
            start=backtest_start_date, end=backtest_end_date
        )

        # GPU 커널에서는 0, 1, 2... 와 같은 정수 인덱스로 날짜를 순회
        trading_date_indices_gpu = cp.arange(len(trading_dates_pd), dtype=cp.int32)

        # 3. Filter the main GPU DataFrame to include only actual trading dates
        #    This ensures the GPU data aligns with our trading date indices.
        #    cuDF는 datetime 객체를 인덱스로 직접 사용할 수 있음
        all_data_gpu = all_data_gpu[
            all_data_gpu.index.get_level_values("date").isin(trading_dates_pd)
        ]

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

        total_returns, daily_values = run_gpu_optimization(
            param_combinations,
            all_data_gpu,
            weekly_filtered_gpu,
            all_tickers,
            trading_date_indices_gpu,
            trading_dates_pd,
            initial_cash,
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        # 6. Results summary
        print(f"\n--- 🎉 GPU 백테스팅 완료 ---")
        print(f"⏱️  총 소요 시간: {elapsed_time:.2f}초")
        print(f"📈 조합 당 평균 시간: {elapsed_time/num_combinations*1000:.2f}ms")

        # ... (결과 출력 로직은 동일) ...
        returns_cpu = total_returns.get()
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

    finally:
        # 작업이 성공하든 실패하든 항상 DB 연결을 닫아줍니다.
        if db_conn:
            db_conn.close()
            print("\nDatabase connection closed.")
