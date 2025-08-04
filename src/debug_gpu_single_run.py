# debug_gpu_single_run.py (수정 후)

"""
This script is used to debug the GPU single run.
It is used to test the GPU single run with the parameters from the config.yaml file.
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

# --- Debug를 위한 단일 파라미터 조합 정의 ---
cpu_test_params = config["strategy_params"]
max_stocks_options = cp.array([cpu_test_params["max_stocks"]], dtype=cp.int32)
order_investment_ratio_options = cp.array(
    [cpu_test_params["order_investment_ratio"]], dtype=cp.float32
)
additional_buy_drop_rate_options = cp.array(
    [cpu_test_params["additional_buy_drop_rate"]], dtype=cp.float32
)
sell_profit_rate_options = cp.array(
    [cpu_test_params["sell_profit_rate"]], dtype=cp.float32
)
priority_map = {"lowest_order": 0, "highest_drop": 1}
priority_val = priority_map.get(cpu_test_params["additional_buy_priority"], 0)
additional_buy_priority_options = cp.array([priority_val], dtype=cp.int32)
grid = cp.meshgrid(
    max_stocks_options,
    order_investment_ratio_options,
    additional_buy_drop_rate_options,
    sell_profit_rate_options,
    additional_buy_priority_options,
)
param_combinations = cp.vstack([item.flatten() for item in grid]).T
num_combinations = param_combinations.shape[0]

print("✅ [DEBUG MODE] Single parameter combination for GPU test:")
print(param_combinations)

# -----------------------------------------------------------------------------
# 2. GPU Data Pre-loader
# -----------------------------------------------------------------------------


# parameter_simulation_gpu.py 와 동일한 함수이므로 코드를 그대로 재사용합니다.
def preload_all_data_to_gpu(conn, start_date, end_date):
    """
    Loads all necessary stock data for the entire backtest period into a
    single cuDF DataFrame, minimizing I/O during simulation.
    """
    # ... (위 parameter_simulation_gpu.py의 함수와 동일한 내용) ...
    print("⏳ Loading all stock data into GPU memory...")
    start_time = time.time()
    query = f"""
    SELECT stock_code AS ticker, date, open_price, high_price, low_price, close_price, volume, atr_14_ratio
    FROM DailyStockPrice JOIN CalculatedIndicators USING (stock_code, date)
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """
    df_pd = pd.read_sql(query, conn, parse_dates=["date"])
    gdf = cudf.from_pandas(df_pd)
    gdf = gdf.set_index(["ticker", "date"])
    end_time = time.time()
    print(
        f"✅ Data loaded to GPU. Shape: {gdf.shape}. Time: {end_time - start_time:.2f}s"
    )
    return gdf


def preload_weekly_filtered_stocks_to_gpu(conn, start_date, end_date):
    """
    Loads all weekly filtered stock codes for the backtest period into a
    cuDF DataFrame.
    """
    # ... (위 parameter_simulation_gpu.py의 함수와 동일한 내용) ...
    print("⏳ Loading weekly filtered stocks data to GPU memory...")
    start_time = time.time()
    query = f"""
    SELECT `filter_date` as date, `stock_code` as ticker FROM WeeklyFilteredStocks
    WHERE `filter_date` BETWEEN '{start_date}' AND '{end_date}'
    """
    df_pd = pd.read_sql(query, conn, parse_dates=["date"])
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


# parameter_simulation_gpu.py 와 동일한 함수이므로 코드를 그대로 재사용합니다.
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
    # ... (위 parameter_simulation_gpu.py의 함수와 동일한 내용) ...
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

        # 2. Generate trading dates... (이하 로직은 parameter_simulation_gpu.py와 동일)
        trading_dates_pd = pd.bdate_range(
            start=backtest_start_date, end=backtest_end_date
        )
        trading_date_indices_gpu = cp.arange(len(trading_dates_pd), dtype=cp.int32)
        all_data_gpu = all_data_gpu[
            all_data_gpu.index.get_level_values("date").isin(trading_dates_pd)
        ]
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

        # ... (결과 출력 로직은 parameter_simulation_gpu.py와 동일) ...
        print(f"\n--- 🎉 GPU 백테스팅 완료 ---")
        print(f"⏱️  총 소요 시간: {elapsed_time:.2f}초")

        returns_cpu = total_returns.get()
        print(f"\n📊 성과 요약 (단일 조합):")
        print(f"   최종 수익률: {returns_cpu[0]*100:.2f}%")

        print(f"\n✅ GPU 디버깅 시스템 테스트 완료!")

    finally:
        if db_conn:
            db_conn.close()
            print("\nDatabase connection closed.")
