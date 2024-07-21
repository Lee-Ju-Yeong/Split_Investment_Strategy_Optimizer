import os
import concurrent.futures
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import configparser
import numpy as np

# MagicSplit_Backtesting_Optimizer 모듈의 함수와 클래스 가져오기
from MagicSplit_Backtesting_Optimizer import (
    Position, Trade, get_stock_codes, load_stock_data_from_mysql, calculate_additional_buy_drop_rate,
    calculate_sell_profit_rate, initial_buy_sell, additional_buy, additional_sell, get_trading_dates_from_db,
    calculate_mdd, portfolio_backtesting, plot_backtesting_results, check_if_already_calculated, 
    run_backtesting_for_period, calculate_average_results, average_cagr_wrapper, backtesting_wrapper
)

# 설정 파일 읽기
config = configparser.ConfigParser()
config.read('config.ini')

# 데이터베이스 연결 정보 설정
db_params = {
    'host': config['mysql']['host'],
    'user': config['mysql']['user'],
    'password': config['mysql']['password'],
    'database': config['mysql']['database'],
    'connection_timeout': 60
}

# 초기 자본 설정
initial_capital = 100000000  # 초기 자본 1억

# 백테스팅 전체 기간 설정
start_date = '2004-01-01'
end_date = '2024-01-01'

per_threshold = 10
pbr_threshold = 1
div_threshold = 1.0
min_additional_buy_drop_rate = 0.005

# 백테스팅 파라미터 옵션 설정
num_splits_options = [20]
buy_threshold_options = [30]
investment_ratio_options = [0.3, ]#0.25
consider_delisting_options = [False]
max_stocks_options = [40,] # 50, 55]

# 파라미터 조합 생성
combinations = [(n, b, i, c, m) for n in num_splits_options for b in buy_threshold_options for i in investment_ratio_options for c in consider_delisting_options for m in max_stocks_options]

# 여러 기간 설정
time_periods = [(2006, 2008), ] #(2008, 2023), (2010, 2023), (2012, 2023), (2014, 2023)

# 파일 저장 경로 설정
current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
results_folder = f'parameter_simulation_{current_time_str}'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 파라미터 조합 엑셀 파일 저장 경로
results_file = os.path.join(results_folder, f'parameter_combinations_{current_time_str}.csv')

def run_backtesting_and_save_results():
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for param in combinations:
            if not check_if_already_calculated(param[0], param[1], param[2], param[3], param[4], results_folder):
                futures.append(executor.submit(
    average_cagr_wrapper, 
    param, 
    time_periods, 
    db_params, 
    per_threshold, 
    pbr_threshold, 
    div_threshold, 
    min_additional_buy_drop_rate, 
    param[3], 
    param[4], 
    results_folder
))
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)

    results_df = pd.DataFrame(results, columns=["num_splits", "buy_threshold", "investment_ratio", "consider_delisting", "max_stocks", "Average_Total_Value", "Average_CAGR", "Average_MDD"])
    results_df.to_csv(results_file, index=False)
    print(results_df.sort_values(by="Average_CAGR", ascending=False))


if __name__ == "__main__":
    run_backtesting_and_save_results()
