import os
import concurrent.futures
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import configparser
import numpy as np
import random

# MagicSplit_Backtesting_Optimizer 모듈의 함수와 클래스 가져오기
from MagicSplit_Backtesting_Optimizer import (
    Position, Trade, get_stock_codes, load_stock_data_from_mysql, calculate_additional_buy_drop_rate,
    calculate_sell_profit_rate, initial_buy_sell, additional_buy, additional_sell, get_trading_dates_from_db,
    calculate_mdd, portfolio_backtesting, plot_backtesting_results
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
}

# 초기 자본 설정
initial_capital = 100000000  # 초기 자본 1억
per_threshold = 10
pbr_threshold = 1
div_threshold = 1.0
min_additional_buy_drop_rate = 0.005
seed=101
# 백테스팅 파라미터 옵션 설정
num_splits_options = [20]
buy_threshold_options = [30]
investment_ratio_options = [0.3,0.25 ]#0.25
consider_delisting_options = [False]
max_stocks_options = [40,] # 50, 55]

# 파라미터 조합 생성
combinations = [(n, b, i, c, m) for n in num_splits_options for b in buy_threshold_options for i in investment_ratio_options for c in consider_delisting_options for m in max_stocks_options]

# 여러 기간 설정
time_periods = [(2006, 2007),(2005, 2006) ] #(2008, 2023), (2010, 2023), (2012, 2023), (2014, 2023)

# 상위 폴더 설정
base_folder = 'parameter_simulation'
if not os.path.exists(base_folder):
    os.makedirs(base_folder)


# 파일 저장 경로 설정
current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
results_folder = os.path.join(base_folder, f'parameter_simulation_{current_time_str}')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 파라미터 조합 엑셀 파일 저장 경로
results_file = os.path.join(base_folder, f'parameter_combinations_{current_time_str}.csv')

def get_most_recent_results_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    if not files:
        return None
    most_recent_file = max(files, key=os.path.getctime)
    return most_recent_file

def check_if_already_calculated(num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks,directory):
    most_recent_file = get_most_recent_results_file(directory)
    if most_recent_file:
        existing_results = pd.read_csv(most_recent_file)
        return not existing_results[
            (existing_results['num_splits'] == num_splits) & 
            (existing_results['buy_threshold'] == buy_threshold) &
            (existing_results['investment_ratio'] == investment_ratio) &
            (existing_results['consider_delisting'] == consider_delisting) &
            (existing_results['max_stocks'] == max_stocks)
        ].empty
    return False

def run_backtesting_for_period(seed,initial_capital,num_splits, buy_threshold, investment_ratio, start_year, end_year, db_params, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, consider_delisting, max_stocks, results_folder):
    random.seed(seed)
    np.random.seed(seed)
    start_date = f'{start_year}-01-01'
    end_date = f'{end_year}-12-31'
    try:
         _, total_portfolio_value, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, all_trading_dates, cagr = portfolio_backtesting(seed,
            initial_capital, num_splits, investment_ratio, buy_threshold, start_date, end_date, db_params, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, consider_delisting, max_stocks, results_folder
        )
    except Exception as e:
        print(f'Error in backtesting for period {start_year}-{end_year}: {e}')
    mdd = calculate_mdd(portfolio_values_over_time)
    plot_backtesting_results(all_trading_dates, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, num_splits, max_stocks, buy_threshold, cagr, mdd, results_folder)
    return total_portfolio_value, cagr, mdd

def calculate_average_results(backtesting_results):
    total_values = np.array([result[0] for result in backtesting_results])
    cagr_values = np.array([result[1] for result in backtesting_results])
    mdd_values = np.array([result[2] for result in backtesting_results])

    average_total_value = np.mean(total_values)
    average_cagr = np.mean(cagr_values)
    average_mdd = np.mean(mdd_values)

    return average_total_value, average_cagr, average_mdd

def average_cagr_wrapper(seed,initial_capital,params, time_periods, db_params, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, consider_delisting, max_stocks, results_folder):
    num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks = params
    period_results = []
    for start_year, end_year in time_periods:
        try:
            total_value, cagr, mdd = run_backtesting_for_period(seed,initial_capital,
                num_splits, buy_threshold, investment_ratio, start_year, end_year, db_params, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, consider_delisting, max_stocks, results_folder
            )
            period_results.append((total_value, cagr, mdd))
        except Exception as e:
            print(f'Error in backtesting for period {start_year}-{end_year}: {e}')
            period_results.append((0, None, None))

    average_results = calculate_average_results(period_results)
    return num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks, average_results[0], average_results[1], average_results[2]

def run_backtesting_and_save_results():
    results = []
    for param in combinations:
        result = average_cagr_wrapper(seed,initial_capital, param, time_periods, db_params, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, param[3], param[4], results_folder)
        results.append(result)

    results_df = pd.DataFrame(results, columns=["num_splits", "buy_threshold", "investment_ratio", "consider_delisting", "max_stocks", "Average_Total_Value", "Average_CAGR", "Average_MDD"])
    results_df.to_csv(results_file, index=False)
    print(results_df.sort_values(by="Average_CAGR", ascending=False))


if __name__ == "__main__":
    run_backtesting_and_save_results()