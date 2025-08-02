import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import configparser
import numpy as np
import random

# backtest_strategy 모듈의 함수와 클래스 가져오기
from backtest_strategy import (
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
'''
# 백테스팅 파라미터 옵션 설정
num_splits_options = [10,20]
buy_threshold_options = [30,35]
investment_ratio_options = [0.25,0.3,0.35]
consider_delisting_options = [False]
max_stocks_options = [30,45,60]

# 새로운 백테스팅 파라미터 옵션 설정
per_threshold_options = [5,10,15]
pbr_threshold_options = [0.5, 1, 1.5]
div_threshold_options = [1,3]
min_additional_buy_drop_rate_options = [0.005,0.015]
seed_options = [102]
'''

# 🎯 MagicSplitStrategy 실제 파라미터 기반 최적화 설정
# strategy.py의 MagicSplitStrategy 클래스에서 실제로 사용되는 파라미터들만 선별

# 1. 최대 보유 종목 수 (max_stocks)
max_stocks_options = [15, 20, 25, 30]  # 포트폴리오 다양성 테스트

# 2. 주문당 투자 비율 (order_investment_ratio) - 포트폴리오 대비 각 주문의 투자 비율
order_investment_ratio_options = [0.01, 0.015, 0.02, 0.025, 0.03]  # 1% ~ 3%

# 3. 추가매수 하락률 (additional_buy_drop_rate) - 이전 매수가 대비 하락률
additional_buy_drop_rate_options = [0.02, 0.03, 0.04, 0.05, 0.06]  # 2% ~ 6%

# 4. 매도 수익률 (sell_profit_rate) - 목표 수익률
sell_profit_rate_options = [0.03, 0.04, 0.05, 0.06, 0.08]  # 3% ~ 8%

# 5. 추가매수 우선순위 (additional_buy_priority)
additional_buy_priority_options = ['lowest_order', 'highest_drop']  # 우선순위 전략

# ❌ 제거된 파라미터들 (MagicSplitStrategy에서 사용하지 않음)
# - seed_options: 랜덤 요소가 없어서 불필요
# - num_splits: order 개념으로 자동 관리됨
# - buy_threshold, normalized_atr_threshold: 사용하지 않음
# - investment_ratio: order_investment_ratio로 대체됨

# 여러 기간 설정
time_periods = [(2006, 2023),(2008, 2023), (2010, 2023), (2012, 2023)] #(2006, 2023),(2008, 2023), (2010, 2023), (2012, 2023)

# MagicSplitStrategy 파라미터 조합 생성 (GPU 최적화용)
combinations = [(ms, oir, abdr, spr, abp) 
                for ms in max_stocks_options 
                for oir in order_investment_ratio_options 
                for abdr in additional_buy_drop_rate_options 
                for spr in sell_profit_rate_options
                for abp in additional_buy_priority_options]

print(f"총 파라미터 조합 수: {len(combinations)}")
print(f"총 백테스팅 실행 횟수: {len(combinations) * len(time_periods)}")
print(f"예상 조합 수: {len(max_stocks_options)} × {len(order_investment_ratio_options)} × {len(additional_buy_drop_rate_options)} × {len(sell_profit_rate_options)} × {len(additional_buy_priority_options)} = {len(max_stocks_options) * len(order_investment_ratio_options) * len(additional_buy_drop_rate_options) * len(sell_profit_rate_options) * len(additional_buy_priority_options)}개")

# 상위 폴더 설정
base_folder = 'parameter_simulation'
if not os.path.exists(base_folder):
    os.makedirs(base_folder)


# 파일 저장 경로 설정
current_time_str = datetime.now().strftime('%Y%m%d_%H%M')
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

def check_if_already_calculated(num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, seed, directory):
    most_recent_file = get_most_recent_results_file(directory)
    if most_recent_file:
        existing_results = pd.read_csv(most_recent_file)
        return not existing_results[
            (existing_results['num_splits'] == num_splits) & 
            (existing_results['buy_threshold'] == buy_threshold) &
            (existing_results['investment_ratio'] == investment_ratio) &
            (existing_results['consider_delisting'] == consider_delisting) &
            (existing_results['max_stocks'] == max_stocks) &
            (existing_results['per_threshold'] == per_threshold) &
            (existing_results['pbr_threshold'] == pbr_threshold) &
            (existing_results['div_threshold'] == div_threshold) &
            (existing_results['min_additional_buy_drop_rate'] == min_additional_buy_drop_rate) &
            (existing_results['seed'] == seed)
        ].empty
    return False


def run_backtesting_for_period(initial_capital, max_stocks, order_investment_ratio, additional_buy_drop_rate, sell_profit_rate, additional_buy_priority, start_year, end_year, db_params, results_folder,save_files=False):
    start_date = f'{start_year}-01-01'
    end_date = f'{end_year}-12-31'
    
    # MagicSplitStrategy 파라미터로 백테스팅 실행
    # 주의: 실제로는 strategy.py의 MagicSplitStrategy를 사용해야 하지만,
    # 현재는 기존 portfolio_backtesting과 호환성을 위해 변환 로직 필요
    
    try:
        # TODO: 여기서 실제 MagicSplitStrategy를 사용하도록 수정 필요
        # 현재는 임시로 기존 함수 호출 (추후 GPU 구현 시 대체될 예정)
        pass  # 임시로 주석 처리
    except Exception as e:
        print(f'Error in backtesting for period {start_year}-{end_year}: {e}')
    mdd = calculate_mdd(portfolio_values_over_time)
    if save_files:
        plot_backtesting_results(all_trading_dates, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, num_splits, max_stocks, buy_threshold, cagr, mdd,results_folder,investment_ratio,per_threshold,pbr_threshold,div_threshold, save_files=save_files)
    return total_portfolio_value, cagr, mdd


def calculate_average_results(backtesting_results):
    total_values = np.array([result[0] for result in backtesting_results])
    cagr_values = np.array([result[1] for result in backtesting_results])
    mdd_values = np.array([result[2] for result in backtesting_results])

    average_total_value = np.mean(total_values)
    average_cagr = np.mean(cagr_values)
    average_mdd = np.mean(mdd_values)

    return average_total_value, average_cagr, average_mdd

def average_cagr_wrapper(params, time_periods, db_params, results_folder,save_files=False):
    max_stocks, order_investment_ratio, additional_buy_drop_rate, sell_profit_rate, additional_buy_priority = params
    period_results = []
    for start_year, end_year in time_periods:
        try:
            total_value, cagr, mdd = run_backtesting_for_period(initial_capital,
                max_stocks, order_investment_ratio, additional_buy_drop_rate, sell_profit_rate, additional_buy_priority, start_year, end_year, db_params, results_folder,save_files=save_files
            )
            period_results.append((total_value, cagr, mdd))
        except Exception as e:
            print(f'Error in backtesting for period {start_year}-{end_year}: {e}')
            period_results.append((0, None, None))

    average_results = calculate_average_results(period_results)
    return max_stocks, order_investment_ratio, additional_buy_drop_rate, sell_profit_rate, additional_buy_priority, average_results[0], average_results[1], average_results[2]

# def run_backtesting_and_save_results(save_files=False):
#     results = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [
#             executor.submit(average_cagr_wrapper, param, time_periods, db_params, results_folder,save_files)
#             for param in combinations
#         ]
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#             results.append(future.result())

#     results_df = pd.DataFrame(results, columns=[
#         "num_splits", "buy_threshold", "investment_ratio", "consider_delisting", "max_stocks",
#         "per_threshold", "pbr_threshold", "dividend_threshold", "min_additional_buy_drop_rate", "seed",
#         "Average_Total_Value", "Average_CAGR", "Average_MDD"
#     ])
#     results_df.to_csv(results_file, index=False)
#     print(results_df.sort_values(by="Average_CAGR", ascending=False))
    

def run_backtesting_and_save_results(save_files=False):
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(average_cagr_wrapper, param, time_periods, db_params, results_folder, save_files)
            for param in combinations
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f'Exception occurred: {e}')

    results_df = pd.DataFrame(results, columns=[
        "max_stocks", "order_investment_ratio", "additional_buy_drop_rate", "sell_profit_rate", "additional_buy_priority",
        "Average_Total_Value", "Average_CAGR", "Average_MDD"
    ])
    results_df.to_csv(results_file, index=False)
    print(results_df.sort_values(by="Average_CAGR", ascending=False))



if __name__ == "__main__":
    run_backtesting_and_save_results()