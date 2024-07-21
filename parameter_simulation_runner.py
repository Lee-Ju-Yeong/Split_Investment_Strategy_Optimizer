import os
import concurrent.futures
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import configparser

# 백테스팅과 관련된 함수들 가져오기
from MagicSplit_Backtesting_Optimizer import (
    portfolio_backtesting, calculate_mdd
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
investment_ratio_options = [0.3, 0.25]
consider_delisting_options = [False]
max_stocks_options = [40, 50, 55]

# 파라미터 조합 생성
combinations = [(n, b, i, c, m) for n in num_splits_options for b in buy_threshold_options for i in investment_ratio_options for c in consider_delisting_options for m in max_stocks_options]

# 여러 기간 설정
time_periods = [(2006, 2023), (2008, 2023), (2010, 2023), (2012, 2023), (2014, 2023)]

# 파일 저장 경로 설정
current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
results_folder = f'parameter_simulation_{current_time_str}'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 파라미터 조합 엑셀 파일 저장 경로
results_file = os.path.join(results_folder, f'parameter_combinations_{current_time_str}.csv')

# 백테스팅 실행 및 결과 저장
def backtesting_wrapper(params):
    num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks = params
    try:
        _, _, portfolio_values_over_time, _, _, _, _, cagr = portfolio_backtesting(
            initial_capital, num_splits, investment_ratio, buy_threshold, start_date, end_date, db_params, 
            per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, consider_delisting, max_stocks
        )
        mdd = calculate_mdd(portfolio_values_over_time)
        return num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks, cagr, mdd
    except Exception as e:
        print(f'Error in backtesting: {e}')
        return num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks, None, None

def run_backtesting_and_save_results():
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for param in combinations:
            futures.append(executor.submit(backtesting_wrapper, param))
        
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)

    results_df = pd.DataFrame(results, columns=["num_splits", "buy_threshold", "investment_ratio", "consider_delisting", "max_stocks", "CAGR", "MDD"])
    results_df.to_csv(results_file, index=False)
    print(results_df.sort_values(by="CAGR", ascending=False))

if __name__ == "__main__":
    run_backtesting_and_save_results()
