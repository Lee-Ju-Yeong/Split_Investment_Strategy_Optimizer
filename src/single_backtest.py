import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import random
import mysql.connector
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import configparser
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

# 백테스팅과 관련된 함수들 가져오기
from backtest_strategy import (
    get_stock_codes, load_stock_data_from_mysql, calculate_additional_buy_drop_rate,
    calculate_sell_profit_rate, additional_buy, additional_sell,
    get_trading_dates_from_db, portfolio_backtesting, calculate_mdd, plot_backtesting_results, get_stock_codes, select_stocks
)

# Read Settings File
config = configparser.ConfigParser()
config.read('config.ini')

# Set up database connection information
db_params = {
    'host': config['mysql']['host'],
    'user': config['mysql']['user'],
    'password': config['mysql']['password'],
    'database': config['mysql']['database'],

}


initial_capital = 100000000  # 초기 자본 1억
# 랜덤 시드 고정
seed=101
results_folder="results_of_single_test"

def single_backtesting(seed,num_splits, buy_threshold, investment_ratio, start_date, end_date, per_threshold, pbr_threshold, div_threshold, normalized_atr_threshold, consider_delisting, max_stocks,save_files=True):
    random.seed(seed)
    np.random.seed(seed)
    positions_dict, total_portfolio_value, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, all_trading_dates, cagr = portfolio_backtesting(seed,
        initial_capital, num_splits, investment_ratio, buy_threshold, start_date, end_date, db_params, per_threshold, pbr_threshold, div_threshold,normalized_atr_threshold, consider_delisting, max_stocks,results_folder,save_files
    )
    mdd = calculate_mdd(portfolio_values_over_time)
    plot_backtesting_results(all_trading_dates, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, num_splits, max_stocks, buy_threshold, cagr, mdd,results_folder,investment_ratio,per_threshold,pbr_threshold,div_threshold, save_files)
    return  positions_dict, total_portfolio_value, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, all_trading_dates, cagr ,mdd




if __name__ == "__main__":
    print("Starting backtesting...")
    num_splits = 20 
    buy_threshold = 32
    investment_ratio = 9.6  # 9.6
    start_date = '2010-01-01'
    end_date = '2024-08-31'
    
    per_threshold = 10
    pbr_threshold = 1.0
    div_threshold = 1
    # min_additional_buy_drop_rate = 0.005
    consider_delisting = False
    max_stocks = 24
    normalized_atr_threshold = 4
    seed=107 
    
    # date = datetime.strptime('2024-09-27', '%Y-%m-%d')
    # entered_stocks = []  # 현재 포트폴리오에 포함된 종목 리스트
    # loaded_stock_data = {}  # 이미 로드된 종목의 데이터 저장용 딕셔너리
    # # 종목 코드 가져오기
    # stock_codes = get_stock_codes(date, per_threshold, pbr_threshold, div_threshold, buy_threshold, db_params, consider_delisting)
    # # 종목 선정 방식 선택 (normalized_atr, correlation, rank_based, random,roc)
    # stock_selection_method = 'normalized_atr'  # 원하는 방식 선택 백테스팅은 안쪽에서 다시 따로  해줘야햄
    # sorted_stock_codes = select_stocks(stock_selection_method, stock_codes, date, db_params, entered_stocks, loaded_stock_data)

    # print(date,"기준 선정된 종목 리스트:", sorted_stock_codes)
    
    
    
    

    positions_dict, total_portfolio_value, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, all_trading_dates, cagr,mdd= single_backtesting(seed,num_splits, buy_threshold, investment_ratio, start_date, end_date, per_threshold, pbr_threshold, div_threshold, normalized_atr_threshold, consider_delisting, max_stocks)
    print(f"최종 포트폴리오 가치: {total_portfolio_value}")
    print(f"CAGR: {cagr}")
    print(f"MDD: {mdd:.2%}")
    print("Backtesting completed")