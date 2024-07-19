import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import mysql.connector
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import configparser

# 백테스팅과 관련된 함수들 가져오기
from MagicSplit_Backtesting_Optimizer import (
    get_stock_codes, load_stock_data_from_mysql, calculate_additional_buy_drop_rate,
    calculate_sell_profit_rate, initial_buy_sell, additional_buy, additional_sell,
    get_trading_dates_from_db, portfolio_backtesting, calculate_mdd, plot_backtesting_results,
    analyze_backtesting_results
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
    'connection_timeout': 60
}

initial_capital = 100000000  # 초기 자본 1억

def single_backtesting(num_splits, buy_threshold, investment_ratio, start_date, end_date, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, consider_delisting, max_stocks):
    positions_dict, total_portfolio_value, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, all_trading_dates, cagr = portfolio_backtesting(
        initial_capital, num_splits, investment_ratio, buy_threshold, start_date, end_date, db_params, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, consider_delisting, max_stocks
    )
    mdd = calculate_mdd(portfolio_values_over_time)
    plot_backtesting_results(all_trading_dates, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals)
    analyze_backtesting_results(positions_dict, total_portfolio_value, cagr, mdd)
    return positions_dict, total_portfolio_value, cagr, mdd

if __name__ == "__main__":
    print("Starting backtesting...")
    num_splits = 20
    buy_threshold = 30
    investment_ratio = 0.3
    start_date = '2004-01-01'
    end_date = '2024-01-01'
    per_threshold = 20
    pbr_threshold = 2
    div_threshold = 1.0
    min_additional_buy_drop_rate = 0.005
    consider_delisting = False
    max_stocks = 20

    single_backtesting(num_splits, buy_threshold, investment_ratio, start_date, end_date, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, consider_delisting, max_stocks)
    print("Backtesting completed")