import pandas as pd
import matplotlib.pyplot as plt
import os
import concurrent.futures
from tqdm import tqdm
import pymysql
import datetime
import random
import configparser
from cryptography.fernet import Fernet
import mysql.connector
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

class Position:
    def __init__(self, buy_price, quantity, order, additional_buy_drop_rate, sell_profit_rate):
        self.buy_price = buy_price
        self.quantity = quantity
        self.order = order
        self.additional_buy_drop_rate = additional_buy_drop_rate
        self.sell_profit_rate = sell_profit_rate

class Trade:
    def __init__(self, date, code, order, quantity, buy_price, sell_price, trade_type, profit, profit_rate, normalized_value, capital, total_portfolio_value):
        self.date = date
        self.code = code
        self.order = order
        self.quantity = quantity
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.trade_type = trade_type
        self.profit = profit
        self.profit_rate = profit_rate
        self.normalized_value = normalized_value
        self.capital = capital
        self.total_portfolio_value = total_portfolio_value

# Function to dynamically import conditions-fulfilling stock codes from the database
def get_stock_codes(date, per_threshold, pbr_threshold, div_threshold, buy_threshold, db_params, consider_delisting):
    conn = mysql.connector.connect(**db_params)
    cursor = conn.cursor()
    date_str = pd.to_datetime(date).strftime('%Y-%m-%d')

    if consider_delisting:
        query = f"""
        SELECT ticker 
        FROM stock_data 
        WHERE date = '{date_str}' 
        AND PER > 0 
        AND PER <= {per_threshold} 
        AND PBR > 0 
        AND PBR <= {pbr_threshold} 
        AND dividend >= {div_threshold}
        AND normalized_value <= {buy_threshold}
        """
    else:
        query = f"""
        SELECT sd.ticker 
        FROM stock_data sd
        LEFT JOIN (
            SELECT ticker, MAX(date) as last_date
            FROM stock_data
            GROUP BY ticker
        ) ld ON sd.ticker = ld.ticker
        WHERE sd.date = '{date_str}' 
        AND sd.PER > 0 
        AND sd.PER <= {per_threshold} 
        AND sd.PBR > 0 
        AND sd.PBR <= {pbr_threshold} 
        AND sd.dividend >= {div_threshold}
        AND sd.normalized_value <= {buy_threshold}
        AND ld.last_date >= '2024-06-01'
        """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=['ticker'])
    cursor.close()
    conn.close()
    return df['ticker'].tolist()



def load_stock_data_from_mysql(ticker, start_date, end_date, db_params):
    conn = mysql.connector.connect(**db_params)
    cursor = conn.cursor()

    # start_date 이전 5년의 데이터를 함께 가져오기 위해 5년 전 날짜 계산
    extended_start_date = pd.to_datetime(start_date) - timedelta(days=252*5)
    extended_start_date_str = extended_start_date.strftime('%Y-%m-%d')
    end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    query = f"""
        SELECT * FROM stock_data 
        WHERE ticker = '{ticker}' AND date BETWEEN '{extended_start_date_str}' AND '{end_date_str}'
    """
    cursor.execute(query)
    
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    
    # 날짜 형식 변환 및 인덱스 설정
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    cursor.close()
    conn.close()

    # 5년 동안의 최저값 계산
    df['five_year_low'] = df['close'].rolling(window=252*5, min_periods=1).min()
    
    # start_date 이후의 데이터만 반환
    return df[df.index >= pd.to_datetime(start_date)]

def calculate_additional_buy_drop_rate(last_buy_price, five_year_low, num_splits):
    return 1 - np.power((five_year_low / last_buy_price), (1 / (num_splits - 1)))

def calculate_sell_profit_rate(buy_profit_rate):
    """
    매수 수익률을 기반으로 매도 수익률을 계산
    :buy_profit_rate: 매수 기준 수익률
    :return: 계산된 매도 수익률
    """
    sell_profit_rate = (1 / (1 - buy_profit_rate)) - 1
    return sell_profit_rate 
    

# 추가된 트레이드 객체를 포함한 함수들
def initial_buy_sell(row, positions, capital, investment_per_split, buy_threshold, buy_signals, sell_signals, code, trading_history, total_portfolio_value,num_splits):
    normalized = row['normalized_value']
    five_year_low = row['five_year_low']
    current_order = len(positions) + 1
    buy_commission_rate = 0.00015
    sell_commission_rate = 0.00015
    sell_tax_rate = 0.0018
    additional_buy_drop_rate = 0
    sell_profit_rate = 0

    if normalized < buy_threshold and len(positions) == 0 and capital >= investment_per_split:
        additional_buy_drop_rate = calculate_additional_buy_drop_rate(row['close'], five_year_low, num_splits)
        sell_profit_rate = calculate_sell_profit_rate(additional_buy_drop_rate)
        quantity = int(investment_per_split / row['close'])
        total_cost = int(row['close'] * quantity * (1 + buy_commission_rate))
        new_position = Position(row['close'], quantity, 1, sell_profit_rate, additional_buy_drop_rate)
        positions.append(new_position)
        capital -= total_cost
        buy_signals.append((row.name, row['close']))

        trade = Trade(row.name, code, 1, quantity, row['close'], None, 'buy', None, None, normalized, capital, total_portfolio_value)
        trading_history.append(trade)

    liquidated = False

    for position in positions:
        if row['close'] > position.buy_price * (1 + position.sell_profit_rate) and position.order == 1:
            total_revenue = int(row['close'] * position.quantity * (1 - sell_commission_rate - sell_tax_rate))
            capital += total_revenue
            positions.remove(position)
            sell_signals.append((row.name, row['close']))
            profit = total_revenue - int(position.buy_price * position.quantity * (1 + buy_commission_rate))
            profit_rate = profit / (position.buy_price * position.quantity)
            trade = Trade(row.name, code, 1, position.quantity, position.buy_price, row['close'], 'sell', profit, profit_rate, normalized, capital, total_portfolio_value)
            trading_history.append(trade)
            liquidated = True

    if liquidated and not positions:
        return positions, capital, code
    else:
        return positions, capital, None

def additional_buy(row, positions, capital, investment_per_split, num_splits, buy_signals, trading_history, total_portfolio_value,code):
    buy_commission_rate = 0.00015
    if positions and len(positions) < num_splits and capital >= investment_per_split:
        last_position = positions[-1]
        if row['close'] <= last_position.buy_price * (1 - last_position.additional_buy_drop_rate):
            quantity = int(investment_per_split / row['close'])
            total_cost = int(row['close'] * quantity * (1 + buy_commission_rate))
            new_position = Position(row['close'], quantity, len(positions) + 1, last_position.sell_profit_rate, last_position.additional_buy_drop_rate)
            positions.append(new_position)
            capital -= total_cost
            buy_signals.append((row.name, row['close']))

            trade = Trade(row.name, code, len(positions), quantity, row['close'], None, 'buy', None, None, row['normalized_value'], capital, total_portfolio_value)
            trading_history.append(trade)

    return positions, capital

def additional_sell(row, positions, capital, sell_signals, trading_history, total_portfolio_value,code):
    sell_commission_rate = 0.00015
    sell_tax_rate = 0.0018
    buy_commission_rate = 0.00015

    for position in positions:
        if row['close'] >= position.buy_price * (1 + position.sell_profit_rate) and position.order > 1:
            total_revenue = int(row['close'] * position.quantity * (1 - sell_commission_rate - sell_tax_rate))
            capital += total_revenue
            positions.remove(position)
            sell_signals.append((row.name, row['close']))
            profit = total_revenue - int(position.buy_price * position.quantity * (1 + buy_commission_rate))
            profit_rate = profit / (position.buy_price * position.quantity)
            trade = Trade(row.name, code, position.order, position.quantity, position.buy_price, row['close'], 'sell', profit, profit_rate, row['normalized_value'], capital, total_portfolio_value)
            trading_history.append(trade)

    return positions, capital


def get_trading_dates_from_db(db_params, start_date, end_date):
    conn = mysql.connector.connect(**db_params)
    cursor = conn.cursor()
    query = f"SELECT DISTINCT date FROM stock_data WHERE date BETWEEN '{start_date}' AND '{end_date}' ORDER BY date"
    cursor.execute(query)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=['date'])
    
    cursor.close()
    conn.close()
    trading_dates = df['date'].tolist()
    return trading_dates

def calculate_mdd(portfolio_values):
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    mdd = np.max(drawdown)
    return mdd

def portfolio_backtesting(initial_capital, num_splits, investment_ratio, buy_threshold, start_date, 
                          end_date, db_params, per_threshold, pbr_threshold, div_threshold, 
                          min_additional_buy_drop_rate, consider_delisting, max_stocks=20):
    total_portfolio_value = initial_capital
    capital = initial_capital
    positions_dict = {}
    buy_signals = []
    sell_signals = []
    portfolio_values_over_time = []
    capital_over_time = []
    portfolio_values_over_time = np.array(portfolio_values_over_time)
    capital_over_time = np.array(capital_over_time)
    entered_stocks = set()
    current_orders_dict = {}
    loaded_stock_data = {}
    trading_history = []
    previous_month = None

    all_trading_dates = get_trading_dates_from_db(db_params, start_date, end_date)

    conn = mysql.connector.connect(**db_params)
    cursor = conn.cursor()
    query = """
        CREATE TEMPORARY TABLE delisted_stocks AS
        SELECT ticker, MAX(date) AS delisted_date
        FROM stock_data
        GROUP BY ticker
        HAVING MAX(date) < '2024-06-01'
    """
    cursor.execute(query)
    cursor.close()
    conn.close()

    for date_str in tqdm(all_trading_dates, desc="Backtesting progress"):
        date = pd.to_datetime(date_str)
        if consider_delisting:
            conn = mysql.connector.connect(**db_params)
            cursor = conn.cursor()
            query = """
                SELECT delisted_stocks.ticker
                FROM delisted_stocks
                WHERE delisted_stocks.delisted_date = %s
            """
            cursor.execute(query, (date_str,))
            delisted_stocks = cursor.fetchall()
            cursor.close()
            conn.close()
            for delisted_stock in delisted_stocks:
                ticker = delisted_stock[0]
                if ticker in positions_dict:
                    last_close_price = loaded_stock_data[ticker].loc[loaded_stock_data[ticker].index[-1], 'close']
                    for position in positions_dict[ticker]:
                        capital += position.quantity * last_close_price
                    del positions_dict[ticker]
                    entered_stocks.discard(ticker)
                    print('상폐로 entered_stocks 삭제 완료')
                    if ticker in current_orders_dict:
                        del current_orders_dict[ticker]
                        print('상폐로 current_orders_dict 삭제 완료')

        current_month = date.month
        if current_month != previous_month:
            investment_per_split = total_portfolio_value * investment_ratio // num_splits
            previous_month = current_month

        for code in list(entered_stocks):
            if date in loaded_stock_data[code].index:
                row = loaded_stock_data[code].loc[date]
                positions = positions_dict.get(code, [])
                positions, capital = additional_sell(row, positions, capital, sell_signals, trading_history, total_portfolio_value,code)
                positions_dict[code] = positions
                if positions:
                    current_order = max(position.order for position in positions)
                    current_orders_dict[code] = current_order
                else:
                    if code in current_orders_dict:
                        del current_orders_dict[code]

        for code in list(entered_stocks):
            if date in loaded_stock_data[code].index:
                row = loaded_stock_data[code].loc[date]
                positions = positions_dict.get(code, [])
                positions, capital, liquidated_code = initial_buy_sell(row, positions, capital, investment_per_split, buy_threshold, buy_signals, sell_signals, code, trading_history, total_portfolio_value,num_splits)
                positions_dict[code] = positions
                if liquidated_code:
                    entered_stocks.discard(liquidated_code)
                    if liquidated_code in current_orders_dict:
                        del current_orders_dict[liquidated_code]
                else:
                    if positions:
                        current_order = max(position.order for position in positions)
                        current_orders_dict[code] = current_order
                    else:
                        if code in current_orders_dict:
                            del current_orders_dict[code]

        if len(entered_stocks) < max_stocks and capital > investment_per_split:
            stock_codes = get_stock_codes(date, per_threshold, pbr_threshold, div_threshold, buy_threshold, db_params, consider_delisting)
            random.shuffle(stock_codes)
            for code in stock_codes:
                if len(entered_stocks) < max_stocks and code not in entered_stocks:
                    loaded_stock_data[code] = load_stock_data_from_mysql(code, start_date, end_date, db_params)
                    if date in loaded_stock_data[code].index:
                        sample_row = loaded_stock_data[code].loc[date]
                        five_year_low = sample_row['five_year_low']
                        last_buy_price = sample_row['close']
                        additional_buy_drop_rate = calculate_additional_buy_drop_rate(last_buy_price, five_year_low, num_splits)
                        if additional_buy_drop_rate >= min_additional_buy_drop_rate:
                            row = loaded_stock_data[code].loc[date]
                            positions = []
                            positions, capital, liquidated_code = initial_buy_sell(row, positions, capital, investment_per_split, buy_threshold, buy_signals, sell_signals, code, trading_history, total_portfolio_value,num_splits)
                            positions_dict[code] = positions
                            if positions:
                                entered_stocks.add(code)
                                current_order = max(position.order for position in positions)
                                current_orders_dict[code] = current_order
                            else:
                                if code in loaded_stock_data:
                                    del loaded_stock_data[code]

        for code in list(entered_stocks):
            if date in loaded_stock_data[code].index:
                row = loaded_stock_data[code].loc[date]
                positions = positions_dict.get(code, [])
                positions, capital = additional_buy(row, positions, capital, investment_per_split, num_splits, buy_signals, trading_history, total_portfolio_value,code)
                positions_dict[code] = positions
                if positions:
                    entered_stocks.add(code)
                    current_order = max(position.order for position in positions)
                    current_orders_dict[code] = current_order
                else:
                    if capital < investment_per_split:
                        entered_stocks.discard(code)

        current_stock_value = np.sum([
            position.quantity * loaded_stock_data[code].loc[date]['close']
            for code, positions in positions_dict.items()
            for position in positions
            if date in loaded_stock_data[code].index
        ])

        total_portfolio_value = np.sum([capital, current_stock_value])
        # Check for NaN values and handle them
        if np.isnan(total_portfolio_value):
            print(f"Warning: NaN detected in total_portfolio_value on {date_str}. Setting to previous value.")
            total_portfolio_value = portfolio_values_over_time[-1] if len(portfolio_values_over_time) > 0 else initial_capital
        
        portfolio_values_over_time = np.append(portfolio_values_over_time, total_portfolio_value)
        capital_over_time = np.append(capital_over_time, capital)

    total_days = (all_trading_dates[-1] - all_trading_dates[0]).days
    print("total_days",total_days)
    total_years = total_days / 365.25
    final_portfolio_value = portfolio_values_over_time[-1]
    cagr = np.power(final_portfolio_value / initial_capital, 1 / total_years) - 1
    print('backtesting complete')



    results_folder = 'results_of_single_test'
    # Create folder if it does not exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)


    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'trading_history_{num_splits}_{max_stocks}_{buy_threshold}_{current_time_str}.xlsx'
    file_path = os.path.join(results_folder, file_name)
    trading_history_df = pd.DataFrame([trade.__dict__ for trade in trading_history])
    trading_history_df.to_excel(file_path, index=False)
    print(f'Trading history saved to {file_name}')

    return positions_dict, total_portfolio_value, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, all_trading_dates, cagr



# 포트폴리오 가치 주석을 위한 함수
def format_currency(value):
    return f'{value:,.0f}₩'

# 백테스팅 결과 시각화 함수
def plot_backtesting_results(all_trading_dates, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, num_splits, max_stocks, buy_threshold, cagr, mdd):
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    
    sns.set(style="whitegrid")  # Set the seaborn style

    plt.figure(figsize=(14, 7))

    # 포트폴리오 가치 및 자본 그래프 그리기
    sns.lineplot(x=all_trading_dates, y=portfolio_values_over_time, label='Portfolio Value', color='blue')
    sns.lineplot(x=all_trading_dates, y=capital_over_time, label='Capital', color='green')

    # 매수 신호 표시
    if buy_signals:
        buy_dates, buy_prices = zip(*buy_signals)
        sns.scatterplot(x=buy_dates, y=buy_prices, marker='^', color='red', label='Buy Signal')

    # 매도 신호 표시
    if sell_signals:
        sell_dates, sell_prices = zip(*sell_signals)
        sns.scatterplot(x=sell_dates, y=sell_prices, marker='v', color='black', label='Sell Signal')

    # CAGR 및 MDD 텍스트 추가 (오른쪽 아래, 축 밖)
    plt.gca().text(1.01, -0.1, f'CAGR: {cagr:.2%}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    plt.gca().text(1.01, -0.15, f'MDD: {mdd:.2%}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    # 마지막 포트폴리오 가치 주석 달기
    last_date = all_trading_dates[-1]
    last_value = portfolio_values_over_time[-1]
    plt.annotate(format_currency(last_value), 
                 xy=(last_date, last_value), 
                 xytext=(last_date, last_value * 0.95),
                 arrowprops=dict(facecolor='blue', shrink=0.05),
                 fontsize=12,
                 color='blue')

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Portfolio Value and Capital Over Time')
    plt.legend()
    plt.grid(True)

    # Save plot as a PNG file
    results_folder = 'results_of_single_test'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'trading_history_{num_splits}_{max_stocks}_{buy_threshold}_{current_time_str}.png'
    file_path = os.path.join(results_folder, file_name)
    plt.savefig(file_path)
    print(f'Plot saved to {file_path}')
    plt.show()
            
def backtesting_wrapper(params):
    num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks = params
    try:
        _, _, portfolio_values_over_time, _, _, _, _, cagr = portfolio_backtesting(
            initial_capital, num_splits, investment_ratio, buy_threshold, start_date, end_date, db_params, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, consider_delisting, max_stocks
        )
        mdd = calculate_mdd(portfolio_values_over_time)
        return num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks, cagr, mdd
    except Exception as e:
        print(f'Error in backtesting: {e}')
        return num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks, None, None

def check_if_already_calculated(num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks):
    most_recent_file = get_most_recent_results_file()
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

def run_backtesting_for_period(num_splits, buy_threshold, investment_ratio, start_year, end_year, db_params, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, consider_delisting, max_stocks):
    start_date = f'{start_year}-01-01'
    end_date = f'{end_year}-12-31'
    _, total_portfolio_value, portfolio_values_over_time, _, _, _, _, cagr = portfolio_backtesting(
        100000000, num_splits, investment_ratio, buy_threshold, start_date, end_date, db_params, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, consider_delisting, max_stocks
    )
    mdd = calculate_mdd(portfolio_values_over_time)
    return total_portfolio_value, cagr, mdd

def calculate_average_results(backtesting_results):
    total_values = np.array([result[0] for result in backtesting_results])
    cagr_values = np.array([result[1] for result in backtesting_results])
    mdd_values = np.array([result[2] for result in backtesting_results])

    average_total_value = np.mean(total_values)
    average_cagr = np.mean(cagr_values)
    average_mdd = np.mean(mdd_values)

    return average_total_value, average_cagr, average_mdd

def average_cagr_wrapper(params):
    num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks = params
    period_results = []
    for start_year, end_year in time_periods:
        try:
            total_value, cagr, mdd = run_backtesting_for_period(
                num_splits, buy_threshold, investment_ratio, start_year, end_year, db_params, per_threshold, pbr_threshold, div_threshold, min_additional_buy_drop_rate, consider_delisting, max_stocks
            )
            period_results.append((total_value, cagr, mdd))
        except Exception as e:
            print(f'Error in backtesting for period {start_year}-{end_year}: {e}')
            period_results.append((0, None, None))

    average_results = calculate_average_results(period_results)
    return num_splits, buy_threshold, investment_ratio, consider_delisting, max_stocks, average_results[0], average_results[1], average_results[2]

# config = configparser.ConfigParser()
# config.read('config.ini')

# db_params = {
#     'host': config['mysql']['host'],
#     'user': config['mysql']['user'],
#     'password': config['mysql']['password'],
#     'database': config['mysql']['database'],
#     'connection_timeout': 60
# }

# random.seed(100)

# initial_capital = 100000000
# num_splits = 10
# investment_ratio = 0.1
# buy_threshold = 30

# start_date = '2004-01-01'
# end_date = '2024-01-01'

# per_threshold = 20
# pbr_threshold = 2
# div_threshold = 0.5
# min_additional_buy_drop_rate = 0.005
# consider_delisting = False
# max_stocks = 40

# # columns = ["num_splits", "buy_threshold", "investment_ratio", "consider_delisting", "max_stocks", "Average_Total_Value", "Average_CAGR", "Average_MDD"]

# # num_splits_options = [20]
# # buy_threshold_options = [30]
# # investment_ratio_options = [0.3, 0.25]
# # consider_delisting_options = [False]
# # max_stocks_options = [40, 50, 55]

# # combinations = [(n, b, i, c, m) for n in num_splits_options for b in buy_threshold_options for i in investment_ratio_options for c in consider_delisting_options for m in max_stocks_options]

# # time_periods = [(2006, 2023), (2008, 2023), (2010, 2023), (2012, 2023), (2014, 2023)]

# # current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
# # results_file = f'average_backtesting_results_{current_time_str}.csv'

# # with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
# #     futures = []
# #     for param in combinations:
# #         if not check_if_already_calculated(param[0], param[1], param[2], param[3], param[4]):
# #             futures.append(executor.submit(average_cagr_wrapper, param))
# #     results = []
# #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
# #         result = future.result()
# #         results.append(result)

# # results_df = pd.DataFrame(results, columns=columns)
# # results_df.to_csv(results_file, index=False)

# # print(results_df.sort_values(by="Average_CAGR", ascending=False))
