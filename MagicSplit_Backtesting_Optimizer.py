import matplotlib
matplotlib.use('Agg')

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

# Position class to store buy details
class Position:
    def __init__(self, buy_price, quantity, order, additional_buy_drop_rate, sell_profit_rate):
        self.buy_price = buy_price
        self.quantity = quantity
        self.order = order
        self.additional_buy_drop_rate = additional_buy_drop_rate
        self.sell_profit_rate = sell_profit_rate

# Trade class to store trade details
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
    
    # Convert date format and set index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    cursor.close()
    conn.close()

    # Calculate the lowest value in 5 years
    df['five_year_low'] = df['close'].rolling(window=252*5, min_periods=1).min()
    
    # Return data after the start date
    return df[df.index >= pd.to_datetime(start_date)]


def calculate_additional_buy_drop_rate(last_buy_price, five_year_low, num_splits):
    return 1 - np.power((five_year_low / last_buy_price), (1 / (num_splits - 1)))

def calculate_sell_profit_rate(buy_profit_rate):
    sell_profit_rate = (1 / (1 - buy_profit_rate)) - 1
    return sell_profit_rate 
    

# Function to handle initial buy and sell
def initial_buy_sell(row, positions, capital, investment_per_split, buy_threshold, buy_signals, sell_signals, code, trading_history, total_portfolio_value,num_splits,save_files):
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
        new_position = Position(row['close'], quantity, 1, additional_buy_drop_rate,sell_profit_rate)
        positions.append(new_position)
        capital -= total_cost
        buy_signals.append((row.name, row['close']))
        if save_files:
            trade = Trade(row.name, code, 1, quantity, row['close'], None, 'buy', None, None, normalized, capital, total_portfolio_value)
            trading_history.append(trade)

    liquidated = False
    
     # Sell from highest order
    positions = sorted(positions, key=lambda x: x.order, reverse=True)
    positions_to_remove = []

    for position in positions:
        if row['close'] > position.buy_price * (1 + position.sell_profit_rate) and position.order == 1:
            total_revenue = int(row['close'] * position.quantity * (1 - sell_commission_rate - sell_tax_rate))
            capital += total_revenue
            positions_to_remove.append(position)
            sell_signals.append((row.name, row['close']))
            profit = total_revenue - int(position.buy_price * position.quantity * (1 + buy_commission_rate))
            profit_rate = profit / (position.buy_price * position.quantity)
            if save_files:
                trade = Trade(row.name, code, 1, position.quantity, position.buy_price, row['close'], 'sell', profit, profit_rate, normalized, capital, total_portfolio_value)
                trading_history.append(trade)
            liquidated = True

    for position in positions_to_remove:
        positions.remove(position)
    if liquidated and not positions:
        return positions, capital, code
    else:
        return positions, capital, None
    
# Function to handle additional buy
def additional_buy(row, positions, capital, investment_per_split, num_splits, buy_signals, trading_history, total_portfolio_value,code,save_files):
    buy_commission_rate = 0.00015
    if positions and len(positions) < num_splits and capital >= investment_per_split:
        last_position = positions[0]
        # print(f"Evaluating additional buy for code {code}: current close = {row['close']}, last buy price = {last_position.buy_price}, order = {last_position.order}, required drop = {last_position.additional_buy_drop_rate}")
        # print(f"Current positions for code {code}: {[p.buy_price for p in positions]}")     
        if row['close'] <= last_position.buy_price * (1 - last_position.additional_buy_drop_rate):
            # print(f"Additional buy condition met for code {code}")
            quantity = int(investment_per_split / row['close'])
            total_cost = int(row['close'] * quantity * (1 + buy_commission_rate))
            new_position = Position(row['close'], quantity, len(positions) + 1, last_position.additional_buy_drop_rate, last_position.sell_profit_rate)
            positions.append(new_position)
            capital -= total_cost
            buy_signals.append((row.name, row['close']))
            if save_files:
                trade = Trade(row.name, code, len(positions), quantity, row['close'], None, 'buy', None, None, row['normalized_value'], capital, total_portfolio_value)
                trading_history.append(trade)
            
            #  # 로그 추가
            # print(f"Updated positions for code {code}: new position buy price = {new_position.buy_price}, additional buy drop rate = {new_position.additional_buy_drop_rate}, total positions = {len(positions)}")

    return positions, capital

# Function to handle additional sell
def additional_sell(row, positions, capital, sell_signals, trading_history, total_portfolio_value,code,save_files):
    sell_commission_rate = 0.00015
    sell_tax_rate = 0.0018
    buy_commission_rate = 0.00015
    
    # Sort positions by order in descending order to sell the most recent buys first
    positions = sorted(positions, key=lambda x: x.order, reverse=True)
    positions_to_remove = []
    
    # Loop through positions to check if they meet the selling conditions
    for position in positions:
        if row['close'] >= position.buy_price * (1 + position.sell_profit_rate) and position.order > 1:
            total_revenue = int(row['close'] * position.quantity * (1 - sell_commission_rate - sell_tax_rate))
            capital += total_revenue
            positions_to_remove.append(position)
            sell_signals.append((row.name, row['close']))
            profit = total_revenue - int(position.buy_price * position.quantity * (1 + buy_commission_rate))
            profit_rate = profit / (position.buy_price * position.quantity)
            if save_files:
                trade = Trade(row.name, code, position.order, position.quantity, position.buy_price, row['close'], 'sell', profit, profit_rate, row['normalized_value'], capital, total_portfolio_value)
                trading_history.append(trade)
    for position in positions_to_remove:
        positions.remove(position)

    return positions, capital


# Function to get trading dates from the database
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

def portfolio_backtesting(seed,initial_capital, num_splits, investment_ratio, buy_threshold, start_date, 
                          end_date, db_params, per_threshold, pbr_threshold, div_threshold, 
                          min_additional_buy_drop_rate, consider_delisting, max_stocks=20,results_folder=None,save_files=True):
    random.seed(seed)
    np.random.seed(seed)
    total_portfolio_value = initial_capital
    capital = initial_capital
    positions_dict = {}
    buy_signals = []
    sell_signals = []
    portfolio_values_over_time = []
    capital_over_time = []
    portfolio_values_over_time = np.array(portfolio_values_over_time)
    capital_over_time = np.array(capital_over_time)
    
    entered_stocks = []
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
                    entered_stocks.remove(ticker)
                    print('상폐로 entered_stocks 삭제 완료')
                    if ticker in current_orders_dict:
                        del current_orders_dict[ticker]
                        print('상폐로 current_orders_dict 삭제 완료')

        current_month = date.month
        if current_month != previous_month:
            investment_per_split = total_portfolio_value * investment_ratio // num_splits
            previous_month = current_month

        for code in entered_stocks:
            if date in loaded_stock_data[code].index:
                row = loaded_stock_data[code].loc[date]
                positions = positions_dict.get(code, [])
                positions, capital = additional_sell(row, positions, capital, sell_signals, trading_history, total_portfolio_value,code,save_files)
                positions_dict[code] = positions
                if positions:
                    current_order = max(position.order for position in positions)
                    current_orders_dict[code] = current_order
                    # print(f"date {date} Updated positions_dict and entered_stocks for code {code}. Current positions: {[p.buy_price for p in positions]}")
                else:
                    if code in current_orders_dict:
                        del current_orders_dict[code]
                        # print(f"Removed code {code} from entered_stocks")        
                        
# Process initial buy or sell actions for each stock code in entered_stocks
        for code in entered_stocks:
            if date in loaded_stock_data[code].index:
                row = loaded_stock_data[code].loc[date]
                positions = positions_dict.get(code, [])
                positions, capital, liquidated_code = initial_buy_sell(row, positions, capital, investment_per_split, buy_threshold, buy_signals, sell_signals, code, trading_history, total_portfolio_value,num_splits,save_files)
                positions_dict[code] = positions
                if liquidated_code:
                    entered_stocks.remove(liquidated_code)
                    if liquidated_code in current_orders_dict:
                        del current_orders_dict[liquidated_code]
                else:
                    if positions:
                        current_order = max(position.order for position in positions)
                        current_orders_dict[code] = current_order
                     

                    else:
                        if code in current_orders_dict:
                            del current_orders_dict[code]
# Enter new stocks if there is room in the portfolio and sufficient capital
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
                            positions, capital, liquidated_code = initial_buy_sell(row, positions, capital, investment_per_split, buy_threshold, buy_signals, sell_signals, code, trading_history, total_portfolio_value,num_splits,save_files)
                            positions_dict[code] = positions
                            if positions:
                                if code not in entered_stocks: 
                                    entered_stocks.insert(0, code)
                                current_order = max(position.order for position in positions)
                                current_orders_dict[code] = current_order
                            else:
                                if code in loaded_stock_data:
                                    del loaded_stock_data[code]

        for code in entered_stocks:
            if date in loaded_stock_data[code].index:
                row = loaded_stock_data[code].loc[date]
                positions = positions_dict.get(code, [])
                positions, capital = additional_buy(row, positions, capital, investment_per_split, num_splits, buy_signals, trading_history, total_portfolio_value,code,save_files)
                positions_dict[code] = positions
                if positions:
                    if code not in entered_stocks: 
                        entered_stocks.insert(0, code)
                    current_order = max(position.order for position in positions)
                    current_orders_dict[code] = current_order
                else:
                    if capital < investment_per_split:
                        entered_stocks.remove(code)
                        print(f"Removed code {code} from entered_stocks due to insufficient capital.")

        current_stock_value = np.sum([
            position.quantity * loaded_stock_data[code].loc[date]['close']
            for code, positions in positions_dict.items()
            for position in positions
            if date in loaded_stock_data[code].index
        ])

        total_portfolio_value = np.sum([capital, current_stock_value])
        # Check for NaN values and handle them
        if np.isnan(total_portfolio_value):
            # print(f"Warning: NaN detected in total_portfolio_value on {date_str}. Setting to previous value.")
            total_portfolio_value = portfolio_values_over_time[-1] if len(portfolio_values_over_time) > 0 else initial_capital
        
        portfolio_values_over_time = np.append(portfolio_values_over_time, total_portfolio_value)
        capital_over_time = np.append(capital_over_time, capital)

    total_days = (all_trading_dates[-1] - all_trading_dates[0]).days
    total_years = total_days / 365.25
    final_portfolio_value = portfolio_values_over_time[-1]
    cagr = np.power(final_portfolio_value / initial_capital, 1 / total_years) - 1
    # print('backtesting complete')


    if results_folder and save_files:
        # Create folder if it does not exist
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        current_time_str = datetime.now().strftime('%Y%m%d_%H%M')
        file_name = f'trading_history_{num_splits}_{max_stocks}_{buy_threshold}_{current_time_str}.xlsx'
        file_path = os.path.join(results_folder, file_name)
        trading_history_df = pd.DataFrame([trade.__dict__ for trade in trading_history])
        trading_history_df.to_excel(file_path, index=False)
        # print(f'Trading history saved to {file_name}')

    return positions_dict, total_portfolio_value, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, all_trading_dates, cagr



# 포트폴리오 가치 주석을 위한 함수
def format_currency(value):
    return f'{value:,.0f}₩'

# 백테스팅 결과 시각화 함수
def plot_backtesting_results(all_trading_dates, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, num_splits, max_stocks, buy_threshold, cagr, mdd,results_folder, save_files=True):
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    
    sns.set(style="whitegrid")  # Set the seaborn style

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
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save plot as a PNG file
    if save_files:
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

    current_time_str = datetime.now().strftime('%Y%m%d_%H%M')
    file_name = f'trading_history_{num_splits}_{max_stocks}_{buy_threshold}_{current_time_str}.png'
    file_path = os.path.join(results_folder, file_name)
    plt.savefig(file_path)
    # print(f'Plot saved to {file_path}')
    plt.close() 
            
