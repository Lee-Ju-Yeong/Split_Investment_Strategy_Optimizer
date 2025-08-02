import pandas as pd
import numpy as np

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

class Portfolio:
    def __init__(self, initial_cash, start_date, end_date):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.start_date = start_date
        self.end_date = end_date
        self.positions = {}
        self.trade_history = []
        self.daily_value_history = []

    def update_cash(self, amount):
        self.cash += amount

    def add_position(self, ticker, position: Position):
        if ticker not in self.positions:
            self.positions[ticker] = []
        self.positions[ticker].append(position)
        self.positions[ticker].sort(key=lambda p: p.order)

    def remove_position(self, ticker, position_to_remove: Position):
        if ticker in self.positions:
            try:
                self.positions[ticker].remove(position_to_remove)
                if not self.positions[ticker]:
                    del self.positions[ticker]
            except ValueError:
                print(f"Warning: 제거하려는 포지션을 {ticker} 종목에서 찾을 수 없습니다.")

    def get_total_value(self, current_date, data_handler):
        total_market_value = 0
        for ticker, positions_list in self.positions.items():
            current_price = data_handler.get_latest_price(current_date, ticker, self.start_date, self.end_date)
            if current_price is not None and not np.isnan(current_price):
                for position in positions_list:
                    total_market_value += position.quantity * current_price
        
        return self.cash + total_market_value

    def record_trade(self, trade: Trade):
        self.trade_history.append(trade)

    def record_daily_value(self, date, value):
        self.daily_value_history.append({'date': date, 'value': value})
