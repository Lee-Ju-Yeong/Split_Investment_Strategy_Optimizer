import pandas as pd
import numpy as np

class Position:
    """
    하나의 매수 포지션 정보를 표현합니다.
    """
    def __init__(self, buy_price, quantity, order, additional_buy_drop_rate, sell_profit_rate):
        self.buy_price = buy_price  # 매수 가격
        self.quantity = quantity  # 매수 수량
        self.order = order  # 매수 순서 (1차, 2차, ...)
        self.additional_buy_drop_rate = additional_buy_drop_rate  # 추가 매수 하락률
        self.sell_profit_rate = sell_profit_rate  # 매도 목표 수익률

class Trade:
    """
    발생한 단일 거래 정보를 기록합니다.
    """
    def __init__(self, date, code, order, quantity, buy_price, sell_price, trade_type, profit, profit_rate, normalized_value, capital, total_portfolio_value):
        self.date = date  # 거래 날짜
        self.code = code  # 종목 코드
        self.order = order  # 거래 순서
        self.quantity = quantity  # 거래 수량
        self.buy_price = buy_price  # 매수 가격
        self.sell_price = sell_price  # 매도 가격
        self.trade_type = trade_type  # 거래 유형 ('buy' 또는 'sell')
        self.profit = profit  # 실현 손익
        self.profit_rate = profit_rate  # 실현 수익률
        self.normalized_value = normalized_value # 거래 시점의 정규화된 가치
        self.capital = capital  # 거래 후 현금 잔고
        self.total_portfolio_value = total_portfolio_value  # 거래 후 총 포트폴리오 가치

class Portfolio:
    """
    포트폴리오의 전체 상태(현금, 포지션, 거래내역 등)를 관리하고,
    가치 평가 및 상태 업데이트를 수행합니다.
    """
    def __init__(self, initial_cash):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # key: ticker, value: list of Position objects
        self.trade_history = [] # list of Trade objects
        self.daily_value_history = [] # 일별 포트폴리오 가치 기록

    def update_cash(self, amount):
        """
        현금을 증감시킵니다.
        매도 시 amount는 양수, 매수 시 amount는 음수입니다.
        """
        self.cash += amount

    def add_position(self, ticker, position: Position):
        """
        특정 종목의 신규 포지션을 추가합니다.
        """
        if ticker not in self.positions:
            self.positions[ticker] = []
        self.positions[ticker].append(position)
        self.positions[ticker].sort(key=lambda p: p.order) # 항상 순서대로 정렬

    def remove_position(self, ticker, position_to_remove: Position):
        """
        매도된 포지션을 제거합니다.
        """
        if ticker in self.positions:
            try:
                self.positions[ticker].remove(position_to_remove)
                # 리스트가 비면 딕셔너리에서 해당 종목 키를 삭제
                if not self.positions[ticker]:
                    del self.positions[ticker]
            except ValueError:
                print(f"Warning: 제거하려는 포지션을 {ticker} 종목에서 찾을 수 없습니다.")

    def get_total_value(self, current_date, data_handler):
        """
        특정 날짜 기준 총 포트폴리오 가치를 계산합니다.
        (현금 + 모든 주식의 현재 평가액)
        """
        total_market_value = 0
        for ticker, positions_list in self.positions.items():
            current_price = data_handler.get_latest_price(current_date, ticker)
            if current_price is not None and not np.isnan(current_price):
                for position in positions_list:
                    total_market_value += position.quantity * current_price
        
        return self.cash + total_market_value

    def record_trade(self, trade: Trade):
        """
        거래 내역을 기록합니다.
        """
        self.trade_history.append(trade)

    def record_daily_value(self, date, value):
        """
        일별 포트폴리오 가치를 기록합니다.
        """
        self.daily_value_history.append({'date': date, 'value': value})
