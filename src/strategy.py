from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .portfolio import Position

class Strategy(ABC):
    """
    모든 매매 전략 클래스가 상속해야 하는 추상 기본 클래스(인터페이스)입니다.
    """
    @abstractmethod
    def generate_signals(self, current_date, portfolio, data_handler):
        raise NotImplementedError("generate_signals() 메소드를 구현해야 합니다.")

class MagicSplitStrategy(Strategy):
    """
    분할 매수/매도 로직을 구현한 구체적인 전략 클래스입니다.
    """
    def __init__(self, initial_capital, num_splits, investment_ratio, max_stocks, consider_delisting=False):
        self.initial_capital = initial_capital
        self.num_splits = num_splits
        self.investment_ratio = investment_ratio
        self.max_stocks = max_stocks
        self.consider_delisting = consider_delisting
        
        self.investment_per_stock = 0
        self.previous_month = -1

    def _calculate_monthly_investment(self, current_date, portfolio, data_handler):
        """월별 총 투자 가능 금액 및 종목당 투자금을 계산합니다."""
        current_month = current_date.month
        if current_month != self.previous_month:
            total_portfolio_value = portfolio.get_total_value(current_date, data_handler)
            available_investment = total_portfolio_value * self.investment_ratio
            self.investment_per_stock = available_investment / self.max_stocks if self.max_stocks > 0 else 0
            self.previous_month = current_month

    def generate_signals(self, current_date, portfolio, data_handler):
        signals = []
        self._calculate_monthly_investment(current_date, portfolio, data_handler)

        # 1. 기존 포지션에 대한 매도 및 추가매수 신호 생성
        self._generate_signals_for_existing_positions(current_date, portfolio, data_handler, signals)

        # 2. 신규 종목 진입을 위한 매수 신호 생성
        self._generate_signals_for_new_entries(current_date, portfolio, data_handler, signals)
        
        return signals

    def _generate_signals_for_existing_positions(self, current_date, portfolio, data_handler, signals):
        """보유 중인 종목에 대해 매도 또는 추가 매수 신호를 생성합니다."""
        for ticker in list(portfolio.positions.keys()):
            stock_data = data_handler.load_stock_data(ticker, '1980-01-01', current_date.strftime('%Y-%m-%d'))
            
            if stock_data is None or stock_data.empty or current_date not in stock_data.index:
                continue

            row = stock_data.loc[current_date]
            positions = portfolio.positions[ticker]
            
            positions_to_sell_additionally = [p for p in positions if row['close_price'] >= p.buy_price * (1 + p.sell_profit_rate) and p.order > 1]
            for p in positions_to_sell_additionally:
                signals.append({'date': current_date, 'ticker': ticker, 'type': 'SELL', 'quantity': p.quantity, 'position': p})

            first_position = next((p for p in positions if p.order == 1), None)
            if first_position and row['close_price'] > first_position.buy_price * (1 + first_position.sell_profit_rate):
                 for p in positions:
                    signals.append({'date': current_date, 'ticker': ticker, 'type': 'SELL', 'quantity': p.quantity, 'position': p})

            last_position = positions[-1] if positions else None
            if last_position and len(positions) < self.num_splits and row['close_price'] <= last_position.buy_price * (1 - last_position.additional_buy_drop_rate):
                investment_per_split = self.investment_per_stock / self.num_splits
                quantity = int(investment_per_split / row['close_price'])
                if quantity > 0:
                    new_pos = Position(row['close_price'], quantity, len(positions) + 1, last_position.additional_buy_drop_rate, last_position.sell_profit_rate)
                    signals.append({'date': current_date, 'ticker': ticker, 'type': 'BUY', 'quantity': quantity, 'position': new_pos})

    def _generate_signals_for_new_entries(self, current_date, portfolio, data_handler, signals):
        """신규로 진입할 종목을 찾아 매수 신호를 생성합니다."""
        if len(portfolio.positions) < self.max_stocks:
            candidate_codes = data_handler.get_filtered_stock_codes(current_date)
            new_candidates = [code for code in candidate_codes if code not in portfolio.positions]
            
            candidate_atrs = []
            for ticker in new_candidates:
                stock_data = data_handler.load_stock_data(ticker, pd.to_datetime('1980-01-01'), current_date)
                if stock_data is not None and not stock_data.empty and 'atr_14_ratio' in stock_data.columns:
                    mask = stock_data.index <= pd.to_datetime(current_date)
                    latest_atr_ratio = stock_data.loc[mask, 'atr_14_ratio'].iloc[-1] if mask.any() else None
                    if pd.notna(latest_atr_ratio):
                        candidate_atrs.append({'ticker': ticker, 'atr_14_ratio': latest_atr_ratio})

            sorted_candidates = sorted(candidate_atrs, key=lambda x: x['atr_14_ratio'], reverse=True)
            
            for candidate in sorted_candidates:
                if len(portfolio.positions) >= self.max_stocks:
                    break
                
                ticker = candidate['ticker']
                stock_data = data_handler.load_stock_data(ticker, pd.to_datetime('1980-01-01'), current_date)
                if stock_data is None or stock_data.empty or current_date not in stock_data.index:
                    continue
                
                row = stock_data.loc[current_date]
                investment_per_split = self.investment_per_stock / self.num_splits
                quantity = int(investment_per_split / row['close_price'])
                
                if quantity > 0:
                    five_year_low = stock_data['close_price'].rolling(window=252*5, min_periods=1).min().iloc[-1]
                    
                    add_buy_rate = 0
                    if self.num_splits > 1 and row['close_price'] > 0 and five_year_low > 0:
                         add_buy_rate = 1 - np.power((five_year_low / row['close_price']), (1 / (self.num_splits - 1)))

                    sell_profit_rate = float('inf')
                    if add_buy_rate > 0 and add_buy_rate < 1:
                        sell_profit_rate = (1 / (1 - add_buy_rate)) - 1
                    
                    new_pos = Position(row['close_price'], quantity, 1, add_buy_rate, sell_profit_rate)
                    signals.append({'date': current_date, 'ticker': ticker, 'type': 'BUY', 'quantity': quantity, 'position': new_pos})
