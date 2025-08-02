from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .portfolio import Position

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, current_date, portfolio, data_handler, end_date):
        raise NotImplementedError("generate_signals() 메소드를 구현해야 합니다.")

class MagicSplitStrategy(Strategy):
    def __init__(self, initial_capital, max_stocks, order_investment_ratio, 
                 additional_buy_drop_rate, sell_profit_rate, backtest_start_date, backtest_end_date,
                 additional_buy_priority='lowest_order', consider_delisting=False):
        self.initial_capital = initial_capital
        self.max_stocks = max_stocks
        self.order_investment_ratio = order_investment_ratio
        self.additional_buy_drop_rate = additional_buy_drop_rate
        self.sell_profit_rate = sell_profit_rate
        self.backtest_start_date = pd.to_datetime(backtest_start_date)
        self.backtest_end_date = pd.to_datetime(backtest_end_date)
        self.additional_buy_priority = additional_buy_priority
        self.consider_delisting = consider_delisting
        self.investment_per_order = 0
        self.previous_month = -1

    def _calculate_monthly_investment(self, current_date, portfolio, data_handler):
        current_month = current_date.month
        if current_month != self.previous_month:
            total_portfolio_value = portfolio.get_total_value(current_date, data_handler)
            self.investment_per_order = total_portfolio_value * self.order_investment_ratio
            self.previous_month = current_month

    def generate_signals(self, current_date, portfolio, data_handler):
        self._calculate_monthly_investment(current_date, portfolio, data_handler)

        sell_signals = []
        buy_signals = []

        existing_pos_signals = self._generate_signals_for_existing_positions(current_date, portfolio, data_handler)
        for s in existing_pos_signals:
            if s['type'] == 'SELL':
                sell_signals.append(s)
            else:
                buy_signals.append(s)

        if len(portfolio.positions) < self.max_stocks:
            new_entry_signals = self._generate_signals_for_new_entries(current_date, portfolio, data_handler)
            buy_signals.extend(new_entry_signals)
        
        buy_signals.sort(key=lambda s: (s['priority_group'], s['sort_metric']))
        
        all_signals = sell_signals + buy_signals
        for signal in all_signals:
            signal['start_date'] = self.backtest_start_date
            signal['end_date'] = self.backtest_end_date
            
        return all_signals

    def _generate_signals_for_existing_positions(self, current_date, portfolio, data_handler):
        signals = []
        for ticker in list(portfolio.positions.keys()):
            stock_data = data_handler.load_stock_data(ticker, self.backtest_start_date, self.backtest_end_date)
            if stock_data is None or stock_data.empty or current_date not in stock_data.index:
                continue

            row = stock_data.loc[current_date]
            positions = portfolio.positions[ticker]
            
            first_pos = next((p for p in positions if p.order == 1), None)

            # 1. 전체 매도 조건 (1차 매수 기준)
            if first_pos and row['close_price'] > first_pos.buy_price * (1 + self.sell_profit_rate):
                for p in positions:
                    signals.append({'date': current_date, 'ticker': ticker, 'type': 'SELL', 'quantity': p.quantity, 'position': p})
            # 2. 전체 매도 조건이 아닐 경우에만, 2차 이상 포지션의 부분 매도 확인
            else:
                positions_to_sell_partially = [p for p in positions if p.order > 1 and row['close_price'] >= p.buy_price * (1 + self.sell_profit_rate)]
                for p in positions_to_sell_partially:
                    signals.append({'date': current_date, 'ticker': ticker, 'type': 'SELL', 'quantity': p.quantity, 'position': p})
            
            # 3. 추가 매수 조건 (매도 조건과 별개로 항상 체크)
            last_pos = positions[-1] if positions else None
            if last_pos and row['close_price'] <= last_pos.buy_price * (1 - self.additional_buy_drop_rate):
                quantity = int(self.investment_per_order / row['close_price'])
                if quantity > 0:
                    new_pos = Position(row['close_price'], quantity, len(positions) + 1, self.additional_buy_drop_rate, self.sell_profit_rate)
                    
                    sort_metric = len(positions) if self.additional_buy_priority == 'lowest_order' else -( (last_pos.buy_price - row['close_price']) / last_pos.buy_price )
                        
                    signals.append({'date': current_date, 'ticker': ticker, 'type': 'BUY', 'quantity': quantity, 
                                    'position': new_pos, 'priority_group': 2, 'sort_metric': sort_metric})
        return signals

    def _generate_signals_for_new_entries(self, current_date, portfolio, data_handler):
        signals = []
        available_slots = self.max_stocks - len(portfolio.positions)
        if available_slots <= 0: return signals

        candidate_codes = data_handler.get_filtered_stock_codes(current_date)
        if not candidate_codes: return signals

        new_candidates = [code for code in candidate_codes if code not in portfolio.positions]
        candidate_atrs = []
        for ticker in new_candidates:
            stock_data = data_handler.load_stock_data(ticker, self.backtest_start_date, self.backtest_end_date)
            if stock_data is not None and not stock_data.empty and 'atr_14_ratio' in stock_data.columns and current_date in stock_data.index:
                latest_atr = stock_data.loc[current_date, 'atr_14_ratio']
                if pd.notna(latest_atr):
                    candidate_atrs.append({'ticker': ticker, 'atr_14_ratio': latest_atr})

        sorted_candidates = sorted(candidate_atrs, key=lambda x: x['atr_14_ratio'], reverse=True)
        
        for candidate in sorted_candidates:
            if len(signals) >= available_slots: break
            
            ticker = candidate['ticker']
            stock_data = data_handler.load_stock_data(ticker, self.backtest_start_date, self.backtest_end_date)
            row = stock_data.loc[current_date]
            if row['close_price'] > 0:
                quantity = int(self.investment_per_order / row['close_price'])
                if quantity > 0:
                    new_pos = Position(row['close_price'], quantity, 1, self.additional_buy_drop_rate, self.sell_profit_rate)
                    signals.append({'date': current_date, 'ticker': ticker, 'type': 'BUY', 'quantity': quantity, 
                                    'position': new_pos, 'priority_group': 1, 'sort_metric': -candidate['atr_14_ratio']})
        return signals
