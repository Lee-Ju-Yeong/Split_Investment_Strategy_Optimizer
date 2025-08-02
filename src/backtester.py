"""
backtester.py

This module contains the backtesting engine for the Magic Split Strategy.
"""

from tqdm import tqdm
import pandas as pd

class BacktestEngine:
    def __init__(self, start_date, end_date, portfolio, strategy, data_handler, execution_handler):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.portfolio = portfolio
        self.strategy = strategy
        self.data_handler = data_handler
        self.execution_handler = execution_handler

    def run(self):
        print("백테스팅 엔진을 시작합니다...")
        
        trading_dates = self.data_handler.get_trading_dates(self.start_date, self.end_date)
        trading_dates = pd.to_datetime(trading_dates)

        for current_date in tqdm(trading_dates, desc="Backtesting Progress"):
            signal_events = self.strategy.generate_signals(current_date, self.portfolio, self.data_handler)
            
            if signal_events:
                for signal in signal_events:
                    self.execution_handler.execute_order(signal, self.portfolio, self.data_handler)

            total_value = self.portfolio.get_total_value(current_date, self.data_handler)
            self.portfolio.record_daily_value(current_date, total_value)

        print("백테스팅이 완료되었습니다.")
        return self.portfolio
