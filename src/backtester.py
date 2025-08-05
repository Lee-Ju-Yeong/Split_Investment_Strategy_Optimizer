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

        # for current_date in tqdm(trading_dates, desc="Backtesting Progress"):
        for i, current_date in enumerate(tqdm(trading_dates, desc="Backtesting Progress")): # tqdm을 enumerate와 함께 사용
            signal_events = self.strategy.generate_signals(current_date, self.portfolio, self.data_handler)
            # --- [DEBUG] 루프 시작 시점 로그 (GPU와 형식 통일) ---
            capital_before_day = self.portfolio.cash
            total_positions_before_day = sum(len(p) for p in self.portfolio.positions.values())
            print(f"\n--- Day {i+1}/{len(trading_dates)}: {current_date.strftime('%Y-%m-%d')} ---")
            print(f"[BEGIN] Capital: {capital_before_day:,.0f} | Total Positions: {total_positions_before_day}")
            # 월 변경 시 리밸런싱 로그 (strategy.py 에서 출력되지만 여기서도 확인 가능)
            if hasattr(self.strategy, 'previous_month') and current_date.month != self.strategy.previous_month:
                # 실제 investment_per_order 값은 strategy 객체에서 가져와야 함
                print(f"  [REBALANCE] Month changed to {current_date.month}. New Investment/Order: {self.strategy.investment_per_order:,.0f}")

            signal_events = self.strategy.generate_signals(current_date, self.portfolio, self.data_handler)
            
            # --- [DEBUG] 신호 이벤트 로그 ---
            if signal_events:
                for signal in signal_events:
                    self.execution_handler.execute_order(signal, self.portfolio, self.data_handler)

            total_value = self.portfolio.get_total_value(current_date, self.data_handler)
            self.portfolio.record_daily_value(current_date, total_value)


            # --- [DEBUG] 루프 종료 시점 로그 (GPU와 형식 통일) ---
            final_capital_of_day = self.portfolio.cash
            stock_value_of_day = total_value - final_capital_of_day
            final_positions_of_day = sum(len(p) for p in self.portfolio.positions.values())
            
            # 거래 발생 여부 확인
            capital_changed = abs(final_capital_of_day - capital_before_day) > 1 # 부동소수점 오차 감안
            if capital_changed:
                print(f"  [TRADE]   Capital changed by: {final_capital_of_day - capital_before_day:,.0f}")
            print(f" [CPU_HOLDINGS] {sorted(self.portfolio.positions.keys())}")
            print(f"[END]   Capital: {final_capital_of_day:,.0f} | Stock Val: {stock_value_of_day:,.0f} | Total Val: {total_value:,.0f} | Positions: {final_positions_of_day}")

        print("백테스팅이 완료되었습니다.")
        return self.portfolio
