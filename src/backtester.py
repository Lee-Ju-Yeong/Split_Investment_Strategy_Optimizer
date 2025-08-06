# backtester.py (수정 필수!)

from tqdm import tqdm
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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
        
        # tqdm의 mininterval을 늘려 로그 출력이 밀리지 않게 함
        for i, current_date in enumerate(tqdm(trading_dates, desc="Backtesting Progress", mininterval=1.0)):
            
            # --- 1. 신호 생성 및 주문 실행 ---
            signal_events = self.strategy.generate_signals(current_date, self.portfolio, self.data_handler)
            
            # 거래 로그는 execution_handler에서 출력되므로 여기서는 생략
            if signal_events:
                for signal in signal_events:
                    self.execution_handler.execute_order(signal, self.portfolio, self.data_handler)

            # --- 2. 일별 포트폴리오 가치 및 상태 기록 ---
            total_value = self.portfolio.get_total_value(current_date, self.data_handler)
            self.portfolio.record_daily_snapshot(current_date, total_value)

            # --- 3. [핵심] 일일 포트폴리오 스냅샷 로그 출력 ---
            if total_value > 0:
                stock_value = total_value - self.portfolio.cash
                # total_value가 0이 되는 엣지 케이스 방지
                cash_ratio = (self.portfolio.cash / total_value) * 100 if total_value else 0
                stock_ratio = (stock_value / total_value) * 100 if total_value else 0

                # format_specifiers for readability
                header = f"\n{'='*120}\n"
                footer = f"\n{'='*120}"
                
                date_str = pd.to_datetime(current_date).strftime('%Y-%m-%d')
                summary_str = (
                    f"Date: {date_str} | Day {i+1}/{len(trading_dates)}\n"
                    f"{'-'*120}\n"
                    f"Total Value: {total_value:,.0f} | "
                    f"Cash: {self.portfolio.cash:,.0f} ({cash_ratio:.1f}%) | "
                    f"Stocks: {stock_value:,.0f} ({stock_ratio:.1f}%)\n"
                    f"Holdings Count: {len(self.portfolio.positions)} Stocks"
                )
                
                log_message = header + summary_str
                
                positions_df = self.portfolio.get_positions_snapshot(current_date, self.data_handler, total_value)
                
                if not positions_df.empty:
                    positions_df['Avg Buy Price'] = positions_df['Avg Buy Price'].map('{:,.0f}'.format)
                    positions_df['Current Price'] = positions_df['Current Price'].map('{:,.0f}'.format)
                    positions_df['Unrealized P/L'] = positions_df['Unrealized P/L'].map('{:,.0f}'.format)
                    positions_df['Total Value'] = positions_df['Total Value'].map('{:,.0f}'.format)
                    positions_df['P/L Rate'] = positions_df['P/L Rate'].map('{:.2%}'.format)
                    positions_df['Weight'] = positions_df['Weight'].map('{:.2%}'.format)
                    
                    log_message += f"\n{'-'*120}\n[Current Holdings]\n"
                    log_message += positions_df.to_string()

                log_message += footer
                tqdm.write(log_message) # tqdm 진행률 바를 깨뜨리지 않고 메시지 출력

        print("\n백테스팅이 완료되었습니다.")
        return self.portfolio