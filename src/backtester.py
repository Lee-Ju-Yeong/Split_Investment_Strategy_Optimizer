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
            debug_ticker = '013570'
            try:
                stock_data = self.data_handler.load_stock_data(debug_ticker, self.start_date, self.end_date)
                # 데이터프레임이 유효하고, 현재 날짜의 데이터가 존재하는지 확인
                if stock_data is not None and not stock_data.empty and current_date in stock_data.index:
                    ohlc = stock_data.loc[current_date]
                    # tqdm 진행률 바와 출력이 겹치지 않도록 tqdm.write() 사용
                    tqdm.write(
                        f"[CPU_DATA_DEBUG] {current_date.strftime('%Y-%m-%d')} | {debug_ticker} | "
                        f"Open={ohlc['open_price']}, High={ohlc['high_price']}, "
                        f"Low={ohlc['low_price']}, Close={ohlc['close_price']}"
                    )
            except Exception:
                # 디버깅 중 에러가 발생해도 백테스트가 멈추지 않도록 pass 처리
                pass
            # --- 1. [변경] 신호 생성 및 실행 로직 분리 ---
            
            # [추가] 오늘 거래가 실행되기 전의 거래 내역 개수를 기록
            num_trades_before = len(self.portfolio.trade_history)

            # 단계 1-1: 매도 신호 생성 및 즉시 실행
            sell_signals = self.strategy.generate_sell_signals(current_date, self.portfolio, self.data_handler, trading_dates, i)
            if sell_signals:
                for signal in sell_signals:
                    self.execution_handler.execute_order(signal, self.portfolio, self.data_handler)

            # 단계 1-2: 매수 신호 생성 및 즉시 실행 (매도가 반영된 최신 포트폴리오 기준)
            buy_signals = self.strategy.generate_buy_signals(current_date, self.portfolio, self.data_handler,trading_dates, i)
            if buy_signals:
                for signal in buy_signals:
                    self.execution_handler.execute_order(signal, self.portfolio, self.data_handler)

            # --- 2. 일별 포트폴리오 가치 및 상태 기록 ---
            total_value = self.portfolio.get_total_value(current_date, self.data_handler)
            self.portfolio.record_daily_snapshot(current_date, total_value)

            # --- 3. [신규 로직] 당일 발생한 모든 거래에 최종 포트폴리오 가치 업데이트 ---
            num_trades_after = len(self.portfolio.trade_history)
            if num_trades_after > num_trades_before:
                for i in range(num_trades_before, num_trades_after):
                    self.portfolio.trade_history[i].total_portfolio_value = total_value

            # --- 4. [핵심] 일일 포트폴리오 스냅샷 로그 출력 ---
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