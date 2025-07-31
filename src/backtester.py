from tqdm import tqdm
import pandas as pd

class BacktestEngine:
    """
    백테스팅의 전체 이벤트 루프를 제어하는 오케스트레이터입니다.
    """
    def __init__(self, start_date, end_date, portfolio, strategy, data_handler, execution_handler):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.portfolio = portfolio
        self.strategy = strategy
        self.data_handler = data_handler
        self.execution_handler = execution_handler

    def run(self):
        """
        백테스팅 이벤트 루프를 실행합니다.
        """
        print("백테스팅 엔진을 시작합니다...")
        
        # 전체 거래일 목록을 데이터 핸들러로부터 가져옵니다.
        trading_dates = self.data_handler.get_trading_dates(self.start_date, self.end_date)

        for current_date in tqdm(trading_dates, desc="Backtesting Progress"):
            # 1. 전략에 따른 신호 생성
            signal_events = self.strategy.generate_signals(current_date, self.portfolio, self.data_handler)
            
            # 2. 생성된 신호를 기반으로 주문 처리
            if signal_events:
                for signal in signal_events:
                    self.execution_handler.execute_order(signal, self.portfolio, self.data_handler)

            # 3. 일별 포트폴리오 가치 업데이트
            total_value = self.portfolio.get_total_value(current_date, self.data_handler)
            self.portfolio.record_daily_value(current_date, total_value)

        print("백테스팅이 완료되었습니다.")
        return self.portfolio
