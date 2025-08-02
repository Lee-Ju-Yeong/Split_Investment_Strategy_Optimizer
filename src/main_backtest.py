"""
main_backtest.py

This module contains the main function for running the backtest for the Magic Split Strategy.
"""

import warnings
import configparser
import pandas as pd
from .data_handler import DataHandler
from .strategy import MagicSplitStrategy
from .portfolio import Portfolio
from .execution import BasicExecutionHandler
from .backtester import BacktestEngine
from .performance_analyzer import PerformanceAnalyzer 

# 프로그램 전역에서 pandas의 UserWarning을 무시하도록 설정
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    db_params = {
        'host': config['mysql']['host'], 'user': config['mysql']['user'],
        'password': config['mysql']['password'], 'database': config['mysql']['database'],
    }
    
    start_date = '2015-01-01'
    end_date = '2015-12-31'
    initial_cash = 10_000_000

    strategy_params = {
        'initial_capital': initial_cash,
        'max_stocks': 20,
        'order_investment_ratio': 0.02,
        'additional_buy_drop_rate': 0.04,
        'sell_profit_rate': 0.04,
        'backtest_start_date': start_date,
        'backtest_end_date': end_date,
        'additional_buy_priority': 'lowest_order',
        'consider_delisting': False
    }

    data_handler = DataHandler(db_config=db_params)
    strategy = MagicSplitStrategy(**strategy_params)
    portfolio = Portfolio(initial_cash=initial_cash, start_date=start_date, end_date=end_date)
    execution_handler = BasicExecutionHandler()

    engine = BacktestEngine(
        start_date=start_date, end_date=end_date,
        portfolio=portfolio, strategy=strategy,
        data_handler=data_handler, execution_handler=execution_handler
    )
    
    final_portfolio = engine.run()

    # 1. PerformanceAnalyzer를 이용한 거시적 성과 리포트 출력
    history_df = pd.DataFrame(final_portfolio.daily_value_history)
    history_df['date'] = pd.to_datetime(history_df['date'])
    history_df.set_index('date', inplace=True)
    daily_values = history_df['value']

    try:
        analyzer = PerformanceAnalyzer(daily_values)
        
        print("\n" + "="*50)
        print("📈 백테스팅 성과 요약 (거시 분석)")
        print("="*50)
        formatted_metrics = analyzer.get_metrics(formatted=True)
        for key, value in formatted_metrics.items():
            print(f"{key:<30}: {value}")
        print("="*50)
        
        analyzer.plot_equity_curve(save_path="latest_single_test_report.png")

    except ValueError as e:
        print(f"성과 분석 중 오류 발생: {e}")

    # 2. 기존 코드를 활용한 미시적 상세 정보 출력 (순서만 뒤로)
    print("\n" + "="*50)
    print("📂 백테스팅 상세 정보 (미시 분석)")
    print("="*50)

    print("\n--- 최종 보유 포지션 ---")
    if not final_portfolio.positions:
        print("보유 중인 포지션이 없습니다.")
    else:
        # 최종 포지션을 보기 좋게 데이터프레임으로 변환
        positions_list = []
        for ticker, positions in final_portfolio.positions.items():
            for pos in positions:
                positions_list.append({
                    '종목코드': ticker,
                    '차수': pos.order,
                    '수량': pos.quantity,
                    '매수가': f"{pos.buy_price:,.0f}"
                })
        positions_df = pd.DataFrame(positions_list)
        print(positions_df.to_string())

    print("\n--- 전체 거래 내역 (최근 20건) ---")
    if not final_portfolio.trade_history:
        print("거래 내역이 없습니다.")
    else:
        trade_df = pd.DataFrame([vars(t) for t in final_portfolio.trade_history])
        # 너무 길 수 있으니 최근 20건만 출력하거나, 파일로 저장
        print(trade_df[['date', 'code', 'trade_type', 'order', 'quantity', 'buy_price', 'sell_price', 'profit']].tail(20).to_string())
        # trade_df.to_csv("latest_trade_history.csv", index=False) # 전체 내역은 파일로 저장

if __name__ == "__main__":
    main()
