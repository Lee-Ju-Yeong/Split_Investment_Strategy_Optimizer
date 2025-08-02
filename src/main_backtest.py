import warnings
import configparser
import pandas as pd
from .data_handler import DataHandler
from .strategy import MagicSplitStrategy
from .portfolio import Portfolio
from .execution import BasicExecutionHandler
from .backtester import BacktestEngine

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

    print("\n--- 백테스팅 결과 ---")
    print(f"초기 자산: {initial_cash:,.0f} 원")
    final_value = final_portfolio.daily_value_history[-1]['value'] if final_portfolio.daily_value_history else initial_cash
    print(f"최종 자산: {final_value:,.0f} 원")

    print("\n--- 최종 보유 포지션 ---")
    if not final_portfolio.positions:
        print("보유 중인 포지션이 없습니다.")
    else:
        for ticker, positions in final_portfolio.positions.items():
            for pos in positions:
                print(f"종목: {ticker}, 차수: {pos.order}, 수량: {pos.quantity}, 매수가: {pos.buy_price:,.0f} 원")

    print("\n--- 전체 거래 내역 ---")
    if not final_portfolio.trade_history:
        print("거래 내역이 없습니다.")
    else:
        trade_df = pd.DataFrame([vars(t) for t in final_portfolio.trade_history])
        print(trade_df.to_string())

if __name__ == "__main__":
    main()
