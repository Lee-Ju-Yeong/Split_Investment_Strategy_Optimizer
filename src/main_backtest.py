import configparser
import pandas as pd
from .data_handler import DataHandler
from .strategy import MagicSplitStrategy
from .portfolio import Portfolio
from .execution import BasicExecutionHandler
from .backtester import BacktestEngine
# from analysis import PerformanceAnalyzer # 추후 구현될 성능 분석 모듈

def main():
    """
    백테스팅 실행을 위한 메인 함수
    """
    # 1. 설정 및 초기화
    config = configparser.ConfigParser()
    config.read('config.ini')

    db_params = {
        'host': config['mysql']['host'],
        'user': config['mysql']['user'],
        'password': config['mysql']['password'],
        'database': config['mysql']['database'],
    }
    
    start_date = '2015-01-01'
    end_date = '2015-12-31'
    initial_cash = 10_000_000

    # 백테스팅 전략 파라미터 설정
    strategy_params = {
        'initial_capital': initial_cash,
        'num_splits': 5,
        'investment_ratio': 0.8,
        'max_stocks': 20,
        'consider_delisting': False
    }

    # 2. 각 컴포넌트 객체 생성
    data_handler = DataHandler(db_config=db_params)
    strategy = MagicSplitStrategy(**strategy_params)
    portfolio = Portfolio(initial_cash=initial_cash)
    execution_handler = BasicExecutionHandler()

    # 3. 백테스팅 엔진에 주입하고 실행
    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        strategy=strategy,
        data_handler=data_handler,
        execution_handler=execution_handler
    )
    
    # 백테스팅 실행
    final_portfolio = engine.run()

    # 4. 결과 분석 및 시각화 (추후 구현)
    print("\n--- 백테스팅 결과 ---")
    print(f"초기 자산: {initial_cash:,.0f} 원")
    
    final_value = final_portfolio.daily_value_history[-1]['value'] if final_portfolio.daily_value_history else initial_cash
    print(f"최종 자산: {final_value:,.0f} 원")

    # performance_analyzer = PerformanceAnalyzer(final_portfolio.daily_value_history)
    # print(f"CAGR: {performance_analyzer.calculate_cagr():.2%}")
    # print(f"MDD: {performance_analyzer.calculate_mdd():.2%}")
    # performance_analyzer.plot_results()

if __name__ == "__main__":
    main()
