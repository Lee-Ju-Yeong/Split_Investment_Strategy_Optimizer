"""
main_backtest.py

This module contains the main runner for executing a single backtest based on a configuration file.
It orchestrates the setup, execution, and analysis of the backtesting process.
"""
import warnings
import pandas as pd
import os
from datetime import datetime

# --- 프로젝트 핵심 모듈 임포트 ---
from .data_handler import DataHandler
from .strategy import MagicSplitStrategy
from .portfolio import Portfolio
from .execution import BasicExecutionHandler
from .backtester import BacktestEngine
from .performance_analyzer import PerformanceAnalyzer 
from .config_loader import load_config

# 프로그램 전역에서 pandas의 UserWarning을 무시하도록 설정
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')


def run_backtest_from_config(config: dict) -> dict:
    """
    설정 딕셔너리를 받아 백테스팅을 실행하고, 분석 결과를 반환합니다.
    이 함수는 app.py 와 같은 다른 모듈에서 호출될 수 있습니다.

    Args:
        config (dict): 설정 정보를 담은 딕셔너리.

    Returns:
        dict: 백테스팅 분석 결과와 파일 경로 등을 담은 딕셔너리.
              오류 발생 시 {"error": "메시지"} 형태의 딕셔너리 반환.
    """
    try:
        # 1. 설정 값 할당
        db_params = config['database']
        backtest_settings = config['backtest_settings']
        strategy_params_from_config = config['strategy_params']
        execution_params = config['execution_params']
        paths = config['paths']
    except KeyError as e:
        return {"error": f"설정 파일에 필요한 키가 없습니다: {e}"}
    # --- ★★★ 1. 설정 파일에서 save_full_trade_history 옵션 읽기 ★★★ ---
    # .get()을 사용하여 키가 없어도 에러가 나지 않고 기본값 False를 사용하도록 함
    should_save_trades = backtest_settings.get('save_full_trade_history', False)
    start_date = backtest_settings['start_date']
    end_date = backtest_settings['end_date']
    initial_cash = backtest_settings['initial_cash']
    
    # 2. 백테스팅 객체 생성
    data_handler = DataHandler(db_config=db_params)
    
    # Strategy 파라미터에 start/end_date 추가
    strategy_params = {**strategy_params_from_config, 'backtest_start_date': start_date, 'backtest_end_date': end_date}
    strategy = MagicSplitStrategy(**strategy_params)
    
    portfolio = Portfolio(initial_cash=initial_cash, start_date=start_date, end_date=end_date)
    execution_handler = BasicExecutionHandler(**execution_params)

    engine = BacktestEngine(
        start_date=start_date, end_date=end_date,
        portfolio=portfolio, strategy=strategy,
        data_handler=data_handler, execution_handler=execution_handler
    )
    
    print("백테스팅 엔진을 실행합니다...")
    final_portfolio = engine.run()

    # 3. 결과 분석 및 리포트 생성
    history_df = pd.DataFrame(final_portfolio.daily_value_history)
    if history_df.empty:
        return {"error": "백테스팅 결과 데이터가 없습니다. 분석을 수행할 수 없습니다."}

    history_df['date'] = pd.to_datetime(history_df['date'])
    history_df.set_index('date', inplace=True)
    daily_values = history_df['value']

    try:
        analyzer = PerformanceAnalyzer(daily_values)
        raw_metrics = analyzer.get_metrics(formatted=False)
        
        # 3-1. 결과 파일 저장 경로 설정
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join(paths.get('results_dir', 'results'), f"run_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        plot_filename = "equity_curve.png"
        trade_filename = "trade_history.csv"
        
        # 3-2. 분석 결과 파일로 저장
        analyzer.plot_equity_curve(
            title=f"Strategy Performance ({start_date} to {end_date})",
            save_path=os.path.join(result_dir, plot_filename)
        )
        
        trade_df = pd.DataFrame([vars(t) for t in final_portfolio.trade_history])
        trade_filepath_for_response = None  # 기본값은 None
        
        if not trade_df.empty and should_save_trades:
            trade_filename = "full_trade_history.csv" # 파일 이름 변경
            trade_filepath = os.path.join(result_dir, trade_filename)
            trade_df.to_csv(trade_filepath, index=False, encoding='utf-8-sig')
            trade_filepath_for_response = trade_filepath.replace('\\', '/')
            print(f"상세 거래 내역이 '{trade_filepath_for_response}'에 저장되었습니다.")

        # 3-3. 미시적 정보(최종 포지션, 거래 내역) 생성
        final_positions_list = []
        if final_portfolio.positions:
            latest_date = pd.to_datetime(end_date)
            for ticker, positions in final_portfolio.positions.items():
                current_price = data_handler.get_latest_price(latest_date, ticker, start_date, end_date)
                for pos in positions:
                    final_positions_list.append({
                        'ticker': ticker,
                        'order': pos.order,
                        'quantity': pos.quantity,
                        'buy_price': pos.buy_price,
                        'current_price': current_price
                    })

        trade_history_list = [vars(t) for t in final_portfolio.trade_history]

        # 4. 다른 모듈(app.py 등)에 전달할 최종 결과 딕셔너리 구성
        response = {
            "success": True,
            "metrics": raw_metrics,
            "plot_file_path": os.path.join(result_dir, plot_filename).replace('\\', '/'),
            "trade_file_path": trade_filepath_for_response,
            "daily_values": daily_values.reset_index().rename(columns={'date': 'x', 'value': 'y'}).to_dict('records'),
            "final_positions": final_positions_list,
            "trade_history": trade_history_list
        }
        return response

    except (ValueError, KeyError) as e:
        return {"error": f"성과 분석 중 오류 발생: {e}"}


def display_results_in_terminal(result: dict):
    """터미널에 백테스팅 결과를 보기 좋게 출력합니다."""
    
    print("\n" + "="*60)
    print("📈 백테스팅 성과 요약 (거시 분석)")
    print("="*60)
    metrics = result['metrics']
    print(f"{'분석 기간':<25}: {metrics['period_start']} ~ {metrics['period_end']}")
    print(f"{'초기 자산':<25}: {metrics['initial_value']:,.0f} 원")
    print(f"{'최종 자산':<25}: {metrics['final_value']:,.0f} 원")
    print("-" * 60)
    print(f"{'최종 누적 수익률':<25}: {metrics['final_cumulative_returns']:.2%}")
    print(f"{'연평균 복리 수익률 (CAGR)':<25}: {metrics['cagr']:.2%}")
    print(f"{'연간 변동성':<25}: {metrics['annualized_volatility']:.2%}")
    print(f"{'최대 낙폭 (MDD)':<25}: {metrics['mdd']:.2%}")
    print(f"{'샤프 지수':<25}: {metrics['sharpe_ratio']:.2f}")
    print(f"{'소르티노 지수':<25}: {metrics['sortino_ratio']:.2f}")
    print(f"{'칼마 지수':<25}: {metrics['calmar_ratio']:.2f}")
    print(f"\n결과 그래프: {result['plot_file_path']}")
    print(f"거래 내역 파일: {result['trade_file_path']}")
    print("="*60)
    
    print("\n" + "="*60)
    print("📂 백테스팅 상세 정보 (미시 분석)")
    print("="*60)

    print("\n--- 최종 보유 포지션 ---")
    positions_df = pd.DataFrame(result.get('final_positions', []))
    if not positions_df.empty:
        # 평가금액, 수익률 컬럼 추가
        positions_df['current_value'] = positions_df['quantity'] * positions_df['current_price']
        positions_df['profit_loss'] = positions_df['current_value'] - (positions_df['quantity'] * positions_df['buy_price'])
        positions_df['profit_rate'] = (positions_df['profit_loss'] / (positions_df['quantity'] * positions_df['buy_price'])).fillna(0)
        
        # 출력 포맷팅
        positions_df['buy_price'] = positions_df['buy_price'].map('{:,.0f}'.format)
        positions_df['current_price'] = positions_df['current_price'].map('{:,.0f}'.format)
        positions_df['current_value'] = positions_df['current_value'].map('{:,.0f}'.format)
        positions_df['profit_loss'] = positions_df['profit_loss'].map('{:,.0f}'.format)
        positions_df['profit_rate'] = positions_df['profit_rate'].map('{:.2%}'.format)

        print(positions_df.to_string())
    else:
        print("보유 중인 포지션이 없습니다.")
    
    print("\n--- 최근 거래 내역 (10건) ---")
    trades_df = pd.DataFrame(result.get('trade_history', []))
    if not trades_df.empty:
        trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
        print(trades_df[['date', 'code', 'trade_type', 'order', 'quantity', 'buy_price', 'sell_price', 'profit']].tail(10).to_string())
    else:
        print("거래 내역이 없습니다.")
    print("="*60)


def main():
    """터미널에서 직접 실행할 때 사용되는 메인 함수."""
    try:
        config = load_config()  # 인자를 비워서 기본값('config/config.yaml')을 사용하도록 함
        result = run_backtest_from_config(config)
        
        if result.get("success"):
            display_results_in_terminal(result)
        else:
            print(f"\n[오류] 백테스팅 실행 중 문제가 발생했습니다: {result.get('error', '알 수 없는 오류')}")

    except FileNotFoundError:
        print("\n[오류] 'config.yaml' 설정 파일을 찾을 수 없습니다. 프로젝트 루트 디렉토리에 파일을 생성해주세요.")
    except Exception as e:
        print(f"\n[치명적 오류] 예기치 않은 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()