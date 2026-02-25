# main_backtest.py (수정된 최종본)

import warnings
# pandas UserWarning을 다른 모듈 임포트 전에 필터링합니다.
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

import logging
import pandas as pd
import os
from datetime import datetime

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/main_backtest.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from .data_handler import DataHandler
from .backtest.cpu.strategy import MagicSplitStrategy
from .backtest.cpu.portfolio import Portfolio
from .backtest.cpu.execution import BasicExecutionHandler
from .backtest.cpu.backtester import BacktestEngine
from .config_loader import load_config
from .price_policy import resolve_price_policy
# company_info_manager는 이제 DataHandler가 내부적으로 사용하므로 여기서 직접 임포트할 필요가 없습니다.

logger = logging.getLogger(__name__)

def run_backtest_from_config(config: dict) -> dict:
    # When called from non-CLI entrypoints (e.g., Flask), logging may not be configured.
    # We only auto-configure if root has no handlers to avoid duplicating external setups.
    root = logging.getLogger()
    if not getattr(root, "_magic_split_logging_configured", False) and not root.handlers:
        from .logging_utils import setup_logging

        setup_logging()

    try:
        db_params = config['database']
        backtest_settings = config['backtest_settings']
        strategy_params_from_config = config['strategy_params']
        execution_params = config['execution_params']
        paths = config['paths']
    except KeyError as e:
        return {"error": f"설정 파일에 필요한 키가 없습니다: {e}"}
        
    should_save_trades = backtest_settings.get('save_full_trade_history', False)
    start_date = backtest_settings['start_date']
    end_date = backtest_settings['end_date']
    initial_cash = backtest_settings['initial_cash']
    
    price_basis, adjusted_gate = resolve_price_policy(strategy_params_from_config)
    logger.info(
        "price policy | basis=%s | adjusted_gate_start=%s",
        price_basis,
        adjusted_gate,
    )

    # DataHandler는 이제 종목명 조회를 위해 CompanyInfo DB를 내부적으로 로드합니다.
    data_handler = DataHandler(
        db_config=db_params,
        price_basis=price_basis,
        adjusted_price_gate_start_date=adjusted_gate,
    )
    
    strategy_params = {**strategy_params_from_config, 'backtest_start_date': start_date, 'backtest_end_date': end_date}
    strategy = MagicSplitStrategy(**strategy_params)
    
    portfolio = Portfolio(initial_cash=initial_cash, start_date=start_date, end_date=end_date)
    execution_handler = BasicExecutionHandler(**execution_params)

    engine = BacktestEngine(
        start_date=start_date, end_date=end_date,
        portfolio=portfolio, strategy=strategy,
        data_handler=data_handler, execution_handler=execution_handler
    )
    
    logger.info("백테스팅 엔진을 실행합니다...")
    final_portfolio = engine.run()

    ### ### 이슈 구현: daily_snapshot_history 사용으로 변경 ### ###
    history_df = pd.DataFrame(final_portfolio.daily_snapshot_history)
    if history_df.empty:
        return {"error": "백테스팅 결과 데이터가 없습니다. 분석을 수행할 수 없습니다."}

    history_df['date'] = pd.to_datetime(history_df['date'])
    history_df.set_index('date', inplace=True)
    
    # daily_values는 이제 history_df의 한 컬럼일 뿐입니다.
    daily_values_for_response = history_df['total_value']

    try:
        from .performance_analyzer import PerformanceAnalyzer

        # PerformanceAnalyzer는 이제 전체 history_df를 받습니다.
        analyzer = PerformanceAnalyzer(history_df)
        raw_metrics = analyzer.get_metrics(formatted=False)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join(paths.get('results_dir', 'results'), f"run_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        plot_filename = "performance_report.png" # 파일 이름 변경
        
        analyzer.plot_equity_curve(
            title=f"Strategy Performance ({start_date} to {end_date})",
            save_path=os.path.join(result_dir, plot_filename)
        )
        
        trade_df = pd.DataFrame([vars(t) for t in final_portfolio.trade_history])
        trade_filepath_for_response = None
        
        if not trade_df.empty and should_save_trades:
            trade_filename = "full_trade_history.csv"
            trade_filepath = os.path.join(result_dir, trade_filename)
            trade_df.to_csv(trade_filepath, index=False, encoding='utf-8-sig')
            trade_filepath_for_response = trade_filepath.replace('\\', '/')
            logger.info("상세 거래 내역이 '%s'에 저장되었습니다.", trade_filepath_for_response)

        final_positions_list = []
        if final_portfolio.positions:
            latest_date = pd.to_datetime(end_date)
            # 최종 스냅샷 계산을 위해 get_positions_snapshot 활용
            final_positions_df = final_portfolio.get_positions_snapshot(latest_date, data_handler, history_df['total_value'].iloc[-1])
            if not final_positions_df.empty:
                 # DataFrame을 list of dicts로 변환
                 final_positions_list = final_positions_df.to_dict('records')

        trade_history_list = [vars(t) for t in final_portfolio.trade_history]

        response = {
            "success": True,
            "metrics": raw_metrics,
            "plot_file_path": os.path.join(result_dir, plot_filename).replace('\\', '/'),
            "trade_file_path": trade_filepath_for_response,
            "daily_values": daily_values_for_response.reset_index().rename(columns={'date': 'x', 'total_value': 'y'}).to_dict('records'),
            "final_positions": final_positions_list,
            "trade_history": trade_history_list
        }
        return response

    except (ValueError, KeyError) as e:
        import traceback
        traceback.print_exc()
        return {"error": f"성과 분석 중 오류 발생: {e}"}

# ... display_results_in_terminal 과 main 함수는 기존과 동일하게 유지 ...
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
    if result['trade_file_path']:
      print(f"거래 내역 파일: {result['trade_file_path']}")
    print("="*60)
    
    print("\n" + "="*60)
    print("📂 백테스팅 상세 정보 (미시 분석)")
    print("="*60)

    print("\n--- 최종 보유 포지션 ---")
    final_positions = result.get('final_positions', [])
    if final_positions:
        positions_df = pd.DataFrame(final_positions)
        # 이미 계산된 값들을 포맷팅만 진행
        positions_df['Avg Buy Price'] = positions_df['Avg Buy Price'].map('{:,.0f}'.format)
        positions_df['Current Price'] = positions_df['Current Price'].map('{:,.0f}'.format)
        positions_df['Unrealized P/L'] = positions_df['Unrealized P/L'].map('{:,.0f}'.format)
        positions_df['Total Value'] = positions_df['Total Value'].map('{:,.0f}'.format)
        positions_df['P/L Rate'] = positions_df['P/L Rate'].map('{:.2%}'.format)
        positions_df['Weight'] = positions_df['Weight'].map('{:.2%}'.format)
        print(positions_df.to_string())
    else:
        print("보유 중인 포지션이 없습니다.")
    
    print("\n--- 최근 거래 내역 (10건) ---")
    trades_df = pd.DataFrame(result.get('trade_history', []))
    if not trades_df.empty:
        trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
        # 요청사항을 반영하여 컬럼 재구성
        display_columns = ['date', 'code', 'name', 'trade_type', 'order', 'reason_for_trade', 'quantity', 'buy_price', 'sell_price', 'commission', 'tax', 'realized_pnl']
        
        # DataFrame에 있는 컬럼만 선택하여 에러 방지
        existing_columns = [col for col in display_columns if col in trades_df.columns]
        print(trades_df[existing_columns].tail(10).to_string())
    else:
        print("거래 내역이 없습니다.")
    print("="*60)


def main():
    """터미널에서 직접 실행할 때 사용되는 메인 함수."""
    try:
        from .logging_utils import setup_logging

        setup_logging()
        config = load_config()
        result = run_backtest_from_config(config)
        
        if result.get("success"):
            display_results_in_terminal(result)
        else:
            logger.error("백테스팅 실행 중 문제가 발생했습니다: %s", result.get("error", "알 수 없는 오류"))

    except FileNotFoundError:
        logger.error("'config/config.yaml' 설정 파일을 찾을 수 없습니다. 프로젝트 루트에 생성해주세요.")
    except Exception:
        logger.exception("백테스팅 실행 중 예외 발생")

if __name__ == "__main__":
    main()
