# main_backtest.py (수정된 최종본)

import warnings
# pandas UserWarning을 다른 모듈 임포트 전에 필터링합니다.
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

import logging
import pandas as pd
import os
import json
from datetime import datetime

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/main_backtest.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from .data_handler import DataHandler, build_pit_failure_record
from .backtest.cpu.strategy import MagicSplitStrategy
from .backtest.cpu.portfolio import Portfolio
from .backtest.cpu.execution import BasicExecutionHandler
from .backtest.cpu.backtester import BacktestEngine
from .config_loader import load_config
from .price_policy import resolve_price_policy
from .universe_policy import resolve_universe_mode
# company_info_manager는 이제 DataHandler가 내부적으로 사용하므로 여기서 직접 임포트할 필요가 없습니다.

logger = logging.getLogger(__name__)

_STRATEGY_PARAM_KEYS = {
    "max_stocks",
    "order_investment_ratio",
    "additional_buy_drop_rate",
    "sell_profit_rate",
    "additional_buy_priority",
    "cooldown_period_days",
    "stop_loss_rate",
    "max_splits_limit",
    "max_inactivity_period",
    "candidate_source_mode",
    "use_weekly_alpha_gate",
    "min_liquidity_20d_avg_value",
    "min_tier12_coverage_ratio",
    "tier_hysteresis_mode",
    "candidate_lookup_error_policy",
}

_EXECUTION_PARAM_KEYS = {
    "buy_commission_rate",
    "sell_commission_rate",
    "sell_tax_rate",
}


def _write_run_manifest(
    *,
    result_dir: str,
    strategy_params: dict,
    backtest_settings: dict,
    universe_mode: str,
    price_basis: str,
    adjusted_gate: str,
    run_metrics: dict | None = None,
    candidate_lookup_summary: dict | None = None,
    safety_guard: dict | None = None,
    status: str = "success",
    error_info: dict | None = None,
) -> str:
    env_universe_mode = os.environ.get("MAGICSPLIT_UNIVERSE_MODE")
    env_config_path = os.environ.get("MAGICSPLIT_CONFIG_PATH")
    manifest = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_type": "cpu_backtest",
        "backtest_window": {
            "start_date": str(backtest_settings.get("start_date")),
            "end_date": str(backtest_settings.get("end_date")),
        },
        "price_policy": {
            "basis": str(price_basis),
            "adjusted_price_gate_start_date": str(adjusted_gate),
        },
        "universe_policy": {
            "resolved_mode": str(universe_mode),
            "config_mode": strategy_params.get("universe_mode"),
            "env_override_value": env_universe_mode,
            "env_override_applied": bool(env_universe_mode and str(env_universe_mode).strip()),
        },
        "config": {
            "config_path": env_config_path or "config/config.yaml",
            "candidate_source_mode": strategy_params.get("candidate_source_mode"),
            "tier_hysteresis_mode": strategy_params.get("tier_hysteresis_mode"),
            "candidate_lookup_error_policy": strategy_params.get("candidate_lookup_error_policy"),
            "frozen_candidate_manifest_mode": strategy_params.get("frozen_candidate_manifest_mode"),
            "frozen_candidate_manifest_path": strategy_params.get("frozen_candidate_manifest_path"),
            "frozen_candidate_manifest_expected_sha256": strategy_params.get(
                "frozen_candidate_manifest_expected_sha256"
            ),
        },
        "env_overrides": {
            "MAGICSPLIT_UNIVERSE_MODE": env_universe_mode,
            "MAGICSPLIT_CONFIG_PATH": env_config_path,
        },
        "status": str(status),
        "run_metrics": run_metrics or {},
        "candidate_lookup": candidate_lookup_summary or {},
        "safety_guard": safety_guard or {},
    }
    if error_info:
        manifest["error_info"] = error_info
    manifest_path = os.path.join(result_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def _build_failure_result(
    *,
    error,
    persist_artifacts: bool,
    strategy_params: dict,
    backtest_settings: dict,
    universe_mode: str,
    price_basis: str,
    adjusted_gate: str,
    paths: dict,
    candidate_lookup_summary: dict | None = None,
) -> dict:
    pit_failure = build_pit_failure_record(error)
    if pit_failure:
        logger.exception(
            "[PITFailure] code=%s stage=%s message=%s details=%s",
            pit_failure.get("code"),
            pit_failure.get("stage"),
            pit_failure.get("message"),
            pit_failure.get("details"),
        )
    else:
        logger.exception("성과 분석 중 오류 발생")

    reasons = []
    if pit_failure:
        reasons.append(f"pit_failure:{pit_failure.get('code')}")
    else:
        reasons.append(f"runtime_error:{type(error).__name__}")
    safety_guard = {
        "degraded_run": True,
        "promotion_blocked": True,
        "reasons": reasons,
    }
    error_info = {
        "message": f"성과 분석 중 오류 발생: {error}",
        "error_type": type(error).__name__,
    }
    if pit_failure:
        error_info["pit_failure"] = pit_failure

    run_manifest_path = None
    if persist_artifacts:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join(paths.get('results_dir', 'results'), f"run_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        run_manifest_path = _write_run_manifest(
            result_dir=result_dir,
            strategy_params=strategy_params,
            backtest_settings=backtest_settings,
            universe_mode=universe_mode,
            price_basis=price_basis,
            adjusted_gate=adjusted_gate,
            run_metrics={},
            candidate_lookup_summary=candidate_lookup_summary or {},
            safety_guard=safety_guard,
            status="failed",
            error_info=error_info,
        )

    response = {
        "error": error_info["message"],
        "error_type": error_info["error_type"],
        "run_manifest_path": run_manifest_path.replace('\\', '/') if run_manifest_path else None,
        "promotion_blocked": True,
        "promotion_block_reasons": list(safety_guard["reasons"]),
        "candidate_lookup_summary": candidate_lookup_summary or {},
    }
    if pit_failure:
        response["pit_failure"] = pit_failure
    return response


def _build_safety_guard(
    *,
    universe_mode: str,
    strategy_params: dict,
    run_metrics: dict | None,
    candidate_lookup_summary: dict | None,
) -> dict:
    reasons = []
    policy = str(strategy_params.get("candidate_lookup_error_policy", "raise")).strip().lower()
    if policy == "skip":
        reasons.append("candidate_lookup_error_policy=skip")
    if str(universe_mode).strip().lower() != "strict_pit":
        reasons.append(f"universe_mode={universe_mode}")

    lookup_errors = 0
    if candidate_lookup_summary:
        try:
            lookup_errors = int(candidate_lookup_summary.get("error_count", 0) or 0)
        except (TypeError, ValueError):
            lookup_errors = 0
    if lookup_errors <= 0:
        try:
            lookup_errors = int((run_metrics or {}).get("source_lookup_error_days", 0) or 0)
        except (TypeError, ValueError):
            lookup_errors = 0
    if lookup_errors > 0:
        reasons.append(f"candidate_lookup_errors={lookup_errors}")

    blocked = bool(reasons)
    return {
        "degraded_run": blocked,
        "promotion_blocked": blocked,
        "reasons": reasons,
    }

def run_backtest_from_config(config: dict, *, persist_artifacts: bool = True) -> dict:
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
    universe_mode = resolve_universe_mode(
        strategy_params_from_config,
        universe_mode=os.environ.get("MAGICSPLIT_UNIVERSE_MODE"),
    )
    
    price_basis, adjusted_gate = resolve_price_policy(strategy_params_from_config)
    logger.info(
        "price policy | basis=%s | adjusted_gate_start=%s",
        price_basis,
        adjusted_gate,
    )
    logger.info("universe policy | mode=%s", universe_mode)

    try:
        # DataHandler는 이제 종목명 조회를 위해 CompanyInfo DB를 내부적으로 로드합니다.
        data_handler = DataHandler(
            db_config=db_params,
            price_basis=price_basis,
            adjusted_price_gate_start_date=adjusted_gate,
            universe_mode=universe_mode,
            strategy_params=strategy_params_from_config,
        )
        
        strategy_params = {
            k: v
            for k, v in strategy_params_from_config.items()
            if k in _STRATEGY_PARAM_KEYS
        }
        strategy_params.update({"backtest_start_date": start_date, "backtest_end_date": end_date})
        strategy = MagicSplitStrategy(**strategy_params)
        
        portfolio = Portfolio(initial_cash=initial_cash, start_date=start_date, end_date=end_date)
        execution_handler = BasicExecutionHandler(
            **{
                k: v
                for k, v in execution_params.items()
                if k in _EXECUTION_PARAM_KEYS
            }
        )

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

        from .performance_analyzer import PerformanceAnalyzer

        # PerformanceAnalyzer는 이제 전체 history_df를 받습니다.
        analyzer = PerformanceAnalyzer(history_df)
        raw_metrics = analyzer.get_metrics(formatted=False)
        run_metrics = getattr(final_portfolio, "run_metrics", {}) or {}
        candidate_lookup_summary_getter = getattr(strategy, "get_candidate_lookup_error_summary", None)
        candidate_lookup_summary = (
            candidate_lookup_summary_getter()
            if callable(candidate_lookup_summary_getter)
            else {}
        )
        frozen_manifest_summary_getter = getattr(
            data_handler,
            "get_frozen_candidate_manifest_summary",
            None,
        )
        if callable(frozen_manifest_summary_getter):
            candidate_lookup_summary["frozen_candidate_manifest"] = (
                frozen_manifest_summary_getter()
            )
        safety_guard = _build_safety_guard(
            universe_mode=universe_mode,
            strategy_params=strategy_params_from_config,
            run_metrics=run_metrics,
            candidate_lookup_summary=candidate_lookup_summary,
        )
        if safety_guard.get("promotion_blocked"):
            logger.warning(
                "승격 차단: degraded run detected (%s)",
                ", ".join(safety_guard.get("reasons", [])) or "unknown reason",
            )
        
        run_manifest_path = None
        plot_file_path = None
        trade_df = pd.DataFrame([vars(t) for t in final_portfolio.trade_history])
        trade_filepath_for_response = None

        if persist_artifacts:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_dir = os.path.join(paths.get('results_dir', 'results'), f"run_{timestamp}")
            os.makedirs(result_dir, exist_ok=True)
            run_manifest_path = _write_run_manifest(
                result_dir=result_dir,
                strategy_params=strategy_params_from_config,
                backtest_settings=backtest_settings,
                universe_mode=universe_mode,
                price_basis=price_basis,
                adjusted_gate=adjusted_gate,
                run_metrics=run_metrics,
                candidate_lookup_summary=candidate_lookup_summary,
                safety_guard=safety_guard,
            )
            logger.info("실행 메타데이터가 '%s'에 저장되었습니다.", run_manifest_path.replace('\\', '/'))

            plot_filename = "performance_report.png" # 파일 이름 변경
            plot_file_path = os.path.join(result_dir, plot_filename).replace('\\', '/')
            analyzer.plot_equity_curve(
                title=f"Strategy Performance ({start_date} to {end_date})",
                save_path=os.path.join(result_dir, plot_filename)
            )

        if persist_artifacts and not trade_df.empty and should_save_trades:
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
            "run_metrics": run_metrics,
            "universe_mode": universe_mode,
            "degraded_run": bool(safety_guard.get("degraded_run", False)),
            "promotion_blocked": bool(safety_guard.get("promotion_blocked", False)),
            "promotion_block_reasons": list(safety_guard.get("reasons", [])),
            "candidate_lookup_summary": candidate_lookup_summary,
            "run_manifest_path": run_manifest_path.replace('\\', '/') if run_manifest_path else None,
            "plot_file_path": plot_file_path,
            "trade_file_path": trade_filepath_for_response,
            "daily_values": daily_values_for_response.reset_index().rename(columns={'date': 'x', 'total_value': 'y'}).to_dict('records'),
            "final_positions": final_positions_list,
            "trade_history": trade_history_list
        }
        return response

    except (ValueError, KeyError) as e:
        candidate_lookup_summary = {}
        strategy_obj = locals().get("strategy")
        data_handler_obj = locals().get("data_handler")
        candidate_lookup_summary_getter = getattr(strategy_obj, "get_candidate_lookup_error_summary", None)
        if callable(candidate_lookup_summary_getter):
            candidate_lookup_summary = candidate_lookup_summary_getter()
        frozen_manifest_summary_getter = getattr(
            data_handler_obj,
            "get_frozen_candidate_manifest_summary",
            None,
        )
        if callable(frozen_manifest_summary_getter):
            candidate_lookup_summary["frozen_candidate_manifest"] = (
                frozen_manifest_summary_getter()
            )
        return _build_failure_result(
            error=e,
            persist_artifacts=persist_artifacts,
            strategy_params=strategy_params_from_config,
            backtest_settings=backtest_settings,
            universe_mode=universe_mode,
            price_basis=price_basis,
            adjusted_gate=adjusted_gate,
            paths=paths,
            candidate_lookup_summary=candidate_lookup_summary,
        )

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
