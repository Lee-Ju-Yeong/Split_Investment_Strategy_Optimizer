"""
This script is used to debug the GPU single run.
It is used to test the GPU single run with the parameters from the config.yaml file.
"""
import time
import argparse
import cupy as cp
import pandas as pd
from sqlalchemy import create_engine
import os
import urllib.parse

# --- 필요한 모듈 추가 임포트 ---
from src.config_loader import load_config
from src.backtest.gpu.engine import run_magic_split_strategy_on_gpu
from src.gpu_execution_policy import build_gpu_execution_params
from src.price_policy import (
    is_adjusted_price_basis,
    resolve_price_policy,
    validate_backtest_window_for_price_policy,
)
from src.tier_hysteresis_policy import normalize_tier_hysteresis_mode
from src.universe_policy import resolve_universe_mode
from src.optimization.gpu.data_loading import (
    build_empty_weekly_filtered_gpu as build_empty_weekly_filtered_gpu_shared,
    preload_all_data_to_gpu as preload_all_data_to_gpu_shared,
    preload_tier_data_to_tensor as preload_tier_data_to_tensor_shared,
)
### 이슈 #3 동기화를 위한 모듈 임포트 ###
from src.performance_analyzer import PerformanceAnalyzer

# -----------------------------------------------------------------------------
# 1. Configuration and Parameter Setup
# -----------------------------------------------------------------------------

# --- 설정 파일 로드 (YAML 로더로 통일) ---
config = load_config()
db_config = config['database']
backtest_settings = config['backtest_settings']
strategy_params = config['strategy_params']
execution_params = config['execution_params'].copy()

# GPU 커널에 쿨다운 기간 전달을 위해 execution_params에 추가
execution_params['cooldown_period_days'] = strategy_params.get('cooldown_period_days', 5)

# URL 인코딩을 포함하여 DB 연결 문자열 생성
db_pass_encoded = urllib.parse.quote_plus(db_config['password'])
db_connection_str = (
    f"mysql+pymysql://{db_config['user']}:{db_pass_encoded}"
    f"@{db_config['host']}/{db_config['database']}"
)

# --- Debug를 위한 단일 파라미터 조합 정의 ---
cpu_test_params = config['strategy_params']
max_stocks_options = cp.array([cpu_test_params['max_stocks']], dtype=cp.int32)
order_investment_ratio_options = cp.array([cpu_test_params['order_investment_ratio']], dtype=cp.float32)
additional_buy_drop_rate_options = cp.array([cpu_test_params['additional_buy_drop_rate']], dtype=cp.float32)
sell_profit_rate_options = cp.array([cpu_test_params['sell_profit_rate']], dtype=cp.float32)

priority_map = {'lowest_order': 0, 'highest_drop': 1}
priority_val = priority_map.get(cpu_test_params['additional_buy_priority'], 0)
additional_buy_priority_options = cp.array([priority_val], dtype=cp.int32)

# --- [New] Load Advanced Risk Management Parameters ---
stop_loss_rate_options = cp.array([cpu_test_params.get('stop_loss_rate', -0.15)], dtype=cp.float32)
max_splits_limit_options = cp.array([cpu_test_params.get('max_splits_limit', 10)], dtype=cp.int32)
max_inactivity_period_options = cp.array([cpu_test_params.get('max_inactivity_period', 90)], dtype=cp.int32)


grid = cp.meshgrid(
    max_stocks_options,
    order_investment_ratio_options,
    additional_buy_drop_rate_options,
    sell_profit_rate_options,
    additional_buy_priority_options,
    stop_loss_rate_options,          
    max_splits_limit_options,        
    max_inactivity_period_options       
)
param_combinations = cp.vstack([item.flatten() for item in grid]).T
num_combinations = param_combinations.shape[0]

print("✅ [DEBUG MODE] Single parameter combination for GPU test:")
print(param_combinations.get())
print(f"✅ Total parameter combinations generated for GPU: {num_combinations}")


# -----------------------------------------------------------------------------
# 2. GPU Data Pre-loader
# -----------------------------------------------------------------------------

def preload_all_data_to_gpu(
    engine,
    start_date,
    end_date,
    *,
    use_adjusted_prices=False,
    adjusted_price_gate_start_date="2013-11-20",
    universe_mode="optimistic_survivor",
):
    return preload_all_data_to_gpu_shared(
        engine,
        start_date,
        end_date,
        use_adjusted_prices=use_adjusted_prices,
        adjusted_price_gate_start_date=adjusted_price_gate_start_date,
        universe_mode=universe_mode,
    )

def preload_weekly_filtered_stocks_to_gpu(engine, start_date, end_date):
    _ = (engine, start_date, end_date)
    print("⏭️ Skipping weekly filtered preload (tier-only runtime path).")
    return build_empty_weekly_filtered_gpu_shared()

def preload_tier_data_to_tensor(
    engine,
    start_date,
    end_date,
    all_tickers,
    trading_dates_pd,
    *,
    universe_mode="optimistic_survivor",
    min_liquidity_20d_avg_value=0,
    min_tier12_coverage_ratio=None,
):
    return preload_tier_data_to_tensor_shared(
        engine,
        start_date,
        end_date,
        all_tickers,
        trading_dates_pd,
        universe_mode=universe_mode,
        min_liquidity_20d_avg_value=min_liquidity_20d_avg_value,
        min_tier12_coverage_ratio=min_tier12_coverage_ratio,
    )

# -----------------------------------------------------------------------------
# 3. GPU Backtesting Kernel
# -----------------------------------------------------------------------------

def run_gpu_backtest_kernel(params_gpu, data_gpu,
                         weekly_filtered_gpu, all_tickers,
                         trading_date_indices_gpu,
                         trading_dates_pd,
                         initial_cash_value,
                         exec_params: dict,
                         debug_mode: bool = False,
                         tier_tensor: cp.ndarray = None # [Issue #67]
                         ):
    """
    GPU-accelerated 백테스팅 커널을 직접 실행합니다.
    """
    print("🚀 Starting GPU backtesting kernel...")
    
    daily_portfolio_values = run_magic_split_strategy_on_gpu(
        initial_cash=initial_cash_value,
        param_combinations=params_gpu,
        all_data_gpu=data_gpu,
        weekly_filtered_gpu=weekly_filtered_gpu,
        trading_date_indices=trading_date_indices_gpu,
        trading_dates_pd_cpu=trading_dates_pd,
        all_tickers=all_tickers,
        max_splits_limit=20,
        execution_params=exec_params,
        debug_mode=debug_mode,
        tier_tensor=tier_tensor
    )
    
    print("🎉 GPU backtesting kernel finished.")
    
    return daily_portfolio_values


def _build_run_execution_params(params_dict: dict, universe_mode: str):
    return build_gpu_execution_params(
        execution_params,
        params_dict,
        universe_mode,
        default_tier_hysteresis_mode=strategy_params.get(
            'tier_hysteresis_mode',
            'strict_hysteresis_v1',
        ),
    )

# 4. [신규] 워커 함수: run_single_backtest
def run_single_backtest(start_date: str, end_date: str, params_dict: dict, initial_cash: float, debug_mode: bool = False):
    """
    주어진 기간과 파라미터로 단일 GPU 백테스트를 수행하고 결과를 반환합니다.
    WFO 오케스트레이터에 의해 호출되는 '워커' 함수입니다.
    """
    print(f"\n" + "="*80)
    print(f"WORKER: Running Single Backtest for {start_date} to {end_date}")
    print(f"Params: {params_dict}")
    print("="*80)
    universe_mode = resolve_universe_mode(
        params_dict,
        universe_mode=os.environ.get("MAGICSPLIT_UNIVERSE_MODE"),
    )
    price_basis, adjusted_gate_start_date = resolve_price_policy(params_dict)
    validate_backtest_window_for_price_policy(
        start_date=start_date,
        end_date=end_date,
        price_basis=price_basis,
        adjusted_price_gate_start_date=adjusted_gate_start_date,
    )
    use_adjusted_prices = is_adjusted_price_basis(price_basis)
    print(f"Universe Policy: mode={universe_mode}")
    print(
        "Price Policy: "
        f"basis={price_basis}, adjusted_gate_start={adjusted_gate_start_date}"
    )

    # 1. 파라미터 딕셔너리를 GPU가 사용할 수 있는 cp.ndarray로 변환
    priority_map = {'lowest_order': 0, 'highest_drop': 1}
    priority_val = priority_map.get(params_dict.get('additional_buy_priority', 'lowest_order'), 0)
    
    param_combinations = cp.array([[
        params_dict['max_stocks'],
        params_dict['order_investment_ratio'],
        params_dict['additional_buy_drop_rate'],
        params_dict['sell_profit_rate'],
        priority_val,
        params_dict['stop_loss_rate'],
        params_dict['max_splits_limit'],
        params_dict['max_inactivity_period'],
    ]], dtype=cp.float32)

    # 2. 데이터 로드 및 준비
    all_data_gpu = preload_all_data_to_gpu(
        db_connection_str,
        start_date,
        end_date,
        use_adjusted_prices=use_adjusted_prices,
        adjusted_price_gate_start_date=adjusted_gate_start_date,
        universe_mode=universe_mode,
    )
    weekly_filtered_gpu = preload_weekly_filtered_stocks_to_gpu(db_connection_str, start_date, end_date)
    
    sql_engine = create_engine(db_connection_str)
    trading_dates_query = f"""
        SELECT DISTINCT date 
        FROM DailyStockPrice 
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    trading_dates_pd_df = pd.read_sql(trading_dates_query, sql_engine, parse_dates=['date'], index_col='date')
    trading_dates_pd = trading_dates_pd_df.index # 이제 DatetimeIndex 객체
    trading_date_indices_gpu = cp.arange(len(trading_dates_pd), dtype=cp.int32)
    
    all_data_gpu = all_data_gpu[all_data_gpu.index.get_level_values('date').isin(trading_dates_pd)]
    all_tickers = sorted(all_data_gpu.index.get_level_values('ticker').unique().to_pandas().tolist())
    print(f"  - Tickers for period: {len(all_tickers)}")
    print(f"  - Trading days for period: {len(trading_dates_pd)}")

    # [Issue #67] Preload Tier Tensor
    tier_tensor = preload_tier_data_to_tensor(
        db_connection_str,
        start_date,
        end_date,
        all_tickers,
        trading_dates_pd,
        universe_mode=universe_mode,
        min_liquidity_20d_avg_value=int(params_dict.get("min_liquidity_20d_avg_value", 0) or 0),
        min_tier12_coverage_ratio=params_dict.get("min_tier12_coverage_ratio"),
    )

    # 3. 백테스팅 커널 실행
    start_time_kernel = time.time()
    
    run_exec_params = _build_run_execution_params(params_dict, universe_mode)
    
    daily_values_result = run_gpu_backtest_kernel(
        param_combinations, 
        all_data_gpu, 
        weekly_filtered_gpu,
        all_tickers, 
        trading_date_indices_gpu,
        trading_dates_pd,
        initial_cash,
        run_exec_params,
        debug_mode=debug_mode,
        tier_tensor=tier_tensor
    )
    end_time_kernel = time.time()
    print(f"  - GPU Kernel Execution Time: {end_time_kernel - start_time_kernel:.2f}s")
    
    # 4. 결과 처리 및 반환
    if daily_values_result is None or daily_values_result.shape[0] == 0:
        print("  - [Warning] Backtest returned no data. Returning empty series.")
        return pd.Series(dtype=float)

    daily_values_cpu = daily_values_result.get()[0] # 첫 번째 (유일한) 시뮬레이션 결과
    equity_curve_series = pd.Series(daily_values_cpu, index=trading_dates_pd)
    
    return equity_curve_series


def _cpu_daily_values_to_series(cpu_result):
    if not cpu_result or not cpu_result.get("success"):
        raise ValueError(f"CPU backtest failed: {cpu_result.get('error') if cpu_result else 'empty result'}")
    daily_values = cpu_result.get("daily_values", [])
    if not daily_values:
        return pd.Series(dtype=float)
    df = pd.DataFrame(daily_values)
    if df.empty or "x" not in df.columns or "y" not in df.columns:
        return pd.Series(dtype=float)
    df["x"] = pd.to_datetime(df["x"])
    df = df.sort_values("x")
    return pd.Series(df["y"].values, index=df["x"])


def run_tier_parity_gate(config_dict, gpu_equity_curve, tolerance=1e-3):
    from src.main_backtest import run_backtest_from_config

    cfg = dict(config_dict)
    cfg["strategy_params"] = dict(cfg.get("strategy_params", {}))
    cfg["strategy_params"]["candidate_source_mode"] = "tier"
    cfg["strategy_params"]["use_weekly_alpha_gate"] = False
    cfg["strategy_params"]["tier_hysteresis_mode"] = normalize_tier_hysteresis_mode(
        cfg["strategy_params"].get("tier_hysteresis_mode", "strict_hysteresis_v1")
    )

    cpu_result = run_backtest_from_config(cfg)
    cpu_curve = _cpu_daily_values_to_series(cpu_result)
    if cpu_curve.empty or gpu_equity_curve.empty:
        raise ValueError("Parity gate input is empty. cpu_curve or gpu_curve has no data.")

    aligned = pd.concat(
        [cpu_curve.rename("cpu"), gpu_equity_curve.rename("gpu")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("Parity gate aligned series is empty.")

    diff = (aligned["cpu"] - aligned["gpu"]).abs()
    mismatches = diff > float(tolerance)
    mismatch_count = int(mismatches.sum())
    print(
        f"[ParityGate] mode=tier points={len(aligned)} "
        f"mismatch_count={mismatch_count} tolerance={tolerance}"
    )
    if mismatch_count > 0:
        first_idx = diff[mismatches].index[0]
        raise AssertionError(
            f"Tier parity gate failed at {pd.to_datetime(first_idx).date()}: "
            f"cpu={aligned.loc[first_idx, 'cpu']}, "
            f"gpu={aligned.loc[first_idx, 'gpu']}, diff={diff.loc[first_idx]}"
        )
    return {"mismatch_count": mismatch_count, "points": len(aligned)}
# -----------------------------------------------------------------------------
# 5. Main Execution Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU single-run debug and optional CPU/GPU parity gate.")
    parser.add_argument("--parity-gate", action="store_true", help="Run tier-only parity gate and fail on mismatch.")
    parser.add_argument("--parity-tolerance", type=float, default=1e-3, help="Absolute tolerance for parity gate.")
    args = parser.parse_args()

    # 이 파일이 단독으로 실행될 때, config.yaml의 설정으로 CPU-GPU 비교 검증을 수행합니다.
    backtest_start_date = backtest_settings['start_date']
    backtest_end_date = backtest_settings['end_date']
    initial_cash = backtest_settings['initial_cash']
    
    print(f"📅 Running Standalone GPU Debug/Verification Run")
    print(f"📅 Period: {backtest_start_date} ~ {backtest_end_date}")
    # config.yaml에서 직접 파라미터 로드
    params_for_debug = config['strategy_params']
    
    # 리팩토링된 워커 함수 호출
    equity_curve = run_single_backtest(
        start_date=backtest_start_date,
        end_date=backtest_end_date,
        params_dict=params_for_debug,
        initial_cash=initial_cash,
        debug_mode=True  # 단독 실행 시에는 항상 상세 로그 출력
    )
    
    # 기존과 동일한 성과 분석 및 출력 로직
    print("\n" + "="*60)
    print("📈 GPU Standalone Run - Performance Summary")
    print("="*60)
    

    if not equity_curve.empty:
        history_df = pd.DataFrame(equity_curve, columns=['total_value'])
        analyzer = PerformanceAnalyzer(history_df)
        metrics = analyzer.get_metrics(formatted=True)

        for key, value in metrics.items():
            print(f"{key:<25}: {value}")
    else:
        print("Error: No backtesting result data to analyze.")

    print("="*60)
    if args.parity_gate:
        run_tier_parity_gate(config, equity_curve, tolerance=args.parity_tolerance)
        print("✅ Tier-only parity gate passed.")

    print(f"\n✅ GPU standalone run and analysis complete!")
