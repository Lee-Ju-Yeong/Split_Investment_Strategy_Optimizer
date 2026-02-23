"""
High-level GPU parameter simulation orchestration.

Issue #69 (PR-9):
- Split legacy `src.parameter_simulation_gpu_lib` into focused modules under
  `src/optimization/gpu/*` while preserving behavior.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

from .analysis import analyze_and_save_results
from .context import PRIORITY_MAP_REV, _ensure_core_deps, _ensure_gpu_deps, _get_context
from .data_loading import (
    preload_all_data_to_gpu,
    preload_tier_data_to_tensor,
    preload_weekly_filtered_stocks_to_gpu,
)
from .kernel import get_optimal_batch_size, run_gpu_optimization


DEFAULT_FALLBACK_TARGET_BATCHES = 8
DEFAULT_FALLBACK_MIN_BATCH_SIZE = 256
DEFAULT_FALLBACK_MAX_BATCH_SIZE = 2048


def _to_positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _resolve_adaptive_fallback_batch_size(num_combinations):
    if num_combinations <= 0:
        return 1

    target_batches = DEFAULT_FALLBACK_TARGET_BATCHES
    adaptive = (num_combinations + target_batches - 1) // target_batches
    adaptive = max(adaptive, DEFAULT_FALLBACK_MIN_BATCH_SIZE)
    adaptive = min(adaptive, DEFAULT_FALLBACK_MAX_BATCH_SIZE)
    return min(adaptive, num_combinations)


def _resolve_batch_size(optimal_batch_size, backtest_settings, num_combinations):
    if optimal_batch_size:
        batch_size = min(int(optimal_batch_size), num_combinations)
        return batch_size, "auto"

    configured_batch_size = _to_positive_int(backtest_settings.get("simulation_batch_size"))
    if configured_batch_size:
        batch_size = min(configured_batch_size, num_combinations)
        return batch_size, "config.simulation_batch_size"

    batch_size = _resolve_adaptive_fallback_batch_size(num_combinations)
    return batch_size, "adaptive-safe-default"


# -----------------------------------------------------------------------------
# Worker: find_optimal_parameters
# -----------------------------------------------------------------------------
def find_optimal_parameters(start_date: str, end_date: str, initial_cash: float):
    """
    주어진 기간 동안 GPU를 사용하여 파라미터 최적화를 실행하고,
    '전체 시뮬레이션 결과'를 DataFrame으로 반환합니다.
    (WFO 오케스트레이터가 이 결과를 받아 분석을 수행합니다.)
    """
    cp, _, create_engine, _ = _ensure_gpu_deps()
    _, pd = _ensure_core_deps()

    ctx = _get_context()
    config = ctx.config
    db_connection_str = ctx.db_connection_str
    backtest_settings = ctx.backtest_settings
    strategy_params = ctx.strategy_params

    # Avoid mutating cached dicts.
    execution_params = dict(ctx.execution_params_base)
    execution_params["cooldown_period_days"] = strategy_params.get("cooldown_period_days", 5)

    print("\n" + "=" * 80)
    print(f"WORKER: Running GPU Simulations for {start_date} to {end_date}")
    print("=" * 80)

    all_data_gpu = preload_all_data_to_gpu(db_connection_str, start_date, end_date)
    weekly_filtered_gpu = preload_weekly_filtered_stocks_to_gpu(db_connection_str, start_date, end_date)

    sql_engine = create_engine(db_connection_str)
    trading_dates_query = f"""
        SELECT DISTINCT date
        FROM DailyStockPrice
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    trading_dates_pd_df = pd.read_sql(trading_dates_query, sql_engine, parse_dates=["date"], index_col="date")
    trading_dates_pd = trading_dates_pd_df.index
    trading_date_indices_gpu = cp.arange(len(trading_dates_pd), dtype=cp.int32)

    all_data_gpu = all_data_gpu[all_data_gpu.index.get_level_values("date").isin(trading_dates_pd)]
    all_tickers = sorted(all_data_gpu.index.get_level_values("ticker").unique().to_pandas().tolist())

    print(f"  - Tickers for period: {len(all_tickers)}")
    print(f"  - Trading days for period: {len(trading_dates_pd)}")

    tier_tensor = preload_tier_data_to_tensor(db_connection_str, start_date, end_date, all_tickers, trading_dates_pd)

    execution_params["candidate_source_mode"] = strategy_params.get("candidate_source_mode", "weekly")
    execution_params["use_weekly_alpha_gate"] = strategy_params.get("use_weekly_alpha_gate", False)
    execution_params["tier_hysteresis_mode"] = strategy_params.get("tier_hysteresis_mode", "legacy")

    fixed_mem = int(all_data_gpu.memory_usage(deep=True).sum() + weekly_filtered_gpu.memory_usage(deep=True).sum())
    fixed_mem += int(tier_tensor.nbytes)

    optimal_batch_size = get_optimal_batch_size(config, len(all_tickers), fixed_mem)
    batch_size, batch_size_source = _resolve_batch_size(
        optimal_batch_size=optimal_batch_size,
        backtest_settings=backtest_settings,
        num_combinations=ctx.num_combinations,
    )
    if batch_size <= 0:
        raise ValueError(f"Invalid batch size resolved: {batch_size}")

    if batch_size_source == "auto":
        print(f"✅ Using automatically calculated optimal batch size: {batch_size}")
    else:
        print(
            "⚠️ Using fallback batch size "
            f"({batch_size_source}): {batch_size}"
        )

    num_batches = (ctx.num_combinations + batch_size - 1) // batch_size
    print(f"  - Total Simulations: {ctx.num_combinations} | Batch Size: {batch_size} | Batches: {num_batches}")

    all_daily_values_list = []
    total_kernel_time = 0.0
    for idx in range(num_batches):
        start_idx = idx * batch_size
        end_idx = min((idx + 1) * batch_size, ctx.num_combinations)
        param_batch = ctx.param_combinations[start_idx:end_idx]

        print(f"\n  --- Running Batch {idx + 1}/{num_batches} (Sims {start_idx}-{end_idx - 1}) ---")

        start_time_kernel = time.time()
        daily_values_batch = run_gpu_optimization(
            param_batch,
            all_data_gpu,
            weekly_filtered_gpu,
            all_tickers,
            trading_date_indices_gpu,
            trading_dates_pd,
            initial_cash,
            execution_params,
            tier_tensor=tier_tensor,
        )
        batch_time = time.time() - start_time_kernel
        total_kernel_time += batch_time
        print(f"  - Batch {idx + 1} Kernel Execution Time: {batch_time:.2f}s")

        all_daily_values_list.append(daily_values_batch)

    print(f"\n  - Total GPU Kernel Execution Time: {total_kernel_time:.2f}s")
    if not all_daily_values_list:
        print("[Error] No simulation results were generated.")
        return {}, pd.DataFrame()

    daily_values_result = cp.vstack(all_daily_values_list)

    best_params_for_log, all_results_df = analyze_and_save_results(
        ctx.param_combinations,
        daily_values_result,
        trading_dates_pd,
        save_to_file=False,
    )

    if "additional_buy_priority" in best_params_for_log:
        best_params_for_log["additional_buy_priority"] = PRIORITY_MAP_REV.get(
            int(best_params_for_log.get("additional_buy_priority", -1)),
            "unknown",
        )

    return best_params_for_log, all_results_df


def main() -> None:
    """
    Standalone execution entry point (kept for `python -m src.parameter_simulation_gpu`).
    """
    ctx = _get_context()
    backtest_start_date = ctx.backtest_settings["start_date"]
    backtest_end_date = ctx.backtest_settings["end_date"]
    initial_cash = ctx.backtest_settings["initial_cash"]

    print("\n" + "=" * 80)
    print(" 실행 모드: 단독 파라미터 최적화 (STANDALONE OPTIMIZATION MODE)")
    print("=" * 80)
    print(" 이 스크립트는 아래 명시된 '단일 고정 기간'에 대해서만 1회 최적화를 수행합니다.")
    print(f"  - 최적화 대상 기간: {backtest_start_date} ~ {backtest_end_date}")

    wfo_settings = ctx.config.get("walk_forward_settings")
    if wfo_settings and wfo_settings.get("total_folds"):
        total_folds = wfo_settings.get("total_folds")
        print("\n [정보] 전체 Walk-Forward 분석을 실행하시려면 아래 명령어를 사용하십시오.")
        print("  - 명령어: python -m src.walk_forward_analyzer")
        print(f"  - 예상 Fold 수: {total_folds} folds")
    print("=" * 80 + "\n")

    best_parameters_found, all_results_df = find_optimal_parameters(
        start_date=backtest_start_date,
        end_date=backtest_end_date,
        initial_cash=initial_cash,
    )

    print("\n" + "=" * 80)
    print("🏆 STANDALONE RUN - BEST PARAMETERS (by Calmar Ratio) 🏆")
    print("=" * 80)
    if best_parameters_found:
        for key, value in best_parameters_found.items():
            if isinstance(value, float):
                print(f"  - {key:<25}: {value:.4f}")
            else:
                print(f"  - {key:<25}: {value}")
    else:
        print("  No valid parameters found.")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"standalone_simulation_results_{timestamp}.csv")
    all_results_df.to_csv(filepath, index=False, float_format="%.4f")
    print(f"\n✅ Full simulation analysis saved to: {filepath}")


__all__ = [
    "find_optimal_parameters",
    "main",
]
