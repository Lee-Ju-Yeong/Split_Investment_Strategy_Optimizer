"""
parameter_simulation_gpu_lib.py

Library implementation for GPU-accelerated parameter simulation.

Issue #60 goal:
- No import-time side effects (no config load, no parameter grid build, no prints).
- Keep standalone execution behavior via `python -m src.parameter_simulation_gpu`.
- Keep public API `find_optimal_parameters(start_date, end_date, initial_cash)` for WFO.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from .config_loader import load_config


PARAM_ORDER: Tuple[str, ...] = (
    "max_stocks",
    "order_investment_ratio",
    "additional_buy_drop_rate",
    "sell_profit_rate",
    "additional_buy_priority",
    "stop_loss_rate",
    "max_splits_limit",
    "max_inactivity_period",
)

PRIORITY_MAP_REV: Dict[int, str] = {0: "lowest_order", 1: "highest_drop"}


def _ensure_core_deps():
    """
    Lazily import core (CPU) deps to keep module import safe even on minimal notebooks.
    """
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Core dependencies are required to run this module. "
            "Install `numpy` and `pandas` in your environment."
        ) from e
    return np, pd


def _raise_missing_gpu_deps(err: Exception) -> None:
    raise ModuleNotFoundError(
        "GPU dependencies are required to run this module. "
        "Activate the RAPIDS/CUDA environment (e.g. `conda activate rapids-env`) "
        "so that `cupy` and `cudf` are available."
    ) from err


def _ensure_gpu_deps():
    """
    Lazily import GPU-only dependencies to keep module import safe on CPU-only notebooks.
    """
    try:
        import cupy as cp  # type: ignore
        import cudf  # type: ignore
    except ModuleNotFoundError as e:
        _raise_missing_gpu_deps(e)

    try:
        from sqlalchemy import create_engine  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "SQLAlchemy is required to run GPU optimization (DB reads). "
            "Install sqlalchemy (and pymysql) in your environment."
        ) from e

    # Note: `src.backtest_strategy_gpu` imports cupy/cudf at module import time,
    # so we must only import it after confirming GPU deps exist.
    from .backtest_strategy_gpu import run_magic_split_strategy_on_gpu

    return cp, cudf, create_engine, run_magic_split_strategy_on_gpu


def _build_db_connection_str(db_config: dict) -> str:
    db_pass_encoded = urllib.parse.quote_plus(db_config["password"])
    return (
        f"mysql+pymysql://{db_config['user']}:{db_pass_encoded}"
        f"@{db_config['host']}/{db_config['database']}"
    )


def _build_param_combinations(param_space_config: dict):
    cp, _, _, _ = _ensure_gpu_deps()
    np, _ = _ensure_core_deps()

    param_options_list = []
    for key in PARAM_ORDER:
        spec = param_space_config[key]
        dtype = (
            np.int32
            if key
            in ["max_stocks", "additional_buy_priority", "max_splits_limit", "max_inactivity_period"]
            else np.float32
        )

        if spec["type"] == "linspace":
            options = np.linspace(spec["start"], spec["stop"], spec["num"], dtype=dtype)
        elif spec["type"] == "list":
            options = np.array(spec["values"], dtype=dtype)
        elif spec["type"] == "range":
            options = np.arange(spec["start"], spec["stop"], spec["step"], dtype=dtype)
        else:
            raise ValueError(f"Unsupported parameter type '{spec['type']}' for '{key}'")

        param_options_list.append(cp.asarray(options))

    grid = cp.meshgrid(*param_options_list)
    param_combinations = cp.vstack([item.flatten() for item in grid]).T
    num_combinations = int(param_combinations.shape[0])
    return param_combinations, num_combinations


@dataclass(frozen=True)
class _SimulationContext:
    config: dict
    backtest_settings: dict
    strategy_params: dict
    execution_params_base: dict
    db_connection_str: str
    param_combinations: Any
    num_combinations: int


@lru_cache(maxsize=1)
def _get_context() -> _SimulationContext:
    """
    Cached, lazy-initialized context.
    Heavy work (config load + parameter grid generation) is done only when needed.
    """
    config = load_config()

    backtest_settings = dict(config["backtest_settings"])
    strategy_params = dict(config["strategy_params"])
    execution_params_base = dict(config["execution_params"])

    db_connection_str = _build_db_connection_str(config["database"])

    param_combinations, num_combinations = _build_param_combinations(config["parameter_space"])
    print(f"✅ Dynamically generated {num_combinations} parameter combinations from config.yaml.")

    return _SimulationContext(
        config=config,
        backtest_settings=backtest_settings,
        strategy_params=strategy_params,
        execution_params_base=execution_params_base,
        db_connection_str=db_connection_str,
        param_combinations=param_combinations,
        num_combinations=num_combinations,
    )


# -----------------------------------------------------------------------------
# GPU Data Pre-loader
# -----------------------------------------------------------------------------
def preload_all_data_to_gpu(engine, start_date, end_date):
    _, cudf, create_engine, _ = _ensure_gpu_deps()
    _, pd = _ensure_core_deps()

    print("⏳ Loading all stock data into GPU memory...")
    start_time = time.time()
    query = f"""
    SELECT
        dsp.stock_code AS ticker,
        dsp.date,
        dsp.open_price,
        dsp.high_price,
        dsp.low_price,
        dsp.close_price,
        dsp.volume,
        ci.atr_14_ratio
    FROM
        DailyStockPrice AS dsp
    LEFT JOIN
        CalculatedIndicators AS ci ON dsp.stock_code = ci.stock_code AND dsp.date = ci.date
    WHERE
        dsp.date BETWEEN '{start_date}' AND '{end_date}'
    """
    sql_engine = create_engine(engine)
    df_pd = pd.read_sql(query, sql_engine, parse_dates=["date"])
    gdf = cudf.from_pandas(df_pd).set_index(["ticker", "date"])
    print(f"✅ Data loaded to GPU. Shape: {gdf.shape}. Time: {time.time() - start_time:.2f}s")
    return gdf


def preload_weekly_filtered_stocks_to_gpu(engine, start_date, end_date):
    _, cudf, create_engine, _ = _ensure_gpu_deps()
    _, pd = _ensure_core_deps()

    print("⏳ Loading weekly filtered stocks data to GPU memory...")
    start_time = time.time()
    extended_start_date = pd.to_datetime(start_date) - timedelta(days=14)
    query = (
        "SELECT `filter_date` as date, `stock_code` as ticker "
        "FROM WeeklyFilteredStocks "
        f"WHERE `filter_date` BETWEEN '{extended_start_date.strftime('%Y-%m-%d')}' AND '{end_date}'"
    )
    sql_engine = create_engine(engine)
    df_pd = pd.read_sql(query, sql_engine, parse_dates=["date"])
    gdf = cudf.from_pandas(df_pd).set_index("date")
    print(f"✅ Weekly filtered stocks loaded to GPU. Shape: {gdf.shape}. Time: {time.time() - start_time:.2f}s")
    return gdf


def preload_tier_data_to_tensor(engine, start_date, end_date, all_tickers, trading_dates_pd):
    """
    Loads DailyStockTier data and converts it to a dense (num_days, num_tickers) int8 tensor.
    Performs forward-fill to ensure PIT compliance (latest <= date).
    """
    cp, _, create_engine, _ = _ensure_gpu_deps()
    _, pd = _ensure_core_deps()

    print("⏳ Loading DailyStockTier data to GPU tensor...")
    start_time = time.time()

    start_date_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    query = f"""
        SELECT date, stock_code as ticker, tier
        FROM DailyStockTier
        WHERE date BETWEEN '{start_date_str}' AND '{end_date_str}'
        UNION ALL
        SELECT t.date, t.stock_code as ticker, t.tier
        FROM DailyStockTier t
        JOIN (
            SELECT stock_code, MAX(date) AS max_date
            FROM DailyStockTier
            WHERE date < '{start_date_str}'
            GROUP BY stock_code
        ) latest ON t.stock_code = latest.stock_code AND t.date = latest.max_date
    """
    sql_engine = create_engine(engine)
    df_pd = pd.read_sql(query, sql_engine, parse_dates=["date"])

    if df_pd.empty:
        print("⚠️ No Tier data found. Returning empty tensor.")
        return cp.zeros((len(trading_dates_pd), len(all_tickers)), dtype=cp.int8)

    df_wide = df_pd.pivot_table(index="date", columns="ticker", values="tier")
    df_reindexed = df_wide.reindex(index=trading_dates_pd, columns=all_tickers).ffill().fillna(0).astype(int)
    tier_tensor = cp.asarray(df_reindexed.values, dtype=cp.int8)

    print(f"✅ Tier data loaded and tensorized. Shape: {tier_tensor.shape}. Time: {time.time() - start_time:.2f}s")
    return tier_tensor


# -----------------------------------------------------------------------------
# GPU Backtesting Kernel Orchestrator
# -----------------------------------------------------------------------------
def run_gpu_optimization(
    params_gpu,
    data_gpu,
    weekly_filtered_gpu,
    all_tickers,
    trading_date_indices_gpu,
    trading_dates_pd,
    initial_cash_value,
    exec_params,
    tier_tensor=None,
):
    cp, _, _, run_magic_split_strategy_on_gpu = _ensure_gpu_deps()

    print("🚀 Starting GPU backtesting kernel...")
    max_splits_from_params = int(cp.max(params_gpu[:, 6]).get())
    daily_portfolio_values = run_magic_split_strategy_on_gpu(
        initial_cash=initial_cash_value,
        param_combinations=params_gpu,
        all_data_gpu=data_gpu,
        weekly_filtered_gpu=weekly_filtered_gpu,
        trading_date_indices=trading_date_indices_gpu,
        trading_dates_pd_cpu=trading_dates_pd,
        all_tickers=all_tickers,
        execution_params=exec_params,
        max_splits_limit=max_splits_from_params,
        tier_tensor=tier_tensor,
    )
    print("🎉 GPU backtesting kernel finished.")
    return daily_portfolio_values


# -----------------------------------------------------------------------------
# Analysis and Result Saving
# -----------------------------------------------------------------------------
def analyze_and_save_results(param_combinations_gpu, daily_values_gpu, trading_dates_pd, save_to_file=True):
    np, pd = _ensure_core_deps()
    from .performance_analyzer import PerformanceAnalyzer

    print("\n--- 🔬 Analyzing detailed performance metrics ---")
    start_time = time.time()

    param_combinations_cpu = param_combinations_gpu.get()
    daily_values_cpu = daily_values_gpu.get()

    results_list = []
    for i in range(daily_values_cpu.shape[0]):
        history_df_mock = pd.DataFrame(
            pd.Series(daily_values_cpu[i], index=trading_dates_pd),
            columns=["total_value"],
        )
        analyzer = PerformanceAnalyzer(history_df_mock)
        results_list.append(analyzer.get_metrics(formatted=False))

    param_names = list(PARAM_ORDER)
    params_df = pd.DataFrame(param_combinations_cpu, columns=param_names)
    metrics_df = pd.DataFrame(results_list)
    full_results_df = pd.concat([params_df, metrics_df], axis=1)

    full_results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sorted_df = full_results_df.sort_values(by="calmar_ratio", ascending=False).dropna(subset=["calmar_ratio"])

    if not sorted_df.empty:
        best_params_dict = sorted_df.iloc[0].to_dict()
    else:
        best_params_dict = {}

    print("\n🏆 Top 10 Performing Parameter Combinations (by Calmar Ratio):")
    display_columns = [
        "calmar_ratio",
        "cagr",
        "mdd",
        "sharpe_ratio",
        "stop_loss_rate",
        "max_splits_limit",
        "max_inactivity_period",
        "sell_profit_rate",
        "additional_buy_drop_rate",
    ]
    display_df = sorted_df.head(10).get(display_columns, pd.DataFrame())
    if not display_df.empty:
        for col in ["cagr", "mdd", "annualized_volatility", "stop_loss_rate"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map("{:.2%}".format)
        for col in ["calmar_ratio", "sharpe_ratio", "sortino_ratio"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map("{:.2f}".format)
        print(display_df.to_string(index=False))

    if save_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"gpu_simulation_results_{timestamp}.csv")
        sorted_df.to_csv(filepath, index=False, float_format="%.4f")
        print(f"\n✅ Full analysis saved to: {filepath}")

    print(f"⏱️  Analysis took: {time.time() - start_time:.2f} seconds.")
    return best_params_dict, sorted_df


# -----------------------------------------------------------------------------
# Optimal Batch Size Calculation
# -----------------------------------------------------------------------------
def get_optimal_batch_size(config, num_tickers, fixed_data_memory_bytes, safety_factor=0.9):
    """
    현재 가용 GPU 메모리를 기반으로 최적의 시뮬레이션 배치 크기를 계산합니다.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        free_memory_mib = int(result.stdout.strip())
        free_memory_bytes = free_memory_mib * 1024 * 1024

        p_space = config["parameter_space"]
        max_stocks = (
            max(p_space["max_stocks"]["values"])
            if p_space["max_stocks"]["type"] == "list"
            else int(p_space["max_stocks"]["stop"])
        )
        max_splits = (
            max(p_space["max_splits_limit"]["values"])
            if p_space["max_splits_limit"]["type"] == "list"
            else int(p_space["max_splits_limit"]["stop"])
        )

        portfolio_state_per_sim = 4 * 4
        positions_state_per_sim = max_stocks * max_splits * 6 * 4
        buy_signals_per_sim = num_tickers * 1
        sell_signals_per_sim = max_stocks * 1

        estimated_mem_per_sim = (
            portfolio_state_per_sim + positions_state_per_sim + buy_signals_per_sim + sell_signals_per_sim
        )
        estimated_mem_per_sim_with_buffer = estimated_mem_per_sim * 1.2  # 20% buffer

        usable_memory = (free_memory_bytes * safety_factor) - fixed_data_memory_bytes
        if usable_memory <= 0:
            raise ValueError("Not enough free memory for simulations.")

        optimal_size = int(usable_memory / estimated_mem_per_sim_with_buffer)

        print("\n--- 📊 Optimal Batch Size Calculation ---")
        print(f"  - Available GPU Memory   : {free_memory_mib} MiB")
        print(f"  - Memory for Fixed Data  : {fixed_data_memory_bytes / (1024 * 1024):.2f} MiB")
        print(f"  - Usable Memory (90% SF) : {usable_memory / (1024 * 1024):.2f} MiB")
        print(f"  - Estimated Mem/Sim (20% Buf): {estimated_mem_per_sim_with_buffer / 1024:.2f} KB")
        print(f"  - Calculated Batch Size  : {usable_memory:.2f} / {estimated_mem_per_sim_with_buffer:.2f} = {optimal_size}")
        print("----------------------------------------\n")

        if optimal_size <= 0:
            raise ValueError("Calculated optimal size is zero or negative.")

        return optimal_size

    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
        print(f"⚠️  Could not execute nvidia-smi or calculate optimal batch size: {e}")
        return None


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

    fixed_mem = int(all_data_gpu.memory_usage(deep=True).sum() + weekly_filtered_gpu.memory_usage(deep=True).sum())
    fixed_mem += int(tier_tensor.nbytes)

    optimal_batch_size = get_optimal_batch_size(config, len(all_tickers), fixed_mem)

    if optimal_batch_size:
        batch_size = min(optimal_batch_size, ctx.num_combinations)
        print(f"✅ Using automatically calculated optimal batch size: {batch_size}")
    else:
        batch_size = backtest_settings.get("simulation_batch_size")
        if batch_size is None or batch_size <= 0:
            batch_size = ctx.num_combinations
        print(f"⚠️ Using fallback batch size from config: {batch_size}")

    num_batches = (ctx.num_combinations + batch_size - 1) // batch_size
    print(f"  - Total Simulations: {ctx.num_combinations} | Batch Size: {batch_size} | Batches: {num_batches}")

    all_daily_values_list = []
    total_kernel_time = 0.0
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, ctx.num_combinations)
        param_batch = ctx.param_combinations[start_idx:end_idx]

        print(f"\n  --- Running Batch {i + 1}/{num_batches} (Sims {start_idx}-{end_idx - 1}) ---")

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
        print(f"  - Batch {i + 1} Kernel Execution Time: {batch_time:.2f}s")

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
