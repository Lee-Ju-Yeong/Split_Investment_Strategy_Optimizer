"""
GPU kernel execution and batch size helpers for parameter simulation.
"""

from __future__ import annotations

import subprocess

from .context import _ensure_gpu_deps


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
# Optimal Batch Size Calculation
# -----------------------------------------------------------------------------
def _query_free_memory_with_nvidia_smi() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    output = result.stdout.strip()
    if not output:
        return None
    return int(output.splitlines()[0]) * 1024 * 1024


def _query_free_memory_with_cupy_runtime() -> int | None:
    try:
        cp, _, _, _ = _ensure_gpu_deps()
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
    except Exception:
        return None
    return int(free_bytes)


def _resolve_free_gpu_memory_bytes() -> tuple[int | None, str]:
    nvidia_smi_bytes = _query_free_memory_with_nvidia_smi()
    if nvidia_smi_bytes is not None:
        return nvidia_smi_bytes, "nvidia-smi"

    runtime_bytes = _query_free_memory_with_cupy_runtime()
    if runtime_bytes is not None:
        return runtime_bytes, "cupy.runtime.memGetInfo"

    return None, "unavailable"


def get_optimal_batch_size(config, num_tickers, fixed_data_memory_bytes, safety_factor=0.9):
    """
    현재 가용 GPU 메모리를 기반으로 최적의 시뮬레이션 배치 크기를 계산합니다.
    """
    try:
        free_memory_bytes, memory_source = _resolve_free_gpu_memory_bytes()
        if free_memory_bytes is None:
            raise ValueError("Unable to resolve free GPU memory from nvidia-smi/cupy runtime.")

        free_memory_mib = free_memory_bytes // (1024 * 1024)

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
        print(f"  - Memory Source          : {memory_source}")
        print(f"  - Available GPU Memory   : {free_memory_mib} MiB")
        print(f"  - Memory for Fixed Data  : {fixed_data_memory_bytes / (1024 * 1024):.2f} MiB")
        print(f"  - Usable Memory (90% SF) : {usable_memory / (1024 * 1024):.2f} MiB")
        print(f"  - Estimated Mem/Sim (20% Buf): {estimated_mem_per_sim_with_buffer / 1024:.2f} KB")
        print(f"  - Calculated Batch Size  : {usable_memory:.2f} / {estimated_mem_per_sim_with_buffer:.2f} = {optimal_size}")
        print("----------------------------------------\n")

        if optimal_size <= 0:
            raise ValueError("Calculated optimal size is zero or negative.")

        return optimal_size

    except ValueError as err:
        print(f"⚠️  Could not calculate optimal batch size: {err}")
        return None


__all__ = [
    "run_gpu_optimization",
    "get_optimal_batch_size",
]
