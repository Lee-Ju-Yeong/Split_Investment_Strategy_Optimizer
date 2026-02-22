"""
GPU kernel execution and batch size helpers for parameter simulation.
"""

from __future__ import annotations

import subprocess

from .context import _ensure_gpu_deps

BYTES_FLOAT32 = 4
BYTES_INT32 = 4
BYTES_BOOL = 1

DEFAULT_FREE_MEMORY_MARGIN_FACTOR = 0.95
DEFAULT_GPU_RESERVED_BYTES = 256 * 1024 * 1024
DEFAULT_PER_SIM_PEAK_BUFFER_FACTOR = 1.25

SELL_2D_TEMP_FLOAT32_COUNT = 2
SELL_3D_TEMP_FLOAT32_COUNT = 2
ADDITIONAL_BUY_TEMP_FLOAT32_COUNT = 4


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
    runtime_bytes = _query_free_memory_with_cupy_runtime()

    if nvidia_smi_bytes is not None and runtime_bytes is not None:
        return min(nvidia_smi_bytes, runtime_bytes), "min(nvidia-smi,cupy.runtime.memGetInfo)"
    if nvidia_smi_bytes is not None:
        return nvidia_smi_bytes, "nvidia-smi"
    if runtime_bytes is not None:
        return runtime_bytes, "cupy.runtime.memGetInfo"

    return None, "unavailable"


def _estimate_per_sim_memory_bytes(*, num_tickers: int, num_trading_days: int, max_splits: int) -> int:
    portfolio_state_per_sim = 2 * BYTES_FLOAT32
    positions_state_per_sim = num_tickers * max_splits * 3 * BYTES_FLOAT32
    cooldown_state_per_sim = num_tickers * BYTES_INT32
    last_trade_day_per_sim = num_tickers * BYTES_INT32
    daily_values_per_sim = num_trading_days * BYTES_FLOAT32

    sell_2d_temp_per_sim = num_tickers * SELL_2D_TEMP_FLOAT32_COUNT * BYTES_FLOAT32
    sell_3d_temp_per_sim = num_tickers * max_splits * SELL_3D_TEMP_FLOAT32_COUNT * BYTES_FLOAT32
    additional_buy_temp_per_sim = num_tickers * ADDITIONAL_BUY_TEMP_FLOAT32_COUNT * BYTES_FLOAT32
    additional_buy_mask_per_sim = num_tickers * max_splits * 2 * BYTES_BOOL

    estimated_mem_per_sim = (
        portfolio_state_per_sim
        + positions_state_per_sim
        + cooldown_state_per_sim
        + last_trade_day_per_sim
        + daily_values_per_sim
        + sell_2d_temp_per_sim
        + sell_3d_temp_per_sim
        + additional_buy_temp_per_sim
        + additional_buy_mask_per_sim
    )
    return int(estimated_mem_per_sim * DEFAULT_PER_SIM_PEAK_BUFFER_FACTOR)


def get_optimal_batch_size(
    config,
    num_tickers,
    num_trading_days,
    max_splits,
    fixed_data_memory_bytes,
    safety_factor=0.85,
):
    """
    현재 가용 GPU 메모리를 기반으로 최적의 시뮬레이션 배치 크기를 계산합니다.
    """
    try:
        free_memory_bytes, memory_source = _resolve_free_gpu_memory_bytes()
        if free_memory_bytes is None:
            raise ValueError("Unable to resolve free GPU memory from nvidia-smi/cupy runtime.")

        free_memory_mib = free_memory_bytes // (1024 * 1024)

        estimated_mem_per_sim_with_buffer = _estimate_per_sim_memory_bytes(
            num_tickers=num_tickers,
            num_trading_days=num_trading_days,
            max_splits=max_splits,
        )
        effective_free_memory = int(free_memory_bytes * DEFAULT_FREE_MEMORY_MARGIN_FACTOR)
        usable_memory = int(effective_free_memory * safety_factor) - fixed_data_memory_bytes - DEFAULT_GPU_RESERVED_BYTES
        if usable_memory <= 0:
            raise ValueError("Not enough free memory for simulations.")

        optimal_size = int(usable_memory / estimated_mem_per_sim_with_buffer)

        print("\n--- 📊 Optimal Batch Size Calculation ---")
        print(f"  - Memory Source          : {memory_source}")
        print(f"  - Available GPU Memory   : {free_memory_mib} MiB")
        print(
            f"  - Effective Free Memory  : {effective_free_memory / (1024 * 1024):.2f} MiB "
            f"(margin={DEFAULT_FREE_MEMORY_MARGIN_FACTOR:.2f})"
        )
        print(f"  - GPU Reserved Memory    : {DEFAULT_GPU_RESERVED_BYTES / (1024 * 1024):.2f} MiB")
        print(f"  - Memory for Fixed Data  : {fixed_data_memory_bytes / (1024 * 1024):.2f} MiB")
        print(f"  - Usable Memory ({safety_factor:.2f} SF): {usable_memory / (1024 * 1024):.2f} MiB")
        print(
            "  - Estimated Mem/Sim "
            f"(peak-buffered): {estimated_mem_per_sim_with_buffer / 1024:.2f} KB"
        )
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
