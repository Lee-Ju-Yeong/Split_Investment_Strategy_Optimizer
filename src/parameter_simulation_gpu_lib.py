"""
parameter_simulation_gpu_lib.py

Thin wrapper module.

Issue #69 (PR-9):
- Keep this module as a backward-compatible wrapper.
- Move implementation into `src.optimization.gpu.*`.

Compatibility:
- Supports `import src.parameter_simulation_gpu_lib` (package import)
- Supports `import parameter_simulation_gpu_lib` when `src/` is on `sys.path` (legacy style)
"""

from __future__ import annotations

if __package__:
    # Preferred: imported as `src.parameter_simulation_gpu_lib`
    from .optimization.gpu.analysis import analyze_and_save_results
    from .optimization.gpu.context import (
        PARAM_ORDER,
        PRIORITY_MAP_REV,
        _build_db_connection_str,
        _build_param_combinations,
        _ensure_core_deps,
        _ensure_gpu_deps,
        _get_context,
        _raise_missing_gpu_deps,
        _SimulationContext,
    )
    from .optimization.gpu.data_loading import (
        preload_all_data_to_gpu,
        preload_tier_data_to_tensor,
        preload_weekly_filtered_stocks_to_gpu,
    )
    from .optimization.gpu.kernel import get_optimal_batch_size, run_gpu_optimization
    from .optimization.gpu.parameter_simulation import find_optimal_parameters, main
else:  # pragma: no cover
    # Legacy: imported as top-level `parameter_simulation_gpu_lib` with `src/` on sys.path.
    from optimization.gpu.analysis import analyze_and_save_results
    from optimization.gpu.context import (
        PARAM_ORDER,
        PRIORITY_MAP_REV,
        _build_db_connection_str,
        _build_param_combinations,
        _ensure_core_deps,
        _ensure_gpu_deps,
        _get_context,
        _raise_missing_gpu_deps,
        _SimulationContext,
    )
    from optimization.gpu.data_loading import (
        preload_all_data_to_gpu,
        preload_tier_data_to_tensor,
        preload_weekly_filtered_stocks_to_gpu,
    )
    from optimization.gpu.kernel import get_optimal_batch_size, run_gpu_optimization
    from optimization.gpu.parameter_simulation import find_optimal_parameters, main

__all__ = [
    "PARAM_ORDER",
    "PRIORITY_MAP_REV",
    "_ensure_core_deps",
    "_raise_missing_gpu_deps",
    "_ensure_gpu_deps",
    "_build_db_connection_str",
    "_build_param_combinations",
    "_SimulationContext",
    "_get_context",
    "preload_all_data_to_gpu",
    "preload_weekly_filtered_stocks_to_gpu",
    "preload_tier_data_to_tensor",
    "run_gpu_optimization",
    "analyze_and_save_results",
    "get_optimal_batch_size",
    "find_optimal_parameters",
    "main",
]
