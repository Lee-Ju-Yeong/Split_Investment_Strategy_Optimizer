"""
Shared context/dependency helpers for GPU parameter simulation.

Issue #69 (PR-9):
- Keep heavy operations lazy (config load, GPU import, param grid build).
- Provide a cached simulation context for runtime entrypoints.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Tuple

try:
    from ...config_loader import load_config
except ImportError:  # pragma: no cover
    # Legacy mode: imported as top-level `optimization.*` with `src/` on sys.path.
    from config_loader import load_config


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
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "Core dependencies are required to run this module. "
            "Install `numpy` and `pandas` in your environment."
        ) from err
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
    except ModuleNotFoundError as err:
        _raise_missing_gpu_deps(err)

    try:
        from sqlalchemy import create_engine  # type: ignore
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "SQLAlchemy is required to run GPU optimization (DB reads). "
            "Install sqlalchemy (and pymysql) in your environment."
        ) from err

    # Note: `src.backtest.gpu.engine` imports cupy/cudf at module import time,
    # so we must only import it after confirming GPU deps exist.
    try:
        from ...backtest.gpu.engine import run_magic_split_strategy_on_gpu
    except ImportError:  # pragma: no cover
        # Legacy mode: imported as top-level `optimization.*` with `src/` on sys.path.
        from backtest.gpu.engine import run_magic_split_strategy_on_gpu

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
            if key in ["max_stocks", "additional_buy_priority", "max_splits_limit", "max_inactivity_period"]
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
]
