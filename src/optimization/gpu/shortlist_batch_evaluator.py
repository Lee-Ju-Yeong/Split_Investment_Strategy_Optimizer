"""
GPU batch evaluator for frozen shortlist candidate evaluation.

This module is intentionally scoped to "evaluate a small fixed candidate list
for one backtest window" so that WFO research lanes can reuse one-time market
data preparation without changing selection semantics.
"""

from __future__ import annotations

import os
import time
import urllib.parse
from typing import TYPE_CHECKING

from ...gpu_execution_policy import build_gpu_execution_params
from ...price_policy import (
    is_adjusted_price_basis,
    resolve_price_policy,
    validate_backtest_window_for_price_policy,
)
from ...universe_policy import resolve_universe_mode
from .kernel import get_optimal_batch_size, prepare_market_data_bundle, run_gpu_optimization
from .parameter_simulation import (
    _is_gpu_oom_error,
    _measure_prepared_market_data_bytes,
    _release_gpu_memory,
    _resolve_batch_size,
)

if TYPE_CHECKING:
    import pandas as pd


_RUNTIME_CONTRACT_KEYS = (
    "universe_mode",
    "price_basis",
    "adjusted_price_gate_start_date",
    "candidate_source_mode",
    "use_weekly_alpha_gate",
    "tier_hysteresis_mode",
    "min_liquidity_20d_avg_value",
    "min_tier12_coverage_ratio",
)

_PRIORITY_MAP = {
    "lowest_order": 0,
    "highest_drop": 1,
    "biggest_drop": 1,
}


def _build_db_connection_str(db_config: dict) -> str:
    password = urllib.parse.quote_plus(str(db_config["password"]))
    return (
        f"mysql+pymysql://{db_config['user']}:{password}"
        f"@{db_config['host']}/{db_config['database']}"
    )


def _normalize_priority_value(value) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _PRIORITY_MAP:
            return int(_PRIORITY_MAP[key])
        if key in {"0", "1"}:
            return int(key)
    try:
        numeric = int(float(value))
    except (TypeError, ValueError) as err:
        raise ValueError(f"Unsupported additional_buy_priority={value!r}") from err
    if numeric not in (0, 1):
        raise ValueError(f"Unsupported additional_buy_priority={value!r}")
    return numeric


def _normalize_priority_label(value) -> str:
    return "highest_drop" if _normalize_priority_value(value) == 1 else "lowest_order"


def _build_runtime_contract(
    params_dict: dict,
    *,
    start_date: str,
    end_date: str,
    execution_params_base: dict,
) -> tuple[dict, dict]:
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
    run_exec_params = build_gpu_execution_params(
        execution_params_base,
        params_dict,
        universe_mode,
        default_tier_hysteresis_mode=str(
            params_dict.get("tier_hysteresis_mode", "strict_hysteresis_v1")
        ),
    )
    contract = {
        "universe_mode": universe_mode,
        "price_basis": price_basis,
        "adjusted_price_gate_start_date": adjusted_gate_start_date,
        "candidate_source_mode": run_exec_params.get("candidate_source_mode"),
        "use_weekly_alpha_gate": bool(run_exec_params.get("use_weekly_alpha_gate", False)),
        "tier_hysteresis_mode": run_exec_params.get("tier_hysteresis_mode"),
        "min_liquidity_20d_avg_value": int(
            params_dict.get("min_liquidity_20d_avg_value", 0) or 0
        ),
        "min_tier12_coverage_ratio": params_dict.get("min_tier12_coverage_ratio"),
    }
    return contract, run_exec_params


def _ensure_homogeneous_runtime_contracts(
    params_list: list[dict],
    *,
    start_date: str,
    end_date: str,
    execution_params_base: dict,
) -> tuple[dict, dict]:
    first_contract = None
    first_exec_params = None
    mismatches = set()
    for params_dict in params_list:
        contract, run_exec_params = _build_runtime_contract(
            params_dict,
            start_date=start_date,
            end_date=end_date,
            execution_params_base=execution_params_base,
        )
        if first_contract is None:
            first_contract = contract
            first_exec_params = run_exec_params
            continue
        for key in _RUNTIME_CONTRACT_KEYS:
            if first_contract.get(key) != contract.get(key):
                mismatches.add(key)
    if mismatches:
        mismatch_list = ",".join(sorted(mismatches))
        raise ValueError(
            "GPU shortlist batch evaluation requires a homogeneous runtime contract. "
            f"Mismatched fields: {mismatch_list}"
        )
    if first_contract is None or first_exec_params is None:
        raise ValueError("No candidate runtime contract resolved for shortlist batch.")
    return first_contract, first_exec_params


def _build_param_combinations(params_list: list[dict]):
    import cupy as cp

    rows = []
    for params_dict in params_list:
        rows.append(
            [
                int(params_dict["max_stocks"]),
                float(params_dict["order_investment_ratio"]),
                float(params_dict["additional_buy_drop_rate"]),
                float(params_dict["sell_profit_rate"]),
                _normalize_priority_value(
                    params_dict.get("additional_buy_priority", "lowest_order")
                ),
                float(params_dict["stop_loss_rate"]),
                int(params_dict["max_splits_limit"]),
                int(params_dict["max_inactivity_period"]),
            ]
        )
    if not rows:
        return cp.empty((0, 8), dtype=cp.float32)
    return cp.asarray(rows, dtype=cp.float32)


def _build_evaluated_shortlist_frame(
    shortlist_df,
    *,
    params_list: list[dict],
    daily_values_result_cpu,
    trading_dates_pd,
    analyzer_cls,
):
    import pandas as pd

    rows = []
    for row_idx, (_, shortlist_row) in enumerate(shortlist_df.iterrows()):
        equity_curve = pd.Series(daily_values_result_cpu[row_idx], index=trading_dates_pd)
        history_df = pd.DataFrame(equity_curve, columns=["total_value"])
        analyzer = analyzer_cls(history_df)
        metrics = dict(analyzer.get_metrics(formatted=False))
        record = dict(shortlist_row.to_dict())
        record.update(params_list[row_idx])
        record.update(metrics)
        rows.append(record)
    return pd.DataFrame(rows)


def _resolve_shortlist_batch_size(
    *,
    config: dict,
    fixed_data_memory_bytes: int,
    num_tickers: int,
    num_trading_days: int,
    num_combinations: int,
) -> tuple[int, str]:
    optimal_batch_size = None
    try:
        optimal_batch_size = get_optimal_batch_size(
            config=config,
            num_tickers=num_tickers,
            fixed_data_memory_bytes=fixed_data_memory_bytes,
            num_trading_days=num_trading_days,
        )
    except Exception as err:
        print(
            "[GPU_WARNING] Could not calculate optimal shortlist batch size. "
            f"Falling back to safe defaults. error={err}"
        )

    batch_size, batch_size_source = _resolve_batch_size(
        optimal_batch_size=optimal_batch_size,
        backtest_settings=dict(config.get("backtest_settings") or {}),
        num_combinations=num_combinations,
    )
    if batch_size <= 0:
        raise ValueError(f"Invalid shortlist batch size resolved: {batch_size}")
    return batch_size, batch_size_source


def evaluate_shortlist_candidates_gpu_batch(
    shortlist_df,
    *,
    start_date: str,
    end_date: str,
    initial_cash: float,
    base_strategy_params: dict,
    config: dict,
    analyzer_cls,
):
    """
    Evaluate a fixed shortlist over one window using single-GPU batching.

    Returns a pandas DataFrame with the same row semantics as the legacy
    sequential evaluation path: one row per shortlist candidate with strategy
    params and analyzed metrics.
    """

    import cupy as cp
    import numpy as np
    import pandas as pd
    from sqlalchemy import create_engine

    from .data_loading import preload_all_data_to_gpu, preload_tier_data_to_tensor

    if shortlist_df.empty:
        return pd.DataFrame()

    params_list = [dict(base_strategy_params) for _ in range(len(shortlist_df))]
    for row_idx, shortlist_row in enumerate(shortlist_df.to_dict("records")):
        params = dict(base_strategy_params)
        params["max_stocks"] = int(shortlist_row["max_stocks"])
        params["order_investment_ratio"] = float(shortlist_row["order_investment_ratio"])
        params["additional_buy_drop_rate"] = float(shortlist_row["additional_buy_drop_rate"])
        params["sell_profit_rate"] = float(shortlist_row["sell_profit_rate"])
        params["additional_buy_priority"] = _normalize_priority_label(
            shortlist_row["additional_buy_priority"]
        )
        params["stop_loss_rate"] = float(shortlist_row["stop_loss_rate"])
        params["max_splits_limit"] = int(shortlist_row["max_splits_limit"])
        params["max_inactivity_period"] = int(shortlist_row["max_inactivity_period"])
        params_list[row_idx] = params

    execution_params_base = dict(config.get("execution_params") or {})
    execution_params_base["cooldown_period_days"] = base_strategy_params.get(
        "cooldown_period_days",
        5,
    )
    runtime_contract, run_exec_params = _ensure_homogeneous_runtime_contracts(
        params_list,
        start_date=start_date,
        end_date=end_date,
        execution_params_base=execution_params_base,
    )

    universe_mode = str(runtime_contract["universe_mode"])
    price_basis = str(runtime_contract["price_basis"])
    adjusted_gate_start_date = str(runtime_contract["adjusted_price_gate_start_date"])
    use_adjusted_prices = is_adjusted_price_basis(price_basis)
    print("\n" + "=" * 80)
    print(f"WORKER: Running GPU Shortlist Batch for {start_date} to {end_date}")
    print(f"Shortlist candidates: {len(params_list)}")
    print(f"Universe Policy: mode={universe_mode}")
    print(
        "Price Policy: "
        f"basis={price_basis}, adjusted_gate_start={adjusted_gate_start_date}"
    )
    print("=" * 80)

    db_connection_str = _build_db_connection_str(dict(config["database"]))
    param_combinations = _build_param_combinations(params_list)
    num_combinations = int(param_combinations.shape[0])

    all_data_gpu = preload_all_data_to_gpu(
        db_connection_str,
        start_date,
        end_date,
        use_adjusted_prices=use_adjusted_prices,
        adjusted_price_gate_start_date=adjusted_gate_start_date,
        universe_mode=universe_mode,
    )

    sql_engine = create_engine(db_connection_str)
    trading_dates_query = f"""
        SELECT DISTINCT date
        FROM DailyStockPrice
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    trading_dates_pd_df = pd.read_sql(
        trading_dates_query,
        sql_engine,
        parse_dates=["date"],
        index_col="date",
    )
    trading_dates_pd = trading_dates_pd_df.index
    if len(trading_dates_pd) == 0:
        raise ValueError("No trading dates resolved for shortlist batch evaluation.")

    trading_date_indices_gpu = cp.arange(len(trading_dates_pd), dtype=cp.int32)
    all_data_gpu = all_data_gpu[
        all_data_gpu.index.get_level_values("date").isin(trading_dates_pd)
    ]
    all_tickers = sorted(
        all_data_gpu.index.get_level_values("ticker").unique().to_pandas().tolist()
    )
    print(f"  - Tickers for period: {len(all_tickers)}")
    print(f"  - Trading days for period: {len(trading_dates_pd)}")

    tier_tensor = preload_tier_data_to_tensor(
        db_connection_str,
        start_date,
        end_date,
        all_tickers,
        trading_dates_pd,
        universe_mode=universe_mode,
        min_liquidity_20d_avg_value=int(
            runtime_contract["min_liquidity_20d_avg_value"] or 0
        ),
        min_tier12_coverage_ratio=runtime_contract["min_tier12_coverage_ratio"],
    )

    prepared_market_data = None
    prepared_bundle_bytes = 0
    prepared_bundle_start = time.time()
    try:
        prepared_market_data = prepare_market_data_bundle(
            all_data_gpu,
            all_tickers,
            trading_dates_pd,
            run_exec_params,
        )
        prepared_bundle_bytes = _measure_prepared_market_data_bytes(
            prepared_market_data,
            num_trading_days=len(trading_dates_pd),
            num_tickers=len(all_tickers),
        )
        print(
            "✅ Reusable market-data bundle prepared for shortlist batch. "
            f"Time: {time.time() - prepared_bundle_start:.2f}s"
        )
    except Exception as err:
        if not _is_gpu_oom_error(err, cp):
            raise
        _release_gpu_memory(cp)
        print(
            "[GPU_WARNING] OOM while preparing reusable shortlist market-data bundle "
            f"after {time.time() - prepared_bundle_start:.2f}s. "
            "Falling back to legacy per-batch preparation."
        )

    fixed_mem = int(all_data_gpu.memory_usage(deep=True).sum())
    fixed_mem += int(tier_tensor.nbytes)
    fixed_mem += int(prepared_bundle_bytes)
    batch_size, batch_size_source = _resolve_shortlist_batch_size(
        config=config,
        fixed_data_memory_bytes=fixed_mem,
        num_tickers=len(all_tickers),
        num_trading_days=len(trading_dates_pd),
        num_combinations=num_combinations,
    )
    if batch_size_source == "auto":
        print(f"✅ Using automatically calculated shortlist batch size: {batch_size}")
    elif batch_size_source == "auto-capped-by-config":
        print(f"✅ Using auto shortlist batch size capped by config: {batch_size}")
    else:
        print(f"⚠️ Using shortlist batch size fallback ({batch_size_source}): {batch_size}")

    daily_values_result_cpu = np.empty(
        (num_combinations, len(trading_dates_pd)),
        dtype=np.float32,
    )
    cursor = 0
    current_batch_size = batch_size
    executed_batches = 0

    while cursor < num_combinations:
        remaining = num_combinations - cursor
        attempt_batch_size = min(current_batch_size, remaining)
        attempt = 0

        while True:
            attempt += 1
            start_idx = cursor
            end_idx = start_idx + attempt_batch_size
            param_batch = param_combinations[start_idx:end_idx]
            start_time_kernel = time.time()
            print(
                "\n  --- Running Shortlist Batch "
                f"{executed_batches + 1} (Candidates {start_idx}-{end_idx - 1}, "
                f"size={attempt_batch_size}, attempt={attempt}) ---"
            )
            try:
                daily_values_batch = run_gpu_optimization(
                    param_batch,
                    all_data_gpu,
                    all_tickers,
                    trading_date_indices_gpu,
                    trading_dates_pd,
                    initial_cash,
                    run_exec_params,
                    tier_tensor=tier_tensor,
                    pit_universe_mask_tensor=None,
                    prepared_market_data=prepared_market_data,
                )
                if hasattr(daily_values_batch, "get"):
                    daily_values_result_cpu[start_idx:end_idx] = daily_values_batch.get()
                else:
                    daily_values_result_cpu[start_idx:end_idx] = np.asarray(
                        daily_values_batch
                    )
                batch_time = time.time() - start_time_kernel
                print(
                    f"  - Shortlist batch {executed_batches + 1} "
                    f"Kernel Execution Time: {batch_time:.2f}s"
                )
                del daily_values_batch
                _release_gpu_memory(cp)
                break
            except Exception as err:
                batch_time = time.time() - start_time_kernel
                if not _is_gpu_oom_error(err, cp):
                    raise
                _release_gpu_memory(cp)
                if attempt_batch_size <= 1:
                    raise RuntimeError(
                        "GPU OOM persists even with shortlist batch_size=1; "
                        "aborting shortlist batch evaluation."
                    ) from err
                reduced_batch_size = max(attempt_batch_size // 2, 1)
                print(
                    "[GPU_WARNING] OOM in shortlist batch "
                    f"{executed_batches + 1} after {batch_time:.2f}s. "
                    f"Reducing batch size {attempt_batch_size} -> {reduced_batch_size} and retrying."
                )
                attempt_batch_size = reduced_batch_size

        cursor = end_idx
        current_batch_size = min(current_batch_size, attempt_batch_size)
        executed_batches += 1

    return _build_evaluated_shortlist_frame(
        shortlist_df,
        params_list=params_list,
        daily_values_result_cpu=daily_values_result_cpu,
        trading_dates_pd=trading_dates_pd,
        analyzer_cls=analyzer_cls,
    )


__all__ = [
    "evaluate_shortlist_candidates_gpu_batch",
    "_build_evaluated_shortlist_frame",
]
