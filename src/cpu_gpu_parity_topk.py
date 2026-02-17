"""
cpu_gpu_parity_topk.py

CPU/GPU parity harness for top-k parameter batches with scenario packs:
- baseline_deterministic
- seeded_stress
- jackknife_drop_topn
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# BOOTSTRAP: allow direct execution (`python src/cpu_gpu_parity_topk.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from .backtest.cpu.backtester import BacktestEngine
from .backtest.cpu.execution import BasicExecutionHandler
from .backtest.cpu.portfolio import Portfolio
from .backtest.cpu.strategy import MagicSplitStrategy
from .config_loader import load_config
from .data_handler import DataHandler
from .optimization.gpu.context import (
    PARAM_ORDER,
    PRIORITY_MAP_REV,
    _build_db_connection_str,
    _ensure_core_deps,
    _ensure_gpu_deps,
)
from .optimization.gpu.data_loading import (
    preload_all_data_to_gpu,
    preload_tier_data_to_tensor,
    preload_weekly_filtered_stocks_to_gpu,
)
from .optimization.gpu.kernel import run_gpu_optimization
from .optimization.gpu.parameter_simulation import find_optimal_parameters


PRIORITY_MAP = {"lowest_order": 0, "highest_drop": 1}
CANDIDATE_MODES = ("weekly", "hybrid_transition", "tier")
INT_PARAM_COLS = {
    "max_stocks",
    "additional_buy_priority",
    "max_splits_limit",
    "max_inactivity_period",
}
FLOAT_PARAM_COLS = {
    "order_investment_ratio",
    "additional_buy_drop_rate",
    "sell_profit_rate",
    "stop_loss_rate",
}
EXECUTION_PARAM_KEYS = ("buy_commission_rate", "sell_commission_rate", "sell_tax_rate")


def _resolve_code_version() -> str:
    for env_key in ("GIT_COMMIT", "CI_COMMIT_SHA", "GITHUB_SHA"):
        value = os.getenv(env_key)
        if value:
            return value
    return "unknown"


def _is_gpu_unavailable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    hints = ("cupy", "cudf", "cuda", "gpu", "rapids")
    return any(hint in message for hint in hints)


def _normalize_priority(value: Any) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key in PRIORITY_MAP:
            return PRIORITY_MAP[key]
        if key in {"0", "1"}:
            return int(key)
        raise ValueError(f"Unsupported additional_buy_priority string: {value}")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported additional_buy_priority value: {value}") from exc
    if int_value not in (0, 1):
        raise ValueError(f"additional_buy_priority must be 0 or 1, got: {int_value}")
    return int_value


def _priority_to_strategy_string(value: Any) -> str:
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        if isinstance(value, str) and value.strip().lower() in PRIORITY_MAP:
            return value.strip().lower()
        return "lowest_order"
    return PRIORITY_MAP_REV.get(int_value, "lowest_order")


def _normalize_param_rows(df: Any) -> Any:
    np, _ = _ensure_core_deps()
    work = df.copy()
    missing = [col for col in PARAM_ORDER if col not in work.columns]
    if missing:
        raise ValueError(f"Missing parameter columns in source dataframe: {missing}")

    work["additional_buy_priority"] = work["additional_buy_priority"].map(_normalize_priority).astype(np.int32)
    for col in INT_PARAM_COLS:
        work[col] = work[col].astype(np.int32)
    for col in FLOAT_PARAM_COLS:
        work[col] = work[col].astype(np.float32)

    if "param_id" not in work.columns:
        work["param_id"] = work.index.astype(np.int32)
    work = work.reset_index(drop=True)
    return work


def _load_topk_params(
    params_csv: Optional[str],
    top_k: int,
    start_date: str,
    end_date: str,
    initial_cash: float,
) -> Any:
    _, pd = _ensure_core_deps()
    if params_csv:
        full_df = pd.read_csv(params_csv)
    else:
        _, full_df = find_optimal_parameters(
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
        )

    if full_df.empty:
        raise ValueError("No parameter rows available for parity run.")

    if "calmar_ratio" in full_df.columns:
        full_df = full_df.sort_values(by="calmar_ratio", ascending=False)
    selected = full_df.head(max(int(top_k), 1)).copy().reset_index(drop=True)
    selected["param_id"] = selected.index.astype(int)
    return _normalize_param_rows(selected)


def _build_scenarios(base_params: Any, scenario: str, seeded_stress_count: int, jackknife_max_drop: int) -> List[Dict[str, Any]]:
    np, _ = _ensure_core_deps()
    scenarios: List[Dict[str, Any]] = []

    if scenario in ("baseline_deterministic", "all"):
        scenarios.append(
            {
                "scenario_type": "baseline_deterministic",
                "seed_id": None,
                "drop_top_n": 0,
                "params_df": base_params.copy().reset_index(drop=True),
            }
        )

    if scenario in ("seeded_stress", "all"):
        stress_count = max(int(seeded_stress_count), 1)
        for seed in range(stress_count):
            rng = np.random.default_rng(seed)
            shuffled_index = rng.permutation(base_params.index.to_numpy())
            scenarios.append(
                {
                    "scenario_type": "seeded_stress",
                    "seed_id": int(seed),
                    "drop_top_n": 0,
                    "params_df": base_params.loc[shuffled_index].reset_index(drop=True),
                }
            )

    if scenario in ("jackknife_drop_topn", "all"):
        max_drop = max(int(jackknife_max_drop), 0)
        for drop_top_n in range(1, max_drop + 1):
            if len(base_params) <= drop_top_n:
                break
            scenarios.append(
                {
                    "scenario_type": "jackknife_drop_topn",
                    "seed_id": None,
                    "drop_top_n": int(drop_top_n),
                    "params_df": base_params.iloc[drop_top_n:].reset_index(drop=True),
                }
            )

    if not scenarios:
        raise ValueError(f"No scenario generated. input scenario={scenario}")
    return scenarios


def _load_gpu_shared_state(
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
    candidate_source_mode: str,
    use_weekly_alpha_gate: bool,
    parity_mode: str,
) -> Dict[str, Any]:
    cp, _, create_engine, _ = _ensure_gpu_deps()
    _, pd = _ensure_core_deps()

    db_connection_str = _build_db_connection_str(config["database"])
    strategy_params = dict(config["strategy_params"])
    execution_params = dict(config["execution_params"])
    execution_params["cooldown_period_days"] = strategy_params.get("cooldown_period_days", 5)
    execution_params["candidate_source_mode"] = candidate_source_mode
    execution_params["use_weekly_alpha_gate"] = bool(use_weekly_alpha_gate)
    execution_params["parity_mode"] = str(parity_mode).strip().lower()
    execution_params["tier_hysteresis_mode"] = strategy_params.get("tier_hysteresis_mode", "legacy")

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
    tier_tensor = preload_tier_data_to_tensor(db_connection_str, start_date, end_date, all_tickers, trading_dates_pd)

    return {
        "db_connection_str": db_connection_str,
        "trading_dates_pd": trading_dates_pd,
        "trading_date_indices_gpu": trading_date_indices_gpu,
        "all_data_gpu": all_data_gpu,
        "weekly_filtered_gpu": weekly_filtered_gpu,
        "all_tickers": all_tickers,
        "tier_tensor": tier_tensor,
        "execution_params": execution_params,
    }


def _run_gpu_curves(shared_state: Dict[str, Any], params_df: Any, initial_cash: float):
    cp, _, _, _ = _ensure_gpu_deps()
    np, _ = _ensure_core_deps()

    params_matrix = params_df.loc[:, list(PARAM_ORDER)].to_numpy(dtype=np.float32)
    params_gpu = cp.asarray(params_matrix)

    daily_values_gpu = run_gpu_optimization(
        params_gpu=params_gpu,
        data_gpu=shared_state["all_data_gpu"],
        weekly_filtered_gpu=shared_state["weekly_filtered_gpu"],
        all_tickers=shared_state["all_tickers"],
        trading_date_indices_gpu=shared_state["trading_date_indices_gpu"],
        trading_dates_pd=shared_state["trading_dates_pd"],
        initial_cash_value=float(initial_cash),
        exec_params=shared_state["execution_params"],
        tier_tensor=shared_state["tier_tensor"],
    )
    return daily_values_gpu.get()


def _run_cpu_curve(
    data_handler: DataHandler,
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
    initial_cash: float,
    param_row: Dict[str, Any],
    candidate_source_mode: str,
    use_weekly_alpha_gate: bool,
) -> Tuple[Any, Any]:
    _, pd = _ensure_core_deps()

    strategy_params = dict(config["strategy_params"])
    strategy_params.update(
        {
            "max_stocks": int(param_row["max_stocks"]),
            "order_investment_ratio": float(param_row["order_investment_ratio"]),
            "additional_buy_drop_rate": float(param_row["additional_buy_drop_rate"]),
            "sell_profit_rate": float(param_row["sell_profit_rate"]),
            "additional_buy_priority": _priority_to_strategy_string(param_row["additional_buy_priority"]),
            "stop_loss_rate": float(param_row["stop_loss_rate"]),
            "max_splits_limit": int(param_row["max_splits_limit"]),
            "max_inactivity_period": int(param_row["max_inactivity_period"]),
            "candidate_source_mode": candidate_source_mode,
            "use_weekly_alpha_gate": bool(use_weekly_alpha_gate),
            "backtest_start_date": start_date,
            "backtest_end_date": end_date,
        }
    )

    execution_params = dict(config["execution_params"])
    execution_params = {key: execution_params[key] for key in EXECUTION_PARAM_KEYS if key in execution_params}

    strategy = MagicSplitStrategy(**strategy_params)
    portfolio = Portfolio(initial_cash=float(initial_cash), start_date=start_date, end_date=end_date)
    execution_handler = BasicExecutionHandler(**execution_params)

    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        strategy=strategy,
        data_handler=data_handler,
        execution_handler=execution_handler,
    )
    final_portfolio = engine.run()
    history_df = pd.DataFrame(final_portfolio.daily_snapshot_history)
    if history_df.empty:
        return pd.Series(dtype=float), pd.DataFrame(columns=["total_value", "cash", "stock_count"])

    history_df["date"] = pd.to_datetime(history_df["date"])
    history_df.set_index("date", inplace=True)
    history_df = history_df.sort_index()

    curve = history_df["total_value"].astype(float)
    snapshots = history_df.reindex(columns=["total_value", "cash", "stock_count"]).copy()
    snapshots["total_value"] = snapshots["total_value"].astype(float)
    snapshots["cash"] = snapshots["cash"].astype(float)
    snapshots["stock_count"] = snapshots["stock_count"].fillna(0).astype(int)
    return curve, snapshots


def _build_value_dump(merged: Any, diffs: Any, first_mismatch_index: int, window_size: int = 2) -> List[Dict[str, Any]]:
    _, pd = _ensure_core_deps()
    start_idx = max(first_mismatch_index - window_size, 0)
    end_idx = min(first_mismatch_index + window_size + 1, len(merged))
    rows: List[Dict[str, Any]] = []
    for idx in range(start_idx, end_idx):
        dt = pd.to_datetime(merged.index[idx]).strftime("%Y-%m-%d")
        rows.append(
            {
                "date": dt,
                "cpu_value": float(merged.iloc[idx]["cpu"]),
                "gpu_value": float(merged.iloc[idx]["gpu"]),
                "abs_diff": float(diffs.iloc[idx]),
            }
        )
    return rows


def _compare_curves(cpu_curve: Any, gpu_curve: Any, cpu_snapshots: Any, tolerance: float) -> Dict[str, Any]:
    np, pd = _ensure_core_deps()
    merged = pd.concat([cpu_curve.rename("cpu"), gpu_curve.rename("gpu")], axis=1, join="inner").dropna()
    if merged.empty:
        return {
            "matched": False,
            "reason": "empty_overlap",
            "cpu_points": int(len(cpu_curve)),
            "gpu_points": int(len(gpu_curve)),
            "first_mismatch_index": None,
            "first_mismatch": None,
            "cpu_state_dump": None,
            "positions_dump": None,
            "value_dump": [],
        }

    diffs = (merged["cpu"] - merged["gpu"]).abs()
    bad = diffs > float(tolerance)
    if bad.any():
        bad_positions = np.where(bad.to_numpy())[0]
        first_idx = int(bad_positions[0])
        first_date = merged.index[first_idx]
        cpu_snapshot = None
        if not cpu_snapshots.empty and first_date in cpu_snapshots.index:
            row = cpu_snapshots.loc[first_date]
            cpu_snapshot = {
                "date": pd.to_datetime(first_date).strftime("%Y-%m-%d"),
                "cash": float(row["cash"]),
                "stock_count": int(row["stock_count"]),
                "total_value": float(row["total_value"]),
            }
        return {
            "matched": False,
            "reason": "tolerance_exceeded",
            "cpu_points": int(len(cpu_curve)),
            "gpu_points": int(len(gpu_curve)),
            "first_mismatch_index": first_idx,
            "first_mismatch": {
                "date": pd.to_datetime(first_date).strftime("%Y-%m-%d"),
                "cpu_value": float(merged.iloc[first_idx]["cpu"]),
                "gpu_value": float(merged.iloc[first_idx]["gpu"]),
                "abs_diff": float(diffs.iloc[first_idx]),
            },
            "cpu_state_dump": cpu_snapshot,
            "positions_dump": {"cpu_stock_count": None if cpu_snapshot is None else cpu_snapshot["stock_count"]},
            "value_dump": _build_value_dump(merged=merged, diffs=diffs, first_mismatch_index=first_idx),
        }

    return {
        "matched": True,
        "reason": "ok",
        "cpu_points": int(len(cpu_curve)),
        "gpu_points": int(len(gpu_curve)),
        "first_mismatch_index": None,
        "first_mismatch": None,
        "cpu_state_dump": None,
        "positions_dump": None,
        "value_dump": [],
    }


def _run_single_scenario(
    scenario: Dict[str, Any],
    config: Dict[str, Any],
    shared_state: Dict[str, Any],
    start_date: str,
    end_date: str,
    initial_cash: float,
    tolerance: float,
    candidate_source_mode: str,
    use_weekly_alpha_gate: bool,
) -> Dict[str, Any]:
    params_df = scenario["params_df"]
    scenario_type = scenario["scenario_type"]
    seed_id = scenario["seed_id"]
    drop_top_n = scenario["drop_top_n"]
    total_params = int(len(params_df))
    progress_interval = 10 if total_params >= 10 else 1
    scenario_started_at = datetime.now()

    print(
        f"[parity_topk] mode={candidate_source_mode}, scenario={scenario_type}, seed_id={seed_id}, "
        f"drop_top_n={drop_top_n}, params={len(params_df)}"
    )

    gpu_daily_values = _run_gpu_curves(shared_state, params_df, initial_cash=initial_cash)
    trading_dates = shared_state["trading_dates_pd"]
    _, pd = _ensure_core_deps()
    data_handler = DataHandler(db_config=config["database"])

    mismatches: List[Dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(params_df.iterrows()):
        row_dict = row.to_dict()
        cpu_curve, cpu_snapshots = _run_cpu_curve(
            data_handler=data_handler,
            config=config,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            param_row=row_dict,
            candidate_source_mode=candidate_source_mode,
            use_weekly_alpha_gate=use_weekly_alpha_gate,
        )
        gpu_curve = pd.Series(gpu_daily_values[row_idx], index=trading_dates)
        cmp_result = _compare_curves(
            cpu_curve=cpu_curve,
            gpu_curve=gpu_curve,
            cpu_snapshots=cpu_snapshots,
            tolerance=tolerance,
        )
        if not cmp_result["matched"]:
            mismatches.append(
                {
                    "candidate_source_mode": candidate_source_mode,
                    "scenario_type": scenario_type,
                    "seed_id": seed_id,
                    "drop_top_n": drop_top_n,
                    "row_index": int(row_idx),
                    "param_id": int(row_dict.get("param_id", row_idx)),
                    "params": {key: row_dict[key] for key in PARAM_ORDER},
                    **cmp_result,
                }
            )
        processed = row_idx + 1
        if processed % progress_interval == 0 or processed == total_params:
            elapsed = datetime.now() - scenario_started_at
            elapsed_sec = max(elapsed.total_seconds(), 1e-6)
            avg_sec_per_param = elapsed_sec / processed
            remaining = total_params - processed
            eta = timedelta(seconds=int(avg_sec_per_param * remaining))
            progress_pct = (processed / total_params) * 100 if total_params else 100.0
            print(
                f"[parity_topk] cpu progress {processed}/{total_params} ({progress_pct:.1f}%) "
                f"mismatches={len(mismatches)} elapsed={elapsed} eta={eta}"
            )

    return {
        "candidate_source_mode": candidate_source_mode,
        "scenario_type": scenario_type,
        "seed_id": seed_id,
        "drop_top_n": drop_top_n,
        "params_count": int(len(params_df)),
        "mismatch_count": int(len(mismatches)),
        "mismatches": mismatches,
    }


def _build_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU/GPU top-k parity harness with scenario packs.")
    parser.add_argument("--start-date", default=defaults["start_date"], help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=defaults["end_date"], help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument("--initial-cash", type=float, default=defaults["initial_cash"], help="Initial cash.")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k parameter rows to verify.")
    parser.add_argument(
        "--params-csv",
        default=None,
        help="Optional CSV path with parameter rows. If omitted, GPU optimization run is used.",
    )
    parser.add_argument(
        "--scenario",
        choices=["baseline_deterministic", "seeded_stress", "jackknife_drop_topn", "all"],
        default="all",
        help="Scenario pack selection.",
    )
    parser.add_argument(
        "--seeded-stress-count",
        type=int,
        default=10,
        help="Number of seeded_stress scenarios (seed=0..N-1).",
    )
    parser.add_argument(
        "--jackknife-max-drop",
        type=int,
        default=3,
        help="Max drop N for jackknife_drop_topn scenario.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Absolute tolerance for parity check.")
    parser.add_argument(
        "--parity-mode",
        choices=["fast", "strict"],
        default="fast",
        help="Parity execution mode. fast=GPU throughput priority, strict=CPU settlement parity priority.",
    )
    parser.add_argument(
        "--candidate-source-mode",
        default=defaults["candidate_source_mode"],
        choices=[*CANDIDATE_MODES, "all"],
        help="Candidate source mode used for both CPU/GPU runs.",
    )
    parser.add_argument(
        "--use-weekly-alpha-gate",
        action="store_true",
        default=defaults["use_weekly_alpha_gate"],
        help="Enable weekly alpha gate for both CPU/GPU runs.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: results/parity_topk_<timestamp>.json",
    )
    parser.set_defaults(fail_on_mismatch=True)
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        dest="fail_on_mismatch",
        help="Raise error when at least one mismatch is found.",
    )
    parser.add_argument(
        "--no-fail-on-mismatch",
        action="store_false",
        dest="fail_on_mismatch",
        help="Do not raise error even when mismatch exists.",
    )
    return parser


def _build_default_out_path() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("results", f"parity_topk_{timestamp}.json")


def _save_json_report(out_path: str, payload: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def main() -> None:
    config = load_config()
    backtest_settings = config["backtest_settings"]
    strategy_params = config["strategy_params"]
    default_mode = strategy_params.get("candidate_source_mode", "tier")
    if default_mode not in CANDIDATE_MODES:
        default_mode = "tier"

    defaults = {
        "start_date": backtest_settings["start_date"],
        "end_date": backtest_settings["end_date"],
        "initial_cash": backtest_settings["initial_cash"],
        "candidate_source_mode": default_mode,
        "use_weekly_alpha_gate": bool(strategy_params.get("use_weekly_alpha_gate", False)),
    }
    args = _build_parser(defaults).parse_args()

    started_at = datetime.now()
    base_params = _load_topk_params(
        params_csv=args.params_csv,
        top_k=args.top_k,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.initial_cash,
    )
    scenarios = _build_scenarios(
        base_params=base_params,
        scenario=args.scenario,
        seeded_stress_count=args.seeded_stress_count,
        jackknife_max_drop=args.jackknife_max_drop,
    )

    candidate_modes = list(CANDIDATE_MODES) if args.candidate_source_mode == "all" else [args.candidate_source_mode]
    mode_reports: List[Dict[str, Any]] = []
    gpu_skip_reasons: Dict[str, str] = {}

    for mode in candidate_modes:
        try:
            shared_state = _load_gpu_shared_state(
                config=config,
                start_date=args.start_date,
                end_date=args.end_date,
                candidate_source_mode=mode,
                use_weekly_alpha_gate=args.use_weekly_alpha_gate,
                parity_mode=args.parity_mode,
            )
        except Exception as exc:
            if _is_gpu_unavailable_error(exc):
                reason = f"gpu_unavailable: {exc}"
                gpu_skip_reasons[mode] = reason
                mode_reports.append(
                    {
                        "candidate_source_mode": mode,
                        "skipped": True,
                        "skip_reason": reason,
                        "scenario_reports": [],
                        "mode_mismatches": 0,
                    }
                )
                continue
            raise

        scenario_reports: List[Dict[str, Any]] = []
        for scenario in scenarios:
            report = _run_single_scenario(
                scenario=scenario,
                config=config,
                shared_state=shared_state,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_cash=args.initial_cash,
                tolerance=args.tolerance,
                candidate_source_mode=mode,
                use_weekly_alpha_gate=args.use_weekly_alpha_gate,
            )
            scenario_reports.append(report)

        mode_mismatches = int(sum(item["mismatch_count"] for item in scenario_reports))
        mode_reports.append(
            {
                "candidate_source_mode": mode,
                "skipped": False,
                "skip_reason": None,
                "scenario_reports": scenario_reports,
                "mode_mismatches": mode_mismatches,
            }
        )

    total_mismatches = int(sum(item["mode_mismatches"] for item in mode_reports))
    finished_at = datetime.now()
    scenario_metadata = []
    for mode_report in mode_reports:
        for scenario_report in mode_report["scenario_reports"]:
            scenario_metadata.append(
                {
                    "candidate_source_mode": mode_report["candidate_source_mode"],
                    "scenario_type": scenario_report["scenario_type"],
                    "seed_id": scenario_report["seed_id"],
                    "drop_top_n": scenario_report["drop_top_n"],
                }
            )

    summary = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": (finished_at - started_at).total_seconds(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_cash": float(args.initial_cash),
        "top_k": int(args.top_k),
        "tolerance": float(args.tolerance),
        "parity_mode": args.parity_mode,
        "scenario_input": args.scenario,
        "candidate_source_mode_input": args.candidate_source_mode,
        "use_weekly_alpha_gate": bool(args.use_weekly_alpha_gate),
        "snapshot_metadata": {
            "generated_at": finished_at.isoformat(),
            "code_version": _resolve_code_version(),
            "params_source": args.params_csv if args.params_csv else "gpu_optimization_result",
            "parameter_columns": list(PARAM_ORDER),
            "param_ids": [int(value) for value in base_params["param_id"].tolist()],
        },
        "scenario_metadata": scenario_metadata,
        "mode_reports": mode_reports,
        "total_modes": len(mode_reports),
        "total_scenarios": len(scenario_metadata),
        "total_mismatches": total_mismatches,
        "skipped": len(gpu_skip_reasons) == len(candidate_modes),
        "skip_reasons": gpu_skip_reasons,
    }

    out_path = args.out or _build_default_out_path()
    _save_json_report(out_path=out_path, payload=summary)

    print(f"[parity_topk] report saved: {out_path}")
    print(f"[parity_topk] total_mismatches={total_mismatches}")
    if summary["skipped"]:
        print("[parity_topk] skipped due to unavailable GPU dependencies.")

    if args.fail_on_mismatch and total_mismatches > 0:
        raise AssertionError(f"Parity mismatch detected: {total_mismatches}")


if __name__ == "__main__":
    main()
