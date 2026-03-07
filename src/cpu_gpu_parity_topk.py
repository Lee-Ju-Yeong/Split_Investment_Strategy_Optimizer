"""
cpu_gpu_parity_topk.py

Issue #56:
CPU(SSOT) vs GPU parity harness for a *batch* of parameter combinations (top-k).

Design goals:
- Avoid importing heavy "standalone scripts" that execute on import.
- Load GPU fixed data once, run GPU for N param rows, then run CPU sequentially and compare.
- Keep defaults conservative (small K, short period) to prevent accidental long runs.

Example:
  PYTHONPATH=$PWD conda run --no-capture-output -n rapids-env \
    python -m src.cpu_gpu_parity_topk \
      --start-date 20150102 --end-date 20150630 \
      --params-csv results/topk_params.csv --topk 5 \
      --tolerance 1e-3 --out /tmp/parity_report.json
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
from dataclasses import dataclass
from datetime import datetime
import os
import urllib.parse

import pandas as pd
from sqlalchemy import create_engine

from .backtest.cpu.backtester import BacktestEngine
from .backtest.cpu.execution import BasicExecutionHandler
from .backtest.cpu.portfolio import Portfolio
from .backtest.cpu.strategy import MagicSplitStrategy
from .config_loader import load_config
from .data_handler import DataHandler
from .optimization.gpu.data_loading import (
    preload_all_data_to_gpu as preload_all_data_to_gpu_shared,
    preload_pit_universe_mask_to_tensor,
)
from .price_policy import is_adjusted_price_basis, resolve_price_policy
from .universe_policy import resolve_universe_mode


def _require_gpu_stack():
    try:
        import cudf  # noqa: F401
        import cupy as cp  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "GPU stack import failed. Run with `conda run -n rapids-env ...` on a CUDA machine."
        ) from exc


def _parse_date(value: str) -> str:
    # Accept YYYYMMDD or YYYY-MM-DD and normalize to YYYY-MM-DD.
    v = value.strip()
    if len(v) == 8 and v.isdigit():
        return datetime.strptime(v, "%Y%m%d").strftime("%Y-%m-%d")
    return datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")


def _build_db_connection_str(db_cfg: dict) -> str:
    pwd = urllib.parse.quote_plus(db_cfg["password"])
    return f"mysql+pymysql://{db_cfg['user']}:{pwd}@{db_cfg['host']}/{db_cfg['database']}"


def _load_trading_dates(sql_engine, start_date: str, end_date: str) -> pd.DatetimeIndex:
    query = """
        SELECT DISTINCT date
        FROM DailyStockPrice
        WHERE date BETWEEN %s AND %s
        ORDER BY date
    """
    df = pd.read_sql(query, sql_engine, params=(start_date, end_date), parse_dates=["date"])
    if df.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(df["date"]).sort_values().unique())


def _load_all_data_to_gpu(
    db_connection_str: str,
    start_date: str,
    end_date: str,
    *,
    use_adjusted_prices: bool,
    adjusted_price_gate_start_date: str,
    universe_mode: str,
):
    return preload_all_data_to_gpu_shared(
        db_connection_str,
        start_date,
        end_date,
        use_adjusted_prices=use_adjusted_prices,
        adjusted_price_gate_start_date=adjusted_price_gate_start_date,
        universe_mode=universe_mode,
    )


def _build_empty_weekly_filtered_gpu():
    import cudf

    df = cudf.DataFrame({"date": [], "ticker": []})
    return df.set_index("date")


def _load_tier_tensor(
    sql_engine,
    start_date: str,
    end_date: str,
    all_tickers: list[str],
    trading_dates_pd,
    pit_universe_mask_tensor,
    min_liquidity_20d_avg_value: int,
    min_tier12_coverage_ratio: float,
):
    import cupy as cp

    query = """
        SELECT date, stock_code AS ticker, tier, liquidity_20d_avg_value
        FROM DailyStockTier
        WHERE date BETWEEN %s AND %s
        UNION ALL
        SELECT t.date, t.stock_code AS ticker, t.tier, t.liquidity_20d_avg_value
        FROM DailyStockTier t
        JOIN (
            SELECT stock_code, MAX(date) AS max_date
            FROM DailyStockTier
            WHERE date < %s
            GROUP BY stock_code
        ) latest ON t.stock_code = latest.stock_code AND t.date = latest.max_date
    """
    df_pd = pd.read_sql(
        query,
        sql_engine,
        params=(start_date, end_date, start_date),
        parse_dates=["date"],
    )
    if df_pd.empty:
        return cp.zeros((len(trading_dates_pd), len(all_tickers)), dtype=cp.int8)

    all_tickers_str = [str(t) for t in all_tickers]
    tier_wide = (
        df_pd.assign(ticker=df_pd["ticker"].astype(str))
        .pivot_table(index="date", columns="ticker", values="tier")
        .sort_index()
    )
    liq_wide = (
        df_pd.assign(ticker=df_pd["ticker"].astype(str))
        .pivot_table(index="date", columns="ticker", values="liquidity_20d_avg_value")
        .sort_index()
    )

    union_index = tier_wide.index.union(pd.DatetimeIndex(trading_dates_pd)).sort_values()
    tier_ffilled = tier_wide.reindex(index=union_index).ffill()
    liq_ffilled = liq_wide.reindex(index=union_index).ffill()

    tier_asof = (
        tier_ffilled.reindex(index=trading_dates_pd, columns=all_tickers_str)
        .fillna(0)
        .astype(int)
    )
    liq_asof = liq_ffilled.reindex(index=trading_dates_pd, columns=all_tickers_str)

    min_liq = max(int(min_liquidity_20d_avg_value or 0), 0)
    if min_liq > 0:
        liq_mask = liq_asof.fillna(-1) >= min_liq
        tier_asof = tier_asof.where(liq_mask, 0)

    min_ratio = float(min_tier12_coverage_ratio or 0.0)
    if min_ratio > 0 and len(all_tickers) > 0:
        pit_mask_df = pd.DataFrame(
            cp.asnumpy(pit_universe_mask_tensor).astype(bool),
            index=pd.DatetimeIndex(trading_dates_pd),
            columns=all_tickers_str,
        )
        tier12_mask = ((tier_asof > 0) & (tier_asof <= 2)) & pit_mask_df
        pit_size = pit_mask_df.sum(axis=1).astype(int)
        ratio = pd.Series(0.0, index=tier_asof.index, dtype=float)
        valid_mask = pit_size > 0
        ratio.loc[valid_mask] = (
            tier12_mask.sum(axis=1).astype(int).loc[valid_mask] / pit_size.loc[valid_mask]
        )
        failed = ratio[valid_mask & (ratio < min_ratio)]
        if not failed.empty:
            fail_date = failed.index[0]
            raise ValueError(
                f"Tier coverage gate failed on {pd.to_datetime(fail_date).date()}: "
                f"tier12_ratio={float(failed.iloc[0]):.4f} < threshold={min_ratio:.4f}"
            )
    return cp.asarray(tier_asof.values, dtype=cp.int8)


def _priority_to_cpu(value) -> str:
    if isinstance(value, str):
        v = value.strip()
        if v in ("lowest_order", "highest_drop"):
            return v
    try:
        return "highest_drop" if int(float(value)) == 1 else "lowest_order"
    except Exception:
        return "lowest_order"


def _priority_to_gpu(value) -> int:
    if isinstance(value, str):
        v = value.strip()
        if v == "highest_drop":
            return 1
        return 0
    try:
        return 1 if int(float(value)) == 1 else 0
    except Exception:
        return 0


@dataclass(frozen=True)
class ParityParamRow:
    max_stocks: int
    order_investment_ratio: float
    additional_buy_drop_rate: float
    sell_profit_rate: float
    additional_buy_priority: str
    stop_loss_rate: float
    max_splits_limit: int
    max_inactivity_period: int

    @staticmethod
    def from_mapping(row: dict) -> "ParityParamRow":
        return ParityParamRow(
            max_stocks=int(row["max_stocks"]),
            order_investment_ratio=float(row["order_investment_ratio"]),
            additional_buy_drop_rate=float(row["additional_buy_drop_rate"]),
            sell_profit_rate=float(row["sell_profit_rate"]),
            additional_buy_priority=_priority_to_cpu(row["additional_buy_priority"]),
            stop_loss_rate=float(row.get("stop_loss_rate", -0.15)),
            max_splits_limit=int(row.get("max_splits_limit", 10)),
            max_inactivity_period=int(row.get("max_inactivity_period", 90)),
        )

    def to_gpu_row(self) -> list[float]:
        return [
            float(self.max_stocks),
            float(self.order_investment_ratio),
            float(self.additional_buy_drop_rate),
            float(self.sell_profit_rate),
            float(_priority_to_gpu(self.additional_buy_priority)),
            float(self.stop_loss_rate),
            float(self.max_splits_limit),
            float(self.max_inactivity_period),
        ]

    def to_strategy_kwargs(self) -> dict:
        return {
            "max_stocks": int(self.max_stocks),
            "order_investment_ratio": float(self.order_investment_ratio),
            "additional_buy_drop_rate": float(self.additional_buy_drop_rate),
            "sell_profit_rate": float(self.sell_profit_rate),
            "additional_buy_priority": str(self.additional_buy_priority),
            "stop_loss_rate": float(self.stop_loss_rate),
            "max_splits_limit": int(self.max_splits_limit),
            "max_inactivity_period": int(self.max_inactivity_period),
            "candidate_source_mode": "tier",
        }


def _load_params_from_csv(path: str, topk: int, sort_by: str | None) -> list[ParityParamRow]:
    df = pd.read_csv(path)
    if df.empty:
        return []
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    df = df.head(max(int(topk), 1))
    return [ParityParamRow.from_mapping(row.to_dict()) for _, row in df.iterrows()]


def _run_cpu_equity_curve(
    data_handler: DataHandler,
    base_config: dict,
    start_date: str,
    end_date: str,
    initial_cash: float,
    row: ParityParamRow,
) -> pd.Series:
    execution_params = dict(base_config["execution_params"])
    strategy_params = dict(base_config.get("strategy_params", {}))
    strategy_params.update(row.to_strategy_kwargs())
    strategy_params["backtest_start_date"] = start_date
    strategy_params["backtest_end_date"] = end_date

    strategy_allowed = {
        key
        for key in inspect.signature(MagicSplitStrategy.__init__).parameters.keys()
        if key != "self"
    }
    strategy_params = {k: v for k, v in strategy_params.items() if k in strategy_allowed}

    execution_allowed = {
        key
        for key in inspect.signature(BasicExecutionHandler.__init__).parameters.keys()
        if key != "self"
    }
    execution_params = {k: v for k, v in execution_params.items() if k in execution_allowed}

    strategy = MagicSplitStrategy(**strategy_params)
    portfolio = Portfolio(initial_cash=initial_cash, start_date=start_date, end_date=end_date)
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
        return pd.Series(dtype=float)
    history_df["date"] = pd.to_datetime(history_df["date"])
    history_df = history_df.sort_values("date").set_index("date")
    curve = history_df.get("total_value")
    if curve is None:
        return pd.Series(dtype=float)
    return pd.Series(curve.values, index=curve.index)


def _compare_curves(cpu_curve: pd.Series, gpu_curve: pd.Series, tolerance: float) -> dict:
    aligned = pd.concat(
        [cpu_curve.rename("cpu"), gpu_curve.rename("gpu")],
        axis=1,
        join="outer",
    ).sort_index()
    if aligned.empty:
        return {"points": 0, "mismatch_count": None, "first_mismatch": None}

    missing_mask = aligned["cpu"].isna() | aligned["gpu"].isna()
    missing_count = int(missing_mask.sum())
    aligned_present = aligned.loc[~missing_mask]
    diff = (aligned_present["cpu"] - aligned_present["gpu"]).abs()
    mismatches = diff > float(tolerance)
    mismatch_count = missing_count + int(mismatches.sum())
    if mismatch_count <= 0:
        return {"points": int(len(aligned)), "mismatch_count": 0, "first_mismatch": None}

    if missing_count > 0:
        first_idx = aligned.loc[missing_mask].index[0]
        return {
            "points": int(len(aligned)),
            "mismatch_count": mismatch_count,
            "first_mismatch": {
                "date": pd.to_datetime(first_idx).strftime("%Y-%m-%d"),
                "cpu": None if pd.isna(aligned.loc[first_idx, "cpu"]) else float(aligned.loc[first_idx, "cpu"]),
                "gpu": None if pd.isna(aligned.loc[first_idx, "gpu"]) else float(aligned.loc[first_idx, "gpu"]),
                "abs_diff": None,
                "reason": "missing_curve_point",
            },
        }

    first_idx = diff[mismatches].index[0]
    return {
        "points": int(len(aligned)),
        "mismatch_count": mismatch_count,
        "first_mismatch": {
            "date": pd.to_datetime(first_idx).strftime("%Y-%m-%d"),
            "cpu": float(aligned.loc[first_idx, "cpu"]),
            "gpu": float(aligned.loc[first_idx, "gpu"]),
            "abs_diff": float(diff.loc[first_idx]),
        },
    }


def _build_parity_summary(rows: list[dict], *, parity_mode: str, universe_mode: str) -> dict:
    failed = 0
    total_mismatches = 0
    failed_indices = []
    first_failed_row = None
    max_abs_diff = 0.0

    for row in rows:
        compare = dict(row.get("compare") or {})
        mismatch_count = int(compare.get("mismatch_count") or 0)
        total_mismatches += mismatch_count
        first_mismatch = compare.get("first_mismatch") or {}
        max_abs_diff = max(max_abs_diff, float(first_mismatch.get("abs_diff") or 0.0))
        if mismatch_count <= 0:
            continue
        failed += 1
        row_index = int(row.get("index", -1))
        failed_indices.append(row_index)
        if first_failed_row is None:
            first_failed_row = {
                "index": row_index,
                "mismatch_count": mismatch_count,
                "first_mismatch": compare.get("first_mismatch"),
            }

    passed = max(len(rows) - failed, 0)
    policy_ready = str(parity_mode).strip().lower() == "strict" and str(universe_mode).strip().lower() == "strict_pit"
    promotion_block_reasons = []
    if not policy_ready:
        promotion_block_reasons.append(
            f"policy_not_release_ready(parity_mode={parity_mode}, universe_mode={universe_mode})"
        )
    if total_mismatches > 0:
        promotion_block_reasons.append(f"curve_mismatch_count={int(total_mismatches)}")
    promotion_block_reasons.append("decision_level_evidence_missing")

    return {
        "failed": int(failed),
        "passed": int(passed),
        "failed_indices": failed_indices,
        "total_mismatches": int(total_mismatches),
        "max_abs_diff": float(max_abs_diff),
        "first_failed_row": first_failed_row,
        "comparison_level": "equity_curve",
        "policy_ready_for_release_gate": bool(policy_ready),
        "curve_level_parity_zero_mismatch": bool(policy_ready and total_mismatches == 0),
        "decision_level_parity_zero_mismatch": False,
        "event_level_diff_collected": False,
        "promotion_blocked": bool(promotion_block_reasons),
        "promotion_block_reasons": promotion_block_reasons,
    }


def _resolve_decision_evidence_indices(rows: list[dict], mode: str) -> list[int]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "off" or not rows:
        return []

    failed_indices = [
        int(row.get("index", -1))
        for row in rows
        if int((row.get("compare") or {}).get("mismatch_count") or 0) > 0
    ]
    available_indices = [int(row.get("index", -1)) for row in rows]

    if normalized_mode == "first_failed":
        return failed_indices[:1]
    if normalized_mode == "representative":
        return failed_indices[:1] or available_indices[:1]
    if normalized_mode == "all":
        return available_indices
    raise ValueError(f"Unsupported decision_evidence_mode={mode!r}")


def _build_decision_evidence_detail_path(report_out: str, row_index: int) -> str:
    base_dir = os.path.dirname(report_out) or "."
    base_name = os.path.splitext(os.path.basename(report_out))[0]
    return os.path.join(base_dir, f"{base_name}.decision_evidence_row{row_index}.json")


def _write_json_artifact(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _summarize_decision_evidence_payload(row_index: int, payload: dict, *, detail_path: str | None) -> dict:
    return {
        "row_index": int(row_index),
        "decision_level_zero_mismatch": bool(payload.get("decision_level_zero_mismatch", False)),
        "comparison_scope": str(payload.get("comparison_scope", "core_trade_events")),
        "release_decision_fields_complete": bool(payload.get("release_decision_fields_complete", False)),
        "sell_mismatched_pairs": int(payload.get("sell_mismatched_pairs", 0) or 0),
        "buy_mismatched_pairs": int(payload.get("buy_mismatched_pairs", 0) or 0),
        "cpu_sell_events_count": int(payload.get("cpu_sell_events_count", 0) or 0),
        "gpu_sell_events_count": int(payload.get("gpu_sell_events_count", 0) or 0),
        "cpu_buy_events_count": int(payload.get("cpu_buy_events_count", 0) or 0),
        "gpu_buy_events_count": int(payload.get("gpu_buy_events_count", 0) or 0),
        "daily_snapshot_mismatched_pairs": int(payload.get("daily_snapshot_mismatched_pairs", 0) or 0),
        "position_snapshot_mismatched_pairs": int(payload.get("position_snapshot_mismatched_pairs", 0) or 0),
        "detail_path": detail_path,
    }


def _apply_decision_evidence(summary: dict, evidence_rows: list[dict], *, total_rows: int) -> dict:
    updated = dict(summary)
    reasons = [
        reason
        for reason in list(updated.get("promotion_block_reasons") or [])
        if reason != "decision_level_evidence_missing"
        and not str(reason).startswith("decision_level_evidence_partial(")
        and not str(reason).startswith("decision_event_mismatch_rows=")
    ]

    checked_count = len(evidence_rows)
    passed_count = sum(1 for row in evidence_rows if row.get("decision_level_zero_mismatch"))
    all_rows_covered = total_rows > 0 and checked_count == total_rows
    release_fields_complete = checked_count > 0 and all(
        bool(row.get("release_decision_fields_complete", False))
        for row in evidence_rows
    )

    updated["event_level_diff_collected"] = checked_count > 0
    updated["decision_evidence_rows_checked"] = int(checked_count)
    updated["decision_evidence_rows_passed"] = int(passed_count)
    updated["decision_evidence_all_rows_covered"] = bool(all_rows_covered)
    updated["decision_evidence_release_fields_complete"] = bool(release_fields_complete)
    updated["decision_level_parity_zero_mismatch"] = bool(
        updated.get("curve_level_parity_zero_mismatch", False)
        and all_rows_covered
        and checked_count > 0
        and passed_count == checked_count
        and release_fields_complete
    )

    if checked_count <= 0:
        reasons.append("decision_level_evidence_missing")
    elif not all_rows_covered:
        reasons.append(f"decision_level_evidence_partial({checked_count}/{total_rows})")
    elif passed_count != checked_count:
        reasons.append(f"decision_event_mismatch_rows={checked_count - passed_count}")
    elif not release_fields_complete:
        reasons.append("decision_fields_not_covered")

    updated["promotion_block_reasons"] = reasons
    updated["promotion_blocked"] = bool(reasons)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU/GPU top-k parity harness (Issue #56).")
    parser.add_argument("--start-date", default=None, help="YYYYMMDD or YYYY-MM-DD (default: config).")
    parser.add_argument("--end-date", default=None, help="YYYYMMDD or YYYY-MM-DD (default: config).")
    parser.add_argument("--initial-cash", type=float, default=None, help="Override initial cash (default: config).")
    parser.add_argument("--params-csv", default=None, help="CSV with param columns (and optionally metrics).")
    parser.add_argument("--topk", type=int, default=3, help="Number of param rows to validate (default: 3).")
    parser.add_argument("--sort-by", default=None, help="Sort CSV by this metric column desc before topk selection.")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Absolute tolerance (default: 1e-3).")
    parser.add_argument(
        "--parity-mode",
        choices=["fast", "strict"],
        default="strict",
        help="GPU parity mode for evaluation path (default: strict).",
    )
    parser.add_argument(
        "--decision-evidence-mode",
        choices=["off", "first_failed", "representative", "all"],
        default="representative",
        help=(
            "Collect event-level CPU/GPU trade diffs using parity_sell_event_dump helpers. "
            "representative=first failed row, or row0 when curves all match."
        ),
    )
    parser.add_argument("--out", default=None, help="Write JSON report to this path.")
    parser.add_argument(
        "--no-fail-on-mismatch",
        action="store_true",
        help="Do not raise AssertionError even if mismatch exists (report still records failures).",
    )
    args = parser.parse_args()

    _require_gpu_stack()
    import cupy as cp
    from .backtest.gpu.engine import run_magic_split_strategy_on_gpu

    config = load_config()
    start_date = _parse_date(args.start_date) if args.start_date else str(config["backtest_settings"]["start_date"])
    end_date = _parse_date(args.end_date) if args.end_date else str(config["backtest_settings"]["end_date"])
    initial_cash = float(args.initial_cash) if args.initial_cash is not None else float(config["backtest_settings"]["initial_cash"])

    if args.params_csv:
        rows = _load_params_from_csv(args.params_csv, topk=int(args.topk), sort_by=args.sort_by)
    else:
        if int(args.topk) != 1:
            print(
                "[cpu_gpu_parity_topk] info: --params-csv not provided. "
                "Using a single parameter row from config.strategy_params."
            )
        rows = [ParityParamRow.from_mapping(config.get("strategy_params", {}))]

    if not rows:
        raise ValueError("No parameter rows to validate.")
    if len(rows) > 10:
        print(f"[cpu_gpu_parity_topk] warning: topk={len(rows)} may take a long time on CPU.")

    # GPU fixed data preload (once)
    db_conn_str = _build_db_connection_str(config["database"])
    sql_engine = create_engine(db_conn_str)
    trading_dates_pd = _load_trading_dates(sql_engine, start_date, end_date)
    if trading_dates_pd.empty:
        raise ValueError("No trading dates in the requested range.")

    strategy_cfg = config.get("strategy_params", {})
    price_basis, adjusted_gate_start_date = resolve_price_policy(strategy_cfg)
    use_adjusted_prices = is_adjusted_price_basis(price_basis)
    universe_mode = resolve_universe_mode(
        strategy_cfg,
        universe_mode=os.environ.get("MAGICSPLIT_UNIVERSE_MODE"),
    )

    all_data_gpu = _load_all_data_to_gpu(
        db_conn_str,
        start_date,
        end_date,
        use_adjusted_prices=use_adjusted_prices,
        adjusted_price_gate_start_date=adjusted_gate_start_date,
        universe_mode=universe_mode,
    )
    if all_data_gpu.empty:
        raise ValueError("No DailyStockPrice rows in the requested range.")

    all_tickers = sorted(all_data_gpu.index.get_level_values("ticker").unique().to_pandas().tolist())
    trading_date_indices_gpu = cp.arange(len(trading_dates_pd), dtype=cp.int32)
    weekly_filtered_gpu = _build_empty_weekly_filtered_gpu()
    pit_universe_mask_tensor = preload_pit_universe_mask_to_tensor(
        db_conn_str,
        start_date,
        end_date,
        all_tickers,
        trading_dates_pd,
    )
    tier_tensor = _load_tier_tensor(
        sql_engine,
        start_date,
        end_date,
        all_tickers,
        trading_dates_pd,
        pit_universe_mask_tensor,
        min_liquidity_20d_avg_value=int(strategy_cfg.get("min_liquidity_20d_avg_value", 0) or 0),
        min_tier12_coverage_ratio=float(strategy_cfg.get("min_tier12_coverage_ratio", 0.45) or 0.45),
    )

    exec_params = copy.deepcopy(config["execution_params"])
    exec_params["cooldown_period_days"] = config.get("strategy_params", {}).get("cooldown_period_days", 5)
    exec_params["candidate_source_mode"] = "tier"
    exec_params["parity_mode"] = str(args.parity_mode).strip().lower()
    exec_params["tier_hysteresis_mode"] = config.get("strategy_params", {}).get("tier_hysteresis_mode", "legacy")
    gpu_curves: list = []
    if exec_params["parity_mode"] == "strict":
        # Strict parity mode favors semantic equivalence over throughput.
        # Run one parameter row at a time so GPU path uses single-sim semantics.
        for row in rows:
            single_param_gpu = cp.asarray([row.to_gpu_row()], dtype=cp.float32)
            daily_values_gpu = run_magic_split_strategy_on_gpu(
                initial_cash=float(initial_cash),
                param_combinations=single_param_gpu,
                all_data_gpu=all_data_gpu,
                weekly_filtered_gpu=weekly_filtered_gpu,
                trading_date_indices=trading_date_indices_gpu,
                trading_dates_pd_cpu=trading_dates_pd,
                all_tickers=all_tickers,
                execution_params=exec_params,
                max_splits_limit=int(row.max_splits_limit),
                tier_tensor=tier_tensor,
                pit_universe_mask_tensor=pit_universe_mask_tensor,
                debug_mode=False,
            )
            gpu_curves.append(daily_values_gpu.get()[0])
    else:
        param_gpu = cp.asarray([row.to_gpu_row() for row in rows], dtype=cp.float32)
        daily_values_gpu = run_magic_split_strategy_on_gpu(
            initial_cash=float(initial_cash),
            param_combinations=param_gpu,
            all_data_gpu=all_data_gpu,
            weekly_filtered_gpu=weekly_filtered_gpu,
            trading_date_indices=trading_date_indices_gpu,
            trading_dates_pd_cpu=trading_dates_pd,
            all_tickers=all_tickers,
            execution_params=exec_params,
            max_splits_limit=int(max(row.max_splits_limit for row in rows)),
            tier_tensor=tier_tensor,
            pit_universe_mask_tensor=pit_universe_mask_tensor,
            debug_mode=False,
        )
        gpu_curves = list(daily_values_gpu.get())

    # CPU runs (sequential), using a single DataHandler instance for caching.
    data_handler = DataHandler(
        db_config=config["database"],
        strategy_params=strategy_cfg,
        universe_mode=universe_mode,
    )

    report = {
        "meta": {
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": initial_cash,
            "requested_topk": int(args.topk),
            "validated_rows": int(len(rows)),
            "topk": int(len(rows)),
            "tolerance": float(args.tolerance),
            "candidate_source_mode": "tier",
            "parity_mode": exec_params["parity_mode"],
            "tier_hysteresis_mode": exec_params["tier_hysteresis_mode"],
            "price_basis": price_basis,
            "adjusted_price_gate_start_date": adjusted_gate_start_date,
            "universe_mode": universe_mode,
        },
        "rows": [],
        "summary": {},
        "evidence_gaps": [
            "event_level_diff_not_collected",
        ],
        "decision_evidence": {
            "mode": str(args.decision_evidence_mode).strip().lower(),
            "rows": [],
        },
    }

    for idx, row in enumerate(rows):
        gpu_curve = pd.Series(gpu_curves[idx], index=trading_dates_pd)
        cpu_curve = _run_cpu_equity_curve(
            data_handler=data_handler,
            base_config=config,
            start_date=start_date,
            end_date=end_date,
            initial_cash=float(initial_cash),
            row=row,
        )
        cmp_result = _compare_curves(cpu_curve, gpu_curve, tolerance=float(args.tolerance))
        failed = int(cmp_result.get("mismatch_count") or 0) > 0
        report["rows"].append(
            {
                "index": int(idx),
                "params": row.to_strategy_kwargs(),
                "compare": cmp_result,
            }
        )

    report["summary"] = _build_parity_summary(
        report["rows"],
        parity_mode=exec_params["parity_mode"],
        universe_mode=universe_mode,
    )
    decision_evidence_mode = str(args.decision_evidence_mode).strip().lower()
    decision_evidence_indices = _resolve_decision_evidence_indices(report["rows"], decision_evidence_mode)
    if decision_evidence_indices:
        from .parity_sell_event_dump import collect_trade_event_parity_report

        row_lookup = {int(row["index"]): row for row in report["rows"]}
        for row_index in decision_evidence_indices:
            row_payload = row_lookup.get(int(row_index))
            if row_payload is None:
                continue
            params = ParityParamRow.from_mapping(row_payload["params"]).to_gpu_row()
            normalized_params = {
                "max_stocks": int(params[0]),
                "order_investment_ratio": float(params[1]),
                "additional_buy_drop_rate": float(params[2]),
                "sell_profit_rate": float(params[3]),
                "additional_buy_priority": int(params[4]),
                "stop_loss_rate": float(params[5]),
                "max_splits_limit": int(params[6]),
                "max_inactivity_period": int(params[7]),
            }
            payload = collect_trade_event_parity_report(
                config=config,
                start_date=start_date,
                end_date=end_date,
                initial_cash=float(initial_cash),
                params=normalized_params,
                candidate_source_mode="tier",
                use_weekly_alpha_gate=False,
                parity_mode=exec_params["parity_mode"],
                universe_mode=universe_mode,
            )
            detail_path = None
            if args.out:
                detail_path = _build_decision_evidence_detail_path(args.out, int(row_index))
                _write_json_artifact(detail_path, payload)
            summary_payload = _summarize_decision_evidence_payload(
                int(row_index),
                payload,
                detail_path=detail_path,
            )
            row_payload["decision_evidence"] = summary_payload
            report["decision_evidence"]["rows"].append(summary_payload)

    report["summary"] = _apply_decision_evidence(
        report["summary"],
        report["decision_evidence"]["rows"],
        total_rows=len(report["rows"]),
    )
    if not report["summary"]["event_level_diff_collected"]:
        report["evidence_gaps"] = ["event_level_diff_not_collected"]
    elif not report["summary"]["decision_evidence_all_rows_covered"]:
        report["evidence_gaps"] = ["decision_level_evidence_partial"]
    else:
        report["evidence_gaps"] = []

    if args.out:
        _write_json_artifact(args.out, report)
        print(f"[cpu_gpu_parity_topk] report_saved path={args.out}")

    if report["summary"]["failed"] > 0 and not args.no_fail_on_mismatch:
        first_failed = next((r for r in report["rows"] if (r["compare"].get("mismatch_count") or 0) > 0), None)
        raise AssertionError(f"CPU/GPU parity failed. first_failed={first_failed}")

    print(f"[cpu_gpu_parity_topk] passed rows={report['summary']['passed']} tolerance={args.tolerance}")


if __name__ == "__main__":
    main()
