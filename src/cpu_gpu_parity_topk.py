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
import json
from dataclasses import dataclass
from datetime import datetime
import os
import urllib.parse

import pandas as pd
from sqlalchemy import create_engine

from .backtester import BacktestEngine
from .config_loader import load_config
from .data_handler import DataHandler
from .execution import BasicExecutionHandler
from .portfolio import Portfolio
from .strategy import MagicSplitStrategy


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
    df = pd.read_sql(query, sql_engine, params=[start_date, end_date], parse_dates=["date"])
    if df.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(df["date"]).sort_values().unique())


def _load_all_data_to_gpu(sql_engine, start_date: str, end_date: str):
    import cudf

    query = """
        SELECT
            dsp.stock_code AS ticker,
            dsp.date,
            dsp.open_price,
            dsp.high_price,
            dsp.low_price,
            dsp.close_price,
            dsp.volume,
            ci.atr_14_ratio
        FROM DailyStockPrice dsp
        LEFT JOIN CalculatedIndicators ci
          ON dsp.stock_code = ci.stock_code AND dsp.date = ci.date
        WHERE dsp.date BETWEEN %s AND %s
    """
    df_pd = pd.read_sql(query, sql_engine, params=[start_date, end_date], parse_dates=["date"])
    if df_pd.empty:
        return cudf.DataFrame()
    return cudf.from_pandas(df_pd).set_index(["ticker", "date"])


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
        params=[start_date, end_date, start_date],
        parse_dates=["date"],
    )
    if df_pd.empty:
        return cp.zeros((len(trading_dates_pd), len(all_tickers)), dtype=cp.int8)

    tier_wide = df_pd.pivot_table(index="date", columns="ticker", values="tier")
    liq_wide = df_pd.pivot_table(index="date", columns="ticker", values="liquidity_20d_avg_value")

    tier_asof = tier_wide.reindex(index=trading_dates_pd, columns=all_tickers).ffill().fillna(0).astype(int)
    liq_asof = liq_wide.reindex(index=trading_dates_pd, columns=all_tickers).ffill()

    min_liq = max(int(min_liquidity_20d_avg_value or 0), 0)
    if min_liq > 0:
        liq_mask = liq_asof.fillna(-1) >= min_liq
        tier_asof = tier_asof.where(liq_mask, 0)

    min_ratio = float(min_tier12_coverage_ratio or 0.0)
    if min_ratio > 0 and len(all_tickers) > 0:
        tier12_count = ((tier_asof > 0) & (tier_asof <= 2)).sum(axis=1).astype(int)
        ratio = tier12_count / float(len(all_tickers))
        failed = ratio[ratio < min_ratio]
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
    execution_params = base_config["execution_params"]
    strategy_params = dict(base_config.get("strategy_params", {}))
    strategy_params.update(row.to_strategy_kwargs())
    strategy_params["backtest_start_date"] = start_date
    strategy_params["backtest_end_date"] = end_date

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
        join="inner",
    ).dropna()
    if aligned.empty:
        return {"points": 0, "mismatch_count": None, "first_mismatch": None}

    diff = (aligned["cpu"] - aligned["gpu"]).abs()
    mismatches = diff > float(tolerance)
    mismatch_count = int(mismatches.sum())
    if mismatch_count <= 0:
        return {"points": int(len(aligned)), "mismatch_count": 0, "first_mismatch": None}

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


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU/GPU top-k parity harness (Issue #56).")
    parser.add_argument("--start-date", default=None, help="YYYYMMDD or YYYY-MM-DD (default: config).")
    parser.add_argument("--end-date", default=None, help="YYYYMMDD or YYYY-MM-DD (default: config).")
    parser.add_argument("--initial-cash", type=float, default=None, help="Override initial cash (default: config).")
    parser.add_argument("--params-csv", default=None, help="CSV with param columns (and optionally metrics).")
    parser.add_argument("--topk", type=int, default=3, help="Number of param rows to validate (default: 3).")
    parser.add_argument("--sort-by", default=None, help="Sort CSV by this metric column desc before topk selection.")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Absolute tolerance (default: 1e-3).")
    parser.add_argument("--out", default=None, help="Write JSON report to this path.")
    args = parser.parse_args()

    _require_gpu_stack()
    import cupy as cp
    from .backtest_strategy_gpu import run_magic_split_strategy_on_gpu

    config = load_config()
    start_date = _parse_date(args.start_date) if args.start_date else str(config["backtest_settings"]["start_date"])
    end_date = _parse_date(args.end_date) if args.end_date else str(config["backtest_settings"]["end_date"])
    initial_cash = float(args.initial_cash) if args.initial_cash is not None else float(config["backtest_settings"]["initial_cash"])

    if args.params_csv:
        rows = _load_params_from_csv(args.params_csv, topk=int(args.topk), sort_by=args.sort_by)
    else:
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

    all_data_gpu = _load_all_data_to_gpu(sql_engine, start_date, end_date)
    if all_data_gpu.empty:
        raise ValueError("No DailyStockPrice rows in the requested range.")

    all_tickers = sorted(all_data_gpu.index.get_level_values("ticker").unique().to_pandas().tolist())
    trading_date_indices_gpu = cp.arange(len(trading_dates_pd), dtype=cp.int32)
    weekly_filtered_gpu = _build_empty_weekly_filtered_gpu()
    strategy_cfg = config.get("strategy_params", {})
    tier_tensor = _load_tier_tensor(
        sql_engine,
        start_date,
        end_date,
        all_tickers,
        trading_dates_pd,
        min_liquidity_20d_avg_value=int(strategy_cfg.get("min_liquidity_20d_avg_value", 0) or 0),
        min_tier12_coverage_ratio=float(strategy_cfg.get("min_tier12_coverage_ratio", 0.0) or 0.0),
    )

    param_gpu = cp.asarray([row.to_gpu_row() for row in rows], dtype=cp.float32)
    exec_params = copy.deepcopy(config["execution_params"])
    exec_params["cooldown_period_days"] = config.get("strategy_params", {}).get("cooldown_period_days", 5)
    exec_params["candidate_source_mode"] = "tier"

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
        debug_mode=False,
    )
    gpu_curves = daily_values_gpu.get()

    # CPU runs (sequential), using a single DataHandler instance for caching.
    data_handler = DataHandler(db_config=config["database"])

    report = {
        "meta": {
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": initial_cash,
            "topk": int(len(rows)),
            "tolerance": float(args.tolerance),
            "candidate_source_mode": "tier",
        },
        "rows": [],
        "summary": {"failed": 0, "passed": 0},
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
        report["summary"]["failed"] += int(failed)
        report["summary"]["passed"] += int(not failed)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[cpu_gpu_parity_topk] report_saved path={args.out}")

    if report["summary"]["failed"] > 0:
        first_failed = next((r for r in report["rows"] if (r["compare"].get("mismatch_count") or 0) > 0), None)
        raise AssertionError(f"CPU/GPU parity failed. first_failed={first_failed}")

    print(f"[cpu_gpu_parity_topk] passed rows={report['summary']['passed']} tolerance={args.tolerance}")


if __name__ == "__main__":
    main()
