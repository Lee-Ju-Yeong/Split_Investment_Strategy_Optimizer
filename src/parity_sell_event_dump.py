"""
parity_sell_event_dump.py

CPU/GPU 매도 이벤트(체결 수량/체결가/정산액) 1:1 덤프 비교 도구.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# BOOTSTRAP: allow direct execution (`python src/parity_sell_event_dump.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    import sys

    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from .backtest.cpu.backtester import BacktestEngine
from .backtest.cpu.execution import BasicExecutionHandler
from .backtest.cpu.portfolio import Portfolio
from .backtest.cpu.strategy import MagicSplitStrategy
from .config_loader import load_config
from .data_handler import DataHandler
from .optimization.gpu.context import _build_db_connection_str, _ensure_core_deps, _ensure_gpu_deps
from .optimization.gpu.data_loading import (
    preload_all_data_to_gpu,
    preload_tier_data_to_tensor,
    preload_weekly_filtered_stocks_to_gpu,
)


@dataclass
class SellEvent:
    source: str
    date: str
    ticker: str
    quantity: int
    execution_price: float
    net_revenue: float
    reason: Optional[str] = None
    split: Optional[int] = None
    order: Optional[int] = None
    trigger_price: Optional[float] = None
    gross_revenue: Optional[float] = None


@dataclass
class BuyEvent:
    source: str
    date: str
    ticker: str
    quantity: int
    execution_price: float
    total_cost: float
    reason: Optional[str] = None
    split: Optional[int] = None
    order: Optional[int] = None
    trigger_price: Optional[float] = None
    gross_cost: Optional[float] = None


def _normalize_priority_for_cpu(value: Any) -> str:
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("lowest_order", "highest_drop"):
            return key
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return "lowest_order"
    return "highest_drop" if iv == 1 else "lowest_order"


def _normalize_param_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "param_id": int(row["param_id"]),
        "max_stocks": int(row["max_stocks"]),
        "order_investment_ratio": float(row["order_investment_ratio"]),
        "additional_buy_drop_rate": float(row["additional_buy_drop_rate"]),
        "sell_profit_rate": float(row["sell_profit_rate"]),
        "additional_buy_priority": int(row["additional_buy_priority"]),
        "stop_loss_rate": float(row["stop_loss_rate"]),
        "max_splits_limit": int(row["max_splits_limit"]),
        "max_inactivity_period": int(row["max_inactivity_period"]),
    }


def _load_param_row(params_csv: str, param_id: Optional[int]) -> Dict[str, Any]:
    _, pd = _ensure_core_deps()
    df = pd.read_csv(params_csv)
    if df.empty:
        raise ValueError(f"params_csv is empty: {params_csv}")

    if "param_id" in df.columns and param_id is not None:
        filtered = df[df["param_id"] == param_id]
        if filtered.empty:
            raise ValueError(f"param_id={param_id} not found in {params_csv}")
        row = filtered.iloc[0].to_dict()
    else:
        row = df.iloc[0].to_dict()
        if "param_id" not in row:
            row["param_id"] = 0
    return _normalize_param_row(row)


def _build_exec_params(
    config: Dict[str, Any],
    *,
    candidate_source_mode: str,
    use_weekly_alpha_gate: bool,
    parity_mode: str,
) -> Dict[str, Any]:
    strategy_params = dict(config["strategy_params"])
    execution_params = dict(config["execution_params"])
    execution_params["cooldown_period_days"] = strategy_params.get("cooldown_period_days", 5)
    execution_params["candidate_source_mode"] = candidate_source_mode
    execution_params["use_weekly_alpha_gate"] = bool(use_weekly_alpha_gate)
    execution_params["parity_mode"] = str(parity_mode).strip().lower()
    execution_params["tier_hysteresis_mode"] = strategy_params.get("tier_hysteresis_mode", "legacy")
    return execution_params


def _run_cpu_and_collect_trade_events(
    *,
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
    initial_cash: float,
    params: Dict[str, Any],
    candidate_source_mode: str,
    use_weekly_alpha_gate: bool,
) -> Tuple[List[SellEvent], List[BuyEvent]]:
    strategy_params = dict(config["strategy_params"])
    strategy_params.update(
        {
            "max_stocks": int(params["max_stocks"]),
            "order_investment_ratio": float(params["order_investment_ratio"]),
            "additional_buy_drop_rate": float(params["additional_buy_drop_rate"]),
            "sell_profit_rate": float(params["sell_profit_rate"]),
            "additional_buy_priority": _normalize_priority_for_cpu(params["additional_buy_priority"]),
            "stop_loss_rate": float(params["stop_loss_rate"]),
            "max_splits_limit": int(params["max_splits_limit"]),
            "max_inactivity_period": int(params["max_inactivity_period"]),
            "candidate_source_mode": candidate_source_mode,
            "use_weekly_alpha_gate": bool(use_weekly_alpha_gate),
            "backtest_start_date": start_date,
            "backtest_end_date": end_date,
        }
    )

    execution_params = dict(config["execution_params"])
    data_handler = DataHandler(db_config=config["database"], load_company_cache=False)
    strategy = MagicSplitStrategy(**strategy_params)
    portfolio = Portfolio(initial_cash=float(initial_cash), start_date=start_date, end_date=end_date)
    execution_handler = BasicExecutionHandler(
        buy_commission_rate=execution_params["buy_commission_rate"],
        sell_commission_rate=execution_params["sell_commission_rate"],
        sell_tax_rate=execution_params["sell_tax_rate"],
    )
    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        strategy=strategy,
        data_handler=data_handler,
        execution_handler=execution_handler,
    )
    final_portfolio = engine.run()

    sell_events: List[SellEvent] = []
    buy_events: List[BuyEvent] = []
    for trade in final_portfolio.trade_history:
        side = str(trade.trade_type).upper()
        if side == "SELL":
            sell_events.append(
                SellEvent(
                    source="cpu",
                    date=trade.date.strftime("%Y-%m-%d"),
                    ticker=str(trade.code),
                    quantity=int(trade.quantity),
                    execution_price=float(trade.sell_price),
                    net_revenue=float(trade.trade_value),
                    reason=str(trade.reason_for_trade),
                    order=int(trade.order),
                    trigger_price=float(trade.trigger_price) if trade.trigger_price is not None else None,
                    gross_revenue=float(trade.quantity * trade.sell_price),
                )
            )
        elif side == "BUY":
            buy_events.append(
                BuyEvent(
                    source="cpu",
                    date=trade.date.strftime("%Y-%m-%d"),
                    ticker=str(trade.code),
                    quantity=int(trade.quantity),
                    execution_price=float(trade.buy_price),
                    total_cost=float(trade.trade_value),
                    reason=str(trade.reason_for_trade),
                    order=int(trade.order),
                    trigger_price=float(trade.trigger_price) if trade.trigger_price is not None else None,
                    gross_cost=float(trade.quantity * trade.buy_price),
                )
            )
    return sell_events, buy_events


def _run_gpu_and_collect_sell_events(
    *,
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
    initial_cash: float,
    params: Dict[str, Any],
    candidate_source_mode: str,
    use_weekly_alpha_gate: bool,
    parity_mode: str,
) -> Tuple[List[SellEvent], List[BuyEvent], str]:
    cp, _, create_engine, run_magic_split_strategy_on_gpu = _ensure_gpu_deps()
    _, pd = _ensure_core_deps()

    db_connection_str = _build_db_connection_str(config["database"])
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

    param_matrix = cp.asarray(
        [
            [
                float(params["max_stocks"]),
                float(params["order_investment_ratio"]),
                float(params["additional_buy_drop_rate"]),
                float(params["sell_profit_rate"]),
                float(params["additional_buy_priority"]),
                float(params["stop_loss_rate"]),
                float(params["max_splits_limit"]),
                float(params["max_inactivity_period"]),
            ]
        ],
        dtype=cp.float32,
    )
    execution_params = _build_exec_params(
        config,
        candidate_source_mode=candidate_source_mode,
        use_weekly_alpha_gate=use_weekly_alpha_gate,
        parity_mode=parity_mode,
    )

    debug_stdout = io.StringIO()
    with redirect_stdout(debug_stdout):
        run_magic_split_strategy_on_gpu(
            initial_cash=float(initial_cash),
            param_combinations=param_matrix,
            all_data_gpu=all_data_gpu,
            weekly_filtered_gpu=weekly_filtered_gpu,
            trading_date_indices=trading_date_indices_gpu,
            trading_dates_pd_cpu=trading_dates_pd,
            all_tickers=all_tickers,
            execution_params=execution_params,
            max_splits_limit=int(params["max_splits_limit"]),
            tier_tensor=tier_tensor,
            debug_mode=True,
        )
    gpu_log_text = debug_stdout.getvalue()
    sell_events = _parse_gpu_sell_events(
        gpu_log_text,
        sell_commission_rate=float(config["execution_params"]["sell_commission_rate"]),
        sell_tax_rate=float(config["execution_params"]["sell_tax_rate"]),
    )
    buy_events = _parse_gpu_buy_events(
        gpu_log_text,
        trading_dates=[dt.strftime("%Y-%m-%d") for dt in trading_dates_pd],
        buy_commission_rate=float(config["execution_params"]["buy_commission_rate"]),
    )
    return sell_events, buy_events, gpu_log_text


def _parse_gpu_sell_events(log_text: str, sell_commission_rate: float, sell_tax_rate: float) -> List[SellEvent]:
    calc_re = re.compile(
        r"^\[GPU_SELL_CALC\]\s+"
        r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<ticker>\S+)"
        r"(?:\s+\(Split\s+(?P<split>\d+)\))?\s+\|\s+"
        r"Qty:\s+(?P<qty>[\d,]+)\s+\*\s+ExecPrice:\s+(?P<price>[\d,]+)\s+=\s+Revenue:\s+(?P<revenue>[\d,]+)"
    )
    price_re = re.compile(
        r"^\[GPU_SELL_PRICE\]\s+"
        r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<ticker>\S+)"
        r"(?:\s+\(Split\s+(?P<split>\d+)\))?\s+Reason:\s+(?P<reason>[^|]+)\|\s+"
        r"Target:\s+(?P<target>[-\d\.]+)\s+->\s+Exec:\s+(?P<exec>[-\d\.]+)\s+\|\s+High:\s+(?P<high>[-\d\.]+)"
    )

    cost_factor = 1.0 - float(sell_commission_rate) - float(sell_tax_rate)
    pending: Optional[SellEvent] = None
    events: List[SellEvent] = []

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        calc_match = calc_re.match(line)
        if calc_match:
            qty = int(calc_match.group("qty").replace(",", ""))
            exec_price = float(calc_match.group("price").replace(",", ""))
            gross_revenue = float(calc_match.group("revenue").replace(",", ""))
            pending = SellEvent(
                source="gpu",
                date=calc_match.group("date"),
                ticker=calc_match.group("ticker"),
                quantity=qty,
                execution_price=exec_price,
                gross_revenue=gross_revenue,
                net_revenue=float(math.floor(gross_revenue * cost_factor)),
                split=int(calc_match.group("split")) if calc_match.group("split") is not None else None,
            )
            continue

        price_match = price_re.match(line)
        if price_match:
            if pending is None:
                pending = SellEvent(
                    source="gpu",
                    date=price_match.group("date"),
                    ticker=price_match.group("ticker"),
                    quantity=0,
                    execution_price=float(price_match.group("exec")),
                    net_revenue=0.0,
                    split=int(price_match.group("split")) if price_match.group("split") is not None else None,
                )
            pending.reason = price_match.group("reason").strip()
            pending.trigger_price = float(price_match.group("target"))
            events.append(pending)
            pending = None

    return events


def _parse_gpu_buy_events(log_text: str, trading_dates: List[str], buy_commission_rate: float) -> List[BuyEvent]:
    new_buy_re = re.compile(
        r"^\[GPU_NEW_BUY_CALC\]\s+"
        r"(?P<day_idx>\d+),\s+Sim\s+0,\s+Stock\s+\d+\((?P<ticker>[^)]+)\)\s+\|\s+"
        r"Invest:\s+(?P<invest>[\d,]+)\s+/\s+ExecPrice:\s+(?P<price>[\d,]+)\s+=\s+Qty:\s+(?P<qty>[\d,]+)"
    )
    add_buy_summary_re = re.compile(
        r"^\[GPU_ADD_BUY_SUMMARY\]\s+Day\s+(?P<day_idx>\d+),\s+Sim\s+0\s+\|\s+Buys:\s+(?P<count>\d+)\s+\|"
    )
    add_buy_detail_re = re.compile(
        r"^└─\s+Stock\s+\d+\((?P<ticker>[^)]+)\)\s+\|\s+Qty:\s+(?P<qty>[\d,]+)\s+@\s+(?P<price>[\d,]+)$"
    )

    def _resolve_date(day_idx: int) -> str:
        if 0 <= day_idx < len(trading_dates):
            return trading_dates[day_idx]
        return f"day_idx_{day_idx}"

    events: List[BuyEvent] = []
    active_add_buy_day: Optional[int] = None

    for raw_line in log_text.splitlines():
        line = raw_line.strip()

        m_new = new_buy_re.match(line)
        if m_new:
            day_idx = int(m_new.group("day_idx"))
            qty = int(m_new.group("qty").replace(",", ""))
            exec_price = float(m_new.group("price").replace(",", ""))
            gross_cost = float(qty * exec_price)
            commission = float(math.floor(gross_cost * float(buy_commission_rate)))
            total_cost = gross_cost + commission
            events.append(
                BuyEvent(
                    source="gpu",
                    date=_resolve_date(day_idx),
                    ticker=m_new.group("ticker"),
                    quantity=qty,
                    execution_price=exec_price,
                    total_cost=total_cost,
                    reason="신규 매수",
                    split=0,
                    gross_cost=gross_cost,
                )
            )
            continue

        m_summary = add_buy_summary_re.match(line)
        if m_summary:
            active_add_buy_day = int(m_summary.group("day_idx"))
            continue

        m_detail = add_buy_detail_re.match(line)
        if m_detail and active_add_buy_day is not None:
            qty = int(m_detail.group("qty").replace(",", ""))
            exec_price = float(m_detail.group("price").replace(",", ""))
            gross_cost = float(qty * exec_price)
            commission = float(math.floor(gross_cost * float(buy_commission_rate)))
            total_cost = gross_cost + commission
            events.append(
                BuyEvent(
                    source="gpu",
                    date=_resolve_date(active_add_buy_day),
                    ticker=m_detail.group("ticker"),
                    quantity=qty,
                    execution_price=exec_price,
                    total_cost=total_cost,
                    reason="추가 매수(하락)",
                    split=None,
                    gross_cost=gross_cost,
                )
            )
            continue

        if line and not line.startswith("└─"):
            active_add_buy_day = None

    return events


def _pair_events(cpu_events: List[SellEvent], gpu_events: List[SellEvent]) -> List[Dict[str, Any]]:
    cpu_sorted = sorted(cpu_events, key=lambda e: (e.date, e.ticker, e.quantity, e.execution_price, e.order or 0))
    gpu_sorted = sorted(gpu_events, key=lambda e: (e.date, e.ticker, e.quantity, e.execution_price, e.split or 0))

    max_len = max(len(cpu_sorted), len(gpu_sorted))
    rows: List[Dict[str, Any]] = []
    for idx in range(max_len):
        cpu = cpu_sorted[idx] if idx < len(cpu_sorted) else None
        gpu = gpu_sorted[idx] if idx < len(gpu_sorted) else None
        row = {
            "pair_index": idx,
            "cpu": None if cpu is None else cpu.__dict__,
            "gpu": None if gpu is None else gpu.__dict__,
            "matched": False,
            "diff": {},
        }
        if cpu is not None and gpu is not None:
            qty_diff = int(cpu.quantity - gpu.quantity)
            price_diff = float(cpu.execution_price - gpu.execution_price)
            net_diff = float(cpu.net_revenue - gpu.net_revenue)
            same_key = (cpu.date == gpu.date) and (cpu.ticker == gpu.ticker)
            row["diff"] = {
                "date_same": cpu.date == gpu.date,
                "ticker_same": cpu.ticker == gpu.ticker,
                "quantity_diff": qty_diff,
                "execution_price_diff": price_diff,
                "net_revenue_diff": net_diff,
            }
            row["matched"] = same_key and qty_diff == 0 and abs(price_diff) < 1e-6 and abs(net_diff) < 1e-6
        rows.append(row)
    return rows


def _pair_buy_events(cpu_events: List[BuyEvent], gpu_events: List[BuyEvent]) -> List[Dict[str, Any]]:
    cpu_sorted = sorted(cpu_events, key=lambda e: (e.date, e.ticker, e.quantity, e.execution_price, e.order or 0))
    gpu_sorted = sorted(gpu_events, key=lambda e: (e.date, e.ticker, e.quantity, e.execution_price))

    max_len = max(len(cpu_sorted), len(gpu_sorted))
    rows: List[Dict[str, Any]] = []
    for idx in range(max_len):
        cpu = cpu_sorted[idx] if idx < len(cpu_sorted) else None
        gpu = gpu_sorted[idx] if idx < len(gpu_sorted) else None
        row = {
            "pair_index": idx,
            "cpu": None if cpu is None else cpu.__dict__,
            "gpu": None if gpu is None else gpu.__dict__,
            "matched": False,
            "diff": {},
        }
        if cpu is not None and gpu is not None:
            qty_diff = int(cpu.quantity - gpu.quantity)
            price_diff = float(cpu.execution_price - gpu.execution_price)
            cost_diff = float(cpu.total_cost - gpu.total_cost)
            same_key = (cpu.date == gpu.date) and (cpu.ticker == gpu.ticker)
            row["diff"] = {
                "date_same": cpu.date == gpu.date,
                "ticker_same": cpu.ticker == gpu.ticker,
                "quantity_diff": qty_diff,
                "execution_price_diff": price_diff,
                "total_cost_diff": cost_diff,
            }
            row["matched"] = same_key and qty_diff == 0 and abs(price_diff) < 1e-6 and abs(cost_diff) < 1e-6
        rows.append(row)
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU/GPU sell-event 1:1 dump comparator")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--params-csv", required=True)
    parser.add_argument("--param-id", type=int, default=None, help="Optional param_id filter in CSV")
    parser.add_argument("--candidate-source-mode", choices=["weekly", "hybrid_transition", "tier"], default="tier")
    parser.add_argument("--parity-mode", choices=["fast", "strict"], default="strict")
    parser.add_argument("--use-weekly-alpha-gate", action="store_true", default=False)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--gpu-log-out", default=None, help="Optional raw GPU debug log path")
    return parser


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config()
    initial_cash = float(config["backtest_settings"]["initial_cash"])

    params = _load_param_row(args.params_csv, args.param_id)
    cpu_sell_events, cpu_buy_events = _run_cpu_and_collect_trade_events(
        config=config,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=initial_cash,
        params=params,
        candidate_source_mode=args.candidate_source_mode,
        use_weekly_alpha_gate=args.use_weekly_alpha_gate,
    )
    gpu_sell_events, gpu_buy_events, gpu_log_text = _run_gpu_and_collect_sell_events(
        config=config,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=initial_cash,
        params=params,
        candidate_source_mode=args.candidate_source_mode,
        use_weekly_alpha_gate=args.use_weekly_alpha_gate,
        parity_mode=args.parity_mode,
    )

    sell_pairs = _pair_events(cpu_sell_events, gpu_sell_events)
    buy_pairs = _pair_buy_events(cpu_buy_events, gpu_buy_events)
    matched_sell_pairs = sum(1 for row in sell_pairs if row["matched"])
    matched_buy_pairs = sum(1 for row in buy_pairs if row["matched"])
    payload = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "params": params,
        "candidate_source_mode": args.candidate_source_mode,
        "parity_mode": args.parity_mode,
        "cpu_sell_events_count": len(cpu_sell_events),
        "gpu_sell_events_count": len(gpu_sell_events),
        "cpu_buy_events_count": len(cpu_buy_events),
        "gpu_buy_events_count": len(gpu_buy_events),
        "sell_paired_count": len(sell_pairs),
        "sell_matched_pairs": matched_sell_pairs,
        "sell_mismatched_pairs": len(sell_pairs) - matched_sell_pairs,
        "buy_paired_count": len(buy_pairs),
        "buy_matched_pairs": matched_buy_pairs,
        "buy_mismatched_pairs": len(buy_pairs) - matched_buy_pairs,
        "cpu_sell_events": [event.__dict__ for event in cpu_sell_events],
        "gpu_sell_events": [event.__dict__ for event in gpu_sell_events],
        "cpu_buy_events": [event.__dict__ for event in cpu_buy_events],
        "gpu_buy_events": [event.__dict__ for event in gpu_buy_events],
        "sell_pairs": sell_pairs,
        "buy_pairs": buy_pairs,
    }
    _save_json(args.out, payload)
    if args.gpu_log_out:
        Path(args.gpu_log_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.gpu_log_out).write_text(gpu_log_text, encoding="utf-8")

    print(f"[sell_event_dump] report saved: {args.out}")
    print(
        "[sell_event_dump] "
        f"cpu_sell_events={len(cpu_sell_events)}, gpu_sell_events={len(gpu_sell_events)}, "
        f"sell_mismatched_pairs={len(sell_pairs) - matched_sell_pairs}"
    )
    print(
        "[sell_event_dump] "
        f"cpu_buy_events={len(cpu_buy_events)}, gpu_buy_events={len(gpu_buy_events)}, "
        f"buy_mismatched_pairs={len(buy_pairs) - matched_buy_pairs}"
    )
    if args.gpu_log_out:
        print(f"[sell_event_dump] gpu_log saved: {args.gpu_log_out}")


if __name__ == "__main__":
    main()
