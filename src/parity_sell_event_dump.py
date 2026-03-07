"""
parity_sell_event_dump.py

CPU/GPU 매도 이벤트(체결 수량/체결가/정산액) 1:1 덤프 비교 도구.
"""

from __future__ import annotations

import argparse
import io
import inspect
import json
import math
import os
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
    build_empty_weekly_filtered_gpu,
    preload_all_data_to_gpu,
    preload_pit_universe_mask_to_tensor,
    preload_tier_data_to_tensor,
)
from .price_policy import is_adjusted_price_basis, resolve_price_policy
from .universe_policy import resolve_universe_mode


_TRIGGER_PRICE_TOLERANCE = 1e-3
_POSITION_AVG_PRICE_TOLERANCE = 2e-3


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


@dataclass
class DailySnapshot:
    source: str
    date: str
    total_value: float
    cash: float
    stock_count: int


@dataclass
class PositionSnapshot:
    source: str
    date: str
    ticker: str
    holdings: int
    quantity: int
    avg_buy_price: float
    current_price: float
    total_value: float


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


def _normalize_priority_for_gpu(value: Any) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "highest_drop":
            return 1
        if key == "lowest_order":
            return 0
    try:
        iv = int(float(value))
    except (TypeError, ValueError):
        return 0
    return 1 if iv == 1 else 0


def _normalize_param_row(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_param_id = row.get("param_id", 0)
    try:
        param_id = int(raw_param_id)
    except (TypeError, ValueError):
        param_id = 0
    return {
        "param_id": param_id,
        "max_stocks": int(row["max_stocks"]),
        "order_investment_ratio": float(row["order_investment_ratio"]),
        "additional_buy_drop_rate": float(row["additional_buy_drop_rate"]),
        "sell_profit_rate": float(row["sell_profit_rate"]),
        "additional_buy_priority": _normalize_priority_for_gpu(row["additional_buy_priority"]),
        "stop_loss_rate": float(row["stop_loss_rate"]),
        "max_splits_limit": int(row["max_splits_limit"]),
        "max_inactivity_period": int(row["max_inactivity_period"]),
    }


def _load_param_row_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    strategy_params = dict(config.get("strategy_params", {}))
    return _normalize_param_row(
        {
            "param_id": 0,
            "max_stocks": strategy_params.get("max_stocks", 20),
            "order_investment_ratio": strategy_params.get("order_investment_ratio", 0.02),
            "additional_buy_drop_rate": strategy_params.get("additional_buy_drop_rate", 0.04),
            "sell_profit_rate": strategy_params.get("sell_profit_rate", 0.04),
            "additional_buy_priority": strategy_params.get("additional_buy_priority", "lowest_order"),
            "stop_loss_rate": strategy_params.get("stop_loss_rate", -0.15),
            "max_splits_limit": strategy_params.get("max_splits_limit", 10),
            "max_inactivity_period": strategy_params.get("max_inactivity_period", 90),
        }
    )


def _load_param_row_from_csv(params_csv: str, param_id: Optional[int]) -> Dict[str, Any]:
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


def _build_cpu_daily_snapshots(final_portfolio: Portfolio) -> List[DailySnapshot]:
    snapshots: List[DailySnapshot] = []
    for raw in list(getattr(final_portfolio, "daily_snapshot_history", [])):
        snapshots.append(
            DailySnapshot(
                source="cpu",
                date=str(raw["date"].strftime("%Y-%m-%d")),
                total_value=float(raw["total_value"]),
                cash=float(raw["cash"]),
                stock_count=int(raw["stock_count"]),
            )
        )
    return snapshots


def _build_cpu_position_snapshots(final_portfolio: Portfolio) -> List[PositionSnapshot]:
    snapshots: List[PositionSnapshot] = []
    for raw in list(getattr(final_portfolio, "daily_positions_snapshot_history", [])):
        date_str = str(raw["date"].strftime("%Y-%m-%d"))
        for position in raw.get("positions", []):
            snapshots.append(
                PositionSnapshot(
                    source="cpu",
                    date=date_str,
                    ticker=str(position["Ticker"]),
                    holdings=int(position["Holdings"]),
                    quantity=int(position["Quantity"]),
                    avg_buy_price=float(position["Avg Buy Price"]),
                    current_price=float(position["Current Price"]),
                    total_value=float(position["Total Value"]),
                )
            )
    return snapshots


def _run_cpu_and_collect_trade_events(
    *,
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
    initial_cash: float,
    params: Dict[str, Any],
    candidate_source_mode: str,
    use_weekly_alpha_gate: bool,
    universe_mode: str,
) -> Tuple[List[SellEvent], List[BuyEvent], List[DailySnapshot], List[PositionSnapshot]]:
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
    strategy_params_for_data = dict(strategy_params)

    execution_params = dict(config["execution_params"])
    strategy_allowed = {
        key
        for key in inspect.signature(MagicSplitStrategy.__init__).parameters.keys()
        if key != "self"
    }
    execution_allowed = {
        key
        for key in inspect.signature(BasicExecutionHandler.__init__).parameters.keys()
        if key != "self"
    }
    strategy_params = {k: v for k, v in strategy_params.items() if k in strategy_allowed}
    execution_params = {k: v for k, v in execution_params.items() if k in execution_allowed}

    data_handler = DataHandler(
        db_config=config["database"],
        strategy_params=strategy_params_for_data,
        universe_mode=universe_mode,
    )
    strategy = MagicSplitStrategy(**strategy_params)
    portfolio = Portfolio(
        initial_cash=float(initial_cash),
        start_date=start_date,
        end_date=end_date,
        capture_daily_positions_snapshot=True,
    )
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
    return (
        sell_events,
        buy_events,
        _build_cpu_daily_snapshots(final_portfolio),
        _build_cpu_position_snapshots(final_portfolio),
    )


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
    universe_mode: str,
) -> Tuple[List[SellEvent], List[BuyEvent], List[DailySnapshot], List[PositionSnapshot], str]:
    cp, _, create_engine, run_magic_split_strategy_on_gpu = _ensure_gpu_deps()
    _, pd = _ensure_core_deps()

    strategy_cfg = dict(config.get("strategy_params", {}))
    price_basis, adjusted_gate_start_date = resolve_price_policy(strategy_cfg)
    use_adjusted_prices = is_adjusted_price_basis(price_basis)
    db_connection_str = _build_db_connection_str(config["database"])
    all_data_gpu = preload_all_data_to_gpu(
        db_connection_str,
        start_date,
        end_date,
        use_adjusted_prices=use_adjusted_prices,
        adjusted_price_gate_start_date=adjusted_gate_start_date,
        universe_mode=universe_mode,
    )
    weekly_filtered_gpu = build_empty_weekly_filtered_gpu()

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
    tier_tensor = preload_tier_data_to_tensor(
        db_connection_str,
        start_date,
        end_date,
        all_tickers,
        trading_dates_pd,
        universe_mode=universe_mode,
    )
    pit_universe_mask_tensor = preload_pit_universe_mask_to_tensor(
        db_connection_str,
        start_date,
        end_date,
        all_tickers,
        trading_dates_pd,
    )

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
            pit_universe_mask_tensor=pit_universe_mask_tensor,
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
    daily_snapshots = _parse_gpu_daily_snapshots(gpu_log_text)
    position_snapshots = _parse_gpu_position_snapshots(gpu_log_text)
    return sell_events, buy_events, daily_snapshots, position_snapshots, gpu_log_text


def _parse_gpu_daily_snapshots(log_text: str) -> List[DailySnapshot]:
    daily_re = re.compile(
        r"^\[PARITY_SNAPSHOT_DAILY\]\s+"
        r"date=(?P<date>\d{4}-\d{2}-\d{2})\|"
        r"total_value=(?P<total_value>[-\d\.]+)\|"
        r"cash=(?P<cash>[-\d\.]+)\|"
        r"stock_count=(?P<stock_count>\d+)$"
    )
    snapshots: List[DailySnapshot] = []
    for raw_line in log_text.splitlines():
        match = daily_re.match(raw_line.strip())
        if not match:
            continue
        snapshots.append(
            DailySnapshot(
                source="gpu",
                date=match.group("date"),
                total_value=float(match.group("total_value")),
                cash=float(match.group("cash")),
                stock_count=int(match.group("stock_count")),
            )
        )
    return snapshots


def _parse_gpu_position_snapshots(log_text: str) -> List[PositionSnapshot]:
    position_re = re.compile(
        r"^\[PARITY_SNAPSHOT_POSITION\]\s+"
        r"date=(?P<date>\d{4}-\d{2}-\d{2})\|"
        r"ticker=(?P<ticker>\S+)\|"
        r"holdings=(?P<holdings>\d+)\|"
        r"quantity=(?P<quantity>\d+)\|"
        r"avg_buy_price=(?P<avg_buy_price>[-\d\.]+)\|"
        r"current_price=(?P<current_price>[-\d\.]+)\|"
        r"total_value=(?P<total_value>[-\d\.]+)$"
    )
    snapshots: List[PositionSnapshot] = []
    for raw_line in log_text.splitlines():
        match = position_re.match(raw_line.strip())
        if not match:
            continue
        snapshots.append(
            PositionSnapshot(
                source="gpu",
                date=match.group("date"),
                ticker=match.group("ticker"),
                holdings=int(match.group("holdings")),
                quantity=int(match.group("quantity")),
                avg_buy_price=float(match.group("avg_buy_price")),
                current_price=float(match.group("current_price")),
                total_value=float(match.group("total_value")),
            )
        )
    return snapshots


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


def _floor_fee_from_gross_cost(gross_cost: float, rate: float) -> float:
    rate_text = f"{float(rate):.12f}".rstrip("0").rstrip(".")
    if not rate_text or rate_text == "0":
        return 0.0
    if "." in rate_text:
        decimals = len(rate_text.split(".", 1)[1])
        scale = 10 ** decimals
        numerator = int(round(float(rate_text) * scale))
    else:
        scale = 1
        numerator = int(rate_text)

    gross_int = int(round(gross_cost))
    return float((gross_int * numerator) // scale)


def _parse_gpu_buy_events(log_text: str, trading_dates: List[str], buy_commission_rate: float) -> List[BuyEvent]:
    new_buy_re = re.compile(
        r"^\[GPU_NEW_BUY_CALC\]\s+"
        r"(?P<day_idx>\d+),\s+Sim\s+0,\s+Stock\s+\d+\((?P<ticker>[^)]+)\)\s+\|\s+"
        r"(?:Target:\s+(?P<target>[-\d\.,]+)\s+\|\s+)?"
        r"Invest:\s+(?P<invest>[\d,]+)\s+/\s+ExecPrice:\s+(?P<price>[\d,]+)\s+=\s+Qty:\s+(?P<qty>[\d,]+)"
    )
    add_buy_summary_re = re.compile(
        r"^\[GPU_ADD_BUY_SUMMARY\]\s+Day\s+(?P<day_idx>\d+),\s+Sim\s+0\s+\|\s+Buys:\s+(?P<count>\d+)\s+\|"
    )
    add_buy_detail_re = re.compile(
        r"^└─\s+Stock\s+\d+\((?P<ticker>[^)]+)\)\s+\|\s+"
        r"Split:\s+(?P<split>\d+)\s+\|\s+"
        r"Target:\s+(?P<target>[-\d\.,]+)\s+\|\s+"
        r"Qty:\s+(?P<qty>[\d,]+)\s+@\s+(?P<price>[\d,]+)$"
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
            commission = _floor_fee_from_gross_cost(gross_cost, float(buy_commission_rate))
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
                    trigger_price=float(str(m_new.group("target")).replace(",", "")) if m_new.group("target") is not None else None,
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
            commission = _floor_fee_from_gross_cost(gross_cost, float(buy_commission_rate))
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
                    split=int(m_detail.group("split")),
                    trigger_price=float(str(m_detail.group("target")).replace(",", "")),
                    gross_cost=gross_cost,
                )
            )
            continue

        if line and not line.startswith("└─"):
            active_add_buy_day = None

    return events


def _normalize_trade_reason(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None

    mapping = {
        "수익 실현": "profit_taking",
        "Profit-Taking": "profit_taking",
        "손절매 (평균가 기준)": "stop_loss",
        "Stop-Loss": "stop_loss",
        "매매 미발생 기간 초과": "inactivity",
        "Inactivity": "inactivity",
        "Tier3": "tier3",
        "신규 진입": "new_buy",
        "신규 매수": "new_buy",
        "추가 매수(하락)": "add_buy_drop",
    }
    return mapping.get(text, text.lower())


def _trigger_price_diff(cpu_value: Optional[float], gpu_value: Optional[float]) -> Optional[float]:
    if cpu_value is None and gpu_value is None:
        return 0.0
    if cpu_value is None or gpu_value is None:
        return None
    return float(cpu_value) - float(gpu_value)


def _sell_order_match(cpu: SellEvent, gpu: SellEvent) -> Optional[bool]:
    if cpu.order is None or gpu.split is None:
        return None
    return int(cpu.order) == int(gpu.split) + 1


def _buy_order_semantics_match(cpu: BuyEvent, gpu: BuyEvent) -> Optional[bool]:
    if cpu.order is None:
        return None

    normalized_reason = _normalize_trade_reason(gpu.reason)
    if normalized_reason == "new_buy":
        return int(cpu.order) == 1
    if normalized_reason == "add_buy_drop":
        if int(cpu.order) < 2:
            return False
        if gpu.split is None:
            return None
        return int(cpu.order) == int(gpu.split) + 1
    return None


def _event_bucket_key(event: Any) -> Tuple[str, str]:
    return str(event.date), str(event.ticker)


def _bucket_events(events: List[Any]) -> Dict[Tuple[str, str], List[Any]]:
    buckets: Dict[Tuple[str, str], List[Any]] = {}
    for event in events:
        buckets.setdefault(_event_bucket_key(event), []).append(event)
    return buckets


def _build_sell_pair_row(pair_index: int, cpu: Optional[SellEvent], gpu: Optional[SellEvent]) -> Dict[str, Any]:
    row = {
        "pair_index": pair_index,
        "cpu": None if cpu is None else cpu.__dict__,
        "gpu": None if gpu is None else gpu.__dict__,
        "matched": False,
        "diff": {},
    }
    if cpu is None or gpu is None:
        return row

    qty_diff = int(cpu.quantity - gpu.quantity)
    price_diff = float(cpu.execution_price - gpu.execution_price)
    net_diff = float(cpu.net_revenue - gpu.net_revenue)
    trigger_diff = _trigger_price_diff(cpu.trigger_price, gpu.trigger_price)
    reason_same = _normalize_trade_reason(cpu.reason) == _normalize_trade_reason(gpu.reason)
    order_match = _sell_order_match(cpu, gpu)
    same_key = (cpu.date == gpu.date) and (cpu.ticker == gpu.ticker)
    row["diff"] = {
        "date_same": cpu.date == gpu.date,
        "ticker_same": cpu.ticker == gpu.ticker,
        "quantity_diff": qty_diff,
        "execution_price_diff": price_diff,
        "net_revenue_diff": net_diff,
        "reason_same": reason_same,
        "trigger_price_diff": trigger_diff,
        "order_match": order_match,
    }
    trigger_same = trigger_diff is not None and abs(trigger_diff) < _TRIGGER_PRICE_TOLERANCE
    order_ok = True if order_match is None else bool(order_match)
    row["matched"] = (
        same_key
        and qty_diff == 0
        and abs(price_diff) < 1e-6
        and abs(net_diff) < 1e-6
        and reason_same
        and trigger_same
        and order_ok
    )
    return row


def _build_buy_pair_row(pair_index: int, cpu: Optional[BuyEvent], gpu: Optional[BuyEvent]) -> Dict[str, Any]:
    row = {
        "pair_index": pair_index,
        "cpu": None if cpu is None else cpu.__dict__,
        "gpu": None if gpu is None else gpu.__dict__,
        "matched": False,
        "diff": {},
    }
    if cpu is None or gpu is None:
        return row

    qty_diff = int(cpu.quantity - gpu.quantity)
    price_diff = float(cpu.execution_price - gpu.execution_price)
    cost_diff = float(cpu.total_cost - gpu.total_cost)
    trigger_diff = _trigger_price_diff(cpu.trigger_price, gpu.trigger_price)
    reason_same = _normalize_trade_reason(cpu.reason) == _normalize_trade_reason(gpu.reason)
    order_match = _buy_order_semantics_match(cpu, gpu)
    same_key = (cpu.date == gpu.date) and (cpu.ticker == gpu.ticker)
    row["diff"] = {
        "date_same": cpu.date == gpu.date,
        "ticker_same": cpu.ticker == gpu.ticker,
        "quantity_diff": qty_diff,
        "execution_price_diff": price_diff,
        "total_cost_diff": cost_diff,
        "reason_same": reason_same,
        "trigger_price_diff": trigger_diff,
        "order_match": order_match,
    }
    trigger_same = trigger_diff is not None and abs(trigger_diff) < _TRIGGER_PRICE_TOLERANCE
    order_ok = True if order_match is None else bool(order_match)
    row["matched"] = (
        same_key
        and qty_diff == 0
        and abs(price_diff) < 1e-6
        and abs(cost_diff) < 1e-6
        and reason_same
        and trigger_same
        and order_ok
    )
    return row


def _sell_match_score(cpu: SellEvent, gpu: SellEvent) -> Tuple[float, ...]:
    trigger_diff = _trigger_price_diff(cpu.trigger_price, gpu.trigger_price)
    order_match = _sell_order_match(cpu, gpu)
    order_penalty = 0 if order_match in (None, True) else 1
    trigger_penalty = 0 if trigger_diff is not None else 1
    return (
        abs(int(cpu.quantity - gpu.quantity)),
        abs(float(cpu.execution_price - gpu.execution_price)),
        abs(float(cpu.net_revenue - gpu.net_revenue)),
        0 if _normalize_trade_reason(cpu.reason) == _normalize_trade_reason(gpu.reason) else 1,
        trigger_penalty,
        abs(trigger_diff) if trigger_diff is not None else 0.0,
        order_penalty,
    )


def _buy_match_score(cpu: BuyEvent, gpu: BuyEvent) -> Tuple[float, ...]:
    trigger_diff = _trigger_price_diff(cpu.trigger_price, gpu.trigger_price)
    order_match = _buy_order_semantics_match(cpu, gpu)
    order_penalty = 0 if order_match in (None, True) else 1
    trigger_penalty = 0 if trigger_diff is not None else 1
    return (
        abs(int(cpu.quantity - gpu.quantity)),
        abs(float(cpu.execution_price - gpu.execution_price)),
        abs(float(cpu.total_cost - gpu.total_cost)),
        0 if _normalize_trade_reason(cpu.reason) == _normalize_trade_reason(gpu.reason) else 1,
        trigger_penalty,
        abs(trigger_diff) if trigger_diff is not None else 0.0,
        order_penalty,
    )


def _pair_bucket_events(
    cpu_events: List[Any],
    gpu_events: List[Any],
    *,
    cpu_sort_key,
    gpu_sort_key,
    build_row,
    match_score,
) -> List[Dict[str, Any]]:
    cpu_buckets = _bucket_events(cpu_events)
    gpu_buckets = _bucket_events(gpu_events)
    bucket_keys = sorted(set(cpu_buckets.keys()) | set(gpu_buckets.keys()))

    rows: List[Dict[str, Any]] = []
    pair_index = 0
    for bucket_key in bucket_keys:
        cpu_bucket = sorted(cpu_buckets.get(bucket_key, []), key=cpu_sort_key)
        gpu_bucket = sorted(gpu_buckets.get(bucket_key, []), key=gpu_sort_key)
        remaining_gpu = list(gpu_bucket)

        for cpu_event in cpu_bucket:
            if not remaining_gpu:
                rows.append(build_row(pair_index, cpu_event, None))
                pair_index += 1
                continue

            best_pos = min(
                range(len(remaining_gpu)),
                key=lambda idx: (match_score(cpu_event, remaining_gpu[idx]), gpu_sort_key(remaining_gpu[idx])),
            )
            gpu_event = remaining_gpu.pop(best_pos)
            rows.append(build_row(pair_index, cpu_event, gpu_event))
            pair_index += 1

        for gpu_event in remaining_gpu:
            rows.append(build_row(pair_index, None, gpu_event))
            pair_index += 1
    return rows


def _pair_events(cpu_events: List[SellEvent], gpu_events: List[SellEvent]) -> List[Dict[str, Any]]:
    return _pair_bucket_events(
        cpu_events,
        gpu_events,
        cpu_sort_key=lambda e: (e.quantity, e.execution_price, e.order or 0, e.trigger_price or 0.0),
        gpu_sort_key=lambda e: (e.quantity, e.execution_price, e.split or 0, e.trigger_price or 0.0),
        build_row=_build_sell_pair_row,
        match_score=_sell_match_score,
    )


def _pair_buy_events(cpu_events: List[BuyEvent], gpu_events: List[BuyEvent]) -> List[Dict[str, Any]]:
    return _pair_bucket_events(
        cpu_events,
        gpu_events,
        cpu_sort_key=lambda e: (e.quantity, e.execution_price, e.order or 0, e.trigger_price or 0.0),
        gpu_sort_key=lambda e: (e.quantity, e.execution_price, e.split or 0, e.trigger_price or 0.0),
        build_row=_build_buy_pair_row,
        match_score=_buy_match_score,
    )


def _pair_keyed_rows(
    cpu_rows: List[Any],
    gpu_rows: List[Any],
    *,
    key_fn,
    build_row,
) -> List[Dict[str, Any]]:
    cpu_map = {key_fn(row): row for row in cpu_rows}
    gpu_map = {key_fn(row): row for row in gpu_rows}
    keys = sorted(set(cpu_map.keys()) | set(gpu_map.keys()))
    return [build_row(index, cpu_map.get(key), gpu_map.get(key)) for index, key in enumerate(keys)]


def _build_daily_snapshot_pair_row(
    pair_index: int,
    cpu: Optional[DailySnapshot],
    gpu: Optional[DailySnapshot],
) -> Dict[str, Any]:
    row = {
        "pair_index": pair_index,
        "cpu": None if cpu is None else cpu.__dict__,
        "gpu": None if gpu is None else gpu.__dict__,
        "matched": False,
        "diff": {},
    }
    if cpu is None or gpu is None:
        return row

    total_value_diff = float(cpu.total_value - gpu.total_value)
    cash_diff = float(cpu.cash - gpu.cash)
    stock_count_diff = int(cpu.stock_count - gpu.stock_count)
    row["diff"] = {
        "date_same": cpu.date == gpu.date,
        "total_value_diff": total_value_diff,
        "cash_diff": cash_diff,
        "stock_count_diff": stock_count_diff,
    }
    row["matched"] = (
        cpu.date == gpu.date
        and abs(total_value_diff) < 1e-6
        and abs(cash_diff) < 1e-6
        and stock_count_diff == 0
    )
    return row


def _build_position_snapshot_pair_row(
    pair_index: int,
    cpu: Optional[PositionSnapshot],
    gpu: Optional[PositionSnapshot],
) -> Dict[str, Any]:
    row = {
        "pair_index": pair_index,
        "cpu": None if cpu is None else cpu.__dict__,
        "gpu": None if gpu is None else gpu.__dict__,
        "matched": False,
        "diff": {},
    }
    if cpu is None or gpu is None:
        return row

    quantity_diff = int(cpu.quantity - gpu.quantity)
    holdings_diff = int(cpu.holdings - gpu.holdings)
    avg_buy_price_diff = float(cpu.avg_buy_price - gpu.avg_buy_price)
    current_price_diff = float(cpu.current_price - gpu.current_price)
    total_value_diff = float(cpu.total_value - gpu.total_value)
    row["diff"] = {
        "date_same": cpu.date == gpu.date,
        "ticker_same": cpu.ticker == gpu.ticker,
        "holdings_diff": holdings_diff,
        "quantity_diff": quantity_diff,
        "avg_buy_price_diff": avg_buy_price_diff,
        "current_price_diff": current_price_diff,
        "total_value_diff": total_value_diff,
    }
    row["matched"] = (
        cpu.date == gpu.date
        and cpu.ticker == gpu.ticker
        and holdings_diff == 0
        and quantity_diff == 0
        and abs(avg_buy_price_diff) < _POSITION_AVG_PRICE_TOLERANCE
        and abs(current_price_diff) < 1e-6
        and abs(total_value_diff) < 1e-6
    )
    return row


def _pair_daily_snapshots(
    cpu_rows: List[DailySnapshot],
    gpu_rows: List[DailySnapshot],
) -> List[Dict[str, Any]]:
    return _pair_keyed_rows(
        cpu_rows,
        gpu_rows,
        key_fn=lambda row: str(row.date),
        build_row=_build_daily_snapshot_pair_row,
    )


def _pair_position_snapshots(
    cpu_rows: List[PositionSnapshot],
    gpu_rows: List[PositionSnapshot],
) -> List[Dict[str, Any]]:
    return _pair_keyed_rows(
        cpu_rows,
        gpu_rows,
        key_fn=lambda row: (str(row.date), str(row.ticker)),
        build_row=_build_position_snapshot_pair_row,
    )


def _snapshots_have_required_fields(rows: List[Any], required_fields: Tuple[str, ...]) -> bool:
    return all(getattr(row, field, None) is not None for row in rows for field in required_fields)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU/GPU sell-event 1:1 dump comparator")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--params-csv", default=None)
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


def collect_trade_event_parity_report(
    *,
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
    initial_cash: float,
    params: Dict[str, Any],
    candidate_source_mode: str,
    use_weekly_alpha_gate: bool,
    parity_mode: str,
    universe_mode: str,
) -> Dict[str, Any]:
    cpu_sell_events, cpu_buy_events, cpu_daily_snapshots, cpu_position_snapshots = _run_cpu_and_collect_trade_events(
        config=config,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        params=params,
        candidate_source_mode=candidate_source_mode,
        use_weekly_alpha_gate=use_weekly_alpha_gate,
        universe_mode=universe_mode,
    )
    gpu_sell_events, gpu_buy_events, gpu_daily_snapshots, gpu_position_snapshots, gpu_log_text = _run_gpu_and_collect_sell_events(
        config=config,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        params=params,
        candidate_source_mode=candidate_source_mode,
        use_weekly_alpha_gate=use_weekly_alpha_gate,
        parity_mode=parity_mode,
        universe_mode=universe_mode,
    )

    sell_pairs = _pair_events(cpu_sell_events, gpu_sell_events)
    buy_pairs = _pair_buy_events(cpu_buy_events, gpu_buy_events)
    daily_snapshot_pairs = _pair_daily_snapshots(cpu_daily_snapshots, gpu_daily_snapshots)
    position_snapshot_pairs = _pair_position_snapshots(cpu_position_snapshots, gpu_position_snapshots)
    matched_sell_pairs = sum(1 for row in sell_pairs if row["matched"])
    matched_buy_pairs = sum(1 for row in buy_pairs if row["matched"])
    matched_daily_snapshot_pairs = sum(1 for row in daily_snapshot_pairs if row["matched"])
    matched_position_snapshot_pairs = sum(1 for row in position_snapshot_pairs if row["matched"])
    sell_mismatched_pairs = len(sell_pairs) - matched_sell_pairs
    buy_mismatched_pairs = len(buy_pairs) - matched_buy_pairs
    daily_snapshot_mismatched_pairs = len(daily_snapshot_pairs) - matched_daily_snapshot_pairs
    position_snapshot_mismatched_pairs = len(position_snapshot_pairs) - matched_position_snapshot_pairs
    snapshot_level_zero_mismatch = (
        daily_snapshot_mismatched_pairs == 0 and position_snapshot_mismatched_pairs == 0
    )
    decision_level_zero_mismatch = (
        sell_mismatched_pairs == 0
        and buy_mismatched_pairs == 0
        and snapshot_level_zero_mismatch
    )
    snapshot_scope_collected = any(
        [
            cpu_daily_snapshots,
            gpu_daily_snapshots,
            cpu_position_snapshots,
            gpu_position_snapshots,
        ]
    )
    daily_snapshot_fields_complete = (
        len(cpu_daily_snapshots) > 0
        and len(cpu_daily_snapshots) == len(gpu_daily_snapshots)
        and _snapshots_have_required_fields(
            cpu_daily_snapshots,
            ("date", "total_value", "cash", "stock_count"),
        )
        and _snapshots_have_required_fields(
            gpu_daily_snapshots,
            ("date", "total_value", "cash", "stock_count"),
        )
    )
    position_snapshot_fields_complete = (
        _snapshots_have_required_fields(
            cpu_position_snapshots,
            ("date", "ticker", "holdings", "quantity", "avg_buy_price", "current_price", "total_value"),
        )
        and _snapshots_have_required_fields(
            gpu_position_snapshots,
            ("date", "ticker", "holdings", "quantity", "avg_buy_price", "current_price", "total_value"),
        )
    )
    comparison_scope = (
        "structured_trade_and_state_snapshots"
        if snapshot_scope_collected
        else "structured_trade_events"
    )

    return {
        "start_date": start_date,
        "end_date": end_date,
        "params": params,
        "candidate_source_mode": candidate_source_mode,
        "parity_mode": parity_mode,
        "universe_mode": universe_mode,
        "comparison_scope": comparison_scope,
        "release_decision_fields_complete": bool(
            daily_snapshot_fields_complete and position_snapshot_fields_complete
        ),
        "cpu_sell_events_count": len(cpu_sell_events),
        "gpu_sell_events_count": len(gpu_sell_events),
        "cpu_buy_events_count": len(cpu_buy_events),
        "gpu_buy_events_count": len(gpu_buy_events),
        "cpu_daily_snapshots_count": len(cpu_daily_snapshots),
        "gpu_daily_snapshots_count": len(gpu_daily_snapshots),
        "cpu_position_snapshots_count": len(cpu_position_snapshots),
        "gpu_position_snapshots_count": len(gpu_position_snapshots),
        "sell_paired_count": len(sell_pairs),
        "sell_matched_pairs": matched_sell_pairs,
        "sell_mismatched_pairs": sell_mismatched_pairs,
        "buy_paired_count": len(buy_pairs),
        "buy_matched_pairs": matched_buy_pairs,
        "buy_mismatched_pairs": buy_mismatched_pairs,
        "daily_snapshot_paired_count": len(daily_snapshot_pairs),
        "daily_snapshot_matched_pairs": matched_daily_snapshot_pairs,
        "daily_snapshot_mismatched_pairs": daily_snapshot_mismatched_pairs,
        "position_snapshot_paired_count": len(position_snapshot_pairs),
        "position_snapshot_matched_pairs": matched_position_snapshot_pairs,
        "position_snapshot_mismatched_pairs": position_snapshot_mismatched_pairs,
        "daily_snapshot_fields_complete": bool(daily_snapshot_fields_complete),
        "position_snapshot_fields_complete": bool(position_snapshot_fields_complete),
        "decision_level_zero_mismatch": decision_level_zero_mismatch,
        "cpu_sell_events": [event.__dict__ for event in cpu_sell_events],
        "gpu_sell_events": [event.__dict__ for event in gpu_sell_events],
        "cpu_buy_events": [event.__dict__ for event in cpu_buy_events],
        "gpu_buy_events": [event.__dict__ for event in gpu_buy_events],
        "cpu_daily_snapshots": [row.__dict__ for row in cpu_daily_snapshots],
        "gpu_daily_snapshots": [row.__dict__ for row in gpu_daily_snapshots],
        "cpu_position_snapshots": [row.__dict__ for row in cpu_position_snapshots],
        "gpu_position_snapshots": [row.__dict__ for row in gpu_position_snapshots],
        "sell_pairs": sell_pairs,
        "buy_pairs": buy_pairs,
        "daily_snapshot_pairs": daily_snapshot_pairs,
        "position_snapshot_pairs": position_snapshot_pairs,
        "gpu_log_text": gpu_log_text,
    }


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config()
    strategy_cfg = dict(config.get("strategy_params", {}))
    universe_mode = resolve_universe_mode(
        strategy_cfg,
        universe_mode=os.environ.get("MAGICSPLIT_UNIVERSE_MODE"),
    )
    initial_cash = float(config["backtest_settings"]["initial_cash"])
    if args.params_csv:
        params = _load_param_row_from_csv(args.params_csv, args.param_id)
    else:
        params = _load_param_row_from_config(config)
    payload = collect_trade_event_parity_report(
        config=config,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=initial_cash,
        params=params,
        candidate_source_mode=args.candidate_source_mode,
        use_weekly_alpha_gate=args.use_weekly_alpha_gate,
        parity_mode=args.parity_mode,
        universe_mode=universe_mode,
    )
    gpu_log_text = payload.pop("gpu_log_text")
    _save_json(args.out, payload)
    if args.gpu_log_out:
        Path(args.gpu_log_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.gpu_log_out).write_text(gpu_log_text, encoding="utf-8")

    print(f"[sell_event_dump] report saved: {args.out}")
    print(
        "[sell_event_dump] "
        f"cpu_sell_events={payload['cpu_sell_events_count']}, gpu_sell_events={payload['gpu_sell_events_count']}, "
        f"sell_mismatched_pairs={payload['sell_mismatched_pairs']}"
    )
    print(
        "[sell_event_dump] "
        f"cpu_buy_events={payload['cpu_buy_events_count']}, gpu_buy_events={payload['gpu_buy_events_count']}, "
        f"buy_mismatched_pairs={payload['buy_mismatched_pairs']}"
    )
    if args.gpu_log_out:
        print(f"[sell_event_dump] gpu_log saved: {args.gpu_log_out}")


if __name__ == "__main__":
    main()
