"""단위 실행기(탐색 전용): 한 상품·원금·시작일·설정·비용의 두 경로 장부를 kodex_grid의 한 설정 실행 부품으로 연다.

kodex_grid.runner는 두 비용을 함께 돌리지만, 이 조사는 비용 하나가 한 단위다(36개 설정은 기본 비용만, 28번은 높은 비용을
처음부터 따로). 장부를 여는 것(runner._Job)과 지표(study._side)는 승인 코드 그대로이고, 새로 더한 것은 비용 하나의 사유
정리(cost_flags)와 혼합 합산용 일별 원화 계열뿐이다. 비교할 수 없는 경로는 수치를 비우고 사유를 남긴다.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import localcontext

from ..kodex_core import study
from ..kodex_core.config import StudyConfig
from ..kodex_core.inputs import EtfInputs
from ..kodex_core.metrics import combined_rows
from ..kodex_grid.runner import Bench, _from, _head, _Job, _reentry_counts, _trade_counts


GAP_REASON = "input_data_gap_after_start"
MISSING = "missing_ledger"
REASON_KEYS = ("failures", "holds", "later")
SERIES_KEYS = (("equity", "equity_krw"), ("cash", "cash_krw"), ("receivables", "receivables_krw"))


@dataclass
class UnitRun:
    """연 장부: 머리 항목과 경로별 장부 셋(코어·단독 보유·분할). 명목 한도 미달이면 장부를 열지 않는다(groups=None)."""

    job: _Job
    head: dict
    cost: str
    groups: dict | None = None


def open_unit(inputs: EtfInputs, config: StudyConfig, start, *, cost, nominal_floor=None, bench=None) -> UnitRun:
    """study.run_start와 같은 순서(변동성 재검사 → 명목 한도 → 장부)로 한 비용의 두 경로 장부를 연다."""
    job = _Job(_from(inputs, start), config, bench or Bench())
    head = {**_head(job, start, nominal_floor), "opened_at": job.data.bars[0].open_at}
    if not head["capacity"]["ok"]:
        return UnitRun(job, head, cost)
    with localcontext(prec=80):
        return UnitRun(job, head, cost, {path: job.ledgers(cost, path) for path in config.paths})


def cost_flags(job: _Job, group, cost) -> dict:
    """study._flags를 비용 하나로 줄인 것: 시작 시 확정 미달(failures), 시작 비교 보류(holds), 개설 뒤 결측(later)."""
    data, split = job.data, group[job.mode]
    later = [GAP_REASON] if data.data_end is not None else []
    holds = [f"{cost}_{name}_{group[name].status}" for name in ("core", "hold") if group[name].status != "ok"]
    failures = []
    if split.status == "volatility_unavailable":
        holds.append(f"{cost}_volatility_unavailable")
    elif split.status == "observation_ended":
        later.append(f"{cost}_annual_volatility_unavailable")
    elif split.status == "cost_guard_start":
        failures.append(f"{cost}_start_rejected_by_cost_guard")
    elif split.status == "start_failed":
        failures.append(f"{cost}_split_start_failed")
    elif group["core"].status == "ok" and not study._start_liquidity(data, job.config, group["core"], split):
        holds.append(f"{cost}_combined_start_liquidity")
    return {"failures": failures, "holds": holds, "later": later}


def _block(rows, symbol) -> dict:
    """한 계좌의 일별 원화 계열: 순자산·현금·확정 미수금·ETF 평가액."""
    block = {name: [row[key] for row in rows] for name, key in SERIES_KEYS}
    return {**block, "holdings": [row["holdings_krw"][symbol] for row in rows]}


def _series(job: _Job, group, comparable):
    """혼합 합산용 계열. 보유 계좌는 그 장부가 만기까지 유효하면 남기고, 전략 계좌는 비교할 수 있을 때만 남긴다."""
    hold = group["hold"]
    if job.data.data_end is not None or not study._valid(hold):
        return None
    symbol, rows = job.data.instrument.symbol, hold.result.valuations
    strategy = combined_rows(group["core"].result, group[job.mode].result) if comparable else None
    return {"at": [row["at"] for row in rows], "hold": _block(rows, symbol),
            "strategy": None if strategy is None else _block(strategy, symbol)}


def _values(side) -> dict:
    if side is None:
        return {"strategy": None, "hold": None, "delta": None}
    return {"strategy": side["strategy"], "hold": side["hold"],
            "delta": {key: side[key] for key in ("delta_cagr", "delta_abs_mdd", "exposure_mean")}}


def _path(run: UnitRun, path) -> dict:
    """한 경로: 사유가 하나도 없고 세 장부가 만기까지 성립할 때만 비교할 수 있다(그때만 수치를 싣는다)."""
    job, group, mode = run.job, run.groups[path], run.job.mode
    reasons = cost_flags(job, group, run.cost)
    side = study._side(job.data, job.config, group, mode)
    if side is None and not any(reasons.values()):
        reasons["failures"].append(MISSING)
    comparable = side is not None and not any(reasons.values())
    closed = [row["date"] for row in group[mode].detail.get("refreshes", []) if row.get("buys_closed")]
    return {"comparable": comparable, "reasons": reasons, **_values(side if comparable else None),
            "hold_only": study._hold_only(job.data, job.config, group),
            "start_orders": study._first_orders(job.data, group["core"], group[mode], group["hold"]),
            "split_trades": _trade_counts(group[mode]), "reentry_counts": _reentry_counts(group[mode]),
            "buys_closed_years": closed, "series": _series(job, group, comparable)}


def _unopened_path() -> dict:
    """명목 한도 미달: 승인 코드와 같이 장부를 열지 않고 시작 시 비교 보류로 남긴다."""
    reasons = {"failures": [], "holds": [study.CAPACITY_HOLD], "later": []}
    return {"comparable": False, "reasons": reasons, **_values(None), "hold_only": None,
            "start_orders": study._not_submitted(study.CAPACITY_HOLD), "split_trades": None,
            "reentry_counts": None, "buys_closed_years": None, "series": None}


def _body(config: StudyConfig, cost) -> dict:
    return {"mode": config.modes[0], "cost": cost, "cost_rate": config.cost(cost),
            "applied_config_sha256": config.applied_sha256(), "capital": dict(config.capital)}


def unit_result(run: UnitRun) -> dict:
    """연 장부를 경로별로 정리한 단위 결과(직렬화 전)."""
    config = run.job.config
    with localcontext(prec=80):
        paths = {path: _unopened_path() if run.groups is None else _path(run, path) for path in config.paths}
    return {**_body(config, run.cost), "held_reason": None, "head": run.head, "paths": paths}


def run_unit(inputs: EtfInputs, config: StudyConfig, start, *, cost, nominal_floor=None, bench=None) -> dict:
    """한 단위 실행: 장부를 열고 경로별로 정리한다."""
    return unit_result(open_unit(inputs, config, start, cost=cost, nominal_floor=nominal_floor, bench=bench))


def held_unit(config: StudyConfig, cost, reason) -> dict:
    """예상한 자료 문제(DataHold)로 적재하지 못한 단위: 수치 없이 사유만 남긴다."""
    return {**_body(config, cost), "held_reason": reason, "head": None, "paths": None}
