"""한 ETF·여러 시작일·여러 ETF 실행과 판정 조립. 실자료 성과 실행은 이 단계에서 하지 않는다."""
from __future__ import annotations

from datetime import datetime
from decimal import localcontext

from .calendar import EVALUATION_KST, KST
from .config import StudyConfig, frozen_identity, load_config
from .inputs import DataHold, EtfInputs, check_volatility, price_mode, starting_at, volatility_digest
from .ledgers import RunSetting, run_hold, run_matched, run_split
from .metrics import account_metrics, combined_rows, ledger_rows, segment_metrics
from .verdict import (CANDIDATE, HOLD, across_starts, candidates, delta, extension_counts, item_status,
                      matched_label, path_verdict)


BASE, STRESS = "base", "stress"
CAPACITY_HOLD = "start_capacity_below_floor"


class InputIdentityError(RuntimeError):
    """등록 종목·시작일·만기·가격 기준과 다른 입력. 자료 보류가 아니라 실행 오류로 멈춘다."""


def _ledgers(data: EtfInputs, config: StudyConfig, cost: str, path: str) -> dict:
    setting = RunSetting(config, path, config.cost(cost))
    runs = {"core": run_hold(data, setting, config.capital["core"]),
            "hold": run_hold(data, setting, config.capital["hold"])}
    for mode in config.modes:
        runs[mode] = run_split(data, RunSetting(config, path, config.cost(cost), mode))
    return runs


def _metrics(data, config, rows, initial):
    options = dict(initial=initial, started_at=data.bars[0].open_at,
                   exposure_min=config.thresholds["exposure_min"])
    return {**account_metrics(rows, [data.instrument], **options),
            "segments": segment_metrics(rows, [data.instrument], segments=config.segments, **options)}


def _buys_closed(runs, mode):
    return {cost: [row["date"] for row in runs[cost][mode].detail.get("refreshes", []) if row.get("buys_closed")]
            for cost in (BASE, STRESS)}


def _order_status(run, day) -> dict:
    """시작일 시가 주문 상태: 장부 미개설 / 주문 미제출 / 제출 후 거절 / 체결(B5)."""
    if run is None:
        return {"status": "no_ledger"}
    if run.result is None:
        return {"status": "not_opened", "reason": run.status}
    buys = [r for r in run.result.trades if r["session_date"] == day and r["side"] == "BUY"]
    if buys:
        return {"status": "filled", "notional": sum(r["price"] * r["quantity"] for r in buys),
                "quantity": sum(r["quantity"] for r in buys)}
    rejected = [r for r in run.result.rejections
                if r["side"] == "BUY" and r["at"].astimezone(KST).date() == day]
    if rejected:
        return {"status": "rejected", "notional": rejected[0]["price"] * rejected[0]["quantity"],
                "reason": rejected[0]["reason"]}
    events = [e for e in run.result.cash_events if e["kind"] in ("first_entry_unavailable", "baseline_unavailable")]
    return {"status": "not_submitted", "reason": events[0]["reason"] if events else "no_order"}


def _first_orders(data, core, split, hold) -> dict:
    """체결된 첫 주문 원금: 코어+분할 합계와 단독 보유. 각 장부의 주문 상태도 함께 남긴다."""
    day = data.bars[0].session_date
    orders = {"core": _order_status(core, day), "split": _order_status(split, day), "hold": _order_status(hold, day)}

    def filled(name):
        return orders[name]["notional"] if orders[name]["status"] == "filled" else None

    core_plus = None if filled("core") is None else filled("core") + (filled("split") or 0)
    return {"core_plus_split": core_plus, "hold": filled("hold"), "orders": orders}


def _not_submitted(reason) -> dict:
    status = {"status": "not_submitted", "reason": reason}
    return {"core_plus_split": None, "hold": None, "orders": {name: status for name in ("core", "split", "hold")}}


def _start_liquidity(data, config, core, split) -> bool:
    """시작일 같은 시각의 코어·분할 첫 매수 합계 ≤ 직전 20거래일 평균 거래대금 × 참여율."""
    first = data.bars[0]
    if first.adv20 is None:
        return False
    total = _first_orders(data, core, split, None)["core_plus_split"]
    return total is not None and total <= first.adv20 * config.participation_rate


def _flags(data, config, runs, mode) -> tuple[list, list, list, dict]:
    failures, holds, later, ends = [], [], [], {}
    if data.data_end is not None:  # E2: 개설 뒤 파일 일봉 결측. 시작 판정·첫 주문은 보존하고 만기 성과는 미산출.
        later.append("input_data_gap_after_start")
        ends["data"] = dict(data.data_end)
    for cost in (BASE, STRESS):
        group = runs[cost]
        for name in ("core", "hold"):
            if group[name].status != "ok":
                holds.append(f"{cost}_{name}_{group[name].status}")
        split = group[mode]
        if split.status == "volatility_unavailable":
            holds.append(f"{cost}_volatility_unavailable")
        elif split.status == "observation_ended":
            later.append(f"{cost}_annual_volatility_unavailable")
            ends[cost] = split.detail
        elif split.status == "cost_guard_start":
            failures.append(f"{cost}_start_rejected_by_cost_guard")
        elif split.status == "start_failed":
            failures.append(f"{cost}_split_start_failed")
        elif group["core"].status == "ok" and not _start_liquidity(data, config, group["core"], split):
            holds.append(f"{cost}_combined_start_liquidity")
    return failures, holds, later, ends


def _valid(run) -> bool:
    return run.status == "ok" and run.result is not None


def _side(data, config, group, mode):
    """전략 지표와 비교는 코어·분할·단독 보유 세 장부가 만기까지 모두 성립한 경우에만 만든다."""
    if data.data_end is not None or not all(_valid(group[name]) for name in ("core", "hold", mode)):
        return None
    rows = combined_rows(group["core"].result, group[mode].result)
    strategy = _metrics(data, config, rows, config.capital["core"] + config.capital["split"])
    hold = _metrics(data, config, group["hold"].result.valuations, config.capital["hold"])
    return {"strategy": strategy, "hold": hold, **delta(strategy, hold)}


def _hold_only(data, config, group):
    """유효한 단독 보유 장부는 전략 성립과 별개로 보조 비교의 기준으로 남긴다."""
    if data.data_end is not None or not _valid(group["hold"]):
        return None
    return _metrics(data, config, group["hold"].result.valuations, config.capital["hold"])


def _matched(data, config, path, base):
    setting = RunSetting(config, path, config.cost(BASE))
    fraction = base["strategy"]["exposure_mean"]
    result = run_matched(data, setting, config.capital["hold"], fraction)
    metrics = _metrics(data, config, result.valuations, config.capital["hold"])
    return {"fraction": fraction, "metrics": metrics, "judgment": False,
            "similar_exposure_and_ahead": matched_label(base["strategy"], metrics, config.matched_band)}


def _path_result(data, config, runs, mode, path, keep):
    failures, holds, later, ends = _flags(data, config, runs, mode)
    base, stress = _side(data, config, runs[BASE], mode), _side(data, config, runs[STRESS], mode)
    if (base is None or stress is None) and not failures and not holds and not later:
        failures = ["missing_ledger"]
    verdict = path_verdict(base, stress, config.thresholds, failures=failures, holds=holds, later_holds=later)
    output = {"verdict": verdict, "base": base, "stress": stress,
              "hold_metrics": {cost: _hold_only(data, config, runs[cost]) for cost in (BASE, STRESS)},
              "observation_end": ends or None,
              "start_orders": {cost: _first_orders(data, runs[cost]["core"], runs[cost][mode], runs[cost]["hold"])
                               for cost in (BASE, STRESS)},
              "reentries": {cost: runs[cost][mode].detail.get("reentries") for cost in (BASE, STRESS)},
              "buys_closed_years": _buys_closed(runs, mode)}
    if base is not None and not holds and not later:
        output["matched_baseline"] = _matched(data, config, path, base)
    if keep:
        output["ledgers"] = {cost: {name: ledger_rows(runs[cost][name].result, data.instrument.symbol)
                                    for name in ("core", "hold", mode) if runs[cost][name].result is not None}
                             for cost in (BASE, STRESS)}
    return output


def _capacity(data, config, nominal_floor) -> dict:
    """확장 행: 시작일 시가 전 직전 20일 평균 거래대금 × 참여율 ≥ 명목 1억 원(등호 허용)."""
    adv20 = data.bars[0].adv20
    with localcontext(prec=80):  # D7: 호출자 Decimal 정밀도와 무관
        limit = None if adv20 is None else adv20 * config.participation_rate
        ok = nominal_floor is None or (limit is not None and limit >= nominal_floor)
    return {"limit_krw": limit, "floor_krw": nominal_floor, "ok": ok}


def _held_items(config, reason) -> dict:
    verdict = {"status": HOLD, "reasons": [reason], "checks": None}
    orders = {cost: _not_submitted(reason) for cost in (BASE, STRESS)}
    return {mode: {"status": HOLD, "paths": {path: {"verdict": verdict, "base": None, "stress": None,
                                                    "start_orders": orders} for path in config.paths}}
            for mode in config.modes}


def run_start(inputs: EtfInputs, config: StudyConfig, start, *, nominal_floor=None, keep_ledgers=False) -> dict:
    data = starting_at(inputs, start)
    try:  # 사용 직전 재검사(B4): 기준 시각은 그해 첫 국내 거래일 시가, +5·+20 시작도 같다.
        check_volatility(data.calendar, data.volatility)
    except ValueError as error:
        raise InputIdentityError(f"{data.instrument.symbol} {start}: {error}") from error
    capacity = _capacity(data, config, nominal_floor)
    last = data.calendar.evaluation_times(data.start, data.end)[-1]
    output = {"symbol": data.instrument.symbol, "start": start, "end": data.end, "last_evaluation": last,
              "data_end": data.data_end,
              "basis": data.basis_label(config.labels), "capacity": capacity,
              "input_provenance": dict(data.provenance)}
    if not capacity["ok"]:
        items = _held_items(config, CAPACITY_HOLD)
        return {**output, "items": items, "candidates": candidates({m: HOLD for m in items}), "warnings": []}
    with localcontext(prec=80):
        runs = {cost: {path: _ledgers(data, config, cost, path) for path in config.paths} for cost in (BASE, STRESS)}
        items = {}
        for mode in config.modes:
            paths = {path: _path_result(data, config, {c: runs[c][path] for c in runs}, mode, path, keep_ledgers)
                     for path in config.paths}
            items[mode] = {"status": item_status(p["verdict"]["status"] for p in paths.values()), "paths": paths}
    return {**output, "items": items, "candidates": candidates({m: items[m]["status"] for m in items}),
            "warnings": sorted({w for c in runs.values() for p in c.values() for r in p.values()
                                if r.result is not None for w in r.result.warnings})}


def check_starts(calendar, starts, offsets) -> None:
    """실행 행의 시작일이 첫 시작일 + 사전 고정 오프셋(국내 거래일)과 같은지 확인한다."""
    expected = [calendar.offset(starts[0], n) for n in offsets]
    if list(starts) != expected:
        raise ValueError(f"row starts {list(starts)} differ from session offsets {expected}")


def verify_identity(inputs: EtfInputs, symbol, start, config: StudyConfig) -> None:
    """정식 진입점: 등록 종목·시작일·정확한 만기 일봉과 평가 시각·가격 기준."""
    problems = []
    if inputs.instrument.symbol != symbol or any(bar.symbol != symbol for bar in inputs.bars):
        problems.append(f"symbol {inputs.instrument.symbol} != registered {symbol}")
    if inputs.start != start:
        problems.append(f"first session {inputs.start} != registered start {start}")
    gap = inputs.data_end
    if gap is not None:  # E2: 개설 뒤 파일 일봉 결측으로 관측이 끊긴 입력. 요청 만기가 등록 만기여야 한다.
        if gap["requested_end"] != config.end:
            problems.append(f"requested end {gap['requested_end']} != registered end {config.end}")
    else:
        if inputs.end != config.end:
            problems.append(f"last bar {inputs.end} != registered end {config.end}")
        last = inputs.calendar.evaluation_times(inputs.start, inputs.end)[-1]
        if last != datetime.combine(config.end, EVALUATION_KST, KST):
            problems.append(f"last evaluation {last} is not {config.end} 18:00 KST")
    if inputs.instrument.price_basis != config.price_basis:
        problems.append(f"price basis {inputs.instrument.price_basis} differs from the registered {config.price_basis}")
    recorded = inputs.provenance.get("volatility_sha256")
    if recorded is not None and recorded != volatility_digest(inputs.volatility):
        problems.append("annual volatility differs from the values built by the pinned loader")
    if problems:
        raise InputIdentityError("; ".join(problems))


def _held_start(start, reason) -> dict:
    return {"start": start, "held_reason": reason, "items": None}


def run_starts(load_start, symbol, config: StudyConfig, starts, *, nominal_floor=None, verify=False,
               keep_ledgers=False) -> dict:
    """시작일마다 따로 적재·실행한다. 예상한 자료 문제(DataHold)만 그 시작일 보류로 남기고,
    그 밖의 오류는 전파한다(D6)."""
    runs, calendar = {}, None
    for start in starts:
        try:
            inputs = load_start(start)
        except DataHold as error:
            runs[start] = _held_start(start, str(error))
            continue
        if verify:
            verify_identity(inputs, symbol, start, config)
        calendar = calendar or inputs.calendar
        runs[start] = run_start(inputs, config, start, nominal_floor=nominal_floor, keep_ledgers=keep_ledgers)
    if calendar is not None:
        check_starts(calendar, starts, config.start_offsets)
    statuses = {mode: {s: (HOLD if r["items"] is None else r["items"][mode]["status"]) for s, r in runs.items()}
                for mode in config.modes}
    labels = {mode: across_starts(statuses[mode]) for mode in config.modes}
    return {"symbol": symbol, "starts": runs, "labels": labels,
            "held_starts": {s: r["held_reason"] for s, r in runs.items() if r["items"] is None},
            "candidates": candidates({mode: labels[mode]["label"] for mode in config.modes}, CANDIDATE)}


def run_symbol(inputs: EtfInputs, config: StudyConfig, starts, *, nominal_floor=None, keep_ledgers=False) -> dict:
    """이미 적재한 한 입력에서 여러 시작일을 실행(합성 시험용, 등록 대조 없음)."""
    return run_starts(lambda start: inputs, inputs.instrument.symbol, config, starts,
                      nominal_floor=nominal_floor, keep_ledgers=keep_ledgers)


def versus_primary_hold(result, comparison) -> dict:
    """확장 ETF 전략 계좌 대 같은 날짜·비용의 KODEX 200 단독 보유(보조 열, 판정 아님).
    확장 전략 계좌(세 장부)와 KODEX 200 단독 보유가 각각 성립한 경우에만 만든다."""
    rows = {}
    for start, value in result["starts"].items():
        reference = comparison["starts"].get(start)
        if value["items"] is None or reference is None or reference["items"] is None:
            continue
        for mode, item in value["items"].items():
            for path, row in item["paths"].items():
                other = reference["items"][mode]["paths"][path].get("hold_metrics", {}).get(BASE)
                if row["base"] is not None and other is not None:
                    rows[f"{start}/{mode}/{path}"] = {**delta(row["base"]["strategy"], other), "judgment": False}
    return rows


def _start_record(row, symbol, start, result, context) -> dict:
    """행별 식별 기록(B7·D8): 실행 식별값·등록 시작일·등록 만기·등록 가격 기준은 보류 행에도 남긴다."""
    record = {"row": row.id, "role": row.role, "symbol": symbol, "registered_start": start, **context}
    if result["items"] is None:
        return {**record, "status": "held", "reason": result["held_reason"]}
    first = result["items"][next(iter(result["items"]))]["paths"]
    path = first[next(iter(first))]
    opening = {cost: {name: order["status"] for name, order in orders["orders"].items()}
               for cost, orders in path["start_orders"].items()}
    splits = {mode: {cost: item["paths"][p]["start_orders"][cost]["orders"]["split"]["status"]
                     for p in item["paths"] for cost in (BASE, STRESS)} for mode, item in result["items"].items()}
    reasons = sorted({r for item in result["items"].values() for p in item["paths"].values()
                      for r in list(p["verdict"]["reasons"]) + list(p["verdict"].get("later_holds", []))})
    provenance = result["input_provenance"]
    return {**record, "status": "loaded", "actual_start": result["start"], "actual_end": result["end"],
            "last_evaluation": result["last_evaluation"], "price_basis": result["basis"]["price_basis"],
            "capacity": result["capacity"], "opening": opening, "split_opening": splits, "reasons": reasons,
            "data_end": result.get("data_end"),
            "input_manifest_sha256": provenance.get("input_manifest_sha256"),
            "extra_inputs": provenance.get("extra_inputs")}


def _context(config, registered=None) -> dict:
    """행 기록용 실행 식별값. 고정본 해시는 실행 전체(run_identity)와 같은 값을 쓴다(E4)."""
    registered = config.registered if registered is None else registered
    frozen = frozen_identity(config)
    return {"applied_config_sha256": frozen["applied_config_sha256"],
            "frozen_config_file_sha256": frozen["frozen_config_file_sha256"],
            "provided_config_file_sha256": frozen["provided_config_file_sha256"],
            "registered_input_manifest_sha256": registered.get("input_manifest_sha256"),
            "registered_end": config.end, "registered_price_basis": config.price_basis}


def run_row(row, load, config: StudyConfig, *, comparison=None, keep_ledgers=False, verify=True,
            context=None) -> dict:
    """load(symbol, row, start) -> EtfInputs. 시작일별 DataHold는 그 시작일의 보류로 기록한다."""
    context = _context(config) if context is None else context
    config = config.with_changes(segments=row.segments)
    floor = config.nominal_floor if row.role == "extension" else None
    results = {}
    for symbol in row.symbols:
        results[symbol] = run_starts(lambda start, s=symbol: load(s, row, start), symbol, config, row.starts,
                                     nominal_floor=floor, verify=verify, keep_ledgers=keep_ledgers)
        if comparison is not None:
            results[symbol]["versus_primary_hold"] = versus_primary_hold(results[symbol], comparison)
    records = [_start_record(row, symbol, start, result["starts"][start], context)
               for symbol, result in results.items() for start in row.starts]
    output = {"row": row.id, "role": row.role, "symbols": results, "records": records}
    if row.role == "extension":
        output["counts"] = extension_counts({s: {m: r["labels"][m]["label"] for m in config.modes}
                                             for s, r in results.items()}, row.denominator)
    return output


def _sources(output):
    return [start["input_provenance"] for row in output["rows"].values() for symbol in row["symbols"].values()
            for start in symbol["starts"].values() if start["items"] is not None]


def plan_identity(config) -> dict:
    """일반 실행 식별값. 외부 입력기를 받는 run_plan은 정식 실행 표시를 만들 수 없다(D2)."""
    return {**frozen_identity(config), "checks": {"via_formal_entrypoint": False}, "formal_run": False,
            "preflight": None}


def _limits(config, output) -> list:
    """결과 한계 문구도 실제 적재 입력의 가격 기준에 맞춘다(B6)."""
    modes = {start["basis"]["price_mode"] for row in output["rows"].values() for symbol in row["symbols"].values()
             for start in symbol["starts"].values() if start["items"] is not None}
    if not modes:  # D8: 적재된 시작일이 없으면 실행 설정의 가격 기준
        modes = {price_mode(config.price_basis)}
    notes = ["이미 본 과거 기간의 판정", "같은 지수·기간의 시작일·경로·M/D는 독립 표본이 아님"]
    if "provider_adjusted" in modes:
        notes.append(config.labels["total_return_note"])
    if "raw" in modes:
        notes.append(config.labels["basis_entitlement"])
    return notes


def execute_plan(load, config: StudyConfig, *, context=None) -> dict:
    """행 실행만 한다(식별값 없음). 결론 첫 행은 KODEX 200 주 행이며 확장·참고 행은 이를 뒤집지 않는다."""
    rows = {row.id: row for row in config.rows}
    output = {"rows": {}}
    for row_id in ("primary", "comparison", "reference_2004"):
        if row_id in rows:
            output["rows"][row_id] = run_row(rows[row_id], load, config, context=context)
    if "extension" in rows:
        comparison = output["rows"].get("comparison", {}).get("symbols", {}).get(config.primary_symbol)
        output["rows"]["extension"] = run_row(rows["extension"], load, config, comparison=comparison,
                                              context=context)
    output["conclusion_first_row"] = "primary"
    output["limits"] = _limits(config, output)
    return output


def run_plan(load, config: StudyConfig) -> dict:
    """외부 입력기를 받는 일반 실행(합성 시험·연구용). formal_run은 항상 False다.
    정식 실행은 formal.run_formal만 만들 수 있다."""
    output = execute_plan(load, config)
    output["run_identity"] = plan_identity(config)
    return output
