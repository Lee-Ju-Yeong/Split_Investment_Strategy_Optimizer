"""설정 결과(runner.run_point를 직렬화한 꼴)에서 판정 입력과 '함께 표기' 항목을 읽는다. 판정은 judge가 한다."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext

from .judge import FLAG_KEYS, cost_valid, reasons_for
from .runner import COSTS


EXACT_DIGITS = 200  # 80자리 지표끼리의 뺄셈이 반올림되지 않는 자릿수
BASE = "base"


@dataclass(frozen=True)
class Frame:
    """한 행이 가져야 할 틀: 등록한 경로, 그 시작일부터의 구간 이름, 종전 한도(표기용)."""

    paths: tuple
    labels: tuple
    limits: dict


def dec(value):
    return value if value is None or isinstance(value, Decimal) else Decimal(value)


def exact_sub(left, right) -> Decimal:
    with localcontext(prec=EXACT_DIGITS):
        return dec(left) - dec(right)


def segment_deltas(side, labels) -> dict:
    """구간별 '전략 연복리 − 단독 보유 연복리'. 두 계좌 모두 등록한 구간의 이름·순서·개수와 같아야 한다."""
    for name in ("strategy", "hold"):
        if [row["segment"] for row in side[name]["segments"]] != list(labels):
            raise ValueError(f"{name} segments differ from the registered {list(labels)}")
    pairs = zip(side["strategy"]["segments"], side["hold"]["segments"])
    return {s["segment"]: exact_sub(s["cagr"], h["cagr"]) for s, h in pairs}


def _path_rows(result) -> dict:
    return result["run_start"]["items"][result["mode"]]["paths"]


def _path_input(row, extra, labels) -> dict:
    flags, base, stress = extra["flags"], row["base"], row["stress"]
    return {"start_holds": list(flags["holds"]), "start_failures": list(flags["failures"]),
            "later_holds": list(flags["later"]),
            "base": None if base is None else {"delta_cagr": dec(base["delta_cagr"]),
                                               "segments": segment_deltas(base, labels)},
            "stress": None if stress is None else {"delta_cagr": dec(stress["delta_cagr"])}}


def point_input(result, frame: Frame) -> dict:
    """judge.setting_verdict에 넘길 KODEX 4절 판정 입력."""
    if result.get("held_reason"):
        return {"held": result["held_reason"], "paths": None}
    rows, extras = _path_rows(result), result["exploration"]
    return {"held": None, "paths": {path: _path_input(rows[path], extras[path], frame.labels)
                                    for path in frame.paths}}


def _extension_path(row, extra) -> dict:
    """기본 비용의 보류·개설 거부·결측 사유(공통 사유 포함)와 기본 비용 전체 연복리 차만 읽는다."""
    flags, base = extra["flags"], row[BASE]
    return {"start_holds": reasons_for(flags["holds"], BASE), "start_failures": reasons_for(flags["failures"], BASE),
            "later_holds": reasons_for(flags["later"], BASE),
            "base_delta": None if base is None else dec(base["delta_cagr"])}


def extension_input(result, frame: Frame) -> dict:
    """judge.extension_point에 넘길 확장 전용 판정 입력: 명목 한도, 기본 비용 사유, 두 기본 경로의 전체 차."""
    if result.get("held_reason"):
        return {"held": result["held_reason"], "capacity_ok": None, "paths": None}
    rows, extras = _path_rows(result), result["exploration"]
    return {"held": None, "capacity_ok": result["run_start"]["capacity"]["ok"] is True,
            "paths": {path: _extension_path(rows[path], extras[path]) for path in frame.paths}}


def _valid_base_delta(point, path):
    """확장 판정과 같은 유효성: 명목 한도를 채웠고 그 경로에 기본 비용 사유도 공통 사유도 없을 때만 값을 싣는다."""
    row = (point["paths"] or {}).get(path)
    if row is None or point["capacity_ok"] is not True or any(row[key] for key in FLAG_KEYS):
        return None
    return row["base_delta"]


def extension_display(result, frame: Frame) -> dict:
    """확장 행에 함께 싣는 값: 명목 한도와 두 기본 경로의 유효한 전체 차뿐이다."""
    point = extension_input(result, frame)
    return {"held_reason": point["held"], "capacity": (result.get("run_start") or {}).get("capacity"),
            "base_deltas": {path: _valid_base_delta(point, path) for path in frame.paths}}


def _side_view(side, labels):
    if side is None:
        return None
    return {"total_delta": dec(side["delta_cagr"]), "segment_deltas": segment_deltas(side, labels),
            "mdd_deepening": dec(side["delta_abs_mdd"]), "exposure_mean": dec(side["exposure_mean"]),
            "strategy_cagr": dec(side["strategy"]["cagr"]), "hold_cagr": dec(side["hold"]["cagr"]),
            "strategy_mdd": dec(side["strategy"]["mdd"]), "hold_mdd": dec(side["hold"]["mdd"]),
            "ending_equity_krw": dec(side["strategy"]["ending_equity_krw"]),
            "evaluation_days": side["strategy"]["evaluation_days"]}


def worst_segments(deltas) -> list:
    """기본 비용 차가 가장 작은 구간. 동률은 모두 싣는다."""
    lowest = min(deltas.values())
    return [label for label, value in deltas.items() if value == lowest]


def legacy_limits(view, limits) -> dict:
    """종전 한도(낙폭 심화 ≤ 5%p, 평균 비중 ≥ 80%)의 충족 여부. 표기만 하고 새 판정에는 쓰지 않는다."""
    return {"mdd_excess_ok": view["mdd_deepening"] <= dec(limits["mdd_excess_max"]),
            "exposure_ok": view["exposure_mean"] >= dec(limits["exposure_min"])}


def _matched_view(row):
    matched = row.get("matched_baseline")
    if matched is None:
        return None
    metrics = matched["metrics"]
    return {"fraction": dec(matched["fraction"]), "cagr": dec(metrics["cagr"]), "mdd": dec(metrics["mdd"]),
            "exposure_mean": dec(metrics["exposure_mean"]),
            "strategy_minus_matched": exact_sub(row["base"]["strategy"]["cagr"], metrics["cagr"]),
            "similar_exposure_and_ahead": matched["similar_exposure_and_ahead"], "judgment": False}


def _share_view(share):
    """분할 몫 누적수익과 같은 원금 단독 보유 누적수익, 그 차(둘 다 만기까지 성립한 경우만)."""
    split, hold = share["split"]["cumulative_return"], share["hold_same_capital"]["cumulative_return"]
    delta = None if split is None or hold is None else exact_sub(split, hold)
    return {"capital_krw": dec(share["split"]["capital_krw"]), "split": dec(split), "hold_same_capital": dec(hold),
            "delta": delta, "split_status": share["split"]["status"],
            "hold_status": share["hold_same_capital"]["status"]}


def _gated(values, valid) -> dict:
    """비용별 값 가운데 유효한 비용의 것만 남기고, 보류·거부·결측 비용은 빈칸으로 둔다."""
    return {cost: (values or {}).get(cost) if valid[cost] else None for cost in COSTS}


def _blank_reasons(entry, valid) -> dict:
    """빈칸으로 둔 비용의 사유(그 비용의 사유와 공통 사유). 유효한 비용은 None."""
    every = [reason for key in FLAG_KEYS for reason in entry[key]]
    return {cost: None if valid[cost] else reasons_for(every, cost) for cost in COSTS}


def _path_display(row, extra, frame: Frame) -> dict:
    """한 경로의 '함께 표기' 항목. 비용별 수치와 건수는 판정과 같은 유효성 검사(judge.cost_valid)를 거친 것만 싣는다.
    시작 주문은 보류·거부 사유의 근거가 되는 개설 사실이라 그대로 싣는다."""
    entry = _path_input(row, extra, frame.labels)
    valid = {cost: cost_valid(entry, cost) for cost in COSTS}
    base = _side_view(row["base"], frame.labels) if valid["base"] else None
    stress = _side_view(row["stress"], frame.labels) if valid["stress"] else None
    shares = {cost: _share_view(share) for cost, share in (extra["split_share"] or {}).items()}
    return {"base": base, "stress": stress, "flags": extra["flags"], "blank": _blank_reasons(entry, valid),
            "total_delta_positive": None if base is None else base["total_delta"] > 0,
            "worst_segments": None if base is None else worst_segments(base["segment_deltas"]),
            "legacy_limits": None if base is None else legacy_limits(base, frame.limits),
            "split_trades": _gated(extra["split_ledger"], valid), "reentries": _gated(extra["reentry_counts"], valid),
            "buys_closed_years": _gated(row.get("buys_closed_years"), valid), "start_orders": row["start_orders"],
            "split_share": _gated(shares, valid), "matched": _matched_view(row) if valid["base"] else None}


def _held_path(reason) -> dict:
    """입력 보류 행의 빈 경로: 등록한 칸을 빈값과 사유로 채운다."""
    empty = dict.fromkeys(("total_delta_positive", "worst_segments", "legacy_limits", "start_orders", "matched"))
    blank = dict.fromkeys(COSTS)
    return {"base": None, "stress": None, "flags": {"failures": [], "holds": [reason], "later": []},
            "blank": {cost: [reason] for cost in COSTS}, **empty, "split_trades": dict(blank),
            "reentries": dict(blank), "buys_closed_years": dict(blank), "split_share": dict(blank)}


def point_display(result, frame: Frame) -> dict:
    """판정이 아닌 '함께 표기' 항목. 보류 행도 등록한 경로의 빈 칸과 사유를 갖는다."""
    reason = result.get("held_reason")
    if reason:
        return {"held_reason": reason, "capacity": None, "paths": {path: _held_path(reason) for path in frame.paths}}
    rows, extras = _path_rows(result), result["exploration"]
    return {"held_reason": None, "capacity": result["run_start"]["capacity"],
            "paths": {path: _path_display(rows[path], extras[path], frame) for path in frame.paths}}

