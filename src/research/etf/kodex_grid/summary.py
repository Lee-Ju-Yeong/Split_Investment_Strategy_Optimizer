"""설명용 요약(설계 v2.1 4절, 판정 아님). 이 표로 영역이나 후보를 바꾸지 않는다."""
from __future__ import annotations

from decimal import localcontext

from .reading import EXACT_DIGITS


AXES = ("core", "kg", "kp", "levels")
BOTH, NOT_BOTH, UNKNOWN = "both_paths_positive", "not_both_positive", "undetermined"


def base_deltas(row) -> dict:
    """경로별 기본 비용 전체 연복리 차. 표기 단계에서 판정과 같은 비용별 유효성 검사를 거친 값만 남아 있고,
    보류·거부·결측으로 유효하지 않은 경로는 None이다."""
    return {path: None if view["base"] is None else view["base"]["total_delta"]
            for path, view in row["display"]["paths"].items()}


def _complete(row) -> bool:
    return all(value is not None for value in base_deltas(row).values())


def blank_reasons(row) -> list:
    """기본 비용 값이 빈칸인 경로의 사유(경로:사유)."""
    return [f"{path}:{reason}" for path, view in row["display"]["paths"].items() if view["base"] is None
            for reason in (view.get("blank") or {}).get("base") or []]


def ranking(rows) -> list:
    """두 경로의 기본 비용 차를 나란히 쓰고, 둘 중 작은 차가 큰 설정부터(동률은 번호순) 정렬한다.
    두 값이 갖춰지지 않은 설정은 끝에 사유와 함께 둔다."""
    ranked = [{"number": row["number"], "deltas": base_deltas(row), "min_delta": min(base_deltas(row).values())}
              for row in sorted(rows, key=lambda row: row["number"]) if _complete(row)]
    ranked.sort(key=lambda item: item["min_delta"], reverse=True)  # 안정 정렬: 원래 값 그대로 비교, 동률은 번호순 유지
    missing = [{"number": row["number"], "deltas": base_deltas(row), "min_delta": None,
                "reasons": blank_reasons(row)} for row in rows if not _complete(row)]
    return ranked + sorted(missing, key=lambda item: item["number"])


def positivity(row) -> str:
    """한 경로라도 유효한 값이 0 이하면 양수 아님, 두 값이 모두 양수면 양수, 그 밖에는 미확정."""
    values = list(base_deltas(row).values())
    if any(value is not None and value <= 0 for value in values):
        return NOT_BOTH
    return BOTH if all(value is not None for value in values) else UNKNOWN


def positive_counts(rows) -> dict:
    """두 경로 모두 전체 차 > 0인 설정 수, 양수가 아닌 설정 수, 양수 여부가 미확정인 설정 수."""
    states = [positivity(row) for row in rows]
    return {"total": len(rows), **{name: states.count(name) for name in (BOTH, NOT_BOTH, UNKNOWN)}}


def _mean(values):
    with localcontext(prec=EXACT_DIGITS):
        return sum(values) / len(values) if values else None


def axis_means(rows, paths) -> dict:
    """축 값별 평균: 경로별로 유효한 차만 산술평균하고 유효 수/예정 수를 붙인다."""
    output = {}
    for axis in AXES:
        values = sorted({row[axis] for row in rows}, key=lambda value: (len(str(value)), str(value)))
        output[axis] = {str(value): _axis_value([row for row in rows if row[axis] == value], paths)
                        for value in values}
    return output


def _axis_value(rows, paths) -> dict:
    cells = {}
    for path in paths:
        valid = [base_deltas(row).get(path) for row in rows if base_deltas(row).get(path) is not None]
        cells[path] = {"mean_delta": _mean(valid), "valid": len(valid), "planned": len(rows)}
    return cells


def sell_pairs(rows) -> list:
    """매도 체결 건수(기본 비용)와 기본 비용 차를 같은 경로끼리 짝짓는다. 건수는 정확히 쓰고 0건을 표시한다."""
    pairs = []
    for row in rows:
        for path, view in row["display"]["paths"].items():
            sells = ((view["split_trades"] or {}).get("base") or {}).get("sells")
            pairs.append({"number": row["number"], "path": path, "sells": sells, "zero_sells": sells == 0,
                          "delta": base_deltas(row).get(path)})
    return pairs


def describe(rows, paths) -> dict:
    return {"judgment": False, "ranking": ranking(rows), "positive_counts": positive_counts(rows),
            "axis_means": axis_means(rows, paths), "sell_pairs": sell_pairs(rows)}
