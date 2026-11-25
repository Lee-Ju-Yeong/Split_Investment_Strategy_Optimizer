"""판정 표(리뷰 5절)와 후보 표시(v1 6절), 시작일 민감도·여러 ETF 시험 표시(v2)."""
from __future__ import annotations

from decimal import Decimal

PASS, FAIL, HOLD = "pass", "fail", "hold"
CANDIDATE, SENSITIVE, INCOMPLETE = "candidate", "start_sensitive_unresolved", "incomplete"


def checks(base: dict, stress: dict, thresholds: dict) -> dict:
    """앞의 세 줄은 문턱 포함, 스트레스는 0 초과만 통과(정확히 0은 미달)."""
    return {
        "exposure": base["exposure_mean"] >= thresholds["exposure_min"],
        "cagr_edge": base["delta_cagr"] >= thresholds["cagr_edge_min"],
        "mdd_excess": base["delta_abs_mdd"] <= thresholds["mdd_excess_max"],
        "stress_cagr_edge": stress["delta_cagr"] > thresholds["stress_cagr_edge_gt"],
    }


def margins(base: dict, stress: dict, thresholds: dict) -> dict:
    """문턱을 얼마만큼 넘었는지(양수 = 충족 쪽). 오차 띠로 통과·미달을 바꾸지 않는다."""
    return {"exposure": base["exposure_mean"] - thresholds["exposure_min"],
            "cagr_edge": base["delta_cagr"] - thresholds["cagr_edge_min"],
            "mdd_excess": thresholds["mdd_excess_max"] - base["delta_abs_mdd"],
            "stress_cagr_edge": stress["delta_cagr"] - thresholds["stress_cagr_edge_gt"]}


def path_verdict(base, stress, thresholds, *, failures=(), holds=(), later_holds=()) -> dict:
    """failures: 시작 시 확정된 개설 미달(비용 조건·최초 분할 매수 불가). holds: 시작 시 비교 불성립·자료 부족.
    later_holds: 개설 뒤 생긴 자료 결측(연초 변동성 등). 시작 시 확정된 미달은 이후 결측으로 보류가 되지 않는다(D4)."""
    if holds:
        return {"status": HOLD, "reasons": list(holds), "checks": None, "later_holds": list(later_holds)}
    if failures:
        return {"status": FAIL, "reasons": list(failures), "checks": None, "later_holds": list(later_holds)}
    if later_holds:
        return {"status": HOLD, "reasons": list(later_holds), "checks": None, "later_holds": list(later_holds)}
    result = checks(base, stress, thresholds)
    return {"status": PASS if all(result.values()) else FAIL,
            "reasons": [name for name, ok in result.items() if not ok], "checks": result,
            "margins": margins(base, stress, thresholds)}


def item_status(path_statuses) -> str:
    """두 장중 경로 모두 통과해야 통과. 하나라도 미달이면 미달, 그 외 보류가 있으면 보류."""
    statuses = list(path_statuses)
    if FAIL in statuses:
        return FAIL
    if HOLD in statuses or not statuses:
        return HOLD
    return PASS


def candidates(items: dict, passing=PASS) -> dict:
    """둘 다 통과하면 공동 후보(M, D 순서). 대표 하나를 고르지 않는다. 다른 항목도 함께 공개한다.
    통과가 없을 때 시작일 민감이 남으면 'unresolved', 보류·불완전이 남으면 'held'. 모두 미달일 때만 'closed'."""
    order = [mode for mode in ("M", "D") if mode in items] + [m for m in items if m not in ("M", "D")]
    passed = [mode for mode in order if items[mode] == passing]
    unresolved = [mode for mode in order if items[mode] == SENSITIVE]
    held = [mode for mode in order if items[mode] in (HOLD, INCOMPLETE)]
    label = "joint" if len(passed) > 1 else "single" if passed else "closed"
    if not passed and unresolved:
        label = "unresolved"
    elif not passed and held:
        label = "held"
    return {"label": label, "passed": passed, "unresolved": unresolved, "held": held,
            "all": {mode: items[mode] for mode in order}}


def across_starts(labels_by_start: dict) -> dict:
    """한 방식(M 또는 D)의 시작일별 라벨 종합. 보류·불성립이 하나라도 있으면 판정 불완전.
    M과 D가 서로 다른 시작일에서 통과한 것을 합치지 않는다(호출자가 방식별로 부른다)."""
    values = list(labels_by_start.values())
    if not values or HOLD in values:
        label = INCOMPLETE
    elif all(value == PASS for value in values):
        label = CANDIDATE
    elif all(value == FAIL for value in values):
        label = FAIL
    else:
        label = SENSITIVE
    return {"label": label, "by_start": dict(labels_by_start)}


def extension_counts(labels: dict, denominator: int) -> dict:
    """labels[symbol][mode] = across_starts 라벨. 분모는 사전 선정 수, 보류·불성립도 분모에 남긴다."""
    if len(labels) != denominator:
        raise ValueError("extension results must cover the frozen list")
    modes = sorted({mode for row in labels.values() for mode in row}, key=lambda m: ("M", "D").index(m)
                   if m in ("M", "D") else 9)
    per_mode = {}
    for mode in modes:
        counts = {name: sum(row[mode] == name for row in labels.values())
                  for name in (CANDIDATE, FAIL, SENSITIVE, INCOMPLETE)}
        # 판정이 확정된 수 = 후보 + 미달. 시작일 민감·불완전은 미확정으로 따로 센다.
        per_mode[mode] = {**counts, "denominator": denominator,
                          "decided": counts[CANDIDATE] + counts[FAIL], "decided_passed": counts[CANDIDATE],
                          "undecided": counts[SENSITIVE] + counts[INCOMPLETE]}
    passed = sorted(s for s, row in labels.items() if CANDIDATE in row.values())
    note = "passed_among_multiple_tests" if 0 < len(passed) < denominator else None
    return {"by_mode": per_mode, "etf_candidates": passed, "etf_candidate_count": len(passed), "note": note,
            "independent_samples": False}


def delta(strategy: dict, hold: dict) -> dict:
    """차이 = 전체 전략 계좌 - 같은 조건 단독 보유 계좌. 낙폭은 절댓값 차(양수면 더 깊음)."""
    return {"delta_cagr": strategy["cagr"] - hold["cagr"],
            "delta_abs_mdd": abs(strategy["mdd"]) - abs(hold["mdd"]),
            "exposure_mean": strategy["exposure_mean"]}


def matched_label(strategy: dict, matched: dict, band: Decimal) -> bool:
    """평균 투자 비중 차 ≤ band 이고 전략 연복리가 높을 때만 '비슷한 비중에서도 앞섬'(주 판정 아님)."""
    return abs(strategy["exposure_mean"] - matched["exposure_mean"]) <= band and strategy["cagr"] > matched["cagr"]
