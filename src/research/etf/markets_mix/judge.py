"""판정(설계 v1.1 6절). 순수 함수이며 장부·자료를 읽지 않는다. 비교는 반올림 전 Decimal로 한다.

차 = 분할 계좌의 전체 기간 연복리 − 같은 상품(같은 배분) 보유의 연복리. 한 행은 필수 경로(단독 2, 혼합 4)의 차를 가진다.
세 결과 문장은 묶음별 주 시작일·기본 비용의 계획한 행 전부(설계 v2: 216행)에만 쓴다. 병기 항목과 28번 민감도는 판정을 바꾸지 않는다.
"""
from __future__ import annotations


INFERIOR, COUNTER, TIE, PATH_DEPENDENT, BLANK = "inferior", "counterexample", "tie", "path_dependent", "blank"
STATUSES = (INFERIOR, COUNTER, TIE, PATH_DEPENDENT, BLANK)
PLUS, MINUS, ZERO_SIGN = "+", "-", "0"
DOWN, FLAT, UP = "down", "flat", "up"
ALL_INFERIOR, HAS_COUNTER, PARTIAL = "all_inferior", "counterexample", "partial"
MET, NOT_MET, OPEN = "met", "not_met", "undetermined"
DEPENDENT, SAME = "dependent", "same"
YES, NO = "yes", "no"  # 민감도 요약: 부호 차이 확인 / 전부 비교해 같음 (나머지는 OPEN)


def sign_of(delta):
    """차의 부호. 값이 없으면 None(0으로 읽지 않는다)."""
    if delta is None:
        return None
    return PLUS if delta > 0 else MINUS if delta < 0 else ZERO_SIGN


def row_status(signs: dict, required) -> str:
    """필수 경로가 하나라도 빠지면 빈칸. 모두 있으면 전부 음수는 열위, 전부 양수는 반례, 양수와 음수가 함께 있으면
    경로 의존, 나머지는 동률."""
    if set(signs) != set(required) or not required:
        raise ValueError("a row needs exactly its required paths")
    found = set(signs.values())
    if None in found:
        return BLANK
    if found in ({MINUS}, {PLUS}):
        return INFERIOR if found == {MINUS} else COUNTER
    return PATH_DEPENDENT if {PLUS, MINUS} <= found else TIE


def row_verdict(deltas: dict, required) -> dict:
    """deltas: {경로: 차 또는 None}. 상태와 경로별 부호를 함께 남긴다."""
    signs = {path: sign_of(deltas.get(path)) for path in deltas}
    return {"status": row_status(signs, required), "signs": signs}


def beat(versus_reference):
    """같은 시작일 KODEX 200 1억 원 보유보다 더 벌었는지(연복리 차 > 0). 견줄 값이 없으면 None."""
    return None if versus_reference is None else versus_reference > 0


def regime(hold_cagr, band):
    """자기 보유 연복리 구분: −band 미만 하락, ±band 이내 횡보, +band 초과 상승."""
    if hold_cagr is None:
        return None
    return DOWN if hold_cagr < -band else UP if hold_cagr > band else FLAT


def _details(counters, sentences) -> str:
    """반례 문장의 [기간·대상]·[설정 수]: counters = ((시작일, 대상, 설정 수), ...)."""
    return ", ".join(sentences["counter_item"].format(start=start, subject=subject, count=count)
                     for start, subject, count in counters)


def conclusion(statuses: dict, expected, sentences, *, counters=()) -> dict:
    """statuses: {행 열쇠: 상태}. 계획한 행이 모두 한 번씩 있어야 문장을 고른다(누락·여분은 오류).
    counters: 반례 행의 (시작일, 대상, 설정 수). 반례 수의 합이 반례 행 수와 같아야 한다."""
    expected = list(expected)
    if set(statuses) != set(expected) or len(set(expected)) != len(expected) or not expected:
        raise ValueError("the conclusion needs every planned primary row exactly once")
    counts = {name: sum(status == name for status in statuses.values()) for name in STATUSES}
    if sum(count for _, _, count in counters) != counts[COUNTER]:
        raise ValueError("counterexample counts must add up to the counterexample rows")
    kind = HAS_COUNTER if counts[COUNTER] else ALL_INFERIOR if counts[INFERIOR] == len(expected) else PARTIAL
    sentence = sentences[kind].format(rows=len(expected), details=_details(counters, sentences), **counts)
    return {"kind": kind, "sentence": sentence, "rows": len(expected), "counts": counts,
            "all_claimed": kind == ALL_INFERIOR, "scope": sentences["scope"]}


def point_meets(deltas: dict, threshold, *, strict: bool):
    """한 점이 모든 경로에서 문턱을 넘는지. 값이 빠진 경로가 있으면 넘지 못한 경로가 없을 때 None(미확정)."""
    present = [value for value in deltas.values() if value is not None]
    failed = any(not (value > threshold if strict else value >= threshold) for value in present)
    if failed:
        return False
    return True if len(present) == len(deltas) and present else None


def region_status(points) -> str:
    """8점 전부 충족 → 충족, 한 점이라도 확정 미충족 → 미충족, 그 밖에 미확정 점이 있으면 미확정."""
    points = list(points)
    if False in points:
        return NOT_MET
    return MET if points and all(point is True for point in points) else OPEN


def regions_table(regions, by_number: dict) -> list:
    """regions: 영역 선언(rank·core·kg·kp·numbers). by_number: {번호: True/False/None}. 영역의 점이 없으면 오류다."""
    table = []
    for region in regions:
        points = {number: by_number[number] for number in region["numbers"]}
        table.append({**region, "status": region_status(points.values()),
                      "met": sum(point is True for point in points.values()), "points": points})
    return table


def dependence(signs: dict) -> dict:
    """같은 경로에서 한 축(시작일 또는 비용)을 따라 본 부호. 다른 부호가 있으면 의존, 값이 빠졌는데 남은 부호가 같으면
    미확정, 모두 있고 같으면 같음."""
    present = {key: value for key, value in signs.items() if value is not None}
    missing = [key for key, value in signs.items() if value is None]
    if len(set(present.values())) > 1:
        mark = DEPENDENT
    else:
        mark = OPEN if missing else SAME
    return {"mark": mark, "signs": dict(signs), "missing": missing}


def sensitivity(table: dict, paths) -> dict:
    """table: {(시작일, 비용): {경로: 부호 또는 None}}. 같은 경로·비용에서 시작일에 따라, 같은 경로·시작일에서 비용에
    따라 부호가 다른지 표시한다."""
    starts, costs = sorted({key[0] for key in table}), sorted({key[1] for key in table})
    by_start = {f"{path}@{cost}": dependence({start: table[(start, cost)][path] for start in starts})
                for path in paths for cost in costs}
    by_cost = {f"{path}@{start}": dependence({cost: table[(start, cost)][path] for cost in costs})
               for path in paths for start in starts}
    return {"start": by_start, "cost": by_cost, "start_dependent": overall(by_start.values()),
            "cost_dependent": overall(by_cost.values())}


def overall(rows) -> str:
    """경로별 표시의 요약: 하나라도 의존이면 예, 전부 비교해 같으면 아니오, 나머지는 미확정."""
    marks = {row["mark"] for row in rows}
    if DEPENDENT in marks:
        return YES
    return NO if marks == {SAME} else OPEN


def review_threshold(base: dict, high: dict, thresholds) -> str:
    """같은 시작일의 모든 경로에서 기본 비용 차가 문턱 이상이고 높은 비용 차가 0보다 클 때만 충족한다."""
    low = point_meets(base, thresholds["review_base_edge_min"], strict=False)
    up = point_meets(high, thresholds["review_high_cost_gt"], strict=True)
    if low is False or up is False:
        return NOT_MET
    return MET if low is True and up is True else OPEN
