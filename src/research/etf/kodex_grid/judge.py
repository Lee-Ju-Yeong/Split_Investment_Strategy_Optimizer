"""새 판정(설계 v2.1 4~7절). 순수 함수이며 장부·자료를 읽지 않는다. 비교는 반올림 전 Decimal로 한다.

KODEX 판정 입력(한 점): {"held": 사유 또는 None, "paths": {경로: 경로 입력}}
  경로 입력: {"start_holds": [...], "start_failures": [...], "later_holds": [...],
             "base": {"delta_cagr": Decimal, "segments": {구간: Decimal}} 또는 None,
             "stress": {"delta_cagr": Decimal} 또는 None}
확장 판정 입력(한 점): {"held": 사유 또는 None, "capacity_ok": bool, "paths": {경로: 확장 경로 입력}}
  확장 경로 입력: 기본 비용·공통 사유만 담은 세 사유 목록과 "base_delta"(기본 비용 전체 연복리 차 또는 None)
"""
from __future__ import annotations


PASS, FAIL, OPEN = "pass", "fail", "undetermined"  # 통과 / 미달 / 미확정
SELECTED, NO_REGION = "selected", "none"
POSITIVE, OBSERVED = "positive", "observed"
FLAG_KEYS = ("start_holds", "start_failures", "later_holds")
COST_PREFIXES = ("base_", "stress_")
CAPACITY_REASON = "start_capacity_below_floor"


def _ordered_state(holds, failures, later) -> tuple:
    """승인 코드 path_verdict에서 순서만 계승한다: 시작 비교 보류 → 시작 시 확정 미달 → 개설 뒤 결측."""
    for status, reasons in ((OPEN, holds), (FAIL, failures), (OPEN, later)):
        if reasons:
            return status, list(reasons)
    return None, []


def reasons_for(reasons, cost) -> list:
    """그 비용의 사유와 공통 사유(비용 표시가 없는 사유)."""
    return [reason for reason in reasons if reason.startswith(f"{cost}_") or not reason.startswith(COST_PREFIXES)]


def cost_valid(row, cost) -> bool:
    """그 비용의 수치가 있고, 그 비용의 보류·거부·결측 사유도 공통 사유도 없다."""
    every = [reason for key in FLAG_KEYS for reason in row[key]]
    return row[cost] is not None and not reasons_for(every, cost)


def path_checks(row, thresholds) -> dict:
    """유효한 비용의 전체 기간 조건(유효하지 않은 비용은 None). 기본 차는 문턱과 같아도 통과, 스트레스 차는 0이면 미달."""
    base = row["base"]["delta_cagr"] >= thresholds["base_edge_min"] if cost_valid(row, "base") else None
    stress = row["stress"]["delta_cagr"] > thresholds["stress_edge_gt"] if cost_valid(row, "stress") else None
    return {"base_edge": base, "stress_edge": stress}


def path_verdict(row, thresholds) -> dict:
    """경로 상태는 사유의 우선순위로 먼저 정하고, 사유가 없을 때만 두 비용의 수치로 정한다."""
    checks = path_checks(row, thresholds)
    status, reasons = _ordered_state(*(row[key] for key in FLAG_KEYS))
    if status is not None:
        return {"status": status, "reasons": reasons, "checks": checks}
    if None in checks.values():
        raise ValueError("a path without values needs a hold or failure reason")
    return {"status": PASS if all(checks.values()) else FAIL,
            "reasons": [name for name, ok in checks.items() if not ok], "checks": checks}


def common_segments(paths, thresholds) -> list:
    """같은 구간에서 주어진 모든 경로의 기본 비용 차가 0보다 클 때만 그 구간을 센다(0은 세지 않는다)."""
    rows = [row["base"]["segments"] for row in paths.values()]
    labels = list(rows[0])
    if any(list(row) != labels for row in rows):
        raise ValueError("paths must report the same segments in the same order")
    return [label for label in labels if all(row[label] > thresholds["segment_edge_gt"] for row in rows)]


def _reasons(paths, status) -> list:
    return [f"{name}:{reason}" for name, row in paths.items() if row["status"] == status for reason in row["reasons"]]


def _known_failures(paths) -> list:
    """경로 상태가 미달이 아니어도 유효한 수치가 이미 조건을 채우지 못했으면 통과할 수 없다."""
    return [f"{name}:{check}" for name, row in paths.items() if row["status"] != FAIL
            for check, ok in row["checks"].items() if ok is False]


def _setting_status(paths, reachable, thresholds) -> tuple:
    """확정 미달이 하나라도 있으면 미달: 경로 미달, 유효한 수치의 조건 미달, 공통 승리 가능 구간 부족.
    결과를 바꿀 여지가 남을 때만 미확정으로 둔다."""
    short = reachable is not None and len(reachable) < thresholds["common_segments_min"]
    failed = _reasons(paths, FAIL) + _known_failures(paths) + (["common_segments"] if short else [])
    if failed:
        return FAIL, failed
    waiting = any(row["status"] == OPEN for row in paths.values())
    return (OPEN, _reasons(paths, OPEN)) if waiting else (PASS, [])


def setting_verdict(point, thresholds) -> dict:
    """한 설정·시작일의 4절 판정. 공통 승리 구간은 기본 비용이 유효한 경로끼리 센다: 모든 경로가 유효하면
    그것이 공통 승리 구간이고, 일부만 유효하면 그 이상은 나올 수 없는 상한(reachable_segments)이다."""
    if point.get("held"):
        return {"status": OPEN, "reasons": [point["held"]], "paths": None, "common_segments": None,
                "reachable_segments": None}
    paths = {name: path_verdict(row, thresholds) for name, row in point["paths"].items()}
    valid = {name: row for name, row in point["paths"].items() if cost_valid(row, "base")}
    reachable = common_segments(valid, thresholds) if valid else None
    status, reasons = _setting_status(paths, reachable, thresholds)
    return {"status": status, "reasons": reasons, "paths": paths, "reachable_segments": reachable,
            "common_segments": reachable if len(valid) == len(paths) else None}


def region_status(statuses) -> str:
    """8점 전부 통과 → 통과, 한 점이라도 확정 미달 → 미달, 그 밖에 미확정 점이 있으면 미확정."""
    statuses = list(statuses)
    if FAIL in statuses:
        return FAIL
    return PASS if statuses and all(status == PASS for status in statuses) else OPEN


def regions_table(regions, status_by_number) -> list:
    """regions: 영역 선언(rank·core·kg·kp·numbers). 영역의 점이 결과에 없으면 오류다."""
    table = []
    for region in regions:
        points = {number: status_by_number[number] for number in region["numbers"]}
        counts = {name: sum(status == name for status in points.values()) for name in (PASS, FAIL, OPEN)}
        table.append({**region, "status": region_status(points.values()), "counts": counts, "points": points})
    return table


def select_region(table) -> dict:
    """순위대로 처음 만나는 확정 통과 영역을 택한다. 그 앞에 미확정 영역이 있으면 선택 미확정,
    확정 통과 없이 미확정이 남아도 선택 미확정, 전부 확정 미달일 때만 영역 없음."""
    for row in sorted(table, key=lambda item: item["rank"]):
        if row["status"] == OPEN:
            return {"state": OPEN, "region": None, "blocked_by_rank": row["rank"]}
        if row["status"] == PASS:
            region = {key: row[key] for key in ("rank", "core", "kg", "kp", "numbers")}
            return {"state": SELECTED, "region": region, "blocked_by_rank": None}
    return {"state": NO_REGION, "region": None, "blocked_by_rank": None}


def isolated_passes(table, status_by_number) -> list:
    """통과했지만 통과 영역에 들지 못한 점. 공개하되 후보로 삼지 않는다."""
    inside = {number for row in table if row["status"] == PASS for number in row["numbers"]}
    return sorted(number for number, status in status_by_number.items() if status == PASS and number not in inside)


def reconfirm_verdict(status_by_pair, pairs) -> dict:
    """KODEX 재확인: 16쌍 중 하나라도 확정 미달이면 미달, 전부 통과면 통과, 나머지는 미확정."""
    pairs = list(pairs)
    if set(status_by_pair) != set(pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("reconfirmation needs every registered setting/start pair exactly once")
    values = [status_by_pair[pair] for pair in pairs]
    status = FAIL if FAIL in values else PASS if all(value == PASS for value in values) else OPEN
    return {"status": status, "pairs": len(pairs),
            "counts": {name: values.count(name) for name in (PASS, FAIL, OPEN)},
            "failed": [pair for pair in pairs if status_by_pair[pair] == FAIL],
            "undetermined": [pair for pair in pairs if status_by_pair[pair] == OPEN]}


def extension_path(row, threshold) -> dict:
    """한 경로의 기본 비용 전체 차. 비교 보류·자료 결측은 미확정, 기본 비용 개설 미달은 확정 미달."""
    status, reasons = _ordered_state(*(row[key] for key in FLAG_KEYS))
    if status is not None:
        return {"status": status, "reasons": reasons}
    if row["base_delta"] is None:
        raise ValueError("a base-cost path without a value needs a hold or failure reason")
    positive = row["base_delta"] > threshold
    return {"status": POSITIVE if positive else FAIL, "reasons": [] if positive else ["base_edge"]}


def _extension_held(point):
    """판정 전에 비교가 성립하지 않는 경우의 사유: 입력 보류, 명목 한도 미달."""
    if point.get("held"):
        return point["held"]
    return None if point["capacity_ok"] else CAPACITY_REASON


def extension_point(point, threshold) -> dict:
    """같은 점에서 한 경로라도 확정 미달이면 확정 미달, 두 경로 모두 양수면 확정 양수, 나머지는 미확정.
    서로 다른 경로의 승리를 합쳐 세지 않는다."""
    held = _extension_held(point)
    if held is not None:
        return {"status": OPEN, "reasons": [held], "paths": None}
    paths = {name: extension_path(row, threshold) for name, row in point["paths"].items()}
    statuses = [row["status"] for row in paths.values()]
    status = FAIL if FAIL in statuses else POSITIVE if all(value == POSITIVE for value in statuses) else OPEN
    reasons = [f"{name}:{reason}" for name, row in paths.items() for reason in row["reasons"]]
    return {"status": status, "reasons": reasons, "paths": paths}


def extension_product(statuses, *, points, points_min) -> dict:
    """상품 하나(분모 8 고정): 확정 양수 P ≥ 6 → 관찰, P + U < 6 → 미달, 나머지는 미확정."""
    statuses = list(statuses)
    if len(statuses) != points:
        raise ValueError(f"an extension product needs all {points} points; the denominator is not reduced")
    positive, undetermined = statuses.count(POSITIVE), statuses.count(OPEN)
    status = OBSERVED if positive >= points_min else FAIL if positive + undetermined < points_min else OPEN
    return {"status": status, "positive": positive, "undetermined": undetermined,
            "failed": statuses.count(FAIL), "denominator": points, "points_min": points_min}


def extension_verdict(products, *, expected) -> dict:
    """두 상품 각각 6/8 이상이어야 '여러 상품에서 관찰'. 한 상품이라도 확정 미달이면 미달."""
    if list(products) != list(expected):
        raise ValueError("extension results must cover the registered products in order")
    statuses = [row["status"] for row in products.values()]
    status = FAIL if FAIL in statuses else OBSERVED if all(value == OBSERVED for value in statuses) else OPEN
    return {"status": status, "products": dict(products), "independent_samples": False}


def conclusion(sentences, selection, *, reconfirm=None, candidate=None) -> dict:
    """미리 정한 결론 문장(7절). 확장 관찰 여부는 별도 칸이며 이 결론을 바꾸지 않는다."""
    if selection["state"] == NO_REGION:
        return {"sentence": sentences["no_basis"], "stage": "primary_start", "candidate": None}
    if selection["state"] == OPEN:
        return {"sentence": sentences["undetermined"], "stage": "primary_start", "candidate": None}
    if reconfirm is None:
        return {"sentence": None, "stage": "primary_start", "candidate": None, "pending": "kodex_reconfirm"}
    if reconfirm["status"] != PASS:
        name = "no_basis" if reconfirm["status"] == FAIL else "undetermined"
        return {"sentence": sentences[name], "stage": "kodex_reconfirm", "candidate": None}
    return {"sentence": sentences["candidate"], "stage": "kodex_reconfirm", "candidate": candidate,
            "limits": list(sentences["candidate_limits"]), "next_step": sentences["candidate_next_step"]}
