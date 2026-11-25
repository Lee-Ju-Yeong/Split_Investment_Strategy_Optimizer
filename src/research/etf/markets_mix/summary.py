"""조립 결과의 표(설계 v2 1·3절, 규칙은 v1.1 6절): 묶음별 주 판정 행과 결과 문장, 대상별 요약과 8점 영역, 28번 민감도.

판정은 각 묶음의 주 시작일·기본 비용의 행에만 쓴다. 여기의 나머지 표는 모두 병기이며 판정을 바꾸지 않는다.
"""
from __future__ import annotations

from collections import Counter

from . import judge
from .identity import MixContext
from .rows import MIX_ROW, SOLO_ROW, Table, row_key


def subjects(ctx: MixContext) -> list:
    """판정 대상(묶음 순서, 묶음 안에서 단독 다음 혼합): (대상, 종류, 이름, 묶음)."""
    found = []
    for cohort in ctx.mix.cohorts:
        found += [(symbol, SOLO_ROW, symbol, cohort) for symbol in cohort.solo]
        found += [(row, MIX_ROW, row.id, cohort) for row in cohort.mixes]
    return found


def primary_slot(ctx: MixContext, cohort, number) -> tuple:
    start, cost = ctx.mix.slots(cohort)[0]
    return number, start, cost


def primary_rows(ctx: MixContext, table: Table) -> list:
    """각 묶음 주 시작일·기본 비용의 모든 행(대상 × 36개 설정)."""
    return [table.row(subject, kind, primary_slot(ctx, cohort, setting.number))
            for subject, kind, _, cohort in subjects(ctx) for setting in ctx.spec.settings()]


def expected_keys(ctx: MixContext) -> list:
    return [row_key(name, primary_slot(ctx, cohort, setting.number))
            for _, _, name, cohort in subjects(ctx) for setting in ctx.spec.settings()]


def _counter_detail(row) -> dict:
    """반례 행에 함께 쓰는 것: 그 계좌가 KODEX 200 보유보다 더 벌었는지, 자기 보유가 하락·횡보·상승 중 무엇이었는지."""
    versus = {path: slot["vs_reference_hold"] for path, slot in row["paths"].items()}
    return {"key": row["key"], "subject": row["subject"], "start": row["start"], "number": row["number"],
            "beat_reference_hold": {path: judge.beat(value) for path, value in versus.items()},
            "own_hold_regime": {path: slot["regime"] for path, slot in row["paths"].items()}}


def _left_open(row) -> dict:
    """열위도 반례도 아닌 행: 상태·경로별 부호·사유."""
    return {"key": row["key"], "subject": row["subject"], "start": row["start"], "number": row["number"],
            "status": row["status"], "signs": row["signs"],
            "reasons": {path: slot["reasons"] for path, slot in row["paths"].items()}}


def _counters(rows) -> list:
    """반례 수: (시작일, 대상, 설정 수), 행 순서대로."""
    found = {}
    for row in rows:
        if row["status"] == judge.COUNTER:
            key = (row["start"], row["subject"])
            found[key] = found.get(key, 0) + 1
    return [(start, subject, count) for (start, subject), count in found.items()]


def _by_cohort(ctx: MixContext, rows) -> list:
    """묶음별 상태 집계."""
    return [{"cohort": cohort.id, "start": cohort.start.isoformat(),
             "rows": sum(row["start"] == cohort.start.isoformat() for row in rows),
             "counts": {name: sum(row["start"] == cohort.start.isoformat() and row["status"] == name for row in rows)
                        for name in judge.STATUSES}} for cohort in ctx.mix.cohorts]


def check_once(rows, expected) -> None:
    """실제 행 열쇠의 출현 횟수가 기대 열쇠마다 정확히 1인지. 사전으로 바꾸기 전에 보므로 중복이 지워지지 않는다."""
    found = Counter(row["key"] for row in rows)
    if found != Counter(expected) or any(count != 1 for count in found.values()):
        raise ValueError("the conclusion needs every planned primary row exactly once")


def conclusion(ctx: MixContext, rows) -> dict:
    """미리 정한 세 문장 가운데 하나. 계획한 주 판정 행이 모두 한 번씩 있어야 고른다."""
    check_once(rows, expected_keys(ctx))
    counters = _counters(rows)
    chosen = judge.conclusion({row["key"]: row["status"] for row in rows}, expected_keys(ctx), ctx.mix.sentences,
                              counters=counters)
    rest = [row for row in rows if row["status"] not in (judge.INFERIOR, judge.COUNTER)]
    return {**chosen, "counterexamples": [_counter_detail(row) for row in rows if row["status"] == judge.COUNTER],
            "counterexample_settings": [{"start": s, "subject": n, "count": c} for s, n, c in counters],
            "by_cohort": _by_cohort(ctx, rows), "left_open": [_left_open(row) for row in rest],
            "limits": list(ctx.mix.sentences["limits"])}


def _regions(ctx: MixContext, rows) -> dict:
    """8점 영역(병기): 모든 경로의 차가 0보다 큰 점과, +1%p 이상인 점으로 각각 본다."""
    thresholds, regions = ctx.mix.thresholds, [region.describe() for region in ctx.spec.regions()]
    deltas = {row["number"]: {path: slot["delta_cagr"] for path, slot in row["paths"].items()} for row in rows}
    above = {n: judge.point_meets(d, thresholds["edge_gt"], strict=True) for n, d in deltas.items()}
    review = {n: judge.point_meets(d, thresholds["review_base_edge_min"], strict=False) for n, d in deltas.items()}
    return {"edge_gt": judge.regions_table(regions, above), "review_base_edge_min": judge.regions_table(regions, review)}


def subject_view(ctx: MixContext, rows) -> dict:
    """한 대상의 36행 요약: 상태별 수, 반례 설정 번호, 자기 보유 구분, 8점 영역."""
    name, kind, start = rows[0]["subject"], rows[0]["kind"], rows[0]["start"]
    counts = {status: sum(row["status"] == status for row in rows) for status in judge.STATUSES}
    regimes = sorted({slot["regime"] for row in rows for slot in row["paths"].values() if slot["regime"]})
    return {"subject": name, "kind": kind, "start": start, "rows": len(rows), "counts": counts,
            "counterexample_numbers": [row["number"] for row in rows if row["status"] == judge.COUNTER],
            "own_hold_regimes": regimes, "regions": _regions(ctx, rows)}


def subject_views(ctx: MixContext, rows) -> list:
    return [subject_view(ctx, [row for row in rows if (row["subject"], row["start"]) == (name, cohort.start.isoformat())])
            for _, _, name, cohort in subjects(ctx)]


def _deltas(row) -> dict:
    return {path: slot["delta_cagr"] for path, slot in row["paths"].items()}


def _reviews(ctx: MixContext, cohort, rows) -> dict:
    """재검토 문턱(시작일별): 모든 경로에서 기본 비용 차가 +1%p 이상이고 높은 비용 차가 양수."""
    base, high = ctx.mix.costs["primary"], ctx.mix.sensitivity["cost"]
    by = {(row["start"], row["cost"]): row for row in rows}
    return {start.isoformat(): judge.review_threshold(_deltas(by[(start.isoformat(), base)]),
                                                      _deltas(by[(start.isoformat(), high)]), ctx.mix.thresholds)
            for start in cohort.starts()}


def sensitivity_view(table: Table, entry) -> dict:
    """28번의 세 시작일 × 두 비용(민감도). 주 판정을 바꾸지 않는다. entry: subjects()의 한 항목."""
    (subject, kind, _, cohort), ctx = entry, table.ctx
    number = ctx.mix.sensitivity["number"]
    rows = [table.row(subject, kind, (number, start, cost)) for start, cost in ctx.mix.slots(cohort)]
    signs = {(row["start"], row["cost"]): row["signs"] for row in rows}
    marks = judge.sensitivity(signs, table.required(kind, subject))
    return {"subject": rows[0]["subject"], "kind": kind, "cohort": cohort.id, "number": number, **marks,
            "review_threshold": _reviews(ctx, cohort, rows), "rows": rows}


def sensitivity_views(ctx: MixContext, table: Table) -> list:
    return [sensitivity_view(table, entry) for entry in subjects(ctx)]
