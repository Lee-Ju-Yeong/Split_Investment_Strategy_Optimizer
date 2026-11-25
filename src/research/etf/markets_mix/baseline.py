"""공개값 재현(설계 v1.1 7절 2·3번). 실자료로 돌리는 것은 이미 공개된 세 점뿐이다.

- 기준점: 069500·2011-01-03·28번. 정식 결과의 같은 경로·비용 값 802항목과 허용오차 0으로 대조한다.
- 확장 두 상품: 229200·169950의 2017-01-02·28번. 같은 정식 결과의 공개값과 같은 방법으로 대조한다.

이 조사의 단위 실행기가 연 장부를 kodex_grid의 조립(승인 코드 study._path_result)에 그대로 넣어 정식 결과의 꼴로 만들고
kodex_grid의 잎 단위 대조를 쓴다. 단위 실행기가 새로 만드는 값(비용 하나의 사유·지표·일별 계열·체결 건수)은 정식 기록의
같은 비용 값과 따로 검산한다. 값은 공개 기록에 싣지 않고 성립 여부만 남긴다. 불일치가 있으면 본실행을 허용하지 않는다.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import time
import traceback

from ..engine_inputs import _sha
from ..kodex_grid import baseline as grid_baseline
from ..kodex_grid.grid import segment_labels
from ..kodex_grid.identity import IdentityError
from ..kodex_grid.judge import reasons_for
from ..kodex_grid.reading import dec, exact_sub
from ..kodex_grid.records import AssemblyError
from ..kodex_grid.runner import COSTS, Bench, PointLedgers, assemble_point
from ..kodex_grid.sealing import (SEALED, OverwriteError, captured_output, digest, dump, error_code, flush_standard,
                                  with_seal, write_once, writer_lock)
from . import permit
from .batch import compact
from .definition import SOLO, Unit, plan_units, unit_config
from .identity import MixContext, check_current, check_frozen, check_loaded, check_synthetic
from .records import UNIT_SCHEMA, Seat, check_record
from .rows import SHOWN, _mixed, reproduced
from .unit import REASON_KEYS, open_unit, unit_result


FAULTS = grid_baseline.FAULTS
FLAG_OF = {"failures": "failures", "holds": "holds", "later": "later"}
REPORT_HOURS = 24  # 예상 경과시간이 이보다 길면 본실행 전에 보고한다
CONDITIONS = ("확장 두 상품의 28번 한 단위(두 경로)의 경과시간이 모든 단위에 이어지고 동시 실행한 작업이 각자 CPU 하나를 "
              "온전히 쓴다는 조건의 추정이다. 상한이 아니다: 설정(차수·간격)과 상품에 따라 단위별 시간은 크게 달라질 수 있다. "
              "본실행 중에는 진행 로그의 기술적 완료시간만으로 남은 시간을 다시 잡는다(eta 명령).")


class BaselineError(RuntimeError):
    """재현 실행의 기술 오류. 공개 경로에는 기술 코드만 남기고 전문은 봉인 로그에 둔다."""


@dataclass(frozen=True)
class Point:
    """재현할 한 점: 상품·시작일과 정식 결과의 그 시작일 기록."""

    symbol: str
    start: date
    formal: dict

    @property
    def name(self) -> str:
        return f"{self.symbol}@{self.start.isoformat()}"


def formal_points(mix, formal_path) -> list:
    """정식 결과 파일의 SHA-256을 등록값과 먼저 맞춘 뒤에만 세 점의 공개 기록을 읽는다."""
    path = Path(formal_path)
    if _sha(path) != mix.approved["formal_result_sha256"]:
        raise IdentityError("formal result file differs from the registered SHA-256; nothing is compared")
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    targets = [("primary", *mix.baseline["primary"])] + [("extension", *pair) for pair in mix.baseline["extension"]]
    return [Point(symbol, start, rows[row]["symbols"][symbol]["starts"][start.isoformat()])
            for row, symbol, start in targets]


def _unit(ctx: MixContext, point: Point, cost) -> Unit:
    return Unit(SOLO, point.symbol, point.start, ctx.mix.sensitivity["number"], cost)


def grid_shape(runs) -> dict:
    """단위 실행기가 연 두 비용의 장부를 kodex_grid의 조립에 그대로 넣는다(정식 결과와 같은 꼴)."""
    job = runs[COSTS[0]].job
    head = {key: value for key, value in runs[COSTS[0]].head.items() if key != "opened_at"}
    if any(run.groups is None for run in runs.values()):
        return assemble_point(PointLedgers(job, head))
    return assemble_point(PointLedgers(job, head, {cost: run.groups for cost, run in runs.items()},
                                       job.split_holds()))


def _same(formal, mine) -> bool:
    """잎 단위로 같은지(수치 칸은 Decimal 값, 그 밖은 원문과 자료형). kodex_grid의 대조 규칙 그대로다."""
    found = grid_baseline.Comparison()
    found.walk(formal, mine, "")
    return not found.mismatches and found.matched > 0


@dataclass(frozen=True)
class Slot:
    """한 비용·한 경로의 정식 기록 행과 단위 기록 행."""

    cost: str
    path: str
    formal: dict
    mine: dict

    @property
    def flags(self) -> dict:
        return grid_baseline.formal_flags(self.formal["verdict"])

    @property
    def usable(self) -> bool:
        return grid_baseline._usable(self.formal, self.flags, self.cost)


def _values_equal(slot: Slot) -> bool:
    """정식 기록이 그 비용을 비교할 수 있으면 단위도 비교 가능이고 전략·보유 지표와 차가 같다. 아니면 단위도 빈칸이다."""
    if not slot.usable:
        return slot.mine["comparable"] is False and slot.mine["strategy"] is None
    mine = {"strategy": slot.mine["strategy"], "hold": slot.mine["hold"], **slot.mine["delta"]}
    return slot.mine["comparable"] is True and _same(slot.formal[slot.cost], mine)


def _reasons_equal(slot: Slot) -> bool:
    expected = {key: reasons_for(slot.flags[key], slot.cost) for key in REASON_KEYS}
    return slot.mine["reasons"] == expected


def _hold_equal(slot: Slot) -> bool:
    formal = slot.formal["hold_metrics"][slot.cost]
    return slot.mine["hold_only"] is None if formal is None else _same(formal, slot.mine["hold_only"])


def _orders_equal(slot: Slot) -> bool:
    return _same(slot.formal["start_orders"][slot.cost], slot.mine["start_orders"])


def _counts_equal(slot: Slot) -> bool:
    """재진입 시도·체결 수, 미체결 사유별 건수, 매수 차단 연도가 정식 기록의 목록과 같은지."""
    attempts, counts = slot.formal["reentries"][slot.cost], slot.mine["reentry_counts"]
    closed = slot.formal["buys_closed_years"][slot.cost] == slot.mine["buys_closed_years"]
    if attempts is None:
        return closed and (counts["attempts"], counts["filled"], counts["unfilled"]) == (None, None, None)
    outcomes = [row["outcome"] for row in attempts]
    unfilled = {name: outcomes.count(name) for name in sorted(set(outcomes)) if name != "filled"}
    return closed and (counts["attempts"], counts["filled"], counts["unfilled"]) == (
        len(outcomes), outcomes.count("filled"), unfilled)


def _events_equal(runs, results) -> bool:
    """체결 건수의 독립 기대값: 장부의 현금 사건에서 센 매수·매도 수와 같은지."""
    for cost, run in runs.items():
        for path, group in (run.groups or {}).items():
            ledger = group[run.job.mode]
            if ledger.result is None:
                continue
            counted, stated = grid_baseline.event_counts(ledger), results[cost]["paths"][path]["split_trades"]
            if (counted["buys"], counted["sells"]) != (stated["buys"], stated["sells"]):
                return False
    return True


def _record(ctx: MixContext, unit: Unit, result) -> dict:
    seat = Seat(ctx, unit)
    return {"schema": UNIT_SCHEMA, "exploration": True, "formal_run": False, "identity": seat.identity(),
            "setting": seat.setting(), "result": result}


def _one_share(ctx: MixContext, record) -> bool:
    """한 몫만 넣은 혼합이 단독과 같은지: 계열 한 벌을 혼합 조립에 넣어 얻은 지표가 단위의 지표와 같다."""
    result = record["result"]
    base, start = ctx.base, date.fromisoformat(result["head"]["start"])
    labels = [(label, label) for label in segment_labels(base.segments, start, base.end)]
    for path, row in result["paths"].items():
        if not row["comparable"]:
            continue
        slot = _mixed(ctx, [(result["head"]["symbol"], record)], (path,), labels=labels)
        same = all(slot[name][key] == dec(row[name][key]) for name in ("strategy", "hold") for key in SHOWN)
        if not same or slot["delta_cagr"] != exact_sub(row["strategy"]["cagr"], row["hold"]["cagr"]):
            return False
    return True


def _passes(ctx: MixContext, unit: Unit, record) -> bool:
    try:
        check_record(Seat(ctx, unit), record)
    except AssemblyError:
        return False
    return True


def _holds(check, items) -> bool:
    """모든 항목에서 성립해야 한다. 항목이 하나도 없거나 값이 없으면 실패다."""
    try:
        return bool(items) and all(check(item) for item in items)
    except FAULTS:
        return False


def _applied(ctx: MixContext, point: Point) -> bool:
    """1억 원 단독 28번의 적용 설정 = 고정 설정에서 재진입 방식만 바꾼 것 = 등록값."""
    config = unit_config(ctx.mix, _unit(ctx, point, COSTS[0]), base=ctx.base, grid=ctx.spec)
    independent = ctx.base.with_changes(modes=(ctx.spec.mode,)).applied_sha256()
    return config.applied_sha256() == independent == ctx.mix.approved["baseline_applied_config_sha256"]


def self_checks(ctx: MixContext, point: Point, measured) -> dict:
    """단위 실행기가 새로 만드는 값의 검산. 값은 싣지 않고 성립 여부만 남긴다."""
    formal, records = point.formal["items"][ctx.spec.mode]["paths"], measured["records"]
    slots = [Slot(cost, path, formal[path], records[cost]["result"]["paths"][path])
             for cost in COSTS for path in formal]
    units = [(_unit(ctx, point, cost), records[cost]) for cost in COSTS]
    return {"applied_config_equals_registered": _applied(ctx, point),
            "unit_values_equal_formal": _holds(_values_equal, slots),
            "unit_reasons_equal_formal": _holds(_reasons_equal, slots),
            "unit_hold_equals_formal": _holds(_hold_equal, slots),
            "unit_start_orders_equal_formal": _holds(_orders_equal, slots),
            "unit_counts_equal_formal": _holds(_counts_equal, slots),
            "trade_counts_equal_ledger_events": measured["events_equal"],
            "series_reproduce_metrics": _holds(lambda pair: reproduced(ctx, pair[1]), units),
            "one_share_mix_equals_solo": _holds(lambda pair: _one_share(ctx, pair[1]), units),
            "unit_passes_record_checks": _holds(lambda pair: _passes(ctx, *pair), units)}


def _timing(bench: Bench, seconds) -> dict:
    return {"total_seconds": seconds, "by_kind_seconds": dict(bench.clock.seconds),
            "by_kind_counts": dict(bench.clock.counts),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def _measure(ctx: MixContext, point: Point) -> dict:
    """한 점을 한 프로세스에서 두 비용 순서로 실행하며 장부 종류별 시간을 잰다."""
    bench, started = Bench(), time.perf_counter()
    with bench.clock("load_inputs"):
        inputs = ctx.source.load(point.symbol, point.start)
    check_loaded(ctx, inputs)
    runs = {}
    for cost in COSTS:
        unit = _unit(ctx, point, cost)
        config = unit_config(ctx.mix, unit, base=ctx.base, grid=ctx.spec)
        runs[cost] = open_unit(inputs, config, point.start, cost=cost, nominal_floor=ctx.mix.floor(unit), bench=bench)
    results = {cost: unit_result(run) for cost, run in runs.items()}
    mine = json.loads(dump(grid_shape(runs)))
    seconds = time.perf_counter() - started
    records = {cost: json.loads(compact(_record(ctx, _unit(ctx, point, cost), results[cost]))) for cost in COSTS}
    return {"mine": mine, "records": records, "events_equal": _events_equal(runs, records_results(records)),
            "timing": _timing(bench, seconds)}


def records_results(records) -> dict:
    return {cost: record["result"] for cost, record in records.items()}


def _evidence(ctx: MixContext, point: Point, measured) -> dict:
    """한 점의 (공개할 증거, 봉인할 불일치 내역, 봉인할 단위 본문)."""
    comparison, details = grid_baseline.compare(point.formal, measured["mine"]["run_start"], ctx.spec.mode)
    texts = {cost: compact(record) for cost, record in measured["records"].items()}
    public = {"symbol": point.symbol, "start": point.start.isoformat(), "number": ctx.mix.sensitivity["number"],
              "comparison": comparison, "self_checks": self_checks(ctx, point, measured),
              "sealed_units": {cost: digest(text) for cost, text in texts.items()}, "timing": measured["timing"]}
    return {"public": public, "details": details, "texts": texts}


_ACTIVE = {}  # 포크한 자식 프로세스가 물려받는 문맥·점·봉인 폴더


def _work(index: int) -> dict:
    """한 점을 실행·대조한다. 실행 중의 출력과 예외 전문은 봉인 로그에만 남기고 기술 코드만 돌려준다."""
    ctx, point, folder = _ACTIVE["ctx"], _ACTIVE["points"][index], _ACTIVE["folder"]
    with captured_output(folder / "sealed" / f"{point.name}.output.log") as sink:
        try:
            return _evidence(ctx, point, _measure(ctx, point))
        except Exception as error:
            traceback.print_exc(file=sink)
            return {"error": error_code(error)}


def _dispatch(count: int, workers: int) -> list:
    if workers <= 1 or count <= 1:
        return [_work(index) for index in range(count)]
    flush_standard()
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(min(workers, count), mp_context=context) as pool:
        return list(pool.map(_work, range(count)))


def _measured(ctx: MixContext, points, folder: Path, *, workers: int) -> list:
    _ACTIVE.update(ctx=ctx, points=points, folder=folder)
    try:
        found = _dispatch(len(points), workers)
    finally:
        _ACTIVE.clear()
    failed = {point.name: row["error"] for point, row in zip(points, found) if "error" in row}
    if failed:
        raise BaselineError(f"baseline run ended with a technical error: {failed}")
    return found


def unit_seconds(points: dict, primary_start: str):
    """주 시작일에서 잰 한 단위(한 비용·두 경로)의 평균 경과시간. 입력 적재 시간은 뺀다."""
    rows = [row["timing"] for row in points.values() if row["start"] == primary_start]
    spent = sum(row["total_seconds"] - row["by_kind_seconds"].get("load_inputs", 0.0) for row in rows)
    return spent / (len(rows) * len(COSTS)) if rows else None


def estimate(seconds, units: int, cpus=None) -> dict:
    """한 단위의 경과시간으로 잡은 조건부 추정과 권장 동시 실행 수. 상한이 아니다."""
    cpus = cpus or os.cpu_count() or 1
    workers = max(1, min(units, cpus - 2 if cpus > 3 else cpus - 1))
    hours = None if seconds is None else math.ceil(units / workers) * seconds / 3600
    return {"per_unit_seconds": seconds, "units": units, "paths_per_unit": 2, "cpus": cpus,
            "recommended_workers": workers, "conditional_hours": hours, "report_threshold_hours": REPORT_HOURS,
            "exceeds_report_threshold": hours is not None and hours > REPORT_HOURS, "conditions": CONDITIONS}


def _allowed(ctx: MixContext, out, points: dict) -> bool:
    """본실행 허용은 실제 묶음에서 세 점 모두 불일치 0건·필수 검산 전부 참일 때만이다(봉인 파일은 쓰기 전이라 뺀다)."""
    problems = [problem for row in points.values() for problem in permit.point_problems(ctx, out, row)
                if "sealed baseline unit" not in problem]
    return ctx.registered and sorted(points) == sorted(permit.expected_points(ctx)) and not problems


def _selftest_seal(ctx: MixContext, out):
    """실제 묶음의 재현은 같은 판본의 합성 시험 통과 기록이 있어야 잰다. 합성 입력은 허가 대상이 아니다."""
    permit.check_folder(out)
    check_current(ctx)
    if not ctx.registered:
        check_synthetic(ctx)
        return None
    check_frozen(ctx)
    return permit.check_selftest(out)["seal"]


def _write(folder: Path, record, found) -> None:
    """봉인 원본(단위 본문·불일치 내역)과 공개 기록을 한 번만 쓴다."""
    for name, row in found.items():
        for cost, text in row["texts"].items():
            write_once(folder / "sealed" / f"{name}-{cost}.unit.json", text, mode=SEALED)
        write_once(folder / "sealed" / f"{name}.mismatches.json", dump(row["details"]), mode=SEALED)
    write_once(folder / "compare.json", dump(record))
    write_once(folder / "report.md", markdown(json.loads(dump(record))))


def run_baseline(ctx: MixContext, formal_path, out, *, workers=3) -> dict:
    """시험 통과 기록 → 정식 결과 해시 → 세 점 실행 → 대조와 검산. 결과는 out/baseline/에 한 번만 쓴다."""
    folder, seal = Path(out) / "baseline", _selftest_seal(ctx, out)
    with writer_lock(out):
        if (folder / "compare.json").exists():
            raise OverwriteError("a baseline reproduction is already recorded here; use a new output folder")
        points, started = formal_points(ctx.mix, formal_path), time.perf_counter()
        found = dict(zip([point.name for point in points], _measured(ctx, points, folder, workers=workers)))
        public = {name: row["public"] for name, row in found.items()}
        seconds = unit_seconds(public, ctx.mix.baseline["extension"][0][1].isoformat())
        record = with_seal({
            "schema": permit.BASELINE_SCHEMA, "exploration": True, "formal_run": False, "identity": ctx.identity,
            "formal_result": {"path": str(formal_path), "sha256": ctx.mix.approved["formal_result_sha256"]},
            "selftest_seal": seal, "points": public, "main_run_allowed": _allowed(ctx, out, public),
            "timing": {"wall_seconds": time.perf_counter() - started, "workers": workers},
            "estimate": estimate(seconds, len(plan_units(ctx.mix, ctx.spec)))})
        _write(folder, record, found)
    return record


def _point_lines(name, row) -> list:
    comparison, timing = row["comparison"], row["timing"]
    return [f"### {name} (설정 {row['number']}번)", "",
            f"- 일치한 잎 {comparison['matched']}개 / 정식 기록의 잎 {comparison['formal_leaves']}개 / 이쪽 잎 "
            f"{comparison['exploration_leaves']}개, 불일치 {len(comparison['mismatches'])}개, 표기만 다른 잎 "
            f"{len(comparison['notation_only'])}개",
            f"- 항목별 일치 수: {comparison['by_item']}",
            f"- 검산(값은 싣지 않음): {row['self_checks']}",
            f"- 소요 {timing['total_seconds']:.1f}초, 최대 메모리 {timing['max_rss_mb']:.0f} MB",
            f"- 장부 종류별 시간(초): {timing['by_kind_seconds']} / 횟수: {timing['by_kind_counts']}",
            *[f"- 불일치 칸: {item['where']} ({item['kind']})" for item in comparison["mismatches"]], ""]


def markdown(record) -> str:
    plan = record["estimate"]
    hours = "측정 없음" if plan["conditional_hours"] is None else f"약 {plan['conditional_hours']:.1f}시간"
    lines = ["# 공개값 재현 (markets-mix 단위 실행기 대 정식 결과)", "",
             f"- 본실행 허용: **{'예' if record['main_run_allowed'] else '아니오 — 본실행 금지'}**",
             f"- 정식 결과 SHA-256: `{record['formal_result']['sha256']}`",
             f"- 세 점 동시 실행 {record['timing']['workers']}개, 전체 {record['timing']['wall_seconds']:.1f}초",
             f"- 본실행 조건부 추정: {plan['units']}단위 × {plan['paths_per_unit']}경로, 한 단위 "
             f"{plan['per_unit_seconds'] or 0:.1f}초, 동시 {plan['recommended_workers']}개에서 {hours}, "
             f"24시간 초과 예상: {plan['exceeds_report_threshold']}",
             f"- 추정의 조건: {plan['conditions']}", ""]
    for name, row in record["points"].items():
        lines += _point_lines(name, row)
    lines += ["## 실행 식별값", ""]
    lines += [f"- {key}: `{value}`" for key, value in record["identity"].items() if key != "inputs"]
    return "\n".join(lines + ["", "## 입력 대조", "", *grid_baseline._input_lines(record["identity"]["inputs"])]) + "\n"
