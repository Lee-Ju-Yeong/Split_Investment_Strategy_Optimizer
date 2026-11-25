"""공개 기준점 대조(설계 v2.1 8.3절).

069500·2011-01-03·M·9차/kg 2/kp 3/코어 80%(28번)를 탐색 실행기로 돌려 정식 결과 파일의 같은 경로·비용 값과
허용오차 0으로 대조한다. 지정한 수치 칸만 Decimal 값으로 대조하고(1.0 = 1.00), 종목코드·해시·상태·날짜·개수는
원문과 자료형으로 대조한다. 새 판정·새 지표는 값을 공개하지 않고, 탐색 출력과 독립된 기대값(정식 기록의 개설 상태·
사유·수치와 장부의 현금 사건)으로 검산한 결과만 남긴다. 불일치·미실행이 있으면 같은 판본의 본실행을 허용하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback

from ..engine_inputs import _sha
from ..kodex_core import study
from . import judge, permit, reading
from .grid import segment_labels, setting_config
from .identity import Context, IdentityError, check_frozen, check_loaded, check_synthetic
from .records import UNIT_SCHEMA, AssemblyError, Seat, check_record, primary_unit
from .runner import COSTS, Bench, PointLedgers, assemble_point, point_ledgers
from .sealing import (SEALED, OverwriteError, captured_output, digest, dump, error_code, with_seal, write_once,
                      writer_lock)


HEAD_KEYS = ("symbol", "start", "end", "last_evaluation", "data_end", "basis", "capacity", "input_provenance",
             "warnings")
NUMERIC_KEYS = frozenset({"limit_krw", "floor_krw", "core_plus_split", "hold", "notional", "cagr", "mdd",
                          "exposure_mean", "ending_equity_krw", "mdd_from_inception_peak", "delta_cagr",
                          "delta_abs_mdd", "fraction", "cagr_edge", "exposure", "mdd_excess", "stress_cagr_edge"})
FAULTS = (KeyError, TypeError, ValueError, AttributeError, ArithmeticError)
REPORT_HOURS = 24  # 예상 경과시간이 이보다 길면 본실행 전에 사용자에게 보고한다
CONDITIONS = ("기준점 한 설정의 경과시간이 모든 설정에 이어지고 동시 실행한 작업이 각자 CPU 하나를 온전히 쓴다는 조건의 "
              "추정이다. 상한이 아니다: 보유 차수·호가 수·전량 매도 뒤 대기기간에 따라 설정별 시간은 크게 달라질 수 있다. "
              "본실행 중에는 진행 로그의 기술적 완료시간만으로 남은 시간을 다시 잡는다(eta 명령).")


class BaselineError(RuntimeError):
    """기준점 실행의 기술 오류. 공개 경로에는 기술 코드만 남기고 전문은 봉인 로그에 둔다."""


def formal_start(spec, formal_path) -> dict:
    """정식 결과 파일의 SHA-256을 등록값과 먼저 맞춘 뒤에만 기준점(주 시작일) 기록을 읽는다."""
    path = Path(formal_path)
    if _sha(path) != spec.approved["formal_result_sha256"]:
        raise IdentityError("formal result file differs from the registered SHA-256; baseline is not compared")
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]["primary"]["symbols"][spec.symbol]["starts"]
    return rows[spec.primary_start.isoformat()]


def _decimal(value):
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def same_leaf(key, formal, mine) -> bool:
    """지정한 수치 칸의 문자열만 Decimal 값으로, 그 밖(종목코드·해시·상태·날짜·개수·참거짓)은 원문과 자료형으로."""
    if key in NUMERIC_KEYS and isinstance(formal, str) and isinstance(mine, str):
        left, right = _decimal(formal), _decimal(mine)
        if left is not None and right is not None:
            return left == right
    return type(formal) is type(mine) and formal == mine


def count_leaves(value) -> int:
    """잎의 수(빈 목록·빈 사전은 한 자리). 대조와 따로 센다."""
    if isinstance(value, dict):
        return sum(count_leaves(item) for item in value.values()) or 1
    if isinstance(value, list):
        return sum(count_leaves(item) for item in value) or 1
    return 1


class Comparison:
    """잎 단위 대조. 사전은 열쇠 집합까지, 목록은 길이와 기록 순서까지 같아야 한다.
    mismatches에는 칸의 경로와 차이의 종류만 싣고, 값은 details(봉인)에만 둔다."""

    def __init__(self):
        self.matched, self.string_identical = 0, 0
        self.mismatches, self.details, self.notation_only = [], [], []

    def walk(self, formal, mine, where) -> None:
        if isinstance(formal, dict) and isinstance(mine, dict):
            self._dicts(formal, mine, where)
        elif isinstance(formal, list) and isinstance(mine, list):
            self._lists(formal, mine, where)
        else:
            self._leaf(formal, mine, where)

    def _differs(self, where, kind, **values) -> None:
        self.mismatches.append({"where": where, "kind": kind})
        self.details.append({"where": where, "kind": kind, **values})

    def _dicts(self, formal, mine, where) -> None:
        for key in sorted(set(formal) | set(mine)):
            if key in formal and key in mine:
                self.walk(formal[key], mine[key], f"{where}/{key}")
            else:
                self._differs(f"{where}/{key}", "missing_in_" + ("exploration" if key in formal else "formal"))
        self._empty(formal, mine)

    def _lists(self, formal, mine, where) -> None:
        if len(formal) != len(mine):
            self._differs(where, "length", formal=len(formal), exploration=len(mine))
        for index, (left, right) in enumerate(zip(formal, mine)):
            self.walk(left, right, f"{where}[{index}]")
        self._empty(formal, mine)

    def _empty(self, formal, mine) -> None:
        """둘 다 빈 목록·빈 사전이면 그 자리 하나가 일치한 것으로 센다."""
        if not formal and not mine:
            self.matched += 1
            self.string_identical += 1

    def _leaf(self, formal, mine, where) -> None:
        key = where.rsplit("/", 1)[-1].split("[")[0]
        if not same_leaf(key, formal, mine):
            self._differs(where, "value", formal=formal, exploration=mine)
            return
        self.matched += 1
        self.string_identical += formal == mine
        if formal != mine:
            self.notation_only.append(where)


def _parts(start, mode) -> dict:
    """대조할 묶음: 머리 항목, 종전 종합 상태, 경로별 항목(이름별로 두 경로를 묶는다)."""
    paths = start["items"][mode]["paths"]
    parts = {"head": {key: start[key] for key in HEAD_KEYS}, "item_status": start["items"][mode]["status"]}
    for name in sorted({key for row in paths.values() for key in row}):
        parts[name] = {path: row.get(name, "<absent>") for path, row in paths.items()}
    return parts


def compare(formal, mine, mode) -> tuple:
    """정식 기록과 탐색 실행 결과(둘 다 직렬화한 꼴)의 대조: (공개 요약, 봉인할 불일치 내역)."""
    total, by_item = Comparison(), {}
    left, right = _parts(formal, mode), _parts(mine, mode)
    for name in sorted(set(left) | set(right)):
        before = total.matched
        total.walk(left.get(name, "<absent>"), right.get(name, "<absent>"), name)
        by_item[name] = total.matched - before
    public = {"matched": total.matched, "string_identical": total.string_identical, "by_item": by_item,
              "formal_leaves": count_leaves(left), "exploration_leaves": count_leaves(right),
              "mismatches": total.mismatches, "notation_only": total.notation_only}
    return public, total.details


# ---------------------------------------------------------------- 독립 검산: 정식 기록만으로 구한 기대 판정

def formal_flags(verdict) -> dict:
    """정식 기록의 종전 경로 판정에서 개설 상태 사유를 되읽는다(보류 → 확정 미달 → 이후 결측 순으로 적혀 있다)."""
    later = list(verdict.get("later_holds") or [])
    if verdict["checks"] is not None:
        return {"holds": [], "failures": [], "later": []}
    reasons = list(verdict["reasons"])
    if verdict["status"] == "fail":
        return {"holds": [], "failures": reasons, "later": later}
    return {"holds": [] if reasons == later else reasons, "failures": [], "later": later}


def _usable(row, flags, cost) -> bool:
    """그 비용의 수치가 있고, 그 비용의 사유도 공통 사유(비용 표시 없는 사유)도 없다."""
    every = flags["holds"] + flags["failures"] + flags["later"]
    blocked = [r for r in every if r.startswith(cost + "_") or not r.startswith(("base_", "stress_"))]
    return row[cost] is not None and not blocked


def _edge(side) -> Fraction:
    return Fraction(side["strategy"]["cagr"]) - Fraction(side["hold"]["cagr"])


def _formal_checks(row, flags, thresholds) -> dict:
    """비용별 전체 기간 조건. 그 비용을 비교할 수 없으면(수치 없음, 그 비용의 사유, 공통 사유) None."""
    base, stress = _usable(row, flags, "base"), _usable(row, flags, "stress")
    return {"base_edge": _edge(row["base"]) >= Fraction(thresholds["base_edge_min"]) if base else None,
            "stress_edge": _edge(row["stress"]) > Fraction(thresholds["stress_edge_gt"]) if stress else None}


def _formal_status(flags, checks) -> str:
    """경로 상태: 시작 비교 보류 → 시작 시 확정 미달 → 이후 결측 → (사유가 없으면) 두 비용의 조건."""
    if flags["holds"] or (flags["later"] and not flags["failures"]):
        return judge.OPEN
    if flags["failures"]:
        return judge.FAIL
    if None in checks.values():
        return "invalid"  # 사유 없이 수치가 빈 경로는 판정할 수 없다
    return judge.PASS if all(checks.values()) else judge.FAIL


def _formal_path(row, thresholds) -> dict:
    """한 경로를 정식 기록만으로: 상태, 비용별 조건, 기본 비용 승리 구간(기본 비용을 비교할 수 있을 때만)."""
    flags = formal_flags(row["verdict"])
    checks = _formal_checks(row, flags, thresholds)
    usable = checks["base_edge"] is not None
    floor = Fraction(thresholds["segment_edge_gt"])  # 구간 차는 등록한 문턱(0)보다 커야 승리 구간이다
    pairs = zip(row["base"]["strategy"]["segments"], row["base"]["hold"]["segments"]) if usable else ()
    wins = [s["segment"] for s, h in pairs if Fraction(s["cagr"]) - Fraction(h["cagr"]) > floor] if usable else None
    return {"status": _formal_status(flags, checks), "checks": checks, "wins": wins}


def expected_paths(formal_paths, thresholds) -> dict:
    """정식 기록만으로 구한 경로별 상태와, 경로별·비용별 비교 가능 여부·전체 기간 조건 결과."""
    rows = {path: _formal_path(row, thresholds) for path, row in formal_paths.items()}
    return {path: {"status": row["status"], "checks": row["checks"]} for path, row in rows.items()}


def expected_verdict(formal_paths, thresholds) -> tuple:
    """판정 모듈을 쓰지 않는 독립 계산: 정식 기록의 개설 상태·사유·수치만으로 4절 상태와 공통 승리 구간."""
    rows = [_formal_path(row, thresholds) for row in formal_paths.values()]
    seen = [row["wins"] for row in rows if row["wins"] is not None]
    reachable = [label for label in seen[0] if all(label in other for other in seen[1:])] if seen else None
    short = reachable is not None and len(reachable) < thresholds["common_segments_min"]
    if any(row["status"] == "invalid" for row in rows):
        return "invalid", None
    lost = any(check is False for row in rows for check in row["checks"].values())
    failed = short or lost or any(row["status"] == judge.FAIL for row in rows)
    waiting = any(row["status"] == judge.OPEN for row in rows)
    status = judge.FAIL if failed else judge.OPEN if waiting else judge.PASS
    return status, reachable if len(seen) == len(rows) else None


# ---------------------------------------------------------------- 독립 검산: 분할 몫·같은 원금 보유·체결 건수

@dataclass(frozen=True)
class Slot:
    """정식 기록에서 수치가 있는 한 자리(경로·비용)와 그 설정."""

    path: str
    cost: str
    row: dict
    config: object

    @property
    def fee(self) -> Fraction:
        return Fraction(self.config.cost(self.cost))

    def order(self, name) -> dict:
        return self.row["start_orders"][self.cost]["orders"][name]

    def capital(self, name) -> Fraction:
        return Fraction(self.config.capital[name])

    def ending(self, name) -> Fraction:
        return Fraction(self.row[self.cost][name]["ending_equity_krw"])


def valued_slots(formal_paths, config) -> list:
    return [Slot(path, cost, row, config) for path, row in formal_paths.items() for cost in COSTS
            if row[cost] is not None]


def _cash_left(slot: Slot, name) -> Fraction:
    """첫 주문 뒤 남은 현금: 원금 − 주문 원금 × (1 + 수수료율)."""
    return slot.capital(name) - Fraction(slot.order(name)["notional"]) * (1 + slot.fee)


def _close_price(slot: Slot) -> Fraction:
    """단독 보유 장부의 종료 순자산에서 마지막 평가가격을 되구한다: (종료 순자산 − 남은 현금) ÷ 수량."""
    return (slot.ending("hold") - _cash_left(slot, "hold")) / slot.order("hold")["quantity"]


def expected_split(slot: Slot) -> dict:
    """분할 몫의 기대값: 전략 종료 순자산 − 코어 종료 순자산(코어 수량 × 마지막 가격 + 남은 현금)."""
    core = slot.order("core")["quantity"] * _close_price(slot) + _cash_left(slot, "core")
    ending = slot.ending("strategy") - core
    return {"core_ending": core, "ending": ending, "return": ending / slot.capital("split") - 1}


def expected_same_capital_hold(slot: Slot) -> dict:
    """같은 원금 단독 보유의 기대값: 분할 원금으로 시작 시가에 살 수 있는 최대 정수 주를 끝까지 보유한다."""
    hold, capital = slot.order("hold"), slot.capital("split")
    price = Fraction(hold["notional"]) / hold["quantity"]
    quantity = int(capital // (price * (1 + slot.fee)))
    ending = quantity * _close_price(slot) + capital - quantity * price * (1 + slot.fee)
    return {"quantity": quantity, "ending": ending, "return": ending / capital - 1}


def _reported(share) -> tuple:
    return Fraction(share["ending_equity_krw"]), Fraction(share["cumulative_return"])


def _split_matches(slot: Slot, extras) -> bool:
    expected, extra = expected_split(slot), extras[slot.path]
    core = Fraction(extra["core_share"][slot.cost]["ending_equity_krw"])
    reported = _reported(extra["split_share"][slot.cost]["split"])
    return core == expected["core_ending"] and reported == (expected["ending"], expected["return"])


def _hold_matches(slot: Slot, extras) -> bool:
    expected = expected_same_capital_hold(slot)
    reported = _reported(extras[slot.path]["split_share"][slot.cost]["hold_same_capital"])
    return reported == (expected["ending"], expected["return"])


def _ending_matches(slot: Slot, extras) -> bool:
    extra = extras[slot.path]
    core = Fraction(extra["core_share"][slot.cost]["ending_equity_krw"])
    return core + _reported(extra["split_share"][slot.cost]["split"])[0] == slot.ending("strategy")


def event_counts(run) -> dict:
    """체결 건수의 독립 기대값: 장부의 현금 사건에서 센다. 매수 − 매도는 남은 차수 수와 같아야 한다."""
    events = [row for row in run.result.cash_events if row["kind"] == "trade"]
    count = {name: sum(row["side"] == name for row in events) for name in ("BUY", "SELL")}
    if count["BUY"] - count["SELL"] != len(run.result.positions):
        raise ValueError("trade events do not explain the remaining positions")
    reasons = [row["reason"] for row in events]
    return {"buys": count["BUY"], "sells": count["SELL"], "reentries": reasons.count("monthly_reentry"),
            "first": reasons.count("manual_first_entry")}


def _outcomes(slot: Slot) -> list:
    return [row["outcome"] for row in slot.row["reentries"][slot.cost]]


def _counts_match(slot: Slot, measured) -> bool:
    mode = measured.mine["mode"]
    counted = event_counts(measured.ledgers.runs[slot.cost][slot.path][mode])
    reported = measured.mine["exploration"][slot.path]["split_ledger"][slot.cost]
    return ((reported["buys"], reported["sells"]) == (counted["buys"], counted["sells"])
            and counted["reentries"] == _outcomes(slot).count("filled")
            and counted["first"] == (slot.order("split")["status"] == "filled"))


def _reentries_match(slot: Slot, extras) -> bool:
    counts, outcomes = extras[slot.path]["reentry_counts"][slot.cost], _outcomes(slot)
    return (counts["attempts"], counts["filled"]) == (len(outcomes), outcomes.count("filled"))


def _holds(check, slots) -> bool:
    """수치가 있는 자리 전부에서 성립해야 한다. 자리가 하나도 없거나 탐색 출력에 값이 없으면 실패다."""
    try:
        return bool(slots) and all(check(slot) for slot in slots)
    except FAULTS:
        return False


def _applied_matches(ctx: Context, mine) -> bool:
    """실제 적용 식별값: 본문의 해시 = 고정 설정에서 재진입 방식만 바꿔 따로 만든 값 = 등록값."""
    independent = ctx.base.with_changes(modes=(ctx.spec.mode,)).applied_sha256()
    return mine["applied_config_sha256"] == independent == ctx.spec.approved["baseline_applied_config_sha256"]


def _new_verdict(ctx: Context, mine) -> dict:
    """탐색 출력에 새 판정을 적용한 결과."""
    labels = segment_labels(ctx.spec.segments, ctx.spec.primary_start, ctx.base.end)
    frame = reading.Frame(tuple(ctx.base.paths), labels, ctx.base.thresholds)
    return judge.setting_verdict(reading.point_input(mine, frame), ctx.spec.thresholds)


def _verdict_checks(ctx: Context, formal_paths, mine) -> dict:
    """탐색 출력의 새 판정을 정식 기록만으로 구한 기대값과 맞춘다: 설정의 상태·공통 구간, 그리고 경로별 상태와
    경로별·비용별 조건 각각. 정식 수치가 유효한 자리를 탐색 쪽 사유가 비우면 최종 상태가 같아도 어긋난다."""
    try:
        verdict, thresholds = _new_verdict(ctx, mine), ctx.spec.thresholds
        found = {path: {"status": row["status"], "checks": row["checks"]} for path, row in verdict["paths"].items()}
        whole = (verdict["status"], verdict["common_segments"]) == expected_verdict(formal_paths, thresholds)
        each = found == expected_paths(formal_paths, thresholds)
    except FAULTS:
        whole = each = False
    return {"verdict_equals_formal_only_expectation": whole,
            "path_cost_checks_equal_formal_only_expectation": each}


def _record_passes(ctx: Context, mine) -> bool:
    """기준점 결과를 1단계의 그 설정 단위로 놓았을 때 재개·조립의 기록 검사를 그대로 통과하는지."""
    spec = ctx.spec
    seat = Seat(ctx, "stage1", primary_unit(spec, spec.baseline_number()))
    record = {"schema": UNIT_SCHEMA, "exploration": True, "formal_run": False, "identity": seat.identity(),
              "setting": spec.setting(seat.unit.number).describe(), "result": mine}
    try:
        check_record(seat, json.loads(dump(record)))
    except AssemblyError:
        return False
    return True


def self_checks(ctx: Context, formal_paths, measured) -> dict:
    """새 판정·새 지표의 검산. 값은 싣지 않고 성립 여부만 남긴다."""
    extras, slots = measured.mine["exploration"] or {}, valued_slots(formal_paths, measured.config)
    return {**_verdict_checks(ctx, formal_paths, measured.mine),
            "baseline_unit_passes_record_checks": _record_passes(ctx, measured.mine),
            "applied_config_equals_registered": _applied_matches(ctx, measured.mine),
            "strategy_ending_equals_core_plus_split": _holds(lambda slot: _ending_matches(slot, extras), slots),
            "split_share_equals_formal_arithmetic": _holds(lambda slot: _split_matches(slot, extras), slots),
            "same_capital_hold_equals_formal_arithmetic": _holds(lambda slot: _hold_matches(slot, extras), slots),
            "trade_counts_equal_ledger_events": _holds(lambda slot: _counts_match(slot, measured), slots),
            "reentry_counts_equal_formal_lists": _holds(lambda slot: _reentries_match(slot, extras), slots),
            "matched_rerun_equal": measured.rerun_equal}


# ---------------------------------------------------------------- 실행·기록

@dataclass
class Measured:
    """기준점 실행: 적용 설정, 연 장부, 결과(직렬화한 꼴), 시간·메모리, 비중 맞춤 재실행 일치 여부."""

    config: object
    ledgers: PointLedgers
    mine: dict
    timing: dict
    rerun_equal: bool


def _matched_rerun(ledgers: PointLedgers, result) -> bool:
    """비중 맞춤 계좌(경로마다 하나)를 따로 한 번 더 실행해 조립 결과와 같은 값인지 확인한다(재현성 확인)."""
    job = ledgers.job
    with localcontext(prec=80):
        return all(study._matched(job.data, job.config, path, row["base"]) == row.get("matched_baseline")
                   for path, row in result["run_start"]["items"][job.mode]["paths"].items()
                   if row["base"] is not None)


def estimate(seconds, settings, cpus=None) -> dict:
    """기준점 한 설정의 경과시간으로 잡은 조건부 추정과 권장 동시 실행 수. 상한이 아니다."""
    cpus = cpus or os.cpu_count() or 1
    workers = max(1, min(settings, cpus - 2 if cpus > 3 else cpus - 1))
    hours = math.ceil(settings / workers) * seconds / 3600
    return {"per_setting_seconds": seconds, "settings": settings, "cpus": cpus, "recommended_workers": workers,
            "conditional_hours": hours, "report_threshold_hours": REPORT_HOURS,
            "exceeds_report_threshold": hours > REPORT_HOURS, "conditions": CONDITIONS}


def _timing(bench: Bench, seconds) -> dict:
    return {"total_seconds": seconds, "by_kind_seconds": dict(bench.clock.seconds),
            "by_kind_counts": dict(bench.clock.counts),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def _measure(ctx: Context) -> Measured:
    """기준점 설정을 한 프로세스에서 순차 실행하며 장부 종류별 시간을 잰다."""
    spec, bench, started = ctx.spec, Bench(), time.perf_counter()
    config = setting_config(ctx.base, spec, spec.setting(spec.baseline_number()))
    with bench.clock("load_inputs"):
        inputs = ctx.source.load(spec.symbol, spec.primary_start)
    check_loaded(ctx, inputs)
    ledgers = point_ledgers(inputs, config, spec.primary_start, bench=bench)
    result = assemble_point(ledgers)
    seconds = time.perf_counter() - started
    with bench.clock("matched_rerun"):  # 총 소요에는 넣지 않는다
        rerun_equal = _matched_rerun(ledgers, result)
    return Measured(config, ledgers, json.loads(dump(result)), _timing(bench, seconds), rerun_equal)


def _sealed_measure(ctx: Context, folder: Path) -> Measured:
    """기준점 실행 중의 출력과 예외 전문은 봉인 로그에만 남기고, 공개 경로에는 기술 코드만 낸다."""
    with captured_output(folder / "sealed" / "output.log") as sink:
        try:
            return _measure(ctx)
        except Exception as error:
            traceback.print_exc(file=sink)
            code = error_code(error)
    raise BaselineError(f"baseline run ended with a technical error: {code}")


def _selftest_seal(ctx: Context, out):
    """실제 묶음의 기준점은 같은 판본의 합성 시험 통과 기록이 있어야 잰다. 합성 입력은 허가 대상이 아니다."""
    permit.check_current(ctx)
    if not ctx.registered:
        check_synthetic(ctx)
        return None
    check_frozen(ctx)
    return permit.check_selftest(out)["seal"]


def _record(ctx: Context, measured: Measured, evidence) -> dict:
    """공개 대조 기록. 본실행 허용은 실제 묶음에서 불일치 0건·필수 검산 전부 참일 때만이다."""
    spec, comparison, checks = ctx.spec, evidence["comparison"], evidence["self_checks"]
    allowed = (ctx.registered and comparison["mismatches"] == []
               and all(checks.get(name) is True for name in permit.REQUIRED_CHECKS))
    target = {**permit.expected_baseline(ctx), "applied_config_sha256": measured.config.applied_sha256()}
    return with_seal({"schema": permit.BASELINE_SCHEMA, "exploration": True, "formal_run": False,
                      "identity": ctx.identity, "baseline": target, **evidence,
                      "timing": measured.timing, "main_run_allowed": allowed,
                      "estimate": estimate(measured.timing["total_seconds"], len(spec.settings()))})


def _evidence(ctx: Context, formal, measured: Measured) -> tuple:
    """(공개할 증거, 봉인할 불일치 내역, 봉인할 단위 본문)."""
    mode, unit_text = ctx.spec.mode, dump(measured.mine)
    comparison, details = compare(formal, measured.mine["run_start"], mode)
    checks = self_checks(ctx, formal["items"][mode]["paths"], measured)
    public = {"comparison": comparison, "self_checks": checks, "sealed_unit_sha256": digest(unit_text)}
    return public, details, unit_text


def _write(folder: Path, record, sealed_texts) -> None:
    """봉인 원본(단위 본문·불일치 내역)과 공개 기록을 한 번만 쓴다."""
    for name, text in sealed_texts.items():
        write_once(folder / "sealed" / name, text, mode=SEALED)
    write_once(folder / "compare.json", dump(record))
    write_once(folder / "report.md", markdown(json.loads(dump(record))))


def run_baseline(ctx: Context, formal_path, out) -> dict:
    """시험 통과 기록 → 정식 결과 해시 → 기준점 실행 → 대조와 독립 검산. 결과는 out/baseline/에 한 번만 쓴다."""
    folder, seal = Path(out) / "baseline", _selftest_seal(ctx, out)
    with writer_lock(out):
        if (folder / "compare.json").exists():
            raise OverwriteError("a baseline comparison is already recorded here; use a new output folder")
        formal = formal_start(ctx.spec, formal_path)
        measured = _sealed_measure(ctx, folder)
        public, details, unit_text = _evidence(ctx, formal, measured)
        source = {"path": str(formal_path), "sha256": ctx.spec.approved["formal_result_sha256"]}
        record = _record(ctx, measured, {**public, "selftest_seal": seal, "formal_result": source})
        _write(folder, record, {"unit.json": unit_text, "mismatches.json": dump(details)})
    return record


def markdown(record) -> str:
    comparison, timing, plan = record["comparison"], record["timing"], record["estimate"]
    lines = ["# 공개 기준점 대조 (탐색 실행기 대 정식 결과)", "",
             f"- 본실행 허용: **{'예' if record['main_run_allowed'] else '아니오 — 본실행 금지'}**",
             f"- 일치한 잎 {comparison['matched']}개(정식 기록의 잎 {comparison['formal_leaves']}개, 문자열까지 같은 잎 "
             f"{comparison['string_identical']}개), 불일치 {len(comparison['mismatches'])}개, "
             f"값은 같고 표기만 다른 잎 {len(comparison['notation_only'])}개",
             f"- 항목별 일치 수: {comparison['by_item']}",
             f"- 검산(값은 싣지 않음): {record['self_checks']}",
             f"- 총 소요 {timing['total_seconds']:.1f}초, 최대 메모리 {timing['max_rss_mb']:.0f} MB",
             f"- 장부 종류별 시간(초): {timing['by_kind_seconds']} / 횟수: {timing['by_kind_counts']}",
             f"- {plan['settings']}개 조건부 추정: 동시 {plan['recommended_workers']}개에서 약 "
             f"{plan['conditional_hours']:.1f}시간, 24시간 초과 예상: {plan['exceeds_report_threshold']}",
             f"- 추정의 조건: {plan['conditions']}",
             f"- 정식 결과 SHA-256: `{record['formal_result']['sha256']}`",
             f"- 기준점 적용 설정 해시: `{record['baseline']['applied_config_sha256']}`", ""]
    lines += [f"- 불일치 칸: {row['where']} ({row['kind']})" for row in comparison["mismatches"]]
    lines += ["", "## 실행 식별값", ""]
    lines += [f"- {key}: `{value}`" for key, value in record["identity"].items() if key != "inputs"]
    return "\n".join(lines + ["", "## 입력 대조", "", *_input_lines(record["identity"]["inputs"])]) + "\n"


def _input_lines(inputs) -> list:
    """등록값과 실제값(8항목), 종목별 가격 기준."""
    actual, registered = inputs.get("actual", {}), inputs.get("registered", {})
    lines = [f"- {key}: 실제 `{value}` / 등록 `{registered.get(key)}` / 일치 {value == registered.get(key)}"
             for key, value in actual.items()]
    lines.append(f"- 등록 입력 그대로: {inputs.get('registered_inputs_frozen')}")
    return lines + [f"- 가격 기준 {symbol}: `{basis}`"
                    for symbol, basis in inputs.get("price_basis_by_symbol", {}).items()]
