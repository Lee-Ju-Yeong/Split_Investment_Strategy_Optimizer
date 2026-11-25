"""봉인 단위 기록의 검사. 재개와 조립이 같은 검사를 쓴다.

머리(식별값)와 설정이 그 자리의 기대값과 같은지, 본문의 종목·시작·종료·평가 시각·재진입 방식·입력 출처·명목 한도·
원금·적용 설정이 머리와 맞는지, 두 경로 × 두 비용 네 자리가 모두 있고 빈칸마다 그 비용의 사유 또는 공통 사유가
있는지, 수치 행의 구간 이름·순서·개수가 등록한 구간과 같은지 본다. 판정은 하지 않는다(기술적 무결성 검사).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
import json

from ..kodex_core.calendar import EVALUATION_KST, KST
from .grid import segment_labels, setting_config
from .identity import PROVENANCE_SAME, Context
from .judge import CAPACITY_REASON, reasons_for
from .reading import Frame
from .runner import COSTS
from .sealing import dump


UNIT_SCHEMA = "kodex-grid-unit-v2"
GAP_REASON = "input_data_gap_after_start"
KODEX_KINDS = ("primary", "reconfirm")
EXTENSION = "extension"
UNOPENED_FLAGS = {"failures": [], "holds": [CAPACITY_REASON], "later": []}  # 명목 한도로 장부를 열지 않은 행
MALFORMED = (KeyError, TypeError, AttributeError, ValueError, ArithmeticError)


class AssemblyError(RuntimeError):
    """누락·여분·중복·식별 불일치·빈 자리·구간 불일치. 조립하지 않고 멈춘다."""


@dataclass(frozen=True)
class Unit:
    kind: str  # primary / reconfirm / extension
    number: int
    symbol: str
    start: date

    @property
    def name(self) -> str:
        prefix = "" if self.kind == "primary" else f"{self.kind}-{self.symbol}-{self.start.isoformat()}-"
        return f"{prefix}setting-{self.number:02d}"


@dataclass(frozen=True)
class Seat:
    """한 단위가 놓일 자리: 어느 실행의 어느 단계·단위인지와, 단계가 덧붙이는 식별값(binding)."""

    ctx: Context
    stage: str
    unit: Unit
    binding: dict = field(default_factory=dict)

    def config(self):
        return setting_config(self.ctx.base, self.ctx.spec, self.ctx.spec.setting(self.unit.number))

    def identity(self) -> dict:
        """이 자리의 단위가 가져야 할 식별값. 적용 설정 해시는 설정 번호로 다시 만든 기대값이다."""
        unit, spec = self.unit, self.ctx.spec
        return {**self.ctx.identity, **self.binding, "stage": self.stage, "unit": unit.name, "kind": unit.kind,
                "number": unit.number, "symbol": unit.symbol, "start": unit.start.isoformat(),
                "end": self.ctx.base.end.isoformat(), "mode": spec.mode,
                "applied_config_sha256": self.config().applied_sha256()}

    def nominal_floor(self):
        """확장 단위에만 적용하는 시작 전 명목 한도(KODEX 주 행·재확인 행에는 없다)."""
        return self.ctx.base.nominal_floor if self.unit.kind == EXTENSION else None

    def frame(self) -> Frame:
        """등록한 경로와, 이 시작일부터 만기까지의 구간 이름. KODEX 행은 등록한 구간 전부를 가져야 한다."""
        spec, base = self.ctx.spec, self.ctx.base
        labels = segment_labels(spec.segments, self.unit.start, base.end)
        if self.unit.kind in KODEX_KINDS and len(labels) != len(spec.segments):
            raise AssemblyError(f"{self.unit.name}: a KODEX row must cover every registered segment")
        return Frame(tuple(base.paths), labels, base.thresholds)


def primary_unit(spec, number: int) -> Unit:
    """주 시작일의 한 설정."""
    return Unit("primary", number, spec.symbol, spec.primary_start)


def _fail(seat: Seat, message: str):
    raise AssemblyError(f"{seat.unit.name}: {message}")


def _same(left, right) -> bool:
    """값과 자료형이 모두 같은지(0과 False, 1과 True를 같게 보지 않는다)."""
    return type(left) is type(right) and left == right


def _check_head(seat: Seat, record) -> None:
    expected = json.loads(dump(seat.identity()))
    if (record.get("schema"), record.get("exploration"), record.get("formal_run")) != (UNIT_SCHEMA, True, False):
        _fail(seat, "not an exploration unit record")
    actual = record.get("identity") or {}
    if actual != expected:
        differing = sorted(key for key in set(expected) | set(actual) if actual.get(key) != expected.get(key))
        _fail(seat, f"identity differs in {differing}")
    if record.get("setting") != seat.ctx.spec.setting(seat.unit.number).describe():
        _fail(seat, "setting parameters differ from the grid")


def _check_body(seat: Seat, result) -> None:
    """본문이 머리와 같은 재진입 방식·적용 설정으로 계산됐는지, 보류와 결과가 함께 있지 않은지."""
    if result.get("mode") != seat.ctx.spec.mode:
        _fail(seat, "result is not the registered reentry mode")
    if result.get("applied_config_sha256") != seat.config().applied_sha256():
        _fail(seat, "body was computed with another applied configuration than its identity")
    held, opened = bool(result.get("held_reason")), result.get("run_start") is not None
    if held and opened:
        _fail(seat, "hold reason and result are both present")
    if not held and not opened:
        _fail(seat, "no result and no hold reason")


def _check_held(seat: Seat, result) -> None:
    """입력 보류: 두 경로 × 두 비용 네 자리에 빈 수치와 같은 사유가 있어야 한다."""
    reason = result["held_reason"]
    expected = {path: {cost: {"values": None, "reason": reason} for cost in COSTS} for path in seat.ctx.base.paths}
    if result.get("held_slots") != expected or result.get("exploration") is not None:
        _fail(seat, "a held input must record four empty slots with its reason and nothing else")


def promised_provenance(seat: Seat) -> dict:
    """머리가 본문의 입력 출처에 약속한 값. 묶음 입력기를 거친 실행(등록 단위)은 열 항목이 모두 있어야 한다.
    묶음 출처가 없는 합성 입력의 머리는 다섯 묶음 항목이 본문에도 없어야 한다는 것만 약속한다."""
    identity = seat.ctx.identity
    inputs = identity["inputs"]
    actual = inputs.get("actual", {})
    same = {key: actual.get(key) for key in PROVENANCE_SAME}
    if None in same.values():
        if seat.ctx.registered:
            _fail(seat, "a registered unit needs the bundle provenance in its identity")
        return same
    return {**same, "market_rules_sha256": identity["market_rules_sha256"],
            "source_sha256": identity["approved_source_sha256"],
            "price_basis": inputs["price_basis_by_symbol"][seat.unit.symbol],
            "manifest_synthetic": actual["synthetic"], "extra_inputs_present": actual["extra_inputs"] != "none",
            "extra_inputs_listed": actual["extra_inputs"] != "none"}


def stated_provenance(opened) -> dict:
    """본문이 적은 입력 출처. 추가 입력 여부는 extra_inputs.present에서, 가격 기준은 basis 칸에서도 읽는다."""
    provenance = opened["input_provenance"]
    extra = provenance.get("extra_inputs") if isinstance(provenance.get("extra_inputs"), dict) else {}
    listed = None if not extra else bool(extra.get("locked_status") or extra.get("distributions"))
    return {**provenance, "extra_inputs_present": extra.get("present"), "extra_inputs_listed": listed,
            "basis_price_basis": opened["basis"].get("price_basis")}


def _check_provenance(seat: Seat, opened) -> None:
    """필수 출처 항목이 본문에 실제로 있고(없으면 불일치) 머리와 값·자료형이 같은지."""
    promised, stated = promised_provenance(seat), stated_provenance(opened)
    approved = seat.ctx.identity["approved_source_sha256"]
    wrong = [key for key, value in promised.items() if not _same(stated.get(key), value)]
    wrong += ["source_sha256"] if stated.get("source_sha256", approved) != approved else []
    if "price_basis" in promised and stated["basis_price_basis"] != promised["price_basis"]:
        wrong.append("basis.price_basis")
    if wrong:
        _fail(seat, f"input provenance differs from the run identity in {sorted(set(wrong))}")


def _check_origin(seat: Seat, opened) -> None:
    """본문의 종목·시작일·재진입 방식·입력 출처가 머리와 같은지."""
    unit = seat.unit
    if (opened["symbol"], opened["start"]) != (unit.symbol, unit.start.isoformat()):
        _fail(seat, "result belongs to another symbol or start")
    if list(opened["items"]) != [seat.ctx.spec.mode]:
        _fail(seat, "result is not the registered reentry mode")
    _check_provenance(seat, opened)


def _unopened(result) -> bool:
    """확장 명목 한도로 장부를 열지 않은 행(시작 시 비교 보류)."""
    return result["run_start"]["capacity"]["ok"] is not True


def _check_capacity(seat: Seat, result) -> None:
    """명목 한도 칸이 그 단위의 등록 한도(확장만 있음)와 같고, 충족 여부가 한도·문턱과 맞는지."""
    capacity, floor = result["run_start"]["capacity"], seat.nominal_floor()
    limit, stated = capacity["limit_krw"], capacity["floor_krw"]
    ok = floor is None or (limit is not None and Decimal(limit) >= floor)
    if (None if floor is None else str(floor)) != stated or capacity["ok"] is not ok:
        _fail(seat, "capacity differs from the registered nominal floor of this unit")
    flagged = [path for path, extra in result["exploration"].items()
               if CAPACITY_REASON in [reason for values in extra["flags"].values() for reason in values]]
    if ok and flagged:
        _fail(seat, "an opened row carries the nominal capacity hold")


def _check_unopened(seat: Seat, result) -> None:
    """장부를 열지 않은 행은 시작 시 비교 보류를 그대로 보존한다: 모든 경로에 그 사유 하나, 네 자리는 빈칸."""
    for path, row in _path_rows(seat, result).items():
        extra = result["exploration"][path]
        observed = [extra[name] for name in ("split_ledger", "reentry_counts", "core_share", "split_share")]
        if extra["flags"] != UNOPENED_FLAGS or any(row[cost] is not None for cost in COSTS) or any(observed):
            _fail(seat, f"{path}: a row held by the nominal capacity must keep only that start hold")


def _evaluated_at(text, day: str) -> bool:
    """평가 시각이 그 날짜의 18시 한국시각인지(날짜·시각·시간대를 모두 본다)."""
    try:
        at = datetime.fromisoformat(text)
        expected = datetime.combine(date.fromisoformat(day), EVALUATION_KST, KST)
    except (TypeError, ValueError):
        return False
    same_zone = at.tzinfo is not None and at.utcoffset() == expected.utcoffset()
    return same_zone and at.replace(tzinfo=None) == expected.replace(tzinfo=None)


def _justified_stop(seat: Seat, result) -> bool:
    """정당한 조기 종료: 요청 만기가 등록 만기이고, 마지막 유효일(= 본문의 종료일)과 그 뒤의 첫 결측일·중단 사유가
    있다. 개설한 장부에는 모든 경로에 결측 사유가 있어야 하고, 열지 않은 행에는 그 표시를 요구하지 않는다."""
    opened, end = result["run_start"], seat.ctx.base.end.isoformat()
    gap = opened["data_end"]
    marked = _unopened(result) or all(GAP_REASON in extra["flags"]["later"]
                                      for extra in result["exploration"].values())
    return (gap.get("requested_end") == end and gap.get("last_valid_session") == opened["end"]
            and opened["end"] < str(gap.get("first_bad_session")) <= end and bool(gap.get("reason")) and marked)


def _check_end(seat: Seat, result) -> None:
    """정상 만기는 등록일, 조기 종료는 마지막 유효일. 마지막 평가는 그 날의 18시 한국시각이어야 한다."""
    opened, end = result["run_start"], seat.ctx.base.end.isoformat()
    proper = opened["end"] == end if opened["data_end"] is None else _justified_stop(seat, result)
    if not proper:
        _fail(seat, "end of the run is neither the registered end nor a justified early stop")
    if not _evaluated_at(opened["last_evaluation"], opened["end"]):
        _fail(seat, "last evaluation is not 18:00 KST of the last session")


def _path_rows(seat: Seat, result) -> dict:
    rows, extras = result["run_start"]["items"][seat.ctx.spec.mode]["paths"], result["exploration"]
    if list(rows) != list(seat.ctx.base.paths) or list(extras) != list(seat.ctx.base.paths):
        _fail(seat, "both intraday paths are required")
    return rows


def slot_problems(rows, extras) -> list:
    """네 자리 가운데 없는 자리와, 빈칸인데 그 비용의 사유도 공통 사유도 없는 자리(0으로 채우지 않는다)."""
    problems = []
    for path, row in rows.items():
        every = [reason for values in extras[path]["flags"].values() for reason in values]
        for cost in COSTS:
            if cost not in row or cost not in row["start_orders"]:
                problems.append(f"{path}/{cost}: slot is missing")
            elif row[cost] is None and not reasons_for(every, cost):
                problems.append(f"{path}/{cost}: empty slot without a reason for that cost")
    return problems


def _check_slots(seat: Seat, result) -> None:
    problems = slot_problems(_path_rows(seat, result), result["exploration"])
    if problems:
        _fail(seat, "; ".join(problems))


def _slot_metrics(row, cost) -> list:
    """한 자리에서 구간 지표를 가진 계좌들: 전략·단독 보유, 보유 단독 기록, (기본 비용) 비중 맞춤 계좌."""
    found = [] if row[cost] is None else [row[cost]["strategy"], row[cost]["hold"]]
    hold_only = (row.get("hold_metrics") or {}).get(cost)
    matched = row.get("matched_baseline") if cost == COSTS[0] else None
    return found + [metrics for metrics in (hold_only, matched and matched["metrics"]) if metrics]


def _check_segments(seat: Seat, result) -> None:
    """수치 행의 구간 이름·순서·개수가 등록한 구간(이 시작일 이후)과 같아야 한다."""
    labels = list(seat.frame().labels)
    for path, row in _path_rows(seat, result).items():
        for cost in COSTS:
            names = [[part["segment"] for part in metrics["segments"]] for metrics in _slot_metrics(row, cost)]
            if any(found != labels for found in names):
                _fail(seat, f"{path}/{cost}: segments differ from the registered {labels}")


def _capitals(extra, cost) -> list:
    """한 경로·한 비용에서 본문이 되적은 원금: (적힌 값, 원금 이름)."""
    core, share = (extra["core_share"] or {}).get(cost), (extra["split_share"] or {}).get(cost)
    stated = [] if core is None else [(core["capital_krw"], "core_krw")]
    return stated + ([] if share is None else [(share[name]["capital_krw"], "split_krw")
                                                for name in ("split", "hold_same_capital")])


def _check_capitals(seat: Seat, result) -> None:
    """코어 몫·분할 몫·같은 원금 보유에 적힌 원금이 그 설정의 코어·분할 원금과 같은지."""
    setting = seat.ctx.spec.setting(seat.unit.number).describe()
    for path, extra in result["exploration"].items():
        pairs = [pair for cost in COSTS for pair in _capitals(extra, cost)]
        if any(Decimal(value) != Decimal(setting[name]) for value, name in pairs):
            _fail(seat, f"{path}: stated capital differs from the setting")


def _check_opened(seat: Seat, result) -> None:
    if result.get("held_slots") is not None:
        _fail(seat, "an opened result must not carry held slots")
    _check_origin(seat, result["run_start"])
    _check_capacity(seat, result)
    if _unopened(result):
        _check_unopened(seat, result)
    _check_end(seat, result)
    _check_slots(seat, result)
    _check_segments(seat, result)
    _check_capitals(seat, result)


def check_record(seat: Seat, record) -> None:
    """머리 → 본문과 머리의 대조 → 네 자리와 구간. 꼴이 깨진 기록도 조립 오류로 멈춘다."""
    try:
        _check_head(seat, record)
        result = record["result"]
        _check_body(seat, result)
        (_check_held if result.get("held_reason") else _check_opened)(seat, result)
    except MALFORMED as error:
        raise AssemblyError(f"{seat.unit.name}: malformed unit record ({type(error).__name__})") from error
