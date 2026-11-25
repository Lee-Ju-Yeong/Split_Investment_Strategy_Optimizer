"""봉인 단위 기록의 검사. 재개와 조립이 같은 검사를 쓴다(판정은 하지 않는다: 기술적 무결성 검사).

머리(식별값)와 설정이 그 자리의 기대값과 같은지, 본문의 상품·시작·종료·평가 시각·재진입 방식·비용·원금·입력 출처·명목 한도가
머리와 맞는지, 두 경로가 모두 있고 비교할 수 없는 경로마다 사유가 있는지, 비교할 수 있는 경로의 지표·구간·일별 계열이
서로 맞는지(길이, 마지막 순자산, 순자산 = 현금 + 미수금 + ETF 평가액) 본다.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import json

from ..kodex_core import study
from ..kodex_grid import records as grid_records
from ..kodex_grid.grid import segment_labels
from ..kodex_grid.records import AssemblyError
from ..kodex_grid.sealing import dump
from .definition import Unit, unit_config
from .identity import MixContext
from .unit import GAP_REASON, REASON_KEYS


UNIT_SCHEMA = "markets-mix-unit-v1"
STAGE = "run"
ACCOUNTS = ("strategy", "hold")
UNOPENED = {"failures": [], "holds": [study.CAPACITY_HOLD], "later": []}
MALFORMED = (KeyError, TypeError, AttributeError, ValueError, ArithmeticError)


@dataclass(frozen=True)
class Seat:
    """한 단위가 놓일 자리: 어느 실행의 어느 단위인지."""

    ctx: MixContext
    unit: Unit

    def config(self):
        return unit_config(self.ctx.mix, self.unit, base=self.ctx.base, grid=self.ctx.spec)

    def floor(self):
        return self.ctx.mix.floor(self.unit)

    def setting(self) -> dict:
        """격자의 설정에 이 계좌가 실제로 쓰는 원금을 적는다."""
        capital = self.config().capital
        return {**self.ctx.spec.setting(self.unit.number).describe(), "core_krw": str(capital["core"]),
                "split_krw": str(capital["split"]), "account_krw": str(capital["hold"])}

    def identity(self) -> dict:
        """이 자리의 단위가 가져야 할 식별값. 적용 설정 해시는 설정 번호와 원금으로 다시 만든 기대값이다."""
        unit, floor = self.unit, self.floor()
        return {**self.ctx.identity, "stage": STAGE, "unit": unit.name, "kind": unit.kind, "number": unit.number,
                "symbol": unit.symbol, "start": unit.start.isoformat(), "end": self.ctx.base.end.isoformat(),
                "mode": self.ctx.spec.mode, "cost": unit.cost, "cost_rate": str(self.config().cost(unit.cost)),
                "nominal_floor_krw": None if floor is None else str(floor),
                "applied_config_sha256": self.config().applied_sha256()}

    def labels(self) -> list:
        """이 시작일부터 만기까지의 구간 이름(고정 설정의 이름)."""
        return list(segment_labels(self.ctx.base.segments, self.unit.start, self.ctx.base.end))


def _fail(seat: Seat, message: str):
    raise AssemblyError(f"{seat.unit.name}: {message}")


def _check_head(seat: Seat, record) -> None:
    expected = json.loads(dump(seat.identity()))
    if (record.get("schema"), record.get("exploration"), record.get("formal_run")) != (UNIT_SCHEMA, True, False):
        _fail(seat, "not an exploration unit record")
    actual = record.get("identity") or {}
    if actual != expected:
        differing = sorted(key for key in set(expected) | set(actual) if actual.get(key) != expected.get(key))
        _fail(seat, f"identity differs in {differing}")
    if record.get("setting") != seat.setting():
        _fail(seat, "setting parameters or capital differ from the plan")


def _check_body(seat: Seat, result) -> None:
    """본문이 머리와 같은 재진입 방식·비용·원금·적용 설정으로 계산됐는지, 보류와 결과가 함께 있지 않은지."""
    config = seat.config()
    stated = (result.get("mode"), result.get("cost"), result.get("cost_rate"), result.get("applied_config_sha256"))
    if stated != (seat.ctx.spec.mode, seat.unit.cost, str(config.cost(seat.unit.cost)), config.applied_sha256()):
        _fail(seat, "body was computed with another mode, cost or applied configuration than its identity")
    if result.get("capital") != {name: str(value) for name, value in config.capital.items()}:
        _fail(seat, "stated capital differs from the plan")
    held, opened = bool(result.get("held_reason")), result.get("head") is not None
    if held == opened or (held and result.get("paths") is not None):
        _fail(seat, "a unit is either a held input (reason only) or an opened result")


def _check_origin(seat: Seat, head) -> None:
    """본문의 상품·시작일·입력 출처가 머리와 같은지(출처 검사는 kodex_grid의 것을 그대로 쓴다)."""
    if (head["symbol"], head["start"]) != (seat.unit.symbol, seat.unit.start.isoformat()):
        _fail(seat, "result belongs to another symbol or start")
    grid_records._check_provenance(seat, head)


def _check_capacity(seat: Seat, result) -> None:
    """명목 한도 칸이 그 단위의 한도(기준 상품에는 없음)와 같고 충족 여부가 맞는지. 열지 않은 단위는 그 보류만 가진다."""
    capacity, floor = result["head"]["capacity"], seat.floor()
    limit = capacity["limit_krw"]
    ok = floor is None or (limit is not None and Decimal(limit) >= floor)
    if (None if floor is None else str(floor)) != capacity["floor_krw"] or capacity["ok"] is not ok:
        _fail(seat, "capacity differs from the nominal floor of this unit")
    unopened = [path for path, row in result["paths"].items() if row["reasons"] == UNOPENED]
    if (not ok and len(unopened) != len(result["paths"])) or (ok and unopened):
        _fail(seat, "the nominal capacity hold must be on every path of an unopened unit and on no other")


def _justified_stop(seat: Seat, result) -> bool:
    """정당한 조기 종료: 요청 만기가 등록 만기이고, 마지막 유효일·그 뒤의 첫 결측일·중단 사유가 있고, 개설한 경로마다
    결측 사유가 있다."""
    head, end = result["head"], seat.ctx.base.end.isoformat()
    gap = head["data_end"]
    marked = all(row["reasons"] == UNOPENED or GAP_REASON in row["reasons"]["later"]
                 for row in result["paths"].values())
    return (gap.get("requested_end") == end and gap.get("last_valid_session") == head["end"]
            and head["end"] < str(gap.get("first_bad_session")) <= end and bool(gap.get("reason")) and marked)


def _check_end(seat: Seat, result) -> None:
    head, end = result["head"], seat.ctx.base.end.isoformat()
    proper = head["end"] == end if head["data_end"] is None else _justified_stop(seat, result)
    if not proper:
        _fail(seat, "end of the run is neither the registered end nor a justified early stop")
    if not grid_records._evaluated_at(head["last_evaluation"], head["end"]):
        _fail(seat, "last evaluation is not 18:00 KST of the last session")


def _block_problem(at, block, metrics):
    """일별 계열이 지표와 맞는지: 길이, 마지막 순자산, 매일 순자산 = 현금 + 미수금 + ETF 평가액."""
    columns = [block[name] for name in ("equity", "cash", "receivables", "holdings")]
    if any(len(column) != len(at) for column in columns) or len(at) != metrics["evaluation_days"]:
        return "series length differs from the evaluation days"
    if Decimal(columns[0][-1]) != Decimal(metrics["ending_equity_krw"]):
        return "last equity differs from the ending equity"
    with localcontext(prec=80):
        broken = any(Decimal(e) != Decimal(c) + Decimal(r) + Decimal(h) for e, c, r, h in zip(*columns))
    return "equity is not cash + receivables + holdings" if broken else None


def _check_valued(seat: Seat, path, row) -> None:
    """비교할 수 있는 경로: 사유가 없고, 두 계좌의 지표·구간·계열이 모두 있고 서로 맞는다."""
    if any(row["reasons"][key] for key in REASON_KEYS):
        _fail(seat, f"{path}: a comparable path must not carry a reason")
    for name in ACCOUNTS:
        metrics = row[name]
        if [part["segment"] for part in metrics["segments"]] != seat.labels():
            _fail(seat, f"{path}/{name}: segments differ from the registered {seat.labels()}")
        problem = _block_problem(row["series"]["at"], row["series"][name], metrics)
        if problem:
            _fail(seat, f"{path}/{name}: {problem}")
    with localcontext(prec=80):
        if Decimal(row["delta"]["delta_cagr"]) != Decimal(row["strategy"]["cagr"]) - Decimal(row["hold"]["cagr"]):
            _fail(seat, f"{path}: stated difference is not strategy minus hold")


def _check_blank(seat: Seat, path, row) -> None:
    """비교할 수 없는 경로: 사유가 있고 전략 수치·차·전략 계열이 비어 있다(0으로 채우지 않는다)."""
    if not any(row["reasons"][key] for key in REASON_KEYS):
        _fail(seat, f"{path}: empty values without a reason")
    filled = [name for name in ("strategy", "hold", "delta") if row[name] is not None]
    if filled or (row["series"] or {}).get("strategy") is not None:
        _fail(seat, f"{path}: a path that cannot be compared must not carry strategy values")


def _check_paths(seat: Seat, result) -> None:
    rows = result["paths"]
    if list(rows) != list(seat.ctx.base.paths):
        _fail(seat, "both intraday paths are required")
    for path, row in rows.items():
        if type(row["comparable"]) is not bool or set(row["reasons"]) != set(REASON_KEYS):
            _fail(seat, f"{path}: comparable flag and three reason lists are required")
        (_check_valued if row["comparable"] else _check_blank)(seat, path, row)


def _check_opened(seat: Seat, result) -> None:
    _check_origin(seat, result["head"])
    _check_paths(seat, result)
    _check_capacity(seat, result)
    _check_end(seat, result)


def check_record(seat: Seat, record) -> None:
    """머리 → 본문과 머리의 대조 → 경로. 꼴이 깨진 기록도 조립 오류로 멈춘다."""
    try:
        _check_head(seat, record)
        result = record["result"]
        _check_body(seat, result)
        if not result.get("held_reason"):
            _check_opened(seat, result)
    except MALFORMED as error:
        raise AssemblyError(f"{seat.unit.name}: malformed unit record ({type(error).__name__})") from error
