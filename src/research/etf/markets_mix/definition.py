"""정의 파일(markets_mix_v2.json): 상품, 시작일 묶음(묶음마다 단독·혼합·민감도 시작일·구간), 원금·비용·문턱과 계획한 단위
(설계 v2 1~3절). 이 문서에 적지 않은 규칙은 v1.1을 따른다."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from ..kodex_core.config import StudyConfig, _plain
from ..kodex_grid.grid import GridSpec, segment_labels, setting_config


DEFINITION_PATH = Path(__file__).with_name("markets_mix_v2.json")
SCHEMA = "markets-mix-v2"
SOLO, SHARE = "solo", "share"
KINDS = (SOLO, SHARE)


@dataclass(frozen=True)
class Unit:
    """한 단위: (종류, 상품, 시작일, 설정 번호, 비용). 단독은 1억 원 계좌, 몫은 5천만 원 계좌다."""

    kind: str
    symbol: str
    start: date
    number: int
    cost: str

    @property
    def name(self) -> str:
        return f"{self.kind}-{self.symbol}-{self.start.isoformat()}-{self.cost}-setting-{self.number:02d}"


@dataclass(frozen=True)
class Mix:
    """혼합 계좌: 상품당 한 몫, 최초 같은 금액, 이체·재조정 없음."""

    id: str
    legs: tuple


@dataclass(frozen=True)
class Period:
    """함께 표기하는 구간. base_label은 같은 창을 가리키는 고정 설정(kodex_core)의 구간 이름이다."""

    label: str
    start: date
    end: date
    base_label: str


@dataclass(frozen=True)
class Cohort:
    """한 주 시작일의 묶음: 판정 대상(단독·혼합), 28번 민감도 시작일(+5·+20), 그 시작일에서 자른 구간, 시작일 규칙."""

    id: str
    start: date
    sensitivity_starts: tuple
    solo: tuple
    mixes: tuple
    periods: tuple
    start_rule: object

    def starts(self) -> tuple:
        return (self.start, *self.sensitivity_starts)

    def legs(self) -> tuple:
        """혼합 몫이 되는 상품(선언 순서, 중복 없음)."""
        found = []
        for row in self.mixes:
            found += [leg for leg in row.legs if leg not in found]
        return tuple(found)

    def subjects(self) -> int:
        return len(self.solo) + len(self.mixes)


@dataclass(frozen=True)
class MixSpec:
    products: tuple
    reference: str
    cohorts: tuple
    end: date
    mode: str
    sensitivity: dict
    capital: dict
    costs: dict
    thresholds: dict
    rows_expected: dict
    baseline: dict
    sentences: dict
    approved: dict
    sha256: str

    def symbols(self) -> tuple:
        return tuple(symbol for symbol, _ in self.products)

    def starts(self) -> tuple:
        """모든 묶음의 주 시작일과 민감도 시작일."""
        return tuple(start for cohort in self.cohorts for start in cohort.starts())

    def cohort_of(self, start: date) -> Cohort:
        found = [cohort for cohort in self.cohorts if start in cohort.starts()]
        if len(found) != 1:
            raise ValueError(f"{start} is not a start of exactly one cohort")
        return found[0]

    def account(self, kind: str) -> Decimal:
        return self.capital[kind]["account"]

    def floor(self, unit: Unit):
        """기준 상품(KODEX 200) 외에는 시작일의 5% 한도가 그 계좌의 원금 이상이어야 한다."""
        return None if unit.symbol == self.reference else self.account(unit.kind)

    def slots(self, cohort: Cohort) -> tuple:
        """한 묶음의 (시작일, 비용) 자리. 첫 자리가 주 판정(주 시작일·기본 비용)이고 나머지는 민감도다."""
        base, high = self.costs["primary"], self.sensitivity["cost"]
        return tuple((start, base) for start in cohort.starts()) + tuple((start, high) for start in cohort.starts())

    def applied_sha256(self) -> str:
        """실제 적용된 정의의 해시(파일 해시 필드 제외). 시험용으로 바꾼 정의를 파일과 구별한다."""
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "sha256"}
        text = json.dumps(_plain(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()


def _cohort_units(mix: MixSpec, cohort: Cohort, grid: GridSpec) -> list:
    """주 자리는 36개 설정, 민감도 자리는 한 설정(28번)만. 기준 보유(KODEX 200 1억 원)는 28번 단독 단위가 싣는다."""
    (start, cost), rest, number = mix.slots(cohort)[0], mix.slots(cohort)[1:], mix.sensitivity["number"]
    units = []
    for kind, symbols in ((SOLO, cohort.solo), (SHARE, cohort.legs())):
        for symbol in symbols:
            units += [Unit(kind, symbol, start, row.number, cost) for row in grid.settings()]
            units += [Unit(kind, symbol, day, number, high) for day, high in rest]
    if mix.reference not in cohort.solo:
        units += [Unit(SOLO, mix.reference, day, number, slot_cost) for day, slot_cost in mix.slots(cohort)]
    return units


def plan_units(mix: MixSpec, grid: GridSpec) -> tuple:
    """계획한 단위 전부(묶음 순서)."""
    return tuple(unit for cohort in mix.cohorts for unit in _cohort_units(mix, cohort, grid))


def reference_unit(mix: MixSpec, start: date, cost: str) -> Unit:
    """같은 시작일·비용의 KODEX 200 1억 원 보유를 싣는 단위(보유 장부는 분할 설정과 무관하다)."""
    return Unit(SOLO, mix.reference, start, mix.sensitivity["number"], cost)


def unit_config(mix: MixSpec, unit: Unit, *, base: StudyConfig, grid: GridSpec) -> StudyConfig:
    """kodex_grid의 그 설정에서 원금만 그 계좌의 것으로 바꾼다. 단독(1억 원)은 kodex_grid의 설정 그대로다."""
    setting = grid.setting(unit.number)
    core, split = mix.capital[unit.kind]["core"][setting.core.label]
    capital = {"core": core, "split": split, "hold": mix.account(unit.kind)}
    return setting_config(base, grid, setting).with_changes(capital=capital)


def period_labels(mix: MixSpec, base: StudyConfig, start: date) -> tuple:
    """그 시작일의 계좌가 가져야 할 (고정 설정의 구간 이름, 표기 이름). 고정 구간을 실제 시작일에서 자른 창이어야 한다."""
    periods, windows = mix.cohort_of(start).periods, {row.label: row for row in base.segments}
    for period in periods:
        fixed = windows.get(period.base_label)
        if fixed is None or max(fixed.start, start) != max(period.start, start) or fixed.end != period.end:
            raise ValueError(f"period {period.label} is not the window of {period.base_label} from {start}")
    labels = tuple(period.base_label for period in periods)
    if segment_labels(base.segments, start, mix.end) != labels:
        raise ValueError(f"an account from {start} does not cover exactly the declared periods")
    return tuple((period.base_label, period.label) for period in periods)


def _capital_problems(mix: MixSpec, grid: GridSpec) -> list:
    problems, mixes = [], [row for cohort in mix.cohorts for row in cohort.mixes]
    for kind in KINDS:
        table = mix.capital[kind]["core"]
        if set(table) != {option.label for option in grid.cores}:
            problems.append(f"{kind}: core weights differ from the grid")
        if any(core + split != mix.account(kind) for core, split in table.values()):
            problems.append(f"{kind}: core + split must equal the account")
    if any(mix.capital[SOLO]["core"].get(option.label) != (option.core, option.split) for option in grid.cores):
        problems.append("solo capital is not the kodex_grid capital")
    if mix.account(SOLO) != grid.account or any(mix.account(SHARE) * len(row.legs) != grid.account for row in mixes):
        problems.append("a mix of equal shares must add up to the solo account")
    return problems


def _cohort_problems(mix: MixSpec, grid: GridSpec) -> list:
    problems, symbols, cohorts = [], mix.symbols(), mix.cohorts
    named = [name for cohort in cohorts for name in (*cohort.solo, *(row.id for row in cohort.mixes))]
    if len({cohort.id for cohort in cohorts}) != len(cohorts) or len(set(mix.starts())) != len(mix.starts()):
        problems.append("cohort ids and starts must be unique across cohorts")
    if len(set(named)) != len(named):
        problems.append("a subject must belong to one cohort only")
    used = {symbol for cohort in cohorts for symbol in (*cohort.solo, *cohort.legs())}
    if not used <= set(symbols) or mix.reference not in symbols or len(set(symbols)) != len(symbols):
        problems.append("subjects and mix legs must be declared products, the reference product included")
    if any(len(cohort.sensitivity_starts) != len(mix.sensitivity["offsets"]) for cohort in cohorts):
        problems.append("each cohort needs one sensitivity start per declared offset")
    counts = {cohort.id: cohort.subjects() * len(grid.settings()) for cohort in cohorts}
    if mix.rows_expected != {**counts, "total": sum(counts.values())}:
        problems.append("declared row counts differ from subjects x settings per cohort")
    return problems


def _plan_problems(mix: MixSpec, grid: GridSpec, base: StudyConfig) -> list:
    problems = []
    if (mix.end, mix.mode) != (base.end, grid.mode) or mix.end != grid.end:
        problems.append("end or reentry mode differs from the grid and base configuration")
    if any(base.cost(name) != rate for name, rate in mix.costs["one_way"].items()):
        problems.append("one-way costs differ from the base configuration")
    if grid.number_of(mix.sensitivity["setting"]) != mix.sensitivity["number"]:
        problems.append("the sensitivity setting number differs from the grid numbering")
    return problems


def check_definition(mix: MixSpec, grid: GridSpec, base: StudyConfig) -> None:
    """정의가 격자·고정 설정과 같은 규칙을 가리키는지. 다르면 장부를 열기 전에 멈춘다."""
    problems = _capital_problems(mix, grid) + _cohort_problems(mix, grid) + _plan_problems(mix, grid, base)
    for start in mix.starts():
        try:
            period_labels(mix, base, start)
        except ValueError as error:
            problems.append(str(error))
    if problems:
        raise ValueError("mix definition disagrees with the grid or base configuration: " + "; ".join(problems))


def _capital(raw) -> dict:
    return {kind: {"account": Decimal(row["account"]),
                   "core": {label: tuple(Decimal(value) for value in pair) for label, pair in row["core"].items()}}
            for kind, row in raw.items()}


def _dates(values) -> tuple:
    return tuple(date.fromisoformat(day) for day in values)


def _cohort(raw) -> Cohort:
    periods = tuple(Period(label, date.fromisoformat(a), date.fromisoformat(b), fixed)
                    for label, a, b, fixed in raw["segments"])
    mixes = tuple(Mix(row["id"], tuple(row["legs"])) for row in raw["mixes"])
    rule = raw["start_rule"] if isinstance(raw["start_rule"], str) else dict(raw["start_rule"])
    return Cohort(raw["id"], date.fromisoformat(raw["start"]), _dates(raw["sensitivity_starts"]), tuple(raw["solo"]),
                  mixes, periods, rule)


def _sensitivity(raw) -> dict:
    return {"setting": dict(raw["setting"]), "number": raw["number"], "offsets": tuple(raw["offsets"]),
            "cost": raw["cost"]}


def _baseline(raw) -> dict:
    primary, extension = raw["primary"], raw["extension"]
    return {"primary": (primary["symbol"], date.fromisoformat(primary["start"])),
            "extension": tuple((symbol, date.fromisoformat(extension["start"])) for symbol in extension["symbols"]),
            "leaves": dict(raw["leaves"])}


def _checked(payload: bytes) -> dict:
    raw = json.loads(payload)
    if raw.get("schema") != SCHEMA or raw.get("exploration") is not True or raw.get("formal_run") is not False:
        raise ValueError("unexpected markets-mix definition (schema, exploration or formal flag)")
    if set(raw["capital_krw"]) != set(KINDS) or raw["costs"]["primary"] not in raw["costs"]["one_way"]:
        raise ValueError("the definition needs solo and share capital and a declared primary cost")
    return raw


def load_definition(path=DEFINITION_PATH) -> MixSpec:
    payload = Path(path).read_bytes()
    raw = _checked(payload)
    costs = {"primary": raw["costs"]["primary"],
             "one_way": {name: Decimal(rate) for name, rate in raw["costs"]["one_way"].items()}}
    return MixSpec(tuple((row["symbol"], row["name"]) for row in raw["products"]), raw["reference_symbol"],
                   tuple(_cohort(row) for row in raw["cohorts"]), date.fromisoformat(raw["end"]), raw["reentry_mode"],
                   _sensitivity(raw["sensitivity"]), _capital(raw["capital_krw"]), costs,
                   {name: Decimal(value) for name, value in raw["thresholds"].items()}, dict(raw["rows_expected"]),
                   _baseline(raw["baseline"]), dict(raw["sentences"]), dict(raw["approved"]),
                   hashlib.sha256(payload).hexdigest())
