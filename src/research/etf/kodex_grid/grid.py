"""격자 정의(kodex_grid_v1.json): 36개 설정 번호, 설정별 적용 설정, 8개 영역과 순위."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from ..kodex_core.config import Segment, StudyConfig, _plain


GRID_PATH = Path(__file__).with_name("kodex_grid_v1.json")
SCHEMA = "kodex-grid-v1"


@dataclass(frozen=True)
class CoreOption:
    """코어·분할 원금. 둘의 합은 비교용 단독 보유 원금(1억 원)과 같다."""

    label: str
    core: Decimal
    split: Decimal


@dataclass(frozen=True)
class Setting:
    number: int
    core: CoreOption
    kg: Decimal
    kp: Decimal
    levels: int

    def describe(self) -> dict:
        return {"number": self.number, "core": self.core.label, "kg": str(self.kg), "kp": str(self.kp),
                "levels": self.levels, "core_krw": str(self.core.core), "split_krw": str(self.core.split)}


@dataclass(frozen=True)
class Region:
    """코어 하나 × 인접 kg 두 값 × 인접 kp 두 값 × 모든 차수 = 8점. rank 1이 최우선이다."""

    rank: int
    core: str
    kg: tuple
    kp: tuple
    numbers: tuple

    def describe(self) -> dict:
        return {"rank": self.rank, "core": self.core, "kg": [str(v) for v in self.kg],
                "kp": [str(v) for v in self.kp], "numbers": list(self.numbers)}


@dataclass(frozen=True)
class GridSpec:
    symbol: str
    primary_start: date
    end: date
    mode: str
    account: Decimal
    cores: tuple
    kgs: tuple
    kps: tuple
    levels: tuple
    segments: tuple
    thresholds: dict
    region: dict
    reconfirm: dict
    extension: dict
    baseline: dict
    approved: dict
    conclusions: dict
    sha256: str

    def __post_init__(self):
        for option in self.cores:
            if option.core + option.split != self.account:
                raise ValueError(f"core {option.label}: core + split must equal the {self.account} account")
        if len(self.settings()) != 36 or any(len(region.numbers) != self.region["points"] for region in self.regions()):
            raise ValueError("the grid must declare 36 settings and eight-point regions")

    def settings(self) -> tuple:
        """(코어, kg, kp, 차수) 사전순 번호. 코어는 선언 순서(50%가 먼저)다."""
        rows = [(core, kg, kp, levels) for core in self.cores for kg in self.kgs for kp in self.kps
                for levels in self.levels]
        return tuple(Setting(number, *row) for number, row in enumerate(rows, start=1))

    def setting(self, number: int) -> Setting:
        found = [row for row in self.settings() if row.number == number]
        if not found:
            raise ValueError(f"setting number {number} is outside the grid")
        return found[0]

    def number_of(self, point: dict) -> int:
        """point: {"core": 코어 표시, "kg": …, "kp": …, "levels": 차수}에 해당하는 설정 번호."""
        key = (point["core"], Decimal(point["kg"]), Decimal(point["kp"]), point["levels"])
        found = [row.number for row in self.settings() if (row.core.label, row.kg, row.kp, row.levels) == key]
        if len(found) != 1:
            raise ValueError(f"no single grid setting for {key}")
        return found[0]

    def regions(self) -> tuple:
        """순위: 코어 우선순위(80% 먼저) → kg 하한 → kp 하한 오름차순. 코어당 4개, 합 8개."""
        rows = [(core, kg, kp) for core in self.region["core_priority"] for kg in self.region["kg_pairs"]
                for kp in self.region["kp_pairs"]]
        return tuple(Region(rank, core, kg, kp, self._members(core, kg, kp))
                     for rank, (core, kg, kp) in enumerate(rows, start=1))

    def _members(self, core, kg_pair, kp_pair) -> tuple:
        return tuple(row.number for row in self.settings()
                     if row.core.label == core and row.kg in kg_pair and row.kp in kp_pair)

    def candidate_number(self, region: Region) -> int:
        """후속 후보 한 점: 선택 영역의 5차·kg 하한·kp 하한."""
        return self.number_of({"core": region.core, "kg": min(region.kg), "kp": min(region.kp),
                               "levels": self.region["candidate_levels"]})

    def baseline_number(self) -> int:
        return self.number_of(self.baseline)

    def applied_sha256(self) -> str:
        """실제 적용된 격자 정의의 해시(파일 해시 필드 제외). 시험용으로 바꾼 정의를 파일과 구별한다."""
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "sha256"}
        text = json.dumps(_plain(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()


def segment_labels(segments, start: date, end: date) -> tuple:
    """그 시작일부터 만기까지의 계좌가 가져야 할 구간 이름(시작 이전에 끝난 구간은 없다)."""
    return tuple(row.label for row in segments if row.end >= start and row.start <= end)


def _decimals(values) -> tuple:
    return tuple(Decimal(value) for value in values)


def _pairs(rows) -> tuple:
    return tuple(_decimals(row) for row in rows)


def _region(raw) -> dict:
    return {"core_priority": tuple(raw["core_priority"]), "kg_pairs": _pairs(raw["kg_pairs"]),
            "kp_pairs": _pairs(raw["kp_pairs"]), "points": raw["points"], "candidate_levels": raw["candidate_levels"]}


def _dates(values) -> tuple:
    return tuple(date.fromisoformat(value) for value in values)


def load_grid(path=GRID_PATH) -> GridSpec:
    payload = Path(path).read_bytes()
    raw = json.loads(payload)
    if raw.get("schema") != SCHEMA or raw.get("exploration") is not True or raw.get("formal_run") is not False:
        raise ValueError("unexpected KODEX grid definition (schema, exploration or formal flag)")
    if raw["numbering"] != ["core", "kg", "kp", "levels"]:
        raise ValueError("setting numbers are defined in (core, kg, kp, levels) order only")
    axes = raw["axes"]
    cores = tuple(CoreOption(row["label"], Decimal(row["core_krw"]), Decimal(row["split_krw"])) for row in axes["core"])
    segments = tuple(Segment(label, date.fromisoformat(a), date.fromisoformat(b)) for label, a, b in raw["segments"])
    thresholds = {name: (value if type(value) is int else Decimal(value)) for name, value in raw["verdict"].items()}
    reconfirm = {"offsets": tuple(raw["reconfirm"]["offsets"]), "starts": _dates(raw["reconfirm"]["starts"])}
    more = raw["extension"]
    extension = {"symbols": tuple(more["symbols"]), "start": date.fromisoformat(more["start"]),
                 "base_edge_gt": Decimal(more["base_edge_gt"]), "points_min": more["points_min"]}
    return GridSpec(raw["symbol"], date.fromisoformat(raw["primary_start"]), date.fromisoformat(raw["end"]),
                    raw["reentry_mode"], Decimal(raw["account_krw"]), cores, _decimals(axes["kg"]),
                    _decimals(axes["kp"]), tuple(axes["levels"]), segments, thresholds, _region(raw["region"]),
                    reconfirm, extension, dict(raw["baseline"]), dict(raw["approved"]), dict(raw["conclusions"]),
                    hashlib.sha256(payload).hexdigest())


def check_base(spec: GridSpec, base: StudyConfig) -> None:
    """격자 선언과 고정 설정이 같은 종목·만기·구간·방식을 가리키는지. 다르면 장부를 열기 전에 멈춘다."""
    problems = []
    if base.primary_symbol != spec.symbol:
        problems.append(f"symbol {base.primary_symbol} != {spec.symbol}")
    if base.end != spec.end or base.segments != spec.segments:
        problems.append("end or segments differ from the grid declaration")
    if base.sha256 != spec.approved["base_config_file_sha256"]:
        problems.append("base configuration file is not the approved kodex_core_v2.json")
    if set(base.paths) != {"O-H-L-C", "O-L-H-C"} or [name for name, _ in base.costs] != ["base", "stress"]:
        problems.append("two intraday paths and base/stress costs are required")
    if problems:
        raise ValueError("grid declaration and base configuration disagree: " + "; ".join(problems))


def setting_config(base: StudyConfig, spec: GridSpec, setting: Setting) -> StudyConfig:
    """고정 설정에서 차수·kg·kp·원금·재진입 방식만 바꾼 변형. 다른 필드는 건드리지 않는다."""
    capital = {"core": setting.core.core, "split": setting.core.split, "hold": spec.account}
    return base.with_changes(levels=setting.levels, kg=setting.kg, kp=setting.kp, capital=capital,
                             modes=(spec.mode,))
