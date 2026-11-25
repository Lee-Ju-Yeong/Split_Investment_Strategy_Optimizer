"""결과 열람 전에 고정한 설계 값(v1→v2→v2.1→v2.2). 숫자는 kodex_core_v2.json에서만 읽는다."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("kodex_core_v2.json")
SCHEMA = "kodex-core-split-v2.2"
VOLATILITY_MODES = ("fixed", "annual")
REENTRY_MODES = ("M", "D", "none")
ROLES = ("primary", "extension", "comparison", "reference")


@dataclass(frozen=True)
class Segment:
    label: str
    start: date
    end: date


@dataclass(frozen=True)
class StudyRow:
    """실행 행: 같은 명단·시작일 셋. 명단은 결과 전에 고정하며 미달은 교체하지 않는다."""

    id: str
    role: str
    symbols: tuple
    starts: tuple
    segments: tuple
    denominator: int | None = None


@dataclass(frozen=True)
class StudyConfig:
    primary_symbol: str
    capital: dict
    levels: int
    kg: Decimal
    kp: Decimal
    costs: tuple
    participation_rate: Decimal
    penetration_ticks: int
    slippage_ticks: int
    paths: tuple
    modes: tuple
    start_offsets: tuple
    volatility_mode: str
    thresholds: dict
    matched_band: Decimal
    segments: tuple
    end: date
    rows: tuple
    labels: dict
    sha256: str
    nominal_floor: Decimal = Decimal("100000000")
    price_basis: str = "naver_provider_adjusted_distribution_unverified"
    registered: dict = field(default_factory=dict)  # 등록 입력 식별값(정식 실행 대조 기준)
    declared_sha256: str = ""  # E4: 파일의 모든 선언(객체에 없는 키 포함)을 정규화한 JSON 해시(공백 무관)

    def __post_init__(self):
        if self.volatility_mode not in VOLATILITY_MODES:
            raise ValueError("volatility_mode must be fixed or annual")
        if any(mode not in REENTRY_MODES for mode in self.modes):
            raise ValueError("reentry modes must be M, D or none")
        if any(type(n) is not int or n < 0 for n in self.start_offsets):
            raise ValueError("start offsets must be nonnegative session counts")
        if any(row.role not in ROLES for row in self.rows):
            raise ValueError("unknown study row role")

    def cost(self, name) -> Decimal:
        return dict(self.costs)[name]

    def with_changes(self, **changes) -> "StudyConfig":
        """검토로 바뀔 수 있는 인자를 명시적으로 바꾼다. sha256은 원본 파일 값이고, 실제 적용 값은
        applied_sha256()으로 구별한다."""
        return replace(self, **changes)

    def applied_sha256(self) -> str:
        """실제 적용된 설정 내용의 해시(원본 파일 해시 필드는 제외)."""
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__
                   if name not in ("sha256", "declared_sha256")}
        text = json.dumps(_plain(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {name: _plain(getattr(value, name)) for name in value.__dataclass_fields__}
    return value if value is None or isinstance(value, (str, int, bool)) else str(value)


def _segments(rows):
    return tuple(Segment(label, date.fromisoformat(start), date.fromisoformat(end)) for label, start, end in rows)


def _rows(raw, sets):
    return tuple(StudyRow(row["id"], row["role"], tuple(row["symbols"]),
                          tuple(date.fromisoformat(day) for day in row["starts"]),
                          sets[row["segments"]], row.get("denominator")) for row in raw)


def declared_digest(raw: dict) -> str:
    """파일 선언 전체의 정규화 해시. 공백·키 순서만 다른 파일은 같고, 어떤 선언이든 값이 바뀌면 다르다."""
    text = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


# 연복리 연 일수는 R2 지표(baseline_report.YEAR_DAYS)가 쓴다. 다른 값을 선언한 파일은 적용되지 않으므로 거부한다.
YEAR_DAYS_DECLARED = "365.25"


def load_config(path=CONFIG_PATH) -> StudyConfig:
    payload = Path(path).read_bytes()
    raw = json.loads(payload)
    if raw.get("schema") != SCHEMA:
        raise ValueError("unexpected KODEX core configuration schema")
    if raw.get("year_days") != YEAR_DAYS_DECLARED:
        raise ValueError(f"year_days {raw.get('year_days')!r} is not applied by the metrics ({YEAR_DAYS_DECLARED})")
    split, sets = raw["split"], {name: _segments(rows) for name, rows in raw["segment_sets"].items()}
    return StudyConfig(
        raw["primary_symbol"], {k: Decimal(v) for k, v in raw["capital_krw"].items()},
        split["levels"], Decimal(split["kg"]), Decimal(split["kp"]),
        tuple((k, Decimal(v)) for k, v in raw["costs_one_way"].items()),
        Decimal(raw["participation_rate"]), raw["penetration_ticks"], raw["slippage_ticks"],
        tuple(raw["paths"]), tuple(raw["reentry_modes"]), tuple(raw["start_offsets"]),
        raw["volatility_mode"], {k: Decimal(v) for k, v in raw["thresholds"].items()},
        Decimal(raw["matched_exposure_band"]), sets["from_2011"], date.fromisoformat(raw["end"]),
        _rows(raw["rows"], sets), dict(raw["labels"]), hashlib.sha256(payload).hexdigest(),
        Decimal(raw["extension_start_capacity_floor_krw"]), raw["expected_price_basis"],
        dict(raw["registered_inputs"]), declared_digest(raw))


def frozen_identity(config: StudyConfig) -> dict:
    """적용 설정이 고정 파일과 같은지. 다르면 설정 변경 실행으로 구별해 표시한다."""
    frozen = load_config()
    applied = config.applied_sha256()
    declared_changed = config.declared_sha256 != frozen.declared_sha256
    return {"frozen_config_file_sha256": frozen.sha256, "provided_config_file_sha256": config.sha256,
            "frozen_declared_sha256": frozen.declared_sha256, "provided_declared_sha256": config.declared_sha256,
            "applied_config_sha256": applied, "frozen_config_applied_sha256": frozen.applied_sha256(),
            "declarations_modified": declared_changed,
            "config_modified": applied != frozen.applied_sha256() or declared_changed}
