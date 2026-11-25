"""단일 국내 ETF 입력 묶음과 분배금 권리 인터페이스(v1 2절, v2 3절)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal
import hashlib

from ..baseline_policy import VolatilityEstimate
from ..engine import _utc
from ..engine_strategy import _normalized_extremes
from ..engine_types import Bar, Instrument, aware, decimal
from .calendar import KrCalendar, coverage_gaps


PROVIDER_BASIS = "provider_adjusted_approximation"  # 공급자 수정 규칙 가격으로 평가, 분배금 반영 범위·시점 미확인
ENTITLEMENT_BASIS = "raw_price_with_distribution_entitlements"
RAW_PRICE_BASES = {"raw", "krx_raw_unadjusted"}


class DataHold(ValueError):
    """예상한 자료 문제(누락·불량 값·달력 부족 등). 이 예외만 시작일 보류로 흡수한다(D6).
    다른 ValueError는 코드 오류로 보고 그대로 전파한다."""


class InputIntegrityError(RuntimeError):
    """고정 입력 명세·파일 해시·배정표 오류. 자료 부족(ValueError)과 달리 보류로 흡수하지 않고 실행을 멈춘다."""


def price_mode(price_basis: str) -> str:
    """가격 기준은 분배금 사건 개수가 아니라 입력 가격 자료의 속성으로 정한다(B6)."""
    return "raw" if price_basis in RAW_PRICE_BASES else "provider_adjusted"


@dataclass(frozen=True)
class Distribution:
    """분배금 권리: 배당락 거래일 시가 보유 수량 × 주당 금액, 지급 시각에 현금."""

    symbol: str
    ex_date: date
    per_share: Decimal
    pay_at: datetime
    source: str

    def __post_init__(self):
        object.__setattr__(self, "per_share", decimal(self.per_share))
        aware(self.pay_at)
        if type(self.ex_date) is not date or self.per_share <= 0 or not self.source:
            raise ValueError("distribution needs ex_date, positive amount and a source")


@dataclass(frozen=True)
class VolatilityUnavailable:
    """직전 251거래일 종가 또는 유효 수익률 200개가 없어 그해 값을 만들 수 없음 → 자료 보류."""

    reason: str


@dataclass(frozen=True)
class ZeroVolatility:
    """251일 자료가 완전하고 유효 수익률이 200개 이상인데 표본 표준편차가 정확히 0.
    자료 결측이 아니다: 시작이면 비용 조건 미달, 연초 갱신이면 신규 매수 닫힘."""

    known_at: datetime
    reason: str = "valid zero volatility: 2x volatility is below positive round-trip costs"

    def __post_init__(self):
        object.__setattr__(self, "known_at", _utc(self.known_at))


VOLATILITY_TYPES = (VolatilityEstimate, VolatilityUnavailable, ZeroVolatility)


@dataclass(frozen=True)
class EtfInputs:
    """volatility[day]: 각 해 첫 국내 거래일 시가 전 R2 정의 변동성. 늦게 시작한 계좌도 그해 값을 쓴다."""

    instrument: Instrument
    bars: tuple
    calendar: KrCalendar
    volatility: dict
    distributions: tuple = ()
    provenance: dict = field(default_factory=dict)
    data_end: dict | None = None  # E2: 개설 뒤 파일 일봉 결측으로 관측이 끊긴 경우의 마지막 유효일·사유

    @property
    def start(self) -> date:
        return self.bars[0].session_date

    @property
    def end(self) -> date:
        return self.bars[-1].session_date

    @property
    def price_mode(self) -> str:
        return price_mode(self.instrument.price_basis)

    @property
    def basis(self) -> str:
        return ENTITLEMENT_BASIS if self.price_mode == "raw" else PROVIDER_BASIS

    def basis_label(self, labels: dict) -> dict:
        """가격 기준 표시는 입력 가격 자료에서 가져온다. 시작 이후 분배금 사건이 0건이어도 원가격이면 원가격 문구."""
        common = {"basis": self.basis, "price_mode": self.price_mode, "price_basis": self.instrument.price_basis,
                  "distribution_events": len(self.distributions)}
        if self.price_mode == "raw":
            return {**common, "label": labels["basis_entitlement"], "total_return_note": None}
        return {**common, "label": labels["basis"], "total_return_note": labels["total_return_note"]}


def _check_sessions(instrument, bars, calendar):
    if instrument.market != "KR" or instrument.currency != "KRW":
        raise ValueError("this connector supports one domestic KRW ETF only")
    if any(bar.symbol != instrument.symbol for bar in bars):
        raise ValueError("bars must belong to the single ETF")
    days = [bar.session_date for bar in bars]
    if days != calendar.window(days[0], days[-1]):
        raise DataHold("bars must cover every domestic session (and only those); missing bars mean judgment is held")
    missing = calendar.missing_years(days[0], days[-1])
    if missing:
        raise DataHold(f"domestic calendar has no sessions in {missing}; judgment held")
    if any((_utc(bar.open_at), _utc(bar.close_at)) != tuple(map(_utc, calendar.times[bar.session_date]))
           for bar in bars):
        raise DataHold("bar session times differ from the domestic calendar")


def volatility_digest(volatility: dict) -> str:
    """연초 변동성 내용 식별값(적재 뒤 바꿔 넣은 값을 정식 진입점에서 찾는다)."""
    rows = sorted((day.isoformat(), type(value).__name__, str(getattr(value, "value", "")),
                   str(getattr(value, "known_at", "")), getattr(value, "reason", "")) for day, value in volatility.items())
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def check_volatility(calendar, volatility):
    _check_volatility(calendar, volatility)


def _check_volatility(calendar, volatility):
    """연초 변동성은 그해 첫 국내 거래일 키에만 두고, 그날 시가 전에 알려진 값이어야 한다(미래 정보 차단)."""
    for day, value in volatility.items():
        if not isinstance(value, VOLATILITY_TYPES):
            raise ValueError("volatility values must be estimate, unavailable or zero markers")
        if day not in calendar.times or calendar.first_session_of_year(day.year) != day:
            raise ValueError(f"volatility key {day} is not the first domestic session of its year")
        known = getattr(value, "known_at", None)
        if known is not None and not known < _utc(calendar.times[day][0]):
            raise ValueError(f"volatility for {day} was known at {known}, not before that session's open")


def _with_distributions(instrument, bars, events):
    if not events:
        return bars
    if instrument.price_basis not in RAW_PRICE_BASES:
        raise ValueError("distribution entitlements need as-traded raw prices, not provider-adjusted prices")
    by_day = {}
    for event in events:
        if event.symbol != instrument.symbol or event.ex_date in by_day:
            raise ValueError("distribution events must be unique for this ETF")
        by_day[event.ex_date] = event
    if set(by_day) - {bar.session_date for bar in bars}:
        raise ValueError("distribution ex-date must be a supplied domestic session")
    return tuple(replace(bar, dividend=by_day[bar.session_date].per_share,
                         dividend_pay_at=by_day[bar.session_date].pay_at)
                 if bar.session_date in by_day else bar for bar in bars)


def prepare(instrument: Instrument, bars, calendar: KrCalendar, volatility: dict, *,
            distributions=(), provenance=None, data_end=None) -> EtfInputs:
    bars = tuple(sorted(bars, key=lambda bar: bar.session_date))
    if not bars:
        raise ValueError("no bars")
    _check_sessions(instrument, bars, calendar)
    if any(bar.dividend for bar in bars) and not distributions:
        raise ValueError("bar dividends must come from the distribution interface")
    gaps = coverage_gaps(calendar, bars[-1].session_date, start=bars[0].session_date)
    if gaps:
        raise DataHold("judgment held: " + "; ".join(gaps))
    _check_volatility(calendar, volatility)
    for bar in bars:
        try:  # R2 체결 단계의 OHLC 허용 범위 오류를 입력 단계(시작일별 보류)로 당긴다.
            _normalized_extremes(bar)
        except ValueError as error:
            raise DataHold(f"{bar.symbol} {bar.session_date} high/low: {error}") from error
    bars = _with_distributions(instrument, bars, tuple(distributions))
    return EtfInputs(instrument, bars, calendar, dict(volatility), tuple(distributions), dict(provenance or {}),
                     None if data_end is None else dict(data_end))


def starting_at(inputs: EtfInputs, start: date) -> EtfInputs:
    """같은 자료의 다른 시작일(v2 1절 오프셋). 시작 전 분배 권리는 버린다."""
    bars = tuple(bar for bar in inputs.bars if bar.session_date >= start)
    if not bars or bars[0].session_date != start:
        raise ValueError("start must be a supplied session")
    events = tuple(event for event in inputs.distributions if event.ex_date >= start)
    return replace(inputs, bars=bars, distributions=events)


def volatility_plan(inputs: EtfInputs, start: date, mode: str) -> dict:
    """시작 해의 첫 거래일 값으로 시작하고, annual이면 이후 매년 첫 거래일 값으로 신규 주문 기준을 바꾼다."""
    key = inputs.calendar.first_session_of_year(start.year)
    refreshes = inputs.calendar.year_firsts(start, inputs.end) if mode == "annual" else []
    unavailable = []
    for day in [key] + refreshes:
        value = inputs.volatility.get(day, VolatilityUnavailable(f"no estimate for {day}"))
        if isinstance(value, VolatilityUnavailable):
            unavailable.append({"date": day, "reason": value.reason})
    return {"start": key, "refreshes": refreshes, "unavailable": unavailable}
