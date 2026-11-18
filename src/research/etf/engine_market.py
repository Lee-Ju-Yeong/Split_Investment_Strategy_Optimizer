"""ETF M2의 날짜별 호가와 Decimal 비용표. 근거/제한은 etf_execution_rules.md 참조.

호가의 on은 거래일, SEC의 on은 charge date(통상 결제일)다. 수정주가에
이 규칙을 적용한 값은 가상 호가이며 과거의 실제 주문가격을 복원하지 않는다.
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from zoneinfo import ZoneInfo

from .engine_types import aware, decimal


_RULES = json.loads(Path(__file__).with_name("market_rules_v1.json").read_text(encoding="utf-8"))


def _market(value: str) -> str:
    if value not in _RULES["market_aliases"]:
        raise ValueError(f"지원하지 않는 시장: {value}")
    return _RULES["market_aliases"][value]


def _date(value: date) -> date:
    if type(value) is not date:
        raise ValueError("on에는 datetime이 아닌 date가 필요합니다")
    return value


def _nonnegative(value, name: str) -> Decimal:
    number = decimal(value)
    if number < 0:
        raise ValueError(f"{name}은 0 이상이어야 합니다")
    return number


def _period(on: date, periods: list[dict], label: str) -> dict:
    stamp = _date(on).isoformat()
    for row in periods:
        if row["from"] <= stamp <= row["through"]:
            return row
    raise ValueError(f"{label} 공식 확인 범위 밖 날짜: {stamp}")


def tick_size(market: str, on: date, price: Decimal) -> Decimal:
    """거래일 on과 자기 통화 표시 가격에 해당하는 호가단위를 반환한다."""
    market = _market(market)
    price = decimal(price)
    if price <= 0:
        raise ValueError("가격은 양수여야 합니다")
    row = _period(on, _RULES["tick_rules"][market]["periods"], f"{market} 호가")
    for band in row["bands"]:
        if band["below"] is None or price < Decimal(band["below"]):
            return Decimal(band["tick"])
    raise ValueError("가격에 해당하는 호가단위가 없습니다")


def round_price(market: str, on: date, price: Decimal, side: str) -> Decimal:
    """BUY는 내림, SELL은 올림. 내림 결과 0인 주문은 가격을 올리지 않고 거부한다."""
    if side not in {"BUY", "SELL"}:
        raise ValueError("side는 BUY 또는 SELL이어야 합니다")
    price = decimal(price)
    tick = tick_size(market, on, price)
    rounding = ROUND_FLOOR if side == "BUY" else ROUND_CEILING
    rounded = (price / tick).to_integral_value(rounding=rounding) * tick
    if rounded <= 0:
        raise ValueError("호가 내림 결과가 0입니다")
    return rounded


def sec_fee_rate(on: date, *, known_at: datetime | None = None) -> Decimal:
    """charge date의 SEC 비율. known_at은 그 시각에 공표된 정보만 허용한다.

    원문의 공표 시각은 확인되지 않아 미국 동부시간 공표일 종료부터 사용한다.
    known_at=None은 사후 정적 비용표 조회이며 신호의 정보 가용성을 검사하지 않는다.
    """
    if known_at is not None:
        aware(known_at)
    row = _period(on, _RULES["sec_fee"]["periods"], "SEC")
    if known_at is not None:
        next_day = date.fromisoformat(row["announced_on"]) + timedelta(days=1)
        available_at = datetime.combine(next_day, time(), ZoneInfo("America/New_York"))
        if known_at < available_at:
            raise ValueError(f"SEC 요율은 판단 시점에 아직 공표되지 않았습니다: {row['announced_on']}")
    return Decimal(row["rate"])


def fee_amount(market, side, on, notional, commission_rate=None, sec_rate_override=None) -> Decimal:
    """연구 위탁수수료와 US SELL의 SEC 비례비용. 통화 최소단위 반올림은 하지 않는다.

    US SELL의 on은 charge date다. 검증된 요율을 전달하거나 합성 사례를 만들 때
    sec_rate_override를 사용한다. 명시적인 0도 역사표 조회를 완전히 우회한다.
    """
    market = _market(market)
    _date(on)
    if side not in {"BUY", "SELL"}:
        raise ValueError("side는 BUY 또는 SELL이어야 합니다")
    notional = _nonnegative(notional, "거래대금")
    commission = _RULES["commission_rates"][market] if commission_rate is None else commission_rate
    rate = _nonnegative(commission, "위탁수수료율")
    override = None if sec_rate_override is None else _nonnegative(sec_rate_override, "SEC 요율")
    if market == "US" and side == "SELL":
        rate += sec_fee_rate(on) if override is None else override
    return notional * rate
