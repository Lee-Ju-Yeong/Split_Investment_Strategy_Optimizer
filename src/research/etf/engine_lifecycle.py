"""명시한 국내 ETF 상폐 공지에 따른 연구 청산. 실제 해지 상환금 모형이 아니다."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from .engine_strategy import _options, _step_price
from .engine_types import Candidate, ZERO, aware


@dataclass(frozen=True)
class DelistingEvent:
    symbol: str
    known_at: datetime
    last_session: date
    delisting_date: date
    source_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "known_at", aware(self.known_at).astimezone(timezone.utc))
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("delisting needs a symbol")
        if type(self.last_session) is not date or type(self.delisting_date) is not date:
            raise ValueError("delisting dates must be date values")
        if self.last_session >= self.delisting_date:
            raise ValueError("last trading session must precede delisting_date")
        if (not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.source_sha256)):
            raise ValueError("delisting needs a source SHA256 digest")


def validate_delistings(events, instruments, bars):
    result = {}
    for event in events:
        if not isinstance(event, DelistingEvent) or event.symbol in result:
            raise ValueError("delistings must contain unique DelistingEvent values")
        if event.symbol not in instruments or instruments[event.symbol].market != "KR":
            raise ValueError("delisting liquidation supports known KR ETF symbols only")
        own_bars = [bar for bar in bars if bar.symbol == event.symbol]
        final = next((bar for bar in own_bars if bar.session_date == event.last_session), None)
        if final is None or any(bar.session_date > event.last_session for bar in own_bars):
            raise ValueError("delisting needs its last normal session and no later bars")
        if event.known_at > final.open_at:
            raise ValueError("delisting announcement must be known before the last session open")
        if final.volume == ZERO or final.limit_locked:
            raise ValueError("delisting last normal session must be tradable")
        result[event.symbol] = event
    return result


def closing_candidate(instrument, bar, positions, options):
    if not positions:
        return None
    context = _options(instrument, bar, options.penetration_ticks, options.slippage_ticks,
                       options.commission_rate, options.sec_rate_override, options.product_sell_tax_rate)
    price = _step_price(context, bar.close, -options.slippage_ticks)
    if price is None:
        raise ValueError("delisting closing price is nonpositive after slippage")
    return Candidate(bar.symbol, "SELL", min(p.level for p in positions), sum(p.quantity for p in positions),
                     price, bar.close_at, "delisting_liquidation")
