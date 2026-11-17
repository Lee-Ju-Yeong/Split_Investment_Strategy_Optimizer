"""M2 연구 엔진의 공통 입력·결과 계약. 금액·비율은 Decimal, 시각은 aware datetime.

이 엔진은 제품 CPU/GPU 엔진과 분리한다. 제품 V2의 정수 비교와 경계값을 보존하기 위해
기준 엔진에서는 Decimal을 쓰며 M3 커널은 이 결과에 맞춰 검산한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal


ZERO = Decimal("0")
ONE = Decimal("1")


def decimal(value) -> Decimal:
    """문자·정수·기존 Decimal을 유한한 Decimal로 정규화한다."""
    if isinstance(value, bool):
        raise ValueError("bool은 금액이나 비율이 아닙니다")
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("금액과 비율은 유한해야 합니다")
    return number


def aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("시각에는 시간대가 있어야 합니다")
    return value


@dataclass(frozen=True)
class Instrument:
    symbol: str
    market: str  # KR / US / GOLD (KRX 금현물)
    currency: str
    tax_category: str = "unknown"
    price_basis: str = "raw"

    def __post_init__(self):
        if not self.symbol or self.market not in {"KR", "US", "GOLD"}:
            raise ValueError("상품 코드와 시장이 필요합니다")
        expected = "USD" if self.market == "US" else "KRW"
        if self.currency != expected:
            raise ValueError("상품 시장과 통화가 다릅니다")


@dataclass(frozen=True)
class Bar:
    symbol: str
    session_date: date
    open_at: datetime
    close_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    previous_close: Decimal | None = None
    adv20: Decimal | None = None  # 직전 20거래일 평균 거래대금, 자기 시장 통화
    limit_locked: bool = False
    dividend: Decimal = ZERO  # 주당 현금, 가격에 이미 반영됐다면 0이어야 함
    dividend_pay_at: datetime | None = None
    split_ratio: Decimal = ONE

    def __post_init__(self):
        aware(self.open_at)
        aware(self.close_at)
        if self.close_at <= self.open_at:
            raise ValueError("마감은 개장보다 늦어야 합니다")
        for name in ("open", "high", "low", "close", "volume", "dividend", "split_ratio"):
            object.__setattr__(self, name, decimal(getattr(self, name)))
        for name in ("previous_close", "adv20"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, decimal(getattr(self, name)))
        if min(self.open, self.high, self.low, self.close) <= 0 or self.volume < 0:
            raise ValueError("가격은 양수, 거래량은 0 이상이어야 합니다")
        if self.high < self.low or self.dividend < 0 or self.split_ratio <= 0:
            raise ValueError("가격 범위나 기업행사 값이 잘못됐습니다")
        if self.dividend_pay_at is not None:
            aware(self.dividend_pay_at)


@dataclass(frozen=True)
class LevelRule:
    level: int
    budget: Decimal  # 자기 시장 통화, 매수 수수료 포함 주문 한도
    drop_rate: Decimal  # 비율: 5% = 0.05
    take_rate: Decimal

    def __post_init__(self):
        if type(self.level) is not int or self.level < 1:
            raise ValueError("차수는 양의 정수여야 합니다")
        for name in ("budget", "drop_rate", "take_rate"):
            object.__setattr__(self, name, decimal(getattr(self, name)))
        if self.budget <= 0 or not ZERO <= self.drop_rate < ONE or self.take_rate < ZERO:
            raise ValueError("주문 예산과 매매 비율이 잘못됐습니다")


@dataclass(frozen=True)
class Position:
    symbol: str
    level: int
    quantity: int
    entry_price: Decimal
    buy_fee: Decimal
    take_rate: Decimal
    opened_at: datetime

    def __post_init__(self):
        aware(self.opened_at)
        if type(self.level) is not int or self.level < 1 or type(self.quantity) is not int or self.quantity <= 0:
            raise ValueError("차수와 보유 수량은 양의 정수여야 합니다")
        for name in ("entry_price", "buy_fee", "take_rate"):
            object.__setattr__(self, name, decimal(getattr(self, name)))
        if self.entry_price <= 0 or self.buy_fee < 0 or self.take_rate < 0:
            raise ValueError("보유 원가·비용·익절 비율이 잘못됐습니다")

    @property
    def principal(self) -> Decimal:
        return self.entry_price * self.quantity


@dataclass(frozen=True)
class Candidate:
    symbol: str
    side: str
    level: int
    quantity: int
    price: Decimal
    at: datetime
    reason: str

    def __post_init__(self):
        aware(self.at)
        object.__setattr__(self, "price", decimal(self.price))
        if self.side not in {"BUY", "SELL"} or self.price <= 0:
            raise ValueError("주문 방향과 가격이 잘못됐습니다")
        if type(self.quantity) is not int or self.quantity <= 0 or type(self.level) is not int or self.level < 1:
            raise ValueError("주문 수량과 차수는 양의 정수여야 합니다")


@dataclass(frozen=True)
class FxRate:
    available_at: datetime
    krw_per_usd: Decimal

    def __post_init__(self):
        aware(self.available_at)
        object.__setattr__(self, "krw_per_usd", decimal(self.krw_per_usd))
        if self.krw_per_usd <= 0:
            raise ValueError("환율은 양수여야 합니다")


@dataclass
class EngineResult:
    trades: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    valuations: list[dict] = field(default_factory=list)
    cash_events: list[dict] = field(default_factory=list)
    final_cash: dict[str, Decimal] = field(default_factory=dict)
    positions: list[Position] = field(default_factory=list)
    tax: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
