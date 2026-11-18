"""학습 자료로 고정하는 M4 기준선. 평가 자료로 맞춘 정책은 진단 전용이다."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import json

from .engine_types import EngineResult, ONE, ZERO, decimal
from .engine import _utc


def _hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _symbols(instruments):
    symbols = tuple(item.symbol for item in instruments)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("baseline basket must contain unique instruments")
    return symbols


def _unit(value, label):
    value = decimal(value)
    if not ZERO <= value <= ONE:
        raise ValueError(f"{label} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class VolatilityEstimate:
    value: Decimal
    known_at: datetime

    def __post_init__(self):
        object.__setattr__(self, "known_at", _utc(self.known_at))
        object.__setattr__(self, "value", decimal(self.value))
        if self.value <= ZERO:
            raise ValueError("volatility must be positive")


@dataclass(frozen=True)
class BaselinePolicy:
    investment_fraction: Decimal
    weights: tuple[tuple[str, Decimal], ...]
    known_at: datetime
    evaluation_start: datetime
    source_sha256: str
    role: str = "primary"
    mode: str = "monthly"

    def __post_init__(self):
        object.__setattr__(self, "known_at", _utc(self.known_at))
        object.__setattr__(self, "evaluation_start", _utc(self.evaluation_start))
        object.__setattr__(self, "investment_fraction", _unit(self.investment_fraction, "investment_fraction"))
        weights = tuple((symbol, _unit(weight, "weight")) for symbol, weight in self.weights)
        object.__setattr__(self, "weights", weights)
        if not weights or len({s for s, _ in weights}) != len(weights):
            raise ValueError("weights must have unique symbols")
        with localcontext() as arithmetic:
            arithmetic.prec = 80
            arithmetic.rounding = ROUND_HALF_EVEN
            if sum((weight for _, weight in weights), ZERO) != ONE:
                raise ValueError("weights must sum to 1")
        if self.role not in {"primary", "reference", "diagnostic"} or self.mode not in {"monthly", "buy_hold"}:
            raise ValueError("invalid baseline role or mode")
        if self.role != "diagnostic" and self.known_at >= self.evaluation_start:
            raise ValueError("primary/reference policy must be known before evaluation_start")
        if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256):
            raise ValueError("source_sha256 must be a SHA256 hex digest")

    @property
    def sha256(self):
        return _hash(self.__dict__)


def _weights(symbols, estimates, cutoff):
    if estimates is not None and set(estimates) != set(symbols):
        raise ValueError("volatilities must cover exactly the basket")
    values = []
    for symbol in symbols:
        estimate = None if estimates is None else estimates[symbol]
        if estimates is not None and (not isinstance(estimate, VolatilityEstimate) or estimate.known_at > cutoff):
            raise ValueError("volatility must be known by the training cutoff")
        values.append(Fraction(1) if estimate is None else 1 / Fraction(estimate.value))
    total = sum(values)
    weights = [Decimal(value.numerator) / value.denominator for value in (item / total for item in values)]
    weights[-1] = ONE - sum(weights[:-1], ZERO)
    return tuple(zip(symbols, weights))


def _exposure_rows(result, symbols, start, end):
    start, end = _utc(start), _utc(end)
    if start > end or not result.valuations:
        raise ValueError("explicit nonempty learning/evaluation window required")
    rows, previous = [], None
    for row in result.valuations:
        at = _utc(row["at"])
        if not start <= at <= end or (previous is not None and at <= previous):
            raise ValueError("valuation times must increase inside the explicit window")
        holdings = row["holdings_krw"]
        if set(holdings) != set(symbols):
            raise ValueError("valuation basket differs from baseline basket")
        equity = decimal(row["equity_krw"])
        invested = sum((decimal(holdings[symbol]) for symbol in symbols), ZERO)
        if equity <= ZERO or any(decimal(value) < ZERO for value in holdings.values()) or invested > equity:
            raise ValueError("exposure requires positive equity and unlevered holdings")
        rows.append((at, equity, tuple((s, decimal(holdings[s])) for s in symbols), invested / equity))
        previous = at
    return rows


def fit_baseline(training_result: EngineResult, instruments, *, training_start, training_end,
                 evaluation_start, volatilities=None) -> BaselinePolicy:
    """학습 결과만 받는다. 위험 예산 짝에는 학습 시점의 변동성 추정값을 명시한다."""
    training_start, training_end, evaluation_start = map(_utc, (training_start, training_end, evaluation_start))
    if training_end >= evaluation_start:
        raise ValueError("training_end must precede evaluation_start")
    with localcontext() as arithmetic:
        arithmetic.prec = 80
        arithmetic.rounding = ROUND_HALF_EVEN
        symbols = _symbols(instruments)
        rows = _exposure_rows(training_result, symbols, training_start, training_end)
        weights = _weights(symbols, volatilities, training_end)
        source = _hash((rows, weights, training_start, training_end, volatilities))
        return BaselinePolicy(sum((row[3] for row in rows), ZERO) / len(rows), weights,
                              training_end, evaluation_start, source)


def fixed_baseline(instruments, *, investment_fraction, known_at, evaluation_start,
                   volatilities=None, mode="monthly") -> BaselinePolicy:
    """프런티어·보유 참고군. 이 정책을 학습 매칭 주 기준선으로 표시하지 않는다."""
    with localcontext() as arithmetic:
        arithmetic.prec = 80
        arithmetic.rounding = ROUND_HALF_EVEN
        weights = _weights(_symbols(instruments), volatilities, _utc(known_at))
        return BaselinePolicy(investment_fraction, weights, known_at, evaluation_start,
                              _hash((investment_fraction, weights, known_at, volatilities)),
                              role="reference", mode=mode)


def diagnostic_baseline(evaluation_result, policy, *, evaluation_end):
    """사후 노출 매칭. 이미 학습에서 고정한 가중만 재사용하고 주 판정을 금지한다."""
    with localcontext() as arithmetic:
        arithmetic.prec = 80
        arithmetic.rounding = ROUND_HALF_EVEN
        evaluation_end = _utc(evaluation_end)
        symbols = tuple(symbol for symbol, _ in policy.weights)
        rows = _exposure_rows(evaluation_result, symbols, policy.evaluation_start, evaluation_end)
        return replace(policy, investment_fraction=sum((row[3] for row in rows), ZERO) / len(rows),
                       known_at=evaluation_end, source_sha256=_hash((policy.sha256, rows)), role="diagnostic")
