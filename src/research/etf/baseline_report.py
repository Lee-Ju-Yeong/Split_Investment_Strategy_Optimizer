"""M4 프런티어 데이터와 쌍대 지표. 보류 판정·통계적 유의성 검정은 수행하지 않는다."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import Decimal, localcontext

from .baseline_engine import run_baseline
from .baseline_policy import diagnostic_baseline
from .engine_types import ONE, ZERO, decimal
from .engine import _utc


FRONTIER_FRACTIONS = tuple(Decimal(value) for value in ("0", ".25", ".5", ".75", "1"))
YEAR_DAYS = Decimal("365.25")


def _days(duration):
    micros = (duration.days * 86400 + duration.seconds) * 1000000 + duration.microseconds
    return Decimal(micros) / Decimal("86400000000")


def _valuation_rows(result, instruments, started_at, ended_at):
    started_at, ended_at = _utc(started_at), _utc(ended_at)
    if ended_at <= started_at or not result.valuations:
        raise ValueError("metrics require an explicit positive observation period")
    symbols, previous = {item.symbol for item in instruments}, started_at
    rows = [{**row, "at": _utc(row["at"])} for row in result.valuations]
    for row in rows:
        at = row["at"]
        if not previous < at <= ended_at or set(row["holdings_krw"]) != symbols:
            raise ValueError("metric valuations must increase within the period and share the basket")
        if decimal(row["equity_krw"]) <= ZERO:
            raise ValueError("CAGR/exposure require positive equity")
        previous = at
    if rows[-1]["at"] != ended_at:
        raise ValueError("ended_at must have an explicit valuation")
    return rows


def _drawdowns(rows, initial_equity, started_at):
    peak, peak_at, drawdown, longest = initial_equity, started_at, ZERO, ZERO
    underwater = False
    for row in rows:
        at, equity = row["at"], decimal(row["equity_krw"])
        drawdown = min(drawdown, equity / peak - ONE)
        if equity < peak:
            underwater = True
            longest = max(longest, _days(at - peak_at))
        else:
            if underwater:
                longest = max(longest, _days(at - peak_at))
            peak, peak_at, underwater = equity, at, False
    return drawdown, longest


def _unrealized_price_pnl(row, instruments):
    by_symbol = {item.symbol: item for item in instruments}
    total = ZERO
    for position in row["positions"]:
        rate = row["fx_rate"] if by_symbol[position.symbol].currency == "USD" else ONE
        total += (row["prices"][position.symbol] - position.entry_price) * position.quantity * rate
    return total


@dataclass(frozen=True)
class Performance:
    cagr: Decimal
    mdd: Decimal
    calmar: Decimal | None
    underwater_days: Decimal
    average_investment_fraction: Decimal
    rejected_orders: int
    unrealized_price_pnl_krw: Decimal
    initial_equity_krw: Decimal
    ending_equity_krw: Decimal


def performance(result, instruments, *, initial_equity_krw, started_at, ended_at):
    """초기 자산을 곡선에 포함한다. 단위는 비율이며 1%는 .01이다.

    CAGR은 실제 경과일/365.25, 수중 기간은 표본 평가 시각의 고점부터 회복/종료까지다.
    미실현 가격손익은 현재 환율로 환산한 보유분 가격 차이이며 수수료·환차익·청산세는 제외한다.
    """
    with localcontext() as arithmetic:
        arithmetic.prec = 80
        started_at, ended_at = _utc(started_at), _utc(ended_at)
        initial = decimal(initial_equity_krw)
        if initial <= ZERO:
            raise ValueError("initial_equity_krw must be positive")
        rows = _valuation_rows(result, instruments, started_at, ended_at)
        ending = decimal(rows[-1]["equity_krw"])
        cagr = (ending / initial) ** (YEAR_DAYS / _days(ended_at - started_at)) - ONE
        mdd, underwater = _drawdowns(rows, initial, started_at)
        exposure = sum((sum((decimal(v) for v in row["holdings_krw"].values()), ZERO)
                        / decimal(row["equity_krw"]) for row in rows), ZERO) / len(rows)
        rejected = sum(started_at <= _utc(row["at"]) <= ended_at for row in result.rejections)
        return Performance(cagr, mdd, cagr / abs(mdd) if mdd else None, underwater, exposure,
                           rejected, _unrealized_price_pnl(rows[-1], instruments), initial, ending)


def paired_difference(strategy, baseline):
    """ΔCAGR과 ΔMDD 모두 양수일 때 수익·낙폭의 동시 개선이다."""
    with localcontext() as arithmetic:
        arithmetic.prec = 80
        return {"delta_cagr": strategy.cagr - baseline.cagr,
                "delta_mdd": abs(baseline.mdd) - abs(strategy.mdd),
                "delta_calmar": (strategy.calmar - baseline.calmar
                                 if strategy.calmar is not None and baseline.calmar is not None else None),
                "delta_underwater_days": strategy.underwater_days - baseline.underwater_days}


def baseline_policies(policy):
    if policy.role == "diagnostic" or policy.mode != "monthly":
        raise ValueError("the comparison suite requires a prospective monthly policy")
    policies = {"matched": policy, "buy_hold": replace(policy, investment_fraction=ONE,
                                                       role="reference", mode="buy_hold")}
    for fraction in FRONTIER_FRACTIONS:
        label = f"frontier_{int(fraction * 100):03d}"
        policies[label] = replace(policy, investment_fraction=fraction, role="reference")
    return policies


def run_baseline_suite(policy, instruments, bars, *, initial_cash, strategy_result=None,
                       diagnostic_end=None, **engine_options):
    """같은 입력·실행 옵션으로 주 기준선, 보유, 5개 프런티어를 실행한다."""
    policies = baseline_policies(policy)
    if strategy_result is not None or diagnostic_end is not None:
        if strategy_result is None or diagnostic_end is None:
            raise ValueError("diagnostic matching needs both strategy_result and diagnostic_end")
        policies["posthoc_diagnostic"] = diagnostic_baseline(strategy_result, policy, evaluation_end=diagnostic_end)
    return {name: (frozen, run_baseline(frozen, instruments, bars, initial_cash=initial_cash, **engine_options))
            for name, frozen in policies.items()}


def comparison_report(strategy_result, baselines, instruments, *, initial_equity_krw, started_at, ended_at):
    """프런티어 그림용 (MDD,CAGR) 좌표와 동시 개선 부호를 포함한 직렬화 가능 자료."""
    started_at, ended_at = _utc(started_at), _utc(ended_at)
    options = dict(initial_equity_krw=initial_equity_krw, started_at=started_at, ended_at=ended_at)
    strategy = performance(strategy_result, instruments, **options)
    times = [_utc(row["at"]) for row in strategy_result.valuations]
    rows = {}
    for name, (policy, result) in baselines.items():
        if [_utc(row["at"]) for row in result.valuations] != times:
            raise ValueError("paired results must have exactly the same evaluation times")
        metrics = performance(result, instruments, **options)
        delta = paired_difference(strategy, metrics)
        rows[name] = {"role": policy.role, "policy_sha256": policy.sha256,
                      "source_sha256": policy.source_sha256, "mode": policy.mode,
                      "target_investment_fraction": policy.investment_fraction,
                      "weights": dict(policy.weights), "performance": asdict(metrics), **delta,
                      "joint_improvement": delta["delta_cagr"] > ZERO and delta["delta_mdd"] > ZERO}
    return {"schema": "etf-m4-v1", "started_at": started_at, "ended_at": ended_at,
            "strategy": asdict(strategy), "baselines": rows, "actual_after_tax_approved": False,
            "statistical_success_evaluated": False}
