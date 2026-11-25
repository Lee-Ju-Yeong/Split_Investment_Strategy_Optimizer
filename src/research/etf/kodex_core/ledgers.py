"""장부 셋(코어·분할·단독 보유)을 R2 엔진 하위 클래스로 실행한다. R2 코드 자체는 바꾸지 않는다."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import timedelta
from decimal import Decimal, localcontext
import inspect

from ..baseline_engine import _BaselineEngine, _validate_run, run_baseline
from ..baseline_policy import BaselinePolicy
from ..engine import _Engine, _Options, _fraction, _validate_options, run_engine
from ..engine_types import Candidate, EngineResult, LevelRule, ONE, ZERO
from ..m5_parameters import CostGuardRejection, entry_rates
from .calendar import KST
from .config import StudyConfig
from .inputs import EtfInputs, VolatilityUnavailable, ZeroVolatility, volatility_plan
from .reentry import ReentrySchedule


@dataclass(frozen=True)
class RunSetting:
    config: StudyConfig
    path: str
    commission: Decimal
    mode: str = "none"  # M / D / none(R2 최초 1회 진입과 같음)


@dataclass
class LedgerRun:
    result: EngineResult | None
    status: str  # ok / cost_guard_start / start_failed / first_buy_failed / volatility_unavailable / observation_ended
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Session:
    """entry_rates가 국내 ETF에서 읽는 두 필드만 가진 주문 시점."""

    session_date: object
    open_at: object


def _defaults(function) -> dict:
    return {name: p.default for name, p in inspect.signature(function).parameters.items()
            if p.default is not inspect.Parameter.empty}


def _options(setting: RunSetting, *, first_entry, first_level_fraction) -> _Options:
    config = setting.config
    options = _Options(setting.path, first_entry, first_level_fraction,
                       _fraction(config.participation_rate, "participation_rate"), ZERO, ZERO, ONE,
                       "pre_tax", False, config.penetration_ticks, config.slippage_ticks,
                       _fraction(setting.commission, "commission_rate"), None, ZERO, "t_plus_2", None, None)
    _validate_options(options)
    return options


def _common(inputs: EtfInputs, setting: RunSetting, capital: Decimal) -> dict:
    times = inputs.calendar.evaluation_times(inputs.start, inputs.end)
    return dict(instruments=[inputs.instrument], bars=list(inputs.bars), initial_cash={"KRW": capital},
                evaluation_times=times, path=setting.path, kr_sessions=list(inputs.calendar.sessions),
                kr_settlement_session_times=inputs.calendar.settlement_times(),
                domestic_settlement="t_plus_2", _evaluation_cutoff=times[-1],
                commission_rate=setting.commission, participation_rate=setting.config.participation_rate,
                penetration_ticks=setting.config.penetration_ticks, slippage_ticks=setting.config.slippage_ticks,
                fx_cost=ZERO, cash_floor_fraction=ZERO, symbol_cap_fraction=ONE)


def split_rates(setting: RunSetting, inputs: EtfInputs, session, estimate):
    """'왕복비용 ≤ kg×변동성 < 100%' 검사와 (drop, take). 실패는 CostGuardRejection."""
    if isinstance(estimate, ZeroVolatility):
        raise CostGuardRejection(f"{inputs.instrument.symbol}: {estimate.reason}")
    candidate = {"kg": setting.config.kg, "kp": setting.config.kp}
    return entry_rates(candidate, None, inputs.instrument, session, estimate, commission_rate=setting.commission)


# 신규 매수 닫힘: 매수가의 1e-24배 이하 가격은 호가상 불가능하므로 하락 조건이 성립할 수 없다(LevelRule은 drop<1).
CLOSED_DROP = Decimal("0.999999999999999999999999")


class _SplitEngine(_Engine):
    """재진입 실패 추적과 연초 변동성 갱신을 더한 R2 엔진.

    갱신 값은 그날 이후 신규 주문(1차·추가 매수 간격, 새 차수 익절률)에만 쓴다. 보유 차수는 Position에
    저장된 매수 당시 익절률을 계속 쓴다. 새해 값이 비용 문턱에 걸리면 신규 매수만 닫고 익절은 허용한다.
    """

    def __init__(self, inputs, options, hooks):
        self.schedule, self.refreshes, self.rates = hooks["schedule"], hooks["refreshes"], hooks["rates"]
        self.current = hooks["initial"]
        self.refresh_log = []
        super().__init__(inputs, options)
        if self.schedule is not None:
            self.schedule.guard = self._reentry_guard

    def _reentry_guard(self, instrument, bar):
        self.rates(_Session(bar.session_date, bar.open_at), self.current)

    def _refresh_budget(self, at, payload):
        super()._refresh_budget(at, payload)
        day = at.astimezone(KST).date()
        if day in self.refreshes:
            self._refresh_volatility(at, day, self.refreshes[day])

    def _set_rates(self, drop, take):
        symbol = next(iter(self.instruments))
        self.rules[symbol] = tuple(replace(rule, drop_rate=drop, take_rate=take) for rule in self.rules[symbol])

    def _refresh_volatility(self, at, day, estimate):
        if isinstance(estimate, VolatilityUnavailable):
            # run_split이 결측 연초 이전에서 실행을 끝내므로 여기 오면 전년 값 거래가 된다.
            raise RuntimeError(f"{day}: annual volatility unavailable inside an active run")
        self.current = estimate
        try:
            drop, take = self.rates(_Session(day, at), estimate)
        except CostGuardRejection as error:
            self._set_rates(CLOSED_DROP, ZERO)
            self._log_refresh(at, day, applied=False, buys_closed=True, reason=str(error))
            return
        self._set_rates(drop, take)
        self._log_refresh(at, day, applied=True, volatility=estimate.value)

    def _log_refresh(self, at, day, **details):
        self.refresh_log.append({"date": day, **details})
        self._cash_event(at, "volatility_refresh", **details)

    def _reentry_order(self, at, bar, candidate):
        if candidate is None:
            self.schedule.failed(bar.session_date, "no_order")
        return super()._reentry_order(at, bar, candidate)

    def _fill(self, at, payload):
        candidate, bar, _ = payload
        before = len(self.result.rejections)
        super()._fill(at, payload)
        if candidate.reason != "monthly_reentry":
            return
        if len(self.result.rejections) > before:
            self.schedule.failed(bar.session_date, self.result.rejections[-1]["reason"])
        else:
            self.schedule.filled(bar.session_date)


def _split_hooks(inputs, setting, rates, plan):
    refreshes = {day: inputs.volatility.get(day, VolatilityUnavailable(f"no estimate for {day}"))
                 for day in plan["refreshes"]}
    schedule = None if setting.mode == "none" else ReentrySchedule(setting.mode, inputs.calendar)
    return {"schedule": schedule, "refreshes": refreshes, "rates": rates,
            "initial": inputs.volatility[plan["start"]]}


def _observation_end(inputs, plan):
    """첫 결측 연초 직전 거래일까지만 분할 계좌가 유효하다. 그 연초 이후로는 전년 값으로 거래하지 않는다."""
    missing = plan["unavailable"][0]
    index = inputs.calendar.sessions.index(missing["date"])
    return {"last_valid_session": inputs.calendar.sessions[index - 1], "missing_volatility_date": missing["date"],
            "reason": missing["reason"]}


def _until(inputs, last_day):
    return replace(inputs, bars=tuple(bar for bar in inputs.bars if bar.session_date <= last_day))


def run_split(inputs: EtfInputs, setting: RunSetting) -> LedgerRun:
    """시간순: 시작 변동성 → 시작 비용 검사 → 첫 주문과 그 뒤 거래. 미래의 연초 결측은 그 직전까지만 관측을
    끊을 뿐, 이미 확정된 시작 판정·첫 주문·이전 거래를 바꾸지 않는다(B2)."""
    config, first = setting.config, inputs.bars[0]
    plan = volatility_plan(inputs, inputs.start, config.volatility_mode)
    if any(row["date"] == plan["start"] for row in plan["unavailable"]):
        return LedgerRun(None, "volatility_unavailable", {"unavailable": plan["unavailable"]})

    def rates(session, estimate):
        return split_rates(setting, inputs, session, estimate)

    try:
        drop, take = rates(first, inputs.volatility[plan["start"]])
    except CostGuardRejection as error:
        return LedgerRun(None, "cost_guard_start", {"reason": str(error)})
    ended = _observation_end(inputs, plan) if plan["unavailable"] else None
    observed = inputs if ended is None else _until(inputs, ended["last_valid_session"])
    if ended is not None:
        plan = {**plan, "refreshes": [day for day in plan["refreshes"] if day <= ended["last_valid_session"]]}
    rules = {inputs.instrument.symbol: tuple(LevelRule(n, ONE, drop, take) for n in range(1, config.levels + 1))}
    hooks = _split_hooks(observed, setting, rates, plan)
    with localcontext(prec=80):  # R2 followup_run과 같이 차수 비율 1/9를 80자리로 만든다.
        fraction = _fraction(ONE / config.levels, "first_level_fraction")
        engine_inputs = dict(_defaults(run_engine), **_common(observed, setting, config.capital["split"]),
                             rules=rules, first_level_fraction=fraction, _reentry_provider=hooks["schedule"])
        options = _options(setting, first_entry=True, first_level_fraction=fraction)
        engine = _SplitEngine(engine_inputs, options, hooks)
        result = engine.run()
    failed = any(row["kind"] == "first_entry_unavailable" for row in result.cash_events)
    detail = {"rates": (drop, take), "volatility_start": plan["start"], "refreshes": engine.refresh_log,
              "reentries": [] if hooks["schedule"] is None else hooks["schedule"].attempts}
    if failed:
        return LedgerRun(result, "start_failed", detail)
    if ended is not None:
        return LedgerRun(result, "observation_ended", {**detail, **ended})
    return LedgerRun(result, "ok", detail)


class _HoldEngine(_BaselineEngine):
    """단독 보유·코어: 첫 시가 최대 정수 주 매수 후 보유. 이 장부가 실제로 분배금을 받은 경우에만
    지급 다음 국내 거래일 시가에 한 번 재투자한다(권리 없는 잔돈으로는 사지 않는다)."""

    def __init__(self, inputs, options, policy, calendar):
        self.calendar, self.reinvest_days = calendar, set()
        super().__init__(inputs, options, policy)

    def _dividend_pay(self, at, key):
        symbol = self.receivables[key]["symbol"]
        super()._dividend_pay(at, key)
        paid = self.result.cash_events[-1]
        if paid["kind"] != "dividend_payment" or paid["amount"] <= ZERO:
            return
        day = self.calendar.next_session(at.astimezone(KST).date())
        if day is not None:
            self.reinvest_days.add(day)
            self.trade_sessions.add((symbol, day))

    def _order(self, at, bar):
        if bar.session_date not in self.reinvest_days:
            return super()._order(at, bar)
        instrument = self.instruments[bar.symbol]
        price = self._open_price(instrument, bar, "BUY")
        quantity = 0 if price is None else self._buy_quantity(bar, price, self.cash[instrument.currency])
        if quantity == 0:
            return None
        level = self.next_level.get(bar.symbol, 0) + 1
        return Candidate(bar.symbol, "BUY", level, quantity, price, at, "dividend_reinvest")


def _hold_policy(inputs: EtfInputs, fraction=ONE, mode="buy_hold") -> BaselinePolicy:
    start = inputs.bars[0].open_at
    return BaselinePolicy(fraction, ((inputs.instrument.symbol, ONE),), start - timedelta(seconds=1), start,
                          "0" * 64, role="reference", mode=mode)


def run_hold(inputs: EtfInputs, setting: RunSetting, capital: Decimal) -> LedgerRun:
    policy = _hold_policy(inputs)
    common = _common(inputs, setting, capital)
    _validate_run(policy, common["instruments"], common["bars"], common["evaluation_times"], (), None)
    engine_inputs = dict(_defaults(run_baseline), **common, policy=policy, _cap_buy_to_cash=False,
                         rules={inputs.instrument.symbol: (LevelRule(1, ONE, ZERO, ZERO),)},
                         initial_positions=(), initial_tax_basis=None, _candidate_provider=None)
    with localcontext(prec=80):
        result = _HoldEngine(engine_inputs, _options(setting, first_entry=False, first_level_fraction=None),
                             policy, inputs.calendar).run()
    first = [row for row in result.trades if row["reason"] == "baseline_buy_hold"]
    ok = bool(first) and first[0]["session_date"] == inputs.start
    return LedgerRun(result, "ok" if ok else "first_buy_failed", {"first_trade": first[0] if first else None})


def run_matched(inputs: EtfInputs, setting: RunSetting, capital: Decimal, fraction: Decimal) -> EngineResult:
    """보조 기준선: 같은 실행의 전략 평균 투자 비중으로 매월 첫 국내 거래일 ETF+현금 재조정(참고값)."""
    common = _common(inputs, setting, capital)
    return run_baseline(_hold_policy(inputs, fraction, "monthly"), common.pop("instruments"), common.pop("bars"),
                        **common)
