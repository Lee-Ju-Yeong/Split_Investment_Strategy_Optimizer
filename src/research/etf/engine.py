"""Decimal ETF reference engine with an explicit UTC event queue.

Intraday paths are assumptions, not observed ticks.  A path-derived order is
planned at the open and committed only at Candidate.at.  Only prices revealed
by an open, assumed path vertex, fill, or close can mark the portfolio.

``us_sessions`` must be actual NYSE session dates, including settlement tails.
``settlement_session_times`` supplies the explicit cash-availability assumption
for each settlement date; no weekday or wall-clock fallback is invented.
``tax_basis_by_bar`` is keyed by (symbol, session_date), ``initial_tax_basis``
by (symbol, level), and ``tax_payments`` by aware payment time (KRW amounts).
After-tax runs are synthetic checks and require ``synthetic_tax_data=True``.
All extra options are named in run_engine; unknown options raise TypeError.

Missing dividend pay dates and unsettled dividend entitlements use the engine's
own pre-tax receivable ledger, without a fabricated TaxLedger payment.  The
corresponding synthetic after-tax run fails.  Raw split events are unsupported;
split-adjusted quotes already include the split and do not change quantities.

Currency conversion uses cash, holdings and FX known at the conversion time.
``equity_krw``/``total_krw`` deduct finalized unpaid tax only, and remain the
budget/limit decision basis.  Current-year realized-gain ``estimated_tax_krw``
is reporting-only: ``equity_after_estimated_tax_krw`` is the corresponding
synthetic after-tax reporting value.  No future gains or year-end tax enter
order signals.  A missed initial entry is recorded and is never retried.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
import heapq

from .engine_market import fee_amount, sec_fee_rate
from .engine_strategy import candidate_for_bar
from .engine_tax import TaxLedger
from .engine_types import (
    Bar, Candidate, EngineResult, FxRate, Instrument, LevelRule, Position,
    ONE, ZERO, aware, decimal,
)


UTC = timezone.utc


def _utc(at: datetime) -> datetime:
    return aware(at).astimezone(UTC)


def _fraction(value, name: str, *, positive=False) -> Decimal:
    value = decimal(value)
    if not ZERO <= value <= ONE or (positive and value == ZERO):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}at most 1")
    return value


@dataclass(frozen=True)
class _Options:
    path: str
    first_entry: bool
    first_level_fraction: Decimal | None
    participation_rate: Decimal
    fx_cost: Decimal
    cash_floor_fraction: Decimal
    symbol_cap_fraction: Decimal
    tax_mode: str
    synthetic_tax_data: bool
    penetration_ticks: int
    slippage_ticks: int
    commission_rate: Decimal | None
    sec_rate_override: Decimal | None
    product_sell_tax_rate: Decimal


def _validate_options(options: _Options) -> None:
    if options.path not in {"O-H-L-C", "O-L-H-C"}:
        raise ValueError("path must be O-H-L-C or O-L-H-C")
    if options.tax_mode not in {"pre_tax", "after_tax"}:
        raise ValueError("tax_mode must be pre_tax or after_tax")
    if options.tax_mode == "after_tax" and not options.synthetic_tax_data:
        raise ValueError("after_tax requires synthetic_tax_data=True; actual after-tax judgment is deferred")
    for name in ("first_entry", "synthetic_tax_data"):
        if type(getattr(options, name)) is not bool:
            raise ValueError(f"{name} must be bool")
    for name in ("penetration_ticks", "slippage_ticks"):
        if type(getattr(options, name)) is not int or getattr(options, name) < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if options.fx_cost == ONE:
        raise ValueError("fx_cost must be less than 1")


def _input_map(instruments, rules):
    by_symbol = {item.symbol: item for item in instruments}
    if not by_symbol or len(by_symbol) != len(instruments):
        raise ValueError("instruments must have unique symbols and must not be empty")
    if set(rules) != set(by_symbol):
        raise ValueError("rules must cover exactly the instrument symbols")
    for symbol, levels in rules.items():
        numbers = sorted(rule.level for rule in levels)
        if numbers != list(range(1, len(numbers) + 1)) or not numbers:
            raise ValueError(f"{symbol}: rules must contain unique contiguous levels starting at 1")
    return by_symbol


def _validate_bars(bars, instruments) -> None:
    seen, previous = set(), {}
    for bar in sorted(bars, key=lambda item: _utc(item.open_at)):
        if bar.symbol not in instruments or type(bar.session_date) is not date:
            raise ValueError("bar has an unknown symbol or invalid session_date")
        key = (bar.symbol, bar.session_date)
        if key in seen or previous.get(bar.symbol, _utc(bar.open_at)) > _utc(bar.open_at):
            raise ValueError(f"{bar.symbol}: duplicate or overlapping sessions")
        if bar.adv20 is not None and bar.adv20 < ZERO:
            raise ValueError("adv20 must be nonnegative")
        seen.add(key)
        previous[bar.symbol] = _utc(bar.close_at)


def _validated_sessions(us_sessions, settlement_times):
    sessions = list(us_sessions)
    if any(type(day) is not date for day in sessions) or len(set(sessions)) != len(sessions):
        raise ValueError("us_sessions must contain unique actual session dates")
    sessions.sort()
    times = {day: _utc(at) for day, at in (settlement_times or {}).items()}
    if any(day not in sessions for day in times):
        raise ValueError("settlement_session_times keys must belong to us_sessions")
    ordered_times = [times[day] for day in sessions if day in times]
    if ordered_times != sorted(ordered_times) or len(set(ordered_times)) != len(ordered_times):
        raise ValueError("settlement availability times must increase with session dates")
    return sessions, times


def _tax_basis_map(values, instruments, *, initial=False):
    result = {}
    for key, value in (values or {}).items():
        if not isinstance(key, tuple) or len(key) != 2 or key[0] not in instruments:
            raise ValueError("tax basis keys must be (known symbol, date/level)")
        valid = type(key[1]) is int and key[1] > 0 if initial else type(key[1]) is date
        if not valid or decimal(value) < ZERO:
            raise ValueError("invalid tax basis date/level or negative amount")
        result[key] = decimal(value)
    return result


class _Engine:
    def __init__(self, inputs, options):
        self.options = options
        self.candidate_provider = inputs["_candidate_provider"]
        self.instruments = _input_map(inputs["instruments"], inputs["rules"])
        self.order = {symbol: index for index, symbol in enumerate(self.instruments)}
        self.rules = {symbol: tuple(sorted(levels, key=lambda rule: rule.level))
                      for symbol, levels in inputs["rules"].items()}
        self.bars = list(inputs["bars"])
        _validate_bars(self.bars, self.instruments)
        self._init_cash(inputs["initial_cash"])
        self._init_fx(inputs["fx_rates"])
        self._init_positions(inputs["initial_positions"])
        self.sessions, self.settlement_times = _validated_sessions(
            inputs["us_sessions"], inputs["settlement_session_times"])
        self.basis = _tax_basis_map(inputs["tax_basis_by_bar"], self.instruments)
        self.initial_basis = _tax_basis_map(inputs["initial_tax_basis"], self.instruments, initial=True)
        if set(self.basis) - {(bar.symbol, bar.session_date) for bar in self.bars}:
            raise ValueError("tax_basis_by_bar references a missing bar")
        if set(self.initial_basis) - set(self.positions):
            raise ValueError("initial_tax_basis references a missing initial position")
        self.result, self.queue, self.serial = EngineResult(), [], 0
        self.pending_cash, self.receivables, self.pending_tax = [], {}, []
        self.first_seen, self.attempted = set(), set()
        self.last_valuation = None
        self.tax = TaxLedger(mode=options.tax_mode, synthetic=options.synthetic_tax_data)
        self._build_queue(inputs)

    def _init_cash(self, initial):
        if set(initial) - {"KRW", "USD"}:
            raise ValueError("initial_cash supports only KRW and USD")
        self.cash = {currency: decimal(initial.get(currency, ZERO)) for currency in ("KRW", "USD")}
        if min(self.cash.values()) < ZERO:
            raise ValueError("initial_cash must be nonnegative")
        self.settled = self.cash.copy()

    def _init_fx(self, rates):
        ordered = sorted(rates, key=lambda item: _utc(item.available_at))
        self.fx_times = [_utc(rate.available_at) for rate in ordered]
        if len(set(self.fx_times)) != len(self.fx_times):
            raise ValueError("fx_rates must have unique availability times")
        self.fx_values = [rate.krw_per_usd for rate in ordered]

    def _init_positions(self, positions):
        self.positions, self.marks = {}, {}
        for position in positions:
            key = (position.symbol, position.level)
            if position.symbol not in self.instruments or key in self.positions:
                raise ValueError("initial positions have unknown or duplicate symbol/level")
            if position.level not in {rule.level for rule in self.rules[position.symbol]}:
                raise ValueError("initial position level has no rule")
            self.positions[key] = position
            previous = self.marks.get(position.symbol)
            if previous is None or previous[1] < _utc(position.opened_at):
                self.marks[position.symbol] = (position.entry_price, _utc(position.opened_at))

    def _push(self, at, priority, kind, payload=None, symbol=None):
        self.serial += 1
        heapq.heappush(self.queue, (_utc(at), priority, self.order.get(symbol, -1),
                                   self.serial, kind, payload))

    def _build_queue(self, inputs):
        for at in inputs["evaluation_times"]:
            self._push(at, 90, "evaluate")
        for at in inputs["conversion_times"]:
            self._push(at, 25, "convert")
        for at, amount in (inputs["tax_payments"] or {}).items():
            if decimal(amount) < ZERO:
                raise ValueError("tax_payments must be nonnegative KRW amounts")
            self._push(at, 25, "tax_payment", decimal(amount))
        self._bar_events()
        self._initial_events()

    def _bar_events(self):
        monthly = {}
        for bar in sorted(self.bars, key=lambda item: (_utc(item.open_at), self.order[item.symbol])):
            at = _utc(bar.open_at)
            monthly.setdefault((bar.session_date.year, bar.session_date.month), at)
            self._push(at, 10, "corporate", bar, bar.symbol)
            self._path_marks(bar)
            self._push(at, 30, "plan", bar, bar.symbol)
        if self.options.first_level_fraction is not None:
            for at in monthly.values():
                self._push(at, 15, "budget")

    def _path_marks(self, bar):
        names = self.options.path.split("-")
        fields = {"O": bar.open, "H": bar.high, "L": bar.low, "C": bar.close}
        duration = _utc(bar.close_at) - _utc(bar.open_at)
        for index, name in enumerate(names):
            at = _utc(bar.open_at) + duration * index / 3
            self._push(at, 20, "mark", (bar.symbol, fields[name]), bar.symbol)

    def _initial_events(self):
        if not self.queue and not self.positions:
            return
        start = self.queue[0][0] if self.queue else max(_utc(p.opened_at) for p in self.positions.values())
        for position in self.positions.values():
            if _utc(position.opened_at) > start:
                raise ValueError("initial positions cannot be opened after the first engine event")
            self._push(start, 0, "initial_tax", position, position.symbol)
        self._validate_corporate_inputs()

    def _validate_corporate_inputs(self):
        if any(bar.dividend and bar.dividend_pay_at is None for bar in self.bars):
            if self.options.tax_mode == "after_tax":
                raise ValueError("after_tax requires actual dividend_pay_at; no fictitious payment date is allowed")
            self._warn("Dividend payment dates are missing: receivables remain unpaid; actual after-tax judgment is unavailable")
        for bar in self.bars:
            if bar.dividend_pay_at is not None and _utc(bar.dividend_pay_at) < _utc(bar.open_at):
                raise ValueError("dividend payment cannot precede ex-date open")
            instrument = self.instruments[bar.symbol]
            if bar.dividend and instrument.price_basis in {
                    "total_return", "total_return_adjusted", "dividend_adjusted", "adjusted_close"}:
                raise ValueError("cash dividends cannot be added to dividend-adjusted prices again")
            if "distribution_unverified" in instrument.price_basis:
                self._warn(f"{bar.symbol}: price-basis distribution treatment is unverified; pre-tax results have a corporate-action limitation")

    def _warn(self, message):
        if message not in self.result.warnings:
            self.result.warnings.append(message)

    def _fx(self, at):
        index = bisect_right(self.fx_times, _utc(at)) - 1
        if index < 0:
            raise ValueError(f"no FX rate published by {_utc(at).isoformat()}")
        return self.fx_values[index]

    def _rate(self, instrument, at):
        return self._fx(at) if instrument.currency == "USD" else ONE

    def _settlement(self, bar):
        day = bar.session_date
        lag = 1 if day >= date(2024, 5, 28) else 2 if day >= date(2017, 9, 5) else 3
        index = bisect_left(self.sessions, day)
        if index == len(self.sessions) or self.sessions[index] != day or index + lag >= len(self.sessions):
            self._warn(f"{bar.symbol} {day}: insufficient actual US sessions; cash/tax settlement remains pending")
            return None, None
        settled_day = self.sessions[index + lag]
        at = self.settlement_times.get(settled_day)
        if at is None:
            self._warn(f"{bar.symbol} {day}: no availability time for settlement {settled_day}; cash/tax settlement remains pending")
        elif at < _utc(bar.close_at):
            raise ValueError("settlement availability cannot precede the trade session close")
        return settled_day, at

    def _sec_override(self, instrument, bar):
        if instrument.market != "US":
            return self.options.sec_rate_override
        if self.options.sec_rate_override is not None:
            return self.options.sec_rate_override
        day, _ = self._settlement(bar)
        if day is None:
            raise ValueError("US fee calculation needs actual settlement sessions or an explicit sec_rate_override")
        return sec_fee_rate(day, known_at=bar.open_at)

    def run(self):
        handlers = {"mark": self._mark, "corporate": self._corporate,
                    "budget": self._refresh_budget, "plan": self._plan,
                    "fill": self._fill, "evaluate": self._evaluate,
                    "convert": self._convert, "settle": self._settle,
                    "initial_tax": self._initial_tax, "trade_tax": self._trade_tax,
                    "dividend_pay": self._dividend_pay, "tax_payment": self._tax_payment}
        while self.queue:
            at, _, _, _, kind, payload = heapq.heappop(self.queue)
            self.tax.advance(at)
            handlers[kind](at, payload)
        self.result.final_cash = self.cash.copy()
        self.result.positions = list(self.positions.values())
        self.result.tax = self.tax.snapshot()
        self.result.tax["pending_settlement_events"] = list(self.pending_tax)
        self.result.tax["engine_receivables"] = {key: item.copy() for key, item in self.receivables.items()}
        if self.result.cash_events:
            self._cash_event(at, "final_state", positions=self.result.positions.copy(),
                             unsettled_cash={currency: self.cash[currency] - self.settled[currency]
                                             for currency in self.cash})
        return self.result

    def _mark(self, at, payload):
        symbol, price = payload
        self.marks[symbol] = (price, at)

    def _values(self, at):
        needs_fx = self.cash["USD"] != ZERO or any(
            self.instruments[p.symbol].currency == "USD" for p in self.positions.values())
        needs_fx = needs_fx or any(item["currency"] == "USD" for item in self.receivables.values())
        fx = self._fx(at) if needs_fx else ONE
        holdings = {symbol: ZERO for symbol in self.instruments}
        for position in self.positions.values():
            price, known_at = self.marks[position.symbol]
            if known_at > at:
                raise ValueError("future initial-position mark cannot be used")
            instrument = self.instruments[position.symbol]
            holdings[position.symbol] += price * position.quantity * (fx if instrument.currency == "USD" else ONE)
        cash_krw = self.cash["KRW"] + self.cash["USD"] * fx
        receivable = sum((item["net"] * (fx if item["currency"] == "USD" else ONE)
                          for item in self.receivables.values()), ZERO)
        tax_summary = self.tax.valuation_summary()
        liability = tax_summary["unpaid_tax_krw"]
        equity = cash_krw + sum(holdings.values(), ZERO) + receivable - liability
        return {"total_krw": equity, "equity_krw": equity, "tax_liability_krw": liability,
                "equity_before_estimated_tax_krw": equity,
                "estimated_tax_krw": tax_summary["estimated_tax_krw"],
                "equity_after_estimated_tax_krw": equity - tax_summary["estimated_tax_krw"],
                "cash_krw": cash_krw, "holdings_krw": holdings,
                "receivables_krw": receivable, "fx_rate": fx}

    def _evaluate(self, at, _):
        values = self._values(at)
        row = {"at": at, **values, "cash": self.cash.copy(),
               "settled_cash": self.settled.copy(),
               "prices": {symbol: price for symbol, (price, known) in self.marks.items() if known <= at},
               "price_times": {symbol: known for symbol, (_, known) in self.marks.items() if known <= at},
               "positions": list(self.positions.values())}
        self.result.valuations.append(row)
        self.last_valuation = row

    def _prior_values(self, at, *, require_prior=False):
        if self.last_valuation is not None and self.last_valuation["at"] < at:
            return self.last_valuation
        if self.result.valuations:
            prior = [row for row in self.result.valuations if row["at"] < at]
            if prior:
                return prior[-1]
        if require_prior:
            raise ValueError("monthly refresh requires an explicit preceding evaluation")
        return self._values(at)

    def _refresh_budget(self, at, _):
        previous = self._prior_values(at, require_prior=hasattr(self, "budget_initialized"))
        amount = previous["total_krw"] * self.options.first_level_fraction / len(self.instruments)
        if amount <= ZERO:
            raise ValueError("monthly level budget must be positive")
        budgets = {}
        for symbol, instrument in self.instruments.items():
            budget = amount / self._rate(instrument, at)
            self.rules[symbol] = tuple(replace(rule, budget=budget) for rule in self.rules[symbol])
            budgets[symbol] = budget
        self.budget_initialized = True
        self._cash_event(at, "budget_refresh", budgets=budgets, source_total_krw=previous["total_krw"])

    def _plan(self, at, bar):
        instrument = self.instruments[bar.symbol]
        first = bar.symbol not in self.first_seen
        self.first_seen.add(bar.symbol)
        positions = [p for p in self.positions.values() if p.symbol == bar.symbol]
        manual_entry = self.options.first_entry and first and not positions
        candidate = self.candidate_provider(
            instrument, bar, positions, self.rules[bar.symbol], path=self.options.path,
            first_entry=self.options.first_entry and first,
            penetration_ticks=self.options.penetration_ticks, slippage_ticks=self.options.slippage_ticks,
            commission_rate=self.options.commission_rate,
            sec_rate_override=self._sec_override(instrument, bar),
            product_sell_tax_rate=self.options.product_sell_tax_rate)
        if candidate is not None:
            if not at <= _utc(candidate.at) <= _utc(bar.close_at) or candidate.symbol != bar.symbol:
                raise ValueError("strategy candidate lies outside its own session")
            self._push(candidate.at, 40, "fill", (candidate, bar, at), bar.symbol)
        elif manual_entry:
            reason = "zero_volume" if bar.volume == ZERO else "limit_locked" if bar.limit_locked else "zero_affordable_quantity"
            self._entry_unavailable(at, bar, reason)

    def _entry_unavailable(self, at, bar, reason):
        self._warn(f"{bar.symbol} {bar.session_date}: initial entry unavailable ({reason}); initial entry is not retried")
        self._cash_event(at, "first_entry_unavailable", symbol=bar.symbol,
                         session_date=bar.session_date, reason=reason, retry=False)

    def _cash_event(self, at, kind, **details):
        self.result.cash_events.append({"at": at, "kind": kind, **details,
                                        "cash": self.cash.copy(), "settled_cash": self.settled.copy()})

    def _reject(self, at, candidate, reason, generated_at):
        self.result.rejections.append({"at": at, "generated_at": generated_at,
                                       "symbol": candidate.symbol, "side": candidate.side,
                                       "level": candidate.level, "quantity": candidate.quantity,
                                       "price": candidate.price, "reason": reason})

    def _fill(self, at, payload):
        candidate, bar, generated_at = payload
        key = (candidate.symbol, bar.session_date)
        if key in self.attempted:
            raise ValueError("more than one order attempt in a symbol/session")
        self.attempted.add(key)
        self.marks[candidate.symbol] = (candidate.price, at)
        reason = self._fill_rejection(at, candidate, bar)
        if reason:
            self._reject(at, candidate, reason, generated_at)
            if candidate.reason == "manual_first_entry":
                self._entry_unavailable(at, bar, reason)
            return
        self._execute(at, candidate, bar, generated_at)

    def _fee(self, candidate, bar):
        instrument = self.instruments[candidate.symbol]
        notional = candidate.price * candidate.quantity
        fee = fee_amount(instrument.market, candidate.side, bar.session_date, notional,
                         commission_rate=self.options.commission_rate,
                         sec_rate_override=self._sec_override(instrument, bar))
        product_tax = notional * self.options.product_sell_tax_rate if candidate.side == "SELL" else ZERO
        return fee, product_tax

    def _fill_rejection(self, at, candidate, bar):
        key = (candidate.symbol, candidate.level)
        if candidate.side == "SELL":
            position = self.positions.get(key)
            if position is None or position.quantity != candidate.quantity:
                return "position_changed"
        elif key in self.positions:
            return "level_occupied"
        if bar.adv20 is None or bar.adv20 <= ZERO:
            return "adv20_unavailable"
        if candidate.price * candidate.quantity > bar.adv20 * self.options.participation_rate:
            return "participation_limit"
        if candidate.side == "BUY":
            return self._buy_rejection(at, candidate, bar)
        return None

    def _buy_rejection(self, at, candidate, bar):
        instrument = self.instruments[candidate.symbol]
        fee, _ = self._fee(candidate, bar)
        cost = candidate.price * candidate.quantity + fee
        budget = next(rule.budget for rule in self.rules[candidate.symbol] if rule.level == candidate.level)
        if cost > budget:
            return "level_budget_exceeded"
        if cost > self.cash[instrument.currency]:
            return "insufficient_cash"
        if self.options.cash_floor_fraction == ZERO and self.options.symbol_cap_fraction == ONE:
            return None
        values = self._values(at)
        rate = self._rate(instrument, at)
        post_total = values["total_krw"] - fee * rate
        if values["cash_krw"] - cost * rate < post_total * self.options.cash_floor_fraction:
            return "cash_floor"
        if values["holdings_krw"][candidate.symbol] + candidate.price * candidate.quantity * rate > post_total * self.options.symbol_cap_fraction:
            return "symbol_cap"
        return None

    def _execute(self, at, candidate, bar, generated_at):
        instrument = self.instruments[candidate.symbol]
        fee, product_tax = self._fee(candidate, bar)
        key = (candidate.symbol, candidate.level)
        if candidate.side == "BUY":
            rule = next(rule for rule in self.rules[candidate.symbol] if rule.level == candidate.level)
            position = Position(candidate.symbol, candidate.level, candidate.quantity, candidate.price, fee, rule.take_rate, at)
            self.positions[key] = position
            self._spend(instrument.currency, position.principal + fee)
        else:
            position = self.positions.pop(key)
        withholding = self._queue_trade_tax(at, instrument, position, candidate, bar, fee)
        if candidate.side == "SELL":
            self._credit_sale(at, instrument, candidate.price * candidate.quantity - fee - product_tax - withholding, bar)
        row = {"at": at, "generated_at": generated_at, "session_date": bar.session_date,
               "symbol": candidate.symbol, "side": candidate.side, "level": candidate.level,
               "quantity": candidate.quantity, "price": candidate.price, "fee": fee,
               "product_sell_tax": product_tax, "withholding": withholding,
               "currency": instrument.currency, "reason": candidate.reason}
        self.result.trades.append(row)
        self._cash_event(at, "trade", **{key: value for key, value in row.items() if key != "at"})

    def _spend(self, currency, amount):
        self.cash[currency] -= amount
        remaining = amount
        for credit in self.pending_cash:
            if credit["currency"] == currency:
                used = min(credit["remaining"], remaining)
                credit["remaining"] -= used
                remaining -= used
        self.settled[currency] -= remaining
        if self.settled[currency] < ZERO:
            raise ValueError("settled-cash accounting became negative")

    def _credit_sale(self, at, instrument, amount, bar):
        if amount < ZERO:
            raise ValueError("sale costs exceed proceeds")
        self.cash[instrument.currency] += amount
        if instrument.market != "US":
            self.settled[instrument.currency] += amount
            return
        day, settled_at = self._settlement(bar)
        credit = {"currency": "USD", "remaining": amount, "settlement_date": day, "settled_at": settled_at}
        self.pending_cash.append(credit)
        if settled_at is not None:
            self._push(settled_at, 5, "settle", credit)

    def _settle(self, at, credit):
        self.settled[credit["currency"]] += credit["remaining"]
        self.pending_cash.remove(credit)
        self._cash_event(at, "settlement", currency=credit["currency"], amount=credit["remaining"],
                         settlement_date=credit["settlement_date"])

    def _initial_tax(self, at, position):
        instrument = self.instruments[position.symbol]
        self.tax.buy(instrument, position, fx_rate=self._rate(instrument, position.opened_at),
                     tax_basis=self.initial_basis.get((position.symbol, position.level)), settled_at=at)

    def _queue_trade_tax(self, at, instrument, position, candidate, bar, fee):
        payload = (instrument, position, candidate, bar, fee)
        if instrument.market != "US":
            return self._trade_tax(at, payload)
        day, settled_at = self._settlement(bar)
        if settled_at is not None:
            self._push(settled_at, 6, "trade_tax", payload, instrument.symbol)
        else:
            self.pending_tax.append({"symbol": instrument.symbol, "side": candidate.side,
                                     "trade_at": at, "settlement_date": day})
        return ZERO

    def _trade_tax(self, at, payload):
        instrument, position, candidate, bar, fee = payload
        basis = self.basis.get((instrument.symbol, bar.session_date))
        rate = self._rate(instrument, at)
        if candidate.side == "BUY":
            self.tax.buy(instrument, position, fx_rate=rate, tax_basis=basis, settled_at=at)
            return ZERO
        return self.tax.sell(instrument, position, candidate.price, fee, at,
                             fx_rate=rate, tax_basis=basis, settled_at=at)

    def _corporate(self, at, bar):
        instrument = self.instruments[bar.symbol]
        if bar.split_ratio != ONE:
            if instrument.price_basis in {"split_adjusted", "split-adjusted",
                                           "yahoo_split_adjusted_quote_with_cash_dividends"}:
                self._warn(f"{bar.symbol}: split-adjusted prices; split_ratio was not applied to quantities again")
            else:
                raise ValueError("raw split events require matching tax-lot split support; use split-adjusted input")
        if bar.dividend:
            self._accrue_dividend(at, bar)

    def _accrue_dividend(self, at, bar):
        quantity = sum(p.quantity for p in self.positions.values() if p.symbol == bar.symbol)
        if not quantity:
            return
        instrument = self.instruments[bar.symbol]
        pay_at = _utc(bar.dividend_pay_at) if bar.dividend_pay_at is not None else None
        if pay_at is not None and pay_at < at:
            raise ValueError("dividend payment cannot precede ex-date open")
        key = f"{bar.symbol}:{bar.session_date.isoformat()}"
        item = {"symbol": bar.symbol, "currency": instrument.currency, "gross": bar.dividend * quantity,
                "net": bar.dividend * quantity, "pay_at": pay_at, "ex_at": at,
                "basis": self.basis.get((bar.symbol, bar.session_date)), "tax_registered": False}
        self.receivables[key] = item
        settled_quantity = self.tax.snapshot()["holdings"].get(bar.symbol, {}).get("quantity", 0)
        if pay_at is not None and settled_quantity == quantity:
            self.tax.accrue_dividend(instrument, item["gross"], at, pay_at,
                                    tax_basis=item["basis"], dividend_id=key)
            item["tax_registered"] = True
            item["net"] = self.tax.snapshot()["receivables"][key]["net"]
        elif self.options.tax_mode == "after_tax":
            raise ValueError("dividend entitlement and settled tax quantity differ; synthetic after_tax is unsupported")
        else:
            self._warn(f"{bar.symbol}: dividend recorded only in pre-tax engine receivables; payment date or settled tax quantity is unavailable")
        self._cash_event(at, "dividend_receivable", receivable_id=key, **item)
        if pay_at is not None:
            self._push(pay_at, 22, "dividend_pay", key, bar.symbol)

    def _dividend_pay(self, at, key):
        item = self.receivables.pop(key)
        instrument = self.instruments[item["symbol"]]
        tax = self.tax.dividend(instrument, item["gross"], at, receivable_id=key,
                                fx_rate=self._rate(instrument, at), tax_basis=item["basis"]) if item["tax_registered"] else ZERO
        amount = item["gross"] - tax
        self.cash[instrument.currency] += amount
        self.settled[instrument.currency] += amount
        self._cash_event(at, "dividend_payment", receivable_id=key, currency=instrument.currency,
                         gross=item["gross"], withholding=tax, amount=amount)

    def _convert(self, at, _):
        values = self._values(at)
        rate = self._fx(at)
        usd_share = Decimal(sum(instrument.currency == "USD" for instrument in self.instruments.values())) / len(self.instruments)
        usd_holdings = sum((values["holdings_krw"][symbol] for symbol, instrument in self.instruments.items()
                            if instrument.currency == "USD"), ZERO)
        target = max(ZERO, values["total_krw"] * usd_share - usd_holdings) / rate
        difference = target - self.cash["USD"]
        if difference > ZERO:
            self._convert_to_usd(at, difference, rate)
        elif difference < ZERO:
            self._convert_to_krw(at, -difference, rate)
        else:
            self._cash_event(at, "fx_conversion", from_currency=None, amount=ZERO, fx_rate=rate, fee_krw=ZERO)

    def _convert_to_usd(self, at, desired, rate):
        amount = min(desired, self.settled["KRW"] / (rate * (ONE + self.options.fx_cost)))
        cost = amount * rate * (ONE + self.options.fx_cost)
        self.cash["KRW"] -= cost
        self.settled["KRW"] -= cost
        self.cash["USD"] += amount
        self.settled["USD"] += amount
        self._cash_event(at, "fx_conversion", from_currency="KRW", to_currency="USD", amount=amount,
                         desired_amount=desired, fx_rate=rate, fee_krw=amount * rate * self.options.fx_cost)

    def _convert_to_krw(self, at, desired, rate):
        amount = min(desired, self.settled["USD"])
        proceeds = amount * rate * (ONE - self.options.fx_cost)
        self.cash["USD"] -= amount
        self.settled["USD"] -= amount
        self.cash["KRW"] += proceeds
        self.settled["KRW"] += proceeds
        self._cash_event(at, "fx_conversion", from_currency="USD", to_currency="KRW", amount=amount,
                         desired_amount=desired, fx_rate=rate, fee_krw=amount * rate * self.options.fx_cost)
        if amount < desired:
            self._warn("Unsettled USD sale proceeds cannot be converted")

    def _tax_payment(self, at, amount):
        if amount > self.cash["KRW"]:
            raise ValueError("insufficient settled KRW cash for explicit tax payment")
        self.tax.pay_tax(amount, at)
        self._spend("KRW", amount)
        self._cash_event(at, "tax_payment", currency="KRW", amount=amount)


def run_engine(
    instruments: list[Instrument], bars: list[Bar], rules: dict[str, tuple[LevelRule, ...]], *,
    initial_cash: dict[str, Decimal], fx_rates: list[FxRate] = (),
    evaluation_times: list[datetime] = (), conversion_times: list[datetime] = (),
    path="O-H-L-C", first_entry=True, initial_positions: list[Position] = (),
    first_level_fraction: Decimal | None = None, participation_rate=Decimal(".05"),
    fx_cost=Decimal(".001"), cash_floor_fraction=ZERO, symbol_cap_fraction=ONE,
    tax_mode="pre_tax", us_sessions: list[date] = (),
    settlement_session_times: dict[date, datetime] | None = None,
    tax_basis_by_bar: dict[tuple[str, date], Decimal] | None = None,
    initial_tax_basis: dict[tuple[str, int], Decimal] | None = None,
    tax_payments: dict[datetime, Decimal] | None = None, synthetic_tax_data=False,
    penetration_ticks=1, slippage_ticks=0, commission_rate=None,
    sec_rate_override=None, product_sell_tax_rate=ZERO, _candidate_provider=candidate_for_bar,
) -> EngineResult:
    """Run one assumed OHLC path; cash orders never shrink to available cash.

    The caller must validate bars against the complete intended date range and
    actual sessions of every market used.  This API processes only the supplied
    Bar event queue; it does not verify market-session completeness.  M3/M4 runs
    on real data must use the strict M1 input adapter or equivalent validation.
    Never infer the intended end date from the last available bar: missing
    sessions must not be treated as frozen prices or a shorter research period.

    conversion_times and evaluation_times are explicit schedules, not generated
    business-day guesses.  Monthly budgets update on the first supplied union
    session, using the preceding explicit valuation (initial known wealth for
    the first month).  None preserves every input LevelRule budget.
    """
    inputs = locals().copy()
    options = _Options(
        path, first_entry, None if first_level_fraction is None else _fraction(first_level_fraction, "first_level_fraction", positive=True),
        _fraction(participation_rate, "participation_rate"), _fraction(fx_cost, "fx_cost"),
        _fraction(cash_floor_fraction, "cash_floor_fraction"), _fraction(symbol_cap_fraction, "symbol_cap_fraction"),
        tax_mode, synthetic_tax_data, penetration_ticks, slippage_ticks,
        None if commission_rate is None else _fraction(commission_rate, "commission_rate"),
        None if sec_rate_override is None else _fraction(sec_rate_override, "sec_rate_override"),
        _fraction(product_sell_tax_rate, "product_sell_tax_rate"))
    _validate_options(options)
    with localcontext() as arithmetic:
        arithmetic.prec = 80
        return _Engine(inputs, options).run()
