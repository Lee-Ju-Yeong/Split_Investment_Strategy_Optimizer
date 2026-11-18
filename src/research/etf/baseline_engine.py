"""M4 기준선: M2 사건·현금·결제·기업행사·세무 원장을 공유한다."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

from .baseline_policy import BaselinePolicy
from .engine import _Engine, _Options, _fraction, _utc, _validate_options
from .engine_strategy import _options, _step_price
from .engine_types import Candidate, LevelRule, Position, ONE, ZERO


class _BaselineEngine(_Engine):
    def __init__(self, inputs, options, policy):
        self.policy = policy
        self.trade_sessions, self.next_level = set(), {}
        self.converted = False
        super().__init__(inputs, options)
        self._cash_event(policy.evaluation_start, "baseline_policy", policy_sha256=policy.sha256,
                         source_sha256=policy.source_sha256, role=policy.role, mode=policy.mode,
                         investment_fraction=policy.investment_fraction, weights=dict(policy.weights))
        if policy.role == "diagnostic":
            self._warn("Post-hoc exposure matching is diagnostic only; it cannot be a primary comparison")

    def _bar_events(self):
        super()._bar_events()
        months, symbols = {}, set()
        for bar in sorted(self.bars, key=lambda row: (_utc(row.open_at), self.order[row.symbol])):
            month = (bar.session_date.year, bar.session_date.month)
            months.setdefault(month, _utc(bar.open_at))
            key = (bar.symbol, month if self.policy.mode == "monthly" else None)
            if key not in symbols:
                self.trade_sessions.add((bar.symbol, bar.session_date))
                symbols.add(key)
        times = list(months.values())
        for at in (times if self.policy.mode == "monthly" else times[:1]):
            self._push(at, 15, "budget")

    def _refresh_budget(self, at, _):
        previous = self._prior_values(at, require_prior=hasattr(self, "targets"))
        if previous["total_krw"] <= ZERO:
            raise ValueError("baseline needs positive equity")
        self.targets = {symbol: previous["total_krw"] * self.policy.investment_fraction * weight
                        for symbol, weight in self.policy.weights}
        self._cash_event(at, "baseline_targets", source_at=previous.get("at"),
                         source_total_krw=previous["total_krw"], targets_krw=self.targets.copy())

    def _held(self, symbol):
        return sorted((p for p in self.positions.values() if p.symbol == symbol), key=lambda p: p.level)

    def _plan(self, at, bar):
        if self._liquidation_session(bar):
            return
        if (bar.symbol, bar.session_date) not in self.trade_sessions:
            return
        if bar.volume == ZERO or bar.limit_locked:
            self._cash_event(at, "baseline_unavailable", symbol=bar.symbol,
                             reason="zero_volume" if bar.volume == ZERO else "limit_locked", retry=False)
            return
        candidate = self._order(at, bar)
        if candidate is not None:
            self._push(at, 40 if candidate.side == "SELL" else 50, "fill", (candidate, bar, at), bar.symbol)

    def _order(self, at, bar):
        instrument = self.instruments[bar.symbol]
        held = self._held(bar.symbol)
        quantity = sum(position.quantity for position in held)
        target = self.targets[bar.symbol] / self._rate(instrument, at)
        side = "SELL" if bar.open * quantity > target else "BUY"
        price = self._open_price(instrument, bar, side)
        if price is None:
            return None
        amount = (max(0, quantity - int(Fraction(target) // Fraction(price))) if side == "SELL"
                  else self._buy_quantity(bar, price, target - price * quantity))
        if amount == 0:
            return None
        level = held[0].level if side == "SELL" else self.next_level.get(bar.symbol, 0) + 1
        return Candidate(bar.symbol, side, level, amount, price, at, "baseline_" + self.policy.mode)

    def _open_price(self, instrument, bar, side):
        context = _options(instrument, bar, self.options.penetration_ticks, self.options.slippage_ticks,
                           self.options.commission_rate, self._sec_override(instrument, bar),
                           self.options.product_sell_tax_rate)
        steps = self.options.slippage_ticks * (1 if side == "BUY" else -1)
        return _step_price(context, bar.open, steps)

    def _buy_quantity(self, bar, price, budget):
        lower, upper = 0, max(0, int(Fraction(budget) // Fraction(price)))
        while lower < upper:
            middle = (lower + upper + 1) // 2
            probe = Candidate(bar.symbol, "BUY", 1, middle, price, bar.open_at, "baseline_probe")
            fee, _ = self._fee(probe, bar)
            if price * middle + fee <= budget:
                lower = middle
            else:
                upper = middle - 1
        return lower

    def _fill_rejection(self, at, candidate, bar):
        if candidate.side == "SELL" and sum(p.quantity for p in self._held(bar.symbol)) < candidate.quantity:
            return "position_changed"
        if bar.adv20 is None or bar.adv20 <= ZERO:
            return "adv20_unavailable"
        if candidate.price * candidate.quantity > bar.adv20 * self.options.participation_rate:
            return "participation_limit"
        return self._buy_rejection(at, candidate, bar) if candidate.side == "BUY" else None

    def _buy_rejection(self, at, candidate, bar):
        instrument = self.instruments[candidate.symbol]
        fee, _ = self._fee(candidate, bar)
        cost = candidate.price * candidate.quantity + fee
        if cost > self.cash[instrument.currency]:
            return "insufficient_cash"
        values, rate = self._values(at), self._rate(instrument, at)
        post_total = values["total_krw"] - fee * rate
        if values["cash_krw"] - cost * rate < post_total * self.options.cash_floor_fraction:
            return "cash_floor"
        if values["holdings_krw"][candidate.symbol] + candidate.price * candidate.quantity * rate > post_total * self.options.symbol_cap_fraction:
            return "symbol_cap"
        return None

    def _sale_lots(self, symbol, quantity):
        sales = []
        for position in self._held(symbol):
            sold = min(quantity, position.quantity)
            sales.append(replace(position, quantity=sold))
            key = (symbol, position.level)
            if sold == position.quantity:
                del self.positions[key]
            else:
                self.positions[key] = replace(position, quantity=position.quantity - sold)
            quantity -= sold
            if quantity == 0:
                return tuple(sales)
        raise ValueError("baseline sale exceeds holdings")

    def _execute(self, at, candidate, bar, generated_at):
        instrument = self.instruments[candidate.symbol]
        fee, product_tax = self._fee(candidate, bar)
        if candidate.side == "BUY":
            self.next_level[bar.symbol] = candidate.level
            position = Position(bar.symbol, candidate.level, candidate.quantity, candidate.price, fee, ZERO, at)
            self.positions[(bar.symbol, position.level)] = position
            self._spend(instrument.currency, position.principal + fee)
        else:
            position = self._sale_lots(bar.symbol, candidate.quantity)
        withholding = self._queue_trade_tax(at, instrument, position, candidate, bar, fee)
        if candidate.side == "SELL":
            self._credit_sale(at, instrument, candidate.price * candidate.quantity - fee - product_tax - withholding, bar)
        self._record_trade(at, candidate, bar, generated_at, fee, product_tax, withholding)

    def _record_trade(self, at, candidate, bar, generated_at, fee, product_tax, withholding):
        row = {"at": at, "generated_at": generated_at, "session_date": bar.session_date,
               "symbol": candidate.symbol, "side": candidate.side, "level": candidate.level,
               "quantity": candidate.quantity, "price": candidate.price, "fee": fee,
               "product_sell_tax": product_tax, "withholding": withholding,
               "currency": self.instruments[bar.symbol].currency, "reason": candidate.reason}
        self.result.trades.append(row)
        self._cash_event(at, "trade", **{key: value for key, value in row.items() if key != "at"})

    def _trade_tax(self, at, payload):
        instrument, position, candidate, bar, fee = payload
        if candidate.side == "BUY":
            return super()._trade_tax(at, payload)
        return self.tax.sell_lots(instrument, position, candidate.price, fee, at,
                                  fx_rate=self._rate(instrument, at),
                                  tax_basis=self.basis.get((instrument.symbol, bar.session_date)), settled_at=at)

    def _convert(self, at, payload):
        if self.policy.mode == "buy_hold" and self.converted:
            return
        super()._convert(at, payload)
        self.converted = True


def _validate_run(policy, instruments, bars, evaluation_times, conversion_times, tax_payments):
    if not isinstance(policy, BaselinePolicy):
        raise ValueError("a frozen BaselinePolicy is required")
    if tuple(item.symbol for item in instruments) != tuple(symbol for symbol, _ in policy.weights):
        raise ValueError("policy basket/order differs from run basket")
    if not bars or {bar.symbol for bar in bars} != {item.symbol for item in instruments}:
        raise ValueError("baseline needs bars for every basket instrument")
    times = [_utc(at) for at in evaluation_times]
    if not times or times != sorted(set(times)) or times[-1] < max(_utc(bar.close_at) for bar in bars):
        raise ValueError("explicit unique increasing evaluations must cover the last supplied session")
    events = times + [_utc(bar.open_at) for bar in bars]
    events += [_utc(at) for at in conversion_times] + [_utc(at) for at in (tax_payments or {})]
    if min(events) < policy.evaluation_start:
        raise ValueError("run events cannot precede the frozen evaluation_start")


def run_baseline(policy, instruments, bars, *, initial_cash, fx_rates=(), evaluation_times=(),
                 conversion_times=(), path="O-H-L-C", participation_rate=Decimal(".05"),
                 fx_cost=Decimal(".001"), cash_floor_fraction=ZERO, symbol_cap_fraction=ONE,
                 tax_mode="pre_tax", us_sessions=(), settlement_session_times=None,
                 tax_basis_by_bar=None, tax_payments=None, synthetic_tax_data=False,
                 penetration_ticks=1, slippage_ticks=0, commission_rate=None,
                 sec_rate_override=None, product_sell_tax_rate=ZERO,
                 kr_sessions=(), kr_settlement_session_times=None,
                 domestic_settlement="legacy_same_day", max_fx_age_days=None, delistings=(),
                 fx_closure_age_days=None):
    """명시한 기간/실제 세션의 완전성 검사는 M1 연결부 등 호출자의 책임이다.

    월별 기준선은 직전 평가액으로 원화 목표를 고정한다. 실제 정수 보유 비중은 비용·
    가격 변화·거절 때문에 목표와 다를 수 있다. buy_hold는 최초 진입/환전만 한다.
    """
    _validate_run(policy, instruments, bars, evaluation_times, conversion_times, tax_payments)
    inputs = locals().copy()
    inputs.update(rules={item.symbol: (LevelRule(1, ONE, ZERO, ZERO),) for item in instruments},
                  initial_positions=(), initial_tax_basis=None, _candidate_provider=None)
    options = _Options(path, False, None, _fraction(participation_rate, "participation_rate"),
                       _fraction(fx_cost, "fx_cost"), _fraction(cash_floor_fraction, "cash_floor_fraction"),
                       _fraction(symbol_cap_fraction, "symbol_cap_fraction"), tax_mode, synthetic_tax_data,
                       penetration_ticks, slippage_ticks,
                       None if commission_rate is None else _fraction(commission_rate, "commission_rate"),
                       None if sec_rate_override is None else _fraction(sec_rate_override, "sec_rate_override"),
                       _fraction(product_sell_tax_rate, "product_sell_tax_rate"), domestic_settlement, max_fx_age_days,
                       fx_closure_age_days)
    _validate_options(options)
    with localcontext() as arithmetic:
        arithmetic.prec = 80
        return _BaselineEngine(inputs, options, policy).run()
