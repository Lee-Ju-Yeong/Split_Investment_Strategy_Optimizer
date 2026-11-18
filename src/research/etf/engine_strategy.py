"""제품 V2 판단을 일봉의 가정 경로에 적용하는 순수 함수.

제품 근거: split-trader aaa3bb9의 product_trading_policy.py 및 accounting.py.
현금과 보유 원장은 바꾸지 않는다. 호출자는 종목·세션당 한 번만 호출하고 최초
수동 진입 기회도 한 번만 전달한다. 일중 시각은 관측값이 아닌 선형 경로 가정이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256

from .engine_market import fee_amount, round_price, tick_size
from .engine_types import Bar, Candidate, Instrument, LevelRule, Position, decimal


ZERO = Decimal("0")
ONE = Decimal("1")
RETURN_UNIT = Fraction(1, 10000)  # 0.01%를 소수 비율로 표시한다.
OHLC_TOLERANCE = Decimal(".0062")


@dataclass(frozen=True)
class _Context:
    instrument: Instrument
    bar: Bar
    penetration: int
    slippage: int
    commission: Decimal | None
    sec: Decimal | None
    tax: Decimal

    def fee(self, side, notional):
        return fee_amount(self.instrument.market, side, self.bar.session_date, notional,
                          commission_rate=self.commission, sec_rate_override=self.sec)


@dataclass(frozen=True)
class _Submission:
    side: str
    order: tuple[int, int, Decimal]
    at: Fraction
    after_boundary: bool = False


def _options(instrument, bar, penetration, slippage, commission, sec, tax):
    for value in (penetration, slippage):
        if type(value) is not int or value < 0:
            raise ValueError("관통·슬리피지 호가 수는 0 이상의 정수여야 합니다")
    converted = [None if value is None else decimal(value) for value in (commission, sec, tax)]
    if any(value is not None and not ZERO <= value < ONE for value in converted):
        raise ValueError("비용 비율은 0 이상 1 미만이어야 합니다")
    if instrument.market == "US" and sec is None:
        raise ValueError("US에는 검증된 결제일의 sec_rate_override가 필요합니다")
    return _Context(instrument, bar, penetration, slippage, *converted)


def _normalized_extremes(bar):
    upper, lower = max(bar.open, bar.close), min(bar.open, bar.close)
    if upper > bar.high * (ONE + OHLC_TOLERANCE):
        raise ValueError("시가·종가의 고가 초과가 허용된 0.62%를 넘었습니다")
    if lower < bar.low * (ONE - OHLC_TOLERANCE):
        raise ValueError("시가·종가의 저가 미달이 허용된 0.62%를 넘었습니다")
    return min(lower, bar.low), max(upper, bar.high)


def _path_nodes(bar, path):
    low, high = _normalized_extremes(bar)
    if path not in {"O-H-L-C", "O-L-H-C"}:
        raise ValueError("경로는 O-H-L-C 또는 O-L-H-C여야 합니다")
    middle = (high, low) if path == "O-H-L-C" else (low, high)
    return tuple((Fraction(index, 3), Fraction(price))
                 for index, price in enumerate((bar.open, *middle, bar.close)))


def _validated_positions(instrument, bar, positions, rules):
    levels = [rule.level for rule in rules]
    if levels != list(range(1, len(rules) + 1)):
        raise ValueError("차수 규칙은 1부터 빠짐없이 오름차순이어야 합니다")
    held = [position for position in positions if position.symbol == instrument.symbol]
    if len({position.level for position in held}) != len(held):
        raise ValueError("같은 종목·차수의 보유가 중복됐습니다")
    if any(position.level > len(rules) or position.opened_at > bar.open_at for position in held):
        raise ValueError("규칙 밖 차수나 개장 뒤에 생긴 보유를 미리 사용할 수 없습니다")
    return held


def _step_price(context, price, steps):
    market, on = context.instrument.market, context.bar.session_date
    for _ in range(abs(steps)):
        if steps > 0:
            price += tick_size(market, on, price)
        else:
            # 경계 바로 아래의 단위를 써야 가격대별 호가 변경도 정확히 건넌다.
            probe = price.next_minus()
            price -= tick_size(market, on, max(probe, Decimal("1E-100")))
        if price <= ZERO:
            return None
    return price


def _limit_price(context, mark, side):
    limit = _rounded_mark(context, Fraction(mark), side)
    return _step_price(context, limit, context.slippage if side == "BUY" else -context.slippage)


def _rounded_mark(context, mark, side):
    approximate = Decimal(mark.numerator) / mark.denominator
    limit = round_price(context.instrument.market, context.bar.session_date, approximate, side)
    direction = 1 if side == "SELL" else -1
    # 나눗셈이 끝나지 않는 소수여도 호가 경계는 정확한 정수비로 비교한다.
    while (Fraction(limit) < mark if side == "SELL" else Fraction(limit) > mark):
        limit = _step_price(context, limit, direction)
        if limit is None:
            raise ValueError("호가 내림 결과가 0입니다")
    previous = _step_price(context, limit, -direction)
    while previous is not None and (Fraction(previous) >= mark if side == "SELL" else Fraction(previous) <= mark):
        limit, previous = previous, _step_price(context, previous, -direction)
    return limit


def _drop_met(previous, price, drop):
    # Fraction은 Decimal의 유한 자릿수를 정확한 정수비로 옮겨 곱셈 반올림을 막는다.
    return (Fraction(price) * previous.quantity <= Fraction(previous.entry_price)
            * previous.quantity * (1 - Fraction(drop)))


def _take_met(context, position, price):
    basis = Fraction(position.entry_price) * position.quantity
    net_rate = 1 - Fraction(context.fee("SELL", ONE)) - Fraction(context.tax)
    profit = Fraction(price) * position.quantity * net_rate - basis - Fraction(position.buy_fee)
    units = profit / basis / RETURN_UNIT
    sign = -1 if units < 0 else 1
    absolute = abs(units)
    rounded = sign * ((2 * absolute.numerator + absolute.denominator) // (2 * absolute.denominator))
    return rounded * RETURN_UNIT >= Fraction(position.take_rate)


def _quantity(context, rule, price):
    lower, upper = 0, Fraction(rule.budget) // Fraction(price)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        gross = price * middle
        if gross + context.fee("BUY", gross) <= rule.budget:
            lower = middle
        else:
            upper = middle - 1
    return lower


def _buy_rule(held, rules):
    by_level = {position.level: position for position in held}
    target = next((rule for rule in rules[1:] if rule.level not in by_level), None)
    previous = None if target is None else by_level.get(target.level - 1)
    return target, previous


def _sale(context, position, mark):
    limit = _limit_price(context, mark, "SELL")
    if limit is not None and _take_met(context, position, mark) and _take_met(context, position, limit):
        return position.level, position.quantity, limit
    return None


def _purchase(context, rule, previous, mark):
    limit = _limit_price(context, mark, "BUY")
    if limit is None or not _drop_met(previous, mark, rule.drop_rate):
        return None
    if not _drop_met(previous, limit, rule.drop_rate):
        return None
    quantity = _quantity(context, rule, limit)
    return (rule.level, quantity, limit) if quantity else None


def _candidate(context, order, side, when, reason):
    level, quantity, price = order
    micros = _duration_microseconds(context.bar)
    elapsed = when * micros
    # 체결 경계 이전으로 잘라 버리지 않도록 datetime의 최소 단위에서 올린다.
    at = context.bar.open_at + timedelta(microseconds=-(-elapsed.numerator // elapsed.denominator))
    return Candidate(context.instrument.symbol, side, level, quantity, price, at, reason)


def _duration_microseconds(bar):
    duration = bar.close_at - bar.open_at
    return (duration.days * 86400 + duration.seconds) * 1000000 + duration.microseconds


def _gap_sale(context, position):
    order = _sale(context, position, context.bar.open)
    price = _step_price(context, context.bar.open, -context.slippage)
    if order is not None and price is not None and _take_met(context, position, price):
        return position.level, position.quantity, price
    return None


def _gap_purchase(context, rule, previous):
    order = _purchase(context, rule, previous, context.bar.open)
    price = _step_price(context, context.bar.open, context.slippage)
    if order is None or price is None or not _drop_met(previous, price, rule.drop_rate):
        return None
    quantity = _quantity(context, rule, price)
    return (rule.level, quantity, price) if quantity else None


def _manual_entry(context, rule):
    price = _step_price(context, context.bar.open, context.slippage)
    quantity = 0 if price is None else _quantity(context, rule, price)
    order = (1, quantity, price) if quantity else None
    return None if order is None else _candidate(context, order, "BUY", Fraction(0), "manual_first_entry")


def _order_at(context, held, rules, mark):
    sales = [_sale(context, position, mark) for position in held]
    sales = [order for order in sales if order is not None]
    if sales:
        order = min(sales, key=lambda row: sha256(f"{context.instrument.symbol}:{row[0]}".encode()).digest())
        return "SELL", order
    rule, previous = _buy_rule(held, rules)
    order = None if previous is None else _purchase(context, rule, previous, mark)
    return None if order is None else ("BUY", order)


def _opening_candidate(context, nodes, held, rules):
    selected = _order_at(context, held, rules, context.bar.open)
    if selected is None:
        return None, False
    side, order = selected
    if side == "SELL":
        position = next(position for position in held if position.level == order[0])
        gap = _gap_sale(context, position)
    else:
        rule, previous = _buy_rule(held, rules)
        gap = _gap_purchase(context, rule, previous)
    if gap is not None:
        return _candidate(context, gap, side, Fraction(0), f"{side.lower()}_condition_gap"), True
    return _fill_submitted(context, nodes, _Submission(side, order, Fraction(0))), True


def _first_fill(nodes, target, after, side):
    mark = Fraction(target)
    for (start, left), (end, right) in zip(nodes, nodes[1:]):
        if end < after:
            continue
        current = max(start, after)
        value = left + (right - left) * (current - start) / (end - start)
        if (value >= mark if side == "SELL" else value <= mark):
            return current
        if left != right and min(value, right) <= mark <= max(value, right):
            return start + (end - start) * (mark - left) / (right - left)
    return None


def _tick_marks(context, nodes):
    low, high = min(price for _, price in nodes), max(price for _, price in nodes)
    mark = round_price(context.instrument.market, context.bar.session_date, Decimal(low.numerator) / low.denominator, "SELL")
    while Fraction(mark) <= high:
        yield mark
        mark = _step_price(context, mark, 1)


def _take_boundary(context, position):
    target = Fraction(position.take_rate) / RETURN_UNIT
    units = -(-target.numerator // target.denominator)
    basis = Fraction(position.entry_price) * position.quantity
    net_rate = 1 - Fraction(context.fee("SELL", ONE)) - Fraction(context.tax)
    if net_rate <= 0:
        return None
    required = basis * (units - Fraction(1, 2)) * RETURN_UNIT
    return (basis + Fraction(position.buy_fee) + required) / (position.quantity * net_rate)


def _signal_boundaries(context, nodes, held, rules):
    boundaries = {Fraction(mark) for mark in _tick_marks(context, nodes)}
    boundaries.update(boundary for position in held
                      if (boundary := _take_boundary(context, position)) is not None)
    rule, previous = _buy_rule(held, rules)
    if previous is not None:
        boundaries.add(Fraction(previous.entry_price) * (1 - Fraction(rule.drop_rate)))
    return boundaries


def _signal_events(nodes, boundaries):
    for (start, left), (end, right) in zip(nodes, nodes[1:]):
        points = {start: left, end: right}
        if left != right:
            for mark in boundaries:
                if min(left, right) < mark < max(left, right):
                    when = start + (end - start) * (mark - left) / (right - left)
                    points[when] = mark
        ordered = sorted(points.items())
        for index, (when, mark) in enumerate(ordered):
            yield when, mark, False
            if index + 1 < len(ordered) and mark != ordered[index + 1][1]:
                # 두 경계 사이에서는 조건·지정가·수량이 일정하다. 경계 직후를 별도 처리한다.
                yield when, (mark + ordered[index + 1][1]) / 2, True


def _fill_submitted(context, nodes, submission):
    steps = context.penetration if submission.side == "SELL" else -context.penetration
    target = _step_price(context, submission.order[2], steps)
    if target is None:
        return None
    filled = _first_fill(nodes, target, submission.at, submission.side)
    if filled == submission.at and submission.after_boundary:
        # 열린 구간에서만 성립한 신호를 경계 시각에 앞당겨 기록하지 않는다.
        micros = _duration_microseconds(context.bar)
        elapsed = submission.at * micros
        after = Fraction(elapsed.numerator // elapsed.denominator + 1, micros)
        filled = _first_fill(nodes, target, after, submission.side) if after <= 1 else None
    if filled is None:
        return None
    return _candidate(context, submission.order, submission.side, filled,
                      f"{submission.side.lower()}_condition_touch")


def _intraday_candidate(context, nodes, held, rules):
    boundaries = _signal_boundaries(context, nodes, held, rules)
    for when, mark, after_boundary in _signal_events(nodes, boundaries):
        selected = _order_at(context, held, rules, mark)
        if selected is not None:
            side, order = selected
            return _fill_submitted(context, nodes, _Submission(side, order, when, after_boundary))
    return None


def candidate_for_bar(instrument: Instrument, bar: Bar, positions: list[Position], rules: tuple[LevelRule, ...], *,
                      path="O-H-L-C", first_entry=False, penetration_ticks=1, slippage_ticks=0,
                      commission_rate=None, sec_rate_override=None, product_sell_tax_rate=ZERO) -> Candidate | None:
    """처음 조건이 성립한 주문 하나를 고정하고, 그 주문의 체결만 반환한다.

    관통은 보정 지정가부터 세며 갭에는 실제 입력 시가를 쓴다. 슬리피지는 매수에
    더하고 매도에서 뺀다. 판단 수익률 반올림과 호가 보정 경계를 흐리지 않도록
    경로의 호가·수익률 경계를 시간순으로 검사한다. 미체결 주문은 당일 다른 주문으로
    바꾸지 않으며 다음 호출로 넘기지 않는다. 비용은 절사 없는 연구용 비례액이다.
    경계 직후에 성립한 즉시 체결은 다음 마이크로초로 기록한다.
    US SEC 요율은 결제일에 맞춘 명시값이 필수다.
    """
    if instrument.symbol != bar.symbol:
        raise ValueError("상품과 일봉의 종목이 다릅니다")
    if type(first_entry) is not bool:
        raise ValueError("first_entry는 bool이어야 합니다")
    with localcontext() as arithmetic:
        arithmetic.prec = 80
        context = _options(instrument, bar, penetration_ticks, slippage_ticks,
                           commission_rate, sec_rate_override, product_sell_tax_rate)
        nodes = _path_nodes(bar, path)
        held = _validated_positions(instrument, bar, positions, rules)
        if bar.volume == ZERO or bar.limit_locked:
            return None
        if any(position.opened_at == bar.open_at for position in held):
            return None
        if first_entry and not held:
            return _manual_entry(context, rules[0]) if rules else None
        opening, submitted = _opening_candidate(context, nodes, held, rules)
        if submitted:
            return opening
        if not held:
            return None
        return _intraday_candidate(context, nodes, held, rules)
