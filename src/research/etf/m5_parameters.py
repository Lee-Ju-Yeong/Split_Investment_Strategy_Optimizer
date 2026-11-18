"""성과 실행 전 고정한 M5 조건과 과거 자료만 사용하는 변동성·비용 검사."""
from __future__ import annotations

from bisect import bisect_left
from datetime import date
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from itertools import product
import json
from pathlib import Path

from .baseline_policy import VolatilityEstimate
from .engine_inputs import _calendar, _prices, _snapshot
from .engine_market import fee_amount, sec_fee_rate
from .engine_types import LevelRule, ONE, ZERO, decimal
from .m5_inputs import read_manifest


CONFIG_PATH = Path(__file__).with_name("m5_search_v1.json")


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["schema"] != "etf-m5-search-v1" or config["status"] != "explore_only_preperformance":
        raise ValueError("invalid M5 search configuration")
    for candidate in config["pilot"]["candidates"]:
        if set(candidate) != set(config["grid"]) or any(candidate[k] not in config["grid"][k] for k in candidate):
            raise ValueError("pilot candidates must belong to the pinned grid")
    if sorted({sum(row.values()) for row in config["market_mix_strata"]}) != config["k_values"]:
        raise ValueError("market strata and k_values differ")
    return config


def grid_candidates(config):
    grid = config["grid"]
    return [dict(zip(grid, values)) for values in product(*grid.values())]


def estimate_volatility(rows, calendar, start):
    """직전 251개 실제 세션에서 최대 250개 로그수익률. 0거래량 인접 쌍 제외."""
    days = sorted(day for day in calendar if day < start)[-251:]
    dated = {date.fromisoformat(row["date"]): row for row in rows if date.fromisoformat(row["date"]) < start}
    if len(days) != 251 or any(day not in dated for day in days):
        raise ValueError("volatility requires 251 complete pre-start sessions")
    with localcontext(prec=80, rounding=ROUND_HALF_EVEN):
        returns = []
        for previous, current in zip(days, days[1:]):
            left, right = dated[previous], dated[current]
            if decimal(left["volume"]) > ZERO and decimal(right["volume"]) > ZERO:
                returns.append((decimal(right["close"]) / decimal(left["close"])).ln())
        if len(returns) < 200:
            raise ValueError("volatility needs at least 200 nonzero-volume return observations")
        mean = sum(returns, ZERO) / len(returns)
        variance = sum(((value - mean) ** 2 for value in returns), ZERO) / (len(returns) - 1)
        return VolatilityEstimate(variance.sqrt(), calendar[days[-1]][1])


def load_volatilities(data_root, manifest_path, symbols, start):
    root = Path(data_root)
    manifest = read_manifest(root, manifest_path, symbols)
    record, selected, _, bundle = _snapshot(root, root / manifest["split_csv"], symbols)
    prices = _prices(root, bundle, record, selected, start)
    calendars = {market: _calendar(root / "fx" / f"calendar_{name}.csv")
                 for market, name in (("KR", "krx"), ("US", "nyse"), ("GOLD", "krx"))}
    return {symbol: estimate_volatility(prices[symbol], calendars[row["market"]], start)
            for symbol, row in selected.items()}


def candidate_rules(candidate, inputs, volatilities):
    """모든 상품에 같은 계수를 쓰며 상품별 계수 조정·반올림·자동 보정은 하지 않는다."""
    with localcontext(prec=80, rounding=ROUND_HALF_EVEN):
        rules = {}
        for instrument in inputs.instruments:
            estimate = volatilities[instrument.symbol]
            first_bar = min((bar for bar in inputs.bars if bar.symbol == instrument.symbol), key=lambda b: b.open_at)
            if estimate.known_at >= first_bar.open_at:
                raise ValueError("volatility must be known before the first open")
            drop, take = decimal(candidate["kg"]) * estimate.value, decimal(candidate["kp"]) * estimate.value
            # 실행 중 미래 비용 변경까지 예측하지 않고 시작 때 알려진 왕복 비용 하한을 검사한다.
            day = _charge_date(inputs, instrument, first_bar)
            sec = sec_fee_rate(day, known_at=first_bar.open_at) if instrument.market == "US" else None
            cost = fee_amount(instrument.market, "BUY", day, ONE) + fee_amount(
                instrument.market, "SELL", day, ONE, sec_rate_override=sec)
            if not cost <= drop < ONE:
                raise ValueError(f"{instrument.symbol}: drop below known round-trip costs or >= 1")
            rules[instrument.symbol] = tuple(LevelRule(n, ONE, drop, take) for n in range(1, candidate["levels"] + 1))
        return rules


def _charge_date(inputs, instrument, bar):
    if instrument.market != "US":
        return bar.session_date
    sessions = inputs.us_sessions
    day = bar.session_date
    lag = 1 if day >= date(2024, 5, 28) else 2 if day >= date(2017, 9, 5) else 3
    index = bisect_left(sessions, day)
    if index == len(sessions) or sessions[index] != day or index + lag >= len(sessions):
        raise ValueError("cost guard needs actual US settlement sessions")
    return sessions[index + lag]
