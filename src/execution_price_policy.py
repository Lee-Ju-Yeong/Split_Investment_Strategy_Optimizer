"""
Shared tick-size and execution-price rounding policy.

CPU execution and CPU strategy call these scalar functions directly. The GPU
path keeps a vectorized twin in ``src/backtest/gpu/utils.py`` that must produce
the same prices; it uses ``normalize_sell_price_tick_rounding`` from here so both
paths accept exactly the same config values.

Buy prices are always rounded up to the tick grid. Sell prices follow
``execution_params.sell_price_tick_rounding``:

- ``"up"`` (default when the key is missing): legacy behaviour, keeps past
  results reproducible. Can overstate sell proceeds when the price basis is not
  on the tick grid (e.g. adjusted prices).
- ``"down"``: conservative; never sells above the price basis, but at least one
  tick so a positive price never settles at 0.
"""

from __future__ import annotations

import math
from typing import Any

SELL_PRICE_TICK_ROUNDING_UP = "up"
SELL_PRICE_TICK_ROUNDING_DOWN = "down"
SUPPORTED_SELL_PRICE_TICK_ROUNDING = (
    SELL_PRICE_TICK_ROUNDING_UP,
    SELL_PRICE_TICK_ROUNDING_DOWN,
)
DEFAULT_SELL_PRICE_TICK_ROUNDING = SELL_PRICE_TICK_ROUNDING_UP

# price / tick is rounded to this many decimals before ceil/floor so float noise
# such as 987.9999999 or 988.0000001 does not move the price by a whole tick.
TICK_RATIO_DECIMALS = 5


def normalize_sell_price_tick_rounding(value: Any = None) -> str:
    if value is None:
        return DEFAULT_SELL_PRICE_TICK_ROUNDING
    mode = str(value).strip().lower()
    if mode not in SUPPORTED_SELL_PRICE_TICK_ROUNDING:
        supported = ", ".join(SUPPORTED_SELL_PRICE_TICK_ROUNDING)
        raise ValueError(
            f"Unsupported sell_price_tick_rounding={value!r}. supported=[{supported}]"
        )
    return mode


def get_tick_size(price) -> int:
    if price < 2000:
        return 1
    if price < 5000:
        return 5
    if price < 20000:
        return 10
    if price < 50000:
        return 50
    if price < 200000:
        return 100
    if price < 500000:
        return 500
    return 1000


def adjust_price_up(price):
    tick_size = get_tick_size(price)
    return math.ceil(round(price / tick_size, TICK_RATIO_DECIMALS)) * tick_size


def adjust_price_down(price):
    tick_size = get_tick_size(price)
    ticks = math.floor(round(price / tick_size, TICK_RATIO_DECIMALS))
    if ticks < 1 and price > 0:
        ticks = 1
    return ticks * tick_size


def adjust_sell_price(price, sell_price_tick_rounding: str):
    if sell_price_tick_rounding == SELL_PRICE_TICK_ROUNDING_DOWN:
        return adjust_price_down(price)
    return adjust_price_up(price)


__all__ = [
    "DEFAULT_SELL_PRICE_TICK_ROUNDING",
    "SELL_PRICE_TICK_ROUNDING_DOWN",
    "SELL_PRICE_TICK_ROUNDING_UP",
    "SUPPORTED_SELL_PRICE_TICK_ROUNDING",
    "TICK_RATIO_DECIMALS",
    "adjust_price_down",
    "adjust_price_up",
    "adjust_sell_price",
    "get_tick_size",
    "normalize_sell_price_tick_rounding",
]
