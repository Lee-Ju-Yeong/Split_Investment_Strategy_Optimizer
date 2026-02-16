"""
daily_stock_tier_batch.py

Thin wrapper module.

Issue #69:
- Keep this module as a backward-compatible wrapper.
- Move implementation to `src.pipeline.daily_stock_tier_batch`.
"""

from __future__ import annotations

from .pipeline.daily_stock_tier_batch import (  # noqa: F401
    DEFAULT_DANGER_LIQUIDITY,
    DEFAULT_ENABLE_INVESTOR_V1_WRITE,
    DEFAULT_FINANCIAL_LAG_DAYS,
    DEFAULT_INVESTOR_FLOW5_THRESHOLD,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_PRIME_LIQUIDITY,
    build_daily_stock_tier_frame,
    fetch_financial_history,
    fetch_investor_history,
    fetch_price_history,
    get_latest_tier_date,
    get_min_price_date,
    get_tier_ticker_universe,
    run_daily_stock_tier_batch,
    upsert_daily_stock_tier,
)

__all__ = [
    "DEFAULT_DANGER_LIQUIDITY",
    "DEFAULT_ENABLE_INVESTOR_V1_WRITE",
    "DEFAULT_FINANCIAL_LAG_DAYS",
    "DEFAULT_INVESTOR_FLOW5_THRESHOLD",
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_PRIME_LIQUIDITY",
    "build_daily_stock_tier_frame",
    "fetch_financial_history",
    "fetch_investor_history",
    "fetch_price_history",
    "get_latest_tier_date",
    "get_min_price_date",
    "get_tier_ticker_universe",
    "run_daily_stock_tier_batch",
    "upsert_daily_stock_tier",
]
