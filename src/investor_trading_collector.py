"""
investor_trading_collector.py

Thin wrapper module.

Issue #69:
- Keep this module as a backward-compatible wrapper.
- Move implementation to `src.data.collectors.investor_trading_collector`.
"""

from __future__ import annotations

from .data.collectors.investor_trading_collector import (  # noqa: F401
    API_CALL_DELAY,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_START_DATE_STR,
    DEFAULT_WORKERS,
    DEFAULT_WRITE_BATCH_SIZE,
    get_investor_date_bounds,
    get_investor_ticker_universe,
    normalize_investor_df,
    run_investor_trading_batch,
    upsert_investor_rows,
)

__all__ = [
    "API_CALL_DELAY",
    "DEFAULT_LOG_INTERVAL",
    "DEFAULT_START_DATE_STR",
    "DEFAULT_WORKERS",
    "DEFAULT_WRITE_BATCH_SIZE",
    "get_investor_date_bounds",
    "get_investor_ticker_universe",
    "normalize_investor_df",
    "run_investor_trading_batch",
    "upsert_investor_rows",
]

