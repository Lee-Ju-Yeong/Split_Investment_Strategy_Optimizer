"""
financial_collector.py

Thin wrapper module.

Issue #69:
- Keep this module as a backward-compatible wrapper.
- Move implementation to `src.data.collectors.financial_collector`.
"""

from __future__ import annotations

from .data.collectors.financial_collector import (  # noqa: F401
    API_CALL_DELAY,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_START_DATE_STR,
    DEFAULT_WORKERS,
    DEFAULT_WRITE_BATCH_SIZE,
    get_financial_ticker_universe,
    get_latest_financial_dates,
    normalize_fundamental_df,
    run_financial_batch,
    upsert_financial_rows,
)

__all__ = [
    "API_CALL_DELAY",
    "DEFAULT_LOG_INTERVAL",
    "DEFAULT_START_DATE_STR",
    "DEFAULT_WORKERS",
    "DEFAULT_WRITE_BATCH_SIZE",
    "get_financial_ticker_universe",
    "get_latest_financial_dates",
    "normalize_fundamental_df",
    "run_financial_batch",
    "upsert_financial_rows",
]

