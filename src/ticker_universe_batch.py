"""
ticker_universe_batch.py

Thin wrapper module.

Issue #69:
- Keep this module as an entrypoint-compatible wrapper.
- Move implementation to `src.pipeline.ticker_universe_batch`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/ticker_universe_batch.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from .pipeline.ticker_universe_batch import (
    DEFAULT_API_CALL_DELAY,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_MARKETS,
    DEFAULT_START_DATE_STR,
    DEFAULT_STEP_DAYS,
    build_snapshot_dates,
    collect_snapshot_rows,
    get_existing_snapshot_dates,
    main,
    rebuild_ticker_universe_history,
    run_snapshot_batch,
    run_ticker_universe_batch,
    upsert_snapshot_rows,
)

__all__ = [
    "DEFAULT_API_CALL_DELAY",
    "DEFAULT_LOG_INTERVAL",
    "DEFAULT_MARKETS",
    "DEFAULT_START_DATE_STR",
    "DEFAULT_STEP_DAYS",
    "build_snapshot_dates",
    "collect_snapshot_rows",
    "get_existing_snapshot_dates",
    "main",
    "rebuild_ticker_universe_history",
    "run_snapshot_batch",
    "run_ticker_universe_batch",
    "upsert_snapshot_rows",
]


if __name__ == "__main__":
    main()

