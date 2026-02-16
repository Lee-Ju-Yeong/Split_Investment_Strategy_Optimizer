"""
ohlcv_batch.py

Thin wrapper module.

Issue #69:
- Keep this module as an entrypoint-compatible wrapper.
- Move implementation to `src.pipeline.ohlcv_batch`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/ohlcv_batch.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from .pipeline.ohlcv_batch import (  # noqa: E402
    DEFAULT_LOG_INTERVAL,
    _build_universe_ranges_from_history_rows,
    _resolve_effective_collection_window,
    get_ohlcv_ticker_universe,
    main,
    normalize_ohlcv_df,
    run_ohlcv_batch,
)

__all__ = [
    "DEFAULT_LOG_INTERVAL",
    "_build_universe_ranges_from_history_rows",
    "_resolve_effective_collection_window",
    "get_ohlcv_ticker_universe",
    "main",
    "normalize_ohlcv_df",
    "run_ohlcv_batch",
]


if __name__ == "__main__":
    main()
