"""
backtester.py

Thin wrapper module.

Issue #69:
- Keep this module as a backward-compatible wrapper.
- Move implementation to `src.backtest.cpu.backtester`.

Compatibility:
- Supports `import src.backtester` (package import)
- Supports `import backtester` when `src/` is on `sys.path` (legacy test style)
"""

from __future__ import annotations

if __package__:
    # Preferred: imported as `src.backtester`
    from .backtest.cpu.backtester import BacktestEngine
else:  # pragma: no cover
    # Legacy: imported as top-level `backtester` with `src/` on sys.path.
    from backtest.cpu.backtester import BacktestEngine

__all__ = [
    "BacktestEngine",
]
