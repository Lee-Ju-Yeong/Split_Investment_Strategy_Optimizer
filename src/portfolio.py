"""
portfolio.py

Thin wrapper module.

Issue #69:
- Keep this module as a backward-compatible wrapper.
- Move implementation to `src.backtest.cpu.portfolio`.

Compatibility:
- Supports `import src.portfolio` (package import)
- Supports `import portfolio` when `src/` is on `sys.path` (legacy test style)
"""

from __future__ import annotations

if __package__:
    # Preferred: imported as `src.portfolio`
    from .backtest.cpu.portfolio import Portfolio, Position, Trade
else:  # pragma: no cover
    # Legacy: imported as top-level `portfolio` with `src/` on sys.path.
    from backtest.cpu.portfolio import Portfolio, Position, Trade

__all__ = [
    "Portfolio",
    "Position",
    "Trade",
]
