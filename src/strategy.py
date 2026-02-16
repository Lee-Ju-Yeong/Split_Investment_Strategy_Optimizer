"""
strategy.py

Thin wrapper module.

Issue #69:
- Keep this module as a backward-compatible wrapper.
- Move implementation to `src.backtest.cpu.strategy`.

Compatibility:
- Supports `import src.strategy` (package import)
- Supports `import strategy` when `src/` is on `sys.path` (legacy test style)
"""

from __future__ import annotations

if __package__:
    # Preferred: imported as `src.strategy`
    from .backtest.cpu.strategy import MagicSplitStrategy, Position, Strategy, logger
else:  # pragma: no cover
    # Legacy: imported as top-level `strategy` with `src/` on sys.path.
    from backtest.cpu.strategy import MagicSplitStrategy, Position, Strategy, logger

__all__ = [
    "MagicSplitStrategy",
    "Position",
    "Strategy",
    "logger",
]
