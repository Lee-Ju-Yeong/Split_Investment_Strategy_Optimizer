"""
execution.py

Thin wrapper module.

Issue #69:
- Keep this module as a backward-compatible wrapper.
- Move implementation to `src.backtest.cpu.execution`.

Compatibility:
- Supports `import src.execution` (package import)
- Supports `import execution` when `src/` is on `sys.path` (legacy test style)
"""

from __future__ import annotations

if __package__:
    # Preferred: imported as `src.execution`
    from .backtest.cpu.execution import BasicExecutionHandler, logger
else:  # pragma: no cover
    # Legacy: imported as top-level `execution` with `src/` on sys.path.
    from backtest.cpu.execution import BasicExecutionHandler, logger

__all__ = [
    "BasicExecutionHandler",
    "logger",
]
