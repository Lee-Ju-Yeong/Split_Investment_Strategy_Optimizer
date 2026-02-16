"""
walk_forward_analyzer.py

Thin wrapper module.

Issue #69:
- Keep this module as an entrypoint-compatible wrapper.
- Move implementation to `src.analysis.walk_forward_analyzer`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/walk_forward_analyzer.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from .analysis.walk_forward_analyzer import (  # noqa: E402
    find_robust_parameters,
    plot_wfo_results,
    run_walk_forward_analysis,
)

__all__ = [
    "find_robust_parameters",
    "plot_wfo_results",
    "run_walk_forward_analysis",
]


if __name__ == "__main__":
    run_walk_forward_analysis()

