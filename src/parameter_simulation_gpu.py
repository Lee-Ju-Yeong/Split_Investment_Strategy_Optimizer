"""
parameter_simulation_gpu.py

Thin wrapper module.

Issue #60:
- Keep this module import-safe (no config load, no GPU imports, no prints).
- Keep public API `find_optimal_parameters()` for `src.walk_forward_analyzer`.
- Keep standalone execution entry point: `python -m src.parameter_simulation_gpu`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/parameter_simulation_gpu.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from .parameter_simulation_gpu_lib import find_optimal_parameters, main

__all__ = ["find_optimal_parameters", "main"]


if __name__ == "__main__":
    main()
