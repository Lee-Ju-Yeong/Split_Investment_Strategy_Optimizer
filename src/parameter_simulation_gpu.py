"""
parameter_simulation_gpu.py

Thin wrapper module.

Issue #69:
- Keep this module as an entrypoint-compatible wrapper.
- Delegate implementation to `src.optimization.gpu.parameter_simulation`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/parameter_simulation_gpu.py`)
# while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from .optimization.gpu.parameter_simulation import (  # noqa: E402
    find_optimal_parameters,
    main,
)

__all__ = [
    "find_optimal_parameters",
    "main",
]


if __name__ == "__main__":
    main()
