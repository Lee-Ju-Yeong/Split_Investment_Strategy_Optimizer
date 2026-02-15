"""
pipeline_batch.py

Thin wrapper module.

Issue #69:
- Keep this module as an entrypoint-compatible wrapper.
- Move implementation to `src.pipeline.batch`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/pipeline_batch.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from .pipeline.batch import main, run_pipeline_batch

__all__ = ["main", "run_pipeline_batch"]


if __name__ == "__main__":
    main()

