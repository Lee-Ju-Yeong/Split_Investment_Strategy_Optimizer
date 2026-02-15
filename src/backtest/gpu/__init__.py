"""
GPU backtest implementation (CuPy/cuDF).

Important: avoid importing GPU deps (cupy/cudf) at package import time unless
explicitly required by a GPU entrypoint.
"""

