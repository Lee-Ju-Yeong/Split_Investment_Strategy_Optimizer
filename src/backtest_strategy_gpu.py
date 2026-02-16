"""
backtest_strategy_gpu.py

Thin wrapper module.

Issue #69 (PR-7):
- Keep this module as a backward-compatible wrapper.
- Move implementation to `src.backtest.gpu.*`.

Compatibility:
- Supports `import src.backtest_strategy_gpu` (package import)
- Supports `import backtest_strategy_gpu` when `src/` is on `sys.path` (legacy style)
"""

from __future__ import annotations

if __package__:
    # Preferred: imported as `src.backtest_strategy_gpu`
    from .backtest.gpu.data import _collect_candidate_atr_asof, create_gpu_data_tensors
    from .backtest.gpu.engine import run_magic_split_strategy_on_gpu
    from .backtest.gpu.logic import (
        _calculate_monthly_investment_gpu,
        _process_additional_buy_signals_gpu,
        _process_new_entry_signals_gpu,
        _process_sell_signals_gpu,
    )
    from .backtest.gpu.utils import (
        _resolve_signal_date_for_gpu,
        _sort_candidates_by_atr_then_ticker,
        adjust_price_up_gpu,
        get_tick_size_gpu,
    )
else:  # pragma: no cover
    # Legacy: imported as top-level `backtest_strategy_gpu` with `src/` on sys.path.
    from backtest.gpu.data import _collect_candidate_atr_asof, create_gpu_data_tensors
    from backtest.gpu.engine import run_magic_split_strategy_on_gpu
    from backtest.gpu.logic import (
        _calculate_monthly_investment_gpu,
        _process_additional_buy_signals_gpu,
        _process_new_entry_signals_gpu,
        _process_sell_signals_gpu,
    )
    from backtest.gpu.utils import (
        _resolve_signal_date_for_gpu,
        _sort_candidates_by_atr_then_ticker,
        adjust_price_up_gpu,
        get_tick_size_gpu,
    )

__all__ = [
    "create_gpu_data_tensors",
    "get_tick_size_gpu",
    "adjust_price_up_gpu",
    "_calculate_monthly_investment_gpu",
    "_process_sell_signals_gpu",
    "_process_additional_buy_signals_gpu",
    "_process_new_entry_signals_gpu",
    "_resolve_signal_date_for_gpu",
    "_sort_candidates_by_atr_then_ticker",
    "_collect_candidate_atr_asof",
    "run_magic_split_strategy_on_gpu",
]
