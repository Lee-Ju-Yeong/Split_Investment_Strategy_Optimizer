"""
Shared helpers for backtest universe-mode policy.
"""

from __future__ import annotations

from typing import Any, Mapping


DEFAULT_UNIVERSE_MODE = "optimistic_survivor"
SUPPORTED_UNIVERSE_MODES = {"strict_pit", "optimistic_survivor"}

_UNIVERSE_MODE_ALIASES = {
    "pit": "strict_pit",
    "pit_strict": "strict_pit",
    "pit_no_forced_tier_liquidation": "strict_pit",
    "survivor": "optimistic_survivor",
    "survivor_only": "optimistic_survivor",
    "research_survivor": "optimistic_survivor",
}


def normalize_universe_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in _UNIVERSE_MODE_ALIASES:
        mode = _UNIVERSE_MODE_ALIASES[mode]
    if mode not in SUPPORTED_UNIVERSE_MODES:
        supported = ", ".join(sorted(SUPPORTED_UNIVERSE_MODES))
        raise ValueError(f"Unsupported universe_mode={value!r}. supported=[{supported}]")
    return mode


def resolve_universe_mode(
    strategy_params: Mapping[str, Any] | None = None,
    *,
    universe_mode: Any = None,
) -> str:
    if universe_mode is not None and str(universe_mode).strip():
        return normalize_universe_mode(universe_mode)

    params = dict(strategy_params or {})
    if "universe_mode" in params and str(params.get("universe_mode", "")).strip():
        return normalize_universe_mode(params.get("universe_mode"))

    return DEFAULT_UNIVERSE_MODE


def is_survivor_optimistic_mode(universe_mode: str) -> bool:
    return normalize_universe_mode(universe_mode) == "optimistic_survivor"


__all__ = [
    "DEFAULT_UNIVERSE_MODE",
    "SUPPORTED_UNIVERSE_MODES",
    "is_survivor_optimistic_mode",
    "normalize_universe_mode",
    "resolve_universe_mode",
]
