"""
Shared helpers for strict-only runtime candidate policy.
"""

from __future__ import annotations

from typing import Any, Tuple


STRICT_CANDIDATE_SOURCE_MODE = "tier"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(
            f"Unsupported boolean-like value for strict runtime policy: {value!r}"
        )
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
        raise ValueError(
            f"Unsupported boolean-like value for strict runtime policy: {value!r}"
        )
    raise ValueError(
        f"Unsupported boolean-like value for strict runtime policy: {value!r}"
    )


def normalize_runtime_candidate_policy(
    candidate_source_mode: Any,
    use_weekly_alpha_gate: Any = False,
) -> Tuple[str, bool]:
    mode = str(candidate_source_mode or STRICT_CANDIDATE_SOURCE_MODE).strip().lower()
    weekly_gate = _coerce_bool(use_weekly_alpha_gate)
    if mode != STRICT_CANDIDATE_SOURCE_MODE or weekly_gate:
        raise ValueError(
            "Unsupported runtime candidate policy: "
            f"candidate_source_mode={candidate_source_mode!r}, "
            f"use_weekly_alpha_gate={use_weekly_alpha_gate!r}. "
            "strict-only runtime requires candidate_source_mode='tier' "
            "and use_weekly_alpha_gate=False."
        )
    return STRICT_CANDIDATE_SOURCE_MODE, False


__all__ = [
    "STRICT_CANDIDATE_SOURCE_MODE",
    "normalize_runtime_candidate_policy",
]
