"""
Shared helpers for backtest price-basis policy.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Any


DEFAULT_PRICE_BASIS = "adjusted"
DEFAULT_ADJUSTED_PRICE_GATE_START_DATE = "2013-11-20"
SUPPORTED_PRICE_BASIS = {"adjusted", "raw"}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_date(value: Any, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty.")

    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"{field_name} must be YYYY-MM-DD or YYYYMMDD. got={value!r}")


def normalize_iso_date(value: Any, field_name: str) -> str:
    return _parse_date(value, field_name).strftime("%Y-%m-%d")


def normalize_price_basis(value: Any) -> str:
    basis = str(value or "").strip().lower()
    if basis not in SUPPORTED_PRICE_BASIS:
        supported = ", ".join(sorted(SUPPORTED_PRICE_BASIS))
        raise ValueError(f"Unsupported price_basis={value!r}. supported=[{supported}]")
    return basis


def is_adjusted_price_basis(price_basis: str) -> bool:
    return normalize_price_basis(price_basis) == "adjusted"


def resolve_price_policy(strategy_params: Mapping[str, Any] | None = None) -> tuple[str, str]:
    params = dict(strategy_params or {})

    if "price_basis" in params:
        price_basis = normalize_price_basis(params.get("price_basis"))
    elif "use_adjusted_price_for_backtest" in params:
        use_adjusted = _coerce_bool(params.get("use_adjusted_price_for_backtest"))
        price_basis = "adjusted" if use_adjusted else "raw"
    else:
        price_basis = DEFAULT_PRICE_BASIS

    gate_start = normalize_iso_date(
        params.get(
            "adjusted_price_gate_start_date",
            DEFAULT_ADJUSTED_PRICE_GATE_START_DATE,
        ),
        field_name="adjusted_price_gate_start_date",
    )
    return price_basis, gate_start


def validate_backtest_window_for_price_policy(
    start_date: Any,
    end_date: Any,
    price_basis: str,
    adjusted_price_gate_start_date: str,
) -> None:
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start > end:
        raise ValueError(f"Invalid date range: start_date({start}) > end_date({end})")

    if not is_adjusted_price_basis(price_basis):
        return

    gate_start = _parse_date(
        adjusted_price_gate_start_date,
        "adjusted_price_gate_start_date",
    )
    if start < gate_start:
        raise ValueError(
            "Adjusted price mode requires start_date >= "
            f"{gate_start.isoformat()}. got={start.isoformat()}"
        )


__all__ = [
    "DEFAULT_ADJUSTED_PRICE_GATE_START_DATE",
    "DEFAULT_PRICE_BASIS",
    "SUPPORTED_PRICE_BASIS",
    "is_adjusted_price_basis",
    "normalize_iso_date",
    "normalize_price_basis",
    "resolve_price_policy",
    "validate_backtest_window_for_price_policy",
]
