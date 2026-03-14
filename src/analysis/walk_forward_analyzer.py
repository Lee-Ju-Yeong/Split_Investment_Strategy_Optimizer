from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..config_loader import load_config

if TYPE_CHECKING:
    import pandas as pd


_CPU_CERT_PARAM_KEYS = (
    "max_stocks",
    "order_investment_ratio",
    "additional_buy_drop_rate",
    "sell_profit_rate",
    "additional_buy_priority",
    "stop_loss_rate",
    "max_splits_limit",
    "max_inactivity_period",
)
_CPU_CERT_FLOAT_KEYS = {
    "order_investment_ratio",
    "additional_buy_drop_rate",
    "sell_profit_rate",
    "stop_loss_rate",
}
_CPU_CERT_INT_KEYS = {
    "max_stocks",
    "max_splits_limit",
    "max_inactivity_period",
}
_CPU_CERT_SORT_COLUMNS = ("calmar_ratio", "cagr", "mdd")
_APPROVAL_GRADE_HOLDOUT_MIN_DAYS = 730
_HOLDOUT_ADEQUACY_FIELDS = (
    "trade_count",
    "closed_trade_count",
    "avg_hold_days",
    "distinct_entry_months",
)
_DEFAULT_PARITY_CANARY_EXCLUDED_RANGES = (
    ("2025-12-01", "2026-01-31"),
)
_SUPPORTED_WFO_LANE_TYPES = (
    "legacy_wfo",
    "promotion_evaluation",
    "research_start_date_robustness",
)
_RESEARCH_WFO_MODE = "frozen_shortlist_multi_anchor_eval"
_PROMOTION_WFO_MODE = "frozen_shortlist_single_anchor_eval"
_SELECTION_CONTRACT_VERSION = "promotion_holdout_selector_v1"


def _coerce_date(value):
    if hasattr(value, "date") and callable(getattr(value, "date")):
        try:
            return value.date()
        except TypeError:
            pass
    return datetime.fromisoformat(str(value)).date()


def _inclusive_day_count(start_date, end_date) -> int:
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if end < start:
        raise ValueError("holdout_end must be on or after holdout_start")
    return int((end - start).days + 1)


def _normalize_contaminated_ranges(ranges) -> list[dict]:
    normalized = []
    for item in list(ranges or []):
        if isinstance(item, dict):
            raw_start = item.get("start")
            raw_end = item.get("end")
        else:
            raw_start, raw_end = item
        start = _coerce_date(raw_start)
        end = _coerce_date(raw_end)
        if end < start:
            raise ValueError("contaminated range end must be on or after start")
        normalized.append({"start": start.isoformat(), "end": end.isoformat()})
    return normalized


def _ranges_overlap(start_date, end_date, range_start, range_end) -> bool:
    return not (end_date < range_start or range_end < start_date)


def evaluate_holdout_policy(
    *,
    holdout_start,
    holdout_end,
    wfo_end=None,
    contaminated_ranges=None,
    adequacy_metrics=None,
    min_length_days: int = _APPROVAL_GRADE_HOLDOUT_MIN_DAYS,
) -> dict:
    start = _coerce_date(holdout_start)
    end = _coerce_date(holdout_end)
    resolved_wfo_end = _coerce_date(wfo_end) if wfo_end is not None else None
    length_days = _inclusive_day_count(start, end)
    normalized_ranges = _normalize_contaminated_ranges(contaminated_ranges)
    overlap = any(
        _ranges_overlap(start, end, _coerce_date(item["start"]), _coerce_date(item["end"]))
        for item in normalized_ranges
    )
    adequacy = dict(adequacy_metrics or {})
    missing_fields = [field for field in _HOLDOUT_ADEQUACY_FIELDS if field not in adequacy]
    reasons = []
    if length_days < int(min_length_days):
        reasons.append(f"holdout_too_short={length_days}<{int(min_length_days)}")
    if resolved_wfo_end is not None and start <= resolved_wfo_end:
        reasons.append("holdout_starts_on_or_before_wfo_end")
    if overlap:
        reasons.append("holdout_range_contaminated")
    if missing_fields:
        reasons.append("missing_adequacy_fields=" + ",".join(missing_fields))
    approval_eligible = not reasons
    return {
        "holdout_start": start.isoformat(),
        "holdout_end": end.isoformat(),
        "wfo_end": resolved_wfo_end.isoformat() if resolved_wfo_end is not None else None,
        "holdout_length_days": int(length_days),
        "holdout_class": "approval_grade" if approval_eligible else "internal_provisional",
        "approval_eligible": bool(approval_eligible),
        "promotion_wfo_end_before_holdout": bool(
            resolved_wfo_end is None or resolved_wfo_end < start
        ),
        "contaminated_ranges": normalized_ranges,
        "contaminated_overlap": bool(overlap),
        "required_adequacy_fields": list(_HOLDOUT_ADEQUACY_FIELDS),
        "missing_adequacy_fields": missing_fields,
        "reasons": reasons,
    }


def build_holdout_manifest(
    *,
    holdout_start,
    holdout_end,
    wfo_end,
    contaminated_ranges=None,
    adequacy_metrics=None,
    holdout_backtest_executed: bool = False,
    min_length_days: int = _APPROVAL_GRADE_HOLDOUT_MIN_DAYS,
) -> dict:
    policy = evaluate_holdout_policy(
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        wfo_end=wfo_end,
        contaminated_ranges=contaminated_ranges,
        adequacy_metrics=adequacy_metrics,
        min_length_days=min_length_days,
    )
    adequacy = dict(adequacy_metrics or {})
    reasons = list(policy["reasons"])
    if not holdout_backtest_executed:
        reasons.append("holdout_backtest_not_executed")
    approval_eligible = bool(policy["approval_eligible"]) and bool(holdout_backtest_executed)
    holdout_class = "approval_grade" if approval_eligible else "internal_provisional"
    return {
        "holdout_start": policy["holdout_start"],
        "holdout_end": policy["holdout_end"],
        "wfo_end": policy["wfo_end"],
        "holdout_date_reuse_forbidden": True,
        "holdout_backtest_executed": bool(holdout_backtest_executed),
        "parity_canary_excluded_ranges": policy["contaminated_ranges"],
        "holdout_class": holdout_class,
        "holdout_length_days": policy["holdout_length_days"],
        "approval_eligible": approval_eligible,
        "promotion_wfo_end_before_holdout": policy["promotion_wfo_end_before_holdout"],
        "required_adequacy_fields": policy["required_adequacy_fields"],
        "missing_adequacy_fields": policy["missing_adequacy_fields"],
        "trade_count": adequacy.get("trade_count"),
        "closed_trade_count": adequacy.get("closed_trade_count"),
        "avg_hold_days": adequacy.get("avg_hold_days"),
        "distinct_entry_months": adequacy.get("distinct_entry_months"),
        "peak_slot_utilization": adequacy.get("peak_slot_utilization"),
        "realized_split_depth": adequacy.get("realized_split_depth"),
        "avg_invested_capital_ratio": adequacy.get("avg_invested_capital_ratio"),
        "cash_drag_ratio": adequacy.get("cash_drag_ratio"),
        "reasons": reasons,
    }


def build_lane_manifest(
    *,
    lane_type: str,
    approval_eligible: bool,
    decision_date=None,
    research_data_cutoff=None,
    promotion_data_cutoff=None,
    shortlist_hash=None,
    publication_lag_policy=None,
    ticker_universe_snapshot_id=None,
    engine_version_hash=None,
    composite_curve_allowed: bool,
    cpu_audit_outcome: str,
    reasons=None,
) -> dict:
    decision = _coerce_date(decision_date or datetime.now()).isoformat()
    evidence_tier = "approval_grade" if approval_eligible else "internal_provisional"
    return {
        "lane_type": str(lane_type),
        "evidence_tier": evidence_tier,
        "approval_eligible": bool(approval_eligible),
        "decision_date": decision,
        "research_data_cutoff": str(research_data_cutoff) if research_data_cutoff else None,
        "promotion_data_cutoff": str(promotion_data_cutoff) if promotion_data_cutoff else None,
        "shortlist_hash": shortlist_hash,
        "publication_lag_policy": publication_lag_policy or "unspecified",
        "ticker_universe_snapshot_id": ticker_universe_snapshot_id,
        "engine_version_hash": engine_version_hash or "unknown",
        "composite_curve_allowed": bool(composite_curve_allowed),
        "cpu_audit_outcome": str(cpu_audit_outcome),
        "reasons": list(reasons or []),
    }


def _build_unconfigured_holdout_manifest(
    *,
    wfo_end,
    contaminated_ranges=None,
) -> dict:
    return {
        "holdout_start": None,
        "holdout_end": None,
        "wfo_end": _coerce_date(wfo_end).isoformat(),
        "holdout_date_reuse_forbidden": True,
        "holdout_backtest_executed": False,
        "parity_canary_excluded_ranges": _normalize_contaminated_ranges(contaminated_ranges),
        "holdout_class": "unconfigured",
        "holdout_length_days": None,
        "approval_eligible": False,
        "promotion_wfo_end_before_holdout": None,
        "required_adequacy_fields": list(_HOLDOUT_ADEQUACY_FIELDS),
        "missing_adequacy_fields": list(_HOLDOUT_ADEQUACY_FIELDS),
        "trade_count": None,
        "closed_trade_count": None,
        "avg_hold_days": None,
        "distinct_entry_months": None,
        "peak_slot_utilization": None,
        "realized_split_depth": None,
        "avg_invested_capital_ratio": None,
        "cash_drag_ratio": None,
        "reasons": ["holdout_window_missing", "holdout_backtest_not_executed"],
    }


def _json_default(value):
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json_artifact(path: str, payload: dict) -> str:
    artifact_path = Path(path)
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return artifact_path.as_posix()


def write_wfo_manifests(
    *,
    results_dir: str,
    lane_manifest: dict,
    holdout_manifest: dict,
    anchor_manifest: dict | None = None,
) -> dict:
    paths = {
        "lane_manifest_path": _write_json_artifact(
            os.path.join(results_dir, "lane_manifest.json"),
            lane_manifest,
        ),
        "holdout_manifest_path": _write_json_artifact(
            os.path.join(results_dir, "holdout_manifest.json"),
            holdout_manifest,
        ),
    }
    if anchor_manifest is not None:
        paths["anchor_manifest_path"] = _write_json_artifact(
            os.path.join(results_dir, "anchor_manifest.json"),
            anchor_manifest,
        )
    return paths


def _resolve_cpu_audit_outcome(cpu_cert_settings: dict, selection_audits: list[dict]) -> str:
    if not cpu_cert_settings.get("enabled"):
        return "disabled"
    if selection_audits and all(
        str(item.get("cpu_audit_outcome") or "").strip().lower() == "pass"
        for item in selection_audits
    ):
        return "pass"
    return "enabled_but_no_selection_audit"


def _resolve_holdout_runtime_settings(wfo_settings: dict) -> dict:
    contaminated_ranges = (
        wfo_settings.get("holdout_contaminated_ranges")
        or wfo_settings.get("parity_canary_excluded_ranges")
        or list(_DEFAULT_PARITY_CANARY_EXCLUDED_RANGES)
    )
    return {
        "holdout_start": wfo_settings.get("holdout_start")
        or wfo_settings.get("internal_provisional_holdout_start"),
        "holdout_end": wfo_settings.get("holdout_end")
        or wfo_settings.get("internal_provisional_holdout_end"),
        "contaminated_ranges": contaminated_ranges,
    }


def _hash_file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json_sha256(payload) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _resolve_lane_type(wfo_settings: dict) -> str:
    lane_type = str(wfo_settings.get("lane_type") or "legacy_wfo").strip() or "legacy_wfo"
    if lane_type not in _SUPPORTED_WFO_LANE_TYPES:
        raise ValueError(
            "Unsupported walk_forward_settings.lane_type. "
            f"Expected one of: {', '.join(_SUPPORTED_WFO_LANE_TYPES)}."
        )
    return lane_type


def _load_frozen_shortlist(
    shortlist_path: str,
    *,
    setting_name: str,
    context_label: str,
):
    import pandas as pd

    path = Path(shortlist_path)
    if not path.exists():
        raise ValueError(f"{setting_name} does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        shortlist_df = pd.read_csv(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        shortlist_df = pd.DataFrame(rows)
    else:
        raise ValueError(f"{setting_name} must be .csv or .json")
    if shortlist_df.empty:
        raise ValueError(f"{context_label} is empty.")
    missing_columns = [key for key in _CPU_CERT_PARAM_KEYS if key not in shortlist_df.columns]
    if missing_columns:
        raise ValueError(
            f"{context_label} is missing required parameter columns: "
            + ",".join(missing_columns)
        )
    shortlist_df = shortlist_df.reset_index(drop=True).copy()
    shortlist_df["shortlist_candidate_id"] = shortlist_df.index + 1
    return shortlist_df


def _resolve_promotion_runtime_settings(wfo_settings: dict) -> dict:
    promotion_mode = str(
        wfo_settings.get("promotion_mode") or _PROMOTION_WFO_MODE
    ).strip() or _PROMOTION_WFO_MODE
    if promotion_mode != _PROMOTION_WFO_MODE:
        raise ValueError(
            "promotion_evaluation only supports "
            f"promotion_mode={_PROMOTION_WFO_MODE}."
        )
    shortlist_path = str(wfo_settings.get("promotion_shortlist_path") or "").strip()
    if not shortlist_path:
        raise ValueError(
            "promotion_evaluation requires walk_forward_settings.promotion_shortlist_path."
        )
    selection_metric = str(
        wfo_settings.get("promotion_selection_metric")
        or wfo_settings.get("cpu_certification_metric")
        or "calmar_ratio"
    ).strip() or "calmar_ratio"
    shortlist_hash = str(wfo_settings.get("shortlist_hash") or "").strip()
    return {
        "promotion_mode": promotion_mode,
        "promotion_shortlist_path": shortlist_path,
        "promotion_selection_metric": selection_metric,
        "shortlist_hash": shortlist_hash or _hash_file_sha256(shortlist_path),
    }


def _resolve_selection_contract_settings(wfo_settings: dict) -> dict:
    settings = dict(wfo_settings.get("selection_contract") or {})
    return {
        "selection_contract_version": str(
            settings.get("version") or _SELECTION_CONTRACT_VERSION
        ).strip() or _SELECTION_CONTRACT_VERSION,
        "hard_gate_version": str(
            settings.get("hard_gate_version") or "promotion_hard_gate_v1"
        ).strip() or "promotion_hard_gate_v1",
        "min_promotion_fold_pass_rate": float(
            settings.get("min_promotion_fold_pass_rate", 0.70)
        ),
        "min_oos_is_calmar_ratio_median": float(
            settings.get("min_oos_is_calmar_ratio_median", 0.60)
        ),
        "max_oos_mdd_depth_p95": float(
            settings.get("max_oos_mdd_depth_p95", 0.25)
        ),
        "per_fold_min_oos_calmar_ratio": float(
            settings.get("per_fold_min_oos_calmar_ratio", 0.0)
        ),
        "reserve_count": max(int(settings.get("reserve_count", 2) or 2), 0),
        "tie_break_rule": [
            "hard_gate_pass desc",
            "promotion_fold_pass_rate desc",
            "promotion_oos_calmar_median desc",
            "promotion_oos_mdd_depth_worst asc",
            "promotion_oos_cagr_median desc",
            "candidate_signature asc",
        ],
    }


def _resolve_research_runtime_settings(wfo_settings: dict) -> dict:
    research_mode = str(
        wfo_settings.get("research_mode") or _RESEARCH_WFO_MODE
    ).strip() or _RESEARCH_WFO_MODE
    if research_mode != _RESEARCH_WFO_MODE:
        raise ValueError(
            "research_start_date_robustness only supports "
            f"research_mode={_RESEARCH_WFO_MODE}."
        )
    shortlist_path = str(wfo_settings.get("research_shortlist_path") or "").strip()
    if not shortlist_path:
        raise ValueError(
            "research_start_date_robustness requires walk_forward_settings.research_shortlist_path."
        )
    anchor_dates = list(wfo_settings.get("research_anchor_start_dates") or [])
    if not anchor_dates:
        raise ValueError(
            "research_start_date_robustness requires walk_forward_settings.research_anchor_start_dates."
        )
    return {
        "research_mode": research_mode,
        "research_shortlist_path": shortlist_path,
        "research_anchor_start_dates": [_coerce_date(item).isoformat() for item in anchor_dates],
        "anchor_set_id": str(wfo_settings.get("anchor_set_id") or "manual_anchor_set").strip(),
        "anchor_spacing_rule": str(
            wfo_settings.get("anchor_spacing_rule") or "manual_explicit"
        ).strip(),
        "coverage_normalized": bool(wfo_settings.get("coverage_normalized", True)),
    }


def _build_current_lane_reasons(
    *,
    lane_type: str,
    total_folds: int,
    overlap_days: int,
    cpu_audit_outcome: str,
) -> list[str]:
    reasons = []
    if lane_type == "legacy_wfo":
        reasons.append("lane_mode_not_separated")
        if int(total_folds) > 1:
            reasons.append("oos_initial_cash_carry_over_enabled")
            reasons.append("composite_curve_mean_aggregation_enabled")
    if int(overlap_days) > 0:
        reasons.append(f"oos_fold_overlap_days={int(overlap_days)}")
    if lane_type == "promotion_evaluation" and cpu_audit_outcome != "pass":
        reasons.append("cpu_audit_required_for_promotion")
    if cpu_audit_outcome not in ("disabled", "pass"):
        reasons.append("cpu_audit_contract_not_met")
    return reasons


def build_anchor_manifest(
    *,
    anchor_set_id: str,
    anchor_dates: list[str],
    anchor_spacing_rule: str,
    minimum_is_length_days: int,
    minimum_oos_length_days: int,
    coverage_normalized: bool,
    shortlist_freeze_mode: str = _RESEARCH_WFO_MODE,
) -> dict:
    return {
        "anchor_set_id": str(anchor_set_id),
        "anchor_dates": [str(item) for item in anchor_dates],
        "anchor_spacing_rule": str(anchor_spacing_rule),
        "minimum_is_length_days": int(minimum_is_length_days),
        "minimum_oos_length_days": int(minimum_oos_length_days),
        "shortlist_freeze_mode": str(shortlist_freeze_mode),
        "coverage_normalized": bool(coverage_normalized),
    }


def _build_legacy_fold_periods(start_date, end_date, total_folds: int, period_length_days: int) -> tuple[list[dict], int]:
    import pandas as pd

    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    total = int(total_folds)
    period_days = int(period_length_days)
    length = pd.Timedelta(days=period_days)
    if total <= 0 or period_days <= 0:
        raise ValueError("total_folds and period_length_days must be positive.")

    max_shift_days = (end - start).days - (period_days - 1)
    if max_shift_days >= period_days:
        shift_days = period_days
    else:
        shift_days = min(max_shift_days, (period_days + 1) // 2 + 1)
        if shift_days < 1:
            shift_days = 1
    shift = pd.Timedelta(days=shift_days)
    last_is_start = end - shift - (length - pd.Timedelta(days=1))
    span_days = (last_is_start - start).days
    if span_days <= 0:
        raise ValueError(
            "Configuration Error: Cannot fit legacy WFO folds within the requested window."
        )

    is_starts = [start]
    if total > 1:
        base_step = span_days // (total - 1)
        remainder = span_days % (total - 1)
        for index in range(1, total):
            add_days = base_step + (1 if index <= remainder else 0)
            is_starts.append(is_starts[-1] + pd.Timedelta(days=add_days))

    fold_periods = []
    for index, is_start in enumerate(is_starts, start=1):
        is_end = is_start + length - pd.Timedelta(days=1)
        oos_start = is_start + shift
        oos_end = oos_start + length - pd.Timedelta(days=1)
        if oos_start < is_start + pd.Timedelta(days=1):
            raise ValueError("Causality violated: OOS must start after IS start.")
        if oos_end > end or is_start < start or is_end > end:
            raise ValueError("Legacy WFO fold is out of the configured date range.")
        fold_periods.append(
            {
                "Fold": index,
                "IS_Start": is_start.date(),
                "IS_End": is_end.date(),
                "OOS_Start": oos_start.date(),
                "OOS_End": oos_end.date(),
            }
        )
    return fold_periods, period_days - shift_days


def _build_promotion_fold_periods(start_date, end_date, total_folds: int, period_length_days: int) -> tuple[list[dict], int]:
    import pandas as pd

    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    total = int(total_folds)
    period_days = int(period_length_days)
    length = pd.Timedelta(days=period_days)
    if total <= 0 or period_days <= 0:
        raise ValueError("total_folds and period_length_days must be positive.")

    total_days = int((end - start).days + 1)
    initial_is_days = total_days - (total * period_days)
    if initial_is_days <= 0:
        raise ValueError(
            "promotion_evaluation requires a longer history window: "
            "total window must be larger than total_folds * period_length_days."
        )

    fold_periods = []
    initial_is_length = pd.Timedelta(days=initial_is_days)
    for index in range(total):
        is_start = start
        is_end = start + initial_is_length + (index * length) - pd.Timedelta(days=1)
        oos_start = is_end + pd.Timedelta(days=1)
        oos_end = oos_start + length - pd.Timedelta(days=1)
        if oos_end > end:
            raise ValueError("promotion_evaluation OOS period exceeds configured end_date.")
        fold_periods.append(
            {
                "Fold": index + 1,
                "IS_Start": is_start.date(),
                "IS_End": is_end.date(),
                "OOS_Start": oos_start.date(),
                "OOS_End": oos_end.date(),
            }
        )
    return fold_periods, 0


def _build_lane_execution_contract(
    lane_type: str,
    *,
    start_date,
    end_date,
    total_folds: int,
    period_length_days: int,
) -> dict:
    if lane_type == "promotion_evaluation":
        fold_periods, overlap_days = _build_promotion_fold_periods(
            start_date,
            end_date,
            total_folds,
            period_length_days,
        )
        return {
            "fold_periods": fold_periods,
            "overlap_days": overlap_days,
            "carry_over_oos_initial_cash": True,
            "composite_curve_allowed": bool(int(total_folds) > 1),
            "curve_aggregation_mode": "stitch_non_overlap",
        }

    fold_periods, overlap_days = _build_legacy_fold_periods(
        start_date,
        end_date,
        total_folds,
        period_length_days,
    )
    return {
        "fold_periods": fold_periods,
        "overlap_days": overlap_days,
        "carry_over_oos_initial_cash": True,
        "composite_curve_allowed": bool(int(total_folds) > 1),
        "curve_aggregation_mode": "mean_overlap",
    }


def _build_research_anchor_contracts(
    anchor_dates: list[str],
    *,
    end_date,
    total_folds: int,
    period_length_days: int,
) -> list[dict]:
    contracts = []
    for anchor_index, anchor_date in enumerate(anchor_dates, start=1):
        fold_periods, overlap_days = _build_promotion_fold_periods(
            anchor_date,
            end_date,
            total_folds,
            period_length_days,
        )
        contracts.append(
            {
                "anchor_id": f"A{anchor_index}",
                "anchor_start_date": _coerce_date(anchor_date).isoformat(),
                "fold_periods": fold_periods,
                "overlap_days": overlap_days,
            }
        )
    return contracts


def _resolve_oos_initial_cash(
    all_oos_curves: list,
    *,
    initial_cash: float,
    carry_over_enabled: bool,
) -> float:
    if not carry_over_enabled or not all_oos_curves:
        return float(initial_cash)
    return float(all_oos_curves[-1].iloc[-1])


def _aggregate_oos_curves(all_oos_curves: list, *, mode: str):
    import pandas as pd

    if not all_oos_curves:
        return pd.Series(dtype=float)
    if mode == "mean_overlap":
        return pd.concat(all_oos_curves).sort_index().groupby(level=0).mean()
    if mode == "stitch_non_overlap":
        combined = pd.concat(all_oos_curves).sort_index()
        if combined.index.duplicated(keep=False).any():
            raise ValueError(
                "promotion_evaluation requires non-overlap OOS periods; duplicated OOS dates detected."
            )
        return combined
    raise ValueError(f"Unsupported curve aggregation mode: {mode}")


def _analyze_equity_curve(curve, analyzer_cls):
    import pandas as pd

    if curve is None or curve.empty:
        raise ValueError("Equity curve is empty; cannot analyze candidate metrics.")
    history_df = pd.DataFrame(curve, columns=["total_value"])
    analyzer = analyzer_cls(history_df)
    return dict(analyzer.get_metrics(formatted=False))


def _evaluate_shortlist_candidates(
    shortlist_df,
    *,
    start_date: str,
    end_date: str,
    initial_cash: float,
    base_strategy_params: dict,
    backtest_runner,
    analyzer_cls,
    metric: str,
):
    import pandas as pd

    rows = []
    for _, shortlist_row in shortlist_df.iterrows():
        params_dict = _extract_strategy_params(shortlist_row.to_dict(), base_strategy_params)
        equity_curve = backtest_runner(
            start_date=start_date,
            end_date=end_date,
            params_dict=params_dict,
            initial_cash=initial_cash,
        )
        metrics = _analyze_equity_curve(equity_curve, analyzer_cls)
        record = dict(shortlist_row.to_dict())
        record.update(params_dict)
        record.update(metrics)
        rows.append(record)
    evaluated_df = pd.DataFrame(rows)
    return _sort_candidate_frame(evaluated_df, metric).reset_index(drop=True)


def _candidate_signature(candidate_row: dict) -> str:
    signature = []
    for key in _CPU_CERT_PARAM_KEYS:
        value = candidate_row.get(key)
        if key == "additional_buy_priority":
            value = _normalize_priority_for_cpu(value)
        signature.append(f"{key}={value}")
    return "|".join(signature)


def _safe_positive_ratio(numerator, denominator):
    try:
        resolved_numerator = float(numerator)
        resolved_denominator = float(denominator)
    except (TypeError, ValueError):
        return None
    if resolved_denominator <= 0.0:
        return None
    return resolved_numerator / resolved_denominator


def _build_metric_snapshot(evaluated_df, *, prefix: str):
    columns = ["shortlist_candidate_id", "cagr", "mdd", "calmar_ratio"]
    snapshot = evaluated_df[columns].copy()
    snapshot = snapshot.rename(
        columns={
            "cagr": f"{prefix}_cagr",
            "mdd": f"{prefix}_mdd",
            "calmar_ratio": f"{prefix}_calmar_ratio",
        }
    )
    return snapshot


def _build_promotion_candidate_fold_metrics(
    shortlist_df,
    *,
    fold_num: int,
    is_start: str,
    is_end: str,
    oos_start: str,
    oos_end: str,
    initial_cash: float,
    base_strategy_params: dict,
    backtest_runner,
    analyzer_cls,
    metric: str,
    selection_settings: dict,
    is_evaluated_df=None,
):
    is_df = is_evaluated_df
    if is_df is None:
        is_df = _evaluate_shortlist_candidates(
            shortlist_df,
            start_date=is_start,
            end_date=is_end,
            initial_cash=initial_cash,
            base_strategy_params=base_strategy_params,
            backtest_runner=backtest_runner,
            analyzer_cls=analyzer_cls,
            metric=metric,
        )
    oos_df = _evaluate_shortlist_candidates(
        shortlist_df,
        start_date=oos_start,
        end_date=oos_end,
        initial_cash=initial_cash,
        base_strategy_params=base_strategy_params,
        backtest_runner=backtest_runner,
        analyzer_cls=analyzer_cls,
        metric=metric,
    )
    merged = shortlist_df.copy()
    merged = merged.merge(
        _build_metric_snapshot(is_df, prefix="is"),
        on="shortlist_candidate_id",
        how="left",
    )
    merged = merged.merge(
        _build_metric_snapshot(oos_df, prefix="oos"),
        on="shortlist_candidate_id",
        how="left",
    )
    merged["fold"] = int(fold_num)
    merged["IS_Start"] = str(is_start)
    merged["IS_End"] = str(is_end)
    merged["OOS_Start"] = str(oos_start)
    merged["OOS_End"] = str(oos_end)
    merged["candidate_signature"] = [
        _candidate_signature(row)
        for row in merged.to_dict("records")
    ]
    merged["oos_is_calmar_ratio"] = [
        _safe_positive_ratio(row.get("oos_calmar_ratio"), row.get("is_calmar_ratio"))
        for row in merged.to_dict("records")
    ]
    merged["oos_mdd_depth"] = merged["oos_mdd"].abs()
    merged["fold_gate_pass"] = (
        (merged["oos_calmar_ratio"].fillna(float("-inf")) >= selection_settings["per_fold_min_oos_calmar_ratio"])
        & (merged["oos_mdd_depth"].fillna(float("inf")) <= selection_settings["max_oos_mdd_depth_p95"])
    )
    return merged


def _summarize_promotion_candidates(fold_metrics_df, selection_settings: dict):
    import pandas as pd

    rows = []
    for _, group in fold_metrics_df.groupby("shortlist_candidate_id", sort=True):
        first_row = group.iloc[0].to_dict()
        ratio_series = pd.to_numeric(group["oos_is_calmar_ratio"], errors="coerce").dropna()
        oos_calmar_series = pd.to_numeric(group["oos_calmar_ratio"], errors="coerce").dropna()
        oos_cagr_series = pd.to_numeric(group["oos_cagr"], errors="coerce").dropna()
        oos_mdd_depth_series = pd.to_numeric(group["oos_mdd_depth"], errors="coerce").dropna()
        fold_pass_rate = float(pd.Series(group["fold_gate_pass"]).mean())
        ratio_median = float(ratio_series.median()) if not ratio_series.empty else None
        oos_mdd_depth_p95 = (
            float(oos_mdd_depth_series.quantile(0.95))
            if not oos_mdd_depth_series.empty
            else None
        )
        oos_mdd_depth_worst = (
            float(oos_mdd_depth_series.max())
            if not oos_mdd_depth_series.empty
            else None
        )
        hard_gate_pass = (
            ratio_median is not None
            and ratio_median >= selection_settings["min_oos_is_calmar_ratio_median"]
            and fold_pass_rate >= selection_settings["min_promotion_fold_pass_rate"]
            and oos_mdd_depth_p95 is not None
            and oos_mdd_depth_p95 <= selection_settings["max_oos_mdd_depth_p95"]
        )
        rows.append(
            {
                "shortlist_candidate_id": int(first_row["shortlist_candidate_id"]),
                "candidate_signature": first_row["candidate_signature"],
                "promotion_fold_count": int(len(group)),
                "promotion_fold_pass_count": int(group["fold_gate_pass"].sum()),
                "promotion_fold_pass_rate": fold_pass_rate,
                "promotion_oos_calmar_median": float(oos_calmar_series.median()),
                "promotion_oos_cagr_median": float(oos_cagr_series.median()),
                "promotion_oos_mdd_depth_p95": oos_mdd_depth_p95,
                "promotion_oos_mdd_depth_worst": oos_mdd_depth_worst,
                "promotion_oos_is_calmar_ratio_median": ratio_median,
                "hard_gate_pass": bool(hard_gate_pass),
                **{key: first_row.get(key) for key in _CPU_CERT_PARAM_KEYS},
            }
        )
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    summary_df = summary_df.sort_values(
        [
            "hard_gate_pass",
            "promotion_fold_pass_rate",
            "promotion_oos_calmar_median",
            "promotion_oos_mdd_depth_worst",
            "promotion_oos_cagr_median",
            "candidate_signature",
        ],
        ascending=[False, False, False, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    summary_df["selection_rank"] = summary_df.index + 1
    return summary_df


def _build_final_candidate_manifest(
    candidate_summary_df,
    *,
    selection_settings: dict,
    shortlist_hash: str | None,
    decision_date,
    research_data_cutoff,
    promotion_data_cutoff,
    holdout_settings: dict,
    engine_version_hash: str | None,
    cpu_audit_required: bool,
):
    if candidate_summary_df.empty:
        raise ValueError("Promotion candidate summary is empty.")
    champion = candidate_summary_df.iloc[0].to_dict()
    reserve_count = selection_settings["reserve_count"]
    reserve_candidate_ids = [
        int(item)
        for item in candidate_summary_df["shortlist_candidate_id"].iloc[1 : 1 + reserve_count].tolist()
    ]
    ranking_records = candidate_summary_df[
        [
            "selection_rank",
            "shortlist_candidate_id",
            "candidate_signature",
            "hard_gate_pass",
            "promotion_fold_pass_rate",
            "promotion_oos_calmar_median",
            "promotion_oos_mdd_depth_worst",
            "promotion_oos_cagr_median",
        ]
    ].to_dict("records")
    champion_payload = {
        "shortlist_candidate_id": int(champion["shortlist_candidate_id"]),
        "candidate_signature": champion["candidate_signature"],
        **{key: champion.get(key) for key in _CPU_CERT_PARAM_KEYS},
    }
    readiness_reasons = []
    if not bool(champion.get("hard_gate_pass")):
        readiness_reasons.append("no_candidate_passed_hard_gate")
    if cpu_audit_required:
        readiness_reasons.append("final_candidate_cpu_audit_not_executed")
    return {
        "selection_contract_version": selection_settings["selection_contract_version"],
        "hard_gate_version": selection_settings["hard_gate_version"],
        "hard_gate_thresholds": {
            "min_promotion_fold_pass_rate": selection_settings["min_promotion_fold_pass_rate"],
            "min_oos_is_calmar_ratio_median": selection_settings["min_oos_is_calmar_ratio_median"],
            "max_oos_mdd_depth_p95": selection_settings["max_oos_mdd_depth_p95"],
            "per_fold_min_oos_calmar_ratio": selection_settings["per_fold_min_oos_calmar_ratio"],
        },
        "tie_break_rule": list(selection_settings["tie_break_rule"]),
        "decision_date": _coerce_date(decision_date or datetime.now()).isoformat(),
        "research_data_cutoff": str(research_data_cutoff) if research_data_cutoff else None,
        "promotion_data_cutoff": str(promotion_data_cutoff) if promotion_data_cutoff else None,
        "holdout_start": holdout_settings.get("holdout_start"),
        "holdout_end": holdout_settings.get("holdout_end"),
        "promotion_shortlist_hash": shortlist_hash,
        "candidate_ranking_hash": _hash_json_sha256(ranking_records),
        "final_candidate_hash": _hash_json_sha256(champion_payload),
        "champion_candidate_id": int(champion["shortlist_candidate_id"]),
        "champion_candidate_signature": champion["candidate_signature"],
        "reserve_candidate_ids": reserve_candidate_ids,
        "champion_hard_gate_pass": bool(champion.get("hard_gate_pass")),
        "cpu_audit_required": bool(cpu_audit_required),
        "cpu_audit_outcome": "pending_final_candidate_audit" if cpu_audit_required else "disabled",
        "holdout_ready": not readiness_reasons,
        "holdout_readiness_reasons": readiness_reasons,
        "engine_version_hash": engine_version_hash or "unknown",
    }


def _build_metric_distribution_summary(metrics_df, metric_columns: tuple[str, ...]) -> dict:
    import pandas as pd

    if metrics_df.empty:
        return {"row_count": 0, "metrics": {}}
    summary = {"row_count": int(len(metrics_df)), "metrics": {}}
    for column in metric_columns:
        if column not in metrics_df.columns:
            continue
        series = pd.to_numeric(metrics_df[column], errors="coerce").dropna()
        if series.empty:
            continue
        summary["metrics"][column] = {
            "min": float(series.min()),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "max": float(series.max()),
        }
    return summary


def _find_selected_finalist_index(finalists_df: "pd.DataFrame", selected_params_dict: dict) -> int:
    selected_signature = _normalize_param_signature(selected_params_dict)
    for idx, signature in enumerate(_build_param_signatures(finalists_df)):
        if signature == selected_signature:
            return int(idx)
    raise ValueError("GPU-selected finalist is missing from CPU audit shortlist.")


def _normalize_priority_for_cpu(value) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in ("lowest_order", "highest_drop"):
            return normalized
    try:
        return "highest_drop" if int(float(value)) == 1 else "lowest_order"
    except (TypeError, ValueError):
        return "lowest_order"


def _normalize_param_signature(mapping: dict) -> tuple:
    signature = []
    for key in _CPU_CERT_PARAM_KEYS:
        value = mapping.get(key)
        if key == "additional_buy_priority":
            signature.append(_normalize_priority_for_cpu(value))
        elif key in _CPU_CERT_INT_KEYS:
            signature.append(int(value))
        elif key in _CPU_CERT_FLOAT_KEYS:
            signature.append(round(float(value), 10))
        else:
            signature.append(value)
    return tuple(signature)


def _build_param_signatures(df: "pd.DataFrame") -> list[tuple]:
    normalized_columns = []
    for key in _CPU_CERT_PARAM_KEYS:
        values = df[key].tolist()
        if key == "additional_buy_priority":
            normalized = [_normalize_priority_for_cpu(value) for value in values]
        elif key in _CPU_CERT_INT_KEYS:
            normalized = [int(value) for value in values]
        elif key in _CPU_CERT_FLOAT_KEYS:
            normalized = [round(float(value), 10) for value in values]
        else:
            normalized = values
        normalized_columns.append(normalized)
    return list(zip(*normalized_columns))


def _sort_candidate_frame(df: "pd.DataFrame", primary_metric: str) -> "pd.DataFrame":
    sort_cols = [primary_metric]
    ascending = [False]
    for column in _CPU_CERT_SORT_COLUMNS:
        if column in df.columns and column != primary_metric:
            sort_cols.append(column)
            ascending.append(False)
    return df.sort_values(sort_cols, ascending=ascending, kind="stable")


def _get_cpu_certification_settings(wfo_settings: dict) -> dict:
    top_n = int(wfo_settings.get("cpu_certification_top_n", 5) or 5)
    metric = str(wfo_settings.get("cpu_certification_metric", "calmar_ratio")).strip() or "calmar_ratio"
    return {
        "enabled": bool(wfo_settings.get("cpu_certification_enabled", False)),
        "top_n": max(top_n, 1),
        "metric": metric,
    }


def build_gpu_finalist_shortlist(
    simulation_results_df: "pd.DataFrame",
    robust_params_dict: dict,
    *,
    top_n: int,
    metric: str,
) -> "pd.DataFrame":
    import pandas as pd

    if simulation_results_df.empty:
        raise ValueError("GPU simulation results are empty; cannot build finalist shortlist.")
    if metric not in simulation_results_df.columns:
        raise ValueError(f"CPU certification metric '{metric}' is missing from GPU results.")

    ranked = simulation_results_df.reset_index().rename(columns={"index": "gpu_result_index"}).copy()
    ranked = _sort_candidate_frame(ranked, metric).reset_index(drop=True)
    ranked["gpu_rank"] = ranked.index + 1

    finalists = ranked.head(max(int(top_n), 1)).copy()
    finalists["selection_reason"] = "gpu_top_n"
    finalist_signatures = {
        signature: idx
        for idx, signature in enumerate(_build_param_signatures(finalists))
    }
    robust_signature = _normalize_param_signature(robust_params_dict)

    if robust_signature in finalist_signatures:
        row_idx = finalist_signatures[robust_signature]
        finalists.at[row_idx, "selection_reason"] = "gpu_top_n+robust_cluster"
        return finalists.reset_index(drop=True)

    ranked_signatures = _build_param_signatures(ranked)
    robust_match_indices = [
        idx for idx, signature in enumerate(ranked_signatures) if signature == robust_signature
    ]
    robust_matches = ranked.iloc[robust_match_indices]
    if not robust_matches.empty:
        robust_row = robust_matches.head(1).copy()
    else:
        robust_row = pd.DataFrame([robust_params_dict])
        robust_row["gpu_result_index"] = pd.NA
        robust_row["gpu_rank"] = pd.NA
    robust_row["selection_reason"] = "robust_cluster"

    finalists = pd.concat([finalists, robust_row], ignore_index=True, sort=False)
    return finalists.reset_index(drop=True)


def _extract_strategy_params(candidate_row: dict, base_strategy_params: dict) -> dict:
    params = dict(base_strategy_params)
    for key in _CPU_CERT_PARAM_KEYS:
        if key not in candidate_row:
            continue
        value = candidate_row[key]
        if key == "additional_buy_priority":
            params[key] = _normalize_priority_for_cpu(value)
        elif key in _CPU_CERT_INT_KEYS:
            params[key] = int(value)
        elif key in _CPU_CERT_FLOAT_KEYS:
            params[key] = float(value)
        else:
            params[key] = value
    return params


def _build_cpu_backtest_config(
    base_config: dict,
    *,
    start_date: str,
    end_date: str,
    initial_cash: float,
    strategy_params: dict,
) -> dict:
    config = {
        "database": dict(base_config["database"]),
        "backtest_settings": dict(base_config["backtest_settings"]),
        "strategy_params": dict(base_config["strategy_params"]),
        "execution_params": dict(base_config["execution_params"]),
        "paths": dict(base_config.get("paths", {})),
    }
    config["backtest_settings"].update(
        {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "initial_cash": float(initial_cash),
            "save_full_trade_history": False,
        }
    )
    config["strategy_params"].update(strategy_params)
    return config


def _cpu_daily_values_to_series(cpu_result: dict) -> "pd.Series":
    import pandas as pd

    if not cpu_result or not cpu_result.get("success"):
        raise ValueError(
            f"CPU backtest failed: {cpu_result.get('error') if cpu_result else 'empty result'}"
        )
    daily_values = cpu_result.get("daily_values", [])
    if not daily_values:
        return pd.Series(dtype=float)
    df = pd.DataFrame(daily_values)
    if df.empty or "x" not in df.columns or "y" not in df.columns:
        return pd.Series(dtype=float)
    df["x"] = pd.to_datetime(df["x"])
    df = df.sort_values("x")
    return pd.Series(df["y"].values, index=df["x"])


def run_cpu_single_backtest(
    base_config: dict,
    *,
    start_date: str,
    end_date: str,
    params_dict: dict,
    initial_cash: float,
) -> tuple["pd.Series", dict]:
    from ..main_backtest import run_backtest_from_config

    strategy_params = _extract_strategy_params(params_dict, base_config["strategy_params"])
    cpu_config = _build_cpu_backtest_config(
        base_config,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        strategy_params=strategy_params,
    )
    cpu_result = run_backtest_from_config(cpu_config, persist_artifacts=False)
    return _cpu_daily_values_to_series(cpu_result), cpu_result


def certify_gpu_finalists_with_cpu(
    base_config: dict,
    finalists_df: "pd.DataFrame",
    *,
    selected_params_dict: dict,
    start_date: str,
    end_date: str,
    initial_cash: float,
    metric: str,
    top_n_requested: int | None = None,
) -> tuple[dict, "pd.DataFrame"]:
    import pandas as pd

    if finalists_df.empty:
        raise ValueError("CPU certification finalists are empty.")
    selected_idx = _find_selected_finalist_index(finalists_df, selected_params_dict)
    cpu_metric_col = f"cpu_{metric}"
    records = []

    for idx, finalist in finalists_df.iterrows():
        finalist_row = finalist.to_dict()
        strategy_params = _extract_strategy_params(finalist_row, base_config["strategy_params"])
        cpu_curve = pd.Series(dtype=float)
        cpu_result = {"success": False, "error": None}
        metrics = {}
        try:
            cpu_curve, cpu_result = run_cpu_single_backtest(
                base_config,
                start_date=start_date,
                end_date=end_date,
                params_dict=strategy_params,
                initial_cash=initial_cash,
            )
            metrics = dict(cpu_result.get("metrics", {}))
        except Exception as exc:  # pragma: no cover - defensive path
            cpu_result = {"success": False, "error": str(exc)}
        record = {
            "gpu_rank": finalist_row.get("gpu_rank"),
            "gpu_result_index": finalist_row.get("gpu_result_index"),
            "selection_reason": finalist_row.get("selection_reason", "gpu_top_n"),
            "is_gpu_selected_candidate": bool(int(idx) == selected_idx),
            "cpu_success": bool(cpu_result.get("success", False)),
            "cpu_degraded_run": bool(cpu_result.get("degraded_run", False)),
            "cpu_promotion_blocked": bool(cpu_result.get("promotion_blocked", False)),
            "cpu_equity_points": int(len(cpu_curve)),
            "cpu_error": cpu_result.get("error"),
        }
        for key in _CPU_CERT_PARAM_KEYS:
            record[key] = strategy_params.get(key)
            if key in finalist_row:
                record[f"gpu_{key}"] = finalist_row.get(key)
        for column in _CPU_CERT_SORT_COLUMNS:
            if column in finalist_row:
                record[f"gpu_{column}"] = finalist_row.get(column)
            record[f"cpu_{column}"] = metrics.get(column)
        if metric in finalist_row and metrics.get(metric) is not None:
            record[f"{metric}_delta_cpu_minus_gpu"] = float(metrics.get(metric)) - float(finalist_row[metric])
        record["cpu_certified"] = record["cpu_success"] and not record["cpu_promotion_blocked"]
        records.append(record)

    certification_df = pd.DataFrame(records)
    failed_df = certification_df[~certification_df["cpu_success"]].copy()
    if not failed_df.empty:
        raise RuntimeError(
            f"CPU certification encountered {len(failed_df)} failed finalists."
        )
    if not certification_df["cpu_certified"].any():
        raise RuntimeError("CPU certification rejected every GPU finalist.")
    if cpu_metric_col not in certification_df.columns:
        raise ValueError(f"CPU certification metric '{metric}' was not produced by CPU backtest.")
    selected_rows = certification_df[certification_df["is_gpu_selected_candidate"]].copy()
    if selected_rows.empty:
        raise RuntimeError("CPU certification lost the GPU-selected finalist during audit.")
    selected_row = selected_rows.iloc[0].to_dict()
    if not bool(selected_row.get("cpu_certified")):
        raise RuntimeError("GPU-selected finalist failed CPU audit.")

    audited_params = _extract_strategy_params(selected_params_dict, base_config["strategy_params"])
    audited_params.update(
        {
            "selection_source": "gpu_selected_finalist_cpu_audited",
            "cpu_certification_metric": metric,
            "cpu_certification_top_n": int(top_n_requested or len(finalists_df)),
            "cpu_certification_shortlist_size": int(len(finalists_df)),
            "cpu_certification_gpu_rank": selected_row.get("gpu_rank"),
            "cpu_audit_outcome": "pass",
            cpu_metric_col: selected_row.get(cpu_metric_col),
            "cpu_cagr": selected_row.get("cpu_cagr"),
            "cpu_mdd": selected_row.get("cpu_mdd"),
        }
    )
    return audited_params, certification_df

# --- Clustering Helper Function ---
def find_robust_parameters(
    simulation_results_df: "pd.DataFrame",
    param_cols: list,
    metric_cols: list,
    k_range: tuple = (2, 11),
    min_cluster_size_ratio: float = 0.05
) -> tuple[dict, "pd.DataFrame | None"]:
    """
    K-Means 클러스터링을 사용하여 시뮬레이션 결과에서 가장 강건한 파라미터 조합을 찾습니다.
    (WFO 파이프라인에 통합하기 위해 시각화 코드는 제거된 버전)
    """
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    print("\n--- 4a. Robust Parameter Search via Clustering ---")
    features = param_cols + metric_cols
    df = simulation_results_df[features].dropna()
    
    if df.empty or len(df) < k_range[0]:
        print("[Warning] Not enough data for clustering. Returning best result by Calmar.")
        best_by_calmar = simulation_results_df.sort_values('calmar_ratio', ascending=False).iloc[0]
        return best_by_calmar.to_dict(), None

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    
    best_k, best_score = -1, -1
    k_candidates = range(k_range[0], min(k_range[1], len(df)))
    for k in k_candidates:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(scaled_features)
        if len(np.unique(cluster_labels)) < 2: continue
        score = silhouette_score(scaled_features, cluster_labels)
        if score > best_score:
            best_score = score
            best_k = k
            
    if best_k == -1: best_k = k_range[0] # Fallback
    print(f"  - Optimal k detected: {best_k} (Silhouette Score: {best_score:.4f})")
    
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    cluster_summary = df.groupby('cluster')[metric_cols].mean()
    cluster_summary['size'] = df['cluster'].value_counts()
    
    # Z-score 계산 시 분모가 0이 되는 경우 방지
    calmar_std = cluster_summary['calmar_ratio'].std()
    denominator = calmar_std if calmar_std > 0 else 1
    cluster_summary['calmar_zscore'] = (cluster_summary['calmar_ratio'] - cluster_summary['calmar_ratio'].mean()) / denominator
    cluster_summary['robustness_score'] = cluster_summary['calmar_zscore'] * np.log1p(cluster_summary['size'])
    
    min_cluster_size = int(len(df) * min_cluster_size_ratio)
    qualified_clusters = cluster_summary[cluster_summary['size'] >= min_cluster_size]
    
    robust_cluster_id = qualified_clusters['robustness_score'].idxmax() if not qualified_clusters.empty else cluster_summary['robustness_score'].idxmax()
    print(f"  - Most robust cluster identified: Cluster {robust_cluster_id}")
    
    robust_cluster_df = df[df['cluster'] == robust_cluster_id]
    centroid = kmeans.cluster_centers_[robust_cluster_id]
    
    nn = NearestNeighbors(n_neighbors=1).fit(scaled_features[df.index.isin(robust_cluster_df.index)])
    _, indices = nn.kneighbors([centroid])
    
    closest_point_index = robust_cluster_df.index[indices[0][0]]
    best_params_series = simulation_results_df.loc[closest_point_index]
    
    # WFO 결과 저장을 위해 클러스터링 결과가 포함된 DF 반환
    clustered_df_full = df.reset_index().merge(simulation_results_df.drop(columns=features, errors='ignore'), left_on='index', right_index=True)
    
    return best_params_series.to_dict(), clustered_df_full
# --- 분석 및 시각화 헬퍼 함수 ---
def plot_wfo_results(final_curve: "pd.Series", params_df: "pd.DataFrame", results_dir: str):
    """최종 WFO 결과(수익곡선, 파라미터 분포)를 시각화하고 저장합니다."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from ..performance_analyzer import PerformanceAnalyzer

    print("\n" + "="*80)
    print("🎨 Generating WFO result plots...")
    print("="*80)
    
    # 1. 최종 WFO Equity Curve 및 MDD 플롯
    history_df = pd.DataFrame(final_curve, columns=['total_value'])
    analyzer = PerformanceAnalyzer(history_df)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Equity Curve
    ax1.set_title('Walk-Forward Optimization Equity Curve', fontsize=16)
    ax1.plot(analyzer.daily_values.index, analyzer.daily_values, color='b', label='Equity Curve')
    ax1.set_ylabel('Portfolio Value'); ax1.legend(loc='upper left'); ax1.grid(True)

    # Drawdown
    drawdown = (analyzer.daily_values - analyzer.daily_values.cummax()) / analyzer.daily_values.cummax()
    ax2.fill_between(drawdown.index, drawdown, 0, color='r', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown'); ax2.set_xlabel('Date'); ax2.legend(loc='upper left'); ax2.grid(True)
    
    plt.tight_layout()
    equity_curve_path = os.path.join(results_dir, "wfo_equity_curve.png")
    plt.savefig(equity_curve_path, dpi=300)
    plt.close()
    print(f"✅ WFO Equity Curve plot saved to: {equity_curve_path}")

    # 2. 파라미터 안정성(분포) 플롯
    numeric_params = params_df.select_dtypes(include='number').columns.drop('fold', errors='ignore')
    if not numeric_params.empty:
        cols = 3
        rows = (len(numeric_params) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()

        for i, param in enumerate(numeric_params):
            sns.histplot(data=params_df, x=param, ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {param}')
        
        for j in range(i + 1, len(axes)): fig.delaxes(axes[j])

        plt.tight_layout()
        param_dist_path = os.path.join(results_dir, "wfo_parameter_distribution.png")
        plt.savefig(param_dist_path, dpi=300); plt.close()
        print(f"✅ Parameter Distribution plot saved to: {param_dist_path}")


def _run_research_start_date_robustness(
    *,
    config: dict,
    wfo_settings: dict,
    backtest_settings: dict,
    initial_cash: float,
    results_dir: str,
    holdout_settings: dict,
    backtest_runner,
    analyzer_cls,
):
    import pandas as pd

    research_settings = _resolve_research_runtime_settings(wfo_settings)
    shortlist_df = _load_frozen_shortlist(
        research_settings["research_shortlist_path"],
        setting_name="research_shortlist_path",
        context_label="research shortlist",
    )
    shortlist_hash = (
        str(wfo_settings.get("shortlist_hash") or "").strip()
        or _hash_file_sha256(research_settings["research_shortlist_path"])
    )
    selection_metric = str(
        wfo_settings.get("research_selection_metric")
        or wfo_settings.get("cpu_certification_metric")
        or "calmar_ratio"
    ).strip() or "calmar_ratio"
    anchor_contracts = _build_research_anchor_contracts(
        research_settings["research_anchor_start_dates"],
        end_date=pd.to_datetime(backtest_settings["end_date"]),
        total_folds=int(wfo_settings["total_folds"]),
        period_length_days=int(wfo_settings["period_length_days"]),
    )

    research_rows = []
    selection_rows = []
    for anchor_contract in anchor_contracts:
        for period in anchor_contract["fold_periods"]:
            fold_num = int(period["Fold"])
            is_start = period["IS_Start"].isoformat()
            is_end = period["IS_End"].isoformat()
            oos_start = period["OOS_Start"].isoformat()
            oos_end = period["OOS_End"].isoformat()

            evaluated_df = _evaluate_shortlist_candidates(
                shortlist_df,
                start_date=is_start,
                end_date=is_end,
                initial_cash=initial_cash,
                base_strategy_params=config["strategy_params"],
                backtest_runner=backtest_runner,
                analyzer_cls=analyzer_cls,
                metric=selection_metric,
            )
            selected_row = evaluated_df.iloc[0].to_dict()
            selected_params = _extract_strategy_params(selected_row, config["strategy_params"])
            selection_rows.append(
                {
                    "anchor_id": anchor_contract["anchor_id"],
                    "anchor_start_date": anchor_contract["anchor_start_date"],
                    "fold": fold_num,
                    "selection_metric": selection_metric,
                    "selected_shortlist_candidate_id": selected_row.get("shortlist_candidate_id"),
                    "shortlist_size": int(len(evaluated_df)),
                    "IS_Start": is_start,
                    "IS_End": is_end,
                    "selected_is_calmar_ratio": selected_row.get("calmar_ratio"),
                    "selected_is_cagr": selected_row.get("cagr"),
                    "selected_is_mdd": selected_row.get("mdd"),
                }
            )

            oos_curve = backtest_runner(
                start_date=oos_start,
                end_date=oos_end,
                params_dict=selected_params,
                initial_cash=initial_cash,
            )
            oos_metrics = _analyze_equity_curve(oos_curve, analyzer_cls)
            research_rows.append(
                {
                    "anchor_id": anchor_contract["anchor_id"],
                    "anchor_start_date": anchor_contract["anchor_start_date"],
                    "fold": fold_num,
                    "IS_Start": is_start,
                    "IS_End": is_end,
                    "OOS_Start": oos_start,
                    "OOS_End": oos_end,
                    "selection_metric": selection_metric,
                    "selected_shortlist_candidate_id": selected_row.get("shortlist_candidate_id"),
                    "shortlist_size": int(len(evaluated_df)),
                    "is_calmar_ratio": selected_row.get("calmar_ratio"),
                    "is_cagr": selected_row.get("cagr"),
                    "is_mdd": selected_row.get("mdd"),
                    "oos_calmar_ratio": oos_metrics.get("calmar_ratio"),
                    "oos_cagr": oos_metrics.get("cagr"),
                    "oos_mdd": oos_metrics.get("mdd"),
                    **{key: selected_params.get(key) for key in _CPU_CERT_PARAM_KEYS},
                }
            )

    metrics_df = pd.DataFrame(research_rows)
    metrics_path = os.path.join(results_dir, "research_anchor_fold_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    selection_df = pd.DataFrame(selection_rows)
    selection_path = os.path.join(results_dir, "research_selection_audit.csv")
    selection_df.to_csv(selection_path, index=False)
    summary_path = _write_json_artifact(
        os.path.join(results_dir, "research_metric_distribution_summary.json"),
        _build_metric_distribution_summary(
            metrics_df,
            ("oos_calmar_ratio", "oos_cagr", "oos_mdd", "is_calmar_ratio"),
        ),
    )

    first_is_lengths = [
        _inclusive_day_count(
            contract["fold_periods"][0]["IS_Start"],
            contract["fold_periods"][0]["IS_End"],
        )
        for contract in anchor_contracts
    ]
    anchor_manifest = build_anchor_manifest(
        anchor_set_id=research_settings["anchor_set_id"],
        anchor_dates=research_settings["research_anchor_start_dates"],
        anchor_spacing_rule=research_settings["anchor_spacing_rule"],
        minimum_is_length_days=min(first_is_lengths),
        minimum_oos_length_days=int(wfo_settings["period_length_days"]),
        coverage_normalized=research_settings["coverage_normalized"],
    )

    holdout_start = holdout_settings["holdout_start"]
    holdout_end = holdout_settings["holdout_end"]
    if holdout_start and holdout_end:
        holdout_manifest = build_holdout_manifest(
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            wfo_end=pd.to_datetime(backtest_settings["end_date"]).date().isoformat(),
            contaminated_ranges=holdout_settings["contaminated_ranges"],
            adequacy_metrics={},
            holdout_backtest_executed=False,
        )
    else:
        holdout_manifest = _build_unconfigured_holdout_manifest(
            wfo_end=pd.to_datetime(backtest_settings["end_date"]).date().isoformat(),
            contaminated_ranges=holdout_settings["contaminated_ranges"],
        )

    lane_manifest = build_lane_manifest(
        lane_type="research_start_date_robustness",
        approval_eligible=False,
        decision_date=wfo_settings.get("decision_date"),
        research_data_cutoff=wfo_settings.get("research_data_cutoff")
        or backtest_settings["end_date"],
        promotion_data_cutoff=wfo_settings.get("promotion_data_cutoff")
        or backtest_settings["end_date"],
        shortlist_hash=shortlist_hash,
        publication_lag_policy=wfo_settings.get("publication_lag_policy"),
        ticker_universe_snapshot_id=wfo_settings.get("ticker_universe_snapshot_id"),
        engine_version_hash=wfo_settings.get("engine_version_hash") or os.environ.get("MAGICSPLIT_ENGINE_VERSION_HASH"),
        composite_curve_allowed=False,
        cpu_audit_outcome="not_applicable",
        reasons=[
            "research_lane_distribution_only",
            "oos_initial_cash_reset_each_fold",
            "single_composite_equity_curve_disabled",
        ],
    )
    manifest_paths = write_wfo_manifests(
        results_dir=results_dir,
        lane_manifest=lane_manifest,
        holdout_manifest=holdout_manifest,
        anchor_manifest=anchor_manifest,
    )
    print(f"✅ Research metrics saved to: {metrics_path}")
    print(f"✅ Research selection audit saved to: {selection_path}")
    print(f"✅ Research metric summary saved to: {summary_path}")
    print(f"✅ Lane manifest saved to: {manifest_paths['lane_manifest_path']}")
    print(f"✅ Holdout manifest saved to: {manifest_paths['holdout_manifest_path']}")
    print(f"✅ Anchor manifest saved to: {manifest_paths['anchor_manifest_path']}")

# --- Orchestrator 메인 로직 ---

def run_walk_forward_analysis():
    """
    Walk-Forward Optimization 프로세스 전체를 총괄하는 오케스트레이터 함수.
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # 실제 워커 함수 및 분석 모듈은 GPU 환경에서만 필요하므로 lazy import
    from ..debug_gpu_single_run import run_single_backtest
    from ..parameter_simulation_gpu import find_optimal_parameters
    from ..performance_analyzer import PerformanceAnalyzer

    # 1. 설정 로드
    config = load_config()
    wfo_settings = config['walk_forward_settings']
    backtest_settings = config['backtest_settings']
    initial_cash = backtest_settings['initial_cash']
    cpu_cert_settings = _get_cpu_certification_settings(wfo_settings)

    # 2. [핵심] 모든 기간 파라미터 자동 계산
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("🚀 Starting Robustness-Focused Walk-Forward Optimization")
    print(
        "[WFO] CPU certification: "
        f"enabled={cpu_cert_settings['enabled']} "
        f"top_n={cpu_cert_settings['top_n']} "
        f"metric={cpu_cert_settings['metric']}"
    )

    # 사용자 설정값 추출
    total_start_date = pd.to_datetime(backtest_settings['start_date'])
    total_end_date = pd.to_datetime(backtest_settings['end_date'])
    total_folds = int(wfo_settings['total_folds'])
    period_length_days = int(wfo_settings['period_length_days'])
    lane_type = _resolve_lane_type(wfo_settings)
    holdout_settings = _resolve_holdout_runtime_settings(wfo_settings)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("results", f"wfo_run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    if lane_type == "research_start_date_robustness":
        return _run_research_start_date_robustness(
            config=config,
            wfo_settings=wfo_settings,
            backtest_settings=backtest_settings,
            initial_cash=initial_cash,
            results_dir=results_dir,
            holdout_settings=holdout_settings,
            backtest_runner=run_single_backtest,
            analyzer_cls=PerformanceAnalyzer,
        )

    promotion_settings = None
    selection_contract_settings = None
    promotion_shortlist_df = None
    lane_shortlist_hash = str(wfo_settings.get("shortlist_hash") or "").strip() or None
    if lane_type == "promotion_evaluation":
        promotion_settings = _resolve_promotion_runtime_settings(wfo_settings)
        selection_contract_settings = _resolve_selection_contract_settings(wfo_settings)
        promotion_shortlist_df = _load_frozen_shortlist(
            promotion_settings["promotion_shortlist_path"],
            setting_name="promotion_shortlist_path",
            context_label="promotion shortlist",
        )
        lane_shortlist_hash = promotion_settings["shortlist_hash"]

    lane_contract = _build_lane_execution_contract(
        lane_type,
        start_date=total_start_date,
        end_date=total_end_date,
        total_folds=total_folds,
        period_length_days=period_length_days,
    )
    fold_periods = lane_contract["fold_periods"]
    overlap_days = int(lane_contract["overlap_days"])

    print("\n--- Calculated Walk-Forward Folds ---")
    print(pd.DataFrame(fold_periods).to_string(index=False))

    print(
        "\n[WFO] "
        f"lane_type={lane_type} "
        f"overlap={overlap_days} days "
        f"carry_over_oos_initial_cash={lane_contract['carry_over_oos_initial_cash']} "
        f"curve_aggregation_mode={lane_contract['curve_aggregation_mode']}"
    )


    
    #  새로운 롤링 윈도우 루프
    all_oos_curves, all_optimal_params, all_selection_audits = [], [], []
    promotion_candidate_fold_frames = []
    
    pbar = tqdm(fold_periods, desc="WFO Progress")
    for period in pbar:
        fold_num, is_start, is_end, oos_start, oos_end = period.values()
        pbar.set_description(f"WFO Fold {fold_num}/{total_folds}")
        selection_audit = None

        print(f"\n--- Fold {fold_num} IS Period: {is_start} ~ {is_end} ---")

        if lane_type == "promotion_evaluation":
            evaluated_df = _evaluate_shortlist_candidates(
                promotion_shortlist_df,
                start_date=is_start.strftime('%Y-%m-%d'),
                end_date=is_end.strftime('%Y-%m-%d'),
                initial_cash=initial_cash,
                base_strategy_params=config["strategy_params"],
                backtest_runner=run_single_backtest,
                analyzer_cls=PerformanceAnalyzer,
                metric=promotion_settings["promotion_selection_metric"],
            )
            evaluation_path = os.path.join(
                results_dir,
                f"fold_{fold_num}_promotion_shortlist_evaluation.csv",
            )
            evaluated_df.to_csv(evaluation_path, index=False)
            print(
                "  - Promotion shortlist evaluation complete. "
                f"Analyzed {len(evaluated_df)} frozen candidates."
            )
            print(f"  - Fold {fold_num} shortlist evaluation saved.")
            promotion_candidate_fold_frames.append(
                _build_promotion_candidate_fold_metrics(
                    promotion_shortlist_df,
                    fold_num=fold_num,
                    is_start=is_start.strftime('%Y-%m-%d'),
                    is_end=is_end.strftime('%Y-%m-%d'),
                    oos_start=oos_start.strftime('%Y-%m-%d'),
                    oos_end=oos_end.strftime('%Y-%m-%d'),
                    initial_cash=initial_cash,
                    base_strategy_params=config["strategy_params"],
                    backtest_runner=run_single_backtest,
                    analyzer_cls=PerformanceAnalyzer,
                    metric=promotion_settings["promotion_selection_metric"],
                    selection_settings=selection_contract_settings,
                    is_evaluated_df=evaluated_df,
                )
            )

            selected_row = evaluated_df.iloc[0].to_dict()
            selected_params_dict = _extract_strategy_params(selected_row, config["strategy_params"])
            selection_audit = {
                "fold": fold_num,
                "selection_source": "promotion_frozen_shortlist_is_eval",
                "selection_metric": promotion_settings["promotion_selection_metric"],
                "selected_shortlist_candidate_id": selected_row.get("shortlist_candidate_id"),
                "shortlist_size": int(len(evaluated_df)),
                "IS_Start": is_start.strftime('%Y-%m-%d'),
                "IS_End": is_end.strftime('%Y-%m-%d'),
                "selected_is_calmar_ratio": selected_row.get("calmar_ratio"),
                "selected_is_cagr": selected_row.get("cagr"),
                "selected_is_mdd": selected_row.get("mdd"),
            }
            gpu_shortlist_source_df = evaluated_df
        else:
            # [MODIFIED] 1. IS 기간의 "전체" 시뮬레이션 결과 확보
            _, is_simulation_results_df = find_optimal_parameters(
                 start_date=is_start.strftime('%Y-%m-%d'),
                 end_date=is_end.strftime('%Y-%m-%d'),
                 initial_cash=initial_cash
             )
            print(f"  - IS simulation complete. Analyzing {len(is_simulation_results_df)} combinations.")

            # [NEW] 2. 클러스터링으로 강건 파라미터 탐색
            robust_params_dict, clustered_df = find_robust_parameters(
                simulation_results_df=is_simulation_results_df,
                param_cols=['additional_buy_drop_rate', 'sell_profit_rate', 'stop_loss_rate', 'max_inactivity_period'],
                metric_cols=['cagr', 'mdd', 'calmar_ratio'],
                k_range=(2, 8),
                min_cluster_size_ratio=0.05
            )

            # 디버깅을 위해 각 폴드의 클러스터링 결과 저장
            if clustered_df is not None:
                fold_cluster_path = os.path.join(results_dir, f"fold_{fold_num}_clustered_results.csv")
                clustered_df.to_csv(fold_cluster_path, index=False)
                print(f"  - Fold {fold_num} clustered analysis saved.")

            selected_params_dict = dict(robust_params_dict)
            gpu_shortlist_source_df = is_simulation_results_df

        if cpu_cert_settings["enabled"]:
            gpu_shortlist_df = build_gpu_finalist_shortlist(
                gpu_shortlist_source_df,
                selected_params_dict,
                top_n=cpu_cert_settings["top_n"],
                metric=cpu_cert_settings["metric"],
            )
            shortlist_path = os.path.join(results_dir, f"fold_{fold_num}_gpu_shortlist.csv")
            gpu_shortlist_df.to_csv(shortlist_path, index=False)
            print(
                f"  - GPU finalists for Fold {fold_num} saved "
                f"({len(gpu_shortlist_df)} rows)."
            )

            selected_params_dict, cpu_certification_df = certify_gpu_finalists_with_cpu(
                config,
                gpu_shortlist_df,
                selected_params_dict=selected_params_dict,
                start_date=is_start.strftime('%Y-%m-%d'),
                end_date=is_end.strftime('%Y-%m-%d'),
                initial_cash=initial_cash,
                metric=cpu_cert_settings["metric"],
                top_n_requested=cpu_cert_settings["top_n"],
            )
            certification_path = os.path.join(results_dir, f"fold_{fold_num}_cpu_certification.csv")
            cpu_certification_df.to_csv(certification_path, index=False)
            print(
                f"  - CPU certification for Fold {fold_num} saved "
                f"({len(cpu_certification_df)} rows)."
            )
            if selection_audit is None:
                selection_audit = {"fold": fold_num}
            selection_audit.update(
                {
                    "selection_source": selected_params_dict.get(
                        "selection_source",
                        "gpu_selected_finalist_cpu_audited",
                    ),
                    "cpu_certification_metric": selected_params_dict.get("cpu_certification_metric"),
                    "cpu_certification_top_n": selected_params_dict.get("cpu_certification_top_n"),
                    "cpu_certification_shortlist_size": selected_params_dict.get("cpu_certification_shortlist_size"),
                    "cpu_certification_gpu_rank": selected_params_dict.get("cpu_certification_gpu_rank"),
                    "cpu_audit_outcome": selected_params_dict.get("cpu_audit_outcome"),
                    "cpu_calmar_ratio": selected_params_dict.get("cpu_calmar_ratio"),
                    "cpu_cagr": selected_params_dict.get("cpu_cagr"),
                    "cpu_mdd": selected_params_dict.get("cpu_mdd"),
                }
            )

        if selection_audit is not None:
            if not cpu_cert_settings["enabled"]:
                selection_audit["cpu_audit_outcome"] = "disabled"
            all_selection_audits.append(selection_audit)

        reported_params_dict = _extract_strategy_params(selected_params_dict, config["strategy_params"])
        reported_params_dict['fold'] = fold_num
        all_optimal_params.append(reported_params_dict)
        print(f"  - Final params for Fold {fold_num} selected.")
        
        print(f"--- Fold {fold_num} OOS Period: {oos_start} ~ {oos_end} ---")
        
        # 3. 찾은 파라미터로 OOS 기간 백테스트
        oos_initial_cash = _resolve_oos_initial_cash(
            all_oos_curves,
            initial_cash=initial_cash,
            carry_over_enabled=lane_contract["carry_over_oos_initial_cash"],
        )
        if cpu_cert_settings["enabled"]:
            oos_equity_curve, _ = run_cpu_single_backtest(
                config,
                start_date=oos_start.strftime('%Y-%m-%d'),
                end_date=oos_end.strftime('%Y-%m-%d'),
                params_dict=selected_params_dict,
                initial_cash=oos_initial_cash,
            )
        else:
            oos_equity_curve = run_single_backtest(
                 start_date=oos_start.strftime('%Y-%m-%d'),
                 end_date=oos_end.strftime('%Y-%m-%d'),
                 params_dict=selected_params_dict,
                 initial_cash=oos_initial_cash
             )
        all_oos_curves.append(oos_equity_curve)    
            
    pbar.close()

    # 5. [수정] 최종 결과 종합 및 분석 (고도화)
    print("\n" + "="*80)
    print("📈 Walk-Forward Analysis Finished. Aggregating results...")
    print("="*80)

    if not all_oos_curves:
        print("[ERROR] No Out-of-Sample results were generated.")
        # 단일 폴드 실행 시 여기로 올 수 있으므로, 파라미터 분석만 수행
    else:
        final_wfo_curve = _aggregate_oos_curves(
            all_oos_curves,
            mode=lane_contract["curve_aggregation_mode"],
        )
        wfo_analyzer = PerformanceAnalyzer(pd.DataFrame(final_wfo_curve, columns=['total_value']))
        
        print("\n--- Final WFO Performance Metrics ---")
        for key, value in wfo_analyzer.get_metrics(formatted=True).items():
            print(f"  {key:<25}: {value}")
        
        curve_filepath = os.path.join(results_dir, "wfo_equity_curve_data.csv")
        final_wfo_curve.to_csv(curve_filepath)
        print(f"\n✅ Final WFO equity curve data saved to: {curve_filepath}")
        plot_wfo_results(final_wfo_curve, pd.DataFrame(all_optimal_params), results_dir)

   
   
    # 5-2. 파라미터 안정성 분석 및 결과 저장
    params_df = pd.DataFrame(all_optimal_params)
    print("\n📊 Optimal Parameter Stability Analysis (Descriptive Stats):")
    numeric_params_df = params_df.drop(columns=['fold'], errors='ignore').select_dtypes(include='number')
    if numeric_params_df.empty:
        print("[INFO] Numeric parameter summary is empty.")
    else:
        print(numeric_params_df.describe())
    
    if all_selection_audits:
        selection_audit_df = pd.DataFrame(all_selection_audits)
        selection_audit_path = os.path.join(results_dir, "wfo_selection_audit.csv")
        selection_audit_df.to_csv(selection_audit_path, index=False)
        print(f"✅ WFO selection audit saved to: {selection_audit_path}")

    params_filepath = os.path.join(results_dir, "wfo_robust_parameters.csv")
    params_df.to_csv(params_filepath, index=False)
   
    print(f"\n✅ Robust parameters for each fold saved to: {params_filepath}")

    if lane_type == "promotion_evaluation" and promotion_candidate_fold_frames:
        promotion_candidate_fold_df = pd.concat(
            promotion_candidate_fold_frames,
            ignore_index=True,
        )
        promotion_candidate_fold_path = os.path.join(
            results_dir,
            "promotion_candidate_fold_metrics.csv",
        )
        promotion_candidate_fold_df.to_csv(promotion_candidate_fold_path, index=False)
        promotion_candidate_summary_df = _summarize_promotion_candidates(
            promotion_candidate_fold_df,
            selection_contract_settings,
        )
        promotion_candidate_summary_path = os.path.join(
            results_dir,
            "promotion_candidate_summary.csv",
        )
        promotion_candidate_summary_df.to_csv(
            promotion_candidate_summary_path,
            index=False,
        )
        final_candidate_manifest = _build_final_candidate_manifest(
            promotion_candidate_summary_df,
            selection_settings=selection_contract_settings,
            shortlist_hash=lane_shortlist_hash,
            decision_date=wfo_settings.get("decision_date"),
            research_data_cutoff=wfo_settings.get("research_data_cutoff")
            or total_end_date.date().isoformat(),
            promotion_data_cutoff=wfo_settings.get("promotion_data_cutoff")
            or total_end_date.date().isoformat(),
            holdout_settings=holdout_settings,
            engine_version_hash=wfo_settings.get("engine_version_hash")
            or os.environ.get("MAGICSPLIT_ENGINE_VERSION_HASH"),
            cpu_audit_required=True,
        )
        final_candidate_manifest_path = _write_json_artifact(
            os.path.join(results_dir, "final_candidate_manifest.json"),
            final_candidate_manifest,
        )
        print(
            f"✅ Promotion candidate fold metrics saved to: {promotion_candidate_fold_path}"
        )
        print(
            f"✅ Promotion candidate summary saved to: {promotion_candidate_summary_path}"
        )
        print(
            f"✅ Final candidate manifest saved to: {final_candidate_manifest_path}"
        )

    holdout_start = holdout_settings["holdout_start"]
    holdout_end = holdout_settings["holdout_end"]
    if holdout_start and holdout_end:
        holdout_manifest = build_holdout_manifest(
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            wfo_end=total_end_date.date().isoformat(),
            contaminated_ranges=holdout_settings["contaminated_ranges"],
            adequacy_metrics={},
            holdout_backtest_executed=False,
        )
    else:
        holdout_manifest = _build_unconfigured_holdout_manifest(
            wfo_end=total_end_date.date().isoformat(),
            contaminated_ranges=holdout_settings["contaminated_ranges"],
        )

    cpu_audit_outcome = _resolve_cpu_audit_outcome(cpu_cert_settings, all_selection_audits)
    lane_reasons = _build_current_lane_reasons(
        lane_type=lane_type,
        total_folds=total_folds,
        overlap_days=overlap_days,
        cpu_audit_outcome=cpu_audit_outcome,
    )
    lane_approval_eligible = bool(holdout_manifest.get("approval_eligible")) and not lane_reasons
    lane_manifest = build_lane_manifest(
        lane_type=lane_type,
        approval_eligible=lane_approval_eligible,
        decision_date=wfo_settings.get("decision_date"),
        research_data_cutoff=wfo_settings.get("research_data_cutoff") or total_end_date.date().isoformat(),
        promotion_data_cutoff=wfo_settings.get("promotion_data_cutoff") or total_end_date.date().isoformat(),
        shortlist_hash=lane_shortlist_hash,
        publication_lag_policy=wfo_settings.get("publication_lag_policy"),
        ticker_universe_snapshot_id=wfo_settings.get("ticker_universe_snapshot_id"),
        engine_version_hash=wfo_settings.get("engine_version_hash") or os.environ.get("MAGICSPLIT_ENGINE_VERSION_HASH"),
        composite_curve_allowed=lane_contract["composite_curve_allowed"],
        cpu_audit_outcome=cpu_audit_outcome,
        reasons=lane_reasons,
    )
    manifest_paths = write_wfo_manifests(
        results_dir=results_dir,
        lane_manifest=lane_manifest,
        holdout_manifest=holdout_manifest,
    )
    print(f"✅ Lane manifest saved to: {manifest_paths['lane_manifest_path']}")
    print(f"✅ Holdout manifest saved to: {manifest_paths['holdout_manifest_path']}")
