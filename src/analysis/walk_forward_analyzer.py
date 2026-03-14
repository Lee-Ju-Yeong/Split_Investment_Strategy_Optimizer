from __future__ import annotations

import json
import os
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
    contaminated_ranges=None,
    adequacy_metrics=None,
    min_length_days: int = _APPROVAL_GRADE_HOLDOUT_MIN_DAYS,
) -> dict:
    start = _coerce_date(holdout_start)
    end = _coerce_date(holdout_end)
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
    if overlap:
        reasons.append("holdout_range_contaminated")
    if missing_fields:
        reasons.append("missing_adequacy_fields=" + ",".join(missing_fields))
    approval_eligible = not reasons
    return {
        "holdout_start": start.isoformat(),
        "holdout_end": end.isoformat(),
        "holdout_length_days": int(length_days),
        "holdout_class": "approval_grade" if approval_eligible else "internal_provisional",
        "approval_eligible": bool(approval_eligible),
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
    min_length_days: int = _APPROVAL_GRADE_HOLDOUT_MIN_DAYS,
) -> dict:
    policy = evaluate_holdout_policy(
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        contaminated_ranges=contaminated_ranges,
        adequacy_metrics=adequacy_metrics,
        min_length_days=min_length_days,
    )
    adequacy = dict(adequacy_metrics or {})
    return {
        "holdout_start": policy["holdout_start"],
        "holdout_end": policy["holdout_end"],
        "wfo_end": _coerce_date(wfo_end).isoformat(),
        "holdout_date_reuse_forbidden": True,
        "parity_canary_excluded_ranges": policy["contaminated_ranges"],
        "holdout_class": policy["holdout_class"],
        "holdout_length_days": policy["holdout_length_days"],
        "approval_eligible": policy["approval_eligible"],
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
        "reasons": policy["reasons"],
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
        "parity_canary_excluded_ranges": _normalize_contaminated_ranges(contaminated_ranges),
        "holdout_class": "unconfigured",
        "holdout_length_days": None,
        "approval_eligible": False,
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
) -> dict:
    return {
        "lane_manifest_path": _write_json_artifact(
            os.path.join(results_dir, "lane_manifest.json"),
            lane_manifest,
        ),
        "holdout_manifest_path": _write_json_artifact(
            os.path.join(results_dir, "holdout_manifest.json"),
            holdout_manifest,
        ),
    }


def _resolve_cpu_audit_outcome(cpu_cert_settings: dict, selection_audits: list[dict]) -> str:
    if not cpu_cert_settings.get("enabled"):
        return "disabled"
    return "pass" if selection_audits else "enabled_but_no_selection_audit"


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


def _build_current_lane_reasons(*, total_folds: int, overlap_days: int) -> list[str]:
    reasons = ["lane_mode_not_separated"]
    if int(total_folds) > 1:
        reasons.append("oos_initial_cash_carry_over_enabled")
        reasons.append("composite_curve_mean_aggregation_enabled")
    if int(overlap_days) > 0:
        reasons.append(f"oos_fold_overlap_days={int(overlap_days)}")
    return reasons


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
    start_date: str,
    end_date: str,
    initial_cash: float,
    metric: str,
    top_n_requested: int | None = None,
) -> tuple[dict, "pd.DataFrame"]:
    import pandas as pd

    if finalists_df.empty:
        raise ValueError("CPU certification finalists are empty.")
    cpu_metric_col = f"cpu_{metric}"
    records = []

    for _, finalist in finalists_df.iterrows():
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
    certified_df = certification_df[certification_df["cpu_certified"]].copy()
    if certified_df.empty:
        raise RuntimeError("CPU certification rejected every GPU finalist.")
    if cpu_metric_col not in certified_df.columns:
        raise ValueError(f"CPU certification metric '{metric}' was not produced by CPU backtest.")

    certified_df = certified_df.sort_values(
        [cpu_metric_col, "cpu_cagr", "cpu_mdd", "gpu_rank"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    winner = certified_df.iloc[0].to_dict()
    winner_params = _extract_strategy_params(winner, base_config["strategy_params"])
    winner_params.update(
        {
            "selection_source": "cpu_certified_finalist",
            "cpu_certification_metric": metric,
            "cpu_certification_top_n": int(top_n_requested or len(finalists_df)),
            "cpu_certification_shortlist_size": int(len(finalists_df)),
            "cpu_certification_gpu_rank": winner.get("gpu_rank"),
            cpu_metric_col: winner.get(cpu_metric_col),
            "cpu_cagr": winner.get("cpu_cagr"),
            "cpu_mdd": winner.get("cpu_mdd"),
        }
    )
    return winner_params, certification_df

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
    total_folds = wfo_settings['total_folds']
    period_length_days = wfo_settings['period_length_days']
    holdout_settings = _resolve_holdout_runtime_settings(wfo_settings)
        
    # --- 확정 WFO 기간 생성 (no-overlap 우선, 불가 시 최소 겹침 + 균등분포) ---
    S = pd.to_datetime(backtest_settings['start_date']).normalize()
    E = pd.to_datetime(backtest_settings['end_date']).normalize()
    N = int(wfo_settings['total_folds'])
    L_days = int(wfo_settings['period_length_days'])
    L = pd.Timedelta(days=L_days)

    if N <= 0 or L_days <= 0:
        raise ValueError("total_folds and period_length_days must be positive.")

    # 1) 무겹침 가능성 평가
    #   d = OOS_Start - IS_Start, 겹침 = L - d
    #   무겹침 필요조건: d >= L
    #   경계조건: last_IS_start = E - d - (L-1) >= S  ->  d <= (E - S).days - (L-1)
    Dmax_days = (E - S).days - (L_days - 1)   # d가 가질 수 있는 최대값(경계 위배 없이)
    d_days = None

    if Dmax_days >= L_days:
        # 여유로움 → 무겹침 채택
        d_days = L_days
    else:
        # 여유 부족 → 겹침 최소(= d 최대)와 균등성의 균형
        # 기본값: 절반쯤 이동(균형) -> 이전에 합의한 d≈L/2 (+1 보정)
        d_days = min(Dmax_days, (L_days + 1) // 2 + 1)
        if d_days < 1:
            d_days = 1  # 인과성 보장

    # 2) 마지막 폴드가 E에 맞도록 IS 최종 시작점 역산
    d = pd.Timedelta(days=d_days)
    last_is_start = E - d - (L - pd.Timedelta(days=1))

    # 3) IS 시작들의 균등 분포
    #    span_days가 작아도 N개 균등 배치(정수 보정: 몫/나머지 방식)
    span_days = (last_is_start - S).days
    print("span_days:",span_days)
    if span_days <= 0:
        raise ValueError(f"Configuration Error: Cannot fit {N} folds. The total period is too short for the given period length ({L_days} days). Please reduce 'total_folds' or 'period_length_days'.")
    if N == 1:
        is_starts = [S]
    else:
        base_step = span_days // (N - 1)
        remainder = span_days % (N - 1)
        is_starts = [S]
        for i in range(1, N):
            add = base_step + (1 if i <= remainder else 0)
            is_starts.append(is_starts[-1] + pd.Timedelta(days=add))

    # 4) 폴드 구간 구성
    fold_periods = []
    for i, is_start in enumerate(is_starts):
        is_end   = is_start + L - pd.Timedelta(days=1)
        oos_start = is_start + d
        oos_end   = oos_start + L - pd.Timedelta(days=1)

        # 안전 체크
        if oos_start < is_start + pd.Timedelta(days=1):
            raise ValueError("Causality violated: OOS must start at least 1 day after IS start.")
        if oos_end > E:
            raise ValueError("Boundary violated: OOS end beyond end_date.")
        if is_start < S or is_end > E:
            raise ValueError("IS period out of bounds.")

        fold_periods.append({
            'Fold': i + 1,
            'IS_Start': is_start.date(), 'IS_End': is_end.date(),
            'OOS_Start': oos_start.date(), 'OOS_End': oos_end.date()
        })

    print("\n--- Calculated Walk-Forward Folds ---")
    print(pd.DataFrame(fold_periods).to_string(index=False))

    # 참고 출력(선택): 실제 겹침일
    overlap_days = L_days - d_days  # (0이면 무겹침)
    print(f"\n[WFO] d = {d_days} days → overlap = {overlap_days} days (per fold)")


    
    #  새로운 롤링 윈도우 루프
    all_oos_curves, all_optimal_params, all_selection_audits = [], [], []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("results", f"wfo_run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    pbar = tqdm(fold_periods, desc="WFO Progress")
    for period in pbar:
        fold_num, is_start, is_end, oos_start, oos_end = period.values()
        pbar.set_description(f"WFO Fold {fold_num}/{total_folds}")

        print(f"\n--- Fold {fold_num} IS Period: {is_start} ~ {is_end} ---")
        
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

        if cpu_cert_settings["enabled"]:
            gpu_shortlist_df = build_gpu_finalist_shortlist(
                is_simulation_results_df,
                robust_params_dict,
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
            selection_audit = {
                "fold": fold_num,
                "selection_source": selected_params_dict.get("selection_source", "cpu_certified_finalist"),
                "cpu_certification_metric": selected_params_dict.get("cpu_certification_metric"),
                "cpu_certification_top_n": selected_params_dict.get("cpu_certification_top_n"),
                "cpu_certification_shortlist_size": selected_params_dict.get("cpu_certification_shortlist_size"),
                "cpu_certification_gpu_rank": selected_params_dict.get("cpu_certification_gpu_rank"),
                "cpu_calmar_ratio": selected_params_dict.get("cpu_calmar_ratio"),
                "cpu_cagr": selected_params_dict.get("cpu_cagr"),
                "cpu_mdd": selected_params_dict.get("cpu_mdd"),
            }
            all_selection_audits.append(selection_audit)

        reported_params_dict = _extract_strategy_params(selected_params_dict, config["strategy_params"])
        reported_params_dict['fold'] = fold_num
        all_optimal_params.append(reported_params_dict)
        print(f"  - Final params for Fold {fold_num} selected.")
        
        if total_folds == 1:
            print("\n[INFO] Single fold run. OOS performance is same as IS robust parameter performance.")
            # 단일 폴드에서는 OOS 커브가 의미 없으므로 IS 결과를 사용 (혹은 생략)
            break

        print(f"--- Fold {fold_num} OOS Period: {oos_start} ~ {oos_end} ---")
        
        # 3. 찾은 파라미터로 OOS 기간 백테스트
        # OOS 기간의 초기 자금은 이전 OOS 기간의 최종 자금으로 연결
        oos_initial_cash = initial_cash if not all_oos_curves else all_oos_curves[-1].iloc[-1]
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
        final_wfo_curve = pd.concat(all_oos_curves).sort_index().groupby(level=0).mean()
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

    holdout_start = holdout_settings["holdout_start"]
    holdout_end = holdout_settings["holdout_end"]
    if holdout_start and holdout_end:
        holdout_manifest = build_holdout_manifest(
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            wfo_end=total_end_date.date().isoformat(),
            contaminated_ranges=holdout_settings["contaminated_ranges"],
            adequacy_metrics={},
        )
        holdout_manifest["reasons"] = list(holdout_manifest.get("reasons", [])) + [
            "holdout_backtest_not_executed",
        ]
    else:
        holdout_manifest = _build_unconfigured_holdout_manifest(
            wfo_end=total_end_date.date().isoformat(),
            contaminated_ranges=holdout_settings["contaminated_ranges"],
        )

    lane_reasons = _build_current_lane_reasons(
        total_folds=total_folds,
        overlap_days=overlap_days,
    )
    lane_approval_eligible = bool(holdout_manifest.get("approval_eligible")) and not lane_reasons
    lane_manifest = build_lane_manifest(
        lane_type=str(wfo_settings.get("lane_type") or "legacy_wfo"),
        approval_eligible=lane_approval_eligible,
        decision_date=wfo_settings.get("decision_date"),
        research_data_cutoff=wfo_settings.get("research_data_cutoff") or total_end_date.date().isoformat(),
        promotion_data_cutoff=wfo_settings.get("promotion_data_cutoff") or total_end_date.date().isoformat(),
        shortlist_hash=wfo_settings.get("shortlist_hash"),
        publication_lag_policy=wfo_settings.get("publication_lag_policy"),
        ticker_universe_snapshot_id=wfo_settings.get("ticker_universe_snapshot_id"),
        engine_version_hash=wfo_settings.get("engine_version_hash") or os.environ.get("MAGICSPLIT_ENGINE_VERSION_HASH"),
        composite_curve_allowed=bool(total_folds > 1),
        cpu_audit_outcome=_resolve_cpu_audit_outcome(cpu_cert_settings, all_selection_audits),
        reasons=lane_reasons,
    )
    manifest_paths = write_wfo_manifests(
        results_dir=results_dir,
        lane_manifest=lane_manifest,
        holdout_manifest=holdout_manifest,
    )
    print(f"✅ Lane manifest saved to: {manifest_paths['lane_manifest_path']}")
    print(f"✅ Holdout manifest saved to: {manifest_paths['holdout_manifest_path']}")
