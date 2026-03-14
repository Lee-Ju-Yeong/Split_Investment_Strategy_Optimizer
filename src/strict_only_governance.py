"""
Helpers for #97 strict-only governance evidence.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import median
import sys
from typing import Iterable, Mapping, Sequence

# BOOTSTRAP: allow direct execution (`python src/strict_only_governance.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"


REQUIRED_FROZEN_MANIFEST_MODES = frozenset({"record_strict", "replay_strict"})
MIN_OBSERVATION_RUNS = 10
MIN_OBSERVATION_WINDOWS = 10
MIN_OBSERVATION_PARITY_SAMPLES = 1
MIN_MATCHED_PARITY_WINDOWS = 1
MAX_P95_EMPTY_ENTRY_DAY_RATE = 0.20
MIN_MEDIAN_TIER1_COVERAGE = 0.55


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nearest_rank_p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(int(len(ordered) * 0.95 + 0.999999) - 1, 0)
    return float(ordered[min(index, len(ordered) - 1)])


def _artifact_rows(artifact: Mapping[str, object]) -> list[dict]:
    decision = artifact.get("decision_evidence") or {}
    rows = decision.get("rows") if isinstance(decision, Mapping) else []
    return [dict(row) for row in (rows or []) if isinstance(row, Mapping)]


def _artifact_mode(rows: Sequence[Mapping[str, object]]) -> str | None:
    for row in rows:
        mode = str(row.get("frozen_candidate_manifest_mode") or "").strip().lower()
        if mode:
            return mode
    return None


def _gate_c_artifact_reasons(index: int, artifact: Mapping[str, object]) -> list[str]:
    summary = artifact.get("summary") or {}
    rows = _artifact_rows(artifact)
    reasons: list[str] = []
    prefix = f"artifact[{index}]"
    if _as_int(summary.get("failed")) > 0:
        reasons.append(f"{prefix}.curve_failures={_as_int(summary.get('failed'))}")
    if not bool(summary.get("policy_ready_for_release_gate", False)):
        reasons.append(f"{prefix}.policy_not_release_ready")
    if not bool(summary.get("curve_level_parity_zero_mismatch", False)):
        reasons.append(f"{prefix}.curve_level_parity_failed")
    if not bool(summary.get("decision_level_parity_zero_mismatch", False)):
        reasons.append(f"{prefix}.decision_level_parity_failed")
    if bool(summary.get("promotion_blocked", True)):
        reasons.append(f"{prefix}.promotion_blocked")
    if not bool(summary.get("decision_evidence_release_fields_complete", False)):
        reasons.append(f"{prefix}.decision_fields_incomplete")
    if _as_int(summary.get("decision_evidence_pit_failure_rows")) > 0:
        reasons.append(f"{prefix}.pit_failure_rows={_as_int(summary.get('decision_evidence_pit_failure_rows'))}")
    if not rows:
        reasons.append(f"{prefix}.decision_rows_missing")
    candidate_failures = sum(
        1
        for row in rows
        if _as_int(row.get("candidate_order_paired_count")) > 0
        and not bool(row.get("candidate_order_zero_mismatch", False))
    )
    if candidate_failures > 0:
        reasons.append(f"{prefix}.candidate_order_mismatch_rows={candidate_failures}")
    row_pit_failures = sum(1 for row in rows if row.get("pit_failure_code"))
    if row_pit_failures > 0:
        reasons.append(f"{prefix}.row_pit_failures={row_pit_failures}")
    return reasons


def evaluate_issue97_gate_c(parity_artifacts: Sequence[Mapping[str, object]]) -> dict:
    seen_modes = set()
    reasons: list[str] = []
    for index, artifact in enumerate(parity_artifacts):
        rows = _artifact_rows(artifact)
        mode = _artifact_mode(rows)
        if mode:
            seen_modes.add(mode)
        reasons.extend(_gate_c_artifact_reasons(index, artifact))
    if len(parity_artifacts) < len(REQUIRED_FROZEN_MANIFEST_MODES):
        reasons.append(
            f"artifact_count={len(parity_artifacts)}<{len(REQUIRED_FROZEN_MANIFEST_MODES)}"
        )
    for mode in sorted(REQUIRED_FROZEN_MANIFEST_MODES - seen_modes):
        reasons.append(f"missing_frozen_manifest_mode={mode}")
    return {
        "approved": len(reasons) == 0,
        "artifacts_checked": int(len(parity_artifacts)),
        "required_frozen_manifest_modes": sorted(REQUIRED_FROZEN_MANIFEST_MODES),
        "seen_frozen_manifest_modes": sorted(seen_modes),
        "reasons": reasons,
    }


def _manifest_is_strict_only(manifest: Mapping[str, object]) -> bool:
    config = manifest.get("config") or {}
    universe = manifest.get("universe_policy") or {}
    return (
        str(config.get("candidate_source_mode") or "").strip().lower() == "tier"
        and str(config.get("tier_hysteresis_mode") or "").strip().lower() == "strict_hysteresis_v1"
        and str(config.get("candidate_lookup_error_policy") or "").strip().lower() == "raise"
        and not bool(config.get("use_weekly_alpha_gate", False))
        and str(universe.get("resolved_mode") or "").strip().lower() == "strict_pit"
    )


def _parity_artifact_clean(artifact: Mapping[str, object]) -> bool:
    summary = artifact.get("summary") or {}
    rows = _artifact_rows(artifact)
    return (
        bool(summary.get("decision_level_parity_zero_mismatch", False))
        and _as_int(summary.get("decision_evidence_pit_failure_rows")) == 0
        and all(not row.get("pit_failure_code") for row in rows)
        and all(bool(row.get("candidate_order_zero_mismatch", False)) for row in rows if _as_int(row.get("candidate_order_paired_count")) > 0)
    )


def _manifest_window_key(manifest: Mapping[str, object]) -> str | None:
    backtest_window = manifest.get("backtest_window") or {}
    start_date = str(backtest_window.get("start_date") or "").strip()
    end_date = str(backtest_window.get("end_date") or "").strip()
    if start_date and end_date:
        return f"{start_date}:{end_date}"
    return None


def _artifact_window_key(artifact: Mapping[str, object]) -> str | None:
    meta = artifact.get("meta") or {}
    start_date = str(meta.get("start_date") or "").strip()
    end_date = str(meta.get("end_date") or "").strip()
    if start_date and end_date:
        return f"{start_date}:{end_date}"
    rows = _artifact_rows(artifact)
    for row in rows:
        detail_path = str(row.get("detail_path") or "").strip()
        if not detail_path:
            continue
        path = Path(detail_path)
        if not path.exists():
            continue
        try:
            detail = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        detail_start = str(detail.get("start_date") or "").strip()
        detail_end = str(detail.get("end_date") or "").strip()
        if detail_start and detail_end:
            return f"{detail_start}:{detail_end}"
    return None


def _lane_manifest_reasons(index: int, manifest: Mapping[str, object]) -> list[str]:
    prefix = f"lane_manifest[{index}]"
    reasons: list[str] = []
    if not bool(manifest.get("approval_eligible", False)):
        reasons.append(f"{prefix}.approval_ineligible")
    evidence_tier = str(manifest.get("evidence_tier") or "").strip().lower()
    if evidence_tier and evidence_tier != "approval_grade":
        reasons.append(f"{prefix}.evidence_tier={evidence_tier}")
    lane_type = str(manifest.get("lane_type") or "").strip().lower()
    cpu_audit_outcome = str(manifest.get("cpu_audit_outcome") or "").strip().lower()
    if lane_type == "promotion_evaluation" and cpu_audit_outcome != "pass":
        reasons.append(f"{prefix}.cpu_audit_required_for_promotion={cpu_audit_outcome or 'missing'}")
    if cpu_audit_outcome and cpu_audit_outcome not in {"disabled", "pass"}:
        reasons.append(f"{prefix}.cpu_audit_outcome={cpu_audit_outcome}")
    for reason in manifest.get("reasons") or []:
        reasons.append(f"{prefix}.reason={reason}")
    return reasons


def _holdout_manifest_reasons(index: int, manifest: Mapping[str, object]) -> list[str]:
    prefix = f"holdout_manifest[{index}]"
    reasons: list[str] = []
    if not bool(manifest.get("approval_eligible", False)):
        reasons.append(f"{prefix}.approval_ineligible")
    holdout_class = str(manifest.get("holdout_class") or "").strip().lower()
    if holdout_class and holdout_class != "approval_grade":
        reasons.append(f"{prefix}.holdout_class={holdout_class}")
    if not bool(manifest.get("holdout_backtest_executed", False)):
        reasons.append(f"{prefix}.holdout_backtest_not_executed")
    before_holdout = manifest.get("promotion_wfo_end_before_holdout")
    if before_holdout is False:
        reasons.append(f"{prefix}.promotion_wfo_end_before_holdout=false")
    for reason in manifest.get("reasons") or []:
        reasons.append(f"{prefix}.reason={reason}")
    return reasons


def summarize_issue97_observation(
    run_manifests: Sequence[Mapping[str, object]],
    *,
    parity_artifacts: Sequence[Mapping[str, object]] | None = None,
    lane_manifests: Sequence[Mapping[str, object]] | None = None,
    holdout_manifests: Sequence[Mapping[str, object]] | None = None,
) -> dict:
    manifests = [dict(doc) for doc in run_manifests]
    parity_docs = list(parity_artifacts or [])
    lane_docs = [dict(doc) for doc in (lane_manifests or [])]
    holdout_docs = [dict(doc) for doc in (holdout_manifests or [])]
    empty_rates = []
    tier1_coverages = []
    non_strict_runs = 0
    failed_runs = 0
    promotion_blocked_runs = 0
    degraded_runs = 0
    fatal_pit_failure_runs = 0
    total_source_lookup_error_days = 0
    total_source_missing_days = 0
    total_source_unknown_days = 0
    total_metrics_cast_error_count = 0
    total_pit_failure_days = 0
    unique_windows = set()
    parity_window_keys = set()
    for manifest in manifests:
        if not _manifest_is_strict_only(manifest):
            non_strict_runs += 1
        window_key = _manifest_window_key(manifest)
        if window_key is not None:
            unique_windows.add(window_key)
        if str(manifest.get("status") or "").strip().lower() != "success":
            failed_runs += 1
        safety_guard = manifest.get("safety_guard") or {}
        run_metrics = manifest.get("run_metrics") or {}
        if bool(safety_guard.get("promotion_blocked", False)):
            promotion_blocked_runs += 1
        if bool(safety_guard.get("degraded_run", False)):
            degraded_runs += 1
        if ((manifest.get("error_info") or {}).get("pit_failure")):
            fatal_pit_failure_runs += 1
        empty_rate = _as_float(run_metrics.get("empty_entry_day_rate"))
        if empty_rate is not None:
            empty_rates.append(empty_rate)
        tier1_coverage = _as_float(run_metrics.get("tier1_coverage"))
        if tier1_coverage is not None:
            tier1_coverages.append(tier1_coverage)
        total_source_lookup_error_days += _as_int(run_metrics.get("source_lookup_error_days"))
        total_source_missing_days += _as_int(run_metrics.get("source_missing_days"))
        total_source_unknown_days += _as_int(run_metrics.get("source_unknown_days"))
        total_metrics_cast_error_count += _as_int(run_metrics.get("metrics_cast_error_count"))
        total_pit_failure_days += sum(
            _as_int(count)
            for count in dict(run_metrics.get("pit_failure_days_by_code") or {}).values()
        )
    for artifact in parity_docs:
        parity_window_key = _artifact_window_key(artifact)
        if parity_window_key is not None:
            parity_window_keys.add(parity_window_key)
    matched_parity_window_keys = parity_window_keys & unique_windows
    parity_mismatch_runs = sum(1 for artifact in parity_docs if not _parity_artifact_clean(artifact))
    p95_empty_entry_day_rate = _nearest_rank_p95(empty_rates)
    median_tier1_coverage = float(median(tier1_coverages)) if tier1_coverages else None
    reasons: list[str] = []
    if len(manifests) < MIN_OBSERVATION_RUNS:
        reasons.append(f"observation_run_count={len(manifests)}<{MIN_OBSERVATION_RUNS}")
    if len(unique_windows) < MIN_OBSERVATION_WINDOWS:
        reasons.append(f"unique_backtest_windows={len(unique_windows)}<{MIN_OBSERVATION_WINDOWS}")
    if len(parity_docs) < MIN_OBSERVATION_PARITY_SAMPLES:
        reasons.append(
            f"parity_sample_count={len(parity_docs)}<{MIN_OBSERVATION_PARITY_SAMPLES}"
        )
    if len(matched_parity_window_keys) < MIN_MATCHED_PARITY_WINDOWS:
        reasons.append(
            "matched_parity_windows="
            f"{len(matched_parity_window_keys)}<{MIN_MATCHED_PARITY_WINDOWS}"
        )
    if len(lane_docs) < 1:
        reasons.append("lane_manifest_count=0<1")
    if len(holdout_docs) < 1:
        reasons.append("holdout_manifest_count=0<1")
    if non_strict_runs > 0:
        reasons.append(f"non_strict_run_count={non_strict_runs}")
    if failed_runs > 0:
        reasons.append(f"failed_run_count={failed_runs}")
    if promotion_blocked_runs > 0:
        reasons.append(f"promotion_blocked_runs={promotion_blocked_runs}")
    if degraded_runs > 0:
        reasons.append(f"degraded_runs={degraded_runs}")
    if total_source_lookup_error_days > 0:
        reasons.append(f"source_lookup_error_days={total_source_lookup_error_days}")
    if total_source_missing_days > 0:
        reasons.append(f"source_missing_days={total_source_missing_days}")
    if total_source_unknown_days > 0:
        reasons.append(f"source_unknown_days={total_source_unknown_days}")
    if total_metrics_cast_error_count > 0:
        reasons.append(f"metrics_cast_error_count={total_metrics_cast_error_count}")
    if total_pit_failure_days > 0:
        reasons.append(f"pit_failure_days={total_pit_failure_days}")
    if fatal_pit_failure_runs > 0:
        reasons.append(f"fatal_pit_failure_runs={fatal_pit_failure_runs}")
    if p95_empty_entry_day_rate is None:
        reasons.append("empty_entry_day_rate_missing")
    elif p95_empty_entry_day_rate > MAX_P95_EMPTY_ENTRY_DAY_RATE:
        reasons.append(
            f"p95_empty_entry_day_rate={p95_empty_entry_day_rate:.4f}>{MAX_P95_EMPTY_ENTRY_DAY_RATE:.2f}"
        )
    if median_tier1_coverage is None:
        reasons.append("tier1_coverage_missing")
    elif median_tier1_coverage < MIN_MEDIAN_TIER1_COVERAGE:
        reasons.append(
            f"median_tier1_coverage={median_tier1_coverage:.4f}<{MIN_MEDIAN_TIER1_COVERAGE:.2f}"
        )
    if parity_mismatch_runs > 0:
        reasons.append(f"parity_mismatch_runs={parity_mismatch_runs}")
    for index, lane_manifest in enumerate(lane_docs):
        reasons.extend(_lane_manifest_reasons(index, lane_manifest))
    for index, holdout_manifest in enumerate(holdout_docs):
        reasons.extend(_holdout_manifest_reasons(index, holdout_manifest))
    return {
        "approved": len(reasons) == 0,
        "observation_mode": "synthetic_window_pack",
        "observation_run_count": int(len(manifests)),
        "unique_backtest_windows": int(len(unique_windows)),
        "strict_only_run_count": int(len(manifests) - non_strict_runs),
        "parity_sample_count": int(len(parity_docs)),
        "lane_manifest_count": int(len(lane_docs)),
        "holdout_manifest_count": int(len(holdout_docs)),
        "parity_window_count": int(len(parity_window_keys)),
        "matched_parity_windows": int(len(matched_parity_window_keys)),
        "promotion_blocked_runs": int(promotion_blocked_runs),
        "degraded_runs": int(degraded_runs),
        "failed_runs": int(failed_runs),
        "fatal_pit_failure_runs": int(fatal_pit_failure_runs),
        "source_lookup_error_days": int(total_source_lookup_error_days),
        "source_missing_days": int(total_source_missing_days),
        "source_unknown_days": int(total_source_unknown_days),
        "metrics_cast_error_count": int(total_metrics_cast_error_count),
        "pit_failure_days": int(total_pit_failure_days),
        "parity_mismatch_runs": int(parity_mismatch_runs),
        "p95_empty_entry_day_rate": None if p95_empty_entry_day_rate is None else round(p95_empty_entry_day_rate, 4),
        "median_tier1_coverage": None if median_tier1_coverage is None else round(median_tier1_coverage, 4),
        "thresholds": {
            "min_observation_runs": MIN_OBSERVATION_RUNS,
            "min_unique_backtest_windows": MIN_OBSERVATION_WINDOWS,
            "min_observation_parity_samples": MIN_OBSERVATION_PARITY_SAMPLES,
            "min_matched_parity_windows": MIN_MATCHED_PARITY_WINDOWS,
            "max_p95_empty_entry_day_rate": MAX_P95_EMPTY_ENTRY_DAY_RATE,
            "min_median_tier1_coverage": MIN_MEDIAN_TIER1_COVERAGE,
        },
        "reasons": reasons,
    }


def _load_json_documents(paths: Iterable[str]) -> list[dict]:
    documents = []
    for raw_path in paths:
        path = Path(raw_path)
        documents.append(json.loads(path.read_text(encoding="utf-8")))
    return documents


def _expand_paths(paths: Sequence[str], globs: Sequence[str]) -> list[str]:
    expanded = list(paths)
    for pattern in globs:
        expanded.extend(sorted(glob.glob(pattern)))
    return expanded


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate #97 Gate C or historical strict-only synthetic sample pack."
    )
    parser.add_argument("--mode", choices=("gate-c", "observation"), required=True)
    parser.add_argument("--parity-json", action="append", default=[], help="Path to parity JSON artifact.")
    parser.add_argument("--run-manifest-json", action="append", default=[], help="Path to run_manifest.json.")
    parser.add_argument("--run-manifest-glob", action="append", default=[], help="Glob for run_manifest.json files.")
    parser.add_argument("--lane-manifest-json", action="append", default=[], help="Path to lane_manifest.json.")
    parser.add_argument("--holdout-manifest-json", action="append", default=[], help="Path to holdout_manifest.json.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    parity_docs = _load_json_documents(args.parity_json)
    if args.mode == "gate-c":
        payload = evaluate_issue97_gate_c(parity_docs)
    else:
        manifest_paths = _expand_paths(args.run_manifest_json, args.run_manifest_glob)
        payload = summarize_issue97_observation(
            _load_json_documents(manifest_paths),
            parity_artifacts=parity_docs,
            lane_manifests=_load_json_documents(args.lane_manifest_json),
            holdout_manifests=_load_json_documents(args.holdout_manifest_json),
        )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if bool(payload.get("approved", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
