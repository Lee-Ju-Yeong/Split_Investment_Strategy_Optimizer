"""
N-window shortlist derivation for standalone simulation bundles.

This module turns one explicit window bundle manifest into a frozen shortlist
that can be handed off to research/promotion WFO without redoing ad-hoc
spreadsheet work. The scope is intentionally narrow:

- No automatic window search
- No automatic weight search
- No synthetic centroid candidates
- Discovery-only provenance artifacts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PARAMETER_KEYS = (
    "max_stocks",
    "order_investment_ratio",
    "additional_buy_drop_rate",
    "sell_profit_rate",
    "additional_buy_priority",
    "stop_loss_rate",
    "max_splits_limit",
    "max_inactivity_period",
)
FLOAT_PARAM_KEYS = {
    "order_investment_ratio",
    "additional_buy_drop_rate",
    "sell_profit_rate",
    "stop_loss_rate",
}
INT_PARAM_KEYS = {
    "max_stocks",
    "max_splits_limit",
    "max_inactivity_period",
}
SUPPORTED_SELECTION_METRICS = (
    "calmar_ratio",
    "cagr",
    "mdd",
    "sharpe_ratio",
    "sortino_ratio",
)
DEFAULT_FAMILY_EXCLUDED_PARAMETERS = ("stop_loss_rate",)
DEFAULT_AGGREGATION_RULE_VERSION = "n_window_rank_robust_v1"
DEFAULT_TIE_BREAK_RULE_VERSION = "n_window_shortlist_tiebreak_v1"
DEFAULT_MANIFEST_VERSION = "n_window_shortlist_source_v1"
DEFAULT_MAX_RANK_PERCENTILE = 15.0
DEFAULT_OPTIONAL_FAIL_BUDGET = 0
DEFAULT_SHORTLIST_SIZE = 10
APPROVAL_COMPATIBLE_MAX_WINDOWS = 3


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    csv_path: Path
    expected_hash: str
    actual_hash: str
    config_path: str | None
    window_role: str
    weight: float
    row_count: int


@dataclass(frozen=True)
class WindowObservation:
    window_id: str
    row: dict[str, str] | None
    selection_score: float
    metric_scores: dict[str, float]
    passed_gate: bool
    fail_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PreparedWindow:
    spec: WindowSpec
    rows: list[dict[str, str]]
    row_map: dict[str, dict[str, str]]
    metric_score_maps: dict[str, dict[str, float]]


def _parse_float(value: Any) -> float:
    return float(str(value).strip())


def _hash_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_priority(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in ("lowest_order", "highest_drop"):
            return stripped
    try:
        return "highest_drop" if int(float(value)) == 1 else "lowest_order"
    except (TypeError, ValueError):
        return "lowest_order"


def _normalize_param_value(key: str, value: Any) -> Any:
    if key == "additional_buy_priority":
        return _normalize_priority(value)
    if key in INT_PARAM_KEYS:
        return int(float(value))
    if key in FLOAT_PARAM_KEYS:
        return round(float(value), 10)
    return value


def _candidate_signature(row: dict[str, Any]) -> str:
    parts = []
    for key in PARAMETER_KEYS:
        parts.append(f"{key}={_normalize_param_value(key, row.get(key))}")
    return "|".join(parts)


def _family_signature(
    row: dict[str, Any],
    *,
    excluded_parameters: tuple[str, ...],
) -> str:
    parts = []
    for key in PARAMETER_KEYS:
        if key in excluded_parameters:
            continue
        parts.append(f"{key}={_normalize_param_value(key, row.get(key))}")
    return "|".join(parts)


def _metric_value(row: dict[str, str], metric: str) -> float:
    if metric not in row:
        raise ValueError(f"row is missing metric column: {metric}")
    return _parse_float(row[metric])


def _parameter_as_numeric(key: str, value: Any) -> float:
    normalized = _normalize_param_value(key, value)
    if key == "additional_buy_priority":
        return 1.0 if normalized == "highest_drop" else 0.0
    return float(normalized)


def _resolve_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if raw_path is None or str(raw_path).strip() == "":
        return None
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _require_window_role(raw_value: Any) -> str:
    role = str(raw_value or "").strip().lower()
    if role not in {"mandatory", "optional"}:
        raise ValueError("window_role must be 'mandatory' or 'optional'.")
    return role


def _require_selection_metric(raw_value: Any) -> str:
    metric = str(raw_value or "calmar_ratio").strip()
    if metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            "selection_metric must be one of: "
            + ", ".join(SUPPORTED_SELECTION_METRICS)
        )
    return metric


def _load_window_specs(bundle_manifest: dict[str, Any], *, manifest_path: Path) -> list[WindowSpec]:
    windows = list(bundle_manifest.get("windows") or [])
    if not windows:
        raise ValueError("window bundle manifest must include at least one window.")

    base_dir = manifest_path.parent
    seen_ids: set[str] = set()
    specs: list[WindowSpec] = []
    mandatory_count = 0

    for item in windows:
        window_id = str(item.get("window_id") or "").strip()
        if not window_id:
            raise ValueError("every window must define window_id.")
        if window_id in seen_ids:
            raise ValueError(f"duplicate window_id: {window_id}")
        seen_ids.add(window_id)

        csv_path = _resolve_path(base_dir, item.get("csv_path"))
        if csv_path is None or not csv_path.is_file():
            raise ValueError(f"window {window_id} csv_path does not exist.")

        expected_hash = str(item.get("expected_hash") or "").strip()
        if not expected_hash:
            raise ValueError(f"window {window_id} is missing expected_hash.")
        actual_hash = _hash_file_sha256(csv_path)
        if actual_hash != expected_hash:
            raise ValueError(f"window {window_id} expected_hash mismatch.")

        role = _require_window_role(item.get("window_role"))
        if role == "mandatory":
            mandatory_count += 1

        rows = _load_rows(csv_path)
        if not rows:
            raise ValueError(f"window {window_id} csv is empty: {csv_path}")

        specs.append(
            WindowSpec(
                window_id=window_id,
                csv_path=csv_path,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
                config_path=str(_resolve_path(base_dir, item.get("config_path"))) if item.get("config_path") else None,
                window_role=role,
                weight=float(item.get("weight", 1.0) or 1.0),
                row_count=len(rows),
            )
        )
    if mandatory_count == 0:
        raise ValueError("window bundle manifest must include at least one mandatory window.")
    return specs


def _rank_scores(rows: list[dict[str, str]], metric: str) -> dict[str, float]:
    ranked = sorted(rows, key=lambda row: _metric_value(row, metric), reverse=True)
    if not ranked:
        return {}
    if len(ranked) == 1:
        return {_candidate_signature(ranked[0]): 1.0}

    scores: dict[str, float] = {}
    denominator = max(len(ranked) - 1, 1)
    for index, row in enumerate(ranked):
        scores[_candidate_signature(row)] = 1.0 - (index / denominator)
    return scores


def _row_passes_thresholds(row: dict[str, str], thresholds: dict[str, float]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for key, raw_threshold in thresholds.items():
        threshold = float(raw_threshold)
        if key.endswith("_min"):
            metric = key[:-4]
            if _metric_value(row, metric) < threshold:
                reasons.append(f"{metric}_min")
        elif key.endswith("_max"):
            metric = key[:-4]
            if _metric_value(row, metric) > threshold:
                reasons.append(f"{metric}_max")
        else:
            raise ValueError(
                "minimum_criteria keys must end with _min or _max. "
                f"Received: {key}"
            )
    return (not reasons), reasons


def _window_observation_for_candidate(
    *,
    prepared_window: PreparedWindow,
    thresholds: dict[str, float],
    max_rank_percentile: float,
    signature: str,
) -> WindowObservation:
    selection_scores = prepared_window.metric_score_maps["selection_metric"]
    selection_score = selection_scores.get(signature, 0.0)
    row = prepared_window.row_map.get(signature)
    metric_scores = {
        metric: score_map.get(signature, 0.0)
        for metric, score_map in prepared_window.metric_score_maps.items()
        if metric != "selection_metric"
    }
    metric_scores["selection_metric"] = selection_score

    if row is None:
        return WindowObservation(
            window_id=prepared_window.spec.window_id,
            row=None,
            selection_score=0.0,
            metric_scores=metric_scores,
            passed_gate=False,
            fail_reasons=("missing_row",),
        )

    fail_reasons: list[str] = []
    rank_percentile = (1.0 - selection_score) * 100.0
    if rank_percentile > max_rank_percentile:
        fail_reasons.append("rank_percentile")
    threshold_pass, threshold_reasons = _row_passes_thresholds(row, thresholds)
    if not threshold_pass:
        fail_reasons.extend(threshold_reasons)

    return WindowObservation(
        window_id=prepared_window.spec.window_id,
        row=row,
        selection_score=selection_score,
        metric_scores=metric_scores,
        passed_gate=not fail_reasons,
        fail_reasons=tuple(sorted(set(fail_reasons))),
    )


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in pairs) / total_weight


def _robust_score(observations: list[tuple[WindowSpec, WindowObservation]]) -> tuple[float, dict[str, float]]:
    selection_pairs = [(obs.selection_score, spec.weight) for spec, obs in observations]
    cagr_pairs = [(obs.metric_scores.get("cagr", 0.0), spec.weight) for spec, obs in observations]
    mdd_pairs = [(obs.metric_scores.get("mdd", 0.0), spec.weight) for spec, obs in observations]
    selection_values = [obs.selection_score for _, obs in observations]

    mean_selection = _weighted_mean(selection_pairs)
    min_selection = min(selection_values) if selection_values else 0.0
    std_selection = statistics.pstdev(selection_values) if len(selection_values) > 1 else 0.0
    mean_cagr = _weighted_mean(cagr_pairs)
    mean_mdd = _weighted_mean(mdd_pairs)

    score = (
        (0.55 * mean_selection)
        + (0.20 * min_selection)
        + (0.15 * mean_mdd)
        + (0.10 * mean_cagr)
        - (0.10 * std_selection)
    )
    return score, {
        "mean_selection_score": mean_selection,
        "min_selection_score": min_selection,
        "selection_score_std": std_selection,
        "mean_cagr_score": mean_cagr,
        "mean_mdd_score": mean_mdd,
    }


def _param_center_value(values: list[Any], key: str) -> float:
    numeric_values = [_parameter_as_numeric(key, value) for value in values]
    return float(statistics.median(numeric_values))


def _param_distance(row: dict[str, Any], center: dict[str, float]) -> float:
    distance = 0.0
    for key in PARAMETER_KEYS:
        normalized = _parameter_as_numeric(key, row.get(key))
        distance += abs(normalized - center[key])
    return distance


def _nearest_family_center_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    center = {
        key: _param_center_value([candidate["params"][key] for candidate in candidates], key)
        for key in PARAMETER_KEYS
    }
    return min(
        candidates,
        key=lambda item: (
            _param_distance(item["params"], center),
            -float(item["robust_score"]),
            -float(item["score_components"]["min_selection_score"]),
            item["candidate_signature"],
        ),
    )


def _representative_window_row(candidate: dict[str, Any]) -> tuple[str, dict[str, str]]:
    observations = [obs for obs in candidate["window_observations"] if obs.row is not None]
    if not observations:
        raise ValueError("candidate is missing all representative rows.")
    mean_score = statistics.fmean(obs.selection_score for obs in observations)
    selected = min(
        observations,
        key=lambda obs: (
            abs(obs.selection_score - mean_score),
            -obs.selection_score,
            obs.window_id,
        ),
    )
    return selected.window_id, dict(selected.row)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _load_bundle_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bundle manifest must be a JSON object.")
    return payload


def _build_shortlist_rows(
    *,
    ranked_candidates: list[dict[str, Any]],
    shortlist_size: int,
) -> list[dict[str, Any]]:
    selected = ranked_candidates[:shortlist_size]
    output: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, start=1):
        window_id, base_row = _representative_window_row(candidate)
        row = dict(base_row)
        row.update(
            {
                "shortlist_rank": index,
                "candidate_signature": candidate["candidate_signature"],
                "family_signature": candidate["family_signature"],
                "robust_score": round(float(candidate["robust_score"]), 10),
                "source_window_count": candidate["source_window_count"],
                "passed_window_count": candidate["passed_window_count"],
                "failed_window_count": candidate["failed_window_count"],
                "representative_window_id": window_id,
                "aggregation_rule_version": DEFAULT_AGGREGATION_RULE_VERSION,
            }
        )
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("shortlist rows must not be empty")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _render_summary(
    *,
    bundle_manifest: dict[str, Any],
    windows: list[WindowSpec],
    shortlist_rows: list[dict[str, Any]],
    optional_fail_budget: int,
    selection_metric: str,
    max_rank_percentile: float,
) -> str:
    lines = [
        "# N-Window Shortlist Derivation Summary",
        "",
        f"- Bundle ID: `{bundle_manifest.get('bundle_id')}`",
        f"- Source mode: `{bundle_manifest.get('source_mode', 'n_window_consensus_mining')}`",
        f"- Window count: `{len(windows)}`",
        f"- Selection metric: `{selection_metric}`",
        f"- Hard gate max rank percentile: `{max_rank_percentile}`",
        f"- Optional fail budget: `{optional_fail_budget}`",
        "",
        "## Window Roles",
        "",
    ]
    for window in windows:
        lines.append(
            "- "
            f"`{window.window_id}` role=`{window.window_role}` "
            f"rows=`{window.row_count}` csv=`{window.csv_path}`"
        )
    lines.extend(
        [
            "",
            "## Shortlist",
            "",
        ]
    )
    for row in shortlist_rows:
        lines.append(
            "- "
            f"rank={row['shortlist_rank']}, score={row['robust_score']:.4f}, "
            f"rep_window=`{row['representative_window_id']}`, "
            f"signature=`{row['candidate_signature']}`"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This artifact is research-only upstream discovery input. It is not approval evidence.",
            "- Window composition comes only from the explicit bundle manifest. No automatic window search is allowed.",
            "- Final shortlist rows are actual rows from source CSVs. Synthetic centroid parameters are forbidden.",
            "",
        ]
    )
    return "\n".join(lines)


def derive_shortlist(
    *,
    bundle_manifest_path: Path,
    out_dir: Path,
    approval_compatible: bool = False,
) -> dict[str, Path]:
    bundle_manifest_path = bundle_manifest_path.resolve()
    out_dir = out_dir.resolve()
    bundle_manifest = _load_bundle_manifest(bundle_manifest_path)
    window_specs = _load_window_specs(bundle_manifest, manifest_path=bundle_manifest_path)
    if approval_compatible and len(window_specs) > APPROVAL_COMPATIBLE_MAX_WINDOWS:
        raise ValueError(
            "approval-compatible mode only supports "
            f"{APPROVAL_COMPATIBLE_MAX_WINDOWS} windows or fewer."
        )

    governance = dict(bundle_manifest.get("governance_gates") or {})
    thresholds = dict(governance.get("minimum_criteria") or {})
    max_rank_percentile = float(governance.get("max_rank_percentile", DEFAULT_MAX_RANK_PERCENTILE))
    optional_fail_budget = int(governance.get("optional_fail_budget", DEFAULT_OPTIONAL_FAIL_BUDGET))

    selection_contract = dict(bundle_manifest.get("selection_contract") or {})
    selection_metric = _require_selection_metric(selection_contract.get("selection_metric"))
    shortlist_size = int(selection_contract.get("shortlist_size", DEFAULT_SHORTLIST_SIZE))
    excluded_parameters = tuple(
        selection_contract.get("family_excluded_parameters") or DEFAULT_FAMILY_EXCLUDED_PARAMETERS
    )

    prepared_windows: list[PreparedWindow] = []
    for spec in window_specs:
        rows = _load_rows(spec.csv_path)
        prepared_windows.append(
            PreparedWindow(
                spec=spec,
                rows=rows,
                row_map={_candidate_signature(row): row for row in rows},
                metric_score_maps={
                    "selection_metric": _rank_scores(rows, selection_metric),
                    "mdd": _rank_scores(rows, "mdd"),
                    "cagr": _rank_scores(rows, "cagr"),
                    "sharpe_ratio": _rank_scores(rows, "sharpe_ratio"),
                },
            )
        )

    all_signatures: set[str] = set()
    for prepared_window in prepared_windows:
        all_signatures.update(prepared_window.row_map.keys())

    candidate_records: list[dict[str, Any]] = []
    for signature in sorted(all_signatures):
        observations = [
            _window_observation_for_candidate(
                prepared_window=prepared_window,
                thresholds=thresholds,
                max_rank_percentile=max_rank_percentile,
                signature=signature,
            )
            for prepared_window in prepared_windows
        ]
        mandatory_failed = [
            obs.window_id
            for prepared_window, obs in zip(prepared_windows, observations)
            if prepared_window.spec.window_role == "mandatory" and not obs.passed_gate
        ]
        optional_failed = [
            obs.window_id
            for prepared_window, obs in zip(prepared_windows, observations)
            if prepared_window.spec.window_role == "optional" and not obs.passed_gate
        ]
        if mandatory_failed or len(optional_failed) > optional_fail_budget:
            continue

        first_row = next((obs.row for obs in observations if obs.row is not None), None)
        if first_row is None:
            continue
        robust_score, score_components = _robust_score(
            [(prepared_window.spec, observation) for prepared_window, observation in zip(prepared_windows, observations)]
        )
        params = {key: first_row[key] for key in PARAMETER_KEYS}
        candidate_records.append(
            {
                "candidate_signature": signature,
                "family_signature": _family_signature(
                    first_row,
                    excluded_parameters=excluded_parameters,
                ),
                "params": params,
                "robust_score": robust_score,
                "score_components": score_components,
                "window_observations": observations,
                "source_window_count": sum(1 for obs in observations if obs.row is not None),
                "passed_window_count": sum(1 for obs in observations if obs.passed_gate),
                "failed_window_count": sum(1 for obs in observations if not obs.passed_gate),
            }
        )

    family_groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidate_records:
        family_groups.setdefault(candidate["family_signature"], []).append(candidate)

    family_representatives = [
        _nearest_family_center_candidate(candidates)
        for candidates in family_groups.values()
    ]
    ranked_candidates = sorted(
        family_representatives,
        key=lambda item: (
            -float(item["robust_score"]),
            -float(item["score_components"]["min_selection_score"]),
            -int(item["passed_window_count"]),
            item["candidate_signature"],
        ),
    )
    shortlist_rows = _build_shortlist_rows(
        ranked_candidates=ranked_candidates,
        shortlist_size=shortlist_size,
    )
    if not shortlist_rows:
        raise ValueError("no shortlist candidates survived the bundle hard gate.")

    candidate_ranking = [
        {
            "candidate_signature": candidate["candidate_signature"],
            "family_signature": candidate["family_signature"],
            "robust_score": round(float(candidate["robust_score"]), 10),
            "passed_window_count": candidate["passed_window_count"],
            "failed_window_count": candidate["failed_window_count"],
            "score_components": candidate["score_components"],
        }
        for candidate in ranked_candidates
    ]

    shortlist_hash = _hash_json_sha256(shortlist_rows)
    bundle_manifest_hash = _hash_file_sha256(bundle_manifest_path)
    freeze_contract_payload = {
        "decision_date": bundle_manifest.get("decision_date"),
        "research_data_cutoff": bundle_manifest.get("research_data_cutoff"),
        "promotion_data_cutoff": bundle_manifest.get("promotion_data_cutoff"),
        "holdout_start": bundle_manifest.get("holdout_start"),
        "holdout_end": bundle_manifest.get("holdout_end"),
        "bundle_manifest_hash": bundle_manifest_hash,
        "shortlist_hash": shortlist_hash,
    }
    source_manifest = {
        "manifest_version": DEFAULT_MANIFEST_VERSION,
        "artifact_role": "n_window_shortlist_source",
        "evidence_tier": "research_only",
        "approval_evidence_allowed": False,
        "approval_compatible_requested": bool(approval_compatible),
        "approval_compatible_validated": bool(
            not approval_compatible or len(window_specs) <= APPROVAL_COMPATIBLE_MAX_WINDOWS
        ),
        "bundle_id": bundle_manifest.get("bundle_id"),
        "source_mode": bundle_manifest.get("source_mode", "n_window_consensus_mining"),
        "source_window_count": len(window_specs),
        "source_windows": [
            {
                "window_id": spec.window_id,
                "csv_path": str(spec.csv_path),
                "expected_hash": spec.expected_hash,
                "actual_hash": spec.actual_hash,
                "config_path": spec.config_path,
                "window_role": spec.window_role,
                "weight": spec.weight,
                "row_count": spec.row_count,
            }
            for spec in window_specs
        ],
        "search_space_hash": _hash_json_sha256(sorted(all_signatures)),
        "runtime_budget": bundle_manifest.get("runtime_budget"),
        "selection_metric": selection_metric,
        "aggregation_rule_version": DEFAULT_AGGREGATION_RULE_VERSION,
        "tie_break_rule_version": DEFAULT_TIE_BREAK_RULE_VERSION,
        "shortlist_hash": shortlist_hash,
        "shortlist_size": len(shortlist_rows),
        "candidate_ranking_hash": _hash_json_sha256(candidate_ranking),
        "bundle_manifest_hash": bundle_manifest_hash,
        "freeze_contract_hash": _hash_json_sha256(freeze_contract_payload),
        "freeze_contract_payload": freeze_contract_payload,
        "git_sha": _git_sha(),
        "engine_version_hash": bundle_manifest.get("engine_version_hash", "unknown"),
        "generated_at_utc": _utc_now_iso(),
        "claim_ceiling": "research_only",
        "derivation_reasons": [
            "approval_workflow_unchanged",
            "window_search_forbidden",
            "actual_row_only",
        ],
    }
    derivation_report = {
        "bundle_id": bundle_manifest.get("bundle_id"),
        "selection_metric": selection_metric,
        "shortlist_size": len(shortlist_rows),
        "candidate_pool_size": len(candidate_records),
        "family_pool_size": len(family_groups),
        "max_rank_percentile": max_rank_percentile,
        "optional_fail_budget": optional_fail_budget,
        "minimum_criteria": thresholds,
        "candidate_ranking": candidate_ranking,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    shortlist_csv_path = out_dir / "shortlist_candidates.csv"
    shortlist_json_path = out_dir / "shortlist_candidates.json"
    source_manifest_path = out_dir / "shortlist_source_manifest.json"
    derivation_report_path = out_dir / "shortlist_derivation_report.json"
    summary_path = out_dir / "shortlist_derivation_summary.md"

    _write_csv(shortlist_csv_path, shortlist_rows)
    _write_json(shortlist_json_path, {"rows": shortlist_rows})
    _write_json(source_manifest_path, source_manifest)
    _write_json(derivation_report_path, derivation_report)
    summary_path.write_text(
        _render_summary(
            bundle_manifest=bundle_manifest,
            windows=window_specs,
            shortlist_rows=shortlist_rows,
            optional_fail_budget=optional_fail_budget,
            selection_metric=selection_metric,
            max_rank_percentile=max_rank_percentile,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "shortlist_csv": shortlist_csv_path,
        "shortlist_json": shortlist_json_path,
        "source_manifest": source_manifest_path,
        "derivation_report": derivation_report_path,
        "summary": summary_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive a frozen shortlist from an explicit N-window bundle.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    derive = subparsers.add_parser("derive-shortlist", help="Create shortlist + provenance artifacts")
    derive.add_argument("--bundle-manifest", required=True, help="Path to window_bundle_manifest.json")
    derive.add_argument("--out-dir", required=True, help="Output directory for shortlist artifacts")
    derive.add_argument(
        "--approval-compatible",
        action="store_true",
        help="Reject N-window bundles that exceed the approval-compatible upper bound.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "derive-shortlist":
        raise ValueError(f"unsupported command: {args.command}")
    outputs = derive_shortlist(
        bundle_manifest_path=Path(args.bundle_manifest),
        out_dir=Path(args.out_dir),
        approval_compatible=bool(args.approval_compatible),
    )
    for label, path in outputs.items():
        print(f"[shortlist_derivation] {label}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
