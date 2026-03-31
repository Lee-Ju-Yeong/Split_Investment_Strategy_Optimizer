"""
Finalize one champion candidate from a promotion compare summary without rerunning WFO.

This command exists for the exact case where:

- promotion compare already evaluated a broad candidate pack
- a representative shortlist was recompressed afterwards
- the team wants to run CPU audit + holdout on the chosen champion directly

It intentionally avoids rerunning the same promotion WFO windows.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config_loader import load_config
from src.analysis import walk_forward_analyzer as wfo


def _load_candidate_summary(path: Path) -> pd.DataFrame:
    summary_df = pd.read_csv(path)
    if summary_df.empty:
        raise ValueError("candidate summary is empty")
    if "shortlist_candidate_id" not in summary_df.columns:
        raise ValueError("candidate summary is missing shortlist_candidate_id")
    return summary_df.reset_index(drop=True)


def _is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _reorder_candidate_summary(
    summary_df: pd.DataFrame,
    *,
    champion_candidate_id: int | None,
    reserve_count: int,
) -> pd.DataFrame:
    ordered = summary_df.copy().reset_index(drop=True)
    if champion_candidate_id is not None:
        selected = ordered[ordered["shortlist_candidate_id"] == champion_candidate_id]
        if selected.empty:
            raise ValueError(
                f"candidate id {champion_candidate_id} is missing from candidate summary"
            )
        ordered = pd.concat(
            [
                selected.iloc[[0]],
                ordered[ordered["shortlist_candidate_id"] != champion_candidate_id],
            ],
            ignore_index=True,
        )

    champion = ordered.iloc[0]
    if not _is_true(champion.get("hard_gate_pass")):
        raise ValueError(
            "final verdict candidate must already pass the promotion hard gate"
        )

    ordered = ordered.copy()
    ordered["selection_rank"] = range(1, len(ordered) + 1)
    ordered["selection_role"] = "ranked_only"
    ordered.loc[0, "selection_role"] = "champion"

    hard_gate_indices = ordered.index[
        ordered["hard_gate_pass"].map(_is_true)
    ].tolist()
    reserve_indices = [idx for idx in hard_gate_indices if idx != 0][:reserve_count]
    for idx in reserve_indices:
        ordered.loc[idx, "selection_role"] = "reserve"
    return ordered


def _default_results_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"final_candidate_verdict_{stamp}"


def _resolve_output_dir(raw_value: str | None) -> Path:
    if raw_value is None or str(raw_value).strip() == "":
        return _default_results_dir()
    return Path(str(raw_value).strip())


def _run_final_verdict(
    *,
    config: dict,
    candidate_summary_df: pd.DataFrame,
    shortlist_path: str,
    shortlist_hash: str,
    results_dir: Path,
) -> dict:
    wfo_settings = dict(config.get("walk_forward_settings") or {})
    backtest_settings = dict(config.get("backtest_settings") or {})
    holdout_settings = wfo._resolve_holdout_runtime_settings(wfo_settings)
    selection_settings = wfo._resolve_selection_contract_settings(wfo_settings)
    length_basis = wfo._resolve_period_length_basis(wfo_settings)
    trading_dates = wfo._load_runtime_trading_dates(
        config=config,
        backtest_settings=backtest_settings,
        wfo_settings=wfo_settings,
        holdout_settings=holdout_settings,
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    initial_cash = float(backtest_settings.get("initial_cash", 10_000_000))
    backtest_start = str(backtest_settings["start_date"])
    backtest_end = str(backtest_settings["end_date"])
    promotion_cutoff = (
        str(wfo_settings.get("promotion_data_cutoff") or "").strip() or backtest_end
    )
    research_cutoff = (
        str(wfo_settings.get("research_data_cutoff") or "").strip() or backtest_end
    )

    final_candidate_manifest = wfo._build_final_candidate_manifest(
        candidate_summary_df,
        selection_settings=selection_settings,
        shortlist_path=shortlist_path,
        shortlist_hash=shortlist_hash,
        decision_date=wfo_settings.get("decision_date"),
        research_data_cutoff=research_cutoff,
        promotion_data_cutoff=promotion_cutoff,
        holdout_settings=holdout_settings,
        engine_version_hash=wfo_settings.get("engine_version_hash")
        or os.environ.get("MAGICSPLIT_ENGINE_VERSION_HASH"),
        cpu_audit_required=True,
    )

    block_reasons = wfo._build_holdout_auto_execute_block_reasons(
        final_candidate_manifest,
        holdout_start=holdout_settings["holdout_start"],
        holdout_end=holdout_settings["holdout_end"],
    )
    if block_reasons:
        cpu_audit_result = {
            "executed": False,
            "cpu_audit_outcome": "blocked_preconditions",
            "promotion_blocked": True,
            "reasons": block_reasons,
            "metrics": {},
        }
        holdout_result = {
            "attempted": False,
            "success": False,
            "blocked": True,
            "candidate_hash_verified": None,
            "reasons": block_reasons,
            "adequacy_metrics": {},
            "metrics": {},
            "summary_path": None,
            "curve_path": None,
        }
    else:
        cpu_audit_result = wfo._run_final_candidate_cpu_audit(
            config,
            final_candidate_manifest,
            start_date=backtest_start,
            end_date=backtest_end,
            initial_cash=initial_cash,
        )
        if cpu_audit_result["cpu_audit_outcome"] == "pass":
            holdout_result = wfo._run_holdout_from_final_candidate_manifest(
                config,
                final_candidate_manifest,
                holdout_start=str(holdout_settings["holdout_start"]),
                holdout_end=str(holdout_settings["holdout_end"]),
                initial_cash=initial_cash,
                results_dir=results_dir.as_posix(),
            )
        else:
            holdout_result = {
                "attempted": False,
                "success": False,
                "blocked": True,
                "candidate_hash_verified": True,
                "reasons": ["final_candidate_cpu_audit_not_pass"],
                "adequacy_metrics": {},
                "metrics": {},
                "summary_path": None,
                "curve_path": None,
            }

    final_candidate_manifest = wfo._update_final_candidate_manifest_for_holdout(
        final_candidate_manifest,
        cpu_audit_result=cpu_audit_result,
        holdout_result=holdout_result,
    )

    final_candidate_manifest_path = wfo._write_json_artifact(
        (results_dir / "final_candidate_manifest.json").as_posix(),
        final_candidate_manifest,
    )

    holdout_manifest = wfo.build_holdout_manifest(
        holdout_start=holdout_settings["holdout_start"],
        holdout_end=holdout_settings["holdout_end"],
        wfo_end=promotion_cutoff,
        contaminated_ranges=holdout_settings["contaminated_ranges"],
        adequacy_metrics=dict(holdout_result.get("adequacy_metrics") or {}),
        adequacy_thresholds=holdout_settings["adequacy_thresholds"],
        waiver_reason=holdout_settings["waiver_reason"],
        min_length_days=holdout_settings["min_length_days"],
        length_basis=length_basis,
        trading_dates=trading_dates,
        holdout_backtest_attempted=bool(holdout_result.get("attempted")),
        holdout_backtest_success=bool(holdout_result.get("success")),
        holdout_backtest_blocked=bool(holdout_result.get("blocked")),
    )
    holdout_manifest.update(
        {
            "holdout_auto_execute": True,
            "holdout_candidate_id": final_candidate_manifest["champion_candidate_id"],
            "holdout_candidate_hash": final_candidate_manifest["final_candidate_hash"],
            "holdout_candidate_hash_verified": holdout_result.get(
                "candidate_hash_verified"
            ),
            "final_candidate_manifest_path": final_candidate_manifest_path,
            "holdout_summary_path": holdout_result.get("summary_path"),
            "holdout_curve_path": holdout_result.get("curve_path"),
            "holdout_metrics": dict(holdout_result.get("metrics") or {}),
        }
    )

    lane_reasons = wfo._build_current_lane_reasons(
        lane_type="promotion_evaluation",
        total_folds=int(wfo_settings.get("total_folds", 1) or 1),
        overlap_days=0,
        cpu_audit_outcome=str(final_candidate_manifest.get("cpu_audit_outcome") or "unknown"),
    )
    lane_reasons.extend(wfo._build_final_candidate_gate_reasons(final_candidate_manifest))
    lane_approval_eligible = bool(holdout_manifest.get("approval_eligible")) and not lane_reasons
    lane_external_claim_eligible = bool(holdout_manifest.get("external_claim_eligible")) and not lane_reasons
    lane_manifest = wfo.build_lane_manifest(
        lane_type="promotion_evaluation",
        approval_eligible=lane_approval_eligible,
        external_claim_eligible=lane_external_claim_eligible,
        decision_date=wfo_settings.get("decision_date"),
        research_data_cutoff=research_cutoff,
        promotion_data_cutoff=promotion_cutoff,
        shortlist_hash=shortlist_hash,
        publication_lag_policy=wfo_settings.get("publication_lag_policy"),
        ticker_universe_snapshot_id=wfo_settings.get("ticker_universe_snapshot_id"),
        engine_version_hash=wfo_settings.get("engine_version_hash")
        or os.environ.get("MAGICSPLIT_ENGINE_VERSION_HASH"),
        composite_curve_allowed=False,
        cpu_audit_outcome=str(final_candidate_manifest.get("cpu_audit_outcome") or "unknown"),
        selection_cpu_check_outcome="not_applicable",
        reasons=lane_reasons,
    )
    manifest_paths = wfo.write_wfo_manifests(
        results_dir=results_dir.as_posix(),
        lane_manifest=lane_manifest,
        holdout_manifest=holdout_manifest,
    )

    promotion_ablation_df = wfo._build_promotion_ablation_summary(
        candidate_summary_df,
        holdout_manifest,
    )
    promotion_ablation_path = results_dir / "promotion_ablation_summary.csv"
    promotion_ablation_df.to_csv(promotion_ablation_path, index=False)

    explanation_report = wfo._build_promotion_explanation_report(
        candidate_summary_df=candidate_summary_df,
        final_candidate_manifest=final_candidate_manifest,
        holdout_manifest=holdout_manifest,
        lane_manifest=lane_manifest,
        selection_settings=selection_settings,
        ablation_df=promotion_ablation_df,
    )
    explanation_report_path = wfo._write_json_artifact(
        (results_dir / "promotion_explanation_report.json").as_posix(),
        explanation_report,
    )
    explanation_md_path = results_dir / "promotion_explanation_summary.md"
    explanation_md_path.write_text(
        wfo._render_promotion_explanation_markdown(explanation_report),
        encoding="utf-8",
    )

    candidate_summary_path = results_dir / "promotion_candidate_summary.csv"
    candidate_summary_df.to_csv(candidate_summary_path, index=False)

    return {
        "results_dir": results_dir.as_posix(),
        "final_candidate_manifest_path": final_candidate_manifest_path,
        "lane_manifest_path": manifest_paths["lane_manifest_path"],
        "holdout_manifest_path": manifest_paths["holdout_manifest_path"],
        "promotion_ablation_summary_path": promotion_ablation_path.as_posix(),
        "promotion_explanation_report_path": explanation_report_path,
        "promotion_explanation_summary_path": explanation_md_path.as_posix(),
        "promotion_candidate_summary_path": candidate_summary_path.as_posix(),
        "champion_candidate_id": int(final_candidate_manifest["champion_candidate_id"]),
        "cpu_audit_outcome": final_candidate_manifest["cpu_audit_outcome"],
        "holdout_execution_status": final_candidate_manifest["holdout_execution_status"],
        "approval_eligible": bool(lane_manifest.get("approval_eligible")),
        "external_claim_eligible": bool(lane_manifest.get("external_claim_eligible")),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CPU audit + holdout verdict from a promotion compare candidate."
    )
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument(
        "--champion-candidate-id",
        type=int,
        default=None,
        help="Optional explicit candidate id to promote. Defaults to the first row.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional config path. Falls back to MAGICSPLIT_CONFIG_PATH or config/config.yaml.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to results/final_candidate_verdict_<timestamp>.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config_path)
    wfo_settings = dict(config.get("walk_forward_settings") or {})
    shortlist_path = str(wfo_settings.get("promotion_shortlist_path") or "").strip()
    shortlist_hash = str(wfo_settings.get("shortlist_hash") or "").strip()
    if not shortlist_path or not shortlist_hash:
        raise ValueError(
            "final_candidate_verdict requires walk_forward_settings.promotion_shortlist_path "
            "and walk_forward_settings.shortlist_hash in config."
        )

    summary_df = _load_candidate_summary(Path(args.summary_csv))
    selection_settings = wfo._resolve_selection_contract_settings(wfo_settings)
    ordered_df = _reorder_candidate_summary(
        summary_df,
        champion_candidate_id=args.champion_candidate_id,
        reserve_count=selection_settings["reserve_count"],
    )

    result = _run_final_verdict(
        config=config,
        candidate_summary_df=ordered_df,
        shortlist_path=shortlist_path,
        shortlist_hash=shortlist_hash,
        results_dir=_resolve_output_dir(args.out_dir),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
