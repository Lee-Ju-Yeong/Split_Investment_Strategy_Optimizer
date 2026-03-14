import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.main_backtest import _write_run_manifest
from src.strict_only_governance import (
    evaluate_issue97_gate_c,
    main,
    summarize_issue97_observation,
)


def _parity_artifact(
    mode: str,
    *,
    candidate_ok: bool = True,
    decision_ok: bool = True,
    start_date: str = "2025-01-01",
    end_date: str = "2025-01-28",
) -> dict:
    return {
        "meta": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "summary": {
            "failed": 0,
            "policy_ready_for_release_gate": True,
            "curve_level_parity_zero_mismatch": True,
            "decision_level_parity_zero_mismatch": decision_ok,
            "decision_evidence_release_fields_complete": True,
            "decision_evidence_pit_failure_rows": 0,
            "promotion_blocked": False,
        },
        "decision_evidence": {
            "rows": [
                {
                    "row_index": 0,
                    "release_decision_fields_complete": True,
                    "candidate_order_paired_count": 12,
                    "candidate_order_zero_mismatch": candidate_ok,
                    "pit_failure_code": None,
                    "frozen_candidate_manifest_mode": mode,
                }
            ]
        },
    }


def _run_manifest(
    *,
    created_at: str = "2026-01-01T00:00:00Z",
    start_date: str = "2026-01-01",
    end_date: str = "2026-01-31",
    strict_only: bool = True,
    status: str = "success",
    promotion_blocked: bool = False,
    degraded_run: bool = False,
    empty_entry_day_rate: float = 0.10,
    tier1_coverage: float = 0.60,
    source_lookup_error_days: int = 0,
    source_missing_days: int = 0,
    source_unknown_days: int = 0,
    metrics_cast_error_count: int = 0,
    pit_failure_days: int = 0,
    fatal_pit_failure: bool = False,
) -> dict:
    config = {
        "candidate_source_mode": "tier" if strict_only else "weekly",
        "use_weekly_alpha_gate": False if strict_only else True,
        "tier_hysteresis_mode": "strict_hysteresis_v1" if strict_only else "legacy",
        "candidate_lookup_error_policy": "raise" if strict_only else "skip",
    }
    manifest = {
        "created_at": created_at,
        "status": status,
        "backtest_window": {"start_date": start_date, "end_date": end_date},
        "config": config,
        "universe_policy": {"resolved_mode": "strict_pit" if strict_only else "optimistic_survivor"},
        "safety_guard": {
            "promotion_blocked": promotion_blocked,
            "degraded_run": degraded_run,
        },
        "run_metrics": {
            "empty_entry_day_rate": empty_entry_day_rate,
            "tier1_coverage": tier1_coverage,
            "source_lookup_error_days": source_lookup_error_days,
            "source_missing_days": source_missing_days,
            "source_unknown_days": source_unknown_days,
            "metrics_cast_error_count": metrics_cast_error_count,
            "pit_failure_days_by_code": {"tier12_coverage_gate_failed": pit_failure_days}
            if pit_failure_days
            else {},
        },
    }
    if fatal_pit_failure:
        manifest["error_info"] = {
            "pit_failure": {
                "code": "tier12_coverage_gate_failed",
                "stage": "tier12_coverage_gate",
            }
        }
    return manifest


def _lane_manifest(*, approval_eligible: bool = True, cpu_audit_outcome: str = "pass") -> dict:
    return {
        "lane_type": "promotion_evaluation",
        "evidence_tier": "approval_grade" if approval_eligible else "internal_provisional",
        "approval_eligible": approval_eligible,
        "cpu_audit_outcome": cpu_audit_outcome,
        "reasons": [] if approval_eligible and cpu_audit_outcome == "pass" else ["lane_mode_not_separated"],
    }


def _holdout_manifest(
    *,
    approval_eligible: bool = True,
    holdout_backtest_executed: bool = True,
    promotion_wfo_end_before_holdout: bool = True,
) -> dict:
    return {
        "holdout_class": "approval_grade" if approval_eligible else "internal_provisional",
        "approval_eligible": approval_eligible,
        "holdout_backtest_executed": holdout_backtest_executed,
        "promotion_wfo_end_before_holdout": promotion_wfo_end_before_holdout,
        "reasons": [] if approval_eligible and holdout_backtest_executed and promotion_wfo_end_before_holdout else ["holdout_backtest_not_executed"],
    }


class TestStrictOnlyGovernance(unittest.TestCase):
    def test_evaluate_issue97_gate_c_requires_record_and_replay(self):
        result = evaluate_issue97_gate_c(
            [
                _parity_artifact("record_strict"),
                _parity_artifact("replay_strict"),
            ]
        )

        self.assertTrue(result["approved"])
        self.assertEqual(result["artifacts_checked"], 2)
        self.assertEqual(result["seen_frozen_manifest_modes"], ["record_strict", "replay_strict"])
        self.assertEqual(result["reasons"], [])

    def test_evaluate_issue97_gate_c_blocks_missing_mode_and_candidate_order_drift(self):
        result = evaluate_issue97_gate_c(
            [
                _parity_artifact("record_strict", candidate_ok=False),
            ]
        )

        self.assertFalse(result["approved"])
        self.assertIn("artifact[0].candidate_order_mismatch_rows=1", result["reasons"])
        self.assertIn("artifact_count=1<2", result["reasons"])
        self.assertIn("missing_frozen_manifest_mode=replay_strict", result["reasons"])

    def test_summarize_issue97_observation_passes_two_week_window(self):
        manifests = [
            _run_manifest(
                created_at="2026-03-07T00:00:00Z",
                start_date=f"2025-{month:02d}-01",
                end_date=f"2025-{month:02d}-28",
            )
            for month in range(1, 11)
        ]

        summary = summarize_issue97_observation(
            manifests,
            parity_artifacts=[
                _parity_artifact(
                    "record_strict",
                    start_date="2025-01-01",
                    end_date="2025-01-28",
                ),
                _parity_artifact(
                    "replay_strict",
                    start_date="2025-02-01",
                    end_date="2025-02-28",
                ),
            ],
            lane_manifests=[_lane_manifest()],
            holdout_manifests=[_holdout_manifest()],
        )

        self.assertTrue(summary["approved"])
        self.assertEqual(summary["observation_mode"], "synthetic_window_pack")
        self.assertEqual(summary["observation_run_count"], 10)
        self.assertEqual(summary["unique_backtest_windows"], 10)
        self.assertEqual(summary["strict_only_run_count"], 10)
        self.assertEqual(summary["matched_parity_windows"], 2)
        self.assertEqual(summary["parity_mismatch_runs"], 0)
        self.assertAlmostEqual(summary["p95_empty_entry_day_rate"], 0.10, places=4)
        self.assertAlmostEqual(summary["median_tier1_coverage"], 0.60, places=4)

    def test_summarize_issue97_observation_flags_non_strict_and_threshold_breach(self):
        manifests = [
            _run_manifest(
                created_at="2026-03-07T00:00:00Z",
                start_date=f"2025-{month:02d}-01",
                end_date=f"2025-{month:02d}-28",
            )
            for month in range(1, 9)
        ]
        manifests.append(
            _run_manifest(
                created_at="2026-03-07T00:00:00Z",
                start_date="2025-09-01",
                end_date="2025-09-28",
                strict_only=False,
                promotion_blocked=True,
                degraded_run=True,
            )
        )
        manifests.append(
            _run_manifest(
                created_at="2026-03-07T00:00:00Z",
                start_date="2025-09-01",
                end_date="2025-09-28",
                empty_entry_day_rate=0.35,
                tier1_coverage=0.40,
                source_missing_days=1,
                source_unknown_days=1,
                metrics_cast_error_count=1,
                pit_failure_days=1,
            )
        )

        summary = summarize_issue97_observation(
            manifests,
            parity_artifacts=[
                _parity_artifact(
                    "record_strict",
                    decision_ok=False,
                    start_date="2024-01-01",
                    end_date="2024-01-28",
                )
            ],
        )

        self.assertFalse(summary["approved"])
        self.assertIn("unique_backtest_windows=9<10", summary["reasons"])
        self.assertIn("matched_parity_windows=0<1", summary["reasons"])
        self.assertIn("non_strict_run_count=1", summary["reasons"])
        self.assertIn("promotion_blocked_runs=1", summary["reasons"])
        self.assertIn("degraded_runs=1", summary["reasons"])
        self.assertIn("source_missing_days=1", summary["reasons"])
        self.assertIn("source_unknown_days=1", summary["reasons"])
        self.assertIn("metrics_cast_error_count=1", summary["reasons"])
        self.assertIn("pit_failure_days=1", summary["reasons"])
        self.assertIn("parity_mismatch_runs=1", summary["reasons"])

    def test_summarize_issue97_observation_flags_provisional_wfo_manifests(self):
        manifests = [
            _run_manifest(
                created_at="2026-03-07T00:00:00Z",
                start_date=f"2025-{month:02d}-01",
                end_date=f"2025-{month:02d}-28",
            )
            for month in range(1, 11)
        ]

        summary = summarize_issue97_observation(
            manifests,
            parity_artifacts=[
                _parity_artifact(
                    "record_strict",
                    start_date="2025-01-01",
                    end_date="2025-01-28",
                )
            ],
            lane_manifests=[_lane_manifest(approval_eligible=False, cpu_audit_outcome="enabled_but_no_selection_audit")],
            holdout_manifests=[_holdout_manifest(approval_eligible=False, holdout_backtest_executed=False)],
        )

        self.assertFalse(summary["approved"])
        self.assertEqual(summary["lane_manifest_count"], 1)
        self.assertEqual(summary["holdout_manifest_count"], 1)
        self.assertIn("lane_manifest[0].approval_ineligible", summary["reasons"])
        self.assertIn(
            "lane_manifest[0].cpu_audit_required_for_promotion=enabled_but_no_selection_audit",
            summary["reasons"],
        )
        self.assertIn("holdout_manifest[0].holdout_backtest_not_executed", summary["reasons"])

    def test_summarize_issue97_observation_requires_lane_and_holdout_manifests(self):
        manifests = [
            _run_manifest(
                created_at="2026-03-07T00:00:00Z",
                start_date=f"2025-{month:02d}-01",
                end_date=f"2025-{month:02d}-28",
            )
            for month in range(1, 11)
        ]

        summary = summarize_issue97_observation(
            manifests,
            parity_artifacts=[
                _parity_artifact(
                    "record_strict",
                    start_date="2025-01-01",
                    end_date="2025-01-28",
                )
            ],
        )

        self.assertFalse(summary["approved"])
        self.assertIn("lane_manifest_count=0<1", summary["reasons"])
        self.assertIn("holdout_manifest_count=0<1", summary["reasons"])

    def test_write_run_manifest_records_use_weekly_alpha_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_run_manifest(
                result_dir=tmpdir,
                strategy_params={
                    "candidate_source_mode": "tier",
                    "use_weekly_alpha_gate": False,
                    "tier_hysteresis_mode": "strict_hysteresis_v1",
                    "candidate_lookup_error_policy": "raise",
                },
                backtest_settings={"start_date": "2026-01-05", "end_date": "2026-01-09"},
                universe_mode="strict_pit",
                price_basis="adjusted",
                adjusted_gate="2013-11-20",
                run_metrics={},
                candidate_lookup_summary={},
                safety_guard={},
            )

            payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            self.assertIn("use_weekly_alpha_gate", payload["config"])
            self.assertFalse(payload["config"]["use_weekly_alpha_gate"])

    def test_cli_gate_c_writes_json_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = Path(tmpdir) / "record.json"
            replay_path = Path(tmpdir) / "replay.json"
            out_path = Path(tmpdir) / "gate_c.json"
            record_path.write_text(json.dumps(_parity_artifact("record_strict")), encoding="utf-8")
            replay_path.write_text(json.dumps(_parity_artifact("replay_strict")), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                rc = main(
                    [
                        "--mode",
                        "gate-c",
                        "--parity-json",
                        str(record_path),
                        "--parity-json",
                        str(replay_path),
                        "--out",
                        str(out_path),
                    ]
                )

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(rc, 0)
            self.assertTrue(payload["approved"])

    def test_cli_observation_supports_glob_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            runs_dir.mkdir()
            parity_path = Path(tmpdir) / "record.json"
            lane_path = Path(tmpdir) / "lane_manifest.json"
            holdout_path = Path(tmpdir) / "holdout_manifest.json"
            parity_path.write_text(
                json.dumps(
                    _parity_artifact(
                        "record_strict",
                        start_date="2025-01-01",
                        end_date="2025-01-28",
                    )
                ),
                encoding="utf-8",
            )
            lane_path.write_text(json.dumps(_lane_manifest()), encoding="utf-8")
            holdout_path.write_text(json.dumps(_holdout_manifest()), encoding="utf-8")
            for idx, month in enumerate(range(1, 11)):
                run_dir = runs_dir / f"run_{idx:02d}"
                run_dir.mkdir()
                (run_dir / "run_manifest.json").write_text(
                    json.dumps(
                        _run_manifest(
                            created_at="2026-03-07T00:00:00Z",
                            start_date=f"2025-{month:02d}-01",
                            end_date=f"2025-{month:02d}-28",
                        )
                    ),
                    encoding="utf-8",
                )
            out_path = Path(tmpdir) / "observation.json"

            with redirect_stdout(io.StringIO()):
                rc = main(
                    [
                        "--mode",
                        "observation",
                        "--run-manifest-glob",
                        str(runs_dir / "run_*" / "run_manifest.json"),
                        "--parity-json",
                        str(parity_path),
                        "--lane-manifest-json",
                        str(lane_path),
                        "--holdout-manifest-json",
                        str(holdout_path),
                        "--out",
                        str(out_path),
                    ]
                )

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(rc, 0)
            self.assertTrue(payload["approved"])
            self.assertEqual(payload["observation_run_count"], 10)

    def test_summarize_issue97_observation_requires_matched_parity_window(self):
        manifests = [
            _run_manifest(
                created_at="2026-03-07T00:00:00Z",
                start_date=f"2025-{month:02d}-01",
                end_date=f"2025-{month:02d}-28",
            )
            for month in range(1, 11)
        ]

        summary = summarize_issue97_observation(
            manifests,
            parity_artifacts=[
                _parity_artifact(
                    "record_strict",
                    start_date="2024-11-01",
                    end_date="2024-11-28",
                )
            ],
        )

        self.assertFalse(summary["approved"])
        self.assertEqual(summary["matched_parity_windows"], 0)
        self.assertIn("matched_parity_windows=0<1", summary["reasons"])

    def test_cli_returns_nonzero_on_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = Path(tmpdir) / "record.json"
            out_path = Path(tmpdir) / "gate_c_fail.json"
            record_path.write_text(json.dumps(_parity_artifact("record_strict")), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                rc = main(
                    [
                        "--mode",
                        "gate-c",
                        "--parity-json",
                        str(record_path),
                        "--out",
                        str(out_path),
                    ]
                )

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(rc, 1)
            self.assertFalse(payload["approved"])


if __name__ == "__main__":
    unittest.main()
