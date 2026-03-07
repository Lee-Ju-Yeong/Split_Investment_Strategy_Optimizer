import unittest
import inspect
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.cpu_gpu_parity_topk import (
    ParityParamRow,
    _apply_decision_evidence,
    _build_parity_summary,
    _compare_curves,
    _load_all_data_to_gpu,
    _load_tier_tensor,
    _resolve_decision_evidence_indices,
    _write_json_artifact,
)


class TestCpuGpuParityTopk(unittest.TestCase):
    def test_gpu_preload_uses_shared_loader_with_price_and_universe_policy(self):
        src = inspect.getsource(_load_all_data_to_gpu)
        self.assertIn("preload_all_data_to_gpu_shared", src)
        self.assertIn("use_adjusted_prices=use_adjusted_prices", src)
        self.assertIn("adjusted_price_gate_start_date=adjusted_price_gate_start_date", src)
        self.assertIn("universe_mode=universe_mode", src)

    def test_param_row_priority_mapping(self):
        row0 = ParityParamRow.from_mapping(
            {
                "max_stocks": 20,
                "order_investment_ratio": 0.02,
                "additional_buy_drop_rate": 0.04,
                "sell_profit_rate": 0.04,
                "additional_buy_priority": 0,
                "stop_loss_rate": -0.15,
                "max_splits_limit": 10,
                "max_inactivity_period": 90,
            }
        )
        self.assertEqual(row0.additional_buy_priority, "lowest_order")
        self.assertEqual(int(row0.to_gpu_row()[4]), 0)

        row1 = ParityParamRow.from_mapping(
            {
                "max_stocks": 20,
                "order_investment_ratio": 0.02,
                "additional_buy_drop_rate": 0.04,
                "sell_profit_rate": 0.04,
                "additional_buy_priority": 1,
                "stop_loss_rate": -0.15,
                "max_splits_limit": 10,
                "max_inactivity_period": 90,
            }
        )
        self.assertEqual(row1.additional_buy_priority, "highest_drop")
        self.assertEqual(int(row1.to_gpu_row()[4]), 1)

    def test_compare_curves(self):
        idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
        cpu = pd.Series([100.0, 101.0, 102.0], index=idx)
        gpu_ok = pd.Series([100.0, 101.0001, 102.0], index=idx)
        gpu_bad = pd.Series([100.0, 105.0, 102.0], index=idx)

        ok = _compare_curves(cpu, gpu_ok, tolerance=1e-3)
        self.assertEqual(ok["mismatch_count"], 0)
        self.assertIsNone(ok["first_mismatch"])

        bad = _compare_curves(cpu, gpu_bad, tolerance=1e-3)
        self.assertGreater(bad["mismatch_count"], 0)
        self.assertIsNotNone(bad["first_mismatch"])

    def test_compare_curves_treats_missing_dates_as_mismatch(self):
        idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
        cpu = pd.Series([100.0, 101.0, 102.0], index=idx)
        gpu_missing = pd.Series([100.0, 102.0], index=idx[[0, 2]])

        result = _compare_curves(cpu, gpu_missing, tolerance=1e-3)

        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(result["first_mismatch"]["date"], "2020-01-03")
        self.assertEqual(result["first_mismatch"]["reason"], "missing_curve_point")

    @patch("src.cpu_gpu_parity_topk.pd.read_sql")
    def test_load_tier_tensor_uses_pit_denominator_for_coverage_gate(self, mock_read_sql):
        try:
            import cupy as cp
        except Exception as exc:  # pragma: no cover - guarded for non-GPU env
            self.skipTest(f"cupy unavailable: {exc}")

        mock_read_sql.return_value = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "ticker": ["C", "D"],
                "tier": [1, 2],
                "liquidity_20d_avg_value": [100, 100],
            }
        )
        trading_dates = pd.DatetimeIndex(pd.to_datetime(["2024-01-02"]))
        try:
            pit_mask = cp.asarray([[1, 1, 0, 0]], dtype=cp.int8)
        except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
            self.skipTest(f"cupy device unavailable: {exc}")

        with self.assertRaisesRegex(ValueError, "Tier coverage gate failed"):
            _load_tier_tensor(
                sql_engine=object(),
                start_date="2024-01-02",
                end_date="2024-01-02",
                all_tickers=["A", "B", "C", "D"],
                trading_dates_pd=trading_dates,
                pit_universe_mask_tensor=pit_mask,
                min_liquidity_20d_avg_value=0,
                min_tier12_coverage_ratio=0.25,
            )

    def test_build_parity_summary_requires_strict_pit_and_zero_mismatch(self):
        rows = [
            {
                "index": 0,
                "compare": {"mismatch_count": 0, "first_mismatch": None},
            },
            {
                "index": 1,
                "compare": {
                    "mismatch_count": 2,
                    "first_mismatch": {"date": "2020-01-03", "abs_diff": 4.0},
                },
            },
        ]

        summary = _build_parity_summary(rows, parity_mode="strict", universe_mode="strict_pit")
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed_indices"], [1])
        self.assertEqual(summary["total_mismatches"], 2)
        self.assertAlmostEqual(summary["max_abs_diff"], 4.0, places=6)
        self.assertFalse(summary["curve_level_parity_zero_mismatch"])
        self.assertFalse(summary["decision_level_parity_zero_mismatch"])
        self.assertTrue(summary["policy_ready_for_release_gate"])
        self.assertEqual(summary["comparison_level"], "equity_curve")
        self.assertTrue(summary["promotion_blocked"])
        self.assertIn("curve_mismatch_count=2", summary["promotion_block_reasons"])
        self.assertIn("decision_level_evidence_missing", summary["promotion_block_reasons"])
        self.assertEqual(summary["first_failed_row"]["index"], 1)

    def test_build_parity_summary_blocks_release_gate_for_non_strict_pit(self):
        rows = [{"index": 0, "compare": {"mismatch_count": 0, "first_mismatch": None}}]

        summary = _build_parity_summary(rows, parity_mode="strict", universe_mode="optimistic_survivor")
        self.assertEqual(summary["failed_indices"], [])
        self.assertEqual(summary["total_mismatches"], 0)
        self.assertAlmostEqual(summary["max_abs_diff"], 0.0, places=6)
        self.assertFalse(summary["curve_level_parity_zero_mismatch"])
        self.assertFalse(summary["policy_ready_for_release_gate"])
        self.assertFalse(summary["decision_level_parity_zero_mismatch"])
        self.assertTrue(summary["promotion_blocked"])
        self.assertIn(
            "policy_not_release_ready(parity_mode=strict, universe_mode=optimistic_survivor)",
            summary["promotion_block_reasons"],
        )

    def test_build_parity_summary_separates_curve_and_decision_level_claims(self):
        rows = [{"index": 0, "compare": {"mismatch_count": 0, "first_mismatch": None}}]

        summary = _build_parity_summary(rows, parity_mode="strict", universe_mode="strict_pit")
        self.assertTrue(summary["curve_level_parity_zero_mismatch"])
        self.assertFalse(summary["decision_level_parity_zero_mismatch"])
        self.assertFalse(summary["event_level_diff_collected"])
        self.assertTrue(summary["promotion_blocked"])
        self.assertIn("decision_level_evidence_missing", summary["promotion_block_reasons"])

    def test_resolve_decision_evidence_indices_uses_first_failed_then_first_row(self):
        rows = [
            {"index": 0, "compare": {"mismatch_count": 0}},
            {"index": 3, "compare": {"mismatch_count": 2}},
            {"index": 5, "compare": {"mismatch_count": 0}},
        ]

        self.assertEqual(_resolve_decision_evidence_indices(rows, "off"), [])
        self.assertEqual(_resolve_decision_evidence_indices(rows, "first_failed"), [3])
        self.assertEqual(_resolve_decision_evidence_indices(rows, "representative"), [3])
        self.assertEqual(_resolve_decision_evidence_indices(rows, "all"), [0, 3, 5])
        self.assertEqual(
            _resolve_decision_evidence_indices(
                [{"index": 7, "compare": {"mismatch_count": 0}}],
                "representative",
            ),
            [7],
        )

    def test_apply_decision_evidence_promotes_only_when_all_rows_are_covered_and_pass(self):
        base_summary = _build_parity_summary(
            [{"index": 0, "compare": {"mismatch_count": 0, "first_mismatch": None}}],
            parity_mode="strict",
            universe_mode="strict_pit",
        )

        partial = _apply_decision_evidence(
            base_summary,
            [],
            total_rows=1,
        )
        self.assertFalse(partial["decision_level_parity_zero_mismatch"])
        self.assertTrue(partial["promotion_blocked"])
        self.assertIn("decision_level_evidence_missing", partial["promotion_block_reasons"])

        complete = _apply_decision_evidence(
            base_summary,
            [
                {
                    "row_index": 0,
                    "decision_level_zero_mismatch": True,
                    "release_decision_fields_complete": True,
                }
            ],
            total_rows=1,
        )
        self.assertTrue(complete["decision_level_parity_zero_mismatch"])
        self.assertFalse(complete["promotion_blocked"])
        self.assertEqual(complete["promotion_block_reasons"], [])

        failed = _apply_decision_evidence(
            base_summary,
            [{"row_index": 0, "decision_level_zero_mismatch": False}],
            total_rows=1,
        )
        self.assertFalse(failed["decision_level_parity_zero_mismatch"])
        self.assertTrue(failed["promotion_blocked"])
        self.assertIn("decision_event_mismatch_rows=1", failed["promotion_block_reasons"])

        partial_scope = _apply_decision_evidence(
            base_summary,
            [{"row_index": 0, "decision_level_zero_mismatch": True}],
            total_rows=1,
        )
        self.assertFalse(partial_scope["decision_level_parity_zero_mismatch"])
        self.assertTrue(partial_scope["promotion_blocked"])
        self.assertIn("decision_fields_not_covered", partial_scope["promotion_block_reasons"])

    def test_write_json_artifact_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "nested" / "parity" / "report.json"
            _write_json_artifact(str(out_path), {"ok": True, "rows": [1, 2]})

            self.assertTrue(out_path.exists())
            self.assertEqual(
                json.loads(out_path.read_text(encoding="utf-8")),
                {"ok": True, "rows": [1, 2]},
            )


if __name__ == "__main__":
    unittest.main()
