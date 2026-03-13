import tempfile
import unittest
from pathlib import Path

from src.issue98_perf_measure import (
    build_env_snapshot_lines,
    build_input_snapshot,
    build_summary,
    parse_run_log,
)


class TestIssue98PerfMeasure(unittest.TestCase):
    def test_build_env_snapshot_lines_includes_required_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("backtest_settings: {}\nstrategy_params: {}\n", encoding="utf-8")

            lines = build_env_snapshot_lines(
                timestamp="20260308_170000",
                label="pr98_demo",
                config_path=config_path,
                git_head="abc123",
                git_branch="feature/demo",
                python_version="3.10.18",
                cupy_version="13.5.1",
                cudf_version="25.06.00",
                gpu_info="NVIDIA GeForce RTX 5060, 581.80, 8151 MiB",
            )

            joined = "\n".join(lines)
            self.assertIn("timestamp=20260308_170000", joined)
            self.assertIn("label=pr98_demo", joined)
            self.assertIn("git_head=abc123", joined)
            self.assertIn("git_branch=feature/demo", joined)
            self.assertIn("cupy 13.5.1", joined)
            self.assertIn("cudf 25.06.00", joined)
            self.assertIn("NVIDIA GeForce RTX 5060", joined)
            self.assertIn(str(config_path), joined)

    def test_build_input_snapshot_reads_issue98_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "backtest_settings:",
                        "  start_date: '2024-01-01'",
                        "  end_date: '2024-02-29'",
                        "  initial_cash: 10000000",
                        "  simulation_batch_size: 90",
                        "strategy_params:",
                        "  candidate_source_mode: tier",
                        "  tier_hysteresis_mode: strict_hysteresis_v1",
                        "  price_basis: adjusted",
                        "  min_tier12_coverage_ratio: 0.2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = build_input_snapshot(
                config_path=config_path,
                label="pr98_demo",
                timestamp="20260308_170500",
                canonical_profile="issue98_profile",
                run_count=2,
                gpu_sample_interval_sec=5,
                kernel_breakdown=True,
            )

            self.assertEqual(snapshot["label"], "pr98_demo")
            self.assertEqual(snapshot["canonical_profile"], "issue98_profile")
            self.assertEqual(snapshot["start_date"], "2024-01-01")
            self.assertEqual(snapshot["end_date"], "2024-02-29")
            self.assertEqual(snapshot["simulation_batch_size"], 90)
            self.assertEqual(snapshot["candidate_source_mode"], "tier")
            self.assertEqual(snapshot["measurement_run_count"], 2)
            self.assertEqual(snapshot["gpu_sample_interval_sec"], 5)
            self.assertTrue(snapshot["kernel_breakdown"])
            self.assertIsNotNone(snapshot["config_sha256"])

    def test_build_summary_aggregates_logs_and_gpu_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            (outdir / "run1.log").write_text(
                "\n".join(
                    [
                        "✅ Data loaded to GPU. Shape: (103367, 9). Time: 30.76s",
                        "✅ Tier data loaded and tensorized. Shape: (41, 2532). Time: 1.61s",
                        "✅ PIT mask loaded and tensorized. Shape: (41, 2532). Time: 0.35s",
                        "✅ Reusable market-data bundle prepared. Time: 2.50s",
                        "✅ GPU Tensors created successfully in 0.08s.",
                        "[GPU_KERNEL_BREAKDOWN] total_loop_s=1119.00 monthly_rebalance_s=0.50 candidate_select_s=10.00 candidate_payload_s=11.00 strict_rerank_s=0.00 sell_s=300.00 new_entry_s=420.00 additional_buy_s=250.00 valuation_s=122.50",
                        "Total GPU Kernel Execution Time: 1114.65s",
                        "Elapsed (wall clock) time (h:mm:ss or m:ss): 19:24.88",
                        "--- Running Batch 1",
                        "--- Running Batch 2",
                        "--- Running Batch 3",
                        "--- Running Batch 4",
                        "⏱️  Analysis took: 0.71 seconds.",
                        "Exit status: 0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (outdir / "run2.log").write_text(
                "\n".join(
                    [
                        "✅ Data loaded to GPU. Shape: (103367, 9). Time: 17.64s",
                        "✅ Tier data loaded and tensorized. Shape: (41, 2532). Time: 1.54s",
                        "✅ PIT mask loaded and tensorized. Shape: (41, 2532). Time: 0.34s",
                        "✅ Reusable market-data bundle prepared. Time: 2.20s",
                        "✅ GPU Tensors created successfully in 0.07s.",
                        "[GPU_KERNEL_BREAKDOWN] total_loop_s=1121.00 monthly_rebalance_s=0.60 candidate_select_s=9.80 candidate_payload_s=10.80 strict_rerank_s=0.00 sell_s=301.00 new_entry_s=421.00 additional_buy_s=251.00 valuation_s=123.00",
                        "Total GPU Kernel Execution Time: 1117.06s",
                        "Elapsed (wall clock) time (h:mm:ss or m:ss): 19:17.62",
                        "--- Running Batch 1",
                        "--- Running Batch 2",
                        "--- Running Batch 3",
                        "--- Running Batch 4",
                        "⏱️  Analysis took: 0.90 seconds.",
                        "Exit status: 0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (outdir / "run1.gpu.csv").write_text(
                "\n".join(
                    [
                        "2026/03/08 15:24:46.000, 3 %, 5 %, 2200 MiB",
                        "2026/03/08 15:24:51.000, 4 %, 6 %, 2210 MiB",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (outdir / "run2.gpu.csv").write_text(
                "\n".join(
                    [
                        "2026/03/08 16:03:46.000, 3 %, 5 %, 2400 MiB",
                        "2026/03/08 16:03:51.000, 3 %, 5 %, 2422 MiB",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = build_summary(
                outdir=outdir,
                canonical_profile="issue98_profile",
                research_only=True,
            )

            self.assertEqual(summary["canonical_profile"], "issue98_profile")
            self.assertTrue(summary["research_only"])
            self.assertAlmostEqual(summary["median_kernel_s"], 1115.855, places=3)
            self.assertAlmostEqual(summary["median_wall_s"], 1161.25, places=2)
            self.assertEqual(summary["run1"]["batch_count"], 4)
            self.assertEqual(summary["run1"]["gpu_mem_used_max_mib"], 2210)
            self.assertEqual(summary["run2"]["gpu_util_median"], 3.0)
            self.assertAlmostEqual(summary["run1"]["stage_breakdown_s"]["all_data_load_s"], 30.76, places=2)
            self.assertAlmostEqual(summary["run2"]["stage_breakdown_s"]["analysis_s"], 0.90, places=2)
            self.assertAlmostEqual(summary["run1"]["pre_kernel_stage_s"], 35.30, places=2)
            self.assertAlmostEqual(summary["median_stage_breakdown_s"]["prepared_bundle_s"], 2.35, places=2)
            self.assertAlmostEqual(summary["median_stage_breakdown_s"]["analysis_s"], 0.805, places=3)
            self.assertAlmostEqual(summary["run1"]["kernel_stage_breakdown_s"]["new_entry_s"], 420.0, places=2)
            self.assertAlmostEqual(summary["median_kernel_stage_breakdown_s"]["sell_s"], 300.5, places=2)

    def test_parse_run_log_sums_multiple_kernel_breakdown_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run1.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[GPU_KERNEL_BREAKDOWN] total_loop_s=10.00 sell_s=1.00 new_entry_s=2.00 additional_buy_s=7.00",
                        "[GPU_KERNEL_BREAKDOWN] total_loop_s=20.00 sell_s=3.50 new_entry_s=4.50 additional_buy_s=12.00",
                        "Total GPU Kernel Execution Time: 30.10s",
                        "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:31.00",
                        "Exit status: 0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            parsed = parse_run_log(log_path)

            self.assertAlmostEqual(parsed["kernel_stage_breakdown_s"]["total_loop_s"], 30.0, places=2)
            self.assertAlmostEqual(parsed["kernel_stage_breakdown_s"]["sell_s"], 4.5, places=2)
            self.assertAlmostEqual(parsed["kernel_stage_breakdown_s"]["new_entry_s"], 6.5, places=2)
            self.assertAlmostEqual(parsed["kernel_stage_breakdown_s"]["additional_buy_s"], 19.0, places=2)


if __name__ == "__main__":
    unittest.main()
