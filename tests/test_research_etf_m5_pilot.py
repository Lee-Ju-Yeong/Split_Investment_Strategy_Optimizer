"""M5 구간 분리·과거 변동성·모든 실패 기록·합성 파일 연결 검사."""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D, localcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from examples.etf_m5_demo import fixture
from src.research.etf.baseline_policy import VolatilityEstimate
from src.research.etf.engine_inputs import M1Inputs
from src.research.etf.engine_types import Bar, Instrument
from src.research.etf.m5_parameters import candidate_rules, estimate_volatility, grid_candidates, load_config
from src.research.etf.m5_pilot import run_pilot


class ParameterTests(unittest.TestCase):
    def samples(self):
        days = [date(2019, 1, 1) + timedelta(days=n) for n in range(251)]
        calendar = {day: (datetime.combine(day, datetime.min.time(), timezone.utc),
                          datetime.combine(day, datetime.max.time(), timezone.utc)) for day in days}
        rows = [{"date": day.isoformat(), "close": "1" if n % 2 == 0 else "2", "volume": "1"}
                for n, day in enumerate(days)]
        return rows, calendar, days[-1] + timedelta(days=1)

    def test_sample_standard_deviation_has_hand_calculated_denominator(self):
        rows, calendar, start = self.samples()
        actual = estimate_volatility(rows, calendar, start)
        with localcontext(prec=80):
            expected = D(2).ln() * (D(250) / 249).sqrt()
        self.assertLess(abs(actual.value - expected), D("1e-75"))
        self.assertLess(actual.known_at.date(), start)

    def test_future_changes_cannot_change_volatility(self):
        rows, calendar, start = self.samples()
        expected = estimate_volatility(rows, calendar, start)
        rows.append({"date": start.isoformat(), "close": "99999999", "volume": "999999"})
        self.assertEqual(estimate_volatility(rows, calendar, start), expected)

    def test_zero_volume_pairs_and_incomplete_history_are_rejected(self):
        rows, calendar, start = self.samples()
        with self.assertRaisesRegex(ValueError, "251"):
            estimate_volatility(rows[1:], calendar, start)
        for row in rows[:60]:
            row["volume"] = "0"
        with self.assertRaisesRegex(ValueError, "200"):
            estimate_volatility(rows, calendar, start)

    def test_zero_volume_excludes_both_incoming_and_following_returns(self):
        rows, calendar, start = self.samples()
        rows[125]["volume"] = "0"
        with localcontext(prec=80):
            expected = D(2).ln() * (D(248) / 247).sqrt()
        actual = estimate_volatility(rows, calendar, start)
        self.assertLess(abs(actual.value - expected), D("1e-75"))

    def test_grid_has_144_unique_candidates_and_pilot_is_subset(self):
        config = load_config()
        candidates = grid_candidates(config)
        self.assertEqual(len({json.dumps(row, sort_keys=True) for row in candidates}), 144)
        for candidate in config["pilot"]["candidates"]:
            self.assertIn(candidate, candidates)

    def test_drop_under_cost_is_rejected_instead_of_clamped(self):
        at = datetime(2024, 6, 3, tzinfo=timezone.utc)
        bar = Bar("KR", at.date(), at, at + timedelta(hours=6), D(1000), D(1010), D(990), D(1000), D(10000))
        inputs = M1Inputs([Instrument("KR", "KR", "KRW")], [bar], [], [], [], [], {}, {})
        estimate = {"KR": VolatilityEstimate(D(".00001"), at - timedelta(days=1))}
        candidate = load_config()["pilot"]["candidates"][0]
        with self.assertRaisesRegex(ValueError, "round-trip"):
            candidate_rules(candidate, inputs, estimate)
        estimate["KR"] = VolatilityEstimate(D(".01"), at)
        with self.assertRaisesRegex(ValueError, "known before"):
            candidate_rules(candidate, inputs, estimate)

    def test_cost_guard_uses_published_rate_on_actual_settlement_date(self):
        at = datetime(2024, 5, 21, 13, tzinfo=timezone.utc)
        bar = Bar("US", at.date(), at, at + timedelta(hours=7), D(100), D(101), D(99), D(100), D(10000))
        sessions = [date(2024, 5, day) for day in (21, 22, 23)]
        inputs = M1Inputs([Instrument("US", "US", "USD")], [bar], [], [], [], sessions, {}, {})
        estimates = {"US": VolatilityEstimate(D(".00141"), at - timedelta(days=1))}
        candidate = load_config()["pilot"]["candidates"][0]
        with self.assertRaisesRegex(ValueError, "round-trip"):
            candidate_rules(candidate, inputs, estimates)
        estimates["US"] = VolatilityEstimate(D(".001428"), at - timedelta(days=1))
        self.assertEqual(candidate_rules(candidate, inputs, estimates)["US"][0].drop_rate, D(".001428"))


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "inputs"
        self.manifest = fixture(self.data)
        self.symbols = ["SYNTHETIC_KR", "SYNTHETIC_US"]
        self.output = self.root / "run"

    def test_complete_synthetic_pilot_records_4_trials_without_selection(self):
        summary = run_pilot(self.data, self.manifest, self.symbols, self.output)
        self.assertEqual(summary["status"], "passed", summary)
        self.assertEqual(summary["succeeded"], 4)
        self.assertTrue(summary["synthetic"])
        self.assertFalse(summary["statistical_success_evaluated"])
        self.assertFalse(summary["nested_validation_performed"])
        events = [json.loads(line) for line in (self.output / "trials.jsonl").read_text().splitlines()]
        self.assertEqual(len(events), 8)
        trial = json.loads((self.output / "candidate-01-O-H-L-C.json").read_text())
        policy = trial["frozen_policy"]
        self.assertLess(policy["known_at"], policy["evaluation_start"])
        self.assertEqual(len(trial["comparison"]["baselines"]), 7)
        self.assertEqual(trial["training_input"]["start"], "2020-01-02")
        self.assertEqual(trial["evaluation_input"]["start"], "2022-01-03")
        with self.assertRaises(FileExistsError):
            run_pilot(self.data, self.manifest, self.symbols, self.output)

    def test_failed_trial_remains_in_denominator_and_remaining_trials_run(self):
        with patch("src.research.etf.m5_pilot._period", return_value=None), patch(
                "src.research.etf.m5_pilot._trial", side_effect=[ValueError("injected failure"), {}, {}, {}]):
            summary = run_pilot(self.data, self.manifest, self.symbols, self.output)
        self.assertEqual((summary["status"], summary["completed"], summary["failed"], summary["succeeded"]),
                         ("failed", 4, 1, 3))
        self.assertIn("injected failure", (self.output / "trials.jsonl").read_text())

    def test_holdout_and_missing_native_real_data_fail_preflight(self):
        summary = run_pilot(self.data, self.manifest, ["SYNTHETIC_HIDDEN"], self.output)
        self.assertEqual((summary["status"], summary["completed"]), ("failed", 0))
        self.assertIn("holdout", (self.output / "trials.jsonl").read_text())
        manifest = json.loads(self.manifest.read_text())
        manifest["synthetic"] = False
        self.manifest.write_text(json.dumps(manifest))
        output = self.root / "real-failed"
        summary = run_pilot(self.data, self.manifest, self.symbols, output)
        self.assertEqual(summary["status"], "failed")
        self.assertIn("requires an explicit verified M3 build", (output / "trials.jsonl").read_text())

    def test_mid_trial_manifest_change_cannot_publish_a_success(self):
        def change(*args):
            value = json.loads(self.manifest.read_text())
            value["synthetic"] = False
            self.manifest.write_text(json.dumps(value))
            return {}
        with patch("src.research.etf.m5_pilot._period", return_value=None), patch(
                "src.research.etf.m5_pilot._trial", side_effect=change):
            summary = run_pilot(self.data, self.manifest, self.symbols, self.output)
        self.assertEqual((summary["status"], summary["succeeded"], summary["failed"]), ("failed", 0, 4))
        self.assertEqual(list(self.output.glob("candidate-*.json")), [])
        self.assertIn("source or input manifest changed", (self.output / "trials.jsonl").read_text())

    def test_unreadable_configuration_still_records_preflight_failure(self):
        with patch("src.research.etf.m5_pilot.load_config", side_effect=FileNotFoundError("config missing")), patch(
                "src.research.etf.m5_pilot._sha", side_effect=FileNotFoundError("hash unavailable")):
            summary = run_pilot(self.data, self.manifest, self.symbols, self.output)
        self.assertEqual(summary["status"], "failed")
        event = json.loads((self.output / "trials.jsonl").read_text())
        self.assertEqual(event["event"], "preflight_failed")
        self.assertIsNone(event["configuration_sha256"])
        self.assertIn("FileNotFoundError", event["hash_errors"]["configuration_sha256"])

    def test_file_pipeline_imports_without_site_packages(self):
        script = ("import sys; import examples.etf_m5_demo; "
                  "assert 'numpy' not in sys.modules and 'pandas' not in sys.modules")
        result = subprocess.run([sys.executable, "-S", "-c", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == "__main__":
    unittest.main()
