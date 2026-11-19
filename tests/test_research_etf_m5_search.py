"""정식 M5 계획(시작일·후보군·바스켓)과 실행기(학습 비중·재개·실패 기록) 검사."""
import copy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.etf_m5_demo import fixture
from src.research.etf.baseline_engine import run_baseline
from src.research.etf.baseline_policy import fixed_baseline
from src.research.etf.engine_types import Instrument
from src.research.etf.m5_parameters import CostGuardRejection
from src.research.etf.m5_pilot import _common, _window_start
from src.research.etf.m5_runner import matched_policy, run_plan, status
from src.research.etf.m5_search import build_plan, draw_basket, eligible, load_formal, start_windows


UTC = timezone.utc
FIRST = date(2020, 1, 1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quick_strategy(inputs, estimates, candidate, path, capital, kernel):
    """실행기 흐름 검사용 빠른 대역. 실제 전략 대신 투자 비중 30% 기준선 원장을 돌려준다."""
    start = _window_start(inputs)
    policy = fixed_baseline(inputs.instruments, investment_fraction=D(".3"),
                            known_at=start - timedelta(seconds=1), evaluation_start=start)
    return run_baseline(policy, inputs.instruments, inputs.bars, **_common(inputs, candidate, path, capital))


class PlanRuleTests(unittest.TestCase):
    def test_windows_start_in_january_and_july_and_never_overlap(self):
        union = [date(year, month, day) for year in (2020, 2021, 2022) for month in (1, 6, 7, 12) for day in (2, 3, 30)]
        windows = start_windows(union, (2020, 7), date(2022, 6, 30))
        self.assertEqual(windows, [(date(2020, 7, 2), date(2021, 6, 30)), (date(2021, 1, 2), date(2021, 12, 30)),
                                   (date(2021, 7, 2), date(2022, 6, 30))])
        self.assertEqual(start_windows(union, (2020, 7), date(2022, 6, 29))[-1][0], date(2021, 1, 2))

    def test_eligibility_uses_only_the_20_sessions_before_the_start(self):
        sessions = [date(2020, 1, 1) + timedelta(days=n) for n in range(30)]
        start = sessions[25]
        rows = {day.isoformat(): ("10", "5") for day in sessions[5:25]}
        item = {"first": "2017-01-01", "rows": rows}
        self.assertTrue(eligible(item, start, sessions, D("50")))
        self.assertFalse(eligible(item, start, sessions, D("50.01")))
        self.assertTrue(eligible(dict(item, rows=dict(rows, **{start.isoformat(): ("1", "0")})), start, sessions, D("50")))
        self.assertFalse(eligible(dict(item, first="2017-02-01"), start, sessions, D("50")))
        missing = {day: value for day, value in rows.items() if day != sessions[10].isoformat()}
        self.assertFalse(eligible(dict(item, rows=missing), start, sessions, None))
        self.assertFalse(eligible(item, sessions[10], sessions, None))

    def test_basket_draw_is_deterministic_and_uses_each_index_group_once(self):
        pool = {"KR": [("K1", "A"), ("K2", "A"), ("K3", "B")], "US": [("U1", "B"), ("U2", "C"), ("U3", "C")]}
        stratum = {"KR": 2, "US": 1, "GOLD": 1}
        basket = draw_basket(pool, stratum, 7, "key", "GOLD")
        self.assertEqual(basket, draw_basket(pool, stratum, 7, "key", "GOLD"))
        self.assertEqual((len(basket), basket[-1], "K3" in basket), (4, "GOLD", True))
        self.assertEqual(len({"K1", "K2"} & set(basket)), 1)
        self.assertIn(basket[2], {"U2", "U3"})
        self.assertIsNone(draw_basket(pool, {"KR": 3}, 7, "key", "GOLD"))
        self.assertGreater(len({tuple(draw_basket(pool, {"KR": 1}, 7, n, "GOLD")) for n in range(40)}), 1)


class MatchedPolicyTests(unittest.TestCase):
    def test_learning_average_drops_values_after_the_evaluation_start(self):
        at = lambda day, hour=9: datetime(2021, 12, day, hour, tzinfo=UTC).isoformat()
        records = [{"sum": "1.0", "count": 4, "tail": [[at(1), "0.2"], [at(2), "0.3"], [at(3), "0.5"]]},
                   {"sum": "0.6", "count": 3, "tail": [[at(28), "0.1"], [at(29), "0.2"], [at(31), "0.3"]]}]
        instruments = [Instrument("A", "KR", "KRW"), Instrument("B", "US", "USD")]
        cutoff = datetime(2021, 12, 31, 1, tzinfo=UTC)
        policy = matched_policy(records, instruments, cutoff)
        self.assertLess(abs(policy.investment_fraction - D("1.3") / 6), D("1e-25"))
        self.assertEqual((policy.known_at.isoformat(), policy.evaluation_start), (at(29), cutoff))
        self.assertEqual(dict(policy.weights), {"A": D("0.5"), "B": D("0.5")})
        later = matched_policy(records, instruments, datetime(2022, 1, 3, tzinfo=UTC))
        self.assertLess(abs(later.investment_fraction - D("1.6") / 7), D("1e-25"))
        with self.assertRaisesRegex(ValueError, "does not end before"):
            matched_policy(records, instruments, datetime(2021, 12, 28, tzinfo=UTC))


class FormalRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "inputs"
        self.manifest = fixture(self.data)
        self.plan = build_plan(self.data, self.manifest, first_start=FIRST, pilot_candidates=True)
        self.plan_path = self.save("plan.json", self.plan)
        self.run = self.root / "run"

    def save(self, name, plan):
        path = self.root / name
        path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
        return path

    def unit(self, start):
        return json.loads((self.run / "units" / "b001" / f"{start}.json").read_text(encoding="utf-8"))

    def test_plan_keeps_holdout_out_and_counts_every_planned_trial(self):
        self.assertEqual(self.plan, build_plan(self.data, self.manifest, first_start=FIRST, pilot_candidates=True))
        self.assertEqual((self.plan["scope"], self.plan["synthetic"]), ("smoke_not_formal", True))
        basket, = self.plan["baskets"]
        self.assertEqual((basket["stratum"], basket["symbols"]), ("KR1+US1", ["SYNTHETIC_KR", "SYNTHETIC_US"]))
        self.assertEqual([run["start"][:7] for run in basket["runs"]],
                         ["2020-01", "2020-07", "2021-01", "2021-07", "2022-01"])
        self.assertEqual(basket["evaluations"], [basket["runs"][-1]["start"]])
        for earlier, later in zip(basket["runs"][::2], basket["runs"][2::2]):
            self.assertLess(earlier["end"], later["start"])
        self.assertEqual(self.plan["counts"], {"baskets": 1, "units": 5, "evaluations": 1, "trials": 20, "comparisons": 4})
        self.assertEqual({key: row["generated"] for key, row in self.plan["shortfalls"].items()},
                         {"KR2": 0, "US2": 0, "KR1+US1": 1, "KR1+US1+GOLD1": 0, "KR2+US2": 0, "KR2+US2+GOLD1": 0})
        self.assertEqual(self.plan["fx"], {"max_age_days": 7, "closure_age_days": 12})
        self.assertNotIn("SYNTHETIC_HIDDEN", json.dumps(self.plan))
        formal = build_plan(self.data, self.manifest)
        self.assertEqual((formal["scope"], len(formal["candidates"])), ("formal", 144))
        with patch("src.research.etf.m5_search.load_formal",
                   return_value=dict(load_formal(), status="draft_preperformance")):
            self.assertEqual(build_plan(self.data, self.manifest)["scope"], "draft_not_formal")
        other = fixture(self.root / "us", us_only=True)
        us_only = build_plan(other.parent, other, first_start=FIRST, pilot_candidates=True)
        self.assertEqual([sorted(b["symbols"]) for b in us_only["baskets"]], [["SYNTHETIC_US", "SYNTHETIC_US_2"]])
        self.assertNotIn("SYNTHETIC_KR", json.dumps(us_only["baskets"]))

    def test_real_strategy_run_learns_exposure_from_the_two_prior_january_runs(self):
        plan = copy.deepcopy(self.plan)
        basket = plan["baskets"][0]
        basket["runs"] = [run for run in basket["runs"] if run["start"][5:7] == "01"]
        plan.update(candidates=plan["candidates"][1:], paths=plan["paths"][:1],
                    counts=dict(plan["counts"], units=3, trials=3, comparisons=1))
        summary = run_plan(self.data, self.manifest, self.save("small.json", plan), self.run)
        self.assertEqual(summary["status"], "passed", summary)
        self.assertEqual((summary["trials_passed"], summary["comparisons_passed"], summary["references_failed"]), (3, 1, 0))
        self.assertFalse(summary["statistical_success_evaluated"] or summary["nested_validation_performed"])
        learned = [self.unit(run["start"])["trials"][0] for run in basket["runs"][:2]]
        evaluation = self.unit(basket["runs"][2]["start"])
        row, = evaluation["trials"]
        self.assertNotIn("comparison", learned[0])
        self.assertEqual(evaluation["input"]["fx_closure_age_days"], 12)
        self.assertEqual(set(evaluation["references"]["O-H-L-C|0.1|0.5"]),
                         {"buy_hold", "frontier_000", "frontier_025", "frontier_050", "frontier_075", "frontier_100"})
        matched = row["comparison"]["matched"]
        expected = sum(D(t["exposure"]["sum"]) for t in learned) / sum(t["exposure"]["count"] for t in learned)
        self.assertLess(abs(D(matched["investment_fraction"]) - expected), D("1e-25"))
        self.assertEqual(matched["known_at"], learned[1]["exposure"]["tail"][-1][0])
        self.assertLess(datetime.fromisoformat(matched["known_at"]),
                        datetime.fromisoformat(basket["runs"][2]["start"] + "T00:00:00+00:00"))
        self.assertLess(abs(D(matched["delta_cagr"]) - (D(row["strategy"]["cagr"]) - D(matched["performance"]["cagr"]))),
                        D("1e-25"))
        self.assertLess(abs(D(matched["delta_mdd"]) - (abs(D(matched["performance"]["mdd"]))
                                                       - abs(D(row["strategy"]["mdd"])))), D("1e-25"))
        self.assertLess(abs(D(row["comparison"]["posthoc_diagnostic"]["investment_fraction"])
                            - D(row["strategy"]["average_investment_fraction"])), D("1e-25"))

    def test_run_resumes_without_rewriting_and_refuses_a_different_plan(self):
        with patch("src.research.etf.m5_runner._strategy", side_effect=quick_strategy):
            partial = run_plan(self.data, self.manifest, self.plan_path, self.run, max_units=2)
            self.assertEqual((partial["status"], partial["units_completed"], partial["units_missing"]), ("incomplete", 2, 3))
            first = self.run / "units" / "b001" / f"{self.plan['baskets'][0]['runs'][0]['start']}.json"
            before = sha(first)
            stale = first.with_name(f"{self.plan['baskets'][0]['runs'][2]['start']}.json.partial")
            stale.write_text("{\"interrupted\":", encoding="utf-8")
            summary = run_plan(self.data, self.manifest, self.plan_path, self.run)
            self.assertEqual(summary["status"], "passed", summary)
            self.assertEqual((summary["trials_passed"], summary["comparisons_passed"], summary["units_failed"]), (20, 4, 0))
            self.assertEqual(before, sha(first))
            self.assertEqual(run_plan(self.data, self.manifest, self.plan_path, self.run), dict(summary))
            with self.assertRaisesRegex(ValueError, "different source, input, plan or build"):
                run_plan(self.data, self.manifest, self.save("other.json", dict(self.plan, posthoc_diagnostic=False)),
                         self.run)
        self.assertEqual(sorted(path.name for path in self.run.rglob("*.partial")), [])

    def test_failed_learning_run_stays_in_the_denominator_and_fails_its_comparison(self):
        calls = []

        def flaky(*args):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("injected failure")
            return quick_strategy(*args)
        with patch("src.research.etf.m5_runner._strategy", side_effect=flaky):
            summary = run_plan(self.data, self.manifest, self.plan_path, self.run)
        self.assertEqual((summary["status"], summary["trials_passed"], summary["comparisons_passed"]), ("failed", 19, 3))
        self.assertEqual((summary["trials_planned"], summary["comparisons_planned"]), (20, 4))
        runs = self.plan["baskets"][0]["runs"]
        self.assertIn("injected failure", self.unit(runs[0]["start"])["trials"][0]["error"])
        failed = self.unit(runs[4]["start"])["trials"][0]
        self.assertEqual((failed["status"], failed["comparison"]["status"]), ("passed", "failed"))
        self.assertIn("learning run", failed["comparison"]["error"])

    def test_cost_guard_rejection_is_counted_apart_from_failures(self):
        calls = []

        def rejecting(*args):
            calls.append(1)
            if len(calls) == 1:
                raise CostGuardRejection("X: drop below known round-trip costs or >= 1")
            return quick_strategy(*args)
        with patch("src.research.etf.m5_runner._strategy", side_effect=rejecting):
            summary = run_plan(self.data, self.manifest, self.plan_path, self.run)
        self.assertEqual((summary["status"], summary["trials_passed"], summary["trials_rejected"]), ("passed", 19, 1))
        self.assertEqual((summary["comparisons_passed"], summary["comparisons_rejected"]), (3, 1))
        runs = self.plan["baskets"][0]["runs"]
        first = self.unit(runs[0]["start"])["trials"][0]
        self.assertEqual((first["status"], "traceback" in first), ("rejected", False))
        self.assertEqual(self.unit(runs[4]["start"])["trials"][0]["comparison"]["status"], "rejected")

    def test_real_data_requires_a_verified_build_and_plan_must_match_inputs(self):
        with self.assertRaisesRegex(ValueError, "verified M3 build"):
            run_plan(self.data, self.manifest, self.save("real.json", dict(self.plan, synthetic=False)),
                     self.root / "real-run")
        empty = dict(self.plan, baskets=[], counts=dict.fromkeys(self.plan["counts"], 0))
        summary = run_plan(self.data, self.manifest, self.save("empty.json", empty), self.root / "empty-run")
        self.assertEqual((summary["status"], summary["units_planned"]), ("failed", 0))
        short = status(dict(self.plan, scope="formal"), self.root / "short-run")
        self.assertEqual((short["status"], short["shortfall_strata"]), ("incomplete", 6))
        for scope in ("formal", "draft_not_formal", "unknown"):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "plan differs|invalid M5 plan"):
                run_plan(self.data, self.manifest, self.save(f"{scope}.json", dict(self.plan, scope=scope)),
                         self.root / f"{scope}-run")
        with self.assertRaisesRegex(ValueError, "different protocol_sha256"):
            run_plan(self.data, self.manifest, self.save("stale.json", dict(self.plan, protocol_sha256="0" * 64)),
                     self.root / "stale-run")


if __name__ == "__main__":
    unittest.main()
