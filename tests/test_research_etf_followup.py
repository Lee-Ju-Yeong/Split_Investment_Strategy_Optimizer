"""R2 sizing, reset, funding and full planned-sample comparisons."""
from datetime import date, timedelta
from decimal import Decimal as D, localcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from dataclasses import asdict
from unittest.mock import patch

from src.research.etf.baseline_engine import run_baseline
from src.research.etf.baseline_policy import VolatilityEstimate, fixed_baseline
from src.research.etf.engine import run_engine
from src.research.etf.engine_types import FxRate, Instrument, LevelRule
from src.research.etf.followup import _index, _unit_digest, _write_or_verify
from src.research.etf.followup_plan import MIXES, PATHS, Setting, quarter_windows, sleeve_counts, weighted_basket
from src.research.etf.followup_report import _rank_key, _score, phase_settings, summarize
from src.research.etf.followup_run import MonthlyEntry, _trial
from src.research.etf.m5_inputs import _controls
from src.research.etf import protocol
from src.research.etf.m5_parameters import entry_rates
from src.research.etf.m5_runner import _dump
from tests.test_research_etf_engine import DAY, at, bar, kr_run, rules


class BudgetAndFundingTests(unittest.TestCase):
    def test_weights_keep_each_sleeve_and_total_across_sizes(self):
        parent = {sleeve: [f"{i}-{sleeve}" for i in range(8)] for sleeve in
                  ("kr_equity", "us_equity", "us_bond")}
        for mix in MIXES:
            for size in (4, 6, 8):
                for family in range(4):
                    with self.subTest(mix=mix, size=size, family=family), localcontext(prec=80):
                        weights = weighted_basket(mix, parent, size, family)
                        self.assertEqual(len(weights), size)
                        self.assertEqual(sum(map(D, weights.values())), 1)
                        for sleeve, expected in MIXES[mix].items():
                            self.assertEqual(sum(D(w) for s, w in weights.items() if s in parent[sleeve]), D(expected))
                        self.assertEqual(sum(sleeve_counts(mix, size, family).values()), size)

    def test_engine_uses_each_weight_for_every_level(self):
        instruments = [Instrument(symbol, "KR", "KRW") for symbol in ("A", "B", "C", "D")]
        result = run_engine(instruments, [bar(item.symbol) for item in instruments],
                            {item.symbol: rules(second="1000") for item in instruments}, first_entry=False,
                            initial_cash={"KRW": D(10000)}, first_level_fraction=D(".08"),
                            level_budget_weights={"A": D(".5"), "B": D(".25"), "C": D(".125"), "D": D(".125")})
        event = next(r for r in result.cash_events if r["kind"] == "budget_refresh")
        self.assertEqual(event["budgets"], {"A": D(400), "B": D(200), "C": D(100), "D": D(100)})

    def test_large_price_drop_can_fill_through_level_ten(self):
        bars = [bar(day=DAY + timedelta(days=i), price=str(100 - 2 * i)) for i in range(10)]
        ten = tuple(LevelRule(n, D(1000), D(".01"), D(".2")) for n in range(1, 11))
        result = kr_run(bars, level_rules=ten, initial_cash={"KRW": D(12000)})
        self.assertEqual([t["level"] for t in result.trades], list(range(1, 11)))
        tight = kr_run(bars, level_rules=ten, initial_cash={"KRW": D(3000)})
        self.assertTrue(any(t["reason"] == "insufficient_cash" for t in tight.rejections))
        self.assertLess(max(t["level"] for t in tight.trades), 6)

    def test_zero_half_full_usd_targets_and_fx_on_cash(self):
        instruments = [Instrument("KR", "KR", "KRW"), Instrument("US", "US", "USD")]
        for target in ("0", ".5", "1"):
            with self.subTest(target=target):
                result = run_engine(instruments, [bar(), bar("US", market="US")], {i.symbol: rules() for i in instruments},
                                    first_entry=False, initial_cash={"KRW": D(10000)}, fx_cost=0,
                                    fx_rates=[FxRate(at(hour=0), D(1000)), FxRate(at(hour=5), D(1100))],
                                    sec_rate_override=0, usd_target_fraction=D(target), conversion_times=[at(hour=1)],
                                    evaluation_times=[at(hour=2), at(hour=7)])
                self.assertEqual([r["total_krw"] for r in result.valuations], [10000, 10000 + D(target) * 1000])

    def test_market_start_uses_krx_and_the_last_union_session(self):
        kr = [date(2020, 10, 5), date(2021, 10, 5)]
        us = [date(2020, 10, 1), date(2021, 10, 4), date(2021, 10, 5)]
        # Restrict the calendar fixture to two years of this quarter.
        kr.append(date(2022, 10, 4))
        windows = quarter_windows({"KR": kr, "US": us})
        self.assertEqual(windows["2020-10"], (kr[0], us[1]))

    def test_final_us_trade_retains_pending_tax_without_future_fx(self):
        us = Instrument("US", "US", "USD")
        next_day = DAY + timedelta(days=1)
        result = run_engine([us], [bar("US", market="US")], {"US": rules()},
                            initial_cash={"USD": D(1000)}, commission_rate=0, sec_rate_override=0,
                            fx_rates=[FxRate(at(), D(1000))], max_fx_age_days=1,
                            us_sessions=[DAY, next_day], settlement_session_times={next_day: at(next_day, 21)},
                            evaluation_times=[at(hour=21)], _evaluation_cutoff=at(hour=21))
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.valuations[-1]["total_krw"], D(1000000))
        self.assertEqual(result.tax["pending_settlement_events"][0]["settlement_date"], next_day)

    def test_buy_hold_does_not_lose_a_whole_symbol_to_fx_rounding(self):
        instruments = [Instrument(s, "US", "USD") for s in ("A", "B")]
        policy = fixed_baseline(instruments, investment_fraction=1, known_at=at() - timedelta(days=1),
                                evaluation_start=at(), mode="buy_hold")
        result = run_baseline(policy, instruments, [bar(i.symbol, market="US", price="1") for i in instruments],
                              initial_cash={"KRW": D(100000)}, fx_rates=[FxRate(at(), D(1000))],
                              conversion_times=[at(hour=13, minute=30)], evaluation_times=[at(hour=21)],
                              commission_rate=0, sec_rate_override=0, usd_target_fraction=1, _cap_buy_to_cash=True)
        self.assertEqual([(t["symbol"], t["quantity"]) for t in result.trades], [("A", 50), ("B", 49)])


class ResetTests(unittest.TestCase):
    def scenario(self):
        dates = [date.fromisoformat(d) for d in ("2024-12-27", "2024-12-30", "2025-01-02", "2025-01-03", "2025-01-06")]
        bars = [bar(day=day, price=price) for day, price in zip(dates, ("100", "110", "100", "110", "120"))]
        calls = []

        def estimate(symbol, start):
            calls.append((symbol, start))
            return VolatilityEstimate(D(".2"), at(dates[1], 6))

        study = SimpleNamespace(calendars={"KR": dict.fromkeys(dates)}, volatility=estimate)
        setting = Setting(4, 5, kg=1, kp=1, entry="monthly_refresh")
        provider = MonthlyEntry(study, None, setting, D(0))
        return dates, bars, calls, provider

    def test_new_profit_target_stored_in_position_and_used_after_fill(self):
        dates, bars, calls, provider = self.scenario()
        result = kr_run(bars, _reentry_provider=provider)
        self.assertEqual(calls, [("KR", dates[2])])
        self.assertEqual([r["session_date"] for r in result.trades], [dates[i] for i in (0, 1, 2, 4)])
        self.assertEqual(result.trades[2]["reason"], "monthly_reentry")
        self.assertEqual(result.positions, [])

    def test_missing_refresh_inputs_consume_the_one_chance(self):
        dates, bars, calls, provider = self.scenario()

        def missing(symbol, day):
            calls.append((symbol, day))
            raise ValueError("missing history")

        provider.study.volatility = missing
        result = kr_run(bars, _reentry_provider=provider)
        self.assertEqual(calls, [("KR", dates[2])])
        events = [e for e in result.cash_events if e["kind"] == "monthly_reentry"]
        self.assertEqual([event["unavailable"] for event in events], ["missing history"])
        self.assertEqual(len(result.trades), 2)

    def test_rate_must_precede_the_order_open(self):
        candle = bar()
        estimate = VolatilityEstimate(D(".2"), candle.open_at)
        with self.assertRaisesRegex(ValueError, "before the first open"):
            entry_rates(asdict(Setting(4, 5)), None, Instrument("KR", "KR", "KRW"), candle, estimate)


class ReportingTests(unittest.TestCase):
    def test_failed_quarter_keeps_the_family_in_the_denominator(self):
        rows = [(dict(family=family), dict(path=path, status="passed",
                 matched=dict(status="passed", delta_cagr=".01", delta_mdd=".04")))
                for family in ("one", "two") for path in PATHS]
        rows.append((dict(family="two"), dict(path=PATHS[0], status="rejected")))
        score = _score(rows)
        self.assertEqual((score["families"], score["improved_both"]), (2, 1))
        self.assertEqual(score["paths"][PATHS[0]]["paired"], 1)

    def test_depth_tie_break_precedes_size(self):
        score = {"improved_both": 0, "paths": {p: {"median_mdd": None} for p in PATHS}}
        smaller = (Setting(4, 10), score)
        shallower = (Setting(8, 5), score)
        self.assertEqual(min([smaller, shallower], key=_rank_key), shallower)
        self.assertEqual(len(phase_settings("sizing")["kr_equity"]), 21)

    def test_sealed_units_identify_expected_trials(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            unit_dir = root / "units"
            unit_dir.mkdir()
            path = unit_dir / "one.json"
            job = {"id": "one", "settings": [asdict(Setting(4, 5))]}
            trial = {"setting": job["settings"][0], "path": PATHS[0]}
            _dump(path, {"unit": dict(job, trials=[trial])})
            with self.assertRaisesRegex(ValueError, "trial list differs"):
                _index(root, [job])
            complete = dict(job, trials=[dict(trial, path=p) for p in PATHS])
            path.unlink()
            _dump(path, {"unit": complete, "sha256": _unit_digest(complete)})
            self.assertEqual(_index(root, [job])[0], [complete])
            path.unlink()
            _dump(path, {"unit": dict(complete, unexpected=True), "sha256": _unit_digest(complete)})
            with self.assertRaisesRegex(ValueError, "checksum"):
                _index(root, [job])
            _write_or_verify(root / "pins.json", {"source": "old"})
            with self.assertRaisesRegex(ValueError, "pinned execution"):
                _write_or_verify(root / "pins.json", {"source": "new"})

    def test_controls_are_checked_on_the_actual_bytes_used(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "controls.json"
            path.write_text("{}")
            manifest = {"controls": path.name, "files": {path.name: "a" * 64}}
            with self.assertRaisesRegex(ValueError, "changed after"):
                _controls(root, manifest, [], DAY, DAY)

    def test_engine_trial_is_consumed_by_report(self):
        study = SimpleNamespace()
        weights = tuple((s, D(".25")) for s in "ABCD")
        context = {"weights": weights, "usd": D(0), "capital": D(100000), "protocol": protocol.load()}

        def inputs(year):
            start = DAY.replace(year=year)
            instruments = [Instrument(s, "KR", "KRW") for s, _ in weights]
            loaded = SimpleNamespace(instruments=instruments, bars=[bar(s, day=start) for s, _ in weights],
                                     evaluation_times=[at(start, hour=9)], conversion_times=(), fx_rates=[])
            loaded.engine_kwargs = lambda: {"evaluation_times": loaded.evaluation_times}
            return loaded, {s: VolatilityEstimate(D(".04"), at(start) - timedelta(days=1)) for s, _ in weights}

        def python_engine(kernel, *args, **kwargs):
            return run_engine(*args, **kwargs)

        rows = []
        with patch("src.research.etf.followup_run.run_fast_engine", python_engine):
            for path in PATHS:
                trial = _trial(study, None, (inputs(2024), [inputs(2022), inputs(2023)]),
                               context, Setting(4, 5, entry="once"), path, {})
                self.assertEqual(trial["matched"]["status"], "passed")
                rows.append(({"family": "example"}, trial))
        self.assertEqual(_score(rows)["families"], 1)

    def test_forward_stress_gate_and_ranked_setting_are_independent_of_forward_rank(self):
        previous = {mix: {"ranked": [{"setting": asdict(Setting(4, 5, entry="once"))}],
                          "volatility_cuts": [".01", ".02"]} for mix in MIXES}
        variants = phase_settings("forward", previous)
        units = []
        stats = dict(cagr=".03", mdd="-.03", average_investment_fraction=".20",
                     max_filled_level=2, mean_usd_cash=0, mean_usd_holdings=0, fx_fees_krw=0,
                     deep_fills=0, insufficient_cash=0, reentry=dict(attempted=0, filled=0,
                                                                    rejected=0, unavailable={}))
        for mix, settings in variants.items():
            rows = []
            for setting in settings:
                for path in PATHS:
                    matched = {"status": "passed", "delta_cagr": ".02", "delta_mdd": ".04"}
                    if setting.stress and path == PATHS[1]:
                        matched["delta_mdd"] = "-.01"
                    rows.append(dict(setting=asdict(setting), path=path, strategy=stats, status="passed", matched=matched))
            units.append(dict(id=mix, family=mix, mix=mix, quarter="2024-01",
                              starting_volatility=".015", trials=rows))
        summaries = summarize("forward", units, previous)
        for result in summaries.values():
            self.assertEqual(result["candidate"]["setting"]["entry"], "once")
            self.assertEqual(result["candidate"]["failed_conditions"], [f"{PATHS[1]}: cost_stress_positive"])
