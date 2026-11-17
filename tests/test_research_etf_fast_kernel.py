"""명시 빌드가 필요한 M3 검사. 누락·건너뜀을 성공으로 처리하지 않는다.

MAGICSPLIT_ETF_M3_BUILD를 지정한 독립 C++ 검사에서 실행한다. M2의 기존 손계산
검사도 같은 기대값으로 네이티브 후보에 적용한다. 제품 CPU 기본 검사와 분리한다.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D, localcontext
from functools import lru_cache
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.research.etf.engine import run_engine
from src.research.etf.engine_strategy import candidate_for_bar
from src.research.etf.engine_types import Bar, FxRate, Instrument, LevelRule, Position
from src.research.etf.fast_build import build_kernel
from src.research.etf.fast_kernel import FastKernel, ROUNDING_MODES, run_fast_engine

OPEN = datetime(2024, 6, 3, tzinfo=timezone.utc)


@lru_cache(maxsize=1)
def kernel():
    return FastKernel(os.environ["MAGICSPLIT_ETF_M3_BUILD"])


def input_case(market, *, price="100", spread="1"):
    instrument = Instrument("ETF", market, "USD" if market == "US" else "KRW", "국내주식형")
    value, width = D(price), D(spread)
    bar = Bar("ETF", OPEN.date(), OPEN, OPEN + timedelta(hours=6), value,
              value + width, value - width, value, D("1000000"), adv20=D("1000000000"))
    rules = tuple(LevelRule(level, value * 100, D(".005"), D(".005")) for level in range(1, 5))
    held = [Position("ETF", 1, 10, value, D(0), D(".005"), OPEN - timedelta(days=1))]
    return instrument, bar, held, rules


class NativeParityTests(unittest.TestCase):
    def assert_candidate(self, inputs, **options):
        options.setdefault("sec_rate_override", D(0))
        expected = candidate_for_bar(*inputs, **options)
        self.assertEqual(kernel().candidate_for_bar(*inputs, **options), expected)
        return expected

    def test_seeded_candidate_parity(self):
        rng = random.Random(20261117)
        for market, price, spread in (("KR", "1999", "25"), ("US", "1", ".01"), ("GOLD", "100000", "200")):
            for index in range(60):
                instrument, bar, held, rules = input_case(market, price=price, spread=spread)
                slots = rng.sample(range(1, 5), rng.randrange(5))
                held = [replace(held[0], level=level, take_rate=D(rng.randrange(0, 150)) / 10000,
                                quantity=rng.randrange(1, 20)) for level in slots]
                rules = tuple(replace(rule, budget=D(rng.randrange(1, 200)) * bar.open,
                                      drop_rate=D(rng.randrange(0, 150)) / 10000) for rule in rules)
                bar = replace(bar, close=bar.low + (bar.high - bar.low) * D(rng.randrange(101)) / 100)
                for path in ("O-H-L-C", "O-L-H-C"):
                    with self.subTest(market=market, case=index, path=path):
                        self.assert_candidate((instrument, bar, held, rules), path=path,
                                              first_entry=not held and index % 2 == 0,
                                              penetration_ticks=index % 3, slippage_ticks=index % 3,
                                              commission_rate=D(".00015"), product_sell_tax_rate=D(".002"))

    def test_tick_table_transition_and_tiny_price(self):
        for market, price, spread in (("KR", "2000", "6"), ("US", "1", ".001"), ("US", ".0002", ".00009")):
            for day in (date(2023, 12, 10), date(2023, 12, 11)):
                instrument, bar, held, rules = input_case(market, price=price, spread=spread)
                bar = replace(bar, session_date=day)
                for path in ("O-H-L-C", "O-L-H-C"):
                    for steps in (0, 1, 2):
                        with self.subTest(market=market, price=price, day=day, path=path, steps=steps):
                            self.assert_candidate((instrument, bar, held, rules), path=path, slippage_ticks=steps)

    def test_no_signal_with_all_levels_held(self):
        inputs = input_case("US")
        held = [replace(inputs[2][0], level=level, take_rate=D(10)) for level in range(1, 5)]
        self.assertIsNone(self.assert_candidate((*inputs[:2], held, inputs[3])))

    def test_manual_open_keeps_original_decimal_digits_and_lazy_date_lookup(self):
        instrument, bar, _, rules = input_case("KR")
        value = D("100." + "0" * 78 + "1")
        bar = replace(bar, session_date=date(2026, 11, 18), open=value, high=value, low=value, close=value)
        actual = self.assert_candidate((instrument, bar, [], rules), first_entry=True, commission_rate=D(0))
        self.assertEqual(actual.price, value)

    def test_decimal80_step_rounding_changes_gap_qualification(self):
        instrument, bar, held, rules = input_case("US")
        value = D("90." + "0" * 80 + "1")
        bar = replace(bar, open=value, high=value, low=value, close=value)
        held = [replace(held[0], entry_price=D("90.01"), quantity=1, take_rate=D(1))]
        rules = tuple(replace(rule, budget=D(1000), drop_rate=D(0)) for rule in rules[:2])
        actual = self.assert_candidate((instrument, bar, held, rules), slippage_ticks=1, commission_rate=D(0))
        self.assertEqual((actual.price, actual.reason), (D("90.01"), "buy_condition_gap"))

    def test_decimal80_quantity_rounds_intermediate_gross_and_fee(self):
        instrument, bar, _, rules = input_case("US")
        value = D("1." + "0" * 80 + "1")
        bar = replace(bar, open=value, high=value, low=value, close=value)
        rules = (replace(rules[0], budget=D("2.0002")),)
        actual = self.assert_candidate((instrument, bar, [], rules), first_entry=True, commission_rate=D(".0001"))
        self.assertEqual(actual.quantity, 2)

    def test_raw_buy_commission_is_rounded_after_multiplying_notional(self):
        instrument, bar, _, rules = input_case("US", price="3", spread="0")
        commission = D("0.123456789012345678901234567890123456789012345678901234567890123456789000000000015")
        budget = D("3.3703703670370370367037037036703703703670370370367037037036703703703670000000000")
        rules = (replace(rules[0], budget=budget),)
        actual = self.assert_candidate((instrument, bar, [], rules), first_entry=True, commission_rate=commission)
        self.assertEqual(actual.quantity, 1)

    def test_callers_decimal_rounding_mode_is_preserved(self):
        instrument, bar, _, rules = input_case("US")
        value = D("1." + "0" * 80 + "1")
        bar = replace(bar, open=value, high=value, low=value, close=value)
        rules = (replace(rules[0], budget=D("2.0002")),)
        for mode in ROUNDING_MODES:
            with self.subTest(rounding=mode), localcontext(rounding=mode):
                actual = self.assert_candidate((instrument, bar, [], rules), first_entry=True, commission_rate=D(".0001"))
                self.assertEqual(actual.quantity, 1 if mode in {"ROUND_UP", "ROUND_CEILING", "ROUND_05UP"} else 2)

    def test_decimal_half_tie_follows_each_rounding_mode(self):
        instrument, bar, _, rules = input_case("US")
        value = D("1." + "0" * 79 + "5")
        bar = replace(bar, open=value, high=value, low=value, close=value)
        rules = (replace(rules[0], budget=D("1.0001")),)
        for mode in ROUNDING_MODES:
            with self.subTest(rounding=mode), localcontext(rounding=mode):
                actual = self.assert_candidate((instrument, bar, [], rules), first_entry=True, commission_rate=D(".0001"))
                rejected = mode in {"ROUND_UP", "ROUND_CEILING", "ROUND_05UP", "ROUND_HALF_UP"}
                self.assertEqual(0 if actual is None else actual.quantity, 0 if rejected else 1)

    def test_unused_manual_penetration_has_no_new_upper_bound(self):
        instrument, bar, _, rules = input_case("US")
        self.assert_candidate((instrument, bar, [], rules), first_entry=True,
                              penetration_ticks=10 ** 30, commission_rate=D(0))

    def test_kernel_is_reentrant_and_leaves_reference_untouched(self):
        inputs = input_case("US")
        expected = candidate_for_bar(*inputs, sec_rate_override=D(0))
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: kernel().candidate_for_bar(*inputs, sec_rate_override=D(0)), range(16)))
        self.assertEqual(results, [expected] * 16)
        self.assertEqual(candidate_for_bar(*inputs, sec_rate_override=D(0)), expected)

    def test_source_and_library_fingerprints_are_required(self):
        for field in ("source_sha256", "library_sha256", "abi"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "bundle"
                shutil.copytree(os.environ["MAGICSPLIT_ETF_M3_BUILD"], directory)
                manifest = json.loads((directory / "build.json").read_text())
                manifest[field] = "invalid"
                (directory / "build.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    FastKernel(directory)

    def test_native_error_does_not_silently_fall_back(self):
        with self.assertRaisesRegex(ValueError, "M3 커널 입력/가격 오류"):
            kernel()._compute(b"malformed")

    def test_existing_build_is_not_replaced(self):
        with self.assertRaises(FileExistsError):
            build_kernel(Path(os.environ["MAGICSPLIT_ETF_M3_BUILD"]))

    def test_all_ledgers_match_on_seeded_mixed_market_runs(self):
        rng = random.Random(20261118)
        instruments = [Instrument("KR", "KR", "KRW", "국내주식형"),
                       Instrument("US", "US", "USD", "해외상장"),
                       Instrument("GOLD", "GOLD", "KRW", "KRX금현물")]
        for path in ("O-H-L-C", "O-L-H-C"):
            for monthly in (None, D(".15")):
                bars, evaluations = [], []
                for index in range(12):
                    opening = OPEN + timedelta(days=index)
                    evaluations.append(opening + timedelta(hours=9))
                    for instrument in instruments:
                        price = D(1000 if instrument.market != "US" else 100) * D(rng.randrange(90, 111)) / 100
                        at = opening + timedelta(hours=13 if instrument.market == "US" else 0)
                        bars.append(Bar(instrument.symbol, opening.date(), at, at + timedelta(hours=6),
                                        price, price * D("1.02"), price * D(".98"), price,
                                        D(1000000), adv20=D(1000000000)))
                rules = {i.symbol: tuple(LevelRule(n, D(500), D(".01"), D(".01")) for n in (1, 2, 3))
                         for i in instruments}
                options = dict(initial_cash={"KRW": D(10000000), "USD": D(10000)},
                               fx_rates=[FxRate(OPEN - timedelta(days=1), D(1300))], path=path,
                               evaluation_times=evaluations, conversion_times=[OPEN + timedelta(hours=10)],
                               first_level_fraction=monthly, sec_rate_override=D(0))
                with self.subTest(path=path, monthly=monthly):
                    self.assertEqual(run_fast_engine(kernel(), instruments, bars, rules, **options),
                                     run_engine(instruments, bars, rules, **options))

    def test_complete_synthetic_tax_result_matches_at_settlement_and_year_end(self):
        instrument, original, held, rules = input_case("US")
        bar = replace(original, open=D(200), high=D(200), low=D(200), close=D(200))
        tomorrow = OPEN + timedelta(days=1)
        evaluations = [OPEN + timedelta(hours=21), tomorrow + timedelta(hours=21),
                       datetime(2025, 1, 1, tzinfo=timezone.utc)]
        for path in ("O-H-L-C", "O-L-H-C"):
            options = dict(initial_cash={"KRW": D(0), "USD": D(0)},
                           initial_positions=[replace(held[0], quantity=100)], first_entry=False,
                           fx_rates=[FxRate(OPEN - timedelta(days=1), D(1000))],
                           us_sessions=[OPEN.date(), tomorrow.date()],
                           settlement_session_times={tomorrow.date(): tomorrow + timedelta(hours=20)},
                           evaluation_times=evaluations, path=path, commission_rate=D(0),
                           sec_rate_override=D(0), tax_mode="after_tax", synthetic_tax_data=True)
            expected = run_engine([instrument], [bar], {"ETF": rules}, **options)
            self.assertEqual(expected.valuations[1]["estimated_tax_krw"], D(1650000))
            self.assertEqual(run_fast_engine(kernel(), [instrument], [bar], {"ETF": rules}, **options), expected)


class NativeOracleSuite(unittest.TestSuite):
    def run(self, result, debug=False):
        compiled = kernel()
        with ExitStack() as stack:
            for module in ORACLES:
                if hasattr(module, "candidate_for_bar"):
                    stack.enter_context(patch.object(module, "candidate_for_bar", compiled.candidate_for_bar))
                if hasattr(module, "run_engine"):
                    stack.enter_context(patch.object(module, "run_engine", lambda *a, **kw: run_fast_engine(compiled, *a, **kw)))
            return super().run(result, debug)


ORACLES = [importlib.import_module(name) for name in (
    "test_research_etf_engine_strategy", "test_research_etf_engine_contract", "test_research_etf_engine")]


def load_tests(loader, standard_tests, pattern):
    suite = NativeOracleSuite([standard_tests])
    for module in ORACLES:
        suite.addTests(loader.loadTestsFromModule(module))
    return suite
