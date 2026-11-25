"""다른 시장과 혼합 계좌 조사(markets_mix) 합성 시험 — 설계 v1.1.

기대 수량·금액은 실행 전에 손으로 계산해 적었다(계산식은 각 시험의 주석). 가격은 2025년 국내 호가(2,000원 이상 5원),
수수료는 기본 0.015%·높은 비용 0.30%. 실자료 성과는 계산하지 않는다. 합성 입력 도우미는 승인된
tests.test_research_etf_kodex_core와 tests.test_research_etf_kodex_grid의 것을 그대로 쓴다.
"""
import atexit
import contextlib
from dataclasses import replace
from datetime import date
from decimal import Decimal, localcontext
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.research.etf.kodex_core import data as loader
from src.research.etf.kodex_core import formal, study
from src.research.etf.kodex_core.config import _plain
from src.research.etf.kodex_core.inputs import DataHold
from src.research.etf.kodex_core.metrics import plain
from src.research.etf.kodex_grid import identity as grid_identity
from src.research.etf.kodex_grid import sealing
from src.research.etf.kodex_grid.grid import load_grid, setting_config
from src.research.etf.kodex_grid.identity import IdentityError
from src.research.etf.kodex_grid.judge import reasons_for
from src.research.etf.kodex_grid.permit import PermitError
from src.research.etf.kodex_grid.records import AssemblyError
from src.research.etf.kodex_grid.runner import Bench, run_point
from src.research.etf.markets_mix import (baseline, batch, cli, eligibility, identity, judge, mixing, permit, records,
                                          report, rows, summary)
from src.research.etf.markets_mix.definition import (SHARE, SOLO, Cohort, Mix, Period, Unit, check_definition,
                                                     load_definition, period_labels, plan_units, reference_unit,
                                                     unit_config)
from src.research.etf.markets_mix.identity import MixSource, open_context, open_synthetic_context
from src.research.etf.markets_mix.unit import cost_flags, open_unit, run_unit, unit_result
from tests.test_research_etf_kodex_core import BASIS, FROZEN, J, SYMBOL, as_symbol, build, synthetic_bundle, weekdays
from tests.test_research_etf_kodex_grid import (PROBE, WORLD_BASE, WORLD_DAYS, WORLD_PRICES, WORLD_SEGMENTS,
                                                WORLD_SPEC, MemorySource, _Workspace, bar, case2, cfg, estimate,
                                                year_boundary)


D = Decimal
SPEC = load_grid()
MIX = load_definition()
PATHS = FROZEN.paths
OHLC, OLHC = PATHS
PAIRS = tuple(f"{a}|{b}" for a in PATHS for b in PATHS)
BASE, STRESS = "base", "stress"
KOSDAQ, CHINA, NASDAQ, BOND = "229200", "169950", "133690", "136340"
WORLD_PERIODS = tuple(Period(f"P{index}", row.start, row.end, row.label)
                      for index, row in enumerate(WORLD_SEGMENTS, start=1))
FIRST, SECOND = MIX.cohorts


def cohort(name, start, solo, mixes, sensitivity=None, rule="prespecified"):
    """합성 달력의 한 묶음. 민감도 시작일은 주 시작일의 +5·+20 거래일."""
    index = WORLD_DAYS.index(start)
    days = sensitivity or (WORLD_DAYS[index + 5], WORLD_DAYS[index + 20])
    return Cohort(name, start, days, tuple(solo), tuple(Mix("+".join(legs), tuple(legs)) for legs in mixes),
                  WORLD_PERIODS, rule)


def world_mix(*cohorts, products=None):
    counts = {row.id: row.subjects() * 36 for row in cohorts}
    symbols = products or [symbol for symbol, _ in MIX.products]
    return replace(MIX, products=tuple((symbol, symbol) for symbol in symbols), cohorts=cohorts, end=WORLD_DAYS[-1],
                   rows_expected={**counts, "total": sum(counts.values())})


# 두 묶음: A(1/2 시작)는 단독 셋과 혼합 하나, B(1/3 시작)는 단독 하나와 기준 상품을 담은 혼합 하나 — 실제 정의와 같은 꼴
WORLD_MIX = world_mix(cohort("A", J(2), (SYMBOL, "229200", "169950"), [("229200", "169950")]),
                      cohort("B", WORLD_DAYS[1], ("133690",), [(SYMBOL, "133690")]))
# 상품별 합성 가격: (기준 가격에 더하는 값, 하루 기울기). 모두 100원 단위라 호가에 맞는다.
SHAPES = {SYMBOL: (0, 0), KOSDAQ: (0, 20), CHINA: (0, -20), NASDAQ: (1000, 50)}
RISING = {symbol: (0, 100) for symbol in SHAPES}
LOG_LINE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ run (solo|share)-\d{6}-\d{4}-\d\d-\d\d-(base|stress)-setting-\d\d "
                      r"(done|skipped|error) \d+\.\ds( E_[A-Z]+)?$")
RESULT_WORDS = ("inferior", "counterexample", "tie", "path_dependent", "blank", "열위", "반례", "동률", "빈칸",
                "cagr", "delta", "equity")


def market(symbol, shape, sigma="0.01", changes=None):
    shift, slope = shape
    prices = {day: str(price + shift + slope * index) for index, (day, price) in enumerate(zip(WORLD_DAYS, WORLD_PRICES))}
    data = build(WORLD_DAYS, prices, sigma=sigma, basis=BASIS, changes=changes)
    return data if symbol == SYMBOL else as_symbol(data, symbol)


def world_inputs(shapes=SHAPES, held=(), mix=None, **overrides):
    """(상품, 시작일)마다 그 상품의 합성 입력. held의 상품은 예상한 자료 문제(DataHold), overrides는 상품별 입력."""
    inputs = {}
    for symbol, shape in shapes.items():
        data = overrides.get(f"s{symbol}") or market(symbol, shape)
        for start in (mix or WORLD_MIX).starts():
            inputs[(symbol, start)] = DataHold("synthetic: no input for this product") if symbol in held else data
    return inputs


def world_context(mix=WORLD_MIX, name="mix-world", **options):
    inputs = world_inputs(mix=mix, **options)
    return open_synthetic_context(MemorySource(inputs, name), mix=mix, spec=WORLD_SPEC, base=WORLD_BASE)


def config_of(kind, number, mix=MIX, base=FROZEN, spec=SPEC, symbol=SYMBOL):
    return unit_config(mix, Unit(kind, symbol, mix.cohorts[0].start, number, BASE), base=base, grid=spec)


def flat(prices, days=None, **options):
    days = days or weekdays(J(2), J(8))
    return build(days, prices, basis=BASIS, **options)


def sealed(folder):
    return {path.name: json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(folder, "units").glob("*.json"))}


def reseal(run_folder, name, change):
    """단위 파일을 고치고 봉인 기록의 해시까지 맞춰 고친다(해시 검사를 지나 기록 검사에 닿게 하는 시험용)."""
    path = Path(run_folder, "units", name)
    record = json.loads(path.read_text(encoding="utf-8"))
    change(record)
    text = batch.compact(record)
    path.write_text(text, encoding="utf-8")
    seal_path = Path(run_folder, "sealed", "seals", name)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal_path.write_text(sealing.dump(sealing.with_seal({**seal, "sha256": sealing.digest(text)})), encoding="utf-8")


# ---------------------------------------------------------------------------
# 정의 파일과 계획한 단위(설계 2~4절)
# ---------------------------------------------------------------------------
class Definition(unittest.TestCase):
    def test_cohorts_products_starts_and_costs_are_the_design(self):
        self.assertEqual(MIX.symbols(), (SYMBOL, KOSDAQ, CHINA, NASDAQ))  # 136340과 그 혼합은 뺐다(설계 v2 2절)
        self.assertEqual([(row.id, row.start, row.sensitivity_starts, row.solo, [(m.id, m.legs) for m in row.mixes])
                          for row in MIX.cohorts],
                         [("2017", date(2017, 1, 2), (date(2017, 1, 9), date(2017, 2, 1)), (SYMBOL, KOSDAQ, CHINA),
                           [("229200+169950", (KOSDAQ, CHINA))]),
                          ("2021", date(2021, 1, 4), (date(2021, 1, 11), date(2021, 2, 1)), (NASDAQ,),
                           [("069500+133690", (SYMBOL, NASDAQ))])])
        self.assertEqual((MIX.end, MIX.mode, MIX.sensitivity["number"], SPEC.baseline_number(), MIX.reference),
                         (date(2026, 11, 13), "M", 28, 28, SYMBOL))
        self.assertEqual(MIX.costs, {"primary": BASE, "one_way": {BASE: D("0.00015"), STRESS: D("0.003")}})
        self.assertEqual([(row.label, row.start, row.end) for row in FIRST.periods],
                         [("2017-2019", date(2017, 1, 1), date(2019, 12, 31)), ("2020-2023", date(2020, 1, 1), date(2023, 12, 31)),
                          ("2024-2026-11-13", date(2024, 1, 1), date(2026, 11, 13))])
        self.assertEqual([(row.label, row.start, row.end) for row in SECOND.periods],
                         [("2021-2023", date(2021, 1, 1), date(2023, 12, 31)), ("2024-2026-11-13", date(2024, 1, 1), date(2026, 11, 13))])
        self.assertEqual(MIX.rows_expected, {"2017": 144, "2021": 72, "total": 216})
        self.assertEqual(SECOND.start_rule["symbol"], NASDAQ)
        check_definition(MIX, SPEC, FROZEN)

    def test_plan_is_334_units_without_duplicates(self):
        """2017: 단독 3 × (36 + 28번 5자리) + 몫 2 × 41 = 205. 2021: 단독 1 × 41 + 몫 2 × 41 + 기준 보유 6 = 129."""
        units = plan_units(MIX, SPEC)
        self.assertEqual((len(units), len(set(units)), len({unit.name for unit in units})), (334, 334, 334))
        by_start = {start: [u for u in units if u.start == start] for start in MIX.starts()}
        self.assertEqual({start: len(found) for start, found in by_start.items()},
                         {date(2017, 1, 2): 5 * 36 + 5 * 1, date(2017, 1, 9): 10, date(2017, 2, 1): 10,
                          date(2021, 1, 4): 3 * 36 + 3 * 1 + 2, date(2021, 1, 11): 8, date(2021, 2, 1): 8})
        primary = [u for u in units if (u.start, u.cost) in ((FIRST.start, BASE), (SECOND.start, BASE))]
        extra = [u for u in units if u not in primary]
        self.assertEqual({u.number for u in extra}, {28})  # 민감도는 28번만
        self.assertEqual({(u.kind, u.symbol) for u in units if u.start == SECOND.start},
                         {(SOLO, NASDAQ), (SHARE, NASDAQ), (SHARE, SYMBOL), (SOLO, SYMBOL)})
        reference = [u for u in units if u.start in SECOND.starts() and (u.kind, u.symbol) == (SOLO, SYMBOL)]
        self.assertEqual(sorted(u.name for u in reference),  # 2021 KODEX 200 1억 보유는 병기 기준일 뿐 판정 행이 아니다
                         sorted(reference_unit(MIX, start, cost).name for start, cost in MIX.slots(SECOND)))
        self.assertNotIn(BOND, {u.symbol for u in units})
        self.assertEqual(MIX.cohort_of(date(2021, 2, 1)).id, "2021")
        with self.assertRaisesRegex(ValueError, "exactly one cohort"):
            MIX.cohort_of(date(2018, 1, 2))

    def test_share_is_the_grid_setting_with_only_the_capital_changed(self):
        """5천만 원 몫 = kodex_grid의 그 설정에서 원금만 바뀐 설정. 단독(1억 원)은 kodex_grid의 설정 그대로다."""
        for row in SPEC.settings():
            grid = setting_config(FROZEN, SPEC, row)
            solo, share = config_of(SOLO, row.number), config_of(SHARE, row.number)
            self.assertEqual(solo.applied_sha256(), grid.applied_sha256())
            fixed = _plain({name: getattr(grid, name) for name in grid.__dataclass_fields__})
            changed = {name for name, value in fixed.items() if _plain(getattr(share, name)) != value}
            self.assertEqual(changed, {"capital"})
            self.assertEqual(share.capital["core"] + share.capital["split"], D(50000000))
            self.assertEqual(share.capital["hold"], D(50000000))
            self.assertEqual({name: value * 2 for name, value in share.capital.items()}, dict(grid.capital))
        self.assertEqual(config_of(SHARE, 1).capital, {"core": D(25000000), "split": D(25000000), "hold": D(50000000)})
        self.assertEqual(config_of(SHARE, 28).capital, {"core": D(40000000), "split": D(10000000), "hold": D(50000000)})
        self.assertEqual(config_of(SOLO, 28).applied_sha256(), MIX.approved["baseline_applied_config_sha256"])
        self.assertEqual(len({config_of(kind, row.number).applied_sha256() for kind in (SOLO, SHARE)
                              for row in SPEC.settings()}), 72)

    def test_nominal_floor_is_the_account_except_for_the_reference_product(self):
        start = FIRST.start
        self.assertIsNone(MIX.floor(Unit(SOLO, SYMBOL, start, 1, BASE)))
        self.assertIsNone(MIX.floor(Unit(SHARE, SYMBOL, start, 1, BASE)))
        self.assertEqual(MIX.floor(Unit(SOLO, NASDAQ, start, 1, BASE)), D(100000000))
        self.assertEqual(MIX.floor(Unit(SHARE, NASDAQ, start, 1, BASE)), D(50000000))

    def test_periods_are_the_base_segments_cut_at_the_real_start(self):
        first = (("2014-2019", "2017-2019"), ("2020-2023", "2020-2023"), ("2024-2026-11-13", "2024-2026-11-13"))
        second = (("2020-2023", "2021-2023"), ("2024-2026-11-13", "2024-2026-11-13"))
        for start in FIRST.starts():
            self.assertEqual(period_labels(MIX, FROZEN, start), first)
        for start in SECOND.starts():
            self.assertEqual(period_labels(MIX, FROZEN, start), second)
        with self.assertRaisesRegex(ValueError, "exactly one cohort"):
            period_labels(MIX, FROZEN, date(2011, 1, 3))

        def with_periods(row, periods):
            return replace(MIX, cohorts=(replace(FIRST, periods=periods) if row == 0 else FIRST,
                                         replace(SECOND, periods=periods) if row == 1 else SECOND))
        with self.assertRaisesRegex(ValueError, "does not cover exactly"):
            period_labels(with_periods(0, FIRST.periods[:2]), FROZEN, FIRST.start)  # 마지막 구간을 빠뜨린 정의
        short = (replace(SECOND.periods[0], end=date(2022, 12, 31)), SECOND.periods[1])
        with self.assertRaisesRegex(ValueError, "is not the window"):
            period_labels(with_periods(1, short), FROZEN, SECOND.start)  # 2021 묶음 구간의 끝이 고정 구간과 다르다
        moved = with_periods(0, (replace(FIRST.periods[0], end=date(2018, 12, 31)), *FIRST.periods[1:]))
        with self.assertRaisesRegex(ValueError, "is not the window"):
            period_labels(moved, FROZEN, FIRST.start)

    def test_changed_declarations_are_rejected_or_identified(self):
        half = {**MIX.capital, SHARE: {**MIX.capital[SHARE], "account": D(40000000)}}
        cases = ((replace(MIX, capital=half), r"core \+ split must equal"),
                 (replace(MIX, costs={**MIX.costs, "one_way": {BASE: D("0.0002"), STRESS: D("0.003")}}), "one-way costs"),
                 (replace(MIX, rows_expected={"2017": 144, "2021": 72, "total": 215}), "declared row counts"),
                 (replace(MIX, rows_expected={"2017": 144, "2021": 108, "total": 252}), "declared row counts"),
                 (replace(MIX, cohorts=(FIRST, replace(SECOND, mixes=(Mix("x", (KOSDAQ, BOND)),)))), "declared products"),
                 (replace(MIX, cohorts=(FIRST, replace(SECOND, start=FIRST.start))), "starts must be unique"),
                 (replace(MIX, cohorts=(FIRST, replace(SECOND, solo=(CHINA,)))), "one cohort only"),
                 (replace(MIX, cohorts=(FIRST, replace(SECOND, sensitivity_starts=SECOND.sensitivity_starts[:1]))),
                  "one sensitivity start per declared offset"),
                 (replace(MIX, end=date(2026, 11, 12)), "end or reentry mode"),
                 (replace(MIX, sensitivity={**MIX.sensitivity, "number": 27}), "sensitivity setting number"),
                 (replace(MIX, reference="000000"), "reference product"))
        for changed, message in cases:
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                check_definition(changed, SPEC, FROZEN)
        other = replace(MIX, thresholds={**MIX.thresholds, "regime_band": D("0.02")})
        self.assertEqual(other.sha256, MIX.sha256)  # 파일 해시는 그대로지만
        self.assertNotEqual(other.applied_sha256(), MIX.applied_sha256())  # 적용 정의 해시로 구별된다
        raw = json.loads(Path(identity.__file__).with_name("markets_mix_v2.json").read_text())
        with self.subTest("formal flag"), contextlib.ExitStack() as stack:
            folder = stack.enter_context(tempfile.TemporaryDirectory())
            path = Path(folder) / "mix.json"
            path.write_text(json.dumps({**raw, "formal_run": True}))
            with self.assertRaisesRegex(ValueError, "formal flag"):
                load_definition(path)

    def test_registered_hashes_are_the_approved_versions(self):
        self.assertEqual(MIX.approved["source_sha256"], loader.source_identity()["source_sha256"])
        self.assertEqual(MIX.approved["kodex_grid_source_sha256"],
                         grid_identity.exploration_source()["exploration_source_sha256"])
        self.assertEqual((MIX.approved["grid_file_sha256"], MIX.approved["formal_result_sha256"]),
                         (SPEC.sha256, SPEC.approved["formal_result_sha256"]))


# ---------------------------------------------------------------------------
# 판정(설계 6절) — 순수 함수
# ---------------------------------------------------------------------------
def verdict(*values, required=None):
    keys = required or (PATHS if len(values) == 2 else PAIRS)
    deltas = {key: None if value is None else D(value) for key, value in zip(keys, values)}
    return judge.row_verdict(deltas, keys)


def statuses(**counts):
    """상태별 행 수로 만든 {행 열쇠: 상태}."""
    names = [name for name, count in counts.items() for _ in range(count)]
    return {f"row-{index:03d}": name for index, name in enumerate(names)}


class RowStatus(unittest.TestCase):
    """필수 경로가 하나라도 빠지면 빈칸. 모두 있으면 전부 음수=열위, 전부 양수=반례, 양수와 음수가 함께=경로 의존, 나머지=동률."""

    def test_five_states_for_solo_rows(self):
        cases = ((("-0.01", "-0.0001"), judge.INFERIOR, ["-", "-"]), (("0.01", "1E-70"), judge.COUNTER, ["+", "+"]),
                 (("0.01", "-0.01"), judge.PATH_DEPENDENT, ["+", "-"]), (("0", "0"), judge.TIE, ["0", "0"]),
                 (("0", "-0.01"), judge.TIE, ["0", "-"]), (("0.01", "0"), judge.TIE, ["+", "0"]),
                 ((None, "-0.01"), judge.BLANK, [None, "-"]), ((None, None), judge.BLANK, [None, None]),
                 ((None, "0.5"), judge.BLANK, [None, "+"]))
        for values, status, signs in cases:
            with self.subTest(values):
                found = verdict(*values)
                self.assertEqual((found["status"], list(found["signs"].values())), (status, signs))
        self.assertEqual(list(verdict("1", "1")["signs"]), list(PATHS))  # 경로별 부호도 남긴다

    def test_mix_rows_need_all_four_path_pairs(self):
        self.assertEqual(verdict("-1", "-1", "-1", "-1")["status"], judge.INFERIOR)
        self.assertEqual(verdict("1", "1", "1", "1")["status"], judge.COUNTER)
        self.assertEqual(verdict("1", "1", "1", "-1")["status"], judge.PATH_DEPENDENT)
        self.assertEqual(verdict("1", "0", "-1", "-1")["status"], judge.PATH_DEPENDENT)  # 양수와 음수가 함께 있으면 0이 있어도
        self.assertEqual(verdict("-1", "-1", "0", "-1")["status"], judge.TIE)
        self.assertEqual(verdict("1", "1", "1", None)["status"], judge.BLANK)  # 세 짝이 양수여도 한 짝이 빠지면 빈칸
        with self.assertRaisesRegex(ValueError, "exactly its required paths"):
            judge.row_verdict({PAIRS[0]: D(1), PAIRS[1]: D(1)}, PAIRS)  # 빠진 경로를 조용히 넘기지 않는다
        with self.assertRaisesRegex(ValueError, "exactly its required paths"):
            judge.row_verdict({**{pair: D(1) for pair in PAIRS}, "extra": D(1)}, PAIRS)

    def test_small_differences_are_not_rounded(self):
        tiny = "0." + "0" * 70 + "1"
        self.assertEqual(verdict(tiny, tiny)["status"], judge.COUNTER)  # 작은 차이도 반례다
        self.assertEqual(verdict("-" + tiny, "-" + tiny)["status"], judge.INFERIOR)


class ConclusionSentences(unittest.TestCase):
    def test_three_sentences(self):
        sentences = MIX.sentences
        every = statuses(inferior=216)
        done = judge.conclusion(every, list(every), sentences)
        self.assertEqual((done["kind"], done["all_claimed"], done["rows"]), (judge.ALL_INFERIOR, True, 216))
        self.assertEqual(done["sentence"], "2017년에 시작한 네 대상과 2021년에 시작한 두 대상에서 36개 설정 모두 같은 상품·배분을 "
                                           "계속 보유했을 때보다 연복리 수익률이 낮았다.")
        self.assertIn("216행에 한정한다", done["scope"])
        one = statuses(inferior=213, counterexample=3)
        counters = (("2017-01-02", CHINA, 2), ("2021-01-04", "069500+133690", 1))
        found = judge.conclusion(one, list(one), sentences, counters=counters)
        self.assertEqual((found["kind"], found["all_claimed"]), (judge.HAS_COUNTER, False))
        self.assertEqual(found["sentence"], "모두 덜 벌었다는 주장은 성립하지 않는다. 2017-01-02 시작 169950에서 2개, "
                                            "2021-01-04 시작 069500+133690에서 1개가 같은 배분 보유를 앞섰다.")
        with self.assertRaisesRegex(ValueError, "add up to the counterexample rows"):
            judge.conclusion(one, list(one), sentences, counters=counters[:1])
        mixed = statuses(inferior=208, tie=3, path_dependent=2, blank=3)
        partial = judge.conclusion(mixed, list(mixed), sentences)
        self.assertEqual((partial["kind"], partial["all_claimed"]), (judge.PARTIAL, False))
        self.assertEqual(partial["sentence"], "216행 중 208행에서 덜 벌었다. 나머지는 동률 3행, 경로에 따라 부호가 갈린 경우 2행, "
                                              "비교 불가 3행이다.")
        self.assertNotIn("모두", partial["sentence"])
        # 반례가 하나라도 있으면 빈칸·동률이 함께 있어도 둘째 문장이다
        both = statuses(inferior=128, counterexample=2, blank=86)
        self.assertEqual(judge.conclusion(both, list(both), sentences, counters=(("d", "s", 2),))["kind"], judge.HAS_COUNTER)

    def test_one_tie_blank_or_path_dependent_row_withdraws_all(self):
        for name in (judge.TIE, judge.BLANK, judge.PATH_DEPENDENT):
            found = statuses(inferior=215, **{name: 1})
            self.assertEqual(judge.conclusion(found, list(found), MIX.sentences)["kind"], judge.PARTIAL)

    def test_missing_extra_or_duplicate_rows_are_refused(self):
        every = statuses(inferior=216)
        keys = list(every)
        for rows_found, expected in (({key: every[key] for key in keys[:-1]}, keys), (every, keys[:-1]),
                                     (every, keys + keys[:1]), ({}, [])):
            with self.assertRaisesRegex(ValueError, "exactly once"):
                judge.conclusion(rows_found, expected, MIX.sentences)

    def test_planned_primary_rows_are_144_plus_72(self):
        """묶음별 주 판정 행 열쇠: 2017 묶음 4대상 × 36, 2021 묶음 2대상 × 36. 민감도 행은 들어가지 않는다."""
        ctx = open_synthetic_context(MemorySource({}, "frame"), mix=MIX)
        keys = summary.expected_keys(ctx)
        self.assertEqual((len(keys), len(set(keys))), (216, 216))
        self.assertEqual(sum(key.endswith("@2017-01-02/base") for key in keys), 144)
        self.assertEqual(sum(key.endswith("@2021-01-04/base") for key in keys), 72)
        self.assertEqual({key.split("/")[0] for key in keys if "2021" in key}, {NASDAQ, "069500+133690"})
        self.assertNotIn("069500/01@2021-01-04/base", keys)  # 2021 KODEX 200 단독은 판정 행이 아니다
        statuses_found = {key: judge.INFERIOR for key in keys}
        for broken in ({key: statuses_found[key] for key in keys[:-1]},
                       {**statuses_found, "069500/28@2017-01-09/base": judge.INFERIOR}):
            with self.assertRaisesRegex(ValueError, "exactly once"):
                judge.conclusion(broken, keys, MIX.sentences)


class SideNotes(unittest.TestCase):
    """병기 항목과 28번 민감도. 판정을 바꾸지 않는다."""

    def test_own_hold_regime_boundaries(self):
        band = MIX.thresholds["regime_band"]
        cases = (("-0.0100001", judge.DOWN), ("-0.01", judge.FLAT), ("0", judge.FLAT), ("0.01", judge.FLAT),
                 ("0.0100001", judge.UP), ("-0.5", judge.DOWN), ("0.5", judge.UP))
        for value, expected in cases:
            self.assertEqual(judge.regime(D(value), band), expected, value)
        self.assertIsNone(judge.regime(None, band))

    def test_region_points_at_zero_and_one_percent_point(self):
        def meets(strict, *values):
            threshold = D(0) if strict else D("0.01")
            return judge.point_meets({index: None if value is None else D(value) for index, value in enumerate(values)},
                                     threshold, strict=strict)
        self.assertEqual([meets(True, "0.001", "0.002"), meets(True, "0", "0.002"), meets(True, None, "0.002"),
                          meets(True, None, "-0.002"), meets(True, None, None)], [True, False, None, False, None])
        # +1%p는 같아도 충족(문턱 포함), 0은 초과여야 한다
        self.assertEqual([meets(False, "0.01", "0.02"), meets(False, "0.0099999", "0.02"), meets(False, None, "0.01")],
                         [True, False, None])
        regions = [region.describe() for region in SPEC.regions()]
        table = judge.regions_table(regions, {n: n in SPEC.regions()[0].numbers for n in range(1, 37)})
        self.assertEqual([row["status"] for row in table][:2], [judge.MET, judge.NOT_MET])
        self.assertEqual(table[0]["met"], 8)
        waiting = judge.regions_table(regions, {n: None if n == 28 else True for n in range(1, 37)})
        self.assertEqual({row["rank"]: row["status"] for row in waiting}[1], judge.OPEN)
        self.assertEqual({row["rank"]: row["status"] for row in waiting}[5], judge.MET)
        with self.assertRaises(KeyError):
            judge.regions_table(regions, {n: True for n in range(1, 36)})

    def test_start_and_cost_dependence_per_path(self):
        """같은 경로·비용에서 시작일에 따라, 같은 경로·시작일에서 비용에 따라 부호가 다르면 의존. 없는 값은 표시한다."""
        starts = ("2017-01-02", "2017-01-09", "2017-02-01")
        table = {(start, cost): {OHLC: "-", OLHC: "-"} for start in starts for cost in (BASE, STRESS)}
        same = judge.sensitivity(table, PATHS)
        self.assertEqual((same["start_dependent"], same["cost_dependent"]), (judge.NO, judge.NO))
        self.assertEqual({row["mark"] for row in same["start"].values()}, {judge.SAME})
        table[("2017-01-09", BASE)] = {OHLC: "+", OLHC: "-"}
        moved = judge.sensitivity(table, PATHS)
        self.assertEqual((moved["start_dependent"], moved["cost_dependent"]), (judge.YES, judge.YES))
        self.assertEqual(moved["start"][f"{OHLC}@{BASE}"]["mark"], judge.DEPENDENT)
        self.assertEqual(moved["start"][f"{OLHC}@{BASE}"]["mark"], judge.SAME)  # 다른 경로의 부호와 섞지 않는다
        self.assertEqual(moved["start"][f"{OHLC}@{STRESS}"]["mark"], judge.SAME)
        self.assertEqual(moved["cost"][f"{OHLC}@2017-01-09"]["mark"], judge.DEPENDENT)
        self.assertEqual(moved["cost"][f"{OHLC}@2017-01-02"]["mark"], judge.SAME)
        table[("2017-01-09", BASE)] = {OHLC: None, OLHC: "-"}
        missing = judge.sensitivity(table, PATHS)
        row = missing["start"][f"{OHLC}@{BASE}"]
        self.assertEqual((row["mark"], row["missing"]), (judge.OPEN, ["2017-01-09"]))  # 비교값이 없으면 미확정으로 남긴다
        self.assertEqual((missing["start_dependent"], missing["cost_dependent"]), (judge.OPEN, judge.OPEN))
        self.assertEqual(judge.dependence({"a": "+", "b": None, "c": "-"})["mark"], judge.DEPENDENT)
        self.assertEqual(judge.dependence({"a": "0", "b": "-"})["mark"], judge.DEPENDENT)  # 0과 음수도 다른 부호다

    def test_review_threshold_needs_every_path_at_both_costs(self):
        def review(base, high):
            pack = lambda values: {index: None if value is None else D(value) for index, value in enumerate(values)}
            return judge.review_threshold(pack(base), pack(high), MIX.thresholds)
        self.assertEqual(review(("0.01", "0.02"), ("0.001", "1E-30")), judge.MET)  # 기본 +1%p는 같아도 충족
        self.assertEqual(review(("0.0099", "0.02"), ("0.001", "0.001")), judge.NOT_MET)
        self.assertEqual(review(("0.01", "0.02"), ("0", "0.001")), judge.NOT_MET)  # 높은 비용 차 0은 양수가 아니다
        self.assertEqual(review(("0.01", "0.02"), (None, "0.001")), judge.OPEN)
        self.assertEqual(review(("0.01", None), ("-0.1", "0.001")), judge.NOT_MET)  # 이미 어긋난 값이 있으면 미충족
        self.assertEqual(review(("0.02",) * 4, ("0.01",) * 4), judge.MET)  # 혼합은 네 경로 짝 모두
        self.assertEqual(review(("0.02", "0.02", "0.02", "0.009"), ("0.01",) * 4), judge.NOT_MET)


# ---------------------------------------------------------------------------
# 단위 실행기: 승인 코드와의 일치, 빈칸과 사유, 일별 계열
# ---------------------------------------------------------------------------
def with_gap(data, last):
    """개설 뒤 일봉 결측으로 관측이 끊긴 입력(그날까지만 유효)."""
    bars = tuple(row for row in data.bars if row.session_date <= last)
    gap = {"last_valid_session": last, "first_bad_session": data.calendar.next_session(last),
           "reason": "synthetic gap", "requested_end": data.end}
    return replace(data, bars=bars, data_end=gap)


def approved_cases():
    """(이름, 입력, 설정, 시작일): 정상, 재진입, 연초 매수 닫힘, 높은 비용 개설 거부, 최초 분할 매수 불가, 단독 보유 첫 매수
    거절, 개설 뒤 결측."""
    normal, base28 = market(SYMBOL, SHAPES[SYMBOL]), cfg("80", "2", "3", 9)
    days = weekdays(J(2), J(10))
    return (("normal", normal, cfg("80", "2", "3", 9, WORLD_BASE), J(2)),
            ("reentry", case2(), base28, J(2)),
            ("closed-year", year_boundary(), cfg("50", "1", "1", 5), date(2024, 12, 2)),
            ("stress-rejected", flat({J(2): "10000"}, sigma="0.002"), base28, J(2)),
            ("split-start-failed", flat({J(2): "3000000", J(3): "3100000"}, days), base28, J(2)),
            ("hold-first-buy", flat({J(2): "10000"}, changes={J(2): {"adv20": D(1644000000)}}), base28, J(2)),
            ("gap", with_gap(flat({J(2): "10000", J(6): "10200"}, days), J(7)), base28, J(2)))


class UnitAgainstApprovedCode(unittest.TestCase):
    """비용 하나의 단위가 승인 코드(kodex_grid.run_point → study._path_result)의 같은 비용 값과 같다."""

    def test_each_cost_equals_the_approved_run(self):
        seen = set()
        for name, data, config, start in approved_cases():
            approved = plain(run_point(data, config, start))
            rows_found, extras = approved["run_start"]["items"]["M"]["paths"], approved["exploration"]
            for cost in (BASE, STRESS):
                mine = plain(run_unit(data, config, start, cost=cost))
                self.assertEqual((mine["cost"], mine["cost_rate"], mine["applied_config_sha256"]),
                                 (cost, str(config.cost(cost)), config.applied_sha256()))
                for path in PATHS:
                    with self.subTest(name=name, cost=cost, path=path):
                        seen.add(self.same_slot(mine["paths"][path], rows_found[path], extras[path], cost))
        self.assertEqual(seen, {True, False})  # 비교 가능한 자리와 빈칸인 자리를 모두 지났다

    def same_slot(self, mine, formal, extra, cost) -> bool:
        reasons = {key: reasons_for(extra["flags"][key], cost) for key in ("failures", "holds", "later")}
        valid = formal[cost] is not None and not any(reasons.values())
        self.assertEqual((mine["reasons"], mine["comparable"]), (reasons, valid))
        if valid:
            self.assertEqual((mine["strategy"], mine["hold"]), (formal[cost]["strategy"], formal[cost]["hold"]))
            self.assertEqual(mine["delta"], {key: formal[cost][key] for key in ("delta_cagr", "delta_abs_mdd", "exposure_mean")})
        else:  # 비교할 수 없으면 승인 코드에 수치가 있어도 싣지 않는다
            self.assertEqual((mine["strategy"], mine["hold"], mine["delta"]), (None, None, None))
            self.assertIsNone((mine["series"] or {}).get("strategy"))
        self.assertEqual(mine["hold_only"], formal["hold_metrics"][cost])
        self.assertEqual(mine["start_orders"], formal["start_orders"][cost])
        self.assertEqual(mine["split_trades"], extra["split_ledger"][cost])
        self.assertEqual(mine["reentry_counts"], extra["reentry_counts"][cost])
        self.assertEqual(mine["buys_closed_years"], formal["buys_closed_years"][cost])
        return valid

    def test_reasons_of_every_case(self):
        """사유는 그 비용의 것만 가진다(다른 비용의 거부를 끌어오지 않는다)."""
        found = {}
        for name, data, config, start in approved_cases():
            for cost in (BASE, STRESS):
                row = run_unit(data, config, start, cost=cost)["paths"][OHLC]
                found[(name, cost)] = [reason for key in ("failures", "holds", "later") for reason in row["reasons"][key]]
        self.assertEqual(found[("normal", BASE)], [])
        self.assertEqual((found[("stress-rejected", BASE)], found[("stress-rejected", STRESS)]),
                         ([], ["stress_start_rejected_by_cost_guard"]))
        self.assertEqual(found[("split-start-failed", BASE)], ["base_split_start_failed"])
        self.assertEqual(found[("hold-first-buy", STRESS)], ["stress_hold_first_buy_failed"])
        self.assertEqual(found[("gap", BASE)], ["input_data_gap_after_start"])

    def test_flags_equal_the_approved_flags_restricted_to_one_cost(self):
        for name, data, config, start in approved_cases():
            runs = {cost: open_unit(data, config, start, cost=cost) for cost in (BASE, STRESS)}
            for path in PATHS:
                group = {cost: runs[cost].groups[path] for cost in runs}
                failures, holds, later, _ = study._flags(runs[BASE].job.data, config, group, "M")
                for cost in (BASE, STRESS):
                    with self.subTest(name=name, cost=cost, path=path):
                        self.assertEqual(cost_flags(runs[cost].job, group[cost], cost),
                                         {"failures": reasons_for(failures, cost), "holds": reasons_for(holds, cost),
                                          "later": reasons_for(later, cost)})

    def test_a_reason_blanks_the_values_even_when_the_ledgers_have_them(self):
        """합계 시작 한도 보류: 세 장부가 만기까지 성립해 승인 코드가 수치를 만들어도, 그 비용의 사유가 있으면 비교하지 않는다."""
        data, config = flat({J(2): "10000"}), cfg("80", "2", "3", 9)
        with patch.object(study, "_start_liquidity", return_value=False):
            row = plain(run_unit(data, config, J(2), cost=BASE))["paths"][OHLC]
            approved = plain(run_point(data, config, J(2)))["run_start"]["items"]["M"]["paths"][OHLC]
        self.assertIsNotNone(approved["base"])
        self.assertEqual((row["reasons"]["holds"], row["comparable"], row["strategy"], row["delta"], row["series"]["strategy"]),
                         (["base_combined_start_liquidity"], False, None, None, None))
        self.assertIsNotNone(row["series"]["hold"])  # 보유 계좌의 계열은 남는다(혼합 보유 쪽은 판정에 쓰지 않는다)

    def test_unit_takes_one_mode_and_rechecks_volatility(self):
        data = flat({J(2): "10000"})
        with self.assertRaisesRegex(ValueError, "exactly one reentry mode"):
            run_unit(data, FROZEN, J(2), cost=BASE)
        late = replace(data, volatility={J(2): estimate(J(3), "0.01")})  # 시가 뒤에 알려진 값
        with self.assertRaises(study.InputIdentityError):
            run_unit(late, cfg("80", "2", "3", 9), J(2), cost=BASE)


class NominalFloorPerAccount(unittest.TestCase):
    """기준 상품 외에는 시작일의 5% 한도가 그 계좌의 원금 이상이어야 한다(같으면 허용). 미달이면 장부를 열지 않는다."""

    def unit(self, adv20, kind, symbol=NASDAQ):
        data = as_symbol(flat({J(2): "10000"}, changes={J(2): {"adv20": D(adv20)}}), symbol)
        unit = Unit(kind, symbol, J(2), 28, BASE)
        return plain(run_unit(data, config_of(kind, 28), J(2), cost=BASE, nominal_floor=MIX.floor(unit)))

    def test_equal_is_allowed_and_one_won_short_is_not(self):
        # 5% 한도 = adv20 × 0.05. 1,000,000,000 → 5천만(같음), 999,999,980 → 49,999,999원
        equal, short = self.unit(1000000000, SHARE), self.unit(999999980, SHARE)
        self.assertEqual((equal["head"]["capacity"]["ok"], equal["head"]["capacity"]["floor_krw"]), (True, "50000000"))
        self.assertTrue(all(row["comparable"] for row in equal["paths"].values()))
        self.assertEqual((short["head"]["capacity"]["ok"], short["head"]["capacity"]["limit_krw"]), (False, "49999999.00"))
        for row in short["paths"].values():
            self.assertEqual((row["comparable"], row["reasons"]),
                             (False, {"failures": [], "holds": ["start_capacity_below_floor"], "later": []}))
            self.assertEqual((row["strategy"], row["series"], row["start_orders"]["orders"]["hold"]["status"]),
                             (None, None, "not_submitted"))

    def test_floor_follows_the_account_and_skips_the_reference_product(self):
        # 한도 7,500만 원: 몫(5천만)은 열리고 단독(1억)은 열리지 않는다
        self.assertTrue(self.unit(1500000000, SHARE)["head"]["capacity"]["ok"])
        self.assertFalse(self.unit(1500000000, SOLO)["head"]["capacity"]["ok"])
        reference = self.unit(1500000000, SOLO, SYMBOL)  # 기준 상품에는 이 문턱이 없다(주문 한도는 장부가 따로 본다)
        self.assertEqual((reference["head"]["capacity"]["ok"], reference["head"]["capacity"]["floor_krw"]), (True, None))
        # 그래도 코어 8천만·보유 1억 원의 첫 주문은 주문별 한도(7,500만 원)에 걸려 비교 보류다
        self.assertEqual(reference["paths"][OHLC]["reasons"]["holds"],
                         ["base_core_first_buy_failed", "base_hold_first_buy_failed"])
        self.assertFalse(reference["paths"][OHLC]["comparable"])


class DailySeries(unittest.TestCase):
    def setUp(self):
        self.data, self.config = market(SYMBOL, SHAPES[SYMBOL]), config_of(SHARE, 7, base=WORLD_BASE, spec=WORLD_SPEC, mix=WORLD_MIX)

    def test_series_is_the_ledger_and_keeps_the_balance(self):
        run = open_unit(self.data, self.config, J(2), cost=BASE)
        result = plain(unit_result(run))
        for path in PATHS:
            series, row, group = result["paths"][path]["series"], result["paths"][path], run.groups[path]
            self.assertEqual(len(series["at"]), len(WORLD_DAYS))
            self.assertEqual(series["hold"]["equity"], [str(v["equity_krw"]) for v in group["hold"].result.valuations])
            pairs = zip(group["core"].result.valuations, group["M"].result.valuations)
            self.assertEqual(series["strategy"]["equity"], [str(core["equity_krw"] + part["equity_krw"]) for core, part in pairs])
            for name in ("strategy", "hold"):
                block = series[name]
                self.assertEqual(block["equity"][-1], row[name]["ending_equity_krw"])
                for equity, cash, receivable, holding in zip(block["equity"], block["cash"], block["receivables"], block["holdings"]):
                    self.assertEqual(D(equity), D(cash) + D(receivable) + D(holding))
        self.assertEqual(result["head"]["opened_at"], str(self.data.bars[0].open_at))

    def test_orders_use_the_real_share_capital_not_half_of_the_solo_account(self):
        """시가 10,005원(수수료 포함 10,006.50075원). 1억: floor(1e8 ÷ 10,006.50075) = 9,993주, 5천만: 4,996주(절반 4,996.5 아님).
        코어 80%: 8천만 → 7,994주, 4천만 → 3,997주. 분할 첫 예산 ÷ 9: 2,222,222원 → 222주, 1,111,111원 → 111주."""
        data = flat({J(2): "10005"})
        expected = {SOLO: (9993, 7994, 222), SHARE: (4996, 3997, 111)}
        for kind, quantities in expected.items():
            orders = run_unit(data, config_of(kind, 28), J(2), cost=BASE)["paths"][OHLC]["start_orders"]["orders"]
            self.assertEqual(tuple(orders[name]["quantity"] for name in ("hold", "core", "split")), quantities)
        self.assertNotEqual(expected[SOLO][0], 2 * expected[SHARE][0])

    def test_changing_future_bars_keeps_earlier_values(self):
        """1/8 이후 일봉을 바꿔도 1/7까지(네 거래일)의 일별 계열과 시작 주문은 같다(두 비용, 두 경로, 두 계좌)."""
        days = weekdays(J(2), J(10))
        prices = {J(2): "10000", J(3): "9800", J(6): "9600", J(7): "9950", J(8): "9750"}
        first, second = flat(prices, days), flat({**prices, J(8): "10300", J(9): "9500"}, days)
        config, compared = config_of(SHARE, 7), 0  # 7번: 코어 50%·kg 2·kp 1·5차
        for cost in (BASE, STRESS):
            one, other = plain(run_unit(first, config, J(2), cost=cost)), plain(run_unit(second, config, J(2), cost=cost))
            for path in PATHS:
                left, right = one["paths"][path], other["paths"][path]
                self.assertEqual(left["start_orders"], right["start_orders"])
                for name in ("strategy", "hold"):
                    for column in ("equity", "cash", "receivables", "holdings"):
                        self.assertEqual(left["series"][name][column][:4], right["series"][name][column][:4])
                        compared += 4
                    self.assertNotEqual(left["series"][name]["equity"], right["series"][name]["equity"])
                legs = [(SYMBOL, left["series"]), (KOSDAQ, right["series"])]  # 더한 계열도 앞부분은 같다
                mixed = mixing.account_rows(legs, "strategy")[:4]
                self.assertEqual([row["equity_krw"] for row in mixed], [2 * D(v) for v in left["series"]["strategy"]["equity"][:4]])
        self.assertEqual(compared, 2 * 2 * 2 * 4 * 4)


# ---------------------------------------------------------------------------
# 혼합 조립(설계 4절, 7절 4·5번)
# ---------------------------------------------------------------------------
def record_of(symbol, data, kind, number, cost=BASE, start=None, mix=MIX, floor=False):
    """단위 기록의 본문(직렬화한 꼴). 식별값 없이 결과만 가진다."""
    start = start or data.start
    unit = Unit(kind, symbol, start, number, cost)
    config = unit_config(mix, unit, base=FROZEN, grid=SPEC)
    unit_floor = mix.floor(unit) if floor else None
    return {"result": json.loads(batch.compact(run_unit(data, config, start, cost=cost, nominal_floor=unit_floor)))}


def frozen_context():
    """고정 정의·고정 설정의 합성 문맥(혼합 조립 함수에 넘길 틀만 쓴다)."""
    return open_synthetic_context(MemorySource({}, "frame"), mix=MIX)


WEEK_LABELS = [("2024-2026-11-13", "2024-2026-11-13")]  # 2025-01의 한 주는 고정 설정의 마지막 구간에 든다


class MixOfTwoShares(unittest.TestCase):
    """두 5천만 원 몫의 일별 원화 계열을 그대로 더한다. 이체·재조정은 없다."""

    def legs(self, number=28):
        first = as_symbol(flat({J(2): "10000"}), KOSDAQ)
        second = as_symbol(flat({J(2): "20000"}), CHINA)
        return [(KOSDAQ, record_of(KOSDAQ, first, SHARE, number)), (CHINA, record_of(CHINA, second, SHARE, number))]

    def test_hand_computed_balances_and_cash_are_preserved(self):
        """가격이 변하지 않는 두 상품(10,000원·20,000원), 28번(코어 4천만·분할 1천만, 첫 예산 1,111,111원).

        보유 몫: 10,000원 → floor(5e7 ÷ 10,001.5) = 4,999주, 현금 5e7 − 4,999 × 10,001.5 = 2,501.5원.
                 20,000원 → floor(5e7 ÷ 20,003) = 2,499주, 현금 5e7 − 2,499 × 20,003 = 12,503원.
        전략 몫: 10,000원 → 코어 3,999주 + 분할 111주 = 4,110주, 현금 5e7 − 4,110 × 10,001.5 = 8,893,835원.
                 20,000원 → 코어 1,999주 + 분할 55주 = 2,054주, 현금 5e7 − 2,054 × 20,003 = 8,913,838원.
        혼합 보유: ETF 49,990,000 + 49,980,000 = 99,970,000원, 현금 15,004.5원, 순자산 99,985,004.5원.
        혼합 전략: ETF 41,100,000 + 41,080,000 = 82,180,000원, 현금 17,807,673원, 순자산 99,987,673원."""
        legs = [(symbol, record["result"]["paths"][OHLC]["series"]) for symbol, record in self.legs()]
        hold, strategy = mixing.account_rows(legs, "hold"), mixing.account_rows(legs, "strategy")
        self.assertEqual(len(hold), 5)
        for row in hold:
            self.assertEqual((row["equity_krw"], row["cash_krw"], row["holdings_krw"]),
                             (D("99985004.5"), D("15004.5"), {KOSDAQ: D(49990000), CHINA: D(49980000)}))
        for row in strategy:
            self.assertEqual((row["equity_krw"], row["cash_krw"], sum(row["holdings_krw"].values())),
                             (D(99987673), D(17807673), D(82180000)))
            self.assertEqual(row["equity_krw"], row["cash_krw"] + row["receivables_krw"] + sum(row["holdings_krw"].values()))

    def test_mix_metrics_use_the_summed_account(self):
        """연복리 = (끝 ÷ 1억)^(365.25 ÷ 경과일) − 1. 경과: 1/2 시가(00:00 UTC) → 1/8 18:00 KST(09:00 UTC) = 6.375일."""
        ctx = frozen_context()
        slots = rows.mix_slots(ctx, self.legs(), WEEK_LABELS)
        self.assertEqual(list(slots), list(PAIRS))
        with localcontext(prec=80):
            power = D("365.25") / D("6.375")
            strategy, hold = (D(99987673) / D(100000000)) ** power - 1, (D("99985004.5") / D(100000000)) ** power - 1
            difference = strategy - hold
        for slot in slots.values():
            self.assertEqual((slot["strategy"]["cagr"], slot["hold"]["cagr"]), (strategy, hold))
            self.assertEqual(slot["delta_cagr"], difference)
            self.assertEqual(slot["strategy"]["exposure_mean"].quantize(D("1E-20")),
                             (D(82180000) / D(99987673)).quantize(D("1E-20")))
            self.assertEqual((slot["strategy"]["ending_equity_krw"], slot["hold"]["ending_equity_krw"]),
                             (D(99987673), D("99985004.5")))
            self.assertEqual(slot["trades"], {"buys": 2, "sells": 0, "by_leg": {KOSDAQ: {"buys": 1, "sells": 0, "status": "ok"},
                                                                               CHINA: {"buys": 1, "sells": 0, "status": "ok"}}})
            self.assertEqual(list(slot["segments"]), ["2024-2026-11-13"])
        self.assertEqual(judge.row_verdict({key: slot["delta_cagr"] for key, slot in slots.items()}, PAIRS)["status"],
                         judge.COUNTER)  # 덜 사서 수수료가 적다: 값이 작아도 양수는 양수다

    def test_one_share_mix_equals_the_solo_account(self):
        """한 상품에 전부 배정한 혼합(몫 하나)은 그 단위의 단독 계좌와 같다(설계 7절 4번)."""
        ctx = frozen_context()
        for kind, number in ((SOLO, 28), (SHARE, 1), (SOLO, 7)):
            record = record_of(SYMBOL, case2(), kind, number)
            solo = rows.solo_slots(record["result"], [("2024-2026-11-13", "2024-2026-11-13")])
            for path in PATHS:
                alone = rows._mixed(ctx, [(SYMBOL, record)], (path,), labels=WEEK_LABELS)
                self.assertEqual({key: alone[key] for key in ("delta_cagr", "strategy", "hold", "mdd_deepening", "segments")},
                                 {key: solo[path][key] for key in ("delta_cagr", "strategy", "hold", "mdd_deepening", "segments")})
                self.assertEqual(alone["trades"], solo[path]["trades"])
            self.assertTrue(rows.reproduced(ctx, record))

    def test_each_share_keeps_its_own_path(self):
        """1번(코어 50%·kg 1·kp 1·5차) 몫: 고가를 먼저 지나면 매수 1·매도 1, 저가를 먼저 지나면 매수 2·매도 1(kodex_grid 표 5와
        같은 봉). 다른 몫은 가격이 변하지 않아 경로와 무관하다. 혼합의 경로 짝은 각 몫이 자기 경로의 계열을 낸다."""
        changes = {**bar(J(3), 10000, 10105, 9895, 10000), **bar(J(6), 10000, 10110, 9895, 10000)}
        moving = record_of(KOSDAQ, as_symbol(flat({J(2): "10000"}, changes=changes), KOSDAQ), SHARE, 1)
        still = record_of(CHINA, as_symbol(flat({J(2): "20000"}), CHINA), SHARE, 1)
        ctx = frozen_context()
        slots = rows.mix_slots(ctx, [(KOSDAQ, moving), (CHINA, still)], WEEK_LABELS)
        self.assertEqual([slots[pair]["trades"]["buys"] for pair in PAIRS], [2, 2, 3, 3])
        self.assertEqual(slots[PAIRS[0]]["delta_cagr"], slots[PAIRS[1]]["delta_cagr"])
        self.assertEqual(slots[PAIRS[2]]["delta_cagr"], slots[PAIRS[3]]["delta_cagr"])
        self.assertNotEqual(slots[PAIRS[0]]["delta_cagr"], slots[PAIRS[2]]["delta_cagr"])
        swapped = rows.mix_slots(ctx, [(CHINA, still), (KOSDAQ, moving)], WEEK_LABELS)  # 몫의 순서를 바꾸면 둘째 경로가 갈린다
        self.assertEqual([swapped[pair]["trades"]["buys"] for pair in PAIRS], [2, 3, 2, 3])
        self.assertEqual(swapped[PAIRS[1]]["strategy"], slots[PAIRS[2]]["strategy"])
        for pair, (left, right) in zip(PAIRS, [(a, b) for a in PATHS for b in PATHS]):
            series = [(KOSDAQ, moving["result"]["paths"][left]["series"]), (CHINA, still["result"]["paths"][right]["series"])]
            by_hand = mixing.account_rows(series, "strategy")[-1]["equity_krw"]
            self.assertEqual(slots[pair]["strategy"]["ending_equity_krw"], by_hand)

    def test_a_share_that_cannot_be_compared_blanks_the_pairs_that_use_it(self):
        """높은 비용에서 개설을 거부당한 몫(σ 0.2%: 2 × 0.3% > 2 × 0.2%)은 그 몫을 쓰는 모든 경로 짝을 빈칸으로 만든다."""
        rejected = record_of(KOSDAQ, as_symbol(flat({J(2): "10000"}, sigma="0.002"), KOSDAQ), SHARE, 28, STRESS)
        fine = record_of(CHINA, as_symbol(flat({J(2): "20000"}), CHINA), SHARE, 28, STRESS)
        slots = rows.mix_slots(frozen_context(), [(KOSDAQ, rejected), (CHINA, fine)], WEEK_LABELS)
        for slot in slots.values():
            self.assertEqual((slot["comparable"], slot["delta_cagr"], slot["strategy"], slot["hold"]), (False, None, None, None))
            self.assertEqual(slot["reasons"], ["229200:stress_start_rejected_by_cost_guard"])
        found = judge.row_verdict({key: slot["delta_cagr"] for key, slot in slots.items()}, PAIRS)
        self.assertEqual((found["status"], set(found["signs"].values())), (judge.BLANK, {None}))
        one_path = json.loads(json.dumps(fine))  # 한 경로만 빈칸인 몫: 그 경로를 쓰는 두 짝만 빈칸이고 행은 빈칸이다
        broken = one_path["result"]["paths"][OLHC]
        broken.update(comparable=False, strategy=None, hold=None, delta=None,
                      reasons={"failures": [], "holds": ["stress_combined_start_liquidity"], "later": []})
        ok = record_of(KOSDAQ, as_symbol(flat({J(2): "10000"}), KOSDAQ), SHARE, 28, STRESS)
        partial = rows.mix_slots(frozen_context(), [(KOSDAQ, ok), (CHINA, one_path)], WEEK_LABELS)
        self.assertEqual([partial[pair]["comparable"] for pair in PAIRS], [True, False, True, False])
        self.assertEqual(partial[PAIRS[1]]["reasons"], ["169950:stress_combined_start_liquidity"])
        signs = judge.row_verdict({key: slot["delta_cagr"] for key, slot in partial.items()}, PAIRS)
        self.assertEqual((signs["status"], list(signs["signs"].values())), (judge.BLANK, ["+", None, "+", None]))

    def test_shares_valued_at_different_times_are_not_added(self):
        short = record_of(KOSDAQ, as_symbol(flat({J(2): "10000"}, weekdays(J(2), J(7))), KOSDAQ), SHARE, 28)
        full = record_of(CHINA, as_symbol(flat({J(2): "20000"}), CHINA), SHARE, 28)
        with self.assertRaisesRegex(ValueError, "same times"):
            rows.mix_slots(frozen_context(), [(KOSDAQ, short), (CHINA, full)], WEEK_LABELS)
        with self.assertRaisesRegex(ValueError, "same times"):
            mixing.combine([])
        late = json.loads(json.dumps(full))
        late["result"]["head"]["opened_at"] = "2025-01-03 00:00:00+00:00"
        with self.assertRaisesRegex(AssemblyError, "open at the same time"):
            rows.mix_slots(frozen_context(), [(KOSDAQ, record_of(KOSDAQ, as_symbol(flat({J(2): "10000"}), KOSDAQ), SHARE, 28)),
                                              (CHINA, late)], WEEK_LABELS)


# ---------------------------------------------------------------------------
# 실행·봉인·재개·조립(설계 7·8절)
# ---------------------------------------------------------------------------
B_START = WORLD_DAYS[1].isoformat()  # 합성 B 묶음의 주 시작일(2025-01-03)


class WorldRun(_Workspace):
    """두 묶음(A: 단독 셋·혼합 하나, B: 단독 하나·혼합 하나)의 합성 묶음을 순차와 병렬로 한 번씩 실행해 봉인하고
    조립한다(334단위, 주 판정 144 + 72행)."""

    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.root, cls.ctx, cls.echoed = Path(cls.shared.name), world_context(), {"one": [], "many": []}
        for name, workers in (("one", 1), ("many", 4)):
            batch.run(cls.ctx, cls.root / name, workers=workers, echo=cls.echoed[name].append)
        cls.results = {name: batch.assemble(cls.ctx, cls.root / name) for name in ("one", "many")}

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def test_sequential_and_parallel_runs_are_the_same_bytes(self):
        one, many = self.root / "one" / "run", self.root / "many" / "run"
        names = sorted(path.name for path in (one / "units").iterdir())
        self.assertEqual((len(names), names), (334, sorted(path.name for path in (many / "units").iterdir())))
        for name in names:
            self.assertEqual((one / "units" / name).read_bytes(), (many / "units" / name).read_bytes())
        for name in ("result.json", "summary.md", "lock.json"):
            self.assertEqual((one / name).read_bytes(), (many / name).read_bytes())
        self.assertEqual(json.loads(sealing.dump(self.results["one"])), json.loads(sealing.dump(self.results["many"])))
        self.assertGreater(len({(one / "units" / name).read_bytes() for name in names}), 320)
        self.assertEqual(list(self.root.rglob(".*.tmp")), [])

    def test_units_are_sealed_marked_and_identified(self):
        folder = self.root / "many" / "run"
        for unit in plan_units(WORLD_MIX, WORLD_SPEC):
            record = json.loads((folder / "units" / f"{unit.name}.json").read_text())
            seat = records.Seat(self.ctx, unit)
            head = record["identity"]
            self.assertEqual((record["exploration"], record["formal_run"], head["exploration"], head["formal_run"], head["scope"]),
                             (True, False, True, False, "synthetic"))
            self.assertEqual((head["applied_config_sha256"], record["result"]["applied_config_sha256"]),
                             (seat.config().applied_sha256(),) * 2)
            self.assertEqual((head["mix_definition_file_sha256"], head["mix_definition_applied_sha256"], head["study"]),
                             (MIX.sha256, WORLD_MIX.applied_sha256(), "markets-mix"))
            self.assertEqual((head["mix_source_sha256"], head["exploration_source_sha256"], head["approved_source_sha256"]),
                             (identity.mix_source()["mix_source_sha256"], MIX.approved["kodex_grid_source_sha256"],
                              MIX.approved["source_sha256"]))
            self.assertEqual((head["kind"], head["symbol"], head["cost"], head["number"], head["start"]),
                             (unit.kind, unit.symbol, unit.cost, unit.number, unit.start.isoformat()))
            self.assertNotIn("timing", record)
            seal = sealing.read_sealed(folder / "sealed" / "seals" / f"{unit.name}.json", batch.SEAL_SCHEMA)
            self.assertEqual((seal["unit"], seal["sha256"]), (unit.name, sealing.file_digest(folder / "units" / f"{unit.name}.json")))
            self.assertEqual(seal["timing"]["by_kind_counts"]["split"], 2)  # 한 단위 = 한 비용의 두 경로
        modes = {(path.stat().st_mode & 0o777) for path in (folder / "units").iterdir()}
        self.assertEqual((modes, (folder / "progress.log").stat().st_mode & 0o077), ({0o600}, 0o044))

    def test_share_units_differ_from_solo_units_only_by_capital(self):
        folder = self.root / "one" / "run" / "units"
        for number in (1, 28):
            solo = json.loads((folder / f"solo-229200-2025-01-02-base-setting-{number:02d}.json").read_text())
            share = json.loads((folder / f"share-229200-2025-01-02-base-setting-{number:02d}.json").read_text())
            self.assertEqual({name: D(value) * 2 for name, value in share["result"]["capital"].items()},
                             {name: D(value) for name, value in solo["result"]["capital"].items()})
            self.assertEqual((solo["identity"]["nominal_floor_krw"], share["identity"]["nominal_floor_krw"]),
                             ("100000000", "50000000"))
            self.assertNotEqual(solo["identity"]["applied_config_sha256"], share["identity"]["applied_config_sha256"])
            same = {key for key in solo["setting"] if solo["setting"][key] == share["setting"][key]}
            self.assertEqual(same, {"number", "core", "kg", "kp", "levels"})

    def test_progress_log_has_only_names_times_and_technical_codes(self):
        for name in ("one", "many"):
            text = (self.root / name / "run" / "progress.log").read_text()
            lines = text.splitlines()
            self.assertEqual((len(lines), lines), (334, self.echoed[name]))
            self.assertTrue(all(LOG_LINE.match(line) for line in lines), lines[:2])
            self.assertFalse([word for word in RESULT_WORDS if word in text.lower()])
            self.assertEqual({line.split()[3] for line in lines}, {"done"})

    def test_rows_come_from_the_sealed_units_of_their_own_cohort(self):
        result, folder = self.results["one"], self.root / "one" / "run" / "units"
        self.assertEqual((len(result["rows"]), result["conclusion"]["rows"]), (216, 216))
        self.assertEqual([row["key"] for row in result["rows"]], summary.expected_keys(self.ctx))
        self.assertEqual([(row["cohort"], row["rows"]) for row in result["conclusion"]["by_cohort"]], [("A", 144), ("B", 72)])
        for key, unit_name, start in (("229200/07@2025-01-02/base", "solo-229200-2025-01-02-base-setting-07", "2025-01-02"),
                                      (f"133690/07@{B_START}/base", f"solo-133690-{B_START}-base-setting-07", B_START)):
            row = next(item for item in result["rows"] if item["key"] == key)
            unit = json.loads((folder / f"{unit_name}.json").read_text())["result"]
            # 기준 보유: 같은 시작일·비용의 KODEX 200 1억 원 단독(28번 단위가 싣는다). B 묶음에는 그 단위만 있다
            reference = json.loads((folder / f"solo-069500-{start}-base-setting-28.json").read_text())["result"]
            for path in PATHS:
                slot, sealed_path = row["paths"][path], unit["paths"][path]
                with localcontext(prec=200):  # 80자리 지표끼리의 뺄셈이 반올림되지 않는 자릿수
                    difference = D(sealed_path["strategy"]["cagr"]) - D(sealed_path["hold"]["cagr"])
                    versus = D(sealed_path["strategy"]["cagr"]) - D(reference["paths"][path]["hold_only"]["cagr"])
                self.assertEqual((slot["delta_cagr"], slot["vs_reference_hold"]), (difference, versus))
                self.assertEqual(D(sealed_path["delta"]["delta_cagr"]), difference)
                self.assertEqual(list(slot["segments"]), ["P1", "P2", "P3", "P4"])
                self.assertEqual(slot["trades"], {key: sealed_path["split_trades"][key] for key in ("buys", "sells")})
            self.assertEqual(row["signs"], {path: judge.sign_of(row["paths"][path]["delta_cagr"]) for path in PATHS})
        self.assertFalse((folder / f"solo-069500-{B_START}-base-setting-07.json").exists())

    def test_reference_hold_does_not_depend_on_the_setting(self):
        """기준 보유를 28번 단위에서 읽는 근거: 같은 시작일·비용의 1억 원 보유는 36개 설정에서 모두 같다."""
        folder = self.root / "one" / "run" / "units"
        holds = {json.dumps(json.loads((folder / f"solo-069500-2025-01-02-base-setting-{n:02d}.json").read_text())
                            ["result"]["paths"][OHLC]["hold_only"], sort_keys=True) for n in range(1, 37)}
        self.assertEqual(len(holds), 1)

    def test_mix_rows_add_the_two_share_units_for_each_path_pair(self):
        result, folder = self.results["one"], self.root / "one" / "run" / "units"
        row = next(item for item in result["rows"] if item["key"] == f"069500+133690/28@{B_START}/base")
        legs = {symbol: json.loads((folder / f"share-{symbol}-{B_START}-base-setting-28.json").read_text())["result"]
                for symbol in (SYMBOL, NASDAQ)}
        self.assertEqual(list(row["paths"]), list(PAIRS))
        for pair, (left, right) in zip(PAIRS, [(a, b) for a in PATHS for b in PATHS]):
            ending = (D(legs[SYMBOL]["paths"][left]["strategy"]["ending_equity_krw"])
                      + D(legs[NASDAQ]["paths"][right]["strategy"]["ending_equity_krw"]))
            hold = (D(legs[SYMBOL]["paths"][left]["hold"]["ending_equity_krw"])
                    + D(legs[NASDAQ]["paths"][right]["hold"]["ending_equity_krw"]))
            self.assertEqual((row["paths"][pair]["strategy"]["ending_equity_krw"], row["paths"][pair]["hold"]["ending_equity_krw"]),
                             (ending, hold))
        self.assertEqual({leg["capital"]["hold"] for leg in legs.values()}, {"50000000"})
        self.assertEqual({leg["head"]["start"] for leg in legs.values()}, {B_START})

    def test_conclusion_side_notes_and_sensitivity_are_published_together(self):
        result = self.results["one"]
        done = result["conclusion"]
        counted = {name: sum(row["status"] == name for row in result["rows"]) for name in judge.STATUSES}
        self.assertEqual((done["counts"], done["kind"]), (counted, judge.HAS_COUNTER))  # 하락하는 합성 상품이 있다
        self.assertEqual(len(done["counterexamples"]), counted[judge.COUNTER])
        self.assertEqual(sum(row["count"] for row in done["counterexample_settings"]), counted[judge.COUNTER])
        self.assertIn("2025-01-02 시작 169950에서 36개", done["sentence"])
        self.assertEqual([(view["subject"], view["start"]) for view in result["subjects"]],
                         [(SYMBOL, "2025-01-02"), (KOSDAQ, "2025-01-02"), (CHINA, "2025-01-02"), ("229200+169950", "2025-01-02"),
                          (NASDAQ, B_START), ("069500+133690", B_START)])
        for view in result["subjects"]:
            self.assertEqual((view["rows"], sum(view["counts"].values())), (36, 36))
            self.assertEqual([len(view["regions"][name]) for name in ("edge_gt", "review_base_edge_min")], [8, 8])
        china = next(view for view in result["subjects"] if view["subject"] == CHINA)
        self.assertEqual((china["own_hold_regimes"], china["counts"][judge.COUNTER]), ([judge.DOWN], 36))
        self.assertEqual([view["cohort"] for view in result["sensitivity"]], ["A"] * 4 + ["B"] * 2)
        for view in result["sensitivity"]:
            row_cohort = WORLD_MIX.cohorts[0] if view["cohort"] == "A" else WORLD_MIX.cohorts[1]
            self.assertEqual([(row["start"], row["cost"]) for row in view["rows"]],
                             [(start.isoformat(), cost) for start, cost in WORLD_MIX.slots(row_cohort)])
            self.assertEqual({row["number"] for row in view["rows"]}, {28})
            self.assertEqual(list(view["review_threshold"]), [start.isoformat() for start in row_cohort.starts()])
        sensitivity_only = {row["key"] for view in result["sensitivity"] for row in view["rows"]} - {row["key"] for row in result["rows"]}
        self.assertEqual(len(sensitivity_only), 30)  # 주 판정과 겹치는 6행 외에 민감도 30행
        text = (self.root / "one" / "run" / "summary.md").read_text()
        for phrase in ("모두 덜 벌었다는 주장은 성립하지 않는다", "## 반례 행", "## 28번 민감도", "A 묶음(2025-01-02 시작) 144행",
                       "B 묶음(2025-01-03 시작) 72행", "exploration=True, formal_run=False", "채권은 판정 불가"):
            self.assertIn(phrase, text)

    def test_resume_skips_sealed_units_and_reassembly_is_the_same(self):
        target = self.out / "copy"
        shutil.copytree(self.root / "one", target)
        (target / "run" / "progress.log").unlink()
        batch.run(self.ctx, target, workers=4, echo=self.lines.append)
        self.assertEqual((len(self.lines), {line.split()[3] for line in self.lines}), (334, {"skipped"}))
        self.assertEqual(json.loads(sealing.dump(batch.assemble(self.ctx, target))), json.loads(sealing.dump(self.results["one"])))


class SentencesEndToEnd(_Workspace):
    def test_all_rising_markets_give_the_first_sentence(self):
        ctx = world_context(shapes=RISING, name="rising")
        batch.run(ctx, self.out, workers=4, echo=self.lines.append)
        done = batch.assemble(ctx, self.out)["conclusion"]
        self.assertEqual((done["kind"], done["counts"][judge.INFERIOR], done["all_claimed"]), (judge.ALL_INFERIOR, 216, True))
        self.assertEqual((done["counterexamples"], done["left_open"]), ([], []))
        self.assertIn("36개 설정 모두 같은 상품·배분을 계속 보유했을 때보다", (self.out / "run" / "summary.md").read_text())

    def test_held_and_unopened_products_leave_blanks_with_reasons(self):
        """자료가 없는 상품(나스닥, B 묶음)과 5% 한도가 7,500만 원인 상품(차이나: 1억 단독은 열지 않고 5천만 몫은 연다)."""
        thin = market(CHINA, RISING[CHINA], changes={day: {"adv20": D(1500000000)} for day in WORLD_DAYS})
        ctx = world_context(shapes=RISING, name="blanks", held=(NASDAQ,), **{f"s{CHINA}": thin})
        batch.run(ctx, self.out, workers=4, echo=self.lines.append)
        result = batch.assemble(ctx, self.out)
        blank = {view["subject"]: view["counts"][judge.BLANK] for view in result["subjects"]}
        self.assertEqual(blank, {SYMBOL: 0, KOSDAQ: 0, CHINA: 36, "229200+169950": 0, NASDAQ: 36, "069500+133690": 36})
        done = result["conclusion"]
        self.assertEqual((done["kind"], done["counts"][judge.BLANK], done["counts"][judge.INFERIOR]), (judge.PARTIAL, 108, 108))
        self.assertEqual(done["sentence"], "216행 중 108행에서 덜 벌었다. 나머지는 동률 0행, 경로에 따라 부호가 갈린 경우 0행, "
                                           "비교 불가 108행이다.")
        self.assertEqual([(row["cohort"], row["counts"][judge.BLANK]) for row in done["by_cohort"]], [("A", 36), ("B", 72)])
        reasons = {row["subject"]: sorted({reason for found in row["reasons"].values() for reason in found})
                   for row in done["left_open"]}
        self.assertEqual(reasons[NASDAQ], ["synthetic: no input for this product"])
        self.assertEqual(reasons[CHINA], ["start_capacity_below_floor"])
        self.assertEqual(reasons["069500+133690"], ["133690:synthetic: no input for this product"])
        self.assertNotIn("229200+169950", reasons)  # 차이나 5천만 원 몫은 열렸다
        text = (self.out / "run" / "progress.log").read_text()
        self.assertTrue(all(LOG_LINE.match(line) for line in text.splitlines()))
        self.assertFalse([word for word in RESULT_WORDS + ("capacity", "synthetic") if word in text.lower()])
        self.assertNotIn("모두", done["sentence"])

    def test_high_cost_rejection_is_blank_only_in_the_sensitivity_rows(self):
        """σ 0.2%인 상품: 기본 비용(2 × 0.015% ≤ kg × σ)은 열리고, 28번의 높은 비용(2 × 0.3% > 2 × 0.2%)은 개설 거부다."""
        calm = market(CHINA, RISING[CHINA], sigma="0.002")
        ctx = world_context(shapes=RISING, name="calm", **{f"s{CHINA}": calm})
        batch.run(ctx, self.out, workers=4, echo=self.lines.append)
        result = batch.assemble(ctx, self.out)
        self.assertEqual(result["conclusion"]["kind"], judge.ALL_INFERIOR)  # 주 판정은 기본 비용의 216행뿐이다
        views = {view["subject"]: view for view in result["sensitivity"]}
        for name, reason in ((CHINA, "stress_start_rejected_by_cost_guard"),
                             ("229200+169950", "169950:stress_start_rejected_by_cost_guard")):
            high = [row for row in views[name]["rows"] if row["cost"] == STRESS]
            self.assertEqual({row["status"] for row in high}, {judge.BLANK})
            self.assertEqual({r for row in high for slot in row["paths"].values() for r in slot["reasons"]}, {reason})
            self.assertEqual(set(views[name]["review_threshold"].values()), {judge.NOT_MET})  # 기본 비용 차가 이미 음수다
            marks = {row["mark"] for row in views[name]["cost"].values()}
            self.assertEqual((marks, views[name]["cost_dependent"]), ({judge.OPEN}, judge.OPEN))
        self.assertEqual({row["mark"] for row in views[KOSDAQ]["cost"].values()}, {judge.SAME})
        self.assertEqual({row["mark"] for row in views[NASDAQ]["cost"].values()}, {judge.SAME})


SMALL_MIX = world_mix(cohort("A", J(2), (SYMBOL, KOSDAQ), [(SYMBOL, KOSDAQ)]), products=(SYMBOL, KOSDAQ))


def small_context(mix=SMALL_MIX, name="small"):
    shapes = {symbol: SHAPES[symbol] for symbol in mix.symbols()}
    inputs = world_inputs(shapes=shapes, mix=mix)
    return open_synthetic_context(MemorySource(inputs, name), mix=mix, spec=WORLD_SPEC, base=WORLD_BASE)


class _Sealed(_Workspace):
    """한 번 실행해 봉인한 작은 묶음(두 상품·한 혼합, 164단위)을 시험마다 복사해 쓴다."""

    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.ctx = small_context()
        batch.run(cls.ctx, Path(cls.shared.name) / "run", workers=4, echo=lambda line: None)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def copy(self, name="run"):
        shutil.copytree(Path(self.shared.name) / "run", self.out / name)
        return self.out / name


SOLO_28 = "solo-229200-2025-01-02-base-setting-28.json"
SHARE_28 = "share-229200-2025-01-02-base-setting-28.json"


class AssemblyRefusals(_Sealed):
    """계획한 단위가 모두 봉인됐을 때만 연다. 누락·여분·중복·식별 불일치·변조는 조립하지 않고 멈춘다."""

    def refused(self, out, message, error=AssemblyError):
        with self.assertRaisesRegex(error, message):
            batch.assemble(self.ctx, out)
        self.assertFalse((out / "run" / "result.json").exists())

    def test_complete_run_assembles(self):
        out = self.copy()
        result = batch.assemble(self.ctx, out)
        self.assertEqual((len(result["rows"]), len(result["units"]), result["conclusion"]["rows"]), (108, 164, 108))
        self.assertEqual(sorted(path.name for path in (out / "run").iterdir() if path.is_file()),
                         ["lock.json", "progress.log", "result.json", "summary.md"])

    def test_missing_extra_and_unsealed_units(self):
        out = self.copy("missing")
        (out / "run" / "units" / SOLO_28).unlink()
        self.refused(out, f"missing {SOLO_28}")
        out = self.copy("unsealed")  # 단위 파일은 있지만 봉인 기록이 없다
        (out / "run" / "sealed" / "seals" / SHARE_28).unlink()
        self.refused(out, "seal record is missing or damaged")
        out = self.copy("extra")
        shutil.copy(out / "run" / "units" / SOLO_28, out / "run" / "units" / "solo-999999-2025-01-02-base-setting-28.json")
        self.refused(out, "unexpected solo-999999")
        out = self.copy("marker")
        (out / "run" / "errors").mkdir()
        (out / "run" / "errors" / "x.error.txt").write_text("x E_ENGINE\n")
        self.refused(out, "error marker x.error.txt")

    def test_duplicate_units_in_the_plan_are_refused(self):
        out = self.copy()
        units = plan_units(SMALL_MIX, WORLD_SPEC)
        with sealing.writer_lock(out), self.assertRaisesRegex(AssemblyError, "duplicate units"):
            batch.read_units(batch.Plan(self.ctx, out, units + units[:1]))
        with sealing.writer_lock(out), self.assertRaisesRegex(AssemblyError, "missing"):
            batch.read_units(batch.Plan(self.ctx, out, units + (Unit(SOLO, CHINA, J(2), 28, BASE),)))

    def test_any_byte_change_is_caught_by_the_sealed_hash(self):
        out = self.copy()
        path = out / "run" / "units" / SOLO_28
        path.write_bytes(path.read_bytes() + b"\n")
        self.refused(out, "content differs from the hash sealed")
        with self.assertRaisesRegex(IdentityError, "sealed unit is not reused"):
            batch.run(self.ctx, out, echo=self.lines.append)  # 재개도 같은 검사를 쓴다

    def test_tampered_identity_setting_or_values(self):
        """봉인 해시까지 맞춰 고친 단위도 기록 검사에서 걸린다."""
        def path(record, name=OHLC):
            return record["result"]["paths"][name]

        def drop_reason(record):
            path(record).update(comparable=False, strategy=None, hold=None, delta=None)
            path(record)["series"]["strategy"] = None

        cases = {
            "other_symbol": (lambda r: r["identity"].update(symbol=CHINA), "identity differs in \\['symbol'\\]"),
            "other_cost": (lambda r: r["identity"].update(cost=STRESS), "identity differs"),
            "other_definition": (lambda r: r["identity"].update(mix_definition_applied_sha256="0" * 64), "identity differs"),
            "formal_flag": (lambda r: r.update(formal_run=True), "not an exploration unit"),
            "setting": (lambda r: r["setting"].update(kg="4"), "setting parameters or capital"),
            "account": (lambda r: r["setting"].update(account_krw="50000000"), "setting parameters or capital"),
            "body_cost": (lambda r: r["result"].update(cost=STRESS), "another mode, cost or applied"),
            "body_hash": (lambda r: r["result"].update(applied_config_sha256="0" * 64), "another mode, cost or applied"),
            "body_capital": (lambda r: r["result"]["capital"].update(hold="50000000"), "stated capital differs"),
            "head_symbol": (lambda r: r["result"]["head"].update(symbol=CHINA), "another symbol or start"),
            "held_and_opened": (lambda r: r["result"].update(held_reason="no input"), "either a held input"),
            "one_path": (lambda r: r["result"]["paths"].pop(OLHC), "both intraday paths"),
            "reason_on_valued": (lambda r: path(r)["reasons"]["holds"].append("base_x"), "must not carry a reason"),
            "blank_without_reason": (drop_reason, "empty values without a reason"),
            "blank_with_values": (lambda r: path(r).update(comparable=False, reasons={"failures": ["x"], "holds": [], "later": []}),
                                  "must not carry strategy values"),
            "segment_removed": (lambda r: path(r)["hold"]["segments"].pop(), "segments differ"),
            "series_short": (lambda r: path(r)["series"]["strategy"]["equity"].pop(), "series length"),
            "series_end": (lambda r: path(r)["series"]["hold"]["equity"].__setitem__(-1, "1"), "last equity differs"),
            "series_balance": (lambda r: path(r)["series"]["hold"]["cash"].__setitem__(3, "1"), "cash \\+ receivables"),
            "delta": (lambda r: path(r)["delta"].update(delta_cagr="0.5"), "not strategy minus hold"),
            "capacity": (lambda r: r["result"]["head"]["capacity"].update(floor_krw=None), "capacity differs"),
            "end": (lambda r: r["result"]["head"].update(end="2025-04-01"), "neither the registered end"),
            "evaluation": (lambda r: r["result"]["head"].update(last_evaluation="2025-04-02 09:00:00+00:00"), "18:00 KST"),
            "malformed": (lambda r: r["result"].pop("paths"), "malformed unit record"),
        }
        for name, (change, message) in cases.items():
            with self.subTest(name):
                out = self.copy(name)
                reseal(out / "run", SOLO_28, change)
                self.refused(out, message)

    def test_values_changed_consistently_are_caught_by_recomputing_from_the_series(self):
        """지표와 차를 서로 맞게 고쳐도, 봉인한 일별 계열로 다시 구한 지표와 다르면 멈춘다."""
        def inflate(record):
            row = record["result"]["paths"][OHLC]
            with localcontext(prec=80):
                row["strategy"]["cagr"] = str(D(row["strategy"]["cagr"]) + 1)
                row["delta"]["delta_cagr"] = str(D(row["strategy"]["cagr"]) - D(row["hold"]["cagr"]))
        out = self.copy()
        reseal(out / "run", SHARE_28, inflate)
        self.refused(out, "do not reproduce the sealed metrics")

    def test_other_definition_or_code_version_is_refused_on_resume_and_assembly(self):
        out = self.copy()
        other = small_context(replace(SMALL_MIX, thresholds={**SMALL_MIX.thresholds, "regime_band": D("0.02")}))
        with self.assertRaisesRegex(IdentityError, "mix_definition_applied_sha256"):
            batch.run(other, out, echo=self.lines.append)
        with self.assertRaisesRegex(AssemblyError, "identity differs"):
            batch.assemble(other, out)
        with patch.object(identity, "mix_source", return_value={"mix_source_sha256": "0" * 64}):
            with self.assertRaisesRegex(IdentityError, "does not describe the current code"):
                batch.assemble(self.ctx, out)
        renamed = small_context(name="other-inputs")
        with self.assertRaisesRegex(AssemblyError, "identity differs in \\['inputs'\\]"):
            batch.assemble(renamed, out)
        self.assertFalse((out / "run" / "result.json").exists())

    def test_engine_error_blocks_assembly_until_recovered(self):
        out = self.copy()
        for folder in ("units", "sealed/seals"):
            (out / "run" / folder / SOLO_28).unlink()
        with patch.object(batch, "run_unit", side_effect=RuntimeError("secret detail 12.5%")):
            with self.assertRaisesRegex(batch.RunError, "1 unit"):
                batch.run(self.ctx, out, echo=self.lines.append)
        self.assertEqual((self.lines[-1].split()[3:], (out / "run" / "errors" / SOLO_28.replace(".json", ".error.txt")).read_text()),
                         (["error", self.lines[-1].split()[4], "E_ENGINE"], f"{SOLO_28[:-5]} E_ENGINE\n"))
        self.assertNotIn("secret", "".join(self.lines) + (out / "run" / "progress.log").read_text())
        self.assertIn("secret detail", (out / "run" / "sealed" / "logs" / SOLO_28.replace(".json", ".log")).read_text())
        self.refused(out, "missing|error marker")
        batch.run(self.ctx, out, echo=self.lines.append)
        self.assertEqual(self.lines[-1].split()[3], "done")
        self.assertEqual(len(batch.assemble(self.ctx, out)["rows"]), 108)

    def test_first_assembly_is_fixed_and_never_overwritten(self):
        out = self.copy()
        first = batch.assemble(self.ctx, out)
        lock = sealing.read_sealed(out / "run" / "lock.json", batch.LOCK_SCHEMA)
        self.assertEqual((lock["units"], lock["chosen"]["kind"]), (first["units"], first["conclusion"]["kind"]))
        self.assertEqual(lock["outputs"]["summary.md"], sealing.file_digest(out / "run" / "summary.md"))
        self.assertEqual(json.loads(sealing.dump(batch.assemble(self.ctx, out))), json.loads(sealing.dump(first)))
        # 기록 검사와 계열 검산을 지나는 변경(체결 건수)도 최초 기록과 다르면 덮어쓰지 않는다
        reseal(out / "run", SOLO_28, lambda r: r["result"]["paths"][OHLC]["split_trades"].update(buys=99))
        with self.assertRaisesRegex(IdentityError, "differs from its first record"):
            batch.assemble(self.ctx, out)
        self.assertEqual(sealing.file_digest(out / "run" / "summary.md"), lock["outputs"]["summary.md"])


class ReviewThresholdRows(unittest.TestCase):
    def test_base_and_high_cost_rows_are_read_in_their_own_roles(self):
        """기본 비용 차 +2%p·높은 비용 차 +0.1%p → 충족. 두 비용을 뒤바꾸면(+0.1%p < +1%p) 미충족이어야 한다."""
        ctx = frozen_context()

        def row(start, cost, delta):
            return {"start": start.isoformat(), "cost": cost, "paths": {path: {"delta_cagr": D(delta)} for path in PATHS}}

        for cohort in MIX.cohorts:
            found = [row(start, BASE, "0.02") for start in cohort.starts()] + [row(start, STRESS, "0.001") for start in cohort.starts()]
            self.assertEqual(summary._reviews(ctx, cohort, found), {start.isoformat(): judge.MET for start in cohort.starts()})
            swapped = [row(start, BASE, "0.001") for start in cohort.starts()] + [row(start, STRESS, "0.02") for start in cohort.starts()]
            self.assertEqual(set(summary._reviews(ctx, cohort, swapped).values()), {judge.NOT_MET})


class ReportFormatting(unittest.TestCase):
    def test_points_and_spans_round_only_the_display(self):
        self.assertEqual([report.points("0.012345"), report.points("-0.00004"), report.points(None)],
                         ["+1.234", "-0.004", "—"])  # %p, 표기만 반올림한다(0.012345 → 1.2345 → 짝수 쪽 1.234)
        self.assertEqual(report._span(["0.01", None, "-0.02"]), "-2.000 ~ +1.000")
        self.assertEqual((report._span(["0.01", "0.01"]), report._span([None, None])), ("+1.000", "—"))
        self.assertEqual((judge.beat(D("1E-30")), judge.beat(D(0)), judge.beat(None)), (True, False, None))
        self.assertEqual(set(report.STATUS), set(judge.STATUSES))


class CohortStartOffsets(_Workspace):
    def test_every_cohort_needs_its_sensitivity_starts_at_plus_5_and_plus_20(self):
        """B 묶음의 민감도 시작일을 +6·+20으로 잘못 선언하면 첫 장부 전에 멈춘다(A 묶음이 맞아도)."""
        wrong = world_mix(WORLD_MIX.cohorts[0], cohort("B", WORLD_DAYS[1], (NASDAQ,), [(SYMBOL, NASDAQ)],
                                                         sensitivity=(WORLD_DAYS[7], WORLD_DAYS[21])))
        ctx = world_context(mix=wrong, name="wrong-offsets")
        with patch.object(batch, "run_unit", side_effect=AssertionError("ledger opened")):
            with self.assertRaisesRegex(ValueError, "differ from session offsets"):
                batch.run(ctx, self.out, echo=self.lines.append)
        self.assertEqual(self.lines, [])
        self.assertFalse(eligibility.collect(ctx)[0]["sensitivity_starts_are_offsets"])


class OutputFolderIsSeparate(_Workspace):
    def test_a_kodex_grid_result_folder_is_never_written(self):
        """kodex_grid의 단계 폴더가 있는 곳에는 시험 기록·재현·자격 집계·단위 어느 것도 쓰지 않는다."""
        (self.out / "stage1").mkdir()
        self.enterContext(patch.object(permit, "TEST_MODULES", ("tests.no_such_module",)))  # 시험 안에서 시험 묶음을 다시 돌리지 않는다
        ctx = small_context()
        calls = (lambda: permit.run_selftest(self.out), lambda: batch.run(ctx, self.out), lambda: batch.assemble(ctx, self.out),
                 lambda: eligibility.run_eligibility(ctx, self.out), lambda: baseline.run_baseline(ctx, "none.json", self.out))
        for call in calls:
            with self.assertRaisesRegex(PermitError, "holds kodex_grid results"):
                call()
        self.assertEqual(sorted(path.name for path in self.out.iterdir()), ["stage1"])
        for name in ("run", "selftest", "baseline", "eligibility"):
            self.assertNotIn(name, ("stage1", "stage2"))


class InnermostBoundary(_Sealed):
    """입력 적재·단위 계산·봉인·열람은 허가와 작성자 잠금을 확인한 뒤에만 한다."""

    def calls(self, plan):
        unit = plan.units[0]
        return (plan.load_inputs, lambda: plan.compute(unit, Bench()), lambda: plan.seal(unit, 0.0),
                lambda: plan.verified(unit), lambda: batch.run_units(plan, 1, self.lines.append),
                lambda: batch.read_units(plan))

    def test_without_the_writer_lock_nothing_is_read_or_written(self):
        out = self.copy()
        before = sorted(str(path) for path in out.rglob("*"))
        for call in self.calls(batch.Plan(self.ctx, out)):
            with self.assertRaisesRegex(PermitError, "writer lock"):
                call()
        self.assertEqual((sorted(str(path) for path in out.rglob("*")), self.lines), (before, []))

    def test_registered_inputs_without_evidence_stop_before_any_load(self):
        """등록한 실제 묶음으로 표시된 문맥은 증거 없이는 잠금을 잡고 있어도 입력을 읽지 못한다."""
        source = MemorySource(world_inputs(shapes={s: SHAPES[s] for s in SMALL_MIX.symbols()}), "claimed", frozen=True)
        source.load = lambda *key: self.fail("an input was loaded without a permit")
        grid = grid_identity._context(source, WORLD_SPEC, WORLD_BASE)
        ctx = identity._context(grid, SMALL_MIX)
        self.assertTrue(ctx.registered)
        out = self.out / "claimed"
        for call in (lambda: batch.run(ctx, out), lambda: batch.assemble(ctx, out)):
            with self.assertRaises(IdentityError):
                call()
        with sealing.writer_lock(out):
            for call in self.calls(batch.Plan(ctx, out)):
                with self.assertRaises(IdentityError):
                    call()
        self.assertEqual(sorted(path.name for path in out.rglob("*")), [".writer.lock"])
        with self.assertRaisesRegex(IdentityError, "does not take the registered bundle"):
            open_synthetic_context(source, mix=SMALL_MIX, spec=WORLD_SPEC, base=WORLD_BASE)


# ---------------------------------------------------------------------------
# 본실행 허가·공개값 재현·자격 집계: 합성 M5 묶음을 '등록한 실제 묶음' 자리에 끼워 입력기·사전 대조·허가를 그대로 거친다
# ---------------------------------------------------------------------------
# 합성 M5 묶음의 달력에는 2017년 1월만 다 있어 2017 묶음 하나로 등록 경로를 시험한다(혼합 둘: 기준 상품을 담은 혼합 포함)
BUNDLE_MIX = replace(MIX, products=MIX.products[:3], rows_expected={"2017": 180, "total": 180},
                     cohorts=(replace(FIRST, mixes=(FIRST.mixes[0], Mix("069500+229200", (SYMBOL, KOSDAQ)))),))
POINTS = ("069500@2011-01-03", "229200@2017-01-02", "169950@2017-01-02")
_BUNDLE = {}  # 합성 묶음과 그 정식 결과 꼴(시험 모듈 전체에서 한 번만 만든다)


class _Registered(_Workspace):
    """합성 묶음과 그 정식 결과 꼴 파일(승인된 formal.run_formal로 만든다)은 묶음마다 한 번만 만든다. 등록 입력값과 고정 정의의
    정식 결과 해시·명단만 시험용으로 바꾸고, 시험 묶음 자리에는 작은 시험 모듈 하나를 끼운다."""

    @classmethod
    def setUpClass(cls):
        if not _BUNDLE:
            shared = tempfile.TemporaryDirectory()
            atexit.register(shared.cleanup)
            root = Path(shared.name) / "bundle"
            manifest = synthetic_bundle(root, synthetic=False)
            loader._source.cache_clear()
            record = plain(formal.run_formal(root, manifest, FROZEN, registered=formal.actual_inputs(root, manifest, FROZEN)))
            loader._source.cache_clear()
            _BUNDLE.update(root=root, manifest=manifest, formal_record=record)
        cls.root, cls.manifest, cls.formal_record = _BUNDLE["root"], _BUNDLE["manifest"], _BUNDLE["formal_record"]

    def setUp(self):
        super().setUp()
        self.module = f"mm_probe_{self.id().rsplit('.', 1)[-1]}"
        probe = self.out / "probe" / f"{self.module}.py"
        probe.parent.mkdir()
        probe.write_text(self.PROBE_TEXT)
        sys.path.insert(0, str(probe.parent))
        self.addCleanup(sys.path.remove, str(probe.parent))
        self.addCleanup(sys.modules.pop, self.module, None)
        self.enterContext(patch.object(permit, "TEST_MODULES", (self.module,)))

    PROBE_TEXT = PROBE

    def formal_file(self, change=None):
        record = json.loads(json.dumps(self.formal_record))
        if change is not None:
            change(record)
        path = self.out / "formal.json"
        path.write_text(json.dumps(record, sort_keys=True))
        return path

    def registered(self, change=None, bundle=None):
        """(문맥, 정식 결과 파일). bundle: 다른 합성 묶음의 (루트, 명세)."""
        root, manifest = bundle or (self.root, self.manifest)
        actual = formal.actual_inputs(root, manifest, FROZEN)
        path = self.formal_file(change)
        leaves = dict(zip(sorted(POINTS), (628, 530, 530)))  # 합성 M5 묶음의 정식 기록 잎 수(실자료 802·712·596에 해당)
        mix = replace(BUNDLE_MIX, approved={**MIX.approved, "formal_result_sha256": sealing.file_digest(path)},
                      baseline={**BUNDLE_MIX.baseline, "leaves": leaves})
        self.enterContext(patch.object(grid_identity, "registered_inputs", return_value=actual))
        self.enterContext(patch.object(identity, "frozen_definition", return_value=mix))
        loader._source.cache_clear()
        return open_context(MixSource(root, manifest, FROZEN, symbols=mix.symbols(), registered=actual)), path

    def permitted(self, eligible=True, **options):
        ctx, path = self.registered(**options)
        out = self.out / "run"
        permit.run_selftest(out)
        record = baseline.run_baseline(ctx, path, out, workers=1)
        if eligible:
            eligibility.run_eligibility(ctx, out)
        return ctx, out, record

    def retouch(self, out, change):
        """재현 기록을 고치고 자기 해시까지 맞춘다(필수 항목 검사에 닿게 하는 시험용)."""
        path = out / "baseline" / "compare.json"
        record = json.loads(path.read_text())
        change(record)
        path.write_text(sealing.dump(sealing.with_seal(record)))


class MainRunPermit(_Registered):
    def test_no_evidence_refuses_run_and_assembly_before_any_load(self):
        ctx, _ = self.registered()
        self.assertEqual((ctx.registered, ctx.identity["scope"]), (True, "registered"))
        with patch.object(MixSource, "load", side_effect=AssertionError("input loaded")):
            for call in (batch.run, batch.assemble):
                with self.assertRaisesRegex(PermitError, "selftest/record.json is missing"):
                    call(ctx, self.out / "run")
            permit.run_selftest(self.out / "run")
            for call in (batch.run, batch.assemble):
                with self.assertRaisesRegex(PermitError, "baseline/compare.json is missing"):
                    call(ctx, self.out / "run")
        self.assertFalse((self.out / "run" / "run").exists())

    def test_complete_evidence_permits_the_run_and_the_assembly(self):
        ctx, out, record = self.permitted()
        self.assertTrue(record["main_run_allowed"])
        batch.run(ctx, out, workers=4, echo=self.lines.append)
        self.assertEqual(len(self.lines), 2 * 3 * 36 + 2 * 3 * 5)  # 세 상품: 단독·몫 108씩 + 28번 추가 30
        result = batch.assemble(ctx, out)
        self.assertEqual((len(result["rows"]), result["identity"]["scope"], result["formal_run"]), (180, "registered", False))
        unit = json.loads((out / "run" / "units" / "share-169950-2017-01-02-base-setting-28.json").read_text())
        provenance = unit["result"]["head"]["input_provenance"]
        self.assertEqual((provenance["manifest_synthetic"], provenance["source_sha256"], unit["identity"]["end"]),
                         (False, MIX.approved["source_sha256"], "2026-11-13"))
        self.assertEqual([part["segment"] for part in unit["result"]["paths"][OHLC]["strategy"]["segments"]],
                         ["2014-2019", "2020-2023", "2024-2026-11-13"])  # 단위 안에서는 고정 설정의 구간 이름 그대로
        row = next(item for item in result["rows"] if item["key"] == "229200+169950/28@2017-01-02/base")
        self.assertEqual(list(row["paths"][PAIRS[0]]["segments"]), ["2017-2019", "2020-2023", "2024-2026-11-13"])

    def test_tampered_or_incomplete_evidence_is_refused(self):
        ctx, out, _ = self.permitted()
        permit.enter(ctx, out)
        original = (out / "baseline" / "compare.json").read_bytes()
        point = POINTS[1]
        changes = {
            "mismatch": lambda r: r["points"][point]["comparison"]["mismatches"].append({"where": "x", "kind": "value"}),
            "fewer_leaves": lambda r: r["points"][point]["comparison"].update(matched=1),
            "item_missing": lambda r: r["points"][point]["comparison"]["by_item"].pop("matched_baseline"),
            "check_false": lambda r: r["points"][point]["self_checks"].update(series_reproduce_metrics=False),
            "check_missing": lambda r: r["points"][point]["self_checks"].pop("one_share_mix_equals_solo"),
            "point_missing": lambda r: r["points"].pop(POINTS[0]),
            "scaled_down": lambda r: r["points"][point]["comparison"].update(
                matched=11, formal_leaves=11, exploration_leaves=11,
                by_item={name: 1 for name in permit.REQUIRED_ITEMS}),
            "other_number": lambda r: r["points"][point].update(number=27),
            "one_cost_sealed": lambda r: r["points"][point]["sealed_units"].pop("stress"),
            "renamed": lambda r: r["points"][point].update(start="2017-01-09"),
            "not_allowed": lambda r: r.update(main_run_allowed=False),
            "other_formal": lambda r: r["formal_result"].update(sha256="0" * 64),
            "other_tests": lambda r: r.update(selftest_seal="0" * 64),
            "other_inputs": lambda r: r["identity"]["inputs"]["actual"].update(daily_sha256="0" * 64),
        }
        for name, change in changes.items():
            with self.subTest(name):
                (out / "baseline" / "compare.json").write_bytes(original)
                self.retouch(out, change)
                with self.assertRaisesRegex(PermitError, "does not permit the main run"):
                    batch.run(ctx, out, echo=self.lines.append)
        self.assertIn(b'"workers": 1', original)
        (out / "baseline" / "compare.json").write_bytes(original.replace(b'"workers": 1', b'"workers": 2'))
        with self.assertRaisesRegex(PermitError, "not an intact"):
            permit.enter(ctx, out)  # 자기 해시까지 맞추지 않은 손질
        (out / "baseline" / "compare.json").write_bytes(original)
        sealed_unit = out / "baseline" / "sealed" / f"{point}-base.unit.json"
        sealed_unit.write_bytes(sealed_unit.read_bytes() + b" ")
        with self.assertRaisesRegex(PermitError, "sealed baseline unit base is missing or changed"):
            permit.enter(ctx, out)
        self.assertEqual(self.lines, [])

    def test_eligibility_record_of_this_version_is_required(self):
        """자격 검산이 실행되지 않았거나, 규칙·시작일 오프셋이 성립하지 않았거나, 다른 판본의 기록이면 시작하지 않는다."""
        ctx, out, _ = self.permitted(eligible=False)
        with self.assertRaisesRegex(PermitError, "eligibility/record.json is missing"):
            batch.run(ctx, out, echo=self.lines.append)
        eligibility.run_eligibility(ctx, out)
        permit.enter(ctx, out)
        path = out / "eligibility" / "record.json"
        original = path.read_bytes()
        changes = {"offsets": (lambda r: r.update(sensitivity_starts_are_offsets=False), "session offsets"),
                   "rule_failed": (lambda r: r.update(start_rules=[{"cohort": "x", "holds": False}]), "start rules"),
                   "identity": (lambda r: r["identity"].update(mix_source_sha256="0" * 64), "identity")}
        for name, (change, message) in changes.items():
            with self.subTest(name):
                record = json.loads(original)
                change(record)
                path.write_text(sealing.dump(sealing.with_seal(record)))
                with self.assertRaisesRegex(PermitError, message):
                    batch.run(ctx, out, echo=self.lines.append)
        self.assertEqual(self.lines, [])
        declared = replace(ctx, mix=replace(ctx.mix, cohorts=(replace(ctx.mix.cohorts[0], start_rule=dict(RULE)),)))
        path.write_bytes(original)
        with self.assertRaisesRegex(PermitError, "start rules of the declared cohorts are missing"):
            permit.check_eligibility(declared, out)
        record = json.loads(original)
        record["start_rules"] = [{"cohort": declared.mix.cohorts[0].id, "holds": False}]
        path.write_text(sealing.dump(sealing.with_seal(record)))
        with self.assertRaisesRegex(PermitError, "does not satisfy its start rule"):
            permit.check_eligibility(declared, out)

    def test_registered_run_needs_the_registered_kodex_grid_version(self):
        ctx, _ = self.registered()
        identity.check_frozen(ctx)
        other = replace(ctx, mix=replace(ctx.mix, approved={**ctx.mix.approved, "kodex_grid_source_sha256": "0" * 64}))
        with self.assertRaisesRegex(IdentityError, "kodex_grid source is not the registered r3 version"):
            identity.check_frozen(other)
        source = replace(ctx, grid=replace(ctx.grid, source=MemorySource({}, "x")))
        with self.assertRaisesRegex(IdentityError, "pinned loader of the declared products|pinned bundle loader"):
            identity.check_frozen(source)

    def test_evidence_of_another_version_is_refused(self):
        ctx, out, _ = self.permitted()
        with patch.object(identity, "mix_source", return_value={"mix_source_sha256": "0" * 64}):
            with self.assertRaisesRegex(IdentityError, "does not describe the current code"):
                batch.run(ctx, out, echo=self.lines.append)
        probe = Path(sys.modules[self.module].__file__)
        probe.write_text(self.PROBE_TEXT + "\n# changed\n")  # 시험 파일이 바뀌면 지난 판본의 증거다
        with self.assertRaisesRegex(PermitError, "differ from this version"):
            batch.run(ctx, out, echo=self.lines.append)
        with self.assertRaisesRegex(PermitError, "differ from this version"):
            baseline.run_baseline(ctx, self.out / "formal.json", self.out / "run")


class BaselineReproduction(_Registered):
    def test_three_public_points_match_leaf_by_leaf(self):
        ctx, out, record = self.permitted()
        self.assertEqual((sorted(record["points"]), record["exploration"], record["formal_run"]), (sorted(POINTS), True, False))
        for name, point in record["points"].items():
            comparison = point["comparison"]
            self.assertEqual((comparison["mismatches"], comparison["matched"]),
                             ([], comparison["formal_leaves"]), name)
            self.assertEqual(comparison["matched"], comparison["exploration_leaves"])
            self.assertTrue(all(comparison["by_item"][item] > 0 for item in permit.REQUIRED_ITEMS))
            self.assertEqual(point["self_checks"], {check: True for check in permit.REQUIRED_CHECKS})
            self.assertEqual(point["timing"]["by_kind_counts"]["split"], 4)  # 두 비용 × 두 경로
            for cost in (BASE, STRESS):
                sealed_unit = out / "baseline" / "sealed" / f"{name}-{cost}.unit.json"
                self.assertEqual((sealing.file_digest(sealed_unit), sealed_unit.stat().st_mode & 0o777),
                                 (point["sealed_units"][cost], 0o600))
        self.assertGreater(record["points"][POINTS[0]]["comparison"]["matched"],
                           record["points"][POINTS[1]]["comparison"]["matched"])  # 2011 시작은 구간이 하나 더 있다
        plan = record["estimate"]
        self.assertEqual((plan["units"], plan["paths_per_unit"], plan["exceeds_report_threshold"]), (246, 2, False))
        public = (out / "baseline" / "compare.json").read_text() + (out / "baseline" / "report.md").read_text()
        self.assertNotIn("cagr", public.replace("cagr_edge", ""))  # 공개 기록에는 일치 수와 성립 여부만 있다
        self.assertIn("본실행 허용: **예**", (out / "baseline" / "report.md").read_text())
        with self.assertRaisesRegex(sealing.OverwriteError, "already recorded"):
            baseline.run_baseline(ctx, self.out / "formal.json", out)

    def test_a_changed_public_value_is_a_listed_mismatch_and_forbids_the_run(self):
        def change(record):
            start = record["rows"]["extension"]["symbols"][KOSDAQ]["starts"]["2017-01-02"]
            side = start["items"]["M"]["paths"][OHLC]["base"]["strategy"]
            side["ending_equity_krw"] = str(D(side["ending_equity_krw"]) + 1)

        ctx, out, record = self.permitted(change=change)
        self.assertFalse(record["main_run_allowed"])
        bad, good = record["points"][POINTS[1]], record["points"][POINTS[2]]
        self.assertEqual(bad["comparison"]["mismatches"],
                         [{"where": f"base/{OHLC}/strategy/ending_equity_krw", "kind": "value"}])
        self.assertEqual((bad["self_checks"]["unit_values_equal_formal"], good["comparison"]["mismatches"],
                          good["self_checks"]["unit_values_equal_formal"]), (False, [], True))
        self.assertIn("본실행 금지", (out / "baseline" / "report.md").read_text())
        self.assertNotIn("ending_equity_krw\":", (out / "baseline" / "compare.json").read_text())  # 값은 봉인 내역에만 있다
        with self.assertRaisesRegex(PermitError, "did not allow the main run|mismatches"):
            batch.run(ctx, out, echo=self.lines.append)
        self.assertEqual(self.lines, [])

    def test_formal_file_hash_is_checked_before_any_ledger(self):
        ctx, path = self.registered()
        permit.run_selftest(self.out / "run")
        path.write_text(path.read_text() + " ")
        with patch.object(MixSource, "load", side_effect=AssertionError("input loaded")):
            with self.assertRaisesRegex(IdentityError, "differs from the registered SHA-256"):
                baseline.run_baseline(ctx, path, self.out / "run")
        self.assertFalse((self.out / "run" / "baseline" / "compare.json").exists())

    def test_baseline_needs_the_test_record_first_and_parallel_equals_sequential(self):
        ctx, path = self.registered()
        with self.assertRaisesRegex(PermitError, "selftest/record.json is missing"):
            baseline.run_baseline(ctx, path, self.out / "a")
        records_found = []
        for name, workers in (("a", 1), ("b", 3)):
            permit.run_selftest(self.out / name)
            records_found.append(baseline.run_baseline(ctx, path, self.out / name, workers=workers))
        for name in POINTS:
            one, many = (record["points"][name] for record in records_found)
            self.assertEqual((one["comparison"], one["self_checks"], one["sealed_units"]),
                             (many["comparison"], many["self_checks"], many["sealed_units"]))

    def test_engine_error_leaves_only_a_technical_code(self):
        ctx, path = self.registered()
        permit.run_selftest(self.out / "run")
        with patch.object(baseline, "open_unit", side_effect=RuntimeError("secret detail 12.5%")):
            with self.assertRaisesRegex(baseline.BaselineError, "E_ENGINE") as caught:
                baseline.run_baseline(ctx, path, self.out / "run", workers=1)
        self.assertNotIn("secret", str(caught.exception))
        self.assertFalse((self.out / "run" / "baseline" / "compare.json").exists())
        self.assertIn("secret detail", (self.out / "run" / "baseline" / "sealed" / f"{POINTS[0]}.output.log").read_text())

    def test_estimate_is_conditional(self):
        plan = baseline.estimate(336.0, 410, cpus=8)
        # 권장 동시 실행 6개: ceil(410 ÷ 6) = 69묶음 × 336초 = 6.44시간
        self.assertEqual((plan["recommended_workers"], plan["units"], plan["exceeds_report_threshold"]), (6, 410, False))
        self.assertAlmostEqual(plan["conditional_hours"], 69 * 336 / 3600)
        self.assertIn("상한이 아니다", plan["conditions"])
        self.assertTrue(baseline.estimate(1500.0, 410, cpus=8)["exceeds_report_threshold"])  # 28.75시간
        self.assertIsNone(baseline.estimate(None, 410, cpus=8)["conditional_hours"])
        points = {"a": {"start": "2017-01-02", "timing": {"total_seconds": 110.0, "by_kind_seconds": {"load_inputs": 10.0}}},
                  "b": {"start": "2011-01-03", "timing": {"total_seconds": 900.0, "by_kind_seconds": {}}}}
        self.assertEqual(baseline.unit_seconds(points, "2017-01-02"), 50.0)  # (110 − 10) ÷ 두 비용


def thin_volume(symbol, volume):
    def mutate(found):
        for row in found:
            if row["symbol"] == symbol:
                row["volume"] = volume
    return mutate


class EligibilityTally(_Registered):
    """자격 집계는 통과 여부와 사유의 종류만 낸다(장부를 열지 않는다)."""

    def test_every_product_start_and_account_is_listed_without_values(self):
        ctx, _ = self.registered()
        opened = AssertionError("ledger opened")
        with patch("src.research.etf.kodex_grid.runner.run_split", side_effect=opened), \
                patch("src.research.etf.kodex_grid.runner.run_hold", side_effect=opened):
            record = eligibility.run_eligibility(ctx, self.out / "run")
        self.assertEqual((record["checked_units"], record["eligible_units"], record["performance_reported"]), (18, 18, False))
        self.assertEqual(record["sensitivity_starts_are_offsets"], True)
        self.assertEqual({(row["symbol"], row["start"], row["kind"]) for row in record["units"]},
                         {(symbol, start.isoformat(), kind) for symbol in BUNDLE_MIX.symbols()
                          for start in BUNDLE_MIX.starts() for kind in (SOLO, SHARE)})
        self.assertEqual({row["floor_applies"] for row in record["units"] if row["symbol"] == SYMBOL}, {False})
        self.assertEqual(len(record["subjects"]), 5 * 3)  # (단독 3 + 혼합 2) × 시작일 3
        self.assertEqual((record["blank_rows"]["primary_rows"], record["blank_rows"]["sensitivity_rows"], record["start_rules"]),
                         (0, 0, []))
        allowed = {"symbol", "start", "kind", "account_krw", "floor_applies", "input_loaded", "bars_to_end",
                   "limit_at_least_account", "eligible", "reasons"}
        self.assertEqual({key for row in record["units"] for key in row}, allowed)  # 한도 금액·가격·수익률 칸이 없다
        text = (self.out / "run" / "eligibility" / "report.md").read_text()
        self.assertIn("자격 있는 (상품, 시작일, 원금): 18 / 18", text)
        with self.assertRaisesRegex(sealing.OverwriteError, "already here"):
            eligibility.run_eligibility(ctx, self.out / "run")

    def test_limit_between_the_two_accounts_fails_only_the_solo_account(self):
        """거래량 120,000주: 직전 20일 평균 거래대금 12.06억 원 × 5% = 6,030만 원. 몫(5천만)은 통과, 단독(1억)은 미달."""
        root = self.out / "thin"
        manifest = synthetic_bundle(root, synthetic=False, mutate=thin_volume(CHINA, 120000))
        ctx, _ = self.registered(bundle=(root, manifest))
        record = eligibility.run_eligibility(ctx, self.out / "run")
        china = {(row["start"], row["kind"]): row for row in record["units"] if row["symbol"] == CHINA}
        for start in BUNDLE_MIX.starts():
            solo, share = china[(start.isoformat(), SOLO)], china[(start.isoformat(), SHARE)]
            self.assertEqual((solo["eligible"], solo["reasons"], solo["limit_at_least_account"]),
                             (False, ["limit_below_account"], False))
            self.assertEqual((share["eligible"], share["reasons"]), (True, []))
        mixes = {row["eligible"] for row in record["subjects"] if row["kind"] == "mix"}
        self.assertEqual((record["eligible_units"], mixes), (15, {True}))
        # 단독 169950의 세 시작일이 빈다: 주 판정 36행, 민감도 28번 세 시작일 × 두 비용 − 주 판정과 겹친 1행 = 5행
        self.assertEqual((record["blank_rows"]["primary_rows"], record["blank_rows"]["sensitivity_rows"]), (36, 5))
        self.assertNotIn("60300000", (self.out / "run" / "eligibility" / "record.json").read_text())

    def test_data_holds_are_reduced_to_a_kind(self):
        self.assertEqual(eligibility.hold_code("calendar/history insufficient for 2017-01-02 volatility: x 9,900"),
                         "history_251_insufficient")
        self.assertEqual(eligibility.hold_code("2017-01-02: 20 prior domestic sessions are required for the liquidity limit"),
                         "liquidity_window_insufficient")
        self.assertEqual(eligibility.hold_code("229200 2017-01-02 close: nonpositive value 0"), "input_data_hold")
        gap = with_gap(market(CHINA, SHAPES[CHINA]), WORLD_DAYS[25])
        ctx = world_context(held=(NASDAQ,), **{f"s{CHINA}": gap})
        body, sealed_text = eligibility.collect(ctx)
        found = {(row["symbol"], row["kind"], row["start"]): row for row in body["units"]}
        nasdaq, china = found[(NASDAQ, SHARE, B_START)], found[(CHINA, SOLO, "2025-01-02")]
        self.assertEqual((nasdaq["reasons"], nasdaq["input_loaded"]), (["input_data_hold"], False))
        self.assertEqual((china["reasons"], china["bars_to_end"]), (["data_gap_after_start"], False))
        self.assertNotIn((CHINA, SOLO, B_START), found)  # 묶음에 없는 상품·시작일은 보지 않는다
        self.assertIn((SYMBOL, SOLO, B_START), found)  # B 묶음의 기준 보유(KODEX 200 1억 원)는 본다
        subjects = {(row["subject"], row["start"]): row["eligible"] for row in body["subjects"] if row["primary"]}
        self.assertEqual(subjects, {(SYMBOL, "2025-01-02"): True, (KOSDAQ, "2025-01-02"): True, (CHINA, "2025-01-02"): False,
                                    ("229200+169950", "2025-01-02"): False, (NASDAQ, B_START): False,
                                    ("069500+133690", B_START): False})
        self.assertEqual(body["blank_rows"]["primary_rows"], 4 * 36)
        self.assertEqual(sealed_text[f"133690@{B_START}"], "synthetic: no input for this product")  # 문구는 봉인 쪽에만
        self.assertNotIn("synthetic", json.dumps(body))

    def test_start_rule_is_the_first_year_start_meeting_both_accounts(self):
        """시작일 규칙: 그해 첫 거래일이고, 그때 알 수 있는 251일 이력과 두 원금 한도를 채우며, 그 전 해들의 첫 거래일은
        채우지 못해야 한다. 합성 달력은 한 해뿐이라 '전 해의 첫 거래일'을 1/3으로 바꿔 넣는다(1/2 시작 묶음)."""
        mix = rule_mix(J(2), since=2024)
        firsts = ({2024: WORLD_DAYS[1], 2025: J(2)}, [])
        thin = market(NASDAQ, SHAPES[NASDAQ], changes={WORLD_DAYS[1]: {"adv20": D(1500000000)}})  # 1/3만 7,500만 원
        def context(nasdaq):
            inputs = world_inputs(shapes={symbol: SHAPES[symbol] for symbol in (SYMBOL, NASDAQ)}, mix=mix,
                                  **{f"s{NASDAQ}": nasdaq})
            inputs[(NASDAQ, WORLD_DAYS[1])] = nasdaq
            return open_synthetic_context(MemorySource(inputs, "rule"), mix=mix, spec=WORLD_SPEC, base=WORLD_BASE)
        with patch.object(eligibility, "_year_firsts", return_value=firsts):
            holds = eligibility.collect(context(thin))[0]["start_rules"]
            fails = eligibility.collect(context(market(NASDAQ, SHAPES[NASDAQ])))[0]["start_rules"]
        self.assertEqual(holds[0]["earlier"], [{"start": WORLD_DAYS[1].isoformat(), "eligible": False,
                                                "reasons": ["limit_below_account"], "bars_to_end": True, "later_reasons": []}])
        self.assertEqual((holds[0]["holds"], holds[0]["chosen"]["eligible"], holds[0]["chosen"]["first_session_of_year"]),
                         (True, True, True))
        self.assertEqual((fails[0]["holds"], fails[0]["earlier"][0]["eligible"]), (False, True))
        self.assertEqual(eligibility.collect(world_context())[0]["start_rules"], [])  # 규칙을 선언한 묶음만 본다

    def test_equal_limit_is_allowed(self):
        ctx = world_context()
        for adv20, expected in ((1000000000, True), (999999980, False)):
            data = as_symbol(flat({J(2): "10000"}, changes={J(2): {"adv20": D(adv20)}}), NASDAQ)
            row = eligibility.unit_row(ctx, Unit(SHARE, NASDAQ, J(2), 28, BASE), data)
            self.assertEqual((row["limit_at_least_account"], row["eligible"]), (expected, expected))
        # 기준 상품(KODEX 200)에는 이 문턱을 적용하지 않는다: 한도가 원금보다 작아도 그 사유로 자격을 잃지 않는다
        low = flat({J(2): "10000"}, changes={J(2): {"adv20": D(999999980)}})
        row = eligibility.unit_row(ctx, Unit(SHARE, SYMBOL, J(2), 28, BASE), low)
        self.assertEqual((row["floor_applies"], row["limit_at_least_account"], row["eligible"], row["reasons"]),
                         (False, False, True, []))


class SyntheticTestRecord(_Registered):
    def test_record_of_a_passing_run_and_refusal_of_hand_written_ones(self):
        record = permit.run_selftest(self.out / "run")
        self.assertEqual((record["passed"], record["tests_run"], record["modules"]), (True, 2, {self.module: 2}))
        self.assertEqual(set(record["version"]), {"approved_source_sha256", "kodex_grid_source_sha256", "mix_source_sha256",
                                                  "grid_file_sha256", "definition_file_sha256", "test_files"})
        self.assertEqual(permit.run_selftest(self.out / "run"), record)  # 다시 돌리지 않고 같은 기록을 검사한다
        path = self.out / "run" / "selftest" / "record.json"
        path.write_text(json.dumps({"schema": permit.SELFTEST_SCHEMA, "passed": True}))
        with self.assertRaisesRegex(PermitError, "not an intact"):
            permit.check_selftest(self.out / "run")
        path.write_text(sealing.dump(sealing.with_seal({**record, "tests_run": 1})))
        with self.assertRaisesRegex(PermitError, "recorded test counts differ"):
            permit.check_selftest(self.out / "run")

    def test_real_test_modules_and_counts(self):
        self.assertEqual(permit.TEST_MODULES, (self.module,))
        with patch.object(permit, "TEST_MODULES", ("tests.test_research_etf_markets_mix",)):
            counted = permit.count_tests(permit.test_file("tests.test_research_etf_markets_mix"))
            loaded = unittest.defaultTestLoader.loadTestsFromName("tests.test_research_etf_markets_mix").countTestCases()
        self.assertEqual(counted, loaded)


class FailingProbe(_Registered):
    PROBE_TEXT = PROBE.replace("self.assertEqual(1 + 1, 2)", "self.assertEqual(1 + 1, 3)")

    def test_failed_run_is_recorded_as_failed_and_permits_nothing(self):
        ctx, path = self.registered()
        with self.assertRaisesRegex(PermitError, "did not pass cleanly"):
            permit.run_selftest(self.out / "run")
        self.assertFalse(json.loads((self.out / "run" / "selftest" / "record.json").read_text())["passed"])
        for call in (lambda: batch.run(ctx, self.out / "run"), lambda: baseline.run_baseline(ctx, path, self.out / "run")):
            with self.assertRaisesRegex(PermitError, "did not pass cleanly"):
                call()


class CommandLine(_Registered):
    def test_arguments_and_commands(self):
        self.assertEqual(cli.COMMANDS, ("selftest", "baseline", "eligibility", "run", "assemble", "eta"))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cli._parser().parse_args(["stage1", "--out", "c"])
        args = cli._parser().parse_args(["run", "--data-root", "a", "--manifest", "b", "--out", "c"])
        self.assertEqual((args.workers, args.formal_result), (1, None))
        with self.assertRaisesRegex(SystemExit, "--formal-result"):
            cli._baseline(None, args)
        for command in ("baseline", "eligibility", "run", "assemble"):
            with self.assertRaisesRegex(SystemExit, f"{command} needs --data-root and --manifest"):
                cli.main([command, "--out", str(self.out)])

    def test_unregistered_bundle_stops_before_any_ledger(self):
        with patch.object(batch, "run_unit", side_effect=AssertionError("ledger opened")):
            with self.assertRaises(formal.InputIntegrityError):  # 합성 묶음은 등록값과 달라 사전 대조에서 멈춘다
                cli.main(["run", "--data-root", str(self.root), "--manifest", str(self.manifest), "--out", str(self.out / "o")])
        self.assertFalse((self.out / "o").exists())

    def test_selftest_and_eta_commands_read_no_bundle(self):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.assertEqual(cli.main(["selftest", "--out", str(self.out / "run")]), 0)
        self.assertIn("selftest: passed tests_run=2", printed.getvalue())
        log = self.out / "run" / "run" / "progress.log"
        log.parent.mkdir(parents=True)
        log.write_text("2026-11-25T00:00:00Z run solo-069500-2017-01-02-base-setting-01 done 300.0s\n"
                       "2026-11-25T00:00:00Z run solo-069500-2017-01-02-base-setting-02 done 100.0s\n"
                       "2026-11-25T00:00:00Z run solo-069500-2017-01-02-base-setting-03 error 1.0s E_ENGINE\n")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.assertEqual(cli.main(["eta", "--out", str(self.out / "run"), "--workers", "6"]), 0)
        found = json.loads(printed.getvalue())
        # 334단위 중 2개 완료: 남은 332개 ÷ 6 = 56묶음 × 평균 200초 = 3.11시간
        self.assertEqual((found["planned"], found["finished"], found["remaining"], found["in_error"]),
                         (334, 2, 332, ["solo-069500-2017-01-02-base-setting-03"]))
        self.assertAlmostEqual(found["remaining_hours"], 56 * 200 / 3600)
        log.write_text(log.read_text() + "note: cagr 3%\n")
        with self.assertRaisesRegex(ValueError, "not a technical progress record"):
            cli.main(["eta", "--out", str(self.out / "run")])


# ---------------------------------------------------------------------------
# 구현 v2 리뷰 합의(review-reconciled-impl-v2.md) 고칠 것 1~7의 합성 반례
# ---------------------------------------------------------------------------
def fake_rows(keys, status=judge.INFERIOR):
    """주 판정 행 열쇠에서 만든 최소 행(조립 뒤의 꼴)."""
    rows = []
    for key in keys:
        subject, rest = key.split("/", 1)
        number, start = rest.split("/")[0].split("@")
        rows.append({"key": key, "subject": subject, "number": int(number), "start": start, "status": status,
                     "signs": {}, "paths": {}})
    return rows


class Fix1DuplicatedActualRow(unittest.TestCase):
    def test_an_actual_row_given_twice_is_refused(self):
        """열위 216행에 첫 행을 한 번 더 넣으면, 사전으로 바꾸기 전에 멈춰야 한다(첫째 문장을 내지 않는다)."""
        ctx = open_synthetic_context(MemorySource({}, "frame"), mix=MIX)
        rows = fake_rows(summary.expected_keys(ctx))
        self.assertEqual(summary.conclusion(ctx, rows)["kind"], judge.ALL_INFERIOR)
        for broken in (rows + rows[:1], rows[:-1] + rows[:1]):
            with self.assertRaisesRegex(ValueError, "exactly once"):
                summary.conclusion(ctx, broken)


RULE = {"kind": "first_year_first_session_with_history_and_limits", "symbol": NASDAQ, "search_from_year": 2025,
        "accounts": [SOLO, SHARE]}


def rule_mix(start, since=2025):
    return world_mix(cohort("B", start, (NASDAQ,), [(SYMBOL, NASDAQ)], rule={**RULE, "search_from_year": since}),
                     products=(SYMBOL, NASDAQ))


class Fix2StartRule(unittest.TestCase):
    def rule(self, mix, **overrides):
        inputs = world_inputs(shapes={symbol: SHAPES[symbol] for symbol in (SYMBOL, NASDAQ)}, mix=mix, **overrides)
        ctx = open_synthetic_context(MemorySource(inputs, "rule"), mix=mix, spec=WORLD_SPEC, base=WORLD_BASE)
        return eligibility.collect(ctx)[0]["start_rules"][0]

    def test_the_chosen_day_must_be_the_first_session_of_its_year(self):
        """합성 달력의 2025년 첫 거래일은 1/2다. 1/3을 '연초 거래일'로 선언하면 조건을 채워도 규칙이 성립하지 않는다."""
        found = self.rule(rule_mix(WORLD_DAYS[1]))
        self.assertFalse(found["holds"])
        self.assertFalse(found["chosen"]["first_session_of_year"])
        self.assertTrue(self.rule(rule_mix(J(2)))["holds"])

    def test_a_later_data_gap_does_not_decide_the_start(self):
        """시작일 선택에는 그때 알 수 있는 251일 이력과 두 원금 한도만 쓴다. 개설 뒤 결측은 따로 남긴다."""
        gap = with_gap(market(NASDAQ, SHAPES[NASDAQ]), WORLD_DAYS[25])
        found = self.rule(rule_mix(J(2)), **{f"s{NASDAQ}": gap})
        self.assertTrue(found["holds"])
        self.assertEqual((found["chosen"]["eligible"], found["chosen"]["bars_to_end"]), (True, False))
        self.assertEqual(found["chosen"]["later_reasons"], ["data_gap_after_start"])

    def test_every_year_from_the_search_start_must_have_a_first_session(self):
        """2024년부터 찾으라고 했는데 그해 거래일이 달력에 없으면 빠뜨리지 않고 규칙 실패로 남긴다."""
        found = self.rule(rule_mix(J(2), since=2024))
        self.assertFalse(found["holds"])
        self.assertEqual(found["missing_years"], [2024])


class Fix3ReferenceHold(_Sealed):
    REFERENCE = "solo-069500-2025-01-02-base-setting-28.json"

    def refused(self, out, message):
        with self.assertRaisesRegex(AssemblyError, message):
            batch.assemble(self.ctx, out)

    def test_changed_reference_hold_value_is_caught(self):
        out = self.copy()

        def change(record):
            hold = record["result"]["paths"][OHLC]["hold_only"]
            hold["cagr"] = str(D(hold["cagr"]) + D("0.001"))
        reseal(out / "run", self.REFERENCE, change)
        self.refused(out, "do not reproduce the sealed metrics")

    def test_hold_is_checked_on_a_path_where_only_the_strategy_is_blank(self):
        def blank(record, shift):
            row = record["result"]["paths"][OLHC]
            row.update(comparable=False, strategy=None, hold=None, delta=None,
                       reasons={"failures": [], "holds": ["base_combined_start_liquidity"], "later": []})
            row["series"]["strategy"] = None
            if shift:  # 80자리 값을 기본 28자리 덧셈으로 바꾸지 않도록, 옮길 때만 고친다
                row["hold_only"]["cagr"] = str(D(row["hold_only"]["cagr"]) + shift)
        out = self.copy("shifted")
        reseal(out / "run", self.REFERENCE, lambda record: blank(record, D("0.001")))
        self.refused(out, "do not reproduce the sealed metrics")
        out = self.copy("kept")  # 보유 값은 그대로 두고 전략만 비우면 조립된다(그 경로의 행은 빈칸)
        reseal(out / "run", self.REFERENCE, lambda record: blank(record, D(0)))
        row = next(item for item in batch.assemble(self.ctx, out)["rows"] if item["key"] == "069500/28@2025-01-02/base")
        self.assertEqual((row["status"], row["signs"][OLHC]), (judge.BLANK, None))


class Fix4DependenceIsThreeValued(unittest.TestCase):
    def test_yes_no_or_undetermined(self):
        starts = ("2017-01-02", "2017-01-09", "2017-02-01")
        table = {(start, cost): {OHLC: "-", OLHC: "-"} for start in starts for cost in (BASE, STRESS)}
        self.assertEqual((judge.sensitivity(table, PATHS)["start_dependent"], judge.sensitivity(table, PATHS)["cost_dependent"]),
                         (judge.NO, judge.NO))
        empty = {**table, **{(start, STRESS): {OHLC: None, OLHC: None} for start in starts}}
        found = judge.sensitivity(empty, PATHS)  # 높은 비용이 모두 빈칸이면 '비용 의존 아니오'가 아니다
        self.assertEqual((found["start_dependent"], found["cost_dependent"]), (judge.OPEN, judge.OPEN))
        moved = {**empty, ("2017-01-09", BASE): {OHLC: "+", OLHC: "-"}}
        found = judge.sensitivity(moved, PATHS)
        self.assertEqual((found["start_dependent"], found["cost_dependent"]), (judge.YES, judge.OPEN))
        self.assertEqual({report.MARK[value] for value in (judge.YES, judge.NO, judge.OPEN)}, {"예", "아니오", "미확정"})


class Fix5ReproductionEvidence(unittest.TestCase):
    def test_unfilled_reasons_are_compared_by_reason(self):
        formal_row = {"reentries": {BASE: [{"outcome": "filled"}, {"outcome": "participation_limit"}]},
                      "buys_closed_years": {BASE: []}}

        def slot(unfilled):
            mine = {"reentry_counts": {"attempts": 2, "filled": 1, "unfilled": unfilled}, "buys_closed_years": []}
            return baseline.Slot(BASE, OHLC, formal_row, mine)
        self.assertTrue(baseline._counts_equal(slot({"participation_limit": 1})))
        self.assertFalse(baseline._counts_equal(slot({"no_order": 1})))

    def test_registered_leaves_per_point(self):
        self.assertEqual(MIX.baseline["leaves"], {"069500@2011-01-03": 802, "169950@2017-01-02": 712,
                                                  "229200@2017-01-02": 596})


class Fix6SealsBeforeBodies(_Sealed):
    def test_a_missing_last_seal_stops_before_any_body_is_read(self):
        out = self.copy()
        plan = batch.Plan(self.ctx, out)
        last = plan.units[-1]
        (out / "run" / "sealed" / "seals" / f"{last.name}.json").unlink()
        real, bodies = Path.read_bytes, []

        def counted(path):
            if "units" in Path(path).parts:
                bodies.append(Path(path).name)
            return real(path)
        with patch.object(Path, "read_bytes", autospec=True, side_effect=counted):
            with self.assertRaisesRegex(AssemblyError, "seal record is missing"):
                batch.assemble(self.ctx, out)
        self.assertEqual(bodies, [])

    def test_a_seal_naming_another_unit_stops_before_any_body_is_read(self):
        out = self.copy()
        plan = batch.Plan(self.ctx, out)
        last, other = plan.units[-1], plan.units[0]
        path = out / "run" / "sealed" / "seals" / f"{last.name}.json"
        seal = json.loads(path.read_text())
        path.write_text(sealing.dump(sealing.with_seal({**seal, "unit": other.name})))
        real, bodies = Path.read_bytes, []

        def counted(file_path):
            if "units" in Path(file_path).parts:
                bodies.append(Path(file_path).name)
            return real(file_path)
        with patch.object(Path, "read_bytes", autospec=True, side_effect=counted):
            with self.assertRaisesRegex(AssemblyError, "seal record"):
                batch.assemble(self.ctx, out)
        self.assertEqual(bodies, [])


class Fix7CopyAbs(unittest.TestCase):
    def test_drawdown_deepening_keeps_digits_beyond_28(self):
        side = {"cagr": "0", "exposure_mean": "1", "ending_equity_krw": "1", "segments": []}
        strategy, hold = {**side, "mdd": "-0.1000000000000000000000000000000001"}, {**side, "mdd": "-0.1"}
        self.assertEqual(rows._valued(strategy, hold, [], trades=None)["mdd_deepening"], D("1E-34"))



# ---------------------------------------------------------------------------
# 구현 v2-r3 확인 합의(review-reconciled-impl-v2r3.md) 네 건의 합성 반례
# ---------------------------------------------------------------------------
def duplicate_row(symbol, day):
    def mutate(found):
        found.append(dict(next(row for row in found if row["symbol"] == symbol and row["date"] == day)))
    return mutate


class R4StartDayUsesOnlyWhatIsKnownAtTheOpen(_Registered):
    def test_a_duplicated_row_after_the_start_does_not_decide_the_start(self):
        """1/2 이전 이력과 두 원금 한도는 그대로 두고 1/9 행 하나만 중복한다. 1/2 시작의 자격은 통과로 남고,
        만기까지의 적재 오류는 따로 기록된다."""
        rule = {**RULE, "symbol": KOSDAQ}
        clean = eligibility._rule_day(self.registered()[0], rule, date(2017, 1, 2))
        self.assertEqual((clean["eligible"], clean["bars_to_end"], clean["later_reasons"]), (True, True, []))
        root = self.out / "dup"
        manifest = synthetic_bundle(root, synthetic=False, mutate=duplicate_row(KOSDAQ, date(2017, 1, 9)))
        found = eligibility._rule_day(self.registered(bundle=(root, manifest))[0], rule, date(2017, 1, 2))
        self.assertEqual((found["eligible"], found["reasons"]), (True, []))
        self.assertEqual((found["bars_to_end"], found["later_reasons"]), (False, ["input_data_hold"]))


class R4HoldIsRequiredWhereTheHoldOrderFilled(_Sealed):
    REFERENCE, refused = Fix3ReferenceHold.REFERENCE, Fix3ReferenceHold.refused

    def test_removing_hold_series_and_hold_only_together_is_refused(self):
        """전략만 개설 거부되고 보유 주문은 체결된 경로에서 보유 계열과 hold_only를 함께 지우면 멈춘다."""
        def strip(record):
            row = record["result"]["paths"][OLHC]
            self.assertEqual(row["start_orders"]["orders"]["hold"]["status"], "filled")
            row.update(comparable=False, strategy=None, hold=None, delta=None, hold_only=None, series=None,
                       reasons={"failures": ["stress_start_rejected_by_cost_guard"], "holds": [], "later": []})
        out = self.copy("stripped")
        reseal(out / "run", self.REFERENCE, strip)
        self.refused(out, "do not reproduce the sealed metrics")


class R4ReentryWithoutAttempts(unittest.TestCase):
    def test_no_formal_attempts_means_all_three_fields_are_empty(self):
        formal_row = {"reentries": {STRESS: None}, "buys_closed_years": {STRESS: []}}

        def slot(counts):
            return baseline.Slot(STRESS, OHLC, formal_row, {"reentry_counts": counts, "buys_closed_years": []})
        self.assertTrue(baseline._counts_equal(slot({"attempts": None, "filled": None, "unfilled": None})))
        self.assertFalse(baseline._counts_equal(slot({"attempts": None, "filled": 5, "unfilled": {"no_order": 7}})))


class R4SealHashShape(_Sealed):
    def test_an_empty_body_hash_stops_before_any_body_is_read(self):
        out = self.copy()
        last = batch.Plan(self.ctx, out).units[-1]
        path = out / "run" / "sealed" / "seals" / f"{last.name}.json"
        seal = json.loads(path.read_text())
        path.write_text(sealing.dump(sealing.with_seal({**seal, "sha256": ""})))
        real, bodies = Path.read_bytes, []

        def counted(file_path):
            if "units" in Path(file_path).parts:
                bodies.append(Path(file_path).name)
            return real(file_path)
        with patch.object(Path, "read_bytes", autospec=True, side_effect=counted):
            with self.assertRaisesRegex(AssemblyError, "seal record"):
                batch.assemble(self.ctx, out)
        self.assertEqual(bodies, [])


if __name__ == "__main__":
    unittest.main()
