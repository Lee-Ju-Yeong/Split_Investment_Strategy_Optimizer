"""KODEX 200 변수 지형 조사(kodex_grid) 합성 시험 — 설계 v2.1, review-astra-design.md 5절의 여섯 묶음.

기대 수량·날짜·판정은 실행 전에 손으로 계산해 적었다(계산식은 각 시험의 주석). 가격은 2025년 국내 호가
(2,000원 이상 5원), 수수료는 기본 0.015%·스트레스 0.30%. 실자료 성과는 계산하지 않는다(실제 묶음은 장부 없이
입력 대조만 한다). 합성 입력 도우미는 승인된 tests.test_research_etf_kodex_core의 것을 그대로 쓴다.
"""
import contextlib
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, localcontext
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from src.research.etf.baseline_policy import VolatilityEstimate
from src.research.etf.kodex_core import data as loader
from src.research.etf.kodex_core import formal, study
from src.research.etf.kodex_core.config import Segment, _plain, load_config
from src.research.etf.kodex_core.data import source_identity
from src.research.etf.kodex_core.inputs import DataHold, InputIntegrityError, VolatilityUnavailable, ZeroVolatility
from src.research.etf.kodex_core.ledgers import RunSetting, run_split
from src.research.etf.kodex_core.metrics import plain
from src.research.etf.kodex_grid import (baseline, cli, identity, judge, permit, progress, reading, records, report,
                                         sealing, stages, summary)
from src.research.etf.kodex_grid.grid import check_base, load_grid, segment_labels, setting_config
from src.research.etf.kodex_grid.identity import BundleSource, IdentityError, open_context, open_synthetic_context
from src.research.etf.kodex_grid.judge import FAIL, OBSERVED, OPEN, PASS, POSITIVE
from src.research.etf.kodex_grid.permit import PermitError
from src.research.etf.kodex_grid.runner import Bench, LedgerCache, NoCache, inputs_digest, open_ledgers, run_point
from tests.test_research_etf_kodex_core import (BASIS, CASE2_DAYS, DEC, FROZEN, J, SYMBOL, YEAR_END, as_symbol, at,
                                                build, synthetic_bundle, trades, weekdays)


D = Decimal
KST = timezone(timedelta(hours=9))
SPEC = load_grid()
BASE, STRESS = D("0.00015"), D("0.003")
PATHS = FROZEN.paths
OHLC, OLHC = PATHS
FEB3, MAR4 = date(2025, 2, 3), date(2025, 3, 4)
REAL_ROOT = Path("/mnt/volume/workspace/split-research/etf/data/etf-m5-r1")


def number(core, kg, kp, levels):
    return SPEC.number_of({"core": core, "kg": kg, "kp": kp, "levels": levels})


def cfg(core, kg, kp, levels, base=FROZEN):
    return setting_config(base, SPEC, SPEC.setting(number(core, kg, kp, levels)))


def estimate(day, sigma):
    return VolatilityEstimate(D(sigma), at(day) - timedelta(hours=12))


def bar(day, o, h, low, c):
    return {day: {"open": D(o), "high": D(h), "low": D(low), "close": D(c)}}


def split(data, config, path=OHLC, cost=BASE):
    return run_split(data, RunSetting(config, path, cost, "M"))


def paths_of(result):
    return result["run_start"]["items"]["M"]["paths"]


def frame_of(result, segments=FROZEN.segments):
    """그 결과가 가져야 할 틀: 두 경로와, 시작일부터 종료일까지 걸친 구간 이름."""
    opened = result["run_start"]
    return reading.Frame(tuple(PATHS), segment_labels(segments, opened["start"], opened["end"]), FROZEN.thresholds)


def verdict_of(result, segments=FROZEN.segments):
    point = reading.point_input(plain(result), frame_of(result, segments))
    return judge.setting_verdict(point, SPEC.thresholds)


def extension_of(result, segments=FROZEN.segments):
    point = reading.extension_input(plain(result), frame_of(result, segments))
    return judge.extension_point(point, SPEC.extension["base_edge_gt"])


def slot_problems(result):
    return records.slot_problems(plain(paths_of(result)), plain(result["exploration"]))


# ---------------------------------------------------------------------------
# 격자 정의와 적용 설정
# ---------------------------------------------------------------------------
class GridDefinition(unittest.TestCase):
    def test_numbers_follow_core_kg_kp_levels_order(self):
        rows = SPEC.settings()
        self.assertEqual([row.number for row in rows], list(range(1, 37)))
        self.assertEqual(rows[0].describe(), {"number": 1, "core": "50", "kg": "1", "kp": "1", "levels": 5,
                                              "core_krw": "50000000", "split_krw": "50000000"})
        # 코어 50%가 1~18, 80%가 19~36. 같은 코어 안에서 kg(6개씩) → kp(2개씩) → 차수 순.
        self.assertEqual((number("50", "1", "1", 9), number("50", "1", "3", 5), number("50", "2", "1", 5)), (2, 3, 7))
        self.assertEqual((number("50", "4", "6", 9), number("80", "1", "1", 5), number("80", "4", "6", 9)), (18, 19, 36))
        self.assertEqual(SPEC.baseline_number(), number("80", "2", "3", 9))
        self.assertEqual(SPEC.baseline_number(), 28)
        self.assertEqual(len({(r.core.label, r.kg, r.kp, r.levels) for r in rows}), 36)

    def test_capital_sums_to_one_account(self):
        for row in SPEC.settings():
            config = setting_config(FROZEN, SPEC, row)
            self.assertEqual(config.capital["core"] + config.capital["split"], D(100000000))
            self.assertEqual(config.capital["hold"], D(100000000))
        self.assertEqual(cfg("50", "1", "1", 5).capital["split"], D(50000000))
        self.assertEqual(cfg("80", "1", "1", 5).capital["split"], D(20000000))

    def test_variants_change_only_five_fields(self):
        frozen = _plain({name: getattr(FROZEN, name) for name in FROZEN.__dataclass_fields__})
        hashes = set()
        for row in SPEC.settings():
            config = setting_config(FROZEN, SPEC, row)
            changed = {name for name, value in frozen.items() if _plain(getattr(config, name)) != value}
            self.assertLessEqual(changed, {"levels", "kg", "kp", "capital", "modes"})
            self.assertEqual(config.modes, ("M",))
            hashes.add(config.applied_sha256())
        self.assertEqual(len(hashes), 36)
        point = cfg("80", "2", "3", 9)  # 공개 기준점: 고정 설정과 재진입 방식(M만)만 다르다
        self.assertEqual({name for name, value in frozen.items() if _plain(getattr(point, name)) != value}, {"modes"})

    def test_regions_and_priority(self):
        regions = SPEC.regions()
        self.assertEqual([(r.rank, r.core, r.kg, r.kp) for r in regions],
                         [(1, "80", (D(1), D(2)), (D(1), D(3))), (2, "80", (D(1), D(2)), (D(3), D(6))),
                          (3, "80", (D(2), D(4)), (D(1), D(3))), (4, "80", (D(2), D(4)), (D(3), D(6))),
                          (5, "50", (D(1), D(2)), (D(1), D(3))), (6, "50", (D(1), D(2)), (D(3), D(6))),
                          (7, "50", (D(2), D(4)), (D(1), D(3))), (8, "50", (D(2), D(4)), (D(3), D(6)))])
        self.assertEqual(regions[0].numbers, (19, 20, 21, 22, 25, 26, 27, 28))
        self.assertEqual(regions[7].numbers, (9, 10, 11, 12, 15, 16, 17, 18))
        self.assertTrue(all(len(r.numbers) == 8 for r in regions))
        # 후속 후보 한 점: 영역의 5차·kg 하한·kp 하한
        self.assertEqual([SPEC.candidate_number(r) for r in regions], [19, 21, 25, 27, 1, 3, 7, 9])
        # 겹치는 점은 같은 설정 번호로 나타난다(기준점 28은 코어 80%의 네 영역 모두에 속한다).
        self.assertEqual([r.rank for r in regions if 28 in r.numbers], [1, 2, 3, 4])

    def test_declaration_matches_approved_code_and_design(self):
        self.assertEqual(SPEC.approved["source_sha256"], source_identity()["source_sha256"])
        self.assertEqual(SPEC.approved["base_config_file_sha256"], FROZEN.sha256)
        check_base(SPEC, FROZEN)
        self.assertEqual((SPEC.symbol, SPEC.primary_start, SPEC.end, SPEC.mode),
                         ("069500", date(2011, 1, 3), date(2026, 11, 13), "M"))
        self.assertEqual(SPEC.reconfirm, {"offsets": (5, 20), "starts": (date(2011, 1, 10), date(2011, 1, 31))})
        self.assertEqual((SPEC.extension["symbols"], SPEC.extension["start"], SPEC.extension["points_min"]),
                         (("229200", "169950"), date(2017, 1, 2), 6))
        self.assertEqual(SPEC.thresholds, {"base_edge_min": D("0.01"), "stress_edge_gt": D(0),
                                           "segment_edge_gt": D(0), "common_segments_min": 3})
        primary = next(row for row in FROZEN.rows if row.id == "primary")
        self.assertEqual(primary.starts, (SPEC.primary_start, *SPEC.reconfirm["starts"]))

    def test_changed_declarations_are_rejected_or_identified(self):
        with self.assertRaisesRegex(ValueError, "end or segments"):
            check_base(SPEC, FROZEN.with_changes(end=date(2026, 11, 12)))
        with self.assertRaisesRegex(ValueError, r"core \+ split"):
            replace(SPEC, account=D(90000000))
        changed = replace(SPEC, thresholds={**SPEC.thresholds, "base_edge_min": D("0.02")})
        self.assertEqual(changed.sha256, SPEC.sha256)  # 파일 해시는 그대로지만
        self.assertNotEqual(changed.applied_sha256(), SPEC.applied_sha256())  # 적용 정의 해시로 구별된다
        with tempfile.TemporaryDirectory() as folder:
            raw = json.loads(Path(identity.__file__).with_name("kodex_grid_v1.json").read_text())
            path = Path(folder) / "grid.json"
            path.write_text(json.dumps({**raw, "formal_run": True}))
            with self.assertRaisesRegex(ValueError, "formal flag"):
                load_grid(path)


# ---------------------------------------------------------------------------
# 묶음 1: 한 설정의 판정(4절)
# ---------------------------------------------------------------------------
LABELS = ("s1", "s2", "s3", "s4")
T = SPEC.thresholds


def path_row(base="0.02", stress="0.005", wins=(1, 2, 3, 4), **flags):
    """wins: 기본 비용 차가 양수인 구간 번호. 나머지 구간은 -0.01."""
    segments = {label: D("0.01") if index in wins else D("-0.01") for index, label in enumerate(LABELS, start=1)}
    return {"start_holds": [], "start_failures": [], "later_holds": [], **flags,
            "base": {"delta_cagr": D(base), "segments": segments}, "stress": {"delta_cagr": D(stress)}}


def point(first, second):
    return {"held": None, "paths": {OHLC: first, OLHC: second}}


class Table1SettingVerdict(unittest.TestCase):
    """경로별 승리 구간이 달라도 '같은 구간에서 두 경로가 함께 이긴' 구간만 센다."""

    def test_common_segments_not_per_path_counts(self):
        # 경로별로는 각각 세 구간을 이겼지만 함께 이긴 구간은 s2·s3 둘뿐 → 미달
        split_wins = judge.setting_verdict(point(path_row(wins=(1, 2, 3)), path_row(wins=(2, 3, 4))), T)
        self.assertEqual((split_wins["status"], split_wins["common_segments"], split_wins["reasons"]),
                         (FAIL, ["s2", "s3"], ["common_segments"]))
        shared = judge.setting_verdict(point(path_row(wins=(1, 2, 3)), path_row(wins=(1, 2, 3, 4))), T)
        self.assertEqual((shared["status"], shared["common_segments"]), (PASS, ["s1", "s2", "s3"]))

    def test_base_edge_boundary_is_inclusive_and_unrounded(self):
        exact = judge.setting_verdict(point(path_row(base="0.01"), path_row(base="0.01")), T)
        self.assertEqual(exact["status"], PASS)
        below = "0.00999999999999999999999999999999999999999999999999999999999999999999999999999"
        short = judge.setting_verdict(point(path_row(base=below), path_row()), T)
        self.assertEqual((short["status"], short["reasons"]), (FAIL, [f"{OHLC}:base_edge"]))
        self.assertEqual(short["paths"][OLHC]["status"], PASS)

    def test_stress_zero_and_segment_zero_do_not_pass(self):
        zero = judge.setting_verdict(point(path_row(stress="0"), path_row()), T)
        self.assertEqual((zero["status"], zero["reasons"]), (FAIL, [f"{OHLC}:stress_edge"]))
        tiny = judge.setting_verdict(point(path_row(stress="1E-70"), path_row(stress="1E-70")), T)
        self.assertEqual(tiny["status"], PASS)
        flat = path_row(wins=(1, 2, 3))
        flat["base"]["segments"]["s3"] = D(0)  # 한 경로의 세 번째 구간 차가 정확히 0 → 그 구간은 세지 않는다
        self.assertEqual(judge.setting_verdict(point(flat, path_row(wins=(1, 2, 3))), T)["common_segments"],
                         ["s1", "s2"])
        self.assertEqual(judge.setting_verdict(point(flat, path_row(wins=(1, 2, 3))), T)["status"], FAIL)

    def test_legacy_limits_are_shown_but_not_conditions(self):
        """낙폭 심화 20%p·평균 비중 50%여도 새 판정은 통과한다(종전 한도는 표기만)."""
        row = path_row()
        row["base"].update(delta_abs_mdd=D("0.20"), exposure_mean=D("0.50"))
        self.assertEqual(judge.setting_verdict(point(row, path_row()), T)["status"], PASS)
        view = {"mdd_deepening": D("0.20"), "exposure_mean": D("0.50")}
        self.assertEqual(reading.legacy_limits(view, FROZEN.thresholds), {"mdd_excess_ok": False, "exposure_ok": False})
        edge = {"mdd_deepening": D("0.05"), "exposure_mean": D("0.80")}
        self.assertEqual(reading.legacy_limits(edge, FROZEN.thresholds), {"mdd_excess_ok": True, "exposure_ok": True})

    def test_state_order_follows_approved_path_verdict(self):
        """시작 비교 보류 → 시작 시 확정 미달 → 개설 뒤 결측. 스트레스에서만 개설 거부여도 4절은 미달."""
        hold, rejected = ["base_hold_first_buy_failed"], ["stress_start_rejected_by_cost_guard"]
        cases = (({"start_holds": hold, "start_failures": rejected}, OPEN),
                 ({"start_failures": rejected, "later_holds": ["input_data_gap_after_start"]}, FAIL),
                 ({"later_holds": ["base_annual_volatility_unavailable"]}, OPEN),
                 ({"start_failures": rejected}, FAIL))
        shown = [None, None, None, list(LABELS)]  # 기본 비용 두 경로가 유효한 넷째 사례(스트레스만 거부)만 공통 구간을 싣는다
        for (flags, expected), common in zip(cases, shown):
            with self.subTest(flags=flags):
                row = {**path_row(**flags), "stress": None}
                self.assertEqual(judge.path_verdict(row, T)["status"], expected)
                self.assertEqual(judge.setting_verdict(point(row, row), T)["status"], expected)
                self.assertEqual(judge.setting_verdict(point(row, row), T)["common_segments"], common)

    def test_fail_on_one_path_beats_undetermined_on_the_other(self):
        held = {**path_row(start_holds=["base_combined_start_liquidity"]), "base": None, "stress": None}
        failed = judge.setting_verdict(point(path_row(base="0.001"), held), T)
        self.assertEqual((failed["status"], failed["reasons"]), (FAIL, [f"{OHLC}:base_edge"]))
        waiting = judge.setting_verdict(point(path_row(), held), T)
        self.assertEqual((waiting["status"], waiting["reasons"]), (OPEN, [f"{OLHC}:base_combined_start_liquidity"]))
        loaded = judge.setting_verdict({"held": "calendar/history insufficient", "paths": None}, T)
        self.assertEqual((loaded["status"], loaded["reasons"]), (OPEN, ["calendar/history insufficient"]))

    def test_empty_values_without_reason_are_errors_not_zero(self):
        with self.assertRaisesRegex(ValueError, "needs a hold or failure reason"):
            judge.path_verdict({**path_row(), "stress": None}, T)
        uneven = path_row()
        del uneven["base"]["segments"]["s4"]
        with self.assertRaisesRegex(ValueError, "same segments"):
            judge.setting_verdict(point(path_row(), uneven), T)


# ---------------------------------------------------------------------------
# 묶음 2: 영역·재확인·확장의 경계(5~7절)
# ---------------------------------------------------------------------------
REGIONS = [region.describe() for region in SPEC.regions()]
PAIRS = [f"{n:02d}@{start}" for start in ("2011-01-10", "2011-01-31") for n in SPEC.regions()[0].numbers]


def ext_row(delta, **flags):
    """확장 경로 입력: 기본 비용·공통 사유 세 목록과 기본 비용 전체 연복리 차."""
    return {"start_holds": [], "start_failures": [], "later_holds": [], **flags,
            "base_delta": None if delta is None else D(delta)}


def ext_point(first, second, capacity_ok=True):
    return {"held": None, "capacity_ok": capacity_ok, "paths": {OHLC: first, OLHC: second}}


def statuses(passed=(), undetermined=()):
    return {n: PASS if n in passed else OPEN if n in undetermined else FAIL for n in range(1, 37)}


def choose(passed=(), undetermined=()):
    table = judge.regions_table(REGIONS, statuses(passed, undetermined))
    return judge.select_region(table), {row["rank"]: row["status"] for row in table}


def members(rank):
    return set(SPEC.regions()[rank - 1].numbers)


class Table2RegionsAndFollowUps(unittest.TestCase):
    def test_seven_of_eight_is_not_a_region(self):
        seven = members(1) - {28}
        selection, ranks = choose(seven)
        self.assertEqual((selection["state"], ranks[1]), ("none", FAIL))
        selection, ranks = choose(seven, undetermined={28})  # 여덟째 점이 미확정이면 영역도 미확정
        self.assertEqual((selection["state"], ranks[1], selection["blocked_by_rank"]), (OPEN, OPEN, 1))

    def test_isolated_nonadjacent_and_mixed_core_points_make_no_region(self):
        isolated = {19, 36, 5}
        table = judge.regions_table(REGIONS, statuses(isolated))
        self.assertEqual(judge.select_region(table)["state"], "none")
        self.assertEqual(judge.isolated_passes(table, statuses(isolated)), [5, 19, 36])
        nonadjacent = {n for n in range(19, 37) if SPEC.setting(n).kg in (D(1), D(4)) and SPEC.setting(n).kp in (D(1), D(3))}
        self.assertEqual((len(nonadjacent), choose(nonadjacent)[0]["state"]), (8, "none"))
        mixed = {n for n in range(1, 37) if SPEC.setting(n).kg in (D(1), D(2)) and SPEC.setting(n).kp in (D(1), D(3))
                 and SPEC.setting(n).levels == 5}  # 두 코어에서 네 점씩
        self.assertEqual((len(mixed), choose(mixed)[0]["state"]), (8, "none"))

    def test_registered_priority_picks_one_region(self):
        selection, _ = choose(members(2) | members(4) | members(5))
        self.assertEqual((selection["state"], selection["region"]["rank"]), ("selected", 2))
        self.assertEqual(selection["region"]["numbers"], [21, 22, 23, 24, 27, 28, 29, 30])
        # 코어 80% → kg 하한 → kp 하한: kg 하한이 낮은 순위 2가 kp 하한이 낮은 순위 3보다 앞선다
        self.assertEqual(choose(members(2) | members(3))[0]["region"]["rank"], 2)
        self.assertEqual(choose(members(1) | members(2) | members(8))[0]["region"]["rank"], 1)
        self.assertEqual(choose(members(5) | members(8))[0]["region"]["rank"], 5)  # 코어 50%만 통과
        self.assertEqual(choose(members(3) | members(5), undetermined=members(8))[0]["region"]["rank"], 3)

    def test_undetermined_and_none_are_distinct(self):
        # 상위 미확정·하위 통과 → 선택 미확정(하위를 고르지 않는다)
        upper_open = (members(1) - {19}) | members(5)
        selection, ranks = choose(upper_open, undetermined={19})
        self.assertEqual((selection["state"], selection["region"], selection["blocked_by_rank"]), (OPEN, None, 1))
        self.assertEqual((ranks[1], ranks[5]), (OPEN, PASS))
        # 8개 영역 전부 확정 미달 → 영역 없음
        selection, ranks = choose()
        self.assertEqual((selection["state"], set(ranks.values())), ("none", {FAIL}))
        # 통과 없이 미확정만 남음 → 선택 미확정. 미확정 점이 있어도 확정 미달 점이 있는 영역은 미달이다
        selection, ranks = choose(undetermined=members(6))
        self.assertEqual((selection["state"], selection["blocked_by_rank"], ranks[6], ranks[5]), (OPEN, 6, OPEN, FAIL))
        sentences = SPEC.conclusions
        self.assertEqual(judge.conclusion(sentences, choose()[0])["sentence"], "이 범위에 후속 조사 근거 없음")
        self.assertEqual(judge.conclusion(sentences, choose(undetermined=members(6))[0])["sentence"], "자료 부족으로 미확정")

    def test_region_points_must_exist(self):
        with self.assertRaises(KeyError):
            judge.regions_table(REGIONS, {n: PASS for n in range(1, 36)})

    def test_reconfirmation_needs_all_sixteen_pairs(self):
        selection = choose(members(1))[0]
        everything = judge.reconfirm_verdict({pair: PASS for pair in PAIRS}, PAIRS)
        self.assertEqual((everything["status"], everything["pairs"]), (PASS, 16))
        done = judge.conclusion(SPEC.conclusions, selection, reconfirm=everything, candidate=19)
        self.assertEqual((done["sentence"], done["candidate"], done["stage"]), ("후속 확인용 후보 1개", 19, "kodex_reconfirm"))
        for late in (PAIRS[3], PAIRS[12]):  # +5 또는 +20의 한 점 미달 → 후보 없음
            one_fail = judge.reconfirm_verdict({pair: FAIL if pair == late else PASS for pair in PAIRS}, PAIRS)
            closed = judge.conclusion(SPEC.conclusions, selection, reconfirm=one_fail, candidate=19)
            self.assertEqual((one_fail["status"], one_fail["failed"], closed["candidate"]), (FAIL, [late], None))
            self.assertEqual((closed["sentence"], closed["stage"]), ("이 범위에 후속 조사 근거 없음", "kodex_reconfirm"))
        waiting = judge.reconfirm_verdict({pair: OPEN if pair == PAIRS[0] else PASS for pair in PAIRS}, PAIRS)
        self.assertEqual(waiting["status"], OPEN)
        self.assertEqual(judge.conclusion(SPEC.conclusions, selection, reconfirm=waiting, candidate=19)["sentence"],
                         "자료 부족으로 미확정")
        both = judge.reconfirm_verdict({**{pair: PASS for pair in PAIRS}, PAIRS[0]: OPEN, PAIRS[1]: FAIL}, PAIRS)
        self.assertEqual(both["status"], FAIL)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            judge.reconfirm_verdict({pair: PASS for pair in PAIRS[:15]}, PAIRS)
        pending = judge.conclusion(SPEC.conclusions, selection)
        self.assertEqual((pending["sentence"], pending["pending"]), (None, "kodex_reconfirm"))

    def test_extension_counts_p_and_u(self):
        def product(positive, undetermined):
            values = [POSITIVE] * positive + [OPEN] * undetermined + [FAIL] * (8 - positive - undetermined)
            return judge.extension_product(values, points=8, points_min=6)
        self.assertEqual([product(*pu)["status"] for pu in ((6, 2), (5, 1), (5, 0), (6, 0), (4, 2), (4, 1), (8, 0))],
                         [OBSERVED, OPEN, FAIL, OBSERVED, OPEN, FAIL, OBSERVED])
        self.assertEqual(product(5, 1), {"status": OPEN, "positive": 5, "undetermined": 1, "failed": 2,
                                         "denominator": 8, "points_min": 6})
        with self.assertRaisesRegex(ValueError, "denominator is not reduced"):
            judge.extension_product([POSITIVE] * 6, points=8, points_min=6)
        names = SPEC.extension["symbols"]
        both = judge.extension_verdict({names[0]: product(6, 2), names[1]: product(7, 0)}, expected=names)
        self.assertEqual((both["status"], both["independent_samples"]), (OBSERVED, False))
        waiting = judge.extension_verdict({names[0]: product(6, 0), names[1]: product(5, 1)}, expected=names)
        failed = judge.extension_verdict({names[0]: product(8, 0), names[1]: product(5, 0)}, expected=names)
        self.assertEqual((waiting["status"], failed["status"]), (OPEN, FAIL))
        with self.assertRaisesRegex(ValueError, "registered products"):
            judge.extension_verdict({names[0]: product(8, 0)}, expected=names)

    def test_wins_on_different_paths_are_not_added(self):
        """경로 하나만 이긴 점이 여섯 개여도(세 점은 한 경로, 세 점은 다른 경로) 확정 양수 점은 0개다."""
        up, down = ext_row("0.004"), ext_row("-0.004")
        points = [ext_point(up, down)] * 3 + [ext_point(down, up)] * 3 + [ext_point(down, down)] * 2
        values = [judge.extension_point(p, D(0))["status"] for p in points]
        self.assertEqual(values, [FAIL] * 8)
        self.assertEqual(judge.extension_product(values, points=8, points_min=6)["status"], FAIL)
        self.assertEqual(judge.extension_point(ext_point(up, up), D(0))["status"], POSITIVE)
        zero = judge.extension_point(ext_point(up, ext_row("0")), D(0))  # 0은 양수가 아니다
        self.assertEqual((zero["status"], zero["reasons"]), (FAIL, [f"{OLHC}:base_edge"]))

    def test_extension_reads_base_cost_only(self):
        """확장 판정 입력은 명목 한도, 기본 비용·공통 사유, 두 기본 경로의 전체 차뿐이다."""
        stress_only = {**path_row(base="0.004", start_failures=["stress_start_rejected_by_cost_guard"]), "stress": None}
        self.assertEqual(judge.setting_verdict(point(stress_only, stress_only), T)["status"], FAIL)  # KODEX 4절
        up = ext_row("0.004")
        self.assertEqual(set(up), {"start_holds", "start_failures", "later_holds", "base_delta"})
        base_rejected = ext_row(None, start_failures=["base_split_start_failed"])
        base_held = ext_row(None, start_holds=["base_combined_start_liquidity"])
        gap = ext_row(None, later_holds=["input_data_gap_after_start"])
        self.assertEqual(judge.extension_point(ext_point(base_rejected, up), D(0))["status"], FAIL)
        for row in (base_held, gap):
            self.assertEqual(judge.extension_point(ext_point(row, up), D(0))["status"], OPEN)
        self.assertEqual(judge.extension_point(ext_point(base_held, ext_row("-0.1")), D(0))["status"], FAIL)
        floor = judge.extension_point(ext_point(up, up, capacity_ok=False), D(0))  # 명목 한도 미달은 U로 센다
        self.assertEqual((floor["status"], floor["reasons"], floor["paths"]), (OPEN, ["start_capacity_below_floor"], None))
        self.assertEqual(judge.extension_point({"held": "no input", "capacity_ok": None, "paths": None}, D(0))["status"], OPEN)
        with self.assertRaisesRegex(ValueError, "needs a hold or failure reason"):
            judge.extension_point(ext_point(ext_row(None), up), D(0))  # 사유 없는 빈값을 0으로 읽지 않는다


# ---------------------------------------------------------------------------
# 묶음 3: kg=1의 비용·자료 경계
# ---------------------------------------------------------------------------
def flat(sigma, days=None, prices=None):
    days = days or weekdays(J(2), J(8))
    return build(days, prices or {J(2): "10000"}, sigma=sigma, basis=BASIS)


class Table3CostGuardAtKgOne(unittest.TestCase):
    """비용 조건: 2 × 편도 수수료율 ≤ kg × σ < 1. kg=1이면 스트레스 하한은 σ=0.60%(등호 통과)."""

    def test_start_boundaries(self):
        config = cfg("50", "1", "3", 5)
        cases = (("0.006", "ok", "ok"), ("0.0059999", "ok", "cost_guard_start"),
                 ("0.0003", "ok", "cost_guard_start"), ("0.00029999", "cost_guard_start", "cost_guard_start"))
        for sigma, base, stress in cases:
            with self.subTest(sigma=sigma):
                data = flat(sigma)
                self.assertEqual((split(data, config).status, split(data, config, cost=STRESS).status), (base, stress))

    def test_gap_of_one_hundred_percent_is_rejected(self):
        for kg, sigma, expected in (("4", "0.25", "cost_guard_start"), ("4", "0.2499", "ok"), ("1", "1", "cost_guard_start")):
            with self.subTest(kg=kg, sigma=sigma):
                config = cfg("50", kg, "3", 5)
                self.assertEqual({split(flat(sigma), config, cost=cost).status for cost in (BASE, STRESS)}, {expected})

    def test_stress_only_rejection_fails_kodex_but_counts_in_extension(self):
        """σ 0.59999%·kg 1: 스트레스만 개설 거부. KODEX 4절은 미달, 확장은 유효한 기본 비용 두 경로로 센다."""
        config = cfg("50", "1", "3", 5)
        # 하락장: 분할이 낮은 값에 더 사고 전체 보유 주식이 단독 보유보다 적어 기본 비용 차가 양수다
        falling = run_point(flat("0.0059999", prices={J(2): "10000", J(6): "9000"}), config, J(2))
        self.assertEqual(falling["exploration"][OHLC]["flags"],
                         {"failures": ["stress_start_rejected_by_cost_guard"], "holds": [], "later": []})
        self.assertEqual((verdict_of(falling)["status"], extension_of(falling)["status"]), (FAIL, POSITIVE))
        self.assertIn(f"{OHLC}:stress_start_rejected_by_cost_guard", verdict_of(falling)["reasons"])
        self.assertIsNone(paths_of(falling)[OHLC]["stress"])
        self.assertEqual(falling["exploration"][OHLC]["split_ledger"]["stress"],
                         {"status": "cost_guard_start", "buys": None, "sells": None})
        # 상승장: 단독 보유(9,998주)가 전략(코어 4,999주 + 분할 999주)보다 많이 오른다 → 기본 비용 차 ≤ 0
        rising = run_point(flat("0.0059999", prices={J(2): "10000", J(6): "10010"}), config, J(2))
        self.assertEqual((verdict_of(rising)["status"], extension_of(rising)["status"]), (FAIL, FAIL))
        self.assertEqual(extension_of(rising)["reasons"], [f"{OHLC}:base_edge", f"{OLHC}:base_edge"])

    def test_closed_buys_take_profit_blocked_reentry_and_reopening(self):
        """스트레스 장부: 2025년 σ 0.5%로 매수 닫힘 → 과거 차수 익절 → 2·3월 월초 재진입 거부 → 2026년 σ 0.6%에 재개.

        1차 997주 = floor(10,000,000 ÷ (10,000 × 1.003)). 익절 순수익률 (10,400×0.997 − 10,000×1.003) ÷ 10,000 = 3.388% ≥ 3%.
        매도 뒤 현금 50,000,000 − 9,999,910 + 10,337,693.6 = 50,337,783.6 → 2026년 1월 예산 10,067,556.72
        → floor(10,067,556.72 ÷ (10,400 × 1.003)) = 965주.
        """
        y25, y26 = date(2025, 1, 2), date(2026, 1, 2)
        days = [date(2024, 12, 2), date(2024, 12, 3), y25, date(2025, 1, 3), FEB3, date(2025, 3, 3), y26, date(2026, 1, 5)]
        data = build(days, {days[0]: "10000", y25: "9800", date(2025, 1, 3): "10400"}, basis=BASIS,
                     extra_vol={y25: estimate(y25, "0.005"), y26: estimate(y26, "0.006")})
        config = cfg("50", "1", "3", 5)
        for path in PATHS:
            with self.subTest(path=path):
                run = split(data, config, path, STRESS)
                # 2025-01-02 시가 9,800은 전년 간격(1%)이면 2차 매수였지만 매수가 닫혀 주문이 없다
                self.assertEqual(trades(run), [(days[0], "BUY", 1, 997, D(10000)), (date(2025, 1, 3), "SELL", 1, 997, D(10400)),
                                               (y26, "BUY", 1, 965, D(10400))])
                attempts = run.detail["reentries"]
                self.assertEqual([row["date"] for row in attempts], [FEB3, date(2025, 3, 3), y26])
                self.assertTrue(all(row["outcome"].startswith("cost_guard") for row in attempts[:2]))
                self.assertEqual(attempts[2]["outcome"], "filled")
                self.assertEqual([(row["date"], row.get("buys_closed", False)) for row in run.detail["refreshes"]],
                                 [(y25, True), (y26, False)])
                self.assertEqual(split(data, config, path).detail["refreshes"][0]["applied"], True)  # 기본 비용은 열려 있다
        result = run_point(data, config, days[0])
        for path in PATHS:
            self.assertEqual(paths_of(result)[path]["buys_closed_years"], {"base": [], "stress": [y25]})
            counts = result["exploration"][path]["reentry_counts"]["stress"]
            self.assertEqual((counts["attempts"], counts["filled"], sum(counts["unfilled"].values())), (3, 1, 2))
            self.assertEqual(result["exploration"][path]["split_ledger"]["stress"], {"status": "ok", "buys": 2, "sells": 1})
            # 기본 비용: 9,800에 2차 1,020주, 10,400에 2차·1차 익절, 3월 월초 재진입 → 매수 3·매도 2·재진입 1/1
            self.assertEqual(result["exploration"][path]["split_ledger"]["base"], {"status": "ok", "buys": 3, "sells": 2})
            self.assertEqual(result["exploration"][path]["reentry_counts"]["base"],
                             {"status": "ok", "attempts": 1, "filled": 1, "unfilled": {}})

    def test_valid_zero_volatility_is_failure_and_missing_is_undetermined(self):
        config, data = cfg("50", "1", "3", 5), flat("0.01")
        zero = replace(data, volatility={J(2): ZeroVolatility(at(J(2)) - timedelta(hours=12))})
        missing = replace(data, volatility={J(2): VolatilityUnavailable("synthetic: 251 sessions required")})
        rejected, held = run_point(zero, config, J(2)), run_point(missing, config, J(2))
        self.assertEqual(rejected["exploration"][OHLC]["flags"]["failures"],
                         ["base_start_rejected_by_cost_guard", "stress_start_rejected_by_cost_guard"])
        self.assertEqual((verdict_of(rejected)["status"], extension_of(rejected)["status"]), (FAIL, FAIL))
        self.assertEqual(held["exploration"][OHLC]["flags"]["holds"],
                         ["base_volatility_unavailable", "stress_volatility_unavailable"])
        self.assertEqual((verdict_of(held)["status"], extension_of(held)["status"]), (OPEN, OPEN))
        for result in (rejected, held):  # 수치는 비고 사유만 남는다
            self.assertEqual((paths_of(result)[OHLC]["base"], paths_of(result)[OHLC]["stress"]), (None, None))
            self.assertEqual(slot_problems(result), [])

    def test_later_gap_does_not_erase_start_rejection(self):
        """스트레스 개설 미달이 시작 시 확정됐으면 이후 연초 변동성 결측이 그 미달을 미확정으로 바꾸지 않는다."""
        dec = date(2024, 12, 30)
        days = weekdays(dec, J(10), {date(2024, 12, 31), J(1)})
        data = build(days, {dec: "10000"}, sigma="0.0059999", basis=BASIS,
                     extra_vol={J(2): VolatilityUnavailable("synthetic missing 251st close")})
        result = run_point(data, cfg("50", "1", "3", 5), dec)
        self.assertEqual(result["exploration"][OHLC]["flags"],
                         {"failures": ["stress_start_rejected_by_cost_guard"], "holds": [],
                          "later": ["base_annual_volatility_unavailable"]})
        self.assertEqual(verdict_of(result)["status"], FAIL)
        self.assertEqual(extension_of(result)["status"], OPEN)  # 기본 비용은 개설 뒤 결측 → 미확정
        gap_only = build(days, {dec: "10000"}, basis=BASIS,
                         extra_vol={J(2): VolatilityUnavailable("synthetic missing 251st close")})
        self.assertEqual(verdict_of(run_point(gap_only, cfg("50", "1", "3", 5), dec))["status"], OPEN)


# ---------------------------------------------------------------------------
# 묶음 4: 코어 50%의 첫 주문과 한도
# ---------------------------------------------------------------------------
def orders(result, cost="base"):
    start = paths_of(result)[OHLC]["start_orders"][cost]
    return ({name: row.get("quantity") for name, row in start["orders"].items()}, start["core_plus_split"], start["hold"])


class Table4FirstOrdersAndLimits(unittest.TestCase):
    """시가 10,000원. 수량 = floor(예산 ÷ (10,000 × (1 + 수수료율)))."""

    def setUp(self):
        self.data = build(weekdays(J(2), J(6)), {J(2): "10000"}, basis=BASIS)

    def test_first_quantities_for_both_core_weights(self):
        expected = {  # (코어, 차수): 기본(코어, 분할, 보유, 코어+분할 원금), 스트레스(같은 순서)
            ("50", 5): ((4999, 999, 9998, 59980000), (4985, 997, 9970, 59820000)),
            ("50", 9): ((4999, 555, 9998, 55540000), (4985, 553, 9970, 55380000)),
            ("80", 5): ((7998, 399, 9998, 83970000), (7976, 398, 9970, 83740000)),
            ("80", 9): ((7998, 222, 9998, 82200000), (7976, 221, 9970, 81970000)),
        }
        for (core, levels), rows in expected.items():
            result = run_point(self.data, cfg(core, "2", "3", levels), J(2))
            for cost, (core_q, split_q, hold_q, total) in zip(("base", "stress"), rows):
                with self.subTest(core=core, levels=levels, cost=cost):
                    self.assertEqual(orders(result, cost), ({"core": core_q, "split": split_q, "hold": hold_q},
                                                            D(total), D(hold_q * 10000)))
            self.assertEqual({row["status"] for row in paths_of(result)[OLHC]["start_orders"]["base"]["orders"].values()},
                             {"filled"})

    def test_combined_start_limit_equal_and_one_won_short(self):
        """코어 4,999주 + 분할 999주 = 59,980,000원. 한도(20일 평균 거래대금 × 5%)와 같으면 허용, 1원 모자라면 보류."""
        config = cfg("50", "2", "3", 5)
        group = open_ledgers(self.data, config, J(2))["runs"]["base"][OHLC]
        for adv20, allowed in ((D(1199600000), True), (D(1199599980), False)):
            bars = (replace(self.data.bars[0], adv20=adv20),) + self.data.bars[1:]
            self.assertEqual(study._start_liquidity(replace(self.data, bars=bars), config, group["core"], group["M"]), allowed)

    def test_hold_first_order_limit_equal_and_one_won_short(self):
        """단독 보유 첫 주문 99,980,000원. 주문을 줄이지 않고, 한도에 1원 모자라면 비교 보류(미확정)다."""
        config = cfg("50", "2", "3", 5)
        equal = run_point(build(weekdays(J(2), J(6)), {J(2): "10000"}, basis=BASIS,
                                changes={J(2): {"adv20": D(1999600000)}}), config, J(2))
        short = run_point(build(weekdays(J(2), J(6)), {J(2): "10000"}, basis=BASIS,
                                changes={J(2): {"adv20": D(1999599980)}}), config, J(2))
        self.assertEqual(equal["exploration"][OHLC]["flags"], {"failures": [], "holds": [], "later": []})
        self.assertEqual(paths_of(equal)[OHLC]["start_orders"]["base"]["orders"]["hold"],
                         {"status": "filled", "notional": D(99980000), "quantity": 9998})
        self.assertEqual(short["exploration"][OHLC]["flags"]["holds"], ["base_hold_first_buy_failed"])
        self.assertEqual(paths_of(short)[OHLC]["start_orders"]["base"]["orders"]["hold"],
                         {"status": "rejected", "notional": D(99980000), "reason": "participation_limit"})
        self.assertEqual((verdict_of(short)["status"], extension_of(short)["status"]), (OPEN, OPEN))
        self.assertIsNone(paths_of(short)[OHLC]["hold_metrics"]["base"])

    def test_extension_nominal_floor_equal_and_one_won_short(self):
        """확장 상품만: 같은 5% 한도가 명목 1억 원 이상이어야 한다. KODEX 주 행에는 이 문턱이 없다."""
        config = cfg("50", "2", "3", 5)
        for adv20, ok in ((D(2000000000), True), (D(1999999980), False)):
            data = build(weekdays(J(2), J(6)), {J(2): "10000"}, basis=BASIS, changes={J(2): {"adv20": adv20}})
            extension = run_point(data, config, J(2), nominal_floor=config.nominal_floor)
            self.assertEqual(extension["run_start"]["capacity"]["ok"], ok)
            primary = run_point(data, config, J(2))
            self.assertTrue(primary["run_start"]["capacity"]["ok"])
            self.assertEqual(primary["exploration"][OHLC]["flags"]["holds"], [])
        self.assertEqual(extension["run_start"]["capacity"]["limit_krw"], D(99999999))
        flags = {"failures": [], "holds": ["start_capacity_below_floor"], "later": []}
        self.assertEqual(extension["exploration"][OLHC], {"flags": flags, "split_ledger": None, "reentry_counts": None,
                                                           "core_share": None, "split_share": None})
        self.assertEqual((verdict_of(extension)["status"], extension_of(extension)["status"]), (OPEN, OPEN))
        self.assertEqual(extension["run_start"], study.run_start(data, config, J(2), nominal_floor=config.nominal_floor))
        self.assertEqual(slot_problems(extension), [])  # 네 자리 모두 빈칸이고 공통 사유(명목 한도)가 있다

    def test_budget_after_month_change_uses_split_equity(self):
        """1/31 종가 9,950: 분할 순자산 = 현금 40,008,501.5 + 999 × 9,950 = 49,948,551.5 → 2월 예산 9,989,710.3.
        2/3 시가 9,800(1차 매수가 대비 −2%)의 2차 = floor(9,989,710.3 ÷ 9,801.47) = 1,019주(첫 예산이면 1,020주)."""
        days = [J(2), J(3), J(31), FEB3, date(2025, 2, 4)]
        data = build(days, {J(2): "10000", J(31): "9950", FEB3: "9800"}, basis=BASIS)
        for path in PATHS:
            run = split(data, cfg("50", "2", "3", 5), path)
            self.assertEqual(trades(run), [(J(2), "BUY", 1, 999, D(10000)), (FEB3, "BUY", 2, 1019, D(9800))])
            refresh = [row for row in run.result.cash_events if row["kind"] == "budget_refresh"]
            self.assertEqual([(row["at"], row["source_total_krw"]) for row in refresh],
                             [(at(J(2)), D(50000000)), (at(FEB3), D("49948551.5"))])
            self.assertEqual(refresh[1]["budgets"][SYMBOL], D("9989710.3"))
        nine = split(data, cfg("50", "2", "3", 9))  # 9차 첫 예산은 5천만 ÷ 9 = 5,555,555.5…원 → 555주
        self.assertEqual(trades(nine)[0], (J(2), "BUY", 1, 555, D(10000)))


# ---------------------------------------------------------------------------
# 묶음 5: 체결·기간·입력
# ---------------------------------------------------------------------------
SEGMENTS = (Segment("S1", date(2025, 1, 1), date(2025, 1, 31)), Segment("S2", date(2025, 2, 1), date(2025, 2, 28)),
            Segment("S3", date(2025, 3, 1), date(2025, 3, 31)), Segment("S4", date(2025, 4, 1), date(2025, 4, 30)))
SEGMENT_DAYS = [J(2), J(3), J(6), J(7), J(8), J(9), J(10), J(31), FEB3, date(2025, 2, 28), date(2025, 3, 3),
                date(2025, 3, 31), date(2025, 4, 1), date(2025, 4, 30)]
SEGMENT_PRICES = {J(2): "10000", FEB3: "10300", date(2025, 3, 3): "10000", date(2025, 4, 1): "10500"}
SEGMENT_BASE = FROZEN.with_changes(end=date(2025, 4, 30), segments=SEGMENTS)


def expected_cagr(ending, starting, since, until):
    """구간 연복리 = (끝 ÷ 처음)^(365.25 ÷ 경과일) − 1 (80자리)."""
    with localcontext(prec=80):
        days = D((until - since).total_seconds()) / D(86400)
        return (D(ending) / D(starting)) ** (D("365.25") / days) - 1


def evening(day):
    return datetime(day.year, day.month, day.day, 18, tzinfo=KST)


def traded_until(run, day):
    return [row for row in run.result.trades if row["session_date"] <= day]


def valued_until(run, moment):
    return [row for row in run.result.valuations if row["at"] <= moment]


def ending(run):
    return run.result.valuations[-1]["equity_krw"]


class Table5FillsPeriodsAndInputs(unittest.TestCase):
    def test_one_tick_penetration_no_replacement_and_path_order(self):
        """kg 1·kp 1(σ 1%), 1차 999주 @10,000. 2차 매수 지정가 9,900(관통 9,895), 1차 익절 지정가 10,105(관통 10,110).

        1/3 봉(고 10,105·저 9,895): 고가를 먼저 지나면 익절 주문이 관통 없이 미체결이고 그날 다른 주문으로 바꾸지 않는다.
        저가를 먼저 지나면 9,900 매수(1,009주 = floor(10,000,000 ÷ 9,901.485))가 9,895 관통으로 체결된다.
        1/6 봉(고 10,110): 고가 먼저면 1차 익절 10,105 체결, 저가 먼저면 2차(9,900) 익절 지정가 10,005 체결.
        """
        changes = {**bar(J(3), 10000, 10105, 9895, 10000), **bar(J(6), 10000, 10110, 9895, 10000)}
        data = build(weekdays(J(2), J(8)), {J(2): "10000"}, changes=changes, basis=BASIS)
        config = cfg("50", "1", "1", 5)
        high_first, low_first = split(data, config, OHLC), split(data, config, OLHC)
        self.assertEqual(trades(high_first), [(J(2), "BUY", 1, 999, D(10000)), (J(6), "SELL", 1, 999, D(10105))])
        self.assertEqual(trades(low_first), [(J(2), "BUY", 1, 999, D(10000)), (J(3), "BUY", 2, 1009, D(9900)),
                                             (J(6), "SELL", 2, 1009, D(10005))])
        self.assertEqual([row["reason"] for row in low_first.result.trades],
                         ["manual_first_entry", "buy_condition_touch", "sell_condition_touch"])
        self.assertEqual(high_first.result.rejections, [])
        result = run_point(data, config, J(2))  # 두 경로의 체결 건수가 따로 기록된다
        self.assertEqual([result["exploration"][path]["split_ledger"]["base"] for path in PATHS],
                         [{"status": "ok", "buys": 1, "sells": 1}, {"status": "ok", "buys": 2, "sells": 1}])

    def test_opening_gaps_fill_at_the_open(self):
        """시가 9,800은 2차 지정가 9,900보다 낮게 열려 시가에 체결(1,020주). 10,400 갭 상승은 하루 한 차수씩 시가 익절."""
        data = build(weekdays(J(2), J(9)), {J(2): "10000", J(3): "9800", J(6): "10400"}, basis=BASIS)
        for path in PATHS:
            run = split(data, cfg("50", "1", "1", 5), path)
            self.assertEqual(trades(run), [(J(2), "BUY", 1, 999, D(10000)), (J(3), "BUY", 2, 1020, D(9800)),
                                           (J(6), "SELL", 2, 1020, D(10400)), (J(7), "SELL", 1, 999, D(10400))])
            self.assertEqual([row["reason"] for row in run.result.trades],
                             ["manual_first_entry", "buy_condition_gap", "sell_condition_gap", "sell_condition_gap"])

    def test_sell_first_when_sell_and_buy_are_both_possible(self):
        """kg 2·kp 1. 1/8 시가 9,750에서 3차(9,600 매수, 순수익 1.53%) 익절과 2차 재매수(1차 10,000 대비 −2.5%)가
        함께 가능하다 → 매도를 택하고, 매수는 다음 날(1,025주 = floor(10,000,000 ÷ 9,751.4625))로 넘어간다."""
        prices = {J(2): "10000", J(3): "9800", J(6): "9600", J(7): "9950", J(8): "9750"}
        data = build(weekdays(J(2), J(10)), prices, basis=BASIS)
        for path in PATHS:
            run = split(data, cfg("50", "2", "1", 5), path)
            self.assertEqual(trades(run), [(J(2), "BUY", 1, 999, D(10000)), (J(3), "BUY", 2, 1020, D(9800)),
                                           (J(6), "BUY", 3, 1041, D(9600)), (J(7), "SELL", 2, 1020, D(9950)),
                                           (J(8), "SELL", 3, 1041, D(9750)), (J(9), "BUY", 2, 1025, D(9750))])
            days = [row["session_date"] for row in run.result.trades]
            self.assertEqual(len(days), len(set(days)))  # 하루 한 주문

    def test_year_first_session_is_month_first_and_refresh_day(self):
        """12/30 분할 순자산 49,797,202.1 → 1월 예산 9,959,440.42. 새 σ 0.5%(간격 1%)로 1/2 시가 9,700에 3차
        floor(9,959,440.42 ÷ 9,701.455) = 1,026주. 2차는 매수 당시 익절 기준 3%를 유지한 채 1/3에 익절한다."""
        changes = bar(J(2), 9700, 10150, 9700, 9800)
        data = build(YEAR_END, {DEC(2): "10000", DEC(3): "9800", J(3): "10150"}, changes=changes, basis=BASIS,
                     extra_vol={J(2): estimate(J(2), "0.005")})
        for path in PATHS:
            run = split(data, cfg("50", "2", "3", 5), path)
            self.assertEqual(trades(run), [(DEC(2), "BUY", 1, 999, D(10000)), (DEC(3), "BUY", 2, 1020, D(9800)),
                                           (J(2), "BUY", 3, 1026, D(9700)), (J(3), "SELL", 2, 1020, D(10150)),
                                           (J(6), "SELL", 3, 1026, D(10150))])
            self.assertEqual([p.take_rate for p in run.result.positions], [D("0.03")])
            refresh = [row for row in run.result.cash_events if row["kind"] == "budget_refresh"]
            self.assertEqual((refresh[1]["at"], refresh[1]["source_total_krw"]), (at(J(2)), D("49797202.1")))

    def test_monthly_reentry_retry_and_unsettled_month_end_balance(self):
        """1/24 전량 매도(결제 2/3) → 1/31 평가에 미결제 매도대금 10,388,041.56이 남는다 → 2/3 거래량 0으로 재진입
        불성립 → 3/3 휴장, 3/4 재진입 floor(10,079,308.612 ÷ 10,401.56) = 969주(스트레스 965주)."""
        changes = {J(31): {"adv20": D(1)}, FEB3: {"volume": D(0)}}
        data = build(CASE2_DAYS, {J(2): "10000", J(24): "10400"}, changes=changes, basis=BASIS)
        config = cfg("50", "2", "3", 5)
        for cost, first, again in ((BASE, 999, 969), (STRESS, 997, 965)):
            for path in PATHS:
                run = split(data, config, path, cost)
                self.assertEqual(trades(run), [(J(2), "BUY", 1, first, D(10000)), (J(24), "SELL", 1, first, D(10400)),
                                               (MAR4, "BUY", 1, again, D(10400))])
                self.assertEqual([(a["date"], a["outcome"]) for a in run.detail["reentries"]],
                                 [(FEB3, "no_order"), (MAR4, "filled")])
        run = split(data, config)
        month_end = next(row for row in run.result.valuations if row["at"] == evening(J(31)))
        self.assertEqual(month_end["cash"]["KRW"] - month_end["settled_cash"]["KRW"], D("10388041.56"))
        refresh = {row["at"]: row["source_total_krw"] for row in run.result.cash_events if row["kind"] == "budget_refresh"}
        self.assertEqual((refresh[at(FEB3)], refresh[at(MAR4)]), (D("50396543.06"), D("50396543.06")))
        result = run_point(data, config, J(2))
        self.assertEqual(result["exploration"][OHLC]["reentry_counts"]["base"],
                         {"status": "ok", "attempts": 2, "filled": 1, "unfilled": {"no_order": 1}})

    def test_segments_start_from_previous_close_and_first_one_from_capital(self):
        """구간을 넘는 매매가 없도록 kg 4·kp 6. 전략 = 5,998주 + 현금 40,011,003, 단독 보유 = 9,998주 + 현금 5,003.
        가격이 그대로거나 내린 구간(S1·S3)은 전략이 앞서고, 오른 구간(S2·S4)은 단독 보유가 앞선다."""
        data = build(SEGMENT_DAYS, SEGMENT_PRICES, basis=BASIS)
        result = run_point(data, cfg("50", "4", "6", 5, SEGMENT_BASE), J(2))
        ends = [(J(31), 10000), (date(2025, 2, 28), 10300), (date(2025, 3, 31), 10000), (date(2025, 4, 30), 10500)]
        for path in PATHS:
            base = paths_of(result)[path]["base"]
            for name, shares, cash in (("strategy", 5998, 40011003), ("hold", 9998, 5003)):
                previous, since = 100000000, at(J(2))  # 첫 구간은 시작 시가 직전 원금부터
                for row, (day, price) in zip(base[name]["segments"], ends):
                    equity = shares * price + cash
                    with self.subTest(path=path, account=name, segment=row["segment"]):
                        self.assertEqual(row["cagr"], expected_cagr(equity, previous, since, evening(day)))
                    previous, since = equity, evening(day)  # 다음 구간은 이 구간 마지막 거래일 18시 순자산부터
            self.assertEqual([row["evaluation_days"] for row in base["hold"]["segments"]], [8, 2, 2, 2])
            self.assertEqual(base["strategy"]["ending_equity_krw"], D(5998 * 10500 + 40011003))
        self.assertEqual(result["run_start"]["last_evaluation"], evening(date(2025, 4, 30)))
        point = reading.point_input(plain(result), frame_of(result, SEGMENTS))
        self.assertEqual(judge.common_segments(point["paths"], T), ["S1", "S3"])
        self.assertEqual(verdict_of(result, SEGMENTS)["status"], FAIL)
        display = reading.point_display(plain(result), frame_of(result, SEGMENTS))["paths"][OHLC]
        self.assertEqual(display["worst_segments"], ["S4"])  # 단독 보유가 가장 크게 앞선 구간(+5% 상승)
        self.assertEqual(display["total_delta_positive"], False)
        share = display["split_share"]["base"]  # 분할 몫 999주 + 현금 40,008,501.5 대 같은 원금(5천만) 보유 4,999주
        self.assertEqual(share["split"], D(999 * 10500 + D("40008501.5")) / D(50000000) - 1)
        self.assertEqual(share["hold_same_capital"], (D(4999 * 10500) + D("2501.5")) / D(50000000) - 1)

    def test_later_starts_use_new_capital_year_first_volatility_and_same_end(self):
        """+5·+20에 해당하는 늦은 시작: 첫 구간만 실제 시작일부터 짧아지고, 변동성은 같은 연초 값, 종료 평가시각은 같다."""
        data = build(SEGMENT_DAYS, SEGMENT_PRICES, basis=BASIS)
        config = cfg("50", "4", "6", 5, SEGMENT_BASE)
        for start, sessions in ((J(9), 3), (J(31), 1)):
            with self.subTest(start=start):
                result = run_point(data, config, start)
                hold = paths_of(result)[OHLC]["base"]["hold"]["segments"]
                self.assertEqual([row["segment"] for row in hold], ["S1", "S2", "S3", "S4"])
                self.assertEqual(hold[0]["evaluation_days"], sessions)
                self.assertEqual(hold[0]["cagr"], expected_cagr(9998 * 10000 + 5003, 100000000, at(start), evening(J(31))))
                self.assertEqual(hold[1]["cagr"], expected_cagr(9998 * 10300 + 5003, 9998 * 10000 + 5003,
                                                                evening(J(31)), evening(date(2025, 2, 28))))
                self.assertEqual(result["run_start"]["last_evaluation"], evening(date(2025, 4, 30)))
                self.assertEqual(orders(result)[0], {"core": 4999, "split": 999, "hold": 9998})
                run = open_ledgers(data, config, start)["runs"]["base"][OHLC]["M"]
                self.assertEqual((run.detail["volatility_start"], run.detail["rates"]), (J(2), (D("0.04"), D("0.06"))))

    def test_changing_future_bars_keeps_earlier_orders_and_valuations(self):
        """1/8 이후 일봉을 바꿔도 1/7까지의 주문·평가는 같다(모든 장부, 두 비용, 두 경로)."""
        config, days = cfg("50", "2", "1", 5), weekdays(J(2), J(10))
        prices = {J(2): "10000", J(3): "9800", J(6): "9600", J(7): "9950", J(8): "9750"}
        first = build(days, prices, basis=BASIS)
        second = build(days, {**prices, J(8): "10300", J(9): "9500"}, basis=BASIS)
        self.assertNotEqual(inputs_digest(first), inputs_digest(second))
        left, right = open_ledgers(first, config, J(2)), open_ledgers(second, config, J(2))
        cut = evening(J(7))
        compared = 0
        for cost in ("base", "stress"):
            for path in PATHS:
                pairs = [(left["runs"][cost][path][name], right["runs"][cost][path][name]) for name in ("core", "hold", "M")]
                pairs.append((left["split_hold"][cost][path], right["split_hold"][cost][path]))
                for one, other in pairs:
                    self.assertEqual(traded_until(one, J(7)), traded_until(other, J(7)))
                    self.assertEqual(valued_until(one, cut), valued_until(other, cut))
                    compared += len(valued_until(one, cut))
        self.assertEqual(compared, 2 * 2 * 4 * 4)
        self.assertNotEqual(trades(left["runs"]["base"][OHLC]["M"]), trades(right["runs"]["base"][OHLC]["M"]))
        a, b = run_point(first, config, J(2)), run_point(second, config, J(2))
        self.assertEqual(paths_of(a)[OHLC]["start_orders"], paths_of(b)[OHLC]["start_orders"])


# ---------------------------------------------------------------------------
# (a) 승인 코드와의 일치, (c) 캐시
# ---------------------------------------------------------------------------
def year_boundary(refreshed="0.005"):
    changes = bar(J(2), 9700, 10150, 9700, 9800)
    return build(YEAR_END, {DEC(2): "10000", DEC(3): "9800", J(3): "10150"}, changes=changes, basis=BASIS,
                 extra_vol={J(2): estimate(J(2), refreshed)})


def case2():
    changes = {J(31): {"adv20": D(1)}, FEB3: {"volume": D(0)}}
    return build(CASE2_DAYS, {J(2): "10000", J(24): "10400"}, changes=changes, basis=BASIS)


class ApprovedCodeParity(unittest.TestCase):
    """(a) 9차·kg 2·kp 3·코어 80%·M을 탐색 실행기로 돌린 값 = 같은 설정의 study.run_start."""

    def cases(self):
        gap = VolatilityUnavailable("synthetic missing 251st close")
        dec = date(2024, 12, 30)
        late = weekdays(dec, J(10), {date(2024, 12, 31), J(1)})
        return {"monthly_reentry": (case2(), CASE2_DAYS[0], None),
                "late_start": (case2(), CASE2_DAYS[5], None),
                "year_boundary": (year_boundary(), DEC(2), None),
                "closed_buys": (year_boundary("0.0001"), DEC(2), None),
                "stress_rejected": (flat("0.002"), J(2), None),
                "hold_rejected": (build(weekdays(J(2), J(10)), {J(2): "10000"}, basis=BASIS,
                                        changes={J(2): {"adv20": D(1644000000)}}), J(2), None),
                "split_start_failed": (build(weekdays(J(2), J(10)), {J(2): "3000000", J(3): "3100000"}, basis=BASIS),
                                       J(2), None),
                "capacity_floor": (flat("0.01"), J(2), D(10) ** 20),
                "capacity_ok": (flat("0.01"), J(2), D(100000000)),
                "observation_ended": (build(late, {dec: "10000", J(2): "9700", J(3): "10500"}, basis=BASIS,
                                            extra_vol={J(2): gap}), dec, None)}

    def test_run_start_values_are_identical(self):
        config = cfg("80", "2", "3", 9)
        self.assertEqual(SPEC.setting(SPEC.baseline_number()).describe(), SPEC.setting(number("80", "2", "3", 9)).describe())
        for name, (data, start, floor) in self.cases().items():
            with self.subTest(name):
                mine = run_point(data, config, start, nominal_floor=floor)
                self.assertEqual(mine["run_start"], study.run_start(data, config, start, nominal_floor=floor))
                self.assertEqual((mine["mode"], mine["applied_config_sha256"]), ("M", config.applied_sha256()))
                self.assertEqual(slot_problems(mine), [])

    def test_other_grid_points_also_match(self):
        for key in (("50", "1", "1", 5), ("50", "4", "6", 9), ("80", "1", "6", 5)):
            config = cfg(*key)
            for data, start in ((case2(), CASE2_DAYS[0]), (year_boundary(), DEC(2))):
                self.assertEqual(run_point(data, config, start)["run_start"], study.run_start(data, config, start))

    def test_runner_takes_one_mode_and_rechecks_volatility(self):
        with self.assertRaisesRegex(ValueError, "exactly one reentry mode"):
            run_point(case2(), FROZEN, CASE2_DAYS[0])
        data = flat("0.01")
        late = replace(data, volatility={J(2): VolatilityEstimate(D("0.01"), at(J(6), 6, 30))})
        with self.assertRaisesRegex(study.InputIdentityError, "not before"):
            run_point(late, cfg("80", "2", "3", 9), J(3))

    def test_observations_agree_with_the_ledgers(self):
        data, config = year_boundary(), cfg("50", "2", "3", 5)
        result, ledgers = run_point(data, config, DEC(2)), open_ledgers(data, config, DEC(2))
        for cost in ("base", "stress"):
            for path in PATHS:
                group, extra = ledgers["runs"][cost][path], result["exploration"][path]
                sides = [row["side"] for row in group["M"].result.trades]
                self.assertEqual(extra["split_ledger"][cost],
                                 {"status": "ok", "buys": sides.count("BUY"), "sells": sides.count("SELL")})
                self.assertEqual(extra["split_share"][cost]["split"]["ending_equity_krw"], ending(group["M"]))
                self.assertEqual(extra["core_share"][cost]["ending_equity_krw"] + ending(group["M"]),
                                 paths_of(result)[path][cost]["strategy"]["ending_equity_krw"])
                self.assertEqual(extra["split_share"][cost]["hold_same_capital"]["ending_equity_krw"],
                                 ending(ledgers["split_hold"][cost][path]))
        base = result["exploration"][OHLC]["split_ledger"]["base"]
        self.assertEqual((base["buys"], base["sells"]), (3, 2))  # 999·1,020·1,026주 매수, 2·3차 익절


def swapped_key(key, swap):
    """캐시 열쇠 (입력, 경로, 비용, 원금, …)에서 원금(8천만 ↔ 1억) 또는 비용(기본 ↔ 스트레스)만 바꾼 열쇠."""
    capitals = {D(80000000): D(100000000), D(100000000): D(80000000)}
    if swap == "capital":
        return (*key[:3], capitals.get(key[3], key[3]), *key[4:])
    return (*key[:2], STRESS if key[2] == BASE else BASE, *key[3:])


class LedgerCacheRules(unittest.TestCase):
    """(c) 보유·코어 장부 캐시는 입력 묶음·종목·시작일·종료일·비용·경로·원금이 모두 같을 때만 다시 쓴다."""

    def settings(self):
        return [cfg(*key) for key in (("50", "2", "3", 5), ("50", "1", "1", 9), ("80", "2", "3", 9), ("80", "4", "6", 5))]

    def test_cached_and_uncached_results_are_equal(self):
        data, shared = year_boundary(), Bench()
        for config in self.settings():
            cached = run_point(data, config, DEC(2), bench=shared)
            fresh = run_point(data, config, DEC(2), bench=Bench(cache=NoCache()))
            self.assertEqual(cached, fresh)
        # 장부 종류: 보유 1억(4) + 코어 5천만(4, 분할 원금 5천만 보유와 같은 열쇠) + 코어 8천만(4) + 분할 원금 2천만 보유(4)
        self.assertEqual((shared.cache.misses, len(shared.cache.runs)), (16, 16))
        self.assertEqual(shared.cache.hits, 4 * 12 - 16)

    def test_core_fifty_reuses_the_core_ledger_for_the_split_share(self):
        bench = Bench()
        ledgers = open_ledgers(year_boundary(), cfg("50", "2", "3", 5), DEC(2), bench=bench)
        for cost in ("base", "stress"):
            for path in PATHS:
                self.assertIs(ledgers["split_hold"][cost][path], ledgers["runs"][cost][path]["core"])
        other = open_ledgers(year_boundary(), cfg("80", "2", "3", 9), DEC(2), bench=Bench())
        self.assertIsNot(other["split_hold"]["base"][OHLC], other["runs"]["base"][OHLC]["core"])
        self.assertEqual(other["split_hold"]["base"][OHLC].result.trades[0]["quantity"], 1999)  # 2천만 ÷ 10,001.5

    def test_different_cost_capital_start_path_or_bundle_never_share(self):
        data, config = year_boundary(), cfg("80", "2", "3", 9)
        bench = Bench()
        first = open_ledgers(data, config, DEC(2), bench=bench)
        kinds = [first["runs"][cost][path][name] for cost in ("base", "stress") for path in PATHS for name in ("core", "hold")]
        self.assertEqual(len({id(run) for run in kinds}), 8)  # 비용 2 × 경로 2 × 원금 2가 모두 다른 장부
        self.assertEqual(bench.cache.misses, 12)
        open_ledgers(data, config, DEC(3), bench=bench)  # 시작일이 다르면 모두 새로 계산
        self.assertEqual((bench.cache.misses, bench.cache.hits), (24, 0))
        other_bundle = replace(data, provenance={"daily_sha256": "f" * 64})  # 같은 날짜·상품이라도 입력 묶음이 다르면
        open_ledgers(other_bundle, config, DEC(2), bench=bench)
        self.assertEqual((bench.cache.misses, bench.cache.hits), (36, 0))
        open_ledgers(year_boundary("0.004"), config, DEC(2), bench=bench)  # 연초 변동성만 달라도 다른 입력으로 본다
        self.assertEqual((bench.cache.misses, bench.cache.hits), (48, 0))
        shorter = replace(data, bars=data.bars[:-1])  # 종료일이 다르면 공유하지 않는다
        open_ledgers(shorter, config, DEC(2), bench=bench)
        self.assertEqual((bench.cache.misses, bench.cache.hits), (60, 0))
        open_ledgers(year_boundary(), config, DEC(2), bench=bench)  # 내용이 모두 같은 입력은 다시 쓴다
        self.assertEqual((bench.cache.misses, bench.cache.hits), (60, 12))

    def test_mixed_cache_entries_are_caught_by_the_uncached_comparison(self):
        """캐시 열쇠가 섞여 다른 원금·비용의 장부가 들어오면 캐시 없는 실행과 값이 달라져 드러난다."""
        data, config = year_boundary(), cfg("80", "2", "3", 9)
        fresh = run_point(data, config, DEC(2), bench=Bench(cache=NoCache()))
        for swap in ("capital", "cost"):
            with self.subTest(swap):
                bench = Bench()
                open_ledgers(data, config, DEC(2), bench=bench)
                keys = list(bench.cache.runs)
                bench.cache.runs = {key: bench.cache.runs[swapped_key(key, swap)] for key in keys}
                self.assertNotEqual(run_point(data, config, DEC(2), bench=bench), fresh)


# ---------------------------------------------------------------------------
# 묶음 6: 재현과 누락 — 단계 실행·봉인·재개·조립
# ---------------------------------------------------------------------------
WORLD_DAYS = weekdays(J(2), date(2025, 2, 7)) + [date(2025, 3, 3), date(2025, 3, 4), date(2025, 4, 1), date(2025, 4, 2)]
WORLD_PRICES = [10000, 9900, 9800, 9700, 9900, 10100, 10300, 10400, 10200, 10000, 9800, 9600, 9800, 10000, 10200, 10400,
                10600, 10400, 10200, 10000, 9900, 10100, 10300, 10500, 10300, 10100, 10000, 10200, 10400, 10300, 10500]
WORLD_SEGMENTS = (Segment("S1", date(2025, 1, 1), date(2025, 1, 31)), Segment("S2", date(2025, 2, 1), date(2025, 2, 28)),
                  Segment("S3", date(2025, 3, 1), date(2025, 3, 31)), Segment("S4", date(2025, 4, 1), date(2025, 4, 30)))
WORLD_BASE = FROZEN.with_changes(end=WORLD_DAYS[-1], segments=WORLD_SEGMENTS)
WORLD_SPEC = replace(SPEC, primary_start=J(2), end=WORLD_DAYS[-1], segments=WORLD_SEGMENTS,
                     reconfirm={"offsets": (5, 20), "starts": (WORLD_DAYS[5], WORLD_DAYS[20])},
                     extension={**SPEC.extension, "start": J(2)})
EVERYTHING_PASSES = {"base_edge_min": D(-1), "stress_edge_gt": D(-1), "segment_edge_gt": D(-1), "common_segments_min": 0}
LOG_LINE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ stage[12] [a-z0-9@-]+ "
                      r"(done|skipped|error) \d+\.\ds( E_[A-Z]+)?$")
VERDICT_WORDS = ("pass", "fail", "undetermined", "hold", "selected", "통과", "미달", "미확정", "cagr", "delta")


def world(sigma="0.01", symbol=SYMBOL):
    data = build(WORLD_DAYS, dict(zip(WORLD_DAYS, map(str, WORLD_PRICES))), sigma=sigma, basis=BASIS)
    return data if symbol == SYMBOL else as_symbol(data, symbol)


class MemorySource:
    """합성 입력기: (종목, 시작일)마다 미리 만든 입력 또는 예상한 자료 문제(DataHold)를 돌려준다."""

    def __init__(self, inputs, name="world", frozen=False):
        self.inputs, self.name, self.frozen = inputs, name, frozen

    def identity(self):
        digests = {f"{symbol}@{start}": "held" if isinstance(value, Exception) else inputs_digest(value)
                   for (symbol, start), value in sorted(self.inputs.items())}
        return {"actual": {"synthetic": True, "name": self.name, "inputs": digests},
                "registered_inputs_frozen": self.frozen}

    def load(self, symbol, start):
        value = self.inputs[(symbol, start)]
        if isinstance(value, Exception):
            raise value
        return value


def world_inputs(sigma="0.01", held=()):
    keys = [(SYMBOL, start) for start in (WORLD_SPEC.primary_start, *WORLD_SPEC.reconfirm["starts"])]
    keys += [(symbol, WORLD_SPEC.extension["start"]) for symbol in WORLD_SPEC.extension["symbols"]]
    return {key: DataHold("synthetic: no input for this start") if key in held else world(sigma, key[0]) for key in keys}


def world_context(spec=WORLD_SPEC, **options):
    name = options.pop("name", "world")
    return open_synthetic_context(MemorySource(world_inputs(**options), name), spec=spec, base=WORLD_BASE)


def sealed(folder):
    """봉인 단위의 기록(실행 시간은 별도 봉인 기록에 있어 단위 내용은 실행마다 같다)."""
    return {path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(Path(folder, "units").glob("*.json"))}


def reseal(stage_folder, name, change):
    """단위 파일을 고치고 봉인 기록의 해시까지 맞춰 고친다(해시 검사를 지나 기록 검사에 닿게 하는 시험용)."""
    path = Path(stage_folder, "units", name)
    record = json.loads(path.read_text(encoding="utf-8"))
    change(record)
    text = sealing.dump(record)
    path.write_text(text, encoding="utf-8")
    seal_path = Path(stage_folder, "sealed", "seals", name)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal_path.write_text(sealing.dump(sealing.with_seal({**seal, "sha256": sealing.digest(text)})), encoding="utf-8")


class _Workspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.lines = []
        loader._source.cache_clear()

    def tearDown(self):
        loader._source.cache_clear()
        self.tmp.cleanup()

    def stage1(self, ctx, name="run", workers=1):
        stages.run_stage1(ctx, self.out / name, workers=workers, echo=self.lines.append)
        return self.out / name


class Table6SequentialAndParallel(_Workspace):
    """(b) 같은 합성 입력에서 순차·병렬 실행의 36개 결과와 선택 영역이 같다."""

    def test_same_units_results_and_selection(self):
        ctx = world_context()
        one, many = self.stage1(ctx, "sequential", 1), self.stage1(ctx, "parallel", 4)
        self.assertEqual(sealed(one / "stage1"), sealed(many / "stage1"))
        self.assertEqual(len(sealed(one / "stage1")), 36)
        for name in sorted(path.name for path in (one / "stage1/units").iterdir()):  # 단위 내용은 실행마다 같은 바이트다
            self.assertEqual((one / "stage1/units" / name).read_bytes(), (many / "stage1/units" / name).read_bytes())
        first, second = stages.assemble1(ctx, one), stages.assemble1(ctx, many)
        for key in ("settings", "regions", "selection", "summary", "isolated_passes"):
            self.assertEqual(json.loads(sealing.dump(first[key])), json.loads(sealing.dump(second[key])))
        self.assertEqual([row["number"] for row in first["settings"]], list(range(1, 37)))
        for name in ("selection.json", "summary.md", "result.json", "lock.json"):
            self.assertEqual((one / "stage1" / name).read_text(), (many / "stage1" / name).read_text())
        self.assertGreater(len({json.dumps(unit["result"], sort_keys=True) for unit in sealed(one / "stage1").values()}), 30)
        for unit in sealed(one / "stage1").values():
            self.assertEqual((unit["exploration"], unit["formal_run"], unit["identity"]["exploration"],
                              unit["identity"]["formal_run"], unit["identity"]["scope"]),
                             (True, False, True, False, "synthetic"))
        self.assertEqual((first["exploration"], first["formal_run"], first["selection"]["exploration"],
                          first["selection"]["formal_run"]), (True, False, True, False))
        self.assertEqual({(row["exploration"], row["formal_run"]) for row in first["settings"]}, {(True, False)})

    def test_units_carry_per_setting_applied_hash_and_both_sources(self):
        ctx = world_context()
        out = self.stage1(ctx, workers=4)
        units = sealed(out / "stage1")
        for row in WORLD_SPEC.settings():
            record = units[f"setting-{row.number:02d}.json"]
            expected = setting_config(WORLD_BASE, WORLD_SPEC, row).applied_sha256()
            self.assertEqual((record["identity"]["applied_config_sha256"], record["setting"]), (expected, row.describe()))
            self.assertEqual(record["result"]["applied_config_sha256"], expected)  # 본문에도 실제로 쓴 적용 해시
            self.assertEqual(record["identity"]["approved_source_sha256"], SPEC.approved["source_sha256"])
            self.assertEqual(record["identity"]["exploration_source_sha256"],
                             identity.exploration_source()["exploration_source_sha256"])
            self.assertEqual(record["identity"]["grid_file_sha256"], SPEC.sha256)
            self.assertEqual(record["identity"]["grid_applied_sha256"], WORLD_SPEC.applied_sha256())
            self.assertEqual((record["identity"]["end"], record["identity"]["mode"]), ("2025-04-02", "M"))
            self.assertNotIn("timing", record)
        self.assertEqual(len({record["identity"]["applied_config_sha256"] for record in units.values()}), 36)
        self.assertEqual(list((out / "stage1").rglob(".*.tmp")), [])  # 임시 파일이 남지 않는다

    def test_each_unit_is_sealed_with_its_content_hash_when_computed(self):
        out = self.stage1(world_context(), workers=4)
        for path in sorted((out / "stage1/units").iterdir()):
            seal = sealing.read_sealed(out / "stage1/sealed/seals" / path.name, stages.SEAL_SCHEMA)
            self.assertEqual((seal["unit"], seal["sha256"]), (path.stem, sealing.file_digest(path)))
            self.assertGreater(seal["timing"]["seconds"], 0)
            self.assertEqual(seal["timing"]["by_kind_counts"]["split"], 4)
            self.assertEqual((path.stat().st_mode & 0o777, (out / "stage1/sealed/seals" / path.name).stat().st_mode & 0o777),
                             (0o600, 0o600))  # 봉인 원본은 공개 로그와 다른 권한으로 둔다
        self.assertEqual((out / "stage1/progress.log").stat().st_mode & 0o077, 0o044)


class Table6SealedProgressLog(_Workspace):
    """(e) 조립 전 진행 로그에는 번호·소요시간·기술 상태 코드만 있다(성과 수치·판정 상태 없음)."""

    def test_log_has_only_technical_codes_even_with_cost_failures_and_holds(self):
        # σ 0.5%: kg 1 설정은 스트레스 개설 미달(판정으로는 미달), 나머지는 수치가 있는 행 → 로그는 구별하지 않는다
        out = self.stage1(world_context(sigma="0.005"), "cost", 2)
        held = self.stage1(world_context(held={(SYMBOL, J(2))}, name="held"), "held", 2)
        for folder in (out, held):
            text = (folder / "stage1/progress.log").read_text()
            lines = text.splitlines()
            self.assertEqual(len(lines), 36)
            for line in lines:
                self.assertRegex(line, LOG_LINE)
                self.assertEqual(line.split()[3], "done")
            self.assertEqual(sorted(line.split()[2] for line in lines), [f"setting-{n:02d}" for n in range(1, 37)])
            self.assertFalse([word for word in VERDICT_WORDS if word in text.lower()])
        self.assertEqual(sorted(self.lines), sorted((out / "stage1/progress.log").read_text().splitlines()
                                                    + (held / "stage1/progress.log").read_text().splitlines()))
        rows = stages.assemble1(world_context(sigma="0.005"), out)["settings"]
        self.assertEqual({row["verdict"]["status"] for row in rows if row["kg"] == "1"}, {FAIL})
        self.assertTrue(all(f"{OHLC}:stress_start_rejected_by_cost_guard" in row["verdict"]["reasons"]
                            for row in rows if row["kg"] == "1"))
        # 스트레스만 거부된 설정에도 기본 비용 두 경로의 공통 승리 구간을 싣는다
        self.assertTrue(all(isinstance(row["verdict"]["common_segments"], list) for row in rows if row["kg"] == "1"))

    def test_resume_skips_sealed_units_and_logs_only_codes(self):
        ctx = world_context()
        out = self.stage1(ctx, workers=4)
        before = {path.name: path.stat().st_mtime_ns for path in (out / "stage1/units").iterdir()}
        (out / "stage1/units/setting-07.json").unlink()  # 봉인까지 가지 못하고 끊긴 단위: 단위도 봉인 기록도 없다
        (out / "stage1/sealed/seals/setting-07.json").unlink()
        self.lines.clear()
        self.stage1(ctx, workers=2)
        codes = {line.split()[2]: line.split()[3] for line in self.lines}
        self.assertEqual((codes["setting-07"], set(codes.values()), len(codes)), ("done", {"done", "skipped"}, 36))
        after = {path.name: path.stat().st_mtime_ns for path in (out / "stage1/units").iterdir()}
        self.assertEqual({name for name in before if before[name] != after[name]}, {"setting-07.json"})

    def test_sealed_unit_that_vanished_is_not_recomputed_and_unsealed_leftover_is(self):
        ctx = world_context()
        out = self.stage1(ctx, workers=4)
        unit, seal = out / "stage1/units/setting-07.json", out / "stage1/sealed/seals/setting-07.json"
        content = unit.read_text()
        unit.unlink()
        with self.assertRaisesRegex(IdentityError, "setting-07: sealed unit file is missing"):
            self.stage1(ctx)
        with self.assertRaisesRegex(stages.AssemblyError, "missing setting-07.json"):
            stages.assemble1(ctx, out)
        unit.write_text("{}")  # 봉인 기록 없이 남은 쓰기는 결과가 아니다: 지우고 다시 계산한다
        seal.unlink()
        self.lines.clear()
        self.stage1(ctx)
        self.assertEqual([line.split()[2:4] for line in self.lines if line.split()[3] == "done"], [["setting-07", "done"]])
        self.assertEqual(unit.read_text(), content)  # 같은 입력·같은 판본이면 같은 내용이다
        stages.assemble1(ctx, out)


LEAK_SCRIPT = """
import os, sys
from pathlib import Path
from unittest.mock import patch
import tests.test_research_etf_kodex_grid as t

real = t.stages.run_point

def noisy(inputs, config, start, **options):
    print("LEAK-PRINT delta_cagr=0.0123 pass")
    print("LEAK-WARN sells=7", file=sys.stderr)
    os.write(1, b"LEAK-FD1\\n")
    os.write(2, b"LEAK-FD2\\n")
    if config.kg == 4:
        raise RuntimeError("LEAK-EXCEPTION scheduled reentry session has no bar")
    return real(inputs, config, start, **options)

with patch.object(t.stages, "run_point", side_effect=noisy):
    try:
        t.stages.run_stage1(t.world_context(), Path(sys.argv[1]), workers=int(sys.argv[2]))
    except t.stages.StageError as error:
        print("stopped:", error)
"""
REPO = Path(__file__).resolve().parents[1]


class Table6WorkerOutputIsSealed(_Workspace):
    """공개 경로(프로세스의 표준출력·표준오류, 진행 로그, 오류 표시)에는 번호·시간·기술 코드만 나온다."""

    def run_noisy(self, workers):
        out = self.out / f"workers-{workers}"
        done = subprocess.run([sys.executable, "-B", "-c", LEAK_SCRIPT, str(out), str(workers)], cwd=REPO,
                              capture_output=True, text=True, timeout=300)
        return out / "stage1", done

    def test_real_process_output_has_no_worker_text_sequential_and_parallel(self):
        for workers in (1, 3):
            with self.subTest(workers=workers):
                folder, done = self.run_noisy(workers)
                self.assertEqual(done.returncode, 0, done.stderr[-500:])
                markers = sorted(path.name for path in (folder / "errors").iterdir())
                failed = [f"setting-{n:02d}" for n in range(1, 37) if WORLD_SPEC.setting(n).kg == D(4)]
                self.assertEqual(markers, [f"{name}.error.txt" for name in failed])
                public = [done.stdout, done.stderr, (folder / "progress.log").read_text()]
                public += [(folder / "errors" / name).read_text() for name in markers]
                self.assertFalse([text for text in public if "LEAK" in text or "Traceback" in text])
                self.assertEqual(done.stderr, "")
                lines = done.stdout.splitlines()
                self.assertEqual(lines[-1], f"stopped: 12 unit(s) ended with a technical error: {failed}")
                self.assertTrue(all(LOG_LINE.match(line) for line in lines[:-1]))
                self.assertEqual(sorted(line.split()[-1] for line in lines[:-1] if " error " in line), ["E_ENGINE"] * 12)
                for name in failed:  # 오류 표시에는 단위 이름과 기술 코드만 있다
                    self.assertEqual((folder / "errors" / f"{name}.error.txt").read_text(), f"{name} E_ENGINE\n")
                    sealed_log = (folder / "sealed/logs" / f"{name}.log").read_text()
                    for text in ("LEAK-PRINT", "LEAK-WARN", "LEAK-FD1", "LEAK-FD2", "LEAK-EXCEPTION", "Traceback"):
                        self.assertIn(text, sealed_log)
                    self.assertEqual((folder / "sealed/logs" / f"{name}.log").stat().st_mode & 0o777, 0o600)
                self.assertIn("LEAK-FD1", (folder / "sealed/logs/setting-01.log").read_text())

    def test_error_codes_carry_no_message(self):
        cases = ((RuntimeError("delta 0.5"), "E_ENGINE"), (ValueError("x"), "E_ENGINE"), (OSError("disk"), "E_OS"),
                 (IdentityError("x"), "E_IDENTITY"), (study.InputIdentityError("x"), "E_IDENTITY"),
                 (InputIntegrityError("x"), "E_INTEGRITY"), (sealing.OverwriteError("x"), "E_OVERWRITE"),
                 (MemoryError(), "E_MEMORY"), (PermitError("x"), "E_IDENTITY"))
        for error, code in cases:
            self.assertEqual(sealing.error_code(error), code)


class Table6AssemblyRefusals(_Workspace):
    """(f) 조립은 누락·여분·중복·식별 불일치·엔진 오류·빈 자리를 실행 오류로 멈춘다. 재개도 같은 검사를 쓴다."""

    def setUp(self):
        super().setUp()
        self.ctx = world_context()
        self.run_dir = self.stage1(self.ctx, workers=4)
        self.folder = self.run_dir / "stage1"
        self.units = self.folder / "units"

    def refused(self, message):
        """조립과 재개가 같은 사유로 멈춘다."""
        with self.assertRaisesRegex(stages.AssemblyError, message):
            stages.assemble1(self.ctx, self.run_dir)
        with self.assertRaisesRegex(IdentityError, "sealed unit is not reused: .*" + message):
            self.stage1(self.ctx)

    def test_complete_run_assembles(self):
        result = stages.assemble1(self.ctx, self.run_dir)
        self.assertEqual(len(result["settings"]), 36)
        for name in ("result.json", "selection.json", "summary.md", "lock.json"):
            self.assertTrue((self.folder / name).is_file())

    def test_missing_extra_and_duplicate_units(self):
        (self.units / "setting-12.json").rename(self.units / "kept.json")
        with self.assertRaisesRegex(stages.AssemblyError, "missing setting-12.json; unexpected kept.json"):
            stages.assemble1(self.ctx, self.run_dir)
        (self.units / "kept.json").rename(self.units / "setting-12.json")
        shutil.copy(self.units / "setting-05.json", self.units / "setting-37.json")
        with self.assertRaisesRegex(stages.AssemblyError, "unexpected setting-37.json"):
            stages.assemble1(self.ctx, self.run_dir)
        (self.units / "setting-37.json").unlink()
        shutil.copy(self.units / "setting-05.json", self.units / "setting-06.json")  # 같은 설정이 두 자리에
        self.refused("setting-06: content differs from the hash sealed when it was computed")
        reseal(self.folder, "setting-06.json", lambda record: None)  # 봉인 기록까지 맞춰도 식별값이 그 자리와 다르다
        self.refused("setting-06: identity differs in .*applied_config_sha256")

    def test_any_change_to_a_sealed_unit_is_caught_by_its_hash(self):
        path = self.units / "setting-09.json"
        original = path.read_text(encoding="utf-8")
        for changed in (original.replace('"levels": 5', '"levels": 9', 1), original + "\n", original[:-2]):
            path.write_text(changed, encoding="utf-8")
            self.refused("setting-09: content differs from the hash sealed when it was computed")
        path.write_text(original, encoding="utf-8")
        (self.folder / "sealed/seals/setting-09.json").write_text('{"sha256": "0"}')
        self.refused("setting-09: seal record is missing or damaged")

    def test_tampered_identity_setting_or_result(self):
        """봉인 해시까지 맞춰 고친 기록도 머리·본문·네 자리·구간 검사에서 멈춘다."""
        def base_of(record, path=OHLC):
            return record["result"]["run_start"]["items"]["M"]["paths"][path]

        def drop_last_segment(record):
            for path in PATHS:
                for account in ("strategy", "hold"):
                    base_of(record, path)["base"][account]["segments"].pop()

        def stress_reason_for_empty_base(record):
            base_of(record).update(base=None)
            record["result"]["exploration"][OHLC]["flags"]["failures"].append("stress_start_rejected_by_cost_guard")

        zeros = "0" * 64
        cases = {
            "applied": (lambda r: r["identity"].update(applied_config_sha256=zeros),
                        r"identity differs in \['applied_config_sha256'\]"),
            "exploration_source": (lambda r: r["identity"].update(exploration_source_sha256=zeros),
                                   "exploration_source_sha256"),
            "approved_source": (lambda r: r["identity"].update(approved_source_sha256=zeros), "approved_source_sha256"),
            "inputs": (lambda r: r["identity"]["inputs"]["actual"].update(name="other"),
                       r"identity differs in \['inputs'\]"),
            "grid": (lambda r: r["identity"].update(grid_applied_sha256=zeros), "grid_applied_sha256"),
            "formal_flag": (lambda r: r.update(formal_run=True), "not an exploration unit record"),
            "setting": (lambda r: r["setting"].update(kg="4"), "setting parameters differ"),
            "other_start": (lambda r: r["result"]["run_start"].update(start="2025-01-03"), "another symbol or start"),
            "other_symbol": (lambda r: r["result"]["run_start"].update(symbol="229200"), "another symbol or start"),
            "other_end": (lambda r: r["result"]["run_start"].update(end="2025-03-31"), "neither the registered end"),
            "other_mode": (lambda r: r["result"].update(mode="D"), "not the registered reentry mode"),
            "other_input": (lambda r: r["result"]["run_start"]["input_provenance"].update(daily_sha256="f" * 64),
                            r"input provenance differs from the run identity in \['daily_sha256'\]"),
            "other_config": (lambda r: r["result"].update(applied_config_sha256=zeros), "another applied configuration"),
            "empty_slot": (lambda r: base_of(r).update(base=None), "empty slot without a reason for that cost"),
            "stress_reason_only": (stress_reason_for_empty_base, f"{OHLC}/base: empty slot without a reason"),
            "missing_path": (lambda r: r["result"]["exploration"].pop(OLHC), "both intraday paths"),
            "missing_cost": (lambda r: base_of(r).pop("stress"), "slot is missing"),
            "no_result": (lambda r: r["result"].update(run_start=None), "no result and no hold reason"),
            "hold_added": (lambda r: r["result"].update(held_reason="synthetic: no input"),
                           "hold reason and result are both present"),
            "fourth_segment_removed": (drop_last_segment, r"segments differ from the registered \['S1', 'S2', 'S3', 'S4'\]"),
            "hold_only_segment_removed": (lambda r: base_of(r)["hold_metrics"]["stress"]["segments"].pop(),
                                          f"{OHLC}/stress: segments differ from the registered"),
            "matched_segment_removed": (lambda r: base_of(r, OLHC)["matched_baseline"]["metrics"]["segments"].pop(0),
                                        f"{OLHC}/base: segments differ from the registered"),
            "held_slots_on_opened": (lambda r: r["result"].update(held_slots={OHLC: {"base": {"values": None, "reason": "x"}}}),
                                     "an opened result must not carry held slots"),
            "evaluation_morning": (lambda r: r["result"]["run_start"].update(last_evaluation="2025-04-02 09:00:00+09:00"),
                                   "last evaluation is not 18:00 KST of the last session"),
            "evaluation_utc": (lambda r: r["result"]["run_start"].update(last_evaluation="2025-04-02 09:00:00+00:00"),
                               "last evaluation is not 18:00 KST"),  # 같은 순간이어도 한국시각 18시로 적혀 있어야 한다
            "evaluation_other_day": (lambda r: r["result"]["run_start"].update(last_evaluation="2025-04-01 18:00:00+09:00"),
                                     "last evaluation is not 18:00 KST"),
            "evaluation_text": (lambda r: r["result"]["run_start"].update(last_evaluation="at the close"),
                                "last evaluation is not 18:00 KST"),
            "floor_on_kodex_row": (lambda r: r["result"]["run_start"]["capacity"].update(floor_krw="100000000"),
                                   "capacity differs from the registered nominal floor"),
            "capacity_not_ok": (lambda r: r["result"]["run_start"]["capacity"].update(ok=False),
                                "capacity differs from the registered nominal floor"),
            "capacity_hold_on_opened": (lambda r: r["result"]["exploration"][OHLC]["flags"]["holds"].append(
                "start_capacity_below_floor"), "an opened row carries the nominal capacity hold"),
            "split_capital": (lambda r: r["result"]["exploration"][OHLC]["split_share"]["base"]["split"].update(
                capital_krw="80000000"), f"{OHLC}: stated capital differs from the setting"),
            "core_capital": (lambda r: r["result"]["exploration"][OLHC]["core_share"]["stress"].update(
                capital_krw="80000000"), f"{OLHC}: stated capital differs from the setting"),
            "malformed": (lambda r: r["result"]["run_start"].pop("items"), "malformed unit record"),
        }
        record = json.loads((self.units / "setting-09.json").read_text())  # 이 단위에는 고칠 대상이 실제로 있다
        self.assertEqual(len(base_of(record)["hold_metrics"]["stress"]["segments"]), 4)
        self.assertEqual(len(base_of(record, OLHC)["matched_baseline"]["metrics"]["segments"]), 4)
        unit, seal = self.units / "setting-09.json", self.folder / "sealed/seals/setting-09.json"
        originals = unit.read_text(encoding="utf-8"), seal.read_text(encoding="utf-8")
        for name, (change, message) in cases.items():
            with self.subTest(name):
                reseal(self.folder, "setting-09.json", change)
                self.refused(message)
                unit.write_text(originals[0], encoding="utf-8")
                seal.write_text(originals[1], encoding="utf-8")
        stages.assemble1(self.ctx, self.run_dir)

    def test_line_endings_are_part_of_the_sealed_bytes(self):
        """줄끝만 LF에서 CRLF로 바꾼 파일: JSON으로는 같은 내용이지만 봉인한 바이트가 아니다. 재개·조립 모두 거부한다."""
        path = self.units / "setting-09.json"
        original = path.read_bytes()
        path.write_bytes(original.replace(b"\n", b"\r\n"))
        self.assertEqual(json.loads(path.read_bytes()), json.loads(original))
        self.assertEqual(path.read_text(), original.decode())  # 글자로 읽으면 차이가 보이지 않는다
        self.refused("setting-09: content differs from the hash sealed when it was computed")
        path.write_bytes(original)
        result = stages.assemble1(self.ctx, self.run_dir)
        lock = sealing.read_sealed(self.folder / "lock.json", stages.LOCK_SCHEMA)
        self.assertEqual(lock["units"]["setting-09"], hashlib.sha256(original).hexdigest())
        for name in ("lock.json", "summary.md", "selection.json", "result.json"):  # 최초 기록과 산출물도 바이트로 맞춘다
            with self.subTest(name):
                target = self.folder / name
                promised = target.read_bytes()
                self.assertEqual(lock["outputs"].get(name, hashlib.sha256(promised).hexdigest()),
                                 hashlib.sha256(promised).hexdigest())
                target.write_bytes(promised.replace(b"\n", b"\r\n"))
                with self.assertRaisesRegex(IdentityError, f"not overwritten: {name} already exists"):
                    stages.assemble1(self.ctx, self.run_dir)
                with self.assertRaisesRegex(IdentityError, f"stage1 differs from its first record in .'{name}'."):
                    stages.run_stage2(self.ctx, self.run_dir)
                target.write_bytes(promised)
        self.assertEqual(json.loads(sealing.dump(stages.assemble1(self.ctx, self.run_dir))), json.loads(sealing.dump(result)))

    def test_head_of_one_unit_with_the_body_of_another(self):
        """1번 파일의 머리는 그대로 두고 본문만 2번 결과로 바꾸면, 봉인 해시를 맞춰도 본문의 적용 설정이 머리와 다르다."""
        other = json.loads((self.units / "setting-02.json").read_text())["result"]
        reseal(self.folder, "setting-01.json", lambda record: record.update(result=other))
        self.refused("setting-01: body was computed with another applied configuration than its identity")

    def test_other_code_inputs_or_grid_version_is_refused_on_resume_and_assembly(self):
        other_inputs = world_context(sigma="0.02")
        other_grid = world_context(replace(WORLD_SPEC, thresholds=EVERYTHING_PASSES))
        for name, ctx in (("inputs", other_inputs), ("grid", other_grid)):
            with self.subTest(name):
                with self.assertRaises(IdentityError):
                    self.stage1(ctx)
                with self.assertRaises(stages.AssemblyError):
                    stages.assemble1(ctx, self.run_dir)
        changed = {"exploration_source_sha256": "1" * 64, "exploration_source_files": []}
        with patch.object(identity, "exploration_source", return_value=changed):  # 탐색 코드가 바뀐 판본
            other_code = world_context()
            with self.assertRaisesRegex(IdentityError, "exploration_source_sha256"):
                self.stage1(other_code)
            with self.assertRaisesRegex(stages.AssemblyError, "exploration_source_sha256"):
                stages.assemble1(other_code, self.run_dir)
        with self.assertRaisesRegex(PermitError, "does not describe the current code"):
            stages.assemble1(other_code, self.run_dir)  # 지난 판본의 식별값을 든 문맥은 진입점에서 멈춘다
        stages.assemble1(self.ctx, self.run_dir)  # 원래 판본으로는 그대로 조립된다

    def test_engine_error_blocks_assembly_until_recovered(self):
        shutil.rmtree(self.folder)
        real = stages.run_point

        def failing(inputs, config, start, **options):
            if (config.levels, config.kg, config.kp, config.capital["core"]) == (9, D(2), D(3), D(80000000)):
                raise RuntimeError("synthetic engine failure")
            return real(inputs, config, start, **options)

        self.lines.clear()
        with patch.object(stages, "run_point", side_effect=failing):
            with self.assertRaisesRegex(stages.StageError, r"1 unit\(s\) ended with a technical error: \['setting-28'\]"):
                self.stage1(self.ctx, workers=3)
        errors = [line.split() for line in self.lines if " error " in line]
        self.assertEqual([(parts[2], parts[3], parts[5]) for parts in errors], [("setting-28", "error", "E_ENGINE")])
        self.assertEqual((self.folder / "errors/setting-28.error.txt").read_text(), "setting-28 E_ENGINE\n")
        self.assertIn("synthetic engine failure", (self.folder / "sealed/logs/setting-28.log").read_text())
        with self.assertRaisesRegex(stages.AssemblyError, "missing setting-28.json; error marker setting-28.error.txt"):
            stages.assemble1(self.ctx, self.run_dir)
        self.lines.clear()
        self.stage1(self.ctx, workers=3)  # 같은 입력으로 복구: 실패한 단위만 다시 계산한다
        self.assertEqual(sorted(line.split()[3] for line in self.lines), ["done"] + ["skipped"] * 35)
        self.assertFalse((self.folder / "errors/setting-28.error.txt").exists())
        self.assertEqual(len(stages.assemble1(self.ctx, self.run_dir)["settings"]), 36)

    def test_interrupted_writes_leave_no_result_and_are_cleared_on_resume(self):
        leftover = self.units / ".setting-03.json.999.tmp"
        leftover.write_text("half written")
        with self.assertRaisesRegex(stages.AssemblyError, "unexpected .setting-03.json.999.tmp"):
            stages.assemble1(self.ctx, self.run_dir)
        stale = self.folder / "errors/setting-04.error.txt"  # 봉인까지 끝난 단위에 지난 시도의 오류 표시가 남은 경우
        stale.parent.mkdir()
        stale.write_text("setting-04 E_ENGINE\n")
        with self.assertRaisesRegex(stages.AssemblyError, "error marker setting-04.error.txt"):
            stages.assemble1(self.ctx, self.run_dir)
        self.stage1(self.ctx)  # 재개할 때(폴더 잠금을 잡은 뒤) 남은 임시 파일과 지난 오류 표시를 지운다
        self.assertEqual((list(self.folder.rglob(".*.tmp")), stale.exists()), ([], False))
        stages.assemble1(self.ctx, self.run_dir)


class WriteOnceAndSingleWriter(_Workspace):
    """기록은 한 번만 쓰고 다른 내용으로 바꾸지 않는다. 출력 폴더의 작성자는 하나다."""

    def test_write_once_never_replaces_other_content(self):
        target = self.out / "once/record.json"
        self.assertTrue(sealing.write_once(target, "first"))
        self.assertFalse(sealing.write_once(target, "first"))  # 같은 내용은 통과(다시 쓰지 않는다)
        with self.assertRaisesRegex(sealing.OverwriteError, "record.json already exists with different content"):
            sealing.write_once(target, "second")
        self.assertEqual((target.read_text(), list(target.parent.glob(".*.tmp"))), ("first", []))
        with patch.object(sealing.os, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                sealing.write_once(self.out / "once/other.json", "text")
        self.assertFalse((self.out / "once/other.json").exists())  # 끊긴 쓰기는 결과 파일을 남기지 않는다
        sealing.clear_temporaries(self.out / "once")
        self.assertEqual(sorted(path.name for path in target.parent.iterdir()), ["record.json"])
        sealing.write_once(self.out / "once/sealed.json", "x", mode=sealing.SEALED)
        self.assertEqual((self.out / "once/sealed.json").stat().st_mode & 0o777, 0o600)
        # 하드 링크를 쓰지 않는다: 네트워크 파일시스템에서 링크 뒤 임시 이름을 지우면 흔적 파일(.nfs…)이 남는다
        self.assertEqual({path.stat().st_nlink for path in target.parent.iterdir()}, {1})

    def test_network_filesystem_leftovers_are_not_results(self):
        """지운 파일 자리에 파일시스템이 남긴 흔적(.nfs…)은 여분 결과 파일로 세지 않는다."""
        ctx = world_context()
        out = self.out / "nfs"
        stages.run_stage1(ctx, out, workers=4, echo=self.lines.append)
        for part in ("units", "errors"):
            (out / "stage1" / part).mkdir(exist_ok=True)
            (out / "stage1" / part / ".nfs000000000020e1c200163cc6").write_text("leftover")
        self.assertEqual(len(stages.assemble1(ctx, out)["settings"]), 36)
        # 실제 흔적의 이름 형식(.nfs + 16진수 24자)만 뺀다. 비슷한 이름의 여분 파일은 계속 예정에 없는 파일이다.
        extras = ("other.json", ".nfsABC", ".nfs-note", ".nfs000000000020e1c200163cc", ".nfs000000000020e1c200163cc6.json",
                  ".nfs000000000020E1C200163CC6", "x.nfs000000000020e1c200163cc6")
        for name in extras:
            with self.subTest(name):
                (out / "stage1/units" / name).write_text("{}")
                with self.assertRaisesRegex(stages.AssemblyError, "^unexpected " + re.escape(name) + "$"):
                    stages.assemble1(ctx, out)
                (out / "stage1/units" / name).unlink()
        (out / "stage1/errors/.nfs-note").write_text("x")
        with self.assertRaisesRegex(stages.AssemblyError, "^error marker .nfs-note$"):
            stages.assemble1(ctx, out)

    def test_sealed_records_detect_hand_edits(self):
        record = sealing.with_seal({"schema": "s", "value": 1})
        path = self.out / "record.json"
        path.write_text(sealing.dump(record))
        self.assertEqual(sealing.read_sealed(path, "s"), record)
        for text in (sealing.dump({**record, "value": 2}), sealing.dump({"schema": "s", "value": 1}), "[]"):
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "not an intact s record"):
                sealing.read_sealed(path, "s")

    def test_one_writer_per_output_folder(self):
        ctx, out = world_context(), self.out / "busy"
        leftover = out / "stage1/units/.setting-03.json.999.tmp"
        leftover.parent.mkdir(parents=True)
        leftover.write_text("being written by the live writer")
        calls = (lambda: stages.run_stage1(ctx, out, echo=self.lines.append), lambda: stages.assemble1(ctx, out),
                 lambda: stages.run_stage2(ctx, out, echo=self.lines.append), lambda: stages.assemble2(ctx, out),
                 lambda: baseline.run_baseline(ctx, self.out / "formal.json", out), lambda: permit.run_selftest(out))
        with open(out / ".writer.lock", "a+") as other:  # 다른 실행이 이 폴더에 쓰고 있다
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for call in calls:
                with self.assertRaisesRegex(sealing.BusyError, "another run is writing"):
                    call()
            self.assertTrue(leftover.exists())  # 살아 있는 작성자의 임시 파일을 지우지 않는다
            self.assertEqual((self.lines, sorted(path.name for path in out.iterdir())), ([], [".writer.lock", "stage1"]))
        stages.run_stage1(ctx, out, workers=4, echo=self.lines.append)  # 잠금이 풀리면 중단된 쓰기를 정리하고 실행한다
        self.assertFalse(leftover.exists())
        self.assertEqual(len(self.lines), 36)

    def test_another_thread_is_an_independent_writer(self):
        """한 스레드가 잠금 구역에 있는 동안 같은 프로세스의 다른 스레드는 들어오지 못한다. 같은 호출 흐름의 중첩만 허용."""
        ctx, out, seen = world_context(), self.out / "threads", {}
        leftover = out / "stage1/units/.setting-03.json.999.tmp"
        leftover.parent.mkdir(parents=True)
        leftover.write_text("being written by the lock owner")

        def take_and_release():
            with sealing.writer_lock(out):
                pass

        def other():
            seen["token"] = sealing.writer_token(out)
            calls = {"lock": take_and_release,
                     "stage1": lambda: stages.run_stage1(ctx, out, echo=self.lines.append),
                     "assemble1": lambda: stages.assemble1(ctx, out),
                     "units": lambda: stages.run_units(stages.stage1_plan(ctx, out), 1, self.lines.append)}
            for name, call in calls.items():
                try:
                    call()
                    seen[name] = "entered"
                except (sealing.BusyError, PermitError) as error:
                    seen[name] = str(error)

        with sealing.writer_lock(out):
            token = sealing.writer_token(out)
            with sealing.writer_lock(out):  # 같은 호출 흐름의 중첩
                self.assertEqual(sealing.writer_token(out), token)
            thread = threading.Thread(target=other)
            thread.start()
            thread.join()
            self.assertTrue(leftover.exists())  # 독립 작성자는 임시 파일 정리 전에 거부된다
            self.assertEqual(sealing.writer_token(out), token)
        self.assertIsNone(seen["token"])
        for name in ("lock", "stage1", "assemble1"):
            self.assertIn("another thread is writing", seen[name])
        self.assertIn("is not held by this call flow", seen["units"])
        self.assertEqual((self.lines, sealing.writer_token(out)), ([], None))
        thread = threading.Thread(target=other)  # 잠금이 풀린 뒤에는 다른 스레드가 작성자가 될 수 있다
        thread.start()
        thread.join()
        self.assertEqual((seen["stage1"], len(self.lines)), ("entered", 36))

    def test_forked_worker_writes_under_its_parents_lock_but_cannot_take_it(self):
        """잠금을 잡은 프로세스가 포크한 작업 자식은 부모의 잠금 아래에 있다. 자식이 따로 잠금을 잡지는 못한다."""
        with sealing.writer_lock(self.out):
            token = sealing.writer_token(self.out)
            child = os.fork()
            if child == 0:
                code = 0 if sealing.writer_token(self.out) == token else 1
                try:
                    with sealing.writer_lock(self.out):
                        code += 2
                except sealing.BusyError:
                    pass
                finally:
                    os._exit(code)
            self.assertEqual(os.waitpid(child, 0)[1], 0)

    def test_lock_is_reentrant_in_one_process_and_released(self):
        with sealing.writer_lock(self.out), sealing.writer_lock(self.out):
            with open(self.out / ".writer.lock", "a+") as other:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with open(self.out / ".writer.lock", "a+") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def fail_setting(record, delta="-5"):
    """한 설정의 기본 비용 전체 차를 문턱 아래로 바꾼다(수치 한 곳)."""
    record["result"]["run_start"]["items"]["M"]["paths"][OHLC]["base"]["delta_cagr"] = delta


class FirstRecordLock(_Workspace):
    """최초 조립이 단위별 내용 해시·식별값·선택(순위와 8개 번호)을 고정한다. 이후에는 같을 때만 통과한다."""

    def setUp(self):
        super().setUp()
        self.ctx = world_context(replace(WORLD_SPEC, thresholds=EVERYTHING_PASSES))
        self.run_dir = self.stage1(self.ctx, workers=4)
        self.folder = self.run_dir / "stage1"
        self.first = stages.assemble1(self.ctx, self.run_dir)
        self.files = {name: (self.folder / name).read_text() for name in ("lock.json", "selection.json", "result.json", "summary.md")}

    def test_first_record_fixes_hashes_identity_and_selection(self):
        lock = sealing.read_sealed(self.folder / "lock.json", stages.LOCK_SCHEMA)
        self.assertEqual(lock["units"], {path.stem: sealing.file_digest(path) for path in (self.folder / "units").iterdir()})
        self.assertEqual(lock["identity"], json.loads(sealing.dump(self.ctx.identity)))
        chosen = lock["chosen"]
        self.assertEqual((chosen["state"], chosen["region"]["rank"], chosen["region"]["numbers"], chosen["candidate_if_reconfirmed"]),
                         ("selected", 1, [19, 20, 21, 22, 25, 26, 27, 28], 19))
        self.assertEqual(lock["outputs"], {name: sealing.digest(text) for name, text in self.files.items() if name != "lock.json"})
        again = stages.assemble1(self.ctx, self.run_dir)  # 같은 내용의 재조립은 통과하고 아무것도 바꾸지 않는다
        self.assertEqual(json.loads(sealing.dump(again)), json.loads(sealing.dump(self.first)))
        self.assertEqual({name: (self.folder / name).read_text() for name in self.files}, self.files)

    def test_changed_unit_cannot_move_the_selection_from_rank_one_to_two(self):
        """19번(순위 1 영역에만 속한 점)의 수치 한 곳을 바꾸면 다시 유도한 선택은 순위 2가 된다. 그 선택으로 덮어쓰지 않는다."""
        reseal(self.folder, "setting-19.json", fail_setting)  # 봉인 해시까지 맞춘 변경
        with self.assertRaisesRegex(IdentityError, "stage1 differs from its first record and is not overwritten"):
            stages.assemble1(self.ctx, self.run_dir)
        for call in (stages.run_stage2, stages.assemble2):
            with self.assertRaisesRegex(IdentityError, r"stage1 differs from its first record in \['lock.json'\]"):
                call(self.ctx, self.run_dir)
        self.assertEqual({name: (self.folder / name).read_text() for name in self.files}, self.files)
        self.assertEqual(json.loads(self.files["selection.json"])["region"]["rank"], 1)
        self.assertFalse((self.run_dir / "stage2").exists())
        (self.folder / "lock.json").unlink()  # 최초 기록이 없다고 치면 같은 변경은 순위 2를 고른다(반례가 실제로 순위를 바꾼다)
        for name in ("selection.json", "result.json", "summary.md"):
            (self.folder / name).unlink()
        self.assertEqual(stages.assemble1(self.ctx, self.run_dir)["selection"]["region"]["rank"], 2)

    def test_outputs_are_not_overwritten_by_other_content(self):
        for name in ("selection.json", "summary.md", "result.json"):
            with self.subTest(name):
                (self.folder / name).write_text(self.files[name] + " ")
                with self.assertRaisesRegex(IdentityError, f"not overwritten: {name} already exists"):
                    stages.assemble1(self.ctx, self.run_dir)
                with self.assertRaisesRegex(IdentityError, f"differs from its first record in .'{name}'."):
                    stages.run_stage2(self.ctx, self.run_dir)
                self.assertEqual((self.folder / name).read_text(), self.files[name] + " ")
                (self.folder / name).write_text(self.files[name])
        (self.folder / "lock.json").unlink()
        with self.assertRaisesRegex(IdentityError, "stage1 has no first record"):
            stages.run_stage2(self.ctx, self.run_dir)

    def test_stage_two_units_and_outputs_are_bound_to_the_first_record(self):
        first_sha = sealing.digest(self.files["lock.json"])
        stages.run_stage2(self.ctx, self.run_dir, workers=4, echo=self.lines.append)
        stage2 = self.run_dir / "stage2"
        for path in (stage2 / "units").iterdir():
            self.assertEqual(json.loads(path.read_text())["identity"]["stage1_lock_sha256"], first_sha)
        result = stages.assemble2(self.ctx, self.run_dir)
        lock = sealing.read_sealed(stage2 / "lock.json", stages.LOCK_SCHEMA)
        self.assertEqual((lock["binding"], len(lock["units"]), lock["chosen"]["candidate"]),
                         ({"stage1_lock_sha256": first_sha}, 32, 19))
        texts = {name: (stage2 / name).read_text() for name in ("result.json", "summary.md", "lock.json")}
        self.assertEqual(json.loads(sealing.dump(stages.assemble2(self.ctx, self.run_dir))), json.loads(sealing.dump(result)))
        self.assertEqual({name: (stage2 / name).read_text() for name in texts}, texts)
        reseal(stage2, f"reconfirm-{SYMBOL}-2025-01-09-setting-19.json", fail_setting)
        with self.assertRaisesRegex(IdentityError, "stage2 differs from its first record and is not overwritten"):
            stages.assemble2(self.ctx, self.run_dir)
        self.assertEqual({name: (stage2 / name).read_text() for name in texts}, texts)


class Table6HoldsAndStageTwo(_Workspace):
    """자료 보류·비용 미달은 사유가 있는 개별 행. 영역 없음·선택 미확정이면 2단계를 실행하지 않는다."""

    def test_data_hold_keeps_every_slot_with_reason_and_selection_is_undetermined(self):
        ctx = world_context(held={(SYMBOL, J(2))}, name="held")
        out = self.stage1(ctx, workers=2)
        reason = "synthetic: no input for this start"
        for unit in sealed(out / "stage1").values():  # 입력 보류도 두 경로 × 두 비용 네 자리에 빈 수치와 사유를 남긴다
            self.assertEqual(unit["result"]["held_slots"], {path: {cost: {"values": None, "reason": reason}
                                                                     for cost in ("base", "stress")} for path in PATHS})
            self.assertEqual((unit["result"]["run_start"], unit["result"]["exploration"]), (None, None))
        result = stages.assemble1(ctx, out)
        self.assertEqual({row["verdict"]["status"] for row in result["settings"]}, {OPEN})
        self.assertEqual({tuple(row["verdict"]["reasons"]) for row in result["settings"]}, {(reason,)})
        self.assertEqual({row["status"] for row in result["regions"]}, {OPEN})
        selection = json.loads((out / "stage1/selection.json").read_text())
        self.assertEqual((selection["state"], selection["blocked_by_rank"], selection["stage2"]), (OPEN, 1, "not_run"))
        self.assertEqual(selection["conclusion"]["sentence"], "자료 부족으로 미확정")
        counts = result["summary"]["positive_counts"]
        self.assertEqual(counts, {"total": 36, "both_paths_positive": 0, "not_both_positive": 0, "undetermined": 36})
        self.assertEqual(stages.run_stage2(ctx, out, echo=self.lines.append), "not_run")
        skipped = json.loads((out / "stage2/result.json").read_text())
        self.assertEqual((skipped["stage2"], skipped["reason"], skipped["conclusion"]["sentence"]),
                         ("not_run", OPEN, "자료 부족으로 미확정"))
        self.assertFalse((out / "stage2/units").exists())
        text = (out / "stage2/summary.md").read_text()
        self.assertIn("자료 부족으로 미확정", text)
        self.assertIn("선택 미확정(자료 부족으로 순위가 앞선 영역의 판정이 남아 있음)이므로 2단계를 실행하지 않았다.", text)
        self.assertNotIn("영역 없음", text)

    def test_held_input_must_keep_four_empty_slots_with_its_reason(self):
        ctx = world_context(held={(SYMBOL, J(2))}, name="held")
        out = self.stage1(ctx, workers=4)
        folder = out / "stage1"
        originals = [(folder / part / "setting-05.json").read_text() for part in ("units", "sealed/seals")]
        cases = (lambda r: r["result"]["held_slots"][OHLC].pop("stress"),
                 lambda r: r["result"]["held_slots"][OLHC]["base"].update(reason="another reason"),
                 lambda r: r["result"]["held_slots"][OLHC]["base"].update(values={"delta_cagr": "0"}),
                 lambda r: r["result"].update(held_slots=None), lambda r: r["result"].update(exploration={}))
        for change in cases:
            reseal(folder, "setting-05.json", change)
            with self.assertRaisesRegex(stages.AssemblyError, "setting-05: a held input must record four empty slots"):
                stages.assemble1(ctx, out)
            for part, text in zip(("units", "sealed/seals"), originals):
                (folder / part / "setting-05.json").write_text(text)
        stages.assemble1(ctx, out)

    def test_all_held_tables_keep_the_registered_paths_and_segments(self):
        """전부 보류여도 등록한 두 경로·네 구간으로 표의 골격을 만들고 빈값·사유로 채운다."""
        ctx = world_context(held={(SYMBOL, J(2))}, name="held")
        out = self.stage1(ctx, workers=4)
        result = stages.assemble1(ctx, out)
        self.assertEqual(result["frame"], {"paths": list(PATHS), "labels": ["S1", "S2", "S3", "S4"]})
        row = result["settings"][0]
        self.assertEqual(list(row["display"]["paths"]), list(PATHS))
        self.assertEqual(row["display"]["paths"][OHLC]["flags"]["holds"], ["synthetic: no input for this start"])
        pairs = result["summary"]["sell_pairs"]
        self.assertEqual((len(pairs), {pair["sells"] for pair in pairs}, {pair["delta"] for pair in pairs}), (72, {None}, {None}))
        self.assertEqual([item["number"] for item in result["summary"]["ranking"]], list(range(1, 37)))
        text = (out / "stage1/summary.md").read_text()
        self.assertIn(f"| 번호 | 경로 | S1 (%p) | S2 (%p) | S3 (%p) | S4 (%p) | 최악 구간 |", text)
        self.assertIn(f"| 1 | {OHLC} | — | — | — | — | — |", text)
        self.assertIn(f"| 36 | {OLHC} | — | — | — | — | — |", text)
        self.assertIn(f"| 기본 차 {OHLC} (%p) | 기본 차 {OLHC} (%p) |", text)
        self.assertIn(f"| 1 | {OHLC} | — | — |", text)  # 매도 건수·차 표에도 예정한 행이 빈값으로 남는다
        # 사유 칸: 판정표 36 + 순위표 72(경로마다) + 빈칸 사유 표 144(72행 × 두 비용)
        self.assertEqual(text.count("synthetic: no input for this start"), 36 + 72 + 144)
        self.assertIn(f"| 36 | {OLHC} | synthetic: no input for this start | synthetic: no input for this start |", text)
        blank = row["display"]["paths"][OHLC]["blank"]
        self.assertEqual(blank, dict.fromkeys(("base", "stress"), ["synthetic: no input for this start"]))

    def test_no_region_is_recorded_differently_from_undetermined(self):
        ctx = world_context(replace(WORLD_SPEC, thresholds={**SPEC.thresholds, "base_edge_min": D(5)}))
        out = self.stage1(ctx, workers=4)
        result = stages.assemble1(ctx, out)
        self.assertEqual({row["verdict"]["status"] for row in result["settings"]}, {FAIL})
        self.assertEqual((result["selection"]["state"], result["selection"]["stage2"]), ("none", "not_run"))
        self.assertEqual(stages.assemble2(ctx, out)["reason"], "none")
        closed = json.loads((out / "stage2/result.json").read_text())
        self.assertEqual((closed["conclusion"]["sentence"], closed["conclusion"]["stage"]),
                         ("이 범위에 후속 조사 근거 없음", "primary_start"))
        text = (out / "stage2/summary.md").read_text()
        self.assertIn("영역 없음(8개 영역 전부 확정 미달)이므로 2단계를 실행하지 않았다.", text)
        self.assertNotIn("선택 미확정", text)
        self.assertIn("2단계: 영역 없음(8개 영역 전부 확정 미달)이므로", (out / "stage1/summary.md").read_text())
        self.assertEqual(stages.run_stage2(ctx, out), "not_run")  # 같은 내용의 재실행은 통과하고 덮어쓰지 않는다

    def selected(self, **options):
        ctx = world_context(replace(WORLD_SPEC, thresholds=EVERYTHING_PASSES), **options)
        out = self.stage1(ctx, workers=4)
        selection = stages.assemble1(ctx, out)["selection"]
        self.assertEqual((selection["state"], selection["region"]["rank"], selection["candidate_if_reconfirmed"]),
                         ("selected", 1, 19))
        return ctx, out

    def test_stage_two_runs_the_locked_eight_points(self):
        ctx, out = self.selected()
        self.lines.clear()
        self.assertEqual(stages.run_stage2(ctx, out, workers=4, echo=self.lines.append), "run")
        names = sorted(path.name for path in (out / "stage2/units").iterdir())
        locked = SPEC.regions()[0].numbers
        expected = [f"reconfirm-{SYMBOL}-{start}-setting-{n:02d}.json" for start in ("2025-01-09", "2025-01-30") for n in locked]
        expected += [f"extension-{symbol}-2025-01-02-setting-{n:02d}.json" for symbol in ("229200", "169950") for n in locked]
        self.assertEqual(names, sorted(expected))
        self.assertTrue(all(LOG_LINE.match(line) and line.split()[3] == "done" for line in self.lines))
        result = stages.assemble2(ctx, out)
        reconfirm, extension = result["kodex_reconfirm"]["verdict"], result["extension"]["verdict"]
        self.assertEqual((reconfirm["status"], reconfirm["pairs"], reconfirm["counts"][PASS]), (PASS, 16, 16))
        self.assertEqual((result["conclusion"]["sentence"], result["conclusion"]["candidate"]), ("후속 확인용 후보 1개", 19))
        self.assertEqual(list(extension["products"]), ["229200", "169950"])
        for symbol, product in extension["products"].items():
            rows = result["extension"]["rows"][symbol]
            self.assertEqual((product["denominator"], product["points_min"], len(rows)), (8, 6, 8))
            self.assertEqual(product["positive"], sum(row["extension_point"]["status"] == POSITIVE for row in rows))
            self.assertEqual(product["positive"] + product["undetermined"] + product["failed"], 8)
            for row in rows:  # 확장 행에는 확장 점의 판정만 있다: KODEX 4절 판정도 구간·스트레스 수치도 없다
                self.assertNotIn("verdict", row)
                self.assertEqual(set(row["display"]), {"held_reason", "capacity", "base_deltas"})
                self.assertEqual((row["display"]["capacity"]["floor_krw"], list(row["display"]["base_deltas"])),
                                 ("100000000", list(PATHS)))
        self.assertEqual(result["extension"]["symbols"], ["229200", "169950"])
        kodex = result["kodex_reconfirm"]["rows"]
        self.assertTrue(all(row["display"]["capacity"]["floor_krw"] is None for row in kodex))  # 주 행에는 명목 문턱 없음
        self.assertTrue(all("verdict" in row and "extension_point" not in row for row in kodex))
        self.assertEqual((result["exploration"], result["formal_run"]), (True, False))
        text = (out / "stage2/summary.md").read_text()
        self.assertIn("후속 확인용 후보 1개", text)
        self.assertIn("- 다음 단계: 새 기간은 별도 결정 후 후보 고정 다음 거래일부터 한 계좌를 추적하고, 예컨대 24개월 만기에 "
                      "기본·스트레스 전체 기준을 한 번 평가해 낙폭과 함께 보고한다", text)
        self.assertEqual(result["conclusion"]["next_step"], SPEC.conclusions["candidate_next_step"])

    def test_stage_two_refuses_a_changed_selection(self):
        ctx, out = self.selected()
        path = out / "stage1/selection.json"
        stored = json.loads(path.read_text())
        stored["region"]["numbers"][0] = 23
        path.write_text(json.dumps(stored))
        with self.assertRaisesRegex(IdentityError, r"stage1 differs from its first record in \['selection.json'\]"):
            stages.run_stage2(ctx, out, echo=self.lines.append)
        self.assertFalse((out / "stage2").exists())

    def test_held_reconfirmation_start_leaves_kodex_undetermined_and_extension_separate(self):
        ctx, out = self.selected(held={(SYMBOL, WORLD_DAYS[20])}, name="late-held")
        stages.run_stage2(ctx, out, workers=4, echo=self.lines.append)
        result = stages.assemble2(ctx, out)
        reconfirm = result["kodex_reconfirm"]["verdict"]
        self.assertEqual((reconfirm["status"], reconfirm["counts"]), (OPEN, {PASS: 8, FAIL: 0, OPEN: 8}))
        self.assertEqual((result["conclusion"]["sentence"], result["conclusion"]["candidate"]), ("자료 부족으로 미확정", None))
        self.assertIn(result["extension"]["verdict"]["status"], (OBSERVED, FAIL, OPEN))  # 확장은 별도 칸에 그대로 남는다
        self.assertEqual(len(result["extension"]["rows"]["229200"]), 8)

    def test_reconfirmation_starts_must_be_the_registered_offsets(self):
        wrong = replace(WORLD_SPEC, thresholds=EVERYTHING_PASSES,
                        reconfirm={"offsets": (5, 20), "starts": (WORLD_DAYS[5], WORLD_DAYS[21])})
        inputs = world_inputs()
        inputs[(SYMBOL, WORLD_DAYS[21])] = world()
        ctx = open_synthetic_context(MemorySource(inputs), spec=wrong, base=WORLD_BASE)
        out = self.stage1(ctx, workers=4)
        stages.assemble1(ctx, out)
        with self.assertRaisesRegex(ValueError, "differ from session offsets"):
            stages.run_stage2(ctx, out, echo=self.lines.append)

    def test_justified_early_stop_is_accepted_and_an_unjustified_one_is_not(self):
        """개설 뒤 자료가 끊긴 입력: 요청 만기·마지막 유효일·중단 사유가 갖춰지면 정당한 조기 종료다."""
        gap = {"last_valid_session": WORLD_DAYS[14], "first_bad_session": WORLD_DAYS[15],
               "reason": "synthetic: price row missing", "requested_end": WORLD_DAYS[-1]}
        cut = replace(world(), bars=world().bars[:15], data_end=gap)
        ctx = open_synthetic_context(MemorySource({**world_inputs(), (SYMBOL, J(2)): cut}, "gap"),
                                     spec=WORLD_SPEC, base=WORLD_BASE)
        out = self.stage1(ctx, workers=4)
        result = stages.assemble1(ctx, out)
        self.assertEqual({row["verdict"]["status"] for row in result["settings"]}, {OPEN})
        self.assertEqual({tuple(row["verdict"]["reasons"]) for row in result["settings"]},
                         {(f"{OHLC}:input_data_gap_after_start", f"{OLHC}:input_data_gap_after_start")})
        unit = sealed(out / "stage1")["setting-01.json"]["result"]["run_start"]
        self.assertEqual((unit["end"], unit["data_end"]["requested_end"]), ("2025-01-22", "2025-04-02"))
        folder = out / "stage1"
        for change, name in ((lambda r: r["result"]["run_start"]["data_end"].update(requested_end="2025-03-31"), "other end"),
                             (lambda r: r["result"]["run_start"]["data_end"].update(reason=""), "no reason"),
                             (lambda r: r["result"]["run_start"]["data_end"].update(last_valid_session="2025-01-21"), "day"),
                             (lambda r: r["result"]["exploration"][OLHC]["flags"].update(later=[]), "no flag")):
            with self.subTest(name):
                original = [(folder / part / "setting-01.json").read_text() for part in ("units", "sealed/seals")]
                reseal(folder, "setting-01.json", change)
                with self.assertRaisesRegex(IdentityError, "neither the registered end nor a justified early stop"):
                    self.stage1(ctx)
                for part, text in zip(("units", "sealed/seals"), original):
                    (folder / part / "setting-01.json").write_text(text)


class Table6StageTwoEdges(_Workspace):
    """2단계의 가장자리: 확장 명목 한도 보류와 이후 자료 결측이 겹친 상품, 같은 설정·종목·시작일의 중복."""

    GAP = {"last_valid_session": WORLD_DAYS[14], "first_bad_session": WORLD_DAYS[15],
           "reason": "synthetic: price row missing", "requested_end": WORLD_DAYS[-1]}

    def selected(self, inputs, name, spec=None):
        spec = replace(spec or WORLD_SPEC, thresholds=EVERYTHING_PASSES)
        ctx = open_synthetic_context(MemorySource(inputs, name), spec=spec, base=WORLD_BASE)
        out = self.stage1(ctx, workers=4)
        self.assertEqual(stages.assemble1(ctx, out)["selection"]["state"], "selected")
        return ctx, out

    def floor_and_gap(self, symbol="169950"):
        """시작 한도 99,999,999원(20일 평균 거래대금 1,999,999,980원 × 5%)이고 15번째 거래일 뒤 일봉이 끊긴 확장 상품."""
        data = world(symbol=symbol)
        bars = (replace(data.bars[0], adv20=D(1999999980)),) + data.bars[1:15]
        return replace(data, bars=bars, data_end=self.GAP)

    def test_capacity_hold_with_a_later_data_gap_still_assembles(self):
        symbol = "169950"
        ctx, out = self.selected({**world_inputs(), (symbol, J(2)): self.floor_and_gap(symbol)}, "floor-gap")
        self.assertEqual(stages.run_stage2(ctx, out, workers=4, echo=self.lines.append), "run")
        result = stages.assemble2(ctx, out)  # 2단계가 끝까지 조립된다
        rows = result["extension"]["rows"][symbol]
        self.assertEqual(len(rows), 8)
        for row in rows:  # 그 점은 미확정이고, 장부를 열지 않았으므로 수치가 없다
            self.assertEqual(row["extension_point"], {"status": OPEN, "reasons": ["start_capacity_below_floor"], "paths": None})
            self.assertEqual(row["display"]["capacity"], {"limit_krw": "99999999.00", "floor_krw": "100000000", "ok": False})
            self.assertEqual(row["display"]["base_deltas"], dict.fromkeys(PATHS))
        products = result["extension"]["verdict"]["products"]
        self.assertEqual(products[symbol], {"status": OPEN, "positive": 0, "undetermined": 8, "failed": 0,
                                            "denominator": 8, "points_min": 6})  # 상품 분모 8 유지
        self.assertEqual((products["229200"]["denominator"], len(result["extension"]["rows"]["229200"])), (8, 8))
        reconfirm = result["kodex_reconfirm"]["verdict"]  # KODEX 재확인도 공개된다
        self.assertEqual((reconfirm["status"], reconfirm["pairs"], result["conclusion"]["candidate"]), (PASS, 16, 19))
        unit = json.loads((out / f"stage2/units/extension-{symbol}-2025-01-02-setting-19.json").read_text())["result"]
        self.assertEqual(unit["run_start"]["data_end"]["last_valid_session"], "2025-01-22")
        self.assertEqual((unit["run_start"]["end"], unit["run_start"]["last_evaluation"]),
                         ("2025-01-22", "2025-01-22 18:00:00+09:00"))
        for path in PATHS:  # 시작 시 비교 보류를 보존한다(개설 뒤 결측 표시로 바꾸지 않는다)
            self.assertEqual(unit["exploration"][path]["flags"],
                             {"failures": [], "holds": ["start_capacity_below_floor"], "later": []})
        text = (out / "stage2/summary.md").read_text()
        self.assertIn(f"- {symbol}: **미확정** (확정 양수 P=0, 미확정 U=8, 확정 미달 0, 분모 8, 기준 6)", text)
        self.assertIn("## KODEX 재확인 (16쌍)", text)

    def test_unopened_row_checks_its_early_stop_without_the_ledger_marker(self):
        symbol = "169950"
        ctx, out = self.selected({**world_inputs(), (symbol, J(2)): self.floor_and_gap(symbol)}, "floor-gap")
        stages.run_stage2(ctx, out, workers=4, echo=self.lines.append)
        folder, name = out / "stage2", f"extension-{symbol}-2025-01-02-setting-19.json"
        start = lambda r: r["result"]["run_start"]  # noqa: E731
        cases = {"no_reason": (lambda r: start(r)["data_end"].update(reason=""), "justified early stop"),
                 "other_end": (lambda r: start(r)["data_end"].update(requested_end="2025-03-31"), "justified early stop"),
                 "other_day": (lambda r: start(r)["data_end"].update(last_valid_session="2025-01-21"), "justified early stop"),
                 "bad_before_valid": (lambda r: start(r)["data_end"].update(first_bad_session="2025-01-22"), "justified early stop"),
                 "evaluation": (lambda r: start(r).update(last_evaluation="2025-04-02 18:00:00+09:00"), "last evaluation is not"),
                 "gap_marker": (lambda r: r["result"]["exploration"][OHLC]["flags"]["later"].append("input_data_gap_after_start"),
                                "a row held by the nominal capacity must keep only that start hold"),
                 "values": (lambda r: start(r)["items"]["M"]["paths"][OLHC].update(base={"delta_cagr": "0.02"}),
                            "a row held by the nominal capacity must keep only that start hold"),
                 "floor_met": (lambda r: start(r)["capacity"].update(ok=True), "capacity differs from the registered nominal floor"),
                 "no_floor": (lambda r: start(r)["capacity"].update(floor_krw=None), "capacity differs from the registered nominal floor")}
        originals = [(folder / part / name).read_bytes() for part in ("units", "sealed/seals")]
        for case, (change, message) in cases.items():
            with self.subTest(case):
                reseal(folder, name, change)
                with self.assertRaisesRegex(stages.AssemblyError, message):
                    stages.assemble2(ctx, out)
                for part, raw in zip(("units", "sealed/seals"), originals):
                    (folder / part / name).write_bytes(raw)
        self.assertEqual(stages.assemble2(ctx, out)["stage2"], "run")

    def test_same_setting_symbol_and_start_twice_is_refused(self):
        """잘못 선언한 격자(확장 명단에 주 종목, 확장 시작일에 재확인 시작일)에서만 생긴다: 같은 설정·종목·시작일이
        재확인 단위와 확장 단위로 두 번 들어오면 한 결과를 두 번 세게 되므로 조립하지 않는다."""
        start = WORLD_DAYS[5]
        spec = replace(WORLD_SPEC, extension={**WORLD_SPEC.extension, "symbols": (SYMBOL, "169950"), "start": start})
        ctx, out = self.selected({**world_inputs(), ("169950", start): world(symbol="169950")}, "duplicate", spec)
        self.assertEqual(stages.run_stage2(ctx, out, workers=4, echo=self.lines.append), "run")
        names = sorted(path.name for path in (out / "stage2/units").iterdir())
        self.assertIn(f"reconfirm-{SYMBOL}-2025-01-09-setting-19.json", names)
        self.assertIn(f"extension-{SYMBOL}-2025-01-09-setting-19.json", names)
        with self.assertRaisesRegex(stages.AssemblyError, "^duplicate setting/symbol/start among the sealed units$"):
            stages.assemble2(ctx, out)
        self.assertFalse((out / "stage2/result.json").exists())


class ValidNumbersInTablesAndSummary(_Workspace):
    """표와 요약에는 판정과 같은 비용별 유효성 검사를 거친 수치만 싣는다. 봉인한 단위 원본은 그대로 둔다."""

    def test_cost_with_a_hold_reason_is_blank_with_its_reason(self):
        ctx = world_context(sigma="0.005")  # kg 1 설정은 스트레스만 개설 거부, 나머지는 두 비용 모두 유효
        out = self.stage1(ctx, workers=4)

        def hold_with_values(record):  # 기본 비용 두 경로에 비교 보류 사유와 +2%p 수치를 함께 둔다
            for path in PATHS:
                record["result"]["exploration"][path]["flags"]["holds"].append("base_combined_start_liquidity")
                record["result"]["run_start"]["items"]["M"]["paths"][path]["base"]["delta_cagr"] = "0.02"

        reseal(out / "stage1", "setting-09.json", hold_with_values)
        result = stages.assemble1(ctx, out)  # 기록 검사는 통과한다(보류 사유와 수치가 함께 있을 수 있다)
        rows = {row["number"]: row for row in result["settings"]}
        held, stress_only = rows[9], rows[1]
        for path in PATHS:
            view = held["display"]["paths"][path]
            self.assertEqual(view["blank"], {"base": ["base_combined_start_liquidity"], "stress": None})
            for name in ("base", "total_delta_positive", "worst_segments", "legacy_limits", "matched"):
                self.assertIsNone(view[name], name)
            for name in ("split_trades", "reentries", "buys_closed_years", "split_share"):
                self.assertIsNone(view[name]["base"], name)
                self.assertIsNotNone(view[name]["stress"], name)  # 다른 비용의 유효한 값은 보존한다
            self.assertIsNotNone(view["stress"])
            self.assertEqual(view["start_orders"]["base"]["orders"]["core"]["status"], "filled")  # 개설 사실은 그대로
            self.assertEqual(held["verdict"]["paths"][path]["checks"]["base_edge"], None)
            kept = stress_only["display"]["paths"][path]  # 스트레스만 거부된 행의 기본 수치는 그대로 유효하다
            self.assertEqual(kept["blank"], {"base": None, "stress": ["stress_start_rejected_by_cost_guard"]})
            self.assertIsNotNone(kept["base"]["total_delta"])
            self.assertEqual((kept["stress"], kept["split_trades"]["stress"], kept["split_share"]["stress"]), (None, None, None))
            self.assertIsNotNone(kept["split_trades"]["base"]["sells"])
        described = result["summary"]
        ranked = {item["number"]: item for item in described["ranking"]}
        self.assertEqual((ranked[9]["min_delta"], ranked[9]["deltas"]), (None, dict.fromkeys(PATHS)))
        self.assertEqual(ranked[9]["reasons"], [f"{path}:base_combined_start_liquidity" for path in PATHS])
        self.assertEqual(described["ranking"][-1]["number"], 9)  # 값이 갖춰지지 않은 설정은 끝에
        self.assertIsNotNone(ranked[1]["min_delta"])
        self.assertEqual(summary.positivity(held), "undetermined")  # '두 경로 모두 양수'에 들어가지 않는다
        counts = described["positive_counts"]
        self.assertEqual((counts["undetermined"], sum(counts[name] for name in ("both_paths_positive", "not_both_positive"))),
                         (1, 35))
        for axis, value in (("kg", "2"), ("kp", "3"), ("levels", "5"), ("core", "50")):
            cells = described["axis_means"][axis][value]
            self.assertEqual({(cell["valid"], cell["planned"]) for cell in cells.values()},
                             {(len([r for r in rows.values() if str(r[axis]) == value]) - 1,
                               len([r for r in rows.values() if str(r[axis]) == value]))})
        pairs = {(item["number"], item["path"]): item for item in described["sell_pairs"]}
        self.assertEqual((pairs[(9, OHLC)]["sells"], pairs[(9, OHLC)]["delta"], pairs[(9, OHLC)]["zero_sells"]), (None, None, False))
        text = (out / "stage1/summary.md").read_text()
        self.assertIn(f"| 9 | {OHLC} | base_combined_start_liquidity | 유효 |", text)
        self.assertIn(f"| 1 | {OLHC} | 유효 | stress_start_rejected_by_cost_guard |", text)
        self.assertNotIn("+2.000", text)  # 보류 사유가 걸린 +2%p는 어느 표에도 실리지 않는다
        unit = json.loads((out / "stage1/units/setting-09.json").read_text())  # 원본은 그대로다
        self.assertEqual(unit["result"]["run_start"]["items"]["M"]["paths"][OHLC]["base"]["delta_cagr"], "0.02")
        self.assertEqual(result["units"]["setting-09"], unit)

    def test_extension_delta_is_shown_only_when_valid(self):
        raw = run_point(case2(), cfg("80", "2", "3", 9), CASE2_DAYS[0], nominal_floor=D(100000000))
        frame, result = frame_of(raw), plain(raw)
        shown = reading.extension_display(result, frame)["base_deltas"]
        self.assertEqual(shown, {path: D(paths_of(result)[path]["base"]["delta_cagr"]) for path in PATHS})
        result["exploration"][OHLC]["flags"]["holds"].append("base_combined_start_liquidity")
        result["exploration"][OLHC]["flags"]["failures"].append("stress_start_rejected_by_cost_guard")
        shown = reading.extension_display(result, frame)["base_deltas"]  # 기본 비용 사유가 걸린 경로만 빈칸
        self.assertEqual(shown, {OHLC: None, OLHC: D(paths_of(result)[OLHC]["base"]["delta_cagr"])})
        result["run_start"]["capacity"]["ok"] = False  # 명목 한도 미달이면 두 경로 모두 빈칸
        self.assertEqual(reading.extension_display(result, frame)["base_deltas"], dict.fromkeys(PATHS))


class DescriptiveSummary(unittest.TestCase):
    """설명용 요약(판정 아님): 정렬·집계 방법은 설계 v2.1 4절 문장대로."""

    def row(self, number, first, second, sells=(3, 0), blank=("base_volatility_unavailable",)):
        """표기 행의 꼴: 유효한 기본 비용은 값과 건수를, 유효하지 않은 기본 비용은 빈칸과 사유를 갖는다."""
        def view(delta, count):
            if delta is None:
                return {"base": None, "blank": {"base": list(blank)}, "split_trades": {"base": None}}
            return {"base": {"total_delta": D(delta)}, "blank": {"base": None}, "split_trades": {"base": {"sells": count}}}
        setting = SPEC.setting(number).describe()
        return {**setting, "display": {"paths": {OHLC: view(first, sells[0]), OLHC: view(second, sells[1])}}}

    def held_row(self, number):
        """입력 보류 행: 등록한 두 경로의 빈 칸."""
        return self.row(number, None, None, blank=("no input",))

    def test_ranking_counts_axis_means_and_sell_pairs(self):
        rows = [self.row(1, "0.010", "0.030"), self.row(2, "0.020", "0.020"), self.row(3, "0.060", "-0.010"),
                self.row(19, "0.020", "0.040"), self.row(20, None, "0.090"),
                self.held_row(21), self.row(22, "0", None)]
        result = summary.describe(rows, PATHS)
        # 둘 중 작은 차: 2번 0.020 = 19번 0.020(번호순) > 1번 0.010 > 3번 -0.010. 값이 갖춰지지 않은 20·21·22번은 끝에.
        self.assertEqual([item["number"] for item in result["ranking"]], [2, 19, 1, 3, 20, 21, 22])
        self.assertEqual(result["ranking"][4]["reasons"], ["O-H-L-C:base_volatility_unavailable"])  # 빈칸인 경로와 그 사유
        self.assertEqual(result["ranking"][5]["reasons"], ["O-H-L-C:no input", "O-L-H-C:no input"])
        # 22번은 한 경로가 이미 0이라 두 경로 모두 양수일 수 없다 → '양수 아님'(미확정이 아니다)
        self.assertEqual(result["positive_counts"], {"total": 7, "both_paths_positive": 3, "not_both_positive": 2,
                                                     "undetermined": 2})
        core = result["axis_means"]["core"]  # 경로별로 유효한 차만 평균, 유효 수/예정 수
        self.assertEqual(core["50"][OHLC], {"mean_delta": D("0.03"), "valid": 3, "planned": 3})  # (0.010 + 0.020 + 0.060) ÷ 3
        self.assertEqual(core["80"][OHLC], {"mean_delta": D("0.010"), "valid": 2, "planned": 4})  # (0.020 + 0) ÷ 2
        self.assertEqual(core["80"][OLHC], {"mean_delta": D("0.065"), "valid": 2, "planned": 4})
        self.assertEqual(list(result["axis_means"]), ["core", "kg", "kp", "levels"])
        pairs = {(item["number"], item["path"]): item for item in result["sell_pairs"]}
        self.assertEqual((pairs[(1, OHLC)]["sells"], pairs[(1, OLHC)]["sells"], pairs[(1, OLHC)]["zero_sells"]), (3, 0, True))
        self.assertEqual((pairs[(3, OLHC)]["delta"], pairs[(20, OHLC)]["delta"]), (D("-0.010"), None))
        self.assertEqual((pairs[(21, OHLC)]["sells"], pairs[(21, OHLC)]["zero_sells"], len(pairs)), (None, False, 14))
        self.assertFalse(result["judgment"])

    def test_ranking_compares_the_original_values_beyond_28_digits(self):
        """28자리 뒤에서만 다른 두 값: 더 큰 2번이 먼저다(값을 반올림한 뒤 비교하지 않는다). 동률은 번호순."""
        small, large = "0.0100000000000000000000000000000000001", "0.0100000000000000000000000000000000002"
        rows = [self.row(1, small, "0.5"), self.row(2, large, "0.5"), self.row(4, "0.3", small), self.row(3, small, "0.4")]
        self.assertEqual([item["number"] for item in summary.ranking(rows)], [2, 1, 3, 4])
        self.assertEqual(summary.ranking(rows)[0]["min_delta"], D(large))

    def test_positive_means_strictly_above_zero(self):
        cases = ((("0", "0.01"), "not_both_positive"), (("1E-40", "0.01"), "both_paths_positive"),
                 (("-1E-40", None), "not_both_positive"), ((None, "0"), "not_both_positive"),
                 ((None, "0.01"), "undetermined"), ((None, None), "undetermined"))
        for (first, second), expected in cases:
            self.assertEqual(summary.positivity(self.row(1, first, second)), expected, (first, second))
        self.assertEqual(summary.positivity(self.held_row(1)), "undetermined")

    def test_displayed_positive_flag_is_strict_at_zero(self):
        """표기용 '전체 차 양수'도 0은 양수가 아니다."""
        raw = run_point(case2(), cfg("80", "2", "3", 9), CASE2_DAYS[0])
        frame, result = frame_of(raw), plain(raw)
        for delta, expected in (("0", False), ("1E-40", True), ("-1E-40", False)):
            paths_of(result)[OHLC]["base"]["delta_cagr"] = delta
            self.assertIs(reading.point_display(result, frame)["paths"][OHLC]["total_delta_positive"], expected)


class SettingFailsWhenPassIsImpossible(unittest.TestCase):
    """결과를 바꿀 여지가 남을 때만 미확정이다. 경로 상태의 우선순위(보류 → 확정 미달 → 이후 결측)는 그대로다."""

    HELD = {"base": None, "stress": None}

    def held(self, **flags):
        return {**path_row(**flags), **self.HELD}

    def test_stress_only_rejection_still_shows_common_segments(self):
        row = {**path_row(wins=(1, 2), start_failures=["stress_start_rejected_by_cost_guard"]), "stress": None}
        verdict = judge.setting_verdict(point(row, {**row}), T)
        self.assertEqual((verdict["status"], verdict["common_segments"]), (FAIL, ["s1", "s2"]))
        self.assertEqual(verdict["paths"][OHLC]["checks"], {"base_edge": True, "stress_edge": None})
        self.assertEqual(verdict["reasons"], [f"{OHLC}:stress_start_rejected_by_cost_guard",
                                              f"{OLHC}:stress_start_rejected_by_cost_guard", "common_segments"])

    def test_one_held_path_and_two_wins_on_the_other_is_a_failure(self):
        hold = self.held(start_holds=["base_combined_start_liquidity"])
        two = judge.setting_verdict(point(path_row(wins=(1, 2)), hold), T)  # 다른 경로가 네 구간을 다 이겨도 공통은 둘
        self.assertEqual((two["status"], two["reasons"], two["common_segments"], two["reachable_segments"]),
                         (FAIL, ["common_segments"], None, ["s1", "s2"]))
        three = judge.setting_verdict(point(path_row(wins=(1, 2, 3)), hold), T)  # 셋이면 아직 통과할 수 있다
        self.assertEqual((three["status"], three["reasons"], three["reachable_segments"]),
                         (OPEN, [f"{OLHC}:base_combined_start_liquidity"], ["s1", "s2", "s3"]))
        for flags in ({"later_holds": ["input_data_gap_after_start"]}, {"later_holds": ["base_annual_volatility_unavailable"]}):
            self.assertEqual(judge.setting_verdict(point(path_row(wins=(3, 4)), self.held(**flags)), T)["status"], FAIL)
        nothing = judge.setting_verdict(point(hold, hold), T)  # 유효한 경로가 없으면 판단할 것이 없다
        self.assertEqual((nothing["status"], nothing["reachable_segments"]), (OPEN, None))

    def test_valid_values_that_already_fail_decide_the_setting(self):
        """경로 상태는 우선순위대로 미확정이어도, 유효한 수치가 조건을 못 채웠으면 통과할 수 없다."""
        stress_waiting = {**path_row(base="0.005", start_holds=["stress_hold_first_buy_failed"]), "stress": None}
        verdict = judge.setting_verdict(point(stress_waiting, path_row()), T)
        self.assertEqual((verdict["paths"][OHLC]["status"], verdict["status"], verdict["reasons"]),
                         (OPEN, FAIL, [f"{OHLC}:base_edge"]))
        base_waiting = {**path_row(stress="0", later_holds=["base_annual_volatility_unavailable"]), "base": None}
        verdict = judge.setting_verdict(point(base_waiting, path_row()), T)
        self.assertEqual((verdict["paths"][OHLC]["status"], verdict["status"], verdict["reasons"]),
                         (OPEN, FAIL, [f"{OHLC}:stress_edge"]))
        fine = {**path_row(start_holds=["stress_hold_first_buy_failed"]), "stress": None}  # 유효한 수치는 조건을 채웠다
        self.assertEqual(judge.setting_verdict(point(fine, path_row()), T)["status"], OPEN)

    def test_values_under_a_hold_or_common_reason_are_not_used(self):
        """그 비용의 보류·거부·결측 사유나 공통 사유가 있는 수치는 유효하지 않다(우선순위를 뒤집지 않는다)."""
        cases = (path_row(base="-1", wins=(), start_holds=["base_combined_start_liquidity"]),
                 path_row(base="-1", wins=(), later_holds=["input_data_gap_after_start"]),
                 path_row(base="-1", wins=(), start_holds=["start_capacity_below_floor"]),
                 {**path_row(start_holds=["base_hold_first_buy_failed"],
                             start_failures=["stress_start_rejected_by_cost_guard"]), **self.HELD})
        for row in cases:
            verdict = judge.setting_verdict(point(row, path_row()), T)
            self.assertEqual((verdict["status"], verdict["paths"][OHLC]["checks"]["base_edge"]), (OPEN, None))
        self.assertTrue(judge.cost_valid(path_row(start_failures=["stress_start_rejected_by_cost_guard"]), "base"))
        self.assertFalse(judge.cost_valid(path_row(start_failures=["stress_start_rejected_by_cost_guard"]), "stress"))
        self.assertEqual(judge.reasons_for(["base_x", "stress_y", "input_data_gap_after_start"], "stress"),
                         ["stress_y", "input_data_gap_after_start"])

    def test_real_ledgers_stress_only_rejection_shows_segments(self):
        """σ 0.59999%·kg 1의 실제 합성 장부: 스트레스만 개설 거부. 기본 비용 두 경로의 공통 구간이 실린다."""
        result = run_point(flat("0.0059999", prices={J(2): "10000", J(6): "9000"}), cfg("50", "1", "3", 5), J(2))
        verdict = verdict_of(result)
        self.assertEqual((verdict["status"], verdict["common_segments"]), (FAIL, ["2024-2026-11-13"]))
        self.assertIn("common_segments", verdict["reasons"])  # 구간이 하나뿐인 짧은 자료라 셋을 채울 수 없다


class ExtensionInputIsNarrow(unittest.TestCase):
    """확장 전용 판정 입력: 명목 한도, 기본 비용의 보류·개설 거부 사유, 두 기본 경로의 전체 연복리 차뿐이다."""

    def test_builds_without_stress_or_segment_data(self):
        raw = run_point(case2(), cfg("80", "2", "3", 9), CASE2_DAYS[0], nominal_floor=D(100000000))
        frame, result = frame_of(raw), plain(raw)
        whole = reading.extension_input(result, frame)
        self.assertEqual((set(whole), whole["capacity_ok"]), ({"held", "capacity_ok", "paths"}, True))
        self.assertEqual(set(whole["paths"][OHLC]), {"start_holds", "start_failures", "later_holds", "base_delta"})
        self.assertEqual(whole["paths"][OHLC]["base_delta"], D(paths_of(result)[OHLC]["base"]["delta_cagr"]))
        for path in PATHS:  # 스트레스의 전략 구간 자료와 기본 비용의 구간 자료를 없애도 확장 입력은 같다
            del paths_of(result)[path]["stress"]["strategy"]["segments"]
            paths_of(result)[path]["base"]["strategy"]["segments"] = []
            paths_of(result)[path]["stress"]["delta_cagr"] = "not a number"
        self.assertEqual(reading.extension_input(result, frame), whole)
        self.assertIn(judge.extension_point(whole, D(0))["status"], (POSITIVE, FAIL))
        self.assertEqual(reading.extension_display(result, frame)["base_deltas"],
                         {path: whole["paths"][path]["base_delta"] for path in PATHS})
        with self.assertRaises((KeyError, ValueError)):
            reading.point_input(result, frame)  # KODEX 4절 입력은 구간 자료가 있어야 만든다

    def test_stress_reasons_are_left_out_and_capacity_is_explicit(self):
        raw = run_point(flat("0.0059999", prices={J(2): "10000", J(6): "9000"}), cfg("50", "1", "3", 5), J(2))
        self.assertEqual(raw["exploration"][OHLC]["flags"]["failures"], ["stress_start_rejected_by_cost_guard"])
        narrow = reading.extension_input(plain(raw), frame_of(raw))
        self.assertEqual([narrow["paths"][path]["start_failures"] for path in PATHS], [[], []])
        self.assertEqual(judge.extension_point(narrow, D(0))["status"], POSITIVE)
        low = build(weekdays(J(2), J(6)), {J(2): "10000"}, basis=BASIS, changes={J(2): {"adv20": D(1999999980)}})
        held = run_point(low, cfg("50", "2", "3", 5), J(2), nominal_floor=D(100000000))
        narrow = reading.extension_input(plain(held), frame_of(held))
        self.assertEqual((narrow["capacity_ok"], judge.extension_point(narrow, D(0))["reasons"]),
                         (False, ["start_capacity_below_floor"]))
        self.assertEqual(reading.extension_input({"held_reason": "no input"}, frame_of(held)),
                         {"held": "no input", "capacity_ok": None, "paths": None})


class SplitShareAtEightyPercent(unittest.TestCase):
    """코어 80%에서 분할 몫 누적수익의 분모는 분할 원금 2천만 원이다(코어 원금 8천만 원이 아니다)."""

    def test_hand_computed_returns(self):
        # 시가 10,000 → 1/6부터 10,500(+5%). kg 4·kp 6이라 추가 매수도 익절도 없다. 수수료 0.015%.
        data = build(weekdays(J(2), J(8)), {J(2): "10000", J(6): "10500"}, basis=BASIS)
        result = run_point(data, cfg("80", "4", "6", 5), J(2))
        for path in PATHS:
            extra = result["exploration"][path]
            self.assertEqual(extra["split_ledger"]["base"], {"status": "ok", "buys": 1, "sells": 0})
            share = extra["split_share"]["base"]
            # 분할: 399주 × 10,500 + (20,000,000 − 3,990,000 × 1.00015) = 4,189,500 + 16,009,401.5
            self.assertEqual(share["split"], {"status": "ok", "capital_krw": D(20000000), "ending_equity_krw": D("20198901.5"),
                                              "cumulative_return": D("0.009945075")})
            # 같은 원금 보유: floor(20,000,000 ÷ 10,001.5) = 1,999주 × 10,500 + (20,000,000 − 19,992,998.5)
            self.assertEqual(share["hold_same_capital"]["ending_equity_krw"], D("20996501.5"))
            self.assertEqual(share["hold_same_capital"]["cumulative_return"], D("0.049825075"))
            # 코어: 7,998주 × 10,500 + (80,000,000 − 79,980,000 × 1.00015) = 83,979,000 + 8,003
            self.assertEqual(extra["core_share"]["base"], {"status": "ok", "capital_krw": D(80000000),
                                                           "ending_equity_krw": D(83987003), "cumulative_return": D("0.0498375375")})
            self.assertEqual(paths_of(result)[path]["base"]["strategy"]["ending_equity_krw"], D("104185904.5"))
        view = reading.point_display(plain(result), frame_of(result))["paths"][OHLC]["split_share"]["base"]
        self.assertEqual((view["capital_krw"], view["split"], view["hold_same_capital"], view["delta"]),
                         (D(20000000), D("0.009945075"), D("0.049825075"), D("-0.03988")))


# ---------------------------------------------------------------------------
# 공개 기준점 대조와 독립 검산
# ---------------------------------------------------------------------------
class ComparisonRules(unittest.TestCase):
    """지정한 수치 칸만 Decimal 값으로 허용오차 0(1.0 = 1.00), 그 밖은 원문과 자료형으로 대조한다."""

    def walk(self, formal_value, mine):
        comparison = baseline.Comparison()
        comparison.walk(formal_value, mine, "root")
        return comparison

    def test_equal_values_and_notation(self):
        formal_value = {"cagr": "1.0", "notional": "1E+2", "status": "filled", "day": "2011-01-03", "count": 3,
                        "none": None, "flag": True, "rows": [{"outcome": "filled"}, {"outcome": "no_order"}],
                        "empty": [], "nothing": {}}
        mine = {**formal_value, "cagr": "1.00", "notional": "100"}
        result = self.walk(formal_value, mine)
        self.assertEqual((result.matched, result.mismatches, result.notation_only), (11, [], ["root/cagr", "root/notional"]))
        self.assertEqual(result.string_identical, 9)  # 표기만 다른 두 잎을 뺀 나머지(빈 목록·빈 사전 포함)
        self.assertEqual((baseline.count_leaves(formal_value), baseline.count_leaves(mine)), (11, 11))

    def test_identifiers_states_and_dates_compare_as_text_and_type(self):
        """종목코드 069500과 69500은 다르다. 수치처럼 보이는 식별 문자열을 수로 바꾸지 않는다."""
        cases = (("symbol", "069500", "69500"), ("status", "1.0", "1.00"), ("sha256", "0e10", "0"),
                 ("price_bundle", "20261115", "20261115.0"), ("start", "2011-01-03", "2011-1-3"),
                 ("quantity", 222, "222"), ("quantity", 222, 222.0), ("evaluation_days", 1, True),
                 ("cagr", "0.1", 0.1), ("cagr", "abc", "ABC"), ("cagr", "NaN", "nan"), ("hold", None, "0"))
        for key, formal_value, mine in cases:
            with self.subTest(key=key, formal=formal_value):
                self.assertFalse(baseline.same_leaf(key, formal_value, mine))
                self.assertEqual(self.walk({key: formal_value}, {key: mine}).mismatches,
                                 [{"where": f"root/{key}", "kind": "value"}])
        for key, formal_value, mine in (("cagr", "0.10", "0.1"), ("hold", "99980000", "9.998E+7"), ("symbol", "069500", "069500"),
                                        ("floor_krw", None, None), ("cagr_edge", True, True), ("quantity", 222, 222)):
            self.assertTrue(baseline.same_leaf(key, formal_value, mine), key)
        listed = self.walk({"reentries": [{"date": "2025-02-03"}]}, {"reentries": [{"date": "2025-2-3"}]})
        self.assertEqual(listed.mismatches, [{"where": "root/reentries[0]/date", "kind": "value"}])

    def test_differences_are_listed_without_values(self):
        cases = (({"a": "1.0"}, {"a": "1.0000000000000000000000000000000000000001"}, "value"),
                 ({"a": "filled"}, {"a": "rejected"}, "value"), ({"a": 1}, {"a": True}, "value"),
                 ({"a": None}, {"a": "0"}, "value"), ({"a": "2011-01-03"}, {"a": "2011-01-04"}, "value"),
                 ({"a": ["x", "y"]}, {"a": ["y", "x"]}, "value"), ({"a": ["x"]}, {"a": ["x", "y"]}, "length"),
                 ({"a": 1, "b": 2}, {"a": 1}, "missing_in_exploration"), ({"a": 1}, {"a": 1, "b": 2}, "missing_in_formal"),
                 ({"a": {"b": 1}}, {"a": [1]}, "value"))
        for formal_value, mine, kind in cases:
            with self.subTest(formal=formal_value, mine=mine):
                result = self.walk(formal_value, mine)
                self.assertEqual(result.mismatches[0]["kind"], kind)
                self.assertEqual({key for row in result.mismatches for key in row}, {"where", "kind"})  # 공개 목록에 값 없음
                self.assertEqual([row["where"] for row in result.details], [row["where"] for row in result.mismatches])

    def test_estimate_is_conditional_and_has_no_upper_bound(self):
        plan = baseline.estimate(1200.0, 36, cpus=8)
        self.assertEqual((plan["recommended_workers"], plan["conditional_hours"], plan["exceeds_report_threshold"]),
                         (6, 2.0, False))
        self.assertFalse([key for key in plan if "upper" in key])
        self.assertIn("상한이 아니다", plan["conditions"])
        self.assertTrue(baseline.estimate(7200.0 * 3, 36, cpus=8)["exceeds_report_threshold"])  # 36시간 > 24시간
        self.assertEqual(baseline.estimate(60.0, 36, cpus=1)["recommended_workers"], 1)


def formal_side(delta, wins):
    """정식 기록 꼴의 한 비용 자리: 전략·단독 보유의 전체 연복리와 네 구간 연복리."""
    def account(total, won):
        return {"cagr": str(total), "segments": [{"segment": label, "cagr": "0.06" if index in won else "0.04"}
                                                 for index, label in enumerate(LABELS, start=1)]}
    return {"strategy": account(D("0.05") + D(delta), wins), "hold": account(D("0.05"), ())}


def formal_row(base="0.02", stress="0.005", wins=(1, 2, 3, 4), **verdict):
    """정식 기록 꼴의 한 경로. verdict를 주지 않으면 사유 없는(수치로 정한) 종전 판정이다."""
    old = {"status": "fail", "reasons": [], "checks": {}, **verdict}
    return {"base": None if base is None else formal_side(base, wins),
            "stress": None if stress is None else formal_side(stress, wins), "verdict": old}


def held_verdict(status, reasons, later=()):
    return {"status": status, "reasons": list(reasons), "checks": None, "later_holds": list(later)}


class ExpectedVerdictFromFormalRecordOnly(unittest.TestCase):
    """기준점의 기대 판정과 공통 구간을 정식 기록의 개설 상태·사유·수치만으로 구한다(판정 모듈과 따로 짠 계산)."""

    def expect(self, first, second):
        return baseline.expected_verdict({OHLC: first, OLHC: second}, T)

    def test_values_only(self):
        self.assertEqual(self.expect(formal_row(wins=(1, 2, 3)), formal_row(wins=(2, 3, 4))), (FAIL, ["s2", "s3"]))
        self.assertEqual(self.expect(formal_row(wins=(1, 2, 3)), formal_row()), (PASS, ["s1", "s2", "s3"]))
        self.assertEqual(self.expect(formal_row(base="0.01"), formal_row(base="0.01")), (PASS, list(LABELS)))
        self.assertEqual(self.expect(formal_row(base="0.0099999"), formal_row()), (FAIL, list(LABELS)))
        self.assertEqual(self.expect(formal_row(stress="0"), formal_row()), (FAIL, list(LABELS)))

    def test_opening_states_and_reasons(self):
        rejected = formal_row(stress=None, wins=(1, 2), **held_verdict("fail", ["stress_start_rejected_by_cost_guard"]))
        self.assertEqual(self.expect(rejected, rejected), (FAIL, ["s1", "s2"]))  # 스트레스만 거부: 공통 구간은 구한다
        hold = formal_row(base=None, stress=None, **held_verdict("hold", ["base_combined_start_liquidity"]))
        self.assertEqual(self.expect(formal_row(wins=(1, 2)), hold), (FAIL, None))
        self.assertEqual(self.expect(formal_row(wins=(1, 2, 3)), hold), (OPEN, None))
        gap = formal_row(base=None, stress=None, **held_verdict("hold", ["input_data_gap_after_start"],
                                                               ["input_data_gap_after_start"]))
        self.assertEqual(baseline.formal_flags(gap["verdict"]),
                         {"holds": [], "failures": [], "later": ["input_data_gap_after_start"]})
        self.assertEqual(self.expect(gap, gap), (OPEN, None))
        self.assertEqual(self.expect(formal_row(stress=None), formal_row()), ("invalid", None))  # 사유 없는 빈 수치

    def test_agrees_with_the_judge_on_the_same_cases(self):
        """같은 사례를 판정 모듈의 입력 꼴로 옮기면 두 계산이 같은 상태·공통 구간을 낸다."""
        hold = (formal_row(base=None, stress=None, **held_verdict("hold", ["base_hold_first_buy_failed"])),
                {**path_row(start_holds=["base_hold_first_buy_failed"]), "base": None, "stress": None})
        rejected = (formal_row(stress=None, wins=(1, 2, 3), **held_verdict("fail", ["stress_start_rejected_by_cost_guard"])),
                    {**path_row(wins=(1, 2, 3), start_failures=["stress_start_rejected_by_cost_guard"]), "stress": None})
        valued = [(formal_row(base=b, stress=s, wins=w), path_row(base=b, stress=s, wins=w))
                  for b in ("0.02", "0.005") for s in ("0.005", "-0.001") for w in ((1, 2, 3, 4), (1, 2), (2, 3, 4))]
        cases = valued + [hold, rejected]
        for first_formal, first_judge in cases:
            for second_formal, second_judge in cases:
                verdict = judge.setting_verdict(point(first_judge, second_judge), T)
                self.assertEqual(self.expect(first_formal, second_formal), (verdict["status"], verdict["common_segments"]))
                found = {path: {"status": row["status"], "checks": row["checks"]} for path, row in verdict["paths"].items()}
                self.assertEqual(baseline.expected_paths({OHLC: first_formal, OLHC: second_formal}, T), found)


def formal_like(start, day):
    return {"rows": {"primary": {"symbols": {SYMBOL: {"starts": {day: start}}}}}}


class BaselineCommand(_Workspace):
    """합성 입력의 기준점 대조: 대조와 독립 검산은 그대로 돌지만 본실행 허가 대상은 아니다."""

    def formal_file(self, mutate=None):
        """정식 결과 꼴의 합성 파일(M·D 모두 실행한 승인 코드 결과)과 그 해시를 등록한 격자 정의."""
        config = setting_config(WORLD_BASE, WORLD_SPEC, WORLD_SPEC.setting(28)).with_changes(modes=("M", "D"))
        start = plain(study.run_start(world(), config, J(2)))
        if mutate is not None:
            mutate(start)
        path = self.out / "formal-result.json"
        path.write_text(json.dumps(formal_like(start, "2025-01-02"), ensure_ascii=False, indent=1, sort_keys=True))
        approved = {**SPEC.approved, "formal_result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "baseline_applied_config_sha256": WORLD_BASE.with_changes(modes=("M",)).applied_sha256()}
        return path, replace(WORLD_SPEC, approved=approved)

    def context(self, spec):
        return open_synthetic_context(MemorySource(world_inputs()), spec=spec, base=WORLD_BASE)

    def test_matching_run_compares_every_leaf_and_every_check_holds(self):
        path, spec = self.formal_file()
        record = baseline.run_baseline(self.context(spec), path, self.out)
        comparison = record["comparison"]
        self.assertEqual((comparison["mismatches"], comparison["notation_only"]), ([], []))
        self.assertEqual({comparison["matched"], comparison["string_identical"], comparison["formal_leaves"],
                          comparison["exploration_leaves"], sum(comparison["by_item"].values())}, {comparison["matched"]})
        self.assertGreater(comparison["matched"], 300)
        self.assertEqual(sorted(comparison["by_item"]), sorted(permit.REQUIRED_ITEMS))
        self.assertTrue(all(count > 0 for count in comparison["by_item"].values()))
        self.assertEqual(record["self_checks"], dict.fromkeys(permit.REQUIRED_CHECKS, True))
        self.assertEqual((record["baseline"]["number"], record["exploration"], record["formal_run"]), (28, True, False))
        self.assertEqual(record["baseline"], permit.expected_baseline(self.context(spec)))
        self.assertEqual((record["main_run_allowed"], record["selftest_seal"]), (False, None))  # 합성 입력은 허가 대상이 아니다
        timing = record["timing"]
        self.assertGreaterEqual(set(timing["by_kind_seconds"]), {"load_inputs", "hold", "core", "split", "split_hold",
                                                                "assemble", "matched_rerun"})
        self.assertEqual((timing["by_kind_counts"]["split"], timing["by_kind_counts"]["matched_rerun"]), (4, 1))
        self.assertGreater(timing["max_rss_mb"], 1)
        self.assertEqual((record["estimate"]["settings"], "upper_hours" in record["estimate"]), (36, False))
        folder = self.out / "baseline"
        self.assertEqual(sorted(path.name for path in folder.iterdir()), ["compare.json", "report.md", "sealed"])
        self.assertEqual(sorted(path.name for path in (folder / "sealed").iterdir()),
                         ["mismatches.json", "output.log", "unit.json"])
        self.assertEqual(((folder / "sealed/unit.json").stat().st_mode & 0o777, (folder / "sealed/mismatches.json").read_text()),
                         (0o600, "[]"))
        self.assertEqual(record["sealed_unit_sha256"], sealing.file_digest(folder / "sealed/unit.json"))
        self.assertEqual(sealing.read_sealed(folder / "compare.json", permit.BASELINE_SCHEMA)["seal"], record["seal"])
        report_text = (folder / "report.md").read_text()
        self.assertIn("본실행 금지", report_text)
        self.assertIn("조건의 추정이다. 상한이 아니다", report_text)

    def test_recorded_baseline_is_never_overwritten(self):
        path, spec = self.formal_file()
        ctx = self.context(spec)
        baseline.run_baseline(ctx, path, self.out)
        before = {name: (self.out / "baseline" / name).read_text() for name in ("compare.json", "report.md")}
        with patch.object(baseline, "point_ledgers", side_effect=AssertionError("ledger opened")):
            with self.assertRaisesRegex(sealing.OverwriteError, "already recorded here; use a new output folder"):
                baseline.run_baseline(ctx, path, self.out)
        self.assertEqual({name: (self.out / "baseline" / name).read_text() for name in before}, before)

    def test_any_difference_is_listed_by_place_only(self):
        def bump(start):
            paths = start["items"]["M"]["paths"]
            value = D(paths[OLHC]["base"]["strategy"]["cagr"])
            paths[OLHC]["base"]["strategy"]["cagr"] = str(value + D("1E-60"))
            paths[OHLC]["start_orders"]["base"]["orders"]["split"]["quantity"] += 1
            paths[OHLC]["buys_closed_years"]["stress"].append("2025-01-02")
        path, spec = self.formal_file(bump)
        record = baseline.run_baseline(self.context(spec), path, self.out)
        where = [row["where"] for row in record["comparison"]["mismatches"]]
        self.assertEqual(sorted(where), sorted([f"base/{OLHC}/strategy/cagr", f"start_orders/{OHLC}/base/orders/split/quantity",
                                                f"buys_closed_years/{OHLC}/stress"]))
        self.assertEqual({key for row in record["comparison"]["mismatches"] for key in row}, {"where", "kind"})
        self.assertFalse(record["main_run_allowed"])
        cagr = json.loads(path.read_text())["rows"]["primary"]["symbols"][SYMBOL]["starts"]["2025-01-02"][
            "items"]["M"]["paths"][OLHC]["base"]["strategy"]["cagr"]
        public = (self.out / "baseline/report.md").read_text() + (self.out / "baseline/compare.json").read_text()
        self.assertNotIn(cagr, public)  # 공개 기록에는 칸의 경로와 차이의 종류만 있다
        self.assertIn(f"- 불일치 칸: base/{OLHC}/strategy/cagr (value)", public)
        details = json.loads((self.out / "baseline/sealed/mismatches.json").read_text())
        self.assertEqual({row["where"]: row.get("formal") for row in details}[f"base/{OLHC}/strategy/cagr"], cagr)

    def test_head_fields_are_compared_against_the_exploration_run(self):
        """머리 항목(종목·평가시각·입력 출처)을 정식 쪽에서만 바꾸면 대조가 실패한다. 종목코드는 원문으로 본다."""
        def change_head(start):
            start["symbol"] = "69500"
            start["last_evaluation"] = "2025-04-03 18:00:00+09:00"
            start["input_provenance"]["calendar_sha256"] = "f" * 64
            start["capacity"]["floor_krw"] = "100000000"
        path, spec = self.formal_file(change_head)
        record = baseline.run_baseline(self.context(spec), path, self.out)
        self.assertEqual({row["where"]: row["kind"] for row in record["comparison"]["mismatches"]},
                         {"head/symbol": "value", "head/last_evaluation": "value", "head/capacity/floor_krw": "value",
                          "head/input_provenance/calendar_sha256": "missing_in_exploration"})
        self.assertFalse(record["main_run_allowed"])

    def test_formal_file_hash_is_checked_before_any_ledger(self):
        path, _ = self.formal_file()
        with patch.object(baseline, "point_ledgers", side_effect=AssertionError("ledger opened")):
            with self.assertRaisesRegex(IdentityError, "registered SHA-256"):
                baseline.run_baseline(self.context(WORLD_SPEC), path, self.out)
        self.assertFalse((self.out / "baseline").exists())

    def checks_after(self, change, name):
        """탐색 쪽 조립 결과만 바꾼 뒤의 검산 결과."""
        path, spec = self.formal_file()
        real = baseline.assemble_point

        def altered(ledgers):
            result = real(ledgers)
            change(result)
            return result

        with patch.object(baseline, "assemble_point", side_effect=altered):
            return baseline.run_baseline(self.context(spec), path, self.out / name)

    def test_exploration_side_hold_reason_fails_the_verdict_check(self):
        """수치가 다 있는 기준점에 탐색 쪽 보류 사유만 덧붙이면 검산을 건너뛰지 않고 실패시킨다."""
        record = self.checks_after(lambda r: r["exploration"][OHLC]["flags"]["holds"].append("base_combined_start_liquidity"),
                                   "hold")
        failed = [name for name, ok in record["self_checks"].items() if not ok]
        self.assertEqual((failed, record["comparison"]["mismatches"], record["main_run_allowed"]),
                         (["verdict_equals_formal_only_expectation", "path_cost_checks_equal_formal_only_expectation"],
                          [], False))

    def test_exploration_side_reason_on_one_cost_fails_the_per_cost_check(self):
        """정식 수치가 유효한 자리를 탐색 쪽 사유 하나가 비우면, 설정의 최종 상태와 공통 구간이 같아도 검산이 실패한다.
        스트레스 비용과 기본 비용 각각."""
        for cost, reason in (("stress", "stress_hold_first_buy_failed"), ("base", "base_hold_first_buy_failed"),
                             ("stress-later", "stress_annual_volatility_unavailable")):
            with self.subTest(cost):
                key = "later" if cost.endswith("later") else "holds"
                record = self.checks_after(lambda r: r["exploration"][OLHC]["flags"][key].append(reason), cost)
                checks = record["self_checks"]
                self.assertIs(checks["path_cost_checks_equal_formal_only_expectation"], False)
                self.assertEqual((record["comparison"]["mismatches"], record["main_run_allowed"]), ([], False))
                self.assertTrue(checks["baseline_unit_passes_record_checks"])  # 기록의 꼴은 멀쩡하다
        # 이 합성 기준점은 기본 비용 조건에서 이미 미달이라, 스트레스 사유 하나로는 최종 상태와 공통 구간이 그대로다
        stress = json.loads((self.out / "stress/baseline/compare.json").read_text())["self_checks"]
        self.assertTrue(stress["verdict_equals_formal_only_expectation"])

    def test_expected_path_checks_from_the_formal_record_only(self):
        rejected = formal_row(stress=None, wins=(1, 2), **held_verdict("fail", ["stress_start_rejected_by_cost_guard"]))
        hold = formal_row(base=None, stress=None, **held_verdict("hold", ["base_combined_start_liquidity"]))
        paths = baseline.expected_paths({OHLC: formal_row(base="0.005", stress="0"), OLHC: rejected}, T)
        self.assertEqual(paths, {OHLC: {"status": FAIL, "checks": {"base_edge": False, "stress_edge": False}},
                                 OLHC: {"status": FAIL, "checks": {"base_edge": True, "stress_edge": None}}})
        paths = baseline.expected_paths({OHLC: formal_row(), OLHC: hold}, T)
        self.assertEqual(paths, {OHLC: {"status": PASS, "checks": {"base_edge": True, "stress_edge": True}},
                                 OLHC: {"status": OPEN, "checks": {"base_edge": None, "stress_edge": None}}})

    def test_new_values_are_checked_against_independent_expectations(self):
        def split_return(result):  # 분할 수익률을 코어 원금으로 나눈 값으로
            share = result["exploration"][OHLC]["split_share"]["base"]["split"]
            share["cumulative_return"] = share["ending_equity_krw"] / D(80000000) - 1

        def hold_return(result):
            result["exploration"][OLHC]["split_share"]["stress"]["hold_same_capital"]["ending_equity_krw"] += 1

        cases = {"split": (split_return, "split_share_equals_formal_arithmetic"),
                 "hold": (hold_return, "same_capital_hold_equals_formal_arithmetic"),
                 "buys": (lambda r: r["exploration"][OHLC]["split_ledger"]["base"].update(buys=0),
                          "trade_counts_equal_ledger_events"),
                 "sells": (lambda r: r["exploration"][OLHC]["split_ledger"]["stress"].update(sells=99),
                           "trade_counts_equal_ledger_events"),
                 "reentries": (lambda r: r["exploration"][OHLC]["reentry_counts"]["base"].update(filled=99),
                               "reentry_counts_equal_formal_lists"),
                 "applied": (lambda r: r.update(applied_config_sha256="0" * 64), "applied_config_equals_registered"),
                 "record": (lambda r: r["exploration"][OHLC]["core_share"]["base"].update(capital_krw=D(1)),
                            "baseline_unit_passes_record_checks"),
                 "missing": (lambda r: r["exploration"][OHLC].update(split_share=None),
                             "split_share_equals_formal_arithmetic")}
        for name, (change, check) in cases.items():
            with self.subTest(name):
                record = self.checks_after(change, name)
                self.assertIs(record["self_checks"][check], False)
                self.assertFalse(record["main_run_allowed"])

    def test_expected_new_values_by_hand(self):
        """분할 몫과 같은 원금 보유의 기대값 산식을 손계산으로 확인한다(80%: 코어 8천만·분할 2천만, 수수료 0.015%)."""
        orders = {"core": {"status": "filled", "quantity": 7998, "notional": "79980000"},
                  "hold": {"status": "filled", "quantity": 9998, "notional": "99980000"},
                  "split": {"status": "filled", "quantity": 399, "notional": "3990000"}}
        row = {"start_orders": {"base": {"orders": orders}},
               "base": {"hold": {"ending_equity_krw": "104984003.00"}, "strategy": {"ending_equity_krw": "104185904.5"}}}
        slot = baseline.Slot(OHLC, "base", row, cfg("80", "4", "6", 5))
        # 보유 남은 현금 100,000,000 − 99,980,000 × 1.00015 = 5,003 → 마지막 가격 (104,984,003 − 5,003) ÷ 9,998 = 10,500
        self.assertEqual(baseline._close_price(slot), 10500)
        self.assertEqual(baseline.expected_split(slot), {"core_ending": 83987003, "ending": D("20198901.5"),
                                                        "return": D("0.009945075")})
        self.assertEqual(baseline.expected_same_capital_hold(slot), {"quantity": 1999, "ending": D("20996501.5"),
                                                                    "return": D("0.049825075")})


# ---------------------------------------------------------------------------
# 식별값·입력기·명령
# ---------------------------------------------------------------------------
class IdentityRecords(_Workspace):
    def test_exploration_source_covers_every_python_file_of_the_package(self):
        package = Path(identity.__file__).resolve().parent
        current = identity.exploration_source()
        self.assertEqual(current["exploration_source_files"], sorted(f"kodex_grid/{p.name}" for p in package.glob("*.py")))
        for name in ("permit.py", "records.py", "sealing.py", "progress.py", "stages.py", "judge.py"):
            self.assertIn(f"kodex_grid/{name}", current["exploration_source_files"])
        self.assertRegex(current["exploration_source_sha256"], r"^[0-9a-f]{64}$")
        copy = self.out / "kodex_grid"
        shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))
        with patch.object(identity, "__file__", str(copy / "identity.py")):
            self.assertEqual(identity.exploration_source(), current)
            (copy / "kodex_grid_v1.json").write_text("{}")  # 격자 정의는 따로 해시한다
            self.assertEqual(identity.exploration_source(), current)
            with open(copy / "judge.py", "a") as stream:
                stream.write("\n# changed\n")
            changed = identity.exploration_source()
            self.assertNotEqual(changed["exploration_source_sha256"], current["exploration_source_sha256"])
            (copy / "extra.py").write_text("")
            self.assertIn("kodex_grid/extra.py", identity.exploration_source()["exploration_source_files"])

    def test_approved_source_is_untouched_and_checked(self):
        approved = source_identity()
        self.assertEqual(approved["source_sha256"], "ef6a285b2d0bef22dcb4e767cfd1172116389859be9ad56babbdab5a51381ffb")
        self.assertFalse([name for name in approved["source_files"] if "kodex_grid" in name])
        wrong = replace(SPEC, approved={**SPEC.approved, "source_sha256": "0" * 64})
        with self.assertRaisesRegex(IdentityError, "approved value"):
            identity.code_identity(wrong, FROZEN)
        ctx = world_context()
        self.assertEqual((ctx.identity["exploration"], ctx.identity["formal_run"], ctx.identity["scope"], ctx.registered),
                         (True, False, "synthetic", False))
        self.assertEqual(ctx.identity["base_config_file_sha256"], FROZEN.sha256)
        self.assertNotEqual(ctx.identity["base_config_applied_sha256"], FROZEN.applied_sha256())  # 합성 기간으로 바꾼 설정

    def test_registered_baseline_hash_is_the_published_one(self):
        """28번의 실제 적용 해시(등록값) = 고정 설정에서 재진입 방식만 M으로 바꾼 값 = 설정 번호로 다시 만든 값."""
        published = "eb00cf47e06cbc962c071768499e17d1f8905f7b0533d20073d8a67ea590f388"
        self.assertEqual(SPEC.approved["baseline_applied_config_sha256"], published)
        self.assertEqual(FROZEN.with_changes(modes=("M",)).applied_sha256(), published)
        self.assertEqual(cfg("80", "2", "3", 9).applied_sha256(), published)


class BundleSourceEndToEnd(_Workspace):
    """합성 M5 묶음(등록값 대체)으로 실제 입력기·사전 대조를 거쳐 1단계를 끝까지 돌린다."""

    def bundle(self, **options):
        root = self.out / "bundle"
        return root, synthetic_bundle(root, **options)

    def source(self, root, manifest, base=FROZEN):
        return BundleSource(root, manifest, base, registered=formal.actual_inputs(root, manifest, FROZEN))

    def test_stage_one_through_the_pinned_loader(self):
        root, manifest = self.bundle(synthetic=False)
        ctx = open_synthetic_context(self.source(root, manifest))
        inputs = ctx.identity["inputs"]
        self.assertEqual((set(inputs["actual"]), inputs["registered_inputs_frozen"], ctx.registered),
                         (set(formal.IDENTITY_KEYS), False, False))
        self.assertEqual(inputs["price_basis_by_symbol"], {symbol: BASIS for symbol in ("069500", "169950", "229200")})
        out = self.stage1(ctx, workers=4)
        result = stages.assemble1(ctx, out)
        self.assertEqual(len(result["settings"]), 36)
        unit = json.loads((out / "stage1/units/setting-28.json").read_text())
        provenance = unit["result"]["run_start"]["input_provenance"]
        self.assertEqual((provenance["manifest_synthetic"], provenance["input_manifest_sha256"], provenance["source_sha256"]),
                         (False, inputs["actual"]["input_manifest_sha256"], SPEC.approved["source_sha256"]))
        loaded = loader.load_etf(root, manifest, SPEC.symbol, SPEC.primary_start, FROZEN.end)
        expected = study.run_start(loaded, cfg("80", "2", "3", 9), SPEC.primary_start)
        self.assertEqual(unit["result"]["run_start"], json.loads(sealing.dump(expected)))
        self.assertEqual((unit["result"]["run_start"]["end"], unit["identity"]["end"]), ("2026-11-13", "2026-11-13"))
        self.assertEqual(result["frame"]["labels"], [segment.label for segment in FROZEN.segments])

    def test_unregistered_synthetic_or_modified_inputs_stop_before_ledgers(self):
        root, manifest = self.bundle(synthetic=False)
        with patch.object(stages, "run_point", side_effect=AssertionError("ledger opened")):
            with self.assertRaisesRegex(InputIntegrityError, "input_manifest_sha256"):
                cli.main(["stage1", "--data-root", str(root), "--manifest", str(manifest), "--out", str(self.out / "o")])
            with self.assertRaisesRegex(study.InputIdentityError, "frozen configuration"):
                open_synthetic_context(self.source(root, manifest, FROZEN.with_changes(end=date(2026, 11, 12))), base=FROZEN)
            with self.assertRaisesRegex(IdentityError, "inputs are not the registered bundle"):
                open_context(self.source(root, manifest))  # 등록 전용 진입점은 합성 묶음을 받지 않는다
        self.assertFalse((self.out / "o").exists())
        shutil.rmtree(root)
        root, manifest = self.bundle(synthetic=True)
        with self.assertRaisesRegex(InputIntegrityError, "synthetic"):
            open_synthetic_context(self.source(root, manifest))

    def test_loaded_input_must_match_the_checked_bundle(self):
        root, manifest = self.bundle(synthetic=False)
        source = self.source(root, manifest)
        open_synthetic_context(source)
        real = identity.load_etf

        def swapped(*args, **options):
            loaded = real(*args, **options)
            return replace(loaded, provenance={**loaded.provenance, "daily_sha256": "f" * 64})

        with patch.object(identity, "load_etf", side_effect=swapped):
            with self.assertRaisesRegex(study.InputIdentityError, "daily_sha256"):
                source.load(SPEC.symbol, SPEC.primary_start)
        self.assertEqual(source.load(SPEC.symbol, SPEC.primary_start).start, SPEC.primary_start)

    def test_command_line_arguments(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cli._parser().parse_args(["formal", "--data-root", "a", "--manifest", "b", "--out", "c"])
        args = cli._parser().parse_args(["stage1", "--data-root", "a", "--manifest", "b", "--out", "c"])
        self.assertEqual((args.workers, args.formal_result, args.stage), (1, None, "stage1"))
        with self.assertRaisesRegex(SystemExit, "--formal-result"):
            cli._baseline(None, args)
        with self.assertRaisesRegex(SystemExit, "stage1 needs --data-root and --manifest"):
            cli.main(["stage1", "--out", str(self.out)])
        self.assertEqual(cli.COMMANDS, ("selftest", "baseline", "stage1", "assemble1", "stage2", "assemble2", "eta"))


# ---------------------------------------------------------------------------
# 본실행 허가: 실제 묶음은 같은 판본의 합성 시험 통과 기록과 기준점 대조 증거가 있어야 들어온다
# ---------------------------------------------------------------------------
PROBE = """import unittest


class Probe(unittest.TestCase):
    def test_one(self):
        self.assertTrue(True)

    def test_two(self):
        self.assertEqual(1 + 1, 2)
"""
STAGE_CALLS = (stages.run_stage1, stages.assemble1, stages.run_stage2, stages.assemble2)


class _Permit(_Workspace):
    """시험 묶음 자리에 작은 시험 모듈 하나를 끼운다(전체 시험 묶음을 시험 안에서 다시 돌리지 않는다)."""

    PROBE_TEXT = PROBE

    def setUp(self):
        super().setUp()
        self.module = f"kg_probe_{self.id().rsplit('.', 1)[-1]}"
        self.probe = self.out / "probe" / f"{self.module}.py"
        self.probe.parent.mkdir()
        self.probe.write_text(self.PROBE_TEXT)
        sys.path.insert(0, str(self.probe.parent))
        self.addCleanup(sys.path.remove, str(self.probe.parent))
        self.addCleanup(sys.modules.pop, self.module, None)
        self.enterContext(patch.object(permit, "TEST_MODULES", (self.module,)))

    def registered(self, thresholds=None, mutate=None):
        """합성 M5 묶음을 '등록한 실제 묶음' 자리에 끼운 문맥과 그 정식 결과 꼴 파일.
        등록 입력값과 고정 격자 정의(정식 결과 해시, 필요하면 문턱)만 시험용으로 바꾸고, 입력기·사전 대조·허가 검사는
        그대로 거친다."""
        root = self.out / "bundle"
        manifest = synthetic_bundle(root, synthetic=False, mutate=mutate)
        actual = formal.actual_inputs(root, manifest, FROZEN)
        loaded = loader.load_etf(root, manifest, SPEC.symbol, SPEC.primary_start, FROZEN.end)
        start = plain(study.run_start(loaded, FROZEN, SPEC.primary_start))
        path = self.out / "formal.json"
        path.write_text(json.dumps(formal_like(start, SPEC.primary_start.isoformat()), sort_keys=True))
        spec = replace(SPEC, approved={**SPEC.approved, "formal_result_sha256": sealing.file_digest(path)},
                       thresholds=thresholds or SPEC.thresholds)
        self.enterContext(patch.object(identity, "registered_inputs", return_value=actual))
        self.enterContext(patch.object(identity, "frozen_grid", return_value=spec))
        loader._source.cache_clear()
        return open_context(BundleSource(root, manifest, FROZEN, registered=actual)), path

    def permitted(self, **options):
        """시험 통과 기록과 기준점 대조 증거를 모두 갖춘 출력 폴더."""
        ctx, formal_path = self.registered(**options)
        out = self.out / "run"
        permit.run_selftest(out)
        record = baseline.run_baseline(ctx, formal_path, out)
        return ctx, out, record

    def retouch(self, out, change):
        """기준점 대조 기록을 고치고 자기 해시까지 맞춘다(필수 항목 검사에 닿게 하는 시험용)."""
        path = out / "baseline/compare.json"
        record = json.loads(path.read_text())
        change(record)
        path.write_text(sealing.dump(sealing.with_seal(record)))


class MainRunPermit(_Permit):
    def test_no_evidence_refuses_every_stage_function(self):
        ctx, _ = self.registered()
        self.assertEqual((ctx.registered, ctx.identity["scope"]), (True, "registered"))
        out = self.out / "run"
        with patch.object(stages, "run_point", side_effect=AssertionError("ledger opened")):
            for call in STAGE_CALLS:
                with self.assertRaisesRegex(PermitError, "selftest/record.json is missing; the main run is not started"):
                    call(ctx, out)
        self.assertFalse(out.exists())
        permit.run_selftest(out)  # 시험 통과 기록만으로는 부족하다: 기준점 대조 증거도 있어야 한다
        for call in STAGE_CALLS:
            with self.assertRaisesRegex(PermitError, "baseline/compare.json is missing"):
                call(ctx, out)
        self.assertEqual(sorted(path.name for path in out.iterdir()), [".writer.lock", "selftest"])

    def test_two_field_permit_file_is_refused(self):
        ctx, _ = self.registered()
        out = self.out / "run"
        permit.run_selftest(out)
        compare = out / "baseline/compare.json"
        compare.parent.mkdir()
        two_fields = {"identity": json.loads(sealing.dump(ctx.identity)), "main_run_allowed": True}
        compare.write_text(json.dumps(two_fields))  # r1의 문턱이 받아들이던 파일
        with self.assertRaisesRegex(PermitError, "baseline/compare.json is not an intact kodex-grid-baseline-v2 record"):
            stages.run_stage1(ctx, out)
        compare.write_text(sealing.dump(sealing.with_seal({"schema": permit.BASELINE_SCHEMA, **two_fields})))
        with self.assertRaises(PermitError) as caught:  # 꼴과 자기 해시를 갖춰도 필수 항목이 없으면 통과하지 못한다
            stages.run_stage1(ctx, out)
        for problem in ("not an exploration record", "formal result hash is not the registered one",
                        "baseline setting or its applied configuration hash", "comparison has mismatches or no mismatch list",
                        "a required comparison item is missing", "matched leaves do not cover the whole formal record",
                        "a required self-check is missing or not true", "not taken after this version's synthetic test record",
                        "sealed baseline unit is missing or changed"):
            self.assertIn(problem, str(caught.exception))
        self.assertFalse((out / "stage1").exists())

    def test_complete_evidence_permits_the_run_and_the_assembly(self):
        ctx, out, record = self.permitted()
        self.assertEqual((record["comparison"]["mismatches"], record["self_checks"], record["main_run_allowed"]),
                         ([], dict.fromkeys(permit.REQUIRED_CHECKS, True), True))
        self.assertEqual(record["selftest_seal"], permit.check_selftest(out)["seal"])
        self.assertEqual(record["baseline"]["applied_config_sha256"], SPEC.approved["baseline_applied_config_sha256"])
        self.assertIn("본실행 허용: **예**", (out / "baseline/report.md").read_text())
        permit.enter(ctx, out)
        stages.run_stage1(ctx, out, workers=4, echo=self.lines.append)
        result = stages.assemble1(ctx, out)
        self.assertEqual((len(result["settings"]), result["identity"]["scope"]), (36, "registered"))
        self.assertEqual(stages.run_stage2(ctx, out, echo=self.lines.append), result["selection"]["stage2"])

    def test_each_missing_piece_of_baseline_evidence_is_refused(self):
        ctx, out, _ = self.permitted()
        original = (out / "baseline/compare.json").read_text()
        cases = {
            "mismatch": (lambda r: r["comparison"]["mismatches"].append({"where": "x", "kind": "value"}), "has mismatches"),
            "no_list": (lambda r: r["comparison"].pop("mismatches"), "no mismatch list"),
            "item": (lambda r: r["comparison"]["by_item"].pop("reentries"), "a required comparison item is missing"),
            "empty_item": (lambda r: r["comparison"]["by_item"].update(base=0), "a required comparison item is missing"),
            "leaves": (lambda r: r["comparison"].update(formal_leaves=r["comparison"]["matched"] + 1), "do not cover"),
            "check_false": (lambda r: r["self_checks"].update(split_share_equals_formal_arithmetic=False), "self-check"),
            "check_gone": (lambda r: r["self_checks"].pop("verdict_equals_formal_only_expectation"), "self-check"),
            "check_truthy": (lambda r: r["self_checks"].update(matched_rerun_equal="yes"), "self-check"),
            "formal_hash": (lambda r: r["formal_result"].update(sha256="0" * 64), "formal result hash"),
            "applied": (lambda r: r["baseline"].update(applied_config_sha256="0" * 64), "applied configuration hash"),
            "number": (lambda r: r["baseline"].update(number=27), "baseline setting"),
            "identity": (lambda r: r["identity"].update(exploration_source_sha256="0" * 64), "identity"),
            "inputs": (lambda r: r["identity"]["inputs"]["actual"].update(daily_sha256="0" * 64), "identity"),
            "selftest": (lambda r: r.update(selftest_seal="0" * 64), "synthetic test record"),
            "unit": (lambda r: r.update(sealed_unit_sha256="0" * 64), "sealed baseline unit"),
            "not_allowed": (lambda r: r.update(main_run_allowed=False), "did not allow the main run"),
            "formal_flag": (lambda r: r.update(formal_run=True), "not an exploration record"),
        }
        for name, (change, message) in cases.items():
            with self.subTest(name):
                self.retouch(out, change)
                for call in STAGE_CALLS:
                    with self.assertRaisesRegex(PermitError, message):
                        call(ctx, out)
                (out / "baseline/compare.json").write_text(original)
        (out / "baseline/sealed/unit.json").write_text("{}")  # 봉인한 기준점 단위가 바뀌어도 거부한다
        with self.assertRaisesRegex(PermitError, "sealed baseline unit is missing or changed"):
            stages.run_stage1(ctx, out)
        self.assertFalse((out / "stage1").exists())

    def test_evidence_of_another_version_is_refused(self):
        ctx, out, _ = self.permitted()
        self.probe.write_text(self.PROBE_TEXT + "\n# edited after the record\n")  # 시험 파일이 바뀐 판본
        with self.assertRaisesRegex(PermitError, "test files differ from this version"):
            stages.run_stage1(ctx, out)
        self.probe.write_text(self.PROBE_TEXT)
        permit.enter(ctx, out)
        changed = {"exploration_source_sha256": "1" * 64, "exploration_source_files": []}
        with patch.object(identity, "exploration_source", return_value=changed):  # 탐색 코드가 바뀐 판본
            with self.assertRaisesRegex(PermitError, "does not describe the current code"):
                stages.run_stage1(ctx, out)
            loader._source.cache_clear()
            newer = open_context(ctx.source)
            with self.assertRaisesRegex(PermitError, "exploration source, grid file or test files differ"):
                stages.run_stage1(newer, out)
        self.assertFalse((out / "stage1").exists())

    def test_baseline_needs_the_test_record_first_and_registered_definitions(self):
        ctx, formal_path = self.registered()
        out = self.out / "run"
        with patch.object(baseline, "point_ledgers", side_effect=AssertionError("ledger opened")):
            with self.assertRaisesRegex(PermitError, "selftest/record.json is missing"):
                baseline.run_baseline(ctx, formal_path, out)
            forged = identity.Context(replace(ctx.spec, thresholds=EVERYTHING_PASSES), ctx.base, ctx.source, ctx.identity)
            with self.assertRaisesRegex(PermitError, "does not describe the current code and definitions"):
                baseline.run_baseline(forged, formal_path, out)
        self.assertFalse(out.exists())

    def test_synthetic_entry_does_not_take_the_registered_bundle(self):
        ctx, _ = self.registered()
        with self.assertRaisesRegex(IdentityError, "the synthetic entry does not take the registered bundle"):
            open_synthetic_context(ctx.source)
        with self.assertRaisesRegex(IdentityError, "a registered run needs the frozen definitions"):
            identity.check_frozen(identity.Context(WORLD_SPEC, ctx.base, ctx.source, ctx.identity))  # 격자 정의를 바꾼 문맥
        relabeled = {**ctx.identity, "scope": "synthetic", "inputs": {**ctx.identity["inputs"], "registered_inputs_frozen": False}}
        forged = identity.Context(ctx.spec, ctx.base, MemorySource(world_inputs()), relabeled)
        self.assertTrue(forged.registered)  # 표시를 바꿔도 입력 식별값이 등록 묶음이면 실제 묶음으로 본다
        for call in STAGE_CALLS:
            with self.assertRaisesRegex(IdentityError, "source is not the pinned bundle loader"):
                call(forged, self.out / "forged")
        self.assertFalse((self.out / "forged").exists())

    def test_input_from_the_registered_bundle_cannot_reach_a_synthetic_run(self):
        """합성 입력기가 등록한 실제 묶음의 출처를 단 입력을 돌려주면 적재 직후에 멈춘다."""
        marked = replace(world(), provenance={"daily_sha256": FROZEN.registered["daily_sha256"]})
        ctx = open_synthetic_context(MemorySource({**world_inputs(), (SYMBOL, J(2)): marked}), spec=WORLD_SPEC, base=WORLD_BASE)
        with patch.object(stages, "run_point", side_effect=AssertionError("ledger opened")):
            with self.assertRaisesRegex(IdentityError, "an input loaded from the registered bundle reached a synthetic run"):
                stages.run_stage1(ctx, self.out / "run", echo=self.lines.append)
        self.assertEqual(self.lines, [])
        with self.assertRaisesRegex(IdentityError, "a registered run needs the frozen definitions"):
            open_context(MemorySource(world_inputs()))  # 등록 전용 진입점은 합성 입력기를 받지 않는다

    @unittest.skipUnless((REAL_ROOT / "m5-inputs.json").is_file(), "pinned data not present")
    def test_real_bundle_without_evidence_is_refused_before_any_ledger(self):
        """실제 고정 묶음: 장부 없이 입력 대조만 한다(성과 계산 없음). 허가 기록이 없으면 어느 단계도 시작하지 않는다."""
        manifest = REAL_ROOT / "m5-inputs.json"
        refuse = AssertionError("ledger opened")
        with patch.object(stages, "run_point", side_effect=refuse), patch.object(baseline, "point_ledgers", side_effect=refuse):
            ctx = open_context(BundleSource(REAL_ROOT, manifest, load_config()))
            inputs = ctx.identity["inputs"]
            self.assertEqual((inputs["actual"], inputs["registered_inputs_frozen"], ctx.registered),
                             (FROZEN.registered, True, True))
            self.assertEqual(inputs["price_basis_by_symbol"],
                             {symbol: FROZEN.price_basis for symbol in ("069500", "169950", "229200")})
            for call in STAGE_CALLS:
                with self.assertRaisesRegex(PermitError, "selftest/record.json is missing"):
                    call(ctx, self.out / "real")
            with self.assertRaisesRegex(PermitError, "selftest/record.json is missing"):
                baseline.run_baseline(ctx, self.out / "formal.json", self.out / "real")
            with self.assertRaisesRegex(IdentityError, "the synthetic entry does not take the registered bundle"):
                open_synthetic_context(BundleSource(REAL_ROOT, manifest, load_config()))
        self.assertFalse((self.out / "real").exists())


class BodyProvenanceAndEvaluationTime(_Workspace):
    """묶음 입력기를 거친 단위(등록 단위)의 본문: 필수 출처 항목이 실제로 있고 머리와 같아야 하며, 마지막 평가는
    마지막 거래일의 18시 한국시각이어야 한다."""

    def setUp(self):
        super().setUp()
        root = self.out / "bundle"
        manifest = synthetic_bundle(root, synthetic=False)
        source = BundleSource(root, manifest, FROZEN, registered=formal.actual_inputs(root, manifest, FROZEN))
        self.ctx = open_synthetic_context(source)
        self.run_dir = self.stage1(self.ctx, workers=4)
        self.folder = self.run_dir / "stage1"

    def refused(self, message):
        with self.assertRaisesRegex(stages.AssemblyError, message):
            stages.assemble1(self.ctx, self.run_dir)
        with self.assertRaisesRegex(IdentityError, "sealed unit is not reused: .*" + message):
            self.stage1(self.ctx)

    def test_untouched_bundle_units_pass_and_state_every_promised_item(self):
        unit = json.loads((self.folder / "units/setting-09.json").read_text())
        seat = stages.stage1_plan(self.ctx, self.run_dir).seat(records.primary_unit(SPEC, 9))
        promised, stated = records.promised_provenance(seat), records.stated_provenance(unit["result"]["run_start"])
        self.assertEqual(sorted(promised), sorted(["input_manifest_sha256", "daily_sha256", "metadata_sha256",
                                                    "calendar_sha256", "price_bundle", "market_rules_sha256",
                                                    "source_sha256", "price_basis", "manifest_synthetic",
                                                    "extra_inputs_present", "extra_inputs_listed"]))
        self.assertEqual({key: stated[key] for key in promised}, promised)
        self.assertEqual((promised["manifest_synthetic"], promised["extra_inputs_present"], promised["price_basis"]),
                         (False, False, BASIS))
        self.assertEqual(unit["result"]["run_start"]["last_evaluation"], "2026-11-13 18:00:00+09:00")
        self.assertEqual(len(stages.assemble1(self.ctx, self.run_dir)["settings"]), 36)

    def test_body_that_contradicts_its_head_is_refused(self):
        def provenance(record):
            return record["result"]["run_start"]["input_provenance"]

        def start(record):
            return record["result"]["run_start"]

        differs = "input provenance differs from the run identity in "
        cases = {
            "market_rules": (lambda r: provenance(r).update(market_rules_sha256="0" * 64), differs + r"\['market_rules_sha256'\]"),
            "market_rules_missing": (lambda r: provenance(r).pop("market_rules_sha256"), differs + r"\['market_rules_sha256'\]"),
            "source_missing": (lambda r: provenance(r).pop("source_sha256"), differs + r"\['source_sha256'\]"),
            "source_changed": (lambda r: provenance(r).update(source_sha256="0" * 64), differs + r"\['source_sha256'\]"),
            "price_basis": (lambda r: provenance(r).update(price_basis="raw"), differs + r"\['price_basis'\]"),
            "price_basis_missing": (lambda r: provenance(r).pop("price_basis"), differs + r"\['price_basis'\]"),
            "basis_label": (lambda r: start(r)["basis"].update(price_basis="raw"), differs + r"\['basis.price_basis'\]"),
            "synthetic": (lambda r: provenance(r).update(manifest_synthetic=True), differs + r"\['manifest_synthetic'\]"),
            "synthetic_zero": (lambda r: provenance(r).update(manifest_synthetic=0), differs + r"\['manifest_synthetic'\]"),
            "synthetic_missing": (lambda r: provenance(r).pop("manifest_synthetic"), differs + r"\['manifest_synthetic'\]"),
            "extra_present": (lambda r: provenance(r)["extra_inputs"].update(present=True), differs + r"\['extra_inputs_present'\]"),
            "extra_listed": (lambda r: provenance(r)["extra_inputs"].update(locked_status=[["2011-01-03", False]]),
                             differs + r"\['extra_inputs_listed'\]"),
            "extra_missing": (lambda r: provenance(r).pop("extra_inputs"),
                              differs + r"\['extra_inputs_listed', 'extra_inputs_present'\]"),
            "daily": (lambda r: provenance(r).update(daily_sha256="f" * 64), differs + r"\['daily_sha256'\]"),
            "calendar_missing": (lambda r: provenance(r).pop("calendar_sha256"), differs + r"\['calendar_sha256'\]"),
            "morning": (lambda r: start(r).update(last_evaluation="2026-11-13 09:00:00+09:00"),
                        "last evaluation is not 18:00 KST of the last session"),
            "utc_clock": (lambda r: start(r).update(last_evaluation="2026-11-13 09:00:00+00:00"), "last evaluation is not"),
            "day_before": (lambda r: start(r).update(last_evaluation="2026-11-12 18:00:00+09:00"), "last evaluation is not"),
            "no_time": (lambda r: start(r).update(last_evaluation=None), "last evaluation is not"),
        }
        name = "setting-09.json"
        originals = [(self.folder / part / name).read_bytes() for part in ("units", "sealed/seals")]
        for case, (change, message) in cases.items():
            with self.subTest(case):
                reseal(self.folder, name, change)
                self.refused(message)
                for part, raw in zip(("units", "sealed/seals"), originals):
                    (self.folder / part / name).write_bytes(raw)
        stages.assemble1(self.ctx, self.run_dir)

    def test_capacity_hold_with_a_data_gap_from_the_pinned_loader(self):
        """고정 입력기가 만든 '명목 한도 미달 + 이후 일봉 누락' 확장 입력의 단위도 기록 검사를 통과하고 미확정으로 읽힌다."""
        def thin_then_broken(rows):
            for row in rows:
                if row["symbol"] == "169950":
                    row["volume"] = 100  # 20일 평균 거래대금 × 5%가 1억 원에 한참 못 미친다
                    if str(row["date"]) == "2019-01-03":
                        row["open"] = ""
        root = self.out / "thin"
        manifest = synthetic_bundle(root, synthetic=False, mutate=thin_then_broken)
        loader._source.cache_clear()
        ctx = open_synthetic_context(BundleSource(root, manifest, FROZEN, registered=formal.actual_inputs(root, manifest, FROZEN)))
        unit = records.Unit("extension", 19, "169950", SPEC.extension["start"])
        seat = records.Seat(ctx, "stage2", unit)
        loaded = ctx.source.load(unit.symbol, unit.start)
        self.assertEqual(loaded.data_end["first_bad_session"], date(2019, 1, 3))
        result = run_point(loaded, seat.config(), unit.start, nominal_floor=seat.nominal_floor())
        record = {"schema": records.UNIT_SCHEMA, "exploration": True, "formal_run": False, "identity": seat.identity(),
                  "setting": SPEC.setting(19).describe(), "result": result}
        stored = json.loads(sealing.dump(record))
        records.check_record(seat, stored)  # 개설한 장부에만 생기는 결측 표시를 요구하지 않는다
        opened = stored["result"]["run_start"]
        self.assertEqual((opened["capacity"]["ok"], opened["data_end"]["requested_end"], opened["end"]),
                         (False, "2026-11-13", opened["data_end"]["last_valid_session"]))
        self.assertTrue(opened["last_evaluation"].startswith(opened["end"] + " 18:00:00+09:00"))
        point = judge.extension_point(reading.extension_input(stored["result"], seat.frame()), D(0))
        self.assertEqual((point["status"], point["reasons"]), (OPEN, ["start_capacity_below_floor"]))
        stored["result"]["run_start"]["data_end"]["reason"] = ""
        with self.assertRaisesRegex(stages.AssemblyError, "justified early stop"):
            records.check_record(seat, stored)


class InnermostBoundary(_Permit):
    """묶음을 실행·재개·조립하는 가장 안쪽 공용 함수도 그 출력 폴더의 허가와 작성자 잠금을 확인한다."""

    def log(self, *row):
        self.lines.append(row)

    def inner_calls(self, ctx, out):
        plan = stages.stage1_plan(ctx, out)
        unit = plan.units[0]
        forged = ({"state": "selected", "region": {"numbers": list(SPEC.regions()[0].numbers)}}, "0" * 64)
        second = stages.stage2_plan(ctx, out, forged)
        return {"run_units": lambda: stages.run_units(plan, 1, self.log), "load_inputs": plan.load_inputs,
                "compute": lambda: plan.compute(unit, Bench()), "seal": lambda: plan.seal(unit, 0.0),
                "verified": lambda: plan.verified(unit), "read_units": lambda: stages.read_units(plan),
                "locked_selection": lambda: stages.locked_selection(ctx, out),
                "stage2_run_units": lambda: stages.run_units(second, 1, self.log),
                "stage2_load_inputs": second.load_inputs, "stage2_read_units": lambda: stages.read_units(second)}

    def test_direct_batch_calls_without_evidence_stop_before_any_side_effect(self):
        """허가 기록이 없는 등록 입력 문맥: 잠금을 잡고 안쪽 함수를 직접 불러도 입력 적재·임시 파일 정리·단위 쓰기 전에 멈춘다."""
        ctx, _ = self.registered()
        out = self.out / "run"
        leftover = out / "stage1/units/.setting-03.json.999.tmp"
        leftover.parent.mkdir(parents=True)
        leftover.write_text("interrupted write")
        refuse = AssertionError("side effect before the permit")
        with patch.object(ctx.source, "load", side_effect=refuse), patch.object(stages, "run_point", side_effect=refuse), \
                patch.object(stages, "clear_temporaries", side_effect=refuse):
            for name, call in self.inner_calls(ctx, out).items():
                with self.subTest(name):
                    with self.assertRaisesRegex(PermitError, "selftest/record.json is missing; the main run is not started"):
                        call()  # 잠금 없이
                    with sealing.writer_lock(out):  # 작성자 잠금을 정상적으로 잡은 뒤에도
                        with self.assertRaisesRegex(PermitError, "selftest/record.json is missing"):
                            call()
        self.assertEqual(leftover.read_text(), "interrupted write")
        files = sorted(str(path.relative_to(out)) for path in out.rglob("*") if path.is_file())
        self.assertEqual((files, self.lines), ([".writer.lock", "stage1/units/.setting-03.json.999.tmp"], []))

    def test_with_evidence_the_lock_and_the_first_selection_are_still_required(self):
        """허가가 있어도 이 호출 흐름이 작성자 잠금을 잡고 있어야 하고, 2단계 묶음은 최초 선택에서 나온 것이어야 한다.
        등록 경로(고정 입력기)의 1·2단계 단위가 강화한 기록 검사를 모두 통과하는 것도 여기서 확인한다."""
        ctx, out, record = self.permitted(thresholds=EVERYTHING_PASSES)
        self.assertEqual(record["self_checks"], dict.fromkeys(permit.REQUIRED_CHECKS, True))
        for name, call in self.inner_calls(ctx, out).items():
            with self.subTest(name):
                with self.assertRaisesRegex(PermitError, "the writer lock of .* is not held by this call flow"):
                    call()
        self.assertFalse((out / "stage1").exists())
        plan = stages.stage1_plan(ctx, out)
        with sealing.writer_lock(out):
            stages.run_units(plan, 4, self.log)  # 허가와 잠금이 있으면 안쪽 함수도 실행된다
            self.assertEqual(len(stages.read_units(plan)[0]), 36)
        selection = stages.assemble1(ctx, out)["selection"]
        self.assertEqual((selection["state"], selection["region"]["numbers"]), ("selected", list(SPEC.regions()[0].numbers)))
        lock_sha = sealing.file_digest(out / "stage1/lock.json")
        other_region = {**selection, "region": {**selection["region"], "numbers": list(SPEC.regions()[1].numbers)}}
        forged = {"other numbers": stages.stage2_plan(ctx, out, (other_region, lock_sha)),
                  "other record": stages.stage2_plan(ctx, out, (selection, "0" * 64))}
        refuse = AssertionError("side effect before the link check")
        with sealing.writer_lock(out), patch.object(ctx.source, "load", side_effect=refuse):
            for name, plan2 in forged.items():
                for call in (lambda: stages.run_units(plan2, 1, self.log), plan2.load_inputs, lambda: stages.read_units(plan2)):
                    with self.subTest(name):
                        with self.assertRaisesRegex(IdentityError, "stage2 units are not the ones locked by the first stage1 record"):
                            call()
        self.assertFalse((out / "stage2").exists())
        self.assertEqual(stages.run_stage2(ctx, out, workers=4, echo=self.lines.append), "run")
        result = stages.assemble2(ctx, out)
        self.assertEqual(result["kodex_reconfirm"]["verdict"]["pairs"], 16)
        for symbol in SPEC.extension["symbols"]:  # 2017-01-02 시작의 확장 단위(세 구간, 명목 한도)도 그대로 조립된다
            self.assertEqual(result["extension"]["verdict"]["products"][symbol]["denominator"], 8)
        unit = json.loads((out / "stage2/units/extension-229200-2017-01-02-setting-19.json").read_text())
        self.assertEqual(unit["identity"]["stage1_lock_sha256"], lock_sha)
        self.assertEqual(unit["result"]["run_start"]["capacity"]["floor_krw"], "100000000")
        self.assertEqual([part["segment"] for part in unit["result"]["run_start"]["items"]["M"]["paths"][OHLC][
            "hold_metrics"]["base"]["segments"]], ["2014-2019", "2020-2023", "2024-2026-11-13"])

    def test_exploration_side_cost_reason_forbids_the_main_run(self):
        """탐색 쪽 한 경로에 한 비용의 보류 사유 하나만 덧붙은 기준점: 값 대조는 불일치 0건이어도 본실행을 허가하지 않는다."""
        ctx, formal_path = self.registered()
        real = baseline.assemble_point
        for cost in ("stress", "base"):
            with self.subTest(cost):
                def altered(ledgers, reason=f"{cost}_hold_first_buy_failed"):
                    result = real(ledgers)
                    result["exploration"][OLHC]["flags"]["holds"].append(reason)
                    return result

                out = self.out / cost
                permit.run_selftest(out)
                with patch.object(baseline, "assemble_point", side_effect=altered):
                    record = baseline.run_baseline(ctx, formal_path, out)
                self.assertEqual((record["comparison"]["mismatches"], record["main_run_allowed"]), ([], False))
                self.assertIs(record["self_checks"]["path_cost_checks_equal_formal_only_expectation"], False)
                for call in STAGE_CALLS:
                    with self.assertRaisesRegex(PermitError, "a required self-check is missing or not true"):
                        call(ctx, out)
                self.assertFalse((out / "stage1").exists())


class SyntheticTestRecord(_Permit):
    """합성 시험 통과 기록: 탐색 소스·시험 파일 해시, 시험 수, 결과. 몇 칸만 적은 파일은 통과하지 못한다."""

    def test_record_of_a_passing_run(self):
        out = self.out / "run"
        record = permit.run_selftest(out)
        self.assertEqual((record["tests_run"], record["modules"], record["passed"]), (2, {self.module: 2}, True))
        self.assertEqual({key: record[key] for key in permit.CLEAN}, permit.CLEAN)
        self.assertEqual(record["version"], permit.version())
        self.assertEqual(set(record["version"]), {"approved_source_sha256", "exploration_source_sha256", "grid_file_sha256",
                                                  "test_files"})
        self.assertEqual(record["version"]["test_files"], {self.module: sealing.file_digest(self.probe)})
        self.assertEqual(permit.check_selftest(out), record)
        text = (out / "selftest/record.json").read_text()
        with patch.object(permit, "_run_suites", side_effect=AssertionError("tests rerun")):
            self.assertEqual(permit.run_selftest(out), record)  # 같은 판본의 기록이 있으면 다시 돌리지 않는다
        self.assertEqual((out / "selftest/record.json").read_text(), text)
        self.assertIn("Ran 2 tests", (out / "selftest/sealed/output.log").read_text())

    def test_hand_written_or_partial_records_are_refused(self):
        out = self.out / "run"
        record = permit.run_selftest(out)
        path = out / "selftest/record.json"
        path.write_text(json.dumps({"passed": True, "tests_run": 2}))
        with self.assertRaisesRegex(PermitError, "selftest/record.json is not an intact kodex-grid-selftest-v1 record"):
            permit.check_selftest(out)
        path.write_text(sealing.dump({**record, "tests_run": 3}))  # 한 칸만 고친 기록(자기 해시가 맞지 않는다)
        with self.assertRaisesRegex(PermitError, "not an intact"):
            permit.check_selftest(out)
        cases = {"passed": ({"passed": False}, "did not pass cleanly"), "failures": ({"failures": 1}, "did not pass cleanly"),
                 "skipped": ({"skipped": 1}, "did not pass cleanly"), "count": ({"tests_run": 1}, "test counts differ"),
                 "modules": ({"modules": {self.module: 5}}, "test counts differ"),
                 "source": ({"version": {**record["version"], "exploration_source_sha256": "0" * 64}}, "differ from this version"),
                 "tests": ({"version": {**record["version"], "test_files": {self.module: "0" * 64}}}, "differ from this version"),
                 "minimal": (dict.fromkeys(set(record) - {"schema", "seal", "passed"}, None), "differ from this version")}
        for name, (change, message) in cases.items():
            with self.subTest(name):
                path.write_text(sealing.dump(sealing.with_seal({**record, **change})))
                with self.assertRaisesRegex(PermitError, message):
                    permit.check_selftest(out)
        path.write_text(sealing.dump(record))
        self.assertEqual(permit.check_selftest(out), record)

    def test_added_test_or_missing_module_changes_the_expected_counts(self):
        out = self.out / "run"
        permit.run_selftest(out)
        self.probe.write_text(self.PROBE_TEXT + "\n    def test_three(self):\n        self.fail('new test')\n")
        with self.assertRaisesRegex(PermitError, "differ from this version; recorded test counts differ"):
            permit.check_selftest(out)
        self.assertEqual(permit.count_tests(self.probe), 3)
        with patch.object(permit, "TEST_MODULES", ("kg_probe_missing_module",)):
            with self.assertRaisesRegex(PermitError, "test module kg_probe_missing_module is not found"):
                permit.check_selftest(out)

    def test_static_count_equals_the_loader_count_for_the_real_test_files(self):
        """허가 검사는 시험 파일을 실행하지 않고 시험 수를 센다. 그 수가 실제 수집 수와 같아야 한다."""
        real = ("tests.test_research_etf_kodex_grid", "tests.test_research_etf_kodex_core")
        self.assertEqual(permit.TEST_MODULES, (self.module,))
        for module in real:
            loaded = unittest.defaultTestLoader.loadTestsFromName(module).countTestCases()
            self.assertEqual(permit.count_tests(permit.test_file(module)), loaded)
        self.assertEqual(permit.count_tests(permit.test_file(real[1])), 94)


class FailingProbe(_Permit):
    PROBE_TEXT = PROBE + "\n    def test_broken(self):\n        self.assertEqual(1, 2)\n"

    def test_failed_run_is_recorded_as_failed_and_permits_nothing(self):
        out = self.out / "run"
        with self.assertRaisesRegex(PermitError, "the recorded run did not pass cleanly"):
            permit.run_selftest(out)
        stored = json.loads((out / "selftest/record.json").read_text())
        self.assertEqual((stored["tests_run"], stored["failures"], stored["passed"]), (3, 1, False))
        with self.assertRaisesRegex(PermitError, "did not pass cleanly"):
            permit.run_selftest(out)  # 실패 기록을 지우고 다시 쓰지 않는다(새 출력 폴더에서 다시 돌린다)


class SkippingProbe(_Permit):
    PROBE_TEXT = PROBE + "\n    @unittest.skip('not run')\n    def test_skipped(self):\n        pass\n"

    def test_skipped_tests_do_not_count_as_a_pass(self):
        with self.assertRaisesRegex(PermitError, "did not pass cleanly"):
            permit.run_selftest(self.out / "run")
        self.assertEqual(json.loads((self.out / "run/selftest/record.json").read_text())["skipped"], 1)


class ProgressAndRemainingTime(_Workspace):
    """남은 시간은 진행 로그의 기술적 완료시간만으로 다시 잡는다(조건부 추정)."""

    def log(self, lines):
        path = self.out / "stage1/progress.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"2026-11-24T00:00:00Z stage1 {line}\n" for line in lines))
        return path

    def test_remaining_time_from_technical_completion_times(self):
        path = self.log(["setting-01 done 600.0s", "setting-02 done 1200.0s", "setting-03 error 3.0s E_ENGINE",
                         "setting-04 skipped 0.0s", "setting-03 done 900.0s", "setting-05 error 1.0s E_OS"])
        eta = progress.remaining(path, planned=36, workers=6)
        self.assertEqual((eta["finished"], eta["remaining"], eta["in_error"], eta["measured_units"]), (4, 32, ["setting-05"], 3))
        self.assertEqual((eta["mean_done_seconds"], eta["max_done_seconds"]), (900.0, 1200.0))
        self.assertEqual(eta["remaining_hours"], 6 * 900.0 / 3600)  # ceil(32 ÷ 6) = 6묶음 × 평균 900초
        self.assertIn("상한이 아니다", eta["assumption"])
        self.assertIsNone(progress.remaining(self.log(["setting-04 skipped 0.0s"]), planned=36, workers=6)["remaining_hours"])

    def test_log_lines_and_refusal_of_non_technical_lines(self):
        lines = []
        log = progress.Progress(self.out / "s", "stage1", lines.append)
        log("setting-01", "done", 12.34)
        log("setting-02", "E_ENGINE", 0.5)
        log("setting-03", "skipped", 0.0)
        self.assertEqual([line.split(" ", 1)[1] for line in lines],
                         ["stage1 setting-01 done 12.3s", "stage1 setting-02 error 0.5s E_ENGINE", "stage1 setting-03 skipped 0.0s"])
        self.assertTrue(all(LOG_LINE.match(line) for line in lines))
        self.assertEqual([row["status"] for row in progress.read_log(self.out / "s/progress.log")], ["done", "error", "skipped"])
        for bad in ("setting-01 done 600.0s delta=0.01", "setting-01 pass 600.0s", "setting-01 error 1.0s ValueError: x"):
            with self.assertRaisesRegex(ValueError, "not a technical progress record"):
                progress.read_log(self.log([bad]))

    def test_eta_command_reads_only_the_progress_log(self):
        self.log([f"setting-{n:02d} done 1000.0s" for n in range(1, 13)])
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(cli.main(["eta", "--out", str(self.out), "--workers", "6"]), 0)
        eta = json.loads(stream.getvalue())
        self.assertEqual((eta["planned"], eta["finished"], eta["remaining"], eta["remaining_hours"]),
                         (36, 12, 24, 4 * 1000.0 / 3600))


if __name__ == "__main__":
    unittest.main()
