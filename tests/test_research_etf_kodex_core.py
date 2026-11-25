"""KODEX 코어 80% + 분할 20% 연결부 합성 시험 (리뷰 ①~⑥ + R2 대조).

기대 수량·날짜·판정은 실행 전에 손으로 계산해 적었다(IMPL-REPORT.md 참조).
가격은 2025년 국내 호가(2,000원 이상 5원)를 쓰고 수수료는 기본 0.015%, 스트레스 0.30%.
"""
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, localcontext
import math
import unittest

from src.research.etf.baseline_policy import VolatilityEstimate
from src.research.etf.engine import run_engine
from src.research.etf.engine_types import Bar, Instrument, LevelRule, ONE
from src.research.etf.kodex_core import check
from src.research.etf.kodex_core.calendar import KrCalendar, coverage_gaps
from src.research.etf.kodex_core.config import load_config
from src.research.etf.kodex_core.inputs import (ENTITLEMENT_BASIS, PROVIDER_BASIS, Distribution,
                                                VolatilityUnavailable, prepare)
from src.research.etf.kodex_core.ledgers import RunSetting, run_hold, run_split
from src.research.etf.kodex_core.study import _start_liquidity, run_row, run_start, run_symbol
from src.research.etf.kodex_core.verdict import (across_starts, candidates, extension_counts, item_status,
                                                 path_verdict)


D = Decimal
UTC = timezone.utc
SYMBOL = "069500"
BASE = D("0.00015")
STRESS = D("0.003")


def at(day, hour=0, minute=0):
    return datetime.combine(day, time(hour, minute), UTC)


def weekdays(start, end, holidays=()):
    days, day = [], start
    while day <= end:
        if day.weekday() < 5 and day not in holidays:
            days.append(day)
        day += timedelta(days=1)
    return days


def make_bar(day, price, **changes):
    price = D(price)
    values = dict(symbol=SYMBOL, session_date=day, open_at=at(day), close_at=at(day, 6, 30), open=price,
                  high=price, low=price, close=price, volume=D(1000000), adv20=D("10000000000000"))
    return Bar(**{**values, **changes})


def instrument(basis="raw"):
    return Instrument(SYMBOL, "KR", "KRW", "국내주식형", basis)


def build(days, prices, *, tail=2, sigma="0.01", changes=None, basis="raw", distributions=(), extra_vol=None):
    """days: 일봉이 있는 거래일, tail: 결제용 뒤쪽 세션 수. prices: day -> 가격 또는 None(직전 유지)."""
    calendar_days = list(days)
    while len(calendar_days) < len(days) + tail:
        calendar_days += weekdays(calendar_days[-1] + timedelta(days=1), calendar_days[-1] + timedelta(days=3))[:1]
    calendar = KrCalendar({d: (at(d), at(d, 6, 30)) for d in calendar_days})
    bars, price = [], None
    for day in days:
        price = prices.get(day, price)
        bars.append(make_bar(day, price, **(changes or {}).get(day, {})))
    volatility = {days[0]: VolatilityEstimate(D(sigma), at(days[0]) - timedelta(hours=12))}
    volatility.update(extra_vol or {})
    return prepare(instrument(basis), bars, calendar, volatility, distributions=distributions)


CONFIG = load_config().with_changes(volatility_mode="fixed", start_offsets=(0,))


def setting(mode="none", path="O-H-L-C", commission=BASE, config=CONFIG):
    return RunSetting(config, path, commission, mode)


def trades(run):
    return [(r["session_date"], r["side"], r["level"], r["quantity"], r["price"]) for r in run.result.trades]


J = lambda d: date(2025, 1, d)  # noqa: E731


class FirstQuantityCase1(unittest.TestCase):
    """① 단독 보유·코어·분할 첫 매수 수량과 잔돈 (시가 10,000원)."""

    def setUp(self):
        self.data = build(weekdays(J(2), J(6)), {J(2): "10000"})

    def test_three_ledgers_first_fill_and_change(self):
        hold = run_hold(self.data, setting(), D(100000000))
        core = run_hold(self.data, setting(), D(80000000))
        split = run_split(self.data, setting())
        expected = {"hold": (hold, 9998, D("14997"), D("5003")), "core": (core, 7998, D("11997"), D("8003")),
                    "split": (split, 222, D("333"), D("17779667"))}
        for name, (run, quantity, fee, cash) in expected.items():
            with self.subTest(name):
                first = run.result.trades[0]
                self.assertEqual((first["session_date"], first["quantity"], first["price"], first["fee"]),
                                 (J(2), quantity, D(10000), fee))
                self.assertEqual(run.result.final_cash["KRW"], cash)
        self.assertNotEqual(core.result.final_cash["KRW"], hold.result.final_cash["KRW"] * D("0.8"))

    def test_independent_checker_agrees(self):
        for budget, quantity, change in (("100000000", 9998, 5003), ("80000000", 7998, 8003),
                                         ("20000000/9", 222, None)):
            result = check.first_quantity(budget, "10000", "0.00015")
            self.assertEqual(result["quantity"], quantity)
            if change is not None:
                self.assertEqual(result["change"], change)

    def test_combined_start_liquidity_boundary(self):
        core = run_hold(self.data, setting(), D(80000000))
        split = run_split(self.data, setting())
        for adv20, allowed in ((D(1644000000), True), (D(1643999980), False)):
            bars = (replace(self.data.bars[0], adv20=adv20),) + self.data.bars[1:]
            self.assertEqual(_start_liquidity(replace(self.data, bars=bars), CONFIG, core, split), allowed)

    def test_hold_first_buy_rejection_holds_comparison(self):
        bars = [replace(self.data.bars[0], adv20=D(1644000000))] + list(self.data.bars[1:])
        data = prepare(instrument(), bars, self.data.calendar, self.data.volatility)
        result = run_start(data, CONFIG, data.start)
        for mode in ("M", "D"):
            path = result["items"][mode]["paths"]["O-H-L-C"]["verdict"]
            self.assertEqual(path["status"], "hold")
            self.assertIn("base_hold_first_buy_failed", path["reasons"])
        self.assertEqual(result["candidates"]["label"], "held")


HOLIDAYS = {J(27), J(28), J(29), J(30), date(2025, 3, 3)}
CASE2_DAYS = weekdays(J(2), date(2025, 3, 6), HOLIDAYS)


def case2(**options):
    changes = {J(31): {"adv20": D(1)}, date(2025, 2, 3): {"volume": D(0)}}
    return build(CASE2_DAYS, {J(2): "10000", J(24): "10400"}, changes=changes, **options)


class ReentryScheduleCase2(unittest.TestCase):
    """② 1/24 전량 매도 → 1/27~30 휴장 → 1/31 거절(참여율) → 2/3 거래량 0 → 3/3 휴장."""

    def test_d_and_m_next_order_dates(self):
        data = case2()
        expected = {
            "D": ([(J(2), "BUY", 1, 222, D(10000)), (J(24), "SELL", 1, 222, D(10400)),
                   (date(2025, 2, 4), "BUY", 1, 214, D(10400))],
                  [(J(31), "participation_limit"), (date(2025, 2, 3), "no_order"), (date(2025, 2, 4), "filled")]),
            "M": ([(J(2), "BUY", 1, 222, D(10000)), (J(24), "SELL", 1, 222, D(10400)),
                   (date(2025, 3, 4), "BUY", 1, 214, D(10400))],
                  [(date(2025, 2, 3), "no_order"), (date(2025, 3, 4), "filled")]),
        }
        for path in CONFIG.paths:
            for mode, (fills, attempts) in expected.items():
                with self.subTest(mode=mode, path=path):
                    run = run_split(data, setting(mode, path))
                    self.assertEqual(trades(run), fills)
                    self.assertEqual([(a["date"], a["outcome"]) for a in run.detail["reentries"]], attempts)

    def test_r2_once_only_never_reenters(self):
        run = run_split(case2(), setting("none"))
        self.assertEqual(len(run.result.trades), 2)

    def test_initial_split_rejection_is_start_failure_without_reentry(self):
        data = build(weekdays(J(2), J(10)), {J(2): "3000000", J(3): "3100000"})
        run = run_split(data, setting("D"))
        self.assertEqual((run.status, run.result.trades, run.detail["reentries"]), ("start_failed", [], []))
        result = run_start(data, CONFIG, data.start)
        for mode in ("M", "D"):
            self.assertEqual(result["items"][mode]["status"], "fail")
            self.assertIn("base_split_start_failed", result["items"][mode]["paths"]["O-H-L-C"]["verdict"]["reasons"])


class TakeProfitCase3(unittest.TestCase):
    """③ 비용 전 상승률만 3%(문턱) 도달한 날은 매도 없음 / 여러 차수 동시 익절 가능일의 순서."""

    def test_gross_only_crossing_does_not_sell(self):
        data = build(weekdays(J(2), J(8)), {J(2): "10000", J(3): "10300", J(6): "10305"})
        for path in CONFIG.paths:
            run = run_split(data, setting("none", path))
            self.assertEqual(trades(run), [(J(2), "BUY", 1, 222, D(10000)), (J(6), "SELL", 1, 222, D(10305))])

    def test_multiple_take_profits_one_per_day_in_hash_order(self):
        data = build(weekdays(J(2), J(10)), {J(2): "10000", J(3): "9800", J(6): "9600", J(7): "10500"})
        expected = [(J(2), "BUY", 1, 222, D(10000)), (J(3), "BUY", 2, 226, D(9800)),
                    (J(6), "BUY", 3, 231, D(9600)), (J(7), "SELL", 2, 226, D(10500)),
                    (J(8), "SELL", 1, 222, D(10500)), (J(9), "SELL", 3, 231, D(10500))]
        for path in CONFIG.paths:
            self.assertEqual(trades(run_split(data, setting("none", path))), expected)


class DistributionCase4(unittest.TestCase):
    """④ 1/6 배당락(주당 100원) 보유 → 1/7 분할 전량 매도 → 1/10 지급일 분할은 현금."""

    def data(self, basis="raw"):
        event = Distribution(SYMBOL, J(6), D(100), at(J(10)), "synthetic")
        return build(weekdays(J(2), J(15)), {J(2): "10000", J(7): "10400"}, basis=basis, distributions=[event])

    def test_split_receives_entitlement_and_keeps_cash_until_schedule(self):
        run = run_split(self.data(), setting("M"))
        self.assertEqual(trades(run), [(J(2), "BUY", 1, 222, D(10000)), (J(7), "SELL", 1, 222, D(10400))])
        events = {row["kind"]: row for row in run.result.cash_events if row["kind"].startswith("dividend")}
        self.assertEqual(events["dividend_receivable"]["gross"], D(22200))
        self.assertEqual(events["dividend_payment"]["amount"], D(22200))
        self.assertEqual(run.result.final_cash["KRW"], D("20110320.68"))
        self.assertEqual(run.detail["reentries"], [])

    def test_core_and_hold_reinvest_next_session_after_payment(self):
        for capital, quantity, cash in ((D(80000000), 77, D("6882.88")), (D(100000000), 96, D("6253.24"))):
            run = run_hold(self.data(), setting(), capital)
            self.assertEqual(trades(run)[-1], (J(13), "BUY", 2, quantity, D(10400)))
            self.assertEqual(run.result.trades[-1]["reason"], "dividend_reinvest")
            self.assertEqual(run.result.final_cash["KRW"], cash)

    def test_provider_mode_adds_no_cash_distribution(self):
        data = build(weekdays(J(2), J(15)), {J(2): "10000", J(7): "10400"}, basis="naver_provider_adjusted_distribution_unverified")
        for run in (run_split(data, setting("M")), run_hold(data, setting(), D(80000000))):
            self.assertFalse([r for r in run.result.cash_events if r["kind"].startswith("dividend")])
        self.assertEqual(run_split(data, setting("M")).result.final_cash["KRW"], D("20088120.68"))

    def test_basis_labels_and_adjusted_price_guard(self):
        self.assertEqual(self.data().basis, ENTITLEMENT_BASIS)
        self.assertEqual(build(weekdays(J(2), J(6)), {J(2): "10000"}).basis, ENTITLEMENT_BASIS)  # 원가격이면 사건 0건도 원가격
        provider = build(weekdays(J(2), J(6)), {J(2): "10000"}, basis="naver_provider_adjusted_distribution_unverified")
        self.assertEqual(provider.basis, PROVIDER_BASIS)
        with self.assertRaises(ValueError):
            self.data("naver_provider_adjusted_distribution_unverified")


class AggregationCase5(unittest.TestCase):
    """⑤ 국내 휴장 제외·월초 예산·일별 합산·네 문턱 등호."""

    def test_domestic_sessions_only_and_monthly_budget(self):
        data = case2()
        run = run_split(data, setting("D"))
        dates = [row["at"].astimezone(timezone(timedelta(hours=9))).date() for row in run.result.valuations]
        self.assertEqual(dates, CASE2_DAYS)
        self.assertFalse(HOLIDAYS & set(dates))
        refresh = next(r for r in run.result.cash_events if r["kind"] == "budget_refresh"
                       and r["at"] == at(date(2025, 2, 3)))
        jan31 = next(r for r in run.result.valuations if r["at"] == datetime(2025, 1, 31, 9, 0, tzinfo=UTC))
        self.assertEqual(refresh["source_total_krw"], jan31["equity_krw"])
        self.assertEqual(jan31["equity_krw"], D("20088120.68"))
        with localcontext(prec=80):
            self.assertEqual(refresh["budgets"][SYMBOL], D("20088120.68") * (ONE / 9))

    def test_daily_sum_matches_independent_checker(self):
        result = run_start(case2(), CONFIG, CASE2_DAYS[0], keep_ledgers=True)
        for mode in ("M", "D"):
            for path, value in result["items"][mode]["paths"].items():
                for cost in ("base", "stress"):
                    ledgers = value["ledgers"][cost]
                    self.assertEqual(check.compare_daily(ledgers["core"], ledgers[mode]), [])
                exposures = check.daily_sum(value["ledgers"]["base"]["core"], value["ledgers"]["base"][mode])
                mean = sum(row["exposure"] for row in exposures) / len(exposures)
                self.assertAlmostEqual(float(mean), float(value["base"]["strategy"]["exposure_mean"]), places=12)

    def test_threshold_equalities(self):
        t = CONFIG.thresholds
        base = {"exposure_mean": D("0.80"), "delta_cagr": D("0.01"), "delta_abs_mdd": D("0.05")}
        exact = path_verdict(base, {"delta_cagr": D("0.0000001")}, t)
        self.assertEqual(exact["status"], "pass")
        self.assertEqual([exact["margins"][k] for k in ("exposure", "cagr_edge", "mdd_excess")], [0, 0, 0])
        failed = path_verdict(base, {"delta_cagr": D(0)}, t)
        self.assertEqual((failed["status"], failed["reasons"]), ("fail", ["stress_cagr_edge"]))
        for name, value in (("exposure_mean", D("0.7999999")), ("delta_cagr", D("0.0099999")),
                            ("delta_abs_mdd", D("0.0500001"))):
            self.assertEqual(path_verdict({**base, name: value}, {"delta_cagr": D(1)}, t)["status"], "fail")
        independent = check.inequalities({**base, "stress_delta_cagr": D(0)}, {k: str(v) for k, v in t.items()})
        self.assertEqual(independent, {"exposure": True, "cagr_edge": True, "mdd_excess": True,
                                       "stress_cagr_edge": False})

    def test_candidate_display_rules(self):
        self.assertEqual(candidates({"M": "pass", "D": "pass"})["passed"], ["M", "D"])
        self.assertEqual(candidates({"D": "pass", "M": "pass"})["label"], "joint")
        self.assertEqual(candidates({"M": "fail", "D": "pass"})["all"], {"M": "fail", "D": "pass"})
        self.assertEqual(candidates({"M": "fail", "D": "hold"})["label"], "held")
        self.assertEqual(item_status(["pass", "hold"]), "hold")
        self.assertEqual(item_status(["pass", "fail"]), "fail")
        self.assertEqual(candidates({"M": "candidate", "D": "fail"}, "candidate")["passed"], ["M"])


class StressStartCase6(unittest.TestCase):
    """⑥ 변동성 0.2%: 2σ=0.4%. 기본 왕복 0.03% 통과, 스트레스 왕복 0.6% 거부 → 미달."""

    def test_stress_rejection_is_failure_not_dropped(self):
        data = build(weekdays(J(2), J(10)), {J(2): "10000"}, sigma="0.002")
        self.assertEqual(run_split(data, setting("M")).status, "ok")
        self.assertEqual(run_split(data, setting("M", commission=STRESS)).status, "cost_guard_start")
        result = run_start(data, CONFIG, data.start)
        for mode in ("M", "D"):
            item = result["items"][mode]
            self.assertEqual(item["status"], "fail")
            for value in item["paths"].values():
                self.assertEqual(value["verdict"]["reasons"], ["stress_start_rejected_by_cost_guard"])
        self.assertEqual(result["candidates"]["label"], "closed")


class AnnualVolatility(unittest.TestCase):
    """v2 2절: 연초 첫 거래일에 새 변동성. 보유 차수 익절 기준은 매수 당시 값 유지."""

    def test_refresh_applies_to_new_orders_only(self):
        days = weekdays(date(2024, 12, 2), J(10), {date(2024, 12, 25), date(2024, 12, 31), J(1)})
        extra = {J(2): VolatilityEstimate(D("0.02"), at(J(2)) - timedelta(hours=12))}
        data = build(days, {days[0]: "10000", J(3): "10305"}, extra_vol=extra)
        run = run_split(data, setting("D", config=CONFIG.with_changes(volatility_mode="annual")))
        self.assertEqual(trades(run), [(days[0], "BUY", 1, 222, D(10000)), (J(3), "SELL", 1, 222, D(10305)),
                                       (J(6), "BUY", 1, 215, D(10305))])
        self.assertEqual(run.detail["refreshes"], [{"date": J(2), "applied": True, "volatility": D("0.02")}])
        self.assertEqual(run.result.positions[0].take_rate, D("0.06"))


class R2Parity(unittest.TestCase):
    """재진입 없음 설정의 분할 장부 = 같은 입력의 R2 run_engine 단독 실행."""

    def data(self):
        days = weekdays(J(2), date(2025, 3, 14), HOLIDAYS)
        bars = []
        for index, day in enumerate(days):
            mid = 10000 * (1 + 0.08 * (math.cos(index / 3) - 1))
            o, c = round(mid / 5) * 5, round(mid * (1 + 0.006 * math.cos(index)) / 5) * 5
            bars.append(make_bar(day, o, close=D(c), high=D(max(o, c) + 150), low=D(min(o, c) - 150)))
        calendar = KrCalendar({d: (at(d), at(d, 6, 30)) for d in days + [date(2025, 3, 17), date(2025, 3, 18)]})
        return prepare(instrument(), bars, calendar, {days[0]: VolatilityEstimate(D("0.01"), at(days[0]) - timedelta(hours=1))})

    def test_same_fills_as_r2_engine(self):
        data = self.data()
        for path in CONFIG.paths:
            mine = run_split(data, setting("none", path))
            drop, take = mine.detail["rates"]
            times = data.calendar.evaluation_times(data.start, data.end)
            with localcontext(prec=80):
                fraction = ONE / 9
            r2 = run_engine([data.instrument], list(data.bars),
                            {SYMBOL: tuple(LevelRule(n, ONE, drop, take) for n in range(1, 10))},
                            initial_cash={"KRW": D(20000000)}, evaluation_times=times, path=path,
                            first_level_fraction=fraction, commission_rate=BASE,
                            kr_sessions=list(data.calendar.sessions),
                            kr_settlement_session_times=data.calendar.settlement_times(),
                            domestic_settlement="t_plus_2", _evaluation_cutoff=times[-1])
            with self.subTest(path=path):
                self.assertGreater(len(r2.trades), 4)
                self.assertEqual(mine.result.trades, r2.trades)
                self.assertEqual(mine.result.rejections, r2.rejections)
                self.assertEqual(mine.result.valuations, r2.valuations)


class StudyAssembly(unittest.TestCase):
    def test_start_rows_segments_and_matched_baseline(self):
        data = case2()
        config = CONFIG.with_changes(start_offsets=(0, 5))
        result = run_symbol(data, config, [CASE2_DAYS[0], CASE2_DAYS[5]])
        self.assertEqual(list(result["starts"]), [CASE2_DAYS[0], CASE2_DAYS[5]])
        late = result["starts"][CASE2_DAYS[5]]["items"]["M"]["paths"]["O-H-L-C"]
        self.assertEqual(late["base"]["strategy"]["segments"][0]["segment"], "2024-2026-11-13")
        self.assertIn("mdd_from_inception_peak", late["base"]["strategy"]["segments"][0])
        self.assertIn("matched_baseline", late)
        self.assertEqual(result["starts"][CASE2_DAYS[0]]["basis"]["label"], CONFIG.labels["basis_entitlement"])  # 원가격 입력
        with self.assertRaises(ValueError):
            run_symbol(data, config, [CASE2_DAYS[0], CASE2_DAYS[6]])

    def test_mixed_start_labels(self):
        self.assertEqual(across_starts({1: "pass", 2: "fail", 3: "pass"})["label"], "start_sensitive_unresolved")
        self.assertEqual(across_starts({1: "pass", 2: "hold", 3: "pass"})["label"], "incomplete")
        self.assertEqual(across_starts({1: "pass", 2: "pass", 3: "pass"})["label"], "candidate")
        self.assertEqual(across_starts({1: "fail", 2: "fail", 3: "fail"})["label"], "fail")
        counts = extension_counts({"229200": {"M": "candidate", "D": "fail"},
                                   "169950": {"M": "incomplete", "D": "candidate"}}, 2)
        self.assertEqual(counts["by_mode"]["M"], {"candidate": 1, "fail": 0, "start_sensitive_unresolved": 0,
                                                  "incomplete": 1, "denominator": 2, "decided": 1,
                                                  "decided_passed": 1, "undecided": 1})
        self.assertEqual((counts["etf_candidate_count"], counts["note"]), (2, None))
        with self.assertRaises(ValueError):
            extension_counts({"229200": {"M": "fail"}}, 2)

    def test_unloadable_symbol_stays_in_denominator(self):
        from src.research.etf.kodex_core.config import StudyRow
        row = StudyRow("extension", "extension", ("A", "B"), (CASE2_DAYS[0],), CONFIG.segments, 2)

        def load(symbol, _row, _start):
            raise DataHold("calendar/history insufficient")

        result = run_row(row, load, CONFIG.with_changes(start_offsets=(0,)))
        self.assertEqual(result["counts"]["by_mode"]["D"]["incomplete"], 2)
        self.assertEqual(result["symbols"]["A"]["held_starts"][CASE2_DAYS[0]], "calendar/history insufficient")


DEC = lambda d: date(2024, 12, d)  # noqa: E731
YEAR_END = weekdays(DEC(2), J(10), {DEC(25), DEC(31), J(1)})


class YearBoundaryAndCostGuard(unittest.TestCase):
    """연초 첫 거래일 = 월초 예산 갱신 = 변동성 갱신일에 기존 차수 익절 가능성과 새 차수 매수가 겹친다."""

    def data(self, refreshed="0.005"):
        estimate = {J(2): VolatilityEstimate(D(refreshed), at(J(2)) - timedelta(hours=12))}
        changes = {J(2): {"open": D(9700), "high": D(10150), "low": D(9700), "close": D(9800)}}
        return build(YEAR_END, {DEC(2): "10000", DEC(3): "9800", J(3): "10150"}, changes=changes, extra_vol=estimate)

    def annual(self, mode="none", path="O-H-L-C", commission=BASE):
        return setting(mode, path, commission, CONFIG.with_changes(volatility_mode="annual"))

    def test_one_order_per_day_and_rates_by_purchase_year(self):
        for path in CONFIG.paths:
            run = run_split(self.data(), self.annual(path=path))
            self.assertEqual(trades(run), [(DEC(2), "BUY", 1, 222, D(10000)), (DEC(3), "BUY", 2, 226, D(9800)),
                                           (J(2), "BUY", 3, 228, D(9700)), (J(3), "SELL", 2, 226, D(10150)),
                                           (J(6), "SELL", 3, 228, D(10150))])
            self.assertEqual([p.take_rate for p in run.result.positions], [D("0.03")])
            for stress in (run_split(self.data(), self.annual(path=path, commission=STRESS)),):
                days = [row["session_date"] for row in stress.result.trades]
                self.assertEqual(len(days), len(set(days)))
                self.assertEqual(trades(stress)[2][:3], (J(2), "BUY", 3))

    def test_refresh_below_cost_closes_new_buys_but_keeps_take_profit(self):
        data = self.data(refreshed="0.0001")
        run = run_split(data, self.annual("D"))
        self.assertEqual(trades(run)[:2], [(DEC(2), "BUY", 1, 222, D(10000)), (DEC(3), "BUY", 2, 226, D(9800))])
        # 신규 매수가 닫혀 시가 3차 매수가 없으므로 같은 날 장중 2차 익절(3% 문턱 → 10,100원 지정가)이 체결된다.
        self.assertEqual(trades(run)[2:], [(J(2), "SELL", 2, 226, D(10100))])
        self.assertTrue(run.detail["refreshes"][0]["buys_closed"])
        result = run_start(data, CONFIG.with_changes(volatility_mode="annual"), data.start)
        self.assertEqual(result["items"]["D"]["paths"]["O-H-L-C"]["buys_closed_years"]["base"], [J(2)])

    def test_closed_buys_block_reentry(self):
        estimate = {J(2): VolatilityEstimate(D("0.0001"), at(J(2)) - timedelta(hours=12))}
        data = build(YEAR_END, {DEC(2): "10000", DEC(3): "10400"}, extra_vol=estimate)
        run = run_split(data, self.annual("D"))
        self.assertEqual(trades(run), [(DEC(2), "BUY", 1, 222, D(10000)), (DEC(3), "SELL", 1, 222, D(10400)),
                                       (DEC(4), "BUY", 1, 213, D(10400))])
        flat = build(YEAR_END, {DEC(2): "10000", J(3): "10400"}, extra_vol=estimate)
        run = run_split(flat, self.annual("D"))
        self.assertEqual(trades(run)[1:], [(J(3), "SELL", 1, 222, D(10400))])
        self.assertTrue(all(a["outcome"].startswith("cost_guard") for a in run.detail["reentries"]))
        self.assertEqual([a["date"] for a in run.detail["reentries"]], [J(6), J(7), J(8), J(9), J(10)][:len(run.detail["reentries"])])

    def test_insufficient_valid_returns_is_data_hold(self):
        from src.research.etf.kodex_core.data import _volatility
        days = weekdays(date(2023, 12, 1), date(2025, 1, 10))
        times = {d: (at(d), at(d, 6, 30)) for d in days}
        rows = [{"date": d.isoformat(), "close": "10000", "volume": "0" if i % 3 == 0 else "100"}
                for i, d in enumerate(days)]
        value = _volatility(rows, times, [J(2)])[J(2)]
        self.assertIsInstance(value, VolatilityUnavailable)
        self.assertIn("200", value.reason)
        start = build(weekdays(J(2), J(10)), {J(2): "10000"})
        start = replace(start, volatility={J(2): value})
        self.assertEqual(run_split(start, setting("M")).status, "volatility_unavailable")
        verdict = run_start(start, CONFIG, J(2))["items"]["M"]["paths"]["O-H-L-C"]["verdict"]
        self.assertEqual((verdict["status"], verdict["reasons"][0]), ("hold", "base_volatility_unavailable"))
        later = replace(self.data(), volatility={**self.data().volatility, J(2): value})
        result = run_start(later, CONFIG.with_changes(volatility_mode="annual"), later.start)
        self.assertIn("base_annual_volatility_unavailable", result["items"]["M"]["paths"]["O-H-L-C"]["verdict"]["reasons"])


class InputGuards(unittest.TestCase):
    def test_missing_session_and_coverage(self):
        days = weekdays(J(2), J(10))
        data = build(days, {J(2): "10000"})
        with self.assertRaises(ValueError):
            prepare(instrument(), data.bars[:2] + data.bars[3:], data.calendar, data.volatility)
        late = KrCalendar({d: (at(d), at(d, 6, 30)) for d in weekdays(date(2026, 11, 16), date(2026, 11, 20))})
        self.assertTrue(coverage_gaps(late, date(2026, 11, 19)))
        # 만기 11/16의 T+2(11/18)가 호가표 끝(11/17) 밖이면 보류 사유(리뷰 권고 6).
        self.assertTrue(coverage_gaps(late, date(2026, 11, 16)))
        fixed = KrCalendar({d: (at(d), at(d, 6, 30)) for d in weekdays(date(2026, 11, 12), date(2026, 11, 20))})
        self.assertEqual(coverage_gaps(fixed, date(2026, 11, 13)), [])
        self.assertTrue(coverage_gaps(late, date(2026, 11, 16), start=date(2010, 12, 30)))

    def test_loader_requires_twenty_prior_sessions_not_three_years(self):
        from src.research.etf.kodex_core.data import _bars
        days = weekdays(date(2024, 11, 1), J(10))
        times = {d: (at(d), at(d, 6, 30)) for d in days}
        dated = {d: {"open": "10000", "high": "10100", "low": "9900", "close": "10000", "volume": "5"} for d in days}
        bars, _ = _bars(instrument(), dated, times, days[20], days[-1], {})
        self.assertEqual((bars[0].session_date, bars[0].adv20), (days[20], D(50000)))
        with self.assertRaises(ValueError):
            _bars(instrument(), dated, times, days[19], days[-1], {})



# ---------------------------------------------------------------------------
# 구현 리뷰 1차(review-reconciled-impl.md) A1~A8, C1 회귀 시험.
# 입력은 astra-impl-check/reproduce.py의 합성 사례를 옮긴 것이다.
# ---------------------------------------------------------------------------
from unittest.mock import patch  # noqa: E402

from src.research.etf.kodex_core import data as loader  # noqa: E402
from src.research.etf.kodex_core.config import StudyRow  # noqa: E402
from src.research.etf.kodex_core.inputs import ZeroVolatility, starting_at  # noqa: E402
from src.research.etf.kodex_core.study import InputIdentityError, run_plan, versus_primary_hold  # noqa: E402

BASIS = "naver_provider_adjusted_distribution_unverified"
PATH = CONFIG.paths[0]
FROZEN = load_config()  # 고정 파일 그대로(annual, 오프셋 0/5/20)
TARGET = "169950"


def as_symbol(template, symbol):
    return replace(template, instrument=replace(template.instrument, symbol=symbol),
                   bars=tuple(replace(b, symbol=symbol) for b in template.bars))


def history_fixture(end):
    """2023-12-01부터 평일 달력(1/1 휴장), 종가 10,000/10,100 교대, 거래량 300,000."""
    sessions = [d for d in weekdays(date(2023, 12, 1), end + timedelta(days=5)) if d != J(1)]
    times = {d: (at(d), at(d, 6, 30)) for d in sessions}
    rows = []
    for i, day in enumerate(sessions):
        if day <= end:
            close = 10000 + (i % 2) * 100
            rows.append({"date": day.isoformat(), "open": str(close), "close": str(close), "high": str(close + 5),
                         "low": str(close - 5), "volume": "300000", "price_basis": BASIS, "corp_action_ok": "True"})
    return rows, times


def fake_source(rows, times, symbol="069500"):
    metadata = {("KR", symbol): {"d1_eligible": "True"}}
    return patch.object(loader, "_source", return_value=({}, metadata, tuple(rows), times, {}))


class ReviewA1NominalCapacity(unittest.TestCase):
    """A1: 확장 행 각 시작일의 '20일 평균 거래대금×5% ≥ 1억'(등호 허용)을 독립 적용."""

    DAYS = weekdays(date(2017, 1, 2), date(2017, 2, 7), {date(2017, 1, 27), date(2017, 1, 30)})

    def run_with_limit(self, limit):
        days = self.DAYS
        starts = tuple(days[i] for i in (0, 5, 20))
        row = next(r for r in CONFIG.rows if r.id == "extension")
        self.assertEqual(starts, row.starts)

        def load(symbol, _row, start):
            changes = {starts[2]: {"adv20": limit / D(".05")}} if symbol == TARGET else {}
            data = build(days, {days[0]: "10000", days[-1]: "9700"}, changes=changes, basis=BASIS)
            return starting_at(as_symbol(data, symbol), start)

        return starts, run_row(row, load, FROZEN.with_changes(end=days[-1]))

    def test_floor_equality_passes_and_one_won_below_holds_only_that_start(self):
        starts, equal = self.run_with_limit(D(100000000))
        self.assertEqual(equal["symbols"][TARGET]["candidates"]["label"], "joint")
        self.assertEqual(equal["counts"]["etf_candidate_count"], 2)
        starts, below = self.run_with_limit(D(99990000))
        target = below["symbols"][TARGET]
        for mode in CONFIG.modes:
            self.assertEqual(target["labels"][mode]["by_start"], {starts[0]: "pass", starts[1]: "pass", starts[2]: "hold"})
            self.assertEqual(target["labels"][mode]["label"], "incomplete")
        late = target["starts"][starts[2]]
        self.assertEqual(late["capacity"], {"limit_krw": D(99990000), "floor_krw": D(100000000), "ok": False})
        self.assertEqual(late["items"]["M"]["paths"][PATH]["verdict"]["reasons"], ["start_capacity_below_floor"])
        self.assertIsNone(late["items"]["M"]["paths"][PATH]["base"])
        self.assertEqual((below["counts"]["etf_candidate_count"], below["counts"]["etf_candidates"]), (1, ["229200"]))
        orders = target["starts"][starts[0]]["items"]["M"]["paths"][PATH]["start_orders"]["base"]
        self.assertEqual((orders["core_plus_split"], orders["hold"]), (D(82200000), D(99980000)))
        self.assertEqual({k: v["status"] for k, v in orders["orders"].items()},
                         {"core": "filled", "split": "filled", "hold": "filled"})
        late_orders = late["items"]["M"]["paths"][PATH]["start_orders"]["base"]
        self.assertEqual({k: v["status"] for k, v in late_orders["orders"].items()},
                         {"core": "not_submitted", "split": "not_submitted", "hold": "not_submitted"})
        self.assertEqual(late_orders["orders"]["core"]["reason"], "start_capacity_below_floor")

    def test_primary_row_has_no_nominal_floor(self):
        data = build(weekdays(J(2), J(10)), {J(2): "10000"}, changes={J(2): {"adv20": D(1999800000)}})
        result = run_start(data, CONFIG, J(2))
        self.assertEqual(result["capacity"], {"limit_krw": D(99990000), "floor_krw": None, "ok": True})
        self.assertNotIn("start_capacity_below_floor", result["items"]["M"]["paths"][PATH]["verdict"]["reasons"])


class ReviewA2VolatilityInput(unittest.TestCase):
    """A2: 251일 전부 유한·양의 종가·0 이상 거래량 확인. 정상 0은 비용 조건, 결측·불량은 자료 보류."""

    def history(self):
        rows, times = history_fixture(J(10))
        past = [r for r in rows if r["date"] < J(2).isoformat()][-251:]
        return [dict(r) for r in past], times

    def test_valid_zero_volatility_is_cost_failure_not_data_hold(self):
        past, times = self.history()
        for row in past:
            row.update(close="10000", volume="1")
        value = loader._volatility(past, times, [J(2)])[J(2)]
        self.assertIsInstance(value, ZeroVolatility)
        data = replace(build(weekdays(J(2), J(8)), {J(2): "10000"}), volatility={J(2): value})
        verdict = run_start(data, CONFIG, J(2))["items"]["M"]["paths"][PATH]["verdict"]
        self.assertEqual(verdict, {"status": "fail", "checks": None, "later_holds": [],
                                   "reasons": ["base_start_rejected_by_cost_guard", "stress_start_rejected_by_cost_guard"]})

    def test_missing_close_or_negative_volume_is_held(self):
        for change in ({"close": "", "volume": "0"}, {"volume": "-1"}, {"close": "0"}, {"close": "nan"}):
            with self.subTest(change):
                past, times = self.history()
                past[22].update(change)
                value = loader._volatility(past, times, [J(2)])[J(2)]
                self.assertIsInstance(value, VolatilityUnavailable)

    def test_two_hundred_valid_returns_boundary_and_short_history(self):
        past, times = self.history()
        for omitted, expected in ((50, VolatilityEstimate), (51, VolatilityUnavailable)):
            rows = [dict(r, volume="0" if i < omitted else "300000") for i, r in enumerate(past)]
            self.assertIsInstance(loader._volatility(rows, times, [J(2)])[J(2)], expected)
        self.assertIsInstance(loader._volatility(past[1:], times, [J(2)])[J(2)], VolatilityUnavailable)

    def test_annual_zero_volatility_closes_new_buys(self):
        zero = ZeroVolatility(at(J(2)) - timedelta(hours=12))
        data = replace(YearBoundaryAndCostGuard().data(), volatility={YEAR_END[0]: YearBoundaryAndCostGuard().data().volatility[YEAR_END[0]], J(2): zero})
        run = run_split(data, setting("D", config=CONFIG.with_changes(volatility_mode="annual")))
        self.assertTrue(run.detail["refreshes"][0]["buys_closed"])
        self.assertFalse([t for t in run.result.trades if t["side"] == "BUY" and t["session_date"] >= J(2)])


class ReviewA3ObservationEnd(unittest.TestCase):
    """A3: 연초 변동성 결측이면 분할 계좌 관측 종료, 전년 값 거래 금지, 불성립 계좌의 비교 수치 없음."""

    def test_missing_annual_volatility_ends_observation(self):
        dec = date(2024, 12, 30)
        days = weekdays(dec, J(10), {date(2024, 12, 31), J(1)})
        data = build(days, {dec: "10000", J(2): "9700", J(3): "10500"},
                     extra_vol={J(2): VolatilityUnavailable("synthetic missing 251st close")}, basis=BASIS)
        run = run_split(data, setting("D", config=FROZEN))
        self.assertEqual(run.status, "observation_ended")
        # 결측 연초 직전(12/30)까지만 관측: 첫 주문만 있고 1/2 이후 거래·평가는 없다.
        self.assertEqual(trades(run), [(dec, "BUY", 1, 222, D(10000))])
        self.assertEqual(run.result.valuations[-1]["at"].date(), dec)
        self.assertEqual({k: run.detail[k] for k in ("last_valid_session", "missing_volatility_date", "reason")},
                         {"last_valid_session": dec, "missing_volatility_date": J(2),
                          "reason": "synthetic missing 251st close"})
        path = run_start(data, FROZEN, dec)["items"]["D"]["paths"][PATH]
        self.assertEqual(path["verdict"]["status"], "hold")
        self.assertIn("base_annual_volatility_unavailable", path["verdict"]["reasons"])
        self.assertEqual((path["base"], path["stress"]), (None, None))
        self.assertEqual(path["observation_end"]["base"]["last_valid_session"], dec)
        self.assertIsNotNone(path["hold_metrics"]["base"])  # 유효한 단독 보유 장부는 독립적으로 남는다.
        self.assertNotIn("matched_baseline", path)

    def test_failed_comparison_account_publishes_no_metrics_or_columns(self):
        days = weekdays(J(2), J(10))
        failed = build(days, {J(2): "10000", J(3): "9600"}, changes={J(2): {"adv20": D(100000000)}}, basis=BASIS)
        normal = build(days, {J(2): "10000", J(3): "9600"}, basis=BASIS)
        first, other = run_start(failed, CONFIG, J(2)), run_start(normal, CONFIG, J(2))
        row = first["items"]["M"]["paths"][PATH]
        self.assertEqual(row["verdict"]["status"], "hold")
        self.assertEqual((row["base"], row["hold_metrics"]["base"]), (None, None))
        cols = versus_primary_hold({"starts": {J(2): first}}, {"starts": {J(2): other}})
        self.assertEqual(cols, {})
        valid = versus_primary_hold({"starts": {J(2): other}}, {"starts": {J(2): first}})
        self.assertEqual(valid, {})  # 기준 KODEX 200 단독 보유가 불성립이면 보조 열도 없다.


class ReviewA4Identity(unittest.TestCase):
    """A4: 정식 진입점의 종목·시작일·만기·가격 기준 대조와 실행 식별값."""

    def plan(self, load, end):
        row = StudyRow("extension", "extension", ("229200", TARGET), (J(2),), CONFIG.segments, 2)
        return run_plan(load, CONFIG.with_changes(rows=(row,), start_offsets=(0,), end=end))

    def test_wrong_symbol_or_short_end_stops(self):
        days = weekdays(J(2), J(13))
        data = build(days, {J(2): "10000"}, basis=BASIS)
        with self.assertRaisesRegex(InputIdentityError, "registered 229200"):
            self.plan(lambda symbol, row, start: data, J(13))
        with self.assertRaisesRegex(InputIdentityError, "registered end"):
            self.plan(lambda symbol, row, start: as_symbol(data, symbol), J(14))
        with self.assertRaisesRegex(InputIdentityError, "price basis"):
            raw = build(days, {J(2): "10000"})
            self.plan(lambda symbol, row, start: as_symbol(raw, symbol), J(13))

    def test_identity_marks_modified_and_unpinned_runs(self):
        def unavailable(symbol, row, start):
            raise DataHold("synthetic: no input")
        frozen = run_plan(unavailable, FROZEN)
        self.assertEqual((frozen["run_identity"]["config_modified"], frozen["run_identity"]["formal_run"]), (False, False))
        changed = self.plan(unavailable, J(14))["run_identity"]
        self.assertTrue(changed["config_modified"])
        self.assertNotEqual(changed["applied_config_sha256"], changed["frozen_config_applied_sha256"])
        self.assertEqual(changed["frozen_config_file_sha256"], CONFIG.sha256)

    def test_loader_rejects_end_missing_from_calendar(self):
        end = date(2025, 2, 7)
        rows, times = history_fixture(end)
        shortened = {d: pair for d, pair in times.items() if d != end}
        with fake_source(rows, shortened):
            with self.assertRaisesRegex(ValueError, "fixed end session"):
                loader.load_etf(".", "pinned.json", "069500", J(2), end)


class ReviewA5StartIsolation(unittest.TestCase):
    """A5: 한 시작일의 일봉 오류는 그 시작일만 보류, 후속 시작일·OHLC 범위 오류도 보류로 기록."""

    def run_row_with(self, mutate):
        end = date(2025, 2, 7)
        rows, times = history_fixture(end)
        sessions = [d for d in sorted(times) if d >= J(2)]
        starts = tuple(sessions[i] for i in (0, 5, 20))
        bad = next(r for r in rows if r["date"] == J(2).isoformat())
        mutate(bad)
        row = StudyRow("primary", "primary", ("069500",), starts, CONFIG.segments)
        with fake_source(rows, times):
            result = run_row(row, lambda s, r, start: loader.load_etf(".", "pinned.json", s, start, end),
                             FROZEN.with_changes(end=end))
        return starts, result["symbols"]["069500"]

    def test_single_price_first_start_holds_only_first_start(self):
        def single_price(row):
            row["high"] = row["low"] = row["open"]
        starts, result = self.run_row_with(single_price)
        by_start = result["labels"]["M"]["by_start"]
        self.assertEqual(by_start[starts[0]], "hold")
        self.assertTrue(all(by_start[s] != "hold" for s in starts[1:]))
        self.assertIn("single-price", result["held_starts"][starts[0]])
        self.assertEqual(result["labels"]["M"]["label"], "incomplete")

    def test_ohlc_tolerance_error_is_held_not_raised(self):
        def outside(row):
            row["high"] = str(int(row["open"]) - 500)
            row["low"] = str(int(row["high"]) - 5)
        starts, result = self.run_row_with(outside)
        self.assertIn("0.62%", result["held_starts"][starts[0]])
        self.assertEqual(set(result["held_starts"]), {starts[0]})


class ReviewA6KnownAt(unittest.TestCase):
    """A6: 연초 변동성은 그해 첫 거래일 시가 전에 알려진 값만, 그해 첫 거래일 키에만."""

    def test_late_estimate_rejected(self):
        days = weekdays(J(2), J(13))
        data = build(days, {J(2): "10000"}, basis=BASIS)
        late = VolatilityEstimate(D(".04"), at(days[4], 6, 30))
        with self.assertRaisesRegex(ValueError, "not before that session's open"):
            prepare(data.instrument, data.bars, data.calendar, {days[0]: late})
        exact = VolatilityEstimate(D(".04"), at(days[0]))
        with self.assertRaisesRegex(ValueError, "not before"):
            prepare(data.instrument, data.bars, data.calendar, {days[0]: exact})
        early = VolatilityEstimate(D(".04"), at(days[0]) - timedelta(microseconds=1))
        self.assertEqual(prepare(data.instrument, data.bars, data.calendar, {days[0]: early}).volatility[days[0]], early)
        with self.assertRaisesRegex(ValueError, "first domestic session"):
            prepare(data.instrument, data.bars, data.calendar, {days[5]: early})


class ReviewA7Summary(unittest.TestCase):
    """A7: 시작일 민감은 최상위에서 'unresolved', 확정 수는 후보·미달만."""

    def test_sensitive_not_hidden_as_closed(self):
        from src.research.etf.kodex_core.verdict import FAIL, INCOMPLETE, SENSITIVE
        summary = candidates({"M": SENSITIVE, "D": FAIL}, "candidate")
        self.assertEqual((summary["label"], summary["unresolved"]), ("unresolved", ["M"]))
        self.assertEqual(candidates({"M": FAIL, "D": FAIL}, "candidate")["label"], "closed")
        self.assertEqual(candidates({"M": "candidate", "D": SENSITIVE}, "candidate")["unresolved"], ["D"])
        counts = extension_counts({"a": {"M": SENSITIVE}, "b": {"M": INCOMPLETE}}, 2)["by_mode"]["M"]
        self.assertEqual((counts["decided"], counts["decided_passed"], counts["undecided"]), (0, 0, 2))


class ReviewA8Distributions(unittest.TestCase):
    """A8: 실제로 분배금을 받은 장부만 재투자, 가격 기준 문구는 입력에서."""

    def test_no_entitlement_no_reinvest(self):
        event = Distribution(SYMBOL, J(2), D(100), at(J(6)), "synthetic")
        data = build(weekdays(J(2), J(10)), {J(2): "10000", J(7): "5000"}, distributions=(event,))
        for capital in (CONFIG.capital["core"], CONFIG.capital["hold"]):
            result = run_hold(data, setting(), capital).result
            self.assertFalse([r for r in result.cash_events if r["kind"] == "dividend_payment"])
            self.assertFalse([t for t in result.trades if t["reason"] == "dividend_reinvest"])
        basis = run_start(data, CONFIG, J(2))["basis"]
        self.assertEqual((basis["basis"], basis["label"], basis["total_return_note"]),
                         (ENTITLEMENT_BASIS, CONFIG.labels["basis_entitlement"], None))
        provider = run_start(build(weekdays(J(2), J(10)), {J(2): "10000"}, basis=BASIS), CONFIG, J(2))["basis"]
        self.assertEqual((provider["label"], provider["price_basis"]), (CONFIG.labels["basis"], BASIS))


class ReviewC1Recovery(unittest.TestCase):
    """C1: 2025년 비용 문턱으로 신규 매수 닫힘 → 2026년 첫 거래일 회복 재진입. M·D × 두 경로 × 두 비용."""

    def test_closed_then_recovered_next_year(self):
        start, end, new = date(2024, 12, 30), date(2026, 1, 6), date(2026, 1, 2)
        days = weekdays(start, end, {date(2024, 12, 31), J(1), date(2025, 12, 31), date(2026, 1, 1)})
        values = {J(2): VolatilityEstimate(D(".0001"), at(start, 6, 30)),
                  new: VolatilityEstimate(D(".01"), at(date(2025, 12, 30), 6, 30))}
        data = build(days, {start: "10000", J(3): "10400"}, extra_vol=values, basis=BASIS)
        annual = CONFIG.with_changes(volatility_mode="annual")
        for mode in CONFIG.modes:
            for path in CONFIG.paths:
                for cost in (BASE, STRESS):
                    with self.subTest(mode=mode, path=path, cost=cost):
                        run = run_split(data, setting(mode, path, cost, annual))
                        after = [(t["session_date"], t["side"], t["level"]) for t in run.result.trades
                                 if t["session_date"] >= J(2)]
                        self.assertEqual(after, [(J(3), "SELL", 1), (new, "BUY", 1)])
                        self.assertEqual([r.get("buys_closed", False) for r in run.detail["refreshes"]], [True, False])



# ---------------------------------------------------------------------------
# 구현 리뷰 2차(review-reconciled-impl-r2.md) B1~B7, C1 회귀 시험.
# 입력은 astra-impl-check-r2/reproduce.py·formal_synthetic.py와 같은 합성 자료다(시장 자료 없음).
# ---------------------------------------------------------------------------
import csv  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import tempfile  # noqa: E402

from src.research.etf import m5_inputs  # noqa: E402
from src.research.etf.kodex_core import formal  # noqa: E402
from src.research.etf.kodex_core.inputs import DataHold, InputIntegrityError  # noqa: E402
from src.research.etf.kodex_core.study import run_starts  # noqa: E402


def _put(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_bundle(root, *, synthetic, price_basis=BASIS, mutate=None):
    """고정 설정의 등록 날짜만 담은 합성 M5 묶음(실제 파일 해시 포함). 2004는 251일 준비 기간이 없다."""
    days = (weekdays(date(2004, 1, 2), date(2004, 2, 4), {date(2004, 1, n) for n in (21, 22, 23)})
            + weekdays(date(2010, 1, 4), date(2010, 12, 31)) + weekdays(date(2011, 1, 3), date(2011, 1, 31))
            + weekdays(date(2017, 1, 2), date(2017, 2, 1), {date(2017, 1, 27), date(2017, 1, 30)})
            + [FROZEN.end, date(2026, 11, 16), date(2026, 11, 17)])
    # 시작~만기 사이 모든 연도에 거래일이 있어야 한다(D5). 그해 첫 사흘 평일을 넣는다.
    for year in list(range(2005, 2010)) + list(range(2012, 2017)) + list(range(2018, 2027)):
        days += weekdays(date(year, 1, 2), date(year, 1, 8))[:3]
    days = sorted(set(days))
    calendar = [{"date": d, "open_utc": at(d).isoformat(), "close_utc": at(d, 6, 30).isoformat()} for d in days]
    for name in ("calendar_krx.csv", "calendar_nyse.csv"):
        _put(root / "fx" / name, calendar)
    _put(root / "fx/fx_usdkrw_smbs_mar.csv", [{"available_at_utc": at(days[0]).isoformat(), "value": "1300"}])
    symbols = sorted({s for row in FROZEN.rows for s in row.symbols})
    meta = root / "metadata/etf_metadata.csv"
    _put(meta, [{"market": "KR", "symbol": s, "d1_eligible": "True", "kr_tax_class": "국내주식형"} for s in symbols])
    split = root / "protocol/synth.csv"
    _put(split, [{"market": "KR", "symbol": s, "split": "explore"} for s in symbols])
    rows = []
    for index, day in enumerate(days):
        close = 10000 + (index % 2) * 100
        for s in symbols:
            rows.append(dict(symbol=s, date=day, open=close, high=close + 5, low=close - 5, close=close,
                             volume=300000, price_basis=price_basis, corp_action_ok="True"))
    if mutate is not None:
        mutate(rows)
    price = root / "prices/runs/generated/kr_daily.csv.gz"
    _put(price, rows)
    price_manifest = price.with_name("manifest.json")
    price_manifest.write_text(json.dumps({"daily_sha256": {"KR": _digest(price)}}))
    (root / "prices/CURRENT").write_text("generated")
    split.with_suffix(".json").write_text(json.dumps({
        "split_csv": split.name, "split_sha256": _digest(split), "protocol_sha256": "generated",
        "inputs": {"metadata_sha256": _digest(meta), "price_manifest_sha256": _digest(price_manifest),
                   "daily_sha256": {"KR": _digest(price)}}}))
    manifest = root / "m5-inputs.json"
    m5_inputs.pin_inputs(root, split, manifest, synthetic=synthetic)
    return manifest


class _Bundle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        loader._source.cache_clear()

    def tearDown(self):
        loader._source.cache_clear()
        self.tmp.cleanup()

    def bundle(self, name="a", **options):
        root = self.root / name
        return root, synthetic_bundle(root, **options)


class ReviewB1FormalRun(_Bundle):
    """B1: 장부를 열기 전 등록 명세·비합성·명단·가격 기준·추가 입력 대조. 오류는 보류로 흡수하지 않는다."""

    def registered(self, root, manifest):
        return {row["item"]: row["actual"] for row in formal.identity_table(
            {}, formal.actual_inputs(root, manifest, FROZEN))}

    def test_matching_unsynthetic_bundle_is_formal_with_2004_held(self):
        root, manifest = self.bundle(synthetic=False)
        out = formal.run_formal(root, manifest, FROZEN, registered=self.registered(root, manifest))
        identity = out["run_identity"]
        # 합성 묶음은 등록값을 바꿔 넣었으므로 그 한 항목만 거짓이고 나머지 대조는 모두 참이다.
        self.assertEqual([k for k, ok in identity["checks"].items() if not ok], ["registered_inputs_frozen"])
        self.assertFalse(identity["formal_run"])
        self.assertTrue(all(row["match"] for row in identity["preflight"]["inputs"]))
        records = [r for row in out["rows"].values() for r in row["records"]]
        self.assertEqual((len(records), sum(r["status"] == "loaded" for r in records)), (15, 12))
        self.assertEqual({r["row"] for r in records if r["status"] == "held"}, {"reference_2004"})

    def test_synthetic_or_unregistered_manifest_stops_before_ledgers(self):
        root, manifest = self.bundle(synthetic=True)
        with self.assertRaisesRegex(InputIntegrityError, "synthetic"):
            formal.run_formal(root, manifest, FROZEN, registered=self.registered(root, manifest))
        root2, manifest2 = self.bundle("b", synthetic=False)
        with self.assertRaisesRegex(InputIntegrityError, "input_manifest_sha256"):
            formal.run_formal(root2, manifest2, FROZEN)  # 고정 설정의 등록값(실제 9e66e5b8…)과 다름

    def test_corrupted_price_file_is_integrity_error_not_hold(self):
        root, manifest = self.bundle(synthetic=False)
        registered = self.registered(root, manifest)
        price = root / "prices/runs/generated/kr_daily.csv.gz"
        price.write_bytes(price.read_bytes() + b"corrupt")
        with self.assertRaises(InputIntegrityError):
            formal.run_formal(root, manifest, FROZEN, registered=registered)
        loader._source.cache_clear()
        row = next(r for r in FROZEN.rows if r.id == "primary")
        with self.assertRaises(InputIntegrityError):  # run_row 안에서도 보류로 바뀌지 않는다
            run_row(row, lambda s, r, start: loader.load_etf(root, manifest, s, start, FROZEN.end), FROZEN)

    def test_extra_inputs_mixed_manifests_and_raw_prices_are_not_formal(self):
        """외부 입력기 결과에 정식 대조를 적용해도 추가 입력·혼합 명세·원가격은 정식이 아니다."""
        root, manifest = self.bundle(synthetic=False)
        registered = self.registered(root, manifest)
        checked = formal.preflight(root, manifest, FROZEN, registered=registered)
        first = FROZEN.rows[0].starts[0]

        def locked(symbol, row, start):
            return loader.load_etf(root, manifest, symbol, start, FROZEN.end, locked_status={first: False})
        out = run_plan(locked, FROZEN)
        self.assertFalse(out["run_identity"]["formal_run"])
        identity = formal._formal_identity(FROZEN, out, checked)
        self.assertFalse(identity["formal_run"])
        self.assertFalse(identity["checks"]["no_extra_inputs"])
        record = out["rows"]["primary"]["records"][0]
        self.assertEqual(record["extra_inputs"]["locked_status"], [(first.isoformat(), False)])
        root_b, manifest_b = self.bundle("b", synthetic=False)

        def mixed(symbol, row, start):
            source = (root_b, manifest_b) if symbol == TARGET else (root, manifest)
            return loader.load_etf(*source, symbol, start, FROZEN.end)
        manifest_b.write_text(manifest_b.read_text() + "\n")
        loader._source.cache_clear()
        identity = formal._formal_identity(FROZEN, run_plan(mixed, FROZEN), checked)
        self.assertFalse(identity["checks"]["manifest_registered"])
        self.assertFalse(identity["checks"]["sources_match_preflight"])
        root_raw, manifest_raw = self.bundle("raw", synthetic=False, price_basis="raw")
        with self.assertRaisesRegex(InputIdentityError, "price basis raw"):
            run_plan(lambda s, r, start: loader.load_etf(root_raw, manifest_raw, s, start, FROZEN.end), FROZEN)

    @unittest.skipUnless(Path("/mnt/volume/workspace/split-research/etf/data/etf-m5-r1/m5-inputs.json").is_file(), "pinned data not present")
    def test_registered_real_inputs_pass_preflight_only(self):
        """실제 고정 묶음은 장부 없이 대조만(성과 계산 없음)."""
        root = Path("/mnt/volume/workspace/split-research/etf/data/etf-m5-r1")
        checked = formal.preflight(root, root / "m5-inputs.json", FROZEN)
        self.assertTrue(all(row["match"] for row in checked["inputs"]))
        self.assertEqual(checked["registered"], FROZEN.registered)

    def test_plain_run_plan_is_never_formal(self):
        root, manifest = self.bundle(synthetic=False)
        out = run_plan(lambda s, r, start: loader.load_etf(root, manifest, s, start, FROZEN.end), FROZEN)
        self.assertFalse(out["run_identity"]["formal_run"])
        self.assertFalse(out["run_identity"]["checks"]["via_formal_entrypoint"])


class ReviewB2Chronology(unittest.TestCase):
    """B2: 미래의 연초 결측을 붙여도 첫 주문·이전 거래·개설 미달 판정이 바뀌지 않는다."""

    def data(self):
        dec = date(2024, 12, 30)
        days = weekdays(dec, J(10), {date(2024, 12, 31), J(1)})
        return dec, build(days, {dec: "10000", J(2): "9700", J(3): "10500"},
                          extra_vol={J(2): VolatilityUnavailable("synthetic missing 251st close")}, basis=BASIS)

    def test_first_orders_and_prior_trades_unchanged(self):
        dec, full = self.data()
        prefix = replace(full, bars=full.bars[:1])
        for mode in CONFIG.modes:
            with_gap = run_start(full, FROZEN, dec)["items"][mode]["paths"][PATH]
            without = run_start(prefix, FROZEN, dec)["items"][mode]["paths"][PATH]
            self.assertEqual(with_gap["start_orders"]["base"]["core_plus_split"], D(82200000))
            self.assertEqual(with_gap["start_orders"], without["start_orders"])
            self.assertEqual(with_gap["start_orders"]["base"]["orders"]["split"]["status"], "filled")
        gap_run = run_split(full, setting("D", config=FROZEN))
        prefix_run = run_split(prefix, setting("D", config=FROZEN))
        self.assertEqual(gap_run.result.trades, prefix_run.result.trades)

    def test_unopenable_account_stays_fail(self):
        dec, full = self.data()
        zero = ZeroVolatility(at(dec) - timedelta(hours=12))
        data = replace(full, volatility={**full.volatility, dec: zero})
        before = run_start(replace(data, bars=data.bars[:1]), FROZEN, dec)["items"]["D"]["status"]
        after = run_start(data, FROZEN, dec)["items"]["D"]["status"]
        self.assertEqual((before, after), ("fail", "fail"))


class ReviewB3InputBoundary(unittest.TestCase):
    """B3: 빈·비수치 값과 음수 거래량은 날짜·필드가 담긴 자료 오류. 그 값을 쓰는 시작일만 보류."""

    END = date(2025, 2, 7)

    def run_with(self, mutate, *, times_change=None):
        rows, times = history_fixture(self.END)
        sessions = [d for d in sorted(times) if J(2) <= d <= self.END]
        starts = tuple(sessions[i] for i in (0, 5, 20))
        target = next(r for r in rows if r["date"] == J(2).isoformat())
        mutate(target)
        row = StudyRow("primary", "primary", ("069500",), starts, CONFIG.segments)
        with fake_source(rows, times if times_change is None else times_change(times)):
            result = run_row(row, lambda s, r, start: loader.load_etf(".", "pinned.json", s, start, self.END),
                             FROZEN.with_changes(end=self.END))
        return starts, result["symbols"]["069500"]

    def test_blank_or_text_open_holds_only_that_start(self):
        for value in ("", "abc"):
            with self.subTest(value=value):
                starts, result = self.run_with(lambda r: r.update(open=value))
                self.assertEqual(set(result["held_starts"]), {starts[0]})
                self.assertIn("open", result["held_starts"][starts[0]])
                self.assertIn("2025-01-02", result["held_starts"][starts[0]])

    def test_negative_volume_holds_every_start_whose_window_uses_it(self):
        starts, result = self.run_with(lambda r: r.update(volume="-1"))
        self.assertEqual(set(result["held_starts"]), set(starts))  # 1/9·1/30의 직전 20일 창에 1/2가 들어 있다
        self.assertTrue(all("volume" in reason for reason in result["held_starts"].values()))

    def test_blank_close_in_twenty_day_window_is_field_error(self):
        starts, result = self.run_with(lambda r: r.update(close=""))
        self.assertTrue(all("close" in reason for reason in result["held_starts"].values()))

    def test_missing_start_year_calendar_is_data_hold(self):
        rows, times = history_fixture(self.END)
        no_year = {d: v for d, v in times.items() if d.year != 2025}
        no_year[date(2026, 1, 2)] = (at(date(2026, 1, 2)), at(date(2026, 1, 2), 6, 30))
        with fake_source(rows, no_year):
            with self.assertRaisesRegex(ValueError, "no sessions"):
                loader.load_etf(".", "pinned.json", "069500", J(2), date(2026, 1, 2))

    def test_field_value_messages(self):
        row = {"symbol": "069500", "open": "", "volume": "-1"}
        with self.assertRaisesRegex(ValueError, "069500 2025-01-02 open: missing or non-numeric"):
            loader.field_value(row, "open", J(2), positive=True)
        with self.assertRaisesRegex(ValueError, "volume: negative"):
            loader.field_value(row, "volume", J(2), positive=False)


class ReviewB4KnownAtAtEntry(unittest.TestCase):
    """B4: 적재 뒤 바꿔 넣은 연초 변동성을 사용 직전에 거부한다(+5 시작도 연초 시가 기준)."""

    def test_late_value_injected_after_prepare_is_rejected(self):
        days = weekdays(J(2), J(13))
        data = build(days, {J(2): "10000"}, basis=BASIS)
        late = VolatilityEstimate(D(".04"), at(days[4], 6, 30))
        bad = starting_at(replace(data, volatility={days[0]: late}), days[5])
        with self.assertRaisesRegex(InputIdentityError, "not before"):
            run_start(bad, CONFIG, days[5])
        row = StudyRow("primary", "primary", (SYMBOL,), (days[5],), CONFIG.segments)
        with self.assertRaises(InputIdentityError):
            run_plan(lambda *args: bad, CONFIG.with_changes(rows=(row,), start_offsets=(0,), end=days[-1]))

    def test_early_but_replaced_value_differs_from_loader_digest(self):
        rows, times = history_fixture(date(2025, 2, 7))
        with fake_source(rows, times):
            loaded = loader.load_etf(".", "pinned.json", "069500", J(2), date(2025, 2, 7))
        swapped = replace(loaded, volatility={J(2): VolatilityEstimate(D(".04"), at(J(2)) - timedelta(days=1))})
        config = FROZEN.with_changes(end=date(2025, 2, 7))
        from src.research.etf.kodex_core.study import verify_identity
        verify_identity(loaded, "069500", J(2), config)
        with self.assertRaisesRegex(InputIdentityError, "annual volatility differs"):
            verify_identity(swapped, "069500", J(2), config)


class ReviewB5OrderStates(unittest.TestCase):
    """B5: 시작 주문 상태(미제출/제출 후 거절/체결)를 구분한다."""

    def test_rejected_and_filled_states(self):
        data = build(weekdays(J(2), J(10)), {J(2): "10000"}, changes={J(2): {"adv20": D(1644000000)}}, basis=BASIS)
        orders = run_start(data, CONFIG, J(2))["items"]["M"]["paths"][PATH]["start_orders"]["base"]["orders"]
        self.assertEqual(orders["hold"]["status"], "rejected")
        self.assertEqual(orders["hold"]["reason"], "participation_limit")
        self.assertEqual(orders["hold"]["notional"], D(99980000))
        self.assertEqual((orders["core"]["status"], orders["split"]["status"]), ("filled", "filled"))

    def test_split_not_submitted_and_not_opened(self):
        data = build(weekdays(J(2), J(10)), {J(2): "3000000"}, basis=BASIS)
        orders = run_start(data, CONFIG, J(2))["items"]["M"]["paths"][PATH]["start_orders"]["base"]["orders"]
        self.assertEqual(orders["split"], {"status": "not_submitted", "reason": "zero_affordable_quantity"})
        zero = build(weekdays(J(2), J(10)), {J(2): "10000"}, sigma="0.002", basis=BASIS)
        stress = run_start(zero, CONFIG, J(2))["items"]["M"]["paths"][PATH]["start_orders"]["stress"]["orders"]
        self.assertEqual(stress["split"], {"status": "not_opened", "reason": "cost_guard_start"})


class ReviewB6PriceBasis(unittest.TestCase):
    """B6: 가격 기준은 입력 속성. 마지막 권리일 뒤 시작해도 원가격 표시, 한계 문구도 가격 기준에 맞춘다."""

    def test_raw_after_last_event_and_limits(self):
        event = Distribution(SYMBOL, J(2), D(100), at(J(6)), "synthetic")
        raw = build(weekdays(J(2), J(10)), {J(2): "10000"}, distributions=(event,))
        after = starting_at(raw, J(3))
        self.assertEqual((after.distributions, after.basis), ((), ENTITLEMENT_BASIS))
        label = run_start(raw, CONFIG, J(3))["basis"]
        self.assertEqual((label["price_mode"], label["distribution_events"], label["total_return_note"]), ("raw", 0, None))
        row = StudyRow("primary", "primary", (SYMBOL,), (J(3),), CONFIG.segments)
        config = CONFIG.with_changes(rows=(row,), start_offsets=(0,), end=J(10), price_basis="raw")
        limits = run_plan(lambda *args: after, config)["limits"]
        self.assertNotIn(CONFIG.labels["total_return_note"], limits)
        self.assertIn(CONFIG.labels["basis_entitlement"], limits)
        provider = build(weekdays(J(2), J(10)), {J(2): "10000"}, basis=BASIS)
        limits = run_plan(lambda *args: provider, CONFIG.with_changes(
            rows=(replace(row, starts=(J(2),)),), start_offsets=(0,), end=J(10)))["limits"]
        self.assertIn(CONFIG.labels["total_return_note"], limits)


class ReviewB7Identity(_Bundle):
    """B7: 실행 전체 등록값·실제값·대조 결과와 행별 기록, 소스 해시의 실제 의존 파일."""

    def test_identity_table_and_row_records(self):
        root, manifest = self.bundle(synthetic=False)
        actual = formal.actual_inputs(root, manifest, FROZEN)
        table = formal.identity_table(FROZEN.registered, actual)
        self.assertEqual([row["item"] for row in table], list(formal.IDENTITY_KEYS))
        self.assertFalse(next(r for r in table if r["item"] == "input_manifest_sha256")["match"])
        self.assertEqual(FROZEN.registered["input_manifest_sha256"][:12], "9e66e5b8f16b")
        out = formal.run_formal(root, manifest, FROZEN, registered=actual)
        files = out["run_identity"]["preflight"]["source_files"]
        for name in ("m5_inputs.py", "price_bundle.py", "kodex_core/formal.py", "engine.py"):
            self.assertIn(name, files)
        self.assertIn("head", out["run_identity"]["preflight"]["git"])
        loaded = next(r for r in out["rows"]["primary"]["records"] if r["status"] == "loaded")
        for key in ("role", "registered_start", "actual_start", "actual_end", "last_evaluation", "price_basis",
                    "capacity", "opening", "reasons", "input_manifest_sha256"):
            self.assertIn(key, loaded)
        self.assertEqual(loaded["actual_end"], FROZEN.end)
        self.assertEqual(loaded["last_evaluation"].astimezone(timezone(timedelta(hours=9))).hour, 18)
        held = out["rows"]["reference_2004"]["records"]
        self.assertTrue(all(r["status"] == "held" and "251" in r["reason"] for r in held))


class ReviewC1ClosedYearDrop(unittest.TestCase):
    """C1(계속): 신규 매수가 닫힌 해에 전년 하락 간격(2%) 이상 떨어진 시가가 와도 추가 매수 0건."""

    def test_no_add_buy_in_closed_year(self):
        start = date(2024, 12, 30)
        days = weekdays(start, J(10), {date(2024, 12, 31), J(1)})
        closed = {J(2): VolatilityEstimate(D(".0001"), at(start, 6, 30))}
        data = build(days, {start: "10000", J(3): "9700", J(6): "9400", J(7): "9100"}, extra_vol=closed, basis=BASIS)
        annual = CONFIG.with_changes(volatility_mode="annual")
        for mode in CONFIG.modes:
            for path in CONFIG.paths:
                for cost in (BASE, STRESS):
                    with self.subTest(mode=mode, path=path, cost=cost):
                        run = run_split(data, setting(mode, path, cost, annual))
                        self.assertEqual([t["level"] for t in run.result.trades if t["side"] == "BUY"], [1])
                        self.assertTrue(run.detail["refreshes"][0]["buys_closed"])
        open_year = replace(data, volatility={**data.volatility, J(2): VolatilityEstimate(D(".01"), at(start, 6, 30))})
        run = run_split(open_year, setting("D", config=annual))
        self.assertEqual([t["level"] for t in run.result.trades if t["side"] == "BUY"], [1, 2, 3, 4])



# ---------------------------------------------------------------------------
# 구현 리뷰 3차(review-reconciled-impl-r3.md) D1~D8, Claude 3차 지적 회귀 시험.
# 입력은 astra-impl-check-r3/의 합성 사례(cache_fixture·formal_check·edges·reproduce)를 옮긴 것이다.
# ---------------------------------------------------------------------------
from decimal import localcontext as _localcontext  # noqa: E402

from src.research.etf.kodex_core import study as study_module  # noqa: E402


class ReviewD1ImmutableInputs(_Bundle):
    """D1: 반환한 달력·출처를 바꿔도 다음 정식 실행과 식별값이 같다."""

    def test_returned_calendar_is_read_only_and_next_run_unchanged(self):
        root, manifest = self.bundle(synthetic=False)
        registered = {row["item"]: row["actual"] for row in formal.identity_table(
            {}, formal.actual_inputs(root, manifest, FROZEN))}
        first = formal.run_formal(root, manifest, FROZEN, registered=registered)
        primary = next(r for r in FROZEN.rows if r.id == "primary")
        loaded = loader.load_etf(root, manifest, "069500", primary.starts[0], FROZEN.end)
        day = date(2015, 6, 1)
        with self.assertRaises(TypeError):
            loaded.calendar.times[day] = (at(day), at(day, 6, 30))
        cached = loader._source(root.resolve(), manifest.resolve(), "069500", FROZEN.end)
        with self.assertRaises(TypeError):
            cached[3][day] = (at(day), at(day, 6, 30))
        with self.assertRaises(TypeError):
            cached[2][0]["close"] = "1"
        loaded.provenance["source_files"].append("injected.py")
        again = loader.load_etf(root, manifest, "069500", primary.starts[0], FROZEN.end)
        self.assertNotIn("injected.py", again.provenance["source_files"])
        second = formal.run_formal(root, manifest, FROZEN, registered=registered)
        self.assertEqual(first["rows"]["primary"]["symbols"]["069500"]["held_starts"], {})
        self.assertEqual(second["rows"]["primary"]["symbols"]["069500"]["held_starts"], {})
        self.assertEqual(first["run_identity"]["checks"], second["run_identity"]["checks"])
        self.assertEqual(first["run_identity"]["preflight"]["inputs"], second["run_identity"]["preflight"]["inputs"])
        self.assertEqual([r for row in first["rows"].values() for r in row["records"]],
                         [r for row in second["rows"].values() for r in row["records"]])


class ReviewD2NoFormalFromExternalLoader(_Bundle):
    """D2: 외부 입력기를 받는 run_plan은 사전 대조표가 있어도 정식 표시를 만들 수 없다."""

    def test_run_plan_rejects_preflight_and_is_never_formal(self):
        import inspect
        self.assertNotIn("preflight", inspect.signature(run_plan).parameters)
        root, manifest = self.bundle(synthetic=False)
        out = run_plan(lambda s, r, start: loader.load_etf(root, manifest, s, start, FROZEN.end), FROZEN)
        self.assertEqual((out["run_identity"]["formal_run"], out["run_identity"]["checks"]),
                         (False, {"via_formal_entrypoint": False}))

    def test_copied_manifest_id_does_not_match_preflight_sources(self):
        root, manifest = self.bundle(synthetic=False)
        other_root, other_manifest = self.bundle("other", synthetic=False)
        price = other_root / "prices/runs/generated/kr_daily.csv.gz"  # 다른 내용의 묶음
        checked = formal.preflight(root, manifest, FROZEN, registered={row["item"]: row["actual"] for row in
                                   formal.identity_table({}, formal.actual_inputs(root, manifest, FROZEN))})

        def load(symbol, row, start):
            obj = loader.load_etf(other_root, other_manifest, symbol, start, FROZEN.end)
            fake = {**obj.provenance, "input_manifest_sha256": checked["registered"]["input_manifest_sha256"],
                    "daily_sha256": "f" * 64}
            return replace(obj, provenance=fake)
        self.assertTrue(price.is_file())
        identity = formal._formal_identity(FROZEN, run_plan(load, FROZEN), checked)
        self.assertTrue(identity["checks"]["manifest_registered"])
        self.assertFalse(identity["checks"]["sources_match_preflight"])
        self.assertFalse(identity["formal_run"])


class ReviewD3RejectBeforeLedgers(_Bundle):
    """D3: 고정 설정·명단·만기·가격 기준이 다르면 첫 장부 전에 멈춘다."""

    def test_modified_plan_stops_before_any_ledger(self):
        root, manifest = self.bundle(synthetic=False)
        registered = {row["item"]: row["actual"] for row in formal.identity_table(
            {}, formal.actual_inputs(root, manifest, FROZEN))}
        primary = FROZEN.rows[0]
        changes = ({"rows": (primary,)}, {"end": date(2026, 11, 12)}, {"price_basis": "raw"})
        for change in changes:
            with self.subTest(change=list(change)):
                with patch.object(study_module, "_ledgers", side_effect=AssertionError("ledger opened")):
                    with self.assertRaisesRegex(InputIdentityError, "frozen configuration"):
                        formal.run_formal(root, manifest, FROZEN.with_changes(**change), registered=registered)
                    with self.assertRaisesRegex(InputIdentityError, "frozen configuration"):
                        formal.preflight(root, manifest, FROZEN.with_changes(**change), registered=registered)

    @unittest.skipUnless(Path("/mnt/volume/workspace/split-research/etf/data/etf-m5-r1/m5-inputs.json").is_file(), "pinned data not present")
    def test_real_preflight_rejects_modified_plan(self):
        root = Path("/mnt/volume/workspace/split-research/etf/data/etf-m5-r1")
        with self.assertRaises(InputIdentityError):
            formal.preflight(root, root / "m5-inputs.json", FROZEN.with_changes(rows=(FROZEN.rows[0],)))


class ReviewD4StartFailurePreserved(unittest.TestCase):
    """D4: 시작 시 한쪽 비용만 개설 미달이어도, 이후 연초 결측이 그 미달을 보류로 바꾸지 않는다."""

    def test_stress_start_failure_survives_future_gap(self):
        dec = date(2024, 12, 30)
        days = weekdays(dec, J(10), {date(2024, 12, 31), J(1)})
        data = build(days, {dec: "10000"}, sigma="0.002",
                     extra_vol={J(2): VolatilityUnavailable("synthetic missing 251st close")}, basis=BASIS)
        prefix = replace(data, bars=data.bars[:1])
        for mode in CONFIG.modes:
            with self.subTest(mode=mode):
                full = run_start(data, FROZEN, dec)["items"][mode]
                short = run_start(prefix, FROZEN, dec)["items"][mode]
                self.assertEqual((full["status"], short["status"]), ("fail", "fail"))
                verdict = full["paths"][PATH]["verdict"]
                self.assertEqual(verdict["reasons"], ["stress_start_rejected_by_cost_guard"])
                self.assertEqual(verdict["later_holds"], ["base_annual_volatility_unavailable"])
                orders = full["paths"][PATH]["start_orders"]
                self.assertEqual(orders["base"]["core_plus_split"], D(82200000))
                self.assertEqual(orders["stress"]["orders"]["split"]["status"], "not_opened")

    def test_later_gap_alone_is_hold(self):
        verdict = path_verdict(None, None, CONFIG.thresholds, later_holds=["base_annual_volatility_unavailable"])
        self.assertEqual(verdict["status"], "hold")


class ReviewD5CalendarYearGap(unittest.TestCase):
    """D5: 시작~만기 사이 달력에 연도 전체가 빠지거나, 일봉이 있는데 달력에 없는 날이 있으면 자료 보류."""

    def fixture(self):
        start, end = date(2024, 12, 30), date(2026, 1, 9)
        days = weekdays(date(2022, 12, 1), end + timedelta(days=7))
        times = {d: (at(d), at(d, 6, 30)) for d in days}
        rows = [dict(symbol="069500", date=str(d), open=str(10000 + (i % 2) * 100), high=str(10105 + (i % 2) * 100),
                     low=str(9995), close=str(10000 + (i % 2) * 100), volume="300000", price_basis=BASIS,
                     corp_action_ok="True") for i, d in enumerate(days)]
        return start, end, times, rows

    def test_rows_without_calendar_year_are_held(self):
        start, end, times, rows = self.fixture()
        with fake_source(rows, times):
            complete = loader.load_etf(".", "pinned.json", "069500", start, end)
        self.assertEqual(sum(b.session_date.year == 2025 for b in complete.bars), 261)
        missing = {d: v for d, v in times.items() if d.year != 2025}
        with fake_source(rows, missing):
            with self.assertRaisesRegex(DataHold, "2025"):
                loader.load_etf(".", "pinned.json", "069500", start, end)
        # 연도는 남아 있고 하루만 달력에서 빠진 경우: 일봉이 있는데 달력에 없는 날
        one = {d: v for d, v in times.items() if d != date(2025, 6, 2)}
        with fake_source(rows, one):
            with self.assertRaisesRegex(DataHold, "not domestic calendar sessions"):
                loader.load_etf(".", "pinned.json", "069500", start, end)
        no_rows = [r for r in rows if not r["date"].startswith("2025")]
        with fake_source(no_rows, missing):
            with self.assertRaisesRegex(DataHold, r"no sessions in \[2025\]"):
                loader.load_etf(".", "pinned.json", "069500", start, end)

    def test_prepare_rejects_year_gap(self):
        days = [d for d in weekdays(date(2024, 12, 2), date(2026, 1, 9)) if d.year != 2025]
        calendar = KrCalendar({d: (at(d), at(d, 6, 30)) for d in days + [date(2026, 1, 12), date(2026, 1, 13)]})
        bars = [make_bar(d, "10000", high=D(10005)) for d in days]
        with self.assertRaisesRegex(DataHold, "2025"):
            prepare(instrument(BASIS), bars, calendar, {})


class ReviewD6ErrorClassification(unittest.TestCase):
    """D6: 정상 0 변동성은 명시 판정, 예상 밖 ValueError는 전파, 자료 문제 사유에 종목·날짜·필드."""

    def history(self):
        rows, times = history_fixture(J(10))
        return rows, times

    def test_unexpected_estimator_error_propagates(self):
        rows, times = self.history()
        with patch.object(loader, "estimate_volatility", side_effect=ValueError("diagnostic")):
            with self.assertRaisesRegex(ValueError, "diagnostic"):
                loader._volatility(rows, times, [J(2)])

    def test_unexpected_liquidity_error_propagates_from_row(self):
        end = date(2025, 2, 7)
        rows, times = history_fixture(end)
        sessions = [d for d in sorted(times) if J(2) <= d <= end]
        row = StudyRow("primary", "primary", ("069500",), tuple(sessions[i] for i in (0, 5, 20)), CONFIG.segments)
        with fake_source(rows, times), patch.object(loader, "_adv20", side_effect=ValueError("diagnostic")):
            with self.assertRaisesRegex(ValueError, "diagnostic"):
                run_row(row, lambda s, r, start: loader.load_etf(".", "pinned.json", s, start, end),
                        FROZEN.with_changes(end=end))

    def test_zero_volatility_needs_identical_valid_returns(self):
        rows, times = self.history()
        past = [dict(r, close="10000", volume="7") for r in rows if r["date"] < J(2).isoformat()][-251:]
        self.assertIsInstance(loader._volatility(past, times, [J(2)])[J(2)], ZeroVolatility)
        past[100]["close"] = "10005"
        self.assertIsInstance(loader._volatility(past, times, [J(2)])[J(2)], VolatilityEstimate)

    def test_window_reason_has_symbol_date_field(self):
        rows, times = self.history()
        past = [dict(r) for r in rows if r["date"] < J(2).isoformat()][-251:]
        past[22]["volume"] = "-1"
        reason = loader._volatility(past, times, [J(2)], "069500")[J(2)].reason
        self.assertIn("069500", reason)
        self.assertIn(past[22]["date"], reason)
        self.assertIn("volume", reason)


class ReviewD7CapacityPrecision(unittest.TestCase):
    """D7: 1억 원 문턱 비교와 직전 20일 평균은 호출자 Decimal 정밀도와 무관하다(1원 경계)."""

    def test_one_won_below_floor_under_any_ambient_precision(self):
        data = build(weekdays(J(2), J(6)), {J(2): "10000"}, basis=BASIS, changes={J(2): {"adv20": D("1999999980")}})
        for precision in (28, 7):
            with self.subTest(precision=precision), _localcontext(prec=precision):
                result = run_start(data, FROZEN, J(2), nominal_floor=FROZEN.nominal_floor)
                self.assertEqual(result["capacity"]["limit_krw"], D("99999999.00"))
                self.assertFalse(result["capacity"]["ok"])
                split = result["items"]["M"]["paths"][PATH]["start_orders"]["base"]["orders"]["split"]
                self.assertEqual(split["status"], "not_submitted")
        exact = build(weekdays(J(2), J(6)), {J(2): "10000"}, basis=BASIS, changes={J(2): {"adv20": D("2000000000")}})
        with _localcontext(prec=7):
            self.assertTrue(run_start(exact, FROZEN, J(2), nominal_floor=FROZEN.nominal_floor)["capacity"]["ok"])

    def test_adv20_exact_under_low_precision(self):
        sessions = weekdays(date(2024, 12, 2), J(2))
        dated = {d: {"symbol": "069500", "close": "99999999", "volume": "20"} for d in sessions}
        with _localcontext(prec=7):
            value = loader._adv20(dated, sessions, len(sessions) - 1)
        self.assertEqual(value, D("1999999980"))


class ReviewD8HeldRecordsAndLimits(unittest.TestCase):
    """D8: 모든 시작일 보류 시 한계 문구는 설정 가격 기준, 보류 행에도 실행·명세 식별값·등록 만기·가격 기준."""

    def test_all_held(self):
        def held(symbol, row, start):
            raise DataHold("synthetic: no input")
        provider = run_plan(held, FROZEN)
        self.assertIn(FROZEN.labels["total_return_note"], provider["limits"])
        raw = run_plan(held, FROZEN.with_changes(price_basis="raw"))
        self.assertNotIn(FROZEN.labels["total_return_note"], raw["limits"])
        self.assertIn(FROZEN.labels["basis_entitlement"], raw["limits"])
        record = provider["rows"]["primary"]["records"][0]
        self.assertEqual(record["status"], "held")
        self.assertEqual((record["registered_end"], record["registered_price_basis"]), (FROZEN.end, FROZEN.price_basis))
        self.assertEqual(record["registered_input_manifest_sha256"], FROZEN.registered["input_manifest_sha256"])
        self.assertEqual(record["applied_config_sha256"], FROZEN.applied_sha256())


class ReviewClaude3ManifestNotSynthetic(_Bundle):
    """Claude 3차: 출처가 합성 명세면 다른 대조가 모두 같아도 정식이 아니다(manifest_not_synthetic 보호)."""

    def test_synthetic_flag_alone_blocks_formal(self):
        root, manifest = self.bundle(synthetic=False)
        actual = formal.actual_inputs(root, manifest, FROZEN)
        checked = formal.preflight(root, manifest, FROZEN, registered=actual)
        provenance = {key: actual[key] for key in ("input_manifest_sha256", "daily_sha256", "calendar_sha256",
                                                   "metadata_sha256", "price_bundle")}
        provenance.update(price_basis=FROZEN.price_basis, extra_inputs={"present": False})

        def output(synthetic):
            start = {"items": {}, "input_provenance": {**provenance, "manifest_synthetic": synthetic}}
            return {"rows": {"primary": {"symbols": {"069500": {"starts": {J(2): start}}}}}}
        good = formal._formal_identity(FROZEN, output(False), checked)["checks"]
        bad = formal._formal_identity(FROZEN, output(True), checked)["checks"]
        self.assertEqual([k for k, ok in good.items() if not ok], ["registered_inputs_frozen"])
        self.assertEqual([k for k, ok in bad.items() if not ok], ["registered_inputs_frozen", "manifest_not_synthetic"])



# ---------------------------------------------------------------------------
# 구현 리뷰 4차(review-reconciled-impl-r4.md) E1~E5 회귀 시험.
# 입력은 astra-impl-check-r4/(data_edges·history_calendar·other_checks·formal_cases)의 합성 사례를 옮긴 것이다.
# ---------------------------------------------------------------------------
from src.research.etf.kodex_core import config as config_module  # noqa: E402


def _row(rows, symbol, day):
    hit, = [r for r in rows if r.get("symbol", "069500") == symbol and str(r["date"]) == day]
    return hit


def _low_sigma(rows):
    """σ≈0.2%: 기본 비용은 개설, 스트레스(왕복 0.6%)는 개설 미달."""
    dates = sorted({str(r["date"]) for r in rows})
    for r in rows:
        price = 10000 + 20 * (dates.index(str(r["date"])) % 2)
        r.update(open=price, high=price + 5, low=price - 5, close=price)


class ReviewE1PriceBasisBeforeLedgers(_Bundle):
    """E1: 등록 명단 전체 일봉의 가격 기준을 첫 장부 전에 확인한다(혼합·단일 원가격 모두 중단)."""

    def registered(self, root, manifest):
        return formal.actual_inputs(root, manifest, FROZEN)

    def test_mixed_or_raw_basis_stops_before_any_ledger(self):
        def mixed(rows):
            _row(rows, TARGET, "2017-01-09")["price_basis"] = "raw"

        def all_raw(rows):
            for r in rows:
                if r["symbol"] == TARGET:
                    r["price_basis"] = "raw"
        for name, mutate in (("mixed", mixed), ("raw", all_raw)):
            with self.subTest(name):
                root, manifest = self.bundle(name, synthetic=False, mutate=mutate)
                with patch.object(study_module, "_ledgers", side_effect=AssertionError("ledger opened")):
                    with self.assertRaisesRegex(InputIdentityError, TARGET):
                        formal.run_formal(root, manifest, FROZEN, registered=self.registered(root, manifest))

    def test_uniform_registered_basis_passes_preflight(self):
        root, manifest = self.bundle(synthetic=False)
        checked = formal.preflight(root, manifest, FROZEN, registered=self.registered(root, manifest))
        self.assertEqual(set(checked["price_basis_by_symbol"].values()), {FROZEN.price_basis})


class ReviewE2FileGapAfterStart(_Bundle):
    """E2: 개설 뒤(2015-01-05) 시가 한 칸이 비어도 시작 판정·첫 주문·스트레스 개설 미달을 보존한다."""

    def formal_primary(self, mutate, name):
        root, manifest = self.bundle(name, synthetic=False, mutate=mutate)
        out = formal.run_formal(root, manifest, FROZEN, registered=formal.actual_inputs(root, manifest, FROZEN))
        return out["rows"]["primary"]["symbols"]["069500"]

    def test_future_missing_open_keeps_start_failure(self):
        def gap(rows):
            _low_sigma(rows)
            _row(rows, "069500", "2015-01-05")["open"] = ""
        normal, broken = self.formal_primary(_low_sigma, "normal"), self.formal_primary(gap, "gap")
        self.assertEqual(broken["held_starts"], {})
        for start in FROZEN.rows[0].starts:
            for mode in CONFIG.modes:
                with self.subTest(start=start, mode=mode):
                    before = normal["starts"][start]["items"][mode]["paths"][PATH]
                    after = broken["starts"][start]["items"][mode]["paths"][PATH]
                    self.assertEqual((before["verdict"]["status"], after["verdict"]["status"]), ("fail", "fail"))
                    self.assertEqual(after["verdict"]["reasons"], ["stress_start_rejected_by_cost_guard"])
                    self.assertIn("input_data_gap_after_start", after["verdict"]["later_holds"])
                    self.assertEqual(before["start_orders"], after["start_orders"])
                    self.assertIsNone(after["base"])
            data_end = broken["starts"][start]["data_end"]
            self.assertEqual((data_end["last_valid_session"], data_end["first_bad_session"]),
                             (date(2015, 1, 2), date(2015, 1, 5)))
            self.assertIn("2015-01-05 open", data_end["reason"])
        first = broken["starts"][FROZEN.rows[0].starts[0]]["items"]["M"]["paths"][PATH]
        self.assertEqual(first["start_orders"]["base"]["core_plus_split"], D(82200000))

    def test_gap_without_start_failure_is_hold(self):
        end = date(2025, 2, 7)
        rows, times = history_fixture(end)
        _row(rows, "069500", "2025-01-20")["open"] = ""
        with fake_source(rows, times):
            data = loader.load_etf(".", "pinned.json", "069500", J(2), end)
        self.assertEqual((data.end, data.data_end["first_bad_session"]), (date(2025, 1, 17), date(2025, 1, 20)))
        verdict = run_start(data, FROZEN.with_changes(end=end), J(2))["items"]["M"]["paths"][PATH]["verdict"]
        self.assertEqual((verdict["status"], verdict["reasons"]), ("hold", ["input_data_gap_after_start"]))
        from src.research.etf.kodex_core.study import verify_identity
        verify_identity(data, "069500", J(2), FROZEN.with_changes(end=end))  # 요청 만기 = 등록 만기

    def test_missing_row_after_start_truncates(self):
        end = date(2025, 2, 7)
        rows, times = history_fixture(end)
        rows = [r for r in rows if r["date"] != "2025-01-21"]
        with fake_source(rows, times):
            data = loader.load_etf(".", "pinned.json", "069500", J(2), end)
        self.assertEqual(data.data_end["last_valid_session"], date(2025, 1, 20))
        self.assertIn("price row missing", data.data_end["reason"])


class ReviewE3PreparationCalendar(unittest.TestCase):
    """E3: 251·20거래일 준비기간에서도 일봉 날짜를 달력과 대조한다(일봉은 있는데 달력에 없으면 보류)."""

    def test_prestart_calendar_removal_is_held(self):
        end = date(2025, 2, 7)
        rows, times = history_fixture(end)
        for removed in (date(2024, 12, 16), date(2024, 3, 4)):  # 20일 창 안, 251일 창 안
            with self.subTest(removed=removed):
                shortened = {d: v for d, v in times.items() if d != removed}
                with fake_source(rows, shortened):
                    with self.assertRaisesRegex(DataHold, "not domestic calendar sessions"):
                        loader.load_etf(".", "pinned.json", "069500", J(2), end)
        with fake_source(rows, times):
            self.assertEqual(loader.load_etf(".", "pinned.json", "069500", J(2), end).start, J(2))


class ReviewE4ConfigDeclarations(_Bundle):
    """E4: 의미 있는 선언(객체에 없는 키 포함)이 바뀐 설정 파일은 첫 장부 전 거부, 공백 차이는 허용하되 구별."""

    def write(self, change=None, indent=None):
        raw = json.loads(config_module.CONFIG_PATH.read_bytes())
        if change:
            raw.update(change)
        path = self.root / "config.json"
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=indent))
        return path

    def test_year_days_change_rejected_at_load(self):
        with self.assertRaisesRegex(ValueError, "year_days"):
            config_module.load_config(self.write({"year_days": "360"}))

    def test_undeclared_key_change_stops_preflight(self):
        root, manifest = self.bundle(synthetic=False)
        actual = formal.actual_inputs(root, manifest, FROZEN)
        changed = config_module.load_config(self.write({"status": "edited"}))
        self.assertTrue(config_module.frozen_identity(changed)["declarations_modified"])
        with patch.object(study_module, "_ledgers", side_effect=AssertionError("ledger opened")):
            with self.assertRaisesRegex(InputIdentityError, "frozen configuration"):
                formal.run_formal(root, manifest, changed, registered=actual)

    def test_whitespace_only_file_is_accepted_and_identified(self):
        spaced = config_module.load_config(self.write(indent=7))
        identity = config_module.frozen_identity(spaced)
        self.assertFalse(identity["config_modified"])
        self.assertNotEqual(identity["provided_config_file_sha256"], identity["frozen_config_file_sha256"])
        self.assertEqual(identity["provided_declared_sha256"], identity["frozen_declared_sha256"])
        context = study_module._context(spaced)
        self.assertEqual(context["frozen_config_file_sha256"], identity["frozen_config_file_sha256"])
        self.assertEqual(context["provided_config_file_sha256"], spaced.sha256)
        root, manifest = self.bundle(synthetic=False)
        formal.preflight(root, manifest, spaced, registered=formal.actual_inputs(root, manifest, FROZEN))


class ReviewE5HighBelowLow(unittest.TestCase):
    """E5: 고가 < 저가는 종목·날짜·필드가 담긴 DataHold, 영향 시작일만 보류(나중 일봉이면 관측 종료)."""

    END = date(2025, 2, 7)

    def test_first_start_held_others_run(self):
        rows, times = history_fixture(self.END)
        _row(rows, "069500", "2025-01-02").update(high="9995", low="10005")
        sessions = [d for d in sorted(times) if J(2) <= d <= self.END]
        starts = tuple(sessions[i] for i in (0, 5, 20))
        row = StudyRow("primary", "primary", ("069500",), starts, CONFIG.segments)
        with fake_source(rows, times):
            result = run_row(row, lambda s, r, start: loader.load_etf(".", "pinned.json", s, start, self.END),
                             FROZEN.with_changes(end=self.END))["symbols"]["069500"]
        self.assertEqual(set(result["held_starts"]), {starts[0]})
        self.assertIn("069500 2025-01-02 high/low", result["held_starts"][starts[0]])

    def test_later_reversed_bar_ends_observation(self):
        rows, times = history_fixture(self.END)
        _row(rows, "069500", "2025-01-15").update(high="9995", low="10005")
        with fake_source(rows, times):
            data = loader.load_etf(".", "pinned.json", "069500", J(2), self.END)
        self.assertEqual(data.data_end["first_bad_session"], date(2025, 1, 15))
        self.assertIn("high/low", data.data_end["reason"])


if __name__ == "__main__":
    unittest.main()
