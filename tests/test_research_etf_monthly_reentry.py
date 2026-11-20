"""실제 엔진의 잔고·거절을 이용한 월초 재진입 실험 경계 검사."""
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace
import unittest

from src.research.etf import engine_strategy
from tests.test_research_etf_engine import bar, kr_run
from tools.research.etf_reentry_compare import MonthlyReentry, summarize


class MonthlyReentryTests(unittest.TestCase):
    def replay(self, bars, *, extra_sessions=(), **options):
        sessions = [b.session_date for b in bars] + list(extra_sessions)
        kernel = MonthlyReentry(engine_strategy, SimpleNamespace(kr_sessions=sessions, us_sessions=[]))
        result = kr_run(bars, _candidate_provider=kernel.candidate_for_bar, **options)
        return result, kernel.counts(result)

    def bars(self):
        return [bar(day=date.fromisoformat(day), price=price) for day, price in (
            ("2024-12-27", "100"), ("2024-12-30", "110"), ("2025-01-02", "100"),
            ("2025-01-03", "110"), ("2025-01-06", "100"), ("2025-02-03", "100"))]

    def test_year_rollover_exit_on_month_end_and_second_cycle(self):
        bars = self.bars()
        result, counts = self.replay(bars)
        self.assertEqual([r["session_date"] for r in result.trades], [b.session_date for b in bars if b != bars[4]])
        self.assertEqual(counts, dict(attempted=2, filled=2, rejected=0, unavailable=0))

    def test_original_rejection_never_creates_chance(self):
        _, counts = self.replay(self.bars(), initial_cash={"KRW": Decimal(1)})
        self.assertEqual(counts["attempted"], 0)

    def test_skipped_market_open_not_deferred(self):
        _, counts = self.replay(self.bars(), extra_sessions=[date(2025, 1, 1)])
        self.assertEqual(counts["attempted"], 0)

    def test_failed_monthly_attempt_is_consumed(self):
        for change, expected in [({"volume": Decimal(0)}, "unavailable"), ({"adv20": Decimal(1)}, "rejected")]:
            with self.subTest(expected):
                bars = self.bars()
                bars[2] = replace(bars[2], **change)
                _, counts = self.replay(bars)
                self.assertEqual(counts["attempted"], 1)
                self.assertEqual(counts[expected], 1)

    def test_independent_market_open_after_december_sale(self):
        domestic = date(2025, 1, 2)
        american = date(2025, 1, 3)
        sessions = SimpleNamespace(kr_sessions=[domestic], us_sessions=[american])
        kernel = MonthlyReentry(engine_strategy, sessions)
        self.assertEqual(kernel.opens["KR"], kernel.opens["GOLD"])
        self.assertEqual(set(kernel.opens["US"].values()), {american})
        self.assertEqual(set(kernel.opens["GOLD"].values()), {domestic})

    def test_both_paths_joint_win_counts_all_planned_units(self):
        def value(won):
            return dict(status="passed", delta={"cagr": Decimal(".01"), "mdd": Decimal(".01") if won else 0,
                                                "trades": 0, "average_investment_fraction": 0},
                        reentry=dict(attempted=1, filled=1, rejected=0, unavailable=0), improved=won)
        rows = [dict(paths={"high": value(True), "low": value(True)}),
                dict(paths={"high": value(True), "low": value(False)}),
                dict(paths={p: dict(status="rejected") for p in ("high", "low")}), dict(paths={})]
        summary = summarize(rows, ["high", "low"])
        self.assertEqual((summary["improved_both_paths"], summary["planned"]), (1, 4))
        self.assertEqual(summary["paths"]["high"]["counts"], dict(passed=2, rejected=1, r1_failed=1))

    def test_no_passed_comparisons_preserves_denominator_and_null_medians(self):
        result = summarize([dict(paths={"high": {"status": "rejected"}})], ["high"])
        self.assertEqual(result["planned"], 1)
        self.assertEqual(result["improved_both_paths"], 0)
        self.assertTrue(all(v is None for v in result["paths"]["high"]["median_delta"].values()))
        self.assertEqual(result["paths"]["high"]["reentry"]["filled"], 0)

    def test_delisting_exit_at_last_tradable_session_has_no_reentry_opportunity(self):
        from src.research.etf.engine_lifecycle import DelistingEvent
        from tools.research.etf_reentry_sample import history
        bars = self.bars()[:2]
        result, counts = self.replay(bars, extra_sessions=[date(2025, 1, 2)],
                                     delistings=[DelistingEvent("KR", bars[0].open_at, bars[-1].session_date,
                                                                bars[-1].session_date + timedelta(days=1), "a" * 64)])
        self.assertEqual(result.trades[-1]["reason"], "delisting_liquidation")
        self.assertEqual(history(result, bars, "KR")["remaining_sessions"], 0)
        self.assertEqual(counts["attempted"], 0)

    def test_settled_sale_converted_before_monthly_open_can_reject_order(self):
        from src.research.etf.engine import run_engine
        from src.research.etf.engine_types import FxRate, Instrument
        from tests.test_research_etf_engine import at, rules
        days = [date.fromisoformat(s) for s in (
            "2024-12-26", "2024-12-27", "2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03")]
        bars = [bar(day=d, price=p) for d, p in zip(days, ["100", "110", "100", "100", "100", "100"])]
        kernel = MonthlyReentry(engine_strategy, SimpleNamespace(kr_sessions=days, us_sessions=[]))
        result = run_engine([Instrument("KR", "KR", "KRW"), Instrument("US", "US", "USD")],
                            bars, {"KR": rules(), "US": rules()}, initial_cash={"KRW": Decimal("1000")},
                            commission_rate=0, fx_cost=0, fx_rates=[FxRate(at(days[0]), Decimal("1000"))],
                            conversion_times=[at(days[2]), at(days[4])], domestic_settlement="t_plus_2",
                            kr_sessions=days, kr_settlement_session_times={days[3]: at(days[3], 8)},
                            _candidate_provider=kernel.candidate_for_bar)
        self.assertEqual([e["amount"] for e in result.cash_events if e["kind"] == "fx_conversion"],
                         [Decimal(0), Decimal(".55")])
        self.assertEqual([r["reason"] for r in result.rejections], ["insufficient_cash"])
        self.assertEqual(kernel.counts(result), dict(attempted=1, filled=0, rejected=1, unavailable=0))
