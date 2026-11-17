"""Independent M4 performance arithmetic, frontier and paired-calendar checks."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import unittest
from zoneinfo import ZoneInfo

from src.research.etf.baseline_report import (
    baseline_policies, comparison_report, paired_difference, performance, run_baseline_suite,
)
from src.research.etf.engine_types import EngineResult, Position
from test_research_etf_baselines import D, JUNE, JULY, KR, at, bar, fit, policy, run


def result_curve(values, *, holdings=None):
    rows = []
    for day, value in values:
        rows.append({"at": at(day, 6), "equity_krw": D(value),
                     "holdings_krw": {"KR": D(holdings or "0")},
                     "positions": [], "prices": {}, "fx_rate": D("1")})
    return EngineResult(valuations=rows)


def metrics(result, *, started_at=None, **options):
    defaults = dict(initial_equity_krw=D("1000"), started_at=started_at or at(),
                    ended_at=result.valuations[-1]["at"])
    return performance(result, [KR], **(defaults | options))


class PerformanceTests(unittest.TestCase):
    def test_actual_elapsed_year_cagr_and_calmar(self):
        start = at(date(2023, 1, 1))
        result = result_curve([(date(2023, 6, 1), "800"), (date(2024, 1, 1), "1100")])
        value = metrics(result, started_at=start)
        self.assertEqual(value.cagr, D(".1"))
        self.assertEqual(value.mdd, D("-.2"))
        self.assertEqual(value.calmar, D(".5"))
        self.assertEqual(value.underwater_days, D("365.25"))

    def test_initial_cost_drawdown_uses_initial_wealth_peak(self):
        value = metrics(result_curve([(JUNE, "990")]))
        self.assertEqual(value.mdd, D("-.01"))

    def test_increasing_cash_curve_has_no_calmar_denominator(self):
        value = metrics(result_curve([(JUNE, "1000"), (JULY, "1100")]))
        self.assertEqual(value.mdd, D("0"))
        self.assertIsNone(value.calmar)
        self.assertEqual(value.underwater_days, D("0"))

    def test_average_exposure_is_not_ratio_of_sums(self):
        value = metrics(result_curve([(JUNE, "1000"), (JULY, "2000")], holdings="500"))
        self.assertEqual(value.average_investment_fraction, D(".375"))

    def test_rejections_count_only_inside_evaluation_window(self):
        result = result_curve([(JUNE, "1000")])
        result.rejections = [{"at": at()}, {"at": at(JUNE + timedelta(days=1))}]
        self.assertEqual(metrics(result).rejected_orders, 1)

    def test_unrealized_price_pnl_uses_terminal_snapshot_not_later_state(self):
        result = run([bar(close=D("120"), high=D("120"))])
        result.positions = []
        self.assertEqual(metrics(result).unrealized_price_pnl_krw, D("100"))

    def test_no_terminal_valuation_is_invented(self):
        with self.assertRaisesRegex(ValueError, "explicit valuation"):
            metrics(result_curve([(JUNE, "1000")]), ended_at=at(JULY))

    def test_nonpositive_equity_and_invalid_calendar_rejected(self):
        for values in ([(JUNE, "0")], [(JULY, "1000"), (JUNE, "1000")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                metrics(result_curve(values))

    def test_delta_signs_reward_lower_drawdown_and_higher_cagr(self):
        base = metrics(result_curve([(JUNE, "800"), (JULY, "1000")]))
        better = replace(base, cagr=base.cagr + D(".01"), mdd=D("-.1"))
        delta = paired_difference(better, base)
        self.assertEqual(delta["delta_cagr"], D(".01"))
        self.assertEqual(delta["delta_mdd"], D(".1"))

    def test_dst_spring_and_fall_measure_elapsed_time_in_utc(self):
        zone = ZoneInfo("America/New_York")
        for start, end, hours in ((datetime(2024, 3, 9, 12, tzinfo=zone), datetime(2024, 3, 11, 12, tzinfo=zone), 47),
                                  (datetime(2024, 11, 2, 12, tzinfo=zone), datetime(2024, 11, 4, 12, tzinfo=zone), 49)):
            result = result_curve([(JUNE, "990")])
            result.valuations[0]["at"] = end
            local = metrics(result, started_at=start, ended_at=end)
            utc_result = EngineResult(valuations=[result.valuations[0] | {"at": end.astimezone(timezone.utc)}])
            universal = metrics(utc_result, started_at=start.astimezone(timezone.utc),
                                ended_at=end.astimezone(timezone.utc))
            with self.subTest(hours=hours):
                self.assertEqual(local, universal)
                self.assertEqual((local.underwater_days * 24).quantize(D(".000001")), D(hours))

    def test_repeated_dst_wall_hour_does_not_reverse_valuation_order(self):
        zone = ZoneInfo("America/New_York")
        result = result_curve([(JUNE, "990"), (JULY, "1000")])
        result.valuations[0]["at"] = datetime(2024, 11, 3, 1, 45, tzinfo=zone, fold=0)
        result.valuations[1]["at"] = datetime(2024, 11, 3, 1, 15, tzinfo=zone, fold=1)
        value = metrics(result, started_at=datetime(2024, 11, 3, 1, 0, tzinfo=zone, fold=0))
        self.assertEqual(value.cagr, D("0"))
        self.assertEqual(value.mdd, D("-.01"))


class FrontierTests(unittest.TestCase):
    def test_suite_has_fixed_five_exposures_and_buy_hold(self):
        policies = baseline_policies(fit())
        self.assertEqual(len(policies), 7)
        self.assertEqual([p.investment_fraction for key, p in policies.items() if key.startswith("frontier_")],
                         [D("0"), D(".25"), D(".5"), D(".75"), D("1")])
        self.assertEqual(policies["matched"].role, "primary")
        self.assertTrue(all(p.role == "reference" for name, p in policies.items() if name != "matched"))
        self.assertEqual(policies["buy_hold"].mode, "buy_hold")

    def test_full_rebalance_and_buy_hold_are_distinct_after_month_boundary(self):
        bars = [bar(high=D("200"), close=D("200")), bar(day=JULY, price="100")]
        suite = run_baseline_suite(fit(), [KR], bars, initial_cash={"KRW": D("1000")},
                                   evaluation_times=[at(JUNE, 7), at(JULY, 7)], commission_rate=D("0"))
        self.assertEqual(len(suite["buy_hold"][1].trades), 1)
        self.assertEqual(len(suite["frontier_100"][1].rejections), 1)
        self.assertEqual(suite["frontier_100"][1].rejections[0]["reason"], "insufficient_cash")

    def test_flat_prices_with_zero_costs_all_frontier_returns_are_zero(self):
        bars = [bar(), bar(day=JULY)]
        suite = run_baseline_suite(fit(), [KR], bars, initial_cash={"KRW": D("1000")},
                                   evaluation_times=[at(JUNE, 7), at(JULY, 7)], commission_rate=D("0"))
        report = comparison_report(run(bars), suite, [KR], initial_equity_krw=D("1000"),
                                   started_at=at(), ended_at=at(JULY, 7))
        self.assertEqual(report["baselines"]["matched"]["role"], "primary")
        for row in report["baselines"].values():
            self.assertEqual(row["performance"]["cagr"], D("0"))
            self.assertEqual(row["performance"]["mdd"], D("0"))
            self.assertFalse(row["joint_improvement"])
        self.assertFalse(report["statistical_success_evaluated"])

    def test_paired_calendar_mismatch_is_rejected(self):
        bars = [bar()]
        result = run(bars)
        different = run(bars, evaluation_times=[at(JUNE, 8)])
        with self.assertRaisesRegex(ValueError, "same evaluation"):
            comparison_report(result, {"matched": (policy(), different)}, [KR], initial_equity_krw=D("1000"),
                              started_at=at(), ended_at=at(JUNE, 7))

    def test_diagnostic_is_explicit_and_cannot_seed_main_suite(self):
        bars = [bar()]
        result = run(bars)
        suite = run_baseline_suite(fit(), [KR], bars, initial_cash={"KRW": D("1000")},
                                   strategy_result=result, diagnostic_end=at(JUNE, 7),
                                   evaluation_times=[at(JUNE, 7)], commission_rate=D("0"))
        self.assertEqual(suite["posthoc_diagnostic"][0].role, "diagnostic")
        with self.assertRaisesRegex(ValueError, "prospective"):
            baseline_policies(suite["posthoc_diagnostic"][0])


if __name__ == "__main__":
    unittest.main()
