from copy import deepcopy
from datetime import date
from decimal import Decimal, localcontext
from fractions import Fraction
import unittest

from src.research.etf.kodex_core.ledgers import run_split
from tests.test_research_etf_kodex_core import CONFIG, J, build, case2, setting, weekdays
from tools.research.kodex_order_size import BuyBudget, RATES, check_budgets, improvement, run_sized_split


class OrderSizeTest(unittest.TestCase):
    def test_one_ninth_reproduces_engine_state_and_events(self):
        data = case2()
        with localcontext(prec=80):
            for path in ("O-H-L-C", "O-L-H-C"):
                settings = setting(mode="M", path=path)
                baseline = run_split(data, settings)
                partial = run_sized_split(data, settings, RATES[0])
                self.assertEqual(partial.detail, baseline.detail)
                for name in ("trades", "cash_events", "rejections", "valuations"):
                    self.assertEqual(getattr(partial.result, name), getattr(baseline.result, name))

    def test_larger_budget_keeps_nine_levels_and_rejects_unaffordable_order(self):
        days = weekdays(J(2), J(8))
        data = build(days, dict(zip(days, map(str, (10000, 9700, 9400, 9100, 8800)), strict=True)))
        settings = setting(mode="M")
        with localcontext(prec=80):
            rate = Fraction(1, 3)
            run = run_sized_split(data, settings, rate)
            self.assertEqual(settings.config.levels, 9)
            self.assertEqual(run.result.trades[0]["quantity"], 666)
            self.assertEqual([row["level"] for row in run.result.trades], [1, 2, 3])
            self.assertEqual([(row["level"], row["reason"]) for row in run.result.rejections],
                             [(4, "insufficient_cash"), (4, "insufficient_cash")])
            order = BuyBudget(Decimal(20_000_000), rate, settings.commission)
            self.assertEqual(check_budgets(run.result, data.bars, order), 1)

    def test_monthly_reentry_uses_updated_split_valuation(self):
        data = case2()
        with localcontext(prec=80):
            rate = Fraction(1, 4)
            settings = setting(mode="M")
            run = run_sized_split(data, settings, rate)
            order = BuyBudget(Decimal(20_000_000), rate, settings.commission)
            self.assertEqual(check_budgets(run.result, data.bars, order), 3)
            reentry, = [row for row in run.result.trades if row["reason"] == "monthly_reentry"]
            self.assertEqual(reentry["session_date"], date(2025, 3, 4))
            budget = next(event["budgets"]["069500"] for event in reversed(run.result.cash_events)
                          if event["kind"] == "budget_refresh" and event["at"] <= reentry["at"])
            cost = reentry["price"] * reentry["quantity"] + reentry["fee"]
            self.assertLessEqual(cost, budget)
            self.assertGreater(cost + reentry["price"], budget)

    def test_one_third_fills_exact_fee_included_boundary(self):
        data = build(weekdays(J(2), J(6)), {J(2): "10000"})
        capital = {"split": Decimal("150022.5"), "core": Decimal(80_000_000), "hold": Decimal("80150022.5")}
        settings = setting(mode="M", config=CONFIG.with_changes(capital=capital))
        run = run_sized_split(data, settings, Fraction(1, 3))
        self.assertEqual(run.result.trades[0]["quantity"], 5)

    def test_budget_audit_detects_lost_refresh_and_one_share_short(self):
        data = case2()
        settings = setting(mode="M")
        rate = Fraction(1, 4)
        run = run_sized_split(data, settings, rate)
        order = BuyBudget(Decimal(20_000_000), rate, settings.commission)
        missing = deepcopy(run.result)
        missing.cash_events.pop(max(i for i, row in enumerate(missing.cash_events) if row["kind"] == "budget_refresh"))
        with self.assertRaisesRegex(ValueError, "registered monthly opens"):
            check_budgets(missing, data.bars, order)
        short = deepcopy(run.result)
        buy = next(row for row in short.cash_events if row["kind"] == "trade" and row["side"] == "BUY")
        buy["quantity"] -= 1
        buy["fee"] = buy["price"] * buy["quantity"] * settings.commission
        with self.assertRaisesRegex(ValueError, "buy quantity"):
            check_budgets(short, data.bars, order)

    def test_diagnostic_requires_both_paths_and_does_not_claim_hold_edge(self):
        def point(base, stress):
            flags = {"start_holds": [], "start_failures": [], "later_holds": []}
            paths = {path: {**flags, "base": {"delta_cagr": base}, "stress": {"delta_cagr": stress}}
                     for path in ("O-H-L-C", "O-L-H-C")}
            return {"paths": paths}

        reference = point("-0.011", "-0.013")
        improved = point("-0.0085", "-0.012")
        self.assertTrue(improvement(improved, reference)["passes"])
        improved["paths"]["O-L-H-C"]["stress"]["delta_cagr"] = "-0.013"
        self.assertFalse(improvement(improved, reference)["passes"])
        improved["paths"]["O-L-H-C"]["stress"] = None
        self.assertIsNone(improvement(improved, reference)["passes"])
