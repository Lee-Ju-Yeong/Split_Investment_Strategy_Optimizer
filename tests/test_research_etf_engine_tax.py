"""세금 문서 사례를 독립 손계산한 기대값. 외부 시세/법률/DB를 사용하지 않는다."""
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
import unittest
from unittest.mock import patch

from src.research.etf.engine_tax import KR_TAX_CATEGORY_ALIASES, TaxLedger
from src.research.etf.engine_types import Instrument, Position

D = Decimal


def at(day):
    return datetime.fromisoformat(day + "T01:00:00+00:00")


def instrument(symbol="A", market="US", category="unknown"):
    return Instrument(symbol, market, "USD" if market == "US" else "KRW", category)


def position(symbol="A", quantity=100, price="100", day="2025-01-02", level=1, fee="0"):
    return Position(symbol, level, quantity, D(price), D(fee), D(".1"), at(day))


def synthetic():
    return TaxLedger("after_tax", synthetic=True)


class TaxLedgerUSHandCalculationTests(unittest.TestCase):
    def test_account_annual_netting_and_single_deduction(self):
        # A: +500만원, B: -100만원 => (400만-250만)*22%=33만원.
        ledger, a, b = synthetic(), instrument(), instrument("B")
        pa, pb = position(), position("B")
        ledger.buy(a, pa, fx_rate=D("1000"))
        ledger.buy(b, pb, fx_rate=D("1000"))
        self.assertEqual(ledger.sell(a, pa, D("150"), D("0"), at("2025-02-01"), fx_rate=D("1000")), D("0"))
        ledger.sell(b, pb, D("90"), D("0"), at("2025-02-02"), fx_rate=D("1000"))
        before = ledger.snapshot()
        self.assertEqual(before["us_realized_krw"], {2025: D("4000000")})
        self.assertEqual(before["unpaid_tax_krw"], D("0"))
        ledger.advance(at("2026-01-01"))
        row = ledger.snapshot()["annual_us_tax"][2025]
        self.assertEqual((row["income_tax_krw"], row["local_tax_krw"]), (D("300000"), D("30000")))
        self.assertEqual(ledger.snapshot()["tax_reserve_krw"], D("330000"))

    def test_each_purchase_fx_and_fees_enter_krw_average(self):
        # 10*100*1000+10수수료*1000=101만, 10*80*1200+5*1200=96.6만.
        # 계좌 평균 원가 98,800/주. 10주 100 매도 - 수수료2, 환율1100 => 109.78만.
        # 따라서 매도 차수가 어느 것이든 109.78만-98.8만 = 109,800원.
        ledger, asset = synthetic(), instrument()
        first = position(quantity=10, fee="10")
        second = position(quantity=10, price="80", day="2025-01-03", level=2, fee="5")
        ledger.buy(asset, first, fx_rate=D("1000"))
        ledger.buy(asset, second, fx_rate=D("1200"))
        ledger.sell(asset, second, D("100"), D("2"), at("2025-01-04"), fx_rate=D("1100"))
        snap = ledger.snapshot()
        self.assertEqual(snap["us_realized_krw"][2025], D("109800"))
        self.assertEqual(snap["holdings"]["A"]["krw_cost"], D("988000"))
        self.assertEqual(snap["holdings"]["A"]["foreign_cost"], D("907.5"))

    def test_fx_only_gain_is_realized(self):
        # 같은 달러 가격: 100*100달러*(1300-1000)=300만원, 공제 후 11만원.
        ledger, asset, lot = synthetic(), instrument(), position()
        ledger.buy(asset, lot, fx_rate=D("1000"))
        ledger.sell(asset, lot, D("100"), D("0"), at("2025-02-01"), fx_rate=D("1300"))
        snap = ledger.snapshot()
        self.assertEqual(snap["us_realized_krw"][2025], D("3000000"))
        self.assertEqual(snap["estimated_us_tax"][2025]["income_tax_krw"], D("100000"))
        self.assertEqual(snap["estimated_us_tax"][2025]["local_tax_krw"], D("10000"))

    def test_december_trade_is_recognized_at_january_settlement(self):
        # 12/31 거래를 가정해도 1/2 결제 사건 호출 전에는 미실현, 귀속은 2026.
        ledger, asset = synthetic(), instrument()
        lot = position(day="2025-12-30")
        ledger.buy(asset, lot, fx_rate=D("1000"), settled_at=at("2025-12-31"))
        self.assertEqual(ledger.snapshot()["us_realized_krw"], {})
        ledger.sell(asset, lot, D("150"), D("0"), at("2026-01-02"),
                    settled_at=at("2026-01-02"), fx_rate=D("1000"))
        self.assertEqual(ledger.snapshot()["us_realized_krw"], {2026: D("5000000")})
        self.assertEqual(ledger.snapshot()["annual_us_tax"], {})

    def test_losses_do_not_carry_to_next_year(self):
        ledger, asset = synthetic(), instrument()
        first = position()
        ledger.buy(asset, first, fx_rate=D("1000"))
        ledger.sell(asset, first, D("90"), D("0"), at("2025-02-01"), fx_rate=D("1000"))
        second = position(day="2026-01-02")
        ledger.buy(asset, second, fx_rate=D("1000"))
        ledger.sell(asset, second, D("130"), D("0"), at("2026-02-01"), fx_rate=D("1000"))
        ledger.advance(at("2027-01-01"))
        self.assertEqual(ledger.snapshot()["unpaid_tax_krw"], D("110000"))

    def test_next_year_payment_is_separate_from_sale_withholding(self):
        ledger, asset, lot = synthetic(), instrument(), position()
        ledger.buy(asset, lot, fx_rate=D("1000"))
        self.assertEqual(ledger.sell(asset, lot, D("150"), D("0"), at("2025-03-01"), fx_rate=D("1000")), D("0"))
        ledger.advance(at("2026-01-01"))
        row = ledger.snapshot()["annual_us_tax"][2025]
        self.assertEqual(str(row["income_due_date"]), "2026-05-31")
        self.assertEqual(str(row["local_due_date"]), "2026-07-31")
        self.assertFalse(row["holiday_adjusted"])
        allocation = ledger.pay_tax(D("500000"), at("2026-06-01"))
        self.assertEqual(allocation, [{"year": 2025, "component": "income", "amount_krw": D("500000")}])
        self.assertEqual(ledger.snapshot()["unpaid_tax_krw"], D("50000"))
        ledger.pay_tax(D("50000"), at("2026-07-31"))
        self.assertEqual(ledger.snapshot()["unpaid_tax_krw"], D("0"))

    def test_partial_payment_and_excess_payment_are_atomic(self):
        ledger, asset, lot = synthetic(), instrument(), position()
        ledger.buy(asset, lot, fx_rate=D("1000"))
        ledger.sell(asset, lot, D("150"), D("0"), at("2025-03-01"), fx_rate=D("1000"))
        before = ledger.snapshot()
        with self.assertRaises(ValueError):
            ledger.pay_tax(D("550001"), at("2026-05-31"))
        self.assertEqual(ledger.snapshot(), before)
        ledger.pay_tax(D("525000"), at("2026-05-31"))
        self.assertEqual(ledger.snapshot()["unpaid_tax_krw"], D("25000"))


class TaxLedgerValuationSummaryTests(unittest.TestCase):
    def test_current_realized_gain_adds_estimate_without_withholding(self):
        # 100주*(150-100)*1000=500만원; (500만-250만)*22%=55만원.
        ledger, asset, lot = synthetic(), instrument(), position()
        ledger.buy(asset, lot, fx_rate=D("1000"))
        self.assertEqual(ledger.valuation_summary()["total_tax_liability_krw"], D("0"))
        withheld = ledger.sell(asset, lot, D("150"), D("0"), at("2025-03-01"), fx_rate=D("1000"))
        self.assertEqual(withheld, D("0"))
        self.assertEqual(ledger.valuation_summary(), {
            "unpaid_tax_krw": D("0"), "estimated_tax_krw": D("550000"),
            "total_tax_liability_krw": D("550000"),
        })
        # 미래 손익은 쓰지 않으며, 새 거래 없이 시간만 전진하면 추정액은 같다.
        ledger.advance(at("2025-10-01"))
        self.assertEqual(ledger.valuation_summary()["estimated_tax_krw"], D("550000"))
        self.assertEqual(ledger.snapshot()["tax_reserve_krw"], D("0"))

    def test_later_realized_losses_reduce_estimate_until_zero(self):
        # +500만원 후 -100만원이면 33만원, 추가 -200만원이면 공제 이하로 0.
        ledger = synthetic()
        assets = [instrument(symbol) for symbol in ("A", "B", "C")]
        lots = [position(symbol) for symbol in ("A", "B", "C")]
        for asset, lot in zip(assets, lots):
            ledger.buy(asset, lot, fx_rate=D("1000"))
        for asset, lot, price, day, expected in zip(
                assets, lots, ("150", "90", "80"),
                ("2025-03-01", "2025-04-01", "2025-05-01"), ("550000", "330000", "0")):
            ledger.sell(asset, lot, D(price), D("0"), at(day), fx_rate=D("1000"))
            summary = ledger.valuation_summary()
            self.assertEqual(summary["estimated_tax_krw"], D(expected))
            self.assertEqual(summary["total_tax_liability_krw"], D(expected))
            self.assertEqual(summary["unpaid_tax_krw"], D("0"))

    def test_year_finalization_keeps_total_liability_continuous(self):
        ledger, asset, lot = synthetic(), instrument(), position()
        ledger.buy(asset, lot, fx_rate=D("1000"))
        ledger.sell(asset, lot, D("150"), D("0"),
                    datetime.fromisoformat("2025-12-31T14:59:59+00:00"), fx_rate=D("1000"))
        before = ledger.valuation_summary()
        # 기존 귀속시각인 한국시간 1/1 00:00에 55만원 추정액이 확정 미납으로 이동.
        ledger.advance(datetime.fromisoformat("2025-12-31T15:00:00+00:00"))
        after = ledger.valuation_summary()
        self.assertEqual(before["estimated_tax_krw"], D("550000"))
        self.assertEqual(after["unpaid_tax_krw"], D("550000"))
        self.assertEqual(after["estimated_tax_krw"], D("0"))
        self.assertEqual(after["total_tax_liability_krw"], before["total_tax_liability_krw"])

    def test_unpaid_and_current_estimate_combine_and_payment_is_neutral(self):
        # 전년 미납55만원 + 올해 (400만-250만)*22%=33만원 -> 총88만원.
        ledger, asset = synthetic(), instrument()
        first, second = position(), position(day="2026-01-02")
        ledger.buy(asset, first, fx_rate=D("1000"))
        ledger.sell(asset, first, D("150"), D("0"), at("2025-03-01"), fx_rate=D("1000"))
        ledger.buy(asset, second, fx_rate=D("1000"))
        ledger.sell(asset, second, D("140"), D("0"), at("2026-03-01"), fx_rate=D("1000"))
        before = ledger.valuation_summary()
        self.assertEqual(before, {"unpaid_tax_krw": D("550000"), "estimated_tax_krw": D("330000"),
                                  "total_tax_liability_krw": D("880000")})
        cash, paid = D("20000000"), D("525000")
        equity_before = cash - before["total_tax_liability_krw"]
        ledger.pay_tax(paid, at("2026-05-31"))
        after = ledger.valuation_summary()
        self.assertEqual(after["unpaid_tax_krw"], D("25000"))
        self.assertEqual(after["estimated_tax_krw"], D("330000"))
        self.assertEqual(after["total_tax_liability_krw"], D("355000"))
        self.assertEqual(cash - paid - after["total_tax_liability_krw"], equity_before)
        ledger.pay_tax(D("25000"), at("2026-07-31"))
        self.assertEqual(ledger.valuation_summary()["total_tax_liability_krw"], D("330000"))

    def test_empty_and_pre_tax_summary_are_all_zero(self):
        zero = {"unpaid_tax_krw": D("0"), "estimated_tax_krw": D("0"),
                "total_tax_liability_krw": D("0")}
        self.assertEqual(synthetic().valuation_summary(), zero)
        ledger, asset, lot = TaxLedger(), instrument(), position()
        ledger.buy(asset, lot, fx_rate=D("1000"))
        ledger.sell(asset, lot, D("150"), D("0"), at("2025-03-01"), fx_rate=D("1000"))
        self.assertEqual(ledger.snapshot()["estimated_us_tax"][2025]["income_tax_krw"], D("500000"))
        self.assertEqual(ledger.valuation_summary(), zero)
        ledger.advance(at("2026-01-01"))
        self.assertEqual(ledger.snapshot()["annual_us_tax"][2025]["income_tax_krw"], D("500000"))
        self.assertEqual(ledger.valuation_summary(), zero)

    def test_summary_does_not_copy_full_ledger_or_change_state(self):
        ledger, asset, lot = synthetic(), instrument(), position()
        ledger.buy(asset, lot, fx_rate=D("1000"))
        ledger.sell(asset, lot, D("150"), D("0"), at("2025-03-01"), fx_rate=D("1000"))
        before = ledger.snapshot()
        with patch("src.research.etf.engine_tax.deepcopy", side_effect=AssertionError("전체 복사 금지")), \
                patch.object(ledger, "snapshot", side_effect=AssertionError("전체 원장 조회 금지")):
            summary = ledger.valuation_summary()
            self.assertTrue(all(isinstance(value, Decimal) for value in summary.values()))
            summary["total_tax_liability_krw"] = D("999")
            self.assertEqual(ledger.valuation_summary()["total_tax_liability_krw"], D("550000"))
        self.assertEqual(ledger.snapshot(), before)


class TaxLedgerDomesticHandCalculationTests(unittest.TestCase):
    def test_strategy_gain_can_be_tax_loss(self):
        # 전략 2차 80->88은 +10%이나 계좌 평균90 대비 -2이므로 세액0.
        ledger, asset = synthetic(), instrument(market="KR", category="기타")
        first, second = position(price="100"), position(price="80", level=2, day="2025-01-03")
        ledger.buy(asset, first, tax_basis=D("100"))
        ledger.buy(asset, second, tax_basis=D("80"))
        self.assertEqual(ledger.sell(asset, second, D("88"), D("0"), at("2025-01-04"), tax_basis=D("88")), D("0"))
        self.assertEqual(ledger.snapshot()["holdings"]["A"]["average_price"], D("90"))
        self.assertEqual(ledger.snapshot()["holdings"]["A"]["effective_tax_basis"], D("90"))

    def test_minimum_of_actual_and_tax_basis_gain(self):
        # 1000좌: 가격이익20/과표이익5 ->5000*.154=770, 반대 ->2000*.154=308.
        for sell_price, sell_basis, expected in (("120", "105", "770"), ("102", "120", "308")):
            with self.subTest(price=sell_price):
                ledger, asset, lot = synthetic(), instrument(market="KR", category="기타"), position(quantity=1000)
                ledger.buy(asset, lot, tax_basis=D("100"))
                self.assertEqual(ledger.sell(asset, lot, D(sell_price), D("0"), at("2025-02-01"), tax_basis=D(sell_basis)), D(expected))

    def test_distribution_carry_retains_untaxed_nineteen(self):
        # 문서 3.4 사례를 1000좌로 확대: 분배 과세1000, 매도 과세20000, 총세액3234.
        ledger, asset, lot = synthetic(), instrument(market="KR", category="기타"), position(quantity=1000)
        ledger.buy(asset, lot, tax_basis=D("100"))
        receipt = ledger.accrue_dividend(asset, D("1000"), at("2025-02-01"), at("2025-02-05"), tax_basis=D("120"))
        self.assertEqual(ledger.snapshot()["holdings"]["A"]["effective_tax_basis"], D("100"))
        self.assertEqual(ledger.snapshot()["receivable_net"]["KRW"], D("846"))
        self.assertEqual(ledger.dividend(asset, D("1000"), at("2025-02-05"), receivable_id=receipt), D("154"))
        self.assertEqual(ledger.sell(asset, lot, D("120"), D("0"), at("2025-02-06"), tax_basis=D("120")), D("3080"))
        self.assertEqual(ledger.snapshot()["financial_income_krw"][2025], D("21000"))

    def test_multiple_distributions_and_negative_basis_gain(self):
        # B100, G=-10 분배5 => B95, 이후 T110 분배2 => G15/과세2, B95 유지.
        ledger, asset, lot = synthetic(), instrument(market="KR", category="기타"), position(quantity=1000)
        ledger.buy(asset, lot, tax_basis=D("100"))
        self.assertEqual(ledger.dividend(asset, D("5000"), at("2025-02-01"), tax_basis=D("90")), D("0"))
        self.assertEqual(ledger.snapshot()["holdings"]["A"]["effective_tax_basis"], D("95"))
        self.assertEqual(ledger.dividend(asset, D("2000"), at("2025-03-01"), tax_basis=D("110")), D("308"))
        self.assertEqual(ledger.snapshot()["holdings"]["A"]["effective_tax_basis"], D("95"))
        self.assertEqual(ledger.sell(asset, lot, D("120"), D("0"), at("2025-04-01"), tax_basis=D("120")), D("3080"))

    def test_domestic_loss_does_not_offset_other_sale(self):
        ledger, asset = synthetic(), instrument(market="KR", category="기타")
        one, two = position(), position(level=2)
        ledger.buy(asset, one, tax_basis=D("100"))
        ledger.buy(asset, two, tax_basis=D("100"))
        ledger.sell(asset, one, D("80"), D("0"), at("2025-02-01"), tax_basis=D("80"))
        self.assertEqual(ledger.sell(asset, two, D("120"), D("0"), at("2025-02-02"), tax_basis=D("120")), D("308"))

    def test_domestic_equity_and_krx_gold_sale_have_no_tax(self):
        for market, category in (("KR", "국내주식형"), ("GOLD", "unknown"), ("GOLD", "기타")):
            with self.subTest(market=market, category=category):
                ledger, asset, lot = synthetic(), instrument(market=market, category=category), position()
                ledger.buy(asset, lot)
                self.assertEqual(ledger.sell(asset, lot, D("200"), D("0"), at("2025-02-01")), D("0"))

    def test_recognized_category_aliases_and_rounding(self):
        for name, canonical in KR_TAX_CATEGORY_ALIASES.items():
            with self.subTest(name=name):
                ledger, asset, lot = synthetic(), instrument(market="KR", category=name), position(quantity=1)
                ledger.buy(asset, lot, tax_basis=D("100"))
                # 10*.154=1.54원 -> 연구의 원 미만 절사 가정으로 1원.
                tax = ledger.sell(asset, lot, D("110"), D("0"), at("2025-02-01"), tax_basis=D("110"))
                self.assertEqual(tax, D("1") if canonical == "기타" else D("0"))


class TaxLedgerDividendHandCalculationTests(unittest.TestCase):
    def test_ex_date_receivable_survives_sale_until_payment(self):
        # 배당락 1000->990에 대해 10분배를 미수금에 기록: 세전1000, 세후998.46/주.
        ledger, asset, lot = synthetic(), instrument(market="KR", category="국내주식형"), position(quantity=100, price="1000")
        ledger.buy(asset, lot)
        receipt = ledger.accrue_dividend(asset, D("1000"), at("2025-12-30"), at("2026-01-05"))
        self.assertEqual(D("99000") + ledger.snapshot()["receivable_net"]["KRW"], D("99846"))
        self.assertEqual(ledger.snapshot()["financial_income_krw"], {})
        ledger.sell(asset, lot, D("990"), D("0"), at("2025-12-31"))
        self.assertEqual(ledger.dividend(asset, D("1000"), at("2026-01-05"), receivable_id=receipt), D("154"))
        self.assertEqual(ledger.snapshot()["receivable_net"]["KRW"], D("0"))
        self.assertEqual(ledger.snapshot()["financial_income_krw"], {2026: D("1000")})

    def test_us_distribution_withholding_and_financial_income(self):
        # $10*.15=$1.5 미국 원천징수, 국내 추가0, 금융소득은 $10*1300=13000원.
        ledger, asset, lot = synthetic(), instrument(), position()
        ledger.buy(asset, lot, fx_rate=D("1000"))
        self.assertEqual(ledger.dividend(asset, D("10"), at("2025-02-01"), fx_rate=D("1300")), D("1.50"))
        self.assertEqual(ledger.snapshot()["financial_income_krw"], {2025: D("13000")})
        self.assertEqual(ledger.snapshot()["us_realized_krw"], {})

    def test_gross_income_threshold_is_reported_without_extra_tax(self):
        ledger, asset, lot = synthetic(), instrument(market="KR", category="국내주식형"), position()
        ledger.buy(asset, lot)
        self.assertEqual(ledger.dividend(asset, D("21000000"), at("2025-02-01")), D("3234000"))
        self.assertEqual(ledger.snapshot()["financial_income_over_threshold"], [2025])


class TaxLedgerValidationTests(unittest.TestCase):
    def test_pre_tax_default_keeps_records_without_cash_taxes(self):
        ledger, asset, lot = TaxLedger(), instrument(market="KR", category="기타"), position()
        ledger.buy(asset, lot)
        self.assertEqual(ledger.dividend(asset, D("1000"), at("2025-02-01")), D("0"))
        self.assertEqual(ledger.sell(asset, lot, D("120"), D("0"), at("2025-03-01")), D("0"))
        snap = ledger.snapshot()
        self.assertFalse(snap["actual_after_tax_approved"])
        self.assertTrue(snap["missing_inputs"])
        self.assertEqual(snap["unpaid_tax_krw"], D("0"))
        self.assertEqual(len(snap["events"]), 3)

    def test_after_tax_requires_explicit_synthetic_mode(self):
        for kwargs in ({"mode": "after_tax"}, {"mode": "other"}, {"synthetic": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TaxLedger(**kwargs)

    def test_missing_basis_category_and_adjusted_prices_fail_closed(self):
        for asset in (instrument(market="KR", category="기타"), instrument(market="KR"),
                      replace(instrument(), price_basis="adjusted")):
            with self.subTest(asset=asset), self.assertRaises(ValueError):
                synthetic().buy(asset, position())
        ledger, asset, lot = synthetic(), instrument(market="KR", category="기타"), position()
        ledger.buy(asset, lot, tax_basis=D("100"))
        before = ledger.snapshot()
        for action in (lambda: ledger.sell(asset, lot, D("110"), D("0"), at("2025-02-01")),
                       lambda: ledger.dividend(asset, D("10"), at("2025-02-01"))):
            with self.assertRaises(ValueError):
                action()
            self.assertEqual(ledger.snapshot(), before)

    def test_split_only_adjustment_does_not_imply_total_return_price(self):
        # split만 보정한 가격은 현금 분배를 포함하지 않으므로 별도 분배 원장이 가능하다.
        ledger, lot = synthetic(), position()
        asset = replace(instrument(), price_basis="split_adjusted")
        ledger.buy(asset, lot)
        self.assertEqual(ledger.dividend(asset, D("10"), at("2025-02-01")), D("1.50"))

    def test_invalid_tax_basis_buy_fx_and_dividend_schedule_are_atomic(self):
        for invalid in (D("-1"), D("NaN"), D("Infinity"), True, 1.5):
            with self.subTest(value=invalid):
                ledger, asset = synthetic(), instrument(market="KR", category="기타")
                before = ledger.snapshot()
                with self.assertRaises(ValueError):
                    ledger.buy(asset, position(), tax_basis=invalid)
                self.assertEqual(ledger.snapshot(), before)
        ledger, asset, lot = synthetic(), instrument(), position()
        for invalid in (D("0"), D("-1"), D("NaN")):
            with self.assertRaises(ValueError):
                ledger.buy(asset, lot, fx_rate=invalid)
        ledger.buy(asset, lot)
        before = ledger.snapshot()
        with self.assertRaises(ValueError):
            ledger.accrue_dividend(asset, D("1"), at("2025-02-02"), at("2025-02-01"))
        self.assertEqual(ledger.snapshot(), before)

    def test_invalid_numeric_inputs_do_not_mutate_ledger(self):
        ledger, asset, lot = synthetic(), instrument(), position()
        ledger.buy(asset, lot)
        invalid = (D("-1"), D("NaN"), D("Infinity"), True, 1.5)
        for value in invalid:
            with self.subTest(value=value):
                for action in (lambda: ledger.sell(asset, lot, value, D("0"), at("2025-02-01")),
                               lambda: ledger.sell(asset, lot, D("110"), value, at("2025-02-01")),
                               lambda: ledger.dividend(asset, value, at("2025-02-01")),
                               lambda: ledger.pay_tax(value, at("2025-02-01"))):
                    before = ledger.snapshot()
                    with self.assertRaises(ValueError):
                        action()
                    self.assertEqual(ledger.snapshot(), before)
        for fx in (D("0"), D("-1"), D("NaN")):
            with self.assertRaises(ValueError):
                ledger.sell(asset, lot, D("110"), D("0"), at("2025-02-01"), fx_rate=fx)

    def test_duplicate_buy_and_oversell_are_rejected(self):
        ledger, asset, lot = synthetic(), instrument(), position(quantity=10)
        ledger.buy(asset, lot)
        for action in (lambda: ledger.buy(asset, lot),
                       lambda: ledger.sell(asset, replace(lot, quantity=11), D("110"), D("0"), at("2025-02-01")),
                       lambda: ledger.sell(asset, replace(lot, entry_price=D("90")), D("110"), D("0"), at("2025-02-01"))):
            with self.assertRaises(ValueError):
                action()
        ledger.sell(asset, replace(lot, quantity=6), D("110"), D("0"), at("2025-02-01"))
        with self.assertRaises(ValueError):
            ledger.sell(asset, replace(lot, quantity=5), D("110"), D("0"), at("2025-02-02"))
        ledger.sell(asset, replace(lot, quantity=4), D("110"), D("0"), at("2025-02-02"))
        self.assertEqual(ledger.snapshot()["holdings"]["A"]["quantity"], 0)
        with self.assertRaises(ValueError):
            ledger.sell(asset, lot, D("110"), D("0"), at("2025-02-03"))

    def test_clock_is_monotonic_and_future_settlement_is_not_leaked(self):
        ledger, asset, lot = synthetic(), instrument(), position()
        ledger.buy(asset, lot)
        before = ledger.snapshot()
        with self.assertRaises(ValueError):
            ledger.sell(asset, lot, D("150"), D("0"), at("2025-12-31"), settled_at=at("2026-01-02"))
        self.assertEqual(ledger.snapshot(), before)
        ledger.advance(at("2025-03-01"))
        for action in (lambda: ledger.advance(at("2025-02-01")),
                       lambda: ledger.snapshot(at=at("2025-02-01")),
                       lambda: ledger.sell(asset, lot, D("150"), D("0"), at("2025-02-01"))):
            with self.assertRaises(ValueError):
                action()
        with self.assertRaises(ValueError):
            ledger.advance(datetime(2025, 4, 1))

    def test_future_buy_and_sale_before_purchase_are_rejected(self):
        ledger, asset, lot = synthetic(), instrument(), position()
        with self.assertRaises(ValueError):
            ledger.buy(asset, lot, settled_at=at("2025-01-01"))
        ledger.buy(asset, lot)
        with self.assertRaises(ValueError):
            ledger.sell(asset, lot, D("110"), D("0"), at("2025-01-01"))

    def test_receivable_duplicate_early_or_wrong_payment_is_rejected(self):
        ledger, asset, lot = synthetic(), instrument(), position()
        ledger.buy(asset, lot)
        receipt = ledger.accrue_dividend(asset, D("100"), at("2025-02-01"), at("2025-02-05"), dividend_id="one")
        before = ledger.snapshot()
        for action in (lambda: ledger.accrue_dividend(asset, D("100"), at("2025-02-01"), at("2025-02-05")),
                       lambda: ledger.dividend(asset, D("100"), at("2025-02-02"), receivable_id=receipt),
                       lambda: ledger.dividend(asset, D("99"), at("2025-02-05"), receivable_id=receipt)):
            with self.assertRaises(ValueError):
                action()
            self.assertEqual(ledger.snapshot(), before)
        ledger.dividend(asset, D("100"), at("2025-02-05"), receivable_id=receipt)
        with self.assertRaises(ValueError):
            ledger.dividend(asset, D("100"), at("2025-02-05"), receivable_id=receipt)

    def test_snapshot_is_detached_and_financial_amounts_are_decimal(self):
        ledger, asset, lot = synthetic(), instrument(), position()
        ledger.buy(asset, lot)
        snap = ledger.snapshot()
        snap["holdings"]["A"]["quantity"] = 999
        snap["events"].clear()
        self.assertEqual(ledger.snapshot()["holdings"]["A"]["quantity"], 100)
        self.assertIsInstance(ledger.snapshot()["holdings"]["A"]["krw_cost"], D)
        self.assertEqual(len(ledger.snapshot()["events"]), 1)

    def test_unsupported_same_day_domestic_multi_sale_is_explicit_failure(self):
        ledger, asset = synthetic(), instrument(market="KR", category="기타")
        first, second = position(), position(level=2)
        ledger.buy(asset, first, tax_basis=D("100"))
        ledger.buy(asset, second, tax_basis=D("100"))
        ledger.sell(asset, first, D("110"), D("0"), at("2025-02-01"), tax_basis=D("110"))
        before = ledger.snapshot()
        with self.assertRaisesRegex(ValueError, "복수"):
            ledger.sell(asset, second, D("110"), D("0"), at("2025-02-01"), tax_basis=D("110"))
        self.assertEqual(ledger.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
