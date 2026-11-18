"""M5 cash-availability, explicit delisting and FX-age hand calculations."""
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
import unittest

from src.research.etf.baseline_engine import run_baseline
from src.research.etf.baseline_policy import fixed_baseline
from src.research.etf.engine import _Engine, run_engine
from src.research.etf.engine_tax import TaxLedger
from src.research.etf.engine_lifecycle import DelistingEvent
from src.research.etf.engine_types import EngineResult, FxRate, Instrument, LevelRule, Position
from test_research_etf_engine import at, bar, held, rules, DAY


D = Decimal
KR = Instrument("KR", "KR", "KRW", "국내주식형")
WED, FRI, MON = date(2024, 6, 5), date(2024, 6, 7), date(2024, 6, 10)


def event(day=DAY, **changes):
    values = dict(symbol="KR", known_at=at(day - timedelta(days=3)), last_session=day,
                  delisting_date=day + timedelta(days=1), source_sha256="a" * 64)
    return DelistingEvent(**(values | changes))


def kr_run(bars, **options):
    values = dict(initial_cash={"KRW": D("0")}, initial_positions=[held(take="100")],
                  commission_rate=D("0"), first_entry=False)
    values.update(options)
    return run_engine([KR], bars, {"KR": values.pop("level_rules", rules(take="100", second="1000"))}, **values)


class DomesticSettlementTests(unittest.TestCase):
    def test_kr_t_plus_2_skips_exchange_holiday(self):
        result = kr_run([bar(day=WED, price="110")], initial_positions=[held(take=".1")],
                        domestic_settlement="t_plus_2", kr_sessions=[DAY, date(2024, 6, 4), WED, FRI, MON],
                        kr_settlement_session_times={MON: at(MON, 8)}, evaluation_times=[at(WED, 9), at(FRI, 9), at(MON, 9)])
        self.assertEqual([r["settled_cash"]["KRW"] for r in result.valuations], [D("0"), D("0"), D("1100")])
        self.assertEqual([r["cash"]["KRW"] for r in result.valuations], [D("1100")] * 3)
        settlement = next(row for row in result.cash_events if row["kind"] == "settlement")
        self.assertEqual(settlement["settlement_date"], MON)

    def test_unsettled_kr_proceeds_cannot_fund_fx_before_t_plus_2(self):
        instruments = [KR, Instrument("US", "US", "USD")]
        result = run_engine(instruments, [bar(day=WED, price="110")],
                            {item.symbol: rules() for item in instruments}, initial_cash={"KRW": D("0")},
                            initial_positions=[held(take=".1")], first_entry=False, commission_rate=D("0"),
                            fx_cost=D("0"), fx_rates=[FxRate(at(WED), D("1000"))],
                            conversion_times=[at(FRI, 1), at(MON, 9)], domestic_settlement="t_plus_2",
                            kr_sessions=[WED, FRI, MON], kr_settlement_session_times={MON: at(MON, 8)})
        conversions = [row for row in result.cash_events if row["kind"] == "fx_conversion"]
        self.assertEqual([row["amount"] for row in conversions], [D("0"), D(".55")])
        self.assertEqual(result.final_cash, {"KRW": D("550"), "USD": D(".55")})

    def test_unsettled_kr_cash_is_reusable_without_double_settlement_credit(self):
        buyer = Instrument("BUYER", "KR", "KRW")
        result = run_engine([KR, buyer], [bar(day=WED, price="110"), bar("BUYER", day=WED)],
                            {item.symbol: rules() for item in (KR, buyer)}, initial_cash={"KRW": D("0")},
                            initial_positions=[held(take=".1")], commission_rate=D("0"),
                            domestic_settlement="t_plus_2", kr_sessions=[WED, FRI, MON],
                            kr_settlement_session_times={MON: at(MON, 8)}, evaluation_times=[at(WED, 9), at(MON, 9)])
        self.assertEqual([(row["side"], row["symbol"]) for row in result.trades], [("SELL", "KR"), ("BUY", "BUYER")])
        self.assertEqual(result.final_cash["KRW"], D("100"))
        self.assertEqual(result.valuations[-1]["settled_cash"]["KRW"], D("100"))

    def test_missing_kr_settlement_tail_or_time_fails(self):
        for options in ({"kr_sessions": [WED, FRI]}, {"kr_sessions": [WED, FRI, MON]}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "KR.*settlement"):
                kr_run([bar(day=WED, price="110")], initial_positions=[held(take=".1")],
                       domestic_settlement="t_plus_2", **options)

    def test_legacy_same_day_cash_and_gold_are_preserved(self):
        result = kr_run([bar(price="110")], initial_positions=[held(take=".1")], evaluation_times=[at(DAY, 9)])
        self.assertEqual(result.valuations[0]["settled_cash"]["KRW"], D("1100"))
        gold = Instrument("GOLD", "GOLD", "KRW")
        result = run_engine([gold], [bar("GOLD", price="110")], {"GOLD": rules()},
                            initial_cash={"KRW": D("0")}, initial_positions=[held("GOLD", take=".1")],
                            domestic_settlement="t_plus_2", commission_rate=D("0"), first_entry=False,
                            evaluation_times=[at(DAY, 9)])
        self.assertEqual(result.valuations[0]["settled_cash"]["KRW"], D("1100"))

    def test_explicit_tax_payment_needs_settled_krw(self):
        with self.assertRaisesRegex(ValueError, "settled KRW"):
            kr_run([bar(day=WED, price="110")], initial_positions=[held(take=".1")],
                   domestic_settlement="t_plus_2", kr_sessions=[WED, FRI, MON],
                   kr_settlement_session_times={MON: at(MON, 8)}, tax_payments={at(FRI, 1): D("100")})

    def test_kr_strict_settlement_does_not_claim_after_tax_withholding_support(self):
        with self.assertRaisesRegex(ValueError, "settlement-date withholding"):
            kr_run([bar()], domestic_settlement="t_plus_2", tax_mode="after_tax", synthetic_tax_data=True)

    def test_t_plus_2_timestamp_cannot_be_relabelled_as_same_day_cash(self):
        with self.assertRaisesRegex(ValueError, "T\\+2 settlement date"):
            kr_run([bar(day=WED, price="110")], initial_positions=[held(take=".1")],
                   domestic_settlement="t_plus_2", kr_sessions=[WED, FRI, MON],
                   kr_settlement_session_times={MON: at(WED, 8)})

    def test_settled_tax_withdrawal_preserves_pending_credit(self):
        # Independent withdrawal state; this does not enable unsupported KR after_tax execution.
        ledger = TaxLedger(mode="after_tax", synthetic=True)
        us = Instrument("US", "US", "USD")
        buy_at, sell_at, pay_at = at(date(2024, 12, 2)), at(date(2024, 12, 3)), at(date(2025, 1, 3))
        position = Position("US", 1, 1000, D("10"), D("0"), D("0"), buy_at)
        ledger.buy(us, position, fx_rate=D("1000"), settled_at=buy_at)
        ledger.sell(us, position, D("20"), D("0"), sell_at, fx_rate=D("1000"), settled_at=sell_at)
        ledger.advance(pay_at)
        engine = _Engine.__new__(_Engine)
        engine.tax, engine.result = ledger, EngineResult()
        engine.cash = {"KRW": D("3100000"), "USD": D("0")}
        engine.settled = {"KRW": D("2000000"), "USD": D("0")}
        engine.pending_cash = [{"currency": "KRW", "remaining": D("1100000")}]
        engine._tax_payment(pay_at, D("500000"))
        self.assertEqual(engine.cash["KRW"], D("2600000"))
        self.assertEqual(engine.settled["KRW"], D("1500000"))
        self.assertEqual(engine.pending_cash[0]["remaining"], D("1100000"))


class DelistingTests(unittest.TestCase):
    def test_multiple_levels_liquidate_once_at_close_with_fee_and_slippage(self):
        positions = [held(quantity=3, level=1, take="100"), held(quantity=2, level=2, take="100")]
        result = kr_run([bar(close=D("120"), high=D("120"))], initial_positions=positions,
                        delistings=[event()], commission_rate=D(".01"), slippage_ticks=1,
                        evaluation_times=[at(DAY, 9)])
        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual((trade["reason"], trade["quantity"], trade["price"], trade["fee"]),
                         ("delisting_liquidation", 5, D("119"), D("5.95")))
        self.assertEqual(trade["at"], bar().close_at)
        self.assertEqual(result.final_cash["KRW"], D("589.05"))
        self.assertEqual(result.positions, [])
        self.assertEqual(result.tax["holdings"]["KR"]["quantity"], 0)

    def test_final_session_suppresses_both_ordinary_sell_and_new_buy(self):
        result = kr_run([bar(price="200")], initial_positions=[held(take=".1")], delistings=[event()])
        self.assertEqual([row["reason"] for row in result.trades], ["delisting_liquidation"])
        empty = kr_run([bar()], initial_cash={"KRW": D("1000")}, initial_positions=[], first_entry=True,
                       delistings=[event()])
        self.assertEqual(empty.trades, [])
        self.assertTrue(any(row["kind"] == "delisting_liquidated" for row in empty.cash_events))

    def test_future_announcement_or_later_bars_are_rejected(self):
        cases = (([bar()], event(known_at=at(DAY, 1))),
                 ([bar(), bar(day=DAY + timedelta(days=1))], event()))
        for bars, notice in cases:
            with self.subTest(notice=notice), self.assertRaises(ValueError):
                kr_run(bars, delistings=[notice])

    def test_untradable_last_session_or_unproven_event_rejected(self):
        for changes in ({"volume": D("0")}, {"limit_locked": True}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "tradable"):
                kr_run([bar(**changes)], delistings=[event()])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            event(source_sha256="unverified")

    def test_participation_rejection_fails_trial_instead_of_omitting_symbol(self):
        with self.assertRaisesRegex(ValueError, "do not exclude"):
            kr_run([bar(adv20=D("1000"))], delistings=[event()])

    def test_baseline_uses_the_same_terminal_sale_and_t_plus_2_cash(self):
        tomorrow = DAY + timedelta(days=1)
        frozen = fixed_baseline([KR], investment_fraction=D("1"), known_at=at(DAY - timedelta(days=1)),
                                evaluation_start=at())
        result = run_baseline(frozen, [KR], [bar(), bar(day=tomorrow, price="120")],
                              initial_cash={"KRW": D("1000")}, commission_rate=D("0"),
                              delistings=[event(day=tomorrow)], domestic_settlement="t_plus_2",
                              kr_sessions=[DAY, tomorrow, WED, FRI], kr_settlement_session_times={FRI: at(FRI, 8)},
                              evaluation_times=[at(DAY, 9), at(tomorrow, 9)])
        self.assertEqual([(t["side"], t["quantity"]) for t in result.trades], [("BUY", 10), ("SELL", 10)])
        self.assertEqual(result.positions, [])
        self.assertEqual(result.valuations[-1]["settled_cash"]["KRW"], D("0"))
        self.assertEqual(result.final_cash["KRW"], D("1200"))

    def test_terminal_sale_funds_same_instant_m2_buy_even_when_buyer_is_first(self):
        july = date(2024, 7, 1)
        instruments = [Instrument("BUYER", "KR", "KRW"), Instrument("DEAD", "KR", "KRW")]
        bars = [bar("DEAD", day=july, price="200"),
                bar("BUYER", day=july, price="50", open_at=at(july, 6), close_at=at(july, 12))]
        notice = event(july, symbol="DEAD")
        result = run_engine(instruments, bars,
                            {"BUYER": rules("500", take="100", second="250"), "DEAD": rules(take="100")},
                            initial_cash={"KRW": D("0")}, first_entry=False, commission_rate=D("0"),
                            initial_positions=[held("BUYER", quantity=5, take="100"), held("DEAD", quantity=5, take="100")],
                            delistings=[notice])
        self.assertEqual([(t["symbol"], t["side"]) for t in result.trades], [("DEAD", "SELL"), ("BUYER", "BUY")])
        self.assertEqual(result.rejections, [])
        self.assertEqual(result.final_cash["KRW"], D("750"))

    def test_terminal_sale_funds_same_instant_baseline_buy(self):
        july = date(2024, 7, 1)
        instruments = [Instrument("BUYER", "KR", "KRW"), Instrument("DEAD", "KR", "KRW")]
        bars = [bar("BUYER"), bar("DEAD"), bar("DEAD", day=july, price="200"),
                bar("BUYER", day=july, price="50", open_at=at(july, 6), close_at=at(july, 12))]
        frozen = fixed_baseline(instruments, investment_fraction=D("1"), known_at=at(DAY - timedelta(days=1)),
                                evaluation_start=at())
        result = run_baseline(frozen, instruments, bars, initial_cash={"KRW": D("1000")},
                              evaluation_times=[at(DAY, 9), at(july, 13)], commission_rate=D("0"),
                              delistings=[event(july, symbol="DEAD")])
        self.assertEqual([(t["symbol"], t["side"]) for t in result.trades[-2:]], [("DEAD", "SELL"), ("BUYER", "BUY")])
        self.assertEqual(result.rejections, [])
        self.assertEqual(result.final_cash["KRW"], D("750"))


class FxAgeTests(unittest.TestCase):
    def test_fx_age_boundary_uses_publication_instant_and_future_quote_is_ignored(self):
        rates = [FxRate(at(), D("1000")), FxRate(at(DAY + timedelta(days=8)), D("9999"))]
        result = run_engine([Instrument("US", "US", "USD")], [], {"US": rules()},
                            initial_cash={"USD": D("1")}, fx_rates=rates,
                            evaluation_times=[at(DAY + timedelta(days=7))], max_fx_age_days=7)
        self.assertEqual(result.valuations[0]["total_krw"], D("1000"))
        with self.assertRaisesRegex(ValueError, "stale FX"):
            run_engine([Instrument("US", "US", "USD")], [], {"US": rules()},
                       initial_cash={"USD": D("1")}, fx_rates=rates,
                       evaluation_times=[at(DAY + timedelta(days=7), 1)], max_fx_age_days=7)

    def test_invalid_age_and_settlement_modes_are_rejected(self):
        for options in ({"max_fx_age_days": True}, {"max_fx_age_days": -1}, {"domestic_settlement": "guess"},
                        {"fx_closure_age_days": 12}, {"max_fx_age_days": 7, "fx_closure_age_days": 6},
                        {"max_fx_age_days": 7, "fx_closure_age_days": True},
                        {"max_fx_age_days": 7, "fx_closure_age_days": 12}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                kr_run([bar()], **options)


class FxClosureTests(unittest.TestCase):
    """2017년 추석: 마지막 MAR 9/29 09:00 KST, 다음 국내 세션·MAR 10/10 09:00 KST."""
    LAST, REOPEN = date(2017, 9, 29), date(2017, 10, 10)
    US = Instrument("US", "US", "USD")

    def value(self, when, *, sessions=(LAST, REOPEN), closure=12, reopened_rate=False, runner=None):
        rates = [FxRate(at(self.LAST), D("1000"))] + ([FxRate(at(self.REOPEN), D("2000"))] if reopened_rate else [])
        options = dict(initial_cash={"USD": D("1")}, fx_rates=rates, evaluation_times=[when],
                       max_fx_age_days=7, fx_closure_age_days=closure, kr_sessions=list(sessions))
        if runner is not None:
            return runner(options).valuations[0]["total_krw"]
        return run_engine([self.US], [], {"US": rules()}, **options).valuations[0]["total_krw"]

    def test_closure_allows_last_rate_until_the_reopening_publication(self):
        nyse_close = at(date(2017, 10, 9), 20)
        self.assertEqual(nyse_close - at(self.LAST), timedelta(days=10, hours=20))
        self.assertEqual(self.value(nyse_close), D("1000"))
        self.assertEqual(self.value(at(date(2017, 10, 9), 23, 59)), D("1000"))
        self.assertEqual(self.value(at(self.REOPEN), reopened_rate=True), D("2000"))

    def test_session_without_a_new_rate_and_closure_cap_still_fail(self):
        with self.assertRaisesRegex(ValueError, "stale FX.*KRX session passed"):
            self.value(at(self.REOPEN))
        with self.assertRaisesRegex(ValueError, "stale FX.*KRX session passed"):
            self.value(at(date(2017, 10, 9), 20), sessions=(self.LAST, date(2017, 10, 2), self.REOPEN))
        with self.assertRaisesRegex(ValueError, "stale FX.*closure age 10 days"):
            self.value(at(date(2017, 10, 9), 20), closure=10)
        with self.assertRaisesRegex(ValueError, "stale FX.*maximum age 7 days"):
            self.value(at(date(2017, 10, 9), 20), closure=None)

    def test_missed_session_inside_the_basic_limit_keeps_existing_behaviour(self):
        monday = date(2017, 10, 2)
        self.assertEqual(self.value(at(monday, 9), sessions=(self.LAST, monday, self.REOPEN)), D("1000"))

    def test_baseline_engine_applies_the_same_closure_rule(self):
        def baseline(options):
            frozen = fixed_baseline([self.US], investment_fraction=D("0"),
                                    known_at=at(self.LAST - timedelta(days=1)), evaluation_start=at(self.LAST))
            return run_baseline(frozen, [self.US], [bar("US", day=self.LAST, market="US")],
                                sec_rate_override=D("0"), **options)
        self.assertEqual(self.value(at(date(2017, 10, 9), 20), runner=baseline), D("1000"))
        with self.assertRaisesRegex(ValueError, "stale FX.*KRX session passed"):
            self.value(at(self.REOPEN), runner=baseline)


if __name__ == "__main__":
    unittest.main()
