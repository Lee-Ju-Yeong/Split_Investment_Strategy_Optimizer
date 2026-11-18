"""UTC order, cash, settlement and valuation regressions for the M2 engine."""
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import unittest
from unittest.mock import patch

from src.research.etf.engine import run_engine
from src.research.etf.engine_tax import TaxLedger
from src.research.etf.engine_types import Bar, FxRate, Instrument, LevelRule, Position


D = Decimal
UTC = timezone.utc
DAY = date(2024, 6, 3)


def at(day=DAY, hour=0, minute=0):
    return datetime.combine(day, time(hour, minute), UTC)


def bar(symbol="KR", day=DAY, price="100", *, market="KR", **changes):
    opening = at(day, 13, 30) if market == "US" else at(day)
    values = dict(symbol=symbol, session_date=day, open_at=opening,
                  close_at=opening + timedelta(hours=6), open=D(price), high=D(price),
                  low=D(price), close=D(price), volume=D("1000000"), adv20=D("1000000000"))
    return Bar(**{**values, **changes})


def rules(budget="1000", *, take=".1", second=None):
    result = (LevelRule(1, D(budget), D("0"), D(take)),)
    return result if second is None else result + (LevelRule(2, D(second), D(".1"), D(".1")),)


def held(symbol="KR", quantity=10, price="100", *, take=".1", level=1):
    return Position(symbol, level, quantity, D(price), D("0"), D(take), at(DAY - timedelta(days=3)))


def fx(value="1000", when=None):
    return FxRate(when or at(DAY - timedelta(days=10)), D(value))


def kr_run(bars, **options):
    instrument = Instrument("KR", "KR", "KRW", "국내주식형")
    config = dict(initial_cash={"KRW": D("1000")}, commission_rate=D("0"))
    config.update(options)
    return run_engine([instrument], bars, {"KR": config.pop("level_rules", rules())}, **config)


class OrderAndBudgetTests(unittest.TestCase):
    def test_first_open_entry_once_no_reentry_after_sale(self):
        result = kr_run([bar(day=DAY + timedelta(days=2), price="90"),
                         bar(day=DAY + timedelta(days=1), price="110"), bar()])
        self.assertEqual([(row["side"], row["quantity"]) for row in result.trades], [("BUY", 10), ("SELL", 10)])
        self.assertEqual(result.final_cash, {"KRW": D("1100"), "USD": D("0")})
        self.assertEqual(result.positions, [])

    def test_failed_first_entry_is_not_retried(self):
        result = kr_run([bar(), bar(day=DAY + timedelta(days=1), price="90")],
                        initial_cash={"KRW": D("950")})
        self.assertEqual(result.trades, [])
        self.assertEqual(len(result.rejections), 1)
        self.assertEqual(result.rejections[0]["quantity"], 10)
        self.assertEqual(result.rejections[0]["reason"], "insufficient_cash")
        unavailable = next(row for row in result.cash_events if row["kind"] == "first_entry_unavailable")
        self.assertEqual(unavailable["reason"], "insufficient_cash")
        self.assertTrue(any("not retried" in warning for warning in result.warnings))

    def test_unavailable_initial_entry_records_reason_without_later_retry(self):
        cases = (({"volume": D("0")}, rules(), "zero_volume"),
                 ({"limit_locked": True}, rules(), "limit_locked"),
                 ({}, rules("50"), "zero_affordable_quantity"))
        for changes, levels, reason in cases:
            with self.subTest(reason=reason):
                result = kr_run([bar(**changes), bar(day=DAY + timedelta(days=1), price="10")],
                                level_rules=levels)
                self.assertEqual(result.trades, [])
                self.assertEqual(result.rejections, [])
                unavailable = [row for row in result.cash_events if row["kind"] == "first_entry_unavailable"]
                self.assertEqual(len(unavailable), 1)
                self.assertEqual(unavailable[0]["reason"], reason)
                self.assertEqual(unavailable[0]["at"], at())
                self.assertFalse(unavailable[0]["retry"])
                self.assertTrue(any(reason in warning and "not retried" in warning for warning in result.warnings))

    def test_participation_rejects_full_order_and_equality_is_allowed(self):
        rejected = kr_run([bar(adv20=D("10000"))])
        allowed = kr_run([bar(adv20=D("20000"))])
        self.assertEqual(rejected.rejections[0]["reason"], "participation_limit")
        self.assertEqual(rejected.rejections[0]["quantity"], 10)
        self.assertEqual(allowed.trades[0]["quantity"], 10)

    def test_missing_adv_is_explicit_rejection(self):
        result = kr_run([bar(adv20=None)])
        self.assertEqual(result.rejections[0]["reason"], "adv20_unavailable")

    def test_cash_and_symbol_limits_reject_without_resizing(self):
        floor = kr_run([bar()], cash_floor_fraction=D(".1"))
        cap = kr_run([bar()], symbol_cap_fraction=D(".5"))
        self.assertEqual(floor.rejections[0]["reason"], "cash_floor")
        self.assertEqual(cap.rejections[0]["reason"], "symbol_cap")
        self.assertEqual(floor.final_cash["KRW"], D("1000"))

    def test_future_sale_cannot_fund_earlier_buy(self):
        instruments = [Instrument("A", "KR", "KRW"), Instrument("B", "KR", "KRW")]
        first = bar("A", high=D("120"))
        second = bar("B", open_at=at(hour=1), close_at=at(hour=7))
        result = run_engine(instruments, [second, first], {"A": rules(), "B": rules()},
                            initial_cash={"KRW": D("0")}, initial_positions=[held("A")],
                            commission_rate=D("0"), evaluation_times=[at(hour=1, minute=3)])
        self.assertEqual([(row["symbol"], row["side"]) for row in result.trades], [("A", "SELL")])
        self.assertEqual(result.trades[0]["generated_at"], at())
        self.assertGreater(result.trades[0]["at"], at(hour=1, minute=3))
        self.assertEqual(result.rejections[0]["symbol"], "B")
        self.assertEqual(result.rejections[0]["at"], at(hour=1))
        self.assertEqual(result.valuations[0]["cash"]["KRW"], D("0"))
        self.assertEqual(result.final_cash["KRW"], D("1100"))

    def test_same_timestamp_preserves_instrument_input_order(self):
        instruments = [Instrument("B", "KR", "KRW"), Instrument("A", "KR", "KRW")]
        result = run_engine(instruments, [bar("A"), bar("B")], {"A": rules(), "B": rules()},
                            initial_cash={"KRW": D("1000")}, commission_rate=D("0"))
        self.assertEqual(result.trades[0]["symbol"], "B")
        self.assertEqual(result.rejections[0]["symbol"], "A")

    def test_monthly_budget_uses_previous_equity_for_all_levels(self):
        january, february = date(2024, 1, 31), date(2024, 2, 1)
        position = replace(held(take="10"), opened_at=at(date(2024, 1, 30)))
        result = kr_run([bar(day=january, close=D("200"), high=D("200")),
                         bar(day=february, price="90")], initial_positions=[position],
                        first_entry=False, first_level_fraction=D(".1"),
                        level_rules=rules(take="10", second="500"),
                        evaluation_times=[at(january, 9)])
        refreshes = [row for row in result.cash_events if row["kind"] == "budget_refresh"]
        self.assertEqual([row["budgets"]["KR"] for row in refreshes], [D("200"), D("300")])
        self.assertEqual(result.trades[0]["quantity"], 3)
        self.assertEqual(result.positions[0], position)
        self.assertEqual(result.positions[1].take_rate, D(".1"))

    def test_fixed_budget_keeps_different_input_level_budgets(self):
        result = kr_run([bar(price="90")], initial_positions=[held(take="10")],
                        first_entry=False, level_rules=rules(take="10", second="500"))
        self.assertEqual(result.trades[0]["quantity"], 5)
        self.assertFalse(any(row["kind"] == "budget_refresh" for row in result.cash_events))

    def test_monthly_budget_requires_a_preceding_evaluation_after_initial_month(self):
        with self.assertRaisesRegex(ValueError, "preceding evaluation"):
            kr_run([bar(day=date(2024, 1, 31)), bar(day=date(2024, 2, 1))],
                   first_entry=False, first_level_fraction=D(".1"))


class FxAndInformationTests(unittest.TestCase):
    def test_currency_cash_isolation(self):
        result = run_engine([Instrument("US", "US", "USD")], [bar("US", market="US")],
                            {"US": rules()}, initial_cash={"KRW": D("1000000")},
                            commission_rate=D("0"), sec_rate_override=D("0"))
        self.assertEqual(result.rejections[0]["reason"], "insufficient_cash")
        self.assertEqual(result.final_cash["KRW"], D("1000000"))

    def test_explicit_conversion_targets_currency_count_share_and_charges_cost(self):
        instruments = [Instrument("KR", "KR", "KRW"), Instrument("US", "US", "USD")]
        result = run_engine(instruments, [], {item.symbol: rules() for item in instruments},
                            initial_cash={"KRW": D("1000000")}, fx_rates=[fx()],
                            conversion_times=[at(hour=1)])
        self.assertEqual(result.final_cash, {"KRW": D("499500"), "USD": D("500")})
        conversion = result.cash_events[0]
        self.assertEqual(conversion["fee_krw"], D("500"))

    def test_no_conversion_is_invented_without_explicit_schedule(self):
        result = run_engine([Instrument("US", "US", "USD")], [], {"US": rules()},
                            initial_cash={"KRW": D("1000000")}, fx_rates=[fx()])
        self.assertEqual(result.final_cash["USD"], D("0"))
        self.assertEqual(result.cash_events, [])

    def test_conversion_uses_current_holdings_cash_and_published_fx_together(self):
        tomorrow = DAY + timedelta(days=1)
        instruments = [Instrument("KR", "KR", "KRW"), Instrument("US", "US", "USD")]
        for next_fx, expected_krw, expected_usd in (("1000", "4000000", "2000"),
                                                 ("2000", "6500000", "750")):
            with self.subTest(next_fx=next_fx):
                result = run_engine(
                    instruments, [bar(), bar("US", market="US")],
                    {"KR": rules("1000000", take="10"), "US": rules("3000", take="10")},
                    initial_cash={"KRW": D("10000000")},
                    fx_rates=[fx(), fx(next_fx, at(tomorrow))], fx_cost=D("0"),
                    conversion_times=[at(hour=1), at(tomorrow, 1)],
                    evaluation_times=[at(hour=9), at(tomorrow, 2)],
                    commission_rate=D("0"), sec_rate_override=D("0"),
                    us_sessions=[DAY, tomorrow], settlement_session_times={tomorrow: at(tomorrow, 20)})
                self.assertEqual([(row["symbol"], row["quantity"]) for row in result.trades], [("KR", 10000), ("US", 30)])
                self.assertEqual(result.final_cash, {"KRW": D(expected_krw), "USD": D(expected_usd)})
                current = result.valuations[-1]
                usd_exposure = current["holdings_krw"]["US"] + current["cash"]["USD"] * D(next_fx)
                self.assertEqual(usd_exposure, current["equity_krw"] * D(".5"))

    def test_unpublished_fx_is_not_usable_for_conversion(self):
        with self.assertRaisesRegex(ValueError, "no FX rate published"):
            run_engine([Instrument("US", "US", "USD")], [], {"US": rules()},
                       initial_cash={"KRW": D("1000")}, fx_rates=[fx(when=at(hour=2))],
                       conversion_times=[at(hour=1)])

    def test_future_us_close_and_fx_do_not_change_kr_18h_equity(self):
        results = []
        for future_close, future_fx in (("100", "1200"), ("150", "9000")):
            results.append(run_engine(
                [Instrument("US", "US", "USD")],
                [bar("US", market="US"), bar("US", DAY + timedelta(days=1), market="US",
                                             high=D(future_close), close=D(future_close))],
                {"US": rules(take="10")}, initial_cash={"USD": D("0")},
                initial_positions=[held("US", take="10")], first_entry=False,
                fx_rates=[fx(), fx(future_fx, at(DAY + timedelta(days=1), 12))],
                evaluation_times=[at(DAY + timedelta(days=1), 9)],
                commission_rate=D("0"), sec_rate_override=D("0")))
        self.assertEqual(results[0].valuations, results[1].valuations)
        self.assertEqual(results[0].valuations[0]["equity_krw"], D("1000000"))
        self.assertEqual(results[0].valuations[0]["price_times"]["US"], at(hour=19, minute=30))

    def test_monthly_us_budget_uses_latest_published_fx_not_future_rate(self):
        result = run_engine(
            [Instrument("US", "US", "USD")], [bar("US", market="US")], {"US": rules()},
            initial_cash={"KRW": D("1000000")}, first_level_fraction=D(".1"), first_entry=False,
            fx_rates=[fx(), fx("9999", at(hour=22))], commission_rate=D("0"), sec_rate_override=D("0"))
        refresh = next(row for row in result.cash_events if row["kind"] == "budget_refresh")
        self.assertEqual(refresh["budgets"]["US"], D("100"))


class SettlementTests(unittest.TestCase):
    def sale(self, *, day=DAY, sessions=None, times=None, **options):
        sessions = sessions if sessions is not None else [day, day + timedelta(days=1)]
        times = times if times is not None else {session: at(session, 20) for session in sessions}
        config = dict(initial_cash={"KRW": D("0")}, initial_positions=[held("US")],
                      first_entry=False, commission_rate=D("0"), sec_rate_override=D("0"),
                      fx_rates=[fx(when=at(date(2010, 1, 1)))], us_sessions=sessions,
                      settlement_session_times=times)
        config.update(options)
        return run_engine([Instrument("KR", "KR", "KRW"), Instrument("US", "US", "USD")],
                          [bar("US", day, price="110", market="US")],
                          {"KR": rules(), "US": rules()}, **config)

    def test_settlement_transitions_use_actual_sessions_including_holidays(self):
        cases = [
            (date(2017, 9, 1), [date(2017, 9, n) for n in (1, 5, 6, 7)], date(2017, 9, 7)),
            (date(2017, 9, 5), [date(2017, 9, n) for n in (5, 6, 7)], date(2017, 9, 7)),
            (date(2024, 5, 24), [date(2024, 5, n) for n in (24, 28, 29)], date(2024, 5, 29)),
            (date(2024, 5, 28), [date(2024, 5, n) for n in (28, 29)], date(2024, 5, 29)),
        ]
        for day, sessions, expected in cases:
            with self.subTest(day=day):
                position = replace(held("US"), opened_at=at(day - timedelta(days=5)))
                result = self.sale(day=day, sessions=sessions, initial_positions=[position])
                settlement = next(row for row in result.cash_events if row["kind"] == "settlement")
                self.assertEqual(settlement["settlement_date"], expected)
                self.assertEqual(settlement["at"], at(expected, 20))

    def test_unsettled_cash_is_reusable_but_not_convertible(self):
        day_after = DAY + timedelta(days=1)
        result = self.sale(conversion_times=[at(day_after, 1), at(day_after, 21)],
                           evaluation_times=[at(hour=21), at(day_after, 2)], fx_cost=D("0"))
        self.assertEqual(result.valuations[1]["cash"]["USD"], D("1100"))
        self.assertEqual(result.valuations[1]["settled_cash"]["USD"], D("0"))
        self.assertEqual(result.final_cash, {"KRW": D("550000"), "USD": D("550")})
        self.assertTrue(any("Unsettled" in message for message in result.warnings))

    def test_reused_unsettled_credit_does_not_settle_a_second_time(self):
        sessions = [DAY, DAY + timedelta(days=1)]
        instruments = [Instrument("A", "US", "USD"), Instrument("B", "US", "USD")]
        result = run_engine(
            instruments, [bar("A", price="110", market="US"),
                          bar("B", market="US", open_at=at(hour=14), close_at=at(hour=20))],
            {"A": rules(), "B": rules()}, initial_cash={"USD": D("0")},
            initial_positions=[held("A")], fx_rates=[fx()], commission_rate=D("0"),
            sec_rate_override=D("0"), us_sessions=sessions,
            settlement_session_times={day: at(day, 21) for day in sessions})
        self.assertEqual([(row["symbol"], row["side"]) for row in result.trades], [("A", "SELL"), ("B", "BUY")])
        self.assertEqual(result.final_cash["USD"], D("100"))
        final = result.cash_events[-1]
        self.assertEqual(final["settled_cash"]["USD"], D("100"))

    def test_insufficient_sessions_keep_explicit_override_sales_unsettled(self):
        result = self.sale(sessions=[DAY], times={DAY: at(hour=20)})
        self.assertEqual(result.final_cash["USD"], D("1100"))
        self.assertEqual(result.cash_events[-1]["settled_cash"]["USD"], D("0"))
        self.assertEqual(len(result.tax["pending_settlement_events"]), 1)
        self.assertTrue(any("insufficient actual US sessions" in warning for warning in result.warnings))

    def test_missing_settlement_time_does_not_invent_cash_availability(self):
        result = self.sale(times={})
        self.assertEqual(result.cash_events[-1]["settled_cash"]["USD"], D("0"))
        self.assertTrue(any("no availability time" in warning for warning in result.warnings))

    def test_missing_sessions_without_explicit_sec_override_fails(self):
        with self.assertRaisesRegex(ValueError, "actual settlement sessions"):
            self.sale(sessions=[DAY], sec_rate_override=None)

    def test_sec_uses_verified_settlement_date_and_signal_knowledge_time(self):
        with patch("src.research.etf.engine.sec_fee_rate", return_value=D(".0000278")) as lookup:
            result = self.sale(sec_rate_override=None)
        self.assertTrue(lookup.called)
        for invocation in lookup.call_args_list:
            self.assertEqual(invocation.args, (DAY + timedelta(days=1),))
            self.assertEqual(invocation.kwargs, {"known_at": at(hour=13, minute=30)})
        self.assertEqual(result.trades[0]["fee"], D(".03058"))

    def test_future_settlement_fx_changes_tax_only_after_settlement(self):
        results = []
        for rate in ("1000", "1500"):
            results.append(self.sale(fx_rates=[fx(), fx(rate, at(DAY + timedelta(days=1), 1))],
                                     evaluation_times=[at(hour=21)]))
        self.assertEqual(results[0].trades, results[1].trades)
        self.assertEqual(results[0].valuations, results[1].valuations)
        self.assertEqual(results[0].tax["us_realized_krw"][2024], D("100000"))
        self.assertEqual(results[1].tax["us_realized_krw"][2024], D("650000"))


class CorporateAndTaxTests(unittest.TestCase):
    def realized_gain_run(self, mode, *, tax_payments=None, extra_evaluations=()):
        tomorrow, july = DAY + timedelta(days=1), date(2024, 7, 1)
        return run_engine(
            [Instrument("US", "US", "USD")],
            [bar("US", price="200", market="US"), bar("US", july, price="200", market="US")],
            {"US": rules()}, initial_cash={"KRW": D("2000000")},
            initial_positions=[held("US", quantity=100)], first_entry=False,
            first_level_fraction=D(".1"), commission_rate=D("0"), sec_rate_override=D("0"),
            fx_rates=[fx()], us_sessions=[DAY, tomorrow],
            settlement_session_times={tomorrow: at(tomorrow, 20)},
            tax_mode=mode, synthetic_tax_data=mode == "after_tax", tax_payments=tax_payments,
            evaluation_times=[at(hour=21), at(tomorrow, 21), at(date(2024, 6, 30), 9), *extra_evaluations])

    def test_estimated_current_year_tax_is_reported_without_changing_book_equity_or_budget(self):
        for mode, estimate in (("pre_tax", D("0")), ("after_tax", D("1650000"))):
            with self.subTest(mode=mode):
                result = self.realized_gain_run(mode)
                before, settled = result.valuations[:2]
                self.assertEqual(before["estimated_tax_krw"], D("0"))
                self.assertEqual(settled["estimated_tax_krw"], estimate)
                self.assertEqual(settled["equity_krw"], D("22000000"))
                self.assertEqual(settled["total_krw"], settled["equity_krw"])
                self.assertEqual(settled["equity_before_estimated_tax_krw"], D("22000000"))
                self.assertEqual(settled["equity_after_estimated_tax_krw"], D("22000000") - estimate)
                refreshes = [row for row in result.cash_events if row["kind"] == "budget_refresh"]
                self.assertEqual(refreshes[-1]["source_total_krw"], D("22000000"))
                self.assertEqual(refreshes[-1]["budgets"]["US"], D("2200"))

    def test_adjusted_reporting_equity_stays_continuous_across_finalization_and_tax_payment(self):
        result = self.realized_gain_run(
            "after_tax", tax_payments={at(date(2025, 5, 31)): D("1650000")},
            extra_evaluations=(at(date(2024, 12, 31), 9), at(date(2025, 1, 1)), at(date(2025, 6, 1))))
        december, january, paid = result.valuations[-3:]
        self.assertEqual(december["equity_krw"], D("22000000"))
        self.assertEqual(december["estimated_tax_krw"], D("1650000"))
        self.assertEqual(january["tax_liability_krw"], D("1650000"))
        self.assertEqual(january["estimated_tax_krw"], D("0"))
        self.assertEqual(paid["tax_liability_krw"], D("0"))
        for row in (december, january, paid):
            self.assertEqual(row["equity_after_estimated_tax_krw"], D("20350000"))
        self.assertEqual(result.final_cash["KRW"], D("350000"))

    def test_repeated_valuation_does_not_copy_the_full_tax_ledger(self):
        original_snapshot = TaxLedger.snapshot
        with patch.object(TaxLedger, "snapshot", autospec=True, side_effect=original_snapshot) as snapshot:
            result = kr_run([bar()], first_entry=False,
                            evaluation_times=[at() + timedelta(minutes=index) for index in range(60)])
        self.assertEqual(len(result.valuations), 60)
        self.assertEqual(snapshot.call_count, 1)  # Final result export only.

    def test_missing_dividend_pay_date_creates_only_nonspendable_receivable(self):
        result = kr_run([bar(dividend=D("1"))], initial_positions=[held(take="10")],
                        initial_cash={"KRW": D("0")}, first_entry=False,
                        evaluation_times=[at(hour=1)])
        self.assertEqual(result.final_cash["KRW"], D("0"))
        self.assertEqual(result.valuations[0]["receivables_krw"], D("10"))
        self.assertEqual(result.valuations[0]["equity_krw"], D("1010"))
        self.assertEqual(result.tax["receivables"], {})
        self.assertEqual(len(result.tax["engine_receivables"]), 1)
        self.assertFalse(any(row["kind"] == "dividend_payment" for row in result.cash_events))

    def test_missing_dividend_pay_date_rejects_synthetic_after_tax(self):
        with self.assertRaisesRegex(ValueError, "dividend_pay_at"):
            kr_run([bar(dividend=D("1"))], tax_mode="after_tax", synthetic_tax_data=True)

    def test_dividend_rights_are_fixed_before_ex_date_orders(self):
        result = kr_run([bar(dividend=D("1"), dividend_pay_at=at(DAY + timedelta(days=1)))],
                        evaluation_times=[at(hour=7)])
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.final_cash["KRW"], D("0"))
        self.assertEqual(result.tax["receivables"], {})

    def test_synthetic_dividend_receivable_and_actual_payment_are_distinct(self):
        result = kr_run([bar(dividend=D("1"), dividend_pay_at=at(DAY + timedelta(days=1)))],
                        initial_positions=[held(take="10")], initial_cash={"KRW": D("100")},
                        first_entry=False, tax_mode="after_tax", synthetic_tax_data=True,
                        evaluation_times=[at(hour=1), at(DAY + timedelta(days=1), 1)])
        self.assertEqual(result.valuations[0]["cash"]["KRW"], D("100"))
        self.assertEqual(result.valuations[0]["receivables_krw"], D("9"))
        self.assertEqual(result.valuations[1]["cash"]["KRW"], D("109"))
        self.assertEqual(result.valuations[1]["receivables_krw"], D("0"))

    def test_m1_split_adjusted_basis_does_not_apply_split_again(self):
        instrument = Instrument("US", "US", "USD", price_basis="yahoo_split_adjusted_quote_with_cash_dividends")
        position = held("US", price="50", take="10")
        result = run_engine([instrument], [bar("US", price="50", market="US", split_ratio=D("2"))],
                            {"US": rules(take="10")}, initial_cash={"USD": D("0")},
                            initial_positions=[position], first_entry=False, fx_rates=[fx()],
                            commission_rate=D("0"), sec_rate_override=D("0"))
        self.assertEqual(result.positions, [position])
        self.assertEqual(result.tax["holdings"]["US"]["quantity"], 10)
        self.assertTrue(any("not applied" in warning for warning in result.warnings))

    def test_raw_split_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(ValueError, "raw split"):
            kr_run([bar(split_ratio=D("2"))])

    def test_cash_dividends_cannot_be_counted_again_in_total_return_prices(self):
        instrument = Instrument("KR", "KR", "KRW", price_basis="total_return_adjusted")
        with self.assertRaisesRegex(ValueError, "dividend-adjusted"):
            run_engine([instrument], [bar(dividend=D("1"))], {"KR": rules()}, initial_cash={"KRW": D("0")})

    def test_unsettled_us_dividend_entitlement_uses_engine_only_in_pre_tax(self):
        tomorrow = DAY + timedelta(days=1)
        data = [bar("US", market="US"),
                bar("US", tomorrow, market="US", dividend=D("1"), dividend_pay_at=at(tomorrow, 22))]
        config = dict(initial_cash={"USD": D("1000")}, fx_rates=[fx()],
                      commission_rate=D("0"), sec_rate_override=D("0"),
                      us_sessions=[DAY, tomorrow, tomorrow + timedelta(days=1)],
                      settlement_session_times={tomorrow: at(tomorrow, 21)})
        result = run_engine([Instrument("US", "US", "USD")], data, {"US": rules(take="10")}, **config)
        self.assertEqual(result.final_cash["USD"], D("10"))
        self.assertTrue(any("settled tax quantity" in warning for warning in result.warnings))
        with self.assertRaisesRegex(ValueError, "entitlement and settled tax quantity differ"):
            run_engine([Instrument("US", "US", "USD")], data, {"US": rules(take="10")},
                       tax_mode="after_tax", synthetic_tax_data=True, **config)

    def test_explicit_tax_payment_updates_cash_and_unpaid_liability(self):
        tomorrow = DAY + timedelta(days=1)
        position = held("US", quantity=100, take=".1")
        result = run_engine(
            [Instrument("US", "US", "USD")], [bar("US", price="200", market="US")],
            {"US": rules()}, initial_cash={"KRW": D("2000000")}, initial_positions=[position],
            first_entry=False, commission_rate=D("0"), sec_rate_override=D("0"), fx_rates=[fx()],
            us_sessions=[DAY, tomorrow], settlement_session_times={tomorrow: at(tomorrow, 21)},
            tax_mode="after_tax", synthetic_tax_data=True,
            evaluation_times=[at(date(2025, 1, 1))], tax_payments={at(date(2025, 5, 31)): D("1650000")})
        self.assertEqual(result.valuations[0]["tax_liability_krw"], D("1650000"))
        self.assertEqual(result.valuations[0]["equity_krw"], D("20350000"))
        self.assertEqual(result.final_cash["KRW"], D("350000"))
        self.assertEqual(result.tax["unpaid_tax_krw"], D("0"))


class InputValidationTests(unittest.TestCase):
    def test_unknown_options_are_rejected(self):
        with self.assertRaises(TypeError):
            kr_run([bar()], ignored_option=True)

    def test_negative_cash_invalid_fraction_and_naive_time_fail(self):
        for config in ({"initial_cash": {"KRW": D("-1")}},
                       {"participation_rate": D("1.1")},
                       {"evaluation_times": [datetime(2024, 6, 3)]},
                       {"tax_mode": "after_tax"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                kr_run([bar()], **config)

    def test_unknown_duplicate_or_overlapping_bar_fails(self):
        for bars in ([bar("UNKNOWN")], [bar(), bar()],
                     [bar(), bar(day=DAY + timedelta(days=1), open_at=at(hour=5), close_at=at(hour=7))]):
            with self.subTest(bars=bars), self.assertRaises(ValueError):
                kr_run(bars)

    def test_duplicate_fx_and_unmapped_settlement_time_fail(self):
        for config in ({"fx_rates": [fx(), fx()]},
                       {"settlement_session_times": {DAY: at(hour=20)}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                kr_run([bar()], **config)

    def test_future_initial_positions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "initial positions"):
            kr_run([bar()], initial_positions=[replace(held(), opened_at=at(hour=1))])

    def test_tax_basis_references_cannot_be_silently_ignored(self):
        for config in ({"tax_basis_by_bar": {("KR", DAY + timedelta(days=1)): D("100")}},
                       {"initial_tax_basis": {("KR", 1): D("100")}}):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "references a missing"):
                kr_run([bar()], **config)

    def test_initial_holdings_are_registered_even_without_market_or_schedule_events(self):
        result = kr_run([], initial_positions=[held()])
        self.assertEqual(result.tax["holdings"]["KR"]["quantity"], 10)

    def test_inputs_are_not_mutated(self):
        cash, levels, position = {"KRW": D("1000")}, rules(second="500"), held(take="10")
        kr_run([bar(price="90")], initial_cash=cash, level_rules=levels,
               initial_positions=[position], first_entry=False)
        self.assertEqual(cash, {"KRW": D("1000")})
        self.assertEqual(levels[1].budget, D("500"))
        self.assertEqual(position.entry_price, D("100"))


if __name__ == "__main__":
    unittest.main()
