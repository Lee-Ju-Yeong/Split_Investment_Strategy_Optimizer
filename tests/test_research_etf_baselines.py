"""M4 hand-calculated allocation, time isolation and shared-ledger regressions."""
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, localcontext
import unittest
from zoneinfo import ZoneInfo

from src.research.etf.baseline_engine import run_baseline
from src.research.etf.baseline_policy import (
    VolatilityEstimate, diagnostic_baseline, fit_baseline, fixed_baseline,
)
from src.research.etf.engine_tax import TaxLedger
from src.research.etf.engine_types import Bar, EngineResult, FxRate, Instrument, Position


D = Decimal
ROUNDING_MODES = ("ROUND_HALF_EVEN", "ROUND_HALF_UP", "ROUND_HALF_DOWN", "ROUND_UP",
                  "ROUND_DOWN", "ROUND_CEILING", "ROUND_FLOOR", "ROUND_05UP")
JUNE, JULY, AUGUST = date(2024, 6, 3), date(2024, 7, 1), date(2024, 8, 1)
KR = Instrument("KR", "KR", "KRW", "국내주식형")


def at(day=JUNE, hour=0):
    return datetime.combine(day, time(hour), timezone.utc)


def bar(symbol="KR", day=JUNE, price="100", **changes):
    values = dict(symbol=symbol, session_date=day, open_at=at(day), close_at=at(day, 6),
                  open=D(price), high=D(price), low=D(price), close=D(price),
                  volume=D("1000000"), adv20=D("1000000000"))
    return Bar(**(values | changes))


def policy(instruments=(KR,), fraction=".5", **options):
    return fixed_baseline(instruments, investment_fraction=D(fraction), known_at=at(JUNE - timedelta(days=1)),
                          evaluation_start=at(), **options)


def run(bars, instruments=(KR,), frozen=None, **options):
    values = dict(initial_cash={"KRW": D("1000")}, commission_rate=D("0"),
                  evaluation_times=sorted({item.close_at + timedelta(hours=1) for item in bars}))
    return run_baseline(frozen or policy(instruments), instruments, bars, **(values | options))


def learning(exposures=(".2", ".6"), symbols=("KR",)):
    rows = []
    for day, exposure in zip((date(2024, 5, 2), date(2024, 5, 3)), exposures):
        invested = D(exposure) * D("1000")
        rows.append({"at": at(day, 9), "equity_krw": D("1000"),
                     "holdings_krw": {symbol: invested / len(symbols) for symbol in symbols}})
    return EngineResult(valuations=rows)


def fit(result=None, instruments=(KR,), **options):
    values = dict(training_start=at(date(2024, 5, 1)), training_end=at(date(2024, 5, 31)),
                  evaluation_start=at())
    return fit_baseline(result or learning(), instruments, **(values | options))


class FrozenPolicyTests(unittest.TestCase):
    def test_realized_daily_exposure_is_arithmetic_mean(self):
        self.assertEqual(fit().investment_fraction, D(".4"))
        self.assertEqual(fit().role, "primary")

    def test_training_end_must_precede_evaluation(self):
        with self.assertRaisesRegex(ValueError, "precede"):
            fit(training_end=at())

    def test_future_learning_row_is_rejected(self):
        result = learning()
        result.valuations[-1]["at"] = at()
        with self.assertRaisesRegex(ValueError, "window"):
            fit(result)

    def test_duplicate_and_reversed_learning_times_rejected(self):
        for reverse in (False, True):
            result = learning()
            if reverse:
                result.valuations.reverse()
            else:
                result.valuations[-1]["at"] = result.valuations[0]["at"]
            with self.subTest(reverse=reverse), self.assertRaises(ValueError):
                fit(result)

    def test_different_basket_and_leveraged_exposure_rejected(self):
        for change in ({"holdings_krw": {"OTHER": D("100")}},
                       {"holdings_krw": {"KR": D("1001")}}, {"equity_krw": D("0")}):
            result = learning()
            result.valuations[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                fit(result)

    def test_equal_weights_sum_exactly_for_three_assets(self):
        instruments = [replace(KR, symbol=symbol) for symbol in "ABC"]
        with localcontext() as context:
            context.prec = 80
            frozen = fit(learning(symbols=tuple("ABC")), instruments)
            self.assertEqual(sum(dict(frozen.weights).values()), D("1"))

    def test_inverse_volatility_uses_only_known_learning_estimates(self):
        instruments = [replace(KR, symbol=symbol) for symbol in "AB"]
        estimates = {"A": VolatilityEstimate(D(".1"), at(date(2024, 5, 1))),
                     "B": VolatilityEstimate(D(".3"), at(date(2024, 5, 1)))}
        frozen = fit(learning(symbols=("A", "B")), instruments, volatilities=estimates)
        self.assertEqual(dict(frozen.weights), {"A": D(".75"), "B": D(".25")})
        estimates["A"] = replace(estimates["A"], known_at=at())
        with self.assertRaisesRegex(ValueError, "cutoff"):
            fit(learning(symbols=("A", "B")), instruments, volatilities=estimates)

    def test_policy_copies_weights_and_has_reproducible_hash(self):
        self.assertEqual(fit().sha256, fit().sha256)
        self.assertNotEqual(fit().sha256, fit(learning((".3", ".6"))).sha256)

    def test_missing_inverse_volatility_estimate_is_not_silently_equal_weighted(self):
        instruments = [replace(KR, symbol=symbol) for symbol in "AB"]
        estimates = {"A": VolatilityEstimate(D(".1"), at(date(2024, 5, 1))), "B": None}
        with self.assertRaisesRegex(ValueError, "volatility"):
            fit(learning(symbols=("A", "B")), instruments, volatilities=estimates)

    def test_inverse_weights_and_policy_hash_are_stable_in_all_rounding_modes(self):
        instruments = [replace(KR, symbol=symbol) for symbol in "AB"]
        known = at(date(2024, 5, 1))
        estimates = {"A": VolatilityEstimate(D(".40"), known), "B": VolatilityEstimate(D(".03"), known)}
        for ordered in (instruments, instruments[::-1]):
            training = learning(symbols=tuple(item.symbol for item in ordered))
            expected_fit = fit(training, ordered, volatilities=estimates)
            expected_fixed = fixed_baseline(ordered, investment_fraction=D(".5"), known_at=known,
                                            evaluation_start=at(), volatilities=estimates)
            for mode in ROUNDING_MODES:
                with self.subTest(mode=mode, order=[i.symbol for i in ordered]), localcontext() as context:
                    context.prec, context.rounding = 7, mode
                    actual_fit = fit(training, ordered, volatilities=estimates)
                    actual_fixed = fixed_baseline(ordered, investment_fraction=D(".5"), known_at=known,
                                                  evaluation_start=at(), volatilities=estimates)
                    self.assertEqual(actual_fit, expected_fit)
                    self.assertEqual(actual_fit.sha256, expected_fit.sha256)
                    self.assertEqual(actual_fixed, expected_fixed)
                    self.assertEqual(replace(expected_fixed), expected_fixed)
                    self.assertEqual((context.prec, context.rounding), (7, mode))

    def test_diagnostic_exposure_is_stable_in_all_rounding_modes(self):
        frozen = fit()
        result = run([bar(), bar(day=JUNE + timedelta(days=1))])
        for row, equity in zip(result.valuations, (D("3"), D("7"))):
            row.update(equity_krw=equity, holdings_krw={"KR": D("1")})
        end = result.valuations[-1]["at"]
        expected = diagnostic_baseline(result, frozen, evaluation_end=end)
        for mode in ROUNDING_MODES:
            with self.subTest(mode=mode), localcontext() as context:
                context.prec, context.rounding = 7, mode
                self.assertEqual(diagnostic_baseline(result, frozen, evaluation_end=end), expected)
                self.assertEqual((context.prec, context.rounding), (7, mode))

    def test_diagnostic_policy_cannot_be_relabeled_as_primary(self):
        result = run([bar()])
        diagnostic = diagnostic_baseline(result, fit(), evaluation_end=at(JUNE, 7))
        self.assertEqual(diagnostic.role, "diagnostic")
        with self.assertRaisesRegex(ValueError, "before"):
            replace(diagnostic, role="primary")
        outcome = run([bar()], frozen=diagnostic)
        self.assertTrue(any("diagnostic only" in message for message in outcome.warnings))

    def test_invalid_policy_fractions_and_volatility_rejected(self):
        for value in ("-0.1", "1.1", "NaN", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(policy(), investment_fraction=value)
        with self.assertRaises(ValueError):
            VolatilityEstimate(D("0"), at())

    def test_policy_information_order_uses_utc_through_dst_fold(self):
        zone = ZoneInfo("America/New_York")
        known = datetime(2024, 11, 3, 1, 30, tzinfo=zone, fold=0)
        start = datetime(2024, 11, 3, 1, 15, tzinfo=zone, fold=1)
        frozen = fixed_baseline([KR], investment_fraction=D(".5"), known_at=known, evaluation_start=start)
        self.assertLess(frozen.known_at, frozen.evaluation_start)
        self.assertEqual(frozen.known_at.utcoffset(), timedelta(0))

    def test_learning_rows_with_repeated_wall_hour_are_ordered_by_instant(self):
        zone = ZoneInfo("America/New_York")
        result = learning()
        result.valuations[0]["at"] = datetime(2024, 11, 3, 1, 45, tzinfo=zone, fold=0)
        result.valuations[1]["at"] = datetime(2024, 11, 3, 1, 15, tzinfo=zone, fold=1)
        frozen = fit(result, training_start=at(date(2024, 11, 3)),
                     training_end=at(date(2024, 11, 3), 8), evaluation_start=at(date(2024, 11, 4)))
        self.assertEqual(frozen.investment_fraction, D(".4"))


class BaselineExecutionTests(unittest.TestCase):
    def test_half_exposure_cash_and_holdings(self):
        result = run([bar()])
        self.assertEqual(result.trades[0]["quantity"], 5)
        self.assertEqual(result.final_cash["KRW"], D("500"))
        self.assertEqual(result.valuations[0]["holdings_krw"]["KR"], D("500"))

    def test_monthly_partial_sale_uses_preceding_valuation_equity(self):
        result = run([bar(high=D("200"), close=D("200")), bar(day=JULY, price="200")])
        self.assertEqual([(t["side"], t["quantity"]) for t in result.trades], [("BUY", 5), ("SELL", 2)])
        targets = [r for r in result.cash_events if r["kind"] == "baseline_targets"]
        self.assertEqual(targets[1]["targets_krw"], {"KR": D("750")})
        self.assertEqual(targets[1]["source_at"], at(JUNE, 7))
        self.assertEqual(result.positions[0].quantity, 3)
        self.assertEqual(result.final_cash["KRW"], D("900"))

    def test_topup_then_one_order_sells_across_lots(self):
        result = run([bar(), bar(day=JULY, price="50"), bar(day=AUGUST, price="200")])
        self.assertEqual([(t["side"], t["quantity"]) for t in result.trades],
                         [("BUY", 5), ("BUY", 5), ("SELL", 9)])
        self.assertEqual([(p.level, p.quantity) for p in result.positions], [(2, 1)])
        self.assertEqual(result.tax["holdings"]["KR"]["quantity"], 1)
        self.assertEqual(result.tax["holdings"]["KR"]["price_cost"], D("75"))
        self.assertEqual(len([e for e in result.tax["events"] if e["kind"] == "sell"]), 1)

    def test_same_timestamp_sells_fund_buys_even_if_buyer_is_first(self):
        instruments = [replace(KR, symbol="B"), replace(KR, symbol="A")]
        bars = [bar("B"), bar("A", close=D("200"), high=D("200")),
                bar("B", JULY), bar("A", JULY, "200")]
        result = run(bars, instruments, policy(instruments, "1"))
        self.assertEqual([(t["symbol"], t["side"], t["quantity"]) for t in result.trades[2:]],
                         [("A", "SELL", 2), ("B", "BUY", 2)])
        self.assertEqual(result.rejections, [])

    def test_buy_hold_never_rebalances_or_reinvests_cash(self):
        result = run([bar(), bar(day=JULY, price="200", dividend=D("10"), dividend_pay_at=at(JULY, 2)),
                      bar(day=AUGUST, price="50")], frozen=policy(fraction="1", mode="buy_hold"))
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.positions[0].quantity, 10)
        self.assertEqual(result.final_cash["KRW"], D("100"))

    def test_zero_exposure_is_krw_cash_only_and_full_exposure_buys(self):
        zero = run([bar()], frozen=policy(fraction="0"))
        full = run([bar()], frozen=policy(fraction="1"))
        self.assertEqual(zero.trades, [])
        self.assertEqual(zero.final_cash["KRW"], D("1000"))
        self.assertEqual(full.trades[0]["quantity"], 10)

    def test_buy_fee_is_included_in_target_budget(self):
        result = run([bar()], frozen=policy(fraction="1"), commission_rate=D(".01"))
        self.assertEqual(result.trades[0]["quantity"], 9)
        self.assertEqual(result.trades[0]["fee"], D("9"))
        self.assertEqual(result.final_cash["KRW"], D("91"))

    def test_participation_rejects_whole_order(self):
        result = run([bar(adv20=D("1000"))])
        self.assertEqual(result.trades, [])
        self.assertEqual(result.rejections[0]["quantity"], 5)
        self.assertEqual(result.rejections[0]["reason"], "participation_limit")

    def test_zero_volume_or_locked_first_open_is_not_retried_that_month(self):
        for change in ({"volume": D("0")}, {"limit_locked": True}):
            result = run([bar(**change), bar(day=JUNE + timedelta(days=1))])
            self.assertEqual(result.trades, [])
            self.assertTrue(any(row["kind"] == "baseline_unavailable" for row in result.cash_events))

    def test_first_own_session_in_month_handles_market_holiday(self):
        instruments = [replace(KR, symbol="A"), replace(KR, symbol="B")]
        result = run([bar("A"), bar("B", JUNE + timedelta(days=1))], instruments,
                     policy(instruments, "1"))
        self.assertEqual([t["quantity"] for t in result.trades], [5, 5])

    def test_missing_preceding_month_evaluation_rejected(self):
        with self.assertRaisesRegex(ValueError, "preceding evaluation"):
            run([bar(), bar(day=JULY)], evaluation_times=[at(JULY, 7)])

    def test_unknown_order_and_events_before_policy_start_rejected(self):
        with self.assertRaisesRegex(ValueError, "basket"):
            run([bar("OTHER")], instruments=[replace(KR, symbol="OTHER")], frozen=policy())
        with self.assertRaisesRegex(ValueError, "precede"):
            run([bar()], conversion_times=[at(JUNE - timedelta(days=1))])

    def test_two_intraday_paths_have_same_close_results_for_open_orders(self):
        bars = [bar(high=D("130"), low=D("80"), close=D("110")), bar(day=JULY, price="95")]
        self.assertEqual(run(bars, path="O-H-L-C"), run(bars, path="O-L-H-C"))

    def test_future_close_cannot_change_prior_order(self):
        first = run([bar(), bar(day=JULY)])
        changed = run([bar(), bar(day=JULY, close=D("999"), high=D("999"))])
        self.assertEqual(first.trades, changed.trades)
        self.assertEqual(first.valuations[0], changed.valuations[0])

    def test_us_cash_requires_explicit_conversion_and_uses_common_cost(self):
        us = Instrument("US", "US", "USD")
        bars = [bar("US", open_at=at(JUNE, 13), close_at=at(JUNE, 19), price="10")]
        options = dict(instruments=[us], frozen=policy([us], "1"),
                       initial_cash={"KRW": D("100000")}, fx_rates=[FxRate(at(), D("1000"))],
                       sec_rate_override=D("0"))
        rejected = run(bars, **options)
        self.assertEqual(rejected.rejections[0]["reason"], "insufficient_cash")
        funded = run(bars, conversion_times=[at(JUNE, 1)], **options)
        self.assertEqual(funded.trades[0]["quantity"], 9)
        conversion = next(r for r in funded.cash_events if r["kind"] == "fx_conversion")
        with localcontext() as context:
            context.prec = 80
            self.assertEqual(conversion["fee_krw"], D("100000") / D("1.001") * D(".001"))

    def test_us_default_fee_uses_verified_settlement_date_without_override(self):
        us = Instrument("US", "US", "USD")
        next_day, july_next = JUNE + timedelta(days=1), JULY + timedelta(days=1)
        bars = [bar("US", price="10", high=D("20"), close=D("20")),
                bar("US", day=JULY, price="20")]
        result = run(bars, [us], policy([us]), initial_cash={"USD": D("100")},
                     fx_rates=[FxRate(at(JUNE - timedelta(days=1)), D("1000"))], commission_rate=None,
                     us_sessions=[JUNE, next_day, JULY, july_next],
                     settlement_session_times={next_day: at(next_day, 8), july_next: at(july_next, 8)})
        self.assertEqual([(row["side"], row["quantity"]) for row in result.trades], [("BUY", 4), ("SELL", 1)])
        self.assertEqual(result.trades[0]["fee"], D(".028"))
        self.assertEqual(result.trades[1]["fee"], D("20") * (D(".0007") + D(".0000278")))

    def test_gold_default_commission_and_tick_slippage(self):
        gold = Instrument("GOLD", "GOLD", "KRW")
        result = run([bar("GOLD", price="1000")], [gold], policy([gold]),
                     initial_cash={"KRW": D("10000")}, commission_rate=None, slippage_ticks=1)
        self.assertEqual(result.trades[0]["price"], D("1010"))
        self.assertEqual(result.trades[0]["quantity"], 4)
        self.assertEqual(result.trades[0]["fee"], D("12.120"))
        self.assertEqual(result.final_cash["KRW"], D("5947.880"))

    def test_future_us_price_and_unpublished_fx_cannot_change_kr_order(self):
        instruments = [KR, Instrument("US", "US", "USD")]
        bars = [bar(price="1000"), bar("US", price="100", open_at=at(JUNE, 13), close_at=at(JUNE, 19))]
        options = dict(instruments=instruments, frozen=policy(instruments),
                       initial_cash={"KRW": D("500000"), "USD": D("500")},
                       evaluation_times=[at(JUNE, 20)], fx_rates=[FxRate(at(), D("1000"))],
                       sec_rate_override=D("0"))
        initial = run(bars, **options)
        changed_bars = [bars[0], replace(bars[1], high=D("999"), close=D("999"))]
        changed = run(changed_bars, **(options | {"fx_rates": options["fx_rates"] +
                                               [FxRate(at(JUNE, 14), D("9000"))]}))
        self.assertEqual(initial.trades, changed.trades)
        self.assertEqual(initial.trades[0]["quantity"], 250)

    def test_us_multiple_lots_settle_after_trade_without_fabricated_cash(self):
        us = Instrument("US", "US", "USD")
        days = [JUNE, JUNE + timedelta(days=1), JULY, JULY + timedelta(days=1),
                AUGUST, AUGUST + timedelta(days=1)]
        bars = [bar("US", price="100"), bar("US", day=JULY, price="50"),
                bar("US", day=AUGUST, price="200")]
        result = run(bars, [us], policy([us]), initial_cash={"USD": D("1000")},
                     fx_rates=[FxRate(at(), D("1000"))], us_sessions=days,
                     settlement_session_times={day: at(day, 8) for day in days[1::2]}, sec_rate_override=D("0"))
        self.assertEqual(result.trades[-1]["quantity"], 9)
        sale = next(row for row in result.cash_events if row["kind"] == "trade" and row["side"] == "SELL")
        self.assertEqual(sale["cash"]["USD"] - sale["settled_cash"]["USD"], D("1800"))
        self.assertEqual(result.tax["holdings"]["US"]["quantity"], 1)
        self.assertEqual(result.tax["us_realized_krw"][2024], D("1125000"))

    def test_cash_floor_and_symbol_cap_reject_without_resizing(self):
        for option, reason in (({"cash_floor_fraction": D(".8")}, "cash_floor"),
                               ({"symbol_cap_fraction": D(".2")}, "symbol_cap")):
            result = run([bar()], **option)
            self.assertEqual(result.rejections[0]["reason"], reason)
            self.assertEqual(result.rejections[0]["quantity"], 5)

    def test_input_policy_bars_and_cash_remain_unchanged(self):
        frozen, bars, cash = policy(), [bar(), bar(day=JULY)], {"KRW": D("1000")}
        fingerprint = frozen.sha256
        first = run(bars, frozen=frozen, initial_cash=cash)
        self.assertEqual(first, run(bars, frozen=frozen, initial_cash=cash))
        self.assertEqual(frozen.sha256, fingerprint)
        self.assertEqual(cash, {"KRW": D("1000")})


class BatchTaxSaleTests(unittest.TestCase):
    def ledger(self):
        instrument = replace(KR, tax_category="기타")
        ledger = TaxLedger(mode="after_tax", synthetic=True)
        first = Position("KR", 1, 5, D("100"), D("0"), D("0"), at())
        second = Position("KR", 2, 5, D("50"), D("0"), D("0"), at(JULY))
        ledger.buy(instrument, first, tax_basis=D("100"))
        ledger.buy(instrument, second, tax_basis=D("50"))
        return ledger, instrument, first, second

    def test_multi_lot_sale_rounds_tax_once_and_keeps_average_basis(self):
        ledger, instrument, first, second = self.ledger()
        tax = ledger.sell_lots(instrument, [first, replace(second, quantity=4)], D("200"), D("0"),
                               at(AUGUST), tax_basis=D("200"))
        self.assertEqual(tax, D("173"))
        self.assertEqual(ledger.snapshot()["holdings"]["KR"]["price_cost"], D("75"))
        self.assertEqual(len([e for e in ledger.snapshot()["events"] if e["kind"] == "sell"]), 1)

    def test_duplicate_lot_is_rejected_before_mutation(self):
        ledger, instrument, first, _ = self.ledger()
        before = ledger.snapshot()
        with self.assertRaises(ValueError):
            ledger.sell_lots(instrument, [first, first], D("200"), D("0"), at(AUGUST), tax_basis=D("200"))
        self.assertEqual(before, ledger.snapshot())

    def test_single_sale_preserves_invalid_instrument_value_error(self):
        ledger, _, first, _ = self.ledger()
        before = ledger.snapshot()
        for invalid in (None, "KR"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Instrument가 필요"):
                ledger.sell(invalid, first, D("200"), D("0"), at(AUGUST))
            self.assertEqual(ledger.snapshot(), before)

    def test_batch_sale_rejects_invalid_instrument_before_lot_access(self):
        ledger, _, first, _ = self.ledger()
        before = ledger.snapshot()
        for invalid in (None, "KR"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Instrument가 필요"):
                ledger.sell_lots(invalid, [first], D("200"), D("0"), at(AUGUST))
            self.assertEqual(ledger.snapshot(), before)

    def test_separate_same_day_other_etf_orders_are_still_rejected(self):
        ledger, instrument, first, second = self.ledger()
        ledger.sell(instrument, first, D("200"), D("0"), at(AUGUST), tax_basis=D("200"))
        with self.assertRaisesRegex(ValueError, "같은 날"):
            ledger.sell(instrument, second, D("200"), D("0"), at(AUGUST, 1), tax_basis=D("200"))

    def test_full_baseline_after_tax_aggregates_multiple_lots(self):
        instrument = replace(KR, tax_category="기타")
        bars = [bar(), bar(day=JULY, price="50"), bar(day=AUGUST, price="200")]
        result = run(bars, [instrument], policy([instrument]), tax_mode="after_tax", synthetic_tax_data=True,
                     tax_basis_by_bar={("KR", day): D(price) for day, price in
                                       ((JUNE, "100"), (JULY, "50"), (AUGUST, "200"))})
        self.assertEqual(result.trades[-1]["withholding"], D("173"))


if __name__ == "__main__":
    unittest.main()
