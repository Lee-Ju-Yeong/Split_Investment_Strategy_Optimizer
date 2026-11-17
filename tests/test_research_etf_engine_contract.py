"""Literal product-rule/hand-calculation oracles; no adjacent-repo import or mock engine.

Expected values were fixed before inspecting implementation. The JSON and companion
document explain the arithmetic; helpers below only construct typed engine inputs.
Missing implementation is an import error, never a skipped or passing contract check.
"""

import json
import unittest
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path

from src.research.etf.engine import run_engine
from src.research.etf.engine_strategy import candidate_for_bar
from src.research.etf.engine_types import Bar, FxRate, Instrument, LevelRule, Position


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "etf_m2_cases.json"
CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
DEFAULTS = CASES["strategy_defaults"]
INTEGRATION = CASES["integration_cases"]
UTC = timezone.utc


def instant(text):
    return datetime.fromisoformat(text)


def position_from_row(row):
    values = {**DEFAULTS["position"], **row}
    return Position(
        symbol=values.get("symbol", "ETF"), level=values["level"], quantity=values["quantity"],
        entry_price=Decimal(values["entry_price"]), buy_fee=Decimal(values["buy_fee"]),
        take_rate=Decimal(values["take_rate"]),
        opened_at=instant(values.get("opened_at", DEFAULTS["opened_at"])),
    )


def rules_from_rows(rows):
    return tuple(LevelRule(row["level"], Decimal(row["budget"]), Decimal(row["drop_rate"]),
                           Decimal(row["take_rate"])) for row in rows)


def bar_from_row(row):
    values = {**DEFAULTS["bar"], **row}
    values["symbol"] = values.get("symbol", "ETF")
    values["session_date"] = date.fromisoformat(values["session_date"])
    for key in ("open_at", "close_at", "dividend_pay_at"):
        if values.get(key) is not None:
            values[key] = instant(values[key])
    return Bar(**values)


def flat_bar(symbol, day, price, *, market="KR", **overrides):
    session = date.fromisoformat(day)
    start, end = (time(13, 30), time(19, 30)) if market == "US" else (time(0), time(6))
    values = {"symbol": symbol, "session_date": day, "open": price, "high": price,
              "low": price, "close": price,
              "open_at": datetime.combine(session, start, UTC).isoformat(),
              "close_at": datetime.combine(session, end, UTC).isoformat()}
    return bar_from_row({**values, **overrides})


def one_rule(budget, *, take="1"):
    return (LevelRule(1, Decimal(budget), Decimal("0"), Decimal(take)),)


def trade_summary(result):
    return [[row["side"], row["level"], row["quantity"], Decimal(row["price"])]
            for row in result.trades]


def literal_trades(rows):
    return [[side, level, quantity, Decimal(price)] for side, level, quantity, price in rows]


class ProductRuleContractTests(unittest.TestCase):
    """Every case invokes the real candidate API with a predeclared literal answer."""

    def check_case(self, case):
        instrument = Instrument(**{**DEFAULTS["instrument"], **case.get("instrument", {})})
        options = {**DEFAULTS["options"], **case.get("options", {})}
        for key in ("commission_rate", "sec_rate_override", "product_sell_tax_rate"):
            options[key] = Decimal(options[key])
        candidate = candidate_for_bar(
            instrument, bar_from_row(case.get("bar", {})),
            [position_from_row(row) for row in case["positions"]],
            rules_from_rows(case.get("rules", DEFAULTS["rules"])), **options,
        )
        self.check_candidate(candidate, case["expected"])

    def check_candidate(self, candidate, expected):
        if expected is None:
            self.assertIsNone(candidate)
            return
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.symbol, "ETF")
        for key in ("side", "level", "quantity"):
            self.assertEqual(getattr(candidate, key), expected[key], key)
        self.assertEqual(candidate.price, Decimal(expected["price"]))
        self.assertEqual(candidate.at, instant(expected["at"]))
        self.assertIs(type(candidate.quantity), int)


def case_test(case):
    def test(self):
        self.check_case(case)
    test.__doc__ = "Literal fixture: " + case["id"]
    return test


for strategy_case in CASES["strategy_cases"]:
    setattr(ProductRuleContractTests, "test_" + strategy_case["id"], case_test(strategy_case))


class PortfolioContractTests(unittest.TestCase):
    def test_first_unfilled_order_blocks_an_opposite_order_later_in_the_day(self):
        case = INTEGRATION["first_unfilled_order_reserves_daily_order_slot"]
        source = next(row for row in CASES["strategy_cases"]
                      if row["id"] == "first_unfilled_sell_cannot_be_replaced_by_later_buy")
        instrument = Instrument(**{**DEFAULTS["instrument"], **source["instrument"]})
        initial = [position_from_row(row) for row in source["positions"]]
        for path in ("O-H-L-C", "O-L-H-C"):
            with self.subTest(path=path):
                result = run_engine(
                    [instrument], [bar_from_row(source["bar"])],
                    {instrument.symbol: rules_from_rows(source["rules"])},
                    initial_cash={"KRW": Decimal(case["initial_krw"]), "USD": Decimal("0")},
                    initial_positions=initial, first_entry=False, path=path,
                    commission_rate=Decimal("0"), penetration_ticks=1,
                )
                self.assertEqual(trade_summary(result), literal_trades(case[path]["expected_trades"]))
                self.assertEqual(result.final_cash["KRW"], Decimal(case[path]["expected_final_krw"]))
                self.assertEqual(sorted(position.level for position in result.positions),
                                 case[path]["expected_position_levels"])
                self.assertEqual(result.rejections, [])

    def test_conversion_target_subtracts_holdings_bought_after_previous_valuation(self):
        case = INTEGRATION["conversion_uses_current_holdings_after_previous_valuation"]
        instruments = [Instrument("KR_ETF", "KR", "KRW"), Instrument("US_ETF", "US", "USD")]
        bars = [flat_bar("KR_ETF", "2024-06-03", "10000", volume="0"),
                flat_bar("US_ETF", "2024-06-03", case["us_price"], market="US"),
                flat_bar("KR_ETF", "2024-06-04", "10000", volume="0"),
                flat_bar("US_ETF", "2024-06-04", case["us_price"], market="US", volume="0")]
        result = run_engine(
            instruments, bars, {"KR_ETF": one_rule("3000000"), "US_ETF": one_rule(case["us_budget"])},
            initial_cash={"KRW": Decimal(case["initial_krw"]), "USD": Decimal("0")},
            fx_rates=[FxRate(instant("2024-06-03T00:00:00+00:00"), Decimal(case["fx"]))],
            conversion_times=[instant(value) for value in case["conversion_times"]],
            evaluation_times=[instant(case["previous_evaluation_at"])], fx_cost=Decimal(case["fx_cost"]),
            commission_rate=Decimal("0"), sec_rate_override=Decimal("0"),
            us_sessions=[date(2024, 6, 3), date(2024, 6, 4)],
            settlement_session_times={date(2024, 6, 4): instant("2024-06-04T13:30:00+00:00")},
        )
        self.assertEqual(trade_summary(result), literal_trades(case["expected_trades"]))
        for currency in ("KRW", "USD"):
            self.assertEqual(result.final_cash[currency], Decimal(case["expected_final_" + currency.lower()]))
        self.assertEqual(result.valuations[0]["equity_krw"], Decimal(case["previous_equity_krw"]))
        self.assertEqual([(position.symbol, position.quantity) for position in result.positions],
                         [("US_ETF", case["us_quantity"])])

    def test_path_choice_allows_only_one_order_per_symbol_per_day(self):
        case = INTEGRATION["daily_order_limit"]
        instrument = Instrument("ETF", "US", "USD")
        bar = bar_from_row({"open": "100", "high": "120", "low": "80", "close": "100"})
        for path in ("O-H-L-C", "O-L-H-C"):
            with self.subTest(path=path):
                result = run_engine(
                    [instrument], [bar], {"ETF": rules_from_rows(DEFAULTS["rules"])},
                    initial_cash={"KRW": Decimal("0"), "USD": Decimal(case["initial_usd"])},
                    initial_positions=[position_from_row({"level": 1, "take_rate": "0.1"})],
                    first_entry=False, path=path, commission_rate=Decimal("0"), sec_rate_override=Decimal("0"),
                    fx_rates=[FxRate(instant("2024-05-31T00:00:00+00:00"), Decimal("1300"))],
                    us_sessions=[date(2024, 6, 3), date(2024, 6, 4)],
                    settlement_session_times={date(2024, 6, 4): instant("2024-06-04T13:30:00+00:00")},
                )
                self.assertEqual(trade_summary(result), literal_trades(case[path]["expected_trades"]))
                self.assertEqual(result.final_cash["USD"], Decimal(case[path]["expected_final_usd"]))
                self.assertEqual(sorted(position.level for position in result.positions),
                                 case[path]["expected_position_levels"])

    def test_first_level_is_entered_once_and_never_refilled_after_sale(self):
        case = INTEGRATION["no_first_level_reentry"]
        instrument = Instrument("KR_ETF", "KR", "KRW")
        bars = [flat_bar("KR_ETF", day, price) for day, price in zip(
            ("2024-06-03", "2024-06-04", "2024-06-05"), case["opens"])]
        result = run_engine(
            [instrument], bars, {instrument.symbol: one_rule(case["budget"], take="0.1")},
            initial_cash={"KRW": Decimal(case["initial_krw"]), "USD": Decimal("0")},
            commission_rate=Decimal(case["commission_rate"]),
        )
        self.assertEqual(trade_summary(result), literal_trades(case["expected_trades"]))
        self.assertEqual(result.final_cash["KRW"], Decimal(case["expected_final_krw"]))
        self.assertEqual(result.positions, [])
        self.assertEqual(result.rejections, [])

    def test_cash_shortfall_rejects_full_order_instead_of_reducing_quantity(self):
        case = INTEGRATION["cash_rejection"]
        instrument = Instrument("KR_ETF", "KR", "KRW")
        result = run_engine(
            [instrument], [flat_bar(instrument.symbol, "2024-06-03", case["price"])],
            {instrument.symbol: one_rule(case["budget"])},
            initial_cash={"KRW": Decimal(case["initial_krw"]), "USD": Decimal("0")},
            commission_rate=Decimal(case["commission_rate"]),
        )
        self.assertEqual(len(result.trades), case["expected_trades"])
        self.assertTrue(result.rejections)
        self.assertEqual(result.final_cash["KRW"], Decimal(case["expected_final_krw"]))
        self.assertEqual(result.positions, [])

    def test_krw_cash_does_not_fund_usd_order_without_conversion(self):
        case = INTEGRATION["currency_isolation"]
        instrument = Instrument("US_ETF", "US", "USD")
        result = run_engine(
            [instrument], [flat_bar(instrument.symbol, "2024-06-03", case["us_open"], market="US")],
            {instrument.symbol: one_rule(case["budget_usd"])},
            initial_cash={"KRW": Decimal(case["initial_krw"]), "USD": Decimal(case["initial_usd"])},
            fx_rates=[FxRate(instant("2024-06-03T00:00:00+00:00"), Decimal(case["fx"]))],
            commission_rate=Decimal("0"), sec_rate_override=Decimal("0"),
        )
        self.assertEqual(len(result.trades), case["expected_trades"])
        self.assertTrue(result.rejections)
        for currency in ("KRW", "USD"):
            self.assertEqual(result.final_cash[currency], Decimal(case["expected_final_" + currency.lower()]))
        self.assertEqual(result.positions, [])

    def test_participation_limit_rejects_order_without_partial_fill(self):
        case = INTEGRATION["participation_rejection"]
        instrument = Instrument("KR_ETF", "KR", "KRW")
        result = run_engine(
            [instrument], [flat_bar(instrument.symbol, "2024-06-03", case["price"], adv20=case["adv20"])],
            {instrument.symbol: one_rule(case["budget"])},
            initial_cash={"KRW": Decimal("200000"), "USD": Decimal("0")},
            commission_rate=Decimal(case["commission_rate"]),
            participation_rate=Decimal(case["participation_rate"]),
        )
        self.assertEqual(len(result.trades), case["expected_trades"])
        self.assertTrue(result.rejections)
        self.assertEqual(result.final_cash["KRW"], Decimal("200000"))
        self.assertEqual(result.positions, [])

    def test_monthly_budget_uses_previous_valuation_and_preserves_held_cost(self):
        case = INTEGRATION["monthly_budget"]
        instrument = Instrument("KR_ETF", "KR", "KRW")
        initial = position_from_row({"symbol": instrument.symbol, "level": 1,
                                    "quantity": case["initial_position_quantity"],
                                    "entry_price": case["initial_entry_price"],
                                    "take_rate": case["initial_take_rate"],
                                    "opened_at": "2024-01-30T00:00:00+00:00"})
        rules = (LevelRule(1, Decimal("1000"), Decimal("0"), Decimal("10")),
                 LevelRule(2, Decimal("1000"), Decimal("0.1"), Decimal("0.1")))
        result = run_engine(
            [instrument], [flat_bar(instrument.symbol, "2024-01-31", case["january_close"]),
                           flat_bar(instrument.symbol, "2024-02-01", case["february_open"])],
            {instrument.symbol: rules}, initial_cash={"KRW": Decimal(case["initial_krw"]), "USD": Decimal("0")},
            initial_positions=[initial], first_entry=False, commission_rate=Decimal("0"),
            first_level_fraction=Decimal(case["fraction"]),
            evaluation_times=[instant("2024-01-31T09:00:00+00:00"), instant("2024-02-01T09:00:00+00:00")],
        )
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0]["side"], "BUY")
        self.assertEqual(result.trades[0]["level"], case["expected_buy_level"])
        self.assertEqual(result.trades[0]["quantity"], case["expected_buy_quantity"])
        self.assertEqual(result.final_cash["KRW"], Decimal(case["expected_final_krw"]))
        held = next(position for position in result.positions if position.level == 1)
        self.assertEqual(held, initial)


class InformationAndSettlementContractTests(unittest.TestCase):
    def test_estimated_tax_reporting_is_continuous_through_year_end(self):
        case = INTEGRATION["year_end_estimated_tax_reporting_continuity"]
        instrument = Instrument("ETF", "US", "USD")
        held = position_from_row({"level": 1, "quantity": case["quantity"],
                                  "entry_price": case["entry_price_usd"], "take_rate": "0.05",
                                  "opened_at": "2024-12-26T13:30:00+00:00"})
        sessions = [date(2024, 12, day) for day in (27, 30, 31)]
        result = run_engine(
            [instrument], [flat_bar("ETF", case["sale_date"], case["sale_price_usd"], market="US")],
            {"ETF": one_rule("10000", take="0.05")},
            initial_cash={"KRW": Decimal("0"), "USD": Decimal("0")},
            initial_positions=[held], first_entry=False, tax_mode="after_tax", synthetic_tax_data=True,
            fx_rates=[FxRate(instant("2024-12-26T00:00:00+00:00"), Decimal(case["fx"]))],
            evaluation_times=[instant(row["at"]) for row in case["evaluations"]],
            commission_rate=Decimal("0"), sec_rate_override=Decimal("0"), us_sessions=sessions,
            settlement_session_times={day: datetime.combine(day, time(13, 30), UTC) for day in sessions},
        )
        self.assertEqual(len(result.valuations), len(case["evaluations"]))
        for actual, expected in zip(result.valuations, case["evaluations"]):
            for key in ("equity_krw", "estimated_tax_krw", "equity_after_estimated_tax_krw"):
                self.assertEqual(actual[key], Decimal(expected[key]), (expected["at"], key))
        for currency in ("KRW", "USD"):
            self.assertEqual(result.final_cash[currency], Decimal(case["expected_final_" + currency.lower()]))
        self.assertEqual(trade_summary(result), [["SELL", 1, 100, Decimal("200")]])

    def test_report_only_estimated_tax_does_not_reduce_budget_or_available_cash(self):
        case = INTEGRATION["estimated_tax_report_does_not_reduce_budget_or_available_cash"]
        instruments = [Instrument("KR_ETF", "KR", "KRW", "국내주식형"), Instrument("ETF", "US", "USD")]
        initial = [position_from_row({"symbol": "KR_ETF", "level": 1, "quantity": 1,
                                      "entry_price": "1000", "take_rate": "10",
                                      "opened_at": "2024-11-26T00:00:00+00:00"}),
                   position_from_row({"level": 1, "quantity": 100, "entry_price": "100", "take_rate": "0.05",
                                      "opened_at": "2024-11-26T13:30:00+00:00"})]
        bars = [flat_bar("ETF", case["sale_date"], case["us_sale_price"], market="US"),
                flat_bar("KR_ETF", "2024-11-29", "1000", volume="0"),
                flat_bar("KR_ETF", "2024-12-02", case["december_kr_open"])]
        rules = (LevelRule(1, Decimal("1000"), Decimal("0"), Decimal("10")),
                 LevelRule(2, Decimal("1000"), Decimal("0.1"), Decimal("0.1")))
        sessions = [date(2024, 11, 27), date(2024, 11, 29), date(2024, 12, 2)]
        result = run_engine(
            instruments, bars, {"KR_ETF": rules, "ETF": one_rule("10000", take="0.05")},
            initial_cash={"KRW": Decimal(case["initial_krw"]), "USD": Decimal("0")},
            initial_positions=initial, first_entry=False, tax_mode="after_tax", synthetic_tax_data=True,
            fx_rates=[FxRate(instant("2024-11-26T00:00:00+00:00"), Decimal(case["fx"]))],
            evaluation_times=[instant(case["previous_evaluation_at"])],
            first_level_fraction=Decimal(case["first_level_fraction"]),
            commission_rate=Decimal("0"), sec_rate_override=Decimal("0"), us_sessions=sessions,
            settlement_session_times={day: datetime.combine(day, time(13, 30), UTC) for day in sessions},
        )
        self.assertEqual(result.valuations[0]["equity_krw"], Decimal(case["previous_equity_krw"]))
        self.assertEqual(result.valuations[0]["estimated_tax_krw"], Decimal(case["estimated_tax_krw"]))
        self.assertEqual(result.valuations[0]["equity_after_estimated_tax_krw"], Decimal(case["report_only_equity_krw"]))
        self.assertEqual(trade_summary(result), literal_trades(case["expected_trades"]))
        for currency in ("KRW", "USD"):
            self.assertEqual(result.final_cash[currency], Decimal(case["expected_final_" + currency.lower()]))
        self.assertEqual(next(position for position in result.positions if position.level == 1), initial[0])
        self.assertEqual(result.rejections, [])

    def future_information_run(self, close, future_fx):
        case = INTEGRATION["future_information"]
        instrument = Instrument("ETF", "US", "USD")
        position = position_from_row({"level": 1, "take_rate": "10"})
        rates = [FxRate(instant("2024-05-31T00:00:00+00:00"), Decimal(case["published_fx"])),
                 FxRate(instant(case["future_fx_available_at"]), Decimal(future_fx))]
        bars = [flat_bar("ETF", "2024-06-03", case["previous_us_close"], market="US"),
                flat_bar("ETF", "2024-06-04", "100", market="US", high=close, close=close)]
        return run_engine(
            [instrument], bars, {"ETF": one_rule("1000", take="10")},
            initial_cash={"KRW": Decimal("0"), "USD": Decimal("0")}, initial_positions=[position],
            first_entry=False, fx_rates=rates, evaluation_times=[instant(case["evaluation_at"])],
            commission_rate=Decimal("0"), sec_rate_override=Decimal("0"),
        )

    def test_future_us_close_cannot_change_earlier_evaluation(self):
        before = self.future_information_run("100", "1400")
        after = self.future_information_run("999", "1400")
        self.assert_earlier_equity(before, after)

    def test_unpublished_fx_cannot_change_earlier_evaluation(self):
        before = self.future_information_run("100", "1400")
        after = self.future_information_run("100", "9999")
        self.assert_earlier_equity(before, after)

    def test_future_us_close_and_fx_joint_change_preserves_earlier_ledger(self):
        before = self.future_information_run("100", "1400")
        after = self.future_information_run("999", "9999")
        self.assert_earlier_equity(before, after)
        self.assertEqual(before.trades, after.trades)
        self.assertEqual(before.rejections, after.rejections)

    def assert_earlier_equity(self, before, after):
        self.assertEqual(len(before.valuations), 1)
        self.assertEqual(before.valuations, after.valuations)
        self.assertEqual(before.valuations[0]["equity_krw"],
                         Decimal(INTEGRATION["future_information"]["expected_equity_krw"]))

    def unsettled_sale_run(self, *, future_fx="1400"):
        case = INTEGRATION["unsettled_sale_not_convertible"]
        instruments = [Instrument("KR_ETF", "KR", "KRW"), Instrument("ETF", "US", "USD")]
        sessions = [date(2024, 6, day) for day in (3, 4, 5)]
        settlement_times = {day: datetime.combine(day, time(13, 30), UTC) for day in sessions}
        rates = [FxRate(instant("2024-05-31T00:00:00+00:00"), Decimal(case["fx"])),
                 FxRate(instant("2024-06-04T00:00:00+00:00"), Decimal(future_fx))]
        bars = [flat_bar("KR_ETF", "2024-06-03", "100"),
                flat_bar("ETF", "2024-06-03", case["sale_price_usd"], market="US"),
                flat_bar("ETF", "2024-06-04", case["sale_price_usd"], market="US")]
        return run_engine(
            instruments, bars, {"KR_ETF": one_rule("1000"), "ETF": one_rule("1000", take="0.05")},
            initial_cash={"KRW": Decimal("0"), "USD": Decimal("0")},
            initial_positions=[position_from_row({"level": 1, "take_rate": "0.05"})],
            first_entry=False, fx_rates=rates, conversion_times=[instant(case["conversion_at"])],
            evaluation_times=[instant("2024-06-03T20:00:00+00:00")],
            us_sessions=sessions, settlement_session_times=settlement_times,
            commission_rate=Decimal("0"), sec_rate_override=Decimal("0"),
        )

    def test_us_unsettled_sale_proceeds_cannot_be_converted(self):
        case = INTEGRATION["unsettled_sale_not_convertible"]
        result = self.unsettled_sale_run()
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0]["side"], "SELL")
        self.assertEqual(result.trades[0]["quantity"], case["sale_quantity"])
        for currency in ("KRW", "USD"):
            self.assertEqual(result.final_cash[currency], Decimal(case["expected_final_" + currency.lower()]))
        self.assertEqual(result.positions, [])
        self.assertEqual(result.rejections, [])

    def test_future_settlement_fx_does_not_reprice_trade_or_earlier_valuation(self):
        before = self.unsettled_sale_run(future_fx="1400")
        after = self.unsettled_sale_run(future_fx="9999")
        self.assertEqual(before.trades, after.trades)
        self.assertEqual(before.valuations, after.valuations)
        self.assertEqual(before.valuations[0]["equity_krw"], Decimal("1430000"))


if __name__ == "__main__":
    unittest.main()
