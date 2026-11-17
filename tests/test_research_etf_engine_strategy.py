"""제품 고정 커밋과 독립 산술로 정한 V2 매매·일봉 체결 사례."""
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D

from src.research.etf.engine_strategy import candidate_for_bar
from src.research.etf.engine_types import Bar, Instrument, LevelRule, Position


DAY = date(2026, 1, 5)
OPEN = datetime(2026, 1, 5, 0, tzinfo=timezone.utc)
US = Instrument("ETF", "US", "USD")
RULES = tuple(LevelRule(level, D("1000"), D(".1"), D(".1")) for level in range(1, 5))


def bar(open="100", high=None, low=None, close=None, **extra):
    close = open if close is None else close
    high = max(D(open), D(close)) if high is None else high
    low = min(D(open), D(close)) if low is None else low
    return Bar("ETF", DAY, OPEN, OPEN + timedelta(hours=9), D(open), D(high), D(low), D(close),
               D("10000"), **extra)


def position(level=1, entry="100", quantity=10, fee="0", take=".1"):
    return Position("ETF", level, quantity, D(entry), D(fee), D(take), OPEN - timedelta(days=1))


def choose(prices, held=(), rules=RULES, **options):
    return candidate_for_bar(US, prices, list(held), rules, commission_rate=D("0"),
                             sec_rate_override=D("0"), **options)


class ManualAndSlotTests(unittest.TestCase):
    def test_first_level_only_explicitly_at_open(self):
        self.assertIsNone(choose(bar("100", high="120", low="70")))
        result = choose(bar("100", high="120", low="70"), first_entry=True)
        self.assertEqual((result.side, result.level, result.quantity, result.price, result.at),
                         ("BUY", 1, 10, D("100"), OPEN))
        self.assertEqual(result.reason, "manual_first_entry")

    def test_first_level_not_refilled_with_other_holdings(self):
        result = choose(bar("90"), [position(2, "200", take="10")], first_entry=True)
        self.assertEqual(result.level, 3)  # 2차가 있으므로 3차는 허용하며 1차는 사지 않는다.
        self.assertIsNone(choose(bar("190"), [position(2, "200", take="10")], first_entry=True))

    def test_lowest_empty_slot_and_immediate_anchor(self):
        result = choose(bar("90"), [position(1), position(3, take="10")])
        self.assertEqual((result.level, result.price), (2, D("90")))
        self.assertIsNone(choose(bar("50"), [position(3, take="10")]))

    def test_missing_second_blocks_fourth(self):
        self.assertIsNone(choose(bar("50"), [position(3, "100", take="10")]))

    def test_filled_slots_have_no_fifth_level(self):
        self.assertIsNone(choose(bar("50"), [position(level, take="10") for level in range(1, 5)]))

    def test_buy_uses_principal_without_buy_fee(self):
        self.assertIsNone(choose(bar("91"), [position(fee="1000")]))
        self.assertEqual(choose(bar("90"), [position(fee="1000")]).level, 2)

    def test_drop_is_not_rounded_to_display_price(self):
        anchor = position(entry="100.0000000000000001", quantity=3)
        self.assertIsNone(choose(bar("90.0000000000000001"), [anchor]))
        self.assertEqual(choose(bar("90"), [anchor]).level, 2)

    def test_quantity_includes_fee_and_is_integral(self):
        rules = (replace(RULES[0], budget=D("99")),)
        result = candidate_for_bar(US, bar("10"), [], rules, first_entry=True,
                                   commission_rate=D(".1"), sec_rate_override=D("0"))
        self.assertEqual(result.quantity, 9)  # 9*10 + 9*.1*10 = 99.
        result = candidate_for_bar(US, bar("10"), [], (replace(rules[0], budget=D("98.99")),),
                                   first_entry=True, commission_rate=D(".1"), sec_rate_override=D("0"))
        self.assertEqual(result.quantity, 8)

    def test_manual_budget_failure_does_not_wait_for_cheaper_price(self):
        self.assertIsNone(choose(bar("100", low="50"), rules=(replace(RULES[0], budget=D("99")),),
                                 first_entry=True))

    def test_fractional_fee_budget_boundary_is_not_truncated(self):
        for budget, expected in (("100.07", 1), ("100.06999999999999999", None)):
            result = candidate_for_bar(US, bar("100"), [], (replace(RULES[0], budget=D(budget)),),
                                       first_entry=True, commission_rate=D(".0007"), sec_rate_override=D("0"))
            self.assertEqual(None if result is None else result.quantity, expected)


class NetReturnTests(unittest.TestCase):
    def test_denominator_is_principal_not_principal_plus_buy_fee(self):
        # (1060 - 1000 - 10) / 1000 = 5%; /1010 would incorrectly reject.
        result = choose(bar("106"), [position(fee="10", take=".05")])
        self.assertEqual(result.side, "SELL")

    def test_half_up_at_exact_half_of_point_zero_one_percent(self):
        # 0.005/100 = .005%, which HALF_UP rounds to .01%.
        held = [position(quantity=1, take=".0001")]
        self.assertEqual(choose(bar("100.005"), held).side, "SELL")
        self.assertIsNone(choose(bar("100.004999999999999999"), held))

    def test_non_grid_target_is_compared_after_return_rounding(self):
        # Decision return is 1.00%; it remains below a 1.001% target.
        self.assertIsNone(choose(bar("101.004"), [position(take=".01001")]))
        self.assertEqual(choose(bar("101.005"), [position(take=".01001")]).side, "SELL")

    def test_all_costs_are_removed_before_rounding(self):
        # Gross1060 - basis1000 - buy1 - sell1.06 - tax2.12 =55.82; 5.58%.
        result = candidate_for_bar(US, bar("106"), [position(fee="1", take=".0558")], RULES,
                                   commission_rate=D(".001"), sec_rate_override=D("0"),
                                   product_sell_tax_rate=D(".002"))
        self.assertEqual(result.side, "SELL")
        result = candidate_for_bar(US, bar("106"), [position(fee="1", take=".0559")], RULES,
                                   commission_rate=D(".001"), sec_rate_override=D("0"),
                                   product_sell_tax_rate=D(".002"))
        self.assertIsNone(result)

    def test_corrected_limit_is_checked_as_well_as_mark(self):
        self.assertIsNone(choose(bar("90"), [position()], slippage_ticks=1))
        self.assertIsNone(choose(bar("110"), [position()], slippage_ticks=1))

    def test_hash_order_not_level_or_profit_order(self):
        # SHA256('ETF:1'), ('ETF:4'), ('ETF:3'), ('ETF:2') ascending.
        result = choose(bar("120"), [position(level) for level in range(2, 5)])
        self.assertEqual(result.level, 4)
        reverse = choose(bar("120"), [position(level) for level in range(4, 1, -1)])
        self.assertEqual(reverse, result)

    def test_stored_take_rate_is_not_replaced_by_new_rule(self):
        self.assertIsNone(choose(bar("110"), [position(take=".2")]))


class PathAndFillTests(unittest.TestCase):
    def test_gap_is_open_and_needs_no_penetration(self):
        sale = choose(bar("120"), [position()])
        self.assertEqual((sale.side, sale.price, sale.at), ("SELL", D("120"), OPEN))
        purchase = choose(bar("80"), [position()])
        self.assertEqual((purchase.side, purchase.price, purchase.quantity, purchase.at),
                         ("BUY", D("80"), 12, OPEN))

    def test_non_tick_aligned_open_keeps_input_open_for_gap_and_manual(self):
        self.assertEqual(choose(bar("120.005"), [position()]).price, D("120.005"))
        self.assertEqual(choose(bar("80.005"), [position()]).price, D("80.005"))
        manual = choose(bar("100.005"), first_entry=True)
        self.assertEqual((manual.price, manual.quantity), (D("100.005"), 9))

    def test_exact_touch_requires_one_tick_penetration(self):
        self.assertIsNone(choose(bar("100", high="110", low="100"), [position()]))
        self.assertIsNone(choose(bar("100", high="100", low="90"), [position()]))
        self.assertEqual(choose(bar("100", high="110.01", low="100"), [position()]).price, D("110"))
        self.assertEqual(choose(bar("100", high="100", low="89.99"), [position()]).price, D("90"))

    def test_zero_penetration_allows_touch(self):
        self.assertEqual(choose(bar("100", high="110"), [position()], penetration_ticks=0).price,
                         D("110"))

    def test_both_paths_use_first_fill_not_daily_sell_priority(self):
        prices = bar("100", high="120", low="80")
        sale = choose(prices, [position()], path="O-H-L-C")
        purchase = choose(prices, [position()], path="O-L-H-C")
        expected = OPEN + timedelta(hours=1, minutes=30, seconds=5, microseconds=400000)
        self.assertEqual((sale.side, sale.price, sale.at), ("SELL", D("110"), expected))
        self.assertEqual((purchase.side, purchase.price, purchase.at), ("BUY", D("90"), expected))

    def test_same_time_sale_precedes_purchase(self):
        rules = (RULES[0], replace(RULES[1], drop_rate=D("0")))
        result = choose(bar("100"), [position(take="0")], rules=rules)
        self.assertEqual(result.side, "SELL")

    def test_future_close_does_not_change_first_segment_sale(self):
        held = [position()]
        before = held.copy()
        result = choose(bar("100", high="120", low="80", close="90"), held)
        changed = choose(bar("100", high="120", low="80", close="115"), held)
        self.assertEqual(changed, result)
        self.assertEqual(held, before)

    def test_slippage_is_adverse_and_still_requires_profitable_final_price(self):
        result = choose(bar("100", high="110.02"), [position()], slippage_ticks=1)
        self.assertEqual((result.side, result.price), ("SELL", D("110")))

    def test_linear_cost_and_return_rounding_choose_first_profitable_tick(self):
        # 105.15 -100 -.07 -(105.15*.0008) =4.99588: 5.00%로 반올림.
        # 105.14에서는 4.985888 ->4.99%; 한 호가 관통은105.16이다.
        held = [position(quantity=1, fee=".07", take=".05")]
        result = candidate_for_bar(US, bar("100", high="105.16"), held, RULES,
                                   commission_rate=D(".0007"), sec_rate_override=D(".0001"))
        self.assertEqual((result.price, result.at), (D("105.15"), OPEN + timedelta(hours=3)))
        self.assertIsNone(candidate_for_bar(US, bar("100", high="105.15"), held, RULES,
                                            commission_rate=D(".0007"), sec_rate_override=D(".0001")))

    def test_kr_tick_band_crossing_uses_lower_band_for_downward_step(self):
        kr = Instrument("ETF", "KR", "KRW")
        rules = (RULES[0], replace(RULES[1], drop_rate=D("0"), budget=D("10000")))
        result = candidate_for_bar(kr, bar("2005", low="1999"), [position(entry="2000", take="10")],
                                   rules, commission_rate=D("0"))
        self.assertEqual((result.side, result.price, result.at),
                         ("BUY", D("2000"), OPEN + timedelta(hours=6)))

    def test_us_one_dollar_tick_boundary_and_microsecond_ceiling(self):
        result = choose(bar(".9998", high="1.01"), [position(entry=".9999", take="0")])
        expected = OPEN + timedelta(seconds=211, microseconds=764706)  # 10800/51초를 마이크로초로 올림.
        self.assertEqual((result.side, result.price, result.at), ("SELL", D(".9999"), expected))

    def test_gold_default_cost_and_ten_won_penetration(self):
        gold = Instrument("ETF", "GOLD", "KRW")
        rules = (replace(RULES[0], budget=D("1000000")),)
        entry = candidate_for_bar(gold, bar("100000"), [], rules, first_entry=True)
        self.assertEqual((entry.price, entry.quantity), (D("100000"), 9))
        # 101600 -100000 -300 -(101600*.003) =995.2: .9952% ->1.00%.
        result = candidate_for_bar(gold, bar("100000", high="101610"),
                                   [position(entry="100000", quantity=1, fee="300", take=".01")], rules)
        self.assertEqual((result.side, result.price, result.at),
                         ("SELL", D("101600"), OPEN + timedelta(hours=3)))

    def test_volume_and_lock_blocks_are_independent(self):
        self.assertIsNone(choose(replace(bar("120"), volume=D("0")), [position()]))
        self.assertIsNone(choose(bar("120", limit_locked=True), [position()]))
        self.assertEqual(choose(bar("120"), [position()]).side, "SELL")

    def test_price_rounding_violation_within_point_six_two_percent(self):
        result = choose(bar("100.62", high="100", low="99", close="100"), first_entry=True)
        self.assertEqual(result.at, OPEN)
        result = choose(bar("99.38", high="101", low="100", close="100"), first_entry=True)
        self.assertEqual(result.at, OPEN)
        with self.assertRaises(ValueError):
            choose(bar("100.62000001", high="100", low="99", close="100"), first_entry=True)
        with self.assertRaises(ValueError):
            choose(bar("99.37999999", high="101", low="100", close="100"), first_entry=True)


class FirstOrderReservationTests(unittest.TestCase):
    def setUp(self):
        self.instrument = Instrument("ETF", "KR", "KRW")
        self.day = date(2025, 3, 4)
        self.open = datetime(2025, 3, 4, tzinfo=timezone.utc)
        self.held = [Position("ETF", 1, 10, D("10000"), D("0"), D(".05"),
                              self.open - timedelta(days=1))]
        self.rules = tuple(LevelRule(level, D("100000"), D(".05"), D(".05"))
                           for level in range(1, 5))

    def prices(self, open="10000", high="10500", low="9400", close="9600"):
        return Bar("ETF", self.day, self.open, self.open + timedelta(hours=6),
                   D(open), D(high), D(low), D(close), D("10000"))

    def select(self, prices, held=None, **options):
        return candidate_for_bar(self.instrument, prices, self.held if held is None else held,
                                 self.rules, commission_rate=D("0"), **options)

    def test_opus_counterexample_unfilled_sale_locks_daily_order(self):
        # 10500 지정가 매도부터 발주한다. 고가10500은10505 관통에 못 미친다.
        # 나중의 저가9400은 BUY2의9495 관통까지 충족해도 새 주문을 허용하지 않는다.
        self.assertIsNone(self.select(self.prices(), path="O-H-L-C"))
        sale = self.select(self.prices(), path="O-H-L-C", penetration_ticks=0)
        self.assertEqual((sale.side, sale.price), ("SELL", D("10500")))

    def test_low_first_path_keeps_first_buy(self):
        # 첫 2시간에10000 ->9400, 최초 매수 체결9495: 505/600*7200=6060초.
        result = self.select(self.prices(), path="O-L-H-C")
        self.assertEqual((result.side, result.level, result.price, result.at),
                         ("BUY", 2, D("9500"), self.open + timedelta(hours=1, minutes=41)))

    def test_unfilled_buy_cannot_be_replaced_by_later_sale(self):
        # 저가9500은 매수9495 관통에 부족하며 이후10600의 익절 기회도 무시한다.
        self.assertIsNone(self.select(self.prices(high="10600", low="9500", close="10000"),
                                      path="O-L-H-C"))

    def test_first_buy_waits_past_sale_opportunity_and_keeps_original_limit(self):
        # 시가89.999의 보정지정가90은 통과하나 불리한 시가90.009는 미달한다.
        # 첫 구간의 익절을 무시하고120 ->88에서89.99에 도달할 때90으로 체결한다.
        # 3시간 +(30.01/32)*10800초 =5:48:48.375.
        result = choose(bar("89.999", high="120", low="88", close="100"),
                        [position()], path="O-H-L-C", slippage_ticks=1)
        self.assertEqual((result.side, result.level, result.price, result.at),
                         ("BUY", 2, D("90"), OPEN + timedelta(hours=5, minutes=48, seconds=48,
                                                              microseconds=375000)))

    def test_simultaneous_sale_conditions_use_hash_before_fill_test(self):
        held = [replace(self.held[0], level=level) for level in (2, 4)]
        # 두 차수의 최초 조건 시각이 같다. ETF:4 해시가ETF:2보다 앞선다.
        self.assertIsNone(self.select(self.prices(), held))
        result = self.select(self.prices(high="10505"), held)
        self.assertEqual((result.side, result.level, result.price, result.at),
                         ("SELL", 4, D("10500"), self.open + timedelta(hours=2)))
        self.assertEqual(self.select(self.prices(high="10505"), list(reversed(held))), result)

    def test_earlier_condition_wins_over_hash_of_later_sale(self):
        held = [replace(self.held[0], level=2),
                replace(self.held[0], level=4, take_rate=D(".05001"))]
        result = self.select(self.prices(high="10510", low="10000", close="10000"), held)
        self.assertEqual((result.side, result.level, result.price), ("SELL", 2, D("10500")))

    def test_condition_times_inside_same_tick_are_not_tied(self):
        # HALF_UP 경계는2차10499.5, 매수비용1원인4차10499.6이다.
        # 둘 다10500 지정가지만 발주는2차가 먼저이며 해시로4차를 고르면 안 된다.
        held = [replace(self.held[0], level=2), replace(self.held[0], level=4, buy_fee=D("1"))]
        result = self.select(self.prices(high="10505", low="10000", close="10000"), held)
        self.assertEqual((result.side, result.level, result.price), ("SELL", 2, D("10500")))

    def test_opening_condition_unfilled_blocks_later_opposite_order(self):
        # 시가110.001의 보정매도가110은 통과하지만 불리한 시가109.991은 미달한다.
        # 고가110.005는110.01 관통에 못 미치므로 뒤의90 매수도 허용하지 않는다.
        self.assertIsNone(choose(bar("110.001", high="110.005", low="80", close="100"),
                                  [position()], slippage_ticks=1))

    def test_opening_hash_winner_is_not_skipped_for_another_gap_fill(self):
        held = [position(2, take=".09"), position(4, take=".1")]
        prices = bar("110.001", high="110.005", low="100", close="100")
        self.assertIsNone(choose(prices, held, slippage_ticks=1))
        filled = choose(replace(prices, high=D("110.01")), held, slippage_ticks=1)
        self.assertEqual((filled.side, filled.level, filled.price, filled.at),
                         ("SELL", 4, D("110"), OPEN + timedelta(hours=3)))

    def test_zero_target_strict_half_up_boundary_keeps_earlier_sale(self):
        # 2차는99.995 직후, 4차는99.996 직후 조건 성립. 음수 반단위는0이 아니다.
        held = [position(2, quantity=1, take="0"), position(4, quantity=1, fee=".001", take="0")]
        result = choose(bar("99.99", high="100.01", low="99.99"), held)
        self.assertEqual((result.side, result.level, result.price), ("SELL", 2, D("100")))

    def test_condition_immediately_after_tick_is_not_recorded_before_submission(self):
        # 110에서는 보정매도가109.99라 미달. 110 직후110이 되고 관통0이면 즉시 체결한다.
        result = choose(bar("100", high="120", low="100"), [position()],
                        slippage_ticks=1, penetration_ticks=0)
        self.assertEqual((result.price, result.at),
                         (D("110"), OPEN + timedelta(hours=1, minutes=30, microseconds=1)))

    def test_next_day_cancels_unfilled_order_and_reconsiders(self):
        self.assertIsNone(self.select(self.prices()))
        next_day = replace(self.prices("9400", "9400", "9400", "9400"),
                           session_date=self.day + timedelta(days=1),
                           open_at=self.open + timedelta(days=1),
                           close_at=self.open + timedelta(days=1, hours=6))
        purchase = self.select(next_day)
        self.assertEqual((purchase.side, purchase.level, purchase.price, purchase.at),
                         ("BUY", 2, D("9400"), next_day.open_at))
        repriced = self.select(replace(next_day, open=D("10520"), high=D("10520"),
                                       low=D("10520"), close=D("10520")))
        self.assertEqual((repriced.side, repriced.price, repriced.at),
                         ("SELL", D("10520"), next_day.open_at))


class InputContractTests(unittest.TestCase):
    def test_us_requires_explicit_settlement_date_sec_rate(self):
        with self.assertRaisesRegex(ValueError, "sec_rate_override"):
            candidate_for_bar(US, bar(), [], RULES)

    def test_symbol_path_and_option_validation(self):
        with self.assertRaises(ValueError):
            choose(replace(bar(), symbol="OTHER"))
        for options in ({"path": "O-C"}, {"penetration_ticks": -1}, {"slippage_ticks": True},
                        {"product_sell_tax_rate": "NaN"}, {"first_entry": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                choose(bar(), **options)

    def test_duplicates_future_positions_and_sparse_rules_fail(self):
        for held, rules in (([position(), position()], RULES),
                            ([replace(position(), opened_at=OPEN + timedelta(hours=1))], RULES),
                            ([], (RULES[0], RULES[2]))):
            with self.subTest(held=held, rules=rules), self.assertRaises(ValueError):
                choose(bar(), held, rules)

    def test_existing_open_trade_blocks_second_trade_in_same_bar(self):
        self.assertIsNone(choose(bar("120"), [replace(position(), opened_at=OPEN)]))


if __name__ == "__main__":
    unittest.main()
