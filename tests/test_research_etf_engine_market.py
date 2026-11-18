"""외부 연결 없이 공식표 경계와 손계산 비용을 검산한다."""
import json
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from src.research.etf.engine_market import fee_amount, round_price, sec_fee_rate, tick_size


D = Decimal
DAY = date(2024, 6, 3)
# 각 행의 공식 효력일과 USD 1백만당 비용. 기대값은 실행 데이터에서 생성하지 않는다.
SEC_TRANSITIONS = (
    ("2011-01-01", "16.90"), ("2011-01-21", "19.20"),
    ("2012-02-21", "18.00"), ("2012-04-01", "22.40"),
    ("2013-05-25", "17.40"), ("2014-03-18", "22.10"),
    ("2015-02-14", "18.40"), ("2016-02-16", "21.80"),
    ("2017-07-04", "23.10"), ("2018-05-22", "13.00"),
    ("2019-04-16", "20.70"), ("2020-02-18", "22.10"),
    ("2021-02-25", "5.10"), ("2022-05-14", "22.90"),
    ("2023-02-27", "8.00"), ("2024-05-22", "27.80"),
    ("2025-05-14", "0.00"), ("2026-04-04", "20.60"),
)


class TickTests(unittest.TestCase):
    def test_kr_legacy_and_stock_tick_change_does_not_change_etf(self):
        for day in (date(2011, 1, 3), date(2023, 1, 25), date(2023, 12, 10)):
            for price in (D("10"), D("1999"), D("2000"), D("500000")):
                with self.subTest(day=day, price=price):
                    self.assertEqual(tick_size("KR", day, price), D("5"))

    def test_kr_final_effective_date_and_price_boundary(self):
        for day in (date(2023, 12, 11), date(2023, 12, 18), date(2026, 11, 17)):
            for price, expected in (("1999.99", "1"), ("2000", "5"), ("500000", "5")):
                with self.subTest(day=day, price=price):
                    self.assertEqual(tick_size("KR", day, D(price)), D(expected))

    def test_gold_open_date_units_and_alias(self):
        for market in ("GOLD", "KRX_GOLD"):
            for day in (date(2014, 3, 24), DAY):
                self.assertEqual(tick_size(market, day, D("100001")), D("10"))
            with self.assertRaisesRegex(ValueError, "확인 범위"):
                tick_size(market, date(2014, 3, 23), D("100001"))

    def test_us_minimum_quotation_increment(self):
        for price, expected in (("0.9999", "0.0001"), ("1", "0.01"), ("123.456", "0.01")):
            self.assertEqual(tick_size("US", DAY, D(price)), D(expected))
        # 2024년에 채택한 $0.005 규칙은 SEC 2026년 연장 명령상 아직 적용되지 않는다.
        self.assertEqual(tick_size("US", date(2026, 11, 17), D("10")), D("0.01"))

    def test_rounding_direction_and_threshold_crossing(self):
        cases = (
            ("KR", "1999.99", "1999", "2000"),
            ("KR", "2000.01", "2000", "2005"),
            ("KR", "2005", "2005", "2005"),
            ("US", "1.999", "1.99", "2.00"),
            ("US", "0.99999", "0.9999", "1.0000"),
            ("GOLD", "100001", "100000", "100010"),
            ("KRX_GOLD", "100001", "100000", "100010"),
        )
        for market, price, buy, sell in cases:
            with self.subTest(market=market, price=price):
                self.assertEqual(round_price(market, DAY, D(price), "BUY"), D(buy))
                self.assertEqual(round_price(market, DAY, D(price), "SELL"), D(sell))

    def test_zero_rounded_buy_is_rejected_instead_of_raising_bid(self):
        with self.assertRaisesRegex(ValueError, "결과가 0"):
            round_price("KR", DAY, D("0.5"), "BUY")
        self.assertEqual(round_price("KR", DAY, D("0.5"), "SELL"), D("1"))

    def test_tick_unverified_dates_fail(self):
        for market in ("KR", "US", "GOLD", "KRX_GOLD"):
            for day in (date(2010, 12, 31), date(2026, 11, 18), date(2030, 1, 1)):
                with self.subTest(market=market, day=day):
                    with self.assertRaisesRegex(ValueError, "확인 범위"):
                        tick_size(market, day, D("100"))

    def test_invalid_tick_inputs_fail(self):
        for price in (D("0"), D("-1"), D("NaN"), D("Infinity"), True):
            with self.subTest(price=price), self.assertRaises(ValueError):
                tick_size("KR", DAY, price)
        with self.assertRaises(ValueError):
            tick_size("ETF", DAY, D("100"))
        with self.assertRaises(ValueError):
            tick_size("KR", datetime(2024, 6, 3), D("100"))
        with self.assertRaises(ValueError):
            round_price("KR", DAY, D("100"), "HOLD")


class SecTests(unittest.TestCase):
    def test_every_sec_transition_and_previous_day(self):
        for index, (stamp, per_million) in enumerate(SEC_TRANSITIONS):
            day = date.fromisoformat(stamp)
            with self.subTest(day=day):
                self.assertEqual(sec_fee_rate(day) * D("1000000"), D(per_million))
                if index:
                    self.assertEqual(sec_fee_rate(day - timedelta(days=1)) * D("1000000"),
                                     D(SEC_TRANSITIONS[index - 1][1]))

    def test_sec_verified_tail_and_unknown_periods(self):
        self.assertEqual(sec_fee_rate(date(2026, 11, 17)), D("0.0000206"))
        for day in (date(2010, 12, 31), date(2026, 11, 18), date(2026, 12, 31)):
            with self.subTest(day=day), self.assertRaisesRegex(ValueError, "확인 범위"):
                sec_fee_rate(day)

    def test_known_at_rejects_future_announcement_instead_of_old_rate(self):
        with self.assertRaisesRegex(ValueError, "아직 공표"):
            sec_fee_rate(date(2025, 5, 14), known_at=datetime(2025, 4, 7, tzinfo=timezone.utc))

    def test_known_at_publication_day_is_conservatively_unavailable(self):
        # 2025-04-08 EDT 하루가 끝나는 시각은 2025-04-09 04:00 UTC.
        boundary = datetime(2025, 4, 9, 4, tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError, "아직 공표"):
            sec_fee_rate(date(2025, 5, 14), known_at=boundary - timedelta(microseconds=1))
        self.assertEqual(sec_fee_rate(date(2025, 5, 14), known_at=boundary), D("0"))

    def test_known_at_handles_winter_eastern_time_and_equivalent_zone(self):
        boundary = datetime(2023, 1, 24, 5, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            sec_fee_rate(date(2023, 2, 27), known_at=boundary - timedelta(microseconds=1))
        kst = boundary.astimezone(timezone(timedelta(hours=9)))
        self.assertEqual(sec_fee_rate(date(2023, 2, 27), known_at=kst), D("0.000008"))

    def test_known_at_looks_up_charge_period_without_using_later_period(self):
        self.assertEqual(sec_fee_rate(date(2025, 5, 13),
                                     known_at=datetime(2025, 4, 1, tzinfo=timezone.utc)), D("0.0000278"))

    def test_sec_dates_and_knowledge_timestamps_require_explicit_types(self):
        with self.assertRaises(ValueError):
            sec_fee_rate(datetime(2024, 6, 3, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            sec_fee_rate(DAY, known_at=datetime(2024, 6, 3))

    def test_data_has_contiguous_periods_units_and_primary_provenance(self):
        path = Path(__file__).parents[1] / "src/research/etf/market_rules_v1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["sec_fee"]["periods"]
        self.assertEqual(len(rows), 18)
        self.assertEqual(rows[0]["from"], "2011-01-01")
        self.assertEqual(rows[-1]["through"], data["verified_on"])
        for index, row in enumerate(rows):
            with self.subTest(row=row["from"]):
                self.assertTrue(row["source_url"].startswith("https://www.sec.gov/"))
                self.assertLess(row["announced_on"], row["effective_from"])
                self.assertLessEqual(row["effective_from"], row["from"])
                self.assertEqual(D(row["per_million_usd"]) / D("1000000"), D(row["rate"]))
                if index:
                    previous_end = date.fromisoformat(rows[index - 1]["through"])
                    self.assertEqual(previous_end + timedelta(days=1), date.fromisoformat(row["from"]))


class FeeTests(unittest.TestCase):
    def test_research_commissions_and_alias_have_fractional_costs(self):
        self.assertEqual(fee_amount("KR", "BUY", DAY, D("1001")), D("0.15015"))
        self.assertEqual(fee_amount("KR", "SELL", DAY, D("1001")), D("0.15015"))
        self.assertEqual(fee_amount("US", "BUY", DAY, D("100.5")), D("0.07035"))
        for market in ("GOLD", "KRX_GOLD"):
            for side in ("BUY", "SELL"):
                self.assertEqual(fee_amount(market, side, DAY, D("333.3")), D("0.9999"))

    def test_us_sell_adds_decimal_sec_proportion(self):
        self.assertEqual(fee_amount("US", "SELL", DAY, D("100.5")), D("0.0731439"))

    def test_official_zero_period_is_distinct_from_unknown_period(self):
        self.assertEqual(fee_amount("US", "SELL", date(2025, 5, 14), D("100")), D("0.07"))
        with self.assertRaisesRegex(ValueError, "확인 범위"):
            fee_amount("US", "SELL", date(2026, 12, 31), D("0"))

    def test_explicit_zero_override_bypasses_historical_lookup(self):
        with patch("src.research.etf.engine_market.sec_fee_rate", side_effect=AssertionError("조회 금지")):
            for override in (D("0"), 0):
                self.assertEqual(fee_amount("US", "SELL", date(2030, 1, 1), D("100"),
                                            sec_rate_override=override), D("0.07"))

    def test_synthetic_commission_and_sec_overrides(self):
        self.assertEqual(fee_amount("US", "SELL", date(2000, 1, 1), D("100"),
                                    commission_rate=D("0"), sec_rate_override=D("0.00002")), D("0.002"))
        self.assertEqual(fee_amount("KR", "BUY", DAY, D("100"), commission_rate=D("0")), D("0"))

    def test_buy_and_other_markets_do_not_query_sec(self):
        with patch("src.research.etf.engine_market.sec_fee_rate", side_effect=AssertionError("조회 금지")):
            self.assertEqual(fee_amount("US", "BUY", date(2030, 1, 1), D("100")), D("0.07"))
            self.assertEqual(fee_amount("KR", "SELL", DAY, D("100"), sec_rate_override=D("0.5")), D("0.015"))

    def test_fees_are_linear_without_broker_rounding_breakpoints(self):
        for market, side in (("KR", "BUY"), ("KR", "SELL"), ("US", "SELL"), ("GOLD", "BUY")):
            with self.subTest(market=market, side=side):
                whole = fee_amount(market, side, DAY, D("1001"))
                parts = fee_amount(market, side, DAY, D("1000.99")) + fee_amount(market, side, DAY, D("0.01"))
                self.assertIsInstance(whole, Decimal)
                self.assertEqual(whole, parts)

    def test_invalid_fee_inputs_fail(self):
        for key in ("notional", "commission_rate", "sec_rate_override"):
            for value in (D("-0.01"), D("NaN"), D("Infinity"), True):
                kwargs = dict(market="US", side="SELL", on=DAY, notional=D("100"))
                kwargs[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    fee_amount(**kwargs)
        with self.assertRaises(ValueError):
            fee_amount("US", "sell", DAY, D("100"))


if __name__ == "__main__":
    unittest.main()
