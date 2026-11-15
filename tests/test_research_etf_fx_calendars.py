"""ETF 연구 환율·금리(fx.py)와 거래 달력(calendars.py) 단위 테스트.

네트워크·외부 자료·exchange_calendars 없이 돈다. 실제 자료 대조 결과는 자료 폴더의
calendar_crosscheck.json, fx_manifest.json에 남는다.
"""
import os
import sys
import tempfile
import unittest
from datetime import date, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.research.etf import calendars as cal  # noqa: E402
from src.research.etf import fx  # noqa: E402

UTC = "UTC"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


class TestLocalToUtc(unittest.TestCase):
    def test_nyse_open_shifts_with_us_daylight_saving(self):
        # 2024: 서머타임 시작 3월 둘째 일요일(3-10)
        self.assertEqual(cal.local_to_utc(date(2024, 3, 8), time(9, 30), "America/New_York").hour, 14)
        self.assertEqual(cal.local_to_utc(date(2024, 3, 11), time(9, 30), "America/New_York").hour, 13)
        # 2007년 규칙 변경 전: 2006년은 4월 첫 일요일(4-2)에 시작
        self.assertEqual(cal.local_to_utc(date(2006, 3, 13), time(9, 30), "America/New_York").hour, 14)
        self.assertEqual(cal.local_to_utc(date(2006, 4, 3), time(9, 30), "America/New_York").hour, 13)
        self.assertEqual(cal.local_to_utc(date(2007, 3, 12), time(9, 30), "America/New_York").hour, 13)
        # 2024: 서머타임 종료 11월 첫 일요일(11-3)
        self.assertEqual(cal.local_to_utc(date(2024, 11, 4), time(16, 0), "America/New_York").hour, 21)

    def test_krx_has_no_daylight_saving(self):
        for d in (date(2024, 1, 3), date(2024, 7, 3)):
            u = cal.local_to_utc(d, time(9, 0), "Asia/Seoul")
            self.assertEqual((u.date(), u.hour), (d, 0))  # 09:00 KST = 같은 날 00:00 UTC


class TestCalendarRows(unittest.TestCase):
    def test_krx_regular_close_extended_2016_08_01(self):
        self.assertEqual(cal.regular_hours("KRX", date(2016, 7, 29)), (time(9), time(15)))
        self.assertEqual(cal.regular_hours("KRX", date(2016, 8, 1)), (time(9), time(15, 30)))

    def test_nyse_early_close_flags(self):
        r = cal._row("NYSE", date(2024, 11, 29), time(9, 30), time(13, 0), "t")
        self.assertTrue(r["early_close"])
        self.assertFalse(r["late_open"])
        self.assertEqual(r["close_utc"], "2024-11-29T18:00:00+00:00")
        self.assertEqual(r["utc_offset_hours"], -5.0)
        self.assertFalse(r["is_dst"])
        summer = cal._row("NYSE", date(2024, 7, 3), time(9, 30), time(13, 0), "t")
        self.assertTrue(summer["is_dst"])
        self.assertEqual(summer["close_utc"], "2024-07-03T17:00:00+00:00")

    def test_krx_csat_day_flags(self):
        r = cal._row("KRX", date(2025, 11, 13), time(10, 0), time(16, 30), "t")
        self.assertTrue(r["late_open"])
        self.assertTrue(r["late_close"])
        self.assertFalse(r["early_close"])
        self.assertEqual(r["open_utc"], "2025-11-13T01:00:00+00:00")

    def test_overrides_close_and_change_hours(self):
        rows = [cal._row("KRX", d, time(9), time(15, 30), "lib")
                for d in (date(2026, 7, 16), date(2026, 7, 17), date(2026, 11, 19))]
        out = cal.apply_overrides(rows, cal.KRX_OVERRIDES)
        self.assertEqual([r["date"] for r in out], ["2026-07-16", "2026-11-19"])
        csat = out[1]
        self.assertEqual((csat["open_local"], csat["close_local"]), ("10:00", "16:30"))
        self.assertTrue(csat["source"].startswith("override:"))

    def test_known_overrides_are_present(self):
        kinds = {(o.day, o.kind) for o in cal.KRX_OVERRIDES}
        self.assertIn((date(2026, 7, 17), "closed"), kinds)
        self.assertIn((date(2026, 11, 19), "hours"), kinds)


def _write_calendar(path: Path, market: str, days, hours) -> pd.DataFrame:
    rows = [cal._row(market, d, *hours, "t") for d in days]
    pd.DataFrame(rows).to_csv(path, index=False)
    return cal.load_calendar(path)


class TestCalendarLoadAndEvents(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        # 2024-07-04 미국 휴장, 2024-07-03 미국 조기 폐장은 여기선 단순화(정상 시각)
        self.krx = _write_calendar(d / "k.csv", "KRX",
                                   [date(2024, 7, 3), date(2024, 7, 4), date(2024, 7, 5)], (time(9), time(15, 30)))
        self.nyse = _write_calendar(d / "n.csv", "NYSE",
                                    [date(2024, 7, 3), date(2024, 7, 5)], (time(9, 30), time(16)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_parses_utc_and_bools(self):
        self.assertEqual(str(self.krx["open_utc"].dt.tz), "UTC")
        self.assertEqual(self.krx["is_dst"].dtype, bool)

    def test_load_rejects_duplicates(self):
        p = Path(self.tmp.name) / "dup.csv"
        rows = [cal._row("KRX", date(2024, 7, 3), time(9), time(15, 30), "t")] * 2
        pd.DataFrame(rows).to_csv(p, index=False)
        with self.assertRaises(ValueError):
            cal.load_calendar(p)

    def test_combined_sessions_union(self):
        comb = cal.combined_sessions(self.krx, self.nyse)
        self.assertEqual(len(comb), 3)
        row = comb[comb["date"] == pd.Timestamp("2024-07-04")].iloc[0]
        self.assertTrue(row["krx_open"])
        self.assertFalse(row["nyse_open"])

    def test_events_are_in_utc_order(self):
        ev = cal.session_events_utc(self.krx, self.nyse)
        self.assertTrue(ev["at_utc"].is_monotonic_increasing)
        seq = list(zip(ev["market"], ev["session_date"].dt.strftime("%m-%d"), ev["event"]))
        # 국내 7-3 마감(06:30 UTC) → 미국 7-3 개장(13:30) → 미국 7-3 마감(20:00) → 국내 7-4 개장(7-4 00:00 UTC)
        i = seq.index(("KRX", "07-03", "close"))
        self.assertEqual(seq[i + 1: i + 4], [("NYSE", "07-03", "open"), ("NYSE", "07-03", "close"),
                                             ("KRX", "07-04", "open")])

    def test_us_close_of_same_date_is_unknown_before_krx_open(self):
        # 국내 7-5 08:00 KST(= 7-4 23:00 UTC)에 알려진 마지막 미국 종가는 7-3(7-4 휴장)
        last = cal.last_session_on_or_before(self.nyse, ts("2024-07-04 23:00"))
        self.assertEqual(last["date"], pd.Timestamp("2024-07-03"))
        # 미국 7-5 종가는 7-5 20:00 UTC 전에는 모른다
        last = cal.last_session_on_or_before(self.nyse, ts("2024-07-05 19:59"))
        self.assertEqual(last["date"], pd.Timestamp("2024-07-03"))
        with self.assertRaises(ValueError):
            cal.last_session_on_or_before(self.nyse, pd.Timestamp("2024-07-05"))

    def test_crosscheck_sessions(self):
        res = cal.crosscheck_sessions(["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-08"],
                                      ["2024-07-02", "2024-07-03", "2024-07-04", "2024-07-05"])
        self.assertEqual(res["overlap"], ("2024-07-02", "2024-07-05"))
        self.assertEqual(res["calendar_only"], [])
        self.assertEqual(res["observed_only"], ["2024-07-04", "2024-07-05"])


class TestFedHolidaysAndRelease(unittest.TestCase):
    def setUp(self):
        self.hol = fx.us_federal_holidays("2015-01-01", "2026-12-31")

    def test_holiday_set(self):
        self.assertIn(date(2024, 1, 15), self.hol)       # MLK
        self.assertIn(date(2021, 7, 5), self.hol)        # 일요일 독립기념일 → 월요일
        self.assertIn(date(2022, 6, 20), self.hol)       # 준틴스 대체
        self.assertIn(date(2018, 12, 24), self.hol)      # 대통령령 휴무
        self.assertNotIn(date(2021, 12, 31), self.hol)   # 토요일 새해 → 금요일 이동분은 연준 영업일

    def test_h10_release_date(self):
        self.assertEqual(fx.h10_release_date(date(2024, 3, 1), self.hol), date(2024, 3, 4))   # 금 → 월
        self.assertEqual(fx.h10_release_date(date(2024, 2, 26), self.hol), date(2024, 3, 4))  # 월 → 다음 월
        self.assertEqual(fx.h10_release_date(date(2024, 1, 12), self.hol), date(2024, 1, 16))  # MLK → 화
        self.assertEqual(fx.h10_release_date(date(2018, 12, 21), self.hol), date(2018, 12, 26))

    def test_next_business_day(self):
        self.assertEqual(fx.next_business_day(date(2024, 7, 3), self.hol), date(2024, 7, 5))
        self.assertEqual(fx.next_business_day(date(2024, 3, 1), self.hol), date(2024, 3, 4))


def _fred_frame(rows):
    return pd.DataFrame({"date": pd.to_datetime([r[0] for r in rows]), "value": [r[1] for r in rows]})


class TestDexkous(unittest.TestCase):
    def setUp(self):
        self.t = fx.dexkous_table(_fred_frame([
            ("2024-02-23", 1331.0), ("2024-02-26", 1332.0), ("2024-03-01", 1334.8),
            ("2024-07-05", 1380.0),
        ]))

    def test_times(self):
        r = self.t.set_index("observation_date")
        self.assertEqual(r.loc["2024-03-01", "observed_at_utc"], ts("2024-03-01 17:00"))   # 정오 EST
        self.assertEqual(r.loc["2024-03-01", "available_at_utc"], ts("2024-03-04 21:15"))  # 16:15 EST
        self.assertEqual(r.loc["2024-07-05", "available_at_utc"], ts("2024-07-08 20:15"))  # 16:15 EDT
        self.assertEqual(r.loc["2024-02-26", "lag_days"], 7)
        self.assertEqual(r.loc["2024-03-01", "lag_days"], 3)

    def test_asof_never_uses_unpublished_value(self):
        self.assertEqual(fx.asof(self.t, ts("2024-03-04 21:14"))["observation_date"], "2024-02-23")
        self.assertEqual(fx.asof(self.t, ts("2024-03-04 21:15"))["observation_date"], "2024-03-01")
        self.assertIsNone(fx.asof(self.t, ts("2024-02-26 00:00")))
        with self.assertRaises(ValueError):
            fx.asof(self.t, pd.Timestamp("2024-03-05"))

    def test_asof_series_matches_asof(self):
        q = [ts("2024-02-26 00:00"), ts("2024-03-04 21:15"), ts("2024-03-04 21:14"), ts("2024-08-01")]
        got = fx.asof_series(self.t, q)
        self.assertTrue(pd.isna(got.iloc[0]))
        self.assertEqual(list(got.iloc[1:]), [1334.8, 1331.0, 1380.0])
        for at, v in zip(q[1:], got.iloc[1:]):
            self.assertEqual(fx.asof(self.t, at)["value"], v)


class TestSmbsAndRates(unittest.TestCase):
    XML = ("<chart><set color='c' label='99.12.30' value='1138.0' />"
           "<set color='c' label='00.01.04' value='1128.0' />"
           "<set color='c' label='24.01.02' value='1289.4' />"
           "<set color='c' label='24.01.03' value='1299.3' /></chart>").encode("euc-kr")

    def test_parse_two_digit_years(self):
        df = fx.parse_smbs_xml(self.XML)
        self.assertEqual(list(df["date"].dt.year), [1999, 2000, 2024, 2024])
        self.assertEqual(df["value"].iloc[-1], 1299.3)

    def test_smbs_available_after_previous_trading_day(self):
        t = fx.smbs_table(fx.parse_smbs_xml(self.XML))
        self.assertEqual(len(t), 3)  # 첫 행은 기준 거래일을 몰라 제외
        r = t.set_index("value_date").loc["2024-01-03"]
        self.assertEqual(r["based_on_trading_date"], "2024-01-02")
        self.assertEqual(r["available_at_utc"], ts("2024-01-02 09:00"))  # 18:00 KST
        self.assertEqual(fx.asof(t, ts("2024-01-02 08:59"))["value_date"], "2024-01-02")

    def test_read_fred_csv_drops_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.csv"
            p.write_text("DATE,X\n2024-07-03,1.0\n2024-07-04,.\n2024-07-05,\n2024-07-08,2.0\n")
            df = fx.read_fred_csv(p)
        self.assertEqual(list(df["value"]), [1.0, 2.0])

    def test_us_daily_rate_next_business_day(self):
        t = fx.us_daily_rate_table(_fred_frame([("2024-07-03", 5.2), ("2024-07-05", 5.21)]))
        self.assertEqual(list(t["available_at_utc"]), [ts("2024-07-05 20:15"), ts("2024-07-08 20:15")])

    def test_monthly_rate_plus_two_months(self):
        t = fx.monthly_rate_table(_fred_frame([("2024-01-01", 3.8), ("2024-12-01", 3.4)]))
        self.assertEqual(list(t["observation_month"]), ["2024-01", "2024-12"])
        self.assertEqual(list(t["available_at_utc"]), [ts("2024-02-29 15:00"), ts("2025-01-31 15:00")])

    def test_compare_fx_aligns_by_seoul_trading_date(self):
        dex = fx.dexkous_table(_fred_frame([("2024-01-02", 1300.0), ("2024-01-03", 1310.0)]))
        smbs = pd.DataFrame({"value_date": ["2024-01-03", "2024-01-04"], "value": [1300.0, 1310.0],
                             "based_on_trading_date": ["2024-01-02", "2024-01-03"]})
        res = fx.compare_fx(dex, smbs, start="2024-01-01")
        self.assertEqual(res["seoul_trading_date"]["mean_abs_pct"], 0.0)
        self.assertGreater(res["same_label_date"]["mean_abs_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
