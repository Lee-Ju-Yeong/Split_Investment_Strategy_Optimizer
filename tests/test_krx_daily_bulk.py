"""
src.pipeline.krx_daily_bulk* 단위 테스트. 외부 호출 0회, pykrx·PyMySQL 불필요(CI 편입 대상).
가짜 DB는 sqlite 메모리 DB라 INSERT 실패 시 실제 롤백까지 확인한다.
리뷰 반례(review-astra.md)마다 회귀 테스트를 둔다.
"""

import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import krx_daily_bulk as bulk
from src.pipeline import krx_daily_bulk_normalize as norm
from src.pipeline import krx_daily_bulk_tier as tier
from src.pipeline import new_key_guard as guard


D = date
A, B, C = "005930", "035720", "000300"
MARKET = {A: "STK", B: "KSQ", C: "STK"}
OLD_DATES = [D(2026, 1, 26), D(2026, 1, 27), D(2026, 1, 28), D(2026, 1, 29), D(2026, 1, 30),
             D(2026, 2, 2), D(2026, 2, 3), D(2026, 2, 4), D(2026, 2, 5), D(2026, 2, 6)]
CLOSE = {A: 70000, B: 1000, C: 2000}


# ---- sqlite 기반 가짜 MySQL 연결 -------------------------------------------

def _param(value):
    return value.isoformat() if isinstance(value, date) else value


class SqliteCursor:
    def __init__(self, conn):
        self.conn = conn
        self.cur = conn.db.cursor()
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.conn.statements.append(sql)
        if sql.strip() == "SELECT DATABASE()":
            self.rows = [(self.conn.name,)]
            return
        self.cur.execute(sql.replace("%s", "?"), tuple(_param(p) for p in params))
        self.rows = self.cur.fetchall()

    def executemany(self, sql, rows):
        self.conn.statements.append(sql)
        if self.conn.fail_on_table and f"`{self.conn.fail_on_table}`" in sql:
            raise sqlite3.IntegrityError("simulated duplicate key")
        self.cur.executemany(sql.replace("%s", "?"), [tuple(_param(v) for v in r) for r in rows])
        self.conn.inserted.append(sql.split("`")[1])

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class SqliteConn:
    def __init__(self, name="stocks_stage"):
        self.name = name
        self.db = sqlite3.connect(":memory:", isolation_level="DEFERRED")
        self.statements, self.fail_on_table, self.inserted = [], None, []
        self.commits = self.rollbacks = 0
        for spec in guard.TABLE_SPECS.values():
            cols = ", ".join(f"`{c}`" for c in spec.columns)
            self.db.execute(f"CREATE TABLE `{spec.table}` ({cols}, PRIMARY KEY ({', '.join(spec.key_columns)}))")
        self.db.commit()

    def cursor(self):
        return SqliteCursor(self)

    def commit(self):
        self.db.commit()
        self.commits += 1

    def rollback(self):
        self.db.rollback()
        self.rollbacks += 1

    def close(self):
        pass

    def rows(self, table, where="1=1"):
        return self.db.execute(f"SELECT * FROM `{table}` WHERE {where}").fetchall()

    def dates(self, table):
        col = guard.TABLE_SPECS[table].date_column
        return sorted({guard.as_date(r[0]) for r in self.db.execute(f"SELECT DISTINCT `{col}` FROM `{table}`")})

    def insert(self, table, rows):
        spec = guard.TABLE_SPECS[table]
        sql = f"INSERT INTO `{table}` VALUES ({', '.join('?' * len(spec.columns))})"
        self.db.executemany(sql, [tuple(_param(v) for v in r) for r in rows])
        self.db.commit()


def seed(conn, codes=(A, B), price=D(2026, 2, 6), mcap=None, fin=None, inv=None, short=D(2026, 2, 3),
         ind=None, tier_=None, extra_price_dates=(), snapshots=(D(2026, 1, 26), D(2026, 2, 2), D(2026, 2, 7))):
    mcap, fin, inv, ind, tier_ = (x or price for x in (mcap, fin, inv, ind, tier_))
    price_dates = [d for d in OLD_DATES if d <= price] + list(extra_price_dates)
    for d in price_dates:
        for c in codes:
            ratio = 2.0 if c == C else 1.0
            close = CLOSE[c]
            conn.insert("DailyStockPrice", [(c, d, close, close, close, close, close * ratio, ratio, close * ratio, close * ratio, close * ratio, 100)])
    for table, through in (("MarketCapDaily", mcap), ("FinancialData", fin), ("InvestorTradingTrend", inv),
                           ("ShortSellingDaily", short), ("CalculatedIndicators", ind), ("DailyStockTier", tier_)):
        for d in [x for x in price_dates if x <= through]:
            conn.insert(table, [_row(table, c, d) for c in codes])
    for s in snapshots:
        conn.insert("TickerUniverseSnapshot", [(s, c, "KOSPI" if MARKET[c] == "STK" else "KOSDAQ", None, "pykrx") for c in codes])
    return conn


def _row(table, code, day):
    return {
        "MarketCapDaily": (code, day, 10**12, 1000, "pykrx"),
        "FinancialData": (code, day, 10.5, 1.4, 5000, 50000, 1400, 2.0, 10.0, "pykrx"),
        "InvestorTradingTrend": (code, day, -300, 200, 100, 0),
        "ShortSellingDaily": (code, day, None, None, None, 1234000, "pykrx"),
        "CalculatedIndicators": (code, day, 1, 1, 0.1, 0.5, 0.5),
        "DailyStockTier": (day, code, 2, "r", 1, None, None, None, None, None, None, None, None),
    }[table]


# ---- 가짜 KRX 원천 -----------------------------------------------------------

def price_raw(rows):
    columns = ["ISU_SRT_CD", "MKT_ID", "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC", "TDD_CLSPRC",
               "CMPPREVDD_PRC", "ACC_TRDVOL", "MKTCAP", "LIST_SHRS"]
    return pd.DataFrame([[f"{v:,}" if isinstance(v, int) else v for v in r] for r in rows], columns=columns)


class FakeSource:
    def __init__(self, codes=(A, B), holidays=(), overrides=None):
        self.codes = list(codes)
        self.calls = []
        self.holidays = set(holidays)
        self.overrides = overrides or {}

    def _maybe(self, key, default):
        value = self.overrides.get(key)
        if isinstance(value, Exception):
            raise value
        return default if value is None else value

    def market_prices(self, ymd):
        self.calls.append(("prices", ymd))
        if ymd in self.holidays:
            rows = [(c, MARKET[c], 0, 0, 0, 0, 0, 0, 0, 0) for c in self.codes]
        else:
            rows = [(c, MARKET[c], CLOSE[c], CLOSE[c], CLOSE[c], CLOSE[c], 0, 100, 10**12, 1000) for c in self.codes]
            rows.append(("100000", "KNX", 500, 500, 500, 500, 0, 10, 5000, 10))
        return self._maybe(("prices", ymd), price_raw(rows))

    def fundamentals(self, ymd):
        self.calls.append(("fundamentals", ymd))
        rows = [(c, "50,000", "10.5", "1.4", "5,000", "2.0", "1,400") for c in self.codes] + [("100000", "1", "1", "1", "1", "1", "1")]
        return self._maybe(("fundamentals", ymd), pd.DataFrame(rows, columns=["ISU_SRT_CD", "BPS", "PER", "PBR", "EPS", "DVD_YLD", "DPS"]))

    def investor_net_values(self, ymd, market_id, code):
        self.calls.append(("investor", ymd, market_id, code))
        values = {7050: "100", 8000: "-300", 9000: "150", 9001: "50"}
        rows = [(c, values[code]) for c in self.codes if MARKET[c] == market_id]
        return self._maybe(("investor", ymd, market_id, code), pd.DataFrame(rows, columns=["ISU_SRT_CD", "NETBID_TRDVAL"]))

    def short_balances(self, ymd, market_code):
        self.calls.append(("short", ymd, market_code))
        market_id = "STK" if market_code == 1 else "KSQ"
        rows = [(c, "10", "1,234,000") for c in self.codes if MARKET[c] == market_id]
        return self._maybe(("short", ymd, market_code), pd.DataFrame(rows, columns=["ISU_CD", "BAL_QTY", "BAL_AMT"]))


def tier_rows(day, codes):
    rows = pd.DataFrame([{c: None for c in guard.TABLE_SPECS["DailyStockTier"].columns} for _ in codes])
    rows["date"], rows["stock_code"], rows["tier"] = day, list(codes), 2
    return rows


def options(end, **kwargs):
    kwargs.setdefault("expected_database", "stocks_stage")
    kwargs.setdefault("run_indicators", False)
    return bulk.BulkOptions(end_date=end, **kwargs)


def run(conn, source, opts, recorder=None):
    plan = bulk.build_plan(conn, opts)
    return plan, bulk.run_daily_bulk(conn, source, opts, plan, recorder)


# ---- 정규화·완전성 -------------------------------------------------------------

class TestMarketCompleteness(unittest.TestCase):
    def test_both_markets_are_required(self):
        raw = FakeSource(codes=(A,)).market_prices("20260209")
        with self.assertRaisesRegex(norm.BulkResponseError, "market missing"):
            norm.normalize_market_day(raw, "20260209")

    def test_unparsable_values_are_damage_not_holiday(self):
        raw = price_raw([(A, "STK", "abc", 0, 0, 0, 0, 0, 0, 0), (B, "KSQ", 0, 0, 0, 0, 0, 0, 0, 0)])
        with self.assertRaisesRegex(norm.BulkResponseError, "unparsable"):
            norm.normalize_market_day(raw, "20260216")
        zero = FakeSource(holidays={"20260216"}).market_prices("20260216")
        self.assertTrue(norm.normalize_market_day(zero, "20260216").is_holiday)

    def test_invalid_or_isin_ticker_codes_are_rejected(self):
        for bad in ("KR7005930003", "5930", None):
            raw = price_raw([(bad, "STK", 1, 1, 1, 1, 0, 1, 1, 1), (B, "KSQ", 1, 1, 1, 1, 0, 1, 1, 1)])
            with self.assertRaisesRegex(norm.BulkResponseError, "invalid ticker"):
                norm.normalize_market_day(raw, "20260209")

    def test_new_alphanumeric_codes_are_accepted(self):
        raw = price_raw([("0001A0", "STK", 1, 1, 1, 1, 0, 1, 1, 1), (B, "KSQ", 1, 1, 1, 1, 0, 1, 1, 1)])
        self.assertIn("0001A0", norm.normalize_market_day(raw, "20260209").codes)

    def test_universe_change_limit(self):
        previous = {f"{i:06d}" for i in range(1000)}
        ok = norm.check_universe_change(previous - {"000001"} | {"999999"}, previous)
        self.assertEqual((ok["added"], ok["removed"]), (["999999"], ["000001"]))
        with self.assertRaisesRegex(norm.BulkResponseError, "universe_change_exceeds_limit"):
            norm.check_universe_change(set(list(previous)[:900]), previous)


class TestTableNormalization(unittest.TestCase):
    def test_fundamental_keeps_existing_rules_and_drops_outside_universe(self):
        out, info = norm.normalize_fundamental_day(FakeSource().fundamentals("20260209"), "20260209", [A, B])
        self.assertEqual(sorted(out["stock_code"]), [A, B])
        self.assertEqual(info["outside_universe"], 1)
        self.assertAlmostEqual(out.set_index("stock_code").loc[A, "roe"], 10.0)

    def test_investor_foreigner_sum_partial_and_outside(self):
        series = bulk._fetch_investor_series(FakeSource(), "20260209")
        series["other_foreigner"][1] = pd.Series([], dtype=float)           # B 기타외국인 응답 없음
        series["institution"][0] = pd.concat([series["institution"][0], pd.Series({"069500": 1.0})])
        out, info = norm.normalize_investor_day(series, "20260209", [A, B])
        self.assertEqual(list(out["stock_code"]), [A])
        self.assertEqual(int(out.iloc[0]["foreigner_net_buy"]), 200)
        self.assertEqual((info["partial"], info["outside_universe"]), ([B], 1))

    def test_short_dash_is_unpublished_not_zero(self):
        frames = [norm.short_response_frame(pd.DataFrame({"ISU_CD": [A], "BAL_AMT": ["-"]}), "s1"),
                  norm.short_response_frame(pd.DataFrame({"ISU_CD": [B], "BAL_AMT": ["0"]}), "s2")]
        out, info = norm.normalize_short_balance_day(frames, "20260209", [A, B, C])
        self.assertEqual(info["unpublished"], [A])
        self.assertEqual(info["missing_from_response"], [C])
        self.assertEqual(list(out["stock_code"]), [B])
        self.assertTrue(out[["short_volume", "short_value", "short_balance"]].isna().all(axis=None))


class TestAdjustedPrices(unittest.TestCase):
    def _prices(self, **change):
        return pd.DataFrame([{"stock_code": c, "open_price": 0 if c == B else 100, "high_price": 110, "low_price": 90,
                              "close_price": 105, "change_vs_prev": change.get(c, 5), "date": "2026-02-09"} for c in ("A", "B", "C", "D", "E", "F")])

    def test_only_unit_ratio_rows_without_event_are_carried(self):
        prev = D(2026, 2, 6)
        prior = {
            "A": norm.PriorPriceState(prev, 100.0, 100.0, 1.0),
            "B": norm.PriorPriceState(prev, 100.0, 200.0, 2.0),        # 비율 2: 단위 미확인
            "C": norm.PriorPriceState(D(2026, 2, 5), 100.0, 100.0, 1.0),  # 공백
            "E": norm.PriorPriceState(prev, 50.0, 50.0, 1.0),          # 기준가 100 != 50
            "F": norm.PriorPriceState(prev, 100.0, None, None),        # NULL 비율
        }
        result = norm.apply_adjusted_carry_forward(self._prices(), prior, prev)
        self.assertEqual(list(result.rows["stock_code"]), ["A"])
        row = result.rows.iloc[0]
        self.assertEqual((row["adj_close"], row["adj_ratio"], row["adj_low"]), (105.0, 1.0, 90.0))
        self.assertEqual(result.quarantined, {"B": "prior_ratio_not_one", "C": "gap_since_prior_row", "D": "no_prior_row",
                                              "E": "base_price_mismatch", "F": "prior_ratio_not_one"})

    def test_unparsable_change_is_quarantined_not_carried(self):
        prev = D(2026, 2, 6)
        prices = self._prices(A=np.nan)
        result = norm.apply_adjusted_carry_forward(prices, {"A": norm.PriorPriceState(prev, 50.0, 50.0, 1.0)}, prev)
        self.assertEqual(result.quarantined["A"], "unparsable_change")

    def test_quarantined_code_stays_quarantined_next_day(self):
        prev = D(2026, 2, 6)
        prior = {"A": norm.PriorPriceState(prev, 100.0, 100.0, 1.0), "B": norm.PriorPriceState(prev, 100.0, 200.0, 2.0)}
        day1 = norm.apply_adjusted_carry_forward(self._prices().head(2), prior, prev)
        prior2 = norm.next_prior_state(day1.rows, D(2026, 2, 9), prior)
        prices2 = self._prices(A=0, B=0).head(2)
        day2 = norm.apply_adjusted_carry_forward(prices2, prior2, D(2026, 2, 9))
        self.assertEqual(day2.quarantined, {"B": "gap_since_prior_row"})


# ---- 보호 경계 ---------------------------------------------------------------

class TestGuard(unittest.TestCase):
    def test_protected_until_is_later_of_max_and_floor(self):
        self.assertEqual(guard.protected_until(D(2026, 2, 3), D(2026, 2, 9)), D(2026, 2, 8))
        self.assertEqual(guard.protected_until(D(2026, 2, 10), D(2026, 2, 9)), D(2026, 2, 10))
        self.assertEqual(guard.protected_until(D(2026, 2, 3), None), D(2026, 2, 3))

    def test_target_database_checks(self):
        with self.assertRaisesRegex(guard.NewKeyViolation, "target_database_mismatch"):
            guard.assert_target_database(SqliteConn("stocks"), "stocks_stage", None, True)
        with self.assertRaisesRegex(guard.NewKeyViolation, "origin_write_requires_write_floor"):
            guard.assert_target_database(SqliteConn("stocks"), "stocks", None, True)
        self.assertEqual(guard.assert_target_database(SqliteConn("stocks"), "stocks", D(2026, 2, 9), True), "stocks")
        self.assertEqual(guard.assert_target_database(SqliteConn("stocks"), "stocks", None, False), "stocks")

    def test_insert_sql_never_updates_existing_keys(self):
        conn = SqliteConn()
        guard.insert_new_keys(conn, "MarketCapDaily", [(A, "2026-02-09", 1, 1, "pykrx")], D(2026, 2, 6))
        sql = conn.statements[-1].upper()
        for forbidden in ("DUPLICATE", "REPLACE", "IGNORE", "UPDATE"):
            self.assertNotIn(forbidden, sql)
        with self.assertRaises(guard.NewKeyViolation):
            guard.insert_new_keys(conn, "MarketCapDaily", [(A, "2026-02-06", 1, 1, "pykrx")], D(2026, 2, 6))


# ---- 계획: 보호일·공매도 지연·종료일 ---------------------------------------

class TestPlan(unittest.TestCase):
    def test_write_floor_keeps_short_feb_4_to_6_out_of_origin(self):
        conn = seed(SqliteConn("stocks"))
        plan = bulk.build_plan(conn, options(D(2026, 2, 13), expected_database="stocks", write_floor=D(2026, 2, 9)))
        self.assertTrue(all(d >= D(2026, 2, 9) for d in plan.short_source_dates))
        self.assertEqual(plan.short_source_dates, [D(2026, 2, 9), D(2026, 2, 10)])
        without_floor = bulk.build_plan(conn, options(D(2026, 2, 13)))
        self.assertEqual(without_floor.short_source_dates[:3], [D(2026, 2, 4), D(2026, 2, 5), D(2026, 2, 6)])

    def test_short_lag_does_not_advance_without_new_prices(self):
        # 반례: 주가 2/6 고정, 공매도 MAX를 3·4·5일로 바꿔도 상한은 계속 2/3이어야 한다.
        for short_max in (D(2026, 2, 3), D(2026, 2, 4), D(2026, 2, 5)):
            conn = seed(SqliteConn(), short=short_max)
            plan, report = run(conn, FakeSource(), options(D(2026, 2, 6)))
            self.assertEqual(plan.short_source_dates, [])
            self.assertEqual(report["short"], [])
            self.assertEqual(conn.dates("ShortSellingDaily")[-1], short_max)

    def test_end_date_and_max_dates_bound_all_steps(self):
        # 반례: 주가가 2/13까지 미리 있고 다른 표가 뒤처진 상태에서 종료일 2/9, --max-dates 1.
        extra = [D(2026, 2, 9), D(2026, 2, 10), D(2026, 2, 11), D(2026, 2, 12), D(2026, 2, 13)]
        conn = seed(SqliteConn(), extra_price_dates=extra, mcap=D(2026, 2, 6), fin=D(2026, 2, 6), inv=D(2026, 2, 6),
                    ind=D(2026, 2, 6), tier_=D(2026, 2, 6))
        opts = options(D(2026, 2, 13), max_dates=1, run_indicators=True, run_tier=True)
        plan = bulk.build_plan(conn, opts)
        self.assertEqual([p.day for p in plan.date_plans], [D(2026, 2, 9)])
        self.assertEqual(plan.run_end, D(2026, 2, 9))
        self.assertTrue(max(plan.short_source_dates) <= D(2026, 2, 4))
        self.assertEqual(plan.indicator_dates, [D(2026, 2, 9)])
        self.assertEqual(plan.tier_dates, [D(2026, 2, 9)])

    def test_empty_tables_stop_before_any_login(self):
        conn = SqliteConn()
        seed(conn)
        conn.db.execute("DELETE FROM ShortSellingDaily")
        conn.db.commit()
        with self.assertRaisesRegex(bulk.BulkStopError, "empty_tables_need_initial_backfill"):
            bulk.build_plan(conn, options(D(2026, 2, 13)))
        bulk.build_plan(conn, options(D(2026, 2, 13), run_short=False))

    def test_floor_leaving_a_gap_is_refused(self):
        conn = seed(SqliteConn("stocks"))
        with self.assertRaisesRegex(bulk.BulkStopError, "write_floor_leaves_gap"):
            bulk.build_plan(conn, options(D(2026, 2, 13), expected_database="stocks", write_floor=D(2026, 2, 11)))

    def test_short_lag_cap_matches_existing_rule(self):
        dates = OLD_DATES[-5:]
        self.assertEqual(bulk.short_lag_cap(dates, 3), D(2026, 2, 3))
        self.assertEqual(bulk.short_lag_cap(dates[:2], 3), D(2026, 2, 2))

    def test_no_external_work_means_no_login_and_zero_estimate(self):
        conn = seed(SqliteConn(), short=D(2026, 2, 3))
        plan = bulk.build_plan(conn, options(D(2026, 2, 6)))
        self.assertFalse(plan.needs_external_calls)
        self.assertEqual(plan.expected_krx_calls, 0)


# ---- 실행 -------------------------------------------------------------------

class TestRun(unittest.TestCase):
    def test_writes_only_new_keys_and_skips_holiday_after_one_call(self):
        conn = seed(SqliteConn())
        source = FakeSource(holidays={"20260210"})
        with tempfile.TemporaryDirectory() as tmp:
            recorder = bulk.RunRecorder(str(Path(tmp) / "run"))
            _, report = run(conn, source, options(D(2026, 2, 11), run_short=False), recorder)
            files = sorted(p.name for p in recorder.path.iterdir())
        self.assertEqual(report["holidays"], ["2026-02-10"])
        self.assertEqual([c for c in source.calls if c[1] == "20260210"], [("prices", "20260210")])
        self.assertEqual(conn.dates("DailyStockPrice")[-2:], [D(2026, 2, 9), D(2026, 2, 11)])
        self.assertIn("day-2026-02-09.json", files)
        self.assertIn("holiday-2026-02-10.json", files)
        self.assertEqual(len(conn.rows("FinancialData", "date='2026-02-09'")), 2)

    def test_real_rollback_when_a_later_insert_fails(self):
        # v3 반례: 이전 시험은 실패 표가 실제 첫 INSERT였다. 네 번째(재무) INSERT를 실패시킨다.
        conn = seed(SqliteConn())
        opts = options(D(2026, 2, 9), run_short=False)
        plan = bulk.build_plan(conn, opts)
        conn.fail_on_table = "FinancialData"
        with self.assertRaises(sqlite3.IntegrityError):
            bulk.run_daily_bulk(conn, FakeSource(), opts, plan)
        self.assertEqual(conn.inserted, ["MarketCapDaily", "TickerUniverseSnapshot", "DailyStockPrice"])
        for table, column in (("MarketCapDaily", "date"), ("TickerUniverseSnapshot", "snapshot_date"),
                              ("DailyStockPrice", "date"), ("FinancialData", "date")):
            self.assertEqual(conn.rows(table, f"{column}='2026-02-09'"), [], table)
        self.assertGreaterEqual(conn.rollbacks, 1)

    def test_failure_keeps_committed_previous_days_only(self):
        conn = seed(SqliteConn())
        source = FakeSource(overrides={("investor", "20260210", "STK", 7050): RuntimeError("simulated")})
        with self.assertRaises(RuntimeError):
            run(conn, source, options(D(2026, 2, 11), run_short=False))
        self.assertEqual(conn.dates("DailyStockPrice")[-1], D(2026, 2, 9))
        self.assertFalse(any(c[1] == "20260211" for c in source.calls))

    def test_first_empty_investor_response_stops_before_next_call(self):
        conn = seed(SqliteConn())
        source = FakeSource(overrides={("investor", "20260209", "STK", 7050): pd.DataFrame()})
        with self.assertRaisesRegex(norm.BulkResponseError, "empty response"):
            run(conn, source, options(D(2026, 2, 9), run_short=False))
        self.assertEqual(len([c for c in source.calls if c[0] == "investor"]), 1)

    def test_partial_investor_rows_stop_by_default(self):
        conn = seed(SqliteConn())
        source = FakeSource(overrides={("investor", "20260209", "KSQ", 9001): pd.DataFrame({"ISU_SRT_CD": [A], "NETBID_TRDVAL": ["1"]})})
        with self.assertRaises(norm.BulkResponseError):
            run(conn, source, options(D(2026, 2, 9), run_short=False))
        self.assertEqual(conn.rows("DailyStockPrice", "date='2026-02-09'"), [])

    def test_quarantined_prices_are_not_written_but_market_rows_are(self):
        conn = seed(SqliteConn(), codes=(A, B, C))
        with tempfile.TemporaryDirectory() as tmp:
            recorder = bulk.RunRecorder(str(Path(tmp) / "run"))
            _, report = run(conn, FakeSource(codes=(A, B, C)), options(D(2026, 2, 9), run_short=False), recorder)
            quarantine = pd.read_csv(recorder.path / "quarantine-2026-02-09.csv", dtype={"stock_code": str})
        self.assertEqual(report["quarantined"], {"2026-02-09": 1})
        self.assertEqual(list(quarantine["stock_code"]), [C])
        self.assertEqual(conn.rows("DailyStockPrice", f"date='2026-02-09' AND stock_code='{C}'"), [])
        self.assertEqual(len(conn.rows("MarketCapDaily", f"date='2026-02-09' AND stock_code='{C}'")), 1)
        self.assertEqual(len(conn.rows("TickerUniverseSnapshot", f"snapshot_date='2026-02-09' AND stock_code='{C}'")), 1)

    def test_stop_policy_halts_on_unverified_units(self):
        conn = seed(SqliteConn(), codes=(A, B, C))
        with self.assertRaisesRegex(bulk.BulkStopError, "adjusted_unit_unverified"):
            run(conn, FakeSource(codes=(A, B, C)), options(D(2026, 2, 9), run_short=False, adjust_event_policy="stop"))

    def test_prior_state_comes_from_first_actual_price_write_day(self):
        # 반례: 시총만 2/4까지라 2/5·2/6은 시세 호출(시총·명단용)만 하고 2/9에 처음 주가를 쓴다.
        conn = seed(SqliteConn(), mcap=D(2026, 2, 4))
        _, report = run(conn, FakeSource(), options(D(2026, 2, 9), run_short=False))
        self.assertEqual(report["quarantined"], {})
        self.assertEqual(len(conn.rows("DailyStockPrice", "date='2026-02-09'")), 2)
        self.assertEqual(conn.dates("MarketCapDaily")[-3:], [D(2026, 2, 5), D(2026, 2, 6), D(2026, 2, 9)])

    def test_holiday_reply_for_stored_trading_day_is_an_error(self):
        conn = seed(SqliteConn(), mcap=D(2026, 2, 5))
        with self.assertRaisesRegex(norm.BulkResponseError, "holiday_response_for_stored_trading_day"):
            run(conn, FakeSource(holidays={"20260206"}), options(D(2026, 2, 6), run_short=False))

    def test_truncated_market_response_stops(self):
        conn = seed(SqliteConn(), codes=(A, B))
        raw = price_raw([(A, "STK", 70000, 70000, 70000, 70000, 0, 100, 10**12, 1000)])
        with self.assertRaisesRegex(norm.BulkResponseError, "market missing"):
            run(conn, FakeSource(overrides={("prices", "20260209"): raw}), options(D(2026, 2, 9), run_short=False))


class TestShortStep(unittest.TestCase):
    def test_short_unpublished_dash_stops(self):
        conn = seed(SqliteConn())
        dash = pd.DataFrame({"ISU_CD": [A], "BAL_QTY": ["-"], "BAL_AMT": ["-"]})
        with self.assertRaisesRegex(norm.BulkResponseError, "short_unpublished_rows"):
            run(conn, FakeSource(overrides={("short", "20260204", 1): dash}), options(D(2026, 2, 12)))
        self.assertEqual(conn.dates("ShortSellingDaily")[-1], D(2026, 2, 3))

    def test_short_missing_codes_stop(self):
        conn = seed(SqliteConn())
        empty_kosdaq = pd.DataFrame({"ISU_CD": ["999999"], "BAL_QTY": ["1"], "BAL_AMT": ["1"]})
        with self.assertRaisesRegex(norm.BulkResponseError, "short_missing_rows"):
            run(conn, FakeSource(overrides={("short", "20260204", 2): empty_kosdaq}), options(D(2026, 2, 9)))

    def test_short_dates_use_full_calendar_after_new_prices(self):
        conn = seed(SqliteConn())
        _, report = run(conn, FakeSource(), options(D(2026, 2, 12)))
        self.assertEqual([d["date"] for d in report["short"]], ["2026-02-04", "2026-02-05", "2026-02-06", "2026-02-09"])


class TestDerivedSteps(unittest.TestCase):
    def test_indicators_save_only_new_dates_within_run_end(self):
        conn = seed(SqliteConn(), extra_price_dates=[D(2026, 2, 9), D(2026, 2, 10)], ind=D(2026, 2, 6))
        index = pd.to_datetime(["2026-02-05", "2026-02-06", "2026-02-09", "2026-02-10"])
        prices = pd.DataFrame({c: [1.0] * 4 for c in ("open_price", "high_price", "low_price", "close_price")} | {"volume": [1] * 4}, index=index)
        plan = bulk.build_plan(conn, options(D(2026, 2, 9), run_indicators=True, run_short=False))
        ctx = bulk.RunContext(conn, bulk.NoExternalSource(), options(D(2026, 2, 9), run_indicators=True), plan)
        with patch("src.indicator_calculator.get_ohlcv_from_db", return_value=prices):
            result = bulk.run_indicator_step(ctx)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(conn.dates("CalculatedIndicators")[-1], D(2026, 2, 9))

    def test_tier_needs_short_source_rows_for_each_date(self):
        conn = seed(SqliteConn("stocks"), extra_price_dates=[D(2026, 2, 9)], mcap=D(2026, 2, 9), fin=D(2026, 2, 9), inv=D(2026, 2, 9),
                    ind=D(2026, 2, 9), tier_=D(2026, 2, 6))
        plan = bulk.build_plan(conn, options(D(2026, 2, 9), expected_database="stocks", write_floor=D(2026, 2, 9), run_tier=True, run_short=False))
        ctx = bulk.RunContext(conn, bulk.NoExternalSource(), options(D(2026, 2, 9), expected_database="stocks", run_tier=True), plan)
        with self.assertRaisesRegex(bulk.BulkStopError, "tier_short_inputs_missing"):
            bulk.run_tier_step(ctx)

    def test_tier_window_passes_lag_universe_history_and_guarded_writer(self):
        conn = seed(SqliteConn(), codes=(A, B, C), extra_price_dates=[D(2026, 2, 9)], tier_=D(2026, 2, 6), short=D(2026, 2, 6))
        conn.db.execute(f"DELETE FROM DailyStockPrice WHERE stock_code='{C}' AND date='2026-02-09'")  # C는 2/6 뒤 상장폐지
        conn.db.commit()
        opts = options(D(2026, 2, 9), run_tier=True, run_short=False, short_lag_trading_days=2)
        plan = bulk.build_plan(conn, opts)
        ctx = bulk.RunContext(conn, bulk.NoExternalSource(), opts, plan)
        captured = {}

        def fake_tier(conn_, **kwargs):
            captured.update(kwargs)
            return {"rows_saved": kwargs["row_writer"](conn_, tier_rows("2026-02-09", [A, B]))}

        with patch("src.pipeline.daily_stock_tier_batch.run_daily_stock_tier_batch", side_effect=fake_tier):
            bulk.run_tier_step(ctx)
        self.assertEqual(len(conn.rows("DailyStockTier", "date='2026-02-09'")), 2)
        self.assertEqual((captured["start_date_str"], captured["end_date_str"]), ("20260209", "20260209"))
        self.assertEqual(captured["short_selling_publication_lag_trading_days"], 2)
        self.assertEqual(captured["ticker_codes"], sorted([A, B]))
        self.assertEqual(captured["price_history_start_date"], D(2026, 1, 26))

    def test_tier_universe_keeps_codes_delisted_inside_window(self):
        conn = seed(SqliteConn(), codes=(A, B, C))
        conn.db.execute(f"DELETE FROM DailyStockPrice WHERE stock_code='{C}' AND date > '2026-02-02'")
        conn.db.commit()
        self.assertEqual(tier.tier_universe_codes(conn, D(2026, 2, 2), D(2026, 2, 6)), sorted([A, B, C]))

    def test_price_history_start_counts_observations(self):
        conn = seed(SqliteConn())
        self.assertEqual(tier.price_history_start(conn, D(2026, 2, 6), observations=3), D(2026, 2, 3))

    def test_compare_tier_frames_uses_float32_and_reports_keys(self):
        cols = list(guard.TABLE_SPECS["DailyStockTier"].columns)
        base = {c: None for c in cols} | {"date": "2026-02-06", "stock_code": A, "tier": 1, "reason": "r", "sbv_ratio": 0.1}
        stored = pd.DataFrame([base, base | {"stock_code": B}])
        same = pd.DataFrame([base | {"sbv_ratio": float(np.float32(0.1))}, base | {"stock_code": B}])
        self.assertTrue(tier.compare_tier_frames(same, stored)["identical"])
        changed = pd.DataFrame([base | {"tier": 2}])
        result = tier.compare_tier_frames(changed, stored)
        self.assertFalse(result["identical"])
        self.assertEqual((result["mismatch"]["tier"]["count"], result["only_stored"]), (1, 1))


# ---- 대조 -------------------------------------------------------------------

class TestVerify(unittest.TestCase):
    def test_compare_is_exact_for_integers_and_stored_precision_for_others(self):
        stored = pd.DataFrame({"stock_code": [A, B], "m": [10**12, 5], "p": [1.000004, 2.0], "f": [float(np.float32(1.4)), 2.0]})
        fetched = pd.DataFrame({"stock_code": [A, B], "m": [10**12 + 10**6, 5], "p": [1.000004, 2.000011], "f": [1.4, 2.0]})
        out = bulk.compare_with_db(fetched, stored, {"m": "int", "p": "dec5", "f": "float32"})
        self.assertEqual(out["mismatch"]["m"], {"count": 1, "sample": [A]})   # 1조 + 100만 원 차이도 잡는다
        self.assertEqual(out["mismatch"]["p"], {"count": 1, "sample": [B]})
        self.assertEqual(out["mismatch"]["f"]["count"], 0)
        self.assertFalse(out["identical"])

    def test_verify_day_identical_and_mismatch(self):
        conn = seed(SqliteConn())
        result = bulk.verify_day(conn, FakeSource(), D(2026, 2, 2))
        self.assertTrue(result["identical"], json.dumps(result, default=str)[:2000])
        self.assertEqual(result["quarantined_count"], 0)
        changed = FakeSource(overrides={("fundamentals", "20260202"): pd.DataFrame(
            [(A, "50,000", "10.6", "1.4", "5,000", "2.0", "1,400"), (B, "50,000", "10.5", "1.4", "5,000", "2.0", "1,400")],
            columns=["ISU_SRT_CD", "BPS", "PER", "PBR", "EPS", "DVD_YLD", "DPS"])})
        result = bulk.verify_day(conn, changed, D(2026, 2, 2))
        self.assertFalse(result["identical"])
        self.assertEqual(result["tables"]["FinancialData"]["mismatch"]["per"]["count"], 1)

    def test_verify_requires_stored_rows_including_snapshot(self):
        conn = seed(SqliteConn())
        with self.assertRaisesRegex(bulk.BulkStopError, "verify_day_without_stored_rows"):
            bulk.verify_day(conn, FakeSource(), D(2026, 2, 3))


# ---- CLI --------------------------------------------------------------------

class FakeSession:
    def __init__(self, *args, **kwargs):
        self.logged_in = False
        self.closed = False

    def login(self):
        self.logged_in = True

    @contextlib.contextmanager
    def installed(self, *_):
        yield self

    def stats(self):
        return {"krx_calls": 0, "logins": int(self.logged_in)}

    def close(self):
        self.closed = True


class TestCli(unittest.TestCase):
    def _main(self, conn, argv, source=None, session=None):
        fake_db = types.ModuleType("src.db_setup")
        fake_db.get_db_connection = lambda: conn
        session = session or FakeSession()
        sink = io.StringIO()
        with patch.dict(sys.modules, {"src.db_setup": fake_db}), \
             patch.object(bulk, "_open_session", return_value=session) as opener, \
             patch.object(bulk, "PykrxDailyBulkSource", return_value=source or FakeSource()), \
             patch.object(bulk, "default_collector_modules", create=True), \
             patch("src.common.krx_session.default_collector_modules", return_value=()), \
             contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            code = bulk.main(argv)
        return code, opener, session, sink.getvalue()

    def test_dry_run_never_opens_session_or_writes(self):
        conn = seed(SqliteConn())
        code, opener, _, _ = self._main(conn, ["--end-date", "20260213", "--expect-database", "stocks_stage", "--dry-run"])
        self.assertEqual(code, 0)
        opener.assert_not_called()
        self.assertFalse(any(s.upper().startswith("INSERT") for s in conn.statements))

    def test_report_dir_must_be_new_and_is_never_overwritten(self):
        conn = seed(SqliteConn())
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "old.json").write_text("{}")
            code, opener, _, out = self._main(conn, ["--end-date", "20260209", "--expect-database", "stocks_stage", "--report-dir", tmp])
            self.assertEqual(code, 3)
            self.assertIn("report_dir_not_empty", out)
            self.assertEqual(Path(tmp, "old.json").read_text(), "{}")
        opener.assert_not_called()

    def test_origin_without_floor_is_refused_before_login(self):
        conn = seed(SqliteConn("stocks"))
        with tempfile.TemporaryDirectory() as tmp:
            code, opener, _, out = self._main(conn, ["--end-date", "20260209", "--expect-database", "stocks", "--report-dir", str(Path(tmp) / "r")])
        self.assertEqual(code, 3)
        self.assertIn("origin_write_requires_write_floor", out)
        opener.assert_not_called()

    def test_run_records_watermarks_from_db_and_no_login_when_idle(self):
        conn = seed(SqliteConn(), short=D(2026, 2, 3))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "r"
            code, opener, _, _ = self._main(conn, ["--end-date", "20260206", "--expect-database", "stocks_stage", "--skip-indicators", "--report-dir", str(target)])
            report = json.loads((target / "report.json").read_text())
        self.assertEqual(code, 0)
        opener.assert_not_called()
        self.assertEqual(report["watermarks_after_db"]["DailyStockPrice"], "2026-02-06")

    def test_run_with_work_logs_in_once_and_reports(self):
        conn = seed(SqliteConn())
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "r"
            code, opener, session, _ = self._main(conn, ["--end-date", "20260209", "--expect-database", "stocks_stage", "--skip-indicators", "--skip-short", "--report-dir", str(target)])
            report = json.loads((target / "report.json").read_text())
        self.assertEqual(code, 0)
        self.assertTrue(session.logged_in and session.closed)
        self.assertEqual(report["watermarks_after_db"]["DailyStockPrice"], "2026-02-09")

    def test_verify_mismatch_exits_nonzero_and_budget_is_fixed(self):
        conn = seed(SqliteConn())
        changed = FakeSource(overrides={("short", "20260202", 1): pd.DataFrame({"ISU_CD": [A], "BAL_QTY": ["1"], "BAL_AMT": ["999"]})})
        with tempfile.TemporaryDirectory() as tmp:
            code, _, _, _ = self._main(conn, ["--expect-database", "stocks_stage", "--verify-date", "--report-dir", str(Path(tmp) / "v")], source=changed)
            self.assertEqual(code, 4)
            code, opener, _, out = self._main(conn, ["--expect-database", "stocks_stage", "--verify-date", "--max-krx-calls", "20", "--report-dir", str(Path(tmp) / "w")])
        self.assertEqual(code, 3)
        self.assertIn("verify_budget_is_fixed_at_15", out)
        self.assertEqual(bulk.VERIFY_KRX_BUDGET, 15)

    def test_unexpected_error_message_is_classified_only(self):
        conn = seed(SqliteConn())
        source = FakeSource(overrides={("prices", "20260209"): RuntimeError("secret-text-from-server")})
        with tempfile.TemporaryDirectory() as tmp:
            code, _, _, out = self._main(conn, ["--end-date", "20260209", "--expect-database", "stocks_stage", "--skip-short", "--skip-indicators", "--report-dir", str(Path(tmp) / "r")], source=source)
        self.assertEqual(code, 1)
        self.assertIn("unexpected:RuntimeError", out)
        self.assertNotIn("secret-text-from-server", out)



# ---- v3 반례 회귀(review-astra-fixes.md) ------------------------------------

class TestV3ModeAndPrechecks(unittest.TestCase):
    _main = TestCli._main

    def test_dry_run_with_verify_is_a_usage_error_with_zero_transmission(self):
        conn = seed(SqliteConn())
        source = FakeSource()
        code, opener, session, out = self._main(conn, ["--expect-database", "stocks_stage", "--dry-run", "--verify-date"], source=source)
        self.assertEqual(code, 3)
        self.assertIn("dry_run_and_verify_are_exclusive", out)
        opener.assert_not_called()
        self.assertFalse(session.logged_in)
        self.assertEqual(source.calls, [])

    def test_verify_checks_stored_rows_before_login(self):
        conn = seed(SqliteConn())
        source = FakeSource()
        with tempfile.TemporaryDirectory() as tmp:
            code, opener, _, out = self._main(conn, ["--expect-database", "stocks_stage", "--verify-date", "20260203", "--report-dir", str(Path(tmp) / "v")], source=source)
        self.assertEqual(code, 3)
        self.assertIn("verify_day_without_stored_rows", out)
        opener.assert_not_called()
        self.assertEqual(source.calls, [])

    def test_verify_budget_and_database_are_checked_before_login(self):
        conn = seed(SqliteConn())
        with tempfile.TemporaryDirectory() as tmp:
            code, opener, _, _ = self._main(conn, ["--expect-database", "other", "--verify-date", "--report-dir", str(Path(tmp) / "v")])
        self.assertEqual(code, 3)
        opener.assert_not_called()

    def test_max_block_events_cli_reaches_policy(self):
        args = bulk.build_arg_parser().parse_args(["--expect-database", "x", "--end-date", "20260209"])
        self.assertEqual(bulk.policy_from_args(args).max_block_events, 1)
        args = bulk.build_arg_parser().parse_args(["--expect-database", "x", "--end-date", "20260209", "--max-block-events", "2"])
        self.assertEqual(bulk.policy_from_args(args).max_block_events, 2)

    def test_origin_floor_earlier_than_feb_9_is_refused(self):
        with self.assertRaisesRegex(guard.NewKeyViolation, "origin_write_floor_before_2026-02-09"):
            guard.assert_target_database(SqliteConn("stocks"), "stocks", D(2026, 2, 4), True)
        guard.assert_target_database(SqliteConn("stocks_stage"), "stocks_stage", D(2026, 2, 4), True)


class TestV3TierAudit(unittest.TestCase):
    def _ctx(self, conn, **kwargs):
        opts = options(D(2026, 2, 10), run_tier=True, run_short=False, short_lag_trading_days=2, **kwargs)
        plan = bulk.build_plan(conn, opts)
        return bulk.RunContext(conn, bulk.NoExternalSource(), opts, plan)

    def _conn(self):
        return seed(SqliteConn(), extra_price_dates=[D(2026, 2, 9), D(2026, 2, 10)], tier_=D(2026, 2, 6), short=D(2026, 2, 6))

    def _run(self, ctx, produced):
        def fake_tier(conn_, **kwargs):
            frames = [tier_rows(day, codes) for day, codes in produced.items()]
            return {"rows_saved": kwargs["row_writer"](conn_, pd.concat(frames, ignore_index=True))}
        with patch("src.pipeline.daily_stock_tier_batch.run_daily_stock_tier_batch", side_effect=fake_tier):
            return bulk.run_tier_step(ctx)

    def test_missing_output_code_stops_that_date_and_keeps_earlier_dates(self):
        conn = self._conn()
        with self.assertRaisesRegex(bulk.BulkStopError, "tier_audit_failed:2026-02-10:\\['missing_from_output'\\]"):
            self._run(self._ctx(conn), {"2026-02-09": [A, B], "2026-02-10": [A]})
        self.assertEqual(len(conn.rows("DailyStockTier", "date='2026-02-09'")), 2)
        self.assertEqual(conn.rows("DailyStockTier", "date='2026-02-10'"), [])

    def test_date_with_no_output_at_all_is_audited(self):
        conn = self._conn()
        with self.assertRaisesRegex(bulk.BulkStopError, "tier_audit_failed:2026-02-10"):
            self._run(self._ctx(conn), {"2026-02-09": [A, B]})

    def test_quarantined_roster_code_without_price_stops_unless_allowed(self):
        conn = self._conn()
        conn.insert("TickerUniverseSnapshot", [(D(2026, 2, 9), c, "KOSPI", None, "pykrx") for c in (A, B, C)])
        with self.assertRaisesRegex(bulk.BulkStopError, "roster_without_price"):
            self._run(self._ctx(conn), {"2026-02-09": [A, B], "2026-02-10": [A, B]})
        self.assertEqual(conn.rows("DailyStockTier", "date='2026-02-09'"), [])
        conn2 = self._conn()
        conn2.insert("TickerUniverseSnapshot", [(D(2026, 2, 9), c, "KOSPI", None, "pykrx") for c in (A, B, C)])
        self._run(self._ctx(conn2, tier_allow_quarantined=True), {"2026-02-09": [A, B], "2026-02-10": [A, B]})
        self.assertEqual(len(conn2.rows("DailyStockTier", "date='2026-02-10'")), 2)

    def test_short_source_with_only_one_valid_row_stops(self):
        conn = self._conn()
        conn.db.execute(f"UPDATE ShortSellingDaily SET short_balance_value=NULL WHERE date='2026-02-05' AND stock_code='{B}'")
        conn.db.commit()
        with self.assertRaisesRegex(bulk.BulkStopError, "short_missing"):
            self._run(self._ctx(conn), {"2026-02-09": [A, B], "2026-02-10": [A, B]})
        self.assertEqual(conn.rows("DailyStockTier", "date='2026-02-09'"), [])


class TestV3Comparator(unittest.TestCase):
    def test_null_pairs_match_and_big_integers_are_exact(self):
        stored = pd.DataFrame({"stock_code": [A, B, C], "v": [None, 2**53, 7], "f": [None, 1.0, 2.0]})
        fetched = pd.DataFrame({"stock_code": [A, B, C], "v": pd.array([None, 2**53 + 1, 7], dtype="Int64"), "f": [np.nan, 1.0, 2.0]})
        out = bulk.compare_with_db(fetched, stored, {"v": "int", "f": "float32"})
        self.assertEqual(out["mismatch"]["v"], {"count": 1, "sample": [B]})
        self.assertEqual(out["mismatch"]["f"]["count"], 0)
        same = bulk.compare_with_db(stored.copy(), stored, {"v": "int", "f": "float32"})
        self.assertTrue(same["identical"])

    def test_float_columns_are_read_as_double(self):
        conn = seed(SqliteConn())
        frame = bulk._db_frame(conn, "FinancialData", D(2026, 2, 2), ["roe", "per"], None, {"roe": "float32", "per": "float32"})
        self.assertTrue(any("CAST(`roe` AS DOUBLE)" in s for s in conn.statements))
        self.assertEqual(frame["roe"].dtype, object)

    def test_quarantined_adjusted_rows_block_overall_pass(self):
        conn = seed(SqliteConn(), codes=(A, B, C))
        result = bulk.verify_day(conn, FakeSource(codes=(A, B, C)), D(2026, 2, 2))
        self.assertFalse(result["identical"])
        self.assertIn("adjusted_quarantined", result["unverified"])
        self.assertEqual(result["adjusted_check"], {"total": 3, "checked": 2, "quarantined": 1, "reasons": {"prior_ratio_not_one": 1}})


class TestV3March14Reproduction(unittest.TestCase):
    def test_windows_match_march14_chunking(self):
        self.assertEqual(tier.march14_windows(D(2026, 1, 5), D(2026, 2, 6)), [(D(2025, 4, 16), D(2026, 2, 6))])
        self.assertEqual(tier.march14_windows(D(2017, 4, 18), D(2017, 4, 20))[0][0], D(2017, 4, 18))

    def test_query_starts_follow_default_batch_arguments(self):
        starts = tier.march14_query_starts(D(2025, 4, 16))
        self.assertEqual(starts["price_short_investor_mcap"], "2021-10-29")
        self.assertEqual(starts["financial"], "2014-09-24")  # 2021-10-29 - (7*366+30)일

    def test_stored_tier_floats_are_read_as_double_and_nulls_compare_equal(self):
        conn = seed(SqliteConn())
        stored = tier._db_tier_frame(conn, D(2026, 2, 6), D(2026, 2, 6), with_computed_at=False)
        self.assertTrue(any("CAST(`sbv_ratio` AS DOUBLE)" in s for s in conn.statements))
        self.assertTrue(tier.compare_tier_frames(stored.copy(), stored)["identical"])
        self.assertEqual(len(tier._mismatch_rows(stored.copy(), stored.assign(computed_at=None))), 0)


class TestV3StepLimits(unittest.TestCase):
    def test_max_dates_applies_to_every_step(self):
        # 반례: 일반 표 2/6, 공매도·지표·Tier 1/26 → --max-dates 1이어도 6·9·9일을 잡던 문제.
        conn = seed(SqliteConn(), short=D(2026, 1, 26), ind=D(2026, 1, 26), tier_=D(2026, 1, 26))
        opts = options(D(2026, 2, 6), max_dates=1, run_indicators=True, run_tier=True)
        plan = bulk.build_plan(conn, opts)
        self.assertEqual(plan.date_plans, [])
        self.assertEqual((len(plan.short_source_dates), len(plan.indicator_dates), len(plan.tier_dates)), (1, 1, 1))
        source = FakeSource()
        bulk.run_daily_bulk(conn, source, replace_opts(opts, run_indicators=False, run_tier=False), plan)
        self.assertEqual([c for c in source.calls if c[0] == "short"], [("short", "20260127", 1), ("short", "20260127", 2)])


def replace_opts(opts, **kwargs):
    from dataclasses import replace
    return replace(opts, **kwargs)


class TestV3ResponseCompleteness(unittest.TestCase):
    def test_code_missing_from_all_investor_keys_is_counted_and_stops(self):
        conn = seed(SqliteConn())
        empty_kosdaq = pd.DataFrame({"ISU_SRT_CD": ["999999"], "NETBID_TRDVAL": ["1"]})
        overrides = {("investor", "20260209", "KSQ", code): empty_kosdaq for code in (7050, 8000, 9000, 9001)}
        with self.assertRaisesRegex(norm.BulkResponseError, "investor_unobserved_rows"):
            run(conn, FakeSource(overrides=overrides), options(D(2026, 2, 9), run_short=False))
        self.assertEqual(conn.rows("DailyStockPrice", "date='2026-02-09'"), [])

    def test_damaged_investor_number_stops_immediately(self):
        conn = seed(SqliteConn())
        source = FakeSource(overrides={("investor", "20260209", "STK", 7050): pd.DataFrame({"ISU_SRT_CD": [A], "NETBID_TRDVAL": ["12a"]})})
        with self.assertRaisesRegex(norm.BulkResponseError, "unparsable"):
            run(conn, source, options(D(2026, 2, 9), run_short=False))
        self.assertEqual(len([c for c in source.calls if c[0] == "investor"]), 1)

    def test_all_dash_prices_are_not_a_holiday(self):
        raw = price_raw([(A, "STK", "-", "-", "-", "-", "-", "-", "1", "1"), (B, "KSQ", "-", "-", "-", "-", "-", "-", "1", "1")])
        with self.assertRaises(norm.BulkResponseError):
            norm.normalize_market_day(raw, "20260209")

    def test_dash_change_is_not_zero_and_is_quarantined(self):
        raw = price_raw([(A, "STK", 70000, 70000, 70000, 70000, "-", 100, 10**12, 1000), (B, "KSQ", 1000, 1000, 1000, 1000, 0, 100, 10**12, 1000)])
        snap = norm.normalize_market_day(raw, "20260209")
        self.assertTrue(np.isnan(snap.prices.set_index("stock_code").loc[A, "change_vs_prev"]))
        prior = {c: norm.PriorPriceState(D(2026, 2, 6), float(CLOSE[c]), float(CLOSE[c]), 1.0) for c in (A, B)}
        result = norm.apply_adjusted_carry_forward(snap.prices, prior, D(2026, 2, 6))
        self.assertEqual(result.quarantined, {A: "unparsable_change"})


class TestV3QuarantineDurability(unittest.TestCase):
    def test_quarantine_file_failure_prevents_commit(self):
        conn = seed(SqliteConn(), codes=(A, B, C))
        with tempfile.TemporaryDirectory() as tmp:
            recorder = bulk.RunRecorder(str(Path(tmp) / "r"))
            with patch.object(bulk.RunRecorder, "write_csv", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    run(conn, FakeSource(codes=(A, B, C)), options(D(2026, 2, 9), run_short=False), recorder)
        self.assertEqual(conn.dates("DailyStockPrice")[-1], D(2026, 2, 6))
        self.assertEqual(bulk.build_plan(conn, options(D(2026, 2, 9), run_short=False)).date_plans[0].day, D(2026, 2, 9))

    def test_stop_policy_saves_quarantine_before_stopping(self):
        conn = seed(SqliteConn(), codes=(A, B, C))
        with tempfile.TemporaryDirectory() as tmp:
            recorder = bulk.RunRecorder(str(Path(tmp) / "r"))
            with self.assertRaises(bulk.BulkStopError):
                run(conn, FakeSource(codes=(A, B, C)), options(D(2026, 2, 9), run_short=False, adjust_event_policy="stop"), recorder)
            saved = pd.read_csv(recorder.path / "quarantine-2026-02-09.csv", dtype={"stock_code": str})
        self.assertEqual(list(saved["stock_code"]), [C])
        self.assertEqual(list(saved["reason"]), ["prior_ratio_not_one"])

    def test_pending_then_done_and_recovery_of_committed_without_done(self):
        conn = seed(SqliteConn())
        with tempfile.TemporaryDirectory() as tmp:
            recorder = bulk.RunRecorder(str(Path(tmp) / "r"))
            run(conn, FakeSource(), options(D(2026, 2, 9), run_short=False), recorder)
            names = sorted(p.name for p in recorder.path.iterdir())
            self.assertIn("day-2026-02-09.pending.json", names)
            self.assertIn("day-2026-02-09.json", names)
            (recorder.path / "day-2026-02-09.json").unlink()   # 커밋 뒤 완료 표시 전에 멈춘 상황
            found = bulk.reconcile_previous_report(conn, str(recorder.path))
        self.assertEqual(found["pending_without_done"][0]["state"], "committed")

    def test_recovery_reports_not_committed_pending(self):
        conn = seed(SqliteConn())
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "day-2026-02-09.pending.json").write_text(json.dumps({"date": "2026-02-09", "tables": ["DailyStockPrice"]}))
            found = bulk.reconcile_previous_report(conn, tmp)
        self.assertEqual(found["pending_without_done"][0]["state"], "not_committed")


if __name__ == "__main__":
    unittest.main()
