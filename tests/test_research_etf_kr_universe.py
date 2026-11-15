import datetime as dt
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.research.etf.kr_universe import (  # noqa: E402
    CANARY_DATE,
    CANARY_DELISTED_CODE,
    EMPTY_RESPONSE_RETRIES,
    build_history,
    collect,
    load_snapshots,
    month_end_dates,
)


def _quotes(codes):
    return pd.DataFrame({
        "ISU_SRT_CD": [c for c, _ in codes],
        "ISU_ABBRV": [n for _, n in codes],
        "TDD_CLSPRC": ["10,000"] * len(codes),
        "ACC_TRDVAL": ["1,234"] * len(codes),
        "MKTCAP": ["5,000,000,000"] * len(codes),
        "IDX_IND_NM": ["지수"] * len(codes),
    })


class FakeKrx:
    """KRX처럼 휴장일에도 목록을 준다. 2024-02에만 있다가 사라지는 B와, 이름이 바뀌는 A.

    empty_times: 날짜별로 앞에서 몇 번 빈 응답을 줄지(일시 실패 흉내).
    """

    def __init__(self, empty_times=None):
        self.calls = []
        self.empty_times = dict(empty_times or {})

    def __call__(self, day):
        self.calls.append(day)
        if self.empty_times.get(day, 0) > 0:
            self.empty_times[day] -= 1
            return pd.DataFrame()
        if day == CANARY_DATE:
            return _quotes([(CANARY_DELISTED_CODE, "폐지예")])
        d = dt.datetime.strptime(day, "%Y%m%d").date()
        if d.month == 1:
            return _quotes([("A", "옛이름")])
        if d.month == 2:
            return _quotes([("A", "옛이름"), ("B", "곧폐지")])
        return _quotes([("A", "새이름"), ("C", "3월상장")][: 1 if d.day < 31 else 2])


class TestKrUniverse(unittest.TestCase):
    def test_month_end_dates_crosses_year(self):
        self.assertEqual(month_end_dates("2023-12", "2024-02"),
                         [dt.date(2023, 12, 31), dt.date(2024, 1, 31), dt.date(2024, 2, 29)])

    def test_collect_uses_exact_month_end_and_caches(self):
        fake = FakeKrx()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            m = collect("2024-01", "2024-03", out, delay=0, fetch=fake, today=dt.date(2024, 4, 10))
            files = sorted(p.name for p in (out / "snapshots").glob("*.csv"))
            # 2024-03-31은 일요일이지만 KRX는 그날 기준 목록을 주므로 그 날짜로 저장한다.
            self.assertEqual(files, ["20240131.csv", "20240229.csv", "20240331.csv"])
            self.assertTrue(all(r["status"] == "downloaded" for r in m["snapshots"]))

            n_calls = len(fake.calls)
            m2 = collect("2024-01", "2024-03", out, delay=0, fetch=fake, today=dt.date(2024, 4, 10))
            self.assertEqual(len(fake.calls), n_calls + 1)  # 확인용 조회 1회만
            self.assertTrue(all(r["status"] == "cached" for r in m2["snapshots"]))

            hist = build_history(load_snapshots(out / "snapshots")).set_index("code")
            self.assertFalse(bool(hist.loc["A", "delisted"]))
            self.assertTrue(bool(hist.loc["A", "name_changed"]))
            self.assertEqual(hist.loc["A", "last_name"], "새이름")
            self.assertTrue(bool(hist.loc["B", "delisted"]))
            self.assertEqual(hist.loc["B", "first_seen"], "2024-02-29")
            self.assertEqual(hist.loc["B", "last_seen"], "2024-02-29")
            self.assertEqual(hist.loc["B", "last_mktcap_krw"], 5_000_000_000)
            self.assertEqual(hist.loc["C", "first_seen"], "2024-03-31")

    def test_incomplete_month_is_not_cached(self):
        fake = FakeKrx()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            m = collect("2024-02", "2024-03", out, delay=0, fetch=fake, today=dt.date(2024, 3, 25))
            self.assertEqual([r["status"] for r in m["snapshots"]], ["downloaded", "month_not_complete"])
            self.assertFalse((out / "snapshots" / "20240331.csv").exists())
            # 월말 뒤 재실행하면 3월 말 목록(C 포함)을 받는다.
            collect("2024-02", "2024-03", out, delay=0, fetch=fake, today=dt.date(2024, 4, 1))
            snap = pd.read_csv(out / "snapshots" / "20240331.csv", dtype=str, encoding="utf-8-sig")
            self.assertEqual(sorted(snap["ISU_SRT_CD"]), ["A", "C"])

    def test_empty_response_retries_same_date_then_fails_without_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            flaky = FakeKrx(empty_times={"20240131": EMPTY_RESPONSE_RETRIES - 1})
            m = collect("2024-01", "2024-01", out, delay=0, fetch=flaky, today=dt.date(2024, 4, 1))
            self.assertEqual(m["snapshots"][0]["status"], "downloaded")
            self.assertTrue(all(c in ("20240131", CANARY_DATE) for c in flaky.calls))

            dead = FakeKrx(empty_times={"20240229": EMPTY_RESPONSE_RETRIES})
            m = collect("2024-02", "2024-02", out, delay=0, fetch=dead, today=dt.date(2024, 4, 1))
            self.assertEqual(m["snapshots"][0]["status"], "empty_response")
            self.assertFalse((out / "snapshots" / "20240229.csv").exists())

    def test_truncated_cached_snapshot_is_refetched(self):
        fake = FakeKrx()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            today = dt.date(2024, 4, 1)
            collect("2024-02", "2024-02", out, delay=0, fetch=fake, today=today)
            path = out / "snapshots" / "20240229.csv"
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8-sig")  # 헤더 + 1행만 남김
            (out / "snapshots" / "20240229.csv.tmp").write_text("partial", encoding="utf-8")

            m = collect("2024-02", "2024-02", out, delay=0, fetch=fake, today=today)
            self.assertEqual(m["snapshots"][0]["status"], "downloaded")
            snap = load_snapshots(out / "snapshots")  # .tmp는 읽지 않는다
            self.assertEqual(sorted(snap["ISU_SRT_CD"]), ["A", "B"])

    def test_canary_stops_when_delisted_etf_missing(self):
        def current_only(day):
            return _quotes([("069500", "KODEX 200")])

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                collect("2024-01", "2024-01", Path(tmp), delay=0, fetch=current_only)

    def test_build_history_counts_gaps(self):
        snaps = pd.concat([
            _quotes([("A", "a")]).assign(trade_date="2024-01-31"),
            _quotes([("B", "b")]).assign(trade_date="2024-02-29"),
            _quotes([("A", "a"), ("B", "b")]).assign(trade_date="2024-03-28"),
        ])
        hist = build_history(snaps).set_index("code")
        self.assertEqual(int(hist.loc["A", "gap_snapshots"]), 1)
        self.assertFalse(bool(hist.loc["A", "delisted"]))


if __name__ == "__main__":
    unittest.main()
