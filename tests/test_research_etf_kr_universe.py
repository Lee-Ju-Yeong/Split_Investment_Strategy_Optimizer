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
    """주말·휴일은 빈 표. 2024-02에만 있다가 사라지는 B와, 이름이 바뀌는 A."""

    def __init__(self):
        self.calls = []

    def __call__(self, day):
        self.calls.append(day)
        if day == CANARY_DATE:
            return _quotes([(CANARY_DELISTED_CODE, "폐지예")])
        d = dt.datetime.strptime(day, "%Y%m%d").date()
        if d.weekday() >= 5 or d == dt.date(2024, 3, 29):
            return pd.DataFrame()
        if d.month == 1:
            return _quotes([("A", "옛이름")])
        if d.month == 2:
            return _quotes([("A", "옛이름"), ("B", "곧폐지")])
        return _quotes([("A", "새이름")])


class TestKrUniverse(unittest.TestCase):
    def test_month_end_dates_crosses_year(self):
        self.assertEqual(month_end_dates("2023-12", "2024-02"),
                         [dt.date(2023, 12, 31), dt.date(2024, 1, 31), dt.date(2024, 2, 29)])

    def test_collect_steps_back_over_holidays_and_caches(self):
        fake = FakeKrx()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            m = collect("2024-01", "2024-03", out, delay=0, fetch=fake)
            files = sorted(p.name for p in (out / "snapshots").glob("*.csv"))
            # 2024-03-31(일)·30(토)·29(가상 휴일) → 28일
            self.assertEqual(files, ["20240131.csv", "20240229.csv", "20240328.csv"])
            self.assertTrue(all(r["status"] == "downloaded" for r in m["snapshots"]))

            n_calls = len(fake.calls)
            m2 = collect("2024-01", "2024-03", out, delay=0, fetch=fake)
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
