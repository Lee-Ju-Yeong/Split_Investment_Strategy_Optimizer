"""연구용 ETF 가격 저장소(M1) 정규화·품질 표시 검사. 네트워크·실제 자료 없이 작은 합성 파일로 돈다."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.research.etf import price_store as ps

KR_ROWS = [
    "date,open,high,low,close,volume",
    "2024-01-02,100,110,95,105,1000",
    "2024-01-03,105,104,100,106,1200",   # 종가 > 고가: 표시만, 고치지 않음
    "2024-01-04,0,0,0,106,0",            # 거래 없는 날: 0 시가 표시
    "2024-01-05,106,150,106,150,900",    # +41.5% 급등: 표시
]
KR_REF = [
    "date,open,high,low,close,volume",
    "2024-01-02,1,1,1,1,1", "2024-01-03,1,1,1,1,1", "2024-01-04,1,1,1,1,1", "2024-01-05,1,1,1,1,1",
]
US_ROWS = [
    "date,open,high,low,close,adj_close,volume,adj_factor,adj_open,adj_high,adj_low,dividend,split",
    "2024-01-02,10,11,9,10.5,10.4,100,0.99,9.9,10.9,8.9,0,",
    "2024-01-03,10.5,11,10,10.8,10.8,120,1,10.5,11,10,0.25,",
    "2024-01-05,10.8,11.2,10.7,11,11,90,1,10.8,11.2,10.7,0,2:1",
]
US_REF = [
    "date,open,high,low,close,adj_close,volume,adj_factor,adj_open,adj_high,adj_low,dividend,split",
    "2024-01-02,1,1,1,1,1,1,1,1,1,1,0,", "2024-01-03,1,1,1,1,1,1,1,1,1,1,0,",
    "2024-01-04,1,1,1,1,1,1,1,1,1,1,0,", "2024-01-05,1,1,1,1,1,1,1,1,1,1,0,",
]


class PriceStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.raw, self.out = root / "raw", root / "out"
        for subdir, files in ((ps.KR_SUBDIR, {"AAA111": KR_ROWS, "069500": KR_REF}),
                              (ps.US_SUBDIR, {"XYZ": US_ROWS, "SPY": US_REF})):
            folder = self.raw / subdir
            folder.mkdir(parents=True)
            for symbol, rows in files.items():
                (folder / f"{symbol}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.out.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_kr_rows_kept_and_flagged_not_repaired(self):
        frame = ps.load_symbol("KR", self.raw / ps.KR_SUBDIR / "AAA111.csv", "AAA111")
        self.assertEqual(len(frame), 4)  # 행을 지우지 않는다
        self.assertEqual(frame["ohlc_ok"].tolist(), [True, False, False, True])
        self.assertEqual(frame.loc[1, "close"], 106.0)  # 값을 고치지 않는다
        self.assertTrue((frame["dividend"] == 0).all())
        self.assertEqual(frame.loc[0, "turnover"], 105 * 1000)
        self.assertEqual(frame.loc[0, "currency"], "KRW")

    def test_us_dividend_and_split_columns(self):
        frame = ps.load_symbol("US", self.raw / ps.US_SUBDIR / "XYZ.csv", "XYZ")
        self.assertEqual(frame["dividend"].tolist(), [0.0, 0.25, 0.0])
        self.assertEqual(frame["split_ratio"].tolist(), [1.0, 1.0, 2.0])
        # 분할 반영 시세(OHLC)를 쓰고 adj_close로 분배금을 다시 더하지 않는다.
        self.assertEqual(frame["close"].tolist(), [10.5, 10.8, 11.0])

    def test_quality_counts_and_reference_gap(self):
        reference = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))
        us = ps.load_symbol("US", self.raw / ps.US_SUBDIR / "XYZ.csv", "XYZ")
        quality = ps.symbol_quality(us, reference)
        self.assertEqual(quality["missing_reference_days_in_span"], 1)  # 01-04 없음
        self.assertEqual(quality["dividend_events"], 1)
        self.assertEqual(quality["split_events"], 1)
        kr = ps.symbol_quality(ps.load_symbol("KR", self.raw / ps.KR_SUBDIR / "AAA111.csv", "AAA111"), reference)
        self.assertEqual(kr["bad_ohlc_rows"], 2)
        self.assertEqual(kr["zero_volume_days"], 1)
        self.assertEqual(kr["large_return_days"], 1)

    def test_build_writes_deterministic_outputs(self):
        first = ps.build(self.raw, self.out)
        digest = {m["market"]: m["daily_sha256"] for m in first["markets"]}
        second = ps.build(self.raw, self.out)
        self.assertEqual(digest, {m["market"]: m["daily_sha256"] for m in second["markets"]})
        with gzip.open(self.out / "kr_daily.csv.gz", "rt", encoding="utf-8") as handle:
            daily = pd.read_csv(handle, dtype={"symbol": str})
        self.assertEqual(list(daily.columns), ps.COLUMNS)
        self.assertEqual(sorted(daily["symbol"].unique()), ["069500", "AAA111"])  # 코드 앞자리 0 보존
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([m["market"] for m in manifest["markets"]], ["KR", "US"])
        inventory = pd.read_csv(self.out / "us_inventory.csv")
        self.assertEqual(set(inventory["symbol"]), {"SPY", "XYZ"})

    def test_symbol_filter_and_missing_source(self):
        summary = ps.build(self.raw, self.out, markets=("US",), symbols=["XYZ"])
        self.assertEqual(summary["markets"][0]["symbols"], 1)
        with self.assertRaises(FileNotFoundError):
            ps.build(self.raw, self.out, markets=("KR",), symbols=["NOPE"])

    def test_paths_require_absolute_env(self):
        from src.research.etf import paths
        import os
        old = os.environ.pop(paths.ENV_DATA_ROOT, None)
        try:
            with self.assertRaises(RuntimeError):
                paths.data_root()
            os.environ[paths.ENV_DATA_ROOT] = "relative/dir"
            with self.assertRaises(RuntimeError):
                paths.data_root()
        finally:
            os.environ.pop(paths.ENV_DATA_ROOT, None)
            if old is not None:
                os.environ[paths.ENV_DATA_ROOT] = old


if __name__ == "__main__":
    unittest.main()
