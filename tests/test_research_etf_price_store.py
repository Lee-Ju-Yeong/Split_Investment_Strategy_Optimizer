"""연구용 ETF 가격 저장소(M1) 정규화·품질 표시 검사. 네트워크·실제 자료 없이 작은 합성 파일로 돈다."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
        self.assertTrue(second["promoted"])
        bundle = ps.current_bundle(self.out)
        self.assertEqual(str(bundle), second["bundle_dir"])
        with gzip.open(bundle / "kr_daily.csv.gz", "rt", encoding="utf-8") as handle:
            daily = pd.read_csv(handle, dtype={"symbol": str})
        self.assertEqual(list(daily.columns), ps.COLUMNS)
        self.assertEqual(sorted(daily["symbol"].unique()), ["069500", "AAA111"])  # 코드 앞자리 0 보존
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([m["market"] for m in manifest["markets"]], ["KR", "US"])
        self.assertEqual(manifest["run_id"], bundle.name)
        self.assertTrue(manifest["scope"]["complete"])
        inventory = pd.read_csv(bundle / "us_inventory.csv")
        self.assertEqual(set(inventory["symbol"]), {"SPY", "XYZ"})

    def test_symbol_filter_and_missing_source(self):
        summary = ps.build(self.raw, self.out, markets=("US",), symbols=["XYZ"])
        self.assertEqual(summary["markets"][0]["symbols"], 1)
        with self.assertRaises(FileNotFoundError):
            ps.build(self.raw, self.out, markets=("KR",), symbols=["NOPE"])

    def _bundle_files(self, bundle: Path) -> dict:
        return {p.name: ps._sha256(p) for p in sorted(bundle.iterdir())}

    def test_partial_run_keeps_current_bundle(self):
        full = ps.build(self.raw, self.out)
        before = self._bundle_files(ps.current_bundle(self.out))
        for markets, symbols in ((("KR",), None), (("KR", "US"), ["AAA111", "XYZ"])):
            partial = ps.build(self.raw, self.out, markets=markets, symbols=symbols)
            self.assertFalse(partial["promoted"])
            self.assertFalse(partial["scope"]["complete"])
            self.assertTrue(Path(partial["bundle_dir"]).is_dir())  # 일부 실행 묶음은 따로 남는다
        self.assertEqual(str(ps.current_bundle(self.out)), full["bundle_dir"])
        self.assertEqual(self._bundle_files(ps.current_bundle(self.out)), before)

    def test_failed_rebuild_keeps_previous_bundle_intact(self):
        ps.build(self.raw, self.out)
        current = ps.current_bundle(self.out)
        before = self._bundle_files(current)
        original = ps.build_market

        def fail_on_us(source, out, symbols=None):
            if source.market == "US":  # KR 파일을 다 쓴 뒤 두 번째 시장에서 실패
                raise OSError("디스크 가득 참(모의)")
            return original(source, out, symbols)

        with mock.patch.object(ps, "build_market", side_effect=fail_on_us):
            with self.assertRaises(OSError):
                ps.build(self.raw, self.out)
        self.assertEqual(ps.current_bundle(self.out), current)
        self.assertEqual(self._bundle_files(current), before)
        self.assertEqual([p.name for p in (self.out / ps.RUNS_DIR).iterdir()], [current.name])  # 임시 폴더 정리

    def test_failed_manifest_verification_does_not_promote(self):
        with mock.patch.object(ps, "_verify_bundle", side_effect=RuntimeError("해시 불일치(모의)")):
            with self.assertRaises(RuntimeError):
                ps.build(self.raw, self.out)
        self.assertFalse((self.out / ps.CURRENT_POINTER).exists())
        with self.assertRaises(FileNotFoundError):
            ps.current_bundle(self.out)
        self.assertEqual(list((self.out / ps.RUNS_DIR).iterdir()), [])

    def test_invalid_corp_actions_flagged_not_normalized(self):
        rows = [
            "date,open,high,low,close,volume,dividend,split",
            "2024-01-02,10,11,9,10.5,100,,",          # 빈 칸 = 사건 없음
            "2024-01-03,10,11,9,10.5,100,abc,",       # 숫자 아님
            "2024-01-04,10,11,9,10.5,100,-0.1,",      # 음수 분배금
            "2024-01-05,10,11,9,10.5,100,N/A,",       # pandas 기본 결측 문자열도 빈 칸으로 보지 않음
            "2024-01-08,10,11,9,10.5,100,0,x:1",      # 분할 형식 오류
            "2024-01-09,10,11,9,10.5,100,0,2",        # 비율 기호 없음
            "2024-01-10,10,11,9,10.5,100,0,0:1",      # 0 비율
            "2024-01-11,10,11,9,10.5,100,0.3,3:2",    # 정상 사건
        ]
        path = self.raw / ps.US_SUBDIR / "BAD.csv"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        frame = ps.load_symbol("US", path, "BAD")
        self.assertEqual(frame["corp_action_ok"].tolist(), [True, False, False, False, False, False, False, True])
        self.assertEqual(frame.loc[0, "dividend"], 0.0)
        self.assertEqual(frame.loc[0, "split_ratio"], 1.0)
        self.assertTrue(frame.loc[1:3, "dividend"].isna().all())  # 0으로 바꾸지 않는다
        self.assertTrue(frame.loc[4:6, "split_ratio"].isna().all())  # 1로 바꾸지 않는다
        self.assertEqual(frame.loc[7, "dividend"], 0.3)
        self.assertEqual(frame.loc[7, "split_ratio"], 1.5)
        quality = ps.symbol_quality(frame, None)
        self.assertEqual(quality["bad_corp_action_rows"], 6)
        self.assertEqual(quality["dividend_events"], 1)
        self.assertEqual(quality["split_events"], 1)  # NaN을 분할 사건으로 세지 않는다
        summary = ps.build(self.raw, self.out, markets=("US",))
        self.assertEqual(summary["markets"][0]["symbols_with_bad_corp_actions"], 1)
        inventory = pd.read_csv(Path(summary["bundle_dir"]) / "us_inventory.csv").set_index("symbol")
        self.assertEqual(inventory.loc["BAD", "bad_corp_action_rows"], 6)
        self.assertEqual(inventory.loc["XYZ", "bad_corp_action_rows"], 0)

    def test_missing_corp_action_columns_are_flagged(self):
        path = self.raw / ps.US_SUBDIR / "NOCOL.csv"
        path.write_text("date,open,high,low,close,volume\n2024-01-02,10,11,9,10.5,100\n", encoding="utf-8")
        frame = ps.load_symbol("US", path, "NOCOL")
        self.assertFalse(frame.loc[0, "corp_action_ok"])  # 열이 없으면 '사건 없음'이 아니라 모름

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
