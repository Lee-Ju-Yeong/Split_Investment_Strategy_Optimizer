"""연구용 ETF 가격 저장소(M1) 정규화·품질 표시 검사. 네트워크·실제 자료 없이 작은 합성 파일로 돈다."""

import contextlib
import gzip
import io
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
        # CURRENT 승격에는 폐지 원천이 필요하다. 일봉 파일이 없는 폐지 1개만 둔 최소 원천(일봉 결과는 그대로).
        kr_universe = root / "kr_universe"
        (kr_universe / ps.NAVER_DAILY_SUBDIR).mkdir(parents=True)
        (kr_universe / ps.KR_DELISTED_HISTORY).write_bytes(
            ("code,last_name,delisted,last_seen\nGONE01,폐지,True,2003-01-31\n").encode("utf-8-sig"))
        self.kr_delisted = ps.DelistedKrSource.from_dir(kr_universe)

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_without_delisted_source_does_not_promote(self):
        # Codex 검토 회귀: 폐지 원천 없이 직접 호출해도 폐지 포함 CURRENT를 생존 전용 묶음으로 바꾸면 안 된다.
        full = ps.build(self.raw, self.out, kr_delisted=self.kr_delisted)
        survivors_only = ps.build(self.raw, self.out)
        self.assertFalse(survivors_only["promoted"])
        self.assertFalse(survivors_only["scope"]["complete"])
        self.assertEqual(str(ps.current_bundle(self.out)), full["bundle_dir"])

    def test_empty_delisted_history_fails(self):
        path = self.kr_delisted.history_csv
        path.write_bytes("code,last_name,delisted,last_seen\nAAA111,생존,False,2024-01-31\n".encode("utf-8-sig"))
        with self.assertRaises(ValueError):
            ps.build(self.raw, self.out, kr_delisted=self.kr_delisted)
        self.assertFalse((self.out / ps.CURRENT_POINTER).exists())

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
        first = ps.build(self.raw, self.out, kr_delisted=self.kr_delisted)
        digest = {m["market"]: m["daily_sha256"] for m in first["markets"]}
        second = ps.build(self.raw, self.out, kr_delisted=self.kr_delisted)
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
        full = ps.build(self.raw, self.out, kr_delisted=self.kr_delisted)
        before = self._bundle_files(ps.current_bundle(self.out))
        for markets, symbols in ((("KR",), None), (("KR", "US"), ["AAA111", "XYZ"])):
            partial = ps.build(self.raw, self.out, markets=markets, symbols=symbols, kr_delisted=self.kr_delisted)
            self.assertFalse(partial["promoted"])
            self.assertFalse(partial["scope"]["complete"])
            self.assertTrue(Path(partial["bundle_dir"]).is_dir())  # 일부 실행 묶음은 따로 남는다
        self.assertEqual(str(ps.current_bundle(self.out)), full["bundle_dir"])
        self.assertEqual(self._bundle_files(ps.current_bundle(self.out)), before)

    def test_failed_rebuild_keeps_previous_bundle_intact(self):
        ps.build(self.raw, self.out, kr_delisted=self.kr_delisted)
        current = ps.current_bundle(self.out)
        before = self._bundle_files(current)
        original = ps.build_market

        def fail_on_us(source, out, symbols=None):
            if source.market == "US":  # KR 파일을 다 쓴 뒤 두 번째 시장에서 실패
                raise OSError("디스크 가득 참(모의)")
            return original(source, out, symbols)

        with mock.patch.object(ps, "build_market", side_effect=fail_on_us):
            with self.assertRaises(OSError):
                ps.build(self.raw, self.out, kr_delisted=self.kr_delisted)
        self.assertEqual(ps.current_bundle(self.out), current)
        self.assertEqual(self._bundle_files(current), before)
        self.assertEqual([p.name for p in (self.out / ps.RUNS_DIR).iterdir()], [current.name])  # 임시 폴더 정리

    def test_failed_manifest_verification_does_not_promote(self):
        with mock.patch.object(ps, "_verify_bundle", side_effect=RuntimeError("해시 불일치(모의)")):
            with self.assertRaises(RuntimeError):
                ps.build(self.raw, self.out, kr_delisted=self.kr_delisted)
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


def naver_xml(symbol, rows, name="테스트 ETF", lead=b""):
    """Naver fchart 응답 모양의 EUC-KR 바이트."""
    items = "".join(f'\n\t\t\t\t<item data="{r}" />' for r in rows)
    text = (f'<?xml version="1.0" encoding="EUC-KR" ?>\n\n<protocol>\n\t\t\t<chartdata symbol="{symbol}" '
            f'name="{name}" count="{len(rows)}" timeframe="day" precision="0" origintime="20240102">'
            f'{items}\n\t</chartdata>\n</protocol>\n')
    return lead + text.encode("euc-kr")


EMPTY_XML = b'\n\n<?xml version="1.0" encoding="EUC-KR" ?>\n<protocol />'
DEL_ROWS = [
    "20240102|100|102|99|101|500",
    "20240103|0|0|0|101|0",          # 중간의 거래 없는 날: 일봉에 남기고 표시만
    "20240104|101|103|100|102|700",
    "20240105|0|0|0|102|0",          # 마지막 행: 폐지 표시 행
]
HISTORY_HEADER = "code,first_name,last_name,index_name,first_seen,last_seen,delisted,flags,has_daily"


class DelistedKrTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.raw, self.out, self.kr_universe = root / "raw", root / "out", root / "kr_universe"
        for subdir, files in ((ps.KR_SUBDIR, {"AAA111": KR_ROWS, "069500": KR_REF}),
                              (ps.US_SUBDIR, {"XYZ": US_ROWS, "SPY": US_REF})):
            folder = self.raw / subdir
            folder.mkdir(parents=True)
            for symbol, rows in files.items():
                (folder / f"{symbol}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.out.mkdir()
        naver = self.kr_universe / ps.NAVER_DAILY_SUBDIR
        naver.mkdir(parents=True)
        (naver / "DEL001.xml").write_bytes(naver_xml("DEL001", DEL_ROWS, name="코덱스 테스트"))
        (naver / "DEL002.xml").write_bytes(EMPTY_XML)
        (naver / "DEL004.xml").write_bytes(naver_xml("OTHER9", DEL_ROWS))
        (naver / "DEL005.xml").write_bytes(
            b'<?xml version="1.0" encoding="EUC-KR" ?><!DOCTYPE p [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]>'
            b'<protocol><chartdata symbol="DEL005"><item data="20240102|1|1|1|1|1" /></chartdata></protocol>')
        history = [HISTORY_HEADER,
                   "DEL001,코덱스 테스트,코덱스 테스트,코스피 200,2024-01-31,2024-01-31,True,,True",
                   "DEL002,빈응답,빈응답,코스피 200,2003-01-31,2003-01-31,True,,False",
                   "DEL003,파일없음,파일없음,코스피 200,2003-01-31,2003-01-31,True,,False",
                   "DEL004,코드다름,코드다름,코스피 200,2024-01-31,2024-01-31,True,,True",
                   "DEL005,엔티티,엔티티,코스피 200,2024-01-31,2024-01-31,True,,True",
                   "AAA111,생존,생존,코스피 200,2024-01-31,2024-01-31,False,,True"]
        (self.kr_universe / ps.KR_DELISTED_HISTORY).write_bytes(("\n".join(history) + "\n").encode("utf-8-sig"))
        snapshots = self.kr_universe / ps.KRX_SNAPSHOT_SUBDIR
        snapshots.mkdir()
        # KRX 실제 종가: DEL001 01-02 종가 101 → 102(Naver가 낮음 = 수정주가 모양), 01-04는 같음
        snap = ["trade_date,ISU_SRT_CD,ISU_ABBRV,TDD_CLSPRC",
                "2024-01-02,DEL001,코덱스 테스트,102", "2024-01-02,AAA111,생존,105",
                "2024-01-04,DEL001,코덱스 테스트,102", "2024-01-04,AAA111,생존,-"]
        (snapshots / "20240102.csv").write_bytes(("\n".join(snap) + "\n").encode("utf-8-sig"))
        self.source = ps.DelistedKrSource.from_dir(self.kr_universe)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_naver_chart_reads_euc_kr_and_empty_response(self):
        chart = ps.parse_naver_chart(naver_xml("DEL001", DEL_ROWS, lead=b"\r\n"))
        self.assertEqual(chart.symbol, "DEL001")
        self.assertEqual(chart.rows[0], ("20240102", "100", "102", "99", "101", "500"))
        empty = ps.parse_naver_chart(EMPTY_XML)
        self.assertIsNone(empty.symbol)
        self.assertEqual(empty.rows, [])

    def test_parse_naver_chart_refuses_entities_and_doctype(self):
        bomb = (b'<?xml version="1.0" encoding="EUC-KR" ?><!DOCTYPE p [<!ENTITY a "aaaaaaaaaa">'
                b'<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]><protocol>&b;</protocol>')
        external = (b'<?xml version="1.0" encoding="EUC-KR" ?><!DOCTYPE p SYSTEM "file:///etc/passwd">'
                    b'<protocol />')
        for payload in (bomb, external):
            with self.assertRaises(ps.UnsafeXmlError):
                ps.parse_naver_chart(payload)

    def test_parse_naver_chart_rejects_malformed_input(self):
        cases = [
            b"<protocol />",                                                     # 선언 없음
            b'<?xml version="1.0" encoding="SHIFT_JIS" ?><protocol />',          # 모르는 인코딩
            b'<?xml version="1.0" encoding="EUC-KR" ?><protocol>\xff\xff</protocol>',  # 깨진 바이트
            naver_xml("DEL001", ["20240102|1|2|3"]),                             # 필드 수 부족
            b'<?xml version="1.0" encoding="EUC-KR" ?><protocol><chartdata symbol="A"/>'
            b'<chartdata symbol="B"/></protocol>',                               # chartdata 2개
        ]
        for payload in cases:
            with self.assertRaises((ValueError, ps.expat.ExpatError), msg=payload[:60]):
                ps.parse_naver_chart(payload)

    def test_delisting_marker_moved_out_of_trading_days(self):
        daily, markers, info = ps.load_naver_delisted(ps.parse_naver_chart(naver_xml("DEL001", DEL_ROWS)), "DEL001")
        self.assertEqual(daily["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-02", "2024-01-03", "2024-01-04"])
        self.assertEqual(daily["ohlc_ok"].tolist(), [True, False, True])  # 중간 0 시가 행은 남기고 표시
        self.assertEqual(daily.loc[1, "close"], 101.0)                    # 값을 고치지 않는다
        self.assertEqual(list(daily.columns), ps.COLUMNS)
        self.assertTrue((daily["listing_status"] == ps.DELISTED).all())
        self.assertTrue((daily["delisting_date"] == pd.Timestamp("2024-01-05")).all())
        self.assertEqual(daily.loc[0, "price_basis"], ps.PRICE_BASIS["KR"])
        self.assertEqual(markers["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-05"])
        self.assertEqual(info, {"delisting_marker_found": True, "delisting_date": "2024-01-05"})
        # 마지막 행이 폐지 표시 모양이 아니면 아무 행도 옮기지 않는다.
        daily2, markers2, info2 = ps.load_naver_delisted(
            ps.parse_naver_chart(naver_xml("DEL001", DEL_ROWS[:3])), "DEL001")
        self.assertEqual(len(daily2), 3)
        self.assertTrue(markers2.empty)
        self.assertEqual(info2, {"delisting_marker_found": False, "delisting_date": None})
        self.assertTrue(daily2["delisting_date"].isna().all())

    def test_build_with_delisted_source(self):
        summary = ps.build(self.raw, self.out, kr_delisted=self.source)
        self.assertTrue(summary["promoted"])
        self.assertTrue(summary["scope"]["kr_delisted"])
        bundle = ps.current_bundle(self.out)
        kr = summary["markets"][0]
        self.assertEqual((kr["symbols"], kr["symbols_listed"], kr["symbols_delisted"]), (3, 2, 1))
        self.assertEqual(kr["delisted_source"]["candidates"], 5)
        self.assertEqual(kr["delisted_source"]["included"], 1)
        self.assertEqual(kr["delisted_source"]["excluded"],
                         {"naver_empty_response": 1, "naver_file_missing": 1, "symbol_mismatch": 1, "unsafe_xml": 1})
        self.assertEqual(kr["delisted_source"]["history_sha256"], ps._sha256(self.source.history_csv))
        self.assertEqual(set(kr["files"]), {"kr_daily.csv.gz", "kr_inventory.csv",
                                            ps.DELISTED_SOURCES_FILE, ps.DELISTING_MARKERS_FILE})
        with gzip.open(bundle / "kr_daily.csv.gz", "rt", encoding="utf-8") as handle:
            daily = pd.read_csv(handle, dtype={"symbol": str}, keep_default_na=False)
        self.assertEqual(list(daily.columns), ps.COLUMNS)
        dl = daily[daily["symbol"] == "DEL001"]
        self.assertEqual(dl["date"].tolist(), ["2024-01-02", "2024-01-03", "2024-01-04"])  # 폐지 표시 행 제외
        self.assertEqual(set(dl["listing_status"]), {"delisted"})
        self.assertEqual(set(dl["delisting_date"]), {"2024-01-05"})
        self.assertEqual(set(daily.loc[daily["symbol"] != "DEL001", "listing_status"]), {"listed"})
        self.assertEqual(set(daily.loc[daily["symbol"] != "DEL001", "delisting_date"]), {""})
        markers = pd.read_csv(bundle / ps.DELISTING_MARKERS_FILE, dtype={"symbol": str})
        self.assertEqual(markers[["symbol", "date", "open"]].values.tolist(), [["DEL001", "2024-01-05", 0.0]])
        sources = pd.read_csv(bundle / ps.DELISTED_SOURCES_FILE, dtype=str, keep_default_na=False).set_index("symbol")
        self.assertEqual(sources.loc["DEL001", "status"], "included")
        self.assertEqual(sources.loc["DEL001", "name"], "코덱스 테스트")  # UTF-8 이름 보존
        self.assertEqual(sources.loc["DEL004", "detail"], "chartdata symbol='OTHER9'")
        self.assertEqual(sources.loc["DEL003", "xml_sha256"], "")
        self.assertEqual(len(sources.loc["DEL001", "xml_sha256"]), 64)
        self.assertNotIn("AAA111", sources.index)  # 생존 행은 폐지 원천 기록에 없다
        inventory = pd.read_csv(bundle / "kr_inventory.csv", dtype={"symbol": str}).set_index("symbol")
        self.assertEqual(inventory.loc["DEL001", "listing_status"], "delisted")
        self.assertEqual(inventory.loc["DEL001", "source"], ps.DELISTED_SOURCE_LABEL)
        self.assertEqual(inventory.loc["DEL001", "source_sha256"], sources.loc["DEL001", "xml_sha256"])
        self.assertEqual(inventory.loc["DEL001", "bad_ohlc_rows"], 1)
        self.assertEqual(inventory.loc["AAA111", "listing_status"], "listed")
        # KRX 실제 종가 대조: DEL001 01-02는 Naver가 낮고 01-04는 같다. 휴장 '-' 칸은 빼고 센다.
        self.assertEqual(inventory.loc["DEL001", "krx_close_compared"], 2)
        self.assertEqual(inventory.loc["DEL001", "krx_close_equal"], 1)
        self.assertAlmostEqual(inventory.loc["DEL001", "krx_close_min_ratio"], 101 / 102)
        self.assertEqual(kr["krx_close_check"]["delisted"]["rows_naver_below_krx"], 1)
        self.assertEqual(kr["krx_close_check"]["listed"]["rows_equal"], 1)
        self.assertEqual(kr["krx_close_check"]["listed"]["rows_naver_above_krx"], 0)

    def test_tampered_side_file_fails_verification(self):
        summary = ps.build(self.raw, self.out, kr_delisted=self.source)
        bundle = Path(summary["bundle_dir"])
        manifest = json.loads((bundle / ps.MANIFEST).read_text(encoding="utf-8"))
        ps._verify_bundle(bundle, manifest)
        with open(bundle / ps.DELISTING_MARKERS_FILE, "a", encoding="utf-8") as handle:
            handle.write("tampered\n")
        with self.assertRaises(RuntimeError):
            ps._verify_bundle(bundle, manifest)

    def test_symbol_filter_reaches_delisted_and_is_partial(self):
        summary = ps.build(self.raw, self.out, markets=("KR",), symbols=["DEL001"], kr_delisted=self.source)
        self.assertFalse(summary["promoted"])
        kr = summary["markets"][0]
        self.assertEqual((kr["symbols_listed"], kr["symbols_delisted"]), (0, 1))
        self.assertEqual(kr["delisted_source"]["candidates"], 1)

    def test_code_in_both_survivor_and_delisted_fails_closed(self):
        ps.build(self.raw, self.out, kr_delisted=self.source)
        current = ps.current_bundle(self.out)
        history = [HISTORY_HEADER, "AAA111,생존,생존,코스피 200,2024-01-31,2024-01-31,True,,True"]
        (self.kr_universe / ps.KR_DELISTED_HISTORY).write_bytes(("\n".join(history) + "\n").encode("utf-8-sig"))
        with self.assertRaises(ValueError):
            ps.build(self.raw, self.out, kr_delisted=self.source)
        self.assertEqual(ps.current_bundle(self.out), current)

    def test_marker_not_after_last_trade_is_excluded(self):
        rows = ["20240104|101|103|100|102|700", "20240103|0|0|0|102|0"]
        (self.kr_universe / ps.NAVER_DAILY_SUBDIR / "DEL001.xml").write_bytes(naver_xml("DEL001", rows))
        summary = ps.build(self.raw, self.out, markets=("KR",), kr_delisted=self.source)
        excluded = summary["markets"][0]["delisted_source"]["excluded"]
        self.assertEqual(excluded.get("invalid_rows"), 1)

    def test_cli_requires_delisted_source(self):
        env = {"MAGICSPLIT_ETF_DATA_ROOT": str(Path(self.tmp.name) / "empty_data"),
               "MAGICSPLIT_ETF_RAW_ROOT": str(self.raw)}
        with mock.patch.dict("os.environ", env), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                ps.main([])
            ps.main(["--markets", "US"])  # KR을 돌리지 않으면 폐지 원천이 필요 없다(일부 실행)
        self.assertFalse((Path(env["MAGICSPLIT_ETF_DATA_ROOT"]) / "prices" / ps.CURRENT_POINTER).exists())


if __name__ == "__main__":
    unittest.main()
