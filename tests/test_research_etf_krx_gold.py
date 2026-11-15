"""KRX 금현물 수동 다운로드 CSV 적재 테스트(네트워크 없음).

표본은 KRX '개별종목 시세 추이' CSV와 같은 형식(cp949, 따옴표, 최신일이 위)이지만 값은 만든 것이다.
KRX 이용약관이 자료 복제·배포를 금지하므로 실제 시세를 저장소에 넣지 않는다.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.research.etf import krx_gold as kg  # noqa: E402

HEADER = "일자,종가,대비,등락률,시가,고가,저가,거래량,거래대금\n"

# 2025-01-02(목) ~ 2025-01-08(수) 국내 거래일 5일. 거래대금 = 거래량 × 대략 종가.
NEWER = (
    '"2025/01/08","106000","1000","0.95","105500","106200","105400","1000","106000000"\n'
    '"2025/01/07","105000","1000","0.96","104500","105100","104400","2000","210000000"\n'
    '"2025/01/06","104000","3000","2.97","101500","104100","101400","1500","156000000"\n'
)
OLDER = (
    '"2025/01/06","104000","3000","2.97","101500","104100","101400","1500","156000000"\n'  # 겹치는 날(같은 값)
    '"2025/01/03","101000","1000","1.00","100500","101200","100400","3000","303000000"\n'
    '"2025/01/02","100000","0","0.00","100000","100300","99800","2500","250000000"\n'
)
CALENDAR = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]


def _csv(body: str, encoding: str = "cp949") -> bytes:
    return (HEADER + body).encode(encoding)


def _source(name: str, data: bytes) -> kg.SourceFile:
    return kg.SourceFile(name, kg._sha256(data), len(data), "2026-11-15T00:00:00+00:00", Path(name))


def _daily(*bodies: str) -> pd.DataFrame:
    parsed = []
    for i, body in enumerate(bodies):
        data = _csv(body)
        parsed.append((_source(f"f{i}.csv", data), kg.parse_krx_gold_csv(data)))
    return kg.normalize(parsed, ingested_at_utc="2026-11-15T00:00:00+00:00")


class ParseTest(unittest.TestCase):
    def test_parses_cp949_krx_format(self):
        frame = kg.parse_krx_gold_csv(_csv(NEWER))
        self.assertEqual(list(frame.columns), list(kg.HEADER_MAP.values()))
        self.assertEqual(len(frame), 3)
        first = frame.iloc[0]
        self.assertEqual(first["date"], pd.Timestamp("2025-01-08"))
        self.assertEqual(
            (first["open"], first["high"], first["low"], first["close"]),
            (105500.0, 106200.0, 105400.0, 106000.0),
        )
        self.assertEqual(first["volume"], 1000.0)
        self.assertEqual(first["value"], 106000000.0)
        self.assertAlmostEqual(first["change_pct"], 0.95)

    def test_accepts_utf8_bom_and_thousands_separator(self):
        body = '"2025/01/02","100,000","0","0.00","100,000","100,300","99,800","2,500","250,000,000"\n'
        frame = kg.parse_krx_gold_csv(b"\xef\xbb\xbf" + (HEADER + body).encode("utf-8"))
        self.assertEqual(frame.iloc[0]["close"], 100000.0)
        self.assertEqual(frame.iloc[0]["value"], 250000000.0)

    def test_blank_cell_becomes_nan(self):
        body = '"2025/01/02","100000","0","0.00","","100300","99800","2500","250000000"\n'
        self.assertTrue(pd.isna(kg.parse_krx_gold_csv(_csv(body)).iloc[0]["open"]))

    def test_missing_column_is_rejected(self):
        bad = "일자,종가,시가,고가,저가\n\"2025/01/02\",\"1\",\"1\",\"1\",\"1\"\n".encode("cp949")
        with self.assertRaises(kg.KrxGoldFormatError):
            kg.parse_krx_gold_csv(bad)

    def test_non_finite_numbers_are_rejected(self):
        # float()가 받아들이는 inf·nan·범위 초과 값은 KRX 원천에 나올 수 없으므로 형식 오류다.
        for token in ("inf", "-Infinity", "nan", "1e999"):
            body = f'"2025/01/02","100000","0","0.00","100000","{token}","99800","2500","250000000"\n'
            with self.subTest(token=token), self.assertRaises(kg.KrxGoldFormatError):
                kg.parse_krx_gold_csv(_csv(body))

    def test_bad_date_is_rejected(self):
        body = '"02-01-2025","100000","0","0.00","100000","100300","99800","2500","250000000"\n'
        with self.assertRaises(kg.KrxGoldFormatError):
            kg.parse_krx_gold_csv(_csv(body))


class NormalizeTest(unittest.TestCase):
    def test_merges_sorts_and_drops_identical_overlap(self):
        daily = _daily(NEWER, OLDER)
        self.assertEqual(list(daily.columns), kg.NORMALIZED_COLUMNS)
        self.assertEqual(daily["date"].dt.strftime("%Y-%m-%d").tolist(), CALENDAR)
        self.assertTrue((daily["product"] == "gold_9999_1kg").all())
        self.assertTrue((daily["price_unit"] == "KRW/g").all())
        self.assertTrue(daily["source_sha256"].str.len().eq(64).all())

    def test_conflicting_overlap_is_rejected(self):
        changed = OLDER.replace('"104000","3000"', '"104010","3010"', 1)
        with self.assertRaises(kg.KrxGoldFormatError):
            _daily(NEWER, changed)

    def test_unknown_product_is_rejected(self):
        data = _csv(NEWER)
        with self.assertRaises(ValueError):
            kg.normalize([(_source("a.csv", data), kg.parse_krx_gold_csv(data))], product="silver")


class QualityTest(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.DatetimeIndex(pd.to_datetime(CALENDAR))

    def test_clean_sample_has_no_blocking_issue(self):
        report, jumps = kg.quality_checks(_daily(NEWER, OLDER), calendar=self.calendar)
        self.assertEqual(report["blocking_issue_count"], 0)
        self.assertEqual(report["warning_count"], 0)
        self.assertEqual(report["calendar_trading_days_in_range"], 5)
        self.assertEqual(report["missing_vs_calendar_dates"], [])
        self.assertEqual(report["rows"], 5)
        self.assertEqual((report["first_date"], report["last_date"]), ("2025-01-02", "2025-01-08"))
        self.assertTrue(jumps.empty)  # 최대 일간 변화 약 2.9%

    def test_missing_trading_day_is_flagged_twice(self):
        # 2025-01-07 행이 빠지면 거래일 대비 누락과 '대비' 불일치(01-08)가 함께 잡힌다.
        newer = "".join(l for l in NEWER.splitlines(keepends=True) if "2025/01/07" not in l)
        report, _ = kg.quality_checks(_daily(newer, OLDER), calendar=self.calendar)
        self.assertEqual(report["missing_vs_calendar_dates"], ["2025-01-07"])
        self.assertEqual(report["change_mismatch_dates"], ["2025-01-08"])
        self.assertEqual(report["blocking_issue_count"], 2)

    def test_extra_non_trading_day_is_flagged(self):
        report, _ = kg.quality_checks(_daily(NEWER, OLDER), calendar=self.calendar.drop(pd.Timestamp("2025-01-03")))
        self.assertEqual(report["extra_vs_calendar_dates"], ["2025-01-03"])

    def test_ohlc_zero_and_missing_values(self):
        daily = _daily(NEWER, OLDER)
        daily.loc[daily["date"] == "2025-01-03", "high"] = 100000.0   # 고가 < 종가
        daily.loc[daily["date"] == "2025-01-06", "low"] = 0.0         # 0 가격
        daily.loc[daily["date"] == "2025-01-07", "volume"] = float("nan")
        report, _ = kg.quality_checks(daily, calendar=self.calendar)
        self.assertEqual(report["ohlc_inconsistent_dates"], ["2025-01-03"])
        self.assertEqual(report["nonpositive_price_dates"], ["2025-01-06"])
        self.assertEqual(report["missing_value_dates"], ["2025-01-07"])
        self.assertGreaterEqual(report["blocking_issue_count"], 3)

    def test_infinite_values_are_blocking(self):
        # 파서를 거치지 않은 표에 무한대가 있어도 정상 판정하지 않는다.
        daily = _daily(NEWER, OLDER)
        daily.loc[daily["date"] == "2025-01-07", "high"] = float("inf")
        daily.loc[daily["date"] == "2025-01-03", "value"] = float("-inf")
        report, _ = kg.quality_checks(daily, calendar=self.calendar)
        self.assertEqual(report["non_finite_value_dates"], ["2025-01-03", "2025-01-07"])
        self.assertGreaterEqual(report["blocking_issue_count"], 2)

    def test_vwap_outside_range_is_warning_only(self):
        daily = _daily(NEWER, OLDER)
        daily.loc[daily["date"] == "2025-01-03", "value"] = 3000 * 99000.0  # 평균가 99,000 < 저가 100,400
        report, _ = kg.quality_checks(daily, calendar=self.calendar)
        self.assertEqual(report["vwap_outside_range_dates"], ["2025-01-03"])
        self.assertEqual(report["blocking_issue_count"], 0)
        self.assertEqual(report["warning_count"], 1)

    def test_jump_and_price_limit(self):
        # 2025-01-09: 106,000 → 95,400 (-10%, 가격제한폭)
        limit_down = '"2025/01/09","95400","-10600","-10.00","95400","95400","95400","500","47700000"\n'
        daily = _daily(limit_down + NEWER, OLDER)
        calendar = self.calendar.append(pd.DatetimeIndex([pd.Timestamp("2025-01-09")]))
        report, jumps = kg.quality_checks(daily, calendar=calendar)
        self.assertEqual(report["jump_dates"], ["2025-01-09"])
        self.assertEqual(report["price_limit_close_dates"], ["2025-01-09"])
        self.assertTrue(bool(jumps.loc[0, "at_price_limit"]))
        self.assertEqual(report["blocking_issue_count"], 0)


class EtfCompareTest(unittest.TestCase):
    def test_identical_returns_have_unit_correlation_and_zero_tracking(self):
        daily = _daily(NEWER, OLDER)
        etf = pd.DataFrame({"date": CALENDAR, "close": (daily["close"] / 10).tolist()})
        result = kg.compare_with_etf(daily, etf)
        self.assertEqual(result["overlap_days"], 5)
        self.assertAlmostEqual(result["daily_return_correlation"], 1.0)
        self.assertAlmostEqual(result["tracking_error_annualized"], 0.0)
        self.assertAlmostEqual(result["tracking_difference_cagr"], 0.0)


class PreserveRawTest(unittest.TestCase):
    def test_copies_bytes_and_keeps_download_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "data_a.csv"
            src.write_bytes(_csv(NEWER))
            os.utime(src, (1_700_000_000, 1_700_000_000))
            kept = kg.preserve_raw(src, tmp / "raw", move=True)
            self.assertFalse(src.exists())
            self.assertEqual(kept.raw_path.read_bytes(), _csv(NEWER))
            self.assertEqual(int(kept.raw_path.stat().st_mtime), 1_700_000_000)
            self.assertEqual(kept.downloaded_at_utc, "2023-11-14T22:13:20+00:00")

    def test_never_overwrites_file_created_by_another_run(self):
        # 다른 실행이 존재 확인 직후 같은 이름으로 먼저 만든 상황: exists()가 False를 돌려줘도
        # 배타적 생성이 실패하고 해시 비교로 멈춰야 한다. 원본도 보존본도 잃지 않는다.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            raw.mkdir()
            (raw / "data_a.csv").write_bytes(_csv(OLDER))   # 먼저 보존된 다른 내용
            src = tmp / "data_a.csv"
            src.write_bytes(_csv(NEWER))
            with mock.patch.object(Path, "exists", return_value=False):
                with self.assertRaises(FileExistsError):
                    kg.preserve_raw(src, raw, move=True)
            self.assertEqual((raw / "data_a.csv").read_bytes(), _csv(OLDER))
            self.assertEqual(src.read_bytes(), _csv(NEWER))

    def test_same_content_already_preserved_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            raw.mkdir()
            (raw / "data_a.csv").write_bytes(_csv(NEWER))
            src = tmp / "data_a.csv"
            src.write_bytes(_csv(NEWER))
            kg.preserve_raw(src, raw, move=True)
            self.assertFalse(src.exists())
            self.assertEqual((raw / "data_a.csv").read_bytes(), _csv(NEWER))


LATER = '"2025/01/09","107000","1000","0.94","106500","107100","106400","1000","107000000"\n'


class IngestTest(unittest.TestCase):
    def _setup_root(self, tmp: Path) -> tuple[Path, Path, Path]:
        inbox, root = tmp / "inbox", tmp / "etf_m1"
        inbox.mkdir()
        (root / "fx").mkdir(parents=True)
        pd.DataFrame({"date": CALENDAR + ["2025-01-09"]}).to_csv(root / "fx" / "calendar_krx.csv", index=False)
        etf = tmp / "411060.csv"
        pd.DataFrame({"date": CALENDAR, "close": [1, 1.01, 1.04, 1.05, 1.06]}).to_csv(etf, index=False)
        return inbox, root, etf

    @staticmethod
    def _outputs(root: Path) -> dict:
        out = root / "krx_gold"
        return {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}

    def test_late_failure_keeps_previous_generation_and_inputs(self):
        # 달력이 깨져 품질 계산 단계에서 실패해도 이전 산출물 세대가 그대로 남고, 새 파일은 옮겨지지 않는다.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inbox, root, etf = self._setup_root(tmp)
            (inbox / "data_a.csv").write_bytes(_csv(NEWER + OLDER))
            with mock.patch.dict(os.environ, {"MAGICSPLIT_ETF_DATA_ROOT": str(root)}):
                kg.ingest([inbox / "data_a.csv"], move=True, etf_path=etf)
                before = self._outputs(root)
                self.assertEqual(set(before), {
                    "gold_9999_1kg_daily.csv", "gold_9999_1kg_jumps.csv",
                    "gold_9999_1kg_quality.json", "gold_9999_1kg_manifest.json",
                })

                (inbox / "data_c.csv").write_bytes(_csv(LATER))
                bad_calendar = tmp / "bad_calendar.csv"
                bad_calendar.write_text("date\nnot-a-date\n", encoding="utf-8")
                with self.assertRaises(Exception):
                    kg.ingest([inbox / "data_c.csv"], move=True, calendar_path=bad_calendar, etf_path=etf)
                self.assertEqual(self._outputs(root), before)   # 임시 파일·잠금 파일도 남지 않는다

                broken_etf = tmp / "broken_etf.csv"
                broken_etf.write_text("when,price\n2025-01-02,1\n", encoding="utf-8")
                with self.assertRaises(Exception):
                    kg.ingest([inbox / "data_c.csv"], move=True, etf_path=broken_etf)
                self.assertEqual(self._outputs(root), before)

                self.assertTrue((inbox / "data_c.csv").exists())
                raw = root / "krx_gold" / "raw" / "gold_9999_1kg"
                self.assertEqual([p.name for p in raw.iterdir()], ["data_a.csv"])

                # 입력을 바로잡으면 새 세대로 한꺼번에 바뀐다.
                result = kg.ingest([inbox / "data_c.csv"], move=True, etf_path=etf)
                self.assertEqual(result["manifest"]["rows"], 6)
                after = self._outputs(root)
                daily = pd.read_csv(io.BytesIO(after["gold_9999_1kg_daily.csv"]))
                quality = json.loads(after["gold_9999_1kg_quality.json"])
                manifest = json.loads(after["gold_9999_1kg_manifest.json"])
                self.assertEqual(daily["date"].iloc[-1], "2025-01-09")
                self.assertEqual(set(daily["ingested_at_utc"]), {manifest["ingested_at_utc"]})
                self.assertEqual(quality["generated_at_utc"], manifest["ingested_at_utc"])

    def test_lock_blocks_concurrent_ingest_and_is_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inbox, root, etf = self._setup_root(tmp)
            (inbox / "data_a.csv").write_bytes(_csv(NEWER))
            lock = root / "krx_gold" / ".gold_9999_1kg.lock"
            with mock.patch.dict(os.environ, {"MAGICSPLIT_ETF_DATA_ROOT": str(root)}):
                lock.parent.mkdir(parents=True)
                lock.write_text("pid=other\n")
                with self.assertRaises(RuntimeError):
                    kg.ingest([inbox / "data_a.csv"], move=True, etf_path=etf)
                self.assertTrue((inbox / "data_a.csv").exists())
                self.assertFalse((root / "krx_gold" / "raw").exists())
                self.assertEqual(lock.read_text(), "pid=other\n")   # 남의 잠금은 지우지 않는다

                lock.unlink()
                kg.ingest([inbox / "data_a.csv"], move=True, etf_path=etf)
                self.assertFalse(lock.exists())
                # 실패한 실행도 자기 잠금은 풀고 끝낸다.
                (inbox / "data_c.csv").write_bytes(_csv(LATER))
                with self.assertRaises(FileNotFoundError):
                    kg.ingest([inbox / "data_c.csv"], etf_path=tmp / "missing_etf.csv")
                self.assertFalse(lock.exists())

    def test_conflicting_inputs_with_same_name_are_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inbox, root, etf = self._setup_root(tmp)
            (inbox / "a").mkdir()
            (inbox / "b").mkdir()
            (inbox / "a" / "data.csv").write_bytes(_csv(NEWER))
            (inbox / "b" / "data.csv").write_bytes(_csv(OLDER))
            with mock.patch.dict(os.environ, {"MAGICSPLIT_ETF_DATA_ROOT": str(root)}):
                with self.assertRaises(FileExistsError):
                    kg.ingest([inbox / "a" / "data.csv", inbox / "b" / "data.csv"], move=True, etf_path=etf)
            self.assertTrue((inbox / "a" / "data.csv").exists() and (inbox / "b" / "data.csv").exists())
            self.assertFalse((root / "krx_gold" / "raw").exists())

    def test_end_to_end_preserves_raw_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inbox, root = tmp / "inbox", tmp / "etf_m1"
            inbox.mkdir()
            (inbox / "data_a.csv").write_bytes(_csv(NEWER))
            (inbox / "data_b.csv").write_bytes(_csv(OLDER))
            (root / "fx").mkdir(parents=True)
            pd.DataFrame({"date": CALENDAR}).to_csv(root / "fx" / "calendar_krx.csv", index=False)
            etf = tmp / "411060.csv"
            pd.DataFrame({"date": CALENDAR, "close": [1, 1.01, 1.04, 1.05, 1.06]}).to_csv(etf, index=False)

            with mock.patch.dict(os.environ, {"MAGICSPLIT_ETF_DATA_ROOT": str(root)}):
                result = kg.ingest(sorted(inbox.glob("*.csv")), move=True, etf_path=etf)
                # 같은 파일을 다시 넣어도(이미 raw에 같은 해시) 문제없다.
                (inbox / "data_a.csv").write_bytes(_csv(NEWER))
                kg.ingest([inbox / "data_a.csv"], move=False, etf_path=etf)
                # 같은 이름·다른 내용은 덮어쓰지 않는다.
                (inbox / "data_a.csv").write_bytes(_csv(OLDER))
                with self.assertRaises(FileExistsError):
                    kg.ingest([inbox / "data_a.csv"], etf_path=etf)

            out = root / "krx_gold"
            raw = out / "raw" / "gold_9999_1kg"
            self.assertEqual(sorted(p.name for p in raw.iterdir()), ["data_a.csv", "data_b.csv"])
            self.assertEqual((raw / "data_b.csv").read_bytes(), _csv(OLDER))
            self.assertFalse((inbox / "data_b.csv").exists())  # --move: 원래 파일 삭제
            daily = pd.read_csv(out / "gold_9999_1kg_daily.csv")
            self.assertEqual(daily["date"].tolist(), CALENDAR)
            self.assertEqual(list(daily.columns), kg.NORMALIZED_COLUMNS)
            manifest = result["manifest"]
            self.assertEqual(manifest["rows"], 5)
            self.assertEqual(len(manifest["raw_files"]), 2)
            self.assertIn("제10조", manifest["terms"])
            quality = result["quality"]
            self.assertEqual(quality["checks"]["blocking_issue_count"], 0)
            self.assertEqual(quality["etf_comparison"]["overlap_days"], 5)
            for name in ("gold_9999_1kg_quality.json", "gold_9999_1kg_manifest.json", "gold_9999_1kg_jumps.csv"):
                self.assertTrue((out / name).exists(), name)

    def test_bad_file_moves_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            good, bad = tmp / "good.csv", tmp / "bad.csv"
            good.write_bytes(_csv(NEWER))
            bad.write_bytes("일자,종가\n\"2025/01/02\",\"1\"\n".encode("cp949"))
            with mock.patch.dict(os.environ, {"MAGICSPLIT_ETF_DATA_ROOT": str(tmp / "root")}):
                with self.assertRaises(kg.KrxGoldFormatError):
                    kg.ingest([good, bad], move=True)
            self.assertTrue(good.exists() and bad.exists())
            self.assertFalse((tmp / "root" / "krx_gold" / "raw").exists())


if __name__ == "__main__":
    unittest.main()
