"""고정 입력 변경·보류 상품·미래 거래대금 누출을 실제 파일 경계에서 검사한다."""
import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from src.research.etf.engine_inputs import load_m1_inputs


def write_rows(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M1InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "prices" / "runs" / "fixture"
        self.start, self.end = date(2024, 6, 3), date(2024, 6, 4)
        day, dates = date(2020, 1, 1), []
        while day <= date(2024, 6, 14):
            if day.weekday() < 5:
                dates.append(day)
            day += timedelta(days=1)
        self.days = dates
        for name, hours in (("krx", (0, 6)), ("nyse", (13, 20))):
            rows = [{"date": day.isoformat(), "open_utc": self.at(day, hours[0]),
                     "close_utc": self.at(day, hours[1])} for day in dates]
            write_rows(self.root / "fx" / f"calendar_{name}.csv", list(rows[0]), rows)
        fx = [{"available_at_utc": self.at(date(2020, 1, 1), 0), "value": "1300"},
              {"available_at_utc": self.at(self.end, 0), "value": "1400"}]
        write_rows(self.root / "fx" / "fx_usdkrw_smbs_mar.csv", list(fx[0]), fx)
        self.meta = [{"market": "KR", "symbol": "069500", "d1_eligible": "True", "kr_tax_class": "국내주식형"},
                     {"market": "US", "symbol": "SPY", "d1_eligible": "True", "kr_tax_class": ""}]
        self.split_rows = [{"market": "KR", "symbol": "069500", "split": "explore"},
                           {"market": "US", "symbol": "SPY", "split": "explore"},
                           {"market": "KR", "symbol": "HIDDEN", "split": "holdout"}]
        self.prices = {market: [{"market": market, "symbol": symbol, "date": day.isoformat(),
                                "open": "1000", "high": "1010", "low": "990", "close": "1000",
                                "volume": "2000000", "dividend": "0", "split_ratio": "1",
                                "corp_action_ok": "True", "price_basis": basis}
                               for day in dates]
                       for market, symbol, basis in (("KR", "069500", "naver_provider_adjusted_distribution_unverified"),
                                                     ("US", "SPY", "yahoo_split_adjusted_quote_with_cash_dividends"))}
        self.split = self.root / "protocol" / "holdout_split_fixture.csv"
        self.save()

    @staticmethod
    def at(day, hour):
        return datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).isoformat()

    def save(self):
        metadata = self.root / "metadata" / "etf_metadata.csv"
        write_rows(metadata, list(self.meta[0]), self.meta)
        write_rows(self.split, list(self.split_rows[0]), self.split_rows)
        hashes = {}
        for market, rows in self.prices.items():
            path = self.bundle / f"{market.lower()}_daily.csv.gz"
            write_rows(path, list(rows[0]), rows)
            hashes[market] = sha(path)
        manifest = self.bundle / "manifest.json"
        manifest.write_text(json.dumps({"daily_sha256": hashes}), encoding="utf-8")
        (self.root / "prices" / "CURRENT").write_text("fixture", encoding="utf-8")
        record = {"split_csv": self.split.name, "split_sha256": sha(self.split),
                  "protocol_sha256": "original-m1-protocol", "inputs": {"metadata_sha256": sha(metadata),
                  "price_manifest_sha256": sha(manifest), "daily_sha256": hashes}}
        self.split.with_suffix(".json").write_text(json.dumps(record), encoding="utf-8")

    def load(self, symbols=None, **options):
        return load_m1_inputs(self.root, symbols or ["069500"], self.start, self.end, self.split, **options)

    def test_reads_only_requested_explore_symbol_with_exact_previous_adv(self):
        result = self.load()
        self.assertEqual([item.symbol for item in result.instruments], ["069500"])
        self.assertEqual(len(result.bars), 2)
        self.assertEqual(result.bars[0].adv20, Decimal("2000000000"))
        self.assertEqual(result.bars[0].previous_close, Decimal("1000"))
        self.assertEqual(result.provenance["m1_split_protocol_sha256"], "original-m1-protocol")
        self.assertNotEqual(result.provenance["run_protocol_sha256"], "original-m1-protocol")
        self.assertIn("calendar_nyse.csv", result.provenance["additional_inputs_sha256"])

    def test_rejects_holdout_before_opening_price_or_metadata(self):
        (self.root / "metadata" / "etf_metadata.csv").unlink()
        with self.assertRaisesRegex(ValueError, "보류"):
            self.load(["HIDDEN"])

    def test_rejects_split_tampering(self):
        with self.split.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "해시"):
            self.load()

    def test_rejects_metadata_tampering(self):
        path = self.root / "metadata" / "etf_metadata.csv"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "해시"):
            self.load()

    def test_rejects_price_bundle_and_daily_tampering(self):
        (self.bundle / "kr_daily.csv.gz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "해시"):
            self.load()

    def test_future_turnover_cannot_change_first_adv(self):
        before = self.load().bars[0].adv20
        for row in self.prices["KR"]:
            if row["date"] >= self.start.isoformat():
                row["volume"] = "999999999"
        self.save()
        self.assertEqual(self.load().bars[0].adv20, before)

    def test_rejects_insufficient_pre_start_history(self):
        self.prices["KR"] = [r for r in self.prices["KR"] if r["date"] >= "2022-01-01"]
        self.save()
        with self.assertRaisesRegex(ValueError, "3년"):
            self.load()

    def test_missing_prior_market_session_cannot_be_replaced_by_an_older_row(self):
        self.prices["KR"] = [row for row in self.prices["KR"] if row["date"] != "2024-05-31"]
        self.save()
        with self.assertRaisesRegex(ValueError, "20거래일"):
            self.load()

    def test_missing_terminal_session_cannot_freeze_a_position(self):
        self.prices["KR"] = [row for row in self.prices["KR"] if row["date"] < self.end.isoformat()]
        self.save()
        with self.assertRaisesRegex(ValueError, "069500.*세션 누락.*상장폐지"):
            self.load()

    def test_delisted_source_ending_before_window_is_rejected(self):
        for row in self.prices["KR"]:
            row["listing_status"] = "delisted"
            row["delisting_date"] = self.end.isoformat()
        self.prices["KR"] = [row for row in self.prices["KR"] if row["date"] < self.end.isoformat()]
        self.save()
        with self.assertRaisesRegex(ValueError, "상장폐지"):
            self.load()

    def test_nontrading_end_date_uses_last_actual_session(self):
        self.end = date(2024, 6, 8)
        result = self.load()
        self.assertEqual(result.bars[-1].session_date, date(2024, 6, 7))

    def truncate_calendar(self, market, through):
        path = self.root / "fx" / f"calendar_{market}.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows = [row for row in rows if row["date"] <= through]
        write_rows(path, list(rows[0]), rows)

    def test_end_beyond_calendar_cannot_silently_shorten_window(self):
        self.end = date(2024, 6, 20)
        with self.assertRaisesRegex(ValueError, "KRX.*요청 종료일.*확인 범위"):
            self.load()

    def test_us_instrument_requires_nyse_calendar_through_end(self):
        self.truncate_calendar("nyse", "2024-06-03")
        with self.assertRaisesRegex(ValueError, "NYSE.*확인 범위"):
            self.load(["SPY"])

    def test_us_fx_schedule_also_requires_krx_calendar_through_end(self):
        self.truncate_calendar("krx", "2024-06-03")
        with self.assertRaisesRegex(ValueError, "KRX.*확인 범위"):
            self.load(["SPY"])

    def test_unused_nyse_calendar_end_does_not_block_kr_only_input(self):
        self.truncate_calendar("nyse", "2024-06-03")
        result = self.load()
        self.assertEqual(len(result.bars), 2)

    def test_zero_volume_zero_ohlc_fails_with_source_context(self):
        for row in self.prices["KR"]:
            row["ohlc_ok"] = "True"
            if row["date"] == self.start.isoformat():
                row.update(open="0", high="0", low="0", volume="0", ohlc_ok="False")
        self.save()
        with self.assertRaisesRegex(ValueError, "069500 2024-06-03.*OHLC.*volume=0.*ohlc_ok=False"):
            self.load()

    def test_rejects_start_liquidity_and_d1_failures(self):
        self.meta[0]["d1_eligible"] = "False"
        self.save()
        with self.assertRaisesRegex(ValueError, "D1"):
            self.load()
        self.meta[0]["d1_eligible"] = "True"
        for row in self.prices["KR"]:
            row["volume"] = "1"
        self.save()
        with self.assertRaisesRegex(ValueError, "거래대금"):
            self.load()

    def test_rejects_unverified_flat_bar_but_accepts_explicit_limit_status(self):
        for row in self.prices["KR"]:
            if row["date"] == self.start.isoformat():
                row["high"] = row["low"] = row["open"] = row["close"]
        self.save()
        with self.assertRaisesRegex(ValueError, "제한폭"):
            self.load()
        result = self.load(locked_status={("069500", self.start): True})
        self.assertTrue(result.bars[0].limit_locked)

    def test_us_calendar_settlement_tail_and_terminal_valuation_are_explicit(self):
        result = self.load(["SPY"])
        self.assertGreater(result.us_sessions[-1], self.end)
        self.assertEqual(result.settlement_session_times[self.end].hour, 20)
        self.assertGreater(result.evaluation_times[-1], result.bars[-1].close_at)

    def test_missing_dividend_payment_stays_missing(self):
        for row in self.prices["US"]:
            if row["date"] == self.end.isoformat():
                row["dividend"] = "1.23"
        self.save()
        result = self.load(["SPY"])
        self.assertIsNone(result.bars[-1].dividend_pay_at)
        self.assertTrue(any("지급일" in warning for warning in result.warnings))

    def test_explicit_dividend_date_and_locked_input_are_fingerprinted(self):
        at = datetime(2024, 6, 10, 15, tzinfo=timezone.utc)
        result = self.load(payment_dates={("069500", self.start): at},
                           locked_status={("069500", self.start): False})
        self.assertEqual(result.provenance["payment_dates"][0]["at"], at.isoformat())
        self.assertFalse(result.provenance["locked_status"][0]["locked"])

    def test_unknown_symbols_duplicates_and_naive_payment_time_fail(self):
        with self.assertRaises(ValueError):
            self.load(["UNKNOWN"])
        with self.assertRaises(ValueError):
            self.load(["069500", "069500"])
        with self.assertRaises(ValueError):
            self.load(payment_dates={("069500", self.start): datetime(2024, 6, 10)})
        with self.assertRaisesRegex(ValueError, "요청한"):
            self.load(locked_status={("UNKNOWN", self.start): False})

    def test_rejects_unusable_corporate_actions(self):
        for row in self.prices["KR"]:
            if row["date"] == self.start.isoformat():
                row["corp_action_ok"] = "False"
        self.save()
        with self.assertRaisesRegex(ValueError, "기업행사"):
            self.load()

    def test_gold_uses_both_partition_and_gram_prices_without_metadata_invention(self):
        self.split_rows.append({"market": "GOLD", "symbol": "KRX_GOLD_9999_1KG", "split": "both"})
        self.save()
        gold = [{key: row[key] for key in ("date", "open", "high", "low", "close", "volume")}
                for row in self.prices["KR"]]
        write_rows(self.root / "krx_gold" / "gold_9999_1kg_daily.csv", list(gold[0]), gold)
        result = self.load(["KRX_GOLD_9999_1KG"])
        self.assertEqual(result.instruments[0].market, "GOLD")
        self.assertEqual(result.instruments[0].price_basis, "raw")
        self.assertEqual(result.bars[0].close, Decimal("1000"))
        self.assertIn("gold_9999_1kg_daily.csv", result.provenance["additional_inputs_sha256"])


if __name__ == "__main__":
    unittest.main()
