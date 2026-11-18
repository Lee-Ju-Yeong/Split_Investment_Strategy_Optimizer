"""전송 목록과 M5 입력의 실제 파일 경계·상폐 근거·변조 거부."""
from datetime import date
import json
from pathlib import Path
import unittest
import zipfile
from unittest.mock import patch

from src.research.etf.m5_inputs import load_m5_inputs, pack_inputs, pin_inputs, read_manifest
import test_research_etf_engine_inputs as fixture
from test_research_etf_engine_inputs import sha, write_rows


class M5InputTests(unittest.TestCase):
    setUp = fixture.M1InputTests.setUp
    save = fixture.M1InputTests.save
    at = staticmethod(fixture.M1InputTests.at)

    def pin(self):
        self.manifest = self.root / "m5-inputs.json"
        return pin_inputs(self.root, self.split, self.manifest, synthetic=True)

    def load(self, symbols=None):
        return load_m5_inputs(self.root, self.manifest, symbols or ["069500"], self.start, self.end)

    def controls(self, values):
        (self.root / "protocol" / "m5_controls.json").write_text(json.dumps(values), encoding="utf-8")

    def test_pinned_inputs_are_portable_and_preserve_original_split(self):
        original = sha(self.split)
        manifest = self.pin()
        result = self.load()
        self.assertTrue(result.provenance["synthetic"])
        self.assertEqual(result.domestic_settlement, "t_plus_2")
        self.assertEqual(result.max_fx_age_days, 7)
        self.assertEqual(result.provenance["evaluation_calendar_sha256"], self.load().provenance["evaluation_calendar_sha256"])
        self.assertEqual(sha(self.split), original)
        self.assertTrue(all(not Path(name).is_absolute() for name in manifest["files"]))
        self.assertEqual(result.provenance["m5_input_manifest_sha256"], sha(self.manifest))

    def test_each_calendar_fx_current_and_prices_are_pinned(self):
        manifest = self.pin()
        for name in manifest["files"]:
            path = self.root / name
            before = path.read_bytes()
            with self.subTest(name=name):
                path.write_bytes(before + b"changed")
                with self.assertRaises(ValueError):
                    self.load()
                path.write_bytes(before)

    def test_holdout_is_rejected_before_any_price_hash_or_price_rows(self):
        self.pin()
        with patch("src.research.etf.m5_inputs._check_hash", wraps=__import__(
                "src.research.etf.m5_inputs", fromlist=["_check_hash"])._check_hash) as check:
            with self.assertRaisesRegex(ValueError, "holdout"):
                self.load(["HIDDEN"])
        self.assertEqual([call.args[0] for call in check.call_args_list], [self.split])

    def test_incomplete_or_inconsistent_manifest_is_rejected(self):
        original = self.pin()
        for change in (lambda m: m["files"].pop("fx/calendar_krx.csv"),
                       lambda m: m["extra_inputs_sha256"].update({"calendar_krx.csv": "0" * 64}),
                       lambda m: m.update(original_split_protocol_sha256="0" * 64),
                       lambda m: m.update(synthetic="False")):
            value = json.loads(json.dumps(original))
            change(value)
            self.manifest.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                self.load()

    def test_path_escape_and_noncanonical_path_rejected(self):
        original = self.pin()
        for name in ("../outside", "/tmp/outside", "fx/../protocol/holdout_split_fixture.csv"):
            value = json.loads(json.dumps(original))
            value["files"][name] = "0" * 64
            self.manifest.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                self.load()

    def test_controls_duplicates_bad_bool_and_naive_pay_time_fail(self):
        lock = {"symbol": "069500", "date": self.start.isoformat(), "locked": False}
        cases = ({"locked_status": [lock, lock]},
                 {"locked_status": [dict(lock, locked="False")]},
                 {"payment_dates": [{"symbol": "069500", "date": self.start.isoformat(), "pay_at": "2024-06-10T15:00:00"}]})
        for values in cases:
            with self.subTest(values=values):
                self.controls(values)
                self.pin()
                with self.assertRaises(ValueError):
                    self.load()
                self.manifest.unlink()

    def test_explicit_delisting_ends_bars_without_price_freeze_or_inference(self):
        self.prices["KR"] = [row for row in self.prices["KR"] if row["date"] <= self.start.isoformat()]
        self.save()
        self.pin()
        with self.assertRaisesRegex(ValueError, "세션 누락"):
            self.load()
        self.manifest.unlink()
        self.controls({"delistings": [{"symbol": "069500", "known_at": "2024-06-02T00:00:00+00:00",
                        "last_session": self.start.isoformat(), "delisting_date": self.end.isoformat(),
                        "source_sha256": "a" * 64}]})
        self.pin()
        result = self.load()
        self.assertEqual(len(result.bars), 1)
        self.assertEqual(result.delistings[0].last_session, self.start)
        self.assertEqual(result.provenance["delisting_sources_sha256"], {"069500": "a" * 64})

    def test_strict_settlement_needs_actual_calendar_tail(self):
        path = self.root / "fx" / "calendar_krx.csv"
        from src.research.etf.engine_inputs import _rows
        rows = [row for row in _rows(path) if row["date"] <= self.end.isoformat()]
        write_rows(path, list(rows[0]), rows)
        self.pin()
        with self.assertRaisesRegex(ValueError, "settlement calendar tail"):
            self.load()

    def test_normal_post_delisting_prices_conflict_but_explicit_zero_marker_is_not_a_bar(self):
        self.controls({"delistings": [{"symbol": "069500", "known_at": "2024-06-02T00:00:00+00:00",
                        "last_session": self.start.isoformat(), "delisting_date": self.end.isoformat(),
                        "source_sha256": "a" * 64}]})
        self.pin()
        with self.assertRaisesRegex(ValueError, "source prices contradict"):
            self.load()
        self.manifest.unlink()
        for row in self.prices["KR"]:
            if row["date"] == self.end.isoformat():
                row.update(open="0", high="0", low="0", volume="0")
        self.save()
        self.pin()
        self.assertEqual(len(self.load().bars), 1)

    def test_manifest_does_not_overwrite_existing_output(self):
        self.pin()
        before = self.manifest.read_bytes()
        with self.assertRaises(FileExistsError):
            self.pin()
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_pack_roundtrip_keeps_relative_paths_and_original_hashes(self):
        manifest = self.pin()
        target = self.root / "transfer.zip"
        packed = pack_inputs(self.root, self.manifest, target)
        self.assertEqual(packed["files"], len(manifest["files"]) + 1)
        restored = self.root / "restored"
        with zipfile.ZipFile(target) as archive:
            archive.extractall(restored)
        loaded = load_m5_inputs(restored, restored / "m5-inputs.json", ["069500"], self.start, self.end)
        self.assertEqual(loaded.provenance["split_sha256"], sha(self.split))
        with self.assertRaises(FileExistsError):
            pack_inputs(self.root, self.manifest, target)


if __name__ == "__main__":
    unittest.main()
