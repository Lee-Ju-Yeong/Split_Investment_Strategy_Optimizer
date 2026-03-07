import os
import sys
import unittest
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_handler import DataHandler, PitRuntimeError


class TestDataHandlerTierApis(unittest.TestCase):
    def setUp(self):
        self.db_config = {
            "host": "fake_host",
            "user": "fake_user",
            "password": "fake_password",
            "database": "fake_db",
        }
        self.pool_patcher = patch("mysql.connector.pooling.MySQLConnectionPool")
        self.mock_pool = self.pool_patcher.start()
        self.mock_conn = MagicMock()
        self.mock_pool.return_value.get_connection.return_value = self.mock_conn
        self.handler = DataHandler(self.db_config)
        self.handler.universe_mode = "strict_pit"

    def tearDown(self):
        self.pool_patcher.stop()

    @patch("pandas.read_sql")
    def test_get_stock_tier_as_of(self, mock_read_sql):
        mock_read_sql.return_value = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "stock_code": "005930",
                    "tier": 1,
                    "reason": "prime_liquidity",
                    "liquidity_20d_avg_value": 123,
                }
            ]
        )
        result = self.handler.get_stock_tier_as_of("005930", "2024-01-04")
        self.assertIsNotNone(result)
        self.assertEqual(result["stock_code"], "005930")
        self.assertEqual(result["tier"], 1)

    @patch("pandas.read_sql")
    def test_get_tiers_as_of(self, mock_read_sql):
        mock_read_sql.return_value = pd.DataFrame(
            [
                {
                    "stock_code": "005930",
                    "date": pd.Timestamp("2024-01-03"),
                    "tier": 1,
                    "reason": "prime_liquidity",
                    "liquidity_20d_avg_value": 100,
                },
                {
                    "stock_code": "000660",
                    "date": pd.Timestamp("2024-01-03"),
                    "tier": 2,
                    "reason": "normal_liquidity",
                    "liquidity_20d_avg_value": 80,
                },
            ]
        )
        result = self.handler.get_tiers_as_of(
            as_of_date="2024-01-04",
            tickers=["005930", "000660"],
            allowed_tiers=[1, 2],
        )
        self.assertEqual(set(result.keys()), {"005930", "000660"})
        self.assertEqual(result["005930"]["tier"], 1)
        self.assertEqual(result["000660"]["tier"], 2)

    def test_get_filtered_stock_codes_with_tier(self):
        with patch.object(
            self.handler, "get_filtered_stock_codes", return_value=["A", "B", "C"]
        ), patch.object(
            self.handler,
            "get_tiers_as_of",
            return_value={
                "A": {"tier": 1},
                "C": {"tier": 2},
            },
        ):
            result = self.handler.get_filtered_stock_codes_with_tier(
                date="2024-01-04",
                allowed_tiers=(1, 2),
            )
        self.assertEqual(result, ["A", "C"])

    @patch("pandas.read_sql")
    def test_get_pit_universe_codes_as_of_uses_snapshot_first(self, mock_read_sql):
        mock_read_sql.side_effect = [
            pd.DataFrame({"stock_code": ["A", "B"]}),
        ]

        codes, source = self.handler.get_pit_universe_codes_as_of("2024-01-04")

        self.assertEqual(codes, ["A", "B"])
        self.assertEqual(source, "SNAPSHOT_ASOF")
        self.assertEqual(mock_read_sql.call_count, 1)

    @patch("pandas.read_sql")
    def test_get_pit_universe_codes_as_of_fallbacks_to_history(self, mock_read_sql):
        mock_read_sql.side_effect = [
            pd.DataFrame({"stock_code": []}),
            pd.DataFrame({"stock_code": ["C"]}),
        ]

        codes, source = self.handler.get_pit_universe_codes_as_of("2024-01-04")

        self.assertEqual(codes, ["C"])
        self.assertEqual(source, "HISTORY_ACTIVE_ASOF")
        self.assertEqual(mock_read_sql.call_count, 2)

    def test_get_candidates_with_tier_fallback_pit_prefers_tier1(self):
        with patch.object(
            self.handler,
            "get_pit_universe_codes_as_of",
            return_value=(["A", "B", "C"], "SNAPSHOT_ASOF"),
        ), patch.object(
            self.handler,
            "get_tiers_as_of",
            return_value={
                "A": {"tier": 1, "liquidity_20d_avg_value": 100},
                "C": {"tier": 1, "liquidity_20d_avg_value": 200},
            },
        ):
            codes, source = self.handler.get_candidates_with_tier_fallback_pit("2024-01-04")

        self.assertEqual(codes, ["A", "C"])
        self.assertEqual(source, "TIER_1_SNAPSHOT_ASOF")

    def test_get_candidates_with_tier_fallback_pit_uses_tier12_fallback(self):
        with patch.object(
            self.handler,
            "get_pit_universe_codes_as_of",
            return_value=(["A", "B", "C"], "SNAPSHOT_ASOF"),
        ), patch.object(
            self.handler,
            "get_tiers_as_of",
            return_value={"B": {"tier": 2, "liquidity_20d_avg_value": 150}},
        ):
            codes, source = self.handler.get_candidates_with_tier_fallback_pit("2024-01-04")

        self.assertEqual(codes, ["B"])
        self.assertEqual(source, "TIER_2_FALLBACK_SNAPSHOT_ASOF")

    def test_get_candidates_with_tier_fallback_pit_applies_min_liquidity(self):
        with patch.object(
            self.handler,
            "get_pit_universe_codes_as_of",
            return_value=(["A", "B"], "SNAPSHOT_ASOF"),
        ), patch.object(
            self.handler,
            "get_tiers_as_of",
            return_value={
                "A": {"tier": 1, "liquidity_20d_avg_value": 80},
                "B": {"tier": 1, "liquidity_20d_avg_value": 120},
            },
        ):
            codes, source = self.handler.get_candidates_with_tier_fallback_pit_gated(
                date="2024-01-04",
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=None,
            )

        self.assertEqual(codes, ["B"])
        self.assertEqual(source, "TIER_1_SNAPSHOT_ASOF")

    def test_get_candidates_with_tier_fallback_pit_coverage_gate_fail_fast(self):
        with patch.object(
            self.handler,
            "get_pit_universe_codes_as_of",
            return_value=(["A", "B", "C", "D"], "SNAPSHOT_ASOF"),
        ), patch.object(
            self.handler,
            "get_tiers_as_of",
            return_value={"A": {"tier": 1, "liquidity_20d_avg_value": 100}},
        ):
            with self.assertRaises(PitRuntimeError) as ctx:
                self.handler.get_candidates_with_tier_fallback_pit_gated(
                    date="2024-01-04",
                    min_liquidity_20d_avg_value=0,
                    min_tier12_coverage_ratio=0.6,
                )
        self.assertEqual(ctx.exception.pit_code, "tier12_coverage_gate_failed")
        self.assertEqual(ctx.exception.pit_stage, "tier12_coverage_gate")

    def test_freeze_tier_candidate_manifest_reuses_cached_payload(self):
        with patch.object(
            self.handler,
            "get_pit_universe_codes_as_of",
            return_value=(["A", "B", "C"], "SNAPSHOT_ASOF"),
        ) as pit_mock, patch.object(
            self.handler,
            "get_tiers_as_of",
            return_value={
                "A": {"tier": 1, "liquidity_20d_avg_value": 120},
                "B": {"tier": 2, "liquidity_20d_avg_value": 140},
            },
        ) as tier_mock:
            summary = self.handler.freeze_tier_candidate_manifest(
                [pd.Timestamp("2024-01-04")],
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=0.3,
            )

            self.assertEqual(summary["days"], 1)
            self.assertEqual(summary["universe_mode"], "strict_pit")
            self.assertEqual(summary["min_liquidity_20d_avg_value"], 100)
            self.assertAlmostEqual(summary["min_tier12_coverage_ratio"], 0.3, places=6)
            pit_mock.reset_mock()
            tier_mock.reset_mock()

            codes, source = self.handler.get_candidates_with_tier_fallback_pit_gated(
                date="2024-01-04",
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=0.3,
            )

        self.assertEqual(codes, ["A"])
        self.assertEqual(source, "TIER_1_SNAPSHOT_ASOF")
        pit_mock.assert_not_called()
        tier_mock.assert_not_called()

    def test_live_tier_candidate_lookup_is_cached_per_run(self):
        with patch.object(
            self.handler,
            "get_pit_universe_codes_as_of",
            return_value=(["A", "B", "C"], "SNAPSHOT_ASOF"),
        ) as pit_mock, patch.object(
            self.handler,
            "get_tiers_as_of",
            return_value={
                "A": {"tier": 1, "liquidity_20d_avg_value": 120},
                "B": {"tier": 2, "liquidity_20d_avg_value": 140},
            },
        ) as tier_mock:
            codes1, source1 = self.handler.get_candidates_with_tier_fallback_pit_gated(
                date="2024-01-04",
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=0.3,
            )
            codes2, source2 = self.handler.get_candidates_with_tier_fallback_pit_gated(
                date="2024-01-04",
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=0.3,
            )

        self.assertEqual(codes1, ["A"])
        self.assertEqual(codes2, ["A"])
        self.assertEqual(source1, "TIER_1_SNAPSHOT_ASOF")
        self.assertEqual(source2, "TIER_1_SNAPSHOT_ASOF")
        pit_mock.assert_called_once()
        tier_mock.assert_called_once()

    def test_frozen_tier_candidate_manifest_is_bypassed_when_gate_changes(self):
        with patch.object(
            self.handler,
            "get_pit_universe_codes_as_of",
            return_value=(["A", "B"], "SNAPSHOT_ASOF"),
        ) as pit_mock, patch.object(
            self.handler,
            "get_tiers_as_of",
            return_value={"A": {"tier": 1, "liquidity_20d_avg_value": 120}},
        ) as tier_mock:
            self.handler.freeze_tier_candidate_manifest(
                [pd.Timestamp("2024-01-04")],
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=None,
            )
            pit_mock.reset_mock()
            tier_mock.reset_mock()

            self.handler.get_candidates_with_tier_fallback_pit_gated(
                date="2024-01-04",
                min_liquidity_20d_avg_value=200,
                min_tier12_coverage_ratio=None,
            )

        pit_mock.assert_called_once()
        tier_mock.assert_called_once()

    def test_frozen_tier_candidate_manifest_is_bypassed_when_universe_mode_changes(self):
        with patch.object(
            self.handler,
            "get_pit_universe_codes_as_of",
            return_value=(["A", "B"], "SNAPSHOT_ASOF"),
        ) as pit_mock, patch.object(
            self.handler,
            "get_tiers_as_of",
            return_value={"A": {"tier": 1, "liquidity_20d_avg_value": 120}},
        ) as tier_mock:
            self.handler.freeze_tier_candidate_manifest(
                [pd.Timestamp("2024-01-04")],
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=None,
            )
            self.handler.universe_mode = "optimistic_survivor"
            pit_mock.reset_mock()
            tier_mock.reset_mock()

            self.handler.get_candidates_with_tier_fallback_pit_gated(
                date="2024-01-04",
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=None,
            )

        pit_mock.assert_called_once()
        tier_mock.assert_called_once()

    def test_prepare_strict_frozen_candidate_manifest_uses_signal_dates_only(self):
        self.handler.frozen_candidate_manifest_mode = "record_strict"
        fake_conn = MagicMock()
        with patch.object(
            self.handler,
            "_begin_snapshot_connection",
            return_value=(fake_conn, {"connection_id": 7, "snapshot_started_at": "ts"}),
        ), patch.object(
            self.handler,
            "_resolve_tier_candidate_payload",
            return_value={
                "candidate_codes": ["A"],
                "source": "TIER_1_SNAPSHOT_ASOF",
                "pit_size": 3,
                "tier1_count": 1,
                "tier12_count": 2,
            },
        ) as resolve_mock:
            summary = self.handler.prepare_strict_frozen_candidate_manifest(
                trading_dates=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                candidate_lookup_error_policy="raise",
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=0.3,
            )

        called_dates = [call.args[0] for call in resolve_mock.call_args_list]
        self.assertEqual(called_dates, ["2024-01-02", "2024-01-03"])
        self.assertEqual(summary["signal_dates"], 2)
        self.assertEqual(summary["snapshot_connection_id"], 7)
        self.assertTrue(self.handler._strict_frozen_candidate_manifest_required)
        fake_conn.rollback.assert_called_once()
        fake_conn.close.assert_called_once()

    def test_prepare_strict_frozen_candidate_manifest_requires_raise_policy(self):
        self.handler.frozen_candidate_manifest_mode = "record_strict"
        with self.assertRaises(PitRuntimeError) as ctx:
            self.handler.prepare_strict_frozen_candidate_manifest(
                trading_dates=pd.to_datetime(["2024-01-02", "2024-01-03"]),
                candidate_lookup_error_policy="skip",
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=0.3,
            )
        self.assertEqual(ctx.exception.pit_code, "strict_frozen_requires_raise_policy")

    def test_strict_frozen_manifest_miss_raises(self):
        self.handler._frozen_tier_candidate_manifest = {"2024-01-02": {"candidate_codes": [], "source": "NO_CANDIDATES", "pit_size": 0, "tier1_count": 0, "tier12_count": 0}}
        self.handler._frozen_tier_candidate_manifest_key = self.handler._build_tier_manifest_key(100, 0.3)
        self.handler._strict_frozen_candidate_manifest_required = True

        with self.assertRaises(PitRuntimeError) as ctx:
            self.handler.get_candidates_with_tier_fallback_pit_gated(
                date="2024-01-03",
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=0.3,
            )
        self.assertEqual(ctx.exception.pit_code, "frozen_manifest_miss")

    def test_prepare_replay_strict_loads_manifest_file(self):
        self.handler.frozen_candidate_manifest_mode = "replay_strict"
        manifest_key = list(self.handler._build_tier_manifest_key(100, 0.3))
        payload = {
            "meta": {"manifest_key": manifest_key, "signal_dates": ["2024-01-02"]},
            "manifest": {
                "2024-01-02": {
                    "candidate_codes": ["A"],
                    "source": "TIER_1_SNAPSHOT_ASOF",
                    "pit_size": 3,
                    "tier1_count": 1,
                    "tier12_count": 2,
                }
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            import json
            json.dump(payload, tmp, ensure_ascii=False)
            tmp_path = tmp.name
        self.addCleanup(lambda: os.path.exists(tmp_path) and os.remove(tmp_path))
        self.handler.frozen_candidate_manifest_path = tmp_path
        self.handler.frozen_candidate_manifest_expected_sha256 = self.handler._hash_file(tmp_path)

        summary = self.handler.prepare_strict_frozen_candidate_manifest(
            trading_dates=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            candidate_lookup_error_policy="raise",
            min_liquidity_20d_avg_value=100,
            min_tier12_coverage_ratio=0.3,
        )

        self.assertEqual(summary["mode"], "replay_strict")
        self.assertEqual(summary["signal_dates"], 1)
        self.assertEqual(self.handler._frozen_tier_candidate_manifest["2024-01-02"]["candidate_codes"], ["A"])

    def test_prepare_replay_strict_requires_expected_sha256(self):
        self.handler.frozen_candidate_manifest_mode = "replay_strict"
        self.handler.frozen_candidate_manifest_path = "/tmp/fake-manifest.json"
        self.handler.frozen_candidate_manifest_expected_sha256 = None

        with self.assertRaises(PitRuntimeError) as ctx:
            self.handler.prepare_strict_frozen_candidate_manifest(
                trading_dates=pd.to_datetime(["2024-01-02", "2024-01-03"]),
                candidate_lookup_error_policy="raise",
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=0.3,
            )
        self.assertEqual(ctx.exception.pit_code, "frozen_manifest_expected_sha_missing")

    def test_prepare_replay_strict_rejects_signal_date_mismatch(self):
        self.handler.frozen_candidate_manifest_mode = "replay_strict"
        manifest_key = list(self.handler._build_tier_manifest_key(100, 0.3))
        payload = {
            "meta": {"manifest_key": manifest_key, "signal_dates": ["2024-01-02"]},
            "manifest": {
                "2024-01-02": {
                    "candidate_codes": ["A"],
                    "source": "TIER_1_SNAPSHOT_ASOF",
                    "pit_size": 3,
                    "tier1_count": 1,
                    "tier12_count": 2,
                }
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            import json
            json.dump(payload, tmp, ensure_ascii=False)
            tmp_path = tmp.name
        self.addCleanup(lambda: os.path.exists(tmp_path) and os.remove(tmp_path))
        self.handler.frozen_candidate_manifest_path = tmp_path
        self.handler.frozen_candidate_manifest_expected_sha256 = self.handler._hash_file(tmp_path)

        with self.assertRaises(PitRuntimeError) as ctx:
            self.handler.prepare_strict_frozen_candidate_manifest(
                trading_dates=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                candidate_lookup_error_policy="raise",
                min_liquidity_20d_avg_value=100,
                min_tier12_coverage_ratio=0.3,
            )
        self.assertEqual(ctx.exception.pit_code, "frozen_manifest_signal_date_set_mismatch")

    def test_invalid_frozen_manifest_mode_is_structured(self):
        with self.assertRaises(PitRuntimeError) as ctx:
            DataHandler._resolve_frozen_candidate_manifest_mode(
                {"frozen_candidate_manifest_mode": "broken_mode"}
            )
        self.assertEqual(ctx.exception.pit_code, "frozen_manifest_mode_invalid")

    def test_invalid_frozen_manifest_expected_sha_is_structured(self):
        with self.assertRaises(PitRuntimeError) as ctx:
            DataHandler._resolve_frozen_candidate_manifest_expected_sha256(
                {"frozen_candidate_manifest_expected_sha256": "not-a-sha"}
            )
        self.assertEqual(ctx.exception.pit_code, "frozen_manifest_expected_sha_invalid")


if __name__ == "__main__":
    unittest.main()
