import unittest
import pandas as pd
from unittest.mock import MagicMock

from src.backtest.cpu.backtester import BacktestEngine


class _DummyPortfolio:
    def __init__(self):
        self.cash = 1_000_000.0
        self.positions = {}
        self.trade_history = []
        self.daily_snapshot_history = []

    def get_total_value(self, _current_date, _data_handler):
        return self.cash

    def record_daily_snapshot(self, current_date, total_value):
        self.daily_snapshot_history.append({"date": pd.to_datetime(current_date), "total_value": total_value})

    def get_positions_snapshot(self, _current_date, _data_handler, _total_value):
        return pd.DataFrame()


class _DummyDataHandler:
    def __init__(self, trading_dates):
        self._trading_dates = pd.DatetimeIndex(trading_dates)
        self.clear_lazy_tier_candidate_cache = MagicMock()
        self.clear_runtime_lookup_cache = MagicMock()
        self.clear_frozen_tier_candidate_manifest = MagicMock()
        self.freeze_tier_candidate_manifest = MagicMock()
        self.prepare_strict_frozen_candidate_manifest = MagicMock(return_value={})
        self.get_candidates_with_tier_fallback_pit_gated = MagicMock()

    def get_trading_dates(self, _start_date, _end_date):
        return self._trading_dates


class _DummyExecutionHandler:
    def execute_order(self, *_args, **_kwargs):
        return None


class _ScriptedStrategy:
    def __init__(self, contexts):
        self.max_stocks = 5
        self.candidate_source_mode = "tier"
        self.candidate_lookup_error_policy = "raise"
        self.min_liquidity_20d_avg_value = 123
        self.min_tier12_coverage_ratio = 0.45
        self._contexts = list(contexts)
        self._idx = 0
        self.last_entry_context = {}

    def generate_sell_signals(self, *_args, **_kwargs):
        return []

    def generate_new_entry_signals(self, *_args, **_kwargs):
        self.last_entry_context = dict(self._contexts[self._idx])
        self._idx += 1
        return []

    def generate_additional_buy_signals(self, *_args, **_kwargs):
        return []


class TestBacktesterEntryMetrics(unittest.TestCase):
    def test_run_prepares_strict_manifest_and_clears_lazy_cache_once(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        data_handler = _DummyDataHandler(dates)
        strategy = _ScriptedStrategy([{}, {}])
        engine = BacktestEngine(
            start_date=dates[0],
            end_date=dates[-1],
            portfolio=_DummyPortfolio(),
            strategy=strategy,
            data_handler=data_handler,
            execution_handler=_DummyExecutionHandler(),
        )

        engine.run()

        data_handler.clear_lazy_tier_candidate_cache.assert_called_once()
        self.assertEqual(data_handler.clear_runtime_lookup_cache.call_count, len(dates))
        data_handler.prepare_strict_frozen_candidate_manifest.assert_called_once()
        call_args, call_kwargs = data_handler.prepare_strict_frozen_candidate_manifest.call_args
        self.assertTrue(call_args[0].equals(dates))
        self.assertEqual(call_kwargs["candidate_lookup_error_policy"], "raise")
        self.assertEqual(call_kwargs["min_liquidity_20d_avg_value"], 123)
        self.assertEqual(call_kwargs["min_tier12_coverage_ratio"], 0.45)
        data_handler.clear_frozen_tier_candidate_manifest.assert_not_called()
        data_handler.freeze_tier_candidate_manifest.assert_not_called()

    def test_entry_metrics_tracks_source_breakdown(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        strategy = _ScriptedStrategy(
            [
                {
                    "tier_source": "TIER_1_SNAPSHOT_ASOF",
                    "raw_candidate_count": 9,
                    "active_candidate_count": 7,
                    "ranked_candidate_count": 6,
                    "selected_count": 0,
                },
                {
                    "tier_source": "NO_CANDIDATES_SNAPSHOT_ASOF",
                    "raw_candidate_count": 0,
                    "active_candidate_count": 0,
                    "ranked_candidate_count": 0,
                    "selected_count": 0,
                },
                {
                    "tier_source": "CANDIDATE_LOOKUP_ERROR",
                    "raw_candidate_count": 0,
                    "active_candidate_count": 0,
                    "ranked_candidate_count": 0,
                    "selected_count": 0,
                    "pit_failure_code": "tier12_coverage_gate_failed",
                    "pit_failure_stage": "tier12_coverage_gate",
                },
            ]
        )
        engine = BacktestEngine(
            start_date=dates[0],
            end_date=dates[-1],
            portfolio=_DummyPortfolio(),
            strategy=strategy,
            data_handler=_DummyDataHandler(dates),
            execution_handler=_DummyExecutionHandler(),
        )

        portfolio = engine.run()
        metrics = getattr(portfolio, "run_metrics", {})

        self.assertEqual(metrics["entry_opportunity_days"], 3)
        self.assertEqual(metrics["candidate_eval_days"], 2)
        self.assertEqual(metrics["empty_entry_days"], 3)
        self.assertAlmostEqual(metrics["empty_entry_day_rate"], 1.0, places=4)
        self.assertAlmostEqual(metrics["tier1_coverage"], 0.3333, places=4)
        self.assertAlmostEqual(metrics["tier2_fallback_rate"], 0.0, places=4)
        self.assertAlmostEqual(metrics["no_candidates_rate"], 0.3333, places=4)
        self.assertAlmostEqual(metrics["source_lookup_error_rate"], 0.3333, places=4)
        self.assertAlmostEqual(metrics["no_signal_date_rate"], 0.0, places=4)
        self.assertAlmostEqual(metrics["source_unknown_rate"], 0.0, places=4)
        self.assertAlmostEqual(metrics["avg_raw_candidates"], 4.5, places=2)
        self.assertAlmostEqual(metrics["avg_active_candidates"], 3.5, places=2)
        self.assertAlmostEqual(metrics["avg_ranked_candidates"], 3.0, places=2)
        self.assertAlmostEqual(metrics["avg_selected_signals"], 0.0, places=2)
        self.assertEqual(metrics["pit_failure_days_by_code"], {"tier12_coverage_gate_failed": 1})
        self.assertEqual(metrics["pit_failure_days_by_stage"], {"tier12_coverage_gate": 1})

    def test_entry_metrics_counts_unknown_source(self):
        dates = pd.to_datetime(["2024-01-02"])
        strategy = _ScriptedStrategy([{}])
        engine = BacktestEngine(
            start_date=dates[0],
            end_date=dates[-1],
            portfolio=_DummyPortfolio(),
            strategy=strategy,
            data_handler=_DummyDataHandler(dates),
            execution_handler=_DummyExecutionHandler(),
        )

        portfolio = engine.run()
        metrics = getattr(portfolio, "run_metrics", {})
        self.assertEqual(metrics["entry_opportunity_days"], 1)
        self.assertEqual(metrics["candidate_eval_days"], 0)
        self.assertAlmostEqual(metrics["source_unknown_rate"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
