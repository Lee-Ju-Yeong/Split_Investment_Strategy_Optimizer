import unittest
import pandas as pd

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

    def get_trading_dates(self, _start_date, _end_date):
        return self._trading_dates


class _DummyExecutionHandler:
    def execute_order(self, *_args, **_kwargs):
        return None


class _ScriptedStrategy:
    def __init__(self, contexts):
        self.max_stocks = 5
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
