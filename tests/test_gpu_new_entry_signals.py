import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import cupy as cp
    from src.backtest.gpu.logic import _process_new_entry_signals_gpu
    _GPU_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
    cp = None
    _process_new_entry_signals_gpu = None
    _GPU_IMPORT_ERROR = exc


class TestGpuNewEntrySignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _GPU_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"GPU runtime unavailable: {_GPU_IMPORT_ERROR}")

    @staticmethod
    def _params(max_stocks: float):
        return cp.asarray(
            [[max_stocks, 0.02, 0.04, 0.04, 0.0, -0.15, 10.0, 90.0]],
            dtype=cp.float32,
        )

    def test_new_entry_stops_when_cash_falls_below_order_budget(self):
        portfolio_state = cp.asarray([[150.0, 100.0]], dtype=cp.float32)
        positions_state = cp.zeros((1, 3, 3, 3), dtype=cp.float32)
        cooldown_state = cp.full((1, 3), -1, dtype=cp.int32)
        last_trade_day_idx_state = cp.full((1, 3), -1, dtype=cp.int32)

        portfolio_state_after, positions_after, last_trade_after = _process_new_entry_signals_gpu(
            portfolio_state=portfolio_state,
            positions_state=positions_state,
            cooldown_state=cooldown_state,
            last_trade_day_idx_state=last_trade_day_idx_state,
            current_day_idx=1,
            cooldown_period_days=5,
            param_combinations=self._params(max_stocks=3),
            current_prices=cp.asarray([30.0, 50.0, 60.0], dtype=cp.float32),
            signal_close_prices=cp.asarray([31.0, 51.0, 61.0], dtype=cp.float32),
            candidate_tickers_for_day=cp.asarray([0, 1, 2], dtype=cp.int32),
            candidate_atrs_for_day=cp.asarray([0.3, 0.2, 0.1], dtype=cp.float32),
            buy_commission_rate=0.0,
            log_buffer=cp.zeros((16, 5), dtype=cp.float32),
            log_counter=cp.zeros((1,), dtype=cp.int32),
            debug_mode=False,
            all_tickers=["A", "B", "C"],
        )

        self.assertEqual(float(portfolio_state_after[0, 0].item()), 60.0)
        self.assertEqual(float(positions_after[0, 0, 0, 0].item()), 3.0)
        self.assertEqual(float(positions_after[0, 1, 0, 0].item()), 0.0)
        self.assertEqual(float(positions_after[0, 2, 0, 0].item()), 0.0)
        self.assertEqual(int(last_trade_after[0, 0].item()), 1)
        self.assertEqual(int(last_trade_after[0, 1].item()), -1)
        self.assertEqual(int(last_trade_after[0, 2].item()), -1)

    def test_new_entry_skips_holding_and_cooldown_candidates(self):
        portfolio_state = cp.asarray([[300.0, 100.0]], dtype=cp.float32)
        positions_state = cp.zeros((1, 3, 3, 3), dtype=cp.float32)
        positions_state[0, 0, 0, 0] = 1.0
        positions_state[0, 0, 0, 1] = 20.0
        positions_state[0, 0, 0, 2] = 0.0
        cooldown_state = cp.asarray([[-1, 0, -1]], dtype=cp.int32)
        last_trade_day_idx_state = cp.full((1, 3), -1, dtype=cp.int32)

        portfolio_state_after, positions_after, _ = _process_new_entry_signals_gpu(
            portfolio_state=portfolio_state,
            positions_state=positions_state,
            cooldown_state=cooldown_state,
            last_trade_day_idx_state=last_trade_day_idx_state,
            current_day_idx=2,
            cooldown_period_days=5,
            param_combinations=self._params(max_stocks=3),
            current_prices=cp.asarray([20.0, 25.0, 40.0], dtype=cp.float32),
            signal_close_prices=cp.asarray([21.0, 26.0, 41.0], dtype=cp.float32),
            candidate_tickers_for_day=cp.asarray([0, 1, 2], dtype=cp.int32),
            candidate_atrs_for_day=cp.asarray([0.3, 0.2, 0.1], dtype=cp.float32),
            buy_commission_rate=0.0,
            log_buffer=cp.zeros((16, 5), dtype=cp.float32),
            log_counter=cp.zeros((1,), dtype=cp.int32),
            debug_mode=False,
            all_tickers=["A", "B", "C"],
        )

        self.assertEqual(float(positions_after[0, 0, 0, 0].item()), 1.0)
        self.assertEqual(float(positions_after[0, 1, 0, 0].item()), 0.0)
        self.assertEqual(float(positions_after[0, 2, 0, 0].item()), 2.0)
        self.assertEqual(float(portfolio_state_after[0, 0].item()), 220.0)


if __name__ == "__main__":
    unittest.main()
