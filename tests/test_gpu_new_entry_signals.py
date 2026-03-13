import os
import sys
import unittest
import math

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

    @staticmethod
    def _params_matrix(max_stocks_list):
        return cp.asarray(
            [[float(max_stocks), 0.02, 0.04, 0.04, 0.0, -0.15, 10.0, 90.0] for max_stocks in max_stocks_list],
            dtype=cp.float32,
        )

    @staticmethod
    def _python_reference_new_entry(
        *,
        portfolio_state,
        positions_state,
        cooldown_state,
        current_day_idx,
        cooldown_period_days,
        param_combinations,
        current_prices,
        candidate_tickers_for_day,
        buy_commission_rate,
    ):
        portfolio_state = portfolio_state.copy()
        positions_state = positions_state.copy()
        last_trade_day_idx_state = [[-1 for _ in range(len(current_prices))] for _ in range(len(portfolio_state))]

        for sim_idx, (cash, order_budget) in enumerate(portfolio_state):
            held_count = sum(any(split[0] > 0 for split in stock_splits) for stock_splits in positions_state[sim_idx])
            available_slots = max(0, int(float(param_combinations[sim_idx][0])) - held_count)
            if available_slots <= 0:
                continue
            for stock_idx in candidate_tickers_for_day:
                if available_slots <= 0:
                    break
                is_holding = any(split[0] > 0 for split in positions_state[sim_idx][stock_idx])
                cooldown_ref = cooldown_state[sim_idx][stock_idx]
                in_cooldown = cooldown_ref >= 0 and (current_day_idx - cooldown_ref) < cooldown_period_days
                if is_holding or in_cooldown:
                    continue
                if cash < order_budget:
                    break
                exec_price = float(current_prices[stock_idx])
                qty = int(math.floor(order_budget / exec_price)) if exec_price > 0 else 0
                if qty <= 0:
                    continue
                total_cost = float(qty * exec_price) + math.floor(float(qty * exec_price) * float(buy_commission_rate))
                if cash < total_cost:
                    continue
                cash -= total_cost
                available_slots -= 1
                positions_state[sim_idx][stock_idx][0][0] = float(qty)
                positions_state[sim_idx][stock_idx][0][1] = exec_price
                positions_state[sim_idx][stock_idx][0][2] = float(current_day_idx)
                last_trade_day_idx_state[sim_idx][stock_idx] = current_day_idx
            portfolio_state[sim_idx][0] = float(cash)

        return portfolio_state, positions_state, last_trade_day_idx_state

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

    def test_multi_sim_active_set_rerank_matches_python_reference(self):
        portfolio_state = cp.asarray(
            [
                [150.0, 100.0],
                [250.0, 100.0],
                [250.0, 100.0],
            ],
            dtype=cp.float32,
        )
        positions_state = cp.zeros((3, 4, 3, 3), dtype=cp.float32)
        positions_state[1, 0, 0, 0] = 1.0
        positions_state[1, 0, 0, 1] = 20.0
        positions_state[1, 0, 0, 2] = 0.0
        cooldown_state = cp.asarray(
            [
                [-1, -1, -1, -1],
                [-1, 0, -1, -1],
                [-1, -1, -1, -1],
            ],
            dtype=cp.int32,
        )
        last_trade_day_idx_state = cp.full((3, 4), -1, dtype=cp.int32)
        current_prices = cp.asarray([30.0, 40.0, 25.0, 80.0], dtype=cp.float32)
        signal_close_prices = cp.asarray([31.0, 41.0, 26.0, 81.0], dtype=cp.float32)
        candidate_tickers_for_day = cp.asarray([0, 1, 2, 3], dtype=cp.int32)

        portfolio_state_after, positions_after, last_trade_after = _process_new_entry_signals_gpu(
            portfolio_state=portfolio_state.copy(),
            positions_state=positions_state.copy(),
            cooldown_state=cooldown_state.copy(),
            last_trade_day_idx_state=last_trade_day_idx_state.copy(),
            current_day_idx=2,
            cooldown_period_days=5,
            param_combinations=self._params_matrix([3, 3, 1]),
            current_prices=current_prices,
            signal_close_prices=signal_close_prices,
            candidate_tickers_for_day=candidate_tickers_for_day,
            candidate_atrs_for_day=cp.asarray([0.4, 0.3, 0.2, 0.1], dtype=cp.float32),
            buy_commission_rate=0.0,
            log_buffer=cp.zeros((32, 5), dtype=cp.float32),
            log_counter=cp.zeros((1,), dtype=cp.int32),
            debug_mode=False,
            all_tickers=["A", "B", "C", "D"],
        )

        expected_portfolio, expected_positions, expected_last_trade = self._python_reference_new_entry(
            portfolio_state=portfolio_state.get().tolist(),
            positions_state=positions_state.get().tolist(),
            cooldown_state=cooldown_state.get().tolist(),
            current_day_idx=2,
            cooldown_period_days=5,
            param_combinations=self._params_matrix([3, 3, 1]).get().tolist(),
            current_prices=current_prices.get().tolist(),
            candidate_tickers_for_day=candidate_tickers_for_day.get().tolist(),
            buy_commission_rate=0.0,
        )

        self.assertEqual(
            [round(float(value), 6) for value in portfolio_state_after[:, 0].get().tolist()],
            [round(float(row[0]), 6) for row in expected_portfolio],
        )
        self.assertEqual(
            positions_after[:, :, 0, 0].get().tolist(),
            [[stock[0][0] for stock in sim] for sim in expected_positions],
        )
        self.assertEqual(
            last_trade_after.get().tolist(),
            expected_last_trade,
        )
        self.assertEqual(
            positions_after[:, :, 0, 0].get().tolist(),
            [
                [3.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 4.0, 1.0],
                [3.0, 0.0, 0.0, 0.0],
            ],
        )

    def test_active_sim_pruning_keeps_cpu_parity_for_budget_and_slot_exhaustion(self):
        portfolio_state = cp.asarray(
            [
                [150.0, 100.0],
                [300.0, 100.0],
                [300.0, 100.0],
                [300.0, 100.0],
            ],
            dtype=cp.float32,
        )
        positions_state = cp.zeros((4, 3, 3, 3), dtype=cp.float32)
        positions_state[2, 0, 0, 0] = 1.0
        positions_state[2, 0, 0, 1] = 20.0
        positions_state[2, 0, 0, 2] = 0.0
        cooldown_state = cp.asarray(
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [0, -1, -1],
            ],
            dtype=cp.int32,
        )
        last_trade_day_idx_state = cp.full((4, 3), -1, dtype=cp.int32)
        current_prices = cp.asarray([30.0, 40.0, 50.0], dtype=cp.float32)
        signal_close_prices = cp.asarray([31.0, 41.0, 51.0], dtype=cp.float32)
        candidate_tickers_for_day = cp.asarray([0, 1, 2], dtype=cp.int32)
        param_combinations = self._params_matrix([3, 3, 1, 1])

        portfolio_state_after, positions_after, last_trade_after = _process_new_entry_signals_gpu(
            portfolio_state=portfolio_state.copy(),
            positions_state=positions_state.copy(),
            cooldown_state=cooldown_state.copy(),
            last_trade_day_idx_state=last_trade_day_idx_state.copy(),
            current_day_idx=2,
            cooldown_period_days=5,
            param_combinations=param_combinations,
            current_prices=current_prices,
            signal_close_prices=signal_close_prices,
            candidate_tickers_for_day=candidate_tickers_for_day,
            candidate_atrs_for_day=cp.asarray([0.4, 0.3, 0.2], dtype=cp.float32),
            buy_commission_rate=0.0,
            log_buffer=cp.zeros((32, 5), dtype=cp.float32),
            log_counter=cp.zeros((1,), dtype=cp.int32),
            debug_mode=False,
            all_tickers=["A", "B", "C"],
        )

        expected_portfolio, expected_positions, expected_last_trade = self._python_reference_new_entry(
            portfolio_state=portfolio_state.get().tolist(),
            positions_state=positions_state.get().tolist(),
            cooldown_state=cooldown_state.get().tolist(),
            current_day_idx=2,
            cooldown_period_days=5,
            param_combinations=param_combinations.get().tolist(),
            current_prices=current_prices.get().tolist(),
            candidate_tickers_for_day=candidate_tickers_for_day.get().tolist(),
            buy_commission_rate=0.0,
        )

        self.assertEqual(
            [round(float(value), 6) for value in portfolio_state_after[:, 0].get().tolist()],
            [round(float(row[0]), 6) for row in expected_portfolio],
        )
        self.assertEqual(
            positions_after[:, :, 0, 0].get().tolist(),
            [[stock[0][0] for stock in sim] for sim in expected_positions],
        )
        self.assertEqual(last_trade_after.get().tolist(), expected_last_trade)
        self.assertEqual(
            positions_after[:, :, 0, 0].get().tolist(),
            [
                [3.0, 0.0, 0.0],
                [3.0, 2.0, 2.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
        )

    def test_new_entry_skips_unaffordable_commission_candidate_and_keeps_scanning(self):
        portfolio_state = cp.asarray([[105.0, 100.0]], dtype=cp.float32)
        positions_state = cp.zeros((1, 2, 3, 3), dtype=cp.float32)
        cooldown_state = cp.full((1, 2), -1, dtype=cp.int32)
        last_trade_day_idx_state = cp.full((1, 2), -1, dtype=cp.int32)
        current_prices = cp.asarray([100.0, 40.0], dtype=cp.float32)
        signal_close_prices = cp.asarray([101.0, 41.0], dtype=cp.float32)
        candidate_tickers_for_day = cp.asarray([0, 1], dtype=cp.int32)

        portfolio_state_after, positions_after, last_trade_after = _process_new_entry_signals_gpu(
            portfolio_state=portfolio_state.copy(),
            positions_state=positions_state.copy(),
            cooldown_state=cooldown_state.copy(),
            last_trade_day_idx_state=last_trade_day_idx_state.copy(),
            current_day_idx=3,
            cooldown_period_days=5,
            param_combinations=self._params(max_stocks=2),
            current_prices=current_prices,
            signal_close_prices=signal_close_prices,
            candidate_tickers_for_day=candidate_tickers_for_day,
            candidate_atrs_for_day=cp.asarray([0.4, 0.3], dtype=cp.float32),
            buy_commission_rate=0.10,
            log_buffer=cp.zeros((16, 5), dtype=cp.float32),
            log_counter=cp.zeros((1,), dtype=cp.int32),
            debug_mode=False,
            all_tickers=["A", "B"],
        )

        expected_portfolio, expected_positions, expected_last_trade = self._python_reference_new_entry(
            portfolio_state=portfolio_state.get().tolist(),
            positions_state=positions_state.get().tolist(),
            cooldown_state=cooldown_state.get().tolist(),
            current_day_idx=3,
            cooldown_period_days=5,
            param_combinations=self._params(max_stocks=2).get().tolist(),
            current_prices=current_prices.get().tolist(),
            candidate_tickers_for_day=candidate_tickers_for_day.get().tolist(),
            buy_commission_rate=0.10,
        )

        self.assertEqual(
            [round(float(value), 6) for value in portfolio_state_after[:, 0].get().tolist()],
            [round(float(row[0]), 6) for row in expected_portfolio],
        )
        self.assertEqual(
            positions_after[:, :, 0, 0].get().tolist(),
            [[stock[0][0] for stock in sim] for sim in expected_positions],
        )
        self.assertEqual(last_trade_after.get().tolist(), expected_last_trade)
        self.assertEqual(float(positions_after[0, 0, 0, 0].item()), 0.0)
        self.assertEqual(float(positions_after[0, 1, 0, 0].item()), 2.0)
        self.assertEqual(float(portfolio_state_after[0, 0].item()), 17.0)

    def test_new_entry_matches_reference_with_heterogeneous_budget_commission_and_active_shrink(self):
        portfolio_state = cp.asarray(
            [
                [150.0, 100.0],
                [230.0, 120.0],
                [340.0, 100.0],
            ],
            dtype=cp.float32,
        )
        positions_state = cp.zeros((3, 3, 3, 3), dtype=cp.float32)
        cooldown_state = cp.full((3, 3), -1, dtype=cp.int32)
        last_trade_day_idx_state = cp.full((3, 3), -1, dtype=cp.int32)
        current_prices = cp.asarray([95.0, 40.0, 30.0], dtype=cp.float32)
        signal_close_prices = cp.asarray([96.0, 41.0, 31.0], dtype=cp.float32)
        candidate_tickers_for_day = cp.asarray([0, 1, 2], dtype=cp.int32)
        param_combinations = self._params_matrix([3, 3, 3])

        portfolio_state_after, positions_after, last_trade_after = _process_new_entry_signals_gpu(
            portfolio_state=portfolio_state.copy(),
            positions_state=positions_state.copy(),
            cooldown_state=cooldown_state.copy(),
            last_trade_day_idx_state=last_trade_day_idx_state.copy(),
            current_day_idx=4,
            cooldown_period_days=5,
            param_combinations=param_combinations,
            current_prices=current_prices,
            signal_close_prices=signal_close_prices,
            candidate_tickers_for_day=candidate_tickers_for_day,
            candidate_atrs_for_day=cp.asarray([0.4, 0.3, 0.2], dtype=cp.float32),
            buy_commission_rate=0.10,
            log_buffer=cp.zeros((32, 5), dtype=cp.float32),
            log_counter=cp.zeros((1,), dtype=cp.int32),
            debug_mode=False,
            all_tickers=["A", "B", "C"],
        )

        expected_portfolio, expected_positions, expected_last_trade = self._python_reference_new_entry(
            portfolio_state=portfolio_state.get().tolist(),
            positions_state=positions_state.get().tolist(),
            cooldown_state=cooldown_state.get().tolist(),
            current_day_idx=4,
            cooldown_period_days=5,
            param_combinations=param_combinations.get().tolist(),
            current_prices=current_prices.get().tolist(),
            candidate_tickers_for_day=candidate_tickers_for_day.get().tolist(),
            buy_commission_rate=0.10,
        )

        self.assertEqual(
            [round(float(value), 6) for value in portfolio_state_after[:, 0].get().tolist()],
            [round(float(row[0]), 6) for row in expected_portfolio],
        )
        self.assertEqual(
            positions_after[:, :, 0, 0].get().tolist(),
            [[stock[0][0] for stock in sim] for sim in expected_positions],
        )
        self.assertEqual(last_trade_after.get().tolist(), expected_last_trade)
        self.assertEqual(
            positions_after[:, :, 0, 0].get().tolist(),
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
            ],
        )

    def test_new_entry_matches_reference_with_heterogeneous_budget_commission_and_active_set_shrink(self):
        portfolio_state = cp.asarray(
            [
                [215.0, 100.0],
                [105.0, 100.0],
                [220.0, 120.0],
            ],
            dtype=cp.float32,
        )
        positions_state = cp.zeros((3, 3, 3, 3), dtype=cp.float32)
        cooldown_state = cp.full((3, 3), -1, dtype=cp.int32)
        last_trade_day_idx_state = cp.full((3, 3), -1, dtype=cp.int32)
        current_prices = cp.asarray([100.0, 40.0, 30.0], dtype=cp.float32)
        signal_close_prices = cp.asarray([101.0, 41.0, 31.0], dtype=cp.float32)
        candidate_tickers_for_day = cp.asarray([0, 1, 2], dtype=cp.int32)
        param_combinations = self._params_matrix([2, 2, 2])

        portfolio_state_after, positions_after, last_trade_after = _process_new_entry_signals_gpu(
            portfolio_state=portfolio_state.copy(),
            positions_state=positions_state.copy(),
            cooldown_state=cooldown_state.copy(),
            last_trade_day_idx_state=last_trade_day_idx_state.copy(),
            current_day_idx=4,
            cooldown_period_days=5,
            param_combinations=param_combinations,
            current_prices=current_prices,
            signal_close_prices=signal_close_prices,
            candidate_tickers_for_day=candidate_tickers_for_day,
            candidate_atrs_for_day=cp.asarray([0.5, 0.4, 0.3], dtype=cp.float32),
            buy_commission_rate=0.10,
            log_buffer=cp.zeros((32, 5), dtype=cp.float32),
            log_counter=cp.zeros((1,), dtype=cp.int32),
            debug_mode=False,
            all_tickers=["A", "B", "C"],
        )

        expected_portfolio, expected_positions, expected_last_trade = self._python_reference_new_entry(
            portfolio_state=portfolio_state.get().tolist(),
            positions_state=positions_state.get().tolist(),
            cooldown_state=cooldown_state.get().tolist(),
            current_day_idx=4,
            cooldown_period_days=5,
            param_combinations=param_combinations.get().tolist(),
            current_prices=current_prices.get().tolist(),
            candidate_tickers_for_day=candidate_tickers_for_day.get().tolist(),
            buy_commission_rate=0.10,
        )

        self.assertEqual(
            [round(float(value), 6) for value in portfolio_state_after[:, 0].get().tolist()],
            [round(float(row[0]), 6) for row in expected_portfolio],
        )
        self.assertEqual(
            positions_after[:, :, 0, 0].get().tolist(),
            [[stock[0][0] for stock in sim] for sim in expected_positions],
        )
        self.assertEqual(last_trade_after.get().tolist(), expected_last_trade)
        self.assertEqual(
            positions_after[:, :, 0, 0].get().tolist(),
            [
                [1.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        )
        self.assertEqual(
            [round(float(value), 6) for value in portfolio_state_after[:, 0].get().tolist()],
            [17.0, 17.0, 110.0],
        )


if __name__ == "__main__":
    unittest.main()
