"""
test_backtest_strategy_gpu.py

Unit tests for the GPU-accelerated backtesting logic to ensure its results
are consistent with the original CPU-based implementation.
"""
import unittest
import numpy as np
import cupy as cp
import pandas as pd

# Import the functions to be tested
from src.backtest.gpu.logic import (
    _calculate_monthly_investment_gpu,
    _process_additional_buy_signals_gpu,
    _process_sell_signals_gpu,
)

# A mock Position class to simulate the structure of the original data
class MockPosition:
    def __init__(self, quantity):
        self.quantity = quantity

def convert_cpu_data_to_gpu(positions_dict, prices_dict, all_codes, max_splits):
    """
    Converts CPU-style data structures (dicts of objects) into GPU-friendly
    CuPy arrays.
    """
    num_stocks = len(all_codes)
    quantities_np = np.zeros((num_stocks, max_splits), dtype=np.float32)
    prices_np = np.zeros(num_stocks, dtype=np.float32)

    code_to_idx = {code: i for i, code in enumerate(all_codes)}

    for code, positions in positions_dict.items():
        stock_idx = code_to_idx[code]
        for i, pos in enumerate(positions):
            if i < max_splits:
                quantities_np[stock_idx, i] = pos.quantity
    
    for code, price in prices_dict.items():
        stock_idx = code_to_idx[code]
        prices_np[stock_idx] = price

    # Transfer NumPy arrays from CPU memory to CuPy arrays in GPU memory
    quantities_gpu = cp.asarray(quantities_np)
    prices_gpu = cp.asarray(prices_np)
    
    return quantities_gpu, prices_gpu

class TestBacktestStrategyGPU(unittest.TestCase):

    def setUp(self):
        """Set up common data for tests."""
        self.all_tickers = ['005930', '373220', '000660', '005380']
        self.max_stocks = len(self.all_tickers)
        self.max_splits_limit = 5

        # --- Mock Data for Monthly Investment Test ---
        self.param_combinations = cp.array([
            # max_stocks, order_inv_ratio, add_buy_drop, sell_profit, add_buy_prio
            [10, 0.02, 0.05, 0.10, 0],  # Sim 0
            [20, 0.03, 0.04, 0.12, 1],  # Sim 1
        ], dtype=cp.float32)
        
        self.num_combinations = self.param_combinations.shape[0]

        # state: [capital, investment_per_order]
        self.portfolio_state = cp.zeros((self.num_combinations, 2), dtype=cp.float32)
        self.portfolio_state[:, 0] = cp.array([1000000, 2000000]) # Initial capital

        # state: [quantity, buy_price]
        self.positions_state = cp.zeros((self.num_combinations, self.max_stocks, self.max_splits_limit, 2), dtype=cp.float32)
        # Sim 0, stock 0 ('005930'), 1st split
        self.positions_state[0, 0, 0, 0] = 10  # quantity
        # Sim 1, stock 1 ('373220'), 1st split
        self.positions_state[1, 1, 0, 0] = 5   # quantity
        # Sim 1, stock 2 ('000660'), 1st split
        self.positions_state[1, 2, 0, 0] = 20  # quantity
        
        self.current_date = pd.to_datetime('2023-01-10')
        
    def test_calculate_portfolio_value_gpu(self):
        # ... (previous test remains here) ...
        pass

    def test_calculate_monthly_investment_gpu(self):
        """
        Tests if the vectorized monthly investment calculation is correct.
        """
        # 1. Manually calculate the expected result with CPU logic
        
        # --- Sim 0 Expected ---
        sim0_capital = 1000000
        sim0_stock_value = 10 * 71000 # 10 shares of '005930' at 71000
        sim0_total_value = sim0_capital + sim0_stock_value
        sim0_inv_ratio = 0.02
        expected_inv_per_order_0 = sim0_total_value * sim0_inv_ratio
        
        # --- Sim 1 Expected ---
        sim1_capital = 2000000
        sim1_stock_value = (5 * 450000) + (20 * 130000) # LG + Hynix
        sim1_total_value = sim1_capital + sim1_stock_value
        sim1_inv_ratio = 0.03
        expected_inv_per_order_1 = sim1_total_value * sim1_inv_ratio

        expected_results = cp.array([
            [sim0_capital, expected_inv_per_order_0],
            [sim1_capital, expected_inv_per_order_1]
        ])
        evaluation_prices = cp.array([71000, 450000, 130000, 200000], dtype=cp.float32)

        # 2. Run the actual GPU function
        updated_portfolio_state = _calculate_monthly_investment_gpu(
            self.portfolio_state,
            self.positions_state,
            self.param_combinations,
            evaluation_prices,
            self.current_date,
            False,
        )

        # 3. Assert that the results are close
        print(f"\\nExpected State (CPU):\\n{expected_results.get()}")
        print(f"Actual State (GPU):\\n{updated_portfolio_state.get()}")
        
        self.assertTrue(
            cp.allclose(expected_results, updated_portfolio_state),
            "GPU monthly investment calculation does not match CPU result."
        )


class TestIssue56TierSignalExecutionParity(unittest.TestCase):
    def _single_param_row(self, *, add_drop=0.05, sell_profit=0.10):
        return cp.array(
            [[10, 0.02, add_drop, sell_profit, 1, -0.50, 10, 999]],
            dtype=cp.float32,
        )

    def test_sell_uses_tminus1_signal_and_executes_at_t0_open(self):
        """
        Regression guard:
        - T-1 high crosses profit target (signal generated)
        - execution policy is open-market, so fill occurs at T0 open
        """
        portfolio_state = cp.array([[1_000_000.0, 100_000.0]], dtype=cp.float32)
        positions_state = cp.zeros((1, 1, 3, 3), dtype=cp.float32)
        positions_state[0, 0, 0, 0] = 10.0   # qty
        positions_state[0, 0, 0, 1] = 100.0  # buy_price
        positions_state[0, 0, 0, 2] = 0.0    # open_day_idx

        cooldown_state = cp.full((1, 1), -1, dtype=cp.int32)
        last_trade_day_idx_state = cp.array([[0]], dtype=cp.int32)
        params = self._single_param_row(sell_profit=0.10)

        portfolio_state_after, positions_after, _, _, sell_mask = _process_sell_signals_gpu(
            portfolio_state=portfolio_state,
            positions_state=positions_state,
            cooldown_state=cooldown_state,
            last_trade_day_idx_state=last_trade_day_idx_state,
            current_day_idx=1,
            param_combinations=params,
            current_open_prices=cp.array([100.0], dtype=cp.float32),
            current_close_prices=cp.array([100.0], dtype=cp.float32),
            current_high_prices=cp.array([105.0], dtype=cp.float32),
            signal_close_prices=cp.array([100.0], dtype=cp.float32),
            signal_high_prices=cp.array([115.0], dtype=cp.float32),    # T-1 high >= target
            signal_day_idx=0,
            sell_commission_rate=0.00015,
            sell_tax_rate=0.0018,
            debug_mode=False,
            all_tickers=["TEST"],
            trading_dates_pd_cpu=pd.DatetimeIndex(["2021-01-05", "2021-01-06"]),
        )

        self.assertGreater(float(portfolio_state_after[0, 0].item()), 1_000_000.0)
        self.assertEqual(float(positions_after[0, 0, 0, 0].item()), 0.0)
        self.assertTrue(bool(sell_mask[0, 0].item()))

    def test_additional_buy_uses_tminus1_low_for_trigger(self):
        """
        Regression guard:
        - trigger = 95 from last buy=100 and drop=5%
        - T-1 low=100 (no signal), even if intraday(T0) low were lower in reality
        """
        portfolio_state = cp.array([[1_000_000.0, 100_000.0]], dtype=cp.float32)
        positions_state = cp.zeros((1, 1, 3, 3), dtype=cp.float32)
        positions_state[0, 0, 0, 0] = 10.0
        positions_state[0, 0, 0, 1] = 100.0
        positions_state[0, 0, 0, 2] = 0.0

        last_trade_day_idx_state = cp.array([[0]], dtype=cp.int32)
        sell_occurred_today_mask = cp.zeros((1, 1), dtype=cp.bool_)
        params = self._single_param_row(add_drop=0.05)

        portfolio_state_after, positions_after, _ = _process_additional_buy_signals_gpu(
            portfolio_state=portfolio_state,
            positions_state=positions_state,
            last_trade_day_idx_state=last_trade_day_idx_state,
            sell_occurred_today_mask=sell_occurred_today_mask,
            current_day_idx=1,
            param_combinations=params,
            current_opens=cp.array([95.0], dtype=cp.float32),
            signal_close_prices=cp.array([100.0], dtype=cp.float32),
            signal_lows=cp.array([100.0], dtype=cp.float32),  # no trigger at T-1
            signal_day_idx=0,
            buy_commission_rate=0.00015,
            log_buffer=cp.zeros((1, 1), dtype=cp.float32),
            log_counter=cp.zeros((1,), dtype=cp.int32),
            debug_mode=False,
            all_tickers=["TEST"],
        )

        self.assertEqual(float(portfolio_state_after[0, 0].item()), 1_000_000.0)
        # split index 1 should remain empty (no additional buy)
        self.assertEqual(float(positions_after[0, 0, 1, 0].item()), 0.0)


if __name__ == '__main__':
    unittest.main()
