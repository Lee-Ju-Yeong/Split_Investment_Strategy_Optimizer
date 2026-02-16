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
from src.backtest.gpu.logic import _calculate_monthly_investment_gpu

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


if __name__ == '__main__':
    unittest.main()
