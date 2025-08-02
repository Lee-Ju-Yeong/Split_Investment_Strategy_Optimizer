"""
backtest_strategy_gpu.py

This module contains the GPU-accelerated versions of backtesting logic 
using CuPy and Numba for massive parallelization.
"""

import cupy as cp
import cudf
import time
import pandas as pd
from sqlalchemy import create_engine

# This function was accidentally removed, re-adding for the unit test.
def calculate_portfolio_value_gpu(capital, quantities, prices):
    """Calculates the total portfolio value for a given date on the GPU."""
    prices_col = prices.reshape(-1, 1)
    position_values = quantities * prices_col
    total_stock_value = cp.sum(position_values)
    total_portfolio_value = capital + total_stock_value
    return total_portfolio_value.get()

# -----------------------------------------------------------------------------
# GPU Backtesting Kernel
# -----------------------------------------------------------------------------

def _calculate_monthly_investment_gpu(
    current_date,
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    param_combinations: cp.ndarray,
    all_data_gpu: cudf.DataFrame,
    all_tickers: list # The ordered list of tickers corresponding to the second dimension of positions_state
):
    """
    Vectorized calculation of monthly investment amounts for all simulations.
    """
    # 1. Get current prices for all stocks
    # Reset index to perform boolean masking on the 'date' column directly
    data_for_lookup = all_data_gpu.reset_index()
    # Filter data up to the current date
    filtered_data = data_for_lookup[data_for_lookup['date'] <= current_date]
    # Get the last available price for each ticker
    latest_prices = filtered_data.groupby('ticker')['close_price'].last()
    
    # Reindex to ensure the price series matches the exact order and size of all_tickers,
    # filling any missing prices with 0 (for stocks with no data on that day).
    price_series = latest_prices.reindex(all_tickers).fillna(0)
    
    # Convert the final price series to a CuPy array.
    prices_gpu = cp.asarray(price_series.values, dtype=cp.float32)
    
    # 2. Calculate current stock value for all simulations
    quantities = positions_state[..., 0] # Shape: (num_combinations, num_stocks, max_splits)
    
    # Reshape prices for broadcasting: (1, num_stocks, 1)
    prices_reshaped = prices_gpu.reshape(1, -1, 1)
    
    # stock_values shape: (num_combinations, num_stocks)
    stock_values = cp.sum(quantities * prices_reshaped, axis=2)
    total_stock_values = cp.sum(stock_values, axis=1, keepdims=True)

    # 3. Calculate total portfolio value
    capital_array = portfolio_state[:, 0:1]
    total_portfolio_values = capital_array + total_stock_values

    # 4. Update investment_per_order in portfolio_state
    order_investment_ratios = param_combinations[:, 1:2]
    investment_per_order = total_portfolio_values * order_investment_ratios
    
    portfolio_state[:, 1:2] = investment_per_order
    
    return portfolio_state


def run_magic_split_strategy_on_gpu(
    initial_capital: float,
    param_combinations: cp.ndarray,
    all_data_gpu: cudf.DataFrame,
    trading_dates: cp.ndarray,
    all_tickers: list,
    max_splits_limit: int = 20
):
    """
    Main GPU-accelerated backtesting function for the MagicSplitStrategy.
    """
    print("🚀 Initializing GPU backtesting environment...")
    num_combinations = param_combinations.shape[0]
    num_trading_days = len(trading_dates)
    
    # --- 1. State Management Arrays ---
    # Portfolio-level state: [0:capital, 1:investment_per_order]
    portfolio_state = cp.zeros((num_combinations, 2), dtype=cp.float32)
    portfolio_state[:, 0] = initial_capital

    # Position-level state: [0: quantity, 1: buy_price]
    max_stocks_param = int(cp.max(param_combinations[:, 0]).get()) # Get max_stocks from user parameters
    num_tickers = len(all_tickers)
    
    # The actual dimension used for arrays must match the full list of tickers
    positions_state = cp.zeros((num_combinations, num_tickers, max_splits_limit, 2), dtype=cp.float32)
    
    daily_portfolio_values = cp.zeros((num_combinations, num_trading_days), dtype=cp.float32)

    print(f"    - State arrays created. Portfolio State Shape: {portfolio_state.shape}")
    print(f"    - Positions State Array Shape: {positions_state.shape}")

    # --- 2. Main Simulation Loop (Vectorized) ---
    previous_month = -1
    for i, current_date_np in enumerate(trading_dates):
        current_date = pd.to_datetime(current_date_np) # Convert numpy.datetime64 to pd.Timestamp
        current_month = current_date.month
        
        if current_month != previous_month:
            print(f"    - Rebalancing for month: {current_month}...")
            portfolio_state = _calculate_monthly_investment_gpu(
                current_date, portfolio_state, positions_state, param_combinations, all_data_gpu, all_tickers
            )
            previous_month = current_month
        
        if (i + 1) % 252 == 0:
            year = current_date.year
            print(f"    - Simulating year: {year} ({i+1}/{num_trading_days})")

    print("🎉 GPU backtesting simulation finished.")
    
    return daily_portfolio_values
