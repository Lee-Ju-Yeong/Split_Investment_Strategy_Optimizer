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


def _process_sell_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray
):
    """
    Vectorized sell signal processing for all simulations.
    
    This function implements the MagicSplitStrategy sell logic:
    1. Full liquidation: If 1st position profit >= sell_profit_rate, sell all positions for that stock
    2. Partial liquidation: If 2nd+ position profit >= sell_profit_rate, sell only that position
    
    Args:
        portfolio_state: (num_combinations, 2) [capital, investment_per_order]
        positions_state: (num_combinations, num_stocks, max_splits, 2) [quantity, buy_price]
        param_combinations: (num_combinations, 5) [max_stocks, order_inv_ratio, add_buy_drop, sell_profit, add_buy_prio]
        current_prices: (num_stocks,) current market prices for all stocks
    
    Returns:
        Updated portfolio_state and positions_state after sell executions
    """
    num_combinations, num_stocks, max_splits, _ = positions_state.shape
    
    # Extract sell profit rates for each simulation: shape (num_combinations, 1, 1)
    sell_profit_rates = param_combinations[:, 3:4].reshape(-1, 1, 1)
    
    # Get quantities and buy prices: shape (num_combinations, num_stocks, max_splits)
    quantities = positions_state[..., 0]
    buy_prices = positions_state[..., 1]
    
    # Reshape current prices for broadcasting: (1, num_stocks, 1)
    current_prices_reshaped = current_prices.reshape(1, -1, 1)
    
    # Calculate current profit rates for all positions
    # Avoid division by zero: only calculate for positions with buy_price > 0
    valid_positions = buy_prices > 0
    profit_rates = cp.zeros_like(buy_prices)
    
    # --- 💡 수정된 부분 시작 💡 ---
    # `current_prices_reshaped`를 `buy_prices`와 동일한 shape으로 브로드캐스팅합니다.
    # (1, 357, 1) shape이 (1000, 357, 20) shape에 맞게 확장됩니다.
    # 이렇게 하면 boolean 인덱싱 전에 shape이 일치하게 됩니다.
    broadcasted_prices = cp.broadcast_to(current_prices_reshaped, buy_prices.shape)
    
    # 이제 broadcasted_prices를 사용하여 계산합니다.
    profit_rates[valid_positions] = (broadcasted_prices[valid_positions] - buy_prices[valid_positions]) / buy_prices[valid_positions]
    # --- 💡 수정된 부분 끝 💡 ---
    
    # --- Step 1: Check for full liquidation conditions ---
    # Get 1st positions (order=1): shape (num_combinations, num_stocks)
    first_positions_profit = profit_rates[:, :, 0]  # First split (index 0)
    first_positions_valid = valid_positions[:, :, 0]
    
    # Full liquidation condition: 1st position profit >= sell_profit_rate
    full_liquidation_mask = (first_positions_profit >= sell_profit_rates.squeeze(-1)) & first_positions_valid
    
    # --- Step 2: Check for partial liquidation conditions ---
    # For 2nd+ positions, check individual profit conditions
    partial_liquidation_mask = (profit_rates >= sell_profit_rates) & valid_positions
    # But exclude 1st positions from partial liquidation (they're handled by full liquidation)
    partial_liquidation_mask[:, :, 0] = False
    
    # If full liquidation is triggered for a stock, disable partial liquidation for that stock
    full_liquidation_expanded = full_liquidation_mask[:, :, cp.newaxis]  # Shape: (num_combinations, num_stocks, 1)
    partial_liquidation_mask = partial_liquidation_mask & (~full_liquidation_expanded)
    
    # --- Step 3: Execute full liquidations ---
    # Calculate proceeds from full liquidations
    full_liquidation_expanded_all = full_liquidation_expanded.repeat(max_splits, axis=2)
    full_liquidation_quantities = quantities * full_liquidation_expanded_all
    full_liquidation_proceeds = cp.sum(full_liquidation_quantities * current_prices_reshaped, axis=(1, 2))  # Shape: (num_combinations,)
    
    # Clear all positions for fully liquidated stocks
    positions_state[full_liquidation_expanded_all, 0] = 0  # Set quantities to 0
    positions_state[full_liquidation_expanded_all, 1] = 0  # Set buy_prices to 0
    
    # --- Step 4: Execute partial liquidations ---
    # Calculate proceeds from partial liquidations
    partial_liquidation_quantities = quantities * partial_liquidation_mask
    partial_liquidation_proceeds = cp.sum(partial_liquidation_quantities * current_prices_reshaped, axis=(1, 2))  # Shape: (num_combinations,)
    
    # Clear partially liquidated positions
    positions_state[partial_liquidation_mask, 0] = 0  # Set quantities to 0
    positions_state[partial_liquidation_mask, 1] = 0  # Set buy_prices to 0
    
    # --- Step 5: Update capital ---
    total_proceeds = full_liquidation_proceeds + partial_liquidation_proceeds
    portfolio_state[:, 0] += total_proceeds  # Add proceeds to capital
    
    return portfolio_state, positions_state


def _process_additional_buy_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray
):
    """
    Vectorized additional buy signal processing for all simulations.
    
    This function implements the MagicSplitStrategy additional buy logic:
    - If current price <= last_buy_price * (1 - additional_buy_drop_rate), trigger additional buy
    
    Args:
        portfolio_state: (num_combinations, 2) [capital, investment_per_order]
        positions_state: (num_combinations, num_stocks, max_splits, 2) [quantity, buy_price]
        param_combinations: (num_combinations, 5) [max_stocks, order_inv_ratio, add_buy_drop, sell_profit, add_buy_prio]
        current_prices: (num_stocks,) current market prices for all stocks
    
    Returns:
        Updated portfolio_state and positions_state after additional buy executions
    """
    num_combinations, num_stocks, max_splits, _ = positions_state.shape
    
    # Extract additional buy drop rates: shape (num_combinations, 1, 1)
    add_buy_drop_rates = param_combinations[:, 2:3].reshape(-1, 1, 1)
    
    # Get investment amounts per order: shape (num_combinations, 1, 1)
    investment_per_order = portfolio_state[:, 1:2].reshape(-1, 1, 1)
    
    # Get current capital: shape (num_combinations,)
    current_capital = portfolio_state[:, 0]
    
    # Reshape current prices: (1, num_stocks, 1)
    current_prices_reshaped = current_prices.reshape(1, -1, 1)
    
    # --- Step 1: Find stocks that have existing positions ---
    quantities = positions_state[..., 0]
    buy_prices = positions_state[..., 1]
    
    # Find the last (highest order) position for each stock in each simulation
    # We'll iterate through splits in reverse to find the last non-zero position
    has_positions = quantities > 0  # Shape: (num_combinations, num_stocks, max_splits)
    
    # Find the last position for each stock (rightmost True in the max_splits dimension)
    last_position_indices = cp.zeros((num_combinations, num_stocks), dtype=cp.int32)
    has_any_position = cp.any(has_positions, axis=2)  # Shape: (num_combinations, num_stocks)
    
    for sim in range(num_combinations):
        for stock in range(num_stocks):
            if has_any_position[sim, stock]:
                # Find the last True position
                positions_for_stock = has_positions[sim, stock, :]
                last_idx = cp.where(positions_for_stock)[0]
                if len(last_idx) > 0:
                    last_position_indices[sim, stock] = last_idx[-1]
    
    # Get the buy price of the last position for each stock
    last_buy_prices = cp.zeros((num_combinations, num_stocks), dtype=cp.float32)
    for sim in range(num_combinations):
        for stock in range(num_stocks):
            if has_any_position[sim, stock]:
                last_split_idx = last_position_indices[sim, stock]
                last_buy_prices[sim, stock] = buy_prices[sim, stock, last_split_idx]
    
    # --- Step 2: Check additional buy conditions ---
    # Trigger condition: current_price <= last_buy_price * (1 - add_buy_drop_rate)
    trigger_prices = last_buy_prices * (1 - add_buy_drop_rates.squeeze(-1))
    current_prices_2d = current_prices_reshaped.squeeze(-1)  # Shape: (1, num_stocks)
    
    additional_buy_condition = (current_prices_2d <= trigger_prices) & has_any_position
    
    # --- Step 3: Check if there's room for additional positions ---
    # Find next available split slot for each stock
    next_split_indices = last_position_indices + 1
    can_add_position = (next_split_indices < max_splits) & additional_buy_condition
    
    # --- Step 4: Check capital availability ---
    # Calculate required capital for additional buys
    quantities_to_buy = investment_per_order.squeeze(-1) / current_prices_2d  # Shape: (num_combinations, num_stocks)
    required_capital_per_stock = quantities_to_buy * current_prices_2d
    
    # Check if simulation has enough capital for each potential buy
    has_capital = current_capital.reshape(-1, 1) >= required_capital_per_stock
    
    # Final condition: all conditions must be met
    final_buy_condition = can_add_position & has_capital
    
    # --- Step 5: Execute additional buys ---
    total_spent = cp.zeros(num_combinations, dtype=cp.float32)
    
    for sim in range(num_combinations):
        for stock in range(num_stocks):
            if final_buy_condition[sim, stock]:
                next_split = next_split_indices[sim, stock]
                if next_split < max_splits:
                    # Calculate quantity and cost
                    stock_price = current_prices[stock]
                    inv_amount = investment_per_order[sim, 0, 0]
                    quantity = int(inv_amount / stock_price)
                    cost = quantity * stock_price
                    
                    # Execute the buy
                    positions_state[sim, stock, next_split, 0] = quantity  # Set quantity
                    positions_state[sim, stock, next_split, 1] = stock_price  # Set buy price
                    total_spent[sim] += cost
    
    # Update capital
    portfolio_state[:, 0] -= total_spent
    
    return portfolio_state, positions_state


def run_magic_split_strategy_on_gpu(
    initial_capital: float,
    param_combinations: cp.ndarray,
    all_data_gpu: cudf.DataFrame,
    trading_date_indices: cp.ndarray,  # 💡 파라미터 이름 변경 (trading_dates -> trading_date_indices)
    trading_dates_pd_cpu: pd.DatetimeIndex, # 💡 새로운 파라미터 추가
    all_tickers: list,
    max_splits_limit: int = 20
):
    """
    Main GPU-accelerated backtesting function for the MagicSplitStrategy.
    """
    print("🚀 Initializing GPU backtesting environment...")
    num_combinations = param_combinations.shape[0]
    num_trading_days = len(trading_date_indices) # 💡 길이는 정수 인덱스 배열 기준
    
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
    # 💡 정수 인덱스(0, 1, 2...)를 순회하도록 루프 변경
    for i, date_idx in enumerate(trading_date_indices):
        # 💡 현재 정수 인덱스를 사용하여 실제 날짜 객체를 CPU의 Pandas DatetimeIndex에서 조회
        # .item()을 사용하여 CuPy 스칼라를 Python 스칼라로 변환
        current_date = trading_dates_pd_cpu[date_idx.item()]
        current_month = current_date.month
        
        # Get current market prices for all stocks
        data_for_lookup = all_data_gpu.reset_index()
        current_day_data = data_for_lookup[data_for_lookup['date'] == current_date]
        if not current_day_data.empty:
            daily_prices = current_day_data.groupby('ticker')['close_price'].last()
            price_series = daily_prices.reindex(all_tickers).fillna(0)
            current_prices = cp.asarray(price_series.values, dtype=cp.float32)
            
            # --- Monthly Rebalance ---
            if current_month != previous_month:
                print(f"    - Rebalancing for month: {current_month}...")
                portfolio_state = _calculate_monthly_investment_gpu(
                    current_date, portfolio_state, positions_state, param_combinations, all_data_gpu, all_tickers
                )
                previous_month = current_month
            
            # --- Process Sell Signals ---
            portfolio_state, positions_state = _process_sell_signals_gpu(
                portfolio_state, positions_state, param_combinations, current_prices
            )
            
            # --- Process Additional Buy Signals ---
            portfolio_state, positions_state = _process_additional_buy_signals_gpu(
                portfolio_state, positions_state, param_combinations, current_prices
            )
            
            # --- Calculate and store daily portfolio values ---
            quantities = positions_state[..., 0]
            current_prices_reshaped = current_prices.reshape(1, -1, 1)
            stock_values = cp.sum(quantities * current_prices_reshaped, axis=(1, 2))
            total_values = portfolio_state[:, 0] + stock_values
            daily_portfolio_values[:, i] = total_values
        
        if (i + 1) % 252 == 0:
            year = current_date.year
            print(f"    - Simulating year: {year} ({i+1}/{num_trading_days})")

    print("🎉 GPU backtesting simulation finished.")
    
    return daily_portfolio_values