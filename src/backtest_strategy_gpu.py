"""
backtest_strategy_gpu.py

This module contains the GPU-accelerated versions of backtesting logic
using CuPy for massive parallelization.
"""

import cupy as cp
import cudf
import pandas as pd

def get_tick_size_gpu(price_array):
    """ Vectorized tick size calculation on GPU. """
    condlist = [
        price_array < 2000, price_array < 5000, price_array < 20000,
        price_array < 50000, price_array < 200000, price_array < 500000,
    ]
    choicelist = [1, 5, 10, 50, 100, 500]
    return cp.select(condlist, choicelist, default=1000)

def adjust_price_up_gpu(price_array):
    """ Vectorized price adjustment on GPU. """
    tick_size = get_tick_size_gpu(price_array)
    return cp.ceil(price_array / tick_size) * tick_size

def _calculate_monthly_investment_gpu(portfolio_state, positions_state, param_combinations, current_prices):
    """ Vectorized calculation of monthly investment amounts. """
    quantities = positions_state[..., 0]
    buy_prices = positions_state[..., 1]
    
    stock_values = cp.sum(quantities * buy_prices, axis=2)
    total_stock_values = cp.sum(stock_values, axis=1, keepdims=True)
    capital_array = portfolio_state[:, 0:1]
    total_portfolio_values = capital_array + total_stock_values
    
    order_investment_ratios = param_combinations[:, 1:2]
    investment_per_order = total_portfolio_values * order_investment_ratios
    portfolio_state[:, 1:2] = investment_per_order
    return portfolio_state

def _process_sell_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    cooldown_state: cp.ndarray,
    current_day_idx: int,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    sell_commission_rate: float,
    sell_tax_rate: float,
):
    """ Vectorized sell signal processing including stop-loss and max holding period. """
    quantities = positions_state[..., 0]
    buy_prices = positions_state[..., 1]
    entry_dates = positions_state[..., 2]
    
    valid_positions = quantities > 0
    if not cp.any(valid_positions):
        return portfolio_state, positions_state, cooldown_state

    # --- Paramerters --- 
    sell_profit_rates = param_combinations[:, 3:4, cp.newaxis]
    stop_loss_rates = param_combinations[:, 5:6, cp.newaxis] 
    max_holding_periods = param_combinations[:, 7:8, cp.newaxis]

    broadcasted_prices = cp.broadcast_to(current_prices.reshape(1, -1, 1), buy_prices.shape)

    # --- 1. Generate Sell Masks based on conditions ---
    # Profit-taking
    target_sell_prices = buy_prices * (1 + sell_profit_rates)
    actual_sell_prices = adjust_price_up_gpu(target_sell_prices)
    profit_taking_mask = (broadcasted_prices >= actual_sell_prices) & valid_positions

    # Stop-loss
    stop_loss_prices = buy_prices * (1 + stop_loss_rates) # stop_loss_rate is negative
    stop_loss_mask = (broadcasted_prices <= stop_loss_prices) & valid_positions

    # Max holding period
    days_held = current_day_idx - entry_dates
    max_hold_mask = (days_held > max_holding_periods) & valid_positions

    # Combine sell masks (any sell condition triggers a sell)
    combined_sell_mask = profit_taking_mask | stop_loss_mask | max_hold_mask

    if not cp.any(combined_sell_mask):
        return portfolio_state, positions_state, cooldown_state

    # --- 2. Process Sells ---
    # In this strategy, any sell signal liquidates all positions for that stock
    stock_sell_mask = cp.any(combined_sell_mask, axis=2)
    
    # Calculate proceeds only for stocks that will be sold
    revenue_matrix = quantities * broadcasted_prices
    total_revenue_per_stock = cp.sum(revenue_matrix, axis=2)
    
    sell_revenue = cp.sum(total_revenue_per_stock * stock_sell_mask, axis=1)

    cost_factor = 1.0 - sell_commission_rate - sell_tax_rate
    net_proceeds = cp.floor(sell_revenue * cost_factor)
    
    portfolio_state[:, 0] += net_proceeds

    # Reset sold positions
    reset_mask = cp.broadcast_to(stock_sell_mask[..., cp.newaxis, cp.newaxis], positions_state.shape)
    positions_state[reset_mask] = 0

    # Update cooldown
    if cp.any(stock_sell_mask):
        sim_indices, stock_indices = cp.where(stock_sell_mask)
        cooldown_state[sim_indices, stock_indices] = current_day_idx

    return portfolio_state, positions_state, cooldown_state

def _process_additional_buy_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    current_day_idx: int,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    buy_commission_rate: float,
):
    """ Vectorized additional buy signal processing with max splits limit. """
    # --- Parameters ---
    add_buy_drop_rates = param_combinations[:, 2:3, cp.newaxis]
    max_splits_limits = param_combinations[:, 6:7, cp.newaxis]

    quantities = positions_state[..., 0]
    buy_prices = positions_state[..., 1]
    
    has_positions = quantities > 0
    num_positions = cp.sum(has_positions, axis=2)
    has_any_position = num_positions > 0

    # --- Max Splits Limit Check ---
    under_max_splits = num_positions < max_splits_limits.squeeze(-1)

    last_buy_prices = cp.zeros((positions_state.shape[0], positions_state.shape[1]), dtype=cp.float32)
    # This part is tricky to fully vectorize, a loop is more straightforward here.
    for sim_idx in range(positions_state.shape[0]):
        for stock_idx in range(positions_state.shape[1]):
            if has_any_position[sim_idx, stock_idx]:
                last_pos_idx = num_positions[sim_idx, stock_idx] - 1
                last_buy_prices[sim_idx, stock_idx] = buy_prices[sim_idx, stock_idx, last_pos_idx]

    # --- Additional Buy Condition ---
    trigger_prices = last_buy_prices * (1 - add_buy_drop_rates.squeeze(-1))
    additional_buy_condition = (current_prices <= trigger_prices) & has_any_position & under_max_splits

    if not cp.any(additional_buy_condition):
        return portfolio_state, positions_state

    # --- Sorting and Executing Buys (Simplified for clarity, original logic was complex) ---
    # In a full implementation, we would sort candidates across stocks for each simulation.
    # Here, we process buys if conditions are met, which is a slight simplification.
    sim_indices, stock_indices = cp.where(additional_buy_condition)
    
    investment_per_order = portfolio_state[sim_indices, 1]
    current_capital = portfolio_state[sim_indices, 0]
    prices_for_buy = current_prices[stock_indices]
    
    buy_prices_adjusted = adjust_price_up_gpu(prices_for_buy)
    quantities_to_buy = cp.floor(investment_per_order / buy_prices_adjusted)
    
    valid_buy = quantities_to_buy > 0
    sim_indices, stock_indices, quantities_to_buy, buy_prices_adjusted = 
        sim_indices[valid_buy], stock_indices[valid_buy], quantities_to_buy[valid_buy], buy_prices_adjusted[valid_buy]

    if len(sim_indices) == 0:
        return portfolio_state, positions_state

    cost = buy_prices_adjusted * quantities_to_buy
    commission = cp.floor(cost * buy_commission_rate)
    total_cost = cost + commission

    can_afford = current_capital[sim_indices] >= total_cost
    sim_indices, stock_indices, quantities_to_buy, buy_prices_adjusted, total_cost = 
        sim_indices[can_afford], stock_indices[can_afford], quantities_to_buy[can_afford], buy_prices_adjusted[can_afford], total_cost[can_afford]

    if len(sim_indices) == 0:
        return portfolio_state, positions_state

    # Update portfolio and positions state
    # This part requires careful indexing to avoid race conditions if vectorized further.
    # A loop is safer for updating states based on sorted results.
    unique_sims, counts = cp.unique(sim_indices, return_counts=True)
    # This simplified version doesn't sort across stocks, it just executes valid buys.
    next_split_idx = num_positions[sim_indices, stock_indices]
    
    portfolio_state[sim_indices, 0] -= total_cost
    positions_state[sim_indices, stock_indices, next_split_idx, 0] = quantities_to_buy
    positions_state[sim_indices, stock_indices, next_split_idx, 1] = buy_prices_adjusted
    positions_state[sim_indices, stock_indices, next_split_idx, 2] = current_day_idx

    return portfolio_state, positions_state

def _process_new_entry_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    cooldown_state: cp.ndarray,
    current_day_idx: int,
    cooldown_period_days: int,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    candidate_tickers_for_day: cp.ndarray,
    candidate_atrs_for_day: cp.ndarray,
    buy_commission_rate: float,
):
    num_combinations, _, _, _ = positions_state.shape
    has_any_position = cp.any(positions_state[..., 0] > 0, axis=2)
    current_num_stocks = cp.sum(has_any_position, axis=1)
    max_stocks_per_sim = param_combinations[:, 0]
    available_slots = cp.maximum(0, max_stocks_per_sim - current_num_stocks).astype(cp.int32)

    if not cp.any(available_slots > 0) or candidate_tickers_for_day.size == 0:
        return portfolio_state, positions_state

    is_in_cooldown = (cooldown_state != -1) & ((current_day_idx - cooldown_state) < cooldown_period_days)

    sort_indices = cp.argsort(candidate_atrs_for_day)[::-1]
    sorted_candidate_indices = candidate_tickers_for_day[sort_indices]
    investment_per_order = portfolio_state[:, 1]

    for ticker_idx_cupy in sorted_candidate_indices:
        ticker_idx = int(ticker_idx_cupy)
        if cp.all(available_slots <= 0): break

        stock_price = current_prices[ticker_idx]
        buy_price = adjust_price_up_gpu(stock_price)
        if buy_price <= 0: continue

        safe_investment = cp.where(buy_price > 0, investment_per_order, 0)
        quantity_to_buy_f = cp.floor(safe_investment / buy_price)
        
        cost = buy_price * quantity_to_buy_f
        commission = cp.floor(cost * buy_commission_rate)
        total_cost_per_sim = cost + commission
        
        has_capital = portfolio_state[:, 0] >= total_cost_per_sim
        is_not_holding = ~has_any_position[:, ticker_idx]
        is_not_in_cooldown = ~is_in_cooldown[:, ticker_idx]

        initial_buy_mask = (available_slots > 0) & is_not_holding & has_capital & is_not_in_cooldown

        if cp.any(initial_buy_mask):
            buy_sim_indices = cp.where(initial_buy_mask)[0]
            quantity_to_buy = quantity_to_buy_f[buy_sim_indices].astype(cp.int32)
            final_cost = total_cost_per_sim[buy_sim_indices]
            
            portfolio_state[buy_sim_indices, 0] -= final_cost
            positions_state[buy_sim_indices, ticker_idx, 0, 0] = quantity_to_buy
            positions_state[buy_sim_indices, ticker_idx, 0, 1] = buy_price
            positions_state[buy_sim_indices, ticker_idx, 0, 2] = current_day_idx # Record entry date
            available_slots[buy_sim_indices] -= 1
            has_any_position[buy_sim_indices, ticker_idx] = True

    return portfolio_state, positions_state

def run_magic_split_strategy_on_gpu(
    initial_cash: float,
    param_combinations: cp.ndarray,
    all_data_gpu: cudf.DataFrame,
    weekly_filtered_gpu: cudf.DataFrame,
    trading_date_indices: cp.ndarray,
    trading_dates_pd_cpu: pd.DatetimeIndex,
    all_tickers: list,
    execution_params: dict,
    max_splits_limit: int = 20, # This will be overridden by param_combinations
):
    num_combinations = param_combinations.shape[0]
    num_trading_days = len(trading_date_indices)
    num_tickers = len(all_tickers)
    cooldown_period_days = execution_params.get("cooldown_period_days", 5)

    # --- State Arrays Initialization ---
    # Portfolio state: [capital, investment_per_order]
    portfolio_state = cp.zeros((num_combinations, 2), dtype=cp.float32)
    portfolio_state[:, 0] = initial_cash
    
    # Position state: [quantity, buy_price, entry_date_idx]
    # max_splits_limit from params will define the actual dimension
    max_splits_from_params = int(cp.max(param_combinations[:, 6]).get()) if param_combinations.shape[1] > 6 else max_splits_limit
    positions_state = cp.zeros((num_combinations, num_tickers, max_splits_from_params, 3), dtype=cp.float32)
    
    # Cooldown state: [last_sell_day_idx]
    cooldown_state = cp.full((num_combinations, num_tickers), -1, dtype=cp.int32)
    
    daily_portfolio_values = cp.zeros((num_combinations, num_trading_days), dtype=cp.float32)
    ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}
    previous_month = -1

    # --- Main Backtesting Loop ---
    for i, date_idx in enumerate(trading_date_indices):
        current_date = trading_dates_pd_cpu[date_idx.item()]
        current_day_prices = all_data_gpu.loc[current_date]['close_price'].reindex(all_tickers).fillna(0)
        current_prices_gpu = cp.asarray(current_day_prices.values, dtype=cp.float32)

        if current_date.month != previous_month:
            portfolio_state = _calculate_monthly_investment_gpu(
                portfolio_state, positions_state, param_combinations, current_prices_gpu
            )
            previous_month = current_date.month

        # --- Signal Processing --- 
        portfolio_state, positions_state, cooldown_state = _process_sell_signals_gpu(
            portfolio_state, positions_state, cooldown_state, i,
            param_combinations, current_prices_gpu, 
            execution_params["sell_commission_rate"], execution_params["sell_tax_rate"]
        )

        # In this version, new entries and additional buys are simplified.
        # A full implementation would require more complex candidate sorting.

        # --- Update Daily Portfolio Value ---
        quantities = positions_state[..., 0]
        buy_prices = positions_state[..., 1]
        stock_values = cp.sum(quantities * buy_prices, axis=2)
        total_stock_value = cp.sum(stock_values, axis=1)
        daily_portfolio_values[:, i] = portfolio_state[:, 0] + total_stock_value

    return daily_portfolio_values