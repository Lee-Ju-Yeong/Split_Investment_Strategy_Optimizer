"""
backtest_strategy_gpu.py

This module contains the GPU-accelerated versions of backtesting logic
using CuPy and Numba for massive parallelization.
"""

import cupy as cp
import cudf
import pandas as pd

def get_tick_size_gpu(price_array):
    """
    주가 배열에 따른 호가 단위 배열을 반환합니다.
    """
    condlist = [
        price_array < 2000,
        price_array < 5000,
        price_array < 20000,
        price_array < 50000,
        price_array < 200000,
        price_array < 500000,
    ]
    choicelist = [
        cp.full_like(price_array, 1),
        cp.full_like(price_array, 5),
        cp.full_like(price_array, 10),
        cp.full_like(price_array, 50),
        cp.full_like(price_array, 100),
        cp.full_like(price_array, 500),
    ]
    return cp.select(condlist, choicelist, default=1000)

def adjust_price_up_gpu(price_array):
    """주어진 가격 배열을 호가 단위에 맞춰 올림 처리합니다."""
    tick_size = get_tick_size_gpu(price_array)
    return cp.ceil(price_array / tick_size) * tick_size

def _calculate_monthly_investment_gpu(
    current_date,
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    param_combinations: cp.ndarray,
    all_data_gpu: cudf.DataFrame,
    all_tickers: list,
):
    data_for_lookup = all_data_gpu.reset_index()
    filtered_data = data_for_lookup[data_for_lookup["date"] <= current_date]
    latest_prices = filtered_data.groupby("ticker")["close_price"].last()
    price_series = latest_prices.reindex(all_tickers).fillna(0)
    prices_gpu = cp.asarray(price_series.values, dtype=cp.float32)
    quantities = positions_state[..., 0]
    prices_reshaped = prices_gpu.reshape(1, -1, 1)
    stock_values = cp.sum(quantities * prices_reshaped, axis=2)
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
    sell_profit_rates = param_combinations[:, 3:4].reshape(-1, 1, 1)
    quantities = positions_state[..., 0]
    buy_prices = positions_state[..., 1]
    broadcasted_prices = cp.broadcast_to(
        current_prices.reshape(1, -1, 1), buy_prices.shape
    )
    valid_positions = buy_prices > 0
    
    target_sell_prices = buy_prices * (1 + sell_profit_rates)
    actual_sell_prices = adjust_price_up_gpu(target_sell_prices)
    
    sell_trigger_condition = (broadcasted_prices >= actual_sell_prices) & valid_positions
    first_position_sell_triggered = sell_trigger_condition[:, :, 0]
    
    partial_sell_mask = sell_trigger_condition.copy()
    partial_sell_mask[:, :, 0] = False
    partial_sell_mask &= ~cp.broadcast_to(
        first_position_sell_triggered[:, :, cp.newaxis], partial_sell_mask.shape
    )
    
    full_liquidation_mask = cp.broadcast_to(
        first_position_sell_triggered[:, :, cp.newaxis], quantities.shape
    )
    
    # --- Net Proceeds Calculation with Floor ---
    cost_factor = 1.0 - sell_commission_rate - sell_tax_rate
    
    full_liquidation_revenue = cp.sum(quantities * actual_sell_prices * full_liquidation_mask, axis=(1, 2))
    partial_sell_revenue = cp.sum(quantities * actual_sell_prices * partial_sell_mask, axis=(1, 2))
    
    net_proceeds = cp.floor((full_liquidation_revenue + partial_sell_revenue) * cost_factor)
    
    portfolio_state[:, 0] += net_proceeds
    
    positions_state[..., 0][partial_sell_mask] = 0
    full_liquidation_stock_mask = cp.broadcast_to(
        first_position_sell_triggered[:, :, cp.newaxis], positions_state[..., 0].shape
    )
    positions_state[..., 0][full_liquidation_stock_mask] = 0
    positions_state[..., 1][full_liquidation_stock_mask] = -1

    if cp.any(first_position_sell_triggered):
        sim_indices, stock_indices = cp.where(first_position_sell_triggered)
        cooldown_state[sim_indices, stock_indices] = current_day_idx

    return portfolio_state, positions_state, cooldown_state

def _process_additional_buy_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    buy_commission_rate: float,
):
    num_combinations, num_stocks, max_splits, _ = positions_state.shape
    add_buy_drop_rates = param_combinations[:, 2:3].reshape(-1, 1, 1)
    investment_per_order = portfolio_state[:, 1:2].reshape(-1, 1, 1)
    current_prices_reshaped = current_prices.reshape(1, -1, 1)
    quantities = positions_state[..., 0]
    buy_prices = positions_state[..., 1]
    has_positions = quantities > 0
    has_any_position = cp.any(has_positions, axis=2)
    last_position_indices = cp.zeros((num_combinations, num_stocks), dtype=cp.int32)
    
    for sim in range(num_combinations):
        for stock in range(num_stocks):
            if has_any_position[sim, stock]:
                last_idx = cp.where(has_positions[sim, stock, :])[0]
                if len(last_idx) > 0:
                    last_position_indices[sim, stock] = last_idx[-1]
                    
    last_buy_prices = cp.zeros((num_combinations, num_stocks), dtype=cp.float32)
    for sim in range(num_combinations):
        for stock in range(num_stocks):
            if has_any_position[sim, stock]:
                last_split_idx = last_position_indices[sim, stock]
                last_buy_prices[sim, stock] = buy_prices[sim, stock, last_split_idx]
                
    trigger_prices = last_buy_prices * (1 - add_buy_drop_rates.squeeze(-1))
    current_prices_2d = current_prices_reshaped.squeeze(-1)
    additional_buy_condition = (current_prices_2d <= trigger_prices) & has_any_position
    
    can_add_position = cp.zeros_like(additional_buy_condition, dtype=cp.bool_)
    next_split_indices_to_buy = cp.full_like(last_position_indices, -1, dtype=cp.int32)
    for sim in range(num_combinations):
        for stock in range(num_stocks):
            if additional_buy_condition[sim, stock]:
                positions_for_stock = positions_state[sim, stock, :, 0]
                empty_slots = cp.where(positions_for_stock == 0)[0]
                if empty_slots.size > 0:
                    first_empty_slot = empty_slots[0]
                    if first_empty_slot < max_splits:
                        can_add_position[sim, stock] = True
                        next_split_indices_to_buy[sim, stock] = first_empty_slot
                        
    initial_buy_condition = can_add_position
    if cp.any(initial_buy_condition):
        sim_indices, stock_indices = cp.where(initial_buy_condition)
        num_existing_splits = cp.sum(has_positions[sim_indices, stock_indices], axis=1)
        last_buy_prices_for_candidates = last_buy_prices[sim_indices, stock_indices]
        current_prices_for_candidates = current_prices[stock_indices]
        drop_rates = cp.zeros_like(last_buy_prices_for_candidates)
        valid_mask = last_buy_prices_for_candidates > 0
        drop_rates[valid_mask] = (last_buy_prices_for_candidates[valid_mask] - current_prices_for_candidates[valid_mask]) / last_buy_prices_for_candidates[valid_mask]
        add_buy_priority_params = param_combinations[sim_indices, 4]
        sort_metric = cp.where(add_buy_priority_params == 0, num_existing_splits.astype(cp.float32), -drop_rates)
        
        candidates_gdf = cudf.DataFrame({
            'sim_idx': sim_indices,
            'sort_metric': sort_metric,
            'stock_idx': stock_indices,
            'next_split_idx': next_split_indices_to_buy[sim_indices, stock_indices]
        })
        sorted_candidates_gdf = candidates_gdf.sort_values(by=['sim_idx', 'sort_metric'], ascending=[True, True])
        
        sorted_sim_indices = sorted_candidates_gdf['sim_idx'].values
        sorted_stock_indices = sorted_candidates_gdf['stock_idx'].values
        sorted_next_split_indices = sorted_candidates_gdf['next_split_idx'].values
        
        for i in range(len(sorted_sim_indices)):
            sim_idx = int(sorted_sim_indices[i])
            stock_idx = int(sorted_stock_indices[i])
            next_split_idx = int(sorted_next_split_indices[i])
            
            current_sim_capital = portfolio_state[sim_idx, 0]
            inv_per_order = investment_per_order[sim_idx, 0, 0]
            current_price = current_prices[stock_idx]
            buy_price = adjust_price_up_gpu(current_price)
            
            if buy_price <= 0: continue
            quantity = cp.floor(inv_per_order / buy_price)
            if quantity <= 0: continue
            
            cost = float(buy_price) * float(quantity)
            commission = cp.floor(cost * buy_commission_rate)
            total_cost = cost + commission
            
            if float(current_sim_capital) >= total_cost:
                portfolio_state[sim_idx, 0] -= total_cost
                positions_state[sim_idx, stock_idx, next_split_idx, 0] = quantity
                positions_state[sim_idx, stock_idx, next_split_idx, 1] = buy_price
                
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
    max_splits_limit: int = 20,
    debug_mode: bool = False,
):
    num_combinations = param_combinations.shape[0]
    num_trading_days = len(trading_date_indices)
    num_tickers = len(all_tickers)
    cooldown_period_days = execution_params.get("cooldown_period_days", 5)

    portfolio_state = cp.zeros((num_combinations, 2), dtype=cp.float32)
    portfolio_state[:, 0] = initial_cash
    positions_state = cp.zeros((num_combinations, num_tickers, max_splits_limit, 2), dtype=cp.float32)
    cooldown_state = cp.full((num_combinations, num_tickers), -1, dtype=cp.int32)
    daily_portfolio_values = cp.zeros((num_combinations, num_trading_days), dtype=cp.float32)
    ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}
    previous_month = -1

    for i, date_idx in enumerate(trading_date_indices):
        current_date = trading_dates_pd_cpu[date_idx.item()]
        current_month = current_date.month
        data_for_lookup = all_data_gpu.reset_index()
        current_day_data = data_for_lookup[data_for_lookup["date"] == current_date]

        if not current_day_data.empty:
            daily_prices = current_day_data.groupby("ticker")["close_price"].last()
            price_series = daily_prices.reindex(all_tickers).fillna(0)
            current_prices = cp.asarray(price_series.values, dtype=cp.float32)

            if current_month != previous_month:
                portfolio_state = _calculate_monthly_investment_gpu(
                    current_date, portfolio_state, positions_state, param_combinations, all_data_gpu, all_tickers
                )
                previous_month = current_month

            weekly_filtered_reset = weekly_filtered_gpu.reset_index()
            past_data = weekly_filtered_reset[weekly_filtered_reset["date"] < current_date]
            candidates_of_the_week = cudf.DataFrame()
            if not past_data.empty:
                most_recent_date_cudf = past_data["date"].max()
                candidates_of_the_week = past_data[past_data["date"] == most_recent_date_cudf]

            candidate_tickers_for_day = cp.array([], dtype=cp.int32)
            candidate_atrs_for_day = cp.array([], dtype=cp.float32)
            if not candidates_of_the_week.empty:
                candidate_tickers_str = candidates_of_the_week["ticker"].to_arrow().to_pylist()
                candidate_indices = [ticker_to_idx.get(t) for t in candidate_tickers_str if ticker_to_idx.get(t) is not None]
                if candidate_indices:
                    data_for_filtering = all_data_gpu.reset_index()
                    mask_ticker = data_for_filtering["ticker"].isin(candidate_tickers_str)
                    mask_date = data_for_filtering["date"] == current_date
                    candidate_data_today = data_for_filtering[mask_ticker & mask_date]
                    if not candidate_data_today.empty:
                        candidate_data_today = candidate_data_today.set_index(["ticker", "date"])
                        valid_candidates = candidate_data_today.dropna(subset=["atr_14_ratio"])
                        if not valid_candidates.empty:
                            valid_tickers_str = valid_candidates.index.get_level_values("ticker").to_arrow().to_pylist()
                            valid_indices = [ticker_to_idx[t] for t in valid_tickers_str]
                            candidate_tickers_for_day = cp.array(valid_indices, dtype=cp.int32)
                            candidate_atrs_for_day = cp.asarray(valid_candidates["atr_14_ratio"].values, dtype=cp.float32)

            portfolio_state, positions_state = _process_new_entry_signals_gpu(
                portfolio_state, positions_state, cooldown_state, i, cooldown_period_days,
                param_combinations, current_prices, candidate_tickers_for_day, candidate_atrs_for_day,
                buy_commission_rate=execution_params["buy_commission_rate"]
            )
            portfolio_state, positions_state = _process_additional_buy_signals_gpu(
                portfolio_state, positions_state, param_combinations, current_prices,
                buy_commission_rate=execution_params["buy_commission_rate"]
            )
            portfolio_state, positions_state, cooldown_state = _process_sell_signals_gpu(
                portfolio_state, positions_state, cooldown_state, i,
                param_combinations, current_prices, execution_params["sell_commission_rate"], 
                execution_params["sell_tax_rate"]
            )

            quantities = positions_state[..., 0]
            current_prices_reshaped = current_prices.reshape(1, -1, 1)
            stock_values = cp.sum(quantities * current_prices_reshaped, axis=(1, 2))
            total_values = portfolio_state[:, 0] + stock_values
            daily_portfolio_values[:, i] = total_values
        else:
            if i > 0:
                daily_portfolio_values[:, i] = daily_portfolio_values[:, i - 1]
            else:
                daily_portfolio_values[:, i] = initial_cash

    return daily_portfolio_values

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
    num_combinations, num_stocks_total, _, _ = positions_state.shape
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
            available_slots[buy_sim_indices] -= 1
            has_any_position[buy_sim_indices, ticker_idx] = True

    return portfolio_state, positions_state
