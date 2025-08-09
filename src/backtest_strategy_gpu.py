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
    last_trade_day_idx_state: cp.ndarray,
    current_day_idx: int,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    sell_commission_rate: float,
    sell_tax_rate: float,
):
    """
    [수정된 로직 v2]
    1. 전체 청산(손절매, 최대 '매매 미발생' 기간) 조건을 먼저 처리합니다.
    2. 그 다음, 청산되지 않은 종목에 한해 부분 수익실현을 처리합니다.
    """
    quantities = positions_state[..., 0]
    buy_prices = positions_state[..., 1]

    valid_positions = quantities > 0
    if not cp.any(valid_positions):
        return portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state

    # --- 파라미터 로드 ---
    sell_profit_rates = param_combinations[:, 3:4, cp.newaxis]
    stop_loss_rates = param_combinations[:, 5:6, cp.newaxis]
    max_inactivity_periods = param_combinations[:, 7:8] # 최대 매매 미발생 기간

    broadcasted_prices = cp.broadcast_to(current_prices.reshape(1, -1, 1), buy_prices.shape)
    
    # 이 날에 매도가 발생한 종목을 추적하기 위한 마스크 (쿨다운 관리용)
    sell_occurred_stock_mask = cp.zeros((positions_state.shape[0], positions_state.shape[1]), dtype=cp.bool_)

    # --- 시나리오 1: 전체 청산 (손절매 또는 최대 매매 미발생생 기간) ---
    stop_loss_prices = buy_prices * (1 + stop_loss_rates)
    stop_loss_position_mask = (broadcasted_prices <= stop_loss_prices) & valid_positions
    stock_stop_loss_mask = cp.any(stop_loss_position_mask, axis=2)

    # [수정 1] '최대 매매 미발생 기간' 계산 로직
    # 마지막 거래가 없었던 종목(-1)은 제외
    has_traded_before = last_trade_day_idx_state != -1
    days_inactive = current_day_idx - last_trade_day_idx_state
    stock_inactivity_mask = (days_inactive > max_inactivity_periods) & has_traded_before
    
    # [수정 2] 종목 단위 청산 마스크 통합
    stock_liquidation_mask = stock_stop_loss_mask | stock_inactivity_mask
    
    if cp.any(stock_liquidation_mask):
        # 청산 대상 종목의 모든 포지션에 대한 수익 계산
        revenue_matrix = quantities * broadcasted_prices
        # 청산 대상 종목(stock_liquidation_mask)만 필터링하여 수익 계산
        liquidation_revenue = cp.sum(revenue_matrix * stock_liquidation_mask[:, :, cp.newaxis], axis=(1, 2))

        cost_factor = 1.0 - sell_commission_rate - sell_tax_rate
        net_proceeds = cp.floor(liquidation_revenue * cost_factor)
        
        # 자본 업데이트
        portfolio_state[:, 0] += net_proceeds
        
        # 포지션 리셋 (청산된 종목의 모든 차수)
        reset_mask = stock_liquidation_mask[:, :, cp.newaxis, cp.newaxis]
        positions_state[reset_mask.broadcast_to(positions_state.shape)] = 0
        
        # 쿨다운용 마스크 업데이트
        sell_occurred_stock_mask |= stock_liquidation_mask
        
        # 전체 청산된 포지션은 이후의 수익실현 대상에서 제외해야 함
        # 현재 positions_state가 0으로 리셋되었으므로, valid_positions를 다시 계산
        valid_positions = positions_state[..., 0] > 0


    # --- 시나리오 2: 부분 매도 (수익 실현) ---
    # 전체 청산되지 않은 유효한 포지션에 대해서만 수익실현 검사
    target_sell_prices = buy_prices * (1 + sell_profit_rates)
    # 호가 단위는 실제 매도가에서 조정되어야 하므로, 여기서는 논리적 트리거 가격만 사용
    profit_taking_mask = (broadcasted_prices >= target_sell_prices) & valid_positions
    
    if cp.any(profit_taking_mask):
        # 실제 매도가는 지정가 주문을 반영하여 호가 단위에 맞게 올림 처리
        actual_sell_prices = adjust_price_up_gpu(target_sell_prices)
        
        # 수익 실현으로 인한 수익 계산
        # profit_taking_mask가 True인 차수들의 수익만 계산
        profit_taking_revenue_matrix = quantities * actual_sell_prices
        total_profit_revenue = cp.sum(profit_taking_revenue_matrix * profit_taking_mask, axis=(1, 2))

        cost_factor = 1.0 - sell_commission_rate - sell_tax_rate
        net_proceeds = cp.floor(total_profit_revenue * cost_factor)
        
        # 자본 업데이트
        portfolio_state[:, 0] += net_proceeds

        # 포지션 리셋 (수익 실현된 '차수'만)
        positions_state[profit_taking_mask] = 0

        # 쿨다운용 마스크 업데이트
        profit_occurred_stock_mask = cp.any(profit_taking_mask, axis=2)
        sell_occurred_stock_mask |= profit_occurred_stock_mask


    # --- 최종 상태 업데이트 (쿨다운 및 마지막 거래일) ---
    if cp.any(sell_occurred_stock_mask):
        sim_indices, stock_indices = cp.where(sell_occurred_stock_mask)
        cooldown_state[sim_indices, stock_indices] = current_day_idx
        # [추가] 매도 발생 시, 마지막 거래일 업데이트
        last_trade_day_idx_state[sim_indices, stock_indices] = current_day_idx

    # [수정] last_trade_day_idx_state 반환
    return portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state

def _process_additional_buy_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    last_trade_day_idx_state: cp.ndarray, # [추가] 파라미터
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
    sim_indices, stock_indices, quantities_to_buy, buy_prices_adjusted = (
        sim_indices[valid_buy], stock_indices[valid_buy], quantities_to_buy[valid_buy], buy_prices_adjusted[valid_buy]
    )

    if len(sim_indices) == 0:
        return portfolio_state, positions_state

    cost = buy_prices_adjusted * quantities_to_buy
    commission = cp.floor(cost * buy_commission_rate)
    total_cost = cost + commission

    can_afford = current_capital[sim_indices] >= total_cost
    sim_indices, stock_indices, quantities_to_buy, buy_prices_adjusted, total_cost = (
        sim_indices[can_afford], stock_indices[can_afford], quantities_to_buy[can_afford], buy_prices_adjusted[can_afford], total_cost[can_afford]
    )
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

    # [추가] 함수 반환 직전에 마지막 거래일 업데이트 로직 추가
    if len(sim_indices) > 0:
        # 이전에 성공적으로 매수한 sim_indices, stock_indices 사용
        last_trade_day_idx_state[sim_indices, stock_indices] = current_day_idx

    return portfolio_state, positions_state, last_trade_day_idx_state # [수정] 반환값 추가

def _process_new_entry_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    cooldown_state: cp.ndarray,
    last_trade_day_idx_state: cp.ndarray,
    current_day_idx: int,
    cooldown_period_days: int,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    candidate_tickers_for_day: cp.ndarray,
    candidate_atrs_for_day: cp.ndarray,
    buy_commission_rate: float,
):
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
            last_trade_day_idx_state[buy_sim_indices, ticker_idx] = current_day_idx

    return portfolio_state, positions_state, last_trade_day_idx_state

# backtest_strategy_gpu.py 파일의 run_magic_split_strategy_on_gpu 함수를 아래 코드로 교체합니다.

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
    debug_mode: bool = False
):
    """
    [전면 수정된 최종 버전]
    GPU 가속화를 사용하여 Magic Split 전략의 전체 백테스팅을 실행합니다.
    - 매도, 추가매수, 신규진입 로직을 모두 포함합니다.
    - '마지막 거래일' 기준의 비활성 기간 규칙을 적용합니다.
    - '현재가'를 사용하여 포트폴리오 가치를 정확히 계산합니다.
    """
    # --- 1. 상태 배열 초기화 ---
    num_combinations = param_combinations.shape[0]
    num_trading_days = len(trading_date_indices)
    num_tickers = len(all_tickers)
    cooldown_period_days = execution_params.get("cooldown_period_days", 5)

    # Portfolio state: [capital, investment_per_order]
    portfolio_state = cp.zeros((num_combinations, 2), dtype=cp.float32)
    portfolio_state[:, 0] = initial_cash
    
    # Position state: [quantity, buy_price, entry_date_idx]
    max_splits_from_params = int(cp.max(param_combinations[:, 6]).get()) if param_combinations.shape[1] > 6 else max_splits_limit
    positions_state = cp.zeros((num_combinations, num_tickers, max_splits_from_params, 3), dtype=cp.float32)
    
    # Cooldown state: [last_sell_day_idx]
    cooldown_state = cp.full((num_combinations, num_tickers), -1, dtype=cp.int32)
    
    # Last trade day state: [last_trade_day_idx]
    last_trade_day_idx_state = cp.full((num_combinations, num_tickers), -1, dtype=cp.int32)
    
    daily_portfolio_values = cp.zeros((num_combinations, num_trading_days), dtype=cp.float32)
    
    # Helper Dictionaries & Variables
    ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}
    all_tickers_gpu = cp.array(list(ticker_to_idx.values()), dtype=cp.int32)
    previous_month = -1

    # --- 2. 메인 백테스팅 루프 ---
    for i, date_idx in enumerate(trading_date_indices):
        current_date = trading_dates_pd_cpu[date_idx.item()]
        
        if debug_mode and (i % 20 == 0 or i == num_trading_days -1): # 디버그 모드에서 진행상황 로깅
            print(f"\n--- Day {i+1}/{num_trading_days}: {current_date.strftime('%Y-%m-%d')} ---")

        # 2-1. 현재 날짜의 데이터 준비 (종가, 후보 종목 등)
        # 현재일의 모든 종목 종가
        current_prices_gpu = cp.asarray(all_data_gpu.loc[current_date]['close_price'].reindex(all_tickers).fillna(0).values, dtype=cp.float32)
        
        # 현재일에 적용할 주간 필터링된 종목 리스트 (가장 최근 필터링 날짜 기준)
        try:
            latest_filter_date = weekly_filtered_gpu.index.asof(current_date)
            if pd.notna(latest_filter_date):
                candidate_tickers_series = weekly_filtered_gpu.loc[latest_filter_date]['ticker']
                candidate_tickers_list = candidate_tickers_series.to_list() if isinstance(candidate_tickers_series, cudf.Series) else [candidate_tickers_series]
                
                # 종목 코드를 인덱스로 변환
                candidate_indices = cp.array([ticker_to_idx.get(t, -1) for t in candidate_tickers_list if t in ticker_to_idx], dtype=cp.int32)
                candidate_tickers_for_day = candidate_indices[candidate_indices != -1]
                
                # 후보 종목의 ATR 값 준비
                candidate_atrs_for_day = cp.asarray(all_data_gpu.loc[(candidate_tickers_list, current_date)]['atr_14_ratio'].fillna(0).values, dtype=cp.float32)
            else:
                candidate_tickers_for_day = cp.array([], dtype=cp.int32)
                candidate_atrs_for_day = cp.array([], dtype=cp.float32)
        except (KeyError, IndexError):
            candidate_tickers_for_day = cp.array([], dtype=cp.int32)
            candidate_atrs_for_day = cp.array([], dtype=cp.float32)

        # 2-2. 월별 투자금 재계산
        if current_date.month != previous_month:
            portfolio_state = _calculate_monthly_investment_gpu(
                portfolio_state, positions_state, param_combinations, current_prices_gpu
            )
            previous_month = current_date.month

        # 2-3. 신호 처리 (매도 -> 추가 매수 -> 신규 진입 순)
        portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state = _process_sell_signals_gpu(
            portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, i,
            param_combinations, current_prices_gpu,
            execution_params["sell_commission_rate"], execution_params["sell_tax_rate"]
        )

        portfolio_state, positions_state, last_trade_day_idx_state = _process_additional_buy_signals_gpu(
            portfolio_state, positions_state, last_trade_day_idx_state, i,
            param_combinations, current_prices_gpu,
            execution_params["buy_commission_rate"]
        )
        
        portfolio_state, positions_state, last_trade_day_idx_state = _process_new_entry_signals_gpu(
            portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, i,
            cooldown_period_days, param_combinations, current_prices_gpu,
            candidate_tickers_for_day, candidate_atrs_for_day,
            execution_params["buy_commission_rate"]
        )

        # 2-4. 일일 포트폴리오 가치 업데이트
        # [수정] '현재가'를 사용하여 주식 평가 가치를 정확히 계산
        stock_quantities = cp.sum(positions_state[..., 0], axis=2)  # shape: (num_sims, num_tickers)
        stock_market_values = stock_quantities * current_prices_gpu   # current_prices_gpu는 브로드캐스팅됨
        total_stock_value = cp.sum(stock_market_values, axis=1)    # shape: (num_sims,)
        
        daily_portfolio_values[:, i] = portfolio_state[:, 0] + total_stock_value

        if debug_mode and (i % 20 == 0 or i == num_trading_days -1):
            capital_snapshot = portfolio_state[0, 0].get()
            stock_val_snapshot = total_stock_value[0].get()
            total_val_snapshot = daily_portfolio_values[0, i].get()
            num_pos_snapshot = cp.sum(cp.any(positions_state[0, :, :, 0] > 0, axis=1)).get()
            print(f" [GPU_HOLDINGS] {cp.where(cp.any(positions_state[0, :, :, 0] > 0, axis=1))[0].get().tolist()}")
            print(f"[END]   Capital: {capital_snapshot:,.0f} | Stock Val: {stock_val_snapshot:,.0f} | Total Val: {total_val_snapshot:,.0f} | Stocks Held: {num_pos_snapshot}")


    # --- 3. 결과 반환 ---
    return daily_portfolio_values