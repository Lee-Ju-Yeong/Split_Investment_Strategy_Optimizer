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
    # [수정] cp.full_like를 사용하여 price_array와 동일한 shape의 배열 리스트를 생성합니다.
    # 이것이 cupy.select가 요구하는 형식입니다.
    choicelist = [
        cp.full_like(price_array, 1),
        cp.full_like(price_array, 5),
        cp.full_like(price_array, 10),
        cp.full_like(price_array, 50),
        cp.full_like(price_array, 100),
        cp.full_like(price_array, 500),
    ]
    return cp.select(condlist, choicelist, default=1000)
    return cp.select(condlist, choicelist, default=1000)

def adjust_price_up_gpu(price_array):
    """ Vectorized price adjustment on GPU. """
    tick_size = get_tick_size_gpu(price_array)
    return cp.ceil(price_array / tick_size) * tick_size

def _calculate_monthly_investment_gpu(portfolio_state, positions_state, param_combinations, current_prices):
    """ Vectorized calculation of monthly investment amounts based on current market value. """
    quantities = positions_state[..., 0]
    
    # [수정] 총 자산 계산 시 매수 평단이 아닌 '현재가'를 사용해야 합니다.
    total_quantities_per_stock = cp.sum(quantities, axis=2)
    stock_market_values = total_quantities_per_stock * current_prices
    total_stock_values = cp.sum(stock_market_values, axis=1, keepdims=True)

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
    quantities_sum = cp.sum(quantities, axis=2)
    # 0으로 나누는 것을 방지하기 위해 0인 경우 1로 설정
    quantities_sum_safe = cp.where(quantities_sum == 0, 1, quantities_sum)
    avg_buy_prices = cp.sum(buy_prices * quantities, axis=2) / quantities_sum_safe
    
    stock_stop_loss_mask = (current_prices <= avg_buy_prices * (1 + stop_loss_rates.squeeze(-1))) & (cp.sum(quantities, axis=2) > 0)
    
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
        # [수정] cp.broadcast_to 함수를 사용하여 AttributeError를 해결합니다.
        positions_state[cp.broadcast_to(reset_mask, positions_state.shape)] = 0
        
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
    last_trade_day_idx_state: cp.ndarray,
    current_day_idx: int,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    buy_commission_rate: float,
):
    """ [완전 벡터화된 최종 로직] 루프를 제거하고 순수 CuPy 연산으로 추가 매수를 처리합니다. """
    add_buy_drop_rates = param_combinations[:, 2:3]
    max_splits_limits = param_combinations[:, 6:7]

    quantities = positions_state[..., 0]
    buy_prices = positions_state[..., 1]
    
    has_positions = quantities > 0
    num_positions = cp.sum(has_positions, axis=2)
    has_any_position = num_positions > 0

    if not cp.any(has_any_position):
        return portfolio_state, positions_state, last_trade_day_idx_state

    # 1. 마지막 매수가 찾기 (완전 벡터화)
    # 각 종목의 마지막 포지션만 True로 마스킹
    last_pos_mask = (cp.cumsum(has_positions, axis=2) == num_positions[:, :, cp.newaxis]) & has_positions
    last_buy_prices = cp.sum(buy_prices * last_pos_mask, axis=2)

    # 2. 추가 매수 조건 확인
    trigger_prices = last_buy_prices * (1 - add_buy_drop_rates)
    under_max_splits = num_positions < max_splits_limits
    additional_buy_mask = (current_prices <= trigger_prices) & has_any_position & under_max_splits
    
    if not cp.any(additional_buy_mask):
        return portfolio_state, positions_state, last_trade_day_idx_state
        
    # 3. 비어있는 가장 낮은 차수 슬롯 찾기 (완전 벡터화)
    is_empty_slot = quantities == 0
    # argmax는 첫 번째 True의 인덱스를 반환. 모두 False이면 0을 반환.
    first_empty_slot_indices = cp.argmax(is_empty_slot, axis=2)
    # 모두 차있는 경우 (모두 False), argmax가 0을 반환하는 엣지 케이스 보정
    fix_mask = (is_empty_slot[:, :, 0] == False) & (first_empty_slot_indices == 0)
    first_empty_slot_indices[fix_mask] = positions_state.shape[2] # max_splits_limit
    next_split_indices = first_empty_slot_indices
    
    # 4. 매수 실행
    sim_indices, stock_indices = cp.where(additional_buy_mask)
    investment_per_order = portfolio_state[sim_indices, 1]
    prices_for_buy = current_prices[stock_indices]
    buy_prices_adjusted = adjust_price_up_gpu(prices_for_buy)
    
    quantities_to_buy = cp.floor(investment_per_order / buy_prices_adjusted)
    quantities_to_buy[buy_prices_adjusted <= 0] = 0
    
    cost = buy_prices_adjusted * quantities_to_buy
    commission = cp.floor(cost * buy_commission_rate)
    total_cost = cost + commission
    
    can_afford = portfolio_state[sim_indices, 0] >= total_cost
    final_buy_mask = (quantities_to_buy > 0) & can_afford

    if not cp.any(final_buy_mask):
        return portfolio_state, positions_state, last_trade_day_idx_state

    # 실제 매수할 대상만 필터링
    buy_sim_indices = sim_indices[final_buy_mask]
    buy_stock_indices = stock_indices[final_buy_mask]
    buy_quantities = quantities_to_buy[final_buy_mask]
    buy_prices_final = buy_prices_adjusted[final_buy_mask]
    buy_total_cost = total_cost[final_buy_mask]
    buy_split_indices = next_split_indices[buy_sim_indices, buy_stock_indices]

    # 5. 상태 업데이트
    # Note: 이 방식은 하루에 여러 종목을 동시에 추가 매수할 경우, 현금이 충분하다는 가정 하에 진행됩니다.
    # CPU 버전도 우선순위에 따라 자금을 소진하므로, 이 벡터화된 접근은 약간의 차이를 보일 수 있으나
    # 대규모 병렬처리를 위한 합리적인 트레이드오프입니다.
    portfolio_state[buy_sim_indices, 0] -= buy_total_cost
    positions_state[buy_sim_indices, buy_stock_indices, buy_split_indices, 0] = buy_quantities
    positions_state[buy_sim_indices, buy_stock_indices, buy_split_indices, 1] = buy_prices_final
    positions_state[buy_sim_indices, buy_stock_indices, buy_split_indices, 2] = current_day_idx
    
    last_trade_day_idx_state[buy_sim_indices, buy_stock_indices] = current_day_idx

    return portfolio_state, positions_state, last_trade_day_idx_state

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
        return portfolio_state, positions_state, last_trade_day_idx_state

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
    debug_mode: bool = False
):
    # --- 1. 상태 배열 초기화 ---
    num_combinations = param_combinations.shape[0]
    num_trading_days = len(trading_date_indices)
    num_tickers = len(all_tickers)
    cooldown_period_days = execution_params.get("cooldown_period_days", 5)

    portfolio_state = cp.zeros((num_combinations, 2), dtype=cp.float32)
    portfolio_state[:, 0] = initial_cash
    
    max_splits_from_params = int(cp.max(param_combinations[:, 6]).get()) if param_combinations.shape[1] > 6 else max_splits_limit
    positions_state = cp.zeros((num_combinations, num_tickers, max_splits_from_params, 3), dtype=cp.float32)
    
    cooldown_state = cp.full((num_combinations, num_tickers), -1, dtype=cp.int32)
    last_trade_day_idx_state = cp.full((num_combinations, num_tickers), -1, dtype=cp.int32)
    daily_portfolio_values = cp.zeros((num_combinations, num_trading_days), dtype=cp.float32)
    
    ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}
    all_data_reset_idx = all_data_gpu.reset_index()
    weekly_filtered_reset_idx = weekly_filtered_gpu.reset_index()
    previous_month = -1

    # --- 2. 메인 백테스팅 루프 ---
    for i, date_idx in enumerate(trading_date_indices):
        current_date = trading_dates_pd_cpu[date_idx.item()]
        
        if debug_mode and (i % 20 == 0 or i == num_trading_days - 1):
            print(f"\n--- Day {i+1}/{num_trading_days}: {current_date.strftime('%Y-%m-%d')} ---")

        # 2-1. 현재 날짜의 가격 및 후보 종목 데이터 준비
        daily_prices_series = all_data_reset_idx[all_data_reset_idx['date'] == current_date].set_index('ticker')['close_price']
        current_prices_gpu = cp.asarray(daily_prices_series.reindex(all_tickers).fillna(0).values, dtype=cp.float32)

        pos = weekly_filtered_reset_idx['date'].searchsorted(current_date, side='right')
        if pos > 0:
            latest_filter_date = weekly_filtered_reset_idx['date'][pos - 1]
            candidates_of_the_week = weekly_filtered_reset_idx[weekly_filtered_reset_idx['date'] == latest_filter_date]
            candidate_tickers_list = candidates_of_the_week['ticker'].to_arrow().to_pylist()
            
            candidate_indices = cp.array([ticker_to_idx.get(t, -1) for t in candidate_tickers_list if t in ticker_to_idx], dtype=cp.int32)
            candidate_tickers_for_day = candidate_indices[candidate_indices != -1]
            
            daily_atr_series = all_data_reset_idx[all_data_reset_idx['date'] == current_date].set_index('ticker')['atr_14_ratio']
            candidate_atrs_for_day = cp.asarray(daily_atr_series.reindex(candidate_tickers_list).fillna(0).values, dtype=cp.float32)
        else:
            candidate_tickers_for_day = cp.array([], dtype=cp.int32)
            candidate_atrs_for_day = cp.array([], dtype=cp.float32)

        # 2-2. 월별 투자금 재계산
        if current_date.month != previous_month:
            portfolio_state = _calculate_monthly_investment_gpu(
                portfolio_state, positions_state, param_combinations, current_prices_gpu
            )
            previous_month = current_date.month

        # 2-3. [수정] 신호 처리 순서를 '신규 -> 추가 -> 매도'로 변경
        portfolio_state, positions_state, last_trade_day_idx_state = _process_new_entry_signals_gpu(
            portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, i,
            cooldown_period_days, param_combinations, current_prices_gpu,
            candidate_tickers_for_day, candidate_atrs_for_day,
            execution_params["buy_commission_rate"]
        )
        
        portfolio_state, positions_state, last_trade_day_idx_state = _process_additional_buy_signals_gpu(
            portfolio_state, positions_state, last_trade_day_idx_state, i,
            param_combinations, current_prices_gpu,
            execution_params["buy_commission_rate"]
        )

        portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state = _process_sell_signals_gpu(
            portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, i,
            param_combinations, current_prices_gpu,
            execution_params["sell_commission_rate"], execution_params["sell_tax_rate"]
        )

        # 2-4. 일일 포트폴리오 가치 업데이트
        stock_quantities = cp.sum(positions_state[..., 0], axis=2)
        stock_market_values = stock_quantities * current_prices_gpu
        total_stock_value = cp.sum(stock_market_values, axis=1)
        
        daily_portfolio_values[:, i] = portfolio_state[:, 0] + total_stock_value

        if debug_mode and (i % 20 == 0 or i == num_trading_days - 1):
            capital_snapshot = portfolio_state[0, 0].get()
            stock_val_snapshot = total_stock_value[0].get()
            total_val_snapshot = daily_portfolio_values[0, i].get()
            num_pos_snapshot = cp.sum(cp.any(positions_state[0, :, :, 0] > 0, axis=1)).get()
            print(f" [GPU_HOLDINGS] {cp.where(cp.any(positions_state[0, :, :, 0] > 0, axis=1))[0].get().tolist()}")
            print(f"[END]   Capital: {capital_snapshot:,.0f} | Stock Val: {stock_val_snapshot:,.0f} | Total Val: {total_val_snapshot:,.0f} | Stocks Held: {num_pos_snapshot}")
    return daily_portfolio_values