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

def adjust_price_up_gpu(price_array):
    """ Vectorized price adjustment on GPU. """
    tick_size = get_tick_size_gpu(price_array)
    return cp.ceil(price_array / tick_size) * tick_size

def _calculate_monthly_investment_gpu(portfolio_state, positions_state, param_combinations, current_prices,current_date,debug_mode):
    """ Vectorized calculation of monthly investment amounts based on current market value. """
    if debug_mode:
        print("\n" + "-"*25)
        print(f"DEBUG: Monthly Rebalance Triggered on {current_date.strftime('%Y-%m-%d')}")
        print("-"*25)

    quantities = positions_state[..., 0]
    
    # [수정] 총 자산 계산 시 매수 평단이 아닌 '현재가'를 사용해야 합니다.
    total_quantities_per_stock = cp.sum(quantities, axis=2)
    stock_market_values = total_quantities_per_stock * current_prices
    total_stock_values = cp.sum(stock_market_values, axis=1, keepdims=True)

    capital_array = portfolio_state[:, 0:1]
    total_portfolio_values = capital_array + total_stock_values
    
    order_investment_ratios = param_combinations[:, 1:2]
    investment_per_order = total_portfolio_values * order_investment_ratios
    if debug_mode:
        sim0_capital = capital_array[0, 0].item()
        sim0_stock_value = total_stock_values[0, 0].item()
        sim0_total_value = total_portfolio_values[0, 0].item()
        sim0_investment_per_order = investment_per_order[0, 0].item()
        
        # 보유 종목의 가격이 0인지 확인하는 핵심 로그
        holding_mask = total_quantities_per_stock[0] > 0
        sim0_holding_quantities = total_quantities_per_stock[0, holding_mask].get()
        sim0_holding_prices = current_prices[holding_mask].get() # current_prices는 1D 배열

        print(f"  Capital (Sim 0)        : {sim0_capital:,.0f}")
        if sim0_holding_quantities.size > 0:
            print(f"  Holding Qtys (Sim 0)     : {sim0_holding_quantities}")
            print(f"  Prices for Holdings (Sim 0): {sim0_holding_prices}")
        print(f"  Total Stock Value (Sim 0): {sim0_stock_value:,.0f}")
        print(f"  Total Portfolio (Sim 0): {sim0_total_value:,.0f}")
        print(f"  => New Investment/Order : {sim0_investment_per_order:,.0f}")
        print("-"*25 + "\n")
    portfolio_state[:, 1:2] = investment_per_order
    return portfolio_state

def _process_sell_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    cooldown_state: cp.ndarray,
    last_trade_day_idx_state: cp.ndarray,
    current_day_idx: int,
    param_combinations: cp.ndarray,
    current_close_prices: cp.ndarray,   # 종가(기존 용도 유지)
    current_high_prices: cp.ndarray,    # intraday high (익절 비교용)
    sell_commission_rate: float,
    sell_tax_rate: float,
    debug_mode: bool = False,
    all_tickers: list = None
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
        # [추가] 당일 매도가 없으므로 False 마스크를 반환
        sell_occurred_stock_mask = cp.zeros((positions_state.shape[0], positions_state.shape[1]), dtype=cp.bool_)
        return portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, sell_occurred_stock_mask

    # --- 파라미터 로드 ---
    sell_profit_rates = param_combinations[:, 3:4, cp.newaxis]
    stop_loss_rates = param_combinations[:, 5:6, cp.newaxis]
    max_inactivity_periods = param_combinations[:, 7:8] # 최대 매매 미발생 기간
    cost_factor = 1.0 - sell_commission_rate - sell_tax_rate
    
    # 이 날에 매도가 발생한 종목을 추적하기 위한 마스크 (쿨다운 관리용)
    sell_occurred_stock_mask = cp.zeros((positions_state.shape[0], positions_state.shape[1]), dtype=cp.bool_)

    # --- 시나리오 1: 전체 청산 (손절매 또는 최대 매매 미발생생 기간) ---
    # (sim, stock) 형태로 현재가를 브로드캐스팅 준비
    current_prices_2d = cp.broadcast_to(current_close_prices, (positions_state.shape[0], positions_state.shape[1]))
    
    # --- 시나리오 1: 전체 청산 (손절매 또는 최대 매매 미발생 기간) ---
    total_quantities = cp.sum(quantities, axis=2)
    has_any_position = total_quantities > 0
    
    # 평균 매수가 계산 (0으로 나누기 방지)
    safe_total_quantities = cp.where(has_any_position, total_quantities, 1)
    avg_buy_prices = cp.sum(buy_prices * quantities, axis=2) / safe_total_quantities
     # 손절매 조건
    stock_stop_loss_mask = (current_prices_2d <= avg_buy_prices * (1 + stop_loss_rates.squeeze(-1))) & has_any_position
    
    # 비활성 기간 조건
    has_traded_before = last_trade_day_idx_state != -1
    days_inactive = current_day_idx - last_trade_day_idx_state
    stock_inactivity_mask = (days_inactive > max_inactivity_periods) & has_traded_before
    
    stock_liquidation_mask = stock_stop_loss_mask | stock_inactivity_mask
    
    if debug_mode and cp.any(stock_liquidation_mask):
        sim0_stop_loss_indices = cp.where(stock_stop_loss_mask[0])[0].get()
        sim0_inactivity_indices = cp.where(stock_inactivity_mask[0])[0].get()
        # [수정] 인덱스를 티커로 변환하여 로그 출력
        if sim0_stop_loss_indices.size > 0:
            tickers_str = ", ".join([f"{idx}({all_tickers[idx]})" for idx in sim0_stop_loss_indices])
            print(f"  [GPU_SELL_DEBUG] Day {current_day_idx}: Stop-Loss triggered for Stocks [{tickers_str}]")
        if sim0_inactivity_indices.size > 0:
            tickers_str = ", ".join([f"{idx}({all_tickers[idx]})" for idx in sim0_inactivity_indices])
            print(f"  [GPU_SELL_DEBUG] Day {current_day_idx}: Inactivity triggered for Stocks [{tickers_str}]")
    if cp.any(stock_liquidation_mask):
        broadcasted_close_prices = cp.broadcast_to(current_close_prices.reshape(1, -1, 1), buy_prices.shape)
        adjusted_liquidation_prices = adjust_price_up_gpu(broadcasted_close_prices)

        # 청산 대상 종목의 모든 포지션에 대한 수익 계산
        revenue_matrix = quantities * adjusted_liquidation_prices
        # 청산 대상 종목(stock_liquidation_mask)만 필터링하여 수익 계산
        liquidation_revenue = cp.sum(revenue_matrix * stock_liquidation_mask[:, :, cp.newaxis], axis=(1, 2))
        

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
    # [유지] 목표가 계산은 이미 단순 계산 방식으로 구현되어 있습니다.
    target_sell_prices = buy_prices * (1 + sell_profit_rates)
    # [유지] 실제 체결가는 목표가를 호가 단위에 맞게 올림 처리합니다.
    execution_sell_prices = adjust_price_up_gpu(target_sell_prices)

    # [수정] 체결 조건: 당일 '고가(high)'가 계산된 체결가에 도달했는지 확인하도록 변경
    high_prices_3d = cp.broadcast_to(current_high_prices.reshape(1, -1, 1), buy_prices.shape) # [수정] close_prices 대신 high_prices 사용
    
    # [유지] 현실적인 백테스팅을 위해 당일(T0) 매수분은 매도 금지
    open_day_idx = positions_state[..., 2]
    sellable_time_mask = open_day_idx < current_day_idx

    # [수정] 체결 마스크 생성 시 high_prices_3d를 사용합니다.
    profit_taking_mask = (high_prices_3d >= execution_sell_prices) & valid_positions & sellable_time_mask

    if debug_mode and cp.any(profit_taking_mask):
        sim0_profit_taking_indices = cp.where(cp.any(profit_taking_mask[0], axis=1))[0].get()
        # [수정] 인덱스를 티커로 변환하여 로그 출력
        if sim0_profit_taking_indices.size > 0:
            tickers_str = ", ".join([f"{idx}({all_tickers[idx]})" for idx in sim0_profit_taking_indices])
            print(f"  [GPU_SELL_DEBUG] Day {current_day_idx}: Profit-Taking triggered for Stocks [{tickers_str}]")
    if cp.any(profit_taking_mask):
        # 수익 실현 금액은 'exec_prices'로 계산
        revenue_matrix = quantities * execution_sell_prices

        # profit_taking_mask가 True인 차수들의 수익만 합산
        total_profit_revenue = cp.sum(revenue_matrix * profit_taking_mask, axis=(1, 2))

        # 비용은 매출액에 일괄 곱(벡터화) — CPU와 동일 효과
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
    return portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, sell_occurred_stock_mask

def _process_additional_buy_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    last_trade_day_idx_state: cp.ndarray,
    sell_occurred_today_mask: cp.ndarray, 
    current_day_idx: int,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    current_lows: cp.ndarray,
    current_highs: cp.ndarray,
    buy_commission_rate: float,
    log_buffer: cp.ndarray,
    log_counter: cp.ndarray,
    debug_mode: bool = False,
    all_tickers: list = None
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
    can_add_buy = ~sell_occurred_today_mask
    
    # [추가] "1차 포지션 존재" 규칙: 1차 매수 수량이 0보다 커야 함
    has_first_split = positions_state[..., 0, 0] > 0

    # [추가] 오늘 신규 진입한 종목은 추가 매수 대상에서 제외 (CPU 로직과 동기화)
    # 1차 매수일이 오늘보다 이전이어야 함.
    open_day_indices = positions_state[..., 2]
    # 각 종목의 첫 번째 포지션(1차 매수)의 개설일 (포지션이 없으면 0)
    # 포지션이 있는 곳은 개설일, 없는 곳은 무한대(inf)로 채워 최소값 계산 시 영향을 주지 않도록 함
    first_open_day_idx = cp.where(has_positions, open_day_indices, cp.inf).min(axis=2)
    is_not_new_today = (first_open_day_idx < current_day_idx)
    
    
    # [수정] is_not_new_today 조건을 최종 마스크에 추가합니다.
    additional_buy_mask = (current_lows <= trigger_prices) & has_any_position & under_max_splits & can_add_buy & is_not_new_today & has_first_split
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
    
    # [수정] 더욱 정교해진 하이브리드 매수 가격 결정 로직 (벡터화)
    targets_for_buy = trigger_prices[sim_indices, stock_indices]
    highs_for_buy = current_highs[stock_indices]
    closes_for_buy = current_prices[stock_indices] # '종가'는 current_prices_gpu 입니다.
    
    # [추가] 조건부 가격 선택: 당일 고가가 목표 매수가보다 낮으면(True) 종가를, 아니면(False) 목표 매수가를 사용
    use_close_price_mask = highs_for_buy < targets_for_buy
    prices_for_buy = cp.where(use_close_price_mask, closes_for_buy, targets_for_buy)
    
    buy_prices_adjusted = adjust_price_up_gpu(prices_for_buy)
    
    # [수정] 수량 계산은 '종가' 기준이 아닌 투자금 기준이므로 변경 없음.
    investment_per_order = portfolio_state[sim_indices, 1]
    quantities_to_buy = cp.floor(investment_per_order / buy_prices_adjusted)
    quantities_to_buy[buy_prices_adjusted <= 0] = 0
    
    cost = buy_prices_adjusted * quantities_to_buy
    commission = cp.floor(cost * buy_commission_rate)
    total_cost = cost + commission
    
    can_afford = portfolio_state[sim_indices, 0] >= total_cost

    final_buy_mask = (quantities_to_buy > 0) & can_afford # [수정] is_within_range 조건 추가

    if not cp.any(final_buy_mask):
        return portfolio_state, positions_state, last_trade_day_idx_state

    # 실제 매수할 대상만 필터링
    buy_sim_indices = sim_indices[final_buy_mask]
    buy_stock_indices = stock_indices[final_buy_mask]
    buy_quantities = quantities_to_buy[final_buy_mask]
    buy_prices_final = buy_prices_adjusted[final_buy_mask]
    buy_total_cost = total_cost[final_buy_mask]
    buy_split_indices = next_split_indices[buy_sim_indices, buy_stock_indices]
    # --- [추가] 5. 상태 업데이트 및 로깅 ---
    capital_before_buy = portfolio_state[buy_sim_indices, 0].copy()
    # [수정] if/else 구조로 변경하여 debug_mode에 따른 로깅을 명확히 함
    if debug_mode:
        # [추가] 상세 디버깅 로그를 위한 데이터 추출
        # prices_for_buy는 GPU 메모리에만 존재하므로, final_buy_mask를 이용해 다시 계산
        sim0_buy_mask = (sim_indices == 0) & final_buy_mask
        if cp.any(sim0_buy_mask):
            sim0_stock_indices = stock_indices[sim0_buy_mask]
            
            # 상세 로그 출력
            print("  [GPU_ADD_BUY_DEBUG] ---------- Additional Buy Details (Sim 0) ----------")
            # CuPy 배열을 순회하면 성능 저하가 있지만, 디버그 모드에서 소량의 데이터에만 적용되므로 허용 가능
            for stock_idx in sim0_stock_indices:
                idx = stock_idx.item()
                ticker_code = all_tickers[idx] # [추가] 티커 코드 조회
                trigger = trigger_prices[0, idx].item()
                high = current_highs[idx].item()
                close = current_prices[idx].item()
                
                # 시나리오 B (갭 하락) 조건
                if high < trigger:
                    price_basis = close
                    scenario = "B (Gap Down)"
                else: # 시나리오 A (스침)
                    price_basis = trigger
                    scenario = "A (Touch)"

                exec_price = adjust_price_up_gpu(cp.array(price_basis)).item()
                
                
                print(f"    - Stock {idx}({ticker_code}): Trigger={trigger:,.0f}, High={high:,.0f}, Close={close:,.0f} | Scenario: {scenario} -> Basis={price_basis:,.0f} -> Exec Price={exec_price:,.0f}")
            print("  --------------------------------------------------------------------")

        sim0_mask = buy_sim_indices == 0
        if cp.any(sim0_mask):
            costs_sim0 = buy_total_cost[sim0_mask]
            cap_before_sim0 = capital_before_buy[sim0_mask]
            stock_indices_sim0 = buy_stock_indices[sim0_mask]

            # 하루에 여러 종목 추가매수가 가능하므로 루프를 사용해 출력
            temp_cap = portfolio_state[0, 0].item()
            for i in range(costs_sim0.size):
                cost_item = costs_sim0[i].item()
                idx = stock_indices_sim0[i].item()
                ticker_code = all_tickers[idx]   
                # 로그 출력 시점에서만 임시로 자본 계산
                temp_cap_after = temp_cap - cost_item
                print(f"[GPU_ADD_BUY] Day {current_day_idx}, Sim 0, Stock {idx}({ticker_code}) | "
                      f"Cost: {cost_item:,.0f} | "
                      f"Cap Before: {temp_cap:,.0f} -> Cap After: {temp_cap_after:,.0f}")
                # 다음 로그를 위해 임시 자본 업데이트
                temp_cap = temp_cap_after
    else: 
        # [수정] 생략되었던 에러 버퍼링 로직의 전체 코드입니다.
        capital_after_buy_prediction = portfolio_state[buy_sim_indices, 0] - buy_total_cost
        error_mask = capital_after_buy_prediction < 0
        num_errors = cp.sum(error_mask).item()

        if num_errors > 0:
            error_sim_indices = buy_sim_indices[error_mask]
            error_capital_before = capital_before_buy[error_mask]
            error_costs = buy_total_cost[error_mask]
            error_stock_indices = buy_stock_indices[error_mask]
            
            start_idx = cp.atomicAdd(log_counter, 0, num_errors)
            
            if start_idx + num_errors < log_buffer.shape[0]:
                log_data = cp.vstack([
                    cp.full(num_errors, current_day_idx, dtype=cp.float32),
                    error_sim_indices.astype(cp.float32),
                    error_stock_indices.astype(cp.float32),
                    error_capital_before,
                    error_costs
                ]).T
                log_buffer[start_idx : start_idx + num_errors] = log_data
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
    current_lows: cp.ndarray,
    current_highs: cp.ndarray,
    candidate_tickers_for_day: cp.ndarray,
    candidate_atrs_for_day: cp.ndarray,
    buy_commission_rate: float,
    log_buffer: cp.ndarray,
    log_counter: cp.ndarray,
    debug_mode: bool = False,
    all_tickers: list = None
):
    has_any_position = cp.any(positions_state[..., 0] > 0, axis=2)
    current_num_stocks = cp.sum(has_any_position, axis=1)
    max_stocks_per_sim = param_combinations[:, 0]
    
    available_slots = cp.maximum(0, max_stocks_per_sim - current_num_stocks).astype(cp.int32)

    if not cp.any(available_slots > 0) or candidate_tickers_for_day.size == 0:
        return portfolio_state, positions_state, last_trade_day_idx_state
    # --- 1. 모든 후보 종목에 대한 매수 조건 일괄 계산 ---
    num_candidates = len(candidate_tickers_for_day)
    # (sim, candidate) 형태로 배열 확장
    sim_indices = cp.arange(param_combinations.shape[0])
    
    # 각 시뮬레이션의 정보를 후보 종목 수만큼 복제/확장
    available_slots_expanded = cp.repeat(cp.maximum(0, max_stocks_per_sim - current_num_stocks), num_candidates)
    investment_per_order_expanded = cp.repeat(portfolio_state[:, 1], num_candidates)
    capital_expanded = cp.repeat(portfolio_state[:, 0], num_candidates)
    
    # 각 후보 종목의 정보를 시뮬레이션 수만큼 복제/확장
    candidate_indices_expanded = cp.tile(candidate_tickers_for_day, len(sim_indices))
    candidate_atrs_expanded = cp.tile(candidate_atrs_for_day, len(sim_indices))
    
    # 후보 종목들의 현재가, 보유여부, 쿨다운 여부 조회
    candidate_prices = current_prices[candidate_indices_expanded]
    candidate_lows = current_lows[candidate_indices_expanded]      # [추가] 후보 종목의 저가 배열 생성
    candidate_highs = current_highs[candidate_indices_expanded]     # [추가] 후보 종목의 고가 배열 생성
    is_holding = has_any_position[cp.repeat(sim_indices, num_candidates), candidate_indices_expanded]
    is_in_cooldown = (cooldown_state[cp.repeat(sim_indices, num_candidates), candidate_indices_expanded] != -1) & \
                     ((current_day_idx - cooldown_state[cp.repeat(sim_indices, num_candidates), candidate_indices_expanded]) < cooldown_period_days)
    # 매수 비용 계산
    buy_prices = adjust_price_up_gpu(candidate_prices)
    quantities = cp.floor(investment_per_order_expanded / buy_prices)
    quantities[buy_prices <= 0] = 0
    
    costs = buy_prices * quantities
    commissions = cp.floor(costs * buy_commission_rate)
    total_costs = costs + commissions

     # 최종 매수 가능 마스크
    # [수정] is_within_range 조건을 최종 마스크에 추가합니다.
    buy_mask = (available_slots_expanded > 0) & ~is_holding & ~is_in_cooldown & (capital_expanded >= total_costs) & (quantities > 0)
    # --- 2. 우선순위에 따라 실제 매수 대상 선정 ---
    # 우선순위 점수 계산 (ATR 높은 순, 점수가 낮을수록 우선)
    priority_scores = cp.full_like(candidate_atrs_expanded, float('inf'))
    priority_scores[buy_mask] = -candidate_atrs_expanded[buy_mask] # ATR이 높을수록 점수가 낮아짐

    # (sim, candidate) 형태로 재구성
    priority_scores_2d = priority_scores.reshape(len(sim_indices), num_candidates)
    
    # 각 시뮬레이션별로 우선순위가 높은 후보의 인덱스 정렬
    sorted_candidate_indices_in_sim = cp.argsort(priority_scores_2d, axis=1)

    # --- 3. 순차적 자본 차감을 통한 최종 매수 실행 ---
    # for 루프를 사용하지만, 이는 시뮬레이션이 아닌 '매수 순서'를 위한 루프이며 훨씬 빠름
    temp_capital = portfolio_state[:, 0].copy()
    temp_available_slots = available_slots.copy()
    for k in range(num_candidates): # 우선순위 k번째 후보부터 순차적으로 검사
        # k번째 우선순위 후보들의 정보
        candidate_indices_k = sorted_candidate_indices_in_sim[:, k] # 각 sim별 k번째 후보의 '후보 리스트 내 인덱스'
        
        # (sim, candidate) 형태의 1D 인덱스로 변환
        flat_indices_k = sim_indices * num_candidates + candidate_indices_k
        
        # 이 후보들이 여전히 매수 가능한지 다시 확인 (업데이트된 자본 기준)
        can_afford = temp_capital >= total_costs[flat_indices_k]
        has_slot = temp_available_slots > 0
        still_valid_mask = buy_mask[flat_indices_k] & can_afford & has_slot[sim_indices]
        
        if not cp.any(still_valid_mask): continue
            
        # 실제 매수가 발생하는 sim 인덱스
        active_sim_indices = sim_indices[still_valid_mask]
        
        # 실제 매수할 종목의 '전체 종목 리스트 내 인덱스'
        final_stock_indices = candidate_indices_expanded[flat_indices_k[still_valid_mask]]
        
        # 매수 정보
        final_costs = total_costs[flat_indices_k[still_valid_mask]]
        final_quantities = quantities[flat_indices_k[still_valid_mask]]
        final_buy_prices = buy_prices[flat_indices_k[still_valid_mask]]

        # --- [수정] 상태 업데이트 블록 ---
        capital_before_buy = temp_capital[active_sim_indices].copy()
        
        temp_capital[active_sim_indices] -= final_costs
        
        positions_state[active_sim_indices, final_stock_indices, 0, 0] = final_quantities
        positions_state[active_sim_indices, final_stock_indices, 0, 1] = final_buy_prices
        positions_state[active_sim_indices, final_stock_indices, 0, 2] = current_day_idx
        
        last_trade_day_idx_state[active_sim_indices, final_stock_indices] = current_day_idx

        # [추가] 매수가 발생한 시뮬레이션의 available_slots를 즉시 1 감소시킴
        temp_available_slots[active_sim_indices] -= 1
        # 2. 조건부 로깅
        if debug_mode:
            # [수정] 'buy_sim_indices'를 올바른 변수인 'active_sim_indices'로 변경합니다.
            sim0_mask = active_sim_indices == 0
            if cp.any(sim0_mask):
                # .get()을 호출하면 CPU로 데이터가 넘어와 동기화 문제가 발생할 수 있으므로,
                # boolean 마스킹을 끝까지 유지한 후 최소한의 데이터만 가져옵니다.
                costs_sim0 = final_costs[sim0_mask]
                cap_before_sim0 = capital_before_buy[sim0_mask]
                # cap_after는 루프 내에서 계속 변하므로 active_sim_indices로 필터링
                cap_after_sim0 = temp_capital[active_sim_indices[sim0_mask]]
                stock_indices_sim0 = final_stock_indices[sim0_mask]
                
                # 하루에 여러 종목 매수가 가능하므로 루프를 사용해 출력
                for i in range(costs_sim0.size):
                    idx = stock_indices_sim0[i].item() # [추가]
                    ticker_code = all_tickers[idx]    # [추가]
                    print(f"[GPU_NEW_BUY] Day {current_day_idx}, Sim 0, Stock {idx}({ticker_code}) | "
                          f"Cost: {costs_sim0[i].item():,.0f} | "
                          f"Cap Before: {cap_before_sim0[i].item():,.0f} -> Cap After: {cap_after_sim0[i].item():,.0f}")
        else: # 에러 버퍼링 모드
            capital_after_buy = temp_capital[active_sim_indices]
            error_mask = capital_after_buy < 0
            num_errors = cp.sum(error_mask).item()
            if num_errors > 0:
                error_sim_indices = active_sim_indices[error_mask]
                error_stock_indices = final_stock_indices[error_mask]
                error_capital_before = capital_before_buy[error_mask]
                error_costs = final_costs[error_mask]

                start_idx = cp.atomicAdd(log_counter, 0, num_errors)
                if start_idx + num_errors < log_buffer.shape[0]:
                    log_data = cp.vstack([
                        cp.full(num_errors, current_day_idx, dtype=cp.float32),
                        error_sim_indices.astype(cp.float32),
                        error_stock_indices.astype(cp.float32),
                        error_capital_before,
                        error_costs
                    ]).T
                    log_buffer[start_idx : start_idx + num_errors] = log_data
        
    # 최종적으로 업데이트된 자본을 원래 상태 배열에 반영
    portfolio_state[:, 0] = temp_capital
    
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
    # [추가] 로그 버퍼 및 카운터 초기화
    # 포맷: [day, sim_idx, stock_idx, capital_before, cost]
    log_buffer = cp.zeros((1000, 5), dtype=cp.float32)
    log_counter = cp.zeros(1, dtype=cp.int32)
    
    ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}
    all_data_reset_idx = all_data_gpu.reset_index()
    weekly_filtered_reset_idx = weekly_filtered_gpu.reset_index()
    # [수정] CPU의 asof() 동작을 GPU에서 정확히 구현하는 3단계 로직
    print("Creating full timeseries grid to simulate CPU's asof logic...")
    
    # 1. 모든 종목과 모든 영업일을 조합하여 전체 그리드 생성
    full_grid = cudf.MultiIndex.from_product(
        [trading_dates_pd_cpu, all_tickers], names=['date', 'ticker']
    ).to_frame(index=False)
    
    # 2. 전체 그리드에 실제 데이터를 left-merge
    #    (실제 데이터가 없는 날짜는 NaN 값을 갖는 행이 생성됨)
    #    'ticker' 컬럼 타입을 통일하여 merge 오류 방지
    all_data_reset_idx['ticker'] = all_data_reset_idx['ticker'].astype('str')
    full_grid['ticker'] = full_grid['ticker'].astype('str')
    merged_data = cudf.merge(full_grid, all_data_reset_idx, on=['date', 'ticker'], how='left')

    # 3. 종목별로 그룹화하여 forward-fill 적용
    merged_data = merged_data.sort_values(by=['ticker', 'date'])
    
    # [수정] 키 컬럼('date', 'ticker')과 값 컬럼을 분리하여 처리 후 재결합
    
    # 3-1. 키 컬럼과 인덱스 보존
    key_cols = merged_data[['date', 'ticker']]
    
    # 3-2. 값 컬럼만 선택하여 ffill 및 bfill 수행
    value_cols = merged_data.drop(columns=['date', 'ticker'])
    filled_values = value_cols.groupby(merged_data['ticker']).ffill()
    
    # 3-3. 보존했던 키 컬럼과 채워진 값 컬럼을 다시 결합
    all_data_filled = cudf.concat([key_cols, filled_values], axis=1)
    
    all_data_reset_idx = all_data_filled.dropna().copy()
    # [추가] <<<<<<< 이 블록을 추가해주세요 >>>>>>>
    print("\n" + "="*80)
    print(f"[GPU DATA-PROBE] 2020-03-17 분기점 분석: ffill 완료 후 데이터 상태")
    print("="*80)
    # CPU/GPU가 서로 다르게 선택했던 종목들을 모두 포함하여 비교
    # GPU가 매수한 종목(234, 267)과 CPU가 매수한 종목(오디텍:080520, 비상교육:100220)을 확인
    try:
        # 이 인덱스는 실제 실행 시 all_tickers 리스트에 따라 달라질 수 있으므로, 방어적으로 코딩
        tickers_to_probe = []
        gpu_bought_indices = [234, 267]
        for idx in gpu_bought_indices:
            if idx < len(all_tickers):
                tickers_to_probe.append(all_tickers[idx])
        
        cpu_bought_tickers = ['080520', '100220']
        tickers_to_probe.extend(cpu_bought_tickers)
        
        # 중복 제거
        tickers_to_probe = sorted(list(set(tickers_to_probe)))
        
        # ffill이 완료된 데이터셋에서 해당 종목들의 2020-03-17 데이터를 조회
        probe_df = all_data_filled[
            (all_data_filled['date'] == '2020-03-17') &
            (all_data_filled['ticker'].isin(tickers_to_probe))
        ]
        
        print("ffill된 데이터셋 조회 결과:")
        print(probe_df.to_pandas().to_string(index=False))

    except Exception as e:
        print(f"GPU 데이터 프로브 중 오류 발생: {e}")
    print("="*80 + "\n")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    print("Full timeseries grid created and filled.")

    previous_month = -1

    # --- 2. 메인 백테스팅 루프 ---
    for i, date_idx in enumerate(trading_date_indices):
        current_date = trading_dates_pd_cpu[date_idx.item()]
        # --- [추가] 데이터 비교를 위한 디버깅 로그 ---
        debug_ticker = '013570'
        if debug_ticker in ticker_to_idx:
            debug_ticker_idx = ticker_to_idx[debug_ticker]
            daily_df = all_data_reset_idx[all_data_reset_idx['date'] == current_date]
            
            # 해당 날짜에 해당 티커 데이터가 있는지 확인
            ticker_data = daily_df[daily_df['ticker'] == debug_ticker]
            if not ticker_data.empty:
                # cudf.Series에서 스칼라 값을 안전하게 추출
                o_price = ticker_data['open_price'].iloc[0]
                h_price = ticker_data['high_price'].iloc[0]
                l_price = ticker_data['low_price'].iloc[0]
                c_price = ticker_data['close_price'].iloc[0]
                print(f"[GPU_DATA_DEBUG] {current_date.strftime('%Y-%m-%d')} | {debug_ticker} | "
                      f"Open={o_price}, High={h_price}, Low={l_price}, Close={c_price}")
        if debug_mode and (i % 20 == 0 or i == num_trading_days - 1):
            print(f"\n--- Day {i+1}/{num_trading_days}: {current_date.strftime('%Y-%m-%d')} ---")

        # 2-1. 현재 날짜의 가격 및 후보 종목 데이터 준비
        # daily_prices_series = all_data_reset_idx[all_data_reset_idx['date'] == current_date].set_index('ticker')['close_price']
        # current_prices_gpu = cp.asarray(daily_prices_series.reindex(all_tickers).fillna(0).values, dtype=cp.float32)
        # [추가] high도 함께 로드
        daily_df = all_data_reset_idx[all_data_reset_idx['date'] == current_date].set_index('ticker')
        daily_close_series = daily_df['close_price']
        daily_high_series  = daily_df['high_price']
        daily_low_series   = daily_df['low_price']

        current_prices_gpu = cp.asarray(daily_close_series.reindex(all_tickers).fillna(0).values, dtype=cp.float32)
        current_highs_gpu  = cp.asarray(daily_high_series .reindex(all_tickers).fillna(0).values, dtype=cp.float32)
        current_lows_gpu   = cp.asarray(daily_low_series.reindex(all_tickers).fillna(0).values, dtype=cp.float32)
    
        past_or_equal_data = weekly_filtered_reset_idx[weekly_filtered_reset_idx['date'] < current_date]
        if not past_or_equal_data.empty:
            latest_filter_date = past_or_equal_data['date'].max()
            candidates_of_the_week = weekly_filtered_reset_idx[weekly_filtered_reset_idx['date'] == latest_filter_date]
            candidate_tickers_list = candidates_of_the_week['ticker'].to_arrow().to_pylist()
            
            daily_atr_series = all_data_reset_idx[all_data_reset_idx['date'] == current_date].set_index('ticker')['atr_14_ratio']
            valid_candidate_atr_series = daily_atr_series.reindex(candidate_tickers_list).dropna()
            if not valid_candidate_atr_series.empty:
                # 2. 유효한 종목 코드와 ATR 값을 각각 추출
                valid_tickers = valid_candidate_atr_series.index.to_arrow().to_pylist()
                valid_atrs = valid_candidate_atr_series.values
                
                # 3. 유효한 종목 코드를 ticker_idx로 변환하여 최종 후보 배열 생성
                candidate_indices = cp.array([ticker_to_idx.get(t, -1) for t in valid_tickers if t in ticker_to_idx], dtype=cp.int32)
                
                # 4. 최종 후보 종목과 순서가 동일한 ATR 배열 생성
                candidate_tickers_for_day = candidate_indices[candidate_indices != -1]
                candidate_atrs_for_day = cp.asarray(valid_atrs, dtype=cp.float32)

                # [방어 코드] 만약의 경우를 대비해 두 배열의 길이가 같은지 확인
                if len(candidate_tickers_for_day) != len(candidate_atrs_for_day):
                    # 이 경우는 거의 발생하지 않지만, 발생 시 디버깅을 위해 경고 추가
                    print(f"Warning: Day {current_day_idx}, Mismatch in candidate arrays length after filtering.")
                    min_len = min(len(candidate_tickers_for_day), len(candidate_atrs_for_day))
                    candidate_tickers_for_day = candidate_tickers_for_day[:min_len]
                    candidate_atrs_for_day = candidate_atrs_for_day[:min_len]
            else:
                candidate_tickers_for_day = cp.array([], dtype=cp.int32)
                candidate_atrs_for_day = cp.array([], dtype=cp.float32)
        else:
            candidate_tickers_for_day = cp.array([], dtype=cp.int32)
            candidate_atrs_for_day = cp.array([], dtype=cp.float32)

        # 2-2. 월별 투자금 재계산
        if current_date.month != previous_month:
            portfolio_state = _calculate_monthly_investment_gpu(
                portfolio_state, positions_state, param_combinations, current_prices_gpu,current_date,debug_mode
            )
            previous_month = current_date.month

         # 매도를 먼저 처리하여 현금과 포트폴리오 슬롯을 확보합니다.
        portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, sell_occurred_today_mask = _process_sell_signals_gpu(
            portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, i,
            param_combinations, 
            current_prices_gpu,                       # current_close_prices 역할
            current_highs_gpu,                        # current_high_prices 역할
            execution_params["sell_commission_rate"], 
            execution_params["sell_tax_rate"],
            debug_mode=debug_mode,
            all_tickers=all_tickers
        )
        
        # 확보된 자원으로 신규 종목 진입을 시도합니다.
        portfolio_state, positions_state, last_trade_day_idx_state = _process_new_entry_signals_gpu(
            portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, i,
            cooldown_period_days, param_combinations, current_prices_gpu,current_lows_gpu,current_highs_gpu,
            candidate_tickers_for_day, candidate_atrs_for_day,
            execution_params["buy_commission_rate"],
            log_buffer, log_counter, debug_mode, all_tickers=all_tickers
        )
        
        # 마지막으로 기존 보유 종목의 추가 매수를 처리합니다.
        portfolio_state, positions_state, last_trade_day_idx_state = _process_additional_buy_signals_gpu(
            portfolio_state, positions_state, last_trade_day_idx_state,sell_occurred_today_mask, i,
            param_combinations, current_prices_gpu,current_lows_gpu,current_highs_gpu,
            execution_params["buy_commission_rate"],
            log_buffer, log_counter, debug_mode, all_tickers=all_tickers
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
            holding_indices = cp.where(cp.any(positions_state[0, :, :, 0] > 0, axis=1))[0].get()
            holdings_str = ", ".join([f"{idx}({all_tickers[idx]})" for idx in holding_indices])
            print(f" [GPU_HOLDINGS] [{holdings_str}]")
            print(f"[END]   Capital: {capital_snapshot:,.0f} | Stock Val: {stock_val_snapshot:,.0f} | Total Val: {total_val_snapshot:,.0f} | Stocks Held: {num_pos_snapshot}")
            
    # [추가] 루프 종료 후, 에러 로그 분석 및 출력
    if not debug_mode and log_counter[0] > 0:
        print("\n" + "="*60)
        print("⚠️  [GPU KERNEL WARNING] Negative Capital Detected!")
        print("="*60)
        num_logs = min(log_counter[0].item(), 1000)
        logs_cpu = pd.DataFrame(
            log_buffer[:num_logs].get(),
            columns=['Day_Idx', 'Sim_Idx', 'Stock_Idx', 'Capital_Before', 'Cost']
        )
        print(f"Total {num_logs} instances of negative capital occurred. Showing first 10:")
        # 정수형으로 변환하여 가독성 향상
        for col in ['Day_Idx', 'Sim_Idx', 'Stock_Idx']:
            logs_cpu[col] = logs_cpu[col].astype(int)
        print(logs_cpu.head(10).to_string(index=False))
        print("\n[Analysis] This suggests that on certain days, multiple parallel buy orders consumed more capital than available.")
        print("="*60)
    return daily_portfolio_values