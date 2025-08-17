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
    # [수정] float32 나눗셈에서 발생할 수 있는 미세한 오차를 보정하기 위해
    # 소수점 5자리에서 반올림(round)한 후 올림(ceil)을 적용합니다.
    # 예: 18430 / 10 = 1843.0000001 -> round -> 1843.0 -> ceil -> 1843.0
    divided = price_array / tick_size
    rounded = cp.round(divided, 5) 
    return cp.ceil(rounded) * tick_size

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
    all_tickers: list = None,
    trading_dates_pd_cpu: pd.DatetimeIndex = None,
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
        if debug_mode:
            sim0_liquidation_mask = stock_liquidation_mask[0]
            if cp.any(sim0_liquidation_mask):
                sim0_indices_to_log = cp.where(sim0_liquidation_mask)[0]
                for idx_cupy in sim0_indices_to_log:
                    idx = idx_cupy.item()
                    ticker = all_tickers[idx]
                    # 청산 기준가는 '당일 종가'
                    target_price = current_close_prices[idx].item()
                    exec_price = adjust_price_up_gpu(current_close_prices[idx]).item()
                    high_price = current_high_prices[idx].item()
                    reason = "Stop-Loss" if stock_stop_loss_mask[0, idx] else "Inactivity"
                    net_proceeds_sim0 = (quantities[0, idx, 0] * exec_price).get() # 간단한 계산
                    print(
                        f"[GPU_SELL_CALC] {trading_dates_pd_cpu[current_day_idx].strftime('%Y-%m-%d')} {ticker} | "
                        f"Qty: {quantities[0, idx, 0].item():,.0f} * ExecPrice: {exec_price:,.0f} = Revenue: {net_proceeds_sim0:,.0f}"
                    )
                    print(
                        f"[GPU_SELL_PRICE] {trading_dates_pd_cpu[current_day_idx].strftime('%Y-%m-%d')} {ticker} "
                        f"Reason: {reason} | "
                        f"Target: {target_price:.2f} -> Exec: {exec_price} | "
                        f"High: {high_price}"
                    )

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
        if debug_mode:
            sim0_profit_taking_mask = profit_taking_mask[0]
            if cp.any(sim0_profit_taking_mask):
                # 수익 실현이 발생한 [stock_idx, split_idx] 쌍을 가져옴
                sim0_stock_indices, sim0_split_indices = cp.where(sim0_profit_taking_mask)
                for i in range(len(sim0_stock_indices)):
                    stock_idx = sim0_stock_indices[i].item()
                    split_idx = sim0_split_indices[i].item()
                    
                    ticker = all_tickers[stock_idx]
                    high_price = current_high_prices[stock_idx].item()
                    target_price = target_sell_prices[0, stock_idx, split_idx].item()
                    exec_price = execution_sell_prices[0, stock_idx, split_idx].item()
                    
                    print(
                        f"[GPU_SELL_PRICE] {trading_dates_pd_cpu[current_day_idx].strftime('%Y-%m-%d')} {ticker} "
                        f"(Split {split_idx}) Reason: Profit-Taking | " # [추가] 몇 번째 차수인지 명시
                        f"Target: {target_price:.2f} -> Exec: {exec_price} | "
                        f"High: {high_price}"
                    )
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

# 기존 _process_additional_buy_signals_gpu 함수를 아래 코드로 전체 교체하십시오.

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
    """ [수정] 순차적 자금 차감을 적용하여 경쟁 조건 버그를 해결한 추가 매수 로직 """
    # --- [유지] 1. 파라미터 및 기본 상태 준비 ---
    add_buy_drop_rates = param_combinations[:, 2:3]
    max_splits_limits = param_combinations[:, 6:7]
    quantities = positions_state[..., 0]
    buy_prices = positions_state[..., 1]
    
    has_positions = quantities > 0
    num_positions = cp.sum(has_positions, axis=2)
    has_any_position = num_positions > 0
    if not cp.any(has_any_position):
        return portfolio_state, positions_state, last_trade_day_idx_state

    # --- [유지] 2. 추가 매수 조건에 맞는 모든 후보 탐색 ---
    last_pos_mask = (cp.cumsum(has_positions, axis=2) == num_positions[:, :, cp.newaxis]) & has_positions
    last_buy_prices = cp.sum(buy_prices * last_pos_mask, axis=2)
    trigger_prices = last_buy_prices * (1 - add_buy_drop_rates)
    under_max_splits = num_positions < max_splits_limits
    can_add_buy = ~sell_occurred_today_mask
    has_first_split = positions_state[..., 0, 0] > 0
    open_day_indices = positions_state[..., 2]
    first_open_day_idx = cp.where(has_positions, open_day_indices, cp.inf).min(axis=2)
    is_not_new_today = (first_open_day_idx < current_day_idx)
    
    initial_buy_mask = (current_lows <= trigger_prices) & has_any_position & under_max_splits & can_add_buy & is_not_new_today & has_first_split
    if not cp.any(initial_buy_mask):
        return portfolio_state, positions_state, last_trade_day_idx_state

    # --- 3. [핵심 수정] 후보들을 우선순위에 따라 정렬 ---
    sim_indices, stock_indices = cp.where(initial_buy_mask)
    
    # 각 후보가 속한 시뮬레이션의 'additional_buy_priority' 파라미터 값을 가져옵니다.
    # 0: lowest_order, 1: highest_drop
    add_buy_priorities = param_combinations[:, 4:5]
    priorities_for_candidates = add_buy_priorities[sim_indices].flatten()

    # 'lowest_order' 점수 계산: 현재 보유한 분할매수 차수 (오름차순 정렬 대상)
    scores_lowest_order = num_positions[sim_indices, stock_indices]

    # 'highest_drop' 점수 계산: 실제 하락률 (내림차순 정렬 대상)
    candidate_last_buy_prices = last_buy_prices[sim_indices, stock_indices]
    candidate_current_prices = current_prices[stock_indices]
    epsilon = 1e-9 # 0으로 나누기 방지
    scores_highest_drop = (candidate_last_buy_prices - candidate_current_prices) / (candidate_last_buy_prices + epsilon)
    
    # 파라미터 값에 따라 최종 우선순위 점수를 선택합니다.
    # lowest_order(0)는 오름차순 정렬해야 하므로 점수를 그대로 사용합니다.
    # highest_drop(1)은 내림차순 정렬해야 하므로, 점수에 음수를 취한 뒤 오름차순 정렬합니다.
    priority_scores = cp.where(priorities_for_candidates == 0,
                               scores_lowest_order,
                               -scores_highest_drop) # 내림차순 정렬을 위해 음수화

     # 1. 2차 정렬 기준: 후보들의 stock_idx (오름차순)
    candidate_stock_indices = stock_indices
    key2_stock_indices = candidate_stock_indices.astype(cp.float32)

    # 2. 1차 정렬 기준: 계산된 우선순위 점수 (오름차순)
    key1_priority_scores = priority_scores

    # [추가] 두 개의 1D 키 배열을 vstack을 사용해 (2, N) 형태의 단일 2D 배열로 쌓습니다.
    # lexsort는 마지막 행부터 정렬하므로, 우선순위가 낮은 키(key2)를 먼저, 높은 키(key1)를 나중에 넣습니다.
    sort_keys_array = cp.vstack((key2_stock_indices, key1_priority_scores))

    # [수정] 단일 2D 배열을 lexsort에 전달합니다.
    sorted_indices = cp.lexsort(sort_keys_array)
    
    # 정렬된 순서대로 후보 정보 재배열
    sorted_sim_indices = sim_indices[sorted_indices]
    sorted_stock_indices = stock_indices[sorted_indices]

    # --- 4. [핵심 수정] 순차적 자금 차감을 통한 최종 매수 실행 ---
    temp_capital = portfolio_state[:, 0].copy()
    
    if debug_mode:
        temp_cap_log = portfolio_state[0, 0].item()

    # 정렬된 후보들을 순회하며 하나씩 매수 시도
    for i in range(len(sorted_indices)):
        sim_idx = sorted_sim_indices[i]
        stock_idx = sorted_stock_indices[i]

        # 이 거래가 현재 자본으로 감당 가능한지 확인
        # (주의: 매번 portfolio_state 원본이 아닌 temp_capital과 비교해야 함)
        investment = portfolio_state[sim_idx, 1] # 투자금은 월별로 고정
        
        # 매수가 결정 (기존 로직과 동일)
        target_price = trigger_prices[sim_idx, stock_idx]
        high_price = current_highs[stock_idx]
        epsilon = cp.float32(1.0) # 최소 가격 단위(1원)를 안전 마진으로 사용
        price_basis = cp.where(high_price <= target_price - epsilon, high_price, target_price)
        
        # [추가] price_basis 검증을 위한 상세 로그
        if debug_mode and sim_idx == 0:
            ticker = all_tickers[stock_idx.item()]
            scenario = "B (Clear Gap Down)" if high_price.item() <= target_price.item() - epsilon.item() else "A (Touch or Close)"
            print(f"  └─ [ADD_BUY_DEBUG] Stock {stock_idx.item()}({ticker}) | Scenario: {scenario} | "
                  f"High: {high_price.item():.2f} vs Target: {target_price.item():.2f} "
                  f"-> Basis: {price_basis.item():.2f}")
            
            
        exec_price = adjust_price_up_gpu(price_basis)
        
        if exec_price <= 0: continue
        
        # 비용 계산
        quantity = cp.floor(investment / exec_price)
        if quantity <= 0: continue
        
        cost = exec_price * quantity
        commission = cp.floor(cost * buy_commission_rate)
        total_cost = cost + commission
        
        # 순차적 자본 확인
        if temp_capital[sim_idx] >= total_cost:
            # 매수 실행: 상태 업데이트
            is_empty_slot = positions_state[sim_idx, stock_idx, :, 0] == 0
            split_idx = cp.argmax(is_empty_slot)
            # 엣지 케이스: 모든 슬롯이 차있는 경우는 initial_buy_mask에서 이미 걸러짐
            
            positions_state[sim_idx, stock_idx, split_idx, 0] = quantity
            positions_state[sim_idx, stock_idx, split_idx, 1] = exec_price
            positions_state[sim_idx, stock_idx, split_idx, 2] = current_day_idx
            last_trade_day_idx_state[sim_idx, stock_idx] = current_day_idx
            
            # [핵심] 임시 자본 즉시 차감
            capital_before_buy = temp_capital[sim_idx].copy() # 로그용
            temp_capital[sim_idx] -= total_cost

            # 디버깅 로그
            if debug_mode and sim_idx == 0:
                ticker_code = all_tickers[stock_idx.item()]
                print(f"[GPU_ADD_BUY_CALC] {current_day_idx}, Sim 0, Stock {stock_idx.item()}({ticker_code}) | "
              f"Invest: {investment.item():,.0f} / ExecPrice: {exec_price.item():,.0f} = Qty: {quantity.item():,.0f}")
                # print(f"[GPU_ADD_BUY] Day {current_day_idx}, Sim 0, Stock {stock_idx.item()}({ticker_code}) | "
                #       f"Cost: {total_cost.item():,.0f} | "
                #       f"Cap Before: {capital_before_buy.item():,.0f} -> Cap After: {temp_capital[sim_idx].item():,.0f}")

    # --- 5. [유지] 최종 자본 상태 반영 ---
    portfolio_state[:, 0] = temp_capital
    return portfolio_state, positions_state, last_trade_day_idx_state

# 기존 _process_new_entry_signals_gpu 함수를 아래 코드로 전체 교체하십시오.

def _process_new_entry_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    cooldown_state: cp.ndarray,
    last_trade_day_idx_state: cp.ndarray,
    current_day_idx: int,
    cooldown_period_days: int,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    # [삭제] current_lows, current_highs
    candidate_tickers_for_day: cp.ndarray,
    candidate_atrs_for_day: cp.ndarray,
    buy_commission_rate: float,
    log_buffer: cp.ndarray,
    log_counter: cp.ndarray,
    debug_mode: bool = False,
    all_tickers: list = None
    # [삭제] trading_dates_pd_cpu
):
    # --- [유지] 0. 진입 조건 확인 ---
    has_any_position = cp.any(positions_state[..., 0] > 0, axis=2)
    current_num_stocks = cp.sum(has_any_position, axis=1)
    max_stocks_per_sim = param_combinations[:, 0]
    available_slots = cp.maximum(0, max_stocks_per_sim - current_num_stocks).astype(cp.int32)

    if not cp.any(available_slots > 0) or candidate_tickers_for_day.size == 0:
        return portfolio_state, positions_state, last_trade_day_idx_state

    # --- [유지] 1. 모든 (시뮬레이션, 후보) 쌍에 대한 기본 정보 계산 ---
    num_simulations = param_combinations.shape[0]
    num_candidates = len(candidate_tickers_for_day)
    
    # (sim, candidate) 형태의 1D 배열 생성
    sim_indices_expanded = cp.repeat(cp.arange(num_simulations), num_candidates)
    candidate_indices_in_list = cp.tile(cp.arange(num_candidates), num_simulations)
    
    # 후보 종목의 실제 티커 인덱스
    candidate_ticker_indices = candidate_tickers_for_day[candidate_indices_in_list]

    # 매수 조건 검사를 위한 배열 확장
    is_holding = has_any_position[sim_indices_expanded, candidate_ticker_indices]
    is_in_cooldown = (cooldown_state[sim_indices_expanded, candidate_ticker_indices] != -1) & \
                     ((current_day_idx - cooldown_state[sim_indices_expanded, candidate_ticker_indices]) < cooldown_period_days)
    
    # 매수 비용 일괄 계산
    investment_per_order = portfolio_state[sim_indices_expanded, 1]
    candidate_prices = current_prices[candidate_ticker_indices]
    buy_prices = adjust_price_up_gpu(candidate_prices)
    quantities = cp.floor(investment_per_order / buy_prices)
    quantities[buy_prices <= 0] = 0
    costs = buy_prices * quantities
    commissions = cp.floor(costs * buy_commission_rate)
    total_costs = costs + commissions

    # --- [유지] 2. 우선순위에 따라 후보 정렬 ---
    priority_scores = cp.full(num_simulations * num_candidates, float('inf'), dtype=cp.float32)
    initial_buy_mask = ~is_holding & ~is_in_cooldown & (quantities > 0)
    priority_scores[initial_buy_mask] = -candidate_atrs_for_day[candidate_indices_in_list[initial_buy_mask]]

    priority_scores_2d = priority_scores.reshape(num_simulations, num_candidates)
    sorted_candidate_indices_in_sim = cp.argsort(priority_scores_2d, axis=1)

    # --- 3. [핵심 수정] 순차적 자본 차감을 통한 최종 매수 실행 ---
    # CPU의 순차적 로직을 모방하기 위해, 우선순위 루프(k)를 유지하되
    # 각 루프에서 자본과 슬롯을 즉시 업데이트하여 다음 루프에 반영합니다.
    temp_capital = portfolio_state[:, 0].copy()
    temp_available_slots = available_slots.copy()
    
    # 디버깅을 위한 임시 로그 변수 (실제 계산과 분리)
    if debug_mode:
        temp_cap_log = portfolio_state[0, 0].item()

    for k in range(num_candidates):
        # k번째 우선순위 후보들의 '후보 리스트 내 인덱스'
        candidate_idx_k = sorted_candidate_indices_in_sim[:, k]
        
        # (sim, candidate) 형태의 1D 인덱스로 변환
        # 각 시뮬레이션의 k번째 우선순위 후보를 가리키는 고유 인덱스
        flat_indices_k = cp.arange(num_simulations) * num_candidates + candidate_idx_k

        # 이 후보들이 여전히 매수 가능한지 '현재 시점'의 자본과 슬롯으로 다시 확인
        can_afford = temp_capital >= total_costs[flat_indices_k]
        has_slot = temp_available_slots > 0
        
        # initial_buy_mask: 보유/쿨다운 등 기본 조건
        # can_afford / has_slot: 동적으로 변하는 자원 조건
        still_valid_mask = initial_buy_mask[flat_indices_k] & can_afford & has_slot
        
        if not cp.any(still_valid_mask):
            continue
            
        # 이번 스텝(k)에서 실제 매수가 발생하는 시뮬레이션들의 인덱스
        active_sim_indices = cp.where(still_valid_mask)[0]
        
        # 매수에 필요한 정보들을 'active_sim_indices'를 이용해 추출
        # 1. 어떤 종목을 살 것인가?
        # flat_indices_k에서 유효한 것들만 필터링
        active_flat_indices = flat_indices_k[active_sim_indices]
        final_stock_indices = candidate_ticker_indices[active_flat_indices]
        
        # 2. 얼마에, 얼마나, 총 비용은?
        final_costs = total_costs[active_flat_indices]
        final_quantities = quantities[active_flat_indices]
        final_buy_prices = buy_prices[active_flat_indices]

        # 3. 상태 업데이트
        capital_before_buy = temp_capital[active_sim_indices].copy() # 로그 기록용
        
        # [핵심] 실제 자본과 슬롯을 '즉시' 차감하여 다음 k 루프에 영향을 줌
        temp_capital[active_sim_indices] -= final_costs
        temp_available_slots[active_sim_indices] -= 1

        positions_state[active_sim_indices, final_stock_indices, 0, 0] = final_quantities
        positions_state[active_sim_indices, final_stock_indices, 0, 1] = final_buy_prices
        positions_state[active_sim_indices, final_stock_indices, 0, 2] = current_day_idx
        last_trade_day_idx_state[active_sim_indices, final_stock_indices] = current_day_idx
        
        # --- 4. [수정] 새로운 로직에 맞는 디버깅 및 에러 로깅 ---
        if debug_mode:
            sim0_mask = cp.isin(active_sim_indices, cp.array([0]))
            if cp.any(sim0_mask):
                costs_sim0 = final_costs[sim0_mask]
                stock_indices_sim0 = final_stock_indices[sim0_mask]
                buy_prices_sim0 = final_buy_prices[sim0_mask]
                quantities_sim0 = final_quantities[sim0_mask]
                
                recorded_quantities = positions_state[0, stock_indices_sim0, 0, 0].get()

                for i in range(costs_sim0.size):
                    idx = stock_indices_sim0[i].item()
                    ticker_code = all_tickers[idx]
                    cost_item = costs_sim0[i].item()
                    buy_price_val = buy_prices_sim0[i].item()
                    
                    cap_before_log = temp_cap_log
                    cap_after_log = temp_cap_log - cost_item
                    
                    
                    expected_quantity = quantities_sim0[i].item()
                    actual_quantity = recorded_quantities[i]
                    
                    print(f"[GPU_NEW_BUY_CALC] {current_day_idx}, Sim 0, Stock {idx}({ticker_code}) | "
          f"Invest: {investment_per_order[active_flat_indices[sim0_mask]][i].item():,.0f} / ExecPrice: {buy_price_val:,.0f} = Qty: {expected_quantity:,.0f}")
                    # print(f"[GPU_NEW_BUY] Day {current_day_idx}, Sim 0, Stock {idx}({ticker_code}) | "
                    #       f"Cost: {cost_item:,.0f} | "
                    #       f"Cap Before: {cap_before_log:,.0f} -> Cap After: {cap_after_log:,.0f}")
                    print(f"  └─ Executed Buy Price Saved to State: {buy_price_val:,.0f}")
                    if abs(expected_quantity - actual_quantity) > 1e-5:
                        print(f"  └─ 🚨 [VERIFICATION FAILED] Expected Quantity: {expected_quantity:,.0f}, "
                              f"Actual Quantity in State: {actual_quantity:,.0f}")
                    else:
                        print(f"  └─ ✅ [VERIFICATION PASSED] Quantity in State: {actual_quantity:,.0f}")
                    
                    temp_cap_log = cap_after_log
        else:
            # 에러 버퍼링 로직 (기존과 유사)
            error_mask = temp_capital[active_sim_indices] < 0
            if cp.any(error_mask):
                error_sim_indices = active_sim_indices[error_mask]
                num_errors = len(error_sim_indices)
                start_idx = cp.atomicAdd(log_counter, 0, num_errors)
                if start_idx + num_errors < log_buffer.shape[0]:
                    log_data = cp.vstack([
                        cp.full(num_errors, current_day_idx, dtype=cp.float32),
                        error_sim_indices.astype(cp.float32),
                        final_stock_indices[error_mask].astype(cp.float32),
                        capital_before_buy[error_mask],
                        final_costs[error_mask]
                    ]).T
                    log_buffer[start_idx : start_idx + num_errors] = log_data

    # --- [유지] 5. 최종 자본 상태 반영 ---
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
                    print(f"Warning: Day {i}, Mismatch in candidate arrays length after filtering.")
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
            all_tickers=all_tickers,
            trading_dates_pd_cpu=trading_dates_pd_cpu
        )
        
        # 확보된 자원으로 신규 종목 진입을 시도합니다.
        portfolio_state, positions_state, last_trade_day_idx_state = _process_new_entry_signals_gpu(
            portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, i,
            cooldown_period_days, param_combinations, current_prices_gpu,
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

        if debug_mode:
            capital_snapshot = portfolio_state[0, 0].get()
            stock_val_snapshot = total_stock_value[0].get()
            total_val_snapshot = daily_portfolio_values[0, i].get()
            num_pos_snapshot = cp.sum(cp.any(positions_state[0, :, :, 0] > 0, axis=1)).get()
            
            # [추가] CPU 로그와 유사한 포맷으로 출력하여 비교 용이성 증대
            header = f"\n{'='*120}\n"
            footer = f"\n{'='*120}"
            date_str = current_date.strftime('%Y-%m-%d')
            
            cash_ratio = (capital_snapshot / total_val_snapshot) * 100 if total_val_snapshot else 0
            stock_ratio = (stock_val_snapshot / total_val_snapshot) * 100 if total_val_snapshot else 0

            summary_str = (
                f"GPU STATE | Date: {date_str} | Day {i+1}/{num_trading_days}\n"
                f"{'-'*120}\n"
                f"Total Value: {total_val_snapshot:,.0f} | "
                f"Cash: {capital_snapshot:,.0f} ({cash_ratio:.1f}%) | "
                f"Stocks: {stock_val_snapshot:,.0f} ({stock_ratio:.1f}%)\n"
                f"Holdings Count: {num_pos_snapshot} Stocks"
            )
            
            log_message = header + summary_str
            
            holding_indices = cp.where(cp.any(positions_state[0, :, :, 0] > 0, axis=1))[0].get()
            if holding_indices.size > 0:
                holdings_str = ", ".join([f"{idx}({all_tickers[idx]})" for idx in holding_indices])
                log_message += f"\n[Current Holdings]\n{holdings_str}"

            log_message += footer
            print(log_message)
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