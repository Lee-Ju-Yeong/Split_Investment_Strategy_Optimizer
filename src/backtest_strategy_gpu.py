"""
backtest_strategy_gpu.py

This module contains the GPU-accelerated versions of backtesting logic
using CuPy for massive parallelization.
"""

import cupy as cp
import cudf
import pandas as pd


def create_gpu_data_tensors(all_data_gpu: cudf.DataFrame, all_tickers: list, trading_dates_pd: pd.Index) -> dict:
    """
    Long-format cuDF를 Wide-format CuPy 텐서 딕셔너리로 변환합니다.
    (num_days, num_tickers) 형태의 행렬을 생성하여 반환합니다.
    """
    print("⏳ Creating wide-format GPU data tensors to eliminate CPU-side slicing...")
    
    # pivot_table이 cudf에서 더 안정적일 수 있음
    pivoted_close = all_data_gpu.pivot_table(index='date', columns='ticker', values='close_price')
    pivoted_high = all_data_gpu.pivot_table(index='date', columns='ticker', values='high_price')
    pivoted_low = all_data_gpu.pivot_table(index='date', columns='ticker', values='low_price')

    # trading_dates_pd와 all_tickers 순서에 맞게 재정렬 및 CuPy로 변환
    # .loc 대신 join을 사용하여 누락된 날짜/티커에 대해 NaN을 보장
    base_df = cudf.DataFrame(index=trading_dates_pd)
    close_tensor = base_df.join(pivoted_close, how='left')[all_tickers].to_cupy().astype(cp.float32)
    high_tensor = base_df.join(pivoted_high, how='left')[all_tickers].to_cupy().astype(cp.float32)
    low_tensor = base_df.join(pivoted_low, how='left')[all_tickers].to_cupy().astype(cp.float32)
    
    # 커널 연산을 위해 NaN을 0으로 대체 (거래정지 등)
    close_tensor = cp.nan_to_num(close_tensor, copy=False)
    high_tensor = cp.nan_to_num(high_tensor, copy=False)
    low_tensor = cp.nan_to_num(low_tensor, copy=False)

    print("✅ GPU Tensors created successfully.")
    return {"close": close_tensor, "high": high_tensor, "low": low_tensor}


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

def _calculate_monthly_investment_gpu(portfolio_state, positions_state, param_combinations, evaluation_prices,current_date,debug_mode):
    """ Vectorized calculation of monthly investment amounts based on current market value. """
    if debug_mode:
        print("\n" + "-"*25)
        print(f"DEBUG: Monthly Rebalance Triggered on {current_date.strftime('%Y-%m-%d')}")
        print("-"*25)

    quantities = positions_state[..., 0]
    
    #  총 자산 계산 시 매수 평단이 아닌 '평가 기준가(전일 종가)'를 사용해야 합니다.
    total_quantities_per_stock = cp.sum(quantities, axis=2)
    # current_prices 대신 evaluation_prices(전일 종가)를 사용
    stock_market_values = total_quantities_per_stock * evaluation_prices
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
        sim0_holding_prices = evaluation_prices[holding_mask].get() # evaluation_prices는 1D 배열

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
    stock_inactivity_mask = (days_inactive >= max_inactivity_periods - 1) & has_traded_before & has_any_position
    stock_liquidation_mask_base = stock_stop_loss_mask | stock_inactivity_mask
    stock_liquidation_mask = stock_liquidation_mask_base
    #  현실적인 손절매 체결 로직을 적용하여 최종 청산 마스크를 결정
    if cp.any(stock_liquidation_mask_base):
        # 현실적인 손절매 체결가(price_basis) 계산
        stop_loss_prices = avg_buy_prices * (1 + stop_loss_rates.squeeze(-1))
        high_prices_2d = cp.broadcast_to(current_high_prices, stop_loss_prices.shape)

        # 시나리오 1(A): 장중 손절가 도달 시, Target Price를 기준가로 사용
        # 시나리오 2(B): 갭하락으로 미도달 시, 당일 종가(current_prices_2d)를 기준가로 사용
        stop_loss_basis = cp.where(high_prices_2d >= stop_loss_prices, stop_loss_prices, current_prices_2d)

        # 최종 청산 기준가(liquidation_price_basis) 결정:
        # - 손절매의 경우: 위에서 계산한 stop_loss_basis 사용
        # - 비활성 청산의 경우: 기존처럼 당일 종가(current_prices_2d) 사용
        liquidation_price_basis = cp.where(stock_stop_loss_mask, stop_loss_basis, current_prices_2d)

        # [핵심] 가격 결정 로직이 체결 가능성을 이미 포함하므로, 최종 마스크는 base 마스크와 동일
        stock_liquidation_mask = stock_liquidation_mask_base
    else:
        # 청산 후보가 없으면 빈 마스크로 초기화
        stock_liquidation_mask = stock_liquidation_mask_base
        
    if debug_mode and cp.any(stock_liquidation_mask):
        sim0_stop_loss_indices = cp.where(stock_stop_loss_mask[0])[0].get()
        sim0_inactivity_indices = cp.where(stock_inactivity_mask[0])[0].get()
        # 인덱스를 티커로 변환하여 로그 출력
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
                    target_price = liquidation_price_basis[0, idx].item()
                    exec_price = adjust_price_up_gpu(liquidation_price_basis[0, idx]).item()
                    high_price = current_high_prices[idx].item()
                    reason = "Stop-Loss" if stock_stop_loss_mask[0, idx] else "Inactivity"
                    # 실제 계산에 사용할 수량을 가져와 정확한 예상 수익 계산
                    qty_to_log = cp.sum(quantities[0, idx, :]).item()
                    net_proceeds_sim0 = qty_to_log * exec_price
                    print(
                        f"[GPU_SELL_CALC] {trading_dates_pd_cpu[current_day_idx].strftime('%Y-%m-%d')} {ticker} | "
                        f"Qty: {qty_to_log:,.0f} * ExecPrice: {exec_price:,.0f} = Revenue: {net_proceeds_sim0:,.0f}"
                    )
                    print(
                        f"[GPU_SELL_PRICE] {trading_dates_pd_cpu[current_day_idx].strftime('%Y-%m-%d')} {ticker} "
                        f"Reason: {reason} | "
                        f"Target: {target_price:.2f} -> Exec: {exec_price} | "
                        f"High: {high_price}"
                    )

        broadcasted_liquidation_prices = cp.broadcast_to(liquidation_price_basis.reshape(positions_state.shape[0], -1, 1), buy_prices.shape)
        adjusted_liquidation_prices = adjust_price_up_gpu(broadcasted_liquidation_prices)

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
    #  목표가 계산은 이미 단순 계산 방식으로 구현되어 있습니다.
    target_sell_prices = buy_prices * (1 + sell_profit_rates)
    # 실제 체결가는 목표가를 호가 단위에 맞게 올림 처리합니다.
    execution_sell_prices = adjust_price_up_gpu(target_sell_prices)

    # 체결 조건: 당일 '고가(high)'가 계산된 체결가에 도달했는지 확인하도록 변경
    high_prices_3d = cp.broadcast_to(current_high_prices.reshape(1, -1, 1), buy_prices.shape) # close_prices 대신 high_prices 사용
    
    #  현실적인 백테스팅을 위해 당일(T0) 매수분은 매도 금지
    open_day_idx = positions_state[..., 2]
    sellable_time_mask = open_day_idx < current_day_idx

    # 체결 마스크 생성 시 high_prices_3d를 사용합니다.
    profit_taking_mask = (high_prices_3d >= execution_sell_prices) & valid_positions & sellable_time_mask

    if debug_mode and cp.any(profit_taking_mask):
        sim0_profit_taking_indices = cp.where(cp.any(profit_taking_mask[0], axis=1))[0].get()
        # 인덱스를 티커로 변환하여 로그 출력
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
    # 백테스트 시작 후 10 거래일 동안만 슬롯 상태를 로깅 (sim 0 기준)
    if debug_mode and current_day_idx < 10:
        # trading_dates_pd_cpu를 가져오기 위해 함수 인자에 추가해야 하지만,
        # 디버깅 편의를 위해 전역에서 접근 가능한 변수를 임시로 사용하거나,
        # 여기서는 날짜 없이 Day Index만 출력합니다.
        log_msg = (
            f"[GPU_SLOT_DEBUG] Day {current_day_idx} | "
            f"MaxStocks: {max_stocks_per_sim[0].item()}, "
            f"CurrentHoldings: {current_num_stocks[0].item()}, "
            f"AvailableSlots: {available_slots[0].item()}"
        )
        print(log_msg)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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

        # 이 후보들이 매수 가능한지 판단할 때 '자금 관리 원칙'을 추가합니다.
        
        # 원칙 1: (CPU와 동일) 전략이 요구하는 이상적인 투자금이 현재 가용 현금보다 많으면 매수하지 않습니다.
        # flat_indices_k에 해당하는 시뮬레이션들의 'investment_per_order' 값을 가져옵니다.
        # portfolio_state의 shape은 (num_sim, 2) 이므로, arange로 sim_indices를 만들어 접근합니다.
        sim_indices_k = cp.arange(num_simulations)
        investment_per_order_k = portfolio_state[sim_indices_k, 1]
        has_sufficient_cash_for_budget = temp_capital >= investment_per_order_k

        # 원칙 2: (기존 로직) 실제 매수 비용을 감당할 수 있어야 합니다.
        can_afford_actual_cost = temp_capital >= total_costs[flat_indices_k]
        
        # 원칙 3: (기존 로직) 포트폴리오에 빈 슬롯이 있어야 합니다.
        has_slot = temp_available_slots > 0
        
        # 모든 원칙을 결합하여 최종 매수 가능 여부를 결정합니다.
        still_valid_mask = initial_buy_mask[flat_indices_k] & has_sufficient_cash_for_budget & can_afford_actual_cost & has_slot

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
    #  로그 버퍼 및 카운터 초기화
    # 포맷: [day, sim_idx, stock_idx, capital_before, cost]
    log_buffer = cp.zeros((1000, 5), dtype=cp.float32)
    log_counter = cp.zeros(1, dtype=cp.int32)
    
    ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}
    all_data_reset_idx = all_data_gpu.reset_index()
    weekly_filtered_reset_idx = weekly_filtered_gpu.reset_index()
    print("Data prepared for GPU backtest. (asof logic will be applied in main loop)")

    previous_prices_gpu = cp.zeros(num_tickers, dtype=cp.float32)
    # --- 2.  메인 루프를 월 블록 단위로 변경 ---
    
    #  각 월의 첫 거래일 인덱스를 미리 계산
    monthly_grouper = trading_dates_pd_cpu.to_series().groupby(pd.Grouper(freq='MS'))
    month_start_indices = monthly_grouper.first().index.map(
        lambda dt: trading_dates_pd_cpu.get_loc(dt)
    ).dropna().astype(int).tolist()
    data_tensors = create_gpu_data_tensors(all_data_gpu.reset_index(), all_tickers, trading_dates_pd_cpu)
    close_prices_tensor = data_tensors["close"]
    high_prices_tensor = data_tensors["high"]
    low_prices_tensor = data_tensors["low"]
    # 월 블록 루프 시작
    for i in range(len(month_start_indices)):
        start_idx = month_start_indices[i]
        end_idx = month_start_indices[i+1] if i + 1 < len(month_start_indices) else num_trading_days
        
        # 월별 투자금 재계산 로직을 월 블록 루프의 시작점으로 이동
        # 평가 기준가는 월 블록 시작일의 전일 종가 또는 초기값
        eval_prices = previous_prices_gpu if start_idx > 0 else cp.zeros(num_tickers, dtype=cp.float32)
        current_rebalance_date = trading_dates_pd_cpu[start_idx]
        
        portfolio_state = _calculate_monthly_investment_gpu(
            portfolio_state, positions_state, param_combinations, eval_prices, current_rebalance_date, debug_mode
        )
        #  디버깅 및 검증을 위한 임시 '일일 루프' (향후 단일 커널로 대체될 부분)
        for day_idx in range(start_idx, end_idx):
            current_date = trading_dates_pd_cpu[day_idx]
            # 텐서에서 하루치 데이터 슬라이싱
            current_prices_gpu = close_prices_tensor[day_idx]
            current_highs_gpu  = high_prices_tensor[day_idx]
            current_lows_gpu   = low_prices_tensor[day_idx]

            # --- 후보군 선정 로직 (기존과 동일) ---
            past_or_equal_data = weekly_filtered_reset_idx[weekly_filtered_reset_idx['date'] < current_date]
            if not past_or_equal_data.empty:
                # 1. 해당 주간의 필터링된 종목 리스트를 가져옴
                latest_filter_date = past_or_equal_data['date'].max()
                candidates_of_the_week = weekly_filtered_reset_idx[weekly_filtered_reset_idx['date'] == latest_filter_date]
                candidate_tickers_list = candidates_of_the_week['ticker'].to_arrow().to_pylist()

                # 2. 전체 데이터에서 현재 날짜 '이하'의 모든 데이터를 가져옴
                daily_data_for_candidates = all_data_reset_idx[all_data_reset_idx['date'] == current_date]
                
                if not daily_data_for_candidates.empty:
                    # 3. 현재 날짜 데이터 중에서 주간 후보군에 해당하는 종목만 필터링
                    candidate_data_today = daily_data_for_candidates[daily_data_for_candidates['ticker'].isin(candidate_tickers_list)]
                    
                    # 4. 유효한 ATR 값을 가진 후보만 최종 선정 (NaN 제거)
                    valid_candidate_atr_series = candidate_data_today.set_index('ticker')['atr_14_ratio'].dropna()

                    if not valid_candidate_atr_series.empty:
                        # 5. 최종 후보 티커와 ATR 값을 cuPy 배열로 변환
                        valid_tickers = valid_candidate_atr_series.index.to_arrow().to_pylist()
                        valid_atrs = valid_candidate_atr_series.values
                        
                        # ticker_to_idx 딕셔너리에 없는 티커는 제외
                        candidate_indices_list = [ticker_to_idx.get(t) for t in valid_tickers if t in ticker_to_idx]
                        
                        # 실제 ATR 값도 ticker_to_idx에 존재하는 티커에 맞춰 필터링
                        valid_atrs_filtered = [atr for t, atr in zip(valid_tickers, valid_atrs) if t in ticker_to_idx]

                        candidate_tickers_for_day = cp.array(candidate_indices_list, dtype=cp.int32)
                        candidate_atrs_for_day = cp.asarray(valid_atrs_filtered, dtype=cp.float32)
                    else:
                        candidate_tickers_for_day = cp.array([], dtype=cp.int32)
                        candidate_atrs_for_day = cp.array([], dtype=cp.float32)
                else:
                    candidate_tickers_for_day = cp.array([], dtype=cp.int32)
                    candidate_atrs_for_day = cp.array([], dtype=cp.float32)
            else:
                candidate_tickers_for_day = cp.array([], dtype=cp.int32)
                candidate_atrs_for_day = cp.array([], dtype=cp.float32)

            # 2-2. 월별 투자금 재계산
            # --- 신호 처리 함수 호출 (기존과 동일) ---
            portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, sell_occurred_today_mask = _process_sell_signals_gpu(
                portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, day_idx,
                param_combinations, current_prices_gpu, current_highs_gpu,
                execution_params["sell_commission_rate"], execution_params["sell_tax_rate"],
                debug_mode=debug_mode, all_tickers=all_tickers, trading_dates_pd_cpu=trading_dates_pd_cpu
            )
            portfolio_state, positions_state, last_trade_day_idx_state = _process_new_entry_signals_gpu(
                portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, day_idx,
                cooldown_period_days, param_combinations, current_prices_gpu,
                candidate_tickers_for_day, candidate_atrs_for_day,
                execution_params["buy_commission_rate"], log_buffer, log_counter, debug_mode, all_tickers=all_tickers
            )
            portfolio_state, positions_state, last_trade_day_idx_state = _process_additional_buy_signals_gpu(
                portfolio_state, positions_state, last_trade_day_idx_state, sell_occurred_today_mask, day_idx,
                param_combinations, current_prices_gpu, current_lows_gpu, current_highs_gpu,
                execution_params["buy_commission_rate"], log_buffer, log_counter, debug_mode, all_tickers=all_tickers
            )
        
            # --- 일일 포트폴리오 가치 업데이트 (기존과 동일) ---
            stock_quantities = cp.sum(positions_state[..., 0], axis=2)
            stock_market_values = stock_quantities * current_prices_gpu
            total_stock_value = cp.sum(stock_market_values, axis=1)
            daily_portfolio_values[:, day_idx] = portfolio_state[:, 0] + total_stock_value
            if debug_mode:
                capital_snapshot = portfolio_state[0, 0].get()
                stock_val_snapshot = total_stock_value[0].get()
                total_val_snapshot = daily_portfolio_values[0, day_idx].get()
                num_pos_snapshot = cp.sum(cp.any(positions_state[0, :, :, 0] > 0, axis=1)).get()
                
                # [추가] CPU 로그와 유사한 포맷으로 출력하여 비교 용이성 증대
                header = f"\n{'='*120}\n"
                footer = f"\n{'='*120}"
                date_str = current_date.strftime('%Y-%m-%d')
                
                cash_ratio = (capital_snapshot / total_val_snapshot) * 100 if total_val_snapshot else 0
                stock_ratio = (stock_val_snapshot / total_val_snapshot) * 100 if total_val_snapshot else 0

                summary_str = (
                    f"GPU STATE | Date: {date_str} | Day {day_idx+1}/{num_trading_days}\n"
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
            # 월 블록의 마지막 날 종가를 다음 리밸런싱을 위한 평가 기준으로 저장
        previous_prices_gpu = close_prices_tensor[end_idx - 1].copy()
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

    
       
