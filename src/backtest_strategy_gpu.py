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


def get_tick_size_gpu(price_array):
    """
    주가 배열에 따른 호가 단위 배열을 반환합니다.
    @cp.fuse() 데코레이터는 CuPy의 엄격한 타입 요구사항으로 인해 제거되었습니다.
    """
    # 조건 리스트는 그대로 둡니다.
    condlist = [
        price_array < 2000,
        price_array < 5000,
        price_array < 20000,
        price_array < 50000,
        price_array < 200000,
        price_array < 500000,
    ]
    # ★★★ 핵심 수정 부분 ★★★
    # 선택지 리스트를 스칼라가 아닌, 'CuPy 배열의 리스트'로 만듭니다.
    # cp.full_like(price_array, 값)은 price_array와 똑같은 모양과 타입의 배열을
    # '값'으로 가득 채워서 만들어줍니다.
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
    all_tickers: list,  # The ordered list of tickers corresponding to the second dimension of positions_state
):
    """
    Vectorized calculation of monthly investment amounts for all simulations.
    """
    # 1. Get current prices for all stocks
    # Reset index to perform boolean masking on the 'date' column directly
    data_for_lookup = all_data_gpu.reset_index()
    # Filter data up to the current date
    filtered_data = data_for_lookup[data_for_lookup["date"] <= current_date]
    # Get the last available price for each ticker
    latest_prices = filtered_data.groupby("ticker")["close_price"].last()

    # Reindex to ensure the price series matches the exact order and size of all_tickers,
    # filling any missing prices with 0 (for stocks with no data on that day).
    price_series = latest_prices.reindex(all_tickers).fillna(0)

    # Convert the final price series to a CuPy array.
    prices_gpu = cp.asarray(price_series.values, dtype=cp.float32)

    # 2. Calculate current stock value for all simulations
    quantities = positions_state[
        ..., 0
    ]  # Shape: (num_combinations, num_stocks, max_splits)

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


# ==============================================================================
# 아래 함수로 기존 _process_sell_signals_gpu 함수를 완전히 대체합니다.
# ==============================================================================
def _process_sell_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    # config.yaml에서 읽어온 실행 파라미터를 추가로 받습니다.
    sell_commission_rate: float,
    sell_tax_rate: float,
    debug_mode: bool = False, # ★★★ 추가
    is_debug_day: bool = False,      
    debug_ticker_idx: int = -1  
):
    """
    Vectorized sell signal processing for all simulations, reflecting exact execution logic.
    CPU(execution.py)의 매도 로직을 GPU로 완벽하게 포팅한 버전입니다.
    """
   
    # --- Step 0: 파라미터 및 상태 준비 ---
    sell_profit_rates = param_combinations[:, 3:4].reshape(-1, 1, 1)  # (comb, 1, 1)
    quantities = positions_state[..., 0]  # (comb, stock, split)
    buy_prices = positions_state[..., 1]  # (comb, stock, split)

    # 현재가가 브로드캐스팅된 배열
    # (1, stock, 1) -> (comb, stock, split)
    broadcasted_prices = cp.broadcast_to(
        current_prices.reshape(1, -1, 1), buy_prices.shape
    )
    
     # ★★★ 함수 내부에 디버깅 블록 추가 ★★★
    if debug_mode and is_debug_day: 
        # 0번 시뮬레이션, 디버그 티커, 1차 매도분(split_idx=0)에 대한 모든 변수 값 출력
        sim_idx, stock_idx, split_idx = 0, debug_ticker_idx, 0
        
        # 1. 입력 값 확인
        bp = buy_prices[sim_idx, stock_idx, split_idx].get()
        spr = sell_profit_rates[sim_idx, 0, 0].get()
        cp_val = current_prices[stock_idx].get()
        
        print("\n--- SELL DEBUGGER (2023-01-06, TickerIdx: 287) ---")
        print(f"  Input -> BuyPrice: {bp}, SellProfitRate: {spr}, CurrentPrice: {cp_val}")

        # 2. 중간 계산 과정 확인
        cost_f = 1.0 - sell_commission_rate - sell_tax_rate
        target_sp = (bp * (1 + spr)) / cost_f
        actual_sp = adjust_price_up_gpu(cp.array([target_sp])).get()[0]

        print(f"  Calc  -> TargetSellPrice: {target_sp}, ActualSellPrice: {actual_sp}")

        # 3. 최종 조건 및 결과 확인
        final_condition = cp_val >= actual_sp
        print(f"  Result-> Sell Condition ({cp_val} >= {actual_sp}): {final_condition}")
        print("---------------------------------------------------\n")

    

    # 매도 대상이 될 수 있는 유효한 포지션 (매수가가 0보다 큼)
    valid_positions = buy_prices > 0

    # --- Step 1: CPU 로직과 동일하게 실제 체결가 및 순수익 계산 ---

    # 1. 비용 팩터 계산
    cost_factor = 1.0 - sell_commission_rate - sell_tax_rate

    # 2. 최소 목표 매도가 계산
    # (comb, stock, split) * (comb, 1, 1) -> (comb, stock, split)
    target_sell_prices = (buy_prices * (1 + sell_profit_rates)) / cost_factor

    # 3. 실제 체결가 결정 (호가 단위 올림)
    actual_sell_prices = adjust_price_up_gpu(target_sell_prices)

    # 4. 매도 체결 조건: 현재가(종가)가 계산된 실제 체결가에 도달했거나 넘어섰는가?
    # (comb, stock, split) >= (comb, stock, split)
    sell_trigger_condition = (
        broadcasted_prices >= actual_sell_prices
    ) & valid_positions

    # --- Step 2: 매도 로직 실행 (부분 매도 / 전체 청산) ---

    # 1. 1차 매도분 청산 조건: 1차 포지션(split_idx=0)의 매도 조건이 충족되었는가?
    # (comb, stock)
    first_position_sell_triggered = sell_trigger_condition[:, :, 0]

    # 2. 부분 매도(2차 이상) 조건: 전체 청산 대상이 아니면서, 개별 매도 조건이 충족되었는가?
    partial_sell_mask = sell_trigger_condition.copy()
    partial_sell_mask[:, :, 0] = False  # 1차 매도분은 부분 매도 대상에서 제외

    # 1차 청산이 발동된 종목은 그 종목의 다른 차수들도 부분 매도 대상에서 제외 (전체 청산되므로)
    # (comb, stock, 1) -> (comb, stock, split)
    partial_sell_mask &= ~cp.broadcast_to(
        first_position_sell_triggered[:, :, cp.newaxis], partial_sell_mask.shape
    )

    # --- Step 3: 매도 대금 계산 및 자본 업데이트 ---

    # 1. 전체 청산될 포지션들의 매도 대금 계산
    full_liquidation_mask = cp.broadcast_to(
        first_position_sell_triggered[:, :, cp.newaxis], quantities.shape
    )

    # --- ★★★ 로그 추가 시작 ★★★ ---
    if debug_mode: # ★★★ 수정: if 문으로 전체 로그 블록 감싸기
        # 0번 시뮬레이션에서 어떤 거래가 일어났는지 확인하여 출력
        
        # 1. 0번 시뮬레이션에서 '부분 매도'가 일어난 포지션 찾기
        sim0_partial_sell_mask = partial_sell_mask[0] # (num_stocks, max_splits)
        if cp.any(sim0_partial_sell_mask):
            # 매도가 일어난 (stock_idx, split_idx) 쌍의 인덱스를 가져옴
            partial_stock_indices, partial_split_indices = cp.where(sim0_partial_sell_mask)
            
            # 각 매도 건에 대해 로그 출력
            for stock_idx, split_idx in zip(partial_stock_indices, partial_split_indices):
                stock_idx, split_idx = int(stock_idx), int(split_idx)
                order_num = split_idx + 1
                qty = int(quantities[0, stock_idx, split_idx])
                close = float(current_prices[stock_idx])
                buy_price = float(buy_prices[0, stock_idx, split_idx])
                sell_price = float(actual_sell_prices[0, stock_idx, split_idx])
                net_revenue = (sell_price * qty) * cost_factor
                
                print(f"  [GPU_TRADE_LOG] TickerIdx: {stock_idx}, Action: SELL, Order: {order_num}, "
                    f"Qty: {qty}, Close: {close:,.0f}, BuyPrice(Original): {buy_price:,.0f}, "
                    f"SellPrice: {sell_price:,.0f}, NetRevenue: {net_revenue:,.0f}")

        # 2. 0번 시뮬레이션에서 '전체 청산'이 일어난 종목 찾기
        sim0_full_liquidation_mask = full_liquidation_mask[0]
        if cp.any(sim0_full_liquidation_mask):
            # 전체 청산이 일어난 종목의 인덱스 (stock_idx)를 가져옴
            full_stock_indices = cp.where(cp.any(sim0_full_liquidation_mask, axis=1))[0]

            for stock_idx in full_stock_indices:
                stock_idx = int(stock_idx)
                # 해당 종목의 모든 유효한 포지션(차수)에 대해 로그 출력
                for split_idx in range(positions_state.shape[2]): # max_splits
                    # 이 차수가 실제로 매도 대상이었는지 확인
                    if quantities[0, stock_idx, split_idx] > 0 and sim0_full_liquidation_mask[stock_idx, split_idx]:
                        order_num = split_idx + 1
                        qty = int(quantities[0, stock_idx, split_idx])
                        close = float(current_prices[stock_idx])
                        buy_price = float(buy_prices[0, stock_idx, split_idx])
                        sell_price = float(actual_sell_prices[0, stock_idx, split_idx])
                        net_revenue = (sell_price * qty) * cost_factor
                        
                        print(f"  [GPU_TRADE_LOG] TickerIdx: {stock_idx}, Action: SELL (Full), Order: {order_num}, "
                            f"Qty: {qty}, Close: {close:,.0f}, BuyPrice(Original): {buy_price:,.0f}, "
                            f"SellPrice: {sell_price:,.0f}, NetRevenue: {net_revenue:,.0f}")
                
                # 종목 청산 로그 (CPU의 Liquidate와 유사)
                print(f"  [GPU_TRADE_LOG] TickerIdx: {stock_idx}, Action: Liquidate")
        # --- ★★★ 로그 추가 끝 ★★★ ---

    # 전체 청산 시, 모든 포지션은 '자신의 계산된 실제 매도가'에 팔린다고 가정
    full_liquidation_raw_proceeds_matrix = (
        quantities * actual_sell_prices * full_liquidation_mask
    )
    full_liquidation_raw_proceeds = cp.sum(
        full_liquidation_raw_proceeds_matrix, axis=(1, 2)
    )

    # 2. 부분 매도될 포지션들의 매도 대금 계산
    partial_sell_raw_proceeds_matrix = (
        quantities * actual_sell_prices * partial_sell_mask
    )
    partial_sell_raw_proceeds = cp.sum(partial_sell_raw_proceeds_matrix, axis=(1, 2))

    # 3. 총 매도 대금을 합산하고 비용을 차감하여 최종 입금액 계산
    total_raw_proceeds = full_liquidation_raw_proceeds + partial_sell_raw_proceeds
    net_proceeds = total_raw_proceeds * cost_factor

    # 4. 자본에 최종 입금액 반영
    portfolio_state[:, 0] += net_proceeds

    # --- Step 4: 포지션 상태 업데이트 ---

    # 1. 부분 매도(2차 이상)가 일어난 포지션의 '수량'만 0으로 설정합니다.
    #    partial_sell_mask는 전체 청산 대상 종목을 이미 제외했으므로 안전합니다.
    positions_state[..., 0][partial_sell_mask] = 0

    # 2. 전체 청산(1차 매도)이 일어난 종목을 처리합니다.
    #    해당 종목의 '모든 차수'에 대해 수량을 0으로, 매수가를 -1로 만듭니다.
    
    # first_position_sell_triggered: (comb, stock) 형태의 2D 마스크
    # 이를 브로드캐스팅하여 (comb, stock, split) 형태의 3D 마스크로 확장합니다.
    full_liquidation_stock_mask = cp.broadcast_to(
        first_position_sell_triggered[:, :, cp.newaxis], positions_state[..., 0].shape
    )
    
    # 청산 대상 종목의 모든 차수 수량을 0으로 설정합니다.
    # 이것이 총자산 계산의 정확성을 보장합니다.
    positions_state[..., 0][full_liquidation_stock_mask] = 0
    
    # 청산 대상 종목의 모든 차수 매수가를 -1로 설정합니다.
    # 이것이 향후 해당 종목이 추가매수/매도 대상에서 제외되도록 보장합니다.
    positions_state[..., 1][full_liquidation_stock_mask] = -1
    
    return portfolio_state, positions_state


def _process_additional_buy_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    buy_commission_rate: float,  # 매수 수수료율 인자
    debug_mode: bool = False, # ★★★ 추가
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
    # --- Step 0: 파라미터 및 상태 준비 ---
    # Extract additional buy drop rates: shape (num_combinations, 1, 1)
    add_buy_drop_rates = param_combinations[:, 2:3].reshape(-1, 1, 1)

    # Get investment amounts per order: shape (num_combinations, 1, 1)
    investment_per_order = portfolio_state[:, 1:2].reshape(-1, 1, 1)

    # Get current capital: shape (num_combinations,)
    current_capital = portfolio_state[:, 0]

    # Reshape current prices: (1, num_stocks, 1)
    current_prices_reshaped = current_prices.reshape(1, -1, 1)

    quantities = positions_state[..., 0]
    buy_prices = positions_state[..., 1]
    # --- Step 1: 추가 매수 조건 탐색 (자본 확인 전) ---
    # 마지막 차수 정보 추출 (이 부분은 성능 개선의 여지가 있지만, 현재는 정확성에 초점)
    # Find the last (highest order) position for each stock in each simulation
    # We'll iterate through splits in reverse to find the last non-zero position
    has_positions = quantities > 0  # Shape: (num_combinations, num_stocks, max_splits)
    has_any_position = cp.any(
        has_positions, axis=2
    )  # Shape: (num_combinations, num_stocks)
    last_position_indices = cp.zeros((num_combinations, num_stocks), dtype=cp.int32)
    # Find the last position for each stock (rightmost True in the max_splits dimension)

    for sim in range(num_combinations):
        for stock in range(num_stocks):
            if has_any_position[sim, stock]:
                # Find the last True position
                last_idx = cp.where(has_positions[sim, stock, :])[0]
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
    # 이 부분을 CPU 로직과 유사하게 수정합니다.
    # GPU 벡터화가 어려우므로, 우선 루프를 사용해 정확성을 확보합니다
    can_add_position = cp.zeros_like(additional_buy_condition, dtype=cp.bool_)
    next_split_indices_to_buy = cp.full_like(
        last_position_indices, -1, dtype=cp.int32
    )  # -1로 초기화

    # 이 루프는 성능 저하를 일으키지만, 정확한 로직 구현을 위해 필수적입니다.
    for sim in range(num_combinations):
        for stock in range(num_stocks):
            if additional_buy_condition[sim, stock]:
                # 해당 종목의 현재 포지션 상태를 가져옴
                positions_for_stock = positions_state[sim, stock, :, 0]

                # 비어있는 첫 번째 슬롯(차수)을 찾음
                empty_slots = cp.where(positions_for_stock == 0)[0]

                if empty_slots.size > 0:
                    first_empty_slot = empty_slots[0]
                    # 비어있는 슬롯이 최대 차수 제한(max_splits) 내에 있는지 확인
                    if first_empty_slot < max_splits:
                        can_add_position[sim, stock] = True
                        next_split_indices_to_buy[sim, stock] = first_empty_slot

    initial_buy_condition = can_add_position
    if cp.any(initial_buy_condition):
        # --- ★★★ 안전한 자본 차감 로직 추가 ★★★ ---

        # 1. 실제 매수 대상의 인덱스 가져오기
        sim_indices, stock_indices = cp.where(initial_buy_condition)

        # 2. 후보군들의 'sort_metric' 계산에 필요한 정보 가져오기
        # 2-1. 현재 보유 차수 계산 (len(positions)에 해당)
        num_existing_splits = cp.sum(has_positions[sim_indices, stock_indices], axis=1)
        # 2-2. 하락률 계산
        last_buy_prices_for_candidates = last_buy_prices[sim_indices, stock_indices]
        current_prices_for_candidates = current_prices[stock_indices]
        # 분모 0 방지
        drop_rates = cp.zeros_like(last_buy_prices_for_candidates)
        valid_mask = last_buy_prices_for_candidates > 0
        drop_rates[valid_mask] = (last_buy_prices_for_candidates[valid_mask] - current_prices_for_candidates[valid_mask]) / last_buy_prices_for_candidates[valid_mask]
        # 3. 각 시뮬레이션의 'additional_buy_priority' 파라미터 가져오기
        # 0: lowest_order, 1: highest_drop
        add_buy_priority_params = param_combinations[sim_indices, 4]
        # 4. 'sort_metric' 최종 계산
        # priority가 0이면 보유 차수, 1이면 (-하락률)을 sort_metric으로 사용
        sort_metric = cp.where(
            add_buy_priority_params == 0,
            num_existing_splits.astype(cp.float32),
            -drop_rates
        )
        # 5. 모든 후보 정보를 cuDF DataFrame으로 변환
        candidates_gdf = cudf.DataFrame({
            'sim_idx': sim_indices,
            'sort_metric': sort_metric,
            'stock_idx': stock_indices,
            'next_split_idx': next_split_indices_to_buy[sim_indices, stock_indices]
        })
        # 6. DataFrame을 시뮬레이션 번호 -> 우선순위(sort_metric) 순으로 정렬
        sorted_candidates_gdf = candidates_gdf.sort_values(by=['sim_idx', 'sort_metric'], ascending=[True, True])
        # 7. 정렬된 각 열을 다시 CuPy 배열로 변환
        sorted_sim_indices = sorted_candidates_gdf['sim_idx'].values
        sorted_stock_indices = sorted_candidates_gdf['stock_idx'].values
        sorted_next_split_indices = sorted_candidates_gdf['next_split_idx'].values
        # 8. CuPy 배열을 순회하며 순차적 매수 실행
        for i in range(len(sorted_sim_indices)):
            sim_idx = int(sorted_sim_indices[i])
            stock_idx = int(sorted_stock_indices[i])
            next_split_idx = int(sorted_next_split_indices[i])
            
            # 현재 시뮬레이션의 최신 자본 상태를 가져옴
            current_sim_capital = portfolio_state[sim_idx, 0]
            
            # 매수에 필요한 정보 계산
            inv_per_order = investment_per_order[sim_idx, 0, 0]
            current_price = current_prices[stock_idx]
            buy_price = adjust_price_up_gpu(current_price)
            
            if buy_price <= 0:
                continue

            # .astype(cp.int32) 대신 int()로 CPU 스칼라로 변환
            quantity = int(cp.floor(inv_per_order / buy_price))
            if quantity <= 0:
                continue
                
            # CPU 스칼라 값으로 최종 비용 계산
            total_cost = (float(buy_price) * quantity) * (1 + buy_commission_rate)

            # 자본이 충분한 경우에만 매수 실행
            if float(current_sim_capital) >= total_cost:
                # --- ★★★ 로그 추가 시작 ★★★ ---
                # 0번 시뮬레이션에 대해서만 로그 출력
                if debug_mode and sim_idx == 0: # ★★★ 수정: debug_mode 조건 추가
                    # 로그에 필요한 변수들 준비
                    order_num = next_split_idx + 1 # 차수는 1부터 시작
                    close_price = float(current_price)
                    bp_float = float(buy_price)
                    
                    print(f"  [GPU_TRADE_LOG] TickerIdx: {stock_idx}, Action: ADD_BUY, Order: {order_num}, "
                          f"Qty: {quantity}, Close: {close_price:,.0f}, BuyPrice: {bp_float:,.0f}, "
                          f"TotalCost: {total_cost:,.0f}")
                # --- ★★★ 로그 추가 끝 ★★★ ---
                # 자본 차감 (원본 portfolio_state를 직접 수정)
                portfolio_state[sim_idx, 0] -= total_cost
                
                # 포지션 업데이트
                positions_state[sim_idx, stock_idx, next_split_idx, 0] = quantity
                positions_state[sim_idx, stock_idx, next_split_idx, 1] = buy_price

    # if 블록이 끝난 후, 최종적으로 portfolio_state와 positions_state를 반환
    return portfolio_state, positions_state


def run_magic_split_strategy_on_gpu(
    initial_cash: float,
    param_combinations: cp.ndarray,
    all_data_gpu: cudf.DataFrame,
    weekly_filtered_gpu: cudf.DataFrame,
    trading_date_indices: cp.ndarray,  # 💡 파라미터 이름 변경 (trading_dates -> trading_date_indices)
    trading_dates_pd_cpu: pd.DatetimeIndex,  # 💡 새로운 파라미터 추가
    all_tickers: list,
    execution_params: dict,  # ★★★ 추가 ★★★
    max_splits_limit: int = 20,
    debug_mode: bool = False, # ★★★ 추가 
):
    """
    Main GPU-accelerated backtesting function for the MagicSplitStrategy.
    """
    if debug_mode:
        print("🚀 Initializing GPU backtesting environment...")
    num_combinations = param_combinations.shape[0]
    num_trading_days = len(trading_date_indices)  # 💡 길이는 정수 인덱스 배열 기준

    # --- 1. State Management Arrays ---
    # Portfolio-level state: [0:capital, 1:investment_per_order]
    portfolio_state = cp.zeros((num_combinations, 2), dtype=cp.float32)
    portfolio_state[:, 0] = initial_cash

    # Position-level state: [0: quantity, 1: buy_price]
    max_stocks_param = int(
        cp.max(param_combinations[:, 0]).get()
    )  # Get max_stocks from user parameters
    if debug_mode:
        print(f"max_stocks_param: {max_stocks_param}")
    num_tickers = len(all_tickers)

    # The actual dimension used for arrays must match the full list of tickers
    positions_state = cp.zeros(
        (num_combinations, num_tickers, max_splits_limit, 2), dtype=cp.float32
    )
    if debug_mode:
        print(f"portfolio_state: {portfolio_state.get()}")
        print(f"positions_state: {cp.any(positions_state > 0).get()}")
    daily_portfolio_values = cp.zeros(
        (num_combinations, num_trading_days), dtype=cp.float32
    )

    if debug_mode:
        print(f"    - State arrays created. Portfolio State Shape: {portfolio_state.shape}")
        print(f"    - Positions State Array Shape: {positions_state.shape}")

    # 💡 티커를 인덱스로 변환하는 딕셔너리를 미리 만들어 성능 향상
    ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}

    # --- 2. Main Simulation Loop (Vectorized) ---
    previous_month = -1
    for i, date_idx in enumerate(trading_date_indices):
        current_date = trading_dates_pd_cpu[date_idx.item()]
        
        
        current_month = current_date.month

        # --- [DEBUG] 루프 시작 시점의 상태 ---
        if debug_mode:
            capital_before_day = portfolio_state[0, 0].get()
            positions_before_day = cp.sum(positions_state[0, :, :, 0] > 0).get()
            print(
                f"\n--- Day {i+1}/{num_trading_days}: {current_date.strftime('%Y-%m-%d')} ---"
            )
            print(
                f"[BEGIN] Capital: {capital_before_day:,.0f} | Total Positions: {positions_before_day}"
            )

        data_for_lookup = all_data_gpu.reset_index()
        current_day_data = data_for_lookup[data_for_lookup["date"] == current_date]

        if not current_day_data.empty:
            daily_prices = current_day_data.groupby("ticker")["close_price"].last()
            price_series = daily_prices.reindex(all_tickers).fillna(0)
            current_prices = cp.asarray(price_series.values, dtype=cp.float32)

            # --- [ACTION] Monthly Rebalance ---
            if current_month != previous_month:
                portfolio_state = _calculate_monthly_investment_gpu(
                    current_date,
                    portfolio_state,
                    positions_state,
                    param_combinations,
                    all_data_gpu,
                    all_tickers,
                )
                if debug_mode:
                    inv_per_order = portfolio_state[0, 1].get()
                    print(
                        f"  [REBALANCE] Month changed to {current_month}. New Investment/Order: {inv_per_order:,.0f}"
                )
                previous_month = current_month

            # --- [ACTION] Sell, Add_Buy, New_Buy ---
            capital_before_actions = portfolio_state[
                0, 0
            ].get()  # 모든 매매 행위 전의 자본

            # 1. Process New Entry Signals
            # (후보군 선정 로직)
            weekly_filtered_reset = weekly_filtered_gpu.reset_index()
            past_data = weekly_filtered_reset[
                weekly_filtered_reset["date"] <= current_date
            ]
            candidates_of_the_week = cudf.DataFrame()
            # candidates_of_the_week가 계산된 직후에 로그를 추가하세요.

            if not past_data.empty:
                most_recent_date_cudf = past_data["date"].max()

                # --- ★★★ AttributeError 수정 ★★★ ---
                # cudf/numpy 날짜 타입을 파이썬 표준 datetime으로 변환
                most_recent_date_pd = pd.to_datetime(most_recent_date_cudf)
                # ---

                candidates_of_the_week = past_data[
                    past_data["date"] == most_recent_date_cudf
                ]
                if len(candidates_of_the_week) > 0 and debug_mode:
                    print(
                        f"  [DEBUG] Current Date: {current_date.strftime('%Y-%m-%d')}, Using Filter Date: {most_recent_date_pd.strftime('%Y-%m-%d')}, Candidates Found: {len(candidates_of_the_week)}"
                    )
            candidate_tickers_for_day = cp.array([], dtype=cp.int32)
            candidate_atrs_for_day = cp.array([], dtype=cp.float32)

            if not candidates_of_the_week.empty:
                candidate_tickers_str = (
                    candidates_of_the_week["ticker"].to_arrow().to_pylist()
                )
                candidate_indices = [
                    ticker_to_idx.get(t)
                    for t in candidate_tickers_str
                    if ticker_to_idx.get(t) is not None
                ]

                if candidate_indices:
                    data_for_filtering = all_data_gpu.reset_index()

                    mask_ticker = data_for_filtering["ticker"].isin(
                        candidate_tickers_str
                    )
                    mask_date = data_for_filtering["date"] == current_date
                    candidate_data_today = data_for_filtering[mask_ticker & mask_date]
                    if debug_mode:
                        print(
                            f"  [DEBUG] Found {len(candidate_data_today)} candidates with today's price data."
                        )
                    if not candidate_data_today.empty:
                        candidate_data_today = candidate_data_today.set_index(
                            ["ticker", "date"]
                        )
                        valid_candidates = candidate_data_today.dropna(
                            subset=["atr_14_ratio"]
                        )
                        if not valid_candidates.empty:
                            valid_tickers_str = (
                                valid_candidates.index.get_level_values("ticker")
                                .to_arrow()
                                .to_pylist()
                            )
                            valid_indices = [
                                ticker_to_idx[t] for t in valid_tickers_str
                            ]

                            candidate_tickers_for_day = cp.array(
                                valid_indices, dtype=cp.int32
                            )
                            candidate_atrs_for_day = cp.asarray(
                                valid_candidates["atr_14_ratio"].values,
                                dtype=cp.float32,
                            )

            # (신규 매수 실행)
            portfolio_state, positions_state = _process_new_entry_signals_gpu(
                portfolio_state,
                positions_state,
                param_combinations,
                current_prices,
                candidate_tickers_for_day,
                candidate_atrs_for_day,
                buy_commission_rate=execution_params["buy_commission_rate"],
                debug_mode=debug_mode,
            )
            # 2. Process Additional Buy Signals
            portfolio_state, positions_state = _process_additional_buy_signals_gpu(
                portfolio_state,
                positions_state,
                param_combinations,
                current_prices,
                buy_commission_rate=execution_params["buy_commission_rate"],
                debug_mode=debug_mode,
            )
            # 3. Process Sell Signals
             # --- ★★★ 디버깅 코드 추가 시작 ★★★ ---
            # 1. 디버깅할 날짜와 티커 인덱스를 지정합니다.
            is_debug_day = current_date.strftime('%Y-%m-%d') == '2023-01-06'
            
            # all_tickers 리스트에서 '120240'의 인덱스를 찾습니다. 없으면 -1.
            debug_ticker_idx = ticker_to_idx.get('120240', -1) 
            
            portfolio_state, positions_state = _process_sell_signals_gpu(
                portfolio_state, positions_state, param_combinations, current_prices,
                execution_params["sell_commission_rate"], execution_params["sell_tax_rate"],
                is_debug_day=is_debug_day,
                debug_ticker_idx=debug_ticker_idx,
                debug_mode=debug_mode,
            )
            # --- ★★★ 디버깅 코드 추가 끝 ★★★ ---
            capital_after_actions = portfolio_state[
                0, 0
            ].get()  # 모든 매매 행위 후의 자본
            if capital_after_actions != capital_before_actions and debug_mode:
                print(
                    f"  [TRADE]   Capital changed by: {capital_after_actions - capital_before_actions:,.0f}"
                )

            # --- [CALC] Calculate and store daily portfolio values ---
            quantities = positions_state[..., 0]
            current_prices_reshaped = current_prices.reshape(1, -1, 1)
            stock_values = cp.sum(quantities * current_prices_reshaped, axis=(1, 2))
            total_values = portfolio_state[:, 0] + stock_values
            daily_portfolio_values[:, i] = total_values

        else:  # 거래 데이터 없는 날
            if i > 0:
                daily_portfolio_values[:, i] = daily_portfolio_values[:, i - 1]
            else:
                daily_portfolio_values[:, i] = initial_cash

        # --- [DEBUG] 루프 종료 시점의 상태 ---
        if debug_mode:
            final_capital_of_day = portfolio_state[0, 0].get()
            final_total_value_of_day = daily_portfolio_values[0, i].get()
            final_stock_value_of_day = final_total_value_of_day - final_capital_of_day
            final_positions_of_day = cp.sum(positions_state[0, :, :, 0] > 0).get()
            # --- ★★★ 로그 추가 시작 ★★★ ---
            # 0번 시뮬레이션의 최종 보유 종목 리스트 출력
            # 1. 현재 어떤 종목을 보유하고 있는지 (종목 단위) boolean 마스크 생성
            has_any_position = cp.any(positions_state[0, :, :, 0] > 0, axis=1)
            # 2. 보유 중인 종목의 인덱스(ticker_idx)를 가져옴
            held_stock_indices = cp.where(has_any_position)[0].get().tolist()
            # 3. 인덱스를 실제 티커 코드로 변환 (all_tickers 리스트 활용)
            held_tickers = sorted([all_tickers[idx] for idx in held_stock_indices])
            print(f"  [GPU_HOLDINGS] {held_tickers}")
            # --- ★★★ 로그 추가 끝 ★★★ ---
            print(
                f"[END]   Capital: {final_capital_of_day:,.0f} | Stock Val: {final_stock_value_of_day:,.0f} | Total Val: {final_total_value_of_day:,.0f} | Positions: {final_positions_of_day}"
            )
        # ---

        if (i + 1) % 252 == 0:
            if debug_mode:
                year = current_date.year
                print(f"    - Simulating year: {year} ({i+1}/{num_trading_days})")

    if debug_mode:
        print("🎉 GPU backtesting simulation finished.")

    return daily_portfolio_values


def _process_new_entry_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    candidate_tickers_for_day: cp.ndarray,  # 오늘 매수 후보군 티커의 '인덱스' 배열
    candidate_atrs_for_day: cp.ndarray,  # 오늘 매수 후보군 티커의 ATR 값 배열
    buy_commission_rate: float,  # ★★★ 추가: 매수 수수료율 인자
    debug_mode: bool = False, # ★★★ 추가
):
    """
    Vectorized new entry signal processing for all simulations.

    This function implements the new entry logic:
    1. Identify simulations with available slots (max_stocks > current_stocks).
    2. For those simulations, select top N candidates based on ATR.
    3. Execute 1st order buy for the selected tickers.
    """
    num_combinations, num_stocks_total, max_splits, _ = positions_state.shape

    # --- Step 1: Calculate available slots for each simulation ---
    # 현재 보유 종목 수 계산 (종목별로 하나라도 포지션이 있으면 1, 아니면 0)
    has_any_position = cp.any(
        positions_state[..., 0] > 0, axis=2
    )  # Shape: (num_combinations, num_stocks_total)
    current_num_stocks = cp.sum(has_any_position, axis=1)  # Shape: (num_combinations,)

    max_stocks_per_sim = param_combinations[:, 0]

    available_slots = cp.maximum(0, max_stocks_per_sim - current_num_stocks).astype(
        cp.int32
    )

    if not cp.any(available_slots > 0) or candidate_tickers_for_day.size == 0:
        return portfolio_state, positions_state

    # --- Step 2: Prepare candidate data ---
    # 오늘 진입 가능한 후보가 없으면 종료
    if candidate_tickers_for_day.size == 0:
        return portfolio_state, positions_state

    # ATR 기준으로 후보군 내림차순 정렬 (이미 정렬되었다고 가정하지만, 안전하게 한번 더)
    sort_indices = cp.argsort(candidate_atrs_for_day)[::-1]
    sorted_candidate_indices = candidate_tickers_for_day[sort_indices]

    # --- Step 3: Iterate through candidates and execute buys ---
    # 이 부분은 순차적으로 처리해야 함 (최상위 후보부터 슬롯을 채워나가야 하므로)
    # 하지만 시뮬레이션 간에는 병렬 처리가 가능

    investment_per_order = portfolio_state[:, 1]  # Shape: (num_combinations,)
    current_capital = portfolio_state[:, 0]  # Shape: (num_combinations,)
    
    if debug_mode: # ★★★ 수정
        print(f"  [NEW_BUY_DEBUG] Candidates to check: {len(sorted_candidate_indices)}")
    

    # 한 번에 한 종목씩 처리
    for ticker_idx_cupy in sorted_candidate_indices:
        ticker_idx = int(ticker_idx_cupy)  # cupy 스칼라를 int로 변환
        # 모든 시뮬레이션이 꽉 찼으면 루프 종료
        if cp.all(available_slots <= 0):
            break

        stock_price = current_prices[ticker_idx]
        # 호가 올림 처리하여 실제 매수가 결정
        buy_price = adjust_price_up_gpu(stock_price)
        if buy_price <= 0:
            continue
        # --- ★★★ 자본 확인 로직 수정 ★★★ ---
        # 1. 수수료를 포함한 총 비용 계산
        safe_investment = cp.where(buy_price > 0, investment_per_order, 0)
        quantity_to_buy_f = cp.floor(safe_investment / buy_price)  # float 수량

        # 수수료 포함 총 비용
        total_cost_per_sim = (buy_price * quantity_to_buy_f) * (1 + buy_commission_rate)

        # 2. 자본 충분 여부 확인
        has_capital = current_capital >= total_cost_per_sim
        # --- ★★★ 수정 끝 ★★★ ---
        # 이 종목을 매수할 수 있는 시뮬레이션의 최종 조건
        # 1. 슬롯이 있고 (available_slots > 0)
        # 2. 이 종목을 보유하지 않았고 (is_not_holding)
        # 3. 자본이 충분한가 (아래에서 계산)
        # 4. 호가 올림 처리된 가격이 0보다 큰가 (아래에서 계산)
        # 5. 수수료 포함 총 비용이 자본보다 작은가 (아래에서 계산)

        is_not_holding = ~has_any_position[:, ticker_idx]
        # --- ★★★ 중복 코드 제거 및 안전 로직 통합 ★★★ ---
        # 1. 초기 매수 조건 마스크
        initial_buy_mask = (available_slots > 0) & is_not_holding & has_capital

        if cp.any(initial_buy_mask):
            buy_sim_indices = cp.where(initial_buy_mask)[0]
            
            # --- ★★★ 수정 시작 ★★★ ---
            # 1. 실제 매수가(buy_price)를 호가 올림 처리하여 결정합니다.
            buy_price = adjust_price_up_gpu(stock_price)

            # 2. 수정된 buy_price로 수량을 다시 계산합니다.
            quantity_to_buy = cp.floor(
                investment_per_order[buy_sim_indices] / buy_price
            ).astype(cp.int32)

            # 3. 수수료를 포함한 최종 비용(total_cost)을 계산합니다.
            total_cost = (buy_price * quantity_to_buy) * (1 + buy_commission_rate)
            # --- ★★★ 로그 추가 시작 ★★★ ---
            # 0번 시뮬레이션이 이번 매수 대상에 포함되는지 확인
            is_sim0_buying = cp.any(buy_sim_indices == 0)
            if is_sim0_buying:
                # buy_sim_indices 배열에서 0번 시뮬레이션의 위치(인덱스)를 찾음
                sim0_idx_in_buy_list = cp.where(buy_sim_indices == 0)[0][0]
                
                # 해당 위치의 수량과 비용 정보를 가져옴
                qty = int(quantity_to_buy[sim0_idx_in_buy_list])
                cost = float(total_cost[sim0_idx_in_buy_list])
                
                # 수량이 0보다 클 때만 로그 출력 (실제 매수가 일어났을 때)
                if qty > 0:
                    print(f"  [GPU_TRADE_LOG] TickerIdx: {ticker_idx}, Action: NEW_BUY, Order: 1, "
                          f"Qty: {qty}, Close: {float(stock_price):,.0f}, BuyPrice: {float(buy_price):,.0f}, "
                          f"TotalCost: {cost:,.0f}")
            # 4. 자본 상태는 'portfolio_state' 원본에서 단 한번만 차감합니다.
            portfolio_state[buy_sim_indices, 0] -= total_cost
            
            # 5. 포지션 상태를 업데이트합니다. (매수가는 호가 적용된 buy_price)
            positions_state[buy_sim_indices, ticker_idx, 0, 0] = quantity_to_buy
            positions_state[buy_sim_indices, ticker_idx, 0, 1] = buy_price
            
            available_slots[buy_sim_indices] -= 1
            has_any_position[buy_sim_indices, ticker_idx] = True
            
            # 6. 중복되는 current_capital 차감 로직을 삭제합니다.
            # current_capital[buy_sim_indices] -= cost # 이 라인 삭제 또는 주석 처리


    return portfolio_state, positions_state
