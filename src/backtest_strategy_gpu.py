"""
backtest_strategy_gpu.py

This module contains the GPU-accelerated versions of backtesting logic
using CuPy for massive parallelization.
"""

import cupy as cp
import cudf
import pandas as pd
import time 

def create_gpu_data_tensors(all_data_gpu: cudf.DataFrame, all_tickers: list, trading_dates_pd: pd.Index, debug_mode: bool = False) -> dict:
    """
    [수정] 인덱스 매핑을 사용하여 Long-format cuDF를 Wide-format CuPy 텐서로 직접 변환합니다.
    이 방식은 pivot/join보다 명시적이고 데이터 정렬 오류에 강건합니다.
    """
    print("⏳ Creating wide-format GPU data tensors using direct index mapping...")
    start_time = time.time()

    num_days = len(trading_dates_pd)
    num_tickers = len(all_tickers)

    # 1. 날짜와 티커를 정수 인덱스로 매핑하는 딕셔너리 생성
    #    trading_dates_pd는 DatetimeIndex, all_tickers는 list 여야 함
    date_map = {date.to_datetime64(): i for i, date in enumerate(trading_dates_pd)}
    ticker_map = {ticker: i for i, ticker in enumerate(all_tickers)}
    
    # cuDF의 map 함수를 사용하기 위해 매핑 딕셔너리를 cudf.Series로 변환
    date_map_gdf = cudf.Series(date_map)
    ticker_map_gdf = cudf.Series(ticker_map)
    
    # 2. 원본 데이터에 정수 인덱스 컬럼 추가
    #    .astype('datetime64[ns]')로 타입을 맞춰줘야 map이 잘 동작함
    all_data_gpu['day_idx'] = all_data_gpu['date'].astype('datetime64[ns]').map(date_map_gdf)
    all_data_gpu['ticker_idx'] = all_data_gpu['ticker'].map(ticker_map_gdf)
    
    # 유효한 인덱스만 필터링
    data_valid = all_data_gpu.dropna(subset=['day_idx', 'ticker_idx'])
    
    # 3. 필요한 각 컬럼에 대해 (num_days, num_tickers) 텐서 생성하고 값 채우기
    tensors = {}
    # [수정] atr_14_ratio를 텐서화 대상에 추가
    for col_name in ['close_price', 'high_price', 'low_price', 'atr_14_ratio']:
        # 0으로 채워진 빈 텐서 생성
        tensor = cp.zeros((num_days, num_tickers), dtype=cp.float32)
        
        # 값을 채워넣을 위치(row, col)와 값(value)을 CuPy 배열로 추출
        day_indices = cp.asarray(data_valid['day_idx'].astype(cp.int32))
        ticker_indices = cp.asarray(data_valid['ticker_idx'].astype(cp.int32))
        
        # 해당 컬럼에 NaN이 아닌 유효한 값이 있는 행만 필터링
        valid_values_mask = data_valid[col_name].notna()
        valid_data_for_col = data_valid[valid_values_mask]

        day_indices_col = cp.asarray(valid_data_for_col['day_idx'].astype(cp.int32))
        ticker_indices_col = cp.asarray(valid_data_for_col['ticker_idx'].astype(cp.int32))
        values_col = cp.asarray(valid_data_for_col[col_name].astype(cp.float32))

        # CuPy의 고급 인덱싱(fancy indexing)을 사용하여 값을 한 번에 할당
        tensor[day_indices_col, ticker_indices_col] = values_col
        
        # 키 이름에서 _price, _ratio 등 접미사 제거
        key_name = col_name.replace('_price', '').replace('_ratio', '')
        tensors[key_name] = tensor



    print(f"✅ GPU Tensors created successfully in {time.time() - start_time:.2f}s.")
    return tensors


def create_daily_candidate_mask(weekly_filtered_gpu: cudf.DataFrame, trading_dates_pd: pd.DatetimeIndex, ticker_to_idx: dict) -> cp.ndarray:
    """
    [신규] 매일의 후보군을 나타내는 Boolean 마스크 텐서를 미리 생성합니다.
    """
    print("⏳ Creating daily candidate mask tensor...")
    start_time = time.time()
    
    num_days = len(trading_dates_pd)
    num_tickers = len(ticker_to_idx)
    
    candidate_mask = cp.zeros((num_days, num_tickers), dtype=cp.bool_)
    
    # 주간 필터링 날짜들을 오름차순으로 정렬
    unique_filter_dates = weekly_filtered_gpu.index.unique().sort_values()
    
    # [수정] asof는 단일 라벨에만 동작하므로, 루프를 통해 각 거래일의 active filter date를 찾음
    # 이 작업은 사전 계산 단계에서 한 번만 수행되므로 성능 영향은 미미함.
    s = pd.Series(unique_filter_dates.to_pandas(), index=unique_filter_dates.to_pandas())
    # [수정] CPU의 '<' 로직과 일치시키기 위해 asof를 적용하기 전 날짜에서 미세한 시간을 뺍니다.
    adjusted_trading_dates = trading_dates_pd - pd.Timedelta(nanoseconds=1)
    daily_active_filter_dates = s.asof(adjusted_trading_dates).values
    
    # 날짜별로 그룹화된 주간 필터링 종목들
    grouped_candidates = weekly_filtered_gpu.groupby(level=0)['ticker'].collect()
    
    # 각 거래일에 대해, 해당하는 주간 필터의 종목들을 마스크에 True로 설정
    for day_idx, current_date in enumerate(trading_dates_pd):
        active_filter_date = daily_active_filter_dates[day_idx]
        
        # pd.NaT는 아무 필터도 적용되지 않은 초기 상태를 의미
        if pd.notna(active_filter_date):
            candidate_tickers_for_day_pd = grouped_candidates.loc[active_filter_date]
            
            # cuDF Series를 Python 리스트로 변환 후, ticker_to_idx로 인덱싱
            candidate_indices = [ticker_to_idx[ticker] for ticker in candidate_tickers_for_day_pd if ticker in ticker_to_idx]
            
            if candidate_indices:
                candidate_mask[day_idx, cp.array(candidate_indices)] = True

    print(f"✅ Daily candidate mask created in {time.time() - start_time:.2f}s.")
    return candidate_mask

def get_tick_size_gpu(price_array):
    """ Vectorized tick size calculation on GPU. """
    # cp.select는 내부적으로 큰 임시 배열들을 생성하여 메모리 사용량이 많습니다.
    # cp.where를 연쇄적으로 사용하여 단일 결과 배열을 점진적으로 채워나가 메모리 사용량을 최소화합니다.
    # 기본값(1000원)으로 결과 배열을 초기화합니다.
    result = cp.full_like(price_array, 1000, dtype=cp.int32)
    
    # 가격이 낮은 조건부터 순서대로 값을 덮어씁니다.
    result = cp.where(price_array < 500000, 500, result)
    result = cp.where(price_array < 200000, 100, result)
    result = cp.where(price_array < 50000, 50, result)
    result = cp.where(price_array < 20000, 10, result)
    result = cp.where(price_array < 5000, 5, result)
    result = cp.where(price_array < 2000, 1, result)
    
    return result

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
    debug_mode: bool = False,
    all_tickers: list = None
):
    """ [수정] cumsum과 searchsorted를 활용한 완전 병렬 추가 매수 로직 """
    # 1. 추가 매수 조건에 맞는 모든 후보 탐색 (기존과 동일)
    add_buy_drop_rates = param_combinations[:, 2:3]
    max_splits_limits = param_combinations[:, 6:7]
    quantities_state = positions_state[..., 0]
    buy_prices_state = positions_state[..., 1]
    
    has_positions = quantities_state > 0
    num_positions = cp.sum(has_positions, axis=2)
    has_any_position = num_positions > 0
    if not cp.any(has_any_position):
        return portfolio_state, positions_state, last_trade_day_idx_state

    last_pos_mask = (cp.cumsum(has_positions, axis=2) == num_positions[:, :, cp.newaxis]) & has_positions
    last_buy_prices = cp.sum(buy_prices_state * last_pos_mask, axis=2)
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

    # 2. 모든 후보에 대한 비용 및 우선순위 계산 (벡터화)
    sim_indices, stock_indices = cp.where(initial_buy_mask)
    
    # 비용 계산
    candidate_investments = portfolio_state[sim_indices, 1]
    candidate_trigger_prices = trigger_prices[sim_indices, stock_indices]
    candidate_highs = current_highs[stock_indices]
    epsilon = cp.float32(1.0)
    price_basis = cp.where(candidate_highs <= candidate_trigger_prices - epsilon, candidate_highs, candidate_trigger_prices)
    exec_prices = adjust_price_up_gpu(price_basis)
    
    quantities = cp.zeros_like(exec_prices, dtype=cp.int32)
    valid_price_mask = exec_prices > 0
    quantities[valid_price_mask] = cp.floor(candidate_investments[valid_price_mask] / exec_prices[valid_price_mask])

    costs = exec_prices * quantities
    commissions = cp.floor(costs * buy_commission_rate)
    total_costs = costs + commissions

    # 우선순위 점수 계산
    add_buy_priorities = param_combinations[sim_indices, 4]
    scores_lowest_order = num_positions[sim_indices, stock_indices]
    candidate_last_buy_prices = last_buy_prices[sim_indices, stock_indices]
    candidate_current_prices = current_prices[stock_indices]
    price_epsilon = 1e-9
    scores_highest_drop = (candidate_last_buy_prices - candidate_current_prices) / (candidate_last_buy_prices + price_epsilon)
    priority_scores = cp.where(add_buy_priorities == 0, scores_lowest_order, -scores_highest_drop)

    # 3. 시뮬레이션 ID와 우선순위로 후보 정렬
    # lexsort는 마지막 행부터 정렬하므로, 우선순위가 낮은 키(stock_indices)를 먼저, 높은 키(sim_indices)를 나중에 넣습니다.
    # (sim_idx 오름차순 -> priority_score 오름차순 -> stock_idx 오름차순)
    sort_keys = cp.vstack((stock_indices, priority_scores, sim_indices))
    sorted_indices = cp.lexsort(sort_keys)
    
    sorted_sims = sim_indices[sorted_indices]
    sorted_stocks = stock_indices[sorted_indices]
    sorted_costs = total_costs[sorted_indices]
    sorted_quantities = quantities[sorted_indices]
    sorted_exec_prices = exec_prices[sorted_indices]

    # 4. 세그먼트화된 누적 합계를 사용해 감당 가능한 매수 결정
    # 각 시뮬레이션 그룹의 시작점을 찾습니다.
    unique_sims, sim_start_indices = cp.unique(sorted_sims, return_index=True)
    
    # 전체 누적 합계 계산
    global_cumsum = cp.cumsum(sorted_costs)
    
    # `repeat`를 사용하여 세그먼트별로 차감할 값을 효율적으로 전파합니다.
    # (maximum.accumulate가 일부 CuPy 버전에서 지원되지 않는 문제를 우회)
    run_lengths = cp.diff(cp.concatenate((sim_start_indices, cp.array([len(sorted_sims)]))))
    run_lengths_list = run_lengths.tolist() # .repeat()를 위해 파이썬 리스트로 변환
    
    # 각 세그먼트에서 빼야 할 값 (첫 세그먼트는 0, 나머지는 이전 세그먼트의 누적 합)
    segment_subtraction_values = cp.concatenate((cp.array([0], dtype=global_cumsum.dtype), global_cumsum[sim_start_indices[1:] - 1]))
    
    prefix_sum_broadcast = cp.repeat(segment_subtraction_values, run_lengths_list)

    # 세그먼트화된 (시뮬레이션별) 누적 합계
    per_sim_cumsum = global_cumsum - prefix_sum_broadcast
    
    # 각 후보에 대해 해당 시뮬레이션의 가용 자본을 broadcast
    sim_capitals = portfolio_state[unique_sims, 0]
    capital_broadcast = cp.repeat(sim_capitals, run_lengths_list)
    
    # 최종 매수 마스크: 시뮬레이션별 누적 비용이 가용 자본을 넘지 않는 후보들
    final_buy_mask = per_sim_cumsum <= capital_broadcast
    final_buy_mask &= (sorted_quantities > 0) # 수량이 0인 매수는 제외

    if not cp.any(final_buy_mask):
        return portfolio_state, positions_state, last_trade_day_idx_state

    # 5. 최종 매수 목록을 기반으로 상태 병렬 업데이트
    # 매수가 실행될 후보들의 정보
    final_sims = sorted_sims[final_buy_mask]
    final_stocks = sorted_stocks[final_buy_mask]
    final_quantities = sorted_quantities[final_buy_mask]
    final_exec_prices = sorted_exec_prices[final_buy_mask]
    final_costs = sorted_costs[final_buy_mask]

    # 자본 업데이트
    # 매수가 발생한 시뮬레이션과 각 시뮬레이션별 총비용 계산
    unique_bought_sims, bought_sim_starts = cp.unique(final_sims, return_index=True)
    total_cost_per_sim = cp.add.reduceat(final_costs, bought_sim_starts)
    
    # `subtract.at`이 float32를 지원하지 않는 문제를 우회하기 위해,
    # 차감할 비용을 담은 임시 배열을 생성한 후, 전체를 한 번에 뺍니다.
    costs_to_subtract = cp.zeros_like(portfolio_state[:, 0])
    costs_to_subtract[unique_bought_sims] = total_cost_per_sim
    portfolio_state[:, 0] -= costs_to_subtract

    # 포지션 업데이트
    # 추가 매수가 들어갈 비어있는 split_idx를 찾습니다.
    # 추가 매수는 종목당 하루에 최대 한 번이므로, 현재 보유 차수가 곧 비어있는 인덱스가 됩니다.
    split_indices = num_positions[final_sims, final_stocks]
    
    positions_state[final_sims, final_stocks, split_indices, 0] = final_quantities
    positions_state[final_sims, final_stocks, split_indices, 1] = final_exec_prices
    positions_state[final_sims, final_stocks, split_indices, 2] = current_day_idx
    
    # 마지막 거래일 업데이트
    # 중복된 (sim, stock)이 있을 수 있으므로 unique 처리 후 업데이트
    unique_final_trades, _ = cp.unique(cp.vstack([final_sims, final_stocks]), axis=1, return_index=True)
    last_trade_day_idx_state[unique_final_trades[0], unique_final_trades[1]] = current_day_idx

    # 디버깅 로그
    if debug_mode and cp.any(cp.isin(final_sims, cp.array([0]))):
        sim0_mask = (final_sims == 0)
        if cp.any(sim0_mask):
            sim0_stocks = final_stocks[sim0_mask]
            sim0_quants = final_quantities[sim0_mask]
            sim0_prices = final_exec_prices[sim0_mask]
            
            capital_after = portfolio_state[0, 0].item()
            
            print(f"[GPU_ADD_BUY_SUMMARY] Day {current_day_idx}, Sim 0 | Buys: {sim0_stocks.size} | Capital After: {capital_after:,.0f}")
            for i in range(sim0_stocks.size):
                ticker_code = all_tickers[sim0_stocks[i].item()]
                print(f"  └─ Stock {sim0_stocks[i].item()}({ticker_code}) | Qty: {sim0_quants[i].item():,.0f} @ {sim0_prices[i].item():,.0f}")

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
    debug_mode: bool = False,
    all_tickers: list = None
):
    """ [최적화] 슬롯과 자본 제약을 모두 고려한 완전 병렬 신규 매수 로직 """
    # --- 1. 진입 조건 및 기본 정보 계산 ---
    has_any_position = cp.any(positions_state[..., 0] > 0, axis=2)
    current_num_stocks = cp.sum(has_any_position, axis=1)
    max_stocks_per_sim = param_combinations[:, 0]
    available_slots = cp.maximum(0, max_stocks_per_sim - current_num_stocks).astype(cp.int32)

    if not cp.any(available_slots > 0) or candidate_tickers_for_day.size == 0:
        return portfolio_state, positions_state, last_trade_day_idx_state

    num_simulations = param_combinations.shape[0]
    num_candidates = len(candidate_tickers_for_day)

    # --- 2. 모든 (시뮬레이션, 후보) 쌍에 대한 매수 조건 및 비용 일괄 계산 ---
    sim_indices_expanded = cp.repeat(cp.arange(num_simulations), num_candidates)
    candidate_indices_in_list = cp.tile(cp.arange(num_candidates), num_simulations)
    candidate_ticker_indices = candidate_tickers_for_day[candidate_indices_in_list]

    # 조건 검사
    is_holding = has_any_position[sim_indices_expanded, candidate_ticker_indices]
    is_in_cooldown = (cooldown_state[sim_indices_expanded, candidate_ticker_indices] != -1) & \
                     ((current_day_idx - cooldown_state[sim_indices_expanded, candidate_ticker_indices]) < cooldown_period_days)
    
    # 비용 계산
    capital_expanded = portfolio_state[sim_indices_expanded, 0]
    investment_per_order = portfolio_state[sim_indices_expanded, 1]
    buy_prices = adjust_price_up_gpu(current_prices[candidate_ticker_indices])
    
    quantities = cp.zeros_like(buy_prices, dtype=cp.int32)
    valid_price_mask = buy_prices > 0
    quantities[valid_price_mask] = cp.floor(investment_per_order[valid_price_mask] / buy_prices[valid_price_mask])
    
    costs = buy_prices * quantities
    commissions = cp.floor(costs * buy_commission_rate)
    total_costs = costs + commissions

    # --- 3. 모든 잠재적 매수 후보 필터링 및 우선순위 부여 ---
    initial_buy_mask = ~is_holding & ~is_in_cooldown & (quantities > 0)
    # CPU와 동일한 자금 관리 원칙 적용: 이상적인 투자금보다 현금이 적으면 매수 시도조차 안함
    initial_buy_mask &= (capital_expanded >= investment_per_order)

    # 실제 후보들의 1D 인덱스
    valid_indices = cp.where(initial_buy_mask)[0]
    if valid_indices.size == 0:
        return portfolio_state, positions_state, last_trade_day_idx_state

    # 후보 정보 필터링
    sim_indices = sim_indices_expanded[valid_indices]
    stock_indices = candidate_ticker_indices[valid_indices]
    costs = total_costs[valid_indices]
    quantities = quantities[valid_indices]
    buy_prices = buy_prices[valid_indices]
    
    # 우선순위 점수 (음수 ATR)
    priority_scores = -candidate_atrs_for_day[candidate_indices_in_list[valid_indices]]

    # --- 4. 슬롯 제약 병렬 처리 ---
    # 시뮬레이션 ID와 우선순위로 정렬
    sort_keys_slot = cp.vstack((stock_indices, priority_scores, sim_indices))
    sorted_indices_slot = cp.lexsort(sort_keys_slot)
    
    sorted_sims_slot = sim_indices[sorted_indices_slot]
    
    # 각 시뮬레이션 그룹 내에서 순위(rank) 계산
    unique_sims_slot, sim_starts_slot = cp.unique(sorted_sims_slot, return_index=True)
    global_rank = cp.arange(len(sorted_sims_slot))
    
    # 각 그룹의 시작 인덱스에 해당하는 글로벌 랭크 값을 그룹 전체에 전파
    start_ranks = cp.zeros_like(global_rank)
    start_ranks[sim_starts_slot] = sim_starts_slot
    # CuPy 12.0+ 에서는 cp.maximum.accumulate 사용 가능
    # 현재는 호환성을 위해 broadcast 방식으로 대체
    run_lengths_slot = cp.diff(cp.concatenate((sim_starts_slot, cp.array([len(sorted_sims_slot)]))))
    broadcasted_starts = cp.repeat(global_rank[sim_starts_slot], run_lengths_slot.tolist())
    
    intra_sim_rank = global_rank - broadcasted_starts
    
    # 각 후보에 대해 해당 시뮬레이션의 가용 슬롯 수를 broadcast
    available_slots_per_sim = available_slots[unique_sims_slot]
    slots_broadcast = cp.repeat(available_slots_per_sim, run_lengths_slot.tolist())
    
    # 슬롯 제약 통과 마스크
    slot_ok_mask = intra_sim_rank < slots_broadcast
    
    # 슬롯 제약을 통과한 후보들만 다시 필터링
    survived_indices_slot = sorted_indices_slot[slot_ok_mask]
    if survived_indices_slot.size == 0:
        return portfolio_state, positions_state, last_trade_day_idx_state

    sim_indices = sim_indices[survived_indices_slot]
    stock_indices = stock_indices[survived_indices_slot]
    costs = costs[survived_indices_slot]
    quantities = quantities[survived_indices_slot]
    buy_prices = buy_prices[survived_indices_slot]
    priority_scores = priority_scores[survived_indices_slot]

    # --- 5. 자본 제약 병렬 처리 (세그먼트 누적 합계) ---
    # 시뮬레이션 ID와 우선순위로 다시 정렬
    sort_keys_capital = cp.vstack((stock_indices, priority_scores, sim_indices))
    sorted_indices_capital = cp.lexsort(sort_keys_capital)

    sorted_sims = sim_indices[sorted_indices_capital]
    sorted_stocks = stock_indices[sorted_indices_capital]
    sorted_costs = costs[sorted_indices_capital]
    sorted_quantities = quantities[sorted_indices_capital]
    sorted_buy_prices = buy_prices[sorted_indices_capital]

    # 시뮬레이션별 누적 합계 계산
    unique_sims, sim_starts = cp.unique(sorted_sims, return_index=True)
    global_cumsum = cp.cumsum(sorted_costs)
    
    run_lengths = cp.diff(cp.concatenate((sim_starts, cp.array([len(sorted_sims)]))))
    segment_sub_vals = cp.concatenate((cp.array([0]), global_cumsum[sim_starts[1:] - 1]))
    prefix_sum_broadcast = cp.repeat(segment_sub_vals, run_lengths.tolist())
    per_sim_cumsum = global_cumsum - prefix_sum_broadcast
    
    # 가용 자본과 비교
    sim_capitals = portfolio_state[unique_sims, 0]
    capital_broadcast = cp.repeat(sim_capitals, run_lengths.tolist())
    
    final_buy_mask = per_sim_cumsum <= capital_broadcast
    
    if not cp.any(final_buy_mask):
        return portfolio_state, positions_state, last_trade_day_idx_state

    # --- 6. 최종 매수 목록 기반 상태 병렬 업데이트 ---
    final_sims = sorted_sims[final_buy_mask]
    final_stocks = sorted_stocks[final_buy_mask]
    final_quantities = sorted_quantities[final_buy_mask]
    final_buy_prices = sorted_buy_prices[final_buy_mask]
    final_costs = sorted_costs[final_buy_mask]

    # 자본 업데이트
    unique_bought_sims, bought_sim_starts = cp.unique(final_sims, return_index=True)
    total_cost_per_sim = cp.add.reduceat(final_costs, bought_sim_starts)
    
    costs_to_subtract = cp.zeros_like(portfolio_state[:, 0])
    costs_to_subtract[unique_bought_sims] = total_cost_per_sim
    portfolio_state[:, 0] -= costs_to_subtract

    # 포지션 업데이트 (신규 매수는 항상 0번 split_idx에 기록)
    positions_state[final_sims, final_stocks, 0, 0] = final_quantities
    positions_state[final_sims, final_stocks, 0, 1] = final_buy_prices
    positions_state[final_sims, final_stocks, 0, 2] = current_day_idx
    
    # 마지막 거래일 업데이트
    last_trade_day_idx_state[final_sims, final_stocks] = current_day_idx

    # 디버깅 로그
    if debug_mode and cp.any(cp.isin(final_sims, cp.array([0]))):
        sim0_mask = (final_sims == 0)
        if cp.any(sim0_mask):
            sim0_stocks = final_stocks[sim0_mask]
            sim0_quants = final_quantities[sim0_mask]
            sim0_prices = final_buy_prices[sim0_mask]
            capital_after = portfolio_state[0, 0].item()
            
            print(f"[GPU_NEW_BUY_SUMMARY] Day {current_day_idx}, Sim 0 | Buys: {sim0_stocks.size} | Capital After: {capital_after:,.0f}")
            for i in range(sim0_stocks.size):
                ticker_code = all_tickers[sim0_stocks[i].item()]
                print(f"  └─ Stock {sim0_stocks[i].item()}({ticker_code}) | Qty: {sim0_quants[i].item():,.0f} @ {sim0_prices[i].item():,.0f}")

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
    # --- 1. 상태 배열 및 기본 변수 초기화 ---
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

    # --- 2. [최적화] 모든 데이터를 루프 시작 전 텐서로 변환 ---
    # 가격 및 ATR 텐서 생성
    data_tensors = create_gpu_data_tensors(all_data_gpu.reset_index(), all_tickers, trading_dates_pd_cpu, debug_mode=debug_mode)
    close_prices_tensor = data_tensors["close"]
    high_prices_tensor = data_tensors["high"]
    low_prices_tensor = data_tensors["low"]
    atr_tensor = data_tensors["atr_14"]

    # 일별 후보군 마스크 텐서 생성
    candidate_mask_tensor = create_daily_candidate_mask(weekly_filtered_gpu, trading_dates_pd_cpu, ticker_to_idx)

    # --- 3. 메인 백테스팅 루프 (월 단위) ---
    previous_prices_gpu = cp.zeros(num_tickers, dtype=cp.float32)
    monthly_grouper = trading_dates_pd_cpu.to_series().groupby(pd.Grouper(freq='MS'))
    month_first_dates = monthly_grouper.first().dropna()
    month_start_indices = trading_dates_pd_cpu.get_indexer(month_first_dates).tolist()

    for i in range(len(month_start_indices)):
        start_idx = month_start_indices[i]
        end_idx = month_start_indices[i+1] if i + 1 < len(month_start_indices) else num_trading_days
        
        # 월별 투자금 재계산
        eval_prices = previous_prices_gpu if start_idx > 0 else cp.zeros(num_tickers, dtype=cp.float32)
        portfolio_state = _calculate_monthly_investment_gpu(
            portfolio_state, positions_state, param_combinations, eval_prices, trading_dates_pd_cpu[start_idx], debug_mode
        )
        
        # 일일 루프
        for day_idx in range(start_idx, end_idx):
            # [최적화] 텐서에서 하루치 데이터 슬라이싱 (매우 빠름)
            current_prices_gpu = close_prices_tensor[day_idx]
            current_highs_gpu  = high_prices_tensor[day_idx]
            current_lows_gpu   = low_prices_tensor[day_idx]

            # [최적화] 사전 생성된 마스크에서 하루치 후보군 슬라이싱 (매우 빠름)
            daily_candidate_mask = candidate_mask_tensor[day_idx]
            candidate_tickers_for_day = cp.where(daily_candidate_mask)[0]

            if candidate_tickers_for_day.size == 0:
                candidate_atrs_for_day = cp.array([], dtype=cp.float32)
            else:
                # ATR 텐서에서 해당 후보들의 ATR 값만 가져옴
                candidate_atrs_for_day = atr_tensor[day_idx, candidate_tickers_for_day]
                # 후보이지만 ATR 값이 없는 경우(NaN) 제외
                valid_atr_mask = ~cp.isnan(candidate_atrs_for_day)
                candidate_tickers_for_day = candidate_tickers_for_day[valid_atr_mask]
                candidate_atrs_for_day = candidate_atrs_for_day[valid_atr_mask]

            # --- 신호 처리 함수 호출 ---
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
                execution_params["buy_commission_rate"], debug_mode=debug_mode, all_tickers=all_tickers
            )
            portfolio_state, positions_state, last_trade_day_idx_state = _process_additional_buy_signals_gpu(
                portfolio_state, positions_state, last_trade_day_idx_state, sell_occurred_today_mask, day_idx,
                param_combinations, current_prices_gpu, current_lows_gpu, current_highs_gpu,
                execution_params["buy_commission_rate"], debug_mode, all_tickers=all_tickers
            )
        
            # --- 일일 포트폴리오 가치 업데이트 ---
            stock_quantities = cp.sum(positions_state[..., 0], axis=2)
            stock_market_values = stock_quantities * current_prices_gpu
            total_stock_value = cp.sum(stock_market_values, axis=1)
            daily_portfolio_values[:, day_idx] = portfolio_state[:, 0] + total_stock_value
            
            if debug_mode:
                # ... (디버그 로그 출력 부분은 변경 없음) ...
                capital_snapshot = portfolio_state[0, 0].get()
                stock_val_snapshot = total_stock_value[0].get()
                total_val_snapshot = daily_portfolio_values[0, day_idx].get()
                num_pos_snapshot = cp.sum(cp.any(positions_state[0, :, :, 0] > 0, axis=1)).get()
                header = f"\n{'='*120}\n"
                footer = f"\n{'='*120}"
                date_str = trading_dates_pd_cpu[day_idx].strftime('%Y-%m-%d')
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

    return daily_portfolio_values
