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


@cp.fuse()
def get_tick_size_gpu(price_array):
    """주가 배열에 따른 호가 단위 배열을 반환합니다."""
    condlist = [
        price_array < 2000,
        price_array < 5000,
        price_array < 20000,
        price_array < 50000,
        price_array < 200000,
        price_array < 500000,
    ]
    choicelist = [1, 5, 10, 50, 100, 500]
    return cp.select(condlist, choicelist, default=1000)


@cp.fuse()
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

    # 1. 전체 청산된 포지션들 정리
    positions_state[full_liquidation_mask, 0] = 0  # quantity to 0
    positions_state[full_liquidation_mask, 1] = 0  # buy_price to 0

    # 2. 부분 매도된 포지션들 정리
    positions_state[partial_sell_mask, 0] = 0
    positions_state[partial_sell_mask, 1] = 0

    return portfolio_state, positions_state


def _process_additional_buy_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
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
    has_any_position = cp.any(
        has_positions, axis=2
    )  # Shape: (num_combinations, num_stocks)

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

    # --- Step 4: Check capital availability ---
    # Calculate required capital for additional buys
    quantities_to_buy = (
        investment_per_order.squeeze(-1) / current_prices_2d
    )  # Shape: (num_combinations, num_stocks)
    required_capital_per_stock = quantities_to_buy * current_prices_2d

    # Check if simulation has enough capital for each potential buy
    has_capital = current_capital.reshape(-1, 1) >= required_capital_per_stock

    # Final condition: all conditions must be met
    final_buy_condition = (
        can_add_position & has_capital
    )  # 이 has_capital은 아직 안전하지 않음
    if cp.any(final_buy_condition):
        # --- ★★★ 안전한 자본 차감 로직 추가 ★★★ ---

        # 1. 실제 매수 대상의 인덱스 가져오기
        sim_indices, stock_indices = cp.where(final_buy_condition)

        # 2. 비용 계산 (벡터화)
        prices_for_buy = current_prices[stock_indices]
        inv_per_order_for_buy = investment_per_order[sim_indices, 0, 0]
        quantities_to_buy = cp.floor(inv_per_order_for_buy / prices_for_buy).astype(
            cp.int32
        )
        costs = quantities_to_buy * prices_for_buy

        # 3. 자본이 충분한지 최종 확인
        capital_for_buy = current_capital[sim_indices]
        final_buy_mask = capital_for_buy >= costs

        # 4. 최종 매수 대상에 대해서만 상태 업데이트 수행
        if cp.any(final_buy_mask):
            final_sim_indices = sim_indices[final_buy_mask]
            final_stock_indices = stock_indices[final_buy_mask]
            final_quantities = quantities_to_buy[final_buy_mask]
            final_costs = costs[final_buy_mask]
            final_next_splits = next_split_indices_to_buy[
                final_sim_indices, final_stock_indices
            ]

            # 포지션 및 자본 업데이트
            # 주의: 이 부분은 고급 인덱싱이며, CuPy 버전에 따라 동작이 다를 수 있음
            # 가장 안전한 방법은 루프를 사용하는 것
            for i in range(len(final_sim_indices)):
                sim_idx = int(final_sim_indices[i])
                stock_idx = int(final_stock_indices[i])
                split_idx = int(final_next_splits[i])

                positions_state[sim_idx, stock_idx, split_idx, 0] = final_quantities[i]
                positions_state[sim_idx, stock_idx, split_idx, 1] = current_prices[
                    stock_idx
                ]
                portfolio_state[sim_idx, 0] -= final_costs[i]

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
):
    """
    Main GPU-accelerated backtesting function for the MagicSplitStrategy.
    """
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
    print(f"max_stocks_param: {max_stocks_param}")
    num_tickers = len(all_tickers)

    # The actual dimension used for arrays must match the full list of tickers
    positions_state = cp.zeros(
        (num_combinations, num_tickers, max_splits_limit, 2), dtype=cp.float32
    )
    print(f"portfolio_state: {portfolio_state.get()}")
    print(f"positions_state: {cp.any(positions_state > 0).get()}")
    daily_portfolio_values = cp.zeros(
        (num_combinations, num_trading_days), dtype=cp.float32
    )

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
        capital_before_day = portfolio_state[0, 0].get()
        positions_before_day = cp.sum(positions_state[0, :, :, 0] > 0).get()
        print(
            f"\n--- Day {i+1}/{num_trading_days}: {current_date.strftime('%Y-%m-%d')} ---"
        )
        print(
            f"[BEGIN] Capital: {capital_before_day:,.0f} | Total Positions: {positions_before_day}"
        )
        # ---

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
                if len(candidates_of_the_week) > 0:
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
                all_tickers,
                sell_commission_rate=execution_params[
                    "sell_commission_rate"
                ],  # ★★★ 추가
                sell_tax_rate=execution_params["sell_tax_rate"],  # ★★★ 추가
            )
            # 2. Process Additional Buy Signals
            portfolio_state, positions_state = _process_additional_buy_signals_gpu(
                portfolio_state, positions_state, param_combinations, current_prices
            )
            # 3. Process Sell Signals
            portfolio_state, positions_state = _process_sell_signals_gpu(
                portfolio_state, positions_state, param_combinations, current_prices
            )

            capital_after_actions = portfolio_state[
                0, 0
            ].get()  # 모든 매매 행위 후의 자본
            if capital_after_actions != capital_before_actions:
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
        final_capital_of_day = portfolio_state[0, 0].get()
        final_stock_value_of_day = (
            stock_values[0].get()
            if "stock_values" in locals() and stock_values.size > 0
            else 0
        )
        final_total_value_of_day = final_capital_of_day + final_stock_value_of_day
        final_positions_of_day = cp.sum(positions_state[0, :, :, 0] > 0).get()

        print(
            f"[END]   Capital: {final_capital_of_day:,.0f} | Stock Val: {final_stock_value_of_day:,.0f} | Total Val: {final_total_value_of_day:,.0f} | Positions: {final_positions_of_day}"
        )
        # ---

        if (i + 1) % 252 == 0:
            year = current_date.year
            print(f"    - Simulating year: {year} ({i+1}/{num_trading_days})")

        print("🎉 GPU backtesting simulation finished.")

    return daily_portfolio_values


def _process_new_entry_signals_gpu(
    portfolio_state: cp.ndarray,
    positions_state: cp.ndarray,
    param_combinations: cp.ndarray,
    current_prices: cp.ndarray,
    candidate_tickers_for_day: cp.ndarray,  # 오늘 매수 후보군 티커의 '인덱스' 배열
    candidate_atrs_for_day: cp.ndarray,  # 오늘 매수 후보군 티커의 ATR 값 배열
    all_tickers: list,
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

    print(f"  [NEW_BUY_DEBUG] Candidates to check: {len(sorted_candidate_indices)}")

    # 한 번에 한 종목씩 처리
    for ticker_idx_cupy in sorted_candidate_indices:
        ticker_idx = int(ticker_idx_cupy)  # cupy 스칼라를 int로 변환
        # 모든 시뮬레이션이 꽉 찼으면 루프 종료
        if cp.all(available_slots <= 0):
            break

        stock_price = current_prices[ticker_idx]
        if stock_price <= 0:
            continue

        # 이 종목을 매수할 수 있는 시뮬레이션의 최종 조건
        # 1. 슬롯이 있고 (available_slots > 0)
        # 2. 이 종목을 보유하지 않았고 (is_not_holding)
        # 3. 자본이 충분한가 (아래에서 계산)
        safe_investment = cp.where(stock_price > 0, investment_per_order, 0)
        required_capital = stock_price * cp.floor(safe_investment / stock_price)
        has_capital = current_capital >= required_capital

        is_not_holding = ~has_any_position[:, ticker_idx]
        # --- ★★★ 중복 코드 제거 및 안전 로직 통합 ★★★ ---
        # 1. 초기 매수 조건 마스크
        initial_buy_mask = (available_slots > 0) & is_not_holding & has_capital

        if cp.any(initial_buy_mask):
            buy_sim_indices = cp.where(initial_buy_mask)[0]
            # 2. 비용 계산
            quantity_to_buy = cp.floor(
                investment_per_order[buy_sim_indices] / stock_price
            ).astype(cp.int32)
            cost = quantity_to_buy * stock_price
            # 3. 자본 상태를 직접 업데이트하며 최종 매수 실행
            portfolio_state[buy_sim_indices, 0] -= cost
            # 4. 나머지 상태 업데이트
            positions_state[buy_sim_indices, ticker_idx, 0, 0] = quantity_to_buy
            positions_state[buy_sim_indices, ticker_idx, 0, 1] = stock_price
            available_slots[buy_sim_indices] -= 1
            has_any_position[buy_sim_indices, ticker_idx] = True
            current_capital[buy_sim_indices] -= cost

    return portfolio_state, positions_state
