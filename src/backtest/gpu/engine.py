"""
GPU backtest engine runner.
"""

from datetime import timedelta
import time

import cudf
import cupy as cp
import pandas as pd

from .data import (
    _collect_candidate_rank_metrics_asof,
    build_ranked_candidate_payload,
    create_gpu_data_tensors,
)
from .logic import (
    _calculate_monthly_investment_gpu,
    _process_additional_buy_signals_gpu,
    _process_new_entry_signals_gpu,
    _process_sell_signals_gpu,
)
from .utils import _resolve_signal_date_for_gpu


def _forward_fill_asof_tensor(price_tensor: cp.ndarray) -> cp.ndarray:
    """
    Forward-fill along day axis for as-of semantics.
    Missing values are represented by 0.0 and remain 0.0 until first valid price.
    """
    if price_tensor.size == 0:
        return price_tensor

    valid_mask = price_tensor > 0
    day_indices = cp.arange(price_tensor.shape[0], dtype=cp.int32).reshape(-1, 1)
    last_valid_day_idx = cp.where(valid_mask, day_indices, 0)
    ticker_indices = cp.arange(price_tensor.shape[1], dtype=cp.int32).reshape(1, -1)

    try:
        last_valid_day_idx = cp.maximum.accumulate(last_valid_day_idx, axis=0)
        return price_tensor[last_valid_day_idx, ticker_indices]
    except NotImplementedError:
        # 일부 CuPy 버전에서 ufunc.accumulate가 미지원이므로 호환 경로를 사용한다.
        filled = price_tensor.copy()
        for day_idx in range(1, filled.shape[0]):
            filled[day_idx] = cp.where(filled[day_idx] > 0, filled[day_idx], filled[day_idx - 1])
        return filled


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
    tier_tensor: cp.ndarray = None # [Issue #67]
):
    # --- 1. 상태 배열 초기화 ---
    num_combinations = param_combinations.shape[0]
    num_trading_days = len(trading_date_indices)
    num_tickers = len(all_tickers)
    cooldown_period_days = execution_params.get("cooldown_period_days", 5)
    
    # Config from exec_params
    candidate_source_mode = execution_params.get("candidate_source_mode", "weekly")
    use_weekly_alpha_gate = execution_params.get("use_weekly_alpha_gate", False)
    parity_mode = str(execution_params.get("parity_mode", "fast")).strip().lower()
    strict_cash_rounding = parity_mode == "strict"
    tier_hysteresis_mode = str(execution_params.get("tier_hysteresis_mode", "legacy")).strip().lower()
    strict_hysteresis_enabled = (
        tier_hysteresis_mode == "strict_hysteresis_v1"
        and candidate_source_mode in {"tier", "hybrid_transition"}
    )
    entry_tier1_only = strict_hysteresis_enabled
    hold_max_tier = 2 if strict_hysteresis_enabled else 0
    force_liquidate_tier3 = strict_hysteresis_enabled
    valid_modes = {'weekly', 'tier', 'hybrid_transition'}
    if candidate_source_mode not in valid_modes:
        print(f"[Warning] Invalid candidate_source_mode '{candidate_source_mode}'. Falling back to 'weekly'.")
        candidate_source_mode = 'weekly'
    if candidate_source_mode in ('tier', 'hybrid_transition') and tier_tensor is None:
        raise ValueError(f"tier_tensor is required when candidate_source_mode='{candidate_source_mode}'")

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
    needs_weekly_candidates = (
        candidate_source_mode == "weekly"
        or (candidate_source_mode == "hybrid_transition" and use_weekly_alpha_gate)
    )
    weekly_filtered_reset_idx = weekly_filtered_gpu.reset_index() if needs_weekly_candidates else None
    print(f"Data prepared for GPU backtest. Mode: {candidate_source_mode}")
    progress_log_interval_days = int(execution_params.get("progress_log_interval_days", 100))
    progress_log_enabled = bool(execution_params.get("progress_log_enabled", True))
    run_start_ts = time.time()
    processed_days = 0

    previous_prices_gpu = cp.zeros(num_tickers, dtype=cp.float32)
    # --- 2.  메인 루프를 월 블록 단위로 변경 ---
    
    #  각 월의 첫 거래일 인덱스를 미리 계산
    monthly_grouper = trading_dates_pd_cpu.to_series().groupby(pd.Grouper(freq='MS'))
    month_first_dates = monthly_grouper.first().dropna()
    month_start_indices = trading_dates_pd_cpu.get_indexer(month_first_dates).tolist()
    data_tensors = create_gpu_data_tensors(all_data_reset_idx, all_tickers, trading_dates_pd_cpu)
    open_prices_tensor = data_tensors["open"]
    close_prices_tensor = data_tensors["close"]
    high_prices_tensor = data_tensors["high"]
    low_prices_tensor = data_tensors["low"]
    if parity_mode == "strict":
        close_prices_tensor = _forward_fill_asof_tensor(close_prices_tensor)
        high_prices_tensor = _forward_fill_asof_tensor(high_prices_tensor)
        low_prices_tensor = _forward_fill_asof_tensor(low_prices_tensor)
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
            signal_date, signal_day_idx = _resolve_signal_date_for_gpu(day_idx, trading_dates_pd_cpu)
            # 텐서에서 하루치 데이터 슬라이싱
            current_opens_gpu = open_prices_tensor[day_idx]
            current_prices_gpu = close_prices_tensor[day_idx]
            current_highs_gpu  = high_prices_tensor[day_idx]
            current_lows_gpu   = low_prices_tensor[day_idx]
            if signal_day_idx >= 0:
                signal_closes_gpu = close_prices_tensor[signal_day_idx]
                signal_highs_gpu = high_prices_tensor[signal_day_idx]
                signal_lows_gpu = low_prices_tensor[signal_day_idx]
                if tier_tensor is not None:
                    signal_tiers_gpu = tier_tensor[signal_day_idx]
                else:
                    signal_tiers_gpu = cp.zeros(num_tickers, dtype=cp.int8)
            else:
                signal_closes_gpu = cp.zeros(num_tickers, dtype=cp.float32)
                signal_highs_gpu = cp.zeros(num_tickers, dtype=cp.float32)
                signal_lows_gpu = cp.zeros(num_tickers, dtype=cp.float32)
                signal_tiers_gpu = cp.zeros(num_tickers, dtype=cp.int8)

            # --- [Issue #67] Candidate Selection Logic ---
            candidate_indices_list = []
            
            # (A) Tier Selection (Primary for Tier/Hybrid)
            if candidate_source_mode in ['tier', 'hybrid_transition'] and tier_tensor is not None:
                # 1. Select Tier 1
                if signal_day_idx >= 0:
                    signal_tiers = tier_tensor[signal_day_idx] # (num_tickers,)
                    tier1_mask = (signal_tiers == 1)

                    if cp.any(tier1_mask):
                        candidate_indices = cp.where(tier1_mask)[0]
                    elif entry_tier1_only:
                        candidate_indices = cp.array([], dtype=cp.int32)
                    else:
                        # Fallback to Tier 2 (<= 2)
                        tier2_mask = (signal_tiers > 0) & (signal_tiers <= 2)
                        candidate_indices = cp.where(tier2_mask)[0]

                    candidate_indices_list = candidate_indices.tolist() # Convert to list for intersection
            
            # (B) Weekly Selection (Primary for Weekly, Gate for Hybrid)
            weekly_indices_list = []
            if needs_weekly_candidates:
                past_or_equal_data = weekly_filtered_reset_idx[weekly_filtered_reset_idx['date'] < current_date]
                if not past_or_equal_data.empty:
                    latest_filter_date = past_or_equal_data['date'].max()
                    candidates_of_the_week = weekly_filtered_reset_idx[weekly_filtered_reset_idx['date'] == latest_filter_date]
                    candidate_tickers_list = candidates_of_the_week['ticker'].to_arrow().to_pylist()
                    weekly_indices_list = [ticker_to_idx.get(t) for t in candidate_tickers_list if t in ticker_to_idx]
            
            # (C) Combine
            final_candidate_indices = []
            
            if candidate_source_mode == 'weekly':
                final_candidate_indices = weekly_indices_list
            elif candidate_source_mode == 'tier':
                final_candidate_indices = candidate_indices_list
            elif candidate_source_mode == 'hybrid_transition':
                if use_weekly_alpha_gate:
                    weekly_index_set = set(weekly_indices_list)
                    final_candidate_indices = [
                        idx for idx in candidate_indices_list if idx in weekly_index_set
                    ]
                else:
                    final_candidate_indices = candidate_indices_list

            # (D) Valid Data Check + deterministic ranking metrics (MarketCap -> ATR -> Ticker)
            if final_candidate_indices and signal_date is not None:
                final_candidate_tickers = [all_tickers[i] for i in final_candidate_indices]
                valid_candidate_metrics_df = _collect_candidate_rank_metrics_asof(
                    all_data_reset_idx=all_data_reset_idx,
                    final_candidate_tickers=final_candidate_tickers,
                    signal_date=signal_date,
                )

                if valid_candidate_metrics_df is None or valid_candidate_metrics_df.empty:
                    candidate_tickers_for_day = cp.array([], dtype=cp.int32)
                    candidate_atrs_for_day = cp.array([], dtype=cp.float32)
                else:
                    candidate_indices_final, valid_atrs_final, ranked_records = build_ranked_candidate_payload(
                        valid_candidate_metrics_df=valid_candidate_metrics_df,
                        ticker_to_idx=ticker_to_idx,
                    )

                    if ranked_records:
                        candidate_tickers_for_day = cp.asarray(candidate_indices_final, dtype=cp.int32)
                        candidate_atrs_for_day = cp.asarray(valid_atrs_final, dtype=cp.float32)
                        if debug_mode:
                            preview = ", ".join([f"{ticker}" for ticker, _, _, _ in ranked_records[:10]])
                            print(
                                f"[GPU_CANDIDATE_DEBUG] {current_date.strftime('%Y-%m-%d')} "
                                f"(signal={signal_date.strftime('%Y-%m-%d')}) "
                                f"ranked={len(ranked_records)} top10=[{preview}]"
                            )
                    else:
                        candidate_tickers_for_day = cp.array([], dtype=cp.int32)
                        candidate_atrs_for_day = cp.array([], dtype=cp.float32)
                        if debug_mode:
                            print(
                                f"[GPU_CANDIDATE_DEBUG] {current_date.strftime('%Y-%m-%d')} "
                                f"(signal={signal_date.strftime('%Y-%m-%d')}) ranked=0"
                            )
            else:
                candidate_tickers_for_day = cp.array([], dtype=cp.int32)
                candidate_atrs_for_day = cp.array([], dtype=cp.float32)
                if debug_mode:
                    signal_str = signal_date.strftime('%Y-%m-%d') if signal_date is not None else "None"
                    print(
                        f"[GPU_CANDIDATE_DEBUG] {current_date.strftime('%Y-%m-%d')} "
                        f"(signal={signal_str}) ranked=0 (no candidates before metric filter)"
                    )

            # 2-2. 월별 투자금 재계산
            # --- 신호 처리 함수 호출 (기존과 동일) ---
            portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, sell_occurred_today_mask = _process_sell_signals_gpu(
                portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, day_idx,
                param_combinations,
                current_opens_gpu, current_prices_gpu, current_highs_gpu,
                signal_closes_gpu, signal_highs_gpu, signal_day_idx,
                execution_params["sell_commission_rate"], execution_params["sell_tax_rate"],
                signal_tiers=signal_tiers_gpu if strict_hysteresis_enabled else None,
                force_liquidate_tier3=force_liquidate_tier3,
                strict_cash_rounding=strict_cash_rounding,
                debug_mode=debug_mode, all_tickers=all_tickers, trading_dates_pd_cpu=trading_dates_pd_cpu
            )
            portfolio_state, positions_state, last_trade_day_idx_state = _process_new_entry_signals_gpu(
                portfolio_state, positions_state, cooldown_state, last_trade_day_idx_state, day_idx,
                cooldown_period_days, param_combinations, current_opens_gpu,
                candidate_tickers_for_day, candidate_atrs_for_day,
                execution_params["buy_commission_rate"], log_buffer, log_counter, debug_mode,
                all_tickers=all_tickers,
                strict_cash_rounding=strict_cash_rounding,
            )
            portfolio_state, positions_state, last_trade_day_idx_state = _process_additional_buy_signals_gpu(
                portfolio_state, positions_state, last_trade_day_idx_state, sell_occurred_today_mask, day_idx,
                param_combinations,
                current_opens_gpu, signal_closes_gpu, signal_lows_gpu, signal_day_idx,
                execution_params["buy_commission_rate"], log_buffer, log_counter, debug_mode,
                all_tickers=all_tickers,
                signal_tiers=signal_tiers_gpu if strict_hysteresis_enabled else None,
                hold_max_tier=hold_max_tier,
                strict_cash_rounding=strict_cash_rounding,
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
            processed_days += 1
            if (
                progress_log_enabled
                and progress_log_interval_days > 0
                and (
                    processed_days % progress_log_interval_days == 0
                    or processed_days == num_trading_days
                )
            ):
                elapsed_sec = max(time.time() - run_start_ts, 1e-6)
                progress = processed_days / max(num_trading_days, 1)
                remaining_days = max(num_trading_days - processed_days, 0)
                eta_sec = (elapsed_sec / processed_days) * remaining_days if processed_days > 0 else 0.0
                print(
                    "[GPU_PROGRESS] "
                    f"{processed_days}/{num_trading_days} ({progress*100:.1f}%) "
                    f"elapsed={timedelta(seconds=int(elapsed_sec))} "
                    f"eta={timedelta(seconds=int(eta_sec))}"
                )
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

    
       
