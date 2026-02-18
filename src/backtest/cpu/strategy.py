"""
strategy.py

This module contains the functions for generating the signals for the Magic Split Strategy.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import uuid
import logging

logger = logging.getLogger(__name__)

class Position:
    """ 포지션 정보를 저장하는 데이터 클래스 """
    def __init__(self, buy_price, quantity, order, additional_buy_drop_rate, sell_profit_rate):
        self.position_id = str(uuid.uuid4())
        self.buy_price = buy_price
        self.quantity = quantity
        self.order = order
        self.additional_buy_drop_rate = additional_buy_drop_rate
        self.sell_profit_rate = sell_profit_rate
        self.open_date = None 

class Strategy(ABC):
    @abstractmethod
    def generate_sell_signals(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None): 
        """매도 관련 신호(수익 실현, 손절 등)를 생성합니다."""
        raise NotImplementedError("generate_sell_signals() 메소드를 구현해야 합니다.")

    # 신규 진입과 추가 매수를 위한 두 개의 구체적인 추상 메소드를 정의합니다.
    @abstractmethod
    def generate_new_entry_signals(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
        """신규 종목 진입 신호를 생성합니다."""
        raise NotImplementedError("generate_new_entry_signals() 메소드를 구현해야 합니다.")

    @abstractmethod
    def generate_additional_buy_signals(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
        """기존 보유 종목의 추가 매수 신호를 생성합니다."""
        raise NotImplementedError("generate_additional_buy_signals() 메소드를 구현해야 합니다.")

class MagicSplitStrategy(Strategy):
    def __init__(
        self,
        max_stocks,
        order_investment_ratio,
        additional_buy_drop_rate,
        sell_profit_rate,
        backtest_start_date,
        backtest_end_date,
        additional_buy_priority="lowest_order",
        cooldown_period_days=5,
        # --- [New] Advanced Risk Management Parameters ---
        stop_loss_rate=-0.15,
        max_splits_limit=10,
        max_inactivity_period=90,
        # --- [Issue #67] Candidate Source Config ---
        candidate_source_mode="weekly", # weekly | hybrid_transition | tier
        use_weekly_alpha_gate=False,
        tier_hysteresis_mode="legacy",  # legacy | strict_hysteresis_v1
    ):
        self.max_stocks = max_stocks
        self.order_investment_ratio = order_investment_ratio
        self.additional_buy_drop_rate = additional_buy_drop_rate
        self.sell_profit_rate = sell_profit_rate
        self.backtest_start_date = pd.to_datetime(backtest_start_date)
        self.backtest_end_date = pd.to_datetime(backtest_end_date)
        self.additional_buy_priority = additional_buy_priority
        self.cooldown_period_days = cooldown_period_days
        self.stop_loss_rate = stop_loss_rate
        self.max_splits_limit = max_splits_limit
        self.max_inactivity_period = max_inactivity_period
        
        # [Issue #67]
        self.candidate_source_mode = candidate_source_mode
        self.use_weekly_alpha_gate = use_weekly_alpha_gate
        self.tier_hysteresis_mode = tier_hysteresis_mode
        
        self.investment_per_order = 0
        self.previous_month = -1
        self.cooldown_tracker = {}  # 매도된 종목 추적

    def _use_strict_hysteresis(self):
        mode = str(self.tier_hysteresis_mode).strip().lower()
        return mode == "strict_hysteresis_v1" and self.candidate_source_mode in {"tier", "hybrid_transition"}

    def _load_tier_map_for_holdings(self, data_handler, signal_date, tickers):
        if not tickers:
            return {}

        get_tiers_as_of = getattr(data_handler, "get_tiers_as_of", None)
        if callable(get_tiers_as_of):
            try:
                tier_rows = get_tiers_as_of(as_of_date=signal_date, tickers=tickers)
                return {
                    code: int(meta["tier"])
                    for code, meta in tier_rows.items()
                    if meta is not None and meta.get("tier") is not None
                }
            except Exception as exc:
                exc_info = logger.isEnabledFor(logging.DEBUG)
                logger.warning(
                    "Tier map lookup failed (%s). Falling back to per-ticker lookup.",
                    exc,
                    exc_info=exc_info,
                )

        get_stock_tier_as_of = getattr(data_handler, "get_stock_tier_as_of", None)
        if not callable(get_stock_tier_as_of):
            return {}

        tier_map = {}
        for ticker in tickers:
            tier_info = get_stock_tier_as_of(ticker, signal_date)
            if tier_info is None or tier_info.get("tier") is None:
                continue
            tier_map[ticker] = int(tier_info["tier"])
        return tier_map

    def _resolve_signal_date(self, current_date, trading_dates, current_day_idx, data_handler):
        if current_day_idx is None:
            return pd.to_datetime(current_date)
        return data_handler.get_previous_trading_date(trading_dates, current_day_idx)

    def _calculate_monthly_investment(self, current_date, current_day_idx, trading_dates, portfolio, data_handler):
        # 전일 날짜를 기준으로 자산을 평가하도록 로직 변경
        current_month = current_date.month
        if current_month != self.previous_month:
            # 첫 거래일에는 전일이 없으므로, 초기 자본을 기준으로 투자금을 계산
            if current_day_idx == 0:
                total_portfolio_value = portfolio.initial_cash
            else:
                # 전일 날짜를 가져옴
                previous_day_date = trading_dates[current_day_idx - 1]
                total_portfolio_value = portfolio.get_total_value(previous_day_date, data_handler)
            
            self.investment_per_order = np.float32(total_portfolio_value) * np.float32(self.order_investment_ratio)
            self.previous_month = current_month
    # 신규 진입 신호만 생성하는 함수
    def generate_new_entry_signals(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
        # 각 신호 생성 함수가 독립적으로 호출되므로, 투자금 계산 로직은 각 함수에 포함되어야 합니다.
        self._calculate_monthly_investment(current_date, current_day_idx, trading_dates, portfolio, data_handler)
        buy_signals = []
        signal_date = self._resolve_signal_date(current_date, trading_dates, current_day_idx, data_handler)
        if signal_date is None:
            return buy_signals

        # 신규 매수 신호 생성 로직 (기존 generate_buy_signals의 2번 로직)
        available_slots = self.max_stocks - len(portfolio.positions)
        if logger.isEnabledFor(logging.DEBUG) and (current_date - self.backtest_start_date).days < 15:
            log_msg = (
                f"[CPU_SLOT_DEBUG] {current_date.strftime('%Y-%m-%d')} | "
                f"MaxStocks: {self.max_stocks}, "
                f"CurrentHoldings: {len(portfolio.positions)}, "
                f"AvailableSlots: {available_slots}"
            )
            logger.debug(log_msg)
        
        if available_slots > 0:
            candidate_codes = []
            
            # --- [Issue #67] Candidate Source Logic Start ---
            mode = self.candidate_source_mode
            
            # 1. Weekly (Legacy)
            if mode == "weekly":
                candidate_codes = data_handler.get_filtered_stock_codes(current_date)
                
            # 2. Tier / Hybrid
            elif mode in ["tier", "hybrid_transition"]:
                get_tier_candidates = getattr(data_handler, "get_candidates_with_tier_fallback", None)
                if callable(get_tier_candidates):
                    try:
                        strict_hysteresis = self._use_strict_hysteresis()
                        try:
                            tier_codes, used_tier = get_tier_candidates(
                                signal_date,
                                allow_tier2_fallback=not strict_hysteresis,
                            )
                        except TypeError:
                            # Backward-compat: legacy signature(date)만 가진 구현체 지원
                            tier_codes, used_tier = get_tier_candidates(signal_date)
                        candidate_codes = tier_codes

                        if used_tier == 'TIER_2_FALLBACK':
                             logger.debug(
                                 "[Strategy] %s | Fallback: Tier 1 (0) -> Tier 2 (%s)",
                                 current_date.date(),
                                 len(candidate_codes),
                             )

                        if strict_hysteresis and used_tier == "TIER_2_FALLBACK":
                            logger.debug(
                                "[Strategy] %s | Strict hysteresis: Tier1 empty -> skip new entry",
                                current_date.date(),
                            )
                            candidate_codes = []
                            used_tier = "NO_TIER1_CANDIDATES"

                        if strict_hysteresis and used_tier == "NO_TIER1_CANDIDATES":
                            logger.debug(
                                "[Strategy] %s | Strict hysteresis: no Tier1 candidates (fail-close)",
                                current_date.date(),
                            )

                        if mode == "hybrid_transition" and self.use_weekly_alpha_gate:
                            weekly_codes = data_handler.get_filtered_stock_codes(current_date)
                            weekly_set = set(weekly_codes)
                            original_count = len(candidate_codes)
                            candidate_codes = [code for code in candidate_codes if code in weekly_set]

                            logger.debug(
                                "[Strategy] %s | Hybrid: Tier(%s, %s) & Weekly(%s) -> Intersection(%s)",
                                current_date.date(),
                                used_tier,
                                original_count,
                                len(weekly_codes),
                                len(candidate_codes),
                            )
                    except Exception as exc:
                        exc_info = logger.isEnabledFor(logging.DEBUG)
                        logger.warning(
                            "Tier candidate lookup failed (%s). Falling back to weekly.",
                            exc,
                            exc_info=exc_info,
                        )
                        candidate_codes = data_handler.get_filtered_stock_codes(current_date)
                else:
                    # Fallback if method missing (should not happen with correct DataHandler)
                    logger.warning("get_candidates_with_tier_fallback missing. Falling back to weekly.")
                    candidate_codes = data_handler.get_filtered_stock_codes(current_date)
            else:
                # Default fallback
                candidate_codes = data_handler.get_filtered_stock_codes(current_date)
            # --- [Issue #67] Logic End ---
            
            active_candidates = []
            for code in candidate_codes:
                # 쿨다운 체크 로직을 GPU와 동일하게 거래일 인덱스 차이로 
                is_in_cooldown = False
                if self.cooldown_tracker.get(code) is not None:
                    if (current_day_idx - self.cooldown_tracker.get(code)) < self.cooldown_period_days:
                        is_in_cooldown = True

                if code in portfolio.positions or is_in_cooldown:
                    continue
                active_candidates.append(code)

            market_cap_map = {}
            get_market_caps_as_of = getattr(data_handler, "get_market_caps_as_of", None)
            if callable(get_market_caps_as_of) and active_candidates:
                try:
                    market_cap_map = get_market_caps_as_of(signal_date, active_candidates)
                except Exception as exc:
                    exc_info = logger.isEnabledFor(logging.DEBUG)
                    logger.warning(
                        "MarketCap as-of lookup failed (%s). Falling back to 0 for ranking.",
                        exc,
                        exc_info=exc_info,
                    )
            
            candidate_atrs = []
            for ticker in active_candidates:
                signal_row = data_handler.get_stock_row_as_of(
                    ticker, signal_date, self.backtest_start_date, self.backtest_end_date
                )
                if signal_row is None or "atr_14_ratio" not in signal_row.index:
                    continue
                latest_atr = signal_row["atr_14_ratio"]
                signal_close = signal_row.get("close_price")
                if pd.notna(latest_atr) and float(latest_atr) > 0 and pd.notna(signal_close):
                    market_cap_raw = market_cap_map.get(ticker)
                    market_cap_q = int(float(market_cap_raw) // 1_000_000) if market_cap_raw and market_cap_raw > 0 else 0
                    atr_q = int(round(float(latest_atr) * 10000))
                    candidate_atrs.append(
                        {
                            "ticker": ticker,
                            "atr_14_ratio": latest_atr,
                            "market_cap_q": market_cap_q,
                            "atr_q": atr_q,
                            "signal_close_price": signal_close,
                        }
                    )
                        
            # GPU와 동일한 결정론 정렬 기준:
            # 1) market_cap_q desc, 2) atr_q desc, 3) ticker asc
            sorted_candidates = sorted(
                candidate_atrs,
                key=lambda x: (-x["market_cap_q"], -x["atr_q"], x["ticker"]),
            )
            # 슬롯이 찰 때까지만 신호 생성
            num_new_entries = 0
            for entry_rank, candidate in enumerate(sorted_candidates):
                if num_new_entries >= available_slots: break
                ticker = candidate["ticker"]
                signal_close_price = candidate["signal_close_price"]
                if signal_close_price > 0 and self.investment_per_order > 0:
                    new_pos = Position(signal_close_price, 0, 1, self.additional_buy_drop_rate, self.sell_profit_rate)
                    buy_signals.append(
                        self._create_buy_signal(
                            current_date,
                            ticker,
                            self.investment_per_order,
                            new_pos,
                            1,
                            entry_rank,
                            "신규 진입",
                            signal_close_price,
                        )
                    )
                    num_new_entries += 1

        # 신규 매수 신호 내에서의 정렬
        buy_signals.sort(key=lambda s: (s["priority_group"], s["sort_metric"], s["ticker"]))
        
        for signal in buy_signals:
            signal["start_date"] = self.backtest_start_date
            signal["end_date"] = self.backtest_end_date
            
        return buy_signals

    # 추가 매수 신호만 생성하는 함수
    def generate_additional_buy_signals(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
        # 투자금 재계산은 신규 매수에서 이미 했을 수 있지만, 이 함수가 단독으로 사용될 수도 있으므로 여기서도 호출합니다.
        self._calculate_monthly_investment(current_date, current_day_idx, trading_dates, portfolio, data_handler)
        buy_signals = []
        signal_date = self._resolve_signal_date(current_date, trading_dates, current_day_idx, data_handler)
        if signal_date is None:
            return buy_signals
        strict_hysteresis = self._use_strict_hysteresis()
        tier_map = {}
        if strict_hysteresis:
            tier_map = self._load_tier_map_for_holdings(
                data_handler,
                signal_date,
                list(portfolio.positions.keys()),
            )

        # 추가 매수 신호 생성 로직
        for ticker in list(portfolio.positions.keys()):
            # 당일 매도된 종목은 추가 매수 안 함
            if self.cooldown_tracker.get(ticker) == current_day_idx:
                continue

            if strict_hysteresis:
                ticker_tier = tier_map.get(ticker)
                if ticker_tier is None or ticker_tier < 1 or ticker_tier > 2:
                    continue
            
            positions = portfolio.positions[ticker]
            # [핵심 수정] GPU의 'is_not_new_today' 규칙과 동일한 보호 장치
            # 이 종목의 첫 번째 매수(order 1)가 오늘 이전에 이루어졌는지 확인합니다.
            first_position = next((p for p in positions if p.order == 1), None)
            if first_position is None or first_position.open_date is None or first_position.open_date >= current_date:
                continue
            
            signal_row = data_handler.get_stock_row_as_of(
                ticker, signal_date, self.backtest_start_date, self.backtest_end_date
            )
            if signal_row is None:
                continue

            if not any(p.order == 1 for p in positions):
                continue

            if len(positions) >= self.max_splits_limit:
                continue    
            
            signal_close = signal_row["close_price"]
            signal_low = signal_row["low_price"]
            if signal_close <= 0:
                continue

            last_pos = portfolio.positions[ticker][-1]
            buy_trigger_price = last_pos.buy_price * (1 - self.additional_buy_drop_rate)
            
            if signal_low <= buy_trigger_price:
                if self.investment_per_order > 0:
                    new_pos = Position(signal_close, 0, len(portfolio.positions[ticker]) + 1, self.additional_buy_drop_rate, self.sell_profit_rate)
                    sort_metric = len(portfolio.positions[ticker]) if self.additional_buy_priority == "lowest_order" else -((last_pos.buy_price - signal_close) / last_pos.buy_price)
                    buy_signals.append(self._create_buy_signal(current_date, ticker, self.investment_per_order, new_pos, 2, sort_metric, "추가 매수(하락)", buy_trigger_price))

        # 추가 매수 신호 내에서의 정렬
        buy_signals.sort(key=lambda s: (s["priority_group"], s["sort_metric"], s["ticker"]))

        for signal in buy_signals:
            signal["start_date"] = self.backtest_start_date
            signal["end_date"] = self.backtest_end_date

        return buy_signals

    


    def _create_sell_signal(self, date, ticker, position, reason, trigger_price):
        return {"date": date, "ticker": ticker, "type": "SELL", "quantity": position.quantity, "position": position, "reason_for_trade": reason, "trigger_price": trigger_price}

    def _create_buy_signal(self, date, ticker, investment_amount, position, priority, sort_metric, reason, trigger_price):
        return {"date": date, "ticker": ticker, "type": "BUY", "investment_amount": investment_amount, "position": position, "priority_group": priority, "sort_metric": sort_metric, "reason_for_trade": reason, "trigger_price": trigger_price}
    def generate_sell_signals(self, current_date, portfolio, data_handler,trading_dates,current_day_idx=None):
        self._calculate_monthly_investment(current_date, current_day_idx, trading_dates, portfolio, data_handler)
        signals = []
        signal_date = self._resolve_signal_date(current_date, trading_dates, current_day_idx, data_handler)
        if signal_date is None:
            return signals
        strict_hysteresis = self._use_strict_hysteresis()
        tier_map = {}
        if strict_hysteresis:
            tier_map = self._load_tier_map_for_holdings(
                data_handler,
                signal_date,
                list(portfolio.positions.keys()),
            )

        for ticker in list(portfolio.positions.keys()):
            row = data_handler.get_stock_row_as_of(
                ticker, signal_date, self.backtest_start_date, self.backtest_end_date
            )
            if row is None:
                continue

            current_price = row["close_price"]
            current_high = row["high_price"]
            if current_price <= 0: continue

            positions = portfolio.positions[ticker]
            avg_buy_price = sum(p.quantity * p.buy_price for p in positions) / sum(p.quantity for p in positions)

            # --- 1. 리스크 관리 조건 확인 (종목 전체 청산) ---
            liquidate = False
            reason = ""
            trigger_price = current_price

            if strict_hysteresis and tier_map.get(ticker) is not None and tier_map.get(ticker) >= 3:
                liquidate = True
                reason = "Tier3 강제 청산"
                trigger_price = current_price

            if not liquidate and current_price <= avg_buy_price * (1.0 + self.stop_loss_rate):
                liquidate = True
                reason = "손절매 (평균가 기준)"
                trigger_price = avg_buy_price * (1.0 + self.stop_loss_rate)

            if not liquidate:
                last_trade_idx = portfolio.last_trade_day_indices.get(ticker)
                if last_trade_idx is not None:
                    # GPU와 동일한 '실제 거래일' 기준으로 비활성 기간 계산
                    days_inactive = current_day_idx - last_trade_idx
                    
                    if days_inactive >= self.max_inactivity_period -1: # GPU 로직과 동일하게 >= 및 -1 조건 사용
                        liquidate = True
                        reason = "매매 미발생 기간 초과"

            if liquidate:
                # 임시 객체 생성 대신, 실제 보유 중인 모든 포지션에 대해 매도 신호를 생성합니다.
                # 이는 execution 단계에서 올바른 position_id를 참조하여 포지션을 제거할 수 있도록 보장합니다.
                for p in positions:
                    signals.append(self._create_sell_signal(current_date, ticker, p, reason, trigger_price))
                
                # 신호가 하나라도 생성되었다면 쿨다운을 설정합니다.
                if positions:
                    self.cooldown_tracker[ticker] = current_day_idx
                
                continue # 이 종목에 대한 다른 매도(수익실현) 검사는 건너뜁니다.
            # reversed()를 사용하여 가장 최근 매수 포지션부터 순회
            for p in reversed(positions):
                # [규칙] 당일 매수한 포지션은 익절 대상에서 제외
                if p.open_date is not None and p.open_date >= current_date:
                    continue
                
                sell_trigger_price = p.buy_price * (1 + self.sell_profit_rate)
                if current_high >= sell_trigger_price:
                    signals.append(self._create_sell_signal(current_date, ticker, p, "수익 실현", sell_trigger_price))
                    # 1차 매도 여부와 상관없이 개별 익절이므로 cooldown만 설정
                    self.cooldown_tracker[ticker] = current_day_idx
        
        for signal in signals:
            signal["start_date"] = self.backtest_start_date
            signal["end_date"] = self.backtest_end_date
        return signals
