"""
strategy.py

This module contains the functions for generating the signals for the Magic Split Strategy.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import uuid

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
        
        self.investment_per_order = 0
        self.previous_month = -1
        self.cooldown_tracker = {}  # 매도된 종목 추적

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
        if (current_date - self.backtest_start_date).days < 15:
            from tqdm import tqdm
            log_msg = (
                f"[CPU_SLOT_DEBUG] {current_date.strftime('%Y-%m-%d')} | "
                f"MaxStocks: {self.max_stocks}, "
                f"CurrentHoldings: {len(portfolio.positions)}, "
                f"AvailableSlots: {available_slots}"
            )
            tqdm.write(log_msg)
        
        if available_slots > 0:
            candidate_codes = data_handler.get_filtered_stock_codes(current_date)
            get_codes_with_tier = getattr(data_handler, "get_filtered_stock_codes_with_tier", None)
            if callable(get_codes_with_tier):
                try:
                    tier_codes = get_codes_with_tier(signal_date, allowed_tiers=(1, 2))
                    if isinstance(tier_codes, (list, tuple, set)):
                        candidate_codes = list(tier_codes)
                except Exception:
                    candidate_codes = data_handler.get_filtered_stock_codes(current_date)
            
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
            
            candidate_atrs = []
            for ticker in active_candidates:
                signal_row = data_handler.get_stock_row_as_of(
                    ticker, signal_date, self.backtest_start_date, self.backtest_end_date
                )
                if signal_row is None or "atr_14_ratio" not in signal_row.index:
                    continue
                latest_atr = signal_row["atr_14_ratio"]
                signal_close = signal_row.get("close_price")
                if pd.notna(latest_atr) and pd.notna(signal_close):
                    candidate_atrs.append(
                        {
                            "ticker": ticker,
                            "atr_14_ratio": latest_atr,
                            "signal_close_price": signal_close,
                        }
                    )
                        
            # GPU와 동일한 정렬 기준 적용 (1. ATR 내림차순, 2. Ticker 오름차순)
            # Python의 stable sort 특성을 활용: 먼저 2차 기준으로 정렬 후, 1차 기준으로 정렬
            candidates_sorted_by_ticker = sorted(candidate_atrs, key=lambda x: x["ticker"])
            sorted_candidates = sorted(candidates_sorted_by_ticker, key=lambda x: x["atr_14_ratio"], reverse=True)
            # 슬롯이 찰 때까지만 신호 생성
            num_new_entries = 0
            for candidate in sorted_candidates:
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
                            -candidate["atr_14_ratio"],
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

        # 추가 매수 신호 생성 로직
        for ticker in list(portfolio.positions.keys()):
            # 당일 매도된 종목은 추가 매수 안 함
            if self.cooldown_tracker.get(ticker) == current_day_idx:
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

            if current_price <= avg_buy_price * (1.0 + self.stop_loss_rate):
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
