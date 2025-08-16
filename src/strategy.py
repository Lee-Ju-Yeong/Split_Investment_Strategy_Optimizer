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
    def generate_sell_signals(self, current_date, portfolio, data_handler):
        """매도 관련 신호(수익 실현, 손절 등)를 생성합니다."""
        raise NotImplementedError("generate_sell_signals() 메소드를 구현해야 합니다.")

    @abstractmethod
    def generate_buy_signals(self, current_date, portfolio, data_handler):
        """매수 관련 신호(신규 진입, 추가 매수)를 생성합니다."""
        raise NotImplementedError("generate_buy_signals() 메소드를 구현해야 합니다.")

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

    def _calculate_monthly_investment(self, current_date, portfolio, data_handler):
        current_month = current_date.month
        if current_month != self.previous_month:
            total_portfolio_value = portfolio.get_total_value(current_date, data_handler)
            self.investment_per_order = total_portfolio_value * self.order_investment_ratio
            self.previous_month = current_month

    # [변경] 매수 신호만 생성하는 함수
    def generate_buy_signals(self, current_date, portfolio, data_handler):
        self._calculate_monthly_investment(current_date, portfolio, data_handler)
        buy_signals = []

        # 1. 추가 매수 신호 생성
        for ticker in list(portfolio.positions.keys()):
            # 당일 매도된 종목은 추가 매수 안 함 (cooldown_tracker에 오늘 날짜가 기록되었는지 확인)
            if self.cooldown_tracker.get(ticker) == current_date:
                continue

            stock_data = data_handler.load_stock_data(ticker, self.backtest_start_date, self.backtest_end_date)
            if stock_data is None or stock_data.empty or current_date not in stock_data.index:
                continue
            # [추가] "1차 포지션 존재" 규칙: 1차 포지션이 없으면 추가 매수 불가
            positions = portfolio.positions[ticker]
            if not any(p.order == 1 for p in positions):
                continue

            if len(positions) >= self.max_splits_limit:
                continue    
            current_close = stock_data.loc[current_date, "close_price"]
            current_low = stock_data.loc[current_date, "low_price"]
            if current_close <= 0: continue

            last_pos = portfolio.positions[ticker][-1]
            buy_trigger_price = last_pos.buy_price * (1 - self.additional_buy_drop_rate)
            # [수정] 매수 조건 판단을 '종가'가 아닌 '저가' 기준으로 변경합니다.
            if current_low <= buy_trigger_price:
                # [유지] 매수 수량 계산은 보수적으로 '종가' 기준을 유지합니다.
                quantity = int(self.investment_per_order / current_close)
                if quantity > 0:
                    new_pos = Position(current_close, quantity, len(portfolio.positions[ticker]) + 1, self.additional_buy_drop_rate, self.sell_profit_rate)
                    sort_metric = len(portfolio.positions[ticker]) if self.additional_buy_priority == "lowest_order" else -((last_pos.buy_price - current_close) / last_pos.buy_price)
                    buy_signals.append(self._create_buy_signal(current_date, ticker, quantity, new_pos, 2, sort_metric, "추가 매수(하락)", buy_trigger_price))

        # 2. 신규 매수 신호 생성
        available_slots = self.max_stocks - len(portfolio.positions)
        if available_slots > 0:
            candidate_codes = data_handler.get_filtered_stock_codes(current_date)
            
            active_candidates = []
            for code in candidate_codes:
                if code in portfolio.positions or (self.cooldown_tracker.get(code) and (np.busday_count(self.cooldown_tracker[code].date(), current_date.date()) < self.cooldown_period_days)):
                    continue
                active_candidates.append(code)

            candidate_atrs = []
            for ticker in active_candidates:
                stock_data = data_handler.load_stock_data(ticker, self.backtest_start_date, self.backtest_end_date)
                if stock_data is not None and not stock_data.empty and "atr_14_ratio" in stock_data.columns and current_date in stock_data.index:
                    latest_atr = stock_data.loc[current_date, "atr_14_ratio"]
                    if pd.notna(latest_atr):
                        candidate_atrs.append({"ticker": ticker, "atr_14_ratio": latest_atr})

            sorted_candidates = sorted(candidate_atrs, key=lambda x: x["atr_14_ratio"], reverse=True)
            
            # 슬롯이 찰 때까지만 신호 생성
            num_new_entries = 0
            for candidate in sorted_candidates:
                if num_new_entries >= available_slots: break
                # ... (신호 생성 로직은 기존과 동일) ...
                ticker = candidate["ticker"]
                stock_data = data_handler.load_stock_data(ticker, self.backtest_start_date, self.backtest_end_date)
                current_price = stock_data.loc[current_date, "close_price"]
                if current_price > 0:
                    quantity = int(self.investment_per_order / current_price)
                    if quantity > 0:
                        new_pos = Position(current_price, quantity, 1, self.additional_buy_drop_rate, self.sell_profit_rate)
                        buy_signals.append(self._create_buy_signal(current_date, ticker, quantity, new_pos, 1, -candidate["atr_14_ratio"], "신규 진입", current_price))
                        num_new_entries += 1

        # 모든 매수 신호를 우선순위에 따라 정렬
        buy_signals.sort(key=lambda s: (s["priority_group"], s["sort_metric"]))
        
        for signal in buy_signals:
            signal["start_date"] = self.backtest_start_date
            signal["end_date"] = self.backtest_end_date
            
        return buy_signals


    def _create_sell_signal(self, date, ticker, position, reason, trigger_price):
        return {"date": date, "ticker": ticker, "type": "SELL", "quantity": position.quantity, "position": position, "reason_for_trade": reason, "trigger_price": trigger_price}

    def _create_buy_signal(self, date, ticker, quantity, position, priority, sort_metric, reason, trigger_price):
        return {"date": date, "ticker": ticker, "type": "BUY", "quantity": quantity, "position": position, "priority_group": priority, "sort_metric": sort_metric, "reason_for_trade": reason, "trigger_price": trigger_price}
    def generate_sell_signals(self, current_date, portfolio, data_handler):
        self._calculate_monthly_investment(current_date, portfolio, data_handler)
        signals = []

        for ticker in list(portfolio.positions.keys()):
            stock_data = data_handler.load_stock_data(ticker, self.backtest_start_date, self.backtest_end_date)
            if stock_data is None or stock_data.empty or current_date not in stock_data.index:
                continue

            row = stock_data.loc[current_date]
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
                last_trade_date = portfolio.last_trade_dates.get(ticker)
                if last_trade_date:
                    days_inactive = np.busday_count(pd.to_datetime(last_trade_date).date(), current_date.date())
                    if days_inactive > self.max_inactivity_period:
                        liquidate = True
                        reason = "매매 미발생 기간 초과"

            if liquidate:
                for p in positions:
                    signals.append(self._create_sell_signal(current_date, ticker, p, reason, trigger_price))
                self.cooldown_tracker[ticker] = current_date
                continue 

            # [수정] reversed()를 사용하여 가장 최근 매수 포지션부터 순회
            for p in reversed(positions):
                # [규칙] 당일 매수한 포지션은 익절 대상에서 제외
                if p.open_date is not None and p.open_date >= current_date:
                    continue
                
                sell_trigger_price = p.buy_price * (1 + self.sell_profit_rate)
                if current_high >= sell_trigger_price:
                    signals.append(self._create_sell_signal(current_date, ticker, p, "수익 실현", sell_trigger_price))
                    # [수정] 1차 매도 여부와 상관없이 개별 익절이므로 cooldown만 설정
                    self.cooldown_tracker[ticker] = current_date
        
        for signal in signals:
            signal["start_date"] = self.backtest_start_date
            signal["end_date"] = self.backtest_end_date
        return signals