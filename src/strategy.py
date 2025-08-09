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

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, current_date, portfolio, data_handler):
        raise NotImplementedError("generate_signals() 메소드를 구현해야 합니다.")

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

    def generate_signals(self, current_date, portfolio, data_handler):
        self._calculate_monthly_investment(current_date, portfolio, data_handler)

        sell_signals = []
        buy_signals = []

        existing_pos_signals = self._generate_signals_for_existing_positions(current_date, portfolio, data_handler)
        sell_signals.extend([s for s in existing_pos_signals if s["type"] == "SELL"])
        buy_signals.extend([s for s in existing_pos_signals if s["type"] == "BUY"])

        if len(portfolio.positions) < self.max_stocks:
            new_entry_signals = self._generate_signals_for_new_entries(current_date, portfolio, data_handler)
            buy_signals.extend(new_entry_signals)

        buy_signals.sort(key=lambda s: (s["priority_group"], s["sort_metric"]))

        all_signals = sell_signals + buy_signals # 매도 신호를 항상 우선 처리
        for signal in all_signals:
            signal["start_date"] = self.backtest_start_date
            signal["end_date"] = self.backtest_end_date

        return all_signals

    def _generate_signals_for_existing_positions(self, current_date, portfolio, data_handler):
        signals = []
        for ticker in list(portfolio.positions.keys()):
            stock_data = data_handler.load_stock_data(ticker, self.backtest_start_date, self.backtest_end_date)
            if stock_data is None or stock_data.empty or current_date not in stock_data.index:
                continue

            row = stock_data.loc[current_date]
            current_price = row["close_price"]
            if current_price <= 0: continue

            positions = portfolio.positions[ticker]
            avg_buy_price = sum(p.quantity * p.buy_price for p in positions) / sum(p.quantity for p in positions)

            # --- 1. 리스크 관리 조건 확인 (종목 전체 청산) ---
            liquidate = False
            reason = ""
            trigger_price = current_price

            # 조건 1: 평균 매수가 대비 손절률 도달
            if current_price <= avg_buy_price * (1.0 + self.stop_loss_rate):
                liquidate = True
                reason = "손절매 (평균가 기준)"
                trigger_price = avg_buy_price * (1.0 + self.stop_loss_rate)

            # 조건 2: 최대 매매 미발생 기간 초과
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
                continue # 청산 신호 발생 시, 다른 신호(수익실현, 추가매수) 생성 안함

            # --- 2. 수익 실현 신호 생성 (가장 나중에 매수한 포지션부터) ---
            # reversed()를 사용하여 LIFO(후입선출) 방식으로 수익 실현
            for p in reversed(positions):
                sell_trigger_price = p.buy_price * (1 + self.sell_profit_rate)
                if current_price >= sell_trigger_price:
                    signals.append(self._create_sell_signal(current_date, ticker, p, "수익 실현", sell_trigger_price))
                    self.cooldown_tracker[ticker] = current_date
                    # break 문을 제거하여 하루에 여러 포지션 동시 수익 실현 가능하게 함

            # --- 3. 추가 매수 신호 생성 ---
            # 당일 매도 신호(수익실현 포함)가 생성된 종목은 추가 매수 안함
            if any(s["ticker"] == ticker and s["type"] == "SELL" for s in signals):
                continue

            # 최대 분할매수 횟수 제한
            if len(positions) >= self.max_splits_limit:
                continue

            if positions:
                last_pos = positions[-1]
                buy_trigger_price = last_pos.buy_price * (1 - self.additional_buy_drop_rate)
                if current_price <= buy_trigger_price:
                    quantity = int(self.investment_per_order / current_price)
                    if quantity > 0:
                        new_pos = Position(current_price, quantity, len(positions) + 1, self.additional_buy_drop_rate, self.sell_profit_rate)
                        sort_metric = len(positions) if self.additional_buy_priority == "lowest_order" else -((last_pos.buy_price - current_price) / last_pos.buy_price)
                        signals.append(self._create_buy_signal(current_date, ticker, quantity, new_pos, 2, sort_metric, "추가 매수(하락)", buy_trigger_price))
        return signals

    def _generate_signals_for_new_entries(self, current_date, portfolio, data_handler):
        signals = []
        available_slots = self.max_stocks - len(portfolio.positions)
        if available_slots <= 0: return signals

        candidate_codes = data_handler.get_filtered_stock_codes(current_date)
        if not candidate_codes: return signals

        active_candidates = []
        for code in candidate_codes:
            if code in portfolio.positions or code in self.cooldown_tracker and (np.busday_count(self.cooldown_tracker[code].date(), current_date.date()) < self.cooldown_period_days):
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

        for candidate in sorted_candidates:
            if len(signals) >= available_slots: break
            ticker = candidate["ticker"]
            stock_data = data_handler.load_stock_data(ticker, self.backtest_start_date, self.backtest_end_date)
            row = stock_data.loc[current_date]
            current_price = row["close_price"]
            if current_price > 0:
                quantity = int(self.investment_per_order / current_price)
                if quantity > 0:
                    new_pos = Position(current_price, quantity, 1, self.additional_buy_drop_rate, self.sell_profit_rate)
                    signals.append(self._create_buy_signal(current_date, ticker, quantity, new_pos, 1, -candidate["atr_14_ratio"], "신규 진입", current_price))
        return signals

    def _create_sell_signal(self, date, ticker, position, reason, trigger_price):
        return {"date": date, "ticker": ticker, "type": "SELL", "quantity": position.quantity, "position": position, "reason_for_trade": reason, "trigger_price": trigger_price}

    def _create_buy_signal(self, date, ticker, quantity, position, priority, sort_metric, reason, trigger_price):
        return {"date": date, "ticker": ticker, "type": "BUY", "quantity": quantity, "position": position, "priority_group": priority, "sort_metric": sort_metric, "reason_for_trade": reason, "trigger_price": trigger_price}
