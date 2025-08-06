"""
strategy.py

This module contains the functions for generating the signals for the Magic Split Strategy.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .portfolio import Position


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
        cooldown_period_days=5,  # 쿨다운 기간 추가
    ):
        self.max_stocks = max_stocks
        self.order_investment_ratio = order_investment_ratio
        self.additional_buy_drop_rate = additional_buy_drop_rate
        self.sell_profit_rate = sell_profit_rate
        self.backtest_start_date = pd.to_datetime(backtest_start_date)
        self.backtest_end_date = pd.to_datetime(backtest_end_date)
        self.additional_buy_priority = additional_buy_priority
        self.investment_per_order = 0
        self.previous_month = -1
        self.cooldown_period_days = cooldown_period_days  # 쿨다운 기간 설정
        self.cooldown_tracker = {}  # 매도된 종목 추적

    def _calculate_monthly_investment(self, current_date, portfolio, data_handler):
        current_month = current_date.month
        if current_month != self.previous_month:
            total_portfolio_value = portfolio.get_total_value(
                current_date, data_handler
            )
            self.investment_per_order = (
                total_portfolio_value * self.order_investment_ratio
            )
            self.previous_month = current_month

    def generate_signals(self, current_date, portfolio, data_handler):
        self._calculate_monthly_investment(current_date, portfolio, data_handler)

        sell_signals = []
        buy_signals = []

        existing_pos_signals = self._generate_signals_for_existing_positions(
            current_date, portfolio, data_handler
        )
        for s in existing_pos_signals:
            if s["type"] == "BUY":
                buy_signals.append(s)
            else:  # 'SELL' 또는 'LIQUIDATE_TICKER' 신호
                sell_signals.append(s)

        if len(portfolio.positions) < self.max_stocks:
            new_entry_signals = self._generate_signals_for_new_entries(
                current_date, portfolio, data_handler
            )
            buy_signals.extend(new_entry_signals)

        buy_signals.sort(key=lambda s: (s["priority_group"], s["sort_metric"]))

        all_signals = buy_signals + sell_signals  # 매수 신호와 매도 신호를 합침(매수 신호가 먼저 오도록 함)
        for signal in all_signals:
            signal["start_date"] = self.backtest_start_date
            signal["end_date"] = self.backtest_end_date

        return all_signals

    def _generate_signals_for_existing_positions(
        self, current_date, portfolio, data_handler
    ):
        signals = []
        for ticker in list(portfolio.positions.keys()):
            stock_data = data_handler.load_stock_data(
                ticker, self.backtest_start_date, self.backtest_end_date
            )
            if (
                stock_data is None
                or stock_data.empty
                or current_date not in stock_data.index
            ):
                continue

            row = stock_data.loc[current_date]
            positions = portfolio.positions[ticker]
            
            for p in positions:
                if row["close_price"] >= p.buy_price * (1 + self.sell_profit_rate):
                    signals.append(
                        {
                            "date": current_date,
                            "ticker": ticker,
                            "type": "SELL",
                            "quantity": p.quantity,
                            "position": p,
                        }
                    )
                    # 매도 신호 발생 시 쿨다운 트래커에 기록
                    self.cooldown_tracker[ticker] = current_date

            if positions:
                last_pos = positions[-1]
                if row["close_price"] <= last_pos.buy_price * (1 - self.additional_buy_drop_rate):
                    if row["close_price"] > 0:
                        quantity = int(self.investment_per_order / row["close_price"])
                        if quantity > 0:
                            new_pos = Position(
                                row["close_price"],
                                quantity,
                                len(positions) + 1,
                                self.additional_buy_drop_rate,
                                self.sell_profit_rate,
                            )
                            sort_metric = (
                                len(positions)
                                if self.additional_buy_priority == "lowest_order"
                                else -((last_pos.buy_price - row["close_price"]) / last_pos.buy_price)
                            )
                            signals.append(
                                {
                                    "date": current_date,
                                    "ticker": ticker,
                                    "type": "BUY",
                                    "quantity": quantity,
                                    "position": new_pos,
                                    "priority_group": 2,
                                    "sort_metric": sort_metric,
                                }
                            )
        return signals

    def _generate_signals_for_new_entries(self, current_date, portfolio, data_handler):
        signals = []
        available_slots = self.max_stocks - len(portfolio.positions)
        if available_slots <= 0:
            return signals

        candidate_codes = data_handler.get_filtered_stock_codes(current_date)
        if not candidate_codes:
            return signals

        # 쿨다운 로직 적용하여 신규 후보 필터링
        active_candidates = []
        for code in candidate_codes:
            if code in portfolio.positions:
                continue
            
            if code in self.cooldown_tracker:
                exit_date = self.cooldown_tracker[code]
                # 영업일 기준으로 쿨다운 기간 체크
                days_since_exit = np.busday_count(exit_date.date(), current_date.date())
                if days_since_exit < self.cooldown_period_days:
                    continue  # 쿨다운 기간이므로 매수 후보에서 제외
            
            active_candidates.append(code)
        
        print(f"  [CPU_NEW_BUY_DEBUG] Total candidates: {len(candidate_codes)}, Active candidates after cooldown: {len(active_candidates)}")

        candidate_atrs = []
        for ticker in active_candidates:
            stock_data = data_handler.load_stock_data(
                ticker, self.backtest_start_date, self.backtest_end_date
            )
            if (
                stock_data is not None
                and not stock_data.empty
                and "atr_14_ratio" in stock_data.columns
                and current_date in stock_data.index
            ):
                latest_atr = stock_data.loc[current_date, "atr_14_ratio"]
                if pd.notna(latest_atr):
                    candidate_atrs.append(
                        {"ticker": ticker, "atr_14_ratio": latest_atr}
                    )

        sorted_candidates = sorted(
            candidate_atrs, key=lambda x: x["atr_14_ratio"], reverse=True
        )
        
        print(f"  [CPU_NEW_BUY_DEBUG] Candidates to check: {len(sorted_candidates)}")

        for candidate in sorted_candidates:
            if len(signals) >= available_slots:
                break

            ticker = candidate["ticker"]
            stock_data = data_handler.load_stock_data(
                ticker, self.backtest_start_date, self.backtest_end_date
            )
            row = stock_data.loc[current_date]
            if row["close_price"] > 0:
                quantity = int(self.investment_per_order / row["close_price"])
                if quantity > 0:
                    new_pos = Position(
                        row["close_price"],
                        quantity,
                        1,
                        self.additional_buy_drop_rate,
                        self.sell_profit_rate,
                    )
                    signals.append(
                        {
                            "date": current_date,
                            "ticker": ticker,
                            "type": "BUY",
                            "quantity": quantity,
                            "position": new_pos,
                            "priority_group": 1,
                            "sort_metric": -candidate["atr_14_ratio"],
                        }
                    )
        return signals