"""
execution.py

This module contains the functions for executing the orders for the Magic Split Strategy.
"""

import math
from abc import ABC, abstractmethod
from .portfolio import Trade


class ExecutionHandler(ABC):
    @abstractmethod
    def execute_order(self, order_event, portfolio, data_handler):
        raise NotImplementedError("execute_order() 메소드를 구현해야 합니다.")


class BasicExecutionHandler(ExecutionHandler):
    def __init__(
        self,
        buy_commission_rate=0.00015,
        sell_commission_rate=0.00015,
        sell_tax_rate=0.0018,
    ):
        self.buy_commission_rate = buy_commission_rate
        self.sell_commission_rate = sell_commission_rate
        self.sell_tax_rate = sell_tax_rate

    def _get_tick_size(self, price):
        """주가에 따른 호가 단위를 반환합니다."""
        if price < 2000:
            return 1
        elif price < 5000:
            return 5
        elif price < 20000:
            return 10
        elif price < 50000:
            return 50
        elif price < 200000:
            return 100
        elif price < 500000:
            return 500
        else:
            return 1000

    def _adjust_price_up(self, price):
        """주어진 가격을 호가 단위에 맞춰 올림 처리합니다."""
        tick_size = self._get_tick_size(price)
        return math.ceil(price / tick_size) * tick_size

    def execute_order(self, order_event, portfolio, data_handler):
        ticker = order_event["ticker"]
        order_type = order_event["type"]
        # --- 💡 수정: LIQUIDATE_TICKER 신호는 quantity, price 등이 없으므로 먼저 처리 ---
        if order_type == "LIQUIDATE_TICKER":
            portfolio.liquidate_ticker(ticker)  # 포트폴리오에서 해당 종목 관리 중단
            # print(f"{order_event['date'].strftime('%Y-%m-%d')}: 종목 {ticker} 포트폴리오에서 청산됨.") # 로그 필요 시 주석 해제
            return  # 이 신호는 매매가 아니므로 여기서 함수 종료

        quantity = order_event["quantity"]
        current_date = order_event["date"]
        start_date = order_event["start_date"]
        end_date = order_event["end_date"]

        current_price = data_handler.get_latest_price(
            current_date, ticker, start_date, end_date
        )

        if current_price is None or math.isnan(current_price):
            return

        if order_type == "BUY":
            # 종가를 기준으로 호가단위를 적용하여 보수적인 매수가격 결정 (종가보다 높은 가장 가까운 호가)
            buy_price = self._adjust_price_up(current_price)

            cost = buy_price * quantity
            total_cost = cost * (1 + self.buy_commission_rate)

            if portfolio.cash >= total_cost:
                portfolio.update_cash(-total_cost)
                # 실제 체결된 가격으로 Position의 buy_price를 업데이트
                order_event["position"].buy_price = buy_price
                portfolio.add_position(ticker, order_event["position"])

                trade = Trade(
                    current_date,
                    ticker,
                    order_event["position"].order,
                    quantity,
                    buy_price,
                    None,
                    "buy",
                    0,
                    0,
                    None,
                    portfolio.cash,
                    portfolio.get_total_value(current_date, data_handler),
                )
                portfolio.record_trade(trade)

        elif order_type == "SELL":
            position_to_sell = order_event["position"]

            # 거래 비용을 모두 제하고도 목표 수익률을 달성하기 위한 최소 목표 매도가를 역산
            cost_factor = 1 - self.sell_commission_rate - self.sell_tax_rate
            target_sell_price = (
                position_to_sell.buy_price * (1 + position_to_sell.sell_profit_rate)
            ) / cost_factor

            # 호가단위를 적용하여 실제 체결 가능한 매도가 결정 (역산된 목표가보다 높거나 같은 가장 가까운 호가)
            sell_price = self._adjust_price_up(target_sell_price)

            # 2. ★★★ 실제 체결 조건 확인 (신규 추가) ★★★
            # 당일 시장 가격(current_price, 종가 기준)이 내가 팔려던 가격(sell_price)에
            # 도달했거나 넘어섰을 때만 매도를 실행한다.
            # 더 보수적이거나 현실적으로 하려면 당일 고가(high_price)와 비교해야 하지만,
            # 현재 데이터 핸들러 구조에서는 종가(current_price)를 사용합니다.
            if current_price >= sell_price:
                # --- 아래는 기존 매도 로직을 if 블록 안으로 이동 ---
                revenue = sell_price * quantity
                total_revenue = revenue * cost_factor

                portfolio.update_cash(total_revenue)
                portfolio.remove_position(ticker, position_to_sell)

                buy_cost = position_to_sell.buy_price * quantity
                profit = total_revenue - buy_cost
                profit_rate = profit / buy_cost if buy_cost != 0 else 0

                trade = Trade(
                    current_date,
                    ticker,
                    position_to_sell.order,
                    quantity,
                    position_to_sell.buy_price,
                    sell_price,
                    "sell",
                    profit,
                    profit_rate,
                    None,
                    portfolio.cash,
                    portfolio.get_total_value(current_date, data_handler),
                )
                portfolio.record_trade(trade)
            else:
                # 조건 미충족 시 아무것도 하지 않고 넘어감 (매도 실패)
                pass
