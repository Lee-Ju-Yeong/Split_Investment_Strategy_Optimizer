"""
execution.py

This module contains the functions for executing the orders for the Magic Split Strategy.
"""

import math
from .portfolio import Trade, Position

# --- 타입 힌트를 위한 임포트 ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .portfolio import Portfolio
    from .data_handler import DataHandler

class BasicExecutionHandler:
    def __init__(self,
                 buy_commission_rate=0.00015,
                 sell_commission_rate=0.00015,
                 sell_tax_rate=0.0018):
        self.buy_commission_rate = buy_commission_rate
        self.sell_commission_rate = sell_commission_rate
        self.sell_tax_rate = sell_tax_rate

    def _get_tick_size(self, price):
        if price < 2000: return 1
        elif price < 5000: return 5
        elif price < 20000: return 10
        elif price < 50000: return 50
        elif price < 200000: return 100
        elif price < 500000: return 500
        else: return 1000

    def _adjust_price_up(self, price):
        tick_size = self._get_tick_size(price)
        return math.ceil(price / tick_size) * tick_size

    def execute_order(self, order_event: dict, portfolio: 'Portfolio', data_handler: 'DataHandler'):
        ticker = order_event["ticker"]
        order_type = order_event["type"]
        current_date = order_event["date"]
        
        ohlc_data = data_handler.get_ohlc_data_on_date(current_date, ticker, order_event["start_date"], order_event["end_date"])

        if ohlc_data is None:
            return

        cash_before = portfolio.cash
        positions_before = portfolio.positions.get(ticker, [])
        quantity_before = sum(p.quantity for p in positions_before)

        if order_type == "BUY":
            self._execute_buy(order_event, portfolio, data_handler, ohlc_data, cash_before, quantity_before)
        elif order_type == "SELL":
            self._execute_sell(order_event, portfolio, data_handler, ohlc_data, cash_before, quantity_before)

    def _execute_buy(self, order_event, portfolio, data_handler, ohlc_data, cash_before, quantity_before):
        ticker = order_event["ticker"]
        quantity = order_event["quantity"]
        
        execution_price = self._adjust_price_up(ohlc_data['close_price'])
        cost = execution_price * quantity
        commission = math.floor(cost * self.buy_commission_rate)
        total_cost = cost + commission

        if portfolio.cash < total_cost:
            return

        portfolio.update_cash(-total_cost)
        
        position_to_add = order_event["position"]
        position_to_add.buy_price = execution_price 
        portfolio.add_position(ticker, position_to_add, order_event["date"])

        cash_after = portfolio.cash
        positions_after = portfolio.positions.get(ticker, [])
        quantity_after = sum(p.quantity for p in positions_after)
        total_invested_cost_after = sum(p.quantity * p.buy_price for p in positions_after)
        avg_buy_price_after = total_invested_cost_after / quantity_after if quantity_after > 0 else 0

        trade = Trade(
            date=order_event["date"],
            code=ticker,
            name=data_handler.get_name_from_ticker(ticker),
            trade_type='BUY',
            order=position_to_add.order,
            reason_for_trade=order_event.get("reason_for_trade", "N/A"),
            trigger_price=order_event.get("trigger_price"),
            open_price=ohlc_data['open_price'],
            high_price=ohlc_data['high_price'],
            low_price=ohlc_data['low_price'],
            close_price=ohlc_data['close_price'],
            quantity=quantity,
            buy_price=execution_price, # BUY-일때는 buy_price에 기록
            sell_price=None,           # BUY-일때는 sell_price는 공란
            commission=commission,
            tax=0,                     # BUY-일때는 tax는 0
            trade_value=total_cost,
            quantity_before=quantity_before,
            quantity_after=quantity_after,
            avg_buy_price_after=avg_buy_price_after,
            realized_pnl=0,
            cash_before=cash_before,
            cash_after=cash_after
        )
        portfolio.record_trade(trade)

    def _execute_sell(self, order_event, portfolio, data_handler, ohlc_data, cash_before, quantity_before):
        ticker = order_event["ticker"]
        position_to_sell = order_event["position"]

        # [수정] trigger_price 기본값 None (없으면 우리가 역산해서 만듦)
        trigger_price = order_event.get("trigger_price", None)

        # === 세후 목표가 역산 & 호가보정 (CPU 표준) ===
        buy = position_to_sell.buy_price
        pr  = position_to_sell.sell_profit_rate
        cost_factor = (1 - self.sell_commission_rate - self.sell_tax_rate)  # e.g. 1 - 0.00015 - 0.0018
        target_raw = buy * (1 + pr)
        target_after_cost = target_raw / cost_factor

        # [추가] 외부에서 trigger가 안 오면 우리가 계산한 값을 사용
        if trigger_price is None:
            effective_trigger = target_after_cost
        else:
            # 보수적으로 더 높은 쪽을 쓰고 싶으면 아래 주석 해제
            # effective_trigger = max(trigger_price, target_after_cost)
            effective_trigger = trigger_price

        execution_price = self._adjust_price_up(effective_trigger)

        # [수정] 체결 가능: 당일 high가 execution_price 이상
        if ohlc_data["high_price"] < execution_price:
            return

        # [수정] 먼저 계산 → 그 다음 로그 (선언 전 참조 방지)
        quantity = position_to_sell.quantity
        revenue = execution_price * quantity
        commission = math.floor(revenue * self.sell_commission_rate)
        tax = math.floor(revenue * self.sell_tax_rate)
        net_revenue = revenue - commission - tax

        # [추가] 디버그 로그 (GPU와 라인 매칭용)
        print(
            f"[CPU_SELL_DEBUG] {order_event['date']} {ticker} "
            f"buy={buy} raw_target={target_raw:.4f} cost_factor={cost_factor:.6f} "
            f"target_after_cost={target_after_cost:.4f} current_close={ohlc_data['close_price']}"
        )
        print(
            f"[CPU_SELL_PRICE] {order_event['date']} {ticker} "
            f"sell_price(adjusted_tick)={execution_price} "
            f"qty={quantity} revenue_gross={revenue} revenue_net={net_revenue} "
            f"open={ohlc_data['open_price']} high={ohlc_data['high_price']}"
        )

        # 정산
        portfolio.update_cash(net_revenue)
        portfolio.remove_position(ticker, position_to_sell, order_event["date"])

        cash_after = portfolio.cash
        positions_after = portfolio.positions.get(ticker, [])
        quantity_after = sum(p.quantity for p in positions_after)

        avg_buy_price_after = 0
        if quantity_after > 0:
            total_invested_cost_after = sum(p.quantity * p.buy_price for p in positions_after)
            avg_buy_price_after = total_invested_cost_after / quantity_after

        buy_cost = buy * quantity
        realized_pnl = net_revenue - buy_cost

        trade = Trade(
            date=order_event["date"],
            code=ticker,
            name=data_handler.get_name_from_ticker(ticker),
            trade_type="SELL",
            order=position_to_sell.order,
            reason_for_trade=order_event.get("reason_for_trade", "N/A"),
            trigger_price=effective_trigger,  # [수정] 실제 사용값 기록
            open_price=ohlc_data["open_price"],
            high_price=ohlc_data["high_price"],
            low_price=ohlc_data["low_price"],
            close_price=ohlc_data["close_price"],
            quantity=quantity,
            buy_price=buy,
            sell_price=execution_price,
            commission=commission,
            tax=tax,
            trade_value=net_revenue,
            quantity_before=quantity_before,
            quantity_after=quantity_after,
            avg_buy_price_after=avg_buy_price_after,
            realized_pnl=realized_pnl,
            cash_before=cash_before,
            cash_after=cash_after,
        )
        portfolio.record_trade(trade)

        # 1차 매수 익절 시 전량 청산 규칙 유지
        if position_to_sell.order == 1:
            portfolio.liquidate_ticker(ticker)