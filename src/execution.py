"""
execution.py

This module contains the functions for executing the orders for the Magic Split Strategy.
"""

import math
import numpy as np
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

    def execute_order(self, order_event: dict, portfolio: 'Portfolio', data_handler: 'DataHandler', current_day_idx):
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
            self._execute_buy(order_event, portfolio, data_handler, ohlc_data, cash_before, quantity_before, current_day_idx)
        elif order_type == "SELL":
            self._execute_sell(order_event, portfolio, data_handler, ohlc_data, cash_before, quantity_before, current_day_idx)

    def _execute_buy(self, order_event, portfolio, data_handler, ohlc_data, cash_before, quantity_before, current_day_idx):
        ticker = order_event["ticker"]
        investment_amount = np.float32(order_event["investment_amount"])
        
        reason = order_event.get("reason_for_trade", "")
        if "추가 매수" in reason:
            # [수정] GPU와 로직 동기화: epsilon을 이용한 절충안 적용
            trigger_price = np.float32(order_event["trigger_price"])
            high_price = np.float32(ohlc_data['high_price'])
            epsilon = np.float32(1.0) # 최소 가격 단위(1원)를 안전 마진으로 사용
            
            # 조건: 고가가 (목표가 - 1원)보다 작거나 같으면 (확실한 갭하락), 기준가는 고가.
            # 그렇지 않으면 (장중 터치 또는 근접), 기준가는 목표가.
            if high_price <= trigger_price - epsilon:
                price_basis = high_price
            else:
                price_basis = trigger_price
        else: # 신규 진입
            price_basis = np.float32(ohlc_data['close_price'])
            
        execution_price = self._adjust_price_up(price_basis)
        # [추가] 최종 체결가를 기준으로 수량 계산 (핵심 변경)
        if execution_price <= 0: return
        quantity = np.int32(math.floor(round(investment_amount / execution_price, 5)))
        if quantity <= 0: return
        
        cost = np.float32(execution_price) * np.float32(quantity)
        commission = np.floor(round(cost * np.float32(self.buy_commission_rate), 5))
        total_cost = cost + commission
        print(
        f"[CPU_BUY_CALC] {order_event['date'].strftime('%Y-%m-%d')} {ticker} | "
        f"Invest: {investment_amount:,.0f} / ExecPrice: {execution_price:,.0f} = Qty: {quantity}"
        )
        if portfolio.cash < total_cost:
            return

        portfolio.update_cash(-total_cost)
        position_to_add = order_event["position"]
        position_to_add.buy_price = execution_price 
        position_to_add.open_date = order_event["date"]
        position_to_add.quantity = quantity
        portfolio.add_position(ticker, position_to_add, order_event["date"], current_day_idx)
        # 매수 거래 성공 시, 마지막 거래일 인덱스를 업데이트합니다.
        portfolio.last_trade_dates[ticker] = order_event["date"]
        portfolio.last_trade_day_indices[ticker] = current_day_idx
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

    def _execute_sell(self, order_event, portfolio, data_handler, ohlc_data, cash_before, quantity_before, current_day_idx):
        ticker = order_event["ticker"]
        position_to_sell = order_event["position"]

        # GPU와 로직 동기화: 세금/수수료 역산 로직을 제거하고, 순수 목표가로 체결가를 계산합니다.
        buy_price = np.float32(position_to_sell.buy_price)
        sell_profit_rate = np.float32(position_to_sell.sell_profit_rate)
        
        # trigger_price는 손절, 기간만료 등 외부에서 명시적으로 지정될 때만 사용합니다.
        # 수익 실현의 경우, 항상 buy_price를 기준으로 계산합니다.
        is_profit_taking = "수익 실현" in order_event.get("reason_for_trade", "")
        is_stop_loss = "손절매" in order_event.get("reason_for_trade", "")
        
        price_basis = np.float32(0.0)
        target_price = np.float32(0.0) # 로그 기록용
        if is_profit_taking:
            # 수익 실현은 기존 로직 유지 (Target 가격에 도달해야만 체결)
            target_price = buy_price * (np.float32(1.0) + sell_profit_rate)
            execution_price = self._adjust_price_up(target_price)
            if ohlc_data["high_price"] < execution_price:
                return # 체결 실패
            price_basis = target_price # 체결 성공 시 기준가는 target_price
        else: # 손절 또는 기간만료 청산
            target_price = np.float32(order_event.get("trigger_price"))
            
            if is_stop_loss:
                # 시나리오 A: 장중 손절가 도달
                if ohlc_data['high_price'] >= target_price:
                    price_basis = target_price
                # 시나리오 B: 갭하락으로 미도달
                else:
                    price_basis = np.float32(ohlc_data['close_price'])
            else: # 기간만료 청산
                price_basis = np.float32(ohlc_data['close_price'])

        execution_price = self._adjust_price_up(price_basis)

        # 계산 로직은 변경 없음
        quantity = position_to_sell.quantity
        revenue = np.float32(execution_price) * np.float32(quantity)
        commission = np.floor(round(revenue * np.float32(self.sell_commission_rate), 5))
        tax = np.floor(round(revenue * np.float32(self.sell_tax_rate), 5))
        net_revenue = revenue - commission - tax
        print(
        f"[CPU_SELL_CALC] {order_event['date'].strftime('%Y-%m-%d')} {ticker} | "
        f"Qty: {quantity} * ExecPrice: {execution_price:,.0f} = Revenue: {revenue:,.0f}"
    )
        print(
            f"[CPU_SELL_PRICE] {order_event['date'].strftime('%Y-%m-%d')} {ticker} "
            f"Reason: {order_event.get('reason_for_trade', 'N/A')} | "
            f"Target: {target_price:.2f} -> Exec: {execution_price} | "
            f"High: {ohlc_data['high_price']}"
        )
        
        # 정산
        portfolio.update_cash(net_revenue)
        portfolio.remove_position(ticker, position_to_sell, order_event["date"], current_day_idx)
        # 매도 거래 성공 시, 마지막 거래일 인덱스를 업데이트합니다.
        portfolio.last_trade_dates[ticker] = order_event["date"] # 부분 매도든 완전 청산이든 마지막 거래일은 오늘임
        portfolio.last_trade_day_indices[ticker] = current_day_idx
        cash_after = portfolio.cash
        positions_after = portfolio.positions.get(ticker, [])
        quantity_after = sum(p.quantity for p in positions_after)

        avg_buy_price_after = 0
        if quantity_after > 0:
            total_invested_cost_after = sum(p.quantity * p.buy_price for p in positions_after)
            avg_buy_price_after = total_invested_cost_after / quantity_after

        buy_cost = buy_price * quantity
        realized_pnl = net_revenue - buy_cost

        trade = Trade(
            date=order_event["date"],
            code=ticker,
            name=data_handler.get_name_from_ticker(ticker),
            trade_type="SELL",
            order=position_to_sell.order,
            reason_for_trade=order_event.get("reason_for_trade", "N/A"),
            trigger_price=target_price,  # [수정] 실제 사용값 기록
            open_price=ohlc_data["open_price"],
            high_price=ohlc_data["high_price"],
            low_price=ohlc_data["low_price"],
            close_price=ohlc_data["close_price"],
            quantity=quantity,
            buy_price=buy_price,
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