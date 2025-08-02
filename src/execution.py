import math
from abc import ABC, abstractmethod
from .portfolio import Trade

class ExecutionHandler(ABC):
    @abstractmethod
    def execute_order(self, order_event, portfolio, data_handler):
        raise NotImplementedError("execute_order() 메소드를 구현해야 합니다.")

class BasicExecutionHandler(ExecutionHandler):
    def __init__(self, buy_commission_rate=0.00015, sell_commission_rate=0.00015, sell_tax_rate=0.0018):
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
        ticker = order_event['ticker']
        order_type = order_event['type']
        quantity = order_event['quantity']
        current_date = order_event['date']
        start_date = order_event['start_date']
        end_date = order_event['end_date']
        
        current_price = data_handler.get_latest_price(current_date, ticker, start_date, end_date)
        
        if current_price is None:
            return

        if order_type == 'BUY':
            # 종가를 기준으로 호가단위를 적용하여 보수적인 매수가격 결정 (종가보다 높은 가장 가까운 호가)
            buy_price = self._adjust_price_up(current_price)
            
            cost = buy_price * quantity
            total_cost = cost * (1 + self.buy_commission_rate)
            
            if portfolio.cash >= total_cost:
                portfolio.update_cash(-total_cost)
                # 실제 체결된 가격으로 Position의 buy_price를 업데이트
                order_event['position'].buy_price = buy_price
                portfolio.add_position(ticker, order_event['position']) 
                
                trade = Trade(current_date, ticker, order_event['position'].order, quantity, buy_price, None, 'buy', 0, 0, None, portfolio.cash, portfolio.get_total_value(current_date, data_handler))
                portfolio.record_trade(trade)

        elif order_type == 'SELL':
            position_to_sell = order_event['position']
            
            # 거래 비용을 모두 제하고도 목표 수익률을 달성하기 위한 최소 목표 매도가를 역산
            cost_factor = 1 - self.sell_commission_rate - self.sell_tax_rate
            target_sell_price = (position_to_sell.buy_price * (1 + position_to_sell.sell_profit_rate)) / cost_factor
            
            # 호가단위를 적용하여 실제 체결 가능한 매도가 결정 (역산된 목표가보다 높거나 같은 가장 가까운 호가)
            sell_price = self._adjust_price_up(target_sell_price)

            revenue = sell_price * quantity
            total_revenue = revenue * cost_factor # 미리 계산해둔 비용 팩터 사용
            
            portfolio.update_cash(total_revenue)
            portfolio.remove_position(ticker, position_to_sell)
            
            buy_cost = position_to_sell.buy_price * quantity
            profit = total_revenue - buy_cost
            profit_rate = profit / buy_cost if buy_cost != 0 else 0
            
            trade = Trade(current_date, ticker, position_to_sell.order, quantity, position_to_sell.buy_price, sell_price, 'sell', profit, profit_rate, None, portfolio.cash, portfolio.get_total_value(current_date, data_handler))
            portfolio.record_trade(trade)
