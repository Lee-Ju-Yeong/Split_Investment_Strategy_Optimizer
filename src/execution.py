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

    def execute_order(self, order_event, portfolio, data_handler):
        ticker = order_event['ticker']
        order_type = order_event['type']
        quantity = order_event['quantity']
        current_date = order_event['date']
        start_date = order_event['start_date']
        end_date = order_event['end_date']
        
        current_price = data_handler.get_latest_price(current_date, ticker, start_date, end_date)
        
        if current_price is None:
            # print(f"Warning: {current_date}에 {ticker}의 가격 정보가 없어 주문을 실행할 수 없습니다.")
            return

        if order_type == 'BUY':
            cost = current_price * quantity
            total_cost = cost * (1 + self.buy_commission_rate)
            
            if portfolio.cash >= total_cost:
                portfolio.update_cash(-total_cost)
                portfolio.add_position(ticker, order_event['position']) 
                
                trade = Trade(current_date, ticker, order_event['position'].order, quantity, current_price, None, 'buy', 0, 0, None, portfolio.cash, portfolio.get_total_value(current_date, data_handler))
                portfolio.record_trade(trade)
            # else:
            #     print(f"Warning: 현금이 부족하여 {ticker} 매수 주문을 실행할 수 없습니다.")

        elif order_type == 'SELL':
            position_to_sell = order_event['position']
            
            revenue = current_price * quantity
            total_revenue = revenue * (1 - self.sell_commission_rate - self.sell_tax_rate)
            
            portfolio.update_cash(total_revenue)
            portfolio.remove_position(ticker, position_to_sell)
            
            buy_cost = position_to_sell.buy_price * quantity
            profit = (current_price * quantity * (1 - self.sell_commission_rate - self.sell_tax_rate)) - buy_cost
            profit_rate = profit / buy_cost if buy_cost != 0 else 0
            
            trade = Trade(current_date, ticker, position_to_sell.order, quantity, position_to_sell.buy_price, current_price, 'sell', profit, profit_rate, None, portfolio.cash, portfolio.get_total_value(current_date, data_handler))
            portfolio.record_trade(trade)
