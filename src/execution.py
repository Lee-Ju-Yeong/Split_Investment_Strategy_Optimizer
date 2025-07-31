from abc import ABC, abstractmethod
from .portfolio import Trade

class ExecutionHandler(ABC):
    """
    모든 실행 핸들러가 상속해야 하는 추상 기본 클래스입니다.
    주문(Order) 이벤트를 받아 처리하고, 실제 거래(Fill) 이벤트를 생성합니다.
    """
    @abstractmethod
    def execute_order(self, order_event, portfolio, data_handler):
        raise NotImplementedError("execute_order() 메소드를 구현해야 합니다.")

class BasicExecutionHandler(ExecutionHandler):
    """
    수수료와 세금을 계산하여 주문을 처리하는 기본적인 실행 핸들러입니다.
    """
    def __init__(self, buy_commission_rate=0.00015, sell_commission_rate=0.00015, sell_tax_rate=0.0018):
        self.buy_commission_rate = buy_commission_rate
        self.sell_commission_rate = sell_commission_rate
        self.sell_tax_rate = sell_tax_rate

    def execute_order(self, order_event, portfolio, data_handler):
        """
        주문 이벤트를 받아 실제 거래를 시뮬레이션하고 포트폴리오 상태를 업데이트합니다.
        
        Args:
            order_event: (ticker, order_type, quantity) 튜플. 예: ('005930', 'BUY', 10)
            portfolio: 현재 포트폴리오 객체
            data_handler: 데이터 핸들러 객체
        """
        ticker, order_type, quantity = order_event['ticker'], order_event['type'], order_event['quantity']
        current_date = order_event['date']
        
        current_price = data_handler.get_latest_price(current_date, ticker)
        
        if current_price is None:
            print(f"Warning: {current_date}에 {ticker}의 가격 정보가 없어 주문을 실행할 수 없습니다.")
            return

        if order_type == 'BUY':
            cost = current_price * quantity
            total_cost = cost * (1 + self.buy_commission_rate)
            
            if portfolio.cash >= total_cost:
                portfolio.update_cash(-total_cost)
                # Position 객체는 Strategy에서 생성되어 order_event에 포함되어야 함
                portfolio.add_position(ticker, order_event['position']) 
                
                # 거래 기록
                trade = Trade(current_date, ticker, order_event['position'].order, quantity, current_price, None, 'buy', 0, 0, None, portfolio.cash, portfolio.get_total_value(current_date, data_handler))
                portfolio.record_trade(trade)
                
            else:
                print(f"Warning: 현금이 부족하여 {ticker} 매수 주문을 실행할 수 없습니다.")

        elif order_type == 'SELL':
            # 매도할 포지션 정보를 order_event에서 가져와야 함
            position_to_sell = order_event['position']
            
            revenue = current_price * quantity
            total_revenue = revenue * (1 - self.sell_commission_rate - self.sell_tax_rate)
            
            portfolio.update_cash(total_revenue)
            portfolio.remove_position(ticker, position_to_sell)
            
            # 실현 손익 계산
            buy_cost = position_to_sell.buy_price * quantity * (1 + self.buy_commission_rate)
            profit = total_revenue - buy_cost
            profit_rate = profit / buy_cost if buy_cost != 0 else 0
            
            # 거래 기록
            trade = Trade(current_date, ticker, position_to_sell.order, quantity, position_to_sell.buy_price, current_price, 'sell', profit, profit_rate, None, portfolio.cash, portfolio.get_total_value(current_date, data_handler))
            portfolio.record_trade(trade)
