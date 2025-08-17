# portfolio.py (수정 완료)

import pandas as pd
import numpy as np
import uuid # 고유 ID 생성을 위해 추가

import uuid # 고유 ID 생성을 위해 추가

# DataHandler 타입 힌트를 위해 추가 (실제 임포트는 아님)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .data_handler import DataHandler

class Position:
    # --- [수정] entry_date 제거 ---
    def __init__(self, buy_price, quantity, order, additional_buy_drop_rate, sell_profit_rate):
        self.position_id = str(uuid.uuid4())
        self.buy_price = buy_price
        self.quantity = quantity
        self.order = order
        self.additional_buy_drop_rate = additional_buy_drop_rate
        self.sell_profit_rate = sell_profit_rate

class Trade:
    def __init__(self, date, code, name, trade_type, order, reason_for_trade,
                 trigger_price, open_price, high_price, low_price, close_price,
                 quantity, buy_price, sell_price, commission, tax, trade_value,
                 quantity_before, quantity_after, avg_buy_price_after,
                 realized_pnl,
                 cash_before, cash_after, total_portfolio_value=None):
        self.date = date
        self.code = code
        self.name = name
        self.trade_type = trade_type
        self.order = order
        self.reason_for_trade = reason_for_trade
        self.trigger_price = trigger_price
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.quantity = quantity
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.commission = commission
        self.tax = tax
        self.trade_value = trade_value
        self.quantity_before = quantity_before
        self.quantity_after = quantity_after
        self.avg_buy_price_after = avg_buy_price_after
        self.realized_pnl = realized_pnl
        self.cash_before = cash_before
        self.cash_after = cash_after
        self.total_portfolio_value = total_portfolio_value

class Portfolio:
    def __init__(self, initial_cash, start_date, end_date):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.start_date = start_date
        self.end_date = end_date
        self.positions = {}
        # --- [추가] 마지막 거래일 추적용 딕셔너리 ---
        self.last_trade_dates = {}
        self.trade_history = []
        self.daily_snapshot_history = []

    def update_cash(self, amount):
        self.cash += amount

    # --- [수정] last_trade_date 갱신 로직 추가 ---
    def add_position(self, ticker, position: Position, trade_date):
        if ticker not in self.positions:
            self.positions[ticker] = []
        self.positions[ticker].append(position)
        self.positions[ticker].sort(key=lambda p: p.order)
        self.last_trade_dates[ticker] = trade_date # 매수 시 마지막 거래일 갱신

    # --- [수정] last_trade_date 갱신/삭제 로직 추가 ---
    def remove_position(self, ticker, position_to_remove: Position, trade_date):
        if ticker in self.positions:
            initial_len = len(self.positions[ticker])
            self.positions[ticker] = [
                p for p in self.positions[ticker] 
                if p.position_id != position_to_remove.position_id
            ]
            
            if len(self.positions[ticker]) == initial_len:
                 print(f"Warning: 제거하려는 포지션(ID: {position_to_remove.position_id})을 {ticker} 종목에서 찾을 수 없습니다.")

            if not self.positions[ticker]:
                # 모든 포지션이 매도된 경우 (완전 청산)
                del self.positions[ticker]
                if ticker in self.last_trade_dates:
                    del self.last_trade_dates[ticker] # 마지막 거래일 기록도 삭제
            else:
                # 일부 포지션만 매도된 경우 (수익 실현)
                self.last_trade_dates[ticker] = trade_date # 마지막 거래일 갱신

    def get_total_value(self, current_date, data_handler: 'DataHandler'):
        total_market_value = np.float32(0.0)
        for ticker, positions_list in self.positions.items():
            current_price = data_handler.get_latest_price(current_date, ticker, self.start_date, self.end_date)
            if current_price is not None and not np.isnan(current_price):
                current_price_f32 = np.float32(current_price)
                for position in positions_list:
                    total_market_value += np.float32(position.quantity) * current_price_f32
        
        return np.float32(self.cash) + total_market_value

    def record_trade(self, trade: Trade):
        self.trade_history.append(trade)

    ### ### [핵심] 일별 스냅샷 기록 기능 ### ###
    def record_daily_snapshot(self, date, total_value):
        """일별 포트폴리오의 주요 정보를 기록합니다."""
        snapshot = {
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'stock_count': len(self.positions)
        }
        self.daily_snapshot_history.append(snapshot)
        
    def liquidate_ticker(self, ticker):
        if ticker in self.positions:
            del self.positions[ticker]

    ### ### [핵심] 일일 스냅샷 상세 데이터 계산 메소드 ### ###
    def get_positions_snapshot(self, current_date, data_handler: 'DataHandler', total_portfolio_value):
        """현재 보유 중인 모든 종목의 상세 정보를 DataFrame으로 반환합니다."""
        if not self.positions:
            return pd.DataFrame()

        snapshot_data = []
        for ticker, positions in self.positions.items():
            current_price = data_handler.get_latest_price(current_date, ticker, self.start_date, self.end_date)
            # 종가를 못 가져오는 경우(거래정지 등)는 스냅샷에서 제외하지 않고, 가격을 0으로 처리하여 표시
            if current_price is None or np.isnan(current_price):
                current_price = 0

            total_quantity = sum(p.quantity for p in positions)
            total_invested_cost = sum(p.quantity * p.buy_price for p in positions)
            avg_buy_price = total_invested_cost / total_quantity if total_quantity > 0 else 0
            
            current_total_value = total_quantity * current_price
            unrealized_pnl = current_total_value - total_invested_cost
            pnl_rate = unrealized_pnl / total_invested_cost if total_invested_cost > 0 else 0
            weight = current_total_value / total_portfolio_value if total_portfolio_value > 0 else 0
            
            snapshot_data.append({
                'Ticker': ticker,
                'Name': data_handler.get_name_from_ticker(ticker) or 'N/A',
                'Holdings': len(positions),
                'Avg Buy Price': avg_buy_price,
                'Current Price': current_price,
                'Unrealized P/L': unrealized_pnl,
                'P/L Rate': pnl_rate,
                'Total Value': current_total_value,
                'Weight': weight
            })
            
        if not snapshot_data:
            return pd.DataFrame()

        df = pd.DataFrame(snapshot_data)
        return df.sort_values(by='Weight', ascending=False).reset_index(drop=True)