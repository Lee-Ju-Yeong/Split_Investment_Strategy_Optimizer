import unittest
from unittest.mock import MagicMock
from datetime import date
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from portfolio import Portfolio, Position, Trade

class TestPortfolio(unittest.TestCase):

    def setUp(self):
        """테스트에 사용할 Portfolio 인스턴스 초기화"""
        self.initial_cash = 100000.0
        self.start_date = '2022-01-01'
        self.end_date = '2022-12-31'
        self.portfolio = Portfolio(self.initial_cash, self.start_date, self.end_date)
        
        # DataHandler 모의 객체 생성
        self.mock_data_handler = MagicMock()

    def test_initial_state(self):
        """포트폴리오 초기 상태를 검증"""
        self.assertEqual(self.portfolio.cash, self.initial_cash)
        self.assertEqual(self.portfolio.positions, {})
        self.assertEqual(self.portfolio.trade_history, [])
        self.assertEqual(self.portfolio.daily_value_history, [])

    def test_update_cash(self):
        """현금 업데이트 기능 검증"""
        self.portfolio.update_cash(-5000)
        self.assertEqual(self.portfolio.cash, self.initial_cash - 5000)
        self.portfolio.update_cash(1000)
        self.assertEqual(self.portfolio.cash, self.initial_cash - 4000)

    def test_add_and_remove_position(self):
        """포지션 추가 및 제거 기능 검증"""
        ticker = '005930'
        position1 = Position(buy_price=70000, quantity=10, order=1, additional_buy_drop_rate=0.05, sell_profit_rate=0.1)
        position2 = Position(buy_price=65000, quantity=5, order=2, additional_buy_drop_rate=0.05, sell_profit_rate=0.1)

        # 포지션 추가
        self.portfolio.add_position(ticker, position1)
        self.assertIn(ticker, self.portfolio.positions)
        self.assertEqual(len(self.portfolio.positions[ticker]), 1)
        self.assertEqual(self.portfolio.positions[ticker][0].quantity, 10)

        # 같은 종목에 다른 차수 포지션 추가 (정렬 확인)
        self.portfolio.add_position(ticker, position2)
        self.assertEqual(len(self.portfolio.positions[ticker]), 2)
        self.assertEqual(self.portfolio.positions[ticker][0].order, 1) # 정렬되어 order 1이 먼저 와야 함
        
        # 포지션 제거
        self.portfolio.remove_position(ticker, position1)
        self.assertEqual(len(self.portfolio.positions[ticker]), 1)
        self.assertEqual(self.portfolio.positions[ticker][0], position2)

        # 마지막 포지션 제거
        self.portfolio.remove_position(ticker, position2)
        self.assertNotIn(ticker, self.portfolio.positions)

    def test_get_total_value(self):
        """포트폴리오 총 가치 계산 기능 검증"""
        # 초기 상태: 현금만 보유
        total_value = self.portfolio.get_total_value(date(2022, 1, 10), self.mock_data_handler)
        self.assertEqual(total_value, self.initial_cash)
        
        # 포지션 추가 후
        ticker1 = '005930'
        ticker2 = '000660'
        pos1 = Position(buy_price=70000, quantity=10, order=1, additional_buy_drop_rate=0.05, sell_profit_rate=0.1)
        pos2 = Position(buy_price=100000, quantity=5, order=1, additional_buy_drop_rate=0.05, sell_profit_rate=0.1)
        self.portfolio.add_position(ticker1, pos1)
        self.portfolio.add_position(ticker2, pos2)
        self.portfolio.update_cash(-(70000 * 10 + 100000 * 5)) # 매수 비용만큼 현금 차감

        # get_latest_price가 반환할 현재가 설정
        self.mock_data_handler.get_latest_price.side_effect = lambda dt, tk, sd, ed: 75000 if tk == ticker1 else 110000
        
        current_date = date(2022, 6, 15)
        total_value = self.portfolio.get_total_value(current_date, self.mock_data_handler)
        
        expected_market_value = (10 * 75000) + (5 * 110000) # 750000 + 550000 = 1300000
        expected_cash = self.initial_cash - (70000 * 10 + 100000 * 5) # 100000 - 1200000 = -1100000
        expected_total_value = expected_cash + expected_market_value
        
        self.assertEqual(total_value, expected_total_value)
        
        # get_latest_price가 None을 반환하는 경우
        self.mock_data_handler.get_latest_price.side_effect = lambda dt, tk, sd, ed: 75000 if tk == ticker1 else None
        total_value_with_none = self.portfolio.get_total_value(current_date, self.mock_data_handler)
        expected_market_value_with_none = 10 * 75000 # ticker2는 가격이 없어 제외
        self.assertEqual(total_value_with_none, expected_cash + expected_market_value_with_none)


    def test_record_trade_and_daily_value(self):
        """거래 및 일별 가치 기록 기능 검증"""
        # 거래 기록
        trade = Trade(date=date(2022, 3, 5), code='005930', order=1, quantity=10, buy_price=70000, sell_price=None, trade_type='BUY', profit=0, profit_rate=0, normalized_value=0.5, capital=100000, total_portfolio_value=100000)
        self.portfolio.record_trade(trade)
        self.assertEqual(len(self.portfolio.trade_history), 1)
        self.assertEqual(self.portfolio.trade_history[0].code, '005930')

        # 일별 가치 기록
        self.portfolio.record_daily_value(date(2022, 3, 5), 100000)
        self.portfolio.record_daily_value(date(2022, 3, 6), 101000)
        self.assertEqual(len(self.portfolio.daily_value_history), 2)
        self.assertEqual(self.portfolio.daily_value_history[1]['value'], 101000)

if __name__ == '__main__':
    unittest.main()
