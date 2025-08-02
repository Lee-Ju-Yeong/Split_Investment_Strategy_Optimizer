import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import configparser
import math

# 경로 설정
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from db_setup import get_db_connection, create_tables
from data_handler import DataHandler
from portfolio import Portfolio, Position, Trade
from strategy import Strategy
from execution import BasicExecutionHandler
from backtester import BacktestEngine

# --- 테스트를 위한 수정된 클래스들 ---
class TestExecutionHandler(BasicExecutionHandler):
    def execute_order(self, order_event, portfolio, data_handler):
        if order_event['type'] == 'BUY':
            buy_price = self._adjust_price_up(order_event['price'])
            cost = buy_price * order_event['quantity']
            total_cost = cost * (1 + self.buy_commission_rate)
            if portfolio.cash >= total_cost:
                portfolio.update_cash(-total_cost)
                order_event['position'].buy_price = buy_price
                portfolio.add_position(order_event['ticker'], order_event['position'])
                # 거래 기록 추가
                trade = Trade(order_event['date'], order_event['ticker'], 1, order_event['quantity'], buy_price, None, 'buy', 0, 0, None, portfolio.cash, 0)
                portfolio.record_trade(trade)

class SimpleBuyStrategy(Strategy):
    def __init__(self, start_date, end_date):
        self.start_date, self.end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        self.invested = False
    def generate_signals(self, current_date, portfolio, data_handler):
        if self.invested: return []
        for code in data_handler.get_filtered_stock_codes(current_date):
            stock_data = data_handler.load_stock_data(code, self.start_date, self.end_date)
            if stock_data is None or stock_data.empty or current_date not in stock_data.index: continue
            latest = stock_data.loc[current_date]
            if pd.notna(latest['ma_5']) and pd.notna(latest['ma_20']) and latest['ma_5'] > latest['ma_20']:
                self.invested = True
                return [{'date': current_date, 'ticker': code, 'type': 'BUY', 'quantity': 10, 
                         'price': latest['close_price'], 'position': Position(0, 10, 1, 0.05, 0.1), 
                         'start_date': self.start_date, 'end_date': self.end_date}]
        return []

class TestBacktestingIntegration(unittest.TestCase):
    TEST_TICKER = '999998'
    START_DATE, END_DATE = '2022-01-01', '2022-03-31'

    @classmethod
    def setUpClass(cls):
        cls.conn = get_db_connection()
        create_tables(cls.conn)
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), '../config.ini'))
        cls.db_config = dict(config['mysql'])

    @classmethod
    def tearDownClass(cls): cls.conn.close()
    def setUp(self):
        self._cleanup_test_data()
        self._prepare_test_data()
        self.patcher = patch.object(DataHandler, 'get_filtered_stock_codes', return_value=[self.TEST_TICKER])
        self.patcher.start()
    def tearDown(self):
        self.patcher.stop()
        self._cleanup_test_data()

    def _cleanup_test_data(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM WeeklyFilteredStocks WHERE stock_code = %s", (self.TEST_TICKER,))
            cur.execute("DELETE FROM DailyStockPrice WHERE stock_code = %s", (self.TEST_TICKER,))
            cur.execute("DELETE FROM CalculatedIndicators WHERE stock_code = %s", (self.TEST_TICKER,))
        self.conn.commit()

    def _prepare_test_data(self):
        with self.conn.cursor() as cur:
            cur.execute("INSERT IGNORE INTO WeeklyFilteredStocks (filter_date, stock_code, company_name) VALUES (%s,%s,%s)",
                        (self.START_DATE, self.TEST_TICKER, '백테스트용주식'))
        
        dates = pd.to_datetime(pd.date_range(self.START_DATE, self.END_DATE, freq='B'))
        close_prices = np.concatenate([np.linspace(12000, 10000, 30), np.linspace(10001, 15000, len(dates)-30)])
        df = pd.DataFrame({'date': dates, 'close_price': close_prices})
        df.set_index('date', inplace=True)
        df['ma_5'] = df['close_price'].rolling(5).mean()
        df['ma_20'] = df['close_price'].rolling(20).mean()
        df.reset_index(inplace=True); df.dropna(inplace=True); df['stock_code'] = self.TEST_TICKER
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        
        df_daily = df.copy()
        df_daily['open_price'], df_daily['high_price'], df_daily['low_price'], df_daily['volume'] = df['close_price'], df['close_price'], df['close_price'], 1000

        with self.conn.cursor() as cur:
            cur.executemany("INSERT INTO DailyStockPrice (stock_code, date, open_price, high_price, low_price, close_price, volume) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                            df_daily[['stock_code', 'date_str', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']].values.tolist())
            cur.executemany("INSERT INTO CalculatedIndicators (stock_code, date, ma_5, ma_20) VALUES (%s,%s,%s,%s)",
                            df[['stock_code', 'date_str', 'ma_5', 'ma_20']].values.tolist())
        self.conn.commit()

    def test_backtesting_buy_scenario(self):
        data_handler = DataHandler(self.db_config)
        portfolio = Portfolio(1000000, self.START_DATE, self.END_DATE)
        strategy = SimpleBuyStrategy(self.START_DATE, self.END_DATE)
        execution_handler = TestExecutionHandler()
        backtester = BacktestEngine(self.START_DATE, self.END_DATE, portfolio, strategy, data_handler, execution_handler)
        
        final_portfolio = backtester.run()

        self.assertGreater(len(final_portfolio.trade_history), 0, "매수 거래가 발생해야 합니다.")
        self.assertEqual(final_portfolio.trade_history[0].code, self.TEST_TICKER)
        
if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBacktestingIntegration))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
