import configparser
import os
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    # Optional in minimal/laptop envs; integration tests should skip gracefully.
    np = None
    pd = None

class TestBacktestingIntegration(unittest.TestCase):
    TEST_TICKER = '999998'
    START_DATE, END_DATE = '2022-01-01', '2022-03-31'
    INITIAL_CASH = 1_000_000
    INVESTMENT_AMOUNT = 1_000_000

    @classmethod
    def setUpClass(cls):
        if pd is None or np is None:
            raise unittest.SkipTest("pandas/numpy are required for DB integration tests.")

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(repo_root, "config.ini")
        if not os.path.exists(config_path):
            raise unittest.SkipTest("config.ini not found. DB integration tests require local MySQL config.")

        # DB drivers are optional for most tests; integration test should skip if missing.
        try:
            import pymysql  # noqa: F401
            import mysql.connector  # noqa: F401
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"DB driver not installed: {exc.name}")

        # Import inside to keep module import safe in non-DB environments.
        from src.backtester import BacktestEngine
        from src.data_handler import DataHandler
        from src.db_setup import create_tables, get_db_connection
        from src.execution import BasicExecutionHandler
        from src.portfolio import Portfolio, Position
        from src.strategy import Strategy

        cls.BacktestEngine = BacktestEngine
        cls.BasicExecutionHandler = BasicExecutionHandler
        cls.DataHandler = DataHandler
        cls.Portfolio = Portfolio
        cls.Position = Position

        config = configparser.ConfigParser()
        config.read(config_path)
        if "mysql" not in config:
            raise unittest.SkipTest("config.ini is missing [mysql] section.")
        cls.db_config = dict(config["mysql"])

        # db_setup.py reads config.ini from CWD; enforce repo_root for this test.
        cwd = os.getcwd()
        os.chdir(repo_root)
        try:
            cls.conn = get_db_connection()
            create_tables(cls.conn)
        except Exception as exc:
            raise unittest.SkipTest(f"Cannot connect to MySQL: {exc}")
        finally:
            os.chdir(cwd)

        PositionImpl = cls.Position

        class SimpleBuyStrategy(Strategy):
            def __init__(self, start_date, end_date, investment_amount):
                self.start_date = pd.to_datetime(start_date)
                self.end_date = pd.to_datetime(end_date)
                self.investment_amount = float(investment_amount)
                self.invested = False

            def generate_sell_signals(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
                return []

            def generate_additional_buy_signals(
                self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None
            ):
                return []

            def generate_new_entry_signals(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
                if self.invested:
                    return []

                for code in data_handler.get_filtered_stock_codes(current_date):
                    row = data_handler.get_stock_row_as_of(code, current_date, self.start_date, self.end_date)
                    if row is None:
                        continue

                    ma_5 = row.get("ma_5")
                    ma_20 = row.get("ma_20")
                    close_price = row.get("close_price")
                    if pd.notna(ma_5) and pd.notna(ma_20) and pd.notna(close_price) and ma_5 > ma_20:
                        self.invested = True
                        position = PositionImpl(
                            buy_price=float(close_price),
                            quantity=0,
                            order=1,
                            additional_buy_drop_rate=0.05,
                            sell_profit_rate=0.10,
                        )
                        return [
                            {
                                "date": current_date,
                                "ticker": code,
                                "type": "BUY",
                                "investment_amount": self.investment_amount,
                                "position": position,
                                "reason_for_trade": "신규 진입",
                                "trigger_price": float(close_price),
                                "start_date": self.start_date,
                                "end_date": self.end_date,
                            }
                        ]
                return []

        cls.SimpleBuyStrategy = SimpleBuyStrategy

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "conn", None) is not None:
            cls.conn.close()

    def setUp(self):
        self._cleanup_test_data()
        self._prepare_test_data()

        self.filtered_codes_patcher = patch.object(
            self.DataHandler, "get_filtered_stock_codes", return_value=[self.TEST_TICKER]
        )
        self.filtered_codes_patcher.start()

    def tearDown(self):
        self.filtered_codes_patcher.stop()
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
        n = len(dates)
        n_down = min(30, max(1, n))
        n_up = n - n_down
        close_prices = np.concatenate(
            [
                np.linspace(12000, 10000, n_down),
                np.linspace(10001, 15000, n_up) if n_up > 0 else np.array([], dtype=float),
            ]
        )
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
        data_handler = self.DataHandler(self.db_config)
        portfolio = self.Portfolio(self.INITIAL_CASH, self.START_DATE, self.END_DATE)
        strategy = self.SimpleBuyStrategy(self.START_DATE, self.END_DATE, self.INVESTMENT_AMOUNT)
        execution_handler = self.BasicExecutionHandler()
        backtester = self.BacktestEngine(self.START_DATE, self.END_DATE, portfolio, strategy, data_handler, execution_handler)
        
        final_portfolio = backtester.run()

        self.assertGreater(len(final_portfolio.trade_history), 0, "매수 거래가 발생해야 합니다.")
        self.assertEqual(final_portfolio.trade_history[0].code, self.TEST_TICKER)
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
