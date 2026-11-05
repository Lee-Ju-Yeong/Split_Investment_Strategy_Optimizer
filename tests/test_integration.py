"""임시 MySQL에서 실제 SQL과 CPU 실행 인터페이스를 확인한다."""

import os
import unittest


class TestBacktestingIntegration(unittest.TestCase):
    TEST_TICKER = "999998"
    COMPANY_NAME = "합성 검사 종목"
    START_DATE, END_DATE = "2022-01-03", "2022-01-05"
    INITIAL_CASH, INVESTMENT_AMOUNT = 1_000_000, 100_000

    @classmethod
    def setUpClass(cls):
        if "SISO_RUN_DB_TESTS" not in os.environ:
            raise unittest.SkipTest("임시 MySQL 검사는 SISO_RUN_DB_TESTS=1을 명시해야 합니다.")
        # 명시적으로 실행한 검사의 의존성/설정/연결 오류는 실패로 전파한다.
        from tools.ci.db_fixture import required_config, prepare_database
        required_config()
        import pandas as pd
        import pymysql
        from src.db_setup import create_tables
        from src.data_handler import DataHandler
        from src.backtest.cpu.backtester import BacktestEngine
        from src.backtest.cpu.execution import BasicExecutionHandler
        from src.backtest.cpu.portfolio import Portfolio, Position
        from src.backtest.cpu.strategy import Strategy

        cls.db_config, cls.conn, version = prepare_database(pymysql.connect, create_tables)
        cls.addClassCleanup(cls.conn.close)
        cls.DataHandler, cls.BacktestEngine = DataHandler, BacktestEngine
        cls.BasicExecutionHandler, cls.Portfolio = BasicExecutionHandler, Portfolio
        cls.dates = list(pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]))
        cls.SimpleBuyStrategy = cls._strategy_type(Strategy, Position)
        message = f"- 실제 MySQL 버전: {version}\n"
        print(message, flush=True)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(message)

    @classmethod
    def _strategy_type(cls, strategy_base, position_type):
        # 제품 Strategy의 추상 계약을 직접 상속해 인터페이스 변경을 감지한다.
        class SimpleBuyStrategy(strategy_base):
            max_stocks = 1

            def __init__(self):
                self.invested = False
                self.calls = []

            def generate_sell_signals(self, current_date, portfolio, data_handler,
                                      trading_dates, current_day_idx=None):
                self.calls.append(("sell", current_date, current_day_idx))
                return []

            def generate_additional_buy_signals(self, current_date, portfolio, data_handler,
                                                trading_dates, current_day_idx=None):
                self.calls.append(("additional", current_date, current_day_idx))
                return []

            def generate_new_entry_signals(self, current_date, portfolio, data_handler,
                                           trading_dates, current_day_idx=None):
                self.calls.append(("new", current_date, current_day_idx))
                if self.invested:
                    return []
                for code in data_handler.get_filtered_stock_codes(current_date):
                    row = data_handler.get_stock_row_as_of(
                        code, current_date, cls.START_DATE, cls.END_DATE,
                    )
                    if row is not None and row["ma_5"] > row["ma_20"]:
                        self.invested = True
                        return [{
                            "date": current_date, "ticker": code, "type": "BUY",
                            "investment_amount": cls.INVESTMENT_AMOUNT,
                            "position": position_type(10000, 0, 1, 0.05, 0.10),
                            "reason_for_trade": "합성 조건 신규 진입", "trigger_price": 10000,
                            "start_date": cls.START_DATE, "end_date": cls.END_DATE,
                        }]
                return []

        return SimpleBuyStrategy

    def setUp(self):
        self.addCleanup(self._cleanup_test_data)
        self._prepare_test_data()

    def _cleanup_test_data(self):
        with self.conn.cursor() as cursor:
            for table in ("WeeklyFilteredStocks", "CalculatedIndicators", "DailyStockPrice", "CompanyInfo"):
                cursor.execute(f"DELETE FROM {table} WHERE stock_code = %s", (self.TEST_TICKER,))
        self.conn.commit()

    def _prepare_test_data(self):
        with self.conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO CompanyInfo (stock_code, company_name) VALUES (%s, %s)",
                (self.TEST_TICKER, self.COMPANY_NAME),
            )
            cursor.execute(
                "INSERT INTO WeeklyFilteredStocks (filter_date, stock_code, company_name) VALUES (%s,%s,%s)",
                ("2021-12-31", self.TEST_TICKER, self.COMPANY_NAME),
            )
            for index, date in enumerate(self.dates):
                self._insert_daily_row(cursor, date)
                cursor.execute(
                    "INSERT INTO CalculatedIndicators (stock_code, date, ma_5, ma_20) VALUES (%s,%s,%s,%s)",
                    (self.TEST_TICKER, date.date(), 9 if index == 0 else 11, 10),
                )
        self.conn.commit()

    def _insert_daily_row(self, cursor, date):
        cursor.execute(
            """INSERT INTO DailyStockPrice
               (stock_code, date, open_price, high_price, low_price, close_price,
                adj_open, adj_high, adj_low, adj_close, adj_ratio, volume)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (self.TEST_TICKER, date.date(), *([10000] * 8), 1, 1000),
        )

    def _make_engine(self):
        handler = self.DataHandler(self.db_config, price_basis="adjusted", universe_mode="strict_pit")
        handler.clear_load_stock_data_cache()
        self.addCleanup(handler.clear_load_stock_data_cache)
        self.assertTrue(handler.has_stored_adj_ohlc)
        self.assertEqual(handler.get_name_from_ticker(self.TEST_TICKER), self.COMPANY_NAME)
        self.assertEqual(handler.get_trading_dates(self.START_DATE, self.END_DATE), self.dates)
        self.assertEqual(handler.get_filtered_stock_codes(self.dates[0]), [self.TEST_TICKER])
        portfolio = self.Portfolio(self.INITIAL_CASH, self.START_DATE, self.END_DATE)
        strategy = self.SimpleBuyStrategy()
        engine = self.BacktestEngine(self.START_DATE, self.END_DATE, portfolio, strategy,
                                     handler, self.BasicExecutionHandler())
        return engine, portfolio, strategy

    def _assert_daily_flow(self, portfolio, strategy):
        expected = [(stage, date, index) for index, date in enumerate(self.dates)
                    for stage in ("sell", "new", "additional")]
        self.assertEqual(strategy.calls, expected)
        self.assertEqual([row["date"] for row in portfolio.daily_snapshot_history], self.dates)

    def test_backtesting_buy_scenario(self):
        engine, portfolio, strategy = self._make_engine()
        result = engine.run()
        self.assertIs(result, portfolio)
        self._assert_daily_flow(portfolio, strategy)
        self.assertEqual(len(portfolio.trade_history), 1)
        trade = portfolio.trade_history[0]
        self.assertEqual((trade.code, trade.name, trade.trade_type),
                         (self.TEST_TICKER, self.COMPANY_NAME, "BUY"))
        self.assertEqual(trade.date, self.dates[1])
        self.assertEqual(trade.quantity, 10)
        self.assertEqual(trade.commission, 15)
        self.assertEqual(portfolio.cash, 899985)
        self.assertEqual(sum(p.quantity for p in portfolio.positions[self.TEST_TICKER]), 10)
        self.assertEqual([row["cash"] for row in portfolio.daily_snapshot_history],
                         [1000000, 899985, 899985])
        self.assertEqual([row["total_value"] for row in portfolio.daily_snapshot_history],
                         [1000000, 999985, 999985])

    def test_no_buy_signal_preserves_cash(self):
        with self.conn.cursor() as cursor:
            cursor.execute("UPDATE CalculatedIndicators SET ma_5=9 WHERE stock_code=%s", (self.TEST_TICKER,))
        self.conn.commit()
        engine, portfolio, strategy = self._make_engine()
        engine.run()
        self._assert_daily_flow(portfolio, strategy)
        self.assertEqual(portfolio.trade_history, [])
        self.assertEqual(portfolio.positions, {})
        self.assertEqual(portfolio.cash, self.INITIAL_CASH)
        self.assertTrue(all(row["cash"] == row["total_value"] == self.INITIAL_CASH
                            for row in portfolio.daily_snapshot_history))

    def test_missing_adjusted_price_is_not_hidden(self):
        with self.conn.cursor() as cursor:
            cursor.execute(
                "UPDATE DailyStockPrice SET adj_close=NULL WHERE stock_code=%s AND date=%s",
                (self.TEST_TICKER, "2022-01-04"),
            )
        self.conn.commit()
        engine, portfolio, _ = self._make_engine()
        expected = "Adjusted price mode found NULL adjusted OHLC values for ticker=999998 at date=2022-01-04"
        with self.assertRaisesRegex(ValueError, expected):
            engine.run()
        self.assertEqual(portfolio.trade_history, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
