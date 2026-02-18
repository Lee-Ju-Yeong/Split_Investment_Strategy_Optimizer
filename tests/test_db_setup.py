import unittest
from unittest.mock import MagicMock
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from db_setup import create_tables


class TestDbSetup(unittest.TestCase):
    def test_create_tables_creates_issue65_schema_and_indexes(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        create_tables(mock_conn)

        executed_sql = [call.args[0] for call in mock_cursor.execute.call_args_list if call.args]
        joined_sql = "\n".join(executed_sql)

        self.assertIn("CREATE TABLE IF NOT EXISTS FinancialData", joined_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS InvestorTradingTrend", joined_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS MarketCapDaily", joined_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS ShortSellingDaily", joined_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS ShortSellingBackfillCoverage", joined_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS DailyStockTier", joined_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS TickerUniverseSnapshot", joined_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS TickerUniverseHistory", joined_sql)
        self.assertIn("adj_close DECIMAL(20, 5) NULL", joined_sql)
        self.assertIn("adj_ratio DECIMAL(20, 10) NULL", joined_sql)
        self.assertIn("ALTER TABLE DailyStockPrice ADD COLUMN adj_close", joined_sql)
        self.assertIn("ALTER TABLE DailyStockPrice ADD COLUMN adj_ratio", joined_sql)
        self.assertIn("CREATE INDEX idx_financial_date_stock", joined_sql)
        self.assertIn("CREATE INDEX idx_investor_date_stock", joined_sql)
        self.assertIn("CREATE INDEX idx_investor_date_flow", joined_sql)
        self.assertIn("CREATE INDEX idx_mcap_date_stock", joined_sql)
        self.assertIn("CREATE INDEX idx_short_date_stock", joined_sql)
        self.assertIn("CREATE INDEX idx_short_cov_status_updated", joined_sql)
        self.assertIn("CREATE INDEX idx_short_cov_stock_status", joined_sql)
        self.assertIn("CREATE INDEX idx_tier_stock_date", joined_sql)
        self.assertIn("CREATE INDEX idx_tier_date_tier_stock", joined_sql)
        self.assertIn("CREATE INDEX idx_tus_stock_date", joined_sql)
        self.assertIn("CREATE INDEX idx_tus_date_market_stock", joined_sql)
        self.assertIn("CREATE INDEX idx_tuh_listed_date", joined_sql)
        self.assertIn("CREATE INDEX idx_tuh_last_seen_date", joined_sql)
        self.assertIn("CREATE INDEX idx_tuh_delisted_date", joined_sql)
        mock_conn.commit.assert_called_once()
        mock_cursor.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
