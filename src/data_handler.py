import pandas as pd
import mysql.connector
from mysql.connector import pooling
from functools import lru_cache
from datetime import timedelta
import configparser

class DataHandler:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection_pool = pooling.MySQLConnectionPool(pool_name="data_pool",
                                                           pool_size=10,
                                                           **self.db_config)

    def get_connection(self):
        return self.connection_pool.get_connection()

    def get_trading_dates(self, start_date, end_date):
        conn = self.get_connection()
        cursor = conn.cursor()
        query = "SELECT DISTINCT date FROM DailyStockPrice WHERE date BETWEEN %s AND %s ORDER BY date"
        cursor.execute(query, (start_date, end_date))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [row[0] for row in rows]

    @lru_cache(maxsize=200)
    def load_stock_data(self, ticker, start_date, end_date):
        conn = self.get_connection()
        extended_start_date = pd.to_datetime(start_date) - timedelta(days=252*10 + 50)
        
        query = """
            SELECT
                dsp.date, dsp.open_price, dsp.high_price, dsp.low_price, dsp.close_price, dsp.volume,
                ci.ma_5, ci.ma_20, ci.atr_14_ratio, ci.price_vs_5y_low_pct, ci.price_vs_10y_low_pct AS normalized_value
            FROM DailyStockPrice dsp
            LEFT JOIN CalculatedIndicators ci ON dsp.stock_code = ci.stock_code AND dsp.date = ci.date
            WHERE dsp.stock_code = %s AND dsp.date BETWEEN %s AND %s
            ORDER BY dsp.date ASC
        """
        df = pd.read_sql(query, conn, params=(ticker, extended_start_date, end_date))
        conn.close()

        if df.empty:
            return df

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        df_filtered = df.loc[start_date:end_date].copy()
        return df_filtered

    def get_latest_price(self, date, ticker, start_date, end_date):
        stock_data = self.load_stock_data(ticker, start_date, end_date)
        
        if stock_data is None or stock_data.empty:
            return None
        
        try:
            # datetime.date 객체를 Timestamp로 변환
            target_date = pd.to_datetime(date)
            return stock_data.asof(target_date)['close_price']
        except (KeyError, IndexError):
            return None

    def get_filtered_stock_codes(self, date):
        conn = self.get_connection()
        date_str = date.strftime('%Y-%m-%d')
        
        query = """
            SELECT stock_code FROM WeeklyFilteredStocks
            WHERE filter_date = (SELECT MAX(filter_date) FROM WeeklyFilteredStocks WHERE filter_date <= %s)
        """
        try:
            df = pd.read_sql(query, conn, params=[date_str])
            return df['stock_code'].tolist()
        finally:
            conn.close()

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    db_params = {
        'host': config['mysql']['host'], 'user': config['mysql']['user'],
        'password': config['mysql']['password'], 'database': config['mysql']['database'],
    }
    data_handler = DataHandler(db_params)
    dates = data_handler.get_trading_dates('2022-01-01', '2022-01-31')
    print("Trading Dates:", dates)
    stock_data = data_handler.load_stock_data('005930', '2022-01-01', '2023-12-31')
    print("\nSamsung Electronics Data (005930):")
    print(stock_data.head())
    price = data_handler.get_latest_price(date(2022, 1, 10), '005930', '2022-01-01', '2023-12-31')
    print(f"\nPrice of 005930 on or before 2022-01-10: {price}")
