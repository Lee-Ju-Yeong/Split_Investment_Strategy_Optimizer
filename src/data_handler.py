import pandas as pd
import mysql.connector
from mysql.connector import pooling
from functools import lru_cache
from datetime import timedelta
import configparser

class DataHandler:
    """
    데이터베이스 연결, 데이터 조회 및 캐싱 등 모든 데이터 관련 작업을 처리합니다.
    """
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection_pool = pooling.MySQLConnectionPool(pool_name="data_pool",
                                                           pool_size=10,
                                                           **self.db_config)
        self.loaded_stock_data = {}

    def get_connection(self):
        return self.connection_pool.get_connection()

    def get_trading_dates(self, start_date, end_date):
        """
        주어진 기간 내의 모든 거래일을 DB에서 조회합니다.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        query = "SELECT DISTINCT date FROM DailyStockPrice WHERE date BETWEEN %s AND %s ORDER BY date"
        cursor.execute(query, (start_date, end_date))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [row[0] for row in rows]

    @lru_cache(maxsize=100) # 종목별 데이터 로딩 결과를 캐싱
    def load_stock_data(self, ticker, start_date, end_date):
        """
        특정 종목의 OHLCV 및 계산된 지표 데이터를 DB에서 로드합니다.
        """
        if ticker in self.loaded_stock_data:
            return self.loaded_stock_data[ticker]

        conn = self.get_connection()
        
        # 지표 계산에 필요한 충분한 과거 데이터를 포함하여 조회
        extended_start_date = pd.to_datetime(start_date) - timedelta(days=252*10 + 50)
        
        query = """
            SELECT
                dsp.date,
                dsp.open_price,
                dsp.high_price,
                dsp.low_price,
                dsp.close_price,
                dsp.volume,
                ci.ma_5,
                ci.ma_20,
                ci.atr_14_ratio,
                ci.price_vs_5y_low_pct,
                ci.price_vs_10y_low_pct AS normalized_value
            FROM DailyStockPrice dsp
            LEFT JOIN CalculatedIndicators ci ON dsp.stock_code = ci.stock_code AND dsp.date = ci.date
            WHERE dsp.stock_code = %s AND dsp.date BETWEEN %s AND %s
            ORDER BY dsp.date ASC
        """
        df = pd.read_sql(query, conn, params=(ticker, extended_start_date, end_date))
        conn.close()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # 실제 백테스팅 기간에 해당하는 데이터만 사용하도록 필터링
        df_filtered = df.loc[start_date:end_date].copy()
        
        self.loaded_stock_data[ticker] = df_filtered
        return df_filtered

    def get_market_data_for_date(self, date):
        """
        특정 날짜의 모든 종목 데이터를 가져옵니다. (필요시 구현)
        - 이 프로젝트에서는 종목별로 데이터를 가져오므로, 이 메소드는 일단 비워둡니다.
        """
        pass
        
    def get_latest_price(self, date, ticker):
        """
        특정 종목의 주어진 날짜 또는 그 이전 가장 가까운 날의 종가를 반환합니다.
        """
        if ticker not in self.loaded_stock_data:
            return None
        
        df = self.loaded_stock_data[ticker]
        
        # asof는 특정 날짜 또는 그 이전의 가장 최근 데이터를 찾아줍니다.
        price = df.loc[:date].tail(1)
        if not price.empty:
            return price['close_price'].iloc[0]
        return None

    def get_filtered_stock_codes(self, date):
        """
        특정 날짜(current_date)를 기준으로, 가장 최근의 필터링 날짜(filter_date)에
        해당하는 종목 코드 리스트를 WeeklyFilteredStocks 테이블에서 조회합니다.
        """
        conn = self.get_connection()
        date_str = date.strftime('%Y-%m-%d')
        
        # 주어진 날짜(date_str)와 같거나 그보다 이전인 가장 마지막 필터링 날짜를 찾고,
        # 해당 날짜에 필터링된 모든 종목 코드를 반환합니다.
        query = """
            SELECT stock_code
            FROM WeeklyFilteredStocks
            WHERE filter_date = (
                SELECT MAX(filter_date)
                FROM WeeklyFilteredStocks
                WHERE filter_date <= %s
            )
        """
        try:
            df = pd.read_sql(query, conn, params=[date_str])
            return df['stock_code'].tolist()
        finally:
            conn.close()

if __name__ == '__main__':
    # 테스트 코드
    config = configparser.ConfigParser()
    config.read('config.ini')

    db_params = {
        'host': config['mysql']['host'],
        'user': config['mysql']['user'],
        'password': config['mysql']['password'],
        'database': config['mysql']['database'],
    }
    
    data_handler = DataHandler(db_params)
    dates = data_handler.get_trading_dates('2022-01-01', '2022-01-31')
    print("Trading Dates:", dates)

    stock_data = data_handler.load_stock_data('005930', '2022-01-01', '2023-12-31')
    print("\nSamsung Electronics Data (005930):")
    print(stock_data.head())

    price = data_handler.get_latest_price('2022-01-10', '005930')
    print(f"\nPrice of 005930 on or before 2022-01-10: {price}")
