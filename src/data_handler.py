# data_handler.py (수정된 최종본)

import pandas as pd
from mysql.connector import pooling
from functools import lru_cache
from datetime import timedelta

# CompanyInfo 캐시를 직접 관리
STOCK_CODE_TO_NAME_CACHE = {}

class DataHandler:
    def __init__(self, db_config):
        self.db_config = db_config
        try:
            self.connection_pool = pooling.MySQLConnectionPool(pool_name="data_pool",
                                                               pool_size=10,
                                                               **self.db_config)
            self._load_company_info_cache()
        except Exception as e:
            print(f"DB 연결 풀 생성 또는 캐시 로딩 실패: {e}")
            raise

    def _load_company_info_cache(self):
        """DB의 CompanyInfo 테이블에서 데이터를 읽어와 인메모리 캐시를 채웁니다."""
        global STOCK_CODE_TO_NAME_CACHE
        print("CompanyInfo 캐시 로딩 중...")
        conn = self.get_connection()
        try:
            # pymysql은 read_sql_table을 지원하지 않으므로 read_sql_query 사용
            df = pd.read_sql_query('SELECT stock_code, company_name FROM CompanyInfo', conn)
            if not df.empty:
                STOCK_CODE_TO_NAME_CACHE = pd.Series(df.company_name.values, index=df.stock_code).to_dict()
                print(f"CompanyInfo 캐시 로드 완료: {len(STOCK_CODE_TO_NAME_CACHE)}개 종목.")
            else:
                print("경고: CompanyInfo 테이블이 비어있습니다. 종목명이 N/A로 표시될 수 있습니다.")
                STOCK_CODE_TO_NAME_CACHE = {}
        except Exception as e:
            print(f"CompanyInfo 캐시 로드 중 오류: {e}")
            STOCK_CODE_TO_NAME_CACHE = {}
        finally:
            conn.close()
    
    def get_name_from_ticker(self, ticker_code):
        """캐시에서 종목코드로 종목명을 조회합니다."""
        return STOCK_CODE_TO_NAME_CACHE.get(ticker_code)

    def get_connection(self):
        return self.connection_pool.get_connection()

    def get_trading_dates(self, start_date, end_date):
        conn = self.get_connection()
        try:
            query = "SELECT DISTINCT date FROM DailyStockPrice WHERE date BETWEEN %s AND %s ORDER BY date"
            # pd.read_sql 사용 시 날짜 파싱이 더 안정적
            df = pd.read_sql(query, conn, params=(start_date, end_date))
            return pd.to_datetime(df['date']).tolist()
        finally:
            conn.close()

    @lru_cache(maxsize=200)
    def load_stock_data(self, ticker, start_date, end_date):
        conn = self.get_connection()
        # 지표 계산에 필요한 충분한 과거 데이터를 위해 시작 날짜 확장
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
        try:
            df = pd.read_sql(query, conn, params=(ticker, extended_start_date, end_date))
            if df.empty:
                return df

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 실제 백테스팅 기간 데이터만 필터링하여 반환
            df_filtered = df.loc[start_date:end_date].copy()
            return df_filtered
        finally:
            conn.close()


    def get_latest_price(self, date, ticker, start_date, end_date):
        stock_data = self.load_stock_data(ticker, start_date, end_date)
        
        if stock_data is None or stock_data.empty:
            return None
        
        try:
            target_date = pd.to_datetime(date)
            # asof: date 또는 그 이전의 가장 마지막 데이터를 찾아줌
            return stock_data.asof(target_date)['close_price']
        except (KeyError, IndexError):
            return None

    def get_ohlc_data_on_date(self, date, ticker, start_date, end_date):
        stock_data = self.load_stock_data(ticker, start_date, end_date)
        if stock_data is None or stock_data.empty:
            return None
        
        try:
            target_date = pd.to_datetime(date)
            # asof를 사용하여 해당 날짜 또는 그 이전의 가장 가까운 데이터를 Series로 반환
            data_row = stock_data.asof(target_date)
            if data_row is None or pd.isna(data_row.name):
                return None
            return data_row
        except (KeyError, IndexError):
            return None

    def get_filtered_stock_codes(self, date):
        conn = self.get_connection()
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        
        query = """
            SELECT stock_code FROM WeeklyFilteredStocks
            WHERE filter_date = (SELECT MAX(filter_date) FROM WeeklyFilteredStocks WHERE filter_date < %s)
        """
        try:
            df = pd.read_sql(query, conn, params=[date_str])
            return df['stock_code'].tolist()
        finally:
            conn.close()