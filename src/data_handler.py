# src/data_handler.py (수정 후)

import pandas as pd
from functools import lru_cache
from datetime import timedelta, date  # date 임포트 추가

# 중앙화된 DB 연결 함수 및 config 로더 의존성 제거
# 이제 이 모듈은 오직 db_setup만 알면 됩니다.
from .db_setup import get_db_connection


class DataHandler:
    def __init__(self):
        # __init__은 더 이상 DB 설정을 받거나 커넥션 풀을 관리하지 않습니다.
        # 모든 DB 연결은 각 함수 내에서 필요할 때마다 생성됩니다.
        pass

    def get_trading_dates(self, start_date, end_date):
        """특정 기간 동안의 모든 거래일을 DB에서 조회합니다."""
        conn = get_db_connection()
        if not conn:
            return []  # 연결 실패 시 빈 리스트 반환

        # DictCursor를 사용하므로, 결과를 딕셔너리로 쉽게 처리할 수 있습니다.
        query = "SELECT DISTINCT `date` FROM DailyStockPrice WHERE `date` BETWEEN %s AND %s ORDER BY `date`"
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, (start_date, end_date))
                rows = cursor.fetchall()
            return [row["date"] for row in rows]
        except Exception as e:
            print(f"거래일 조회 중 오류 발생: {e}")
            return []
        finally:
            if conn:
                conn.close()

    @lru_cache(maxsize=200)
    def load_stock_data(self, ticker, start_date, end_date):
        """
        특정 종목의 OHLCV 및 지표 데이터를 조회합니다.
        LRU 캐시를 사용하여 동일한 인자 호출 시 DB 조회를 생략하고 메모리에서 즉시 반환합니다.
        """
        conn = get_db_connection()
        if not conn:
            return pd.DataFrame()  # 연결 실패 시 빈 데이터프레임 반환

        # 지표 계산에 필요한 과거 데이터를 포함하여 조회 범위를 확장
        extended_start_date = pd.to_datetime(start_date) - timedelta(days=252 * 10 + 50)

        # 쿼리에서 테이블/컬럼 이름에 백틱(`)을 사용하여 예약어와의 충돌을 방지
        query = """
            SELECT
                dsp.date, dsp.open_price, dsp.high_price, dsp.low_price, dsp.close_price, dsp.volume,
                ci.ma_5, ci.ma_20, ci.atr_14_ratio, ci.price_vs_10y_low_pct AS normalized_value
            FROM DailyStockPrice AS dsp
            LEFT JOIN CalculatedIndicators AS ci ON dsp.stock_code = ci.stock_code AND dsp.date = ci.date
            WHERE dsp.stock_code = %s AND dsp.date BETWEEN %s AND %s
            ORDER BY dsp.date ASC
        """
        try:
            # pd.read_sql은 pymysql connection 객체와 완벽하게 호환됩니다.
            df = pd.read_sql(
                query, conn, params=(ticker, extended_start_date, end_date)
            )
        except Exception as e:
            print(f"종목 데이터 로드 중 오류 발생 (Ticker: {ticker}): {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        # 실제 백테스팅 기간에 해당하는 데이터만 필터링하여 반환
        df_filtered = df.loc[start_date:end_date].copy()
        return df_filtered

    def get_latest_price(self, date, ticker, start_date, end_date):
        """
        특정 날짜 또는 그 이전의 가장 최근 종가를 반환합니다.
        이 함수는 내부적으로 load_stock_data를 호출하므로 DB 연결 코드가 직접 필요 없습니다.
        """
        stock_data = self.load_stock_data(ticker, start_date, end_date)

        if stock_data is None or stock_data.empty:
            return None

        try:
            target_date = pd.to_datetime(date)
            # asof: target_date 시점 또는 그 이전의 가장 가까운 데이터를 찾아줌
            return stock_data.asof(target_date)["close_price"]
        except (KeyError, IndexError):
            # 해당 날짜 이전 데이터가 없는 경우
            return None

    def get_filtered_stock_codes(self, date):
        """특정 날짜를 기준으로 유효한 매수 후보 종목 리스트를 반환합니다."""
        conn = get_db_connection()
        if not conn:
            return []

        date_str = date.strftime("%Y-%m-%d")

        # 서브쿼리를 사용하여 특정 날짜(date_str)와 같거나 그 이전인 가장 최근 필터링 날짜의 종목들을 가져옴
        query = """
            SELECT stock_code FROM WeeklyFilteredStocks
            WHERE filter_date = (SELECT MAX(filter_date) FROM WeeklyFilteredStocks WHERE filter_date <= %s)
        """
        try:
            df = pd.read_sql(query, conn, params=[date_str])
            return df["stock_code"].tolist()
        except Exception as e:
            print(f"필터링된 종목 코드 조회 중 오류 발생: {e}")
            return []
        finally:
            if conn:
                conn.close()


# --- 테스트용 main 실행 블록 수정 ---
if __name__ == "__main__":
    # 이제 DataHandler는 초기화 시 인자가 필요 없습니다.
    data_handler = DataHandler()

    # 테스트 코드
    print("--- 거래일 조회 테스트 ---")
    dates = data_handler.get_trading_dates("2023-01-01", "2023-01-31")
    print("2023년 1월 거래일:", dates)

    print("\n--- 종목 데이터 로드 테스트 (005930 삼성전자) ---")
    stock_data = data_handler.load_stock_data("005930", "2023-01-01", "2023-12-31")
    print(stock_data.head())

    print("\n--- 특정일 가격 조회 테스트 ---")
    price = data_handler.get_latest_price(
        date(2023, 1, 10), "005930", "2023-01-01", "2023-12-31"
    )
    print(f"2023-01-10 또는 그 이전 삼성전자 종가: {price}")

    print("\n--- 필터링된 종목 조회 테스트 ---")
    filtered_stocks = data_handler.get_filtered_stock_codes(date(2023, 6, 15))
    print(
        f"2023-06-15 기준 매수 후보군: {filtered_stocks[:10]}..."
    )  # 너무 많을 수 있으니 10개만 출력
