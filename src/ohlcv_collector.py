# ohlcv_collector.py

import pandas as pd
from pykrx import stock
import time
# from sqlalchemy.exc import IntegrityError # SQLAlchemy 의존성 제거
# from sqlalchemy import text # SQLAlchemy 의존성 제거
from datetime import datetime, timedelta

# 설정값
API_CALL_DELAY = 0.3
DEFAULT_PYKRX_START_DATE_STR = "19800101"

def get_latest_ohlcv_date_for_ticker(conn, ticker_code): # engine 대신 conn (MySQL connection)
    """
    DailyStockPrice 테이블에서 특정 종목의 가장 최근 데이터 날짜를 조회합니다.
    데이터가 없으면 None을 반환합니다.
    conn: pymysql connection 객체
    """
    try:
        with conn.cursor() as cur:
            sql = "SELECT MAX(date) FROM DailyStockPrice WHERE stock_code = %s"
            cur.execute(sql, (ticker_code,))
            result = cur.fetchone()
        if result and result[0]: # result가 None이 아니고, 첫 번째 값도 None이 아닐 때
            return pd.to_datetime(result[0]).date()
        return None
    except Exception as e:
        print(f"    [오류] {ticker_code}의 마지막 OHLCV 날짜 조회 중: {e}")
        return None

def collect_and_save_ohlcv_for_filtered_stocks(conn, company_manager, overall_end_date_str): # engine 대신 conn
    """
    WeeklyFilteredStocks에 있는 모든 고유 종목에 대해 OHLCV 데이터를 수집하여
    DailyStockPrice 테이블에 저장합니다. 각 종목은 데이터 시작점부터 수집합니다.

    Args:
        conn (pymysql.connections.Connection): DB 연결 객체.
        company_manager (module): company_info_manager 모듈 (get_name_from_ticker 사용)
        overall_end_date_str (str): 데이터 수집의 전체 종료일 (YYYYMMDD 형식).
    """
    print("\n" + "="*50)
    print("STEP 3: OHLCV 데이터 수집 및 DailyStockPrice DB 저장 시작")
    print("="*50)

    print("  OHLCV 수집 대상 종목코드 로드 중 (WeeklyFilteredStocks)...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT stock_code FROM WeeklyFilteredStocks")
            result = cur.fetchall()
        if not result:
            print("  WeeklyFilteredStocks 테이블에 수집 대상 종목이 없습니다. OHLCV 수집을 건너뜁니다.")
            return
        unique_ticker_list = [row[0] for row in result]
        print(f"  총 {len(unique_ticker_list)}개의 고유 종목에 대해 OHLCV 수집 예정.")
    except Exception as e:
        print(f"  [오류] WeeklyFilteredStocks 테이블에서 종목코드 로드 중: {e}")
        return

    overall_end_dt_obj = datetime.strptime(overall_end_date_str, "%Y%m%d").date()

    for i, ticker_code in enumerate(unique_ticker_list):
        stock_name = company_manager.get_name_from_ticker(ticker_code) if company_manager else "N/A"
        if not stock_name or stock_name == "N/A":
            stock_name = "N/A"
            # print(f"    [경고] {ticker_code}에 대한 종목명을 CompanyInfo 캐시에서 찾을 수 없습니다.") # 필요시 로깅

        print(f"  ({i+1}/{len(unique_ticker_list)}) {ticker_code} ({stock_name}) OHLCV 수집 중...")

        last_saved_db_date = get_latest_ohlcv_date_for_ticker(conn, ticker_code)

        effective_start_date_pykrx = DEFAULT_PYKRX_START_DATE_STR
        if last_saved_db_date:
            if last_saved_db_date >= overall_end_dt_obj:
                print(f"    - {ticker_code}: 이미 최신 데이터 보유 ({last_saved_db_date}). 건너뜁니다.")
                continue
            effective_start_dt = last_saved_db_date + timedelta(days=1)
            effective_start_date_pykrx = effective_start_dt.strftime("%Y%m%d")

        if pd.to_datetime(effective_start_date_pykrx) > pd.to_datetime(overall_end_date_str):
            print(f"    - {ticker_code}: 수집 시작일({effective_start_date_pykrx})이 종료일({overall_end_date_str}) 이후입니다. 건너뜁니다.")
            continue

        try:
            time.sleep(API_CALL_DELAY)
            df_ohlcv = stock.get_market_ohlcv(effective_start_date_pykrx, overall_end_date_str, ticker_code)

            if df_ohlcv.empty:
                print(f"    - {ticker_code}: {effective_start_date_pykrx} ~ {overall_end_date_str} 기간에 추가 OHLCV 데이터 없음.")
                continue

            df_ohlcv.reset_index(inplace=True)
            df_ohlcv.rename(columns={
                '날짜': 'date', '시가': 'open_price', '고가': 'high_price',
                '저가': 'low_price', '종가': 'close_price', '거래량': 'volume'
            }, inplace=True)
            df_ohlcv['stock_code'] = ticker_code
            # 날짜를 MySQL 'DATE' 타입에 맞는 문자열로 변환
            df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date']).dt.strftime('%Y-%m-%d')

            df_ohlcv_to_save = df_ohlcv[['stock_code', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']]
            
            rows_to_insert = df_ohlcv_to_save.to_records(index=False).tolist()

            if not rows_to_insert:
                print(f"    - {ticker_code}: DB에 저장할 변환된 데이터가 없습니다.")
                continue

            saved_count_for_ticker = 0
            with conn.cursor() as cur:
                # MySQL에서는 INSERT IGNORE 사용
                sql_insert = """
                    INSERT IGNORE INTO DailyStockPrice (stock_code, date, open_price, high_price, low_price, close_price, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                try:
                    # executemany 사용으로 성능 향상 기대
                    cur.executemany(sql_insert, rows_to_insert)
                    saved_count_for_ticker = cur.rowcount # executemany의 rowcount는 실제 영향받은 행 수
                    conn.commit()
                    if saved_count_for_ticker > 0:
                        print(f"    - {ticker_code}: OHLCV 데이터 {saved_count_for_ticker} 건 DB 저장 완료 (총 {len(rows_to_insert)}건 시도).")
                    elif not df_ohlcv_to_save.empty: # 데이터는 있었으나 중복 등으로 저장 안된 경우
                        print(f"    - {ticker_code}: OHLCV 데이터 {len(rows_to_insert)} 건 있었으나, 신규 저장된 데이터 없음 (중복 가능성).")
                except Exception as e_insert_many:
                    print(f"    [오류] {ticker_code} 데이터 일괄 저장 중: {e_insert_many}")
                    conn.rollback()

        except Exception as e_pykrx:
            print(f"    [오류] {ticker_code} OHLCV 데이터 수집 중: {e_pykrx}")
            continue

    print("--- OHLCV 데이터 수집 및 저장 완료 ---")