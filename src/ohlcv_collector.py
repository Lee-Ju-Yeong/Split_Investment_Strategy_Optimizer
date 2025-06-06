# ohlcv_collector.py

import pandas as pd
from pykrx import stock
import time
import numpy as np # numpy 임포트 추가
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

def collect_and_save_ohlcv_for_filtered_stocks(conn, company_manager, overall_end_date_str, force_recollect=False):
    """
    WeeklyFilteredStocks에 있는 모든 고유 종목에 대해 OHLCV 데이터를 수집하여
    DailyStockPrice 테이블에 저장합니다.
    (수정) pykrx API가 약 3000 영업일치의 데이터만 제공하는 것으로 보이므로,
           전체 기간을 한번에 요청하는 단순한 방식으로 되돌립니다.

    Args:
        conn (pymysql.connections.Connection): DB 연결 객체.
        company_manager (module): company_info_manager 모듈 (get_name_from_ticker 사용)
        overall_end_date_str (str): 데이터 수집의 전체 종료일 (YYYYMMDD 형식).
        force_recollect (bool): True이면, 수집 대상 종목의 기존 데이터를 모두 삭제하고 다시 수집합니다.
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

    # 강제 재수집 옵션이 켜져 있으면, 대상 종목들의 기존 데이터를 모두 삭제
    if force_recollect and unique_ticker_list:
        print("\n  [주의] FORCE_RECOLLECT_OHLCV 옵션이 활성화되었습니다.")
        print(f"  DailyStockPrice 테이블에서 {len(unique_ticker_list)}개 종목의 기존 OHLCV 데이터를 모두 삭제합니다.")
        try:
            with conn.cursor() as cur:
                placeholders = ', '.join(['%s'] * len(unique_ticker_list))
                sql_delete = f"DELETE FROM DailyStockPrice WHERE stock_code IN ({placeholders})"
                deleted_count = cur.execute(sql_delete, tuple(unique_ticker_list))
                conn.commit()
                print(f"  성공적으로 {deleted_count}개의 기존 OHLCV 레코드를 삭제했습니다.")
        except Exception as e_delete:
            print(f"  [오류] 기존 OHLCV 데이터 삭제 중: {e_delete}")
            conn.rollback()
            return

    overall_end_dt_obj = datetime.strptime(overall_end_date_str, "%Y%m%d").date()

    for i, ticker_code in enumerate(unique_ticker_list):
        stock_name = company_manager.get_name_from_ticker(ticker_code) if company_manager else "N/A"
        print(f"\n  ({i+1}/{len(unique_ticker_list)}) {ticker_code} ({stock_name}) OHLCV 수집 중...")

        last_saved_db_date = None
        if not force_recollect:
             last_saved_db_date = get_latest_ohlcv_date_for_ticker(conn, ticker_code)

        effective_start_date_pykrx = DEFAULT_PYKRX_START_DATE_STR
        if last_saved_db_date:
            if last_saved_db_date >= overall_end_dt_obj:
                print(f"    - {ticker_code}: 이미 최신 데이터 보유 ({last_saved_db_date}). 건너뜁니다.")
                continue
            effective_start_dt = last_saved_db_date + timedelta(days=1)
            effective_start_date_pykrx = effective_start_dt.strftime("%Y%m%d")
        
        try:
            time.sleep(API_CALL_DELAY)
            df_ohlcv = stock.get_market_ohlcv(effective_start_date_pykrx, overall_end_date_str, ticker_code)

            if df_ohlcv.empty:
                print(f"    - {ticker_code}: {effective_start_date_pykrx} ~ {overall_end_date_str} 기간에 추가 데이터 없음.")
                continue

            df_ohlcv.replace({np.nan: None}, inplace=True)
            df_ohlcv.reset_index(inplace=True)
            df_ohlcv.rename(columns={
                '날짜': 'date', '시가': 'open_price', '고가': 'high_price',
                '저가': 'low_price', '종가': 'close_price', '거래량': 'volume'
            }, inplace=True)
            df_ohlcv['stock_code'] = ticker_code
            df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date']).dt.strftime('%Y-%m-%d')
            
            df_ohlcv_to_save = df_ohlcv[['stock_code', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']]
            rows_to_insert = df_ohlcv_to_save.to_records(index=False).tolist()

            if not rows_to_insert:
                print(f"    - {ticker_code}: DB에 저장할 변환된 데이터가 없습니다.")
                continue

            with conn.cursor() as cur:
                sql_insert = """
                    INSERT IGNORE INTO DailyStockPrice (stock_code, date, open_price, high_price, low_price, close_price, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                try:
                    cur.executemany(sql_insert, rows_to_insert)
                    saved_count = cur.rowcount
                    conn.commit()
                    if saved_count > 0:
                        print(f"    - {ticker_code}: OHLCV 데이터 {saved_count} 건 DB 저장 완료.")
                    elif not df_ohlcv_to_save.empty:
                        print(f"    - {ticker_code}: 신규 저장 데이터 없음 (중복).")
                except Exception as e_insert:
                    print(f"    [오류] {ticker_code} 데이터 저장 중: {e_insert}")
                    conn.rollback()

        except Exception as e_pykrx:
            print(f"    [오류] {ticker_code} OHLCV 데이터 수집 중: {e_pykrx}")
            continue

    print("\n--- OHLCV 데이터 수집 및 저장 완료 ---")