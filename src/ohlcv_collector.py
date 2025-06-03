# ohlcv_collector.py

import pandas as pd
from pykrx import stock
import time
from sqlalchemy.exc import IntegrityError # SQLAlchemy 사용 시 DB 제약조건 위반 에러
from sqlalchemy import text # 로우 SQL 실행용
from datetime import datetime, timedelta

# 설정값 (실제로는 config 파일이나 main_script에서 파라미터로 받는 것이 좋음)
API_CALL_DELAY = 0.3  # KRX 서버 부하 방지를 위한 API 호출 간격 (초)
# pykrx는 이 날짜 이전 데이터는 없다고 가정하고, 실제 데이터 시작일부터 가져옴
DEFAULT_PYKRX_START_DATE_STR = "19800101"

def get_latest_ohlcv_date_for_ticker(engine, ticker_code):
    """
    DailyStockPrice 테이블에서 특정 종목의 가장 최근 데이터 날짜를 조회합니다.
    데이터가 없으면 None을 반환합니다.
    """
    try:
        with engine.connect() as connection:
            # text()를 사용하여 SQL 인젝션 방지 및 파라미터 바인딩
            query = text("SELECT MAX(date) FROM DailyStockPrice WHERE stock_code = :ticker")
            result = connection.execute(query, {"ticker": ticker_code}).scalar_one_or_none()
        if result:
            return pd.to_datetime(result).date() # datetime.date 객체로 반환
        return None
    except Exception as e:
        print(f"    [오류] {ticker_code}의 마지막 OHLCV 날짜 조회 중: {e}")
        return None

def collect_and_save_ohlcv_for_filtered_stocks(engine, company_manager, overall_end_date_str):
    """
    WeeklyFilteredStocks에 있는 모든 고유 종목에 대해 OHLCV 데이터를 수집하여
    DailyStockPrice 테이블에 저장합니다. 각 종목은 데이터 시작점부터 수집합니다.

    Args:
        engine (sqlalchemy.engine.Engine): DB 연결 엔진.
        company_manager (module): company_info_manager 모듈 (get_name_from_ticker 사용)
        overall_end_date_str (str): 데이터 수집의 전체 종료일 (YYYYMMDD 형식).
    """
    print("\n" + "="*50)
    print("STEP 3: OHLCV 데이터 수집 및 DailyStockPrice DB 저장 시작")
    print("="*50)

    # 1. WeeklyFilteredStocks 테이블에서 수집 대상 고유 종목코드 리스트 가져오기
    print("  OHLCV 수집 대상 종목코드 로드 중 (WeeklyFilteredStocks)...")
    try:
        with engine.connect() as connection:
            query = text("SELECT DISTINCT stock_code FROM WeeklyFilteredStocks")
            result = connection.execute(query).fetchall()
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
        # company_manager에서 종목명 가져오기 (로깅 및 확인용)
        stock_name = company_manager.get_name_from_ticker(ticker_code)
        if not stock_name: # 혹시 CompanyInfo 캐시에 없는 경우
            stock_name = "N/A"
            print(f"    [경고] {ticker_code}에 대한 종목명을 CompanyInfo 캐시에서 찾을 수 없습니다.")

        print(f"  ({i+1}/{len(unique_ticker_list)}) {ticker_code} ({stock_name}) OHLCV 수집 중...")

        # 2. DB에서 해당 종목의 가장 최근 저장된 날짜 확인
        last_saved_db_date = get_latest_ohlcv_date_for_ticker(engine, ticker_code)

        # 3. 수집할 데이터의 실제 시작일 결정
        effective_start_date_pykrx = DEFAULT_PYKRX_START_DATE_STR
        if last_saved_db_date:
            # 이미 데이터가 있다면, 마지막 저장일 다음 날부터 수집 시도
            # 만약 마지막 저장일이 overall_end_dt_obj와 같거나 미래면 이미 최신이므로 스킵
            if last_saved_db_date >= overall_end_dt_obj:
                print(f"    - {ticker_code}: 이미 최신 데이터 보유 ({last_saved_db_date}). 건너뜁니다.")
                continue
            effective_start_dt = last_saved_db_date + timedelta(days=1)
            effective_start_date_pykrx = effective_start_dt.strftime("%Y%m%d")

        # 4. 수집할 데이터의 종료일은 overall_end_date_str
        #    effective_start_date_pykrx가 overall_end_date_str보다 미래이면 수집할 데이터 없음
        if pd.to_datetime(effective_start_date_pykrx) > pd.to_datetime(overall_end_date_str):
            print(f"    - {ticker_code}: 수집 시작일({effective_start_date_pykrx})이 종료일({overall_end_date_str}) 이후입니다. 건너뜁니다.")
            continue

        try:
            time.sleep(API_CALL_DELAY)
            # pykrx는 시작일을 매우 과거로 주면 해당 종목 데이터의 시작일부터 가져옴
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
            df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date']).dt.date # datetime.date 객체로 변환

            # DB에 저장할 컬럼만 선택 (스키마에 adj_close_price가 없다면 제외)
            df_ohlcv_to_save = df_ohlcv[['stock_code', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']]

            # DB에 저장 (로우 레벨 SQL INSERT OR IGNORE 사용)
            saved_count = 0
            with engine.connect() as connection:
                for _, row_data in df_ohlcv_to_save.iterrows():
                    # stock_code, date가 PK이므로, 이 조합이 같으면 IGNORE됨
                    stmt = text("""
                        INSERT OR IGNORE INTO DailyStockPrice (stock_code, date, open_price, high_price, low_price, close_price, volume)
                        VALUES (:stock_code, :date, :open_price, :high_price, :low_price, :close_price, :volume)
                    """)
                    try:
                        # row_data가 Series이므로 딕셔너리로 변환하여 바인딩
                        connection.execute(stmt, row_data.to_dict())
                        # IGNORE 되었는지 여부를 확인하기는 어려움. 실제 INSERT된 row 수를 알 수 없음.
                        # 여기서는 시도한 것으로 간주. 더 정확한 카운트를 원하면 다른 방식 필요.
                        saved_count += 1 # 일단 시도한 횟수로 카운트
                    except Exception as e_insert_row:
                         print(f"    [오류] {ticker_code}의 {row_data['date']} 데이터 저장 중: {e_insert_row}")
                connection.commit()

            # 실제 INSERT된 row 수를 정확히 알기 어려우므로, 가져온 데이터 수로 로깅
            if not df_ohlcv_to_save.empty:
                 print(f"    - {ticker_code}: OHLCV 데이터 {len(df_ohlcv_to_save)} 건에 대해 DB 저장 시도 완료.")

        except Exception as e_pykrx:
            print(f"    [오류] {ticker_code} OHLCV 데이터 수집 중: {e_pykrx}")
            continue # 다음 티커로

    print("--- OHLCV 데이터 수집 및 저장 완료 ---")