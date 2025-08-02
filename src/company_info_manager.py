"""
company_info_manager.py

This module contains the functions for managing the company information cache.
"""

import pandas as pd
from pykrx import stock
import time
from datetime import datetime
# from sqlalchemy import text # SQLAlchemy 의존성 제거
from .db_setup import get_db_connection # MySQL 연결 함수 임포트

# 전역 변수로 사용할 인메모리 캐시
STOCK_NAME_TO_CODE_CACHE = {}
STOCK_CODE_TO_NAME_CACHE = {}
STOCK_CODE_TO_MARKET_CACHE = {} # 종목코드 -> 시장구분 캐시


# --- 1. CompanyInfo DB 및 캐시 관리 함수들 ---

def save_company_info_to_db(conn, company_data_list): # engine 대신 conn (MySQL connection)
    """
    CompanyInfo 테이블에 여러 종목 정보를 저장 (INSERT IGNORE 사용).
    company_data_list: [{'stock_code': ..., 'company_name': ..., 'market_type': ..., 'last_updated': ...}, ...]
    conn: pymysql connection 객체
    """
    if not company_data_list:
        return

    saved_count = 0
    try:
        with conn.cursor() as cur:
            for company_data in company_data_list:
                sql = """
                    INSERT IGNORE INTO CompanyInfo (stock_code, company_name, market_type, last_updated)
                    VALUES (%s, %s, %s, %s)
                """
                
                last_updated_str = company_data.get('last_updated')
                if isinstance(last_updated_str, datetime):
                    last_updated_str = last_updated_str.strftime('%Y-%m-%d %H:%M:%S')

                values = (
                    company_data.get('stock_code'),
                    company_data.get('company_name'),
                    company_data.get('market_type'),
                    last_updated_str
                )
                cur.execute(sql, values)
                saved_count += cur.rowcount # INSERT IGNORE시 실제 삽입된 경우 1, 아니면 0
            conn.commit()
        print(f"  CompanyInfo DB에 {saved_count}개 신규 종목 정보 저장 완료 (총 {len(company_data_list)}건 시도).")
    except Exception as e:
        print(f"  CompanyInfo DB 저장 중 오류: {e}")
        conn.rollback() # 오류 발생 시 롤백

def update_company_info_from_pykrx(conn, target_date_str=None): # engine 대신 conn
    """
    pykrx를 사용하여 최신 (또는 지정일 기준) 종목 정보를 가져와
    CompanyInfo DB 테이블을 업데이트하고 내부 캐시도 갱신합니다.
    target_date_str: YYYYMMDD 형식. None이면 pykrx가 내부적으로 최근 영업일 사용.
    conn: pymysql connection 객체
    """
    print(f"CompanyInfo 업데이트 시작 (기준일: {target_date_str if target_date_str else '최근 영업일'})...")

    all_market_tickers = {}
    for market_code in ["KOSPI", "KOSDAQ"]:
        try:
            tickers = stock.get_market_ticker_list(date=target_date_str, market=market_code)
            all_market_tickers[market_code] = tickers
            print(f"  {market_code} 시장 티커 {len(tickers)}개 로드 완료.")
            time.sleep(0.5)
        except Exception as e:
            print(f"  {market_code} 시장 티커 로드 중 오류: {e}")
            all_market_tickers[market_code] = []

    companies_to_upsert_list = []
    processed_tickers = set()

    for market_type, tickers in all_market_tickers.items():
        for ticker_code in tickers:
            if ticker_code in processed_tickers:
                continue
            processed_tickers.add(ticker_code)
            try:
                company_name = stock.get_market_ticker_name(ticker_code)
                if company_name:
                    companies_to_upsert_list.append({
                        'stock_code': ticker_code,
                        'company_name': company_name,
                        'market_type': market_type,
                        'last_updated': datetime.now()
                    })
                time.sleep(0.02)
            except Exception:
                pass

    if not companies_to_upsert_list:
        print("  업데이트할 새로운 종목 정보가 없습니다.")
        load_company_info_cache_from_db(conn) # conn 전달
        print(f"CompanyInfo 업데이트 대상 없음. 캐시만 재로드 완료.")
        return

    print(f"  총 {len(companies_to_upsert_list)}개의 회사 정보를 DB에 저장합니다...")
    save_company_info_to_db(conn, companies_to_upsert_list) # conn 전달

    load_company_info_cache_from_db(conn) # conn 전달
    print(f"CompanyInfo 업데이트 및 캐시 재로드 완료.")

def load_company_info_cache_from_db(conn): # engine 대신 conn
    """ DB의 CompanyInfo 테이블에서 데이터를 읽어와 인메모리 캐시를 채웁니다. """
    global STOCK_NAME_TO_CODE_CACHE, STOCK_CODE_TO_NAME_CACHE, STOCK_CODE_TO_MARKET_CACHE
    try:
        # pd.read_sql_table은 SQLAlchemy engine용. pd.read_sql_query 사용
        df = pd.read_sql_query('SELECT stock_code, company_name, market_type FROM CompanyInfo', conn)
        if not df.empty:
            STOCK_NAME_TO_CODE_CACHE = pd.Series(df.stock_code.values, index=df.company_name).to_dict()
            STOCK_CODE_TO_NAME_CACHE = pd.Series(df.company_name.values, index=df.stock_code).to_dict()
            STOCK_CODE_TO_MARKET_CACHE = pd.Series(df.market_type.values, index=df.stock_code).to_dict()
            print(f"CompanyInfo 캐시 로드 완료: {len(STOCK_NAME_TO_CODE_CACHE)}개 종목.")
        else:
            print("CompanyInfo 테이블이 비어있습니다. 캐시 로드 실패.")
            STOCK_NAME_TO_CODE_CACHE = {}
            STOCK_CODE_TO_NAME_CACHE = {}
            STOCK_CODE_TO_MARKET_CACHE = {}
    except Exception as e:
        print(f"CompanyInfo 캐시 로드 중 오류: {e}. 캐시가 비어있을 수 있습니다.")
        STOCK_NAME_TO_CODE_CACHE = {}
        STOCK_CODE_TO_NAME_CACHE = {}
        STOCK_CODE_TO_MARKET_CACHE = {}


def get_ticker_from_name(stock_name):
    """ 캐시에서 종목명으로 종목코드를 조회합니다. 없으면 None 반환. """
    return STOCK_NAME_TO_CODE_CACHE.get(stock_name)

def get_name_from_ticker(ticker_code):
    """ 캐시에서 종목코드로 종목명을 조회합니다. 없으면 None 반환. """
    return STOCK_CODE_TO_NAME_CACHE.get(ticker_code)

def get_market_from_ticker(ticker_code):
    """ 캐시에서 종목코드로 시장구분을 조회합니다. 없으면 None 반환. """
    return STOCK_CODE_TO_MARKET_CACHE.get(ticker_code)


# --- 메인 프로그램 실행부 (예시) ---
if __name__ == "__main__":
    # MySQL 연결 테스트
    conn = None # conn 변수 초기화
    try:
        conn = get_db_connection()
        print("MySQL DB 연결 성공!")

        # db_setup.py의 create_tables를 여기서 직접 호출하거나,
        # main_script.py에서 호출하도록 구성할 수 있음.
        # 여기서는 테스트 목적으로 CompanyInfo 업데이트 및 캐시 로드만 수행
        
        today_str = datetime.today().strftime("%Y%m%d")
        print("\nCompanyInfo DB 업데이트를 시도합니다...")
        update_company_info_from_pykrx(conn, target_date_str=today_str)
        print("CompanyInfo DB 업데이트 시도 완료.")

        print("\n프로그램 시작: CompanyInfo 캐시 로딩...")
        load_company_info_cache_from_db(conn)
        
        if STOCK_NAME_TO_CODE_CACHE:
            samsung_electronics_code = get_ticker_from_name("삼성전자")
            if samsung_electronics_code:
                print(f"\n삼성전자 종목코드: {samsung_electronics_code}")
                print(f"  종목명: {get_name_from_ticker(samsung_electronics_code)}")
                print(f"  시장: {get_market_from_ticker(samsung_electronics_code)}")
            else:
                print("\n삼성전자 정보를 캐시에서 찾을 수 없습니다.")
        else:
            print("\nCompanyInfo 캐시가 비어있어 조회를 수행할 수 없습니다.")

    except Exception as e:
        print(f"메인 실행 중 오류: {e}")
    finally:
        if conn:
            conn.close()
            print("\nMySQL DB 연결 해제됨.")