import pandas as pd
from pykrx import stock
import time
from datetime import datetime
from sqlalchemy import text # save_company_info_to_db 에서 사용
# from db_setup import get_db_connection # 기존 DB 연결 함수 사용 또는 SQLAlchemy 엔진 직접 사용

# 전역 변수로 사용할 인메모리 캐시
STOCK_NAME_TO_CODE_CACHE = {}
STOCK_CODE_TO_NAME_CACHE = {}
STOCK_CODE_TO_MARKET_CACHE = {} # 종목코드 -> 시장구분 캐시


# --- 1. CompanyInfo DB 및 캐시 관리 함수들 ---

def save_company_info_to_db(engine, company_data_list):
    """
    CompanyInfo 테이블에 여러 종목 정보를 저장 (INSERT OR IGNORE).
    company_data_list: [{'stock_code': ..., 'company_name': ..., 'market_type': ..., 'last_updated': ...}, ...]
    """
    if not company_data_list:
        return

    # df_companies = pd.DataFrame(company_data_list) # save_company_info_to_db 함수 내부에서 처리할 수 있음
                                                 # 또는 여기서 만들어서 넘겨도 무방. 여기서는 리스트를 넘김.

    try:
        # SQLite의 경우 'INSERT OR IGNORE'를 사용하려면 로우 레벨 SQL 또는 SQLAlchemy Core 사용
        with engine.connect() as connection:
            for company_data in company_data_list: # 딕셔너리 리스트를 직접 순회
                stmt = text("""
                    INSERT OR IGNORE INTO CompanyInfo (stock_code, company_name, market_type, last_updated)
                    VALUES (:stock_code, :company_name, :market_type, :last_updated)
                """)
                # last_updated를 문자열로 변환하여 DB 호환성 확보
                company_data_for_sql = company_data.copy()
                if isinstance(company_data_for_sql.get('last_updated'), datetime):
                    company_data_for_sql['last_updated'] = company_data_for_sql['last_updated'].strftime('%Y-%m-%d %H:%M:%S')
                
                connection.execute(stmt, company_data_for_sql)
            connection.commit()
        print(f"  CompanyInfo DB에 {len(company_data_list)}개 종목 정보 저장/갱신 시도 완료.")
    except Exception as e:
        print(f"  CompanyInfo DB 저장 중 오류: {e}")

def update_company_info_from_pykrx(engine, target_date_str=None):
    """
    pykrx를 사용하여 최신 (또는 지정일 기준) 종목 정보를 가져와
    CompanyInfo DB 테이블을 업데이트하고 내부 캐시도 갱신합니다.
    target_date_str: YYYYMMDD 형식. None이면 pykrx가 내부적으로 최근 영업일 사용.
    engine: SQLAlchemy 엔진 객체
    """
    # 전역 캐시 변수 사용 명시 (함수 내에서 수정 시 필요)
    # global STOCK_NAME_TO_CODE_CACHE, STOCK_CODE_TO_NAME_CACHE, STOCK_CODE_TO_MARKET_CACHE
    # 위 전역 변수 직접 수정은 load_company_info_cache_from_db에서 하므로 여기선 불필요

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

    companies_to_upsert_list = [] # DB에 저장할 최종 리스트
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
                        'last_updated': datetime.now() # datetime 객체로 저장
                    })
                time.sleep(0.02) # 이전 0.05에서 줄임 (테스트 필요)
            except Exception:
                pass

    if not companies_to_upsert_list:
        print("  업데이트할 새로운 종목 정보가 없습니다.")
        # 캐시 재로드는 DB 변경이 없어도 현재 캐시 상태를 명확히 하기 위해 호출 가능
        load_company_info_cache_from_db(engine)
        print(f"CompanyInfo 업데이트 대상 없음. 캐시만 재로드 완료.")
        return

    # ------------------- 수정된 부분 ------------------- #
    # companies_to_upsert_list를 save_company_info_to_db 함수에 전달하여 DB 저장
    print(f"  총 {len(companies_to_upsert_list)}개의 회사 정보를 DB에 저장합니다...")
    save_company_info_to_db(engine, companies_to_upsert_list)
    # ------------------------------------------------- #

    # DB 업데이트 후 캐시를 DB 기준으로 다시 로드 (일관성 유지)
    load_company_info_cache_from_db(engine)
    print(f"CompanyInfo 업데이트 및 캐시 재로드 완료.")

def load_company_info_cache_from_db(engine):
    """ DB의 CompanyInfo 테이블에서 데이터를 읽어와 인메모리 캐시를 채웁니다. """
    global STOCK_NAME_TO_CODE_CACHE, STOCK_CODE_TO_NAME_CACHE, STOCK_CODE_TO_MARKET_CACHE
    try:
        df = pd.read_sql_table('CompanyInfo', engine)
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
    from sqlalchemy import create_engine
    db_path = "stock_backtesting.db" 
    engine = create_engine(f'sqlite:///{db_path}')
    
    # 테이블이 없다면 생성 (간단한 예시, 실제로는 alembic 등 마이그레이션 도구 사용 권장)
    # 또는 별도의 setup_database.py 스크립트로 관리
    with engine.connect() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS CompanyInfo (
                stock_code VARCHAR(6) PRIMARY KEY,
                company_name TEXT,
                market_type VARCHAR(10),
                last_updated DATETIME
            )
        """))
        connection.commit()
    
    # 1. CompanyInfo DB 및 캐시 업데이트 (필요시 실행)
    today_str = datetime.today().strftime("%Y%m%d")
    print("CompanyInfo DB 업데이트를 시도합니다...")
    update_company_info_from_pykrx(engine, target_date_str=today_str)
    print("CompanyInfo DB 업데이트 시도 완료.")

    # 2. 프로그램 시작 시 DB에서 캐시 로드 (필수)
    print("\n프로그램 시작: CompanyInfo 캐시 로딩...")
    load_company_info_cache_from_db(engine)
    
    # 3. 캐시 사용 예시
    if STOCK_NAME_TO_CODE_CACHE: # 캐시가 로드되었는지 확인
        samsung_electronics_code = get_ticker_from_name("삼성전자")
        if samsung_electronics_code:
            print(f"\n삼성전자 종목코드: {samsung_electronics_code}")
            print(f"  종목명: {get_name_from_ticker(samsung_electronics_code)}")
            print(f"  시장: {get_market_from_ticker(samsung_electronics_code)}")
        else:
            print("\n삼성전자 정보를 캐시에서 찾을 수 없습니다. DB 업데이트 및 캐시 로드를 확인하세요.")
    else:
        print("\nCompanyInfo 캐시가 비어있어 조회를 수행할 수 없습니다.")