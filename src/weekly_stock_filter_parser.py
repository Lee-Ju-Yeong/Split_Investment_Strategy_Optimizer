import pandas as pd
import os
import glob
from datetime import datetime, timedelta
import time
from pykrx import stock
from sqlalchemy import create_engine, text # text는 SQL 직접 실행 시 필요

# --- 0. 설정 ---
# DB 파일 경로 (SQLite 사용)
DB_FILE_PATH = "stock_backtesting.db"
ENGINE = create_engine(f'sqlite:///{DB_FILE_PATH}')

# HTS 조건검색 결과 CSV 파일들이 저장된 폴더
CONDITION_SEARCH_FILES_FOLDER = 'E:/AI/pythonProject/venv/masicsplit/data/raw_data' # 실제 경로로 수정

# 처리된 데이터 저장 폴더 (CSV 저장 시)
PROCESSED_DATA_FOLDER = os.path.join(os.path.dirname(CONDITION_SEARCH_FILES_FOLDER), "processed_data")

# 인메모리 캐시 (프로그램 실행 시 DB에서 로드)
STOCK_NAME_TO_CODE_CACHE = {}
STOCK_CODE_TO_NAME_CACHE = {}
STOCK_CODE_TO_MARKET_CACHE = {}

# --- 1. CompanyInfo DB 및 캐시 관리 함수들 ---

def save_company_info_to_db(engine, company_data_list):
    """
    CompanyInfo 테이블에 여러 종목 정보를 저장 (INSERT OR IGNORE).
    company_data_list: [{'stock_code': ..., 'company_name': ..., 'market_type': ..., 'last_updated': ...}, ...]
    """
    if not company_data_list:
        return

    df_companies = pd.DataFrame(company_data_list)
    try:
        # SQLite의 경우 'INSERT OR IGNORE'를 사용하려면 로우 레벨 SQL 또는 SQLAlchemy Core 사용
        with engine.connect() as connection:
            for _, row in df_companies.iterrows():
                stmt = text("""
                    INSERT OR IGNORE INTO CompanyInfo (stock_code, company_name, market_type, last_updated)
                    VALUES (:stock_code, :company_name, :market_type, :last_updated)
                """)
                connection.execute(stmt, row.to_dict())
            connection.commit()
        print(f"  CompanyInfo DB에 {len(df_companies)}개 종목 정보 저장/갱신 시도 완료.")
    except Exception as e:
        print(f"  CompanyInfo DB 저장 중 오류: {e}")


def update_company_info_from_pykrx(engine, target_date_str=None):
    """
    pykrx를 사용하여 최신 (또는 지정일 기준) 종목 정보를 가져와
    CompanyInfo DB 테이블을 업데이트하고 내부 캐시도 갱신합니다.
    target_date_str: YYYYMMDD 형식. None이면 pykrx가 내부적으로 최근 영업일 사용.
    """
    global STOCK_NAME_TO_CODE_CACHE, STOCK_CODE_TO_NAME_CACHE, STOCK_CODE_TO_MARKET_CACHE
    print(f"CompanyInfo 업데이트 시작 (기준일: {target_date_str if target_date_str else '최근 영업일'})...")

    all_market_tickers = {}
    companies_to_save_db = [] # DB에 새로 저장하거나 업데이트할 목록

    for market_code in ["KOSPI", "KOSDAQ"]:
        try:
            tickers = stock.get_market_ticker_list(date=target_date_str, market=market_code)
            all_market_tickers[market_code] = tickers
            print(f"  {market_code} 시장 티커 {len(tickers)}개 로드 완료.")
            time.sleep(0.5)
        except Exception as e:
            print(f"  {market_code} 시장 티커 로드 중 오류: {e}")
            all_market_tickers[market_code] = []

    for market_type, tickers in all_market_tickers.items():
        for ticker_code in tickers:
            # 이미 캐시에 있거나, DB에서 로드된 정보가 최신이면 API 호출 스킵 가능 (여기서는 단순화)
            try:
                company_name = stock.get_market_ticker_name(ticker_code)
                if company_name:
                    current_time = datetime.now()
                    # 캐시 업데이트
                    STOCK_NAME_TO_CODE_CACHE[company_name] = ticker_code
                    STOCK_CODE_TO_NAME_CACHE[ticker_code] = company_name
                    STOCK_CODE_TO_MARKET_CACHE[ticker_code] = market_type
                    
                    companies_to_save_db.append({
                        'stock_code': ticker_code,
                        'company_name': company_name,
                        'market_type': market_type,
                        'last_updated': current_time
                    })
                time.sleep(0.02) # API 호출 최소화 (0.05 -> 0.02)
            except Exception:
                pass # 종목명 가져오기 실패 시 무시

    if companies_to_save_db:
        save_company_info_to_db(engine, companies_to_save_db)
    else:
        print("  업데이트할 새로운 종목 정보가 없습니다.")
    
    # DB 업데이트 후 캐시를 DB 기준으로 다시 로드하는 것이 데이터 일관성에 좋음
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
    return STOCK_NAME_TO_CODE_CACHE.get(stock_name)

def get_name_from_ticker(ticker_code):
    return STOCK_CODE_TO_NAME_CACHE.get(ticker_code)

# --- 2. HTS CSV 파싱 및 종목코드 매핑 함수 ---

def parse_hts_csv_and_map_tickers(file_path, engine):
    """
    HTS 조건검색 CSV 파일을 파싱하고, 각 종목명을 종목코드로 매핑합니다.
    매핑에 실패한 종목은 (선택적으로) pykrx로 조회하여 CompanyInfo에 추가 시도.
    결과는 (필터링 날짜, 종목코드, 종목명) 딕셔너리 리스트로 반환합니다.
    """
    file_name = os.path.basename(file_path)
    parsed_data_for_file = [] # 초기화
    filter_date_obj = None # 초기화

    try:
        # 파일명에서 필터링 날짜(금요일) 추출 (예: "2024-05-31_FinancialSafetyFilter.csv")
        filter_date_str = file_name.split('_')[0]
        filter_date_obj = datetime.strptime(filter_date_str, "%Y-%m-%d").date()

        # CSV 파일 읽기
        # - skiprows=2: 상단 2줄 건너뛰기 (사용자 설명 기반)
        # - index_col=False: 첫 번째 열을 인덱스로 사용하지 않음
        # - encoding='cp949': HTS CSV의 일반적인 인코딩
        # - header=None: skiprows 이후 첫 줄을 헤더로 인식하지 않도록 하고, 숫자로 컬럼을 참조
        # - on_bad_lines='skip': 잘못된 줄은 건너뛰도록 처리 (유연성 확보)
        df_raw = pd.read_csv(file_path, skiprows=2, index_col=False, encoding='cp949', header=None, on_bad_lines='skip')
        # 종목명이 있는 컬럼 식별 (사용자 설명: 두 번째 컬럼, pandas에서는 1번 인덱스)
        stock_name_column_index = 1
        
        if df_raw.empty or df_raw.shape[1] <= stock_name_column_index:
            print(f"    [정보] {file_name}: 데이터가 없거나 종목명 컬럼 불충분.")
            return filter_date_obj, []

        raw_stock_names = df_raw.iloc[:, stock_name_column_index]
        stock_names_in_file = raw_stock_names.dropna().astype(str).str.strip().unique().tolist()
        stock_names_in_file = [name for name in stock_names_in_file if name]

        if not stock_names_in_file:
            print(f"    [정보] {file_name}: 유효한 종목명 없음.")
            return filter_date_obj, []
        
        print(f"  > {file_name}: 필터링 날짜: {filter_date_str}, 원본 종목 수: {len(stock_names_in_file)}")
        
        newly_added_companies_for_db = []
        for stock_name in stock_names_in_file:
            ticker_code = get_ticker_from_name(stock_name) # 캐시에서 조회

            if not ticker_code: # 캐시에 없는 경우
                print(f"    [정보] {file_name}: '{stock_name}' 캐시에 없음. pykrx로 조회 시도...")
                try:
                    # 해당 필터링 날짜 기준으로 종목코드 검색 시도
                    # 주의: 이 부분은 API 호출을 유발하므로, 너무 많은 종목이 캐시에 없다면 느려질 수 있음
                    # pykrx의 get_market_ticker_list는 특정 날짜의 "상장 목록"을 주므로,
                    # 정확한 매핑을 위해서는 해당 날짜의 KOSPI/KOSDAQ 전체 목록을 가져와 비교해야 함.
                    # 여기서는 간단하게 가장 최근일 기준의 모든 티커를 가져와 이름을 비교하는 방식을 사용
                    # (initialize_ticker_name_map 가 충분히 최신 정보를 가지고 있다면 이 부분이 덜 필요함)
                    
                    # 임시방편: 만약 캐시에 없다면, 최근일 기준으로 전체 마켓에서 찾아보기 (비효율적)
                    # 더 좋은 방법은 initialize_ticker_name_map이 주기적으로 돌아서 캐시를 최신으로 유지하는 것
                    temp_ticker = None
                    date_for_pykrx_lookup = filter_date_str.replace('-', '') # pykrx용 날짜 형식
                    
                    # 이 루프는 매우 비효율적이므로, 실제 운영에서는 사용하지 않거나,
                    # initialize_ticker_name_map()이 잘 동작하도록 하는 것이 중요.
                    # for market_temp in ["KOSPI", "KOSDAQ"]:
                    #     tickers_temp = stock.get_market_ticker_list(date=date_for_pykrx_lookup, market=market_temp)
                    #     for t_code in tickers_temp:
                    #         if stock.get_market_ticker_name(t_code) == stock_name:
                    #             temp_ticker = t_code
                    #             newly_added_companies_for_db.append({
                    #                 'stock_code': temp_ticker,
                    #                 'company_name': stock_name,
                    #                 'market_type': market_temp, # 정확한 시장 정보
                    #                 'last_updated': datetime.now()
                    #             })
                    #             # 캐시에도 즉시 추가
                    #             STOCK_NAME_TO_CODE_CACHE[stock_name] = temp_ticker
                    #             STOCK_CODE_TO_NAME_CACHE[temp_ticker] = stock_name
                    #             STOCK_CODE_TO_MARKET_CACHE[temp_ticker] = market_temp
                    #             print(f"      - pykrx로 '{stock_name}' -> '{temp_ticker}' 매핑 성공 및 캐시/DB추가 대상 등록.")
                    #             break
                    #         time.sleep(0.01)
                    #     if temp_ticker: break
                    # ticker_code = temp_ticker
                    # 위 로직 대신, 캐시에 없으면 일단 건너뛰고 나중에 일괄 업데이트 하는 것이 나을 수 있음

                    if not ticker_code: # 여전히 못 찾았다면
                         print(f"    [경고] {file_name}: '{stock_name}'의 종목코드를 최종적으로 찾지 못했습니다.")

                except Exception as e_pykrx:
                    print(f"    [오류] {file_name}: pykrx로 '{stock_name}' 조회 중 오류 - {e_pykrx}")
            
            if ticker_code:
                parsed_data_for_file.append({
                    'filter_date': filter_date_obj,
                    'stock_code': ticker_code,
                    'stock_name': stock_name
                })
        
        # 실시간으로 추가된 종목 정보 DB에 저장
        if newly_added_companies_for_db:
            save_company_info_to_db(engine, newly_added_companies_for_db)

        return filter_date_obj, parsed_data_for_file

    except Exception as e_main:
        print(f"    [오류] {file_name}: CSV 파싱 및 매핑 중 주요 예외 발생 - {e_main}")
        return filter_date_obj, [] # 오류 시 빈 리스트 반환 (날짜는 유지 가능하면 유지)

# --- 3. WeeklyFilteredStocks DB 저장 함수 ---

def save_weekly_filtered_stocks_to_db(engine, weekly_data_list):
    """ WeeklyFilteredStocks 테이블에 여러 주간 필터링 종목 정보를 저장 (INSERT OR IGNORE) """
    if not weekly_data_list:
        return

    df_weekly = pd.DataFrame(weekly_data_list)
    # filter_date를 문자열로 변환 (DB에 DATE 타입으로 저장하기 위함)
    df_weekly['filter_date'] = df_weekly['filter_date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)

    try:
        with engine.connect() as connection:
            for _, row in df_weekly.iterrows():
                # company_name은 WeeklyFilteredStocks 테이블 스키마에 따라 포함 여부 결정
                stmt = text("""
                    INSERT OR IGNORE INTO WeeklyFilteredStocks (filter_date, stock_code, company_name)
                    VALUES (:filter_date, :stock_code, :stock_name)
                """)
                connection.execute(stmt, row.to_dict())
            connection.commit()
        print(f"  WeeklyFilteredStocks DB에 {len(df_weekly)}개 필터링 종목 정보 저장/갱신 시도 완료.")
    except Exception as e:
        print(f"  WeeklyFilteredStocks DB 저장 중 오류: {e}")


# --- 4. 메인 실행 로직 ---
if __name__ == "__main__":
    # A. 프로그램 시작 시: CompanyInfo 캐시 초기화
    #    (DB에 데이터가 없다면 먼저 update_company_info_from_pykrx 실행 필요)
    print("="*50)
    print("STEP 1: CompanyInfo 캐시 로딩 (DB에서)")
    print("="*50)
    load_company_info_cache_from_db(ENGINE)

    # 만약 캐시가 비었다면 (최초 실행 등), CompanyInfo DB를 채우는 것을 권장
    if not STOCK_NAME_TO_CODE_CACHE:
        print("\n" + "="*50)
        print("WARNING: CompanyInfo 캐시가 비어있습니다.")
        print("최초 실행 시 또는 DB 업데이트가 필요한 경우, 아래 update_company_info_from_pykrx 함수를 실행하세요.")
        print("이 작업은 시간이 오래 걸릴 수 있으며, 일일 API 호출 제한에 유의해야 합니다.")
        print("주석을 해제하고 실행하거나, 별도의 스크립트로 관리하는 것을 권장합니다.")
        print("="*50)
        # print("\nCompanyInfo DB 및 캐시 업데이트 실행 중... (시간 소요 예상)")
        # update_company_info_from_pykrx(ENGINE, target_date_str=datetime.today().strftime("%Y%m%d")) # 오늘 기준
        # # 업데이트 후 캐시를 다시 로드해야 함
        # load_company_info_cache_from_db(ENGINE)
        # if not STOCK_NAME_TO_CODE_CACHE:
        #     print("CompanyInfo 업데이트 후에도 캐시가 비어있습니다. DB 상태 및 pykrx 호출을 확인하세요.")
        #     # 여기서 프로그램을 종료하거나, 매핑 없이 진행할지 결정
        #     # exit()

    print("\n" + "="*50)
    print(f"STEP 2: 주간 필터링 CSV 파일 파싱 및 종목코드 매핑 시작")
    print(f"대상 폴더: {CONDITION_SEARCH_FILES_FOLDER}")
    print("="*50)
    
    file_pattern = os.path.join(CONDITION_SEARCH_FILES_FOLDER, "*_FinancialSafetyFilter.csv")
    csv_files = sorted(glob.glob(file_pattern))

    if not csv_files:
        print(f"지정된 폴더 '{CONDITION_SEARCH_FILES_FOLDER}'에 '*_FinancialSafetyFilter.csv' 패턴의 파일이 없습니다.")
    
    all_parsed_and_mapped_data = [] # 초기화

    for file_path in csv_files:
        print(f"\n--- 파일 처리: {os.path.basename(file_path)} ---")
        filter_date, mapped_stock_list = parse_hts_csv_and_map_tickers(file_path, ENGINE)

        if mapped_stock_list: # 성공적으로 파싱되고 매핑된 데이터가 있을 경우
            all_parsed_and_mapped_data.extend(mapped_stock_list)
        elif filter_date: # 날짜는 가져왔지만 종목이 없는 경우
            print(f"  > {filter_date}: 해당 날짜에 유효하게 매핑된 종목 없음.")
        else: # 날짜 추출도 실패한 경우
             print(f"  > {os.path.basename(file_path)}: 파일 처리 실패 (날짜 추출 불가).")


    if all_parsed_and_mapped_data:
        # B. 파싱 및 매핑된 결과를 WeeklyFilteredStocks DB 테이블에 저장
        print("\n" + "="*50)
        print(f"STEP 3: 매핑된 주간 필터링 결과를 WeeklyFilteredStocks DB에 저장")
        print("="*50)
        save_weekly_filtered_stocks_to_db(ENGINE, all_parsed_and_mapped_data)

        # C. (선택적) 결과를 CSV 파일로도 저장
        final_df_for_csv = pd.DataFrame(all_parsed_and_mapped_data)
        os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)
        output_csv_path = os.path.join(PROCESSED_DATA_FOLDER, "mapped_weekly_filtered_stocks_FINAL.csv")
        try:
            final_df_for_csv.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"\n매핑된 주간 필터링 종목 데이터가 다음 파일로도 저장되었습니다: {output_csv_path}")
        except Exception as e:
            print(f"\n매핑된 데이터 CSV 저장 중 오류 발생: {e}")
            
        print("\n--- 최종 처리된 주간 필터링 종목 데이터 (DB 저장 내용 샘플) ---")
        print(final_df_for_csv.head())
            
    else:
        print("\n\n--- 최종적으로 처리할 유효한 주간 필터링 종목 데이터가 없습니다. ---")

    print("\n모든 작업 완료.")