from sqlalchemy import create_engine
from datetime import datetime
import os
# 다른 모듈에서 함수 임포트
from company_info_manager import (
    update_company_info_from_pykrx, # 필요시 CompanyInfo DB 업데이트용
    load_company_info_cache_from_db, # 필수: 캐시 로드
    STOCK_NAME_TO_CODE_CACHE # 캐시 상태 확인용 (선택적)
)
from weekly_stock_filter_parser import process_all_hts_csv_files
from sqlalchemy import create_engine
from datetime import datetime

import company_info_manager
import weekly_stock_filter_parser
import ohlcv_collector 


# --- 0. 설정 ---
DB_FILE_PATH = "stock_backtesting.db" # 실제 DB 파일 경로
ENGINE = create_engine(f'sqlite:///{DB_FILE_PATH}')
# HTS CSV 파일들이 저장된 폴더
CONDITION_SEARCH_FILES_FOLDER = 'E:/AI/pythonProject/venv/masicsplit/data/raw_data' # 실제 경로로 수정
# 처리된 CSV 파일 (필터링 결과 + 종목코드) 저장 폴더 (hts_parser가 사용)
PROCESSED_DATA_FOLDER = os.path.join(os.path.dirname(CONDITION_SEARCH_FILES_FOLDER), "processed_data")
os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True) # 폴더 없으면 생성

UPDATE_COMPANY_INFO_DB = False  # CompanyInfo DB를 업데이트할지 여부
COLLECT_OHLCV_DATA = True       # OHLCV 데이터를 수집할지 여부 (새로 추가)

if __name__ == "__main__":
    print("="*50)
    print("STEP 0: (선택적) CompanyInfo DB 업데이트 확인 및 실행")
    print("="*50)
    # 예: 마지막 업데이트 날짜를 DB나 파일에 기록해두고, 일정 기간이 지났으면 업데이트 실행
    # 또는, 스크립트 실행 시 인자로 업데이트 여부를 받거나, 설정 파일에서 읽어오기
    # 여기서는 수동으로 주석을 해제하여 업데이트한다고 가정
    if UPDATE_COMPANY_INFO_DB:
        print("CompanyInfo DB를 최신 정보로 업데이트합니다. (시간 소요 예상)")
        # CompanyInfo 테이블이 없다면 생성하는 로직 필요 (company_info_manager 또는 여기서)
        company_info_manager.update_company_info_from_pykrx(ENGINE, target_date_str=datetime.today().strftime("%Y%m%d"))
    else:
        print("CompanyInfo DB 업데이트는 건너뜁니다. 기존 DB를 사용합니다.")


    print("\n" + "="*50)
    print("STEP 1: CompanyInfo 캐시 로딩 (DB에서)")
    print("="*50)
    load_company_info_cache_from_db(ENGINE)

    if not company_info_manager.STOCK_NAME_TO_CODE_CACHE: # 직접 참조
        print("CRITICAL: CompanyInfo 캐시가 비어있거나 로드에 실패했습니다. 프로그램 실행에 문제가 있을 수 있습니다.")
        # 강력한 경고 또는 프로그램 중단 고려
        # exit("CompanyInfo 캐시 로드 실패. 프로그램 중단.")
    else:
        print(f"CompanyInfo 캐시 로드 성공. {len(company_info_manager.STOCK_NAME_TO_CODE_CACHE)}개 종목 매핑됨.")

    print("\n" + "="*50)
    print(f"STEP 2: 주간 필터링 CSV 파일 파싱 및 종목코드 매핑, WeeklyFilteredStocks DB 저장 시작")
    print(f"대상 폴더: {CONDITION_SEARCH_FILES_FOLDER}")
    print("="*50)
    # weekly_stock_filter_parser 모듈의 함수명에 맞게 호출
    # 예: weekly_stock_filter_parser.process_all_hts_csv_files(ENGINE, CONDITION_SEARCH_FILES_FOLDER, PROCESSED_DATA_FOLDER, company_info_manager)
    # 위 함수는 company_info_manager를 내부적으로 사용하여 종목코드 매핑을 해야 함
    weekly_stock_filter_parser.process_all_hts_csv_files(
        engine=ENGINE,
        csv_folder_path=CONDITION_SEARCH_FILES_FOLDER,
        processed_data_folder_path=PROCESSED_DATA_FOLDER, # 이 파라미터가 필요없다면 제거
        company_manager_module=company_info_manager # 종목코드 매핑에 사용
    )
    print("주간 필터링 CSV 처리 및 WeeklyFilteredStocks DB 저장 완료.")


    # --- 새로운 STEP 추가 ---
    if COLLECT_OHLCV_DATA:
        ohlcv_collector.collect_and_save_ohlcv_for_filtered_stocks(
            engine=ENGINE,
            company_manager=company_info_manager, # 종목명 로깅 등에 사용
            overall_end_date_str=datetime.today().strftime("%Y%m%d") # 오늘까지 수집
        )
    else:
        print("\nOHLCV 데이터 수집은 건너뜁니다.")
    # ----------------------

    print("\n모든 데이터 파이프라인 작업 완료.")