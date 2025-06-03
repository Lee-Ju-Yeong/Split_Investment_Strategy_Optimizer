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

# --- 0. 설정 ---
DB_FILE_PATH = "stock_backtesting.db"
ENGINE = create_engine(f'sqlite:///{DB_FILE_PATH}')
CONDITION_SEARCH_FILES_FOLDER = 'E:/AI/pythonProject/venv/masicsplit/data/raw_data'
PROCESSED_DATA_FOLDER = os.path.join(os.path.dirname(CONDITION_SEARCH_FILES_FOLDER), "processed_data")
UPDATE_COMPANY_INFO_DB = False

if __name__ == "__main__":
    print("="*50)
    print("STEP 0: (선택적) CompanyInfo DB 업데이트 확인 및 실행")
    print("="*50)
    # 예: 마지막 업데이트 날짜를 DB나 파일에 기록해두고, 일정 기간이 지났으면 업데이트 실행
    # 또는, 스크립트 실행 시 인자로 업데이트 여부를 받거나, 설정 파일에서 읽어오기
    # 여기서는 수동으로 주석을 해제하여 업데이트한다고 가정
    if UPDATE_COMPANY_INFO_DB: # UPDATE_COMPANY_INFO_DB = True 또는 False
        print("CompanyInfo DB를 최신 정보로 업데이트합니다. (시간 소요 예상)")
        update_company_info_from_pykrx(ENGINE, target_date_str=datetime.today().strftime("%Y%m%d"))
    else:
        print("CompanyInfo DB 업데이트는 건너뜁니다. 기존 DB를 사용합니다.")


    print("\n" + "="*50)
    print("STEP 1: CompanyInfo 캐시 로딩 (DB에서)")
    print("="*50)
    load_company_info_cache_from_db(ENGINE)

    if not STOCK_NAME_TO_CODE_CACHE:
        print("WARNING: CompanyInfo 캐시가 비어있거나 로드에 실패했습니다.")
        print("         CompanyInfo DB가 비어있을 수 있습니다. `company_info_manager.py`를 직접 실행하여 DB를 채우거나,")
        print("         위 STEP 0의 업데이트 로직을 활성화하여 실행해보세요.")
        # 여기서 프로그램을 중단하거나, 매핑 없이 진행할 수 없으므로 강력한 경고 또는 종료 필요
        # exit("CompanyInfo 캐시 로드 실패. 프로그램 중단.")
    else:
        print(f"CompanyInfo 캐시 로드 성공. {len(STOCK_NAME_TO_CODE_CACHE)}개 종목 매핑됨.")


    print("\n" + "="*50)
    print(f"STEP 2: 주간 필터링 CSV 파일 파싱 및 종목코드 매핑, DB 저장 시작")
    print(f"대상 폴더: {CONDITION_SEARCH_FILES_FOLDER}")
    print("="*50)
    process_all_hts_csv_files(ENGINE, CONDITION_SEARCH_FILES_FOLDER, PROCESSED_DATA_FOLDER)

    print("\n모든 작업 완료.")