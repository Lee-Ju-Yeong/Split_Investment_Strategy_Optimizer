from datetime import datetime
import os
import warnings

# 모든 경고 메시지 억제 (pykrx, pandas 등)
warnings.filterwarnings('ignore')
# 다른 모듈에서 함수 임포트
from .company_info_manager import (
    update_company_info_from_pykrx, # 필요시 CompanyInfo DB 업데이트용
    load_company_info_cache_from_db, # 필수: 캐시 로드
    STOCK_NAME_TO_CODE_CACHE # 캐시 상태 확인용 (선택적)
)
from .weekly_stock_filter_parser import process_all_hts_csv_files
from sqlalchemy import create_engine
from datetime import datetime

# db_setup에서 DB 연결 및 테이블 생성 함수 임포트
from .db_setup import get_db_connection, create_tables 

# 각 기능 모듈 임포트
from . import company_info_manager
from . import weekly_stock_filter_parser
from . import ohlcv_collector
from . import indicator_calculator
from . import filtered_stock_loader

# --- 0. 설정 ---
# HTS CSV 파일들이 저장된 폴더 (사용자 환경에 맞게 설정 필요)
CONDITION_SEARCH_FILES_FOLDER = 'E:/AI/pythonProject/venv/masicsplit/data/raw_data'
# 처리된 CSV 파일 (필터링 결과 + 종목코드) 저장 폴더
PROCESSED_DATA_FOLDER = os.path.join(os.path.dirname(CONDITION_SEARCH_FILES_FOLDER), "processed_data")
# 최종 필터링된 주간 종목 리스트 CSV 파일
FILTERED_STOCKS_CSV_PATH = os.path.join(PROCESSED_DATA_FOLDER, 'mapped_weekly_filtered_stocks_FINAL.csv')
os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True) # 폴더 없으면 생성

# --- 실행 플래그 ---
USE_GPU = False                       # True로 설정 시 지표 계산에 GPU를 사용
UPDATE_COMPANY_INFO_DB = False       # True로 설정 시 CompanyInfo DB를 최신 정보로 업데이트
PROCESS_HTS_CSV_FILES = False      # True로 설정 시 주간 필터링 CSV 파싱 및 DB 저장 실행
LOAD_FILTERED_STOCKS_CSV = True    # True로 설정 시 최종 필터링된 CSV를 DB에 적재/업데이트
COLLECT_OHLCV_DATA = True          # True로 설정 시 OHLCV 데이터 수집 및 DB 저장 실행
FORCE_RECOLLECT_OHLCV = False      # True로 설정 시, 기존 OHLCV 데이터를 모두 지우고 처음부터 다시 수집
CALCULATE_INDICATORS = True        # True로 설정 시 기술적/변동성 지표 계산 및 저장 실행

if __name__ == "__main__":
    db_connection = None  # DB 커넥션 객체 초기화
    try:
        # --- DB 연결 및 테이블 생성 ---
        db_connection = get_db_connection() # MySQL DB 연결
        if db_connection:
            print("MySQL 데이터베이스 연결 성공.")
            create_tables(db_connection) # 필요한 모든 테이블 확인/생성
            print("데이터베이스 테이블 확인/생성 완료.")
        else:
            print("CRITICAL: MySQL 데이터베이스 연결에 실패했습니다. 프로그램을 종료합니다.")
            exit()

        # --- STEP 0: (선택적) CompanyInfo DB 업데이트 ---
        print("\n" + "="*50)
        print("STEP 0: CompanyInfo DB 업데이트 확인 및 실행")
        print("="*50)
        if UPDATE_COMPANY_INFO_DB:
            print("CompanyInfo DB를 최신 정보로 업데이트합니다. (시간 소요 예상)")
            company_info_manager.update_company_info_from_pykrx(db_connection, target_date_str=datetime.today().strftime("%Y%m%d"))
        else:
            print("CompanyInfo DB 업데이트는 건너뜁니다. 기존 DB를 사용합니다.")

        # --- STEP 1: CompanyInfo 캐시 로딩 ---
        print("\n" + "="*50)
        print("STEP 1: CompanyInfo 캐시 로딩 (DB에서)")
        print("="*50)
        company_info_manager.load_company_info_cache_from_db(db_connection)
        if not company_info_manager.STOCK_NAME_TO_CODE_CACHE:
            print("CRITICAL: CompanyInfo 캐시가 비어있거나 로드에 실패했습니다.")
        else:
            print(f"CompanyInfo 캐시 로드 성공. {len(company_info_manager.STOCK_NAME_TO_CODE_CACHE)}개 종목 매핑됨.")

        # --- STEP 2: 주간 필터링 CSV 처리 및 DB 저장 ---
        if PROCESS_HTS_CSV_FILES:
            print("\n" + "="*50)
            print(f"STEP 2: 주간 필터링 CSV 파일 파싱 및 종목코드 매핑, WeeklyFilteredStocks DB 저장 시작")
            print(f"대상 폴더: {CONDITION_SEARCH_FILES_FOLDER}")
            print("="*50)
            weekly_stock_filter_parser.process_all_hts_csv_files(
                conn=db_connection, # MySQL connection 전달
                csv_folder_path=CONDITION_SEARCH_FILES_FOLDER,
                processed_data_folder_path=PROCESSED_DATA_FOLDER,
                company_manager_module=company_info_manager 
            )
            print("주간 필터링 CSV 처리 및 WeeklyFilteredStocks DB 저장 완료.")
        else:
            print("\nSTEP 2: 주간 필터링 CSV 처리는 건너뜁니다.")

        # --- STEP 2.5: 최종 필터링된 주간 종목 리스트 DB에 적재 ---
        if LOAD_FILTERED_STOCKS_CSV:
            print("\n" + "="*50)
            print(f"STEP 2.5: 최종 필터링된 주간 종목 CSV 파일 DB 적재 시작")
            print(f"대상 파일: {FILTERED_STOCKS_CSV_PATH}")
            print("="*50)
            db_engine = filtered_stock_loader.get_db_engine()
            filtered_stock_loader.load_filtered_stocks_to_db(FILTERED_STOCKS_CSV_PATH, db_engine)
        else:
            print("\nSTEP 2.5: 최종 필터링 CSV DB 적재는 건너뜁니다.")

        # --- STEP 3: OHLCV 데이터 수집 및 DB 저장 ---
        if COLLECT_OHLCV_DATA:
            print("\n" + "="*50)
            print("STEP 3: OHLCV 데이터 수집 및 DailyStockPrice DB 저장 시작")
            print("="*50)
            ohlcv_collector.collect_and_save_ohlcv_for_filtered_stocks(
                conn=db_connection, # MySQL connection 전달
                company_manager=company_info_manager,
                overall_end_date_str=datetime.today().strftime("%Y%m%d"),
                force_recollect=FORCE_RECOLLECT_OHLCV # 재수집 플래그 전달
            )
            print("OHLCV 데이터 수집 및 DailyStockPrice DB 저장 완료.")
        else:
            print("\nSTEP 3: OHLCV 데이터 수집은 건너뜁니다.")

        # --- STEP 4: 기술적/변동성 지표 계산 및 저장 ---
        if CALCULATE_INDICATORS:
            indicator_calculator.calculate_and_store_indicators_for_all(db_connection, use_gpu=USE_GPU)
        else:
            print("\nSTEP 4: 기술적/변동성 지표 계산은 건너뜁니다.")

        print("\n모든 데이터 파이프라인 작업 완료.")

    except Exception as e:
        print(f"\n[메인 스크립트 오류] 예외 발생: {e}")
        import traceback
        traceback.print_exc() # 상세 오류 스택 출력

    finally:
        if db_connection:
            db_connection.close()
            print("\nMySQL 데이터베이스 연결 해제됨.")