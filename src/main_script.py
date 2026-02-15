"""
main_script.py

This module contains the main script for running the data pipeline for the Magic Split Strategy.
"""

import warnings
from datetime import datetime

# 모든 경고 메시지 억제 (pykrx, pandas 등)
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/main_script.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

# db_setup에서 DB 연결 및 테이블 생성 함수 임포트
from .db_setup import get_db_connection, create_tables 

# 각 기능 모듈 임포트
from . import company_info_manager
from . import weekly_stock_filter_parser
from . import ohlcv_collector
from . import indicator_calculator
from . import filtered_stock_loader
from .config_loader import load_config, resolve_project_path

if __name__ == "__main__":
    db_connection = None  # DB 커넥션 객체 초기화
    try:
        config = load_config()
        pipeline = config.get("data_pipeline") or {}
        pipeline_paths = pipeline.get("paths") or {}
        pipeline_flags = pipeline.get("flags") or {}

        # --- 0. 설정 (config.yaml 기반) ---
        condition_search_files_folder_str = pipeline_paths.get("condition_search_files_folder")
        if not condition_search_files_folder_str:
            raise KeyError("config.yaml에 `data_pipeline.paths.condition_search_files_folder`가 필요합니다.")

        condition_search_files_folder = resolve_project_path(condition_search_files_folder_str)

        processed_data_folder_str = pipeline_paths.get("processed_data_folder")
        processed_data_folder = (
            resolve_project_path(processed_data_folder_str)
            if processed_data_folder_str
            else condition_search_files_folder.parent / "processed_data"
        )

        filtered_stocks_csv_path_str = pipeline_paths.get("filtered_stocks_csv_path")
        filtered_stocks_csv_path = (
            resolve_project_path(filtered_stocks_csv_path_str)
            if filtered_stocks_csv_path_str
            else processed_data_folder / "mapped_weekly_filtered_stocks_FINAL.csv"
        )

        processed_data_folder.mkdir(parents=True, exist_ok=True)

        # --- 실행 플래그 (config.yaml 기반) ---
        use_gpu = bool(pipeline_flags.get("use_gpu", False))
        update_company_info_db = bool(pipeline_flags.get("update_company_info_db", False))
        process_hts_csv_files = bool(pipeline_flags.get("process_hts_csv_files", False))
        load_filtered_stocks_csv = bool(pipeline_flags.get("load_filtered_stocks_csv", True))
        collect_ohlcv_data = bool(pipeline_flags.get("collect_ohlcv_data", True))
        force_recollect_ohlcv = bool(pipeline_flags.get("force_recollect_ohlcv", False))
        calculate_indicators = bool(pipeline_flags.get("calculate_indicators", True))

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
        if update_company_info_db:
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
        if process_hts_csv_files:
            print("\n" + "="*50)
            print(f"STEP 2: 주간 필터링 CSV 파일 파싱 및 종목코드 매핑, WeeklyFilteredStocks DB 저장 시작")
            print(f"대상 폴더: {condition_search_files_folder}")
            print("="*50)
            weekly_stock_filter_parser.process_all_hts_csv_files(
                conn=db_connection, # MySQL connection 전달
                csv_folder_path=str(condition_search_files_folder),
                processed_data_folder_path=str(processed_data_folder),
                company_manager_module=company_info_manager 
            )
            print("주간 필터링 CSV 처리 및 WeeklyFilteredStocks DB 저장 완료.")
        else:
            print("\nSTEP 2: 주간 필터링 CSV 처리는 건너뜁니다.")

        # --- STEP 2.5: 최종 필터링된 주간 종목 리스트 DB에 적재 ---
        if load_filtered_stocks_csv:
            print("\n" + "="*50)
            print(f"STEP 2.5: 최종 필터링된 주간 종목 CSV 파일 DB 적재 시작")
            print(f"대상 파일: {filtered_stocks_csv_path}")
            print("="*50)
            db_engine = filtered_stock_loader.get_db_engine(config.get("database"))
            filtered_stock_loader.load_filtered_stocks_to_db(str(filtered_stocks_csv_path), db_engine)
        else:
            print("\nSTEP 2.5: 최종 필터링 CSV DB 적재는 건너뜁니다.")

        # --- STEP 3: OHLCV 데이터 수집 및 DB 저장 ---
        if collect_ohlcv_data:
            print("\n" + "="*50)
            print("STEP 3: OHLCV 데이터 수집 및 DailyStockPrice DB 저장 시작")
            print("="*50)
            ohlcv_collector.collect_and_save_ohlcv_for_filtered_stocks(
                conn=db_connection, # MySQL connection 전달
                company_manager=company_info_manager,
                overall_end_date_str=datetime.today().strftime("%Y%m%d"),
                force_recollect=force_recollect_ohlcv # 재수집 플래그 전달
            )
            print("OHLCV 데이터 수집 및 DailyStockPrice DB 저장 완료.")
        else:
            print("\nSTEP 3: OHLCV 데이터 수집은 건너뜁니다.")

        # --- STEP 4: 기술적/변동성 지표 계산 및 저장 ---
        if calculate_indicators:
            indicator_calculator.calculate_and_store_indicators_for_all(db_connection, use_gpu=use_gpu)
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
