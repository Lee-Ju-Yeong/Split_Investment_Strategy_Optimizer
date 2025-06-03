import pandas as pd
import os
import glob
from datetime import datetime
from sqlalchemy import create_engine, text # ENGINE은 main_script에서 주입받거나 여기서 생성

# company_info_manager.py 에서 필요한 함수만 임포트
from company_info_manager import get_ticker_from_name # 캐시 로드는 main_script.py에서 수행

# --- 설정 (이 파일에서 직접 실행할 경우를 대비, main_script.py에서 주입받는 것이 더 좋음) ---
# DB_FILE_PATH = "stock_backtesting.db"
# ENGINE = create_engine(f'sqlite:///{DB_FILE_PATH}') # main_script.py에서 ENGINE을 넘겨받도록 수정 권장

# --- HTS CSV 파싱 및 종목코드 매핑 함수 ---
def parse_single_hts_csv_file(file_path):
    file_name = os.path.basename(file_path)
    parsed_data_for_file = []
    filter_date_obj = None

    try:
        filter_date_str = file_name.split('_')[0]
        filter_date_obj = datetime.strptime(filter_date_str, "%Y-%m-%d").date()

        df_raw = pd.read_csv(file_path, skiprows=2, index_col=False, encoding='cp949', header=None, on_bad_lines='skip')
        stock_name_column_index = 1
        
        if df_raw.empty or df_raw.shape[1] <= stock_name_column_index:
            # print(f"    [정보] {file_name}: 데이터가 없거나 종목명 컬럼 불충분.") # 로깅은 main_script 레벨에서 하거나 logging 모듈 사용
            return filter_date_obj, []

        raw_stock_names = df_raw.iloc[:, stock_name_column_index]
        stock_names_in_file = raw_stock_names.dropna().astype(str).str.strip().unique().tolist()
        stock_names_in_file = [name for name in stock_names_in_file if name]

        if not stock_names_in_file:
            # print(f"    [정보] {file_name}: 유효한 종목명 없음.")
            return filter_date_obj, []
        
        successful_mappings = 0
        for stock_name in stock_names_in_file:
            ticker_code = get_ticker_from_name(stock_name) # 임포트된 함수 사용
            
            if ticker_code:
                parsed_data_for_file.append({
                    'filter_date': filter_date_obj,
                    'stock_code': ticker_code,
                    'stock_name': stock_name
                })
                successful_mappings += 1
            # else: 매핑 실패 시 로깅은 호출하는 쪽에서 담당하거나, 여기서 간단히 할 수 있음
        
        # 매핑 성공/실패에 대한 로깅은 여기서 간단히 하거나, 반환값에 포함시켜 호출부에서 처리
        if successful_mappings < len(stock_names_in_file) and len(stock_names_in_file) > 0:
            print(f"    [주의] {file_name}: {len(stock_names_in_file)}개 중 {successful_mappings}개 매핑됨.")
        elif successful_mappings == 0 and len(stock_names_in_file) > 0:
            print(f"    [경고] {file_name}: 매핑된 종목 없음. CompanyInfo 캐시 확인 필요.")

        return filter_date_obj, parsed_data_for_file

    except Exception as e_main:
        print(f"    [오류] {file_name}: CSV 파싱 및 매핑 중 주요 예외 발생 - {e_main}")
        return filter_date_obj, [] # filter_date_obj는 유지하여 어떤 파일에서 오류났는지 알 수 있게

# --- WeeklyFilteredStocks DB 저장 함수 ---
def save_weekly_filtered_stocks_to_db(engine, weekly_data_list):
    if not weekly_data_list:
        return

    df_weekly = pd.DataFrame(weekly_data_list)
    df_weekly['filter_date'] = df_weekly['filter_date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)

    try:
        with engine.connect() as connection:
            for _, row in df_weekly.iterrows():
                stmt = text("""
                    INSERT OR IGNORE INTO WeeklyFilteredStocks (filter_date, stock_code, company_name)
                    VALUES (:filter_date, :stock_code, :stock_name)
                """)
                connection.execute(stmt, row.to_dict())
            connection.commit()
        print(f"  WeeklyFilteredStocks DB에 {len(df_weekly)}개 필터링 종목 정보 저장/갱신 시도 완료.")
    except Exception as e:
        print(f"  WeeklyFilteredStocks DB 저장 중 오류: {e}")

# 이 파일이 직접 실행될 때의 로직 (또는 main_script.py에서 호출될 함수)
def process_all_hts_csv_files(engine, csv_folder_path, processed_data_folder_path=None): # processed_data_folder_path는 선택적
    file_pattern = os.path.join(csv_folder_path, "*_FinancialSafetyFilter.csv")
    csv_files = sorted(glob.glob(file_pattern))

    if not csv_files:
        print(f"지정된 폴더 '{csv_folder_path}'에 '*_FinancialSafetyFilter.csv' 패턴의 파일이 없습니다.")
        return
    
    all_parsed_and_mapped_data = []

    for file_path in csv_files:
        # print(f"\n--- HTS CSV 파일 처리: {os.path.basename(file_path)} ---") # 상세 로깅은 main_script에서
        filter_date, mapped_stock_list = parse_single_hts_csv_file(file_path)

        if mapped_stock_list: # 데이터가 있을 때만 추가
            all_parsed_and_mapped_data.extend(mapped_stock_list)
        # filter_date만 있고 mapped_stock_list가 비어있는 경우는 이미 parse_single_hts_csv_file에서 로깅됨

    if all_parsed_and_mapped_data:
        print(f"\n총 {len(all_parsed_and_mapped_data)}개의 매핑된 주간 필터링 결과를 WeeklyFilteredStocks DB에 저장합니다...")
        save_weekly_filtered_stocks_to_db(engine, all_parsed_and_mapped_data)

        if processed_data_folder_path: # CSV 저장 경로가 주어졌을 때만 저장
            final_df_for_csv = pd.DataFrame(all_parsed_and_mapped_data)
            os.makedirs(processed_data_folder_path, exist_ok=True)
            output_csv_path = os.path.join(processed_data_folder_path, "mapped_weekly_filtered_stocks_FINAL.csv")
            try:
                final_df_for_csv.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
                print(f"매핑된 주간 필터링 종목 데이터가 다음 파일로도 저장되었습니다: {output_csv_path}")
            except Exception as e:
                print(f"매핑된 데이터 CSV 저장 중 오류 발생: {e}")
    else:
        print("\n최종적으로 처리할 유효한 주간 필터링 종목 데이터가 없습니다.")

# --- hts_csv_parser.py의 if __name__ == "__main__": 블록은 main_script.py로 이동 또는 삭제 ---
# 만약 이 파일을 단독 테스트하고 싶다면, 임시로 ENGINE을 여기서 생성하고
# company_info_manager의 load_company_info_cache_from_db를 먼저 호출해야 함.
# 예:
# if __name__ == "__main__":
#     from company_info_manager import load_company_info_cache_from_db
#     temp_engine = create_engine(f'sqlite:///stock_backtesting.db')
#     load_company_info_cache_from_db(temp_engine) # 캐시 로드
#     if STOCK_NAME_TO_CODE_CACHE: # company_info_manager의 전역 변수
#          process_all_hts_csv_files(temp_engine, 'E:/AI/pythonProject/venv/masicsplit/data/raw_data', 'E:/AI/pythonProject/venv/masicsplit/data/processed_data')
#     else:
#          print("테스트 실행 실패: CompanyInfo 캐시 로드 필요.")