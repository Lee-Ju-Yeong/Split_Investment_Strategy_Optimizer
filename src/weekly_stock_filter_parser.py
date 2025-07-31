# weekly_stock_filter_parser.py
import pandas as pd
import os
import glob
from datetime import datetime
# from sqlalchemy import text # SQLAlchemy 의존성 제거

# company_info_manager.py 에서 필요한 함수/변수 임포트
# 캐시는 company_info_manager 모듈 내의 전역 변수로 관리되며,
# main_script.py에서 load_company_info_cache_from_db()를 호출하여 미리 채워짐.
from .company_info_manager import get_ticker_from_name

def parse_single_hts_csv_file(file_path):
    """
    단일 HTS CSV 파일을 파싱하여 (필터링 날짜, [매핑된 종목 정보 딕셔너리 리스트])를 반환합니다.
    매핑된 종목 정보: {'filter_date': ..., 'stock_code': ..., 'stock_name': ...}
    """
    file_name = os.path.basename(file_path)
    parsed_data_for_file = []
    filter_date_obj = None
    unmapped_stock_names = [] # 매핑 실패한 종목명 기록

    try:
        # 파일명에서 필터링 날짜 추출 (YYYY-MM-DD 형식으로 가정)
        filter_date_str = file_name.split('_')[0]
        filter_date_obj = datetime.strptime(filter_date_str, "%Y-%m-%d").date()

        # CSV 파일 읽기:
        # - 첫 번째 줄은 비어있고, 두 번째 줄은 헤더 설명, 실제 데이터는 세 번째 줄부터.
        # - 종목명은 B열 (0부터 시작하는 인덱스로는 1).
        # - 빈 줄이나 NaN 값을 가진 행은 건너뜀.
        # - on_bad_lines='skip'은 잘못된 형식의 줄을 만났을 때 무시 (경고 발생 가능).
        # - usecols=[1] 로 종목명 컬럼만 읽어오거나, header=None 후 iloc으로 접근
        try:
            # skiprows를 사용하여 실제 데이터가 시작되는 행 이전은 건너뜀
            # 예시 CSV 파일은 3번째 행부터 실제 종목명 데이터 시작 (0-indexed로는 2)
            # 첫 번째 컬럼은 비어있고 두 번째 컬럼(인덱스 1)에 종목명
            df_raw = pd.read_csv(file_path, skiprows=2, header=None, usecols=[1],
                                 encoding='cp949', on_bad_lines='warn') # 'warn'으로 변경하여 문제 인지
        except pd.errors.EmptyDataError:
            # print(f"    [정보] {file_name}: 파일이 비어있습니다.") # 로깅은 호출부에서
            return filter_date_obj, [], [] # 날짜, 성공 리스트, 실패 리스트
        except Exception as e_read:
            print(f"    [오류] {file_name}: CSV 파일 읽기 중 오류 발생 - {e_read}")
            return filter_date_obj, [], []

        if df_raw.empty or df_raw.iloc[:, 0].isnull().all():
            # print(f"    [정보] {file_name}: 유효한 데이터가 없습니다.")
            return filter_date_obj, [], []

        # 첫 번째 (유일한) 컬럼에서 종목명 추출 및 전처리
        stock_names_in_file = df_raw.iloc[:, 0].dropna().astype(str).str.strip().unique().tolist()
        stock_names_in_file = [name for name in stock_names_in_file if name] # 빈 문자열 제거

        if not stock_names_in_file:
            # print(f"    [정보] {file_name}: 파일에서 유효한 종목명을 찾을 수 없습니다.")
            return filter_date_obj, [], []

        for stock_name in stock_names_in_file:
            ticker_code = get_ticker_from_name(stock_name) # company_info_manager의 캐시 사용

            if ticker_code:
                parsed_data_for_file.append({
                    'filter_date': filter_date_obj,
                    'stock_code': ticker_code,
                    'stock_name': stock_name # 원본 종목명 저장
                })
            else:
                unmapped_stock_names.append(stock_name)
        
        return filter_date_obj, parsed_data_for_file, unmapped_stock_names

    except ValueError as e_date: # 날짜 파싱 오류 등
        print(f"    [오류] {file_name}: 파일명에서 날짜 파싱 중 오류 - {e_date}")
        return None, [], [] # 날짜 객체 생성 실패 시 None 반환
    except Exception as e_main:
        print(f"    [오류] {file_name}: CSV 파싱 및 매핑 중 주요 예외 발생 - {e_main}")
        # filter_date_obj가 설정되었을 수도, 안되었을 수도 있음
        return filter_date_obj if 'filter_date_obj' in locals() else None, [], []


def save_weekly_filtered_stocks_to_db(conn, weekly_data_list):
    """
    파싱 및 매핑된 주간 필터링 종목 정보 리스트를 DB에 저장합니다.
    weekly_data_list: [{'filter_date': datetime.date, 'stock_code': str, 'stock_name': str}, ...]
    conn: pymysql connection 객체
    """
    if not weekly_data_list:
        return 0 # 저장된 행 수 반환

    saved_rows = 0
    try:
        with conn.cursor() as cur:
            for data_dict in weekly_data_list:
                sql = """
                    INSERT IGNORE INTO WeeklyFilteredStocks (filter_date, stock_code, company_name)
                    VALUES (%s, %s, %s)
                """
                
                # 날짜 객체를 MySQL에 적합한 문자열로 변환 (YYYY-MM-DD)
                filter_date_str = data_dict['filter_date'].strftime('%Y-%m-%d')

                values = (
                    filter_date_str,
                    data_dict['stock_code'],
                    data_dict['stock_name'] # DB의 company_name 컬럼에 해당
                )
                cur.execute(sql, values)
                saved_rows += cur.rowcount # INSERT IGNORE시 실제 삽입된 행 수
            conn.commit()
        print(f"  WeeklyFilteredStocks DB에 총 {len(weekly_data_list)}건 중 {saved_rows}건 신규 저장 완료.")
        return saved_rows
    except Exception as e:
        print(f"  [오류] WeeklyFilteredStocks DB 저장 중: {e}")
        conn.rollback()
        return 0


def process_all_hts_csv_files(conn, csv_folder_path, processed_data_folder_path=None, company_manager_module=None):
    """
    지정된 폴더 내의 모든 HTS 조건검색 CSV 파일을 일괄 처리하여,
    파싱 및 종목코드 매핑 후 WeeklyFilteredStocks DB에 저장합니다.
    conn: pymysql connection 객체
    company_manager_module: 현재는 get_ticker_from_name을 직접 임포트하여 사용하므로 이 파라미터는 사용되지 않음.
    """
    # `company_manager_module` 파라미터는 현재 구현에서는 직접 사용되지 않지만,
    # 만약 `get_ticker_from_name`이 `company_manager_module.get_ticker_from_name` 형태로
    # 호출되어야 한다면 필요합니다. 현재는 `from .company_info_manager import get_ticker_from_name`
    # 으로 직접 임포트하여 전역 캐시를 사용하고 있으므로, 이 파라미터는 로깅이나
    # 다른 목적으로 전달될 수 있습니다. (또는 제거 가능)

    file_pattern = os.path.join(csv_folder_path, "*_FinancialSafetyFilter.csv")
    csv_files = sorted(glob.glob(file_pattern)) # 날짜순 정렬 효과

    if not csv_files:
        print(f"  [정보] 지정된 폴더 '{csv_folder_path}'에 '{os.path.basename(file_pattern)}' 패턴의 파일이 없습니다.")
        return

    all_parsed_and_mapped_data = []
    total_files_processed = 0
    total_stocks_found_in_files = 0
    total_stocks_mapped = 0
    all_unmapped_details = {} # {filename: [unmapped_stock_names]}

    print(f"  총 {len(csv_files)}개의 HTS CSV 파일 처리 시작...")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        total_files_processed += 1
        # print(f"    - 파일 처리 중: {file_name}") # 상세 로깅은 필요시 활성화

        filter_date, mapped_stock_list, unmapped_names = parse_single_hts_csv_file(file_path)

        if filter_date is None and not mapped_stock_list and not unmapped_names:
            print(f"    [경고] {file_name}: 파일 처리 중 심각한 오류 발생 (날짜 정보 없음). 건너뜁니다.")
            continue

        total_stocks_found_in_files += (len(mapped_stock_list) + len(unmapped_names))
        total_stocks_mapped += len(mapped_stock_list)

        if mapped_stock_list:
            all_parsed_and_mapped_data.extend(mapped_stock_list)
        
        if unmapped_names:
            all_unmapped_details[file_name] = unmapped_names
            # print(f"    [주의] {file_name}: 다음 종목명 매핑 실패 - {unmapped_names}") # 상세 로깅

    print(f"  HTS CSV 파일 처리 완료: {total_files_processed}개 파일 처리됨.")
    print(f"    - 파일에서 발견된 총 종목명(유니크) 수 합계: {total_stocks_found_in_files}") # 실제로는 파일 내 unique 수의 합계
    print(f"    - 성공적으로 종목코드로 매핑된 항목 수: {total_stocks_mapped}")

    if all_unmapped_details:
        print("    --- 매핑 실패 상세 ---")
        for fname, names in all_unmapped_details.items():
            print(f"      {fname}: {', '.join(names)}")
        print("    --------------------")
        print("    [조치 필요] 위 종목들은 CompanyInfo DB에 없거나, 종목명이 일치하지 않을 수 있습니다.")
        print("              `company_info_manager.update_company_info_from_pykrx` 실행 및 캐시 재로드를 고려하세요.")


    if all_parsed_and_mapped_data:
        print(f"\n  총 {len(all_parsed_and_mapped_data)}개의 매핑된 주간 필터링 결과를 WeeklyFilteredStocks DB에 저장합니다...")
        saved_count_in_db = save_weekly_filtered_stocks_to_db(conn, all_parsed_and_mapped_data)

        if processed_data_folder_path: # CSV 저장 경로가 주어졌을 때만 저장
            final_df_for_csv = pd.DataFrame(all_parsed_and_mapped_data)
            # filter_date를 문자열로 변환하여 CSV 저장 (DB에는 DATE 타입으로 저장됨)
            final_df_for_csv['filter_date'] = final_df_for_csv['filter_date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)
            os.makedirs(processed_data_folder_path, exist_ok=True)
            output_csv_path = os.path.join(processed_data_folder_path, "mapped_weekly_filtered_stocks_FINAL.csv")
            try:
                final_df_for_csv.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
                print(f"  매핑된 주간 필터링 종목 데이터가 다음 파일로도 저장되었습니다: {output_csv_path}")
            except Exception as e_csv_save:
                print(f"  [오류] 매핑된 데이터 CSV 저장 중: {e_csv_save}")
    else:
        print("\n  최종적으로 WeeklyFilteredStocks DB에 저장할 유효한 주간 필터링 종목 데이터가 없습니다.")