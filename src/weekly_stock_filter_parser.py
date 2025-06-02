import pandas as pd
import os
import glob
from datetime import datetime

# CSV 파일이 저장된 폴더 (사용자가 제공한 경로)
# UI에서 @raw_data로 전달된 경로는 /e:/AI/pythonProject/venv/masicsplit/data/raw_data 입니다.
# Windows Python 환경에서 올바르게 인식하도록 경로 수정
CONDITION_SEARCH_FILES_FOLDER = 'E:/AI/pythonProject/venv/masicsplit/data/raw_data'

def parse_hts_condition_csv(file_path):
    """
    HTS 조건검색 결과 CSV 파일을 파싱하여 (필터링 날짜, 종목명 리스트)를 반환합니다.
    제공된 CSV 예시 파일 구조 및 사용자 설명에 맞춰져 있습니다.
    """
    file_name = os.path.basename(file_path)
    filter_date_obj = None # 초기화
    stock_names = [] # 초기화

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
        df = pd.read_csv(file_path, skiprows=2, index_col=False, encoding='cp949', header=None, on_bad_lines='skip')

        # 종목명이 있는 컬럼 식별 (사용자 설명: 두 번째 컬럼, pandas에서는 1번 인덱스)
        stock_name_column_index = 1
        
        if df.empty or df.shape[1] <= stock_name_column_index:
            print(f"    [정보] 파일에 데이터가 없거나 종목명 컬럼이 충분하지 않습니다: {file_name}")
            return filter_date_obj, []

        # 종목명 추출 및 정제
        raw_stock_names = df.iloc[:, stock_name_column_index]
        stock_names = raw_stock_names.dropna().astype(str).str.strip().unique().tolist()
        
        # 비어있는 문자열 최종 제거
        stock_names = [name for name in stock_names if name]

        if not stock_names:
            print(f"    [정보] 유효한 종목명이 없습니다: {file_name}")
        
        # 종목코드 추출 로직 (주석 처리 - 필요시 컬럼명/인덱스 확인 후 활성화)
        # stock_codes = []
        # stock_code_column_index = 0 # 예시: 종목코드가 첫 번째 컬럼에 있다면
        # if df.shape[1] > stock_code_column_index:
        #     raw_stock_codes = df.iloc[:, stock_code_column_index]
        #     stock_codes = raw_stock_codes.dropna().astype(str).str.strip().unique().tolist()
        #     stock_codes = [code for code in stock_codes if code] # 비어있는 코드 제거
        #     if not stock_codes:
        #         print(f"    [정보] 유효한 종목코드가 없습니다: {file_name}")
        # else:
        #     print(f"    [정보] 종목코드 컬럼이 충분하지 않습니다: {file_name}")
        #
        # # 여기서 stock_names와 stock_codes를 필요에 따라 함께 반환하거나 구조화할 수 있습니다.
        # # 예: return filter_date_obj, list(zip(stock_names, stock_codes)) 또는 [{'name': n, 'code': c}, ...]


    except pd.errors.EmptyDataError:
        print(f"    [정보] 빈 파일입니다 (EmptyDataError): {file_name}")
    except FileNotFoundError:
        print(f"    [오류] 파일을 찾을 수 없습니다: {file_name}")
        return None, []
    except ValueError as ve:
        print(f"    [오류] 파일명에서 날짜를 추출하거나 데이터 변환 중 오류: {file_name} - {ve}")
        if filter_date_obj is None and '_FinancialSafetyFilter.csv' in file_name:
            try:
                filter_date_str = file_name.split('_')[0]
                filter_date_obj = datetime.strptime(filter_date_str, "%Y-%m-%d").date()
            except:
                 pass
        return filter_date_obj, []
    except Exception as e:
        print(f"    [오류] CSV 파일 파싱 중 예상치 못한 오류 발생: {file_name} - {e}")
        if filter_date_obj is None and '_FinancialSafetyFilter.csv' in file_name:
            try:
                filter_date_str = file_name.split('_')[0]
                filter_date_obj = datetime.strptime(filter_date_str, "%Y-%m-%d").date()
            except:
                 pass
        return filter_date_obj, []

    return filter_date_obj, stock_names # 종목코드 포함 시 반환값 수정 필요

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    print(f"DEBUG: CONDITION_SEARCH_FILES_FOLDER = {CONDITION_SEARCH_FILES_FOLDER}")
    print(f"DEBUG: Folder exists? {os.path.exists(CONDITION_SEARCH_FILES_FOLDER)}")
    # CONDITION_SEARCH_FILES_FOLDER 내의 모든 *_FinancialSafetyFilter.csv 파일을 찾음
    # glob.escape()를 제거하고 직접 경로를 합칩니다.
    file_pattern = os.path.join(CONDITION_SEARCH_FILES_FOLDER, "*_FinancialSafetyFilter.csv")
    print(f"DEBUG: file_pattern = {file_pattern}")
    csv_files = sorted(glob.glob(file_pattern))
    print(f"DEBUG: Found {len(csv_files)} files: {csv_files[:5]}") # 처음 5개 파일명만 출력

    if not csv_files:
        print(f"지정된 폴더 '{CONDITION_SEARCH_FILES_FOLDER}'에 '*_FinancialSafetyFilter.csv' 패턴의 파일이 없습니다.")
    
    all_filtered_data = [] # (날짜, 종목명) 등을 담을 리스트

    for file_path in csv_files:
        print(f"\n--- 파일 처리 시작: {os.path.basename(file_path)} ---")
        filter_date, stock_name_list = parse_hts_condition_csv(file_path) # 종목코드 포함 시 stock_info_list 등으로 변경

        if filter_date:
            if stock_name_list: # 종목명 리스트 (또는 종목 정보 리스트)가 실제로 있을 때
                print(f"  > 필터링 날짜: {filter_date}, 추출된 종목 수: {len(stock_name_list)}")
                # print(f"  > 종목 리스트 (상위 5개): {stock_name_list[:5]}...") 
                for stock_name in stock_name_list: # 종목코드 포함 시 stock_info 등으로 변경
                    # 종목코드 포함 시: all_filtered_data.append({'filter_date': filter_date, 'stock_name': stock_info['name'], 'stock_code': stock_info['code']})
                    all_filtered_data.append({'filter_date': filter_date, 'stock_name': stock_name})
            else:
                print(f"  > 필터링 날짜: {filter_date}, 추출된 종목 없음")
        else:
            print(f"  > 파일명에서 유효한 날짜를 추출하지 못했습니다. ({os.path.basename(file_path)})")
            
    if all_filtered_data:
        parsed_df = pd.DataFrame(all_filtered_data)
        print("\n\n--- 전체 파싱 결과 (샘플) ---")
        
        # 날짜별 그룹화하여 종목명 리스트(또는 종목 정보 리스트) 확인
        # 종목코드를 포함했다면, 'stock_name' 대신 다른 방식으로 요약 필요할 수 있음
        summary_df = parsed_df.groupby('filter_date')['stock_name'].apply(lambda x: list(x) if x.notna().any() else []).reset_index()
        print(summary_df.head())
        
        # 예시: 특정 날짜의 종목 리스트 확인
        # if not summary_df.empty:
        #     sample_date_to_check = summary_df['filter_date'].iloc[0]
        #     print(f"\n--- {sample_date_to_check} 종목 리스트 ---")
        #     # 종목코드 포함 시: print(parsed_df[parsed_df['filter_date'] == sample_date_to_check][['stock_name', 'stock_code']].to_dict('records'))
        #     print(parsed_df[parsed_df['filter_date'] == sample_date_to_check]['stock_name'].tolist())

        # 전체 데이터를 CSV로 저장 (선택적)
        output_csv_path = os.path.join(os.path.dirname(CONDITION_SEARCH_FILES_FOLDER), "processed_data", "parsed_weekly_filtered_stocks.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        try:
            parsed_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"\n파싱된 전체 데이터가 다음 파일로 저장되었습니다: {output_csv_path}")
        except Exception as e:
            print(f"\n파싱된 데이터 저장 중 오류 발생: {e}")
            
    else:
        print("\n\n--- 최종 파싱된 데이터가 없습니다. ---") 