# src/filtered_stock_loader.py
import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/filtered_stock_loader.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

import pandas as pd
import urllib.parse
from sqlalchemy import create_engine, text
from tqdm import tqdm

def get_db_engine(db_config: dict | None = None):
    """SQLAlchemy 엔진을 생성하여 반환합니다.

    Default config source:
    - `config/config.yaml` -> `database` section
    """
    if db_config is None:
        from .config_loader import load_config

        config = load_config()
        db_config = dict(config.get("database") or {})

    required = ["host", "user", "password", "database"]
    missing = [k for k in required if not db_config.get(k)]
    if missing:
        raise KeyError(f"DB 설정 누락: {missing}. config.yaml의 `database` 섹션을 확인하세요.")

    user = db_config["user"]
    password = urllib.parse.quote_plus(db_config["password"])
    host = db_config["host"]
    database = db_config["database"]

    # SQLAlchemy를 사용하여 데이터베이스 연결 엔진 생성
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}?charset=utf8mb4")
    return engine

def load_filtered_stocks_to_db(csv_path, engine):
    """
    필터링된 주간 종목 리스트 CSV 파일을 데이터베이스에 로드합니다.
    (filter_date, stock_code)가 PK이므로, 중복된 데이터는 덮어씁니다.
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_path)
        print(f"'{csv_path}'에서 {len(df)}개의 레코드를 읽었습니다.")

        # 컬럼 이름 확인
        required_columns = ['filter_date', 'stock_code']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV 파일에 필수 컬럼({', '.join(required_columns)})이 없습니다.")

        # 데이터 타입 변환
        df['filter_date'] = pd.to_datetime(df['filter_date']).dt.date

        # 데이터를 임시 테이블에 먼저 저장
        temp_table_name = 'temp_weekly_filtered_stocks'
        df.to_sql(temp_table_name, engine, if_exists='replace', index=False)
        print(f"임시 테이블 '{temp_table_name}'에 데이터 저장을 완료했습니다.")
        
        # INSERT ... ON DUPLICATE KEY UPDATE 쿼리를 사용하여 데이터 병합
        with engine.connect() as conn:
            # 트랜잭션 시작
            with conn.begin():
                # stock_name이 없는 경우를 대비하여 stock_code만 사용
                merge_sql = text(f"""
                INSERT INTO WeeklyFilteredStocks (filter_date, stock_code, company_name)
                SELECT filter_date, stock_code, stock_name FROM {temp_table_name}
                ON DUPLICATE KEY UPDATE
                company_name = VALUES(company_name);
                """)
                conn.execute(merge_sql)
            print("기존 'WeeklyFilteredStocks' 테이블과 데이터 병합을 완료했습니다.")

            # 임시 테이블 삭제
            conn.execute(text(f"DROP TABLE {temp_table_name}"))
            print(f"임시 테이블 '{temp_table_name}'을 삭제했습니다.")
            
        print("데이터베이스에 필터링된 종목 저장을 성공적으로 완료했습니다.")
        return True

    except FileNotFoundError:
        print(f"오류: CSV 파일을 찾을 수 없습니다 - '{csv_path}'")
        return False
    except Exception as e:
        print(f"데이터베이스에 저장하는 중 오류 발생: {e}")
        return False

def main():
    """메인 실행 함수"""
    from .config_loader import load_config, resolve_project_path

    config = load_config()
    pipeline_paths = (config.get("data_pipeline") or {}).get("paths") or {}
    csv_file_path = pipeline_paths.get("filtered_stocks_csv_path", "data/processed_data/mapped_weekly_filtered_stocks_FINAL.csv")

    db_engine = get_db_engine(config.get("database"))
    
    print("주간 필터링 종목 DB 적재 작업을 시작합니다.")
    load_filtered_stocks_to_db(str(resolve_project_path(csv_file_path)), db_engine)
    print("작업을 종료합니다.")

if __name__ == "__main__":
    main()
