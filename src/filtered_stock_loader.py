# src/filtered_stock_loader.py (수정 후)

import pandas as pd
from .db_setup import get_db_connection


def load_filtered_stocks_to_db(csv_path: str):
    """
    필터링된 주간 종목 리스트 CSV 파일을 데이터베이스에 로드합니다.
    INSERT IGNORE를 사용하여 (filter_date, stock_code) PK 중복을 무시합니다.
    """
    conn = get_db_connection()
    if not conn:
        print("DB 연결 실패. 작업을 중단합니다.")
        return False

    try:
        df = pd.read_csv(csv_path)
        print(f"'{csv_path}'에서 {len(df)}개의 레코드를 읽었습니다.")

        required_columns = ["filter_date", "stock_code", "stock_name"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"CSV 파일에 필수 컬럼({', '.join(required_columns)})이 없습니다."
            )

        df["filter_date"] = pd.to_datetime(df["filter_date"]).dt.date
        df.rename(columns={"stock_name": "company_name"}, inplace=True)
        # DB 컬럼 순서에 맞게 DataFrame 컬럼 재정렬
        df = df[["filter_date", "stock_code", "company_name"]]

        print("'WeeklyFilteredStocks' 테이블에 데이터 병합을 시작합니다...")

        # df.to_sql()을 executemany로 대체
        sql = "INSERT IGNORE INTO WeeklyFilteredStocks (filter_date, stock_code, company_name) VALUES (%s, %s, %s)"
        data_tuples = [tuple(row) for row in df.itertuples(index=False)]

        with conn.cursor() as cursor:
            inserted_rows = cursor.executemany(sql, data_tuples)
        conn.commit()

        print(f"데이터 병합 완료. {inserted_rows}개의 신규 레코드가 추가되었습니다.")
        return True

    except FileNotFoundError:
        print(f"오류: CSV 파일을 찾을 수 없습니다 - '{csv_path}'")
        return False
    except Exception as e:
        print(f"데이터베이스에 저장하는 중 오류 발생: {e}")
        if conn:
            conn.rollback()  # 오류 발생 시 롤백
        return False
    finally:
        if conn:
            conn.close()


def main():
    """메인 실행 함수"""
    csv_file_path = "data/processed_data/mapped_weekly_filtered_stocks_FINAL.csv"
    print("주간 필터링 종목 DB 적재 작업을 시작합니다.")
    load_filtered_stocks_to_db(csv_file_path)
    print("작업을 종료합니다.")


if __name__ == "__main__":
    main()
