# src/db_setup.py (수정 후)

import pymysql
from .config_loader import load_config  # configparser 대신 yaml 로더 사용


def get_db_connection():
    """
    config.yaml 파일에서 DB 설정을 읽어와 pymysql 연결 객체를 반환합니다.
    이 함수가 프로젝트 전체의 유일한 DB 연결 통로가 됩니다.
    """
    try:
        # config_loader가 프로젝트 루트 기준 'config/config.yaml'을 찾도록 설정되어 있습니다.
        config = load_config("config.yaml")
        db_config = config["database"]

        conn = pymysql.connect(
            host=db_config["host"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
            charset="utf8mb4",  # utf8mb4로 설정하여 이모지 등 다국어 지원 강화
            cursorclass=pymysql.cursors.DictCursor,  # 결과를 딕셔너리 형태로 받으면 코드 작성이 편리해짐
        )
        return conn
    except FileNotFoundError:
        print(
            "[치명적 오류] 'config.yaml' 설정 파일을 찾을 수 없습니다. 프로젝트 루트에 파일이 있는지 확인해주세요."
        )
        return None
    except KeyError as e:
        print(f"[치명적 오류] 'config.yaml'에 필요한 설정 키가 없습니다: {e}")
        return None
    except Exception as e:
        print(f"[치명적 오류] 데이터베이스 연결에 실패했습니다: {e}")
        return None


def create_tables(conn):
    """
    프로젝트에 필요한 모든 테이블이 존재하는지 확인하고 없으면 생성합니다.
    (현재 프로젝트 구조에 맞춰 불필요한 테이블 정의 제거)
    """
    # 각 SQL 쿼리는 단일 실행 시 세미콜론(;)이 필요 없습니다.
    # VARCHAR(6) -> VARCHAR(20)으로 통일하여 혹시 모를 더 긴 티커 코드에 대비합니다.

    queries = [
        """
        CREATE TABLE IF NOT EXISTS CompanyInfo (
            stock_code VARCHAR(20) PRIMARY KEY,
            company_name VARCHAR(255),
            market_type VARCHAR(50),
            last_updated DATETIME
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS WeeklyFilteredStocks (
            filter_date DATE,
            stock_code VARCHAR(20),
            company_name VARCHAR(255),
            PRIMARY KEY (filter_date, stock_code)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS DailyStockPrice (
            stock_code VARCHAR(20),
            date DATE,
            open_price DECIMAL(20, 5),
            high_price DECIMAL(20, 5),
            low_price DECIMAL(20, 5),
            close_price DECIMAL(20, 5),
            volume BIGINT,
            PRIMARY KEY (stock_code, date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS CalculatedIndicators (
            stock_code VARCHAR(20),
            date DATE,
            ma_5 FLOAT NULL,
            ma_20 FLOAT NULL,
            atr_14_ratio FLOAT NULL,
            price_vs_5y_low_pct FLOAT NULL,
            price_vs_10y_low_pct FLOAT NULL,
            PRIMARY KEY (stock_code, date)
        )
        """,
    ]

    try:
        with conn.cursor() as cur:
            for query in queries:
                cur.execute(query)
        conn.commit()
    except Exception as e:
        print(f"테이블 생성 중 오류 발생: {e}")
        conn.rollback()
        # 오류 발생 시 더 이상 진행하지 않도록 예외를 다시 발생시킬 수 있습니다.
        raise e
