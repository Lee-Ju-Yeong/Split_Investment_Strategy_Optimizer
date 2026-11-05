"""합성 통합 검사만을 위한 연결 제한. 제품 설정 파일을 읽지 않는다."""

import os


EXPECTED = {
    "host": "127.0.0.1",
    "user": "siso_test",
    "database": "siso_test_integration",
}


def required_config(environ=None):
    env = os.environ if environ is None else environ
    if env.get("SISO_RUN_DB_TESTS") != "1":
        raise ValueError("SISO_RUN_DB_TESTS=1을 명시해야 합니다.")
    config = {key: env.get("SISO_TEST_MYSQL_" + key.upper(), "")
              for key in (*EXPECTED, "port", "password")}
    for key, expected in EXPECTED.items():
        if config[key] != expected:
            raise ValueError(f"검사용 {key}는 {expected}만 허용합니다.")
    if not config["password"]:
        raise ValueError("SISO_TEST_MYSQL_PASSWORD가 필요합니다.")
    if not config["port"].isascii() or not config["port"].isdigit():
        raise ValueError("SISO_TEST_MYSQL_PORT는 명시적인 TCP 포트여야 합니다.")
    config["port"] = int(config["port"])
    if not 1 <= config["port"] <= 65535:
        raise ValueError("검사용 TCP 포트 범위가 잘못됐습니다.")
    return config


def verify_empty_database(connection):
    """변경 SQL 전에 실제 연결 대상과 기존 자료 부재를 확인한다."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT DATABASE(), CURRENT_USER(), VERSION()")
        database, account, version = cursor.fetchone()
        if database != EXPECTED["database"] or account.split("@", 1)[0] != EXPECTED["user"]:
            raise ValueError("실제 MySQL DB/사용자가 전용 검사 설정과 다릅니다.")
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = %s",
            (EXPECTED["database"],),
        )
        if cursor.fetchone()[0] != 0:
            raise ValueError("비어 있지 않은 DB에서는 통합 검사를 시작하지 않습니다.")
    return version


def prepare_database(connect, create_tables):
    config = required_config()
    connection = connect(**config, charset="utf8mb4", connect_timeout=5,
                         read_timeout=10, write_timeout=10)
    try:
        version = verify_empty_database(connection)
        create_tables(connection)
        return config, connection, version
    except BaseException:
        connection.close()
        raise
