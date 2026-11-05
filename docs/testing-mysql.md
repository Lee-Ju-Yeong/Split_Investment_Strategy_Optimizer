# 임시 MySQL로 확인하는 CPU 통합 검사

[CPU 기본 검사](../tests/README.md)는 DB 없이 실행합니다. 이 문서는 **실제 MySQL 저장·조회와 CPU 실행 인터페이스**를 합성 자료로 확인하는 별도 경로입니다. 실제 투자 전략 전체·GPU·운영 DB 검증은 아닙니다.

## GitHub 실행 기록 읽기

[DB integration](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/actions/workflows/db-integration.yml)은 main 대상 PR·main 변경·수동 실행에서 동작합니다. 무료 Ubuntu 24.04 / Python 3.12 실행기에 매번 빈 MySQL 8.4 서비스를 만들고 작업 종료 시 폐기합니다. 운영 인증정보나 외부 계정을 사용하지 않습니다.

연결 제한·실패 처리 검사와 실제 DB 시나리오는 별도 단계입니다. 결과 요약의 후보 SHA·실제 checkout SHA·의존성·MySQL 버전·검사 수·종료 코드를 함께 확인하세요. PR에서는 임시 병합 커밋을 검사하므로 후보와 checkout SHA가 다를 수 있습니다. 작업은 10분, 검사 프로세스는 2분 제한입니다.

2026-11-06 최초 실제 확인은 Python 3.12.16·MySQL 8.4.13에서 **연결 제한 14개, 아래 DB 시나리오 3개 통과(실패·오류·건너뜀 0)**입니다. [실행 로그·요약](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/actions/runs/42119887317), [후보·리뷰 기록](../todos/done_2026_02_15-issue59-tests-split-no-db-gpu.md)에서 범위를 확인할 수 있습니다.

| 합성 시나리오 | 기대 결과 |
| --- | --- |
| 가격 10,000원, 투자금 100,000원, 초기 현금 1,000,000원 | 10주 매수, 수수료 15원, 잔여 현금 899,985원 |
| 모든 거래일에 매수 조건 불충족 | 거래 0건, 현금·보유 상태 유지 |
| 2022-01-04 수정주가 누락 | 종목 999998·누락 날짜를 포함한 명확한 오류 |

실제 DataHandler·BacktestEngine·Portfolio·BasicExecutionHandler를 사용합니다. 전략은 검사할 신호를 만드는 작은 Strategy 구현입니다. 회사명·수정주가 열·거래일·후보 SQL 조회·일별 기록과 매도→신규 매수→추가 매수 호출 순서도 확인합니다. 이 자료는 수익률 예측이나 전체 MagicSplitStrategy의 검증 결과가 아닙니다.

## 같은 명령 실행하기

Linux와 Docker가 준비된 **검사용 개발 환경**에서 실행하는 예입니다. 기존 가상환경을 덮어쓰지 말고 별도 작업 폴더에서 준비합니다. 설치에는 인터넷이 필요합니다.

```bash
python3.12 -m venv .venv-db
source .venv-db/bin/activate
python -m pip install -r requirements-db-test.txt
python -m pip check

# 이 괄호 안의 임시 컨테이너만 종료 시 삭제합니다.
(
  set -eu
  siso_mysql_container=$(docker run --rm -d \
    -e MYSQL_ROOT_PASSWORD=ephemeral-root-only \
    -e MYSQL_DATABASE=siso_test_integration \
    -e MYSQL_USER=siso_test \
    -e MYSQL_PASSWORD=ephemeral-test-only \
    -p 127.0.0.1::3306 \
    mysql:8.4@sha256:a81d043930e51745522bd5c430961cf8286f0a23f0a9a2a6e0108f7ea71b0902)
  trap 'docker rm -f "$siso_mysql_container" >/dev/null' EXIT
  siso_ready=0
  for siso_attempt in $(seq 1 30); do
    if docker exec "$siso_mysql_container" mysqladmin ping -h 127.0.0.1 --silent >/dev/null 2>&1; then
      siso_ready=1
      break
    fi
    sleep 2
  done
  test "$siso_ready" = 1
  export SISO_RUN_DB_TESTS=1
  export SISO_TEST_MYSQL_HOST=127.0.0.1
  export SISO_TEST_MYSQL_PORT=$(docker port "$siso_mysql_container" 3306/tcp | cut -d: -f2)
  export SISO_TEST_MYSQL_USER=siso_test
  export SISO_TEST_MYSQL_PASSWORD=ephemeral-test-only
  export SISO_TEST_MYSQL_DATABASE=siso_test_integration
  python tools/ci/run_db_checks.py
)
```

같은 컨테이너에서 두 번 실행하면 이미 테이블이 있으므로 거부합니다. 매 실행마다 위처럼 새 컨테이너를 만드세요. 예제 비밀번호는 폐기되는 검사용 값이며 다른 DB에 사용하지 않습니다. GitHub에서도 동일한 Python 검사 명령을 사용하며 컨테이너 준비는 workflow의 service가 담당합니다. 위 수동 Docker 준비 절차와 Windows/macOS 실행은 별도 실측 범위입니다.

## 연결과 실패의 경계

- 일반 config.ini·config/config.yaml·MAGICSPLIT_CONFIG_PATH는 읽지 않습니다. 전용 변수만 사용하고 host=127.0.0.1, user=siso_test, database=siso_test_integration으로 제한합니다.
- 실제 DB·사용자와 기존 테이블 부재를 SELECT로 확인하기 전에는 스키마 생성·삽입·삭제를 하지 않습니다. 제한을 피하려고 기존 DB 이름이나 계정을 바꾸지 않습니다.
- DB 실행을 지정하지 않은 일반 unittest 탐색은 명확한 사유로 건너뜁니다. 명시적으로 실행한 뒤의 의존성·설정·연결·SQL 오류는 실패입니다.
- 공개 검사 명령은 검사 0개·누락·중복·건너뜀·결과 기록 누락·시간 초과를 실패로 처리합니다. 오류 로그 문자열로 성공을 판단하지 않습니다.
- 성공 여부와 실제 실행 링크는 [작업 기록](../todos/2026_11_06-project-maintenance-execution.md#db)을 따릅니다. GPU·실제 수집·전체 제품 환경은 별도입니다.
