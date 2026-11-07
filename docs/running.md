# 전체 실행에 필요한 환경과 설정

**외부 계정·DB·GPU 없이 확인하는 경로는 [CPU 예제와 테스트 안내](../tests/README.md)에 있습니다.** 이 문서는 실제 데이터 수집·CPU 시뮬레이션·GPU 탐색을 준비할 때 읽습니다. 아래 전체 실행 경로는 2026-11-06 문서 정비에서 재실행하지 않았습니다.

## 환경을 구분하기

| 범위 | 환경과 의존성 | 이번 확인 |
| --- | --- | --- |
| 외부 연결 없는 CPU 예제·고정 검사 | Python 3.12, `requirements-review.txt` | [72개·예제 2개 구성과 실행 기록](../tests/README.md#cpu-example). 전체 제품 환경과 구분 |
| 전체 CPU 실행 | `requirement.txt`의 라이브러리, MySQL, 실행에 맞는 설정·입력 데이터 | 미실행 |
| GPU·WFO 실행 | 별도 CUDA/RAPIDS, CuPy·cuDF, 호환 GPU·드라이버, 데이터·설정 | 미실행 |
| 실제 수집 | 해당 제공처에 허용된 접근과 인증, DB, 요청 간격·조회 범위 설정 | 미실행 |

`requirement.txt`는 이름 그대로 단수형 파일입니다. 전체 의존성의 완전한 버전 잠금 파일은 아닙니다. 이미 준비한 별도 CPU 가상환경에서 설치하는 명령은 `python -m pip install -r requirement.txt`입니다. 확인용 최소 환경과 구분하고 기존 사용 환경을 덮어쓰지 않습니다.

기존 `environment.yml`은 Python 3.9를 지정합니다. 그러나 `src/config_loader.py`의 `str | None` 등 현재 코드에는 더 새로운 Python 문법이 있습니다. 따라서 이 파일만으로 현재 전체 실행 환경이 재현된다고 안내하지 않습니다. 이번에는 기존 환경 파일을 이관하거나 변경하지 않았습니다.

`rapids-env`는 기존 기록에 사용된 환경 이름입니다. 저장소의 `environment.yml`이나 `requirements-review.txt`로 생성되지 않습니다.

## 설정 파일 두 경로

CPU 실행 설정과 일부 수집·스키마 도구의 DB 설정 경로가 아직 다릅니다. YAML만 복사했다고 모든 DB 도구가 준비되지는 않습니다.

| 읽는 코드 | 설정 | 용도 |
| --- | --- | --- |
| `src.config_loader.load_config()` | 저장소 루트의 `config/config.yaml`; `MAGICSPLIT_CONFIG_PATH`로 변경 가능 | CPU 백테스트·GPU/WFO 등의 설정 |
| `src.db_setup.get_db_connection()` | 실행 디렉터리의 `config.ini`, `[mysql]` 섹션 | 스키마 도구와 이 함수를 호출하는 수집·배치 경로 |

저장소 루트에서 YAML 예제를 복사하고 실행 목적에 맞게 수정합니다.

```bash
cp config/config.example.yaml config/config.yaml
```

`config.ini` 형식은 다음과 같습니다. 실제 값은 본인에게 사용 권한이 있는 **격리된 검사용 DB**에 맞춰 로컬에서만 채웁니다.

```ini
[mysql]
host = <TEST_DB_HOST>
port = 3306
user = <TEST_DB_USER>
password = <TEST_DB_PASSWORD>
database = <TEST_DB_NAME>
```

두 파일은 Git 제외 대상입니다. 공개 문서·PR·로그에는 실제 인증정보를 넣지 않습니다. YAML의 `database`와 INI가 같은 DB를 의도한다면 두 설정을 따로 대조해야 합니다.

## 스키마 준비와 데이터 수집

아래 명령은 **DB를 변경**합니다. 검사용 DB와 접속 경로를 확인한 뒤 필요한 것만 실행합니다. 실행 중인 수집 작업을 중복 시작하지 않습니다.

```bash
# 저장소 루트, DB 드라이버와 config.ini가 준비된 환경
python -c "from src.db_setup import get_db_connection, create_tables; conn=get_db_connection(); create_tables(conn); conn.close()"
```

날짜 자리표시자를 실제 값으로 바꿔 사용하는 배치 명령 템플릿:

```text
python -m src.ticker_universe_batch --mode backfill --start-date <YYYYMMDD> --end-date <YYYYMMDD> --step-days 7 --workers 1
python -m src.ohlcv_batch --start-date <YYYYMMDD> --end-date <YYYYMMDD> --log-interval 20
python -m src.pipeline_batch --mode backfill --start-date <YYYYMMDD> --end-date <YYYYMMDD>
python -m src.pipeline_batch --mode daily --end-date <YYYYMMDD>
```

필요한 데이터 범위와 배치 옵션은 [DB 스키마](database/schema.md), [종목 등급 규칙](database/daily_stock_tier_rules.md), [운영 문서](operations/README.md)를 함께 확인합니다. 과거 수집 기록의 성공은 현재 제공처의 접속 가능성을 보장하지 않습니다. 인증 실패나 빈 응답을 데이터 부재로 단정하지 않습니다.

`src.main_script`는 과거 HTS CSV 처리 등을 포함한 기존 일반 파이프라인입니다. 처음 방문자의 기본 실행 명령으로 사용하지 않습니다.

## CPU·GPU·기간별 평가 실행

데이터와 설정이 준비된 각 환경에서 사용하는 진입점입니다. 결과 파일은 보통 Git에서 제외된 `results/` 아래에 생성됩니다.

```bash
# 준비된 CPU 환경
python -m src.main_backtest

# 별도로 준비한 GPU 환경에서 실행
python -m src.debug_gpu_single_run
python -m src.parameter_simulation_gpu
python -m src.walk_forward_analyzer
```

GPU 탐색 → 후보 고정 → CPU 검산 → 마지막 평가 구간 검증의 세부 조건은 [WFO 운영 안내](operations/2026-03-14-wfo-approval-runbook.md)에 있습니다. 구현된 기능과 실제 전략 승인 결과는 구분합니다.

과거 성능 재측정에는 추가 입력이 필요합니다.

```text
python -m src.issue98_perf_measure --config-path <동일_측정_설정.yaml> --canonical-profile issue98_janfeb2024_multibatch_research020 --label <새_실행명> --runs 2
```

기본 설정 경로인 `config/config.issue98_perf_multibatch_janfeb_2024_research020.yaml`과 당시 `results/` 원본은 공개 저장소에 없습니다. 일반 예제 설정으로 대체한 실행을 PR #105의 동일 조건 재현이라고 부르지 않습니다. [당시 조건과 한계](engineering-evidence.md#gpu)

## 기타 실행 경계

- GitHub Pages는 문서를 게시하며 시뮬레이션 웹 앱을 실행하지 않습니다.
- `src.app`은 별도 Flask 실행 코드입니다. 현재 직접 실행부는 `0.0.0.0:5000`, `debug=True`이므로 검토 없이 운영 공개 서버로 실행하지 않습니다. 이번에는 웹 UI를 실행·수정하지 않았습니다.
- DB 통합 검사는 제품 설정과 분리된 [임시 MySQL 안내](testing-mysql.md)를 따릅니다. 제품 DB 연결 API는 그대로이며 테스트 쪽의 잘못된 호출을 제거했습니다. 임시 DB의 합성 시나리오 결과는 운영 DB·전체 전략 검증과 구분합니다.
- 실제 KRX 수집·운영 DB·GPU 회귀·성능 측정·장기 재계산과 Windows/macOS 실행은 이번 검증 범위에 포함되지 않습니다.
