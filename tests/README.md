# 작은 CPU 예제와 테스트 실행

외부 계정·DB·GPU 없이 시작하려면 아래 경로를 사용하세요. 패키지 설치에는 인터넷이 필요합니다. **실행 확인 환경은 Linux·Python 3.12.3**이며 Windows/macOS에서는 이번에 실행하지 않았습니다.

## 1. 처음 설치하기

```bash
git clone https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer.git
cd Split_Investment_Strategy_Optimizer

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-review.txt
python -m pip check
```

이후 명령은 모두 저장소 루트에서 실행합니다. 이미 `.venv`를 쓰고 있다면 덮어쓰지 말고 별도 작업 폴더나 격리 환경을 사용하세요.

[확인용 의존성](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/requirements-review.txt)은 NumPy 2.2.6, pandas 2.2.3, Requests 2.32.5입니다. 전체 제품 환경을 설치하는 파일과 구분합니다. 이 경로에서는 설정 파일 복사, DB 생성, CUDA 설치가 필요하지 않습니다.

<a id="example"></a>
## 2. 가상 데이터로 반영 시점 확인하기

```bash
python -m examples.short_selling_lag_demo
```

[예제 소스](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/examples/short_selling_lag_demo.py)는 가상 종목 `SAMPLE`과 명시적인 6거래일을 만들고, 기존 `build_daily_stock_tier_frame`을 호출합니다. `sbv_ratio`는 공매도 잔고금액을 시가총액으로 나눈 비율입니다. 예제에서는 3천만 ÷ 10억 = 0.03입니다.

예상 결과:

| 계산 날짜 | 당일 반영: 비교용 | 3거래일 뒤 반영: 현재 임시 정책 |
| --- | --- | --- |
| 2024-01-02 | 0.03 | 미반영 |
| 2024-01-03 | 미반영 | 미반영 |
| 2024-01-04 | 미반영 | 미반영 |
| 2024-01-05 | 미반영 | 0.03 |
| 2024-01-08 | 미반영 | 미반영 |
| 2024-01-09 | 미반영 | 미반영 |

`미반영`은 값 0과 다릅니다. 이 예제에는 기준일 데이터가 한 건만 있으며, 다른 날짜로 값을 채우지 않습니다. 실행 끝에 `확인 완료`가 출력되고 종료 코드가 0이어야 합니다. 기대한 날짜·값이 다르거나 기본 지연 정책이 바뀌면 예제가 실패합니다.

이 예제는 **데이터 반영 시점**을 설명합니다. 여섯 행만으로 장기 종목 등급이나 투자 성과를 평가하지 않으며 거래·수익률도 계산하지 않습니다. 3거래일은 프로젝트의 임시 정책이고, 실제 공개 시점이 항상 3거래일이라는 의미가 아닙니다.

## 3. CPU 기본 검사 실행

```bash
python tools/ci/run_cpu_checks.py
```

CI와 같은 명령입니다. 아래 세 모듈과 위 예제를 각각 별도 프로세스에서 실행하며, 동시에 실행하는 작업은 최대 3개입니다. 모듈·예제마다 2분 제한을 둡니다. 전체 `test_*.py`를 수집하지 않으므로 이 명령으로 GPU·DB 검사가 실행되지 않습니다.

각 모듈에서 수집한 검사 식별자를 실제 시작·종료 기록과 대조합니다. 검사 0개·누락·중복·가져오기 실패, 실패·오류·건너뜀·예상된 실패, 결과 기록 누락, 비정상 종료·시간 초과는 모두 실패입니다. 한 작업이 실패해도 다른 독립 작업의 결과를 모으며, **모든 검사와 예제가 성공해야 종료 코드 0**입니다. 아래 34개는 최초 도입 기준이며 선택된 모듈에 추가된 검사도 자동 포함합니다.

특정 모듈의 상세 출력을 따로 보고 싶다면 기존 명령도 사용할 수 있습니다. 이 개별 `unittest` 명령은 기본 동작상 건너뜀을 허용하므로 CI 성공 판정에는 위 통합 명령을 사용합니다.

```bash
python -m unittest discover -s tests -p 'test_portfolio.py' -v
python -m unittest discover -s tests -p 'test_short_selling_collector.py' -v
python -m unittest discover -s tests -p 'test_daily_stock_tier_batch.py' -v
```

| 검사 | 확인하는 동작 | 2026-11-06 실행 결과 |
| --- | --- | --- |
| `test_portfolio.py` | 현금·보유 상태, 자산 평가, 기록, 가격 일괄 조회 | 8개 통과 |
| `test_short_selling_collector.py` | 일부 요청 실패, 성공 데이터 보존, 실패 집계, 빈 구간 기록, 재실행 범위 | 7개 통과 |
| `test_daily_stock_tier_batch.py` | 종목 등급 계산, 데이터 반영 시점, 저장 행 묶음 처리·누락값 정리 | 19개 통과 |

최초 도입 기준은 **34개 통과, 실패 0, 건너뜀 0**입니다. DB·API는 모의 객체로 대체하거나 사용하지 않습니다. 수집기 검사는 의도적으로 실패를 일으키므로 `simulated chunk failure`, `errors=1` 등의 로그가 출력됩니다. 통합 명령은 로그 문구가 아닌 구조화된 검사 결과와 종료 코드로 판정합니다.

이 결과를 전체 제품 검사, 실제 데이터 제공처 접속, GPU 성능 재측정으로 확대하지 않습니다. 최초 수동 실행은 [안내 정비 기록](../todos/done_2026_11_06-public-project-refresh.md), 자동 검사 도입은 [CI 작업 기록](../todos/done_2026_11_06-cpu-checks-ci.md)에 남깁니다.

### GitHub에서 자동 실행하는 범위

[CPU checks 실행 기록](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/actions/workflows/cpu-checks.yml)에서 확인할 수 있습니다.

- main 대상 PR의 생성·갱신·재개, main push, Actions 화면의 수동 실행이 대상입니다. 문서 변경에도 실행합니다.
- 무료 표준 `ubuntu-24.04` 실행기와 Python 3.12를 사용합니다. 실행마다 새 가상환경에 확인용 의존성을 설치하고 `pip check`를 수행합니다.
- **도구 자체 검사 → 위 CPU 검사·예제** 순서입니다. 자체 검사는 임시 자료로 실패·누락·취소 처리를 확인하며 제품 검사 수와 따로 집계합니다.
- 작업 전체는 10분 제한이며 같은 PR·브랜치에 새 실행이 생기면 이전 실행을 취소합니다. 취소·미실행은 통과가 아닙니다.
- 로그·작업 요약에 후보 SHA, 실제 checkout SHA, Python·직접 의존성 버전과 결과를 남깁니다. PR은 GitHub의 임시 병합 커밋을 검사하므로 후보 SHA와 checkout SHA가 다를 수 있습니다.
- GPU·실제 DB·외부 수집·성능 측정·문서 링크 자동 검사는 제외합니다. 병합 필수 검사 설정은 별도로 지정하지 않았습니다.

도구만 수정할 때 자체 검사를 실행하는 명령:

```bash
python -m unittest discover -s tools/ci/tests -p 'test_run_cpu_checks.py' -v
```

## 4. 별도 환경에서 확인할 검사

DB 통합 검사는 새 격리 환경에서 확인했습니다. 나머지 아래 범위는 이번에 실행하지 않았습니다. 자세한 준비 조건은 [전체 실행 안내](../docs/running.md)를 확인하세요.

| 범위 | 필요한 환경·조건 | 명령 또는 시작점 |
| --- | --- | --- |
| CPU의 다른 검사 | 해당 모듈의 DB 드라이버·설정 관련 의존성 등 추가 확인 | `test_point_in_time.py`, `test_data_handler.py` 등은 최소 설치에 포함하지 않음 |
| GPU 회귀 검사 | CUDA GPU, 호환되는 CuPy·cuDF 환경 | `python -m unittest discover -s tests -p 'test_backtest_strategy_gpu.py' -v` |
| DB 통합 검사 | 전용 임시 MySQL과 합성 자료. CPU 기본 검사와 별도 | [준비·실행·실패 기준](../docs/testing-mysql.md), `python tools/ci/run_db_checks.py` |
| 전체 검사 | GPU 의존성과 격리 DB를 포함한 전체 실행 환경 | `python -m unittest discover -s tests -p 'test_*.py' -v` |

`test_integration.py`는 일반 제품 설정을 읽지 않고 전용 변수·실제 DB/사용자·빈 스키마를 확인한 후에만 합성 행을 삽입·삭제합니다. 연결 인자 불일치를 테스트 쪽에서 제거했으며, DB 실행을 명시한 뒤의 준비 오류는 건너뛰지 않습니다. [DB integration 실행 기록](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/actions/workflows/db-integration.yml)에서 연결 제한 검사와 실제 3개 시나리오의 결과를 따로 확인합니다.

## 5. 결과를 읽는 기준

- 실행한 명령·커밋·환경·검사 수와 함께 결과를 기록합니다.
- `skipped`, 의존성 누락, 중단은 통과로 세지 않습니다.
- GPU 검사를 CPU 흉내 객체로 통과시켜 실제 GPU 검사처럼 소개하지 않습니다.
- [문서 화면 검사](../docs/pages-maintenance.md)는 제품 검사를 대신하지 않습니다.
- GitHub Actions의 `CPU checks`는 위 세 모듈·예제와 도구 자체 검사, `DB integration`은 임시 MySQL의 합성 CPU 통합 검사, `Secret Scan`은 비밀정보 검사, `pages build and deployment`는 문서 배포입니다. 각각의 성공을 다른 범위의 성공으로 해석하지 않습니다.

[프로젝트 소개](../README.md) · [세 가지 구현 근거](../docs/engineering-evidence.md) · [문서 안내](../docs/README.md)
