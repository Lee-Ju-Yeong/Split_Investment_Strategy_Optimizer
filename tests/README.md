# 작은 CPU 예제와 테스트 실행

외부 계정·DB·GPU 없이 시작하려면 아래 경로를 사용하세요. 패키지 설치에는 인터넷이 필요합니다. **실행 확인 환경은 Linux·Python 3.12.3**이며 Windows/macOS에서는 이번에 실행하지 않았습니다.

## 1. 처음 설치하기

```bash
git clone https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer.git
cd Split_Investment_Strategy_Optimizer

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-review.txt
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

## 3. 기존 단위 테스트 실행

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

총 **34개 통과, 실패 0, 건너뜀 0**입니다. DB·API는 모의 객체로 대체하거나 사용하지 않습니다. 수집기 검사는 의도적으로 실패를 일으키므로 `simulated chunk failure`, `errors=1` 등의 로그가 출력됩니다. 최종 `unittest` 결과의 `OK`와 종료 코드를 확인하세요.

세 검사는 별도 프로세스로 병렬 실행할 수 있습니다. 이 결과를 전체 제품 검사, 실제 데이터 제공처 접속, GPU 성능 재측정으로 확대하지 않습니다. 실행 기준과 기록은 [이번 정비 기록](../todos/2026_11_06-public-project-refresh.md)에 있습니다.

## 4. 별도 환경에서 확인할 검사

아래 항목은 이번 정비에서 실행하지 않았습니다. 자세한 준비 조건은 [전체 실행 안내](../docs/running.md)를 확인하세요.

| 범위 | 필요한 환경·조건 | 명령 또는 시작점 |
| --- | --- | --- |
| CPU의 다른 검사 | 해당 모듈의 DB 드라이버·설정 관련 의존성 등 추가 확인 | `test_point_in_time.py`, `test_data_handler.py` 등은 최소 설치에 포함하지 않음 |
| GPU 회귀 검사 | CUDA GPU, 호환되는 CuPy·cuDF 환경 | `python -m unittest discover -s tests -p 'test_backtest_strategy_gpu.py' -v` |
| DB 통합 검사 | 격리된 검사용 MySQL, 설정·인터페이스 사전 점검 | `python -m unittest discover -s tests -p 'test_integration.py' -v` |
| 전체 검사 | GPU 의존성과 격리 DB를 포함한 전체 실행 환경 | `python -m unittest discover -s tests -p 'test_*.py' -v` |

`test_integration.py`는 테스트 행을 삽입·삭제하는 코드가 있습니다. 운영 DB에 연결해 실행하지 마세요. 또한 현재 소스에는 `get_db_connection(cls.db_config)` 호출과 인자를 받지 않는 `src.db_setup.get_db_connection()` 정의의 불일치가 남아 있습니다. 준비 단계 예외가 `SkipTest`로 바뀔 수 있으므로 **환경을 갖췄다는 이유만으로 해당 검사가 수행됐다고 보지 않습니다.** 이번에는 제품·기존 테스트 코드를 고치지 않았습니다.

## 5. 결과를 읽는 기준

- 실행한 명령·커밋·환경·검사 수와 함께 결과를 기록합니다.
- `skipped`, 의존성 누락, 중단은 통과로 세지 않습니다.
- GPU 검사를 CPU 흉내 객체로 통과시켜 실제 GPU 검사처럼 소개하지 않습니다.
- GitHub Actions의 `Secret Scan`은 비밀정보 검사, `pages build and deployment`는 문서 배포입니다. 현재 제품 단위 테스트 CI는 별도로 구성돼 있지 않습니다.

[프로젝트 소개](../README.md) · [세 가지 구현 근거](../docs/engineering-evidence.md) · [문서 안내](../docs/README.md)
