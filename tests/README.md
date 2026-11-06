# 작은 CPU 예제와 테스트 실행

외부 계정·DB·GPU 없이 시작하려면 아래 경로를 사용하세요. 패키지 설치에는 인터넷이 필요합니다. 확인 환경은 **Ubuntu 24.04·Python 3.12**이며 세부 버전·SHA는 CI 실행 요약에 남깁니다. Windows/macOS에서는 이번에 실행하지 않았습니다.

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

[확인용 의존성](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/requirements-review.txt)은 NumPy 2.2.6, pandas 2.2.3, Requests 2.32.5, mysql-connector-python 26.10.0, tqdm 4.67.1입니다. 실제 전략이 가져오는 데이터 계층 오류 도우미와 진행 표시에 드라이버·tqdm이 필요합니다. **드라이버 설치와 DB 접속은 별개**이며 이 경로는 제품 설정·DB 서버·CUDA를 사용하지 않습니다.

<a id="cpu-example"></a>
## 2. 가상 데이터로 CPU 분할투자 끝까지 실행하기

```bash
python -m examples.cpu_split_demo
# 선택: 반드시 새 디렉터리. 기존 경로는 덮어쓰지 않습니다.
python -m examples.cpu_split_demo --output-dir results/cpu-split-demo
```

[예제 소스](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/examples/cpu_split_demo.py)는 기존 `MagicSplitStrategy`·`BacktestEngine`·`BasicExecutionHandler`·`Portfolio`를 그대로 호출합니다. 가격과 사전 구성한 종목 등급만 메모리 조회기로 제공합니다. 제품 YAML/INI·실제 DB·네트워크·GPU는 사용하지 않습니다.

가상 종목 `SYNTH001`의 입력은 아래와 같습니다. 거래일은 직접 명시하며 외부 달력을 조회하지 않습니다.

| 날짜 | 시가 | 고가 | 저가 | 종가 |
| --- | ---: | ---: | ---: | ---: |
| 2024-01-02 | 10000 | 10100 | 9900 | 10000 |
| 2024-01-03 | 10000 | 10100 | 9400 | 9500 |
| 2024-01-04 | 9400 | 9600 | 9300 | 9400 |
| 2024-01-05 | 10000 | 10500 | 9900 | 10400 |
| 2024-01-08 | 10400 | 11200 | 10300 | 11100 |
| 2024-01-09 | 11100 | 11200 | 11000 | 11100 |
| 2024-01-10 | 11100 | 11200 | 11000 | 11100 |

초기 현금 100만원, 종목 1개, 월 기준 자산의 10%를 한 번의 주문 금액으로 사용합니다. 이번 기간의 주문 금액은 10만원입니다. 최대 2회 분할, 직전 매수가 대비 5% 하락 시 추가 매수, 개별 매수분 10% 익절, 재진입 대기 5거래일, 손절 -15%, 비활성 기준 90거래일입니다. 추가 매수는 낮은 분할 차수 우선(`lowest_order`), 신규 진입은 가상 등급 1만 허용하는 기존 `tier`·`strict_hysteresis_v1` 정책이며 조회 오류는 실패입니다.

매일 등급 1, ATR 비율 0.1, 시가총액 10억원, 가격 매력도 0.8·신뢰도 1.0·수급 지표 0.01을 사전 입력합니다. 이 예제는 지표나 등급의 산정 과정을 검증하지 않습니다. 가격 기준을 바꾸는 기업행사도 없는 가상 자료입니다.

**전일 조건으로 신호를 만들고 당일 시가를 호가 단위로 올려 체결합니다.** 목표 익절 가격이 실제 체결가라는 뜻은 아닙니다. 아래 수수료·세금은 기존 코드의 예시 모델이며 현재 거래 요율을 안내하는 표가 아닙니다. 매수·매도 수수료는 각각 0.015%, 매도 세금은 0.18%입니다. 매수 수수료는 버림, 매도는 기존 float32 모델로 순수취액을 버림한 뒤 공제액을 배분합니다.

기대 거래 결과:

| 거래일 | 신호 기준일 | 동작 | 수량 | 체결가 | 수수료 | 세금 | 이후 현금 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 01-03 | 01-02 | 1차 매수 | 10 | 10000 | 15 | 0 | 899985 |
| 01-04 | 01-03 | 2차 매수 | 10 | 9400 | 14 | 0 | 805971 |
| 01-08 | 01-05 | 2차분 매도 | 10 | 10400 | 15 | 188 | 909768 |
| 01-09 | 01-08 | 1차분 매도 | 10 | 11100 | 16 | 201 | 1020551 |

| 날짜 | 현금 | 보유 수량 | 보유 평가액 | 총자산 |
| --- | ---: | ---: | ---: | ---: |
| 01-02 | 1000000 | 0 | 0 | 1000000 |
| 01-03 | 899985 | 10 | 95000 | 994985 |
| 01-04 | 805971 | 20 | 188000 | 993971 |
| 01-05 | 805971 | 20 | 208000 | 1013971 |
| 01-08 | 909768 | 10 | 111000 | 1020768 |
| 01-09 | 1020551 | 0 | 0 | 1020551 |
| 01-10 | 1020551 | 0 | 0 | 1020551 |

첫날은 전일 자료가 없어 거래하지 않고, 마지막 날은 재진입 대기 중이어서 현금을 유지합니다.

실행 마지막에 거래 4건·최종 현금/자산 1,020,551원·수수료 60원·세금 389원·순손익 20,551원을 확인합니다.

위 값은 실제 거래·일별 보유 기록에서 만들며 고정 기대값과 모두 대조합니다. `확인 완료`와 종료 코드 0이 합격 기준입니다. 순손익은 최종 자산−초기 현금으로 계산하며 매수 수수료도 포함합니다. 매도 기록의 `realized_pnl` 합 20,580원에는 매수 수수료 29원이 빠져 있으므로 순손익과 단순히 같다고 표시하지 않습니다.

CSV는 UTF-8 BOM으로 저장해 Excel에서 한글을 읽기 쉽게 합니다. 날짜는 `YYYY-MM-DD`, 원화·수량은 정수이며 실행 시간·무작위 ID는 넣지 않습니다.

- `trades.csv`: `trade_date`, `signal_date`, `ticker`, `side`(BUY/SELL), `split`, `quantity`, `price`, `commission`, `tax`, `cash_after`.
- `daily_equity.csv`: `date`, `cash`, `quantity`, `market_value`, `total_value`.

기본 실행은 파일을 만들지 않습니다. 선택 저장은 기존 디렉터리를 거부하므로 반복 저장하려면 새 경로를 지정합니다. 두 CSV는 같은 실행의 콘솔 값과 일치합니다. 기대값 불일치·입력 누락·저장 실패는 비정상 종료이며 성공으로 표시하지 않습니다. 첫날과 마지막 무거래일을 포함한 7일 전체 입력을 먼저 확인합니다. 생성물은 기존 `results/` 제외 규칙에 따라 Git에 넣지 않습니다.

이 결과는 **가상 자료로 기존 CPU 매매 경로를 끝까지 재현한 범위**입니다. 수익성·실거래·SQL 조회·종목 선정·GPU·전체 기간 시뮬레이션 검증과 구분합니다. [새 회귀 검사](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/tests/test_cpu_split_demo.py)에서 미래 자료 변경, 입력 누락, DB·네트워크 차단, 반복·CSV·오류 종료도 확인합니다.

<a id="example"></a>
## 3. 가상 데이터로 반영 시점 확인하기

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

## 4. CPU 기본 검사 실행

```bash
python tools/ci/run_cpu_checks.py
```

CI와 같은 명령입니다. 아래 8개 모듈과 예제 2개를 각각 별도 프로세스에서 실행하며, 동시에 실행하는 작업은 최대 3개입니다. 모듈·예제마다 2분 제한을 둡니다. 전체 `test_*.py`를 수집하지 않으므로 이 명령으로 GPU·DB 검사가 실행되지 않습니다.

각 모듈에서 수집한 검사 식별자를 실제 시작·종료 기록과 대조합니다. 검사 0개·누락·중복·가져오기 실패, 실패·오류·건너뜀·예상된 실패, 결과 기록 누락, 비정상 종료·시간 초과는 모두 실패입니다. 한 작업이 실패해도 다른 독립 작업의 결과를 모으며, **모든 검사와 예제가 성공해야 종료 코드 0**입니다. 아래 34개는 최초 도입 기준이며 선택된 모듈에 추가된 검사도 자동 포함합니다.

특정 모듈의 상세 출력을 따로 보고 싶다면 기존 명령도 사용할 수 있습니다. 이 개별 `unittest` 명령은 기본 동작상 건너뜀을 허용하므로 CI 성공 판정에는 위 통합 명령을 사용합니다.

```bash
python -m unittest discover -s tests -p 'test_portfolio.py' -v
python -m unittest discover -s tests -p 'test_short_selling_collector.py' -v
python -m unittest discover -s tests -p 'test_daily_stock_tier_batch.py' -v
```

| 검사 | 확인하는 동작 | 현재 검사 구성 |
| --- | --- | --- |
| `test_portfolio.py` | 현금·보유 상태, 자산 평가, 기록, 가격 일괄 조회 | 8개 통과 |
| `test_short_selling_collector.py` | 일부 요청 실패, 성공 데이터 보존, 실패 집계, 빈 구간 기록, 재실행 범위 | 7개 통과 |
| `test_daily_stock_tier_batch.py` | 종목 등급 계산, 데이터 반영 시점, 저장 행 묶음 처리·누락값 정리 | 19개 통과 |
| `test_cpu_candidate_priority.py` | 후보 순위·추가 매수·일괄 조회 | 6개 |
| `test_cpu_strategy_entry_context.py` | 첫날·등급 정책·조회 오류·진입 상태 | 6개 |
| `test_backtester_entry_metrics.py` | 엔진의 후보 준비·상태 집계 | 3개 |
| `test_execution_halt_guard.py` | 거래 불가 시가 처리 | 3개 |
| `test_cpu_split_demo.py` | 합성 매매·현금·입력·외부 연결 없음·CLI·CSV | 20개 |

최초 도입은 34개였으며 현재 구성은 **기존 34개 + 관련 CPU 18개 + 새 예제 검사 20개 = 72개**와 예제 2개입니다. 실패·건너뜀은 0이어야 하며 이후 선택 모듈에 추가된 검사도 자동 포함합니다. 실제 엔진·전략·체결·자산 계산을 예제에서 호출하고, 데이터만 가상 입력으로 제공합니다. 다른 단위 검사에서는 DB·API를 모의 객체로 대체하거나 사용하지 않습니다. 수집기 검사는 의도적으로 실패를 일으키므로 `simulated chunk failure`, `errors=1` 등의 로그가 출력됩니다. 통합 명령은 로그 문구가 아닌 구조화된 검사 결과와 종료 코드로 판정합니다.

이 결과를 전체 제품 검사, 실제 데이터 제공처 접속, GPU 성능 재측정으로 확대하지 않습니다. 최초 수동 실행은 [안내 정비 기록](../todos/done_2026_11_06-public-project-refresh.md), 자동 검사 도입은 [CI 작업 기록](../todos/done_2026_11_06-cpu-checks-ci.md)에 남깁니다.

### GitHub에서 자동 실행하는 범위

[CPU checks 실행 기록](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/actions/workflows/cpu-checks.yml)에서 확인할 수 있습니다.

- main 대상 PR의 생성·갱신·재개, main push, Actions 화면의 수동 실행이 대상입니다. 문서 변경에도 실행합니다.
- 무료 표준 `ubuntu-24.04` 실행기와 Python 3.12를 사용합니다. 실행마다 새 가상환경에 확인용 의존성을 설치하고 `pip check`를 수행합니다.
- **도구 자체 검사 → 위 CPU 검사·예제** 순서입니다. 자체 24개는 임시 자료로 실패·누락·취소·예제 CLI 인수 격리와 복원을 확인하며 제품 검사 수와 따로 집계합니다.
- 작업 전체는 10분 제한이며 같은 PR·브랜치에 새 실행이 생기면 이전 실행을 취소합니다. 취소·미실행은 통과가 아닙니다.
- 로그·작업 요약에 후보 SHA, 실제 checkout SHA, Python·직접 의존성 버전과 결과를 남깁니다. PR은 GitHub의 임시 병합 커밋을 검사하므로 후보 SHA와 checkout SHA가 다를 수 있습니다.
- GPU·실제 DB·외부 수집·성능 측정·문서 링크 자동 검사는 제외합니다. 병합 필수 검사 설정은 별도로 지정하지 않았습니다.

도구만 수정할 때 자체 검사를 실행하는 명령:

```bash
python -m unittest discover -s tools/ci/tests -p 'test_run_cpu_checks.py' -v
```

## 5. 별도 환경에서 확인할 검사

DB 통합 검사는 새 격리 환경에서 확인했습니다. 나머지 아래 범위는 이번에 실행하지 않았습니다. 자세한 준비 조건은 [전체 실행 안내](../docs/running.md)를 확인하세요.

| 범위 | 필요한 환경·조건 | 명령 또는 시작점 |
| --- | --- | --- |
| CPU의 다른 검사 | 해당 모듈의 DB 드라이버·설정 관련 의존성 등 추가 확인 | `test_point_in_time.py`, `test_data_handler.py` 등은 최소 설치에 포함하지 않음 |
| GPU 회귀 검사 | CUDA GPU, 호환되는 CuPy·cuDF 환경 | `python -m unittest discover -s tests -p 'test_backtest_strategy_gpu.py' -v` |
| DB 통합 검사 | 전용 임시 MySQL과 합성 자료. CPU 기본 검사와 별도 | [준비·실행·실패 기준](../docs/testing-mysql.md), `python tools/ci/run_db_checks.py` |
| 전체 검사 | GPU 의존성과 격리 DB를 포함한 전체 실행 환경 | `python -m unittest discover -s tests -p 'test_*.py' -v` |

`test_integration.py`는 일반 제품 설정을 읽지 않고 전용 변수·실제 DB/사용자·빈 스키마를 확인한 후에만 합성 행을 삽입·삭제합니다. 연결 인자 불일치를 테스트 쪽에서 제거했으며, DB 실행을 명시한 뒤의 준비 오류는 건너뛰지 않습니다. [DB integration 실행 기록](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/actions/workflows/db-integration.yml)에서 연결 제한 검사와 실제 3개 시나리오의 결과를 따로 확인합니다.

## 6. 결과를 읽는 기준

- 실행한 명령·커밋·환경·검사 수와 함께 결과를 기록합니다.
- `skipped`, 의존성 누락, 중단은 통과로 세지 않습니다.
- GPU 검사를 CPU 흉내 객체로 통과시켜 실제 GPU 검사처럼 소개하지 않습니다.
- [문서 화면 검사](../docs/pages-maintenance.md)는 제품 검사를 대신하지 않습니다.
- GitHub Actions의 `CPU checks`는 위 8개 모듈·예제 2개와 도구 자체 검사, `DB integration`은 임시 MySQL의 합성 CPU 통합 검사, `Secret Scan`은 비밀정보 검사, `pages build and deployment`는 문서 배포입니다. 각각의 성공을 다른 범위의 성공으로 해석하지 않습니다.

[프로젝트 소개](../README.md) · [세 가지 구현 근거](../docs/engineering-evidence.md) · [문서 안내](../docs/README.md)
