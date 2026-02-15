# 테스트 인터페이스 갱신 (test_integration.py) (Issue #59)
(현재 파일 이름은 YYYY_MM_DD-issue<이슈번호>-<issue_name_only_english>.md 로 지정)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/59`
- 목표: `tests/test_integration.py`를 최신 Strategy/Execution 인터페이스에 맞게 정비
- 범위:
  - Strategy 추상 메서드 시그니처와 테스트 더블 정합성 맞추기
  - 테스트용 실행 핸들러/전략을 최신 실행 흐름(매도 -> 신규 -> 추가)으로 수정
  - 불필요한 모듈/의존성 제거
- 완료 조건:
  - 테스트가 최신 코드 기준으로 정상 실행
  - Strategy 인터페이스 변경 시 테스트가 즉시 반영되도록 구조 개선
- (추가) 노트북 개발 제약(로컬 DB/GPU/의존 패키지 부재)에서도 `unittest discover`가 깨지지 않게 스킵 경계 정리

## 1. 배경(현재 이슈의 대략적인 이전 맥락)
- `BacktestEngine`가 하루 단위로 다음 순서로 신호를 생성/실행하도록 변경됨: `sell -> new_entry -> additional_buy` (`src/backtester.py`)
- `Strategy` 인터페이스가 단일 `generate_signals()`에서 3개 메서드로 분리됨 (`src/strategy.py`)
  - `generate_sell_signals(...)`
  - `generate_new_entry_signals(...)`
  - `generate_additional_buy_signals(...)`
- `BasicExecutionHandler.execute_order(...)` 시그니처가 `current_day_idx`를 요구하고, 주문 payload 키가 변경됨 (`src/execution.py`)
- 최근 코드에서 `src/`는 패키지로 취급되며(상대 import 사용), 테스트가 `sys.path`로 `src/`를 직접 넣어 top-level import 하는 방식은 깨지기 쉬움

## 1. 현재 이슈 및 현상, 디버그 했던 내용
### 1-1. 인터페이스 불일치로 인한 테스트 더블 붕괴
- `tests/test_integration.py`의 `SimpleBuyStrategy`가 구 인터페이스(`generate_signals`)만 구현 → 최신 `Strategy` 추상 메서드 3개 미구현으로 인스턴스화 불가
- `tests/test_integration.py`의 `TestExecutionHandler.execute_order(...)` 시그니처가 최신 엔진 호출과 불일치(`current_day_idx` 누락)
- 주문 딕셔너리 키가 최신 실행 로직이 기대하는 형태와 불일치(`investment_amount`, `reason_for_trade`, `trigger_price` 등)

### 1-2. 노트북 환경(로컬 DB/GPU/의존패키지 부재)에서 discover가 깨지는 문제
- DB 통합 테스트는 `config.ini`/DB 연결/DB 드라이버(pymysql or mysql-connector)가 없으면 실패해야 하는데, 현재는 자동 스킵 경계가 없어 `python -m unittest discover`가 깨질 수 있음
- GPU 의존 테스트(`cupy`, `cudf`)도 유사하게 import 단계에서 실패하면 전체 discover를 망가뜨릴 수 있음(옵션 범위)

### 1-3. <디버그내용1>(생략가능)
- 구체적인 코드나, 재현가능한 스크립트, db 조회 쿼리 등으로 진단한 결과
- <디버그내용1내용1>
- <디버그내용1내용2>
```
<예시코드1>
```
- ...
#### 1-3-1. <기타 참고해야할 로직/원칙/세부사항1>(생략가능)
- <참고해야할 로직/원칙/세부사항1내용1>
```
<예시코드2>
```
- ...

---

## 2. 목표(해결하고자 하는 목표)

<목표 핵심 정리>
- `tests/test_integration.py`를 최신 `Strategy` 인터페이스(3단계 신호 생성) 및 `BasicExecutionHandler` 호출 규약에 맞춰 수정
- 테스트 더블(전략/실행핸들러)을 최소화하여, 인터페이스가 변하면 테스트가 즉시 깨지도록 정렬
- 로컬 DB/의존패키지가 없는 환경에서는 DB 통합 테스트가 명확한 메시지로 skip되도록 개선(노트북 개발 생산성)

### 2-1. (사람이 생각하기에) 우선적으로 참조할 파일 (이 파일들 이외에 자율적으로 더 찾아봐야 함)
- `tests/test_integration.py`
- `src/strategy.py`
- `src/backtester.py`
- `src/execution.py`
- `src/portfolio.py`
- `src/data_handler.py`
- `src/db_setup.py`
- `tests/README.md`

---

(하위 부분은 사람이 작성하는게 아니라 AI 가 작성하는 부분)

# AI 결과

## 3. (AI가 생각하기에) 이슈의 원인으로 의심되는 부분들
- `tests/test_integration.py`가 구 인터페이스(`generate_signals`) 기반으로 작성되어, 최신 `Strategy` 추상 메서드(3개)와 불일치
- `BacktestEngine`이 `execute_order(..., current_day_idx)`로 호출하는데 테스트 더블 `TestExecutionHandler.execute_order(...)`는 인자 누락
- 최신 `src/execution.py`가 상대 import를 사용(`from .portfolio ...`)하는데, 테스트는 `sys.path`에 `src/`를 직접 넣어 top-level import를 수행 → import 경로가 취약
- 노트북/미니멀 환경에서 DB/GPU/의존 패키지 부재 시, import 단계에서 실패하면 `unittest discover` 자체가 깨짐 (스킵 경계 부재)

## 4. (AI가 진행한) 디버그 과정
- `tests/test_integration.py` 실행 시 환경에 따라 import 단계에서 실패 가능
  - 시스템 Python: `pandas` 미설치로 `ModuleNotFoundError: pandas`
  - `rapids-env`: `pymysql` 미설치로 `ModuleNotFoundError: pymysql` (`src/db_setup.py`의 top-level import 때문)
- 코드 점검으로 확인한 인터페이스 변화
  - `src/strategy.py`: `Strategy`가 3개 추상 메서드를 요구
  - `src/backtester.py`: `generate_sell_signals` -> `generate_new_entry_signals` -> `generate_additional_buy_signals` 순서로 호출, `execute_order(..., current_day_idx)` 호출

## 5. (AI가) 파악한 이슈의 원인
- 통합 테스트(`tests/test_integration.py`)가 최신 런타임 계약(Strategy/Execution/Import style)과 불일치하여 깨짐
- DB/GPU/optional deps 없는 환경에서 “실행 불가 테스트는 skip”로 처리돼야 하는데, 모듈 import 단계에서 하드 실패하는 구조라 개발 경험이 나쁨

---

## 6. 생각한 수정 방안들
(수정 방안 최소 3가지)
(작성 시: 파일경로:위치 + 무엇을 + 어떻게 + 왜. 코드 전체 복사 금지)

### 6-1. (권장) 인터페이스 정합 + 런타임 Skip 경계 추가
- `tests/test_integration.py`: 최신 `Strategy` 인터페이스 3개 메서드 구현으로 `SimpleBuyStrategy` 갱신
- `tests/test_integration.py`: `execute_order(..., current_day_idx)` 시그니처에 맞춰 실행 핸들러(가능하면 `BasicExecutionHandler` 자체) 사용
- `tests/test_integration.py`: 모듈 import 단계에서 DB 연결/드라이버 import를 하지 않고, `setUpClass`에서 조건 점검 후 `unittest.SkipTest`로 명확히 skip
  - 조건 예: `config.ini` 존재 여부, DB 드라이버 설치 여부, 실제 DB 연결 성공 여부
- (옵션) `tests/test_backtest_strategy_gpu.py`: `cupy/cudf` 미설치 시 모듈 import 실패 대신 skip 처리
- 왜: 이슈 #59의 “최신 인터페이스 정비”를 직접 해결하면서도, 노트북(무DB/무GPU) 환경에서 discover가 깨지지 않도록 함

### 6-2. 통합 테스트를 순수 유닛 테스트로 전환(DB 모킹)
- `tests/test_integration.py`: DB를 직접 쓰지 않고 `DataHandler`를 전부 `MagicMock`으로 대체
- 장점: 어디서나 실행 가능, 빠름
- 단점: SQL/DB 연동/캐시 로딩 등 실제 통합 결함을 잡지 못함 (통합 테스트 의미 약화)

### 6-3. 통합 테스트 실행 프로파일 분리(디스커버리에서 제외)
- `tests/test_integration.py`를 별도 디렉토리(예: `tests/integration/`)로 이동하거나 파일명을 `test_` 패턴에서 제외
- 기본 `unittest discover`에서는 빠른 단위 테스트만 실행, DB/GPU 통합은 별도 명령으로만 실행
- 장점: 구분이 명확하고 로컬 개발 경험이 좋음
- 단점: 통합 테스트가 덜 실행되어 bit-rot 위험 증가, 테스트 실행 문서/CI 정비 필요

---

## 7. 최종 결정된 수정 방안 (AI 가 자동 진행하면 안되고 **무조건**/**MUST** 사람에게 선택/결정을 맡겨야 한다)
(작성 시: 파일경로:위치 + 무엇을 + 어떻게 + 왜. 코드 전체 복사 금지)
- 선택(사용자 결정): **6-1. 인터페이스 정합 + 런타임 Skip 경계 추가**
  - `tests/test_integration.py`: 최신 `Strategy`(3단계 신호) + `BasicExecutionHandler.execute_order(..., current_day_idx)` 계약에 맞게 테스트 더블을 교체/정리
  - `tests/test_integration.py`: 로컬 DB/설정/드라이버가 없으면 import 단계 하드 실패 대신 `SkipTest`로 명확히 skip

### 7-1. 최종 결정 이유(근본 해결)
- 이슈 #59 목표(최신 인터페이스 정비)를 직접 해결하면서도, 노트북(무DB) 환경에서 `unittest`가 깨지지 않게 함
- 통합 테스트의 성격(실제 DB 연동)을 유지하면서, “실행 불가 환경”만 정확히 skip하여 유지보수/신뢰성 균형이 좋음
- 모킹으로 통합 의미를 희석(6-2)하거나, 파일 이동으로 실행 빈도를 떨어뜨리는(6-3) 리스크를 회피

---

## 8. 코드 수정 요약
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
(최종 방안이 결정되면 checkbox 로 checklist 를 먼저 작성한 후 코드 수정이 진행되면 경과를 기록한다)
(작성 시: 파일경로:라인 + 무엇을 + 어떻게. 코드 전체 복사 금지)
- `tests/test_integration.py`를 최신 인터페이스에 맞추고, DB/설정/드라이버 미존재 시 graceful skip 되도록 수정
### 8-1. 통합 테스트 인터페이스 정비
- [x] `tests/test_integration.py`: `SimpleBuyStrategy`를 최신 `Strategy` 인터페이스(3 메서드)로 변경
- [x] `tests/test_integration.py`: 주문 payload를 최신 `BasicExecutionHandler` 기대값(`investment_amount`, `reason_for_trade`, `trigger_price`, `start_date/end_date`)으로 변경
- [x] `tests/test_integration.py`: `config.ini`/DB 드라이버/DB 연결 실패 시 `SkipTest` 처리(모듈 import는 성공)
- [x] `tests/test_integration.py`: `sys.path` hack 제거하고 `src.*` 패키지 import로 정리

---

## 9. 문제 해결에 참고
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
(문제 해결에 참고했던 issue 번호가 포함된 링크 or commit hash)
- issue: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/59`
