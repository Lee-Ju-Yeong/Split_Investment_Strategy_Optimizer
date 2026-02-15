# 구조화된 로깅 도입 및 하드코딩 디버그 출력 제거 (Issue #55)
(현재 파일 이름은 YYYY_MM_DD-issue<이슈번호>-<issue_name_only_english>.md 로 지정)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/55`
- 목표: 표준 `logging`으로 출력 수준(verbosity)을 제어하고, 하드코딩 디버그 출력을 정리
- 범위:
  - 프로젝트 공용 로깅 설정 추가
  - 핵심 루프의 `print`/`tqdm.write`를 로거로 교체
  - 디버그 출력은 레벨/플래그로 제어
  - `src/backtester.py` 하드코딩 디버그 티커 출력 제거 또는 파라미터화
- 완료 조건:
  - 로그 레벨로 출력량 제어 가능
  - 하드코딩 디버그 티커 없음
  - 진행바(tqdm)와 로깅이 충돌하지 않음

## 1. 배경(현재 이슈의 대략적인 이전 맥락)
- 현재 프로젝트는 주요 실행 모듈에서 `print(...)` / `tqdm.write(...)`가 혼재되어 있어,
  - 출력량 제어가 어렵고
  - 진행바(tqdm)와 출력이 섞여 가독성이 떨어지며
  - 디버그 코드(특정 ticker 하드코딩 등)가 런타임에 상시 수행되는 문제가 있음
- 특히 `src/backtester.py`는 루프마다 하드코딩 디버그 티커를 조회/출력하고, 일일 스냅샷을 매일 출력하여 운영/개발 로그가 과다해질 수 있음

#### 1-1. 참고(헤치면 안 되는 핵심 원칙)
- 이 이슈는 “로깅/출력 방식”만 정리하며, 전략/체결/신호/DB 로직의 기능적 의미 변경은 금지
- tqdm 진행바를 유지하되, 로그 출력이 진행바를 깨지 않도록 해야 함
---

## 2. 요구사항(구현하고자 하는 필요한 기능)
### 2-1. 공용 로깅 설정(표준 logging)
- 프로젝트 공통 `setup_logging()` 제공
- 기본 출력 포맷/레벨 설정 가능
- 레벨/플래그는 최소한 환경변수로 제어 가능해야 함(예: `LOG_LEVEL=INFO|DEBUG`)

### 2-2. tqdm 친화적 로깅
- 진행바가 활성화된 상태에서도 로그가 진행바를 깨지 않도록 처리
- 권장: `tqdm.write` 기반 핸들러 또는 `tqdm.contrib.logging` 활용(추가 의존성 없이)

### 2-3. 핵심 루프 출력 정리
- `src/backtester.py`: `print`/`tqdm.write` 기반 디버그/일일 스냅샷 출력은 logger로 이동
  - 일일 스냅샷(대량 출력)은 기본적으로 `DEBUG` 레벨로 내리고, `INFO`에서는 요약만 출력하는 방향 고려
- `src/strategy.py`, `src/execution.py`, `src/main_backtest.py` 등 핵심 실행 경로의 `print`/`tqdm.write`를 logger로 치환

### 2-4. 하드코딩 디버그 티커 제거/파라미터화
- `src/backtester.py`의 `debug_ticker = '013570'` 제거
- 필요 시 환경변수(예: `BACKTEST_DEBUG_TICKER=...`)로 지정될 때만 debug 출력 수행

#### 2-4-1. 우선 참조 파일
- `src/backtester.py`
- `src/execution.py`
- `src/strategy.py`
- `src/main_backtest.py`
- (공용 로깅 유틸 추가 예정) `src/logging_utils.py` (신규)

---

(하위 부분은 사람이 작성하는게 아니라 AI 가 작성하는 부분)

# AI 결과

## 3. (AI가 확인한) 기존 코드/구현의 핵심내용들/의도들
- `src/backtester.py`
  - `print("백테스팅 엔진을 시작합니다...")`, `print("백테스팅이 완료되었습니다.")` 등 직접 출력
  - 루프 내부에 `debug_ticker = '013570'` 하드코딩 + OHLC를 `tqdm.write`로 출력
  - 일일 포트폴리오 스냅샷을 매일 `tqdm.write`로 출력(매우 큰 멀티라인 문자열)
- `src/strategy.py`
  - 초기 15일 슬롯 상태를 `tqdm.write`로 출력 (`[CPU_SLOT_DEBUG] ...`)
  - Tier/hybrid fallback 관측 로그도 `tqdm.write`로 출력
- `src/execution.py`
  - 매수/매도 계산 로그를 `print`로 출력 (`[CPU_BUY_CALC]`, `[CPU_SELL_CALC]`, `[CPU_SELL_PRICE]`)
- `src/main_backtest.py`
  - 엔진 실행/파일 저장 등 상태 메시지를 `print`로 출력
- 현재 `src/` 내에 `logging` 기반의 공용 설정 모듈은 없음(대부분 print 기반)

---

## 4. 생각한 수정 방안들 (ai 가 생각하기에) 구현에 필요한 핵심 변경점
(수정 방안 최소 3가지)
(작성 시: 파일경로:위치 + 무엇을 + 어떻게 + 왜. 코드 전체 복사 금지)
### 4-1. (권장) stdlib `logging` + tqdm-safe Handler 도입(무의존성)
- `src/logging_utils.py` (신규): `setup_logging()` + `TqdmLoggingHandler`(emit에서 `tqdm.write`) 제공
  - 환경변수로 제어: `LOG_LEVEL`(기본 INFO), `LOG_FORMAT`(예: `plain|json` 선택)
- `src/backtester.py`: `print/tqdm.write`를 `logger.info/debug`로 치환
  - `debug_ticker='013570'` 제거 → `BACKTEST_DEBUG_TICKER` 환경변수로 지정될 때만 동작
  - 일일 스냅샷은 `DEBUG`로 내리고 `logger.isEnabledFor(DEBUG)`일 때만 문자열 생성
- `src/strategy.py`, `src/execution.py`, `src/main_backtest.py`: `print/tqdm.write`를 logger로 치환
- 왜: 추가 deps 없이 요구사항을 모두 충족(레벨 제어, 하드코딩 제거, tqdm 충돌 최소화)

### 4-2. `loguru` 도입(간단하지만 의존성 추가)
- `loguru`로 로거를 통일하고, `tqdm.write` sink로 연결하여 진행바 충돌 회피
- 장점: 코드 변경이 단순하고 포맷이 좋음
- 단점: 런타임 의존성 추가(환경/배포/CI 영향)

### 4-3. `structlog` 기반 JSON 구조화 로깅(의존성 추가 + 학습비용)
- 구조화(키-값) 로깅을 “기본”으로 두고, 콘솔은 pretty, 파일은 JSON으로 저장
- 장점: 로그 분석/적재(ELK/Datadog 등)에 최적
- 단점: 도입 비용이 높고 현재 프로젝트에 과한 변경일 수 있음

---

## 5. 최종 결정된 수정 방안 (AI 가 자동 진행하면 안되고 **무조건**/**MUST** 사람에게 선택/결정을 맡겨야 한다)
(작성 시: 파일경로:위치 + 무엇을 + 어떻게 + 왜. 코드 전체 복사 금지)
- 선택(사용자 결정): **4-1. stdlib `logging` + tqdm-safe Handler 도입(무의존성)**
  - `src/logging_utils.py`: `setup_logging()` + `TqdmLoggingHandler` 추가
  - 핵심 루프(`src/backtester.py`)와 실행 경로(`src/main_backtest.py`, `src/strategy.py`, `src/execution.py`)의 `print/tqdm.write`를 logger로 치환
  - 디버그 출력은 `LOG_LEVEL=DEBUG` 및 `BACKTEST_DEBUG_TICKER`(옵션)으로 제어

### 5-1. 최종 결정 이유
- 추가 의존성 없이(무GPU/무DB 노트북 환경 포함) 바로 적용 가능
- 로그 레벨로 출력량 제어가 가능해지고, 디버그 출력이 기본 경로에 상시 섞이는 문제를 해결
- tqdm 진행바와 로그 출력 충돌을 최소화할 수 있음(핸들러에서 `tqdm.write` 사용)

---

## 6. 코드 수정 요약
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
(최종 방안이 결정되면 checkbox 로 checklist 를 먼저 작성한 후 코드 수정이 진행되면 경과를 기록한다)
(작성 시: 파일경로:라인 + 무엇을 + 어떻게. 코드 전체 복사 금지)
- stdlib `logging` 기반 공용 설정을 추가하고, 핵심 루프의 `print/tqdm.write` 디버그 출력을 로거로 이동(레벨/환경변수로 제어).

### 6-1. 공용 로깅 설정 추가
- [x] `src/logging_utils.py:72` 공용 `setup_logging()` + `TqdmLoggingHandler` + `JsonLogFormatter` 추가
  - env var: `LOG_LEVEL`, `LOG_FORMAT`, `LOG_FILE`, `LOG_USE_TQDM`
  - `force=True` 재설정 시 기존 핸들러 `close()` 후 제거(장기 실행 프로세스 리소스 누수 방지)
  - `tqdm.write(..., file=stream)`로 핸들러의 stream 설정을 존중

### 6-2. 핵심 루프/실행 경로 출력 정리
- [x] `src/main_backtest.py:178` CLI 실행 진입점에서 `setup_logging()` 호출 + 상태 출력 `print` -> `logger.info/error/exception`
  - `display_results_in_terminal()`의 `print`는 “사용자용 리포트 출력”이라 유지
  - `run_backtest_from_config()`는 root handler가 비어있을 때만 자동 `setup_logging()` (Flask 등 비-CLI 경로 대비)
- [x] `src/backtester.py:34` 시작/종료 `print` 제거, 하드코딩 `debug_ticker='013570'` 제거
  - `BACKTEST_DEBUG_TICKER`가 있고 `LOG_LEVEL=DEBUG`일 때만 OHLC 디버그 로그 출력
  - 일일 포트폴리오 스냅샷(대량 출력)도 `DEBUG`일 때만 생성/출력
- [x] `src/strategy.py:112` `tqdm.write/print` 기반 관측 로그를 `logger.debug/warning`으로 교체
  - 초기 15일 슬롯 디버그는 `DEBUG`에서만 출력
  - fallback warning은 기본 WARNING에선 traceback 미출력, DEBUG에서만 `exc_info` 포함
- [x] `src/execution.py:92` 매수/매도 계산 `print`를 `logger.debug`로 교체
  - `DEBUG`가 아닐 때는 인자/포맷 평가가 발생하지 않도록 게이팅

---

## 7. 문제 해결에 참고
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
(문제 해결에 참고했던 issue 번호가 포함된 링크 or commit hash)
- issue: https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/55
- commit: (작업 완료 후 기록)
