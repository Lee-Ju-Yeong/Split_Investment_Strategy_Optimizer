# feat(data): Point-in-Time 규칙 명문화 및 룩어헤드 방지 테스트 추가
(현재 파일 이름은 YYYY_MM_DD-issue<이슈번호>-<issue_name_only_english>.md 로 지정)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/64`
- 백테스트/실거래 정합성을 위해 Point-in-Time(PIT) 규칙을 프로젝트 전역 기준으로 정리하고, 룩어헤드 위반을 테스트로 차단해야 함
- 상장폐지 "결과 라벨" 사용, 공시 시점 미반영 등으로 생존편향/룩어헤드 위험이 존재함

## 1. 배경(현재 이슈의 대략적인 이전 맥락)
- 현재 프로젝트는 `DailyStockPrice` + `CalculatedIndicators` 중심의 백테스트 경로가 메인이고, 재무/수급 데이터 확장은 진행 중인 상태
- `TODO.md`에 P0 항목으로 PIT 규칙 명문화 및 룩어헤드 방지 테스트가 등록되어 있으며(이슈 #64), 데이터 확장 이슈(#65~#68)의 선행 조건임
- 원격 저장소는 `llm.md` 단일 소스 + `AGENTS.md`/`GEMINI.md` symlink 구조로 운영 중

## 1. 현재 이슈 및 현상, 디버그 했던 내용
### 1-1. 백테스트 시점 규칙이 코드/문서에 분산
- PIT 기준(당일 가용 정보만 사용)이 단일 문서/단일 테스트로 강제되지 않아, 향후 기능 추가 시 룰 이탈 가능성이 있음
- `lag`(T+1 또는 공시 +45일) 정책이 구현 수준에서 일관되게 강제되지 않음

### 1-2. 데이터 확장 전에 룰 미정의 시 리스크
- `FinancialData`/`InvestorTradingTrend` 조인 확대 시, 공시 시점 처리 실수로 룩어헤드가 쉽게 유입될 수 있음
- Tier 사전계산 도입 시, "미래에 상폐되지 않은 종목" 같은 결과 라벨이 섞이면 성능이 왜곡됨

### 1-3. 이슈 확인 결과
- GitHub 이슈 #64는 open 상태이며 목표/범위/완료 조건이 정의되어 있음
- `todo-management` 스크립트로 확인 시, 기존 `todos/` 디렉토리에 이슈 #64 전용 TODO 파일은 없음

#### 1-3-1. 기타 참고해야할 로직/원칙/세부사항1
- 데이터 조인 경로: `src/data_handler.py`
- 스키마 정의: `src/db_setup.py`
- 전략 후보군 조회/랭킹: `src/strategy.py`
- 파이프라인 오케스트레이션: `src/main_script.py`

---

## 2. 목표(해결하고자 하는 목표)

PIT(시점 정합성) 규칙을 문서와 테스트로 고정해, 데이터 확장/전략 고도화 과정에서 룩어헤드 및 생존편향이 재발하지 않도록 한다.
- 당일 시점에 알 수 없는 정보가 백테스트 입력으로 사용되지 않도록 정책을 명문화
- `lag` 규칙(T+1 또는 공시 +45일)을 코드/테스트 관점에서 검증 가능한 형태로 정의
- 위반 시 자동 실패하는 테스트를 추가해 회귀를 방지

### 2-1. (사람이 생각하기에) 우선적으로 참조할 파일 (이 파일들 이외에 자율적으로 더 찾아봐야 함)
- `TODO.md`
- `src/data_handler.py`
- `src/db_setup.py`
- `src/strategy.py`
- `src/main_script.py`
- `tests/test_data_handler.py`
- `docs/MAGIC_SPLIT_STRATEGY_PRINCIPLES.md`

### 2-2. 요구사항(구현하고자 하는 필요한 기능)
- PIT 규칙 문서화: "결과 라벨 사용 금지", "당일 가용 정보만 사용"을 명시
- `lag` 정책 명시: 최소 T+1, 재무 공시는 보수적 +45일 기본
- 데이터 조회 가드: 기준일 이후 데이터 사용 차단 로직(또는 검증 훅) 설계
- 테스트 추가: 룩어헤드 시나리오 재현 테스트(정상/위반 케이스)
- 완료 조건 점검표: 이슈 #64의 완료 조건과 1:1 매핑

---

(하위 부분은 사람이 작성하는게 아니라 AI 가 작성하는 부분)

# AI 결과

## 3. (AI가 생각하기에) 이슈의 원인으로 의심되는 부분들
- `src/strategy.py:121` 기준으로 신규 진입 신호가 `current_date`의 지표(`ma_5`, `ma_20`)를 그대로 사용함
- `src/strategy.py:220`/`src/strategy.py:265` 기준으로 익절/추가매수 판단에 당일 `high`/`low`를 사용하고, 같은 루프에서 즉시 체결되어 intraday 정보 선행사용 가능성이 큼
- `src/data_handler.py:71`의 동일일자 JOIN(`DailyStockPrice` + `CalculatedIndicators`)은 지표 생성 시점과 매매 시점을 분리하지 않으면 EOD look-ahead로 이어질 수 있음
- `src/data_handler.py:125`의 유니버스 조회는 as-of 구조를 일부 갖췄지만, `WeeklyFilteredStocks` 생성 규칙이 PIT 기준으로 강제되는지 테스트로 보장되지 않음
- `tests/`에 PIT 룰 위반을 재현/차단하는 회귀 테스트가 없어 향후 기능 추가 시 재발 가능성이 높음

## 4. (AI가 진행한) 디버그 과정
- `src/data_handler.py` 분석:
  - `load_stock_data()`가 시작일 이전 버퍼 데이터를 포함해 로드 후 기간 필터링하는 흐름 확인
  - `get_filtered_stock_codes()`가 `filter_date < current_date`로 과거 주차를 가져오는 쿼리임을 확인
- `src/backtester.py` 분석:
  - 일자 루프 내에서 신호 생성 후 같은 날짜에 매수/매도 실행되는 순서를 확인
- `src/strategy.py` 분석:
  - 신규 진입, 추가매수, 익절 판단 모두 `current_date` 캔들 값을 직접 참조함을 확인
- `tests/test_data_handler.py` 분석:
  - as-of 반환 테스트는 있으나, 신호-체결 시점 분리나 룩어헤드 위반 케이스 테스트가 없음
- 룩어헤드 재현 시나리오 설계:
  - `T`일에만 MA 교차가 발생하는 fixture를 두고 `T`일 즉시 진입을 금지해야 하는 테스트 케이스를 정의

## 5. (AI가) 파악한 이슈의 원인
- **핵심 원인 1: Signal/Execution 시점 분리 부재**
  - 현재 구조는 `T`일 정보로 `T`일 체결이 가능해지는 경로가 존재함
- **핵심 원인 2: PIT 규칙의 코드 레벨 단일 가드 부재**
  - 전략/데이터 핸들러에서 일관된 `as_of_date` 정책을 강제하는 공통 인터페이스가 없음
- **핵심 원인 3: 룰 위반을 잡아내는 테스트 부재**
  - 위반 시나리오가 테스트로 고정되지 않아 리팩토링/기능 확장 시 회귀 위험이 큼

---

## 6. 생각한 수정 방안들
### 6-1. 방안 A: `T+1 execution` 강제 (가장 보수적)
- 파일경로: `src/backtester.py`, `src/strategy.py`
- 무엇을: `T`일에는 신호만 생성하고, 실제 체결은 `T+1 open`에서 실행하도록 주문 큐를 도입
- 어떻게:
```python
# pseudo
signals_t = strategy.generate_signals(date_t, ...)
order_queue[date_t_plus_1].extend(signals_t)
execute_orders(order_queue[date_t], price_basis="open")
```
- 왜: intraday `high/low/close`를 본 뒤 같은 날 체결하는 경로를 원천 차단함
- 트레이드오프: 구현 범위가 큼(백테스터 이벤트 플로우 변경), 성과지표가 보수적으로 하락 가능

### 6-2. 방안 B: `T`일 체결 유지 + 신호 입력을 `T-1`로 고정 (중간 난이도)
- 파일경로: `src/strategy.py`, `src/data_handler.py`
- 무엇을: 신호 산출 시 데이터 참조를 `as_of_date = current_date - 1 business day`로 고정
- 어떻게:
```python
# pseudo
as_of = prev_trading_day(current_date)
row = stock_data.loc[as_of]
if row["ma_5"] > row["ma_20"]:
    emit_buy_signal(execution_date=current_date)
```
- 왜: 현 구조를 크게 깨지 않으면서 PIT 위반을 줄일 수 있음
- 트레이드오프: 당일 급변 반영력이 떨어지고, 거래일 캘린더 처리 로직이 추가로 필요

### 6-3. 방안 C: PIT Guard + 회귀 테스트 우선 (최소 침습)
- 파일경로: `src/data_handler.py`, `tests/test_point_in_time.py`, `tests/test_data_handler.py`
- 무엇을: 기준일 이후 데이터 접근 시 예외/경고를 발생시키는 Guard와 위반 재현 테스트를 먼저 추가
- 어떻게:
```python
# pseudo
def assert_point_in_time(df, as_of_date):
    if df.index.max() > as_of_date:
        raise PointInTimeViolation(...)
```
- 왜: 빠르게 안전장치를 만들고, 이후 A/B안을 단계적으로 적용할 수 있음
- 트레이드오프: Guard만으로는 전략 로직의 구조적 룩어헤드를 완전히 제거하지 못함

### 6-4. 방안 D: 유니버스/재무 데이터 `as_of` 스냅샷 명시 (확장 대비)
- 파일경로: `src/db_setup.py`, `src/data_handler.py`, 향후 `financial_collector.py`
- 무엇을: `WeeklyFilteredStocks`/재무 데이터에 `as_of_date` 또는 `effective_from/to`를 명시하고 JOIN 기준을 통일
- 어떻게:
```sql
-- pseudo
SELECT ... WHERE as_of_date <= :current_date ORDER BY as_of_date DESC LIMIT 1
```
- 왜: 향후 `FinancialData`, `InvestorTradingTrend` 확장 시 생존편향/룩어헤드를 구조적으로 차단
- 트레이드오프: 스키마/배치 파이프라인 변경이 필요하며 초기 마이그레이션 비용이 큼

---

## 7. 최종 결정된 수정 방안 (AI 가 자동 진행하면 안되고 **무조건**/**MUST** 사람에게 선택/결정을 맡겨야 한다)
- 최종 선택: **방안 C -> 방안 B 순차 적용**
  - 1단계(C): `src/data_handler.py`에 PIT Guard(`assert_point_in_time`, `get_stock_row_as_of`) 추가 + 회귀 테스트 신설
  - 2단계(B): `src/strategy.py`에서 신호 산출 기준일을 `T-1`로 고정하고, 체결일은 기존처럼 `T` 유지

### 7-1. 최종 결정 이유 1 (안전장치 선적용)
- 데이터 접근 레이어에 Guard를 먼저 넣으면 향후 재무/수급 조인 확장 시에도 동일한 PIT 규칙을 재사용할 수 있음
- 룩어헤드 위반을 테스트로 먼저 고정해두면 전략 리팩토링 중 회귀를 조기에 탐지 가능

### 7-2. 최종 결정 이유 2 (구조 변경 최소화)
- `BacktestEngine` 이벤트 플로우를 크게 바꾸지 않고도, 신호 입력 시점만 `T-1`로 이동해 핵심 룩어헤드를 줄일 수 있음
- 즉시 적용 가능하며 기존 CPU/GPU 비교 검증 파이프라인에 미치는 충격이 상대적으로 작음

### 7-3. 잔여 리스크/후속 과제
- `T+1 execution` 전면 전환(A안)은 아직 미적용이므로, 체결 모델 현실화는 후속 이슈로 별도 관리 필요
- `as_of` 스키마(D안)는 데이터셋 확장 이슈(#65~#67) 진행 시 병행 설계 필요

---

## 8. 코드 수정 요약
- 한 줄 요약: PIT Guard를 데이터 접근 계층에 추가하고, 전략 신호 기준일을 `T-1`로 이동했으며, 회귀 테스트를 신설했다.
### 8-1. 데이터 접근 계층 PIT Guard
- [x] `src/data_handler.py` 에서 `PointInTimeViolation`, `assert_point_in_time`, `get_stock_row_as_of` 추가
  - `index.asof()` 기반으로 실제 사용 row index를 검증하여 미래 데이터 참조를 차단
- [x] `src/data_handler.py` 에서 `get_latest_price`, `get_ohlc_data_on_date`를 `get_stock_row_as_of` 경유로 통일
  - 가격 조회/OHLC 조회 모두 같은 PIT 검증 경로를 사용하도록 정리

### 8-2. 전략 로직 T-1 신호 기준 적용
- [x] `src/strategy.py` 에서 `_resolve_signal_date` 추가
  - `current_day_idx` 기준으로 신호 기준일을 `T-1`로 계산
- [x] `src/strategy.py` 에서 신규/추가/매도 신호 생성 시 `get_stock_row_as_of(..., signal_date, ...)` 사용
  - 신호 산출은 과거 가용 정보 기준으로 제한하고 주문일은 기존 `current_date` 유지

### 8-3. 회귀 테스트 추가
- [x] `tests/test_point_in_time.py` 신규 추가
  - `DataHandler` PIT row 조회, `PointInTimeViolation` 검증, 신규 진입 신호의 `T-1` 기준 동작 검증
- [x] 테스트 실행
  - `conda run -n rapids-env python -m unittest tests.test_point_in_time -v`
  - `conda run -n rapids-env python -m unittest tests.test_data_handler.TestDataHandler.test_get_latest_price tests.test_data_handler.TestDataHandler.test_get_latest_price_no_data -v`

---

## 9. 문제 해결에 참고
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
(문제 해결에 참고했던 issue 번호가 포함된 링크 or commit hash)
- issue: https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/64
