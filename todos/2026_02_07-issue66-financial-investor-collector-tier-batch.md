# feat(pipeline): 재무/수급 수집기 분리 및 DailyStockTier 사전 계산 배치 도입
(현재 파일 이름은 YYYY_MM_DD-issue<이슈번호>-<issue_name_only_english>.md 로 지정)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/66`
- 런타임 병목을 줄이고 재현성을 높이기 위해, 재무/수급 데이터 수집과 Tier 계산을 사전 배치 체계로 전환
- 백필(backfill)과 일배치(daily batch) 경로를 분리해 운영 안정성과 재실행 가능성을 확보

## 1. 배경(현재 이슈의 대략적인 이전 맥락)
- 이 프로젝트는 `DailyStockPrice`/`CalculatedIndicators` 중심의 백테스트 경로가 핵심이며, 최근 이슈 #65에서 `FinancialData`, `InvestorTradingTrend`, `DailyStockTier` 스키마를 선반영
- 백테스트 런타임에서 조건 완화 탐색을 반복하면 DB I/O와 계산량이 급증해 속도와 재현성이 떨어지는 문제가 확인됨
- Point-in-Time 규칙(T-1 신호, look-ahead bias 방지)은 이슈 #64에서 보강되었고, 이제 데이터 파이프라인 단의 확장이 필요

## 1. 현재 이슈 및 현상, 디버그 했던 내용
### 1-1. 런타임 병목
- 재무/수급/위험 조건을 런타임에 매일 계산하면 백테스트 반복 수행 시 시간이 크게 증가
- 전략 검증 속도가 느려지면서 파라미터 탐색 횟수가 제한됨
### 1-2. 수집/계산 책임 혼재
- 단일 수집 경로에 역할이 섞여 있어 백필과 일배치 운영 흐름이 명확하지 않음
- 실패 지점 분리가 어려워 재실행 시 복구 비용이 커짐
### 1-3. Tier 조회 경로 부재
- `DailyStockTier` 스키마는 준비되었지만 실제 계산/적재/조회 배치 파이프라인이 아직 없음

---

## 2. 목표(해결하고자 하는 목표)

재무/수급 수집과 Tier 계산을 배치 파이프라인으로 분리해, 백테스트 런타임에서는 사전 계산된 결과를 조회만 하도록 전환한다.
- 재무/수급 데이터 수집기를 모듈 분리하여 데이터 적재 책임을 명확히 구분
- 위험 프록시 기반 Tier를 백필/일배치로 사전 계산하여 실행 성능과 재현성 개선
- 운영 시나리오(초기 백필, 정기 일배치, 장애 재실행)를 명시해 실전 운영 가능 상태 확보

### 2-1. (사람이 생각하기에) 우선적으로 참조할 파일 (이 파일들 이외에 자율적으로 더 찾아봐야 함)
- `src/db_setup.py`
- `src/data_handler.py`
- `src/main_script.py`
- `src/stock_data_collector.py`
- `src/ohlcv_collector.py`
- `src/ticker_collector.py`
- `TODO.md`

### 2-2. 요구사항(구현하고자 하는 필요한 기능)
- `src/financial_collector.py` 신설: `FinancialData` 적재(백필/일배치 공용 진입점)
- `src/investor_trading_collector.py` 신설: `InvestorTradingTrend` 적재(백필/일배치 공용 진입점)
- `src/daily_stock_tier_batch.py` 신설: 위험 프록시(관리/정지, 저유동성, 자본잠식+lag)로 `DailyStockTier` 계산/적재
- 백필 스크립트/엔트리포인트 정리: 과거 기간 일괄 재계산 가능
- 일배치 실행 경로 추가: 신규 거래일 데이터만 증분 처리
- `src/data_handler.py` 조회 API 확장: 백테스트에서 Tier 필터를 단순 조회로 사용 가능
- 최소 검증 테스트 추가: Tier 계산 및 조회 경로의 PIT 규칙 위반 방지 확인

---

(하위 부분은 사람이 작성하는게 아니라 AI 가 작성하는 부분)

# AI 결과

## 3. (AI가 생각하기에) 이슈의 원인으로 의심되는 부분들
- `src/main_script.py:84`~`130` 파이프라인이 `WeeklyFilteredStocks -> OHLCV -> Indicators`까지만 연결되어 있고, `FinancialData`/`InvestorTradingTrend`/`DailyStockTier` 적재 단계가 없음
- `src/stock_data_collector.py:48`~`57` 에서 수집 단계 자체에 PER/PBR/DIV 하드 필터를 적용해 데이터 자산화를 저해하며, 전략 변경 시 재백필 비용을 증가시킴
- `src/data_handler.py:144`~`157` 는 `WeeklyFilteredStocks` 조회만 제공하고 Tier 조회 API가 없어 백테스트 런타임에서 Tier 기반 단순 조회 모델을 사용할 수 없음
- `src/db_setup.py:112`~`154` 에 스키마/인덱스는 준비되었으나 소비 코드가 없어 실제 성능/재현성 개선 효과가 아직 미반영 상태
- `docs/database/schema.md:51` 이후 문서가 최신 스키마(`InvestorTradingTrend`, `DailyStockTier`)를 반영하지 않아 운영/검증 기준이 코드와 분리됨

## 4. (AI가 진행한) 디버그 과정 및 생각한 수정 방안들 구현에 필요한 핵심 변경점
### 4-1. 현재 파이프라인 엔트리/책임 분석
- `src/main_script.py:100`~`124` 에서 Step 2.5 이후 Step 3(OHLCV)로 바로 진행되며, 신규 컬렉터 삽입 지점은 Step 2.5 직후가 가장 자연스러움
- `src/data_pipeline.py:7`~`19` 는 레거시 엔트리로 남아 있고 `stock_data_collector` 중심 호출이라 신규 배치 아키텍처와 분리 운영 필요성이 확인됨

### 4-2. 증분/멱등 패턴 재사용 가능성 점검
- `src/ohlcv_collector.py:15`~`31` 의 `MAX(date)` 워터마크 패턴, `src/ohlcv_collector.py:122`~`130` 의 `INSERT IGNORE` 패턴을 신규 컬렉터에 동일하게 적용 가능
- `src/indicator_calculator.py:74`~`105` 의 마지막 계산일 기준 증분 저장 패턴을 Tier 재계산 구간(`lookback`) 처리에 재사용 가능

### 4-3. 백테스트 연동 시 PIT/성능 제약 확인
- `src/strategy.py:72`~`76` 가 신호일을 T-1로 강제하고 있어, Tier 조회도 동일한 `signal_date` 기준(`date <= as_of_date`)으로 맞춰야 PIT 일관성 유지
- `src/data_handler.py:58`~`66` 의 `assert_point_in_time`를 Tier 조회 API에도 재사용 가능
- 핵심 변경점은 “런타임 반복 계산 제거 + as-of 일괄 조회”이며, 종목별 `MAX(date<=as_of)` 쿼리를 지원해야 누락/왜곡을 줄일 수 있음

### 4-4. 문서/운영 갭 확인
- `docs/database/schema.md`에 신규 2개 테이블 설명이 누락되어 운영 인수인계 리스크가 존재
- 배치 실패 복구 기준(`MAX(date)` 재시작, 최근 N일 재적재 옵션) 문서화가 필요

## 5. (AI가) 파악한 이슈의 원인
- 근본 원인은 **수집/계산 책임이 런타임과 단일 스크립트에 혼재**되어 있고, 배치 단위(백필/일배치)로 분리되지 않은 구조임
- 두 번째 원인은 **스키마 준비와 실행 코드 간 단절**로, DB 확장이 실제 전략/백테스트 성능 개선으로 이어지지 못하는 상태임
- 세 번째 원인은 **PIT-aware Tier 조회 API 부재**로, Tier를 도입해도 look-ahead 없이 안전하게 소비할 경로가 미정인 점임

---

## 6. 생각한 수정 방안들
### 6-1. 방안 A (권장 MVP): `main_script.py`에 신규 배치 Step 직접 연결
- 파일경로: `src/main_script.py`, `src/financial_collector.py`, `src/investor_trading_collector.py`, `src/daily_stock_tier_batch.py`, `src/data_handler.py`
- 무엇을: 기존 Step 2.5와 Step 3 사이에 재무/수급 수집 + Tier 계산 Step을 추가
- 어떻게:
```
if COLLECT_FINANCIAL_DATA: run_financial_batch(mode="daily")
if COLLECT_INVESTOR_TREND: run_investor_batch(mode="daily")
if CALCULATE_DAILY_TIER: run_tier_batch(mode="daily", lookback_days=20)
```
- 왜: 현재 구조를 크게 흔들지 않고, 백필/일배치 분기와 멱등 적재를 가장 빠르게 도입 가능
- 장점: 구현 속도 빠름, 기존 운영 플래그와 일관
- 단점: `main_script.py`가 더 비대해질 수 있음

### 6-2. 방안 B: 배치 오케스트레이터 분리(`pipeline_batch.py`) + 모드 기반 실행
- 파일경로: `src/pipeline_batch.py`(신설), `src/main_script.py`(호출 위임), 신규 collector 3종
- 무엇을: 백필/일배치를 하나의 오케스트레이터에서 `mode=backfill|daily`로 통합 제어
- 어떻게:
```
python -m src.pipeline_batch --mode backfill --start 20150101 --end 20260207
python -m src.pipeline_batch --mode daily --end 20260207
```
- 왜: 스케줄링/운영 관점에서 실행 단위를 명확히 분리하고, 장애 복구 자동화를 쉽게 하기 위함
- 장점: 운영 명령어가 명확, 향후 크론/에어플로우 연동 쉬움
- 단점: 엔트리포인트 증가, 초기 구조 작업량 증가

### 6-3. 방안 C: Collector Registry 패턴 도입
- 파일경로: `src/collector_runner.py`(신설), 기존/신규 collector 인터페이스 정리
- 무엇을: `run_backfill`/`run_daily` 공통 인터페이스를 두고 수집기를 등록형으로 실행
- 어떻게:
```
runner.register("financial", financial_collector)
runner.register("investor", investor_collector)
runner.register("tier", tier_batch)
runner.run(mode="daily")
```
- 왜: 수집기 확장(예: OpenDart, 공매도, 공시 리스크) 시 구조적 확장성을 확보하기 위함
- 장점: 확장성/유지보수성 우수
- 단점: 현재 이슈 범위 대비 설계 비용 큼

### 6-4. 공통 핵심 변경 항목 (세 방안 공통)
- `src/data_handler.py`: `get_stock_tier_as_of`, `get_tiers_as_of`, `get_filtered_stock_codes_with_tier` 추가
- `src/strategy.py`: 신규 진입 후보군에서 Tier 1/2 우선 조회 적용(신호일 T-1 기준 유지)
- `src/db_setup.py`/`docs/database/schema.md`: 신규 테이블 컬럼/인덱스/운영 규칙 동기화
- `tests/`: Tier as-of 조회, PIT 위반 방지, Tier fallback(1->2) 동작 검증 테스트 추가

---

## 7. 최종 결정된 수정 방안 (AI 가 자동 진행하면 안되고 **무조건**/**MUST** 사람에게 선택/결정을 맡겨야 한다)
- 사용자 결정: **방안 B 채택** (`pipeline_batch.py` 기반 오케스트레이션 분리)
- 파일경로: `src/pipeline_batch.py`, `src/financial_collector.py`, `src/investor_trading_collector.py`, `src/daily_stock_tier_batch.py`, `src/data_handler.py`, `src/strategy.py`
- 무엇을: 백필/일배치를 독립 엔트리포인트로 분리하고, 재무/수급 수집 + Tier 사전 계산을 단일 오케스트레이터로 묶음
- 어떻게: `python -m src.pipeline_batch --mode daily|backfill` 명령에서 모듈별 실행/스킵/파라미터를 제어
- 왜: 기존 `main_script.py`의 역할을 유지하면서 운영 배치만 분리해, 속도/재현성 개선과 운영 자동화 용이성을 동시에 확보
### 7-1. 최종 결정 이유 1 (운영 독립성)
- 백필과 일배치 명령을 분리해 장애 복구/재실행/스케줄러 연동이 쉬움
- 전략/백테스트 경로(`main_backtest.py`)와 데이터 적재 경로를 분리하여 장애 전파 범위를 줄임
### 7-2. 최종 결정 이유 2 (리스크 제어)
- `stock_data_collector.py`를 즉시 대체하지 않고 신규 경로를 병행해 회귀 리스크를 낮춤
- `DataHandler`와 테스트를 함께 확장해 런타임 전환 준비를 단계적으로 수행 가능
### 7-3. 최종 결정 이유 3 (확장성)
- 이후 OpenDart/공시/리스크 프록시를 collector 단위로 추가할 수 있는 구조를 선점

---

## 8. 코드 수정 요약
- 배치 오케스트레이터 분리 + 수집기 2종 + Tier 사전계산 + Tier 조회 API + 테스트를 일괄 반영
### 8-1. 오케스트레이터/수집 배치 추가
- [x] `src/pipeline_batch.py:19` 에서 `run_pipeline_batch` 추가
  - `mode=daily|backfill`, `--skip-*`, `--lookback-days`, `--financial-lag-days`를 지원
- [x] `src/financial_collector.py:149` 에서 `run_financial_batch` 추가
  - `FinancialData`를 워터마크(`MAX(date)`) 기반 증분 + `ON DUPLICATE KEY UPDATE` 저장
- [x] `src/investor_trading_collector.py:139` 에서 `run_investor_trading_batch` 추가
  - 투자자 순매수 컬럼 정규화 후 `InvestorTradingTrend` 업서트 저장

### 8-2. Tier 사전 계산 및 조회 API 추가
- [x] `src/daily_stock_tier_batch.py:141` 에서 `build_daily_stock_tier_frame` 추가
  - 20일 평균 거래대금 기반 Tier 산정 + 재무 lag 기반 financial risk override 적용
- [x] `src/daily_stock_tier_batch.py:244` 에서 `run_daily_stock_tier_batch` 추가
  - 백필/일배치 시작일 해석, 재계산 범위 확장, `DailyStockTier` 업서트 저장
- [x] `src/data_handler.py:160` 에서 `get_stock_tier_as_of`, `get_tiers_as_of`, `get_filtered_stock_codes_with_tier` 추가
  - as-of 조회에서 PIT 검증을 재사용하고 런타임 단순 조회 경로 제공
- [x] `src/strategy.py:113` 에서 신규 진입 후보군 Tier 필터 연동(가능 시)
  - Tier API 사용 가능하면 `allowed_tiers=(1,2)`로 후보군 축소, 실패 시 기존 경로 fallback

### 8-3. 검증/문서 동기화
- [x] `tests/test_pipeline_batch.py:11` 오케스트레이터 호출/입력 검증 테스트 추가
- [x] `tests/test_daily_stock_tier_batch.py:12` Tier 계산 로직(유동성 + financial risk) 테스트 추가
- [x] `tests/test_data_handler_tier.py:13` Tier as-of API 테스트 추가
- [x] `docs/database/schema.md:88` `InvestorTradingTrend`, `DailyStockTier` 섹션 및 인덱스 문서화

### 8-4. 운영 적용 체크리스트
- [ ] 운영 DB에 스키마 반영: `python -c "from src.db_setup import get_db_connection, create_tables; conn=get_db_connection(); create_tables(conn); conn.close()"`
- [ ] 초기 백필 실행: `python -m src.pipeline_batch --mode backfill --start-date <YYYYMMDD> --end-date <YYYYMMDD>`
- [ ] 일배치 전환: `python -m src.pipeline_batch --mode daily --end-date <YYYYMMDD>`

---

## 9. 문제 해결에 참고
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
(문제 해결에 참고했던 issue 번호가 포함된 링크 or commit hash)
- issue: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/64`
- issue: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/65`
- issue: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/66`
