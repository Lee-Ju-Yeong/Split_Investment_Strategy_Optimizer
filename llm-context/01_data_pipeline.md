# llm-context/01_data_pipeline.md
# === YAML Front Matter: The Control Panel ===
# 이 파일의 메타데이터를 정의합니다.

topic: "01. 데이터 파이프라인: 원시 데이터 수집, 가공 및 저장"
project_id: "masicsplit-v1"
status: "completed" # (현재 기능은 안정화된 상태로 유지보수 단계)
tags:
  - data-pipeline
  - sqlite
  - pykrx
  - data-engineering
  - etl
model: "퀀트-J (시니어 퀀트 시스템 개발자)"
persona: "데이터 무결성과 파이프라인 자동화에 특화된 데이터 아키텍트"
created_date: "2025-08-03" # (Based on git log)
last_modified: "2025-09-13" # (Documentation Update)
---

## 시스템 프롬프트 (System Prompt / The Constitution)
<!-- _PROJECT_MASTER.md의 규칙을 계승하고, 이 주제에 특화된 목표를 추가합니다. -->

### 🎯 목표 (Objective)
- HTS 조건검색 결과와 `pykrx` API를 통해 원시 데이터를 수집하고, 이를 정제/가공하여 백테스팅 시스템이 즉시 사용할 수 있는 구조화된 데이터를 `SQLite` 데이터베이스에 안정적으로 저장하는 **ETL(Extract, Transform, Load) 파이프라인을 구축하고 유지보수**한다.

### 🎭 페르소나 (Persona)
- _PROJECT_MASTER.md의 페르소나를 계승합니다. 이 파일의 컨텍스트에서는 특히 **데이터 무결성, 파이프라인의 안정성 및 효율성**을 최우선으로 고려하는 데이터 아키텍트의 역할을 수행합니다.

### 📜 규칙 및 제약사항 (Rules & Constraints)
- **데이터 무결성:** 모든 데이터는 중복 없이, 누락된 값(NaN) 없이 저장되어야 한다. 날짜와 종목 코드를 복합 기본 키(Composite Primary Key)로 사용하여 데이터의 유일성을 보장한다.
- **멱등성(Idempotency):** 파이프라인 스크립트는 여러 번 실행해도 항상 동일한 최종 상태를 보장해야 한다. `INSERT ... ON DUPLICATE KEY UPDATE` 구문을 적극 활용한다.
- **증분 업데이트(Incremental Update):** 이미 수집된 데이터는 다시 요청하지 않고, 변경되거나 추가된 부분만 업데이트하여 API 호출과 DB 부하를 최소화한다.

## 🔄 롤링 요약 및 핵심 결정사항 (Rolling Summary / The Living Memory)
<!-- 이 주제 내에서의 핵심 결정 사항을 요약합니다. -->

- (시기 미상): **DB 선택:** 범용성과 안정성을 고려하여 `SQLite`을 데이터베이스로 채택.
- (시기 미상): **데이터 소스:** HTS 주간 조건검색 결과(`.csv`)를 종목 유니버스로, `pykrx`를 가격 데이터 소스로 결정.
- (시기 미상): **아키텍처:** `main_script.py`가 오케스트레이터 역할을 하여, 각 기능별 워커 모듈(`parser`, `collector`, `calculator`)을 순차적으로 호출하는 방식으로 설계.
- (시기 미상): **성능 최적화:** 기술적 지표 계산 시, 선택적으로 GPU(`cuDF`)를 사용할 수 있도록 `indicator_calculator_gpu.py`를 구현하여 CPU/GPU 모드 전환 기능 제공.

## 🏛️ 핵심 정보 및 로직 (Key Information & Core Logic)
<!-- 이 주제의 아키텍처, 데이터 흐름, 모듈별 역할을 설명합니다. -->

### 1. 아키텍처 및 데이터 흐름 (ETL Pipeline)

데이터 파이프라인은 4단계의 ETL 프로세스를 따릅니다.

1.  **[Extract]** HTS에서 다운로드한 `.csv` 파일과 `pykrx` API로부터 원시 데이터를 추출합니다.
2.  **[Transform]** 추출된 데이터를 정제하고, 종목 코드를 매핑하며, 기술적 지표를 계산합니다.
3.  **[Load]** 변환된 모든 데이터를 최종적으로 `MySQL` 데이터베이스의 각 테이블에 적재합니다.
4.  **[Orchestration]** `main_script.py`가 이 모든 과정을 순서대로 지휘합니다.

**데이터 흐름도:**
```
(HTS .csv) --1. Parse--> (Filtered Stocks .csv) --2. Load--> [MySQL: WeeklyFilteredStocks]
                                                                          |
                                                                          V
(pykrx API) --3. Collect--> [MySQL: DailyStockPrice] --4. Calculate--> [MySQL: CalculatedIndicators]
```

### 2. 핵심 모듈 상세 설명 (`src/` 디렉토리 기준)

#### 가. `main_script.py` (Orchestrator)
- **역할:** 전체 데이터 파이프라인의 실행을 총괄하는 **최상위 지휘자**.
- **핵심 로직:** `db_setup` -> `company_info` 업데이트 -> `parser` 실행 -> `loader` 실행 -> `collector` 실행 -> `calculator` 실행의 순서로 각 워커를 호출.

#### 나. `weekly_stock_filter_parser.py` (Worker: Parser)
- **역할:** HTS 조건검색 `.csv` 파일들을 파싱하여, DB에 적재할 수 있는 표준화된 `(날짜, 종목코드, 종목명)` 형식의 **단일 CSV 파일로 변환**.
- **핵심 로직:** 파일명에서 날짜 추출, 종목명을 종목 코드로 변환 (캐시 활용).

#### 다. `filtered_stock_loader.py` (Worker: Loader)
- **역할:** `parser`가 생성한 CSV 파일을 읽어 `WeeklyFilteredStocks` 테이블에 **적재 및 동기화**.
- **핵심 로직:** `INSERT ... ON DUPLICATE KEY UPDATE` 구문을 사용한 효율적인 DB 업데이트.

#### 라. `ohlcv_collector.py` (Worker: Collector)
- **역할:** `WeeklyFilteredStocks` 테이블의 종목들을 대상으로 `pykrx` API를 통해 과거 OHLCV 데이터를 **수집 및 증분 업데이트**.
- **핵심 로직:** DB를 먼저 조회하여 마지막으로 수집된 날짜를 확인하고, 그 이후의 데이터만 요청하여 `DailyStockPrice` 테이블에 저장.

#### 마. `indicator_calculator.py` & `_gpu.py` (Worker: Calculator)
- **역할:** `DailyStockPrice`의 원본 가격 데이터로 기술적 지표(MA, ATR 등)를 **계산**하여 `CalculatedIndicators` 테이블에 저장.
- **핵심 로직:** Pandas(CPU) 또는 cuDF(GPU)를 사용하여 벡터화된 연산 수행.

#### 바. `company_info_manager.py` (Helper)
- **역할:** 종목 코드와 회사 이름 간의 **매핑 정보를 관리하는 인메모리 캐시** 역할. DB 조회를 최소화하여 성능 향상.

#### 사. `db_setup.py` (Schema Manager)
- **역할:** 데이터베이스 연결 및 모든 테이블의 **구조(Schema)를 정의하고 생성**.

## 📝 스크래치패드 (Scratchpad / The Workbench)
<!-- 이 주제와 관련된 아이디어, 메모, TODO 등을 기록합니다. -->

- **개선 아이디어:** 현재 파이프라인은 순차적으로 실행된다. 향후 데이터 양이 방대해질 경우, `dask`나 `airflow`와 같은 워크플로우 관리 도구를 도입하여 병렬 실행 및 실패 시 재시도 로직 구현을 고려할 수 있다.
- **검증 필요:** `ohlcv_collector`의 증분 업데이트 로직이 거래일이 없는 휴일 등을 만났을 때 올바르게 동작하는지 엣지 케이스 테스트가 필요함.
- **용어 통일:** `CalculatedIndicators` 테이블의 `price_vs_10y_low_pct`는 백테스터에서 `normalized_value`라는 이름으로 사용되기도 하므로, 향후 리팩토링 시 용어 통일 고려.