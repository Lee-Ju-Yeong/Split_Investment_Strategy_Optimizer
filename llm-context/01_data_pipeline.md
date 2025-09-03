# llm-context/01_data_pipeline.md
# === YAML Front Matter ===
topic: "01. 데이터 파이프라인: 원시 데이터 수집 및 가공"
status: "completed" # (현재 기능은 안정화된 상태)
tags:
  - data-pipeline
  - mysql
  - pykrx
  - data-engineering
  - parsing
---
# === System Prompt / Core Instructions ===
# 이 파일의 목적은 데이터 파이프라인을 구성하는 모든 코드의 역할과 데이터 흐름을 명확히 문서화하는 것입니다.

# === Rolling Summary & Key Decisions ===
# 데이터 파이프라인 구축 과정에서의 주요 결정 사항입니다.

- **DB 선택:** 범용성과 안정성을 고려하여 `MySQL`을 데이터베이스로 채택.
- **데이터 소스:**
  1.  **종목 유니버스:** HTS에서 다운로드한 주간 조건검색 결과(`.csv`)를 초기 종목 소스로 사용.
  2.  **기업 정보 및 가격 데이터:** `pykrx` 라이브러리를 통해 KOSPI/KOSDAQ의 공식 데이터를 수집.
- **아키텍처:** `main_script.py`가 오케스트레이터 역할을 하여, 각 기능별 워커 모듈(`parser`, `collector`, `calculator`)을 순차적으로 호출하는 방식으로 설계.
- **성능 최적화:** 대량의 기술적 지표 계산 시, 선택적으로 GPU(`cuDF`)를 사용할 수 있도록 `indicator_calculator_gpu.py`를 구현하여 CPU/GPU 모드 전환 기능 제공.

---
# === Key Information & Core Logic ===
# 이 섹션은 데이터 파이프라인의 아키텍처와 각 모듈의 핵심 역할을 설명합니다.

## 1. 아키텍처 및 데이터 흐름

데이터 파이프라인은 외부의 정형/비정형 데이터를 수집하여 백테스팅 시스템이 사용할 수 있는 깨끗하고 구조화된 데이터로 변환 후, `MySQL` 데이터베이스에 저장하는 것을 목표로 합니다. 전체 프로세스는 아래와 같은 순서로 진행됩니다.

**[HTS .csv 파일] → ① Parser → [Filtered Stocks .csv] → ①.⑤ Loader → ② Collector (Info) →  → ④ Calculator (Indicators) → [MySQL DB]**



---

## 2. 핵심 모듈 상세 설명 (`src/` 디렉토리 기준)

### 가. `main_script.py` (Orchestrator)
- **역할:** 전체 데이터 파이프라인의 실행을 총괄하는 **최상위 지휘자(오케스트레이터)**입니다.
- **핵심 로직:**
    1.  DB 연결을 설정하고 필요한 모든 테이블이 존재하는지 확인/생성 (`db_setup.py` 호출).
    2.  `company_info_manager.py`를 호출하여 최신 종목 정보를 DB에 업데이트하고, 캐시를 로드합니다.
    3.  `weekly_stock_filter_parser.py`를 실행하여 HTS `.csv` 파일들을 파싱하고, 필터링된 종목 리스트를 `WeeklyFilteredStocks` 테이블에 저장합니다.
    4.  `ohlcv_collector.py`를 실행하여 위에서 필터링된 종목들의 전체 기간 OHLCV 데이터를 `pykrx`로부터 수집하고 `DailyStockPrice` 테이블에 저장합니다.
    5.  `indicator_calculator.py`를 실행하여 저장된 가격 데이터 기반으로 기술적 지표(MA, ATR 등)를 계산하고 `CalculatedIndicators` 테이블에 저장합니다.
- **실행 플래그:** `USE_GPU`, `UPDATE_COMPANY_INFO_DB` 등의 boolean 플래그를 통해 각 단계를 선택적으로 실행할 수 있는 유연성을 제공합니다.

### 나. `company_info_manager.py` (Helper)
- **역할:** 한국 주식 시장의 모든 종목 코드와 회사 이름 간의 **매핑 정보를 관리**합니다.
- **핵심 기능:**
    - `update_company_info_from_pykrx`: `pykrx`를 통해 최신 상장 정보를 가져와 `CompanyInfo` 테이블을 업데이트합니다.
    - `load_company_info_cache_from_db`: DB의 `CompanyInfo` 테이블 데이터를 프로그램 실행 시 **인메모리 캐시(Dictionary)**로 로드하여, 이후의 잦은 DB 조회를 방지하고 성능을 향상시킵니다.

### 다. `weekly_stock_filter_parser.py` (Worker: Parser)
- **역할:** 특정 포맷을 가진 HTS 조건검색 결과 `.csv` 파일들을 파싱하여 **최종 필터링 종목 리스트 CSV 파일을 생성**합니다.
- **핵심 로직:**
    1.  지정된 폴더 내의 모든 `.csv` 파일을 순회합니다.
    2.  파일명에서 '필터링 날짜'를 추출합니다.
    3.  파일 내용에서 '종목명' 리스트를 추출합니다.
    4.  `company_info_manager`의 캐시를 사용하여 '종목명'을 '종목 코드'로 변환합니다.
    5.  ** 최종적으로 `(날짜, 종목코드, 종목명)` 데이터를 **종합 CSV 파일로 출력**합니다. (DB 저장 역할은 `filtered_stock_loader.py`로 이전)
    
    

### 라. `ohlcv_collector.py` (Worker: Collector)
- **역할:** `WeeklyFilteredStocks` 테이블에 있는 모든 종목에 대해 `pykrx` API를 사용하여 **과거 OHLCV(시가, 고가, 저가, 종가, 거래량) 데이터를 수집**합니다.
- **핵심 로직:**
    - `get_latest_ohlcv_date_for_ticker`: DB를 먼저 조회하여 각 종목별로 이미 데이터가 어디까지 수집되었는지 확인합니다.
    - `pykrx.stock.get_market_ohlcv`: 마지막으로 저장된 날짜 다음날부터 데이터를 요청하여 **증분 업데이트(Incremental Update)**를 수행, 불필요한 API 호출을 최소화합니다.
    - 수집된 데이터는 `DailyStockPrice` 테이블에 저장됩니다.

### 마. `indicator_calculator.py` & `indicator_calculator_gpu.py` (Worker: Calculator)
- **역할:** `DailyStockPrice` 테이블의 원본 가격 데이터를 바탕으로, 전략에 필요한 **파생 데이터(기술적 지표)를 계산**합니다.
- **`indicator_calculator.py` (CPU):**
    - `pandas` 라이브러리의 `rolling()` 함수 등을 사용하여 이동평균, ATR, 주가 위치 비율 등을 계산합니다.
    - 단일 종목에 대해 순차적으로 계산을 수행합니다.
- **`indicator_calculator_gpu.py` (GPU):**
    - Pandas DataFrame을 `cuDF` DataFrame으로 변환하여 GPU 메모리로 전송합니다.
    - CPU 버전과 거의 동일한 코드로 GPU 상에서 **병렬 계산**을 수행하여 훨씬 빠른 속도를 제공합니다.
    - 계산된 결과는 다시 Pandas DataFrame으로 변환되어 `CalculatedIndicators` 테이블에 저장됩니다.

### 바. `db_setup.py` (Schema)
- **역할:** 데이터베이스 연결(`get_db_connection`) 및 모든 테이블의 **구조(Schema)를 정의하고 생성(`create_tables`)**하는 역할을 담당합니다.
- **주요 테이블:** `CompanyInfo`, `WeeklyFilteredStocks`, `DailyStockPrice`, `CalculatedIndicators` 등.

### 사. `filtered_stock_loader.py` (Worker: Loader)
- **역할:** `weekly_stock_filter_parser`가 생성한 최종 필터링 CSV 파일을 읽어 `WeeklyFilteredStocks` 테이블에 **적재(Load) 및 업데이트**합니다.
- **핵심 로직직:** `INSERT ... ON DUPLICATE KEY UPDATE` 구문을 사용하여, 새로운 데이터는 추가하고 기존 데이터는 갱신하는 방식으로 DB와 CSV 파일 간의 데이터 동기화를 효율적으로 처리합니다.


---
# === Scratchpad / Notes Area ===
- **개선 아이디어:** 현재 데이터 파이프라인은 순차적으로 실행된다. 향후 데이터 양이 방대해질 경우, `dask`나 `airflow`와 같은 워크플로우 관리 도구를 도입하여 각 스텝의 병렬 실행 및 실패 시 재시도 로직을 구현하는 것을 고려해볼 수 있다.
- **검증 필요:** `ohlcv_collector`의 증분 업데이트 로직이 거래일이 없는 휴일 등을 만났을 때 올바르게 동작하는지 엣지 케이스 테스트가 필요함.
- `CalculatedIndicators` 테이블에 저장되는 지표 중 `price_vs_10y_low_pct`는 백테스터에서 `normalized_value`라는 이름으로 사용되기도 합니다. 