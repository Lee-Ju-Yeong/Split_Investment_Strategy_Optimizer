# 데이터베이스 스키마 정의서

기준 코드: `src/db_setup.py`  
이 문서는 코드의 실제 `CREATE TABLE`/인덱스와 일치하도록 유지합니다.

## 1. Core Backtest Tables

### 1-1. `CompanyInfo`
종목 기본 정보 캐시 테이블.

| 컬럼명 | 타입 | 제약 | 설명 |
| :-- | :-- | :-- | :-- |
| `stock_code` | `VARCHAR(20)` | `PRIMARY KEY` | 종목 코드 |
| `company_name` | `VARCHAR(255)` |  | 종목명 |
| `market_type` | `VARCHAR(50)` |  | 시장 구분 |
| `last_updated` | `DATETIME` |  | 마지막 갱신 시각 |

### 1-2. `WeeklyFilteredStocks`
주간 필터링 유니버스 저장 테이블.

| 컬럼명 | 타입 | 제약 | 설명 |
| :-- | :-- | :-- | :-- |
| `filter_date` | `DATE` | `PRIMARY KEY` | 필터 기준일 |
| `stock_code` | `VARCHAR(20)` | `PRIMARY KEY` | 종목 코드 |
| `company_name` | `VARCHAR(255)` |  | 종목명 |

### 1-3. `DailyStockPrice`
일별 OHLCV 저장 테이블.

| 컬럼명 | 타입 | 제약 | 설명 |
| :-- | :-- | :-- | :-- |
| `stock_code` | `VARCHAR(20)` | `PRIMARY KEY` | 종목 코드 |
| `date` | `DATE` | `PRIMARY KEY` | 기준일 |
| `open_price` | `DECIMAL(20,5)` |  | 시가 |
| `high_price` | `DECIMAL(20,5)` |  | 고가 |
| `low_price` | `DECIMAL(20,5)` |  | 저가 |
| `close_price` | `DECIMAL(20,5)` |  | 종가 |
| `adj_close` | `DECIMAL(20,5)` | `NULL` | 수정종가 (파생/보정용) |
| `adj_ratio` | `DECIMAL(20,10)` | `NULL` | 원종가 대비 수정비율 |
| `volume` | `BIGINT` |  | 거래량 |

### 1-4. `CalculatedIndicators`
백테스트용 선계산 지표 테이블.

| 컬럼명 | 타입 | 제약 | 설명 |
| :-- | :-- | :-- | :-- |
| `stock_code` | `VARCHAR(6)` | `PRIMARY KEY` | 종목 코드 |
| `date` | `DATE` | `PRIMARY KEY` | 기준일 |
| `ma_5` | `FLOAT` | `NULL` | 5일 이동평균 |
| `ma_20` | `FLOAT` | `NULL` | 20일 이동평균 |
| `atr_14_ratio` | `FLOAT` | `NULL` | ATR 비율 |
| `price_vs_5y_low_pct` | `FLOAT` | `NULL` | 5년 저점 대비 위치 |
| `price_vs_10y_low_pct` | `FLOAT` | `NULL` | 10년 저점 대비 위치 |

## 2. Strategy Expansion Tables

### 2-1. `FinancialData`
재무 팩터 저장 테이블(PER/PBR/EPS/BPS/DPS/DIV/ROE).

| 컬럼명 | 타입 | 제약 | 설명 |
| :-- | :-- | :-- | :-- |
| `stock_code` | `VARCHAR(20)` | `PRIMARY KEY` | 종목 코드 |
| `date` | `DATE` | `PRIMARY KEY` | 기준일 |
| `per` | `FLOAT` | `NULL` | PER |
| `pbr` | `FLOAT` | `NULL` | PBR |
| `eps` | `FLOAT` | `NULL` | EPS |
| `bps` | `FLOAT` | `NULL` | BPS |
| `dps` | `FLOAT` | `NULL` | DPS |
| `div_yield` | `FLOAT` | `NULL` | 배당수익률 |
| `roe` | `FLOAT` | `NULL` | ROE |
| `source` | `VARCHAR(50)` | `DEFAULT 'pykrx'` | 데이터 소스 |
| `updated_at` | `DATETIME` | `ON UPDATE CURRENT_TIMESTAMP` | 갱신 시각 |

인덱스:
- `idx_financial_date_stock` ON `(date, stock_code)`

### 2-2. `InvestorTradingTrend`
투자자별 순매수 저장 테이블.

| 컬럼명 | 타입 | 제약 | 설명 |
| :-- | :-- | :-- | :-- |
| `stock_code` | `VARCHAR(20)` | `PRIMARY KEY` | 종목 코드 |
| `date` | `DATE` | `PRIMARY KEY` | 기준일 |
| `individual_net_buy` | `BIGINT` | `DEFAULT 0` | 개인 순매수 |
| `foreigner_net_buy` | `BIGINT` | `DEFAULT 0` | 외국인 순매수 |
| `institution_net_buy` | `BIGINT` | `DEFAULT 0` | 기관 순매수 |
| `total_net_buy` | `BIGINT` | `DEFAULT 0` | 합계 순매수 |
| `updated_at` | `DATETIME` | `ON UPDATE CURRENT_TIMESTAMP` | 갱신 시각 |

인덱스:
- `idx_investor_date_stock` ON `(date, stock_code)`
- `idx_investor_date_flow` ON `(date, foreigner_net_buy, institution_net_buy)`

### 2-3. `DailyStockTier`
사전 계산된 일별 종목 등급 저장 테이블.

| 컬럼명 | 타입 | 제약 | 설명 |
| :-- | :-- | :-- | :-- |
| `date` | `DATE` | `PRIMARY KEY` | 기준일 |
| `stock_code` | `VARCHAR(20)` | `PRIMARY KEY` | 종목 코드 |
| `tier` | `TINYINT` | `NOT NULL` | 1(Prime)/2(Normal)/3(Danger) |
| `reason` | `VARCHAR(255)` |  | 등급 사유 |
| `liquidity_20d_avg_value` | `BIGINT` | `NULL` | 20일 평균 거래대금 |
| `computed_at` | `DATETIME` | `DEFAULT CURRENT_TIMESTAMP` | 계산 시각 |

인덱스:
- `idx_tier_stock_date` ON `(stock_code, date)`
- `idx_tier_date_tier_stock` ON `(date, tier, stock_code)`

### 2-4. `TickerUniverseSnapshot`
시점(PIT) 기준 종목 유니버스 스냅샷 테이블.

| 컬럼명 | 타입 | 제약 | 설명 |
| :-- | :-- | :-- | :-- |
| `snapshot_date` | `DATE` | `PRIMARY KEY` | 스냅샷 기준일 |
| `stock_code` | `VARCHAR(20)` | `PRIMARY KEY` | 종목 코드 |
| `market_type` | `VARCHAR(20)` |  | 시장 구분(KOSPI/KOSDAQ/KONEX) |
| `company_name` | `VARCHAR(255)` | `NULL` | 종목명 |
| `source` | `VARCHAR(50)` | `DEFAULT 'pykrx'` | 데이터 소스 |
| `updated_at` | `DATETIME` | `ON UPDATE CURRENT_TIMESTAMP` | 갱신 시각 |

인덱스:
- `idx_tus_stock_date` ON `(stock_code, snapshot_date)`
- `idx_tus_date_market_stock` ON `(snapshot_date, market_type, stock_code)`

### 2-5. `TickerUniverseHistory`
상장/상폐를 포함한 종목 이력 집계 테이블.

| 컬럼명 | 타입 | 제약 | 설명 |
| :-- | :-- | :-- | :-- |
| `stock_code` | `VARCHAR(20)` | `PRIMARY KEY` | 종목 코드 |
| `listed_date` | `DATE` | `NOT NULL` | 최초 관측(상장)일 |
| `last_seen_date` | `DATE` | `NOT NULL` | 마지막 관측일 |
| `delisted_date` | `DATE` | `NULL` | 상장폐지일(추정/확정 반영) |
| `latest_market_type` | `VARCHAR(20)` | `NULL` | 최신 시장 구분 |
| `latest_company_name` | `VARCHAR(255)` | `NULL` | 최신 종목명 |
| `source` | `VARCHAR(50)` | `DEFAULT 'snapshot_agg'` | 집계 소스 |
| `updated_at` | `DATETIME` | `ON UPDATE CURRENT_TIMESTAMP` | 갱신 시각 |

인덱스:
- `idx_tuh_listed_date` ON `(listed_date)`
- `idx_tuh_last_seen_date` ON `(last_seen_date)`
- `idx_tuh_delisted_date` ON `(delisted_date)`

## 3. Legacy Tables

아래 테이블은 레거시 파이프라인(`src/data_pipeline.py`, `src/stock_data_collector.py`)에서 사용합니다.

### 3-1. `stock_data`
레거시 통합 가격/재무 저장 테이블.

### 3-2. `ticker_list`
레거시 티커 마스터 테이블.

### 3-3. `ticker_status`
레거시 수집 상태 테이블.

## 4. 배치 운용 메모
- 스키마 반영: `python -c "from src.db_setup import get_db_connection, create_tables; conn=get_db_connection(); create_tables(conn); conn.close()"`
- 백필/일배치: `python -m src.pipeline_batch --mode backfill|daily ...`
- 유니버스 배치: `python -m src.ticker_universe_batch --mode backfill|daily ...`
- OHLCV 백필: `python -m src.ohlcv_batch --start-date <YYYYMMDD> --end-date <YYYYMMDD> ...`
- 운영 전후 검증: `SHOW TABLES`, `SHOW INDEX FROM <table>`로 인덱스 존재 확인
