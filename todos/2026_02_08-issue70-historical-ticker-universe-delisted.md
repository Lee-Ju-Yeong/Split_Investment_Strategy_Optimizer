# feat(data): 상폐 포함 Historical Ticker Universe 구축
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/70`
- 목적: 상장폐지 종목 누락으로 인한 survivorship bias를 줄이기 위해 시점기준 티커 유니버스를 구축

## 1. 배경
- 현재 OHLCV 수집 유니버스는 `WeeklyFilteredStocks`/`CompanyInfo` 기반이며, 과거 상폐 종목 누락 가능성이 존재
- 장기 백테스트 신뢰도를 높이려면 상장/상폐 이력 기반 유니버스가 필요

## 2. 목표
- `TickerUniverseSnapshot` + `TickerUniverseHistory`를 구축해 기간 교집합 기반 수집 토대를 마련
- 기존 파이프라인과 충돌 없이 단계적으로 전환

## 3. 범위
### 3-1. Phase 1 (이번 단계)
- [x] `db_setup.py`에 `TickerUniverseSnapshot`, `TickerUniverseHistory` 테이블 및 인덱스 추가
- [x] 스냅샷 수집 배치(`ticker_universe_batch.py`) 추가
- [x] 스냅샷으로부터 히스토리(`listed_date`, `delisted_date`) 집계 로직 추가
- [x] 검증 SQL/운영 절차 문서화

### 3-2. Phase 2 (후속)
- [x] `ohlcv_batch` 유니버스 소스를 History 기반으로 전환
- [x] `start_date~end_date` 교집합 기준으로 수집 대상 산정

## 4. 완료 조건
- [x] 스키마 생성 및 인덱스 반영 확인
- [x] 배치 1회 실행 후 샘플 상폐 종목이 History에 존재함을 확인
- [x] TODO/README에 운영 순서 반영

## 5. 검증 쿼리
- [x] `SELECT COUNT(*) FROM TickerUniverseSnapshot;`
- [x] `SELECT COUNT(*) FROM TickerUniverseHistory;`
- [x] `SELECT COUNT(*) FROM TickerUniverseHistory WHERE listed_date > delisted_date;`
- [x] `SELECT COUNT(*) FROM TickerUniverseHistory WHERE delisted_date IS NULL;`

## 6. 운영 절차 (권장안 A: 개발/검증 우선, 운영 순차)
1. 스키마 반영: `python -c "from src.db_setup import get_db_connection, create_tables; conn=get_db_connection(); create_tables(conn); conn.close()"`
2. Phase 1 백필: `python -m src.ticker_universe_batch --mode backfill --start-date 20100101 --end-date 20260207 --step-days 7 --workers 1`
3. 검증 SQL 실행: 5장 쿼리 + 샘플 상폐 종목 확인
4. 일배치 전환: `python -m src.ticker_universe_batch --mode daily --end-date <YYYYMMDD>`

## 7. 실행 로그 (샘플 검증)
- 스모크(일배치): `python -m src.ticker_universe_batch --mode daily --end-date 20260207 --workers 1 --log-interval 1`
- 스모크(병렬 백필): `python -m src.ticker_universe_batch --mode backfill --start-date 20260201 --end-date 20260207 --step-days 3 --workers 2 --log-interval 1`
- 검증 결과:
  - `TickerUniverseSnapshot`: `8320`
  - `TickerUniverseHistory`: `2774`
  - `listed_date > delisted_date`: `0`
  - `delisted_date IS NULL`: `2773`
  - 상폐(비활성) 샘플: `stock_code=454640`, `delisted_date=2026-02-01`

## 8. Phase 2 반영 메모
- `src/ohlcv_batch.py`:
  - `TickerUniverseHistory` 우선 조회(`listed_date`, `last_seen_date`, `delisted_date`) + legacy fallback(`WeeklyFilteredStocks`/`CompanyInfo`)
  - 티커별 lifetime과 요청 구간 `[start_date, end_date]` 교집합으로 수집 범위 산정
  - `resume` 시 교집합 범위 내에서 `latest_saved_date + 1`부터 이어받기
- `tests/test_ohlcv_batch.py`:
  - 교집합 산정, history 우선 선택, legacy fallback, resume window 테스트 추가
