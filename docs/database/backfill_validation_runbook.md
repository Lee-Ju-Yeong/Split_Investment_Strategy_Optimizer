# OHLCV Backfill Validation Runbook

`DailyStockPrice` 전기간 재적재(History 유니버스 + `adjusted=False`) 이후 정합성을 빠르게 확인하기 위한 운영 체크리스트.

## 1) 실행 중 모니터링

```bash
ps -ef | rg 'src\.ohlcv_batch' | rg -v rg
```

```bash
# 현재 적재 행수/커버리지 확인 (진행 중에도 조회 가능)
python - <<'PY'
from src.db_setup import get_db_connection
conn = get_db_connection()
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM DailyStockPrice")
rows = cur.fetchone()[0]
cur.execute("SELECT COUNT(DISTINCT stock_code), MIN(date), MAX(date) FROM DailyStockPrice")
tickers, dmin, dmax = cur.fetchone()
print({"rows": rows, "tickers": tickers, "min_date": dmin, "max_date": dmax})
cur.close()
conn.close()
PY
```

## 2) 완료 직후 필수 검증(SQL)

```sql
SELECT @@hostname AS host, @@datadir AS datadir, DATABASE() AS db_name;
```

```sql
SELECT COUNT(*) AS rows_total,
       COUNT(DISTINCT stock_code) AS tickers_total,
       MIN(date) AS min_date,
       MAX(date) AS max_date
FROM DailyStockPrice;
```

```sql
SELECT stock_code, COUNT(*) AS row_count
FROM DailyStockPrice
GROUP BY stock_code
ORDER BY row_count DESC
LIMIT 20;
```

```sql
-- PK(date, stock_code) 기준 중복 여부 (0이어야 정상)
SELECT COUNT(*) - COUNT(DISTINCT CONCAT(date, ':', stock_code)) AS duplicate_like_rows
FROM DailyStockPrice;
```

```sql
-- 미래 데이터 유입 차단 확인
SELECT COUNT(*) AS future_rows
FROM DailyStockPrice
WHERE date > CURDATE();
```

## 3) 완료 후 후속 순서

1. `DailyStockPrice` 검증 통과 확인
2. `FinancialData`/`InvestorTradingTrend` 최신 구간 증분 동기화
3. `DailyStockTier` 재계산 (`daily_stock_tier_batch`)
4. `CalculatedIndicators` 재계산 (가격 재적재 구간 기준)

## 4) 병렬 실행 금지 항목 (백필 도중)

- 다른 OHLCV 백필/일배치 동시 실행
- 대량 DDL/인덱스 재생성/테이블 truncate
- 동일 테이블에 대량 update를 유발하는 보정 배치(`adj_close`/`adj_ratio`)

## 5) 장애 복구 원칙

- 기본은 `resume=True` 재실행 (동일 명령 재사용)
- 재현성 검증이 필요하면 해당 구간만 삭제 후 재실행
- `--allow-legacy-fallback`은 운영 안정화 전까지 기본 비활성 유지
