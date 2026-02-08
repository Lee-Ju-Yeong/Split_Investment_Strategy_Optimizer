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

## 6) `CalculatedIndicators` 재계산 실행안

`DailyStockPrice`를 KRX raw 기준으로 재적재한 뒤에는 기존 `CalculatedIndicators`가 stale일 수 있으므로 재계산이 필요하다.

### 6-1) 권장: 전체 재계산 (정합성 우선)

```bash
PYTHONPATH=$PWD conda run --no-capture-output -n rapids-env python -u - <<'PY'
from src.db_setup import get_db_connection
from src import indicator_calculator

conn = get_db_connection()
try:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE CalculatedIndicators")
    conn.commit()
    indicator_calculator.calculate_and_store_indicators_for_all(conn, use_gpu=True)
finally:
    conn.close()
PY
```

### 6-2) 대안: 최근 구간만 재계산 (시간 단축)

```bash
PYTHONPATH=$PWD conda run --no-capture-output -n rapids-env python -u - <<'PY'
from src.db_setup import get_db_connection
from src import indicator_calculator

DELETE_FROM = "2024-01-01"

conn = get_db_connection()
try:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM CalculatedIndicators WHERE date >= %s", (DELETE_FROM,))
    conn.commit()
    indicator_calculator.calculate_and_store_indicators_for_all(conn, use_gpu=True)
finally:
    conn.close()
PY
```

## 7) Tier 규칙 튜닝 초안 (read-only)

아래 초안은 현재 통계(최근 20거래일 평균 거래대금 분위수, 재무 위험 비율, 수급 양수 비율) 기반이다.

### 7-1) 변수 의미

- `lookback_days`
  - 의미: 평균 거래대금 계산에 사용하는 롤링 윈도우(거래일 수)
  - 기본값: `20`
  - 영향: 값이 크면 안정적(완만), 값이 작으면 최근 급변에 민감

- `financial_lag_days`
  - 의미: 재무 데이터의 공시 지연(PIT)을 반영하기 위한 lag 일수
  - 기본값: `45`
  - 영향: 값이 작으면 룩어헤드 위험 증가, 값이 크면 반영 속도 저하

- `danger_liquidity`
  - 의미: 20일 평균 거래대금이 이 값 미만이면 기본 Tier를 `3 (Danger)`로 분류
  - 기본값: `300,000,000`
  - 영향: 값을 높이면 보수적(제외 종목 증가), 낮추면 유니버스 확대

- `prime_liquidity`
  - 의미: 20일 평균 거래대금이 이 값 이상이면 기본 Tier를 `1 (Prime)`로 분류
  - 기본값: `1,000,000,000`
  - 영향: 값을 높이면 Prime이 줄고, 낮추면 Prime이 늘어남

- 현재 Tier 산정 규칙
  - `avg_20d_value >= prime_liquidity` → `tier=1`
  - `avg_20d_value < danger_liquidity` → `tier=3`
  - 그 외 → `tier=2`
  - 추가로 `bps <= 0` 또는 `roe < 0`이면 `tier=3`으로 override

### 7-2) 임계값 후보(A/B/C)

- A안(보수): `danger_liquidity=500,000,000`, `prime_liquidity=2,000,000,000`
- B안(기본): `danger_liquidity=300,000,000`, `prime_liquidity=1,000,000,000`
- C안(완화): `danger_liquidity=100,000,000`, `prime_liquidity=800,000,000`

`DailyStockTier`에 쓰기 전, read-only로 tier 분포를 비교한다.

```bash
PYTHONPATH=$PWD conda run --no-capture-output -n rapids-env python -u - <<'PY'
from src.db_setup import get_db_connection
from src.daily_stock_tier_batch import fetch_price_history, fetch_financial_history, build_daily_stock_tier_frame

START = "2024-01-01"
END = "2026-02-06"
SCENARIOS = {
    "A_conservative": {"lookback_days": 20, "financial_lag_days": 45, "danger_liquidity": 500_000_000, "prime_liquidity": 2_000_000_000},
    "B_baseline": {"lookback_days": 20, "financial_lag_days": 45, "danger_liquidity": 300_000_000, "prime_liquidity": 1_000_000_000},
    "C_relaxed": {"lookback_days": 20, "financial_lag_days": 45, "danger_liquidity": 100_000_000, "prime_liquidity": 800_000_000},
}

conn = get_db_connection()
try:
    price = fetch_price_history(conn, START, END)
    fin = fetch_financial_history(conn, END)
    for name, cfg in SCENARIOS.items():
        df = build_daily_stock_tier_frame(price_df=price, financial_df=fin, **cfg)
        dist = (df["tier"].value_counts(normalize=True).sort_index() * 100).round(2).to_dict()
        print(name, {"rows": len(df), "tier_pct": dist})
finally:
    conn.close()
PY
```

분포 비교 후 최종안을 고르면 그때 `pipeline_batch`로 실제 재적재한다.

## 8) Investor 포함 read-only 검증 결과 (2026-02-08)

Codex 2개 + Gemini(`gemini-3-pro-preview`) 교차 검토 후, 아래 3개 시나리오를 read-only로 비교했다.

### 8-1) 비교 시나리오

- A안: `danger=100m`, `prime=1b` + `tier1`에서 `flow_ratio20 < -0.02`이면 `tier2` 강등
- B안: `danger=300m`, `prime=1b` + `flow_ratio20 <= -0.03`이면 `tier`를 1단계 강등
- C안: `danger=300m`, `prime=1b` + `tier2`에서 `flow5 < -500,000,000`이면 `tier3` 강등

### 8-2) read-only 결과 요약

- A안: `tier1=22.23%`, `tier2=45.38%`, `tier3=32.39%`, `flow_impact=9.79%`, `churn=2.90%`
- B안: `tier1=23.76%`, `tier2=23.45%`, `tier3=52.79%`, `flow_impact=13.77%`, `churn=3.82%`
- C안: `tier1=32.03%`, `tier2=18.80%`, `tier3=49.17%`, `flow_impact=1.88%`, `churn=2.59%`

관찰:
- A/B는 수급 규칙 영향도가 커서(`flow_impact` 과다) v1 최소 규칙으로는 보수적이지 않다.
- C는 영향도가 낮고(`1.88%`) 분포 교란이 작아 v1 운영 안정성에 유리하다.
- 현재 데이터 구간에서 `forward 20d downside monotonicity`는 모든 시나리오에서 미충족이었고, 이는 임계값 문제와 별개로 데이터 커버리지/구조 영향이 크므로 참고 지표로만 사용한다.

### 8-3) v1 확정안 (운영 기준)

- `lookback_days = 20`
- `financial_lag_days = 45`
- `danger_liquidity = 300,000,000`
- `prime_liquidity = 1,000,000,000`
- 재무 위험 override: `bps <= 0 OR roe < 0 => tier=3`
- 수급 최소 규칙(v1): `tier2`에서만 `flow5 < -500,000,000`이면 `tier3` 강등
- 결측 규칙: `InvestorTradingTrend` 미존재 날짜/종목은 수급 규칙 미적용(기본 tier 유지)

### 8-4) 적용 순서

1. 위 파라미터로 read-only shadow를 1주간 유지
2. `flow_impact`, `tier churn`, `tier distribution` 일별 점검
3. 이상 없으면 `daily_stock_tier_batch`에 수급 규칙을 write 경로로 반영
