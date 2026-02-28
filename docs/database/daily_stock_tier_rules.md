# DailyStockTier Rules (Operational)

Last Updated: 2026-02-28  
Source of Truth (code): `src/pipeline/daily_stock_tier_batch.py`

## 1) 기본 파라미터

- `lookback_days`: `20`
- `danger_liquidity`: `300,000,000`
- `prime_liquidity`: `1,000,000,000`
- `financial_lag_days`: `1`
- `tier1_growth_roe_min`: `10.0`
- `tier1_growth_bps_min`: `0.0`
- `enable_sbv_tier_overlay`: `True`
- `sbv_tier1_demote_threshold`: `0.0139`
- `sbv_tier3_threshold`: `0.0272`
- `sbv_valid_coverage_threshold`: `0.90`

## 2) Tier 산정 순서

### Step A. 유동성 기반 기본 Tier

- `liquidity_20d_avg_value >= prime_liquidity` -> `tier=1`
- `liquidity_20d_avg_value < danger_liquidity` -> `tier=3`
- 그 외 -> `tier=2`

### Step B. 재무 리스크 강등 (Tier3 override)

- `bps <= 0` 또는 `roe < 0` 이면 `tier=3`

### Step C. Tier1 품질 게이트

- Tier1 후보는 아래를 모두 만족해야 유지:
  - `div_yield > 0`
  - `roe >= 10`
  - `bps > 0`
- 미충족 시 `tier=2` 강등

### Step D. (옵션) Investor v1 overlay

- `enable_investor_v1_write=True`인 경우만 적용
- Tier2에서 `flow5 < investor_flow5_threshold` 이면 `tier=3`

### Step E. SBV overlay (유효일에만 적용)

- 날짜별 `sbv_ratio` non-null coverage를 계산
- `coverage >= sbv_valid_coverage_threshold`인 날짜만 SBV 규칙 적용
- 유효일에서:
  - `sbv_ratio >= sbv_tier3_threshold` -> `tier=3`
  - `tier=1` 이면서 `sbv_ratio >= sbv_tier1_demote_threshold` -> `tier=2`

## 3) Tier3가 되는 대표 케이스

- 저유동성: `liquidity_20d_avg_value < 300,000,000`
- 재무위험: `bps <= 0` 또는 `roe < 0`
- (옵션) 수급위험: Tier2 + `flow5` 임계치 하회
- (SBV 유효일) 공매도 과열: `sbv_ratio >= 0.0272`

## 4) reason 토큰

- 기본: `prime_liquidity`, `normal_liquidity`, `low_liquidity`
- 강등/오버레이: `financial_risk`, `div_zero_or_negative`, `tier1_quality_gate_failed`, `investor_flow5`, `sbv_ratio_elevated`, `sbv_ratio_extreme`

## 5) 재계산 명령

```bash
conda run -n rapids-env python -m src.pipeline.batch \
  --mode backfill \
  --start-date 20131120 \
  --end-date 20260224 \
  --skip-financial \
  --skip-investor \
  --financial-lag-days 1
```

참고: `DailyStockTier`의 최대 계산일은 원천 테이블(`DailyStockPrice`, `FinancialData`, `MarketCapDaily`, `ShortSellingDaily`) 최신일에 의해 제한된다.

## 6) 변경 계획 (구현 전)

아래는 **코드 반영 전 계획안**이며, 아직 구현 완료 상태가 아니다.

### 6-1. 설계 원칙

- PIT(lookahead 금지) 유지: `delisted_date`를 실시간 판정 입력으로 사용하지 않음.
- CPU/GPU parity 우선: 복잡 로직은 런타임이 아니라 Tier 배치에서 선계산.
- 런타임 단순화: CPU/GPU는 `DailyStockTier` 조회 중심으로 유지.
- 스키마 안정성: 당장은 Tier 값(`1~3`) 유지, `reason` 토큰 확장 위주.

### 6-2. Tier 배치 고도화 후보 (우선 구현 대상)

- `SBV regime-aware overlay`:
  - 기존 절대 임계(`0.0139`, `0.0272`) + 일자 단면 percentile(`p95`, `p99`) 병행.
- `Liquidity decay gate`:
  - `liquidity_20d_avg_value`와 `60/120일 평균 대비 붕괴 비율` 동시 사용.
- `Price distress gate`:
  - 단기 급락/변동성 스트레스(가격 기반) 조건 시 강등.
- `Financial momentum gate`:
  - 절대값(ROE/BPS) 외에 악화 속도(Delta/변화율) 반영.
- `Data reliability penalty`:
  - 핵심 팩터 결측/신뢰도 저하 시 보수 강등.
- `Quarantine/Hysteresis`:
  - 위험 이벤트 발생 후 N일 재진입 제한(배치 기준 reason 추적).

### 6-3. reason 토큰 확장 계획

- 신규 후보 토큰:
  - `sbv_ratio_regime_elevated`
  - `sbv_ratio_regime_extreme`
  - `liquidity_decay`
  - `price_distress`
  - `financial_momentum_deterioration`
  - `data_reliability_penalty`
  - `risk_quarantine_active`

## 7) CPU/GPU 정합성 계획 (구현 전)

- `open<=0` 체결 불가(No Fill) 정책 유지.
- 쿨다운 갱신 시점 정합화:
  - CPU: 신호 생성 시점이 아니라 **실제 체결 성공 시점** 갱신으로 정렬 예정.
  - GPU: 현재 체결 가능 마스크 기반 갱신과 동일 기준 유지.
- 유니버스 경계 정합화:
  - `delisted_date` 경계식(`>=` vs `>`) 일관화 검토.

## 8) 테스트 계획 (구현 전)

- Tier 배치 규칙 테스트:
  - NaN 처리 일관성, 롤링 경계(off-by-one), 복합 reason 우선순위.
- CPU/GPU parity 테스트:
  - 동일 입력에서 Tier/후보군/체결 결과 일치 여부.
  - 거래정지(open=0) 구간에서 매도 실패 시 쿨다운 동작 일치 여부.
- PIT/유니버스 테스트:
  - 상폐일 경계 케이스 포함.

## 9) 롤아웃 계획 (구현 전)

- Step 1. `shadow`:
  - 신규 reason/tier 영향만 기록하고 전략 진입 규칙에는 미반영.
- Step 2. `gated`:
  - 제한 종목군/기간에서만 반영, 기존 정책과 A/B 비교.
- Step 3. `default`:
  - parity/성능/운영 지표 통과 시 기본값 전환.

핵심 모니터링:
- `tier1_coverage`, `tier2_fallback_rate`, `empty_entry_day_rate`, `tier_churn_rate`
- `liquidation_unfillable_count`, `factor_missing_ratio`, `pit_violation_count`

## 10) 의사결정 필요 항목

- `False Positive`(우량주 과배제) vs `False Negative`(부실주 미검출) 우선순위
- 배치 계산 부하 허용 범위(백필 시간 증가 허용치)
- Quarantine 기간(N일) 및 해제 조건
- SBV percentile 임계값(`p95/p99`)의 기본값
