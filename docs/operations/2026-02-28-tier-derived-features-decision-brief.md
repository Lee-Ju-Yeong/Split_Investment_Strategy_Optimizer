# Tier/Derived Features Decision Brief

Last Updated: 2026-02-28
Related: `#71`, `#101`

## 1) 목적

- 목표 A: 파라미터 최적화 성과를 높이되 과최적화를 피한다.
- 목표 B: PIT 원칙을 지키면서 상장폐지 위험 종목이 `Tier1`에 들어올 가능성을 낮춘다.
- 목표 C: CPU/GPU parity를 깨지 않는 방식으로 확장한다.

## 2) 현재까지 확정된 사실

- Tier 계산 최신일은 원천 데이터 기준 `2026-02-06`.
- 현재 `Tier1` 품질 게이트:
  - `2Y(div_yield>=0 AND roe>=5 AND bps>=0) AND effective_price_vs_5y<=0.33`
- 현재 `Tier3` 강등 핵심:
  - `bps <= 0 OR roe < 0` (즉시)
  - `2Y(roe<=0 OR bps<=0)` (`financial_distress_2y`)
  - `flow20_mcap` 하위 꼬리 연속 진입 (`flow20_mcap_tail`)
  - `sbv_ratio >= 0.0272` 연속일 조건 (`sbv_ratio_extreme`)
- 추가매수/청산 정책:
  - 추가매수는 `tier<=2`만 허용
  - Tier 기반 강제청산은 비활성(손절/비활성기간으로만 청산)
- 데이터 커버리지(2013-11-20~2026-02-06):
  - `InvestorTradingTrend` row coverage vs `DailyStockPrice`: `94.69%`
  - `MarketCapDaily` row coverage vs `DailyStockPrice`: `99.99%`
  - `shares_outstanding` non-null coverage vs `DailyStockPrice`: `99.99%`
- `CalculatedIndicators`는 현재 raw OHLC 기반 계산 경로다.
  - 즉, `price_basis=adjusted` 백테스트와 일부 지표 기준이 혼합될 수 있음.

## 3) 합의된 설계 원칙

- Tier는 Safety gate(하드 필터), Ranking은 Alpha score(소프트 정렬)로 분리.
- 조인/롤링이 큰 파생변수는 런타임 계산 금지, `DailyStockTier` 배치 선계산.
- `delisted_date` 등 사후 확정값은 실시간 gate 입력으로 직접 사용하지 않음.
- 결측은 0으로 채우지 않는다.
  - `missing_flag`: 데이터가 없으면 `1`, 있으면 `0`
  - `confidence`: 데이터가 충분/신선하면 `1`에 가깝고, 결측이 많으면 `0`에 가까운 신뢰도 점수
  - 활용: 점수는 `raw_score * confidence`로 약화하고, `missing_flag=1`이면 Tier1 진입을 제한한다.
- SBV/수급은 coverage gate를 통과한 날짜에만 강하게 반영.

## 4) 의사결정 상태 (2026-02-28)

### 4-1. 확정

1. Tier 체계:
   - `3-tier 유지`
2. Tier 강제청산:
   - 기본 운영에서 비활성 (`force_tier_liquidation = false`)
3. 백테스트 우선 경로:
   - 빠른 탐색은 `survivor-only`를 우선 허용
   - 단, 최종 승격 전에는 PIT 검증 모드 재평가를 필수로 둠

### 4-2. 적용 상태 (코드 반영)

1. Ranking 기본축(동점 포함):
   - `entry_score = 0.50*cheap_effective + 0.30*flow_score + 0.20*atr_score`
   - 정렬: `entry_score desc -> market_cap desc -> ticker asc`
2. Tier3 수급 강등:
   - `flow20_mcap` 하위 꼬리 구간 연속 진입 시 `tier=3`
3. 지표 기준(raw/adjusted):
   - `effective_price_vs_5y`는 `2014-11-19`까지 raw 지표, `2014-11-20`부터 adjusted 재계산값을 사용
4. 세부 기준 Source of Truth:
   - `docs/database/daily_stock_tier_rules.md`

## 5) 파생변수 활용안 (Tier vs Ranking)

### 5-1. Tier에 넣는 변수 (Safety / Hard gate 우선)

- Tier 목적: "이 종목을 살 수 있는가?"
- 원칙: 상폐/환금성/급격한 훼손 리스크를 먼저 차단

1. `capital_impairment_flag` (`bps <= 0`)
   - 사용: 위험 tier로 강등(기본 정책에서는 Entry/Add 차단)
2. `liquidity_20d_avg_value`
   - 사용: 저유동성 차단 (`<50M~100M`은 위험 강등, `<300M`은 Watch)
3. `sbv_ratio`
   - 사용: 공매도 과열 강등 (`>=0.0272` 위험, Tier1에서 `>=0.0139`면 Tier2 강등)
4. `tier1_quality_pass` (`2Y(div_yield>=0 AND roe>=5 AND bps>=0) AND effective_price_vs_5y<=0.33`)
   - 사용: Tier1 진입 필수 조건
5. `turnover_20d` (도입 권장)
   - 사용: 거래 가능성 보강 게이트(결측/극저 값은 Tier1 제외)
6. `drawdown_20d`, `atr_spike_20` (도입 권장)
   - 사용: 단기 붕괴/변동성 쇼크 구간 Watch 또는 위험 강등
7. `flow20_mcap` (도입 권장)
   - 사용: 중기 수급 급이탈 구간은 Tier1 제외 또는 Watch 강등

### 5-2. Ranking에 넣는 변수 (Alpha / Soft score)

- Ranking 목적: "통과한 종목 중 무엇을 먼저 살 것인가?"
- 원칙: 하드 제외 대신 점수 가중으로 우선순위 차등

1. `cheap_effective = cheap_score * cheap_score_confidence` (핵심)
2. `flow5_mcap` (단기 수급 모멘텀)
3. `ma_array_score` 또는 `price_mom_1m/12m` (추세)
4. `atr_14_ratio` (변동성 점수; 높은 변동 우선)
5. `market_cap` 또는 `size_bucket` (대형주 편향 제어용 tie-break)

### 5-3. 추천 기본 점수식 (초안)

- 후보군: `Tier1` 우선, 부족 시 `Tier2` fallback
- 정렬 점수(예시):
  - `entry_score = 0.50*cheap_effective + 0.30*flow5_mcap_rank + 0.20*atr_rank`
  - 동점 시 `market_cap desc`, 최종 동점은 `ticker asc`
- 결측 처리:
  - 결측은 0으로 채우지 않고 `missing_flag` 기록
  - `confidence`로 가중 축소(`raw_score * confidence`)
  - coverage gate 미달 신호는 해당 날짜 비활성
## 6) 후속 작업 체크리스트

### 6-1. MVP (권장: 1~2주)

- [ ] Tier/Ranking 분리 정책을 코드 상수/설정으로 고정
- [ ] `DailyStockTier` 파생변수 추가(우선 8~10개)
  - `liquidity_decay_20_60`, `turnover_20d`, `flow5_mcap`, `flow20_mcap`
  - `drawdown_20d`, `atr_spike_20`, `sbv_mom_5d`
  - `earnings_yield`, `book_to_price`
- [ ] 결측 정책 공통 함수 도입(`safe_div`, `missing_flag`, `confidence`, coverage gate)
- [ ] CPU/GPU 공통 조회 경로에 신규 컬럼 연결
- [ ] 단위 테스트/회귀 테스트 추가

### 6-2. Phase 2 (선택)

- [ ] `4-tier` 도입 시 엔진 파라미터화(`entry_max_tier`, `add_max_tier`, `liquidation_min_tier`)
- [ ] `volatility_zscore`, `flow_zscore`, `shares_dilution_60d` 등 확장 변수 추가
- [ ] WFO 탐색공간에 tier 임계 파라미터 포함

## 7) 검증 게이트 (운영 반영 전)

- [ ] PIT 위반 0건
- [ ] CPU/GPU parity mismatch 0건 (결정 경로 기준)
- [ ] Backfill 후 지표 커버리지 리포트 확인
- [ ] OOS 기준에서 기존 대비 리스크 지표(MDD/tail) 개선 확인

## 8) 참고 문서

- `docs/database/daily_stock_tier_rules.md`
- `docs/operations/gemini_derived_features_review_package.md`
- `todos/2026_02_08-issue71-pykrx-tier-v2-data-roadmap.md`
