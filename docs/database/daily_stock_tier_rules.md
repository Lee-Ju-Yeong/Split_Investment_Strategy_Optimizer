# DailyStockTier Rules (Operational)

Last Updated: 2026-02-28  
Source of Truth (code): `src/pipeline/daily_stock_tier_batch.py`

## 1) 이 문서가 설명하는 것

`DailyStockTier`는 "해당 날짜에 이 종목이 얼마나 안전/유동적인가?"를 1~3 단계로 미리 계산해둔 테이블입니다.

- `tier=1`: 상대적으로 양호 (신규 진입 우선)
- `tier=2`: 보통 (조건부 후보)
- `tier=3`: 위험/회피 대상 (신규/추매 제한, 기본은 보유 허용)

핵심은 런타임에서 복잡 계산을 반복하지 않고, 배치에서 미리 계산한 Tier를 CPU/GPU가 공통으로 쓰는 것입니다.

### 1-1) 운영 정책 업데이트 (2026-02-28)

- Tier 기반 **강제청산은 기본 비활성**으로 운영한다.
- Tier는 기본적으로 `Entry/Add` 제어용으로 사용한다.
- 백테스트는 목적별로 2모드로 분리한다.
  - `Research(빠른 탐색)`: survivor-only (상폐 미발생 종목군)
  - `Validation(현실 검증)`: PIT universe (상폐 포함, 강제청산 없이 검증)
- 주의: survivor-only 결과는 `survivorship bias`가 있으므로 최종 채택 기준으로 단독 사용하지 않는다.

### 1-2) Tier 행동 규칙 잠금 (2026-02-28)

- `Tier1`: 신규진입 허용
- `Tier2`: 추가매수 허용
- `Tier3`: 추가매수 유보, `max_inactivity_period`와 연동해 정리
- 비고: Tier 기반 강제청산은 기본 비활성

### 1-3) Entry 랭킹 규칙 잠금 (2026-02-28)

- 순차 조건(cheap -> flow -> atr) 대신 합산 점수 사용
- 점수식:
  - `entry_score = 0.50*cheap_effective + 0.30*flow5_mcap_rank + 0.20*atr_rank`
  - `cheap_effective = cheap_score * cheap_score_confidence`
  - `flow5_mcap_rank`: 당일 후보군 내 `flow5_mcap` 백분위 랭크(0~1)
  - `atr_rank`: 당일 후보군 내 `atr_14_ratio` 백분위 랭크(0~1)
- 정렬: `entry_score desc` -> 동점 시 `market_cap desc` -> 최종 `ticker asc`

## 2) 초보자용 핵심 개념 2개

### `lookback_days`가 뭐야?

- 의미: "최근 며칠을 평균내서 볼지"를 정하는 창(window) 길이
- 현재 값: `20`
- 이 프로젝트에서의 사용: `liquidity_20d_avg_value` 계산
- 해석: 최근 20거래일 평균 거래대금을 보고 유동성을 판단

### `flow5`가 뭐야?

- 의미: 최근 5거래일 동안의 외국인+기관 순매수 누적값
- 계산식:
  - `flow = foreigner_net_buy + institution_net_buy`
  - `flow5 = 최근 5일 flow 합`
- 해석:
  - `flow5`가 크게 음수면, 최근 5일 동안 외국인/기관이 순매도 우위였다는 뜻
  - 옵션 규칙에서 Tier2를 Tier3로 강등하는 보조 신호로 사용

## 3) 변수 사전 (Tier/Ranking 후보 전체)

이 섹션은 "지금 DB에서 바로 쓸 수 있는 변수"와 "지금 데이터로 바로 파생 가능한 변수"를 모두 정리합니다.

### 3-1) DB에 이미 저장된 원천 변수 (Raw Features)

| 분류 | 변수 | 출처 테이블 | 설명 | Tier/Ranking 활용 예시 |
|---|---|---|---|---|
| 가격 | `open_price`, `high_price`, `low_price`, `close_price` | `DailyStockPrice` | 일별 OHLC | 변동성/갭/추세 파생 |
| 가격(수정) | `adj_open`, `adj_high`, `adj_low`, `adj_close`, `adj_ratio` | `DailyStockPrice` | 수정주가 및 비율 | 수정주가 기준 백테스트/지표 |
| 거래 | `volume` | `DailyStockPrice` | 일별 거래량 | 유동성/거래정지 유사 신호 |
| 기술지표 | `ma_5`, `ma_20` | `CalculatedIndicators` | 이동평균 | 추세 필터, 괴리율 |
| 기술지표 | `atr_14_ratio` | `CalculatedIndicators` | ATR 기반 변동성 비율 | 변동성 기반 랭킹/리스크 |
| 밸류 위치 | `price_vs_5y_low_pct` | `CalculatedIndicators` | 5년 저점 대비 위치 | 저평가/반등 후보 보조 |
| 밸류 위치 | `price_vs_10y_low_pct` | `CalculatedIndicators` | 10년 저점 대비 위치 | 장기 저점 근접도 |
| 재무 | `per`, `pbr` | `FinancialData` | 밸류에이션 | 저평가 스코어 |
| 재무 | `eps`, `bps`, `dps` | `FinancialData` | 주당지표 | 수익성/안정성 보조 |
| 재무 | `div_yield` | `FinancialData` | 배당수익률 | Tier1 품질 게이트 |
| 재무 | `roe` | `FinancialData` | 자기자본이익률 | Tier1 품질/리스크 게이트 |
| 수급 | `individual_net_buy` | `InvestorTradingTrend` | 개인 순매수 | 역지표/수급 분해 |
| 수급 | `foreigner_net_buy`, `institution_net_buy` | `InvestorTradingTrend` | 외국인/기관 순매수 | `flow`, `flow5` 파생 |
| 수급 | `total_net_buy` | `InvestorTradingTrend` | 합계 순매수 | 수급 강도 필터 |
| 시총 | `market_cap` | `MarketCapDaily` | 시가총액 | 대형주 편향 제어/랭킹 |
| 주식수 | `shares_outstanding` | `MarketCapDaily` | 상장주식수 | 회전율/유동성 파생 |
| 공매도 | `short_volume`, `short_value` | `ShortSellingDaily` | 공매도 거래량/거래대금 | 공매도 압력 파생 |
| 공매도 | `short_balance`, `short_balance_value` | `ShortSellingDaily` | 공매도 잔고/잔고금액 | `sbv_ratio` 핵심 분자 |
| Tier 결과 | `tier`, `reason` | `DailyStockTier` | 배치 계산된 최종 등급/사유 | 전략 필터 직접 사용 |
| Tier 결과 | `liquidity_20d_avg_value` | `DailyStockTier` | 20일 평균 거래대금 | 유동성 필터/랭킹 |
| Tier 결과 | `sbv_ratio` | `DailyStockTier` | 공매도잔고금액/시총 | 공매도 오버레이 |
| Tier 결과 | `flow5_mcap` | `DailyStockTier` | 5일 외국인+기관 순매수/시총 비율 | 수급 점수 랭킹 |
| Tier 결과 | `pbr_discount`, `per_discount`, `div_premium` | `DailyStockTier` | 밸류/배당 파생 점수(0~1) | cheap 계열 랭킹 |
| Tier 결과 | `cheap_score`, `cheap_score_confidence`, `cheap_score_version` | `DailyStockTier` | 저평가 종합점수/신뢰도/버전 | 우선순위 정렬 핵심 |
| 유니버스 | `snapshot_date`, `market_type` | `TickerUniverseSnapshot` | 시점별 상장 유니버스 | PIT 후보군 제한 |
| 상장이력 | `listed_date`, `last_seen_date`, `delisted_date` | `TickerUniverseHistory` | 상장/관측/상폐 이력 | 백테스트 라벨링/검증 |

### 3-2) 지금 데이터로 바로 파생 가능한 변수 (Derived Features)

아래는 별도 외부데이터 없이 현재 DB만으로 계산 가능한 후보입니다.

| 분류 | 변수(예시) | 계산식(개념) | 용도 |
|---|---|---|---|
| 유동성 | `trading_value` | `close_price * volume` | 거래대금 기준 필터 |
| 유동성 | `liquidity_20d_avg_value` | `trading_value` 20일 평균 | Tier 기본 분류 |
| 유동성 | `liquidity_decay_60_20` | `MA20 / MA60 - 1` | 유동성 붕괴 추세 |
| 수급 | `flow` | `foreigner_net_buy + institution_net_buy` | 당일 수급 방향 |
| 수급 | `flow5`, `flow20` | `flow` 5일/20일 누적 | 수급 압력 강도 |
| 수급 | `flow5_mcap` | `flow5 / market_cap` | 시총 정규화 수급 강도 |
| 수급 | `flow_zscore` | 최근 N일 `flow` 표준화 | 급격한 수급 이상치 |
| 공매도 | `sbv_ratio` | `short_balance_value / market_cap` | 공매도 과열 신호 |
| 공매도 | `short_turnover_ratio` | `short_value / market_cap` | 공매도 거래 압력 |
| 공매도 | `short_volume_share` | `short_volume / volume` | 거래량 대비 공매도 비중 |
| 밸류 | `earnings_yield` | `1 / per` (`per>0`) | 수익수익률 관점 |
| 밸류 | `book_to_price` | `1 / pbr` (`pbr>0`) | 장부가치 관점 |
| 밸류 | `pbr_discount`, `per_discount`, `div_premium` | 과거 분포 내 pct-rank 기반 | cheap_score 구성요소 |
| 밸류 | `cheap_score` | 가중합(`pbr`,`per`,`div`) | 랭킹 우선순위 |
| 밸류 | `cheap_effective` | `cheap_score * confidence` | 결측 보정 점수 |
| 품질 | `financial_risk_flag` | `bps<=0 OR roe<0` | Tier3 즉시 강등 |
| 품질 | `financial_distress_2y` | `2Y(roe<=0 OR bps<=0)` | Tier3 장기 부진 강등 |
| 품질 | `tier1_quality_pass` | `2Y(div_yield>=0 AND roe>=5 AND bps>=0) AND effective_price_vs_5y<=0.33` | Tier1 품질 게이트 |
| 가격행동 | `gap_down_pct` | `(open - prev_close)/prev_close` | distress 보조 |
| 가격행동 | `drawdown_20d` | 최근 20일 고점 대비 하락률 | 위험구간 식별 |
| 가격행동 | `atr_spike` | `atr_14_ratio`의 rolling z-score | 변동성 급등 탐지 |
| 규모/편향 | `size_bucket` | `market_cap` 분위수 버킷 | 대형주 편향 완화 |

### 3-3) 정책 파라미터 (Threshold/Rule Knobs)

아래 값들은 "데이터"가 아니라 규칙의 스위치/임계값입니다.

| 파라미터 | 의미 | 현재 값 |
|---|---|---|
| `lookback_days` | 유동성 평균 윈도우 길이 | `20` |
| `danger_liquidity` | Tier3 저유동 임계 | `300,000,000` |
| `prime_liquidity` | Tier1 유동성 임계 | `1,000,000,000` |
| `financial_lag_days` | 재무 데이터 시차(lag) | `1` |
| `tier1_growth_roe_min` | Tier1 최소 ROE | `5.0` |
| `tier1_growth_bps_min` | Tier1 최소 BPS 하한 | `0.0` |
| `tier1_quality_lookback_days` | Tier1 연속성 확인(거래일) | `504` |
| `tier1_position_gate_max_pct` | 5Y 저점 위치 상한(비율) | `0.33` |
| `tier1_position_gate_start_date` | 5Y 위치 계산 원천을 RAW→Adjusted로 전환하는 날짜 | `2014-11-20` |
| `tier1_position_lookback_days` | 5Y 위치 계산 롤링 윈도우 | `1260` |
| `tier1_position_min_periods_days` | 5Y 위치 계산 최소 관측치 | `252` |
| `enable_investor_v1_write` | 수급 오버레이 사용 여부 | `False`(기본) |
| `investor_flow5_threshold` | Tier2->3 수급 강등 임계 | `-500,000,000` |
| `enable_sbv_tier_overlay` | SBV 오버레이 사용 여부 | `True` |
| `sbv_tier1_demote_threshold` | Tier1->2 SBV 강등 임계 | `0.0139` |
| `sbv_tier3_threshold` | Tier3 SBV 강등 임계 | `0.0272` |
| `sbv_valid_coverage_threshold` | SBV 적용 최소 커버리지 | `0.90` |
| `sbv_tier3_consecutive_days` | SBV Tier3 강등 연속일 | `3` |
| `tier3_flow20_quantile` | flow20_mcap 하위 꼬리 분위수 | `0.10` |
| `tier3_flow20_valid_coverage_threshold` | flow20_mcap 적용 최소 커버리지 | `0.70` |
| `tier3_flow20_consecutive_days` | flow20_mcap Tier3 강등 연속일 | `5` |

### 3-4) 빠른 실무 가이드: 무엇부터 Tier/Ranking에 넣을까?

초기(연산비용 낮음):

- Tier: `liquidity_20d_avg_value`, `financial_risk_flag`, `sbv_ratio`
- Ranking: `cheap_effective`, `flow5_mcap`, `atr_14_ratio` (동점 시 `market_cap`)

중기(연산비용 중간):

- Tier: `liquidity_decay_60_20`, `flow5/flow20`
- Ranking: `gap_down_pct`, `drawdown_20d`, `short_volume_share`

고도화(검증 필요):

- `flow_zscore`, `atr_spike`, 복합 hazard score

## 4) Tier 산정 방법론 (Step-by-step)

### Step A. 유동성으로 1차 분류

- `liquidity_20d_avg_value >= prime_liquidity` -> `tier=1`
- `liquidity_20d_avg_value < danger_liquidity` -> `tier=3`
- 그 사이 -> `tier=2`

직관: 거래가 잘 되는 종목은 우선순위를 높이고, 거래가 너무 빈약한 종목은 위험으로 분류합니다.

### Step B. 재무 리스크 강등 (Tier3 override)

- `bps <= 0` 또는 `roe < 0`이면 `tier=3`
- 추가로 `2Y(roe<=0 OR bps<=0)`이면 `tier=3` (`financial_distress_2y`)

직관: 유동성이 좋아도 재무가 매우 나쁘면 최종적으로 위험군으로 강등합니다.

### Step C. Tier1 품질 게이트

Tier1 후보는 아래 조건을 모두 만족해야 Tier1 유지:

- 최근 `tier1_quality_lookback_days`(기본 504거래일) 동안:
  - `div_yield >= 0`
  - `roe >= 5`
  - `bps >= 0`
- 당일 위치 게이트:
  - 하드게이트는 전 기간 유지: `effective_price_vs_5y <= 0.33`(33%)
  - `effective_price_vs_5y` 산식/원천:
    - `2014-11-19`까지: `CalculatedIndicators.price_vs_5y_low_pct` (RAW OHLC 기반)
    - `2014-11-20`부터: `DailyStockPrice.adj_close/adj_low/adj_high`로 배치 시 재계산
      (252거래일부터 시작해 최대 1260거래일까지 증분 롤링)

미충족 시 `tier=2`로 강등.

직관: "거래량만 큰 종목"을 Tier1에 그대로 두지 않고, 기본 재무 품질을 한 번 더 확인합니다.

### Step D. (옵션) Investor v1 overlay

- 조건: `enable_investor_v1_write=True`일 때만 실행
- 규칙: Tier2 종목에서 `flow5 < investor_flow5_threshold`면 `tier=3`

직관: 최근 수급이 매우 나쁜 종목은 중립(Tier2)에서 위험(Tier3)으로 한 단계 더 낮춥니다.

### Step D-2. (구현 예정) Tier1 `flow20_mcap` 게이트

- 규칙: `flow20_mcap >= 0`인 종목만 Tier1 유지
- 적용 조건: Investor coverage gate 통과일
- coverage 미통과일: 이 게이트는 비활성

### Step D-3. Tier3 수급 꼬리 강등 (flow20_mcap)

- 날짜별 `flow20_mcap` 커버리지 계산
- `coverage >= tier3_flow20_valid_coverage_threshold`인 날짜만 적용
- 적용일에서 날짜별 하위 `tier3_flow20_quantile` 꼬리 구간을 계산
- 해당 꼬리 구간이 `tier3_flow20_consecutive_days` 연속이면 `tier=3` (`flow20_mcap_tail`)

### Step E. SBV overlay (유효일에만 적용)

먼저 "해당 날짜의 SBV 데이터가 충분히 있는가?"를 확인합니다.

- 날짜별 `sbv_ratio` non-null coverage 계산
- `coverage >= sbv_valid_coverage_threshold`인 날짜만 SBV 규칙 적용

유효일에서:

- `sbv_ratio >= sbv_tier3_threshold`가 `sbv_tier3_consecutive_days` 연속이면 `tier=3`
- `tier=1`이면서 `sbv_ratio >= sbv_tier1_demote_threshold` -> `tier=2`

직관: 공매도 잔고가 과열된 종목은 보수적으로 강등합니다.

## 5) Tier3가 되는 대표 케이스

- 저유동성: `liquidity_20d_avg_value < 300,000,000`
- 재무위험: `bps <= 0` 또는 `roe < 0`
- 재무 장기부진: `2Y(roe<=0 OR bps<=0)`
- 수급 꼬리: `flow20_mcap` 하위 꼬리 구간 연속 진입
- (옵션) 수급위험: Tier2 + `flow5` 임계치 하회
- (SBV 유효일) 공매도 과열: `sbv_ratio >= 0.0272`
- 운영 해석: 기본 정책에서 Tier3는 "신규/추매 금지 우선"이며 자동 강제청산 트리거로 사용하지 않는다.

## 6) reason 토큰 설명

| reason 토큰 | 의미 |
|---|---|
| `prime_liquidity` | 유동성 기준 Tier1 |
| `normal_liquidity` | 유동성 기준 Tier2 |
| `low_liquidity` | 유동성 기준 Tier3 |
| `financial_risk` | 재무 리스크로 Tier3 강등 |
| `financial_distress_2y` | 재무 장기 부진(2년 연속)으로 Tier3 강등 |
| `tier1_quality_gate_failed` | Tier1 품질 게이트 실패 |
| `flow20_mcap_tail` | flow20_mcap 하위 꼬리 연속 진입으로 Tier3 강등 |
| `investor_flow5` | 수급(5일 누적) 악화로 강등 |
| `sbv_ratio_elevated` | SBV가 높아 Tier1에서 Tier2로 강등 |
| `sbv_ratio_extreme` | SBV가 매우 높아 Tier3로 강등 |

## 7) 재계산 명령

```bash
conda run -n rapids-env python -m src.pipeline.batch \
  --mode backfill \
  --start-date 20131120 \
  --end-date 20260224 \
  --skip-financial \
  --skip-investor \
  --financial-lag-days 1
```

참고:

- `DailyStockTier`의 최대 계산일은 원천 테이블(`DailyStockPrice`, `FinancialData`, `MarketCapDaily`, `ShortSellingDaily`) 최신일에 의해 제한됩니다.
- `financial_lag_days=1`이면 날짜 `D`의 Tier는 재무 데이터를 `D-1`까지 사용합니다.

## 8) 변경 계획 (구현 전)

아래는 코드 반영 전 계획안입니다.

### 8-1. 설계 원칙

- PIT(lookahead 금지) 유지: `delisted_date`를 실시간 판정 입력으로 사용하지 않음
- CPU/GPU parity 우선: 복잡 로직은 런타임보다 Tier 배치 선계산 우선
- 런타임 단순화: CPU/GPU는 `DailyStockTier` 조회 중심 유지
- 스키마 안정성: 당장은 Tier 값 `1~3` 유지, `reason` 토큰 확장 위주

### 8-2. 배치 고도화 후보

- `SBV regime-aware overlay`: 절대 임계 + 일자 단면 percentile(`p95`, `p99`)
- `Liquidity decay gate`: 60/120일 추세 악화 반영
- `Price distress gate`: 단기 급락/변동성 스트레스 반영
- `Financial momentum gate`: 절대값 외 악화 속도 반영
- `Data reliability penalty`: 결측/신뢰도 저하 페널티
- `Quarantine/Hysteresis`: 위험 이벤트 후 N일 재진입 제한

## 9) CPU/GPU 정합성 계획 (구현 전)

- `open<=0` 체결 불가(No Fill) 정책 유지
- 쿨다운 갱신은 "신호 발생"이 아니라 "실제 체결 성공" 기준으로 정합화 검토
- 유니버스 상폐 경계(`>=` vs `>`) 조건식 일관화 검토

## 10) 테스트/롤아웃 계획 (구현 전)

- 테스트:
  - NaN 처리 일관성
  - 롤링 경계(off-by-one)
  - 복합 reason 우선순위
  - 거래정지(open=0) 구간 parity
- 롤아웃:
  - `shadow` -> `gated` -> `default`
- 핵심 모니터링:
  - `tier1_coverage`, `tier2_fallback_rate`, `empty_entry_day_rate`, `tier_churn_rate`
  - `liquidation_unfillable_count`, `factor_missing_ratio`, `pit_violation_count`

## 11) 백테스트 모드 정책 (2026-02-28)

- `survivor_only` (연구용):
  - 목적: 빠른 파라미터 탐색
  - 특징: 상폐 미발생 종목만 사용
  - 한계: 결과 낙관편향(생존자 편향)
- `pit_no_forced_tier_liquidation` (검증용):
  - 목적: 운영 반영 전 현실 검증
  - 특징: PIT 유니버스(상폐 포함) + Tier 강제청산 비활성
  - 채택 기준: 최종 승격 판단은 이 모드 결과를 우선
