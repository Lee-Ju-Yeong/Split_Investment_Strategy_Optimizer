# feat(planning): pykrx 확장 데이터셋 + Tier v2 로드맵
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/71`
- 목적: pykrx에서 추가 수집 가능한 데이터를 Tier 고도화에 단계적으로 반영하기 위한 실행 로드맵 고정

## 0. 진행 현황 (2026-02-14)
- [x] Phase P0 테이블 DDL/인덱스 추가: `MarketCapDaily`, `ShortSellingDaily`
- [x] 수집 워커 추가: `src/market_cap_collector.py`, `src/short_selling_collector.py`
- [x] 배치 엔트리 확장: `src/pipeline_batch.py`에 `--run-marketcap`, `--run-shortsell` 옵션 추가
- [ ] pykrx source health-check preflight 유틸/가드(전량 empty 패턴 fail-fast) 공통화

## 1. 배경
- 현재 운영 축은 `DailyStockPrice` / `FinancialData` / `InvestorTradingTrend` / `DailyStockTier`
- Tier는 유동성 + 재무 위험 중심이며, 수급/공매도/외국인 보유/섹터 히스토리 반영이 제한적
- 대량 백필 진행 중이므로, 신규 데이터는 우선순위 기반 단계 도입이 필요

## 2. 목표
1. 추가 수집 DB 우선순위(P0/P1/P2) 확정
2. Tier 반영 순서(MVP -> Stage2 -> Stage3) 확정
3. PIT/운영 규칙(announce/effective, 결측 대응) 고정
4. Optuna 기반 파라미터 탐색 적용 범위/전제조건 고정

## 3. 데이터 우선순위
### 3-0. PyKrx API 매핑 (구현용, 2026-02-14 업데이트)

아래는 각 데이터셋을 어떤 pykrx API로 적재할지의 "원천 매핑"이다.

- 교차검토(Gemini) 요약:
  - P0(필수): `MarketCapDaily`, `ShortSellingDaily`
  - P1(성능/해석력 개선): `ForeignOwnershipDaily`, `SectorClassificationHistory`(스냅샷 + SCD Type2)
  - 제외/보류:
    - `get_stock_major_changes`: 원천 자체에 데이터가 없는 것으로 확인(health-check/preflight에 포함 금지)
    - OHLCV/가격변동: 이미 `DailyStockPrice`에서 처리(중복 수집 금지)
    - 투자자 매매/순매수: 이미 `InvestorTradingTrend`에서 처리(중복 수집은 drift/불일치 위험)
    - ETF/ETN/ELW: Tier v2(주식 후보군/리스크) MVP 범위 밖(필요 시 별도 이슈로 분리)

- `MarketCapDaily`
  - API: `pykrx.stock.get_market_cap(<date>)` (전종목), `get_market_cap_by_ticker(<start>, <end>, <ticker>)`
  - 비고: `market_cap`, `shares_outstanding`는 수급/공매도 지표의 정규화 분모로 사용
- `ShortSellingDaily`
  - API(권장): `pykrx.stock.get_shorting_status_by_date(<start>, <end>, <ticker>)` (기간/종목)
  - API(보조): `get_shorting_volume_by_ticker(<date>, <market>)`, `get_shorting_value_by_ticker(<date>, <market>)` (단, volume/value만; balance 계열은 별도 처리 필요)
  - 비고: 공매도는 KRX 정책상 최근일(T+2) 지연이 있으므로 preflight에서 날짜 선택을 보수적으로 한다.
- `ForeignOwnershipDaily`
  - API: `pykrx.stock.get_exhaustion_rates_of_foreign_investment(<date>, <market>)`
  - 비고: 외국인 "flow(순매수)"와 "stock(보유/한도)"를 분리해 해석하기 위한 신호
- `SectorClassificationHistory`
  - API: `pykrx.stock.get_market_sector_classifications(<date>)`
  - 비고: `get_stock_major_changes`가 원천 자체 empty인 상황에서는, 섹터 히스토리를 "스냅샷 수집 + 변경 감지(SCD Type2)"로 재구축한다.
- `IndexDaily`, `IndexConstituentHistory`
  - API: `pykrx.stock.get_index_ohlcv(<start>, <end>, <index_ticker>)`, `get_index_ticker_list(<date>)`, `get_index_portfolio_deposit_file(<index_ticker>, <date?>)`

### 3-1. Phase P0 (필수)
- `MarketCapDaily`
  - 필드: `date`, `stock_code`, `market_cap`, `shares_outstanding`
  - 용도: 수급/공매도 신호 시총 정규화
- `ShortSellingDaily`
  - 필드: `date`, `stock_code`, `short_volume`, `short_value`, `short_balance`, `short_balance_value`
  - 용도: 하방 압력/리스크 경보

### 3-2. Phase P1 (성능 개선)
- `ForeignOwnershipDaily`
  - 필드: `date`, `stock_code`, `foreign_holding_shares`, `foreign_exhaustion_rate`
  - 용도: 순매수(flow)와 보유잔량(stock) 분리 해석
- `SectorClassificationHistory`
  - 필드: `stock_code`, `sector_code`, `sector_name`, `announce_date`, `effective_date`, `end_date`
  - 용도: 섹터 상대강도 및 구조 변화 반영
  - 수집/적재 방식(권장):
    - `get_market_sector_classifications(date)`를 주기적으로(예: weekly) 스냅샷 저장
    - 종목별로 `sector_code/sector_name` 변경 감지 시 기존 row의 `end_date`를 닫고 신규 row를 생성(SCD Type2)
    - pykrx 원천에서 `announce_date/effective_date`를 직접 제공하지 않으므로, 기본은 `effective_date=snapshot_date`, `announce_date=NULL` 또는 `announce_date=effective_date`로 저장한다.

### 3-3. Phase P2 (고도화)
- `IndexDaily`
- `IndexConstituentHistory`
- 용도: 레짐/상대강도/멀티팩터 가중치 고도화

### 3-4. Phase P3 (Optimization, 후행)
- `Optuna Study` (Tier v2 score/threshold 탐색)
  - 용도: Tier v2 임계값/가중치 자동 탐색
  - 전제: `#67` 후보군 모드 고정, `#56` parity pass, 백필/일배치 안정화 이후 실행
  - Scope v1:
    - Allowed: Tier v2 수치 파라미터(임계값/가중치), robust selection 수치 파라미터
    - Forbidden: `candidate_source_mode`, 체결/수수료/호가/lag 규칙, WFO fold 분할 규칙

## 4. Tier 반영 순서
### 4-1. MVP
- 기존(유동성 + 재무 위험) + 외국인/기관 순매수 강도

### 4-2. Stage2
- 시총 정규화 수급
- 공매도 압력(거래/잔고)
- 외국인 한도 소진율

### 4-3. Stage3
- 섹터 상대강도
- 인덱스 레짐
- WFO 기반 가중치 튜닝

### 4-4. Stage4 (Optuna 적용)
- `#68` robust score/hard gate를 objective/constraint로 사용
- `optuna_enabled=false` 기본값 유지, 실험 브랜치/실험 config에서만 활성화
- 산출물: best trial + top-n trial + search space + seed/기간 메타데이터 저장
- 추천 평가 구조:
  - Deterministic baseline(결정론적 점수 선택)으로 비교 기준 고정
  - Seeded stress test(고정 seed 다회)로 단일 종목/시드 민감도 점검
  - Jackknife(상위 기여 종목 제거 재실행)로 outlier 의존성 점검

## 5. PIT/운영 규칙
- 변경성 데이터는 `announce_date`, `effective_date` 저장
- 재무/분류 데이터는 lag 반영 후 신호 계산
- 결측 시 가중치 재정규화, 핵심 신호 다중 결측 시 `Tier3` 강등
- 배치 실패는 부분성공 플래그 + 최근 N일 재수집으로 복구
- pykrx source health-check preflight를 수집 배치 공통 가드로 고정
  - 최소 체크(권장):
    - `get_market_ticker_list(<date>) > 0`
    - 샘플 `get_market_cap(<date>)` non-empty
    - 샘플 `get_market_fundamental(<date>)` non-empty
    - 섹터 스냅샷을 도입한 경우 `get_market_sector_classifications(<date>)` non-empty
    - 공매도는 `T+2` 지연을 고려해 `today-3~today-10` 범위의 적절한 거래일로 샘플 호출
  - fail 조건:
    - 전량 empty 응답 패턴 감지 시 해당 배치 `blocked/skip` 처리(무의미한 full run 방지)
  - 주의:
    - `get_stock_major_changes`는 원천 자체에 데이터가 없는 것으로 확인된 상태이므로, 다른 수집 배치의 preflight에 포함하지 않는다(별도 external_blocked 처리).
- Optuna는 lookahead 방지 규칙을 동일 적용하고, 학습/검증 구간 분리(WFO fold 단위)로만 실행
- Optuna 결과 운영 반영은 feature flag로 점진 전환하고, 이상 징후 시 legacy 설정으로 즉시 롤백

### 5-1. Optuna Preflight Gate (Non-negotiable)
- `#67` 후보군 모드/정책 고정(`candidate_source_mode`, `tier=1 -> <=2 fallback`)
- `#56` 최신 parity 인증 증적 존재 + mismatch `0건`
- `#68` hard gate 설정값 존재(`median(OOS/IS)`, `fold_pass_rate`, `OOS_MDD_p95`)
- 데이터 스냅샷 고정: 테이블 row/min/max/hash 기록(드리프트 감지 시 run 무효)
- 실행 재현성 확인: 동일 trial 재실행 hash 불일치 시 즉시 중단/무효화

## 6. 체크리스트
- [ ] P0 테이블 DDL/인덱스 확정
- [ ] 수집 배치 엔트리(`pipeline_batch`) 확장(일/주/월)
- [ ] Tier v2 read-only 실험 스크립트 추가
- [ ] PIT/왜곡 방지 검증 항목 테스트화
- [ ] pykrx source health-check 유틸/가드 추가(전량 empty 패턴 fail-fast)
- [ ] `MarketCapDaily` 적재 단계에서 `Common Stock`만 포함되도록 종목 마스터/유니버스와 조인해 제외 규칙 고정(ETF/ETN/ELW/SPAC 등)
- [ ] 거래정지/비정상 거래일 파생 플래그(halt/zero-volume) 정책 정의(매수 제한/리스크 대응용)
- [ ] (선택) `get_market_fundamental(date)` 기반 `FundamentalDaily` 병행 수집 여부 결정(일별 trailing PER/PBR 등)
- [ ] `SectorClassificationHistory` 스냅샷 수집 + SCD Type2 적재 워커 추가
- [ ] Optuna 실험 스크립트/설정 추가(`robust_score` objective, seed 고정)
- [ ] Optuna 전제조건 체크(후보군 모드/Parity/데이터 커버리지) 자동 가드 추가
- [ ] Optuna 산출물 저장 규격 정의(`trial params`, `score`, `gate pass/fail`, `metadata`)
- [ ] Optuna invalid trial 기준 명문화(`INVALID_REPRO`, `INVALID_PARITY`, `INVALID_DATA`)
- [ ] Optuna run manifest 저장(`config hash`, `data hash`, `env fingerprint`, `git sha`)
- [ ] mode 전환(`hybrid_transition` -> `tier`) 시 study 분리 강제(혼합 비교 금지)
- [ ] `docs/database/schema.md` 및 `TODO.md` 동기화

## 7. 완료 기준
- P0 적재 일배치 1주 무장애
- Tier v2가 기존 대비 최소 1개 리스크 지표 개선(MDD 또는 tail loss)
- PIT 위반 테스트 0건
- Optuna ON/OFF 실험 재현 가능(seed/기간/모드 고정) 및 legacy 롤백 절차 문서화 완료
- Optuna 승격 전 하드게이트 동시 통과: parity `0건` + robust gate 3종 통과 + stress/jackknife 통과

## 8. 운영 결정 메모 (2026-02-22)

### 8-1. 데이터 품질 확인 결과(기준일: 2026-02-03 단면)
- `ShortSellingDaily`는 현재 실질적으로 `short_balance_value`만 안정적으로 적재됨
  - `short_volume/short_value/short_balance`는 운영 DB에서 비어 있는 상태
- 따라서 Tier v2 공매도 신호는 임시로 `sbv_ratio = short_balance_value / market_cap` 단일 경로를 사용
- 분포(2721 종목):
  - `sbv_ratio` p75=`0.0047`, p95=`0.0146`, p99=`0.0297`
  - `liquidity_20d_avg_value` p50=`855,682,139`, p75=`4,740,323,724`

### 8-2. 다중 에이전트 검토 요약
- 검토 방식: 내부 3-agent(`reviewer`, `explorer`, `default`)로 `Conservative/Balanced/Aggressive` 비교
- 합의:
  - Hard-risk는 Tier3를 먼저 적용해야 함 (`bps<=0`, `roe<0`, 비정상/결측, 과도한 `sbv_ratio`, 과도한 ATR)
  - Tier1은 고유동성 + 낮은 `sbv_ratio` 조건으로 좁게 정의
  - 현재 단계에서 운영 기본안은 `Balanced`가 가장 현실적
- 분할 결과(샘플 시뮬레이션, 2721 종목):
  - Conservative: `Tier1=282`, `Tier2=666`, `Tier3=1773`
  - Balanced: `Tier1=298`, `Tier2=878`, `Tier3=1545`
  - Aggressive: `Tier1=974`, `Tier2=808`, `Tier3=939`

### 8-3. Balanced 기준(임시 운영안)
- Tier3 (hard-risk 우선):
  - `liquidity_20d_avg_value < 855,682,139` 또는 결측
  - `market_cap <= 0` 또는 결측
  - `sbv_ratio >= 0.0146` 또는 계산불가
  - `atr_14_ratio >= 0.4` 또는 결측
  - `bps <= 0` 또는 `roe < 0`
- Tier1:
  - `liquidity_20d_avg_value >= 4,740,323,724`
  - `sbv_ratio < 0.0047`
  - `atr_14_ratio < 0.2`
  - `bps > 0`, `roe >= 0` (결측은 Tier1 제외)
- Tier2:
  - 위 조건에 해당하지 않는 나머지

```sql
CASE
  WHEN liquidity_20d_avg_value IS NULL OR liquidity_20d_avg_value < 855682139 THEN 3
  WHEN market_cap IS NULL OR market_cap <= 0 THEN 3
  WHEN short_balance_value IS NULL THEN 3
  WHEN atr_14_ratio IS NULL OR atr_14_ratio >= 0.4 THEN 3
  WHEN (bps IS NOT NULL AND bps <= 0) OR (roe IS NOT NULL AND roe < 0) THEN 3
  WHEN (short_balance_value / market_cap) >= 0.0146 THEN 3
  WHEN liquidity_20d_avg_value >= 4740323724
       AND (short_balance_value / market_cap) < 0.0047
       AND atr_14_ratio < 0.2
       AND (bps IS NULL OR bps > 0)
       AND (roe IS NULL OR roe >= 0) THEN 1
  ELSE 2
END AS tier
```

### 8-4. 후속 액션
- [ ] `DailyStockTier` 계산 경로에 위 Balanced 규칙을 `read-only shadow`로 추가
- [ ] shadow 결과(`tier 분포`, `기존 대비 이동률`, `최근 20영업일 안정성`) 검증 후 default 전환 여부 결정
- [ ] `short_volume/short_value/short_balance` 수집 정상화 및 컬럼 매핑 drift 방지 가드 추가

## 9. 코드 수정 요약 (2026-02-22)

### 9-1. 신규 진입 우선순위 및 필터링
- [x] `src/backtest/cpu/strategy.py`에서 Tier 후보 중 보유 종목/쿨다운 종목을 제외한 뒤 정렬하도록 유지/명시
- [x] `src/backtest/cpu/strategy.py` 신규 진입 정렬 기준을 `ATR -> market_cap -> ticker` 순으로 변경
- [x] `src/backtest/gpu/engine.py`, `src/backtest/gpu/utils.py` 정렬 기준을 CPU와 동일하게 `ATR -> market_cap -> ticker`로 통일
- [x] `src/data_handler.py` `load_stock_data` 쿼리에 `MarketCapDaily.market_cap` 조인을 추가해 CPU 랭킹 분모 확보

### 9-2. 운영 모니터링 지표
- [x] `src/backtest/cpu/backtester.py`에 `empty_entry_day_rate`, `tier1_coverage`, `tier2_fallback_rate` 집계 추가
- [x] `src/backtest/cpu/backtester.py` 실행 종료 시 `self.last_run_metrics` 및 `portfolio.run_metrics`에 지표 저장

### 9-3. 회귀 테스트 보강
- [x] `tests/test_issue67_tier_universe.py`에 `ATR 동률 시 market_cap -> ticker` 우선순위 테스트 추가
- [x] `tests/test_issue67_tier_universe.py`에 보유/쿨다운 제외 선행 필터 테스트 추가
- [x] `src/backtest/gpu/data.py`에 `_collect_candidate_atr_asof` 호환 헬퍼를 추가해 기존 테스트 경로와 호환 유지
