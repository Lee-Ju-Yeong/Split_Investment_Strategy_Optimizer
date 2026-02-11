# feat(planning): pykrx 확장 데이터셋 + Tier v2 로드맵
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/71`
- 목적: pykrx에서 추가 수집 가능한 데이터를 Tier 고도화에 단계적으로 반영하기 위한 실행 로드맵 고정

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
  - 최소 체크: `get_market_ticker_list(<date>) > 0`, 샘플 `get_stock_major_changes(<largecap>)` 응답 확인
  - fail 조건: 전량 empty 응답 패턴 감지 시 해당 배치 `blocked/skip` 처리(무의미한 full run 방지)
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
