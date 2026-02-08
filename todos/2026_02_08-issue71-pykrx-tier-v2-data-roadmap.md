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

## 5. PIT/운영 규칙
- 변경성 데이터는 `announce_date`, `effective_date` 저장
- 재무/분류 데이터는 lag 반영 후 신호 계산
- 결측 시 가중치 재정규화, 핵심 신호 다중 결측 시 `Tier3` 강등
- 배치 실패는 부분성공 플래그 + 최근 N일 재수집으로 복구

## 6. 체크리스트
- [ ] P0 테이블 DDL/인덱스 확정
- [ ] 수집 배치 엔트리(`pipeline_batch`) 확장(일/주/월)
- [ ] Tier v2 read-only 실험 스크립트 추가
- [ ] PIT/왜곡 방지 검증 항목 테스트화
- [ ] `docs/database/schema.md` 및 `TODO.md` 동기화

## 7. 완료 기준
- P0 적재 일배치 1주 무장애
- Tier v2가 기존 대비 최소 1개 리스크 지표 개선(MDD 또는 tail loss)
- PIT 위반 테스트 0건
