# Work Note: `ShortSellingDaily` Publication Lag / PIT

> Type: `implementation`
> Status: `draft`
> Priority: `P0`
> Last updated: 2026-03-07
> Related issues: `#67`
> Gate status: `open`

## 1. Summary
- What: `ShortSellingDaily` 데이터를 `DailyStockTier`에 반영할 때, 실제 가용 시점보다 이르게 쓰지 않도록 PIT 계약을 고정합니다.
- Why: 현재 수집기는 lag를 보수적으로 보지만, tier batch는 same-date join으로 `sbv_ratio`를 만들고 있어 룩어헤드 후보가 됩니다.
- Current status: 문제 정의와 체크리스트만 정리된 초안입니다.
- Next action: `ShortSellingDaily.date`가 거래일인지 공시 가능일인지 먼저 확정합니다.

## 2. Fixed Problem Statement
- 수집기에는 `lag_trading_days` 개념이 이미 있습니다.
- 그러나 tier batch는 `ShortSellingDaily.date == MarketCapDaily.date` 기준으로 same-date join을 하고 있습니다.
- 이 구조가 진짜 PIT-safe인지 아직 명확하지 않습니다.

## 3. Current Plan
- [ ] `ShortSellingDaily.date` 의미 확정
- [ ] `publication_lag_trading_days` 정책 고정
- [ ] Tier join 규칙 결정
  - `same-date` 유지 금지 여부
  - `available_from_date` 저장 여부
- [ ] `sbv_ratio` shadow 재계산 경로 정의
- [ ] 회귀 테스트 추가
- [ ] backfill / recompute / rollback 계획 수립

## 4. Acceptance Criteria
- `DailyStockTier.sbv_ratio`가 미래 공시 데이터를 참조하지 않음
- representative sample에서 `future_reference_count=0` 또는 동등한 PIT 증적 존재
- overlay 수정 후 CPU/GPU candidate selection parity에 신규 mismatch를 만들지 않음
- shadow diff와 backfill 영향 요약이 문서 또는 운영 아티팩트로 남음

## 5. Open Questions
- pykrx short selling source의 날짜는 거래일인가, 공개 가능일인가?
- lag는 단일 거래일 상수로 충분한가?
- same-date join을 유지해야 할 운영상 이유가 있는가?
