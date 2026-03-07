# fix(pit): `ShortSellingDaily` 공시 시차(lag) 검증 및 `DailyStockTier` 반영 보정
- 작성일: 2026-03-07
- 목적: 공매도 잔고/잔고금액 데이터가 실제 가용 시점보다 이르게 Tier 계산에 반영되지 않도록 PIT 계약을 명시하고 보정

## 1. 배경
- 현재 수집기는 `lag_trading_days`를 사용해 안전 종료일을 보수적으로 낮춘다.
- 그러나 Tier 계산 경로는 `ShortSellingDaily.date`를 같은 날짜 `MarketCapDaily.date`와 조인하고, 동일 날짜 `DailyStockTier`에 `sbv_ratio`를 반영한다.
- 이 구조는 `date`가 거래일이라면 룩어헤드 바이어스 후보가 된다.

## 2. 현재 관측 근거
- 수집기 lag 적용:
  - `src/short_selling_collector.py:_cap_end_date_by_short_selling_lag()`
  - `src/short_selling_collector.py:run_short_selling_batch(..., lag_trading_days=...)`
- Tier same-date join:
  - `src/pipeline/daily_stock_tier_batch.py:fetch_short_balance_ratio_inputs()`
  - `src/pipeline/daily_stock_tier_batch.py:_apply_short_balance_ratio()`

## 3. 이번 라운드 결론
- 멀티에이전트/Codex/Gemini 공통 결론:
  - `ShortSellingDaily.date`가 거래일 기준이라면, Tier 반영 시 publication lag 모델링이 필요하다.
  - 이 항목은 성능/리팩터링이 아니라 데이터 정합성/PIT 계약 이슈로 취급해야 한다.

## 4. 초안 체크리스트
- [ ] `ShortSellingDaily.date` 의미를 코드/문서 기준으로 확정
  - 거래일(`transaction_date`)인지
  - 공시일(`publication_date`)인지
- [ ] `publication_lag_trading_days` 정책 고정
  - 기본값은 보수적으로 `3 trading days` 검토
  - `lag_trading_days`와의 관계를 문서화
- [ ] Tier join 규칙 결정
  - `same-date` 유지 금지 여부 판단
  - `available_from_date` 저장 vs Tier 쿼리 시 lag shift 적용 중 선택
- [ ] `sbv_ratio` shadow 재계산 경로 정의
  - 기존 결과와 diff 리포트
  - Tier 이동률 / 최근 20영업일 안정성 비교
- [ ] 회귀 테스트 추가
  - `D` 의사결정이 `<= D` 시점 가용 데이터만 사용함을 검증
  - `D vs D+lag` 케이스 분리
- [ ] 운영 재적재 계획 수립
  - backfill/recompute 범위
  - 롤백 기준

## 5. Acceptance Criteria (초안)
- [ ] `DailyStockTier`의 `sbv_ratio`는 signal/decision date 기준 미래 공시 데이터를 참조하지 않는다.
- [ ] representative sample에서 `future_reference_count=0` 또는 동등한 PIT 증적을 남긴다.
- [ ] Tier overlay 수정 후 CPU/GPU candidate selection parity에 신규 mismatch를 만들지 않는다.
- [ ] shadow diff 결과와 backfill 영향 요약을 `todos/` 또는 운영 문서에 기록한다.

## 6. 오픈 질문
- [ ] pykrx short selling source의 `날짜` 컬럼은 거래일인가, 공개 가능일인가?
- [ ] lag가 종목/시장에 따라 다를 수 있는가, 아니면 단일 거래일 lag로 충분한가?
- [ ] Tier batch에서 same-date join을 유지해야 할 사유가 있는가, 아니면 availability date 모델이 더 명확한가?
