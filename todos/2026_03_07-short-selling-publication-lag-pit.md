# Work Note: `ShortSellingDaily` Publication Lag / PIT

> Type: `implementation`
> Status: `in_progress`
> Priority: `P0`
> Last updated: 2026-03-13
> Related issues: `#67`
> Gate status: `open`

## 1. Summary
- What: `ShortSellingDaily` 데이터를 `DailyStockTier`에 반영할 때, 실제 가용 시점보다 이르게 쓰지 않도록 PIT 계약을 고정합니다.
- Why: 현재 수집기는 lag를 보수적으로 보지만, tier batch는 same-date join으로 `sbv_ratio`를 만들고 있어 룩어헤드 후보가 됩니다.
- Current status: 비인증 KRX/pykrx 경로는 이 호스트에서 `400 LOGOUT`/empty로 깨지고, 로그인 세션 주입 smoke는 성공했습니다. 따라서 lag 판단은 인증 세션 기준으로만 진행하고, 임시 운영정책은 `lag=3 + same-date 금지`로 둡니다.
- Next action: 인증 smoke를 증적 경로로 고정하고, `DailyStockTier`에 `publication_lag_trading_days=3` 임시 정책을 반영합니다.

## 2. Fixed Problem Statement
- 수집기에는 `lag_trading_days` 개념이 이미 있습니다.
- 그러나 tier batch는 `ShortSellingDaily.date == MarketCapDaily.date` 기준으로 same-date join을 하고 있습니다.
- 이 구조가 진짜 PIT-safe인지 아직 명확하지 않습니다.

## 3. 2026-03-13 접근성 증적
- 비인증 collector preflight는 이 호스트에서 실패했습니다.
  - 에러: `[short_selling_collector] KRX endpoint blocked (http_status=400, bld=dbms/MDC/STAT/srt/MDCSTAT30001)`
  - 의미: 기본 collector/probe 경로로는 최근 가용 시점을 판단할 수 없습니다.
- 비인증 `pykrx.stock.get_shorting_status_by_date(...)`는 최근/과거 날짜 모두 `rows=0`이었지만, 이것은 데이터 부재보다 접근 실패일 가능성이 큽니다.
- 인증 세션을 `pykrx`에 주입하는 smoke는 성공했습니다.
  - 스크립트: [check_pykrx_login_session.py](/root/projects/Split_Investment_Strategy_Optimizer/tools/operations/check_pykrx_login_session.py)
  - 관측:
    - `login_ok=True`
    - `stock.get_etf_ticker_list(20260227)` 정상
    - `stock.get_shorting_status_by_date(20260227, 20260227, "005930")` rows=`1`
- 로그인 세션으로 직접 `getJsonData.cmd` short endpoint를 치면 `{"OutBlock_1": [], "CURRENT_DATETIME": ...}` 형태가 왔습니다.
  - 따라서 현재 환경에서는 **직접 short endpoint probe보다, 로그인 세션 주입 후 pykrx smoke가 더 신뢰할 만한 체크**입니다.
- 외부 참고:
  - pykrx upstream 이슈 `#276`, `#278`에서 2026-02-27 이후 `400 LOGOUT`, `403`, 로그인 세션 주입 우회, 로그인 후에도 `CD003` 가능성이 보고되었습니다.
  - 링크:
    - `#276`: https://github.com/sharebook-kr/pykrx/issues/276
    - `#278`: https://github.com/sharebook-kr/pykrx/issues/278

## 4. Interim Decision
- `publication_lag_trading_days`는 **임시 운영정책으로 `3` 유지**합니다.
- `DailyStockTier.sbv_ratio`의 **same-date 반영은 금지** 방향으로 수정합니다.
- 현재 호스트에서 lag를 추가 실험으로 줄이려면, 반드시 로그인 세션 주입 smoke를 먼저 통과해야 합니다.
- 비인증 경로의 `rows=0` 또는 `empty`는 lag 근거로 사용하지 않습니다.

## 5. Current Plan
- [ ] `ShortSellingDaily.date` 의미 확정
- [x] 비인증/인증 접근 경로 분리 문서화
- [x] 로그인 세션 주입 smoke 경로 확보
- [ ] `publication_lag_trading_days` 임시 정책 `3` 고정
- [ ] Tier join 규칙 결정
  - `same-date` 유지 금지 여부
  - `available_from_date` 저장 여부
- [ ] `sbv_ratio` shadow 재계산 경로 정의
- [ ] 회귀 테스트 추가
- [ ] backfill / recompute / rollback 계획 수립

## 6. Evidence Collection Rule
- 공매도 데이터 접근성 확인은 아래 순서로 합니다.
  1. `tools/operations/check_pykrx_login_session.py` 실행
  2. `login_ok=True` 확인
  3. `ETF ticker list` smoke 정상 확인
  4. `short status` smoke 정상 확인
- 위 4개 중 하나라도 실패하면, 해당 환경의 공매도 응답은 lag 근거로 쓰지 않습니다.
- 로그인 세션 없이 직접 endpoint probe만 성공/실패한 결과는 참고용으로만 남기고 정책 결정에는 사용하지 않습니다.

## 7. Acceptance Criteria
- `DailyStockTier.sbv_ratio`가 미래 공시 데이터를 참조하지 않음
- representative sample에서 `future_reference_count=0` 또는 동등한 PIT 증적 존재
- overlay 수정 후 CPU/GPU candidate selection parity에 신규 mismatch를 만들지 않음
- shadow diff와 backfill 영향 요약이 문서 또는 운영 아티팩트로 남음

## 8. Open Questions
- pykrx short selling source의 날짜는 거래일인가, 공개 가능일인가?
- lag는 단일 거래일 상수로 충분한가?
- same-date join을 유지해야 할 운영상 이유가 있는가?
