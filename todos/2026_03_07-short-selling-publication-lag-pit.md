# Work Note: `ShortSellingDaily` Publication Lag / PIT

> Type: `implementation`
> Status: `in_progress`
> Priority: `P0`
> Last updated: 2026-03-14
> Related issues: `#67`
> Gate status: `open`

## 1. Summary
- What: `ShortSellingDaily` 데이터를 `DailyStockTier`에 반영할 때, 실제 가용 시점보다 이르게 쓰지 않도록 PIT 계약을 고정합니다.
- Why: 현재 수집기는 lag를 보수적으로 보지만, tier batch는 same-date join으로 `sbv_ratio`를 만들고 있어 룩어헤드 후보가 됩니다.
- Current status: 비인증 KRX/pykrx 경로는 이 호스트에서 `400 LOGOUT`/empty로 깨지고, 로그인 세션 주입 smoke는 성공했습니다. `DailyStockTier` 코드에도 임시 운영정책 `lag=3 + same-date 금지`를 반영했고, shadow diff까지 완료했습니다. 현재는 `DailyStockTier` backfill을 진행 중입니다.
- Next action: backfill을 `2013-11-20 ~ 2026-02-06` 범위로 끝내고, 완료 후 최신 날짜/row count/샘플 검증을 남깁니다.

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
- [x] `publication_lag_trading_days` 임시 정책 `3` 고정
- [x] Tier join 규칙 결정
  - `same-date` 유지 금지
  - `DailyStockTier` 계산 시점에는 가용일(`available trading date`) 기준 반영
- [x] `sbv_ratio` shadow 재계산 경로 정의
- [x] 회귀 테스트 추가
- [x] backfill / recompute / rollback 계획 수립

## 6. Evidence Collection Rule
- 공매도 데이터 접근성 확인은 아래 순서로 합니다.
  1. `tools/operations/check_pykrx_login_session.py` 실행
  2. `login_ok=True` 확인
  3. `ETF ticker list` smoke 정상 확인
  4. `short status` smoke 정상 확인
- 위 4개 중 하나라도 실패하면, 해당 환경의 공매도 응답은 lag 근거로 쓰지 않습니다.
- 로그인 세션 없이 직접 endpoint probe만 성공/실패한 결과는 참고용으로만 남기고 정책 결정에는 사용하지 않습니다.

## 7. Code Summary
- [daily_stock_tier_batch.py](/root/projects/Split_Investment_Strategy_Optimizer/src/pipeline/daily_stock_tier_batch.py)
  - `short_selling_publication_lag_trading_days` 인자를 `build_daily_stock_tier_frame(...)`, `run_daily_stock_tier_batch(...)`에 추가했습니다.
  - `ShortSellingDaily.date`는 merge 전에 trading-day 기준으로 `lag=3`만큼 앞으로 밀어 `available date`에 반영하도록 바꿨습니다.
- [test_daily_stock_tier_batch.py](/root/projects/Split_Investment_Strategy_Optimizer/tests/test_daily_stock_tier_batch.py)
  - 기존 SBV overlay 테스트는 `lag=0`을 명시해 기존 수학/임계값 의미를 고정했습니다.
  - 새 lag 테스트를 추가해 same-date에는 `sbv_ratio`가 비어 있고, 가용일에만 반영되는지 확인했습니다.


## 8. Shadow Diff Execution
- 목적: 같은 입력 데이터에 대해 `lag=0`과 `lag=3`을 둘 다 계산하고, `DailyStockTier.sbv_ratio / tier / reason`이 실제로 얼마나 달라지는지 read-only로 비교합니다.
- 실행 명령:
```bash
cd /root/projects/Split_Investment_Strategy_Optimizer
CONDA_NO_PLUGINS=true conda run -n rapids-env \
  python -m src.pipeline.daily_stock_tier_shadow \
  --start-date 20240101 \
  --end-date 20240229 \
  --base-lag-days 0 \
  --lag-days 3
```
- 산출물:
  - `summary.json`: 영향 행 수, 영향 날짜 수, `sbv_ratio` 출현/소멸 수, tier/reason 변경 수
  - `daily_impact.csv`: 날짜별 영향 건수
  - `tier_transition.csv`: `tier_base -> tier_lagged` 전이 건수
  - `affected_rows_sample.csv`: 영향 행 샘플
- 해석 기준:
  - `sbv_appeared_rows`, `sbv_disappeared_rows`가 크면 same-date 반영이 실제 결과에 영향을 준다는 뜻입니다.
  - `tier_changed_rows > 0`이면 `DailyStockTier` 재계산 범위를 실제로 정해야 합니다.
  - `affected_dates`가 최근 구간에만 몰리면 부분 recompute 후보, 장기 구간 전반에 퍼지면 backfill 범위를 더 넓게 봅니다.

## 9. Shadow Diff Result (2026-03-13)
- 실행 아티팩트:
  - outdir: `results/daily_stock_tier_shadow_diff/sbv_lag_shadow_20251201_20260206_20260313_234234`
  - summary: `results/daily_stock_tier_shadow_diff/sbv_lag_shadow_20251201_20260206_20260313_234234/summary.json`
- 요약:
  - `rows_compared = 130562`
  - `affected_rows = 70045`
  - `affected_dates = 47`
  - `first_affected_date = 2025-12-01`
  - `last_affected_date = 2026-02-06`
  - `sbv_appeared_rows = 8315`
  - `sbv_disappeared_rows = 89`
  - `sbv_changed_value_rows = 61641`
  - `tier_changed_rows = 302`
  - `reason_changed_rows = 453`
- 해석:
  - `sbv_ratio` 자체는 꽤 넓게 달라졌습니다.
  - 하지만 최종 `tier` 변경은 `302 rows`로 제한적이었습니다.
  - 즉, 정책 수정은 PIT 관점에서 필요하지만 운영 충격은 제한적이라고 볼 수 있습니다.

## 10. Recompute Range Decision
- 원본 `ShortSellingDaily`는 재수집/재적재하지 않습니다.
- 재계산 대상은 `DailyStockTier` 파생 결과물입니다.
- latest raw date 확인:
  - `ShortSellingDaily max_date = 2026-02-03`
  - `DailyStockPrice/MarketCap/Investor/Financial max_date = 2026-02-06`
- 운영 정책이 `lag=3 trading days`이므로, 현재 데이터셋에서 안전하게 재계산 가능한 `DailyStockTier` 최신 날짜는 `2026-02-06`입니다.
- 따라서 backfill 범위는:
  - `start_date = 2013-11-20`
  - `end_date = 2026-02-06`

## 11. Backfill Progress (2026-03-14)
- `chunk-days=30` initial run으로 시작했고, 이후 `chunk-days=365`로 전환해 중복 계산 오버헤드를 줄였습니다.
- `chunk-days=365` 로그 기준 완료 구간:
  - `2014-04-19 .. 2015-04-18`
  - `2015-04-19 .. 2016-04-17`
  - `2016-04-18 .. 2017-04-17`
- 따라서 재시작 기준은 `2017-04-18`입니다.
- 재개 명령:
```bash
cd /root/projects/Split_Investment_Strategy_Optimizer
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rapids-env \
  python -u -m src.tier_backfill_window \
  --start-date 20170418 \
  --end-date 20260206 \
  --chunk-days 365 | tee logs/tier_backfill_window_20170418_20260206_chunk365_$(date +%Y%m%d_%H%M%S).log
```
- 절전 방지가 필요하면 `systemd-inhibit` 래핑을 같이 권장합니다.

## 12. Backfill Decision Rule
- `ShortSellingDaily` 원본 테이블은 재수집/재적재하지 않습니다.
- backfill 대상은 `DailyStockTier` 파생 결과물입니다.
- shadow diff 결과를 보고 아래처럼 결정합니다.
  1. 영향이 최근 구간에만 집중: 최근 `DailyStockTier`만 recompute
  2. 영향이 장기 구간에 넓게 분포: windowed backfill로 `DailyStockTier` 재계산
  3. `tier_changed_rows=0`이고 `sbv_ratio` shift만 관측: 운영 영향이 작더라도 PIT 계약상 recompute 범위를 문서로 남깁니다.

## 13. Acceptance Criteria
- `DailyStockTier.sbv_ratio`가 미래 공시 데이터를 참조하지 않음
- representative sample에서 `future_reference_count=0` 또는 동등한 PIT 증적 존재
- overlay 수정 후 CPU/GPU candidate selection parity에 신규 mismatch를 만들지 않음
- shadow diff와 backfill 영향 요약이 문서 또는 운영 아티팩트로 남음

## 14. Open Questions
- pykrx short selling source의 날짜는 거래일인가, 공개 가능일인가?
- lag는 단일 거래일 상수로 충분한가?
- same-date join을 유지해야 할 운영상 이유가 있는가?
