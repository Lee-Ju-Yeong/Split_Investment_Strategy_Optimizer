# Issue #67: PIT Universe + DailyStockTier Runtime Standardization

> Type: `implementation`
> Status: `in progress`
> Priority: `P0`
> Last updated: 2026-03-07
> Related issues: `#67`, `#56`, `#98`
> Gate status: `open`

## 1. One-Page Summary
- What: CPU/GPU 후보군 선택을 `KRX PIT + DailyStockTier` 기준으로 완전히 통일하는 문서입니다.
- Why: candidate policy가 한쪽이라도 다르면 parity와 WFO 승격 판단이 동시에 흔들립니다.
- Current status: tier-only runtime path와 coverage gate는 대부분 정리됐지만, runtime candidate gate parity와 frozen PIT manifest는 아직 남았습니다.
- Next action: optimizer/runtime path에도 CPU와 같은 candidate gate를 태우고, run-scoped manifest cache를 도입합니다.

## 2. Fixed Rules
- `candidate_source_mode=tier`가 기본 경로입니다.
- `signal_date=T-1`, `as-of <= signal_date` 규칙을 지킵니다.
- `min_tier12_coverage_ratio=0.45`를 운영 하한으로 봅니다.
- `#56`의 release parity gate와 충돌하면 이 문서의 변경도 승격할 수 없습니다.

## 3. Current Plan
- [x] tier-only runtime path 고정
- [x] ATR as-of / `signal_date(T-1)` 정합 반영
- [x] coverage report + `--fail-on-gate` 보강
- [x] empirical threshold `0.45` 고정
- [ ] runtime candidate gate parity
- [ ] frozen PIT candidate manifest
- [ ] CPU/GPU candidate order direct validation
- [ ] PIT 실패 시 예외/로그 표준화

## 4. Key Evidence
- 현재 확인된 것:
  - `candidate_source_mode` 정규화
  - tier coverage gate와 sampled report 동기화
  - `WeeklyFilteredStocks` dead branch cleanup
- 아직 필요한 것:
  - optimizer/runtime 경로의 `min_liquidity_20d_avg_value`
  - `min_tier12_coverage_ratio` 완전 동일화
  - run 중 DB drift를 막는 frozen manifest

## 5. Reading Guide
- 지금 무엇이 열려 있는지만 보려면 `3. Current Plan`을 먼저 읽으세요.
- 아래 이력은 왜 threshold와 runtime 정책이 현재처럼 고정됐는지 추적할 때만 읽으면 됩니다.

## 6. Detailed History And Working Log
### 0. 진행 현황 (2026-02-09)
- 현재 작업 브랜치: `feature/issue67-a-universe-tier-phase1` (`main` 직접 작업 금지 준수)
- 완료(코드 반영):
  - CPU/전략 후보군 분기: `weekly | tier | hybrid_transition` + `use_weekly_alpha_gate`
  - GPU 후보군 Phase A parity: `signal_date(T-1)` + 결정론 정렬 + ATR `as-of(<=)` + Tier preload(30일 가정 제거)
  - 안정성 가드: invalid mode 시 `weekly` fallback, `tier/hybrid`에서 `tier_tensor` 누락 fail-fast
  - CPU 예외 가드: Tier 조회 실패 시 `weekly` fallback
- 완료(검증):
  - `tests/test_issue67_tier_universe.py` 확장(전략 모드, T-1 helper, 정렬 helper, ATR as-of helper, Tier 예외 fallback)
  - 로컬 테스트: `python -m unittest tests.test_issue67_tier_universe tests.test_data_handler_tier tests.test_pipeline_batch -v` 통과(17 tests)
- 업데이트(2026-02-17):
  - `tests/test_issue67_tier_universe.py` 추가 확장:
    - invalid `candidate_source_mode` -> weekly fallback
    - 동일 ATR 후보군 ticker tie-break 결정론 검증
    - `generate_additional_buy_signals`의 T-1 조회 인자 검증
    - T+0 진입 추가매수 차단 / T+1 허용 검증
  - `docs/MAGIC_SPLIT_STRATEGY_PRINCIPLES.md`에 Entry/Hold hysteresis 현재 동작과 Tier3 정책 gap 명시
  - 검증: `conda run --no-capture-output -n rapids-env python -m unittest tests.test_issue67_tier_universe -v` 통과(16 tests)
- 업데이트(2026-02-17, strict hysteresis v1):
  - `strategy_params.tier_hysteresis_mode` 도입(`legacy | strict_hysteresis_v1`)
  - strict 모드 동작:
    - Entry: Tier1 only, Tier1 empty면 신규진입 skip
    - Hold/Add: T-1 Tier<=2만 추가 매수 허용
    - Sell: Tier3 forced liquidation 경로는 커널 지원만 남기고 runtime 기본값은 비활성
  - CPU/GPU 동시 반영:
    - CPU: `MagicSplitStrategy`에 Tier map 기반 Hold/Add 게이트 추가
    - GPU: candidate fallback 제거(strict), add-buy tier gate 반영, tier3 liquidation mask는 shadow capability로 유지
  - 검증: `conda run --no-capture-output -n rapids-env python -m unittest tests.test_issue67_tier_universe tests.test_backtest_strategy_gpu -v` 통과(25 tests)
- 업데이트(2026-03-07, regression sync):
  - `tests/test_issue67_tier_universe.py`를 현재 runtime 정책에 맞게 정리
    - invalid `candidate_source_mode`는 `weekly` fallback이 아니라 `tier` 강제 경로로 검증
    - CPU-side 테스트가 CUDA 미지원 환경에서도 돌도록 GPU helper import를 lazy-load + skip 처리
    - strategy fixture를 PIT gated candidate API / `portfolio.cash` 경로에 맞게 보강
  - `tests/test_data_handler_tier.py`는 `strict_pit` 전제로 재정렬해 survivor-only 추가 query로 인한 오탐을 제거
  - `src/debug_gpu_single_run.py`, `src/parity_sell_event_dump.py`도 `WeeklyFilteredStocks` preload를 중단하고 tier-only runtime path에 맞춰 empty frame을 사용
  - 검증: `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_issue67_tier_universe tests.test_data_handler_tier tests.test_cpu_strategy_entry_context tests.test_universe_policy` 통과(43 tests, 4 skipped)
- 업데이트(2026-03-07, weekly gate cleanup):
  - `src/optimization/gpu/parameter_simulation.py`는 `candidate_source_mode`를 runtime에서 `tier`로 정규화한 뒤 `WeeklyFilteredStocks` preload를 항상 skip
  - `src/backtest/gpu/engine.py`에서도 weekly/hybrid 교집합 분기를 제거하고 Tier 후보군만 사용
  - 의도: `use_weekly_alpha_gate`가 살아 있는 것처럼 보이는 false signal 제거
- 업데이트(2026-03-07, coverage gate/report closeout):
  - `src/tier_coverage_report.py`에 `required_tier12_ratio`, `coverage_gate_pass`, summary JSON, `--fail-on-gate`를 추가
  - 단위 테스트 추가: `tests/test_tier_coverage_report.py`
  - 의미: `DataHandler` runtime gate와 같은 기준을 구간 리포트/백필 검증에도 적용
- 업데이트(2026-03-07, coverage report gate sync):
  - `src/tier_coverage_report.py`에 `coverage_gate_pass`, `required_tier12_ratio`, summary JSON, `--fail-on-gate`를 추가
  - `tests/test_tier_coverage_report.py`로 liquidity filter 반영, failed date summary, empty-row 기본값을 고정
  - 의도: `DataHandler._enforce_tier12_coverage_gate()`와 같은 기준을 오프라인 coverage 점검 CLI에서도 동일하게 확인
  - 추가 안전장치: `--fail-on-gate`는 `--step-days 1`에서만 허용하여 샘플링 리포트를 승격 게이트로 오해하지 않도록 제한
- 업데이트(2026-03-07, empirical threshold fixation):
  - 운영 기본 `min_tier12_coverage_ratio`를 `0.45`로 상향
  - 실측 근거(as-of sampled windows):
    - `2024-12-30..2025-01-13`: 최저 `0.456809`
    - `2015-01-12..2015-01-26`: 최저 `0.477972`
    - `2023-11-06..2023-11-20`: 최저 `0.487128`
  - exact-snapshot proxy(20-snapshot step) 최저값도 약 `0.4525`로 확인되어 `0.45`를 운영 하한으로 채택
- 남은 핵심:
  - `tier` 경로 기준 end-to-end CPU/GPU parity 하네스(#56) (`tests/test_cpu_gpu_parity_topk.py`)
    - 주의: parity 하네스의 "구현/진척 관리"는 #56에서 진행하고, #67에서는 "승격 게이트(DoD)"로만 참조한다.
  - 장기 구간(예: `2013-11-20~`) 백테스트를 위한 `DailyStockTier` 커버리지 확장(backfill)

## 0-1. 진행 현황 업데이트 (2026-02-11)
- [x] `DataHandler.get_pit_universe_codes_as_of()` 추가: `TickerUniverseSnapshot latest(as-of)` 우선, empty 시 `TickerUniverseHistory active(as-of)` fallback
- [x] `DataHandler.get_candidates_with_tier_fallback_pit()` 추가: PIT 유니버스 내부에서 `tier=1 -> tier<=2 fallback`
- [x] `MagicSplitStrategy`가 `get_candidates_with_tier_fallback_pit` 우선 호출(미지원 시 기존 API fallback)하도록 반영
- [x] `A안` 단일 경로 고정:
  - `candidate_source_mode` 기본값을 `tier`로 승격
  - CPU/GPU에서 `weekly/hybrid_transition` 입력 시 경고 후 `tier`로 강제
  - Tier 조회 실패 시 `weekly` fallback 제거(빈 후보군 반환)
- [x] 회귀 테스트 추가:
  - `tests/test_data_handler_tier.py`: PIT universe 조회/ fallback, PIT tier fallback 경로
  - `tests/test_issue67_tier_universe.py`: strategy tier/PIT API 우선 호출, 비-tier 모드 강제 tier 정규화 검증
- [x] Coverage/Liquidity gate 반영:
  - `DataHandler.get_candidates_with_tier_fallback_pit_gated()` 추가
  - 구간별 coverage 로그(`date`, `tier1_count`, `tier12_count`, `universe_count`) 출력
  - `min_tier12_coverage_ratio` 미달 시 fail-fast 예외
  - `min_liquidity_20d_avg_value` 기반 후보군 필터링
- [x] GPU preload에도 동일 게이트 반영:
  - Tier tensor 생성 시 as-of liquidity mask 적용
  - coverage gate(`min_tier12_coverage_ratio`) 미달 시 즉시 실패

## 0-2. 진행 현황 업데이트 (2026-02-14)
- [x] 현재 작업 브랜치: `feature/issue67-a-universe-tier-phase2`
- [x] `A안` 단일 경로 고정(재확인): `weekly/hybrid_transition` 경로는 제거/비활성화가 목표이며, 실행 시 `tier`로 정규화
- [x] Tier 커버리지 리포트/튜닝 보조 CLI 추가:
  - `src/tier_coverage_report.py`: PIT 유니버스 내부에서 `tier1/tier<=2` 커버리지 리포트(csv/table)
- 운영 기본값은 `min_tier12_coverage_ratio=0.45`로 고정(2026-03-07 실측 반영)
- parity harness(`src.cpu_gpu_parity_topk`)의 Tier coverage gate도 PIT universe count를 분모로 사용하도록 맞춰, coverage report와 같은 threshold 의미를 공유
- [x] 장기 backfill을 위한 Tier 배치 성능 보강:
  - `src/daily_stock_tier_batch.py`: `FinancialData` 조회에 `start_date` 윈도우 + as-of seed(최신 1행) 추가
  - `src/tier_backfill_window.py`: DailyStockTier windowed backfill runner 추가(대용량 OOM 방지)
- [x] `DailyStockTier` 장기 커버리지 확장 진행(운영 DB 실측):
  - 완료: `2013-11-20..2013-12-30`, `2014-01-02..2014-12-30`
  - 부분 완료: `2015-01-02..2015-06-29`
  - 기존 보유: `2024-01-02..2026-02-06`
  - 미충족: `2015-06-30..2023-12-31` (장기 백테스트/coverage gate 판단을 위해 backfill 필요)

## 1. 배경
- 현재 CPU/GPU가 `WeeklyFilteredStocks`를 서로 다르게 사용하고 있어 후보군 일관성이 약함
- 전략 강건성 목표(OOS) 관점에서 `Legacy list` 의존을 줄이고 PIT + Tier 규칙으로 통일 필요
- `TickerUniverseSnapshot/History` + `DailyStockTier`는 이미 운영 파이프라인에 존재

## 2. 브랜치 규칙 (필수)
- [x] `main` 브랜치에서 직접 구현 금지
- [x] 기능 브랜치에서만 진행 후 PR로 병합
- [x] 현재 적용 브랜치: `feature/issue67-a-universe-tier-phase2`

## 3. 구현 범위
### 3-1. DataHandler 후보군 API 표준화
- [x] `tier` 단일 경로 정규화(A안 고정): 비-tier 입력은 경고 후 `tier`로 강제
- [x] `tier=1` 우선, empty면 `tier<=2` fallback 규칙 고정
- [ ] PIT 기준 `as_of_date` 검증 실패 시 예외/로그 표준화

### 3-2. CPU 전략 경로 반영
- [x] `MagicSplitStrategy`에서 후보군 조회를 `candidate_source_mode`로 분기
- [x] 전환기 기본값 `hybrid_transition` 반영(`config/config.example.yaml`)
- [x] 최종 전환 후 기본값 `tier` 승격
- [x] `use_weekly_alpha_gate` 플래그 도입 (`weekly`를 보조 신호로만 사용)

### 3-3. GPU 경로 반영
- [x] `WeeklyFilteredStocks` 전용 로더 외 `DailyStockTier` 기반 로더 추가
- [x] GPU 후보군 생성 로직에 `tier=1 -> <=2 fallback` 동일 규칙 적용
- [ ] CPU/GPU 동일 입력일 때 동일 후보군이 생성되는지 검증

### 3-4. 운용/검증 가드
- [x] 구간별 Tier 커버리지 리포트 추가 (`date`, `tier1_count`, `tier12_count`)
- [x] 커버리지 미달 임계값(설정 기반) 시 fail-fast (단, 운영 기본값은 gate off)
- [x] 후보군 커버리지/튜닝 리포트 CLI 추가(`src/tier_coverage_report.py`)
- [x] coverage summary/fail-on-gate 추가: `--summary-out`, `--fail-on-gate`, `coverage_gate_pass`

### 3-5. 후보군 선택 강건성 규칙(결정론 baseline)
- [x] `random-only` 후보군 선택 금지(운영/최적화 기준)
- [x] 동일 입력일 때 동일 `Top-K`가 재현되도록 점수식/정렬 키/동점 규칙 고정
- [x] Entry/Hold hysteresis 규칙 명문화/구현:
  - config: `tier_hysteresis_mode = legacy | strict_hysteresis_v1`
  - strict 모드: `Entry=tier1 only`, `Hold/Add=tier<=2`, `Tier3 forced liquidation`은 runtime 기본값에서 비활성
- [x] 현재 구현 기준 hysteresis 동작 문서/테스트 보강(2026-02-17)
- [ ] `Top-K` 구성 로그에 `score`, `rank`, `tie_break_key` 저장(실험 재현용)

### 3-6. Hybrid 성능 최적화 단계 계획 (2026-02-09 업데이트)
- [x] Phase A (Parity blocker 우선): GPU 후보군 기준일을 `signal_date(T-1)`로 통일
- [x] Phase A (Parity blocker 우선): CPU/GPU 공통 정렬 규칙 고정(`score desc`, `ticker asc`)
- [x] Phase A (Parity blocker 우선): ATR 조회를 `signal_date as-of(<=)`로 맞춰 결측일에서도 직전값 사용
- [x] Phase A (Parity blocker 우선): Tier preload를 `start 이전 latest 1행 + 기간 데이터`로 변경(30일 가정 제거)
- [ ] (De-scoped) Hybrid 경로가 제거된 이후에는 Phase B/C 최적화의 우선순위를 낮춘다.
- [ ] 정책 고정: `nondeterministic set mode`는 도입하지 않고, 필요 시 `deterministic_fast`만 허용
- [ ] 승격 게이트: `tier` parity mismatch `0건` + 재실행 hash 일치 + (필요 시) 성능 개선 실측

## 4. 테스트 범위
- [ ] `tests/test_data_handler_tier.py` 확장: (tier-only) 후보군 조회 + 게이트 회귀 테스트
- [x] `tests/test_tier_coverage_report.py` 추가: sampled coverage summary/gate helper 회귀 테스트
- [x] `tests/test_strategy.py`(또는 신규): `tier=1 -> <=2 fallback` 분기 테스트
- [ ] `tests/test_backtest_universe_mode.py` 신규: 동일 날짜 CPU 후보군 정합성
- [ ] `tests/test_backtest_universe_mode.py` 확장: 동일 입력/시드에서 `Top-K` 결정론 재현성 테스트
- [ ] `tests/test_backtest_universe_mode.py` 확장: GPU 후보군 `signal_date(T-1)` 정합성 테스트
- [ ] `tests/test_cpu_gpu_parity_topk.py`에 모드별/시나리오별 candidate order 검증 추가
- [x] `tests/test_issue67_tier_universe.py` 확장: `signal_date(T-1)`/정렬 helper 동작 검증
- [ ] GPU smoke: `debug_gpu_single_run` 모드별 1회 실행

## 5. 완료 기준 (Definition of Done)
- [ ] `candidate_source_mode=tier`에서 CPU/GPU 후보군 로직 동일
- [ ] `#56` top-k parity가 `tier` 모드에서 mismatch `0건`
- [ ] 후보군 선택 경로가 결정론 baseline으로 고정(`random-only` 경로 없음)
- [ ] `deterministic_fast` 경로(도입 시)에서 parity/재현성/성능 게이트 동시 통과
- [ ] 운영 문서(`TODO.md`, 이슈 본문, 실행 커맨드) 갱신 완료

## 6. 제외 범위
- Tier 계산식 자체 변경(임계값 재설계)은 #71 범위
- WFO robust scoring 변경은 #68 범위
