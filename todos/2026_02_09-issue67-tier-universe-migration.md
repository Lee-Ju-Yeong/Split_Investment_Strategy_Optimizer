# feat(backtest): A안 전환 - KRX PIT + DailyStockTier 후보군 표준화 (Issue #67)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/67`
- 작성일: 2026-02-09
- 결정: 최종 `A안`(KRX PIT + Tier), 전환기 `C안`(Hybrid) 제한 허용

## 0. 진행 현황 (2026-02-09)
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
    - Sell: T-1 Tier3 강제 청산
  - CPU/GPU 동시 반영:
    - CPU: `MagicSplitStrategy`에 Tier map 기반 Hold/Add/Sell 게이트 추가
    - GPU: candidate fallback 제거(strict), add-buy tier gate, tier3 liquidation mask 추가
  - 검증: `conda run --no-capture-output -n rapids-env python -m unittest tests.test_issue67_tier_universe tests.test_backtest_strategy_gpu -v` 통과(25 tests)
- 남은 핵심:
  - 모드별 end-to-end CPU/GPU parity 하네스(`tests/test_backtest_universe_mode.py`, `tests/test_cpu_gpu_parity_topk.py`)
  - Phase B 계측/Phase C 구조 최적화

## 1. 배경
- 현재 CPU/GPU가 `WeeklyFilteredStocks`를 서로 다르게 사용하고 있어 후보군 일관성이 약함
- 전략 강건성 목표(OOS) 관점에서 `Legacy list` 의존을 줄이고 PIT + Tier 규칙으로 통일 필요
- `TickerUniverseSnapshot/History` + `DailyStockTier`는 이미 운영 파이프라인에 존재

## 2. 브랜치 규칙 (필수)
- [x] `main` 브랜치에서 직접 구현 금지
- [x] 기능 브랜치에서만 진행 후 PR로 병합
- [x] 현재 적용 브랜치: `feature/issue67-a-universe-tier-phase1` (권장 브랜치)
- [ ] 차기 단계 브랜치: `feature/issue67-a-universe-tier-phase2`

## 3. 구현 범위
### 3-1. DataHandler 후보군 API 표준화
- [ ] `candidate_source_mode` 기반 조회 API 추가
  - `weekly`
  - `hybrid_transition`
  - `tier`
- [x] `tier=1` 우선, empty면 `tier<=2` fallback 규칙 고정
- [ ] PIT 기준 `as_of_date` 검증 실패 시 예외/로그 표준화

### 3-2. CPU 전략 경로 반영
- [x] `MagicSplitStrategy`에서 후보군 조회를 `candidate_source_mode`로 분기
- [x] 전환기 기본값 `hybrid_transition` 반영(`config/config.example.yaml`)
- [ ] 최종 전환 후 기본값 `tier` 승격
- [x] `use_weekly_alpha_gate` 플래그 도입 (`weekly`를 보조 신호로만 사용)

### 3-3. GPU 경로 반영
- [x] `WeeklyFilteredStocks` 전용 로더 외 `DailyStockTier` 기반 로더 추가
- [x] GPU 후보군 생성 로직에 `tier=1 -> <=2 fallback` 동일 규칙 적용
- [ ] CPU/GPU 동일 입력일 때 동일 후보군이 생성되는지 검증

### 3-4. 운용/검증 가드
- [ ] 구간별 Tier 커버리지 리포트 추가 (`date`, `tier1_count`, `tier12_count`)
- [ ] 커버리지 미달 임계값(설정 기반) 시 fail-fast
- [x] 모드별 실행 요약(`weekly/hybrid/tier`) 로그에 명시

### 3-5. 후보군 선택 강건성 규칙(결정론 baseline)
- [x] `random-only` 후보군 선택 금지(운영/최적화 기준)
- [x] 동일 입력일 때 동일 `Top-K`가 재현되도록 점수식/정렬 키/동점 규칙 고정
- [x] Entry/Hold hysteresis 규칙 명문화/구현:
  - config: `tier_hysteresis_mode = legacy | strict_hysteresis_v1`
  - strict 모드: `Entry=tier1 only`, `Hold/Add=tier<=2`, `tier3 forced liquidation`
- [x] 현재 구현 기준 hysteresis 동작 문서/테스트 보강(2026-02-17)
- [ ] `Top-K` 구성 로그에 `score`, `rank`, `tie_break_key` 저장(실험 재현용)

### 3-6. Hybrid 성능 최적화 단계 계획 (2026-02-09 업데이트)
- [x] Phase A (Parity blocker 우선): GPU 후보군 기준일을 `signal_date(T-1)`로 통일
- [x] Phase A (Parity blocker 우선): CPU/GPU 공통 정렬 규칙 고정(`score desc`, `ticker asc`)
- [x] Phase A (Parity blocker 우선): ATR 조회를 `signal_date as-of(<=)`로 맞춰 결측일에서도 직전값 사용
- [x] Phase A (Parity blocker 우선): Tier preload를 `start 이전 latest 1행 + 기간 데이터`로 변경(30일 가정 제거)
- [ ] Phase B (계측): 후보군 단계별 프로파일 로그 추가
  - `tier_select_ms`, `weekly_gate_ms`, `atr_filter_ms`, `host_transfer_count`
- [ ] Phase C (구조 최적화): `weekly_mask_tensor(day,ticker)` 사전 생성
- [ ] Phase C (구조 최적화): `atr_tensor(day,ticker)` 사전 생성 후 `DataFrame isin/filter` 경로 제거
- [ ] Phase C (구조 최적화): hybrid 교집합은 GPU mask 연산(`tier_mask & weekly_mask`)으로 처리
- [ ] 정책 고정: `nondeterministic set mode`는 도입하지 않고, 필요 시 `deterministic_fast`만 허용
- [ ] 승격 게이트: 3모드(`weekly/hybrid_transition/tier`) parity mismatch `0건` + 재실행 hash 일치 + 성능 개선 실측

## 4. 테스트 범위
- [ ] `tests/test_data_handler_tier.py` 확장: mode별 후보군 조회 테스트
- [x] `tests/test_strategy.py`(또는 신규): `tier=1 -> <=2 fallback` 분기 테스트
- [ ] `tests/test_backtest_universe_mode.py` 신규: 동일 날짜 CPU 후보군 정합성
- [ ] `tests/test_backtest_universe_mode.py` 확장: 동일 입력/시드에서 `Top-K` 결정론 재현성 테스트
- [ ] `tests/test_backtest_universe_mode.py` 확장: GPU 후보군 `signal_date(T-1)` 정합성 테스트
- [ ] `tests/test_cpu_gpu_parity_topk.py`에 모드별/시나리오별 candidate order 검증 추가
- [x] `tests/test_issue67_tier_universe.py` 확장: `signal_date(T-1)`/정렬 helper 동작 검증
- [ ] GPU smoke: `debug_gpu_single_run` 모드별 1회 실행

## 5. 완료 기준 (Definition of Done)
- [ ] `candidate_source_mode=tier`에서 CPU/GPU 후보군 로직 동일
- [ ] `candidate_source_mode=hybrid_transition`에서 정책이 문서와 일치
- [ ] `#56` top-k parity가 `tier` 모드에서 mismatch `0건`
- [ ] 후보군 선택 경로가 결정론 baseline으로 고정(`random-only` 경로 없음)
- [ ] hybrid 경로에서 `signal_date(T-1)` 기준이 CPU/GPU 모두 동일하게 적용
- [ ] `deterministic_fast` 경로(도입 시)에서 parity/재현성/성능 게이트 동시 통과
- [ ] 운영 문서(`TODO.md`, 이슈 본문, 실행 커맨드) 갱신 완료

## 6. 제외 범위
- Tier 계산식 자체 변경(임계값 재설계)은 #71 범위
- WFO robust scoring 변경은 #68 범위
