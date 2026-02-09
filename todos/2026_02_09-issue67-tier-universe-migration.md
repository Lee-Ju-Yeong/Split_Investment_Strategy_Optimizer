# feat(backtest): A안 전환 - KRX PIT + DailyStockTier 후보군 표준화 (Issue #67)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/67`
- 작성일: 2026-02-09
- 결정: 최종 `A안`(KRX PIT + Tier), 전환기 `C안`(Hybrid) 제한 허용

## 1. 배경
- 현재 CPU/GPU가 `WeeklyFilteredStocks`를 서로 다르게 사용하고 있어 후보군 일관성이 약함
- 전략 강건성 목표(OOS) 관점에서 `Legacy list` 의존을 줄이고 PIT + Tier 규칙으로 통일 필요
- `TickerUniverseSnapshot/History` + `DailyStockTier`는 이미 운영 파이프라인에 존재

## 2. 브랜치 규칙 (필수)
- [ ] `main` 브랜치에서 직접 구현 금지
- [ ] 기능 브랜치에서만 진행 후 PR로 병합
- [ ] 권장 브랜치:
  - `feature/issue67-a-universe-tier-phase1`
  - `feature/issue67-a-universe-tier-phase2`

## 3. 구현 범위
### 3-1. DataHandler 후보군 API 표준화
- [ ] `candidate_source_mode` 기반 조회 API 추가
  - `weekly`
  - `hybrid_transition`
  - `tier`
- [ ] `tier=1` 우선, empty면 `tier<=2` fallback 규칙 고정
- [ ] PIT 기준 `as_of_date` 검증 실패 시 예외/로그 표준화

### 3-2. CPU 전략 경로 반영
- [ ] `MagicSplitStrategy`에서 후보군 조회를 `candidate_source_mode`로 분기
- [ ] 기본값은 전환기 `hybrid_transition`, 최종 전환 후 `tier`
- [ ] `use_weekly_alpha_gate` 플래그 도입 (`weekly`를 보조 신호로만 사용)

### 3-3. GPU 경로 반영
- [ ] `WeeklyFilteredStocks` 전용 로더 외 `DailyStockTier` 기반 로더 추가
- [ ] GPU 후보군 생성 로직에 `tier=1 -> <=2 fallback` 동일 규칙 적용
- [ ] CPU/GPU 동일 입력일 때 동일 후보군이 생성되는지 검증

### 3-4. 운용/검증 가드
- [ ] 구간별 Tier 커버리지 리포트 추가 (`date`, `tier1_count`, `tier12_count`)
- [ ] 커버리지 미달 임계값(설정 기반) 시 fail-fast
- [ ] 모드별 실행 요약(`weekly/hybrid/tier`) 로그에 명시

## 4. 테스트 범위
- [ ] `tests/test_data_handler_tier.py` 확장: mode별 후보군 조회 테스트
- [ ] `tests/test_strategy.py`(또는 신규): `tier=1 -> <=2 fallback` 분기 테스트
- [ ] `tests/test_backtest_universe_mode.py` 신규: 동일 날짜 CPU 후보군 정합성
- [ ] GPU smoke: `debug_gpu_single_run` 모드별 1회 실행

## 5. 완료 기준 (Definition of Done)
- [ ] `candidate_source_mode=tier`에서 CPU/GPU 후보군 로직 동일
- [ ] `candidate_source_mode=hybrid_transition`에서 정책이 문서와 일치
- [ ] `#56` top-k parity가 `tier` 모드에서 mismatch `0건`
- [ ] 운영 문서(`TODO.md`, 이슈 본문, 실행 커맨드) 갱신 완료

## 6. 제외 범위
- Tier 계산식 자체 변경(임계값 재설계)은 #71 범위
- WFO robust scoring 변경은 #68 범위
