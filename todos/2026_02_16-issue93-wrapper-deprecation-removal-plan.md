# Wrapper deprecation/removal plan after Issue #69 (Issue #93)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/93`
- 요약:
  - Issue #69 이후 남아있는 compatibility wrapper를 단계적으로 정리
  - deprecation -> removal 2단계 정책으로 진행
  - Stage 2 제거 완료일: `2026-02-17`

## 목표
- 내부 코드/테스트의 wrapper 의존 제거를 완료하고, 제거 가능한 wrapper를 안전하게 삭제한다.

## 범위
### 유지 필수 wrapper (삭제 금지)
- `src.pipeline_batch`
- `src.ticker_universe_batch`
- `src.ohlcv_batch`
- `src.walk_forward_analyzer`
- `src.parameter_simulation_gpu`
- `src.parameter_simulation_gpu_lib`

### 제거 완료 wrapper (Stage 2)
- `src.backtester`
- `src.strategy`
- `src.portfolio`
- `src.execution`
- `src.backtest_strategy_gpu`
- `src.daily_stock_tier_batch`
- `src.financial_collector`
- `src.investor_trading_collector`

## 완료 조건(DoD)
- 내부 코드에서 조건부 wrapper import 0건
- 테스트 스위트(신규 구현 경로 기준) 통과
- 제거 대상 wrapper 삭제 및 관련 compat 테스트 정책 정리
- deprecation/migration 문서 업데이트

## 체크리스트
- [x] Deprecation 정책/일정 문서화
  - `docs/refactoring/wrapper-deprecation.md`에 Phase 1/2 정책 및 검증 커맨드 문서화
- [x] 조건부 wrapper 사용처 탐지 규칙(CI/테스트) 추가
  - `tests/test_wrapper_usage.py` AST 가드 추가 (`src` 런타임 모듈의 조건부 wrapper import 금지)
- [x] 조건부 wrapper 제거 패치
  - 대상 8개 wrapper 파일 삭제 완료
- [x] wrapper compat 테스트/문서 정리
  - 테스트: `tests/test_issue69_cpu_backtest_wrapper_compat.py`, `tests/test_issue69_entrypoint_compat.py`를 canonical import 기준으로 갱신
  - 문서: `docs/refactoring/issue69-import-path-mapping.md`, `docs/refactoring/wrapper-deprecation.md`, `docs/database/backfill_validation_runbook.md`, `llm.md` 갱신
- [x] 지정 테스트 통과 기록
  - `python -m unittest tests.test_wrapper_usage -v` 통과
  - `python -m unittest tests.test_issue69_cpu_backtest_wrapper_compat tests.test_issue69_entrypoint_compat -v` 통과
