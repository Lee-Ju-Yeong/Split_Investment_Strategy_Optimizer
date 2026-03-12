# Issue #93: Wrapper Deprecation / Removal Plan

> Type: `reference`
> Status: `done`
> Priority: `archive`
> Last updated: 2026-03-07
> Related issues: `#93`, `#69`
> Gate status: `closed`

## 1. Summary
- What: package restructure 이후 남아 있던 compatibility wrapper를 정리한 작업입니다.
- Why: canonical import 경로를 고정하고, 실제 운영 entrypoint만 남기기 위해 필요했습니다.
- Current status: 완료된 문서입니다. 현재 active focus 대상이 아닙니다.
- Next action: 신규 wrapper drift가 생길 때만 참고합니다.

## 2. Final Outcome
- 유지 필수 wrapper는 남겼습니다.
  - `src.pipeline_batch`
  - `src.ticker_universe_batch`
  - `src.ohlcv_batch`
  - `src.walk_forward_analyzer`
  - `src.parameter_simulation_gpu`
  - `src.parameter_simulation_gpu_lib`
- 제거 대상 wrapper는 정리했습니다.
- 관련 테스트와 문서도 canonical 경로 기준으로 갱신했습니다.

## 3. Evidence
- 테스트:
  - `python -m unittest tests.test_wrapper_usage -v`
  - `python -m unittest tests.test_issue69_cpu_backtest_wrapper_compat tests.test_issue69_entrypoint_compat -v`
- 참고 문서:
  - `docs/refactoring/wrapper-deprecation.md`
  - `docs/refactoring/issue69-import-path-mapping.md`

## 4. Reading Rule
- 현재 wrapper 관련 신규 작업이 없다면 이 문서는 다시 읽지 않아도 됩니다.
- 다만 canonical entrypoint drift를 의심할 때는 참고 가치가 있습니다.
