# Wrapper Deprecation Plan (#93)

## Scope
- Target issue: `#93`
- Goal: remove internal dependence on conditional compatibility wrappers while keeping user-facing entrypoints stable.

## Wrapper Policy
- Keep wrappers (do not remove):
  - `src.pipeline_batch`
  - `src.ticker_universe_batch`
  - `src.ohlcv_batch`
  - `src.walk_forward_analyzer`
  - `src.parameter_simulation_gpu`
  - `src.parameter_simulation_gpu_lib`
- Removed in Stage 2 (`2026-02-17`):
  - `src.backtester`
  - `src.strategy`
  - `src.portfolio`
  - `src.execution`
  - `src.backtest_strategy_gpu`
  - `src.daily_stock_tier_batch`
  - `src.financial_collector`
  - `src.investor_trading_collector`
- Migration targets:
  - CPU backtester: `src.backtest.cpu.*`
  - GPU backtest logic: `src.backtest.gpu.*`
  - Tier batch: `src.pipeline.daily_stock_tier_batch`
  - Collectors: `src.data.collectors.*`

## Phase 1 (Guard)
- Status: completed.
- Guard implementation (`tests/test_wrapper_usage.py`):
  - `tests/test_wrapper_usage.py`
  - AST-based import scan (`import`, `from ... import ...`)
  - Removed wrapper file existence check

## Phase 2 (Removal)
- Status: completed (`2026-02-17`).
- Applied:
  - Removed 8 conditional wrappers from `src/`
  - Updated compatibility tests to canonical import paths
  - Updated active docs (`llm.md`, `TODO.md`, runbook/import mapping)

## Verification
- Run:
  - `python -m unittest tests.test_wrapper_usage -v`
  - `python -m unittest tests.test_issue69_cpu_backtest_wrapper_compat tests.test_issue69_entrypoint_compat -v`
