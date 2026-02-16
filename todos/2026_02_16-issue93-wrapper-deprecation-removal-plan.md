# Wrapper deprecation/removal plan after Issue #69 (Issue #93)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/93`
- 요약:
  - Issue #69 이후 남아있는 compatibility wrapper를 단계적으로 정리
  - 즉시 삭제가 아닌 deprecation -> removal 2단계로 진행

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

### 조건부 제거 대상 wrapper
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
- [ ] Deprecation 정책/일정 문서화
- [ ] 조건부 wrapper 사용처 탐지 규칙(CI/테스트) 추가
- [ ] 조건부 wrapper 제거 패치
- [ ] wrapper compat 테스트/문서 정리
- [ ] 지정 테스트 통과 기록
