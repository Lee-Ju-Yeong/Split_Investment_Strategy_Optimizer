# Issue #69 Import Path Mapping

- Scope: `src` 패키지 구조 재편 이슈(#69)에서 변경된 import 경로 매핑
- Status date: 2026-02-17
- Source todo: `todos/2026_02_16-issue69-src-package-restructure-breakdown.md`
- Update: Issue #93 Stage 2에서 조건부 wrapper 8종 제거 완료

## 1. 원칙

- 유지 대상 wrapper는 엔트리포인트 호환에 필요한 최소 집합만 유지합니다(6.1 참조).
- 신규/내부 구현 import는 하위 패키지 구현 경로를 우선 사용합니다.
- 엔트리포인트 실행 커맨드(`python -m src.<entrypoint>`)는 기존과 동일하게 유지합니다.
- Issue #93 Stage 2 이후 아래 경로는 호환 import를 지원하지 않습니다.
  - `src.backtester`, `src.strategy`, `src.portfolio`, `src.execution`
  - `src.backtest_strategy_gpu`
  - `src.daily_stock_tier_batch`, `src.financial_collector`, `src.investor_trading_collector`

## 2. 경로 매핑 표

| 영역 | 기존/호환 경로 | 신규 구현 경로 | 상태 | 비고 |
| --- | --- | --- | --- | --- |
| Pipeline orchestrator | `src.pipeline_batch` | `src.pipeline.batch` | Done | wrapper 유지 |
| Ticker universe batch | `src.ticker_universe_batch` | `src.pipeline.ticker_universe_batch` | Done | wrapper 유지 |
| OHLCV batch | `src.ohlcv_batch` | `src.pipeline.ohlcv_batch` | Done | wrapper 유지 |
| Daily tier batch | `src.daily_stock_tier_batch` | `src.pipeline.daily_stock_tier_batch` | Removed (#93 Stage 2) | wrapper 삭제(2026-02-17) |
| Financial collector | `src.financial_collector` | `src.data.collectors.financial_collector` | Removed (#93 Stage 2) | wrapper 삭제(2026-02-17) |
| Investor collector | `src.investor_trading_collector` | `src.data.collectors.investor_trading_collector` | Removed (#93 Stage 2) | wrapper 삭제(2026-02-17) |
| WFO analyzer | `src.walk_forward_analyzer` | `src.analysis.walk_forward_analyzer` | Done | wrapper 유지 |
| CPU backtester engine | `src.backtester` | `src.backtest.cpu.backtester` | Removed (#93 Stage 2) | wrapper 삭제(2026-02-17) |
| CPU strategy | `src.strategy` | `src.backtest.cpu.strategy` | Removed (#93 Stage 2) | wrapper 삭제(2026-02-17) |
| CPU portfolio | `src.portfolio` | `src.backtest.cpu.portfolio` | Removed (#93 Stage 2) | wrapper 삭제(2026-02-17) |
| CPU execution | `src.execution` | `src.backtest.cpu.execution` | Removed (#93 Stage 2) | wrapper 삭제(2026-02-17) |
| GPU optimization library | `src.parameter_simulation_gpu_lib` | `src.optimization.gpu.*` | Done | wrapper 유지 |
| GPU optimization entrypoint | `src.parameter_simulation_gpu` | `src.parameter_simulation_gpu_lib -> src.optimization.gpu.parameter_simulation` | Done | import-safe wrapper 유지 (#60) |
| GPU strategy kernel | `src.backtest_strategy_gpu` | `src.backtest.gpu.*` | Removed (#93 Stage 2) | wrapper 삭제(2026-02-17) |

## 3. import 사용 가이드

### 3.1 `src` 내부 코드(신규/수정 코드)

- 구현 모듈을 직접 import 하세요.
- 예시:

```python
# before
from src.financial_collector import run_financial_batch

# after
from src.data.collectors.financial_collector import run_financial_batch
```

```python
# before
from src.backtester import BacktestEngine

# after
from src.backtest.cpu.backtester import BacktestEngine
```

## 3.2 외부 호출/기존 스크립트 호환

- 아래 6개 엔트리포인트 wrapper만 호환 유지됩니다.
- `src.pipeline_batch`, `src.ticker_universe_batch`, `src.ohlcv_batch`
- `src.walk_forward_analyzer`, `src.parameter_simulation_gpu`, `src.parameter_simulation_gpu_lib`
- 아래 8개 wrapper는 제거되었으므로 구현 경로로 직접 전환해야 합니다.
- `src.backtester`, `src.strategy`, `src.portfolio`, `src.execution`
- `src.backtest_strategy_gpu`, `src.daily_stock_tier_batch`
- `src.financial_collector`, `src.investor_trading_collector`
- 예시:

```python
from src.pipeline_batch import run_pipeline_batch
from src.parameter_simulation_gpu import find_optimal_parameters
```

## 4. 엔트리포인트 호환 커맨드(유지)

- `python -m src.pipeline_batch`
- `python -m src.ticker_universe_batch`
- `python -m src.ohlcv_batch`
- `python -m src.walk_forward_analyzer`
- `python -m src.parameter_simulation_gpu`
- `python -m src.main_backtest`
- `python -m src.main_script`

## 5. PR-12 및 #93 Stage 2 적용 결과

- `src` 내부 wrapper 참조 제거
  - `src.main_backtest` -> `src.backtest.cpu.*` 직접 import
- 테스트 import 전환(호환성 테스트 제외)
  - `tests/test_pipeline_batch.py` -> `src.pipeline.batch`
  - `tests/test_ticker_universe_batch.py` -> `src.pipeline.ticker_universe_batch`
  - `tests/test_ohlcv_batch.py` -> `src.pipeline.ohlcv_batch`
  - `tests/test_daily_stock_tier_batch.py` -> `src.pipeline.daily_stock_tier_batch`
  - `tests/test_collector_normalization.py` -> `src.data.collectors.*`
  - `tests/test_integration.py` -> `src.backtest.cpu.*`
  - `tests/test_point_in_time.py` -> `src.backtest.cpu.strategy`
  - `tests/test_issue67_tier_universe.py` -> `src.backtest.cpu.*`, `src.backtest.gpu.*`
  - `tests/test_portfolio.py` -> `src.backtest.cpu.portfolio`
  - `tests/test_backtest_strategy_gpu.py` -> `src.backtest.gpu.logic`
- #93 Stage 2 반영
  - 조건부 wrapper 8개 삭제
  - `tests/test_issue69_cpu_backtest_wrapper_compat.py`를 canonical import + removed wrapper 검증으로 전환
  - `tests/test_issue69_entrypoint_compat.py`에서 `src.pipeline.daily_stock_tier_batch` 직접 검증
  - `tests/test_wrapper_usage.py`에 removed wrapper 파일 존재 금지 가드 추가

## 6. Wrapper 정리 목록 (2026-02-17)

### 6.1 유지 필수 (현 단계 제거 불가)

| Wrapper | 유지 사유 |
| --- | --- |
| `src.pipeline_batch` | 엔트리포인트 커맨드 호환 (`python -m src.pipeline_batch`) |
| `src.ticker_universe_batch` | 엔트리포인트/운영 스크립트 호환 |
| `src.ohlcv_batch` | 엔트리포인트/운영 스크립트 호환 |
| `src.walk_forward_analyzer` | 엔트리포인트/사이드이펙트 가드 테스트 |
| `src.parameter_simulation_gpu` | public API(`find_optimal_parameters`) + import-safe 규칙(#60) |
| `src.parameter_simulation_gpu_lib` | 패키지/legacy wrapper 호환 테스트 유지 |

### 6.2 제거 완료 (Issue #93 Stage 2)

| Removed wrapper | canonical import path |
| --- | --- |
| `src.backtester` | `src.backtest.cpu.backtester` |
| `src.strategy` | `src.backtest.cpu.strategy` |
| `src.portfolio` | `src.backtest.cpu.portfolio` |
| `src.execution` | `src.backtest.cpu.execution` |
| `src.backtest_strategy_gpu` | `src.backtest.gpu.*` |
| `src.daily_stock_tier_batch` | `src.pipeline.daily_stock_tier_batch` |
| `src.financial_collector` | `src.data.collectors.financial_collector` |
| `src.investor_trading_collector` | `src.data.collectors.investor_trading_collector` |
