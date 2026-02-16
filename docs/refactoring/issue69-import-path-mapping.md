# Issue #69 Import Path Mapping

- Scope: `src` 패키지 구조 재편 이슈(#69)에서 변경된 import 경로 매핑
- Status date: 2026-02-16
- Source todo: `todos/2026_02_16-issue69-src-package-restructure-breakdown.md`

## 1. 원칙

- 기존 top-level 모듈(`src/*.py`)은 호환성 wrapper로 유지합니다.
- 신규/내부 구현 import는 하위 패키지 구현 경로를 우선 사용합니다.
- 엔트리포인트 실행 커맨드(`python -m src.<entrypoint>`)는 기존과 동일하게 유지합니다.
- `backtest_strategy_gpu`는 wrapper를 유지하고 실제 구현은 `src.backtest.gpu.*`를 사용합니다.

## 2. 경로 매핑 표

| 영역 | 기존/호환 경로 | 신규 구현 경로 | 상태 | 비고 |
| --- | --- | --- | --- | --- |
| Pipeline orchestrator | `src.pipeline_batch` | `src.pipeline.batch` | Done | wrapper 유지 |
| Ticker universe batch | `src.ticker_universe_batch` | `src.pipeline.ticker_universe_batch` | Done | wrapper 유지 |
| OHLCV batch | `src.ohlcv_batch` | `src.pipeline.ohlcv_batch` | Done | wrapper 유지 |
| Daily tier batch | `src.daily_stock_tier_batch` | `src.pipeline.daily_stock_tier_batch` | Done | wrapper 유지 |
| Financial collector | `src.financial_collector` | `src.data.collectors.financial_collector` | Done | wrapper 유지 |
| Investor collector | `src.investor_trading_collector` | `src.data.collectors.investor_trading_collector` | Done | wrapper 유지 |
| WFO analyzer | `src.walk_forward_analyzer` | `src.analysis.walk_forward_analyzer` | Done | wrapper 유지 |
| CPU backtester engine | `src.backtester` | `src.backtest.cpu.backtester` | Done | wrapper + legacy import 호환 |
| CPU strategy | `src.strategy` | `src.backtest.cpu.strategy` | Done | wrapper + legacy import 호환 |
| CPU portfolio | `src.portfolio` | `src.backtest.cpu.portfolio` | Done | wrapper + legacy import 호환 |
| CPU execution | `src.execution` | `src.backtest.cpu.execution` | Done | wrapper + legacy import 호환 |
| GPU optimization library | `src.parameter_simulation_gpu_lib` | `src.optimization.gpu.*` | Done | wrapper 유지 |
| GPU optimization entrypoint | `src.parameter_simulation_gpu` | `src.parameter_simulation_gpu_lib -> src.optimization.gpu.parameter_simulation` | Done | import-safe wrapper 유지 (#60) |
| GPU strategy kernel | `src.backtest_strategy_gpu` | `src.backtest.gpu.*` | Done | wrapper + legacy import 호환 |

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

- 기존 top-level 경로 import는 계속 동작합니다.
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

## 5. 남은 작업

- 없음 (Issue #69 분해 단계 기준)
