# llm-context/03_gpu_optimization.md
# === YAML Front Matter ===
topic: "03. GPU 최적화: 대규모 병렬 시뮬레이션"
status: "completed" # (현재 기능은 안정화된 상태)
tags:
  - gpu
  - cuda
  - cupy
  - cudf
  - optimization
  - parallel-computing
---
# === System Prompt / Core Instructions ===
# 이 파일의 목적은 CPU 기반 백테스팅 로직이 어떻게 GPU 환경에서 대규모 병렬 처리를 위해 변환되고 실행되는지, 그 아키텍처와 핵심 기술을 문서화하는 것입니다.

# === Rolling Summary & Key Decisions ===
# GPU 최적화 시스템 개발 과정에서의 주요 결정 사항입니다.

- **핵심 기술 채택:** NVIDIA CUDA 생태계와의 완벽한 호환성과 NumPy/Pandas와 유사한 개발 편의성을 제공하는 `CuPy`와 `cuDF`를 핵심 라이브러리로 채택.
- **아키텍처:** **상태 배열(State Arrays)** 기반 설계를 채택. 모든 시뮬레이션의 상태(포트폴리오, 포지션 등)를 거대한 다차원 CuPy 배열로 표현하고, 모든 거래 로직을 이 배열에 대한 **완전 벡터화(Fully Vectorized)** 연산으로 구현. 이는 GPU의 SIMD(Single Instruction, Multiple Data) 아키텍처 성능을 극대화함.
- **데이터 전처리:** 백테스팅 시작 전, 필요한 모든 데이터를 `preload_all_data_to_gpu` 함수를 통해 **단 한 번에 GPU 메모리로 전송**. 이는 백테스팅 루프 중 발생하는 CPU-GPU 간 데이터 전송 오버헤드를 원천적으로 제거하기 위한 결정.
- **경쟁 조건(Race Condition) 해결:** 병렬 매수 시나리오에서 발생하는 '자본금 동시 사용' 문제를 해결하기 위해, 매수 후보들을 우선순위에 따라 정렬한 후, **순차적 자금 차감(Sequential Capital Deduction)** 로직을 GPU 커널 내에 구현. 이는 병렬성의 이점을 유지하면서도 계산의 정확성을 보장하는 하이브리드 접근 방식임.
- **검증 프로세스:** `debug_gpu_single_run.py`를 개발하여, 단일 파라미터에 대한 GPU 실행 결과가 `02_cpu_backtester.md`의 결과와 100% 일치함을 검증하는 것을 최종 안정화 기준으로 설정.

---
# === Key Information & Core Logic ===
# 이 섹션은 GPU 최적화 시스템의 아키텍처와 핵심 원리를 설명합니다.

## 1. 아키텍처 및 핵심 원리: 벡터화(Vectorization)

CPU 백테스터가 **"하나의 시나리오를 시간 순서대로"** 처리하는 반면, GPU 최적화 시스템은 **"수천 개의 시나리오를 한순간에 동시에"** 처리합니다. 이 거대한 패러다임 전환의 핵심은 **벡터화**입니다.

**상태 배열 (State Arrays):**
모든 시뮬레이션의 상태는 거대한 CuPy 배열로 통합 관리됩니다.
- `portfolio_state (N, 2)`: N개 시뮬레이션의 [현금, 1회 주문금액] 상태.
- `positions_state (N, M, P, 3)`: N개 시뮬레이션, M개 종목, P개 분할매수 차수의 [수량, 매수가, 진입일] 상태. 
- **(N: `num_combinations`, M: `num_tickers`, P: `max_splits_limit`)**

**벡터화된 연산:**
`for` 루프를 사용하여 각 시뮬레이션, 각 종목을 순회하는 대신, 모든 로직은 이 거대한 배열 전체에 적용되는 단일 연산으로 변환됩니다.
- **CPU (Loop):** `for sim in simulations: if sim.price > 100: ...`
- **GPU (Vectorized):** `sell_mask = prices_array > 100`

이러한 접근 방식은 GPU의 수천 개 코어를 동시에 활용하여, CPU로는 수일이 걸릴 계산을 수 분 내에 완료할 수 있게 합니다.

## 2. 핵심 모듈 상세 설명 (`src/` 디렉토리 기준)

### 가. `parameter_simulation_gpu.py` (Orchestrator / WFO Worker 1)
- **역할:** 대규모 파라미터 최적화를 시작하는 **최상위 실행 스크립트**이자, WFO 분석의 **'최적화 워커'**.
- **핵심 로직:**
    1.  **파라미터 공간 정의:** `max_stocks_options`, `stop_loss_rate_options` 등 테스트할 파라미터 값의 범위를 CuPy 배열로 정의.
    2.  `cp.meshgrid`: 정의된 모든 파라미터 값들의 모든 가능한 조합(Cartesian Product)을 생성하여, `param_combinations` 배열을 구축.
    3.  **데이터 사전 로드:** `preload_..._to_gpu` 함수들을 호출하여 백테스팅에 필요한 모든 데이터를 GPU 메모리에 미리 로드.
    4.  **커널 실행:** 준비된 파라미터 조합과 데이터를 `backtest_strategy_gpu.py`의 메인 함수(`run_magic_split_strategy_on_gpu`)에 전달하여 대규모 병렬 백테스팅을 시작.
    5.  **결과 분석:** 모든 시뮬레이션이 끝나면 반환된 `daily_portfolio_values` 배열을 분석하여, 성과가 가장 좋은 파라미터 조합을 찾아내고 결과를 `.csv` 파일로 저장.
    6.  **`find_optimal_parameters` 함수:** WFO 오케스트레이터가 호출할 수 있도록, 위 로직을 재사용 가능한 함수로 캡슐화.

### 나. `backtest_strategy_gpu.py` (GPU Kernel)
- **역할:** **GPU 최적화 시스템의 심장.** CPU 백테스터의 모든 로직(전략, 포트폴리오, 실행)을 CuPy 배열 연산만으로 재구현한 **핵심 커널**.
- **핵심 함수/로직:**
    - `run_magic_split_strategy_on_gpu()`: GPU 백테스팅의 메인 루프. 거래일을 순회하며 각 단계별 처리 함수를 호출.
    - `_process_sell_signals_gpu()`: 모든 시뮬레이션의 모든 포지션에 대해 매도 조건(수익 실현, 손절매 등)을 **동시에** 검사하는 벡터화된 로직.
    - `_process_new_entry_signals_gpu()`: 모든 시뮬레이션에 대해 신규 매수 후보를 우선순위에 따라 정렬하고, 순차적 자금 차감 로직을 통해 **경쟁 조건(Race Condition) 없이** 병렬로 매수 처리.
    - `_process_additional_buy_signals_gpu()`: 모든 보유 종목에 대해 추가 매수 조건을 **동시에** 검사하고, 마찬가지로 정렬 및 순차적 자금 차감 로직으로 처리.
    - **상태 관리:** 모든 함수는 거대한 상태 배열(`portfolio_state`, `positions_state` 등)을 입력으로 받아, 연산을 통해 수정된 새로운 상태 배열을 반환하는 형태로 작동. **클래스나 객체 인스턴스를 전혀 사용하지 않음**.

### 다. `debug_gpu_single_run.py` (Validator / WFO Worker 2)
- **역할:** 단일 파라미터 셋에 대한 GPU 커널의 실행 결과를 CPU 백테스터의 결과와 비교하여, **로직의 정합성을 검증**하는 디버깅 및 테스트 도구. WFO 분석의 **'단일 실행 워커'** 역할도 수행.
- **핵심 로직:**
    1.  `config.yaml`에서 `strategy_params`에 정의된 **단일 파라미터 조합**을 로드.
    2.  이 단일 조합을 `param_combinations` 배열로 변환 (시뮬레이션 개수 N=1).
    3.  `backtest_strategy_gpu.py`의 커널을 `debug_mode=True`로 실행하여, 상세한 내부 동작 로그를 터미널에 출력.
    4.  최종 성과 지표(CAGR, MDD 등)를 `PerformanceAnalyzer`로 계산하여 출력, `main_backtest.py` 실행 결과와 비교할 수 있도록 함.
    5.  **`run_single_backtest` 함수:** WFO 오케스트레이터가 호출할 수 있도록, 위 로직을 재사용 가능한 함수로 캡슐화.

---
# === Scratchpad / Notes Area ===
- **성능 병목 현상:** 현재 시스템에서 가장 시간이 많이 소요되는 부분은 `backtest_strategy_gpu.py`의 메인 루프임. 향후 추가 최적화가 필요하다면, `cupy.RawKernel`을 사용하여 CUDA C++ 코드를 직접 작성하는 것을 고려해볼 수 있으나, 현재 `CuPy` 수준에서도 충분한 성능을 보임.
- **정밀도 문제:** `float32`를 사용하여 발생하는 부동소수점 표현 오차는 의도된 동작이며, 백테스팅 결과의 유의미한 차이를 유발하지 않음을 확인함. 출력단에서 포맷팅하여 가독성을 확보하는 것으로 충분함.
- **가장 어려웠던 점:** 병렬 매수 로직에서 발생하는 경쟁 조건(Race Condition)을 해결하는 것. 순위화(Ranking)와 순차적 차감(Sequential Deduction)을 결합한 하이브리드 방식으로 해결하여 병렬성과 정확성을 모두 확보함.