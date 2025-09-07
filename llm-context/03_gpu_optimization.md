# aicontext/03_gpu_optimization.md
# === YAML Front Matter: The Control Panel ===
# 이 파일의 메타데이터를 정의합니다.

topic: "03. GPU 최적화: 대규모 병렬 백테스팅 엔진"
project_id: "masicsplit-v1"
status: "completed" # (현재 기능은 안정화된 상태로 유지보수 단계)
tags:
  - gpu
  - cuda
  - cupy
  - cudf
  - optimization
  - parallel-computing
model: "퀀트-J (시니어 퀀트 시스템 개발자)"
persona: "대규모 병렬 컴퓨팅 및 GPU 커널 최적화에 특화된 시스템 아키텍트"
created_date: "2025-08-31" # (Based on git log)
last_modified: "2025-09-06" # (Documentation Update)
---

## 시스템 프롬프트 (System Prompt / The Constitution)
<!-- _PROJECT_MASTER.md의 규칙을 계승하고, 이 주제에 특화된 목표를 추가합니다. -->

### 🎯 목표 (Objective)
- CPU 백테스터에서 검증된 전략 로직을, 수천 개의 파라미터 조합에 대해 **동시에(in parallel)** 실행할 수 있는 GPU 기반 병렬 시뮬레이션 엔진으로 변환하고 최적화한다.
- 최종적으로, WFO(Walk-Forward Optimization) 파이프라인의 핵심 **'워커(Worker)'** 모듈로서, 지정된 기간과 파라미터 공간에 대한 초고속 최적화 임무를 수행할 수 있도록 시스템을 안정화한다.

### 🎭 페르소나 (Persona)
- _PROJECT_MASTER.md의 페르소나를 계승합니다. 이 파일의 컨텍스트에서는 특히 **벡터화(Vectorization), 병렬 알고리즘, 그리고 GPU 메모리 관리**를 최우선으로 고려하는 병렬 컴퓨팅 아키텍트의 역할을 수행합니다.

### 📜 규칙 및 제약사항 (Rules & Constraints)
- **벡터화 원칙:** 모든 로직은 순차적 `for` 루프를 최대한 배제하고, `CuPy`와 `cuDF`를 사용한 벡터화 연산으로 구현되어야 한다. 이는 GPU의 SIMD(Single Instruction, Multiple Data) 아키텍처 성능을 극대화하기 위함이다.
- **CPU-GPU 통신 최소화:** 백테스팅 루프 중에는 CPU와 GPU 간의 데이터 전송이 발생해서는 안 된다. 모든 필요한 데이터는 **사전 로딩(Preloading)**되어야 한다.
- **상태 관리:** 클래스나 객체 인스턴스 대신, 모든 시뮬레이션의 상태는 거대한 다차원 **'상태 배열(State Arrays)'**로 관리되어야 한다.

## 🔄 롤링 요약 및 핵심 결정사항 (Rolling Summary / The Living Memory)
<!-- 이 주제 내에서의 핵심 결정 사항을 요약합니다. -->

- (시기 미상): **핵심 기술 채택:** NVIDIA CUDA 생태계와의 호환성과 NumPy/Pandas 유사성을 제공하는 `CuPy`와 `cuDF`를 핵심 라이브러리로 채택.
- (시기 미상): **아키텍처:** **상태 배열(State Arrays)** 기반 설계를 채택하여, 모든 거래 로직을 완전 벡터화된(Fully Vectorized) 연산으로 구현.
- (시기 미상): **데이터 전처리:** 백테스팅 시작 전, 필요한 모든 데이터를 단 한 번에 GPU 메모리로 전송하는 **사전 로딩(Preloading)** 방식을 채택하여 루프 내 통신 오버헤드를 제거.
- (시기 미상): **경쟁 조건(Race Condition) 해결:** 병렬 매수 시나리오에서 발생하는 '자본금 동시 사용' 문제를 해결하기 위해, 우선순위 정렬 후 **순차적 자금 차감** 로직을 GPU 커널 내에 구현하는 하이브리드 접근 방식 채택.
- (시기 미상): **검증 프로세스:** `debug_gpu_single_run.py`를 개발하여, 단일 파라미터에 대한 GPU 실행 결과가 CPU 백테스터의 결과와 100% 일치함을 최종 안정화 기준으로 설정.
- (2025-09-07): **[성능 고도화] Host-side 병목 현상 해결:**
    - **문제 정의:** Python(Host)의 일별 `for` 루프 내에서 반복적인 데이터 슬라이싱 작업이 GPU 유휴 시간(idle time)을 유발하는 핵심 병목임을 식별.
    - **해결책 채택:** 백테스팅 시작 전, 전체 기간의 가격 데이터를 `(날짜, 종목)` 형태의 2D CuPy 텐서로 미리 변환하는 **'데이터 텐서화'** 아키텍처를 도입. 이를 통해 루프 내 CPU 작업량을 최소화하고 GPU 대기 시간을 획기적으로 단축.
    - **정합성 재확보:** 텐서화 과정에서 발생한 데이터 불일치 문제를 **'직접 인덱스 매핑'** 방식으로 해결하여, 최적화 이후에도 CPU-GPU 결과의 완전한 정합성을 보장함.

## 🏛️ 핵심 정보 및 로직 (Key Information & Core Logic)
<!-- 이 주제의 아키텍처, 데이터 흐름, 모듈별 역할을 설명합니다. -->

### 1. 핵심 원리: 상태 배열 기반 벡터화 (State Array-based Vectorization)

CPU 백테스터가 **"하나의 시나리오를 시간 순서대로"** 처리하는 반면, GPU 엔진은 **"수천 개의 시나리오를 한순간에 동시에"** 처리합니다. 이는 모든 상태를 거대한 다차원 CuPy 배열로 표현하고, 모든 로직을 이 배열 전체에 대한 단일 연산으로 변환함으로써 가능해집니다.

-   **`portfolio_state (N, ...)`:** N개 시뮬레이션의 포트폴리오 상태 (현금 등)
-   **`positions_state (N, ...)`:** N개 시뮬레이션의 모든 포지션 상태 (수량, 매수가 등)
    *(N: `num_combinations`)*

### 2. 핵심 모듈 상세 설명 (`src/` 디렉토리 기준)

#### 가. `parameter_simulation_gpu.py` (Orchestrator / WFO Worker 1)
- **역할:** 대규모 파라미터 최적화를 시작하는 **최상위 실행 스크립트**이자, WFO의 **'최적화 워커'**.
- **책임:** 파라미터 공간 정의, 모든 조합 생성(`meshgrid`), 데이터 사전 로딩, GPU 커널 실행, 최종 결과 분석 및 저장.

#### 나. `backtest_strategy_gpu.py` (GPU Kernel)
- **역할:** GPU 최적화 시스템의 **심장**. CPU 백테스터의 모든 로직을 CuPy 배열 연산만으로 재구현한 핵심 커널.
- **책임:** 거래일을 순회하는 메인 루프를 실행하며, 매도/신규매수/추가매수 신호 처리를 위한 벡터화된 함수들을 순차적으로 호출. 모든 상태는 상태 배열을 통해 관리 (클래스/객체 미사용).

#### 다. `debug_gpu_single_run.py` (Validator / WFO Worker 2)
- **역할:** 단일 파라미터에 대한 GPU 커널의 실행 결과를 CPU 백테스터와 비교하여 **로직의 정합성을 검증**하는 디버깅 도구이자, WFO의 **'단일 실행 워커'**.
- **책임:** `config.yaml`의 단일 파라미터를 로드하여 GPU 커널을 실행하고, 최종 성과를 `PerformanceAnalyzer`로 계산하여 출력.

## 💬 대화 기록 (Conversation Log / The Transcript)
<!-- 이 파일은 시스템의 핵심 설계를 문서화하는 데 중점을 두므로, 직접적인 대화 기록은 생략합니다. -->

## 📝 스크래치패드 (Scratchpad / The Workbench)
<!-- 이 주제와 관련된 아이디어, 메모, TODO 등을 기록합니다. -->

- **성능 병목:** **[해결됨]** Host-side의 일일 데이터 슬라이싱 병목은 '데이터 텐서화'를 통해 해결됨. 현재 남은 주요 병목은 `_process_new_entry_signals_gpu`와 `_process_additional_buy_signals_gpu` 내부의 우선순위 기반 순차 처리 루프임.
- **추가 최적화 방안:** 현재의 '월 블록 + Host 일일 루프' 구조에서 Host 일일 루프를 제거하고, `cupy.RawKernel`을 사용하여 한 달치 시뮬레이션을 GPU 내에서 모두 처리하는 **'월 블록 커널화'** 를 구현하면 수 배 이상의 추가 성능 향상을 기대할 수 있음 (단, 개발 난이도 매우 높음).
- **정밀도:** `float32` 사용으로 인한 부동소수점 오차는 백테스팅 결과의 유의미한 차이를 유발하지 않음을 확인함.
- **핵심 과제:** 병렬 매수 로직에서 발생하는 경쟁 조건(Race Condition)을 해결하는 것이 가장 큰 기술적 난관이었음. 순위화(Ranking)와 순차적 차감(Sequential Deduction)을 결합한 하이브리드 방식으로 해결.