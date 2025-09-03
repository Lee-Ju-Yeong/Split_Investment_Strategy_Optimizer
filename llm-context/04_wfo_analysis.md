# llm-context/04_wfo_analysis.md
# === YAML Front Matter ===
topic: "04. Walk-Forward 분석: 전략 강건성 검증"
status: "completed" # (초기 버전 기능 구현 완료)
tags:
  - walk-forward-optimization
  - wfo
  - overfitting
  - robustness
  - strategy-validation
---
# === System Prompt / Core Instructions ===
# 이 파일의 목적은 Walk-Forward Optimization(WFO) 시스템의 설계 사상, 아키텍처, 실행 방법, 그리고 결과 해석 방법을 명확히 문서화하는 것입니다.
# 이 시스템은 우리 투자 전략이 실전에 투입될 수 있는지 여부를 판단하는 최종 관문입니다.

# === Rolling Summary & Key Decisions ===
# WFO 시스템 개발 과정에서의 주요 결정 사항입니다.

- **WFO 모델 채택:** 파트너님의 의도를 반영하여, **'Unanchored Rolling Window (비고정 롤링 윈도우)'** 방식을 최종 WFO 모델로 채택. 이 모델은 학습(IS) 기간과 검증(OOS) 기간이 함께 시간의 흐름에 따라 이동하여, 변화하는 시장 환경에 대한 전략의 적응력을 가장 현실적으로 테스트함.
- **아키텍처 설계:** **'오케스트레이터-워커(Orchestrator-Worker)'** 아키텍처를 채택.
    - **Orchestrator (`walk_forward_analyzer.py`):** 전체 WFO 프로세스(Fold 분할, 루프, 결과 종합)를 지휘.
    - **Worker 1 (`parameter_simulation_gpu.py`):** IS 기간의 최적 파라미터 탐색 임무 수행.
    - **Worker 2 (`debug_gpu_single_run.py`):** OOS 기간의 단일 백테스트 임무 수행.
    - 이 설계는 각 모듈의 책임을 명확히 분리하고 코드의 재사용성을 극대화함.
- **설정 중앙화:** WFO의 모든 동작 규칙(Fold 수, 기간 길이, 이동 간격 등)을 `config.yaml`의 `walk_forward_settings` 섹션에서 중앙 관리하도록 설계하여, 코드 수정 없이 다양한 시나리오를 테스트할 수 있는 유연성을 확보.
- **핵심 결과물 정의:**
    1.  **최종 WFO Equity Curve:** 모든 OOS 기간의 수익 곡선을 하나로 연결한, 전략의 최종 실효성 지표.
    2.  **파라미터 안정성 리포트:** 각 Fold에서 선택된 최적 파라미터의 분포를 분석하여, 파라미터가 특정 기간에만 의존적인지(불안정) 아니면 시점에 관계없이 일관적인지(안정)를 판단.

---
# === Key Information & Core Logic ===
# 이 섹션은 WFO 시스템의 설계 사상과 각 모듈의 역할을 설명합니다.

## 1. 설계 사상: 왜 Walk-Forward 분석이 필요한가?

전체 과거 기간에 대해 단 한 번의 최적화를 수행하는 것은 **과최적화(Overfitting)**의 위험이 매우 높습니다. 이는 특정 과거 데이터에만 완벽하게 들어맞는 '정답' 파라미터를 찾아낼 뿐, 해당 파라미터가 미래의 새로운 시장 환경에서도 유효할 것이라는 보장을 해주지 못합니다.

Walk-Forward 분석은 이 문제를 해결하기 위해, 인간이 과거 데이터로 학습하고 미래를 예측하는 과정을 모방합니다.

**WFO 프로세스:**
1.  **학습 (In-Sample):** 과거의 특정 기간(예: 2015-2020년) 데이터로 가장 좋은 성과를 내는 최적 파라미터를 찾는다.
2.  **검증 (Out-of-Sample):** 위에서 찾은 파라미터를 학습에 사용되지 않은, 시간적으로 약간 뒤따라오는 미래 기간(예: 2015년 7월-2021년 6월)에 적용하여 실제 성과를 테스트한다.
3.  **이동 (Walk Forward):** 학습과 검증 기간을 함께 시간의 흐름에 따라 뒤로 이동시킨 후(예: 6개월), 1~2번 과정을 반복한다.

이 과정을 여러 번 반복하여 얻어진 **모든 검증(OOS) 기간의 성과만을 연결**한 것이 바로 전략의 진정한 장기 성과, 즉 **WFO Equity Curve**입니다.

## 2. 핵심 모듈 상세 설명 (`src/` 디렉토리 기준)

### 가. `walk_forward_analyzer.py` (Orchestrator)
- **역할:** **WFO 프로젝트의 총사령관.**
- **핵심 로직:**
    1.  `config.yaml`을 로드하여 `total_folds`, `is_delta`, `oos_delta`, `step_delta` 등 WFO 실행 규칙을 설정.
    2.  `total_folds` 만큼 `for` 루프를 실행.
    3.  **루프 내부:**
        a. 현재 Fold 번호에 맞는 IS 및 OOS 기간을 정확히 계산.
        b. `parameter_simulation_gpu.py`의 `find_optimal_parameters` 함수를 **IS 기간**과 함께 호출하여 **최적화 워커**에게 임무 부여.
        c. 반환받은 최적 파라미터를 `debug_gpu_single_run.py`의 `run_single_backtest` 함수에 **OOS 기간**과 함께 전달하여 **단일 실행 워커**에게 임무 부여.
        d. 반환받은 OOS 수익 곡선을 `all_oos_curves` 리스트에 저장.
    4.  **루프 종료 후:**
        a. `all_oos_curves` 리스트의 모든 수익 곡선을 시간 순서대로 연결하여 `final_wfo_curve` 생성.
        b. `PerformanceAnalyzer`를 통해 최종 WFO 곡선의 성과 지표를 계산하고 출력.
        c. `all_optimal_params` 리스트를 DataFrame으로 변환하여 파라미터 안정성을 분석하고, 시각화(`plot_wfo_results`) 및 파일 저장.

### 나. `parameter_simulation_gpu.py` (Worker 1: Optimizer)
- **역할:** 오케스트레이터로부터 특정 IS 기간을 할당받아, 해당 기간 내에서 **최고의 성과를 내는 파라미터 셋을 찾아 보고하는 '최적화 전문가'**.
- **WFO 내에서의 동작:** `find_optimal_parameters` 함수가 호출되면, 주어진 기간 내에서만 대규모 GPU 시뮬레이션을 실행하고, 가장 성과가 좋았던 단 하나의 파라미터 딕셔너리와 전체 분석 DataFrame을 반환.

### 다. `debug_gpu_single_run.py` (Worker 2: Backtester)
- **역할:** 오케스트레이터로부터 특정 OOS 기간과 IS 기간에서 찾은 최적 파라미터를 할당받아, **해당 조건으로만 단일 백테스트를 수행하고 그 결과를 보고하는 '실행 전문가'**.
- **WFO 내에서의 동작:** `run_single_backtest` 함수가 호출되면, 주어진 파라미터와 기간으로 단일 GPU 백테스트를 수행하고, 결과물인 일별 수익 곡선(Pandas Series)을 반환.

### 라. `config.yaml` (`walk_forward_settings` 섹션)
- **역할:** WFO 시스템의 모든 동작을 제어하는 **설정 파일**.
- **주요 키:**
    - `total_folds`: 총 몇 번의 학습/검증 사이클을 반복할지 결정.
    - `oos_start_offset_days`: OOS 기간이 IS 기간 시작일로부터 며칠 후에 시작될지(시간차)를 결정.
    - `out_of_sample_period_days`: OOS 기간의 '길이'를 결정.
    - `step_size_days`: 한 사이클이 끝난 후, 다음 학습/검증 기간을 얼마나 뒤로 이동시킬지 결정.

---
# === Scratchpad / Notes Area ===
- **결과 해석 가이드:**
    - **WFO Calmar Ratio > 1.0:** 일반적으로 전략이 장기적으로 강건하다고 판단할 수 있는 긍정적인 신호.
    - **파라미터 분포가 특정 값에 집중:** 최적 파라미터가 안정적이라는 의미. 예를 들어, 모든 Fold에서 `stop_loss_rate`가 -0.15 근처에서 선택된다면, 이 값은 신뢰할 수 있음.
    - **파라미터 분포가 넓게 퍼짐:** 최적 파라미터가 학습 기간에 따라 크게 변동한다는 의미이며, 이는 전략이 불안정하고 과최적화되기 쉽다는 위험 신호.
- **현재 과제:** 초기 실행에서 발견된 과최적화 문제를 해결해야 함. `parameter_simulation_gpu.py`의 파라미터 탐색 공간을 더 현실적인 범위로 좁히고, `total_folds`를 늘려 더 많은 기간에 대한 테스트를 수행하여 파라미터 안정성을 확보하는 것이 다음 단계임.