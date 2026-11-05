# llm-context/04_wfo_analysis.md

> **과거 설계 자료입니다.** 본문은 작성 당시의 설명을 보존합니다. SQLite 사용·완료 상태·AI 역할 표현을 현재 구현이나 작성자의 경력으로 해석하지 않습니다. 현재 안내는 [이 폴더 안내](README.md), [프로젝트 상태](../TODO.md)를 확인합니다.
# === YAML Front Matter: The Control Panel ===
# 이 파일의 메타데이터를 정의합니다.

topic: "04. Walk-Forward 분석: 전략 강건성 검증"
project_id: "masicsplit-v1"
status: "in-progress" # (핵심 로직 고도화 진행 중)
tags:
  - walk-forward-optimization
  - wfo
  - overfitting
  - robustness
  - strategy-validation
  - clustering
model: "퀀트-J (시니어 퀀트 시스템 개발자)"
persona: "과최적화를 방지하고 전략의 강건성을 정량적으로 검증하는 퀀트 분석가"
created_date: "2025-09-01" # (Based on git log)
last_modified: "2025-09-13" # (Documentation Update)
---

## 시스템 프롬프트 (System Prompt / The Constitution)
<!-- _PROJECT_MASTER.md의 규칙을 계승하고, 이 주제에 특화된 목표를 추가합니다. -->

### 🎯 목표 (Objective)
- Walk-Forward Optimization(WFO) 프레임워크를 통해, 투자 전략이 과거 데이터에 과최적화되지 않았는지 검증하고, 다양한 시장 국면에서의 현실적인 기대 성과를 측정한다.
- **[핵심]** 각 학습(IS) 구간에서 '최고 성과'가 아닌, **'가장 강건한(Robust) 파라미터 군집(Cluster)'**을 통계적으로 찾아내어, 전략의 실전 생존 가능성을 극대화한다.

## 🔄 롤링 요약 및 핵심 결정사항 (Rolling Summary / The Living Memory)
<!-- 이 주제 내에서의 핵심 결정 사항을 요약합니다. -->

- (시기 미상): **WFO 모델 채택:** **'제약조건 기반의 제어된 겹침(Constraint-based Controlled Overlap)'** 모델을 채택하여, 제한된 데이터 기간 내에서 분석의 현실성과 효율성을 모두 확보.
- (시기 미상): **아키텍처 설계:** **'오케스트레이터-워커'** 아키텍처를 채택. (`walk_forward_analyzer.py`, `parameter_simulation_gpu.py`, `debug_gpu_single_run.py`)
- (2025-09-04): **설정 간소화:** `config.yaml`의 WFO 설정을 `total_folds`와 `period_length_days` 단 두 개로 단순화하여, 사용자 편의성과 설정의 명확성을 극대화.
- (2025-09-06): **[핵심 방법론 진화]** WFO의 목표를 '최고 성과(뾰족한 산봉우리) 파라미터 찾기'에서, **각 IS 시뮬레이션 결과 전체를 통계적으로 분석하여 '가장 강건한 파라미터 군집(넓고 높은 고원)'의 중심을 찾는 방식**으로 최종 확정.

## 🏛️ 핵심 정보 및 로직 (Key Information & Core Logic)
<!-- 이 주제의 아키텍처, 데이터 흐름, 모듈별 역할을 설명합니다. -->

### 1. 설계 사상: "최고"가 아닌 "최선"을 찾는 여정

WFO는 과거 데이터로 학습(In-Sample)하고, 학습에 사용되지 않은 미래 데이터로 검증(Out-of-Sample)하는 과정을 반복하여 과최적화를 방지합니다. 우리 시스템은 여기서 한 단계 더 나아가, 학습 과정에서 단순히 '1등'을 뽑는 것이 아니라, **성과가 좋은 파라미터들이 형성하는 안정적인 '고원'을 찾아냅니다.**

**고도화된 WFO 프로세스:**
1.  **학습 (In-Sample):** 과거 특정 기간에 대해 수천 개의 파라미터 조합으로 시뮬레이션을 실행한다.
2.  **강건성 분석 (Robustness Analysis):** 시뮬레이션 결과 전체를 통계적으로 분석(예: 클러스터링)하여, 성과가 안정적으로 높은 **'파라미터 군집'의 중심값**을 가장 강건한 파라미터로 도출한다.
3.  **검증 (Out-of-Sample):** 위에서 찾은 **'강건한' 파라미터**를 다음 OOS 기간에 적용하여 실제 성과를 테스트한다.
4.  **이동 (Walk Forward):** 이 과정을 모든 Fold에 걸쳐 반복하여, **'강건한 파라미터로만 운영했을 때'**의 현실적인 장기 성과를 최종 측정한다.

### 2. 핵심 모듈 상세 설명 (`src/` 디렉토리 기준)

#### 가. `walk_forward_analyzer.py` (Orchestrator)
- **역할:** WFO 프로젝트의 **총사령관**.
- **핵심 로직 (고도화된 버전):**
    1.  `config.yaml`을 읽어 WFO 기간을 **자동으로 사전 계산**.
    2.  `total_folds` 만큼 `for` 루프를 실행.
    3.  **루프 내부:**
        a. `parameter_simulation_gpu.py`를 호출하여 IS 기간의 **전체 시뮬레이션 결과 DataFrame**을 반환받음.
        b. **(NEW)** `find_robust_parameters()` 헬퍼 함수를 호출하여, 반환된 DataFrame을 분석하고 **'강건한 파라미터 셋'**을 추출.
        c. 추출된 '강건한 파라미터'를 `debug_gpu_single_run.py`에 전달하여 **OOS 백테스트**를 실행.
        d. OOS 수익 곡선과 '강건한 파라미터'를 리스트에 저장.
    4.  **루프 종료 후:** 최종 수익 곡선과 **'강건한 파라미터들의 시계열 안정성'**을 종합 분석.

#### 나. `parameter_simulation_gpu.py` (Worker 1: Mass Simulator)
- **역할:** 특정 IS 기간에 대한 **대규모 병렬 시뮬레이션을 수행하고, 그 전체 결과를 보고하는 '데이터 생산자'**.
- **WFO 내에서의 동작:** `find_optimal_parameters` 함수는 이제 (1등 파라미터, **전체 시뮬레이션 결과 DataFrame**) 두 가지를 모두 반환. 오케스트레이터는 이 중 **전체 결과 DataFrame**을 핵심적으로 사용.

#### 다. `debug_gpu_single_run.py` (Worker 2: Backtester)
- **역할:** 오케스트레이터로부터 **'강건한' 파라미터**와 OOS 기간을 받아, 단일 백테스트를 수행하고 결과를 보고하는 **'실행 전문가'**. (기존 역할과 동일)

#### 라. `(신설 예정)` `robustness_analyzer.py` (Helper: Analyst)
- **역할:** `find_robust_parameters`와 같은 분석 함수를 포함하는 **통계 분석 헬퍼 모듈**.
- **핵심 로직:** 필터링, 클러스터링(K-Means), 분포 분석(중앙값/최빈값) 등의 방법론을 구현하여, 시뮬레이션 결과로부터 가장 강건한 파라미터 셋을 도출.

## 💬 대화 기록 (Conversation Log / The Transcript)
<!-- 이 주제에 대한 구체적인 대화는 이 파일에서 진행됩니다. -->

## 📝 스크래치패드 (Scratchpad / The Workbench)
<!-- 이 주제와 관련된 아이디어, 메모, TODO 등을 기록합니다. -->

- **핵심 과제:** `find_robust_parameters_from_simulation` 함수를 구현하고 WFO 파이프라인에 통합하는 것.
- **연구 워크플로우:**
    1.  **[연구실]** Jupyter Notebook(Colab)에서 단일 IS 시뮬레이션 결과(`.csv`)를 분석하여, 클러스터링 기반의 강건성 탐색 방법론을 프로토타이핑하고 검증한다.
    2.  **[자동화]** 연구 단계에서 검증된 로직(최적 k 자동 선택 포함)을 `robustness_analyzer.py` 모듈로 개발한다.
    3.  **[통합]** `walk_forward_analyzer.py`가 새로운 분석 모듈을 호출하도록 수정하여, 전체 '강건성 중심 WFO' 파이프라인을 완성한다.
