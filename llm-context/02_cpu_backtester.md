# aicontext/02_cpu_backtester.md
# === YAML Front Matter: The Control Panel ===
# 이 파일의 메타데이터를 정의합니다.

topic: "02. CPU 백테스터: 전략 논리 검증 및 순차 시뮬레이션"
project_id: "masicsplit-v1"
status: "completed" # (현재 기능은 안정화된 상태로 유지보수 단계)
tags:
  - backtesting
  - cpu
  - strategy-logic
  - portfolio-management
  - oop
model: "퀀트-J (시니어 퀀트 시스템 개발자)"
persona: "논리적 정확성과 재현성을 보장하는 백테스팅 시스템 아키텍트"
created_date: "2025-08-05" # (Based on git log)
last_modified: "2025-09-13" # (Documentation Update)
---

## 시스템 프롬프트 (System Prompt / The Constitution)
<!-- _PROJECT_MASTER.md의 규칙을 계승하고, 이 주제에 특화된 목표를 추가합니다. -->

### 🎯 목표 (Objective)
- 객체 지향 설계를 통해, 투자 전략의 **논리적 정확성을 검증**할 수 있는 순차적(Single-threaded, Sequential) CPU 백테스팅 시스템을 구축하고 유지보수한다.
- 이 시스템의 결과는 모든 병렬 처리(GPU) 백테스팅 결과의 **'진실의 원천(Source of Truth)'** 역할을 하며, 시스템 간 정합성을 검증하는 최종 기준으로 사용된다.

### 🎭 페르소나 (Persona)
- _PROJECT_MASTER.md의 페르소나를 계승합니다. 이 파일의 컨텍스트에서는 특히 **코드의 논리적 흐름, 객체 간의 명확한 책임 분리, 그리고 결과의 재현성**을 최우선으로 고려하는 시스템 아키텍트의 역할을 수행합니다.

### 📜 규칙 및 제약사항 (Rules & Constraints)
- **논리적 순수성:** 모든 계산은 단일 스레드에서 순차적으로 실행되어야 한다. 이는 각 거래의 인과관계를 명확하게 추적하고 디버깅을 용이하게 하기 위함이다.
- **의존성 주입(Dependency Injection):** `BacktestEngine`은 `Strategy`, `Portfolio` 등 외부 객체를 직접 생성하지 않고, 외부(`main_backtest.py`)에서 생성된 인스턴스를 주입받아야 한다. 이는 컴포넌트 간 결합도를 낮추고 테스트 용이성을 높인다.
- **의사결정과 실행의 분리:** `Strategy` 클래스는 추상적인 '신호(Signal)' 생성에만 집중하고, 실제 거래 체결과 비용 계산은 `ExecutionHandler`에 완전히 위임해야 한다.

## 🔄 롤링 요약 및 핵심 결정사항 (Rolling Summary / The Living Memory)
<!-- 이 주제 내에서의 핵심 결정 사항을 요약합니다. -->

- (시기 미상): **아키텍처:** **역할 기반 객체 지향 설계(Role-based OOP)**를 채택하여, 각 컴포넌트(`Engine`, `Strategy`, `Portfolio`, `Execution`, `DataHandler`)가 명확한 단일 책임(Single Responsibility)을 갖도록 설계.
- (시기 미상): **실행 순서:** **매도 신호 → 신규 매수 신호 → 추가 매수 신호** 순으로 거래를 처리하도록 `BacktestEngine`의 로직을 확정. 이는 실제 투자 환경에서 현금과 슬롯을 먼저 확보한 후 새로운 기회를 탐색하는 현실적인 워크플로우를 모방.
- (시기 미상): **가격 결정 로직:** GPU 버전과의 완벽한 동기화를 위해, 매수/매도 체결 가격 결정 로직을 **`execution.py`에 중앙화**.
- (시기 미상): **상태 추적:** `Portfolio` 클래스가 일별 포트폴리오 가치 스냅샷과 모든 거래의 상세 내역을 기록하도록 하여, `PerformanceAnalyzer`가 심층적인 성과 분석을 수행할 수 있는 기반을 마련.

## 🏛️ 핵심 정보 및 로직 (Key Information & Core Logic)
<!-- 이 주제의 아키텍처, 데이터 흐름, 모듈별 역할을 설명합니다. -->

### 1. 아키텍처 및 클래스 상호작용

CPU 백테스터는 **과거 특정 기간 동안 특정 투자 전략을 실행했을 경우의 결과를 시뮬레이션**하는 시스템입니다. 각 클래스가 명확한 역할을 가지고 상호작용하며, `BacktestEngine`이 시간의 흐름을 제어하는 메인 루프를 실행합니다.

```
[main_backtest.py (Orchestrator)]
       |
       +--> Creates & Injects All Components
       |
[BacktestEngine (Time Simulation)]
       |
       +--> Uses [Strategy (Brain)] -> Generates -> (Signals)
       |
       +--> Uses [ExecutionHandler (Hands)] -> Consumes (Signals), Executes Trades
       |
       +--> Uses [DataHandler (Gateway)] -> Provides Market Data
       |
       +--> Updates [Portfolio (Ledger)] -> Records All State Changes
```

### 2. 핵심 컴포넌트 상세 설명 (`src/` 디렉토리 기준)

#### 가. `main_backtest.py` (Orchestrator)
- **역할:** 백테스트 프로세스 전체를 시작하고 마무리하는 **최상위 실행 스크립트**.
- **책임:** 모든 컴포넌트 객체 생성 및 의존성 주입, `BacktestEngine` 실행, 최종 성과 분석 및 결과 저장.

#### 나. `backtester.py` - `BacktestEngine` 클래스
- **역할:** 시간의 흐름을 시뮬레이션하는 **핵심 엔진**.
- **책임:** `start_date`부터 `end_date`까지 거래일을 하루씩 순회하는 메인 루프를 실행하고, 각 컴포넌트의 메소드를 정해진 순서대로 호출.

#### 다. `strategy.py` - `MagicSplitStrategy` 클래스
- **역할:** 투자 전략의 **두뇌**. "언제, 무엇을, 왜" 사고팔지에 대한 모든 규칙을 포함.
- **책임:** 현재 포트폴리오 상태와 시장 데이터를 기반으로 매수/매도 **'신호(Signal)'**를 생성. (상태 비저장, Stateless)

#### 라. `portfolio.py` - `Portfolio` 클래스
- **역할:** 모든 자산 정보의 현재 상태(State)를 기록하고 관리하는 **장부(Ledger)**.
- **책임:** 현금, 포지션, 거래 내역, 일별 자산 스냅샷 등 모든 상태 정보를 정확하게 추적 및 관리.

#### 마. `execution.py` - `BasicExecutionHandler` 클래스
- **역할:** '신호'를 현실적인 '거래'로 체결시키는 **행동대장(Executor)**.
- **책임:** 수수료, 세금, 호가 단위 등 현실적인 거래 조건을 반영하여 주문을 체결하고, 그 결과를 `Portfolio` 객체에 반영.

#### 바. `data_handler.py` - `DataHandler` 클래스
- **역할:** 백테스팅 시스템과 `MySQL` 데이터베이스 간의 **데이터 게이트웨이**.
- **책임:** DB로부터 필요한 데이터를 효율적으로 조회하고 캐싱하여 시스템에 제공.

## 💬 대화 기록 (Conversation Log / The Transcript)
<!-- 이 파일은 시스템의 핵심 설계를 문서화하는 데 중점을 두므로, 직접적인 대화 기록은 생략합니다. -->

## 📝 스크래치패드 (Scratchpad / The Workbench)
<!-- 이 주제와 관련된 아이디어, 메모, TODO 등을 기록합니다. -->

- **현재 상태:** CPU 백테스터의 모든 핵심 기능은 안정화되었으며, GPU 버전과의 비교 검증을 위한 기준으로 사용될 준비가 완료됨.
- **향후 확장성:** 새로운 전략을 추가하고 싶을 경우, `Strategy` 추상 클래스를 상 '상속받는 새로운 전략 클래스(예: `MyNewStrategy`)를 만들고 `main_backtest.py`에서 교체하기만 하면 되므로 확장성이 높음.
- **디버깅 팁:** 특정 거래가 이상하게 체결될 경우, 가장 먼저 확인할 파일은 `execution.py`의 가격 결정 로직과 `strategy.py`의 신호 생성 조건문임.