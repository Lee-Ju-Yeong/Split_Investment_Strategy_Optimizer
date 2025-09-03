# llm-context/02_cpu_backtester.md
# === YAML Front Matter ===
topic: "02. CPU 백테스터: 전략 논리 검증 및 시뮬레이션"
status: "completed" # (현재 기능은 안정화된 상태)
tags:
  - backtesting
  - cpu
  - strategy-logic
  - portfolio-management
  - oop
---
# === System Prompt / Core Instructions ===
# 이 파일의 목적은 CPU 백테스터를 구성하는 각 클래스의 역할과 상호작용, 그리고 데이터 흐름을 명확히 문서화하는 것입니다.
# CPU 백테스터는 모든 GPU 로직의 '진실의 원천(Source of Truth)' 역할을 합니다.

# === Rolling Summary & Key Decisions ===
# CPU 백테스터 개발 과정에서의 주요 결정 사항입니다.

- **아키텍처:** **역할 기반 객체 지향 설계(Role-based OOP)**를 채택하여, 각 컴포넌트(`Engine`, `Strategy`, `Portfolio`, `Execution`, `DataHandler`)가 명확한 단일 책임(Single Responsibility)을 갖도록 설계. 이는 코드의 유지보수성과 확장성을 극대화함.
- **실행 순서:** **매도 신호 → 신규 매수 신호 → 추가 매수 신호** 순으로 거래를 처리하도록 `BacktestEngine`의 로직을 확정. 이는 실제 투자 환경에서 현금과 슬롯을 먼저 확보한 후 새로운 기회를 탐색하는 현실적인 워크플로우를 모방.
- **가격 결정 로직:** GPU 버전과의 완벽한 동기화를 위해, 매수/매도 체결 가격 결정 로직을 **`execution.py`에 중앙화**. 특히, 추가 매수와 손절매 시나리오에서 `target_price`와 당일 `high/low/close` 가격을 복합적으로 고려하는 정교한 로직을 적용.
- **상태 추적:** `Portfolio` 클래스가 일별 포트폴리오 가치 스냅샷(`daily_snapshot_history`)과 모든 거래의 상세 내역(`trade_history`)을 기록하도록 하여, `PerformanceAnalyzer`가 심층적인 성과 분석을 수행할 수 있는 기반을 마련.

---
# === Key Information & Core Logic ===
# 이 섹션은 CPU 백테스터의 아키텍처와 각 클래스의 핵심 역할을 설명합니다.

## 1. 아키텍처 및 워크플로우

CPU 백테스터는 **과거 특정 기간 동안 특정 투자 전략을 실행했을 경우 어떤 결과가 나왔을지를 시뮬레이션**하는 시스템입니다. 이 시스템의 핵심 목표는 **전략의 논리적 정확성을 검증**하는 것이며, 모든 계산은 단일 스레드 환경에서 순차적으로 실행되어 각 거래의 인과관계를 명확하게 추적할 수 있습니다.

**워크플로우:**
1.  `main_backtest.py`가 `config.yaml`을 읽어 모든 설정을 초기화하고, 각 핵심 클래스의 인스턴스를 생성.
2.  `BacktestEngine`이 지정된 기간의 모든 거래일을 순회하는 메인 루프를 시작.
3.  **[매일 반복]**
    a. `Strategy`가 현재 포트폴리오 상태(`Portfolio`)와 시장 데이터(`DataHandler`)를 바탕으로 매수/매도 신호를 생성.
    b. `BacktestEngine`이 생성된 신호를 `ExecutionHandler`에 전달.
    c. `ExecutionHandler`가 신호를 바탕으로 가상 주문을 체결하고, 거래 비용(수수료/세금)을 계산.
    d. 체결된 거래 결과를 `Portfolio`에 반영 (현금 변경, 포지션 추가/제거).
    e. `Portfolio`가 당일의 최종 자산 가치(스냅샷)를 기록.
4.  모든 거래일 순회가 끝나면, `main_backtest.py`가 `PerformanceAnalyzer`를 통해 최종 성과를 분석하고 리포트를 생성.

**클래스 간 상호작용 다이어그램:**

```
[main_backtest.py] -> creates -> [BacktestEngine]
                             |
                             +--> uses -> [Strategy]
                             |
                             +--> uses -> [ExecutionHandler] -> updates -> [Portfolio]
                             |
                             +--> uses -> [DataHandler] -> provides data to all
```

## 2. 핵심 컴포넌트 상세 설명 (`src/` 디렉토리 기준)

### 가. `main_backtest.py` (Orchestrator)
- **역할:** CPU 백테스트 프로세스 전체를 시작하고 마무리하는 **최상위 실행 스크립트**.
- **핵심 로직:**
    - `config.yaml` 로드 및 모든 파라미터 설정.
    - `DataHandler`, `Strategy`, `Portfolio`, `ExecutionHandler`, `BacktestEngine` 등 모든 핵심 객체를 생성하고 의존성을 주입.
    - `BacktestEngine.run()`을 호출하여 백테스팅 시작.
    - 백테스트 종료 후, `PerformanceAnalyzer`에 최종 `Portfolio` 객체를 전달하여 성과 분석 및 결과(그래프, CSV) 저장을 지시.

### 나. `backtester.py` - `BacktestEngine` 클래스
- **역할:** 시간의 흐름을 시뮬레이션하는 **핵심 엔진**.
- **핵심 로직:**
    - `__init__`: 필요한 모든 객체(Portfolio, Strategy 등)를 외부에서 주입받음.
    - `run()`: `start_date`부터 `end_date`까지 DB에 존재하는 실제 거래일을 하루씩 순회하는 **메인 루프**를 실행.
    - 루프 내부에서 `Strategy` 객체를 호출하여 그날의 거래 신호를 받아오고, 이를 `ExecutionHandler`에 전달하여 실행을 위임.
    - 매일 루프가 끝날 때마다 `Portfolio.record_daily_snapshot()`을 호출하여 일별 자산 상태를 기록.

### 다. `strategy.py` - `MagicSplitStrategy` 클래스
- **역할:** 투자 전략의 **두뇌**. **"언제(When), 무엇을(What), 왜(Why)"** 사고팔아야 하는지에 대한 모든 의사결정 규칙을 포함.
- **핵심 로직:**
    - `generate_sell_signals()`: 보유 중인 포지션을 순회하며 수익 실현, 손절매, 최대 보유 기간 초과 등 매도 조건을 검사하고 매도 신호 생성.
    - `generate_new_entry_signals()`: 현재 보유 종목 수, 쿨다운 기간 등을 고려하여 포트폴리오에 새로 편입할 종목에 대한 신규 매수 신호 생성.
    - `generate_additional_buy_signals()`: 기존 보유 종목이 특정 하락률을 만족했을 때 추가 매수(물타기) 신호를 생성.
    - **상태 비저장(Stateless):** 전략 자체는 상태를 가지지 않으며, 매일 `Portfolio` 객체로부터 최신 상태를 전달받아 의사결정을 내립니다. 전략은 '무엇을, 언제 사고팔지'에 대한 **추상적인 신호(Signal)**를 생성할 뿐, '어떻게 체결할지'에 대한 구체적인 거래 로직은 `ExecutionHandler`에 완전히 위임합니다. 이는 **의사결정과 실행의 분리**라는 중요한 설계 원칙을 따릅니다.

### 라. `portfolio.py` - `Portfolio` 클래스
- **역할:** 모든 자산 정보의 **현재 상태(State)**를 기록하고 관리하는 **장부**.
- **핵심 로직:**
    - `initial_cash`, `cash`: 초기 자본과 현재 현금을 추적.
    - `positions`: 현재 보유 중인 모든 주식 포지션을 딕셔너리 형태로 관리 (`{'ticker': [Position, Position, ...]}`).
    - `add_position`, `remove_position`: `ExecutionHandler`의 거래 체결 결과에 따라 포지션을 추가하거나 제거.
    - `get_total_value()`: 현재 현금과 보유 주식의 시장 가치를 합산하여 총 포트폴리오 가치를 계산.
    - `record_daily_snapshot`, `record_trade`: 모든 일별 자산 변화와 개별 거래 내역을 `daily_snapshot_history`와 `trade_history` 리스트에 누적하여 기록.

### 마. `execution.py` - `BasicExecutionHandler` 클래스
- **역할:** `Strategy`가 생성한 추상적인 '신호'를 현실적인 '거래'로 체결시키는 **행동대장**.
- **핵심 로지:**
    - `execute_order()`: 매수/매도 신호를 받아 해당 주문을 처리.
    - **현실성 반영:**
        - **수수료 및 세금:** `buy_commission_rate`, `sell_tax_rate` 등을 적용하여 거래 비용을 정확히 계산.
        - **호가 단위:** `_get_tick_size`, `_adjust_price_up` 메소드를 통해 한국 주식 시장의 호가 단위를 적용, 체결가를 현실적으로 보정.
        - **체결 가격 로직:** 신규 매수(당일 종가 기준), 추가 매수(장중 저가와 타겟가 비교), 수익 실현(장중 고가와 타겟가 비교) 등 시나리오에 따라 다른 가격 결정 로직을 적용.

### 바. `data_handler.py` - `DataHandler` 클래스
- **역할:** 백테스팅 시스템과 `MySQL` 데이터베이스 간의 **데이터 게이트웨이**.
- **핵심 로직:**
    - `__init__`: DB 커넥션 풀을 생성하고 `CompanyInfo` 데이터를 캐싱하여 성능 최적화.
    - `load_stock_data()`: 특정 종목의 지정된 기간 데이터를 DB에서 조회. `lru_cache`를 사용하여 반복적인 DB 조회를 방지.
    - `get_latest_price()`: 특정 날짜를 기준으로 가장 최근의 종가를 제공 (거래가 없던 휴일 등을 처리).
    - `get_filtered_stock_codes()`: 특정 날짜에 유효한 투자 대상 종목군을 DB에서 조회.

---
# === Scratchpad / Notes Area ===
- **현재 상태:** CPU 백테스터의 모든 핵심 기능은 안정화되었으며, GPU 버전과의 비교 검증을 위한 기준으로 사용될 준비가 완료됨.
- **향후 확장성:** 새로운 전략을 추가하고 싶을 경우, `Strategy` 추상 클래스를 상속받는 새로운 전략 클래스(예: `MyNewStrategy`)를 만들고 `main_backtest.py`에서 교체하기만 하면 되므로 확장성이 높음.
- **디버깅 팁:** 특정 거래가 이상하게 체결될 경우, 가장 먼저 확인할 파일은 `execution.py`의 가격 결정 로직과 `strategy.py`의 신호 생성 조건문임.

