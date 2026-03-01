# 매직스플릿 투자 전략 원칙 (Magic Split Strategy Principles)

## I. 자금 관리 원칙 (Capital Management)

### 1. 월 1회 투자금 재산정 (Monthly Rebalancing of Investment Amount)

-   **원칙:** 매월 첫 거래일에 포트폴리오의 총자산(현금 + 주식 평가액)을 기준으로 해당 월에 사용할 '분할 매수 1회당 투자금'을 재산정합니다.
-   **계산 시점:** 매월 첫 거래일 시작 시.
-   **계산 기준 자산:** 재산정일의 **전일(D-1)** 종가 기준 총 포트폴리오 평가액.
    -   백테스트 첫날에는 전일이 없으므로, 초기 예탁금을 기준으로 계산합니다.
-   **계산식:** `1회당 투자금 = 전일 기준 총 포트폴리오 평가액 * order_investment_ratio`
-   **적용:** 이렇게 계산된 `1회당 투자금`은 다음 재산정일까지 모든 신규 및 추가 매수 주문에 동일하게 적용됩니다.

> #### **CPU vs. GPU 구현 차이**
>
> -   **CPU:** `strategy.py`의 `_calculate_monthly_investment` 함수에서 `portfolio.get_total_value(previous_day_date, ...)`를 호출하여 객체 지향적으로 계산합니다.
> -   **GPU:** `backtest_strategy_gpu.py`의 `_calculate_monthly_investment_gpu` 함수에서 모든 시뮬레이션의 자산 정보를 담은 `portfolio_state`와 `positions_state` 배열을 사용하여 벡터화 연산으로 동시에 계산합니다.
> -   **결과는 동일:** 두 방식 모두 '전일 종가 기준'으로 평가하며, 로직상 차이는 없습니다.

## II. 종목 선정 원칙 (Universe Selection)

### 1. Tier 후보군 필터링

-   **원칙:** 신규 진입 후보군은 `DailyStockTier` 기반으로 구성합니다.
-   **적용:** 신호일(T-1) 기준 `Tier1`을 우선 사용하고, strict 모드가 아니며 `Tier1`이 비어 있을 때만 `Tier2` fallback을 허용합니다.
-   **비고:** `candidate_source_mode='weekly'`는 운영 경로에서 더 이상 1차 후보군으로 사용하지 않습니다.

### 2. 신규 진입 후보 선정 (New Entry)

-   **대상:** Tier 후보군 중 미보유 + 쿨다운이 아닌 종목.
-   **우선순위:** `entry_composite_score` 내림차순
    1. `0.50 * cheap_effective`
    2. `0.30 * flow_score` (`flow5_mcap` 단면 백분위)
    3. `0.20 * atr_score` (`atr_14_ratio` 단면 백분위)
    4. 동점 시 `market_cap` 내림차순
    5. 최종 동점 시 `ticker` 오름차순
-   **선정 개수:** `available_slots`까지 상위 후보를 채웁니다.

> #### **CPU vs. GPU 구현 차이**
>
> -   **CPU:** `strategy.py`에서 후보별 점수를 계산한 뒤 복합 정렬을 수행합니다.
> -   **GPU:** 동일 정렬 키(복합 점수 -> 시총 -> 티커)를 벡터화 경로에 적용합니다.
> -   **결과 목표:** 동일 입력/설정에서 결정 경로 parity를 유지합니다.

### 3. Entry/Hold 히스테리시스

-   **정책 모드:** `strategy_params.tier_hysteresis_mode`로 제어합니다.
    - `legacy`(기본): Entry는 `tier=1 -> empty면 tier<=2 fallback`
    - `strict_hysteresis_v1`: Entry는 Tier1 only(비면 skip), Tier2 fallback 차단
-   **Entry 경로:** `candidate_source_mode`가 `tier`/`hybrid_transition`일 때 Tier 후보군을 사용합니다. strict 모드에서는 `get_candidates_with_tier_fallback` 결과가 `TIER_2_FALLBACK`인 경우 신규 진입을 생성하지 않습니다.
-   **Hold/Add 경로:** `generate_additional_buy_signals`는 T+0 진입 제외(T+1부터 허용), `cooldown_tracker`, `max_splits_limit`, `additional_buy_drop_rate`를 적용합니다. 모드와 무관하게 T-1 Tier가 `1~2`인 보유 종목에만 추가 매수를 허용합니다.
-   **Tier3 리스크 경로:** Tier 기반 강제청산은 비활성입니다. 청산은 손절/비활성기간(`max_inactivity_period`) 규칙으로만 수행합니다.

## III. 매수 원칙 (Buy Principles)

### 1. 공통 규칙

-   **매수 수량:** `floor(1회당 투자금 / 최종 매수가)`
-   **매수 비용:** `(매수가 * 수량) + 수수료`
-   **수수료:** `floor((매수가 * 수량) * buy_commission_rate)`
-   **자금 확인:** 매수 시점의 보유 현금이 `총 매수 비용`보다 많거나 같아야 주문이 체결됩니다.
    -   **[핵심]** CPU와 GPU 모두, 여러 종목을 동시에 매수할 때 자금 부족으로 인한 문제를 피하기 위해 **우선순위가 높은 주문부터 순차적으로 자금을 차감**하는 로직을 사용합니다.

### 2. 신규 진입 (1차 매수)

-   **시점:** 장 마감 시.
-   **매수가 결정:**
    1.  **기준가(Price Basis):** 당일 **종가(Close Price)**.
    2.  **최종 매수가(Execution Price):** 기준가를 호가 단위에 맞춰 올림(`adjust_price_up`) 처리한 가격.

### 3. 추가 매수 (분할 매수 / Magic Split)

-   **조건:**
    1.  보유 중인 종목의 **당일 저가(Low Price)** 가 `매수 트리거 가격`에 도달하거나 하회.
    2.  `매수 트리거 가격 = 해당 종목의 마지막 분할 매수 단가 * (1 - additional_buy_drop_rate)`
    3.  당일 해당 종목에 대한 매도(수익실현, 손절 등)가 없었을 것.
    4.  총 분할 매수 횟수가 `max_splits_limit` 파라미터를 초과하지 않을 것.
    5.  **[핵심]** 당일 신규 진입한 종목은 추가 매수 대상에서 제외. (T+1 부터 가능)

-   **매수가 결정:**
    1.  **기준가(Price Basis):**
        -   **시나리오 A (장중 터치):** 당일 고가(High)가 `매수 트리거 가격`보다 높거나 같으면, 기준가는 `매수 트리거 가격`이 됩니다.
        -   **시나리오 B (갭 하락):** 당일 고가(High)가 `매수 트리거 가격`보다 낮으면(도달 실패), 기준가는 당일 **고가(High)** 가 됩니다.
    2.  **최종 매수가(Execution Price):** 위에서 결정된 기준가를 호가 단위에 맞춰 올림(`adjust_price_up`) 처리한 가격.

-   **추가 매수 우선순위:** 하루에 여러 종목이 추가 매수 조건에 해당될 경우, `additional_buy_priority` 파라미터에 따라 우선순위가 결정됩니다.
    -   `lowest_order`: 보유한 분할 매수 차수가 **가장 적은** 종목을 우선.
    -   `highest_drop`: 마지막 매수 단가 대비 현재가의 하락률이 **가장 큰** 종목을 우선.

> #### **CPU vs. GPU 구현 차이**
>
> -   **매수가 결정:** CPU(`execution.py`)와 GPU(`_process_additional_buy_signals_gpu`) 모두 시나리오 A/B를 구분하는 동일한 로직을 사용합니다. GPU에서는 `cp.where`를 사용해 벡터화 연산으로 처리합니다.
> -   **우선순위 처리 및 자금 경쟁:**
>     -   **CPU:** 모든 후보 신호를 생성한 뒤, 정해진 우선순위에 따라 **하나의 리스트로 정렬**합니다. 그 후 리스트의 맨 위부터 하나씩 순회하며 자금을 확인하고 차감합니다.
>     -   **GPU:** `_process_additional_buy_signals_gpu` 함수 내에서 모든 후보를 찾고, CPU와 동일한 기준으로 정렬합니다. 그 후 **`for` 루프를 순회**하며 임시 자본(`temp_capital`)을 순차적으로 차감하며 매수를 실행합니다. 이는 GPU의 병렬성을 다소 희생하더라도 CPU와 100% 동일한 결과를 보장하기 위한 핵심적인 설계입니다. (※ 현재 이 부분이 성능 병목 지점으로, 향후 `cumsum`을 이용한 병렬 알고리즘으로 개선될 예정입니다.)

## IV. 매도 원칙 (Sell Principles)

### 1. 공통 규칙

-   **매도 금액:** `floor((매도 체결가 * 수량) * (1 - sell_commission_rate - sell_tax_rate))`
-   **쿨다운 (Cooldown):** 특정 종목이 매도(수익실현, 손절 등)되면, 해당 종목은 `cooldown_period_days` 파라미터로 지정된 기간 동안 신규 매수 후보에서 제외됩니다.

### 2. 전체 청산 (Liquidation)

-   **대상:** 특정 종목의 모든 분할매수 포지션.
-   **조건 (아래 중 하나라도 해당되면 즉시 청산):**
    1.  **손절매 (Stop-Loss):**
        -   **조건:** 당일 **종가(Close)** 가 `해당 종목의 평균 매수 단가 * (1 + stop_loss_rate)` 이하로 하락.
        -   **매도가 결정:**
            1.  **기준가(Price Basis) 결정:** 추가 매수와 동일한 시나리오 A/B 로직을 적용합니다.
                -   **A (장중 도달):** 당일 고가(High)가 손절매 가격 이상이면, 기준가는 `손절매 가격`.
                -   **B (갭 하락):** 당일 고가(High)가 손절매 가격 미만이면, 기준가는 `당일 종가`.
            2.  **최종 매도가(Execution Price):** 위에서 결정된 기준가를 호가 단위에 맞춰 올림(`adjust_price_up`) 처리한 가격.
    2.  **최대 매매 미발생 기간 초과 (Inactivity):**
        -   **조건:** 해당 종목의 마지막 거래(매수 또는 매도)일로부터 `max_inactivity_period` 거래일 이상 경과.
        -   **매도가 결정:** `당일 종가`.

### 3. 부분 매도 (수익 실현 / Profit Taking)

-   **대상:** 각 분할매수 포지션 개별.
-   **조건:**
    1.  당일 **고가(High)** 가 `해당 포지션의 매수 단가 * (1 + sell_profit_rate)` 이상 도달.
    2.  **[핵심]** 당일 매수한 포지션은 수익 실현 대상에서 제외. (T+1 부터 가능)
-   **매도가 결정:**
    1.  **기준가:** `해당 포지션 매수 단가 * (1 + sell_profit_rate)`
    2.  **최종 매도가:** 기준가를 호가 단위에 맞춰 올림(`adjust_price_up`) 처리.

> #### **CPU vs. GPU 구현 차이**
>
> -   **로직:** 매도와 관련된 모든 조건(수익실현, 손절, 비활성) 및 체결가 결정 로직은 CPU와 GPU 간에 **완벽히 동일**합니다.
> -   **구현:** CPU는 객체와 `for` 루프를 사용하고, GPU는 `valid_positions`, `profit_taking_mask`, `stock_liquidation_mask` 등 boolean 마스크와 벡터화 연산을 사용하여 병렬로 처리하는 점만 다릅니다.

## V. 실행 및 계산 디테일

### 1. 호가 단위 (Tick Size)

-   **원칙:** 모든 매수/매도 주문의 최종 체결가는 한국 주식 시장의 호가 단위를 따릅니다. 계산된 가격(기준가)을 실제 체결 가능한 가장 가까운 높은 가격으로 올림 처리합니다.
-   **적용:** `adjust_price_up` 함수를 통해 구현되며, CPU와 GPU 모두 동일한 가격 구간별 호가 단위를 적용합니다.

| 가격 구간             | 호가 단위 |
| --------------------- | --------- |
| 2,000원 미만          | 1원       |
| 2,000원 ~ 4,995원     | 5원       |
| 5,000원 ~ 19,990원    | 10원      |
| 20,000원 ~ 49,950원   | 50원      |
| 50,000원 ~ 199,900원  | 100원     |
| 200,000원 ~ 499,500원 | 500원     |
| 500,000원 이상        | 1,000원   |

### 2. 부동소수점 정밀도 (Floating Point Precision)

-   **원칙:** CPU와 GPU 간의 계산 결과 일관성을 확보하기 위해, 모든 계산 과정에서 `float32` 타입을 명시적으로 사용합니다.
-   **특수 처리:** `float32` 타입의 나눗셈에서 발생할 수 있는 미세한 오차(예: `184.3000001`)가 올림/내림 결정에 영향을 주는 것을 방지하기 위해, 호가 단위 계산(`adjust_price_up`) 시 **소수점 5자리에서 반올림**하는 과정을 추가하여 오차를 보정합니다. 이 처리는 CPU와 GPU 양쪽에 모두 동일하게 적용되어 있습니다.
