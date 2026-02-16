# 전략/파라미터 운영 기준서 (Strategy Parameter Playbook)

기준일: 2026-02-16  
상태: v1 (운영 전 검증 단계)

## 1) 문서 목적

본 문서는 매직스플릿 전략의 핵심 의사결정(종목 선정, Tier, 매수/매도, 최적화/승격 게이트)을
한 곳에서 확인하기 위한 운영 기준서입니다.

- 소스 오브 트루스:
  - 로직: `src/backtest/cpu/strategy.py`, `src/backtest/cpu/backtester.py`
  - 최적화 정렬: `src/optimization/gpu/analysis.py`
  - 우선순위/로드맵: `TODO.md`

## 2) 2026-02-16 합의 결정 (v1)

1. 데이터 학습/검증 시작일은 `2013-11-20` 이후로 고정한다.
2. 후보군 모드는 `candidate_source_mode="tier"`를 기본으로 사용한다.
3. 파라미터 탐색 랭킹은 `Calmar Ratio` 단일 기준으로 유지한다.
4. 승격 게이트는 OOS 강건성 기준을 사용한다.
5. 초기 범위는 공통 파라미터(shared) 최적화로 제한하고 종목별/차수별 최적화는 보류한다.
6. CPU parity 검증 범위는 `Top-100` 파라미터로 설정한다.
7. v1 승격 게이트의 `OOS MDD p95` 기준은 `<= 30%`를 적용한다.

## 3) 종목 선정/Tier 기준

### 3-1. 후보군 소스

- 기본값: `tier`
- 구현 파라미터: `strategy_params.candidate_source_mode`
- 관련 로직:
  - `src/backtest/cpu/strategy.py`의 `mode in ["tier", "hybrid_transition"]` 분기
  - `get_candidates_with_tier_fallback(signal_date)` 호출

### 3-2. Tier fallback 규칙

- 1차: `tier=1` 후보군 조회
- fallback: 비어 있으면 `tier<=2` 후보군 조회
- `signal_date(T-1)` 기준 PIT 조회를 사용한다.

### 3-3. 신규 진입 후보 우선순위

1. ATR(14) 비율 내림차순
2. 종목코드 오름차순(동점 처리)
3. `max_stocks - 현재 보유 수` 만큼만 진입

## 4) 매수/매도 기준 (요약)

### 4-1. 실행 순서

- 일자별 처리 순서: `매도 -> 신규 매수 -> 추가 매수`
- 관련 로직: `src/backtest/cpu/backtester.py`

### 4-2. 신규 매수

- 후보군: Tier 후보군 중 미보유 + 쿨다운 아님
- 주문 금액: 월 1회 재산정된 `investment_per_order`
- 체결 전제: 현금/수수료 조건 충족 시 체결

### 4-3. 추가 매수

- 트리거: `당일 저가 <= 마지막 매수가 * (1 - additional_buy_drop_rate)`
- 제한:
  - 당일 매도 종목 제외
  - 당일 신규진입 종목 제외
  - `len(positions) < max_splits_limit`
- 우선순위:
  - `lowest_order` 또는 `highest_drop`

### 4-4. 매도

- 손절: 평균매수가 기준 `stop_loss_rate`
- 비활성 청산: `max_inactivity_period`
- 익절: 개별 포지션 기준 `sell_profit_rate`
- 공통: 쿨다운 적용

## 5) 파라미터 운영 범위

### 5-1. v1 탐색 대상 (shared only)

- `max_stocks`
- `order_investment_ratio`
- `additional_buy_drop_rate`
- `sell_profit_rate`
- `additional_buy_priority`
- `stop_loss_rate`
- `max_splits_limit`
- `max_inactivity_period`

### 5-2. v1 고정/금지

- 고정:
  - `candidate_source_mode="tier"`
  - `use_weekly_alpha_gate=false`
- 보류(추후 이슈):
  - 종목별 파라미터 상이 적용
  - 차수(order)별 개별 파라미터 최적화

## 6) 비용 시나리오 (2B 합의)

기본 config 비용(`buy=0.00015`, `sell=0.00015`, `tax=0.0018`)에 슬리피지 스트레스를 추가한다.

1. `S0_BASE`:
   - buy=0.00015, sell=0.00015, tax=0.0018
   - 총 비용(단순합 기준): 약 0.21%
2. `S1_MODERATE`:
   - buy=0.00065, sell=0.00065, tax=0.0018
   - 총 비용(단순합 기준): 약 0.31%
3. `S2_SEVERE`:
   - buy=0.00115, sell=0.00115, tax=0.0018
   - 총 비용(단순합 기준): 약 0.41%

참고: `S1/S2`는 운영 기본값 변경이 아니라 강건성 테스트용 시나리오이다.

## 7) 최적화/승격 게이트

### 7-1. 탐색 목표 (Search Objective)

- 정렬 기준: `calmar_ratio` 내림차순
- 구현 위치: `src/optimization/gpu/analysis.py`

### 7-2. 승격 게이트 (Promotion Gate)

아래 조건을 모두 통과해야 `GO`:

1. `median(OOS/IS, Calmar) >= 0.60`
2. `fold pass rate >= 70%`
3. `OOS MDD p95 <= 30%`
4. CPU/GPU parity mismatch `0건`
5. `Top-100` 파라미터 CPU 재검증 완료

## 8) 실행 체크리스트 (요약)

1. `config/config.yaml` 백업
2. `candidate_source_mode="tier"` / `use_weekly_alpha_gate=false` 확인
3. 백테스트 시작일 `2013-11-20` 이후 설정
4. `S0/S1/S2` 비용 시나리오별 WFO 실행
5. 상위 100개 파라미터 CPU parity 검증
6. 게이트 판정표 작성 후 `GO/NO-GO` 결정

## 9) 변경 관리

- 본 문서 변경 시 `TODO.md`의 관련 노트/이슈 상태를 함께 갱신한다.
- 전략/체결 로직 변경 없이 문서/선정·검증 계층부터 적용한다.
