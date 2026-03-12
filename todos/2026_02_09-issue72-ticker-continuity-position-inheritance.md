# Issue #72: Ticker Continuity & Position Inheritance

> Type: `implementation`
> Status: `planned`
> Priority: `P1`
> Last updated: 2026-03-07
> Related issues: `#72`, `#70`, `#67`
> Gate status: `not started`

## 1. Summary
- What: 합병, 분할, 지주사 전환, 티커 변경이 있어도 포지션과 tier/universe 판단을 끊기지 않게 만드는 작업입니다.
- Why: 지금은 ticker를 완전히 독립된 자산으로 보기 때문에, 실질적으로 같은 경제적 실체여도 포지션이 강제로 끊깁니다.
- Current status: 문제 정의만 되어 있고, 실제 데이터 모델과 acceptance fixture는 아직 없습니다.
- Next action: 대표 케이스를 수집하고 `TickerAncestry` 수준의 데이터 모델 초안을 고정합니다.

## 2. Scope And Constraints
- In scope:
  - 대표 이벤트 샘플링
  - continuity 데이터 모델 설계
  - CPU/GPU acceptance fixture 정의
- Out of scope:
  - 모든 기업 이벤트의 자동 완전 추론
  - 우선순위가 낮은 특수 케이스의 즉시 구현
- Constraints:
  - 포지션 연속성은 수량, 평단, tier 판단에 모두 영향을 줍니다.
  - 단순 티커 치환이 아니라 exchange ratio 같은 수학적 보정 규칙이 필요할 수 있습니다.

## 3. Current Plan
- [ ] 대표 케이스 5~10개 수집
  - 예: 티커 변경, 합병, 분할, 지주사 전환
- [ ] continuity 데이터 모델 초안 작성
  - 후보: `TickerAncestry`
- [ ] 포지션 carryover acceptance fixture 설계
  - CPU
  - GPU
- [ ] exchange ratio 보정 규칙 정의
- [ ] tier / universe continuity 규칙 정의

## 4. Gate And Evidence
- Acceptance criteria:
  - continuity 적용 전후 차이를 설명할 샘플 fixture가 존재
  - carryover 후에도 cash/quantity/avg price accounting이 깨지지 않음
  - tier/universe continuity가 PIT 규칙을 위반하지 않음

## 5. Notes
- 이 문서는 아직 설계 초안입니다.
- 실제 구현을 시작하기 전까지는 `대표 케이스`, `보정 규칙`, `acceptance fixture` 3가지를 먼저 확정해야 합니다.
