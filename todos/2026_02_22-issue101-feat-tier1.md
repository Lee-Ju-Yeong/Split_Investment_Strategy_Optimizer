# Issue #101: Tier1 선택 편향 완화와 분포 기반 강건 최적화

> Type: `research`
> Status: `planned`
> Priority: `P2`
> Last updated: 2026-03-07
> Related issues: `#101`, `#68`, `#67`, `#56`
> Gate status: `not started`

## 1. Summary
- What: 특정 종목이 반복 선택되는 편향보다, `theta x scenario x fold` 분포에서 잘 버티는 파라미터를 고르는 프레임으로 옮기려는 연구입니다.
- Why: 현재 결정론 정렬은 재현성은 좋지만, 조건 자체의 일반화 성능보다 특정 종목 포함 여부에 성과가 과민할 수 있습니다.
- Current status: 방향과 선택지 정리까지는 끝났고, 실제 운영 트랙은 아직 고르지 않았습니다.
- Next action: `shadow -> gated -> default` 방식으로 갈지, 더 가벼운 편향 완화안으로 시작할지 결정합니다.

## 2. Fixed Rules
- `candidate_source_mode=tier` 유지
- `signal_date=T-1`, PIT/as-of 조회 유지
- execution/fee/tick/rounding 규칙은 탐색 대상에서 제외
- parity mismatch, coverage 악화, entry 지표 악화 시 즉시 rollback 가능해야 함

## 3. Candidate Directions
- `A. 저비용 완화`
  - quota나 cap 편향 완화로 결정론을 유지
- `B. 분포 평가 중심`
  - `theta x omega x fold` 단위 평가
- `C. hard gate 중심 운영형`
  - robust score + 운영 gate + `shadow -> gated -> default`

## 4. Recommended Starting Point
- 현재 권고는 `C`입니다.
- 이유:
  - 운영 판단으로 바로 연결하기 쉽습니다.
  - 기존 런타임 지표(`empty_entry_day_rate`, `tier1_coverage`, `tier2_fallback_rate`)를 재사용할 수 있습니다.
  - `#68`의 robust score 레이어와 연결하기 좋습니다.

## 5. Current Plan
- [ ] 평가 단위 고정
  - `theta`
  - `omega`
  - `fold`
- [ ] hard gate 초안 고정
  - `median(OOS/IS) >= 0.60`
  - `fold_pass_rate >= 70%`
  - `OOS_MDD_p95 <= 25%`
  - `P95(empty_entry_day_rate) <= 0.20`
  - `median(tier1_coverage) >= 0.55`
- [ ] scenario set 상한 고정
- [ ] 결과 저장 스키마 고정
- [ ] shadow 관찰 기간과 rollback trigger 고정

## 6. Gate And Evidence
- 시작 전 선행 조건:
  - `#67` 정책 안정화
  - `#56` parity strict 유지
- 완료 판단:
  - 분포 기반 선택이 실제 운영 gate와 함께 설명 가능
  - 결과 저장 규격과 replay 규격이 존재
  - shadow 관찰 기준이 문서화됨
