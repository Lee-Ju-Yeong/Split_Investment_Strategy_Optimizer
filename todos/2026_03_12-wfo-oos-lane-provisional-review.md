# WFO / OOS Lane 임시 합의안

> Type: `review`
> Status: `draft`
> Priority: `P2`
> Last updated: `2026-03-14`
> Related issues: `#68`, `#98`, `#56`
> Gate status: `temporary design consensus only; not implemented`

## Summary
- What:
  - 이 문서는 `parameter simulation`, `WFO`, `OOS`를 어떤 순서와 규칙으로 써야 하는지 정리한 임시 합의안이다.
- Why:
  - 연구용 실험 결과와 승인용 증거를 섞어 쓰면 성과가 실제보다 좋아 보일 수 있기 때문이다.
- Current status:
  - `promotion evaluation lane`과 `research start-date robustness lane`을 분리하는 방향에는 합의가 있다.
  - 다만 아직 코드는 이 규칙을 완전히 강제하지 못하므로, 지금 단계는 `설계 메모`이지 `즉시 운영 규칙`은 아니다.
  - 추가 숙의 결과, 이 전략은 자본을 천천히 나눠 넣는 구조라서 `1년 미만 holdout`은 최종 승인용으로 약하다는 쪽으로 의견이 모였다.
  - 따라서 앞으로의 기본 정책은 `approval-grade final untouched holdout >= 24개월`로 두고, 현재 `2025-01-01 ~ 2025-11-30` 구간은 `internal provisional holdout`으로 낮춰 부르는 것이 안전하다는 정리가 추가되었다.
- Next action:
  - `#68`에서 `lane_mode`, `hard gate`, `CPU audit`, `holdout adequacy`, `artifact guardrail`을 코드와 문서에서 함께 잠근다.

## 1. 초심자용 한 페이지 설명
### 1-1. 이 문서를 한 줄로
- 좋은 후보를 먼저 넓게 찾고, 그다음 더 엄격한 방식으로 다시 검증하고, 마지막에는 따로 빼 둔 최신 구간으로 최종 시험을 본다.

### 1-2. 세 단계로 보면 쉽다
1. `Research lane`
   - 후보를 넓게 찾는 단계다.
   - 질문:
     - `어느 시점에 시작해도 성과가 너무 흔들리지 않는가`
2. `Promotion evaluation lane`
   - research에서 추린 후보를 더 엄격하게 보는 단계다.
   - 질문:
     - `시간이 앞으로 가도 이 후보가 계속 버티는가`
3. `Final untouched holdout`
   - 끝까지 따로 남겨 둔 마지막 시험지다.
   - 질문:
     - `정말 안 본 최신 구간에서도 여전히 통하는가`

### 1-3. 가장 중요한 결론만 6줄
- `parameter simulation`은 후보를 많이 찾는 단계다.
- `WFO`는 과거에서 고른 후보가 미래 구간에서도 버티는지 보는 단계다.
- `Research lane`은 `시작 시점 민감도`를 보는 단계다.
- `Promotion lane`은 `시간 전이 검증`을 보는 단계다.
- 앞으로의 기본 정책은 `approval-grade final untouched holdout >= 24개월`이다.
- 현재 저장소에서 실제로 남아 있는 구간 `2025-01-01 ~ 2025-11-30`은 `internal provisional holdout`으로 본다.
- `2020 crash`는 꼭 검증하지만, `final untouched holdout`이라고 부르지는 않는다.

### 1-4. 자주 나오는 말 아주 쉽게 풀기
- `IS (In-Sample)`:
  - 후보를 고를 때 보는 구간
- `OOS (Out-of-Sample)`:
  - 후보를 고른 뒤 처음 보는 검증 구간
- `WFO (Walk-Forward Optimization)`:
  - `IS -> OOS`를 시간을 앞으로 옮겨 가며 여러 번 반복하는 방식
- `Anchored WFO`:
  - 시작일은 고정하고, 학습 구간만 점점 늘려 가는 WFO
- `Multi-anchor Anchored WFO`:
  - 시작일을 몇 개로 바꿔 가며 Anchored WFO를 여러 번 반복하는 연구용 방식
- `Stress pack`:
  - 폭락장처럼 특별히 힘든 구간만 따로 보는 추가 시험
- `Plateau`:
  - 1등 한 점만 좋은 것이 아니라, 주변 설정들도 함께 괜찮은 넓은 구간
- `Holdout`:
  - 개발 중간에는 쓰지 않고 마지막에만 보는 최종 시험용 데이터
- `Internal provisional holdout`:
  - holdout처럼 따로 빼 두긴 했지만, 길이나 이력 때문에 `release-grade final proof`라고 강하게 부르지는 않는 구간
- `Audit`:
  - 다시 최적화하는 것이 아니라, 이미 고른 후보가 기준 엔진에서도 맞는지 확인하는 검산 절차

## 2. 이 문서가 답하려는 질문
### 2-1. 질문은 사실 2개다
- 질문 A:
  - `시간이 앞으로 가도 이 전략이 버티는가`
- 질문 B:
  - `투자를 시작한 날짜가 달라도 결과가 너무 흔들리지 않는가`

### 2-2. 왜 질문을 나눠야 하나
- 질문 A와 질문 B는 비슷해 보이지만 다르다.
- 질문 A는 `시간 전이 검증`이다.
  - 예:
    - 2019년에는 좋았는데 2022년 긴축장에서도 버티는가
- 질문 B는 `시작 시점 민감도` 검증이다.
  - 예:
    - 2018년에 시작하든 2020년에 시작하든 크게 다르지 않은가
- 이 둘을 한 실험으로 한꺼번에 설명하려고 하면 해석이 꼬이기 쉽다.
- 그래서 문서에서는 두 질문을 다른 lane으로 분리한다.

## 3. 지금 합의한 구조
### 3-1. Research lane
- 목적:
  - `시작 시점 민감도`를 본다.
- 권장 구조:
  - `multi-anchor Anchored WFO`
- 공식 research mode:
  - `frozen_shortlist_multi_anchor_eval`
- 아주 쉽게 말하면:
  - 한 번 뽑아 둔 shortlist를 여러 시작일(anchor)에서 반복 평가해 본다.
- 핵심 규칙:
  - anchor 집합은 실행 전에 미리 정한다.
  - 모든 anchor는 같은 `shortlist_hash`를 평가한다.
  - 각 fold는 모두 같은 `initial_cash`로 시작한다.
  - 이전 fold의 마지막 돈을 다음 fold에 넘기지 않는다.
  - `단일 합성 equity curve`를 만들지 않는다.
  - 결과는 `anchor별/fold별 metric 분포`로 본다.
- 이 lane에서 할 수 있는 말:
  - `시작 시점이 바뀌어도 성과 분포가 과하게 흔들리지 않았다`
- 이 lane에서 하면 안 되는 말:
  - `이 결과가 곧 release-grade final proof다`

### 3-2. Promotion evaluation lane
- 목적:
  - `시간 전이 검증`을 본다.
- 권장 구조:
  - `single-anchor`, `non-overlap Anchored WFO`
- 아주 쉽게 말하면:
  - 한 고정 출발점에서 시간을 앞으로 걸어가며, 다른 장세에서도 계속 버티는지 본다.
- 핵심 규칙:
  - `strict_pit`
  - `candidate_source_mode=tier`
  - 같은 fold 안에서 `IS`와 `OOS`는 겹치지 않는다.
  - promotion lane의 `OOS` 날짜는 fold끼리 재사용하지 않는다.
  - CPU `audit`은 `pass/fail`에 가깝게 쓴다.
- 이 lane에서 할 수 있는 말:
  - `고정 출발점에서 시간 전이 검증을 통과했다`

### 3-3. Final untouched holdout
- 목적:
  - 최종 승인 직전 마지막 시험을 본다.
- approval-grade 기본 규칙:
  - `24개월 이상의 연속된 untouched 구간`
- 아주 쉽게 말하면:
  - 이 전략은 한 번에 풀인하지 않고 천천히 진입하고, 추가매수와 청산도 시간이 걸리므로, 짧은 구간만 보면 `들어가는 능력`만 보고 `빠져나오는 능력`은 덜 볼 수 있다.
- 현재 저장소 상태:
  - `2025-01-01 ~ 2025-11-30`는 남겨 둔 최신 구간이긴 하지만, 길이가 짧아서 `approval-grade final proof`보다는 `internal provisional holdout`으로 부르는 것이 안전하다.
- 이 구간은 아래 용도로 미리 쓰면 안 된다.
  - research lane 관찰
  - promotion WFO OOS
  - shortlist 수정 근거
- 즉, holdout은 `끝까지 남겨 둔 시험지`여야 한다.

### 3-4. 왜 2년 holdout이 필요해 보이나
- 이 전략은 `order_investment_ratio` 비율로 자본을 나눠 넣고, 추가매수도 여러 번 일어날 수 있다.
- 그래서 1년 안쪽 구간은 아래 위험이 있다.
  - 포트폴리오가 아직 충분히 차기 전에 끝날 수 있다.
  - 하락장에서 물량만 모으고, 반등이나 강제 청산으로 `빠져나오는 구간`은 충분히 못 볼 수 있다.
  - 종료 시점에 미청산 포지션이 많아 `평가상 좋아 보이는 착시`가 남을 수 있다.
- 그래서 문서의 기본 방향은:
  - `approval-grade final untouched holdout >= 24개월`
  - 다만 짧은 holdout은 `internal provisional` 또는 예외 승인(pack)으로만 취급

### 3-5. 왜 `2024-01-01 ~ 2025-12-31`을 지금 바로 untouched라고 부르면 안 되나
- 직감적으로는 2년이 좋아 보이지만, 현재 저장소 이력상 그 구간은 `완전히 안 본 새 시험지`가 아니다.
- 이유:
  - `2024-01-01 ~ 2024-12-31`는 이미 promotion WFO 예시의 최신 OOS로 써 왔다.
  - `2025-12-01 ~ 2026-01-31`는 parity canary와 release-readiness 확인에 사용된 적이 있다.
- 따라서 `2024-01-01 ~ 2025-12-31`를 지금 와서 `final untouched holdout`이라고 다시 부르면 더 엄격해지는 것이 아니라, 오히려 말이 꼬인다.
- 안전한 표현은 이렇다.
  - `앞으로는 24개월 untouched holdout을 기본 목표로 삼는다`
  - `하지만 현재 cycle의 2025 구간은 internal provisional holdout으로 남긴다`

### 3-6. 왜 `2025-12-01 ~ 2026-01-31`은 canonical holdout이 아닌가
- 이 구간은 이미 `parity canary`와 `release-readiness` 확인에 사용된 적이 있다.
- 여기서 중요한 것은 `parity`라는 단어 자체가 아니라, 그 기간 데이터를 이미 판단과 검산에 써버렸다는 점이다.
- 그래서 이 구간은 `완전히 안 본 순수한 holdout`이라고 강하게 주장하기 어렵다.
- 따라서 이 구간은 `untouched holdout`에서 제외한다.

## 4. Research lane을 왜 이렇게 설계하나
### 4-1. Anchored WFO 하나만으로는 부족한 이유
- `Anchored WFO` 하나는 이런 질문에 강하다.
  - `한 고정 출발점에서 시간이 지나도 버티는가`
- 하지만 이런 질문에는 충분하지 않다.
  - `내가 어느 날짜에 투자를 시작해도 비슷한가`
- 그래서 `start-date robustness`를 보려면 시작일을 몇 개로 바꿔 가며 같은 실험을 반복해야 한다.
- 그 역할을 하는 것이 `multi-anchor Anchored WFO`다.

### 4-2. 왜 `multi-anchor Anchored WFO`가 직관적인가
- 각 anchor는 `하나의 투자 시작 시점 가정`이라고 생각하면 된다.
- 예:
  - Anchor A:
    - `2014-01-01`부터 시작한다고 가정
  - Anchor B:
    - `2015-01-01`부터 시작한다고 가정
  - Anchor C:
    - `2016-01-01`부터 시작한다고 가정
- 이렇게 하면 아래 질문을 자연스럽게 볼 수 있다.
  - `시작을 1년 늦춰도 괜찮은가`
  - `과거 정보가 조금 줄어도 괜찮은가`

### 4-3. 왜 research lane에서 `단일 합성 equity curve`를 금지하나
- research lane은 여러 anchor와 여러 fold를 본다.
- 이 결과를 하나의 curve로 평균내거나 이어 붙이면 실제보다 덜 위험해 보일 수 있다.
- 쉽게 말하면:
  - 여러 번 본 시험 점수를 평균내고
  - 그것을 실전 한 번 본 시험 성적표처럼 보여 주는 꼴이 된다.
- 그래서 research lane의 기본 산출물은 `분포`여야 한다.
- 예:
  - fold별 수익률 분포
  - anchor별 MDD 분포
  - worst-case fold
  - pass rate

### 4-4. research lane에서 꼭 지켜야 할 최소 규칙
- 각 fold는 동일 `initial_cash`로 시작한다.
- 이전 fold 종료 자금을 다음 fold로 carry-over 하지 않는다.
- anchor별 refit, anchor별 재최적화는 공식 research mode에 넣지 않는다.
- 공식 research mode는 `frozen_shortlist_multi_anchor_eval` 하나로 잠근다.
- `anchor_specific_refit_exploration`은 별도 탐색 노트는 가능하지만 공식 evidence lane에는 넣지 않는다.

## 5. Promotion lane은 왜 따로 남겨 두나
### 5-1. research lane을 이미 봤는데 또 왜 필요하나
- research lane은 `후보를 찾고 거르는 단계`다.
- promotion lane은 `그 후보가 정말 승인 심사를 통과할 만한가`를 보는 단계다.
- 즉 둘은 같은 일을 두 번 하는 것이 아니라, 다른 질문에 답한다.

### 5-2. promotion lane이 의미 있으려면
- shortlist를 research 단계에서 먼저 freeze 해야 한다.
- promotion lane은 research보다 더 엄격한 규칙을 써야 한다.
- promotion 결과를 보고 다시 무한히 후보를 바꾸지 않아야 한다.
- 즉 promotion lane은 `재탐색`이 아니라 `정당성 점검`이어야 한다.

### 5-3. research와 promotion은 어떻게 이어지나
1. `research_data_cutoff <= 2024-12-31`
2. shortlist freeze
3. `promotion evaluation lane`
4. `final untouched holdout` 또는 `internal provisional holdout`

## 6. audit, hard gate, robust score를 쉽게 설명하면
### 6-1. audit
- `audit`은 다시 최적화하는 과정이 아니다.
- 이미 뽑힌 후보가 CPU `SSOT`에서도 맞는지 확인하는 `검산`이다.
- 좋은 audit:
  - parity mismatch 확인
  - 규칙 위반 확인
  - `pass/fail` 기록
- 나쁜 audit:
  - CPU 결과를 보고 사실상 새 후보를 다시 고르는 것

### 6-2. hard gate
- `hard gate`는 1차 탈락 기준이다.
- 아주 쉽게 말하면:
  - `최소한 이 정도는 버텨야 다음 단계로 갈 수 있다`

### 6-3. robust score
- `robust score`는 최종 당락을 바로 정하는 점수가 아니다.
- 역할:
  - `hard gate`를 통과한 후보들 안에서 tie-break
- 즉 순서는 이렇게 본다.
  1. 먼저 gate를 통과했는가
  2. 통과한 후보끼리만 robust score를 비교한다

## 7. 날짜와 예시로 보면 더 쉽다
### 7-1. canonical 날짜 partition
- `Development + Research + Promotion WFO`:
  - `2013-11-20 ~ 2024-12-31`
- `Current internal provisional holdout`:
  - `2025-01-01 ~ 2025-11-30`
- `Strict untouched claim에서 제외`:
  - `2025-12-01 ~ 2026-01-31`
- `Future approval-grade target`:
  - `24개월 이상의 untouched 연속 구간`

### 7-2. Promotion lane 예시
- 아래 표는 `promotion evaluation lane` 예시다.

| Fold | IS Period | OOS Period | 쉬운 설명 |
| --- | --- | --- | --- |
| 1 | `2014-01-01 ~ 2018-12-31` | `2019-01-01 ~ 2019-12-31` | 평시 검증 |
| 2 | `2014-01-01 ~ 2019-12-31` | `2020-01-01 ~ 2020-12-31` | crash / 팬데믹 |
| 3 | `2014-01-01 ~ 2020-12-31` | `2021-01-01 ~ 2021-12-31` | 유동성 장세 |
| 4 | `2014-01-01 ~ 2021-12-31` | `2022-01-01 ~ 2022-12-31` | 긴축 하락장 |
| 5 | `2014-01-01 ~ 2022-12-31` | `2023-01-01 ~ 2023-12-31` | 회복기 |
| 6 | `2014-01-01 ~ 2023-12-31` | `2024-01-01 ~ 2024-12-31` | 최신 장세 |

### 7-3. Research lane 예시
- 아래 표는 `research start-date robustness lane` 예시다.
- 모든 anchor는 같은 `shortlist_hash`를 평가한다.
- 모든 anchor는 `2024-12-31` 이전 데이터만 사용한다.

| Anchor | 첫 IS 시작일 | 쉬운 설명 |
| --- | --- | --- |
| A | `2014-01-01` | 가장 긴 과거 정보를 가진 시작점 |
| B | `2015-01-01` | 시작을 1년 늦췄을 때도 비슷한지 보기 |
| C | `2016-01-01` | 더 짧은 과거 정보로 시작해도 버티는지 보기 |

- 이 lane의 해석 예시:
  - `A/B/C 결과가 비슷하다`
    - 시작 시점 민감도가 상대적으로 낮다
  - `A는 좋고 C는 많이 나쁘다`
    - 시작 시점 민감도가 크다

### 7-4. 2020 crash는 어떻게 다루나
- `2020-03`은 빼지 않는다.
- 다만 아래처럼 부른다.
  - `WFO validation fold`
  - `stress-labeled regime`
- 아래 주장은 금지한다.
  - `2020년 OOS는 final untouched holdout이다`
  - `2020년 WFO OOS와 stress 결과를 합산한 KPI가 최종 승인 성능이다`

### 7-5. holdout이 충분한지 무엇을 같이 보나
- 기간만 길다고 자동으로 충분한 것은 아니다.
- 이 전략은 천천히 진입하고, 추가매수와 청산이 시간이 걸리므로 아래도 같이 봐야 한다.
  - `trade_count`
  - `closed_trade_count`
  - `avg_hold_days`
  - `distinct_entry_months`
  - `peak_slot_utilization`
  - `realized_split_depth`
- 그리고 이 전략에서는 `끝날 때 얼마나 안 팔렸는가` 하나보다 `기간 동안 자본이 얼마나 실제로 배치되었는가`를 더 중요하게 본다.
- 그래서 보조 관찰용으로는 아래가 더 자연스럽다.
  - `avg_invested_capital_ratio`
  - `cash_drag_ratio`

## 8. 구현용 메모
### 8-1. hard gate 초안
- 아래는 아직 `#68` 공식안이 아니라 임시 출발점이다.
- 해석 규칙:
  - `hard gate`가 1차 selector다.
  - `robust score`는 `hard gate 통과 후보` 안에서만 tie-break로 쓴다.
- 후보 gate 예시:
  - `median(OOS/IS) >= 0.60`
  - `fold_pass_rate >= 70%`
  - `OOS_MDD_p95 <= 25%`
- 추가 stress reject 예시:
  - `2020` 또는 `2022` fold에서 `MDD > 30%`
  - `Recovery Factor < 1.0`

### 8-2. manifest는 왜 남기나
- 목적:
  - 나중에 `이 결과가 어떤 규칙으로 만들어졌는지`를 증명하기 위해서다.
- 쉽게 말하면:
  - `실험 설명서`와 `날짜 경계 증명서`를 함께 남기는 것이다.

### 8-3. 남겨야 하는 manifest
- `lane_manifest.json`
  - 이 run이 research인지 promotion인지, 승인 가능한 evidence인지 등을 기록한다.
- `holdout_manifest.json`
  - holdout 구간과 WFO 종료일을 기록한다.
- `anchor_manifest.json`
  - 어떤 anchor 집합을 썼는지 기록한다.

### 8-4. 필수 필드
- `lane_manifest.json`
  - `lane_type`
  - `evidence_tier`
  - `approval_eligible`
  - `decision_date`
  - `research_data_cutoff`
  - `promotion_data_cutoff`
  - `shortlist_hash`
  - `publication_lag_policy`
  - `ticker_universe_snapshot_id`
  - `engine_version_hash`
  - `composite_curve_allowed`
  - `cpu_audit_outcome`
- `holdout_manifest.json`
  - `holdout_start`
  - `holdout_end`
  - `wfo_end`
  - `holdout_date_reuse_forbidden`
  - `parity_canary_excluded_ranges`
  - `holdout_class`
    - 예:
      - `approval_grade`
      - `internal_provisional`
  - `trade_count`
  - `closed_trade_count`
  - `distinct_entry_months`
  - `avg_invested_capital_ratio`
  - `cash_drag_ratio`
- `anchor_manifest.json`
  - `anchor_set_id`
  - `anchor_dates`
  - `anchor_spacing_rule`
  - `minimum_is_length_days`
  - `minimum_oos_length_days`
  - `shortlist_freeze_mode=frozen_shortlist_multi_anchor_eval`
  - `coverage_normalized`

## 9. 지금 구현에 대한 경고
- 현재 WFO 구현은 아직 이 문서의 철학을 그대로 강제하지 않는다.
- 특히 아래가 남아 있다.
  - overlap 경로 허용
    - [walk_forward_analyzer.py](/root/projects/Split_Investment_Strategy_Optimizer/src/analysis/walk_forward_analyzer.py:542)
  - OOS 초기자금 carry-over
    - [walk_forward_analyzer.py](/root/projects/Split_Investment_Strategy_Optimizer/src/analysis/walk_forward_analyzer.py:662)
  - 중복 OOS 날짜 평균 합성 curve
    - [walk_forward_analyzer.py](/root/projects/Split_Investment_Strategy_Optimizer/src/analysis/walk_forward_analyzer.py:691)
  - CPU certification 일부 rerank 성격
    - [walk_forward_analyzer.py](/root/projects/Split_Investment_Strategy_Optimizer/src/analysis/walk_forward_analyzer.py:620)
- 따라서 이 문서는 `즉시 운영 승인`이 아니라 `공식 구현 전에 맞춰 둘 설계 메모`다.

## 10. 다음 공식화 후보
1. `#68` 문서에서 `lane_mode`와 날짜 경계를 공식 규칙으로 잠근다.
2. research lane을 `frozen_shortlist multi-anchor Anchored WFO`로 공식화한다.
3. promotion lane을 `single-anchor non-overlap Anchored WFO`로 공식화한다.
4. research lane에서 `단일 합성 equity curve`를 코드로 차단한다.
5. `approval-grade holdout >= 24개월` 기본 정책과 `internal provisional holdout` 예외 표기를 문서와 코드에서 같이 고정한다.
6. `hard gate`와 `robust score tie-break` 역할을 코드와 문서에서 하나로 맞춘다.
7. `CPU certification`을 rerank가 아니라 `audit/pass-fail`로 고정한다.
8. `lane_manifest.json`, `holdout_manifest.json`, `anchor_manifest.json` 포맷을 구현에 연결한다.

## 11. 참고 문서
- [#68 Robust WFO / Ablation](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_02_09-issue68-robust-wfo-ablation.md)
- [#98 GPU Throughput Refactor](/root/projects/Split_Investment_Strategy_Optimizer/todos/done_2026_02_17-issue98-gpu-throughput-refactor.md)
- [WFO analysis context](/root/projects/Split_Investment_Strategy_Optimizer/llm-context/04_wfo_analysis.md)
