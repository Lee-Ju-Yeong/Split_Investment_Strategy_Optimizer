# WFO Shortlist Derivation Review

> Type: `review`
> Status: `recorded`
> Priority: `P2`
> Last updated: `2026-03-19`
> Related issues: `#68`, `#98`
> Gate status: `official workflow unchanged; revisit only with same-cutoff A/B evidence`

## Summary
### What
- 이 문서는 `shortlist(압축된 후보 목록)`를 계속 `parameter simulation(후보를 넓게 찾는 단계)`에서 먼저 만들지, 아니면 `research WFO(연구용 WFO)` 단계에서 `full-grid`로 직접 만들지에 대한 현재 결론을 기록한다.

### Why
- 같은 논의가 반복되는 이유는 두 주장 모두 일부 진실을 갖고 있기 때문이다.
- `single-window shortlist`는 시점 과적합과 false negative(실제로는 괜찮은 후보를 너무 일찍 버리는 문제) 위험이 있다.
- 반대로 `research-stage WFO full-grid`는 더 넓게 보일 수 있지만, discovery와 approval evidence를 섞으면 `OOS(Out-of-Sample)`가 사실상 selector(선정 기준)로 바뀌는 문제가 생긴다.

### Current status
- 현재 공식 경로는 유지한다.
  1. `parameter simulation`
  2. `shortlist freeze`
  3. `research WFO`
  4. `promotion WFO`
  5. `CPU audit`
  6. `holdout`
- `research-stage WFO full-grid`는 **공식 기본 경로**로 채택하지 않는다.
- 현재 상태 판정은 `execute`도 `hold`도 아닌 `additional verification`이다.

### Next action
- 먼저 `same-cutoff A/B`를 돌려 현재 shortlist derivation(후보 추출 방식)이 실제로 promotion-grade 후보를 반복적으로 놓치는지 검증한다.

## 1. 한 페이지 결론
- `single-window shortlist` 우려는 타당하다.
  - 특정 historical window에서 약한 조합이 다른 대부분의 기간에서는 강할 수 있다.
- 하지만 이 우려만으로 `research-stage WFO full-grid`를 바로 공식 경로로 올리면 안 된다.
  - 그 순간 `WFO OOS`가 검증 데이터가 아니라 selector가 될 수 있다.
- 따라서 지금 단계의 권고는 이렇다.
  - 공식 workflow는 유지한다.
  - 먼저 `simulation ranking quality`와 `same-cutoff A/B evidence`를 확보한다.
  - 그 결과가 충분히 강할 때만 `discovery-only WFO full-grid`를 별도 source로 다시 검토한다.

## 2. 왜 단순 찬반으로 끝내면 안 되나
### 2-1. `single-window shortlist` 쪽 진짜 문제
- 한 시점 또는 한 historical path에 더 잘 맞는 후보가 shortlist에 과대표집될 수 있다.
- 반대로 그 구간에서는 약하지만 여러 시기에서 더 안정적인 후보는 너무 일찍 탈락할 수 있다.
- 즉, 이 구조는 `cheap proposal generator`로는 유용하지만, `sole exclusion gate`로 쓰기에는 약할 수 있다.

### 2-2. `research-stage WFO full-grid` 쪽 진짜 문제
- `research WFO` 안에서 top 8을 뽑기 시작하면, fold/anchor OOS가 shortlist membership에 직접 영향을 준다.
- 그러면 `연구용 분포 관찰`이 `후보 선정`으로 바뀌고, 같은 날짜가 여러 번 selector에 기여할 수도 있다.
- 이 경우 approval evidence의 의미가 약해지고, 이후 promotion/holdout 해석도 더 까다로워진다.

### 2-3. 그래서 지금 결론이 `유지 + 검증`인 이유
- 지금 필요한 것은 “새 lane을 믿고 열자”가 아니라 “현재 lane이 실제로 어떤 후보를 놓치고 있는지 같은 cutoff에서 증명하자”이다.
- 즉, 지금의 논점은 철학 싸움보다 `증적 부족`에 가깝다.

## 3. 이번 검토의 consensus
### 3-1. 합의점
- 현재 공식 workflow는 바꾸지 않는다.
- `single-window shortlist`는 temporal brittleness가 있을 수 있으므로, blind faith로 두면 안 된다.
- `research-stage WFO full-grid`가 허용되더라도 approval lane 안으로 들어오면 안 된다.
- promotion lane은 계속 `frozen_shortlist_single_anchor_eval` 성격을 유지해야 한다.
- holdout은 계속 `untouched final test`로 남아 있어야 한다.

### 3-2. 사실상 해소된 이견
- 초기 1차 의견에서는 일부가 `optional discovery lane` 설계를 먼저 열어도 된다고 봤다.
- 재숙의 후 최종 정리는 더 보수적으로 모였다.
  - **지금 당장은 lane 설계보다 A/B 검증과 metric alignment가 먼저**다.

### 3-3. 남아 있는 잔여 리스크
- `Selection bias`
  - research 실패 시 파라미터만 바꿔 재시도를 반복하면 OOS를 selector처럼 쓰게 된다.
- `Metric mismatch`
  - simulation에서 잘 뽑는 기준과 promotion에서 살아남는 기준이 다르면 좋은 후보를 미리 잃는다.
- `Single history bias`
  - 특정 outlier regime이 전체 simulation score를 왜곡할 수 있다.

## 4. 현재 결정
### 4-1. Decision
- 현재 결정:
  - `official workflow unchanged`
- 운영 문장으로는 이렇게 고정한다.
  - `parameter simulation -> frozen shortlist -> research/promotion WFO -> holdout`
- 이 문장을 뒤집는 논의는 아래 증적이 없으면 다시 열지 않는다.

### 4-2. Repeat-Discussion Guardrail
- 아래 증적이 없으면, 이 논의의 기본 답은 계속 `공식 경로 유지`다.
1. 같은 `decision_date` / 같은 `cutoff` / 같은 `promotion dates` / 같은 `holdout dates` 조건의 `same-cutoff A/B`
2. 고정된 `search space hash`, `runtime budget`, `anchor set`
3. deterministic rerun 가능 증적
4. 현재 shortlist derivation이 실제로 eventual promotion-grade 후보를 반복적으로 놓친다는 증적
5. full-grid discovery가 downstream `promotion hard gate`, `CPU audit`, `holdout` 품질을 실제로 개선한다는 증적

## 5. 나중에 다시 열리면 허용 가능한 형태
### 5-1. 허용 범위
- 다시 열리더라도 최대 허용 범위는 이것이다.
  - `discovery-only source`
- 즉:
  - `research-stage WFO full-grid`는 **shortlist를 만드는 upstream discovery source**로만 검토할 수 있다.
  - `promotion lane`이나 `holdout lane`을 대체하는 공식 경로가 되어서는 안 된다.

### 5-2. 그때도 유지해야 할 계약
- promotion은 계속 frozen shortlist만 평가한다.
- holdout 날짜는 shortlist derivation에 쓰지 않는다.
- selector는 composite curve, carry-over path, explanation artifact가 아니라 fold-level metrics만 써야 한다.
- raw top-8이 아니라 `gate first -> robust ranking -> plateau diversified representatives`가 원칙이어야 한다.

### 5-3. 필요해 보이는 provenance artifact
- future follow-up에서는 아래 같은 provenance가 필요하다.
  - `shortlist_source_manifest.json`
  - `source_mode=single_window_parameter_simulation|early_wfo_full_grid`
  - `source_run_id`
  - `source_windows`
  - `anchor_set_id`
  - `selection_metric`
  - `shortlist_hash`
- 후속 공식안은 별도 문서로 정리한다.
  - [N-window shortlist contract v1](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_19-n-window-shortlist-contract-v1.md)

## 6. 지금 바로 할 다음 액션
### 6-1. `P0` Simulation metric alignment
- `parameter simulation` 정렬 기준이 promotion hard gate와 얼마나 어긋나는지 확인한다.
- 질문:
  - simulation 상위 후보가 promotion에서는 얼마나 자주 광탈하는가

### 6-2. `P1` Same-cutoff A/B
- 동일 cutoff에서 아래 둘을 비교한다.
  - 현재 shortlist derivation
  - 개선된 derivation
- 최소 비교 항목:
  - shortlist overlap
  - false-negative rate
  - promotion pass yield
  - CPU audit outcome
  - holdout outcome

### 6-3. `P2` Retry policy
- research 실패 시 재시도 한도와, 실패 후 `parameter retune` 대신 `strategy logic review`로 넘어가는 규칙을 명시한다.

### 6-4. `P3` Provenance 기록
- freeze contract만이 아니라 `shortlist derivation provenance`도 남긴다.

## 7. 이 문서를 언제 다시 읽나
- 누군가 `WFO full-grid가 더 논리적으로 안전하지 않나?`라고 다시 묻는 경우
- `shortlist`를 왜 수동/frozen artifact로 넘기느냐는 질문이 나오는 경우
- `research lane` 안에서 바로 top 8을 뽑자는 제안이 다시 나오는 경우
- promotion lane과 discovery lane의 경계를 다시 설명해야 하는 경우

## 8. 관련 문서
- [WFO Approval Workflow Runbook](/root/projects/Split_Investment_Strategy_Optimizer/docs/operations/2026-03-14-wfo-approval-runbook.md)
- [WFO / OOS Lane 임시 합의안](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_12-wfo-oos-lane-provisional-review.md)
- [WFO 여정과 현재 상태](/root/projects/Split_Investment_Strategy_Optimizer/docs/briefings/2026-03-14-wfo-journey-and-status.md)
- [GPU-native WFO v2 Design Note](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_06-gpu-native-wfo-v2-design.md)
- [N-window shortlist contract v1](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_19-n-window-shortlist-contract-v1.md)
