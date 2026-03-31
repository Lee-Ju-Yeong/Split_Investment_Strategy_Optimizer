# Cluster V1 Shortlist Diversification PoC

> Type: `research`
> Status: `proposed`
> Priority: `P1`
> Last updated: `2026-03-24`
> Related issues: `#68`, `#98`
> Gate status: `discovery-only compare mode`

## 1. 한 페이지 결론
- 이 문서는 `N-window shortlist derivation`에 `cluster_v1`를 바로 기본값으로 넣기 전에, 작은 비교 실험(PoC)으로 검증하기 위한 설계 초안이다.
- 핵심 목적은 `더 좋은 1등을 자동으로 찾기`가 아니라 `shortlist 후보의 성격 다양화`다.
- `cluster_v1`를 쓰더라도 최종 freeze 후보는 반드시 `actual row-based medoid(클러스터 중심에 가장 가까운 실제 후보)`만 허용한다.
- `cluster centroid(가상의 평균 파라미터)`는 분석용으로만 보고, shortlist나 promotion 입력으로는 쓰지 않는다.
- 이 PoC는 `discovery-only`이며, `promotion WFO`, `CPU audit`, `holdout` 로직은 바꾸지 않는다.

## 2. 왜 이 PoC가 필요한가
- 현재 `family_excluded_parameters` 방식은 설명 가능성이 높고 안전하다.
- 하지만 실제 dual-window core6 결과를 보면 후보 6개가 여전히 비슷한 성격으로 몰릴 수 있다.
- 그래서 다음 질문을 작은 실험으로 확인하려는 것이다.
  - clustering을 넣으면 정말 더 다양한 후보가 나오나?
  - 그 다양성이 downstream promotion에서 실제로 도움이 되나?
  - 아니면 복잡도만 늘고 성과는 비슷한가?

## 3. 이 PoC가 답하려는 질문
1. `current family grouping`보다 `cluster_v1`가 shortlist 다양성을 실제로 늘리는가?
2. 다양성이 늘어난 결과가 `promotion pass yield`나 `eventual champion recall`에도 도움이 되는가?
3. clustering을 쓰더라도 `freeze contract(후보 봉인 규칙)`와 `provenance(출처 추적)`를 그대로 유지할 수 있는가?

## 4. 범위와 비범위
### 4-1. 포함
- `src/analysis/shortlist_derivation.py` 내부의 선택형 모드 추가
- `window_bundle_manifest.json` / `shortlist_source_manifest.json`에 clustering 메타데이터 기록
- `same-cutoff A/B` 비교 실험

### 4-2. 제외
- `promotion lane`에서 clustering 재실행
- `holdout`에서 여러 후보를 다시 비교하는 것
- `k auto-search(군집 수 자동 탐색)`
- `metric-based clustering feature(성과 지표를 군집 feature로 직접 쓰는 것)`
- `synthetic centroid freeze`

## 5. 권장 설계
### 5-1. 모드 이름
- `family_grouping_mode = explicit_key_v1`
  - 현재 기본값
- `family_grouping_mode = cluster_v1`
  - 선택형 실험 모드

### 5-2. cluster_v1 입력 원칙
- cluster feature는 `parameter-only`로 제한한다.
- 1차 권장 feature:
  - `order_investment_ratio`
  - `additional_buy_drop_rate`
  - `sell_profit_rate`
  - `max_inactivity_period`
- 보조 참고:
  - `stop_loss_rate`는 저가중치 보조축 후보
  - `max_stocks`, `max_splits_limit`, `additional_buy_priority`는 현재 grid에서 고정 폭이 좁으면 제외 가능

### 5-3. cluster_v1 금지 원칙
- `calmar_ratio`, `cagr`, `mdd` 같은 metric을 cluster feature로 직접 넣지 않는다.
- 이유:
  - upstream shortlist 단계가 다시 `성과 selector`가 되기 쉽다.
  - downstream WFO의 selector와 역할이 겹친다.

### 5-4. representative rule
- 각 클러스터에서 아래 둘 중 하나만 허용:
  - `actual_row_medoid`
  - `nearest_valid_member`
- 기본값:
  - `actual_row_medoid`

## 6. manifest에 남겨야 할 정보
### 6-1. window bundle 쪽
```json
{
  "selection_contract": {
    "selection_metric": "calmar_ratio",
    "shortlist_size": 6,
    "family_grouping_mode": "cluster_v1",
    "cluster_contract": {
      "cluster_contract_version": "cluster_v1",
      "cluster_feature_keys": [
        "order_investment_ratio",
        "additional_buy_drop_rate",
        "sell_profit_rate",
        "max_inactivity_period"
      ],
      "cluster_feature_weights": {
        "order_investment_ratio": 1.0,
        "additional_buy_drop_rate": 1.0,
        "sell_profit_rate": 1.0,
        "max_inactivity_period": 1.0
      },
      "cluster_distance_metric": "weighted_l1",
      "cluster_count_mode": "fixed_k",
      "cluster_count_k": 6,
      "cluster_random_seed": 42,
      "representative_selection_rule": "actual_row_medoid"
    }
  }
}
```

### 6-2. shortlist source manifest 쪽
- `family_grouping_mode`
- `cluster_contract_version`
- `cluster_feature_keys`
- `cluster_feature_weights`
- `cluster_distance_metric`
- `cluster_count_mode`
- `cluster_count_k`
- `cluster_random_seed`
- `representative_selection_rule`
- `cluster_assignment_hash`
- `shortlist_source_manifest_hash`

## 7. 비교 실험 설계
### 7-1. 실험군
1. `baseline_current`
   - 현재 방식
   - 예: `family_excluded_parameters=["stop_loss_rate"]`
2. `baseline_included`
   - 새 explicit 방식
   - 예: `family_included_parameters=["sell_profit_rate","max_splits_limit","max_inactivity_period"]`
3. `cluster_v1`
   - parameter-only clustering + medoid representative

### 7-2. 고정할 것
- 같은 standalone CSV bundle
- 같은 `mandatory/optional` window 구성
- 같은 `selection_metric`
- 같은 `shortlist_size`
- 같은 cutoff

### 7-3. 비교 지표
- `shortlist overlap`
- `family_pool_size`
- `promotion pass yield`
- `eventual champion recall`
- `CPU audit outcome`
- `holdout outcome`

## 8. 성공 기준
- `cluster_v1`가 baseline보다 shortlist 다양성을 명확히 높인다.
- promotion에서 `hard_gate_pass candidate` 수가 줄지 않거나, 의미 있게 늘어난다.
- CPU audit / holdout 경로에서 provenance 문제가 생기지 않는다.
- 결과 설명이 너무 어려워지지 않는다.

## 9. 실패 기준
- shortlist는 달라졌지만 promotion/holdout 결과 개선이 거의 없다.
- cluster feature/seed/k 설정에 따라 결과가 쉽게 흔들린다.
- source manifest만 보고는 왜 후보가 뽑혔는지 설명하기 어렵다.

## 10. 구현 순서
1. `contract` 문서에 `cluster_v1`를 optional compare mode로 잠근다.
2. `shortlist_derivation.py`에 `cluster_v1` experimental path를 추가한다.
3. `shortlist_source_manifest.json`에 clustering 메타데이터를 남긴다.
4. `same-cutoff 3-arm A/B`를 돌린다.
5. 결과가 이기면 그때 기본값 승격 여부를 다시 논의한다.

## 11. 지금 시점의 추천
- 당장은 `family_included_parameters`를 먼저 써보는 것이 더 단순하고 안전하다.
- `cluster_v1`는 그 다음 단계의 비교 실험으로 여는 것이 맞다.
