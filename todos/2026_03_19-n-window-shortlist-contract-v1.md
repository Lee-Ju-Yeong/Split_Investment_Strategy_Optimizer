# N-Window Shortlist Contract v1

> Type: `research`
> Status: `proposed`
> Priority: `P1`
> Last updated: `2026-03-19`
> Related issues: `#68`, `#98`
> Gate status: `discovery-only allowed; approval workflow unchanged`

## Summary
### What
- 이 문서는 `standalone simulation 결과 CSV 여러 개`를 묶어 `frozen shortlist(고정된 후보 목록)`를 만드는 `N-window shortlist derivation(여러 기간을 함께 보는 후보 추출)`의 공식 경계와 v1 계약을 기록한다.

### Why
- 지금은 dual-window만 보고 있지만, 앞으로 `3개 이상 window`나 다른 조합이 추가될 수 있다.
- 이때 스크립트가 `유리한 window 조합을 스스로 찾는 selector(선정기)`가 되면 approval chain(승인 경로)의 의미가 무너진다.
- 그래서 지금 단계에서 `무엇까지 일반화할지`와 `무엇은 금지할지`를 먼저 잠가야 한다.

### Current status
- 현재 방향은 `execute`다.
- 단, 이 문서가 여는 범위는 `discovery-only shortlist source(연구용 후보 압축 입력)`까지다.
- `research WFO`, `promotion WFO`, `CPU audit`, `holdout`의 공식 approval workflow는 바꾸지 않는다.

### Next action
- `window_bundle_manifest.json` / `shortlist_source_manifest.json` 스키마를 고정하고, `src/analysis/shortlist_derivation.py` 같은 별도 오케스트레이터를 추가한다.

## 1. 한 페이지 결론
- `N-window scoring/gating`은 일반화해도 된다.
- 하지만 `window search(어떤 창 조합이 유리한지 자동으로 찾는 기능)`는 금지한다.
- 입력 window는 반드시 `window_bundle_manifest.json`으로 명시적으로 고정한다.
- 최종 shortlist는 `cluster centroid(가상의 평균 파라미터)`가 아니라 `실제로 CSV에 존재하는 row(actual row)`만 freeze한다.
- `mandatory windows는 all-pass`, `optional windows는 small fail budget`을 허용한다.
- `standalone discovery artifact`는 끝까지 `research-only`로 취급하며, 그 자체를 approval evidence로 쓰면 안 된다.

## 2. 이 계약이 여는 것과 닫는 것
### 2-1. 허용하는 것
- 여러 standalone CSV를 함께 읽어 `family dedupe(비슷한 후보 묶기)`를 수행하는 것
- `hard gate -> robust ranking -> plateau diversified representatives` 순서로 후보를 줄이는 것
- `window_bundle_manifest.json`, `shortlist_source_manifest.json`, `shortlist_candidates.csv/json`를 남기는 것
- `same-cutoff A/B` 검증을 위해 `1-window vs N-window derivation`을 비교하는 것

### 2-2. 금지하는 것
- 스크립트가 `window 조합`, `window weight`, `shortlist_size`, `selection_metric`, `aggregation rule`을 자동 탐색하는 것
- `promotion OOS`, `holdout`, `explanation report`, `ablation report`, `composite curve`를 shortlist selector 입력으로 쓰는 것
- centroid를 실제 freeze candidate로 승격하는 것
- promotion lane 안에서 shortlist 밖의 새 후보를 다시 찾는 것

## 3. 공식 경계
### 3-1. Workflow boundary
- 공식 경로는 그대로 유지한다.
  1. `parameter simulation`
  2. `N-window derivation(선택 사항, discovery-only)`
  3. `shortlist freeze`
  4. `research WFO`
  5. `promotion WFO`
  6. `CPU audit`
  7. `holdout`
- 즉, `N-window derivation`은 `upstream discovery source`일 뿐이고, approval lane 자체가 아니다.

### 3-2. Approval-compatible upper bound
- v1에서 `approval-compatible` 모드는 보수적으로 둔다.
  - 권장: `N <= 3`
  - `N > 3`는 일단 `discovery-only`로만 허용한다.
- 이유:
  - window 수가 너무 많아지면 `research WFO selector`와 경계가 흐려지고, false negative와 selection bias를 함께 관리하기가 더 어려워진다.

## 4. Hard Gate / Score 원칙
### 4-1. Window roles
- `mandatory`
  - 반드시 통과해야 하는 핵심 창
- `optional`
  - score / tie-break를 보조하는 창
  - 단, `mandatory fail`을 구제하면 안 된다.

### 4-2. Hard Gate 원칙
- 기본 원칙:
  - `mandatory windows = all-pass`
  - `optional windows = fail budget 허용`
- 해석:
  - `intersection_all`을 그대로 쓰면 false negative가 너무 커질 수 있다.
  - 반대로 모든 창을 느슨하게 평균 내면 특정 regime blind 후보가 살아남는다.

### 4-3. Score 원칙
- `mean = 평균 품질`
- `q25 = 하방 안정성`
- `min = catastrophic veto(재난성 경고)`
- `std = 창 사이 흔들림 페널티`
- v1 기본 방향:
  - `hard gate`를 먼저 적용하고
  - 통과 후보만 `robust score`로 정렬한다.

## 5. Family / Representative 규칙
### 5-1. Family dedupe
- raw top-k를 그대로 쓰지 않는다.
- `stop_loss_rate`처럼 식별력이 약한 축은 family key에서 제외할 수 있다.
- 대신 final freeze 시점에는 대표 row를 1개만 복원한다.

### 5-2. Representative 선택
- `cluster centroid`는 분석용으로만 허용한다.
- 최종 freeze는 아래만 허용한다.
  - `actual row-based medoid`
  - `nearest valid member`
- 이유:
  - centroid는 실제 시뮬레이션 그리드에 없을 수 있고
  - approval chain의 hash / rerun / audit 재현성을 깨뜨릴 수 있다.

## 6. Provenance contract
### 6-1. 필수 산출물
- `window_bundle_manifest.json`
- `shortlist_source_manifest.json`
- `shortlist_candidates.csv`
- `shortlist_candidates.json`
- `shortlist_derivation_report.json`
- `shortlist_derivation_summary.md`

### 6-2. `window_bundle_manifest.json` 핵심 필드
- `bundle_id`
- `source_mode`
- `decision_date`
- `research_data_cutoff`
- `promotion_data_cutoff`
- `holdout_start`
- `holdout_end`
- `window_policy_id`
- `window_overlap_policy`
- `windows[]`
  - `window_id`
  - `csv_path`
  - `expected_hash`
  - `config_path`
  - `window_role`
  - `weight`

### 6-3. `shortlist_source_manifest.json` 핵심 필드
- `manifest_version`
- `artifact_role`
- `evidence_tier`
- `approval_evidence_allowed`
- `source_window_count`
- `source_windows`
- `search_space_hash`
- `runtime_budget`
- `selection_metric`
- `aggregation_rule_version`
- `tie_break_rule_version`
- `shortlist_hash`
- `shortlist_size`
- `candidate_ranking_hash`
- `bundle_manifest_hash`
- `freeze_contract_hash`
- `git_sha`
- `engine_version_hash`
- `generated_at_utc`

### 6-4. 최소 예시
```json
{
  "bundle_id": "issue98_dual_window_v1",
  "source_mode": "n_window_consensus_mining",
  "decision_date": "2026-03-19",
  "research_data_cutoff": "2024-12-31",
  "promotion_data_cutoff": "2024-12-31",
  "holdout_start": "2025-01-01",
  "holdout_end": "2025-12-31",
  "governance_gates": {
    "max_rank_percentile": 15.0,
    "optional_fail_budget": 0,
    "minimum_criteria": {
      "calmar_ratio_min": 0.30
    }
  },
  "selection_contract": {
    "selection_metric": "calmar_ratio",
    "shortlist_size": 6,
    "family_excluded_parameters": ["stop_loss_rate"]
  },
  "windows": [
    {
      "window_id": "window_2015_2019",
      "csv_path": "results/standalone_simulation_results_20260316_224937.csv",
      "expected_hash": "<sha256>",
      "window_role": "mandatory",
      "weight": 1.0
    },
    {
      "window_id": "window_2018_2022",
      "csv_path": "results/standalone_simulation_results_20260319_135008.csv",
      "expected_hash": "<sha256>",
      "window_role": "mandatory",
      "weight": 1.0
    }
  ]
}
```

## 7. 구현 방향
### 7-1. 파일 경계
- 기존 [issue98_combo_mining_report.py](/root/projects/Split_Investment_Strategy_Optimizer/src/analysis/issue98_combo_mining_report.py)는 `single-window reporter`로 유지한다.
- 신규 오케스트레이터 예시:
  - `src/analysis/shortlist_derivation.py`

### 7-2. 권장 호출 형태
```bash
python -m src.analysis.issue98_combo_mining_report \
  --csv-path results/standalone_simulation_results_*.csv \
  --report-dir results/windows/window_01

python -m src.analysis.shortlist_derivation derive-shortlist \
  --bundle-manifest results/shortlist_derivation/window_bundle_manifest.json \
  --approval-compatible \
  --out-dir results/shortlist_derivation/run_001
```

### 7-3. CLI 원칙
- `--left-csv`, `--right-csv` 같은 dual 전용 플래그는 만들지 않는다.
- repeatable input 또는 manifest input만 허용한다.
- 조합 자동탐색 옵션은 v1에서 열지 않는다.

## 8. 검증 계약
- `same-cutoff A/B`로 기존 derivation과 새 derivation을 비교한다.
- 최소 비교 항목:
  - `shortlist overlap`
  - `false-negative rate`
  - `promotion pass yield`
  - `CPU audit outcome`
  - `holdout outcome`
- `approval-compatible` 모드는 아래가 깨지면 reject다.
  - source manifest 누락
  - hash mismatch
  - promotion/holdout 날짜 overlap
  - post-decision regeneration

## 9. 지금 바로 할 일
1. 이 문서를 `#68` 관련 정책 문서에서 공식 링크로 연결한다.
2. `window_bundle_manifest.json` / `shortlist_source_manifest.json` 스키마를 실제 JSON 예시로 추가한다.
3. `src/analysis/shortlist_derivation.py` 구현을 시작한다.
4. `strict_only_governance`에 provenance 검사 규칙을 추가한다.

## 10. 관련 문서
- [WFO shortlist derivation review](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_16-wfo-shortlist-derivation-review.md)
- [WFO Approval Workflow Runbook](/root/projects/Split_Investment_Strategy_Optimizer/docs/operations/2026-03-14-wfo-approval-runbook.md)
- [Robust WFO / Ablation](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_02_09-issue68-robust-wfo-ablation.md)
