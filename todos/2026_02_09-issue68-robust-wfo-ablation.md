# Issue #68: Robust WFO / Ablation

> Type: `implementation`
> Status: `in_progress`
> Priority: `P2`
> Last updated: `2026-03-14`
> Related issues: `#68`, `#56`, `#67`, `#101`
> Gate status: `implementation in progress`

## Summary
- What:
  - 단일 최고점 파라미터가 아니라, 여러 장세와 여러 검증 구간에서도 다시 설명 가능한 `robust` 후보를 고르는 체계를 만든다.
- Why:
  - 지금 방식은 `calmar_ratio` 중심이라, 후보가 넓은 `plateau`에서 안정적인지 설명하기 어렵다.
- Current status:
  - import-safe 기반 정리는 끝났다.
  - `lane_manifest.json`, `holdout_manifest.json` helper와 JSON 저장 연결은 시작됐다.
  - 현재 WFO run은 결과 폴더에 manifest를 남기지만, 아직 `legacy_wfo/internal_provisional` 상태를 정직하게 기록하는 단계다.
  - 추가로 `holdout_backtest_executed`, `promotion_WFO_end < holdout_start` 같은 현재 상태 라벨도 더 정직하게 남기도록 강화 중이다.
  - CPU 단계는 `GPU-selected finalist -> CPU pass/fail audit` 쪽으로 1차 정리됐다.
  - `strict_only_governance` observation gate는 `lane_manifest.json`, `holdout_manifest.json`이 없으면 clean pass가 되지 않도록 막기 시작했다.
  - `strict_only_governance` observation gate는 이제 `final_candidate_manifest.json`도 함께 읽어서 champion hard gate, final candidate CPU audit, holdout execution 상태를 직접 이유(reason)로 반영할 수 있다.
  - `promotion_evaluation`은 `promotion_shortlist_path`를 받아 frozen shortlist 기반 single-anchor non-overlap anchored WFO를 실행할 수 있다.
  - `promotion_evaluation`은 이제 `decision_date(후보 고정 기준일)`가 없으면 실행 자체를 시작하지 않는다.
  - promotion lane은 CPU audit이 `pass`가 아니면 clean approval lane으로 읽지 않도록 reason을 남긴다.
  - `promotion_candidate_summary.csv`와 `final_candidate_manifest.json`이 추가되어, holdout 직전 `single champion + reserve provenance` 계약을 artifact로 남기기 시작했다.
  - `final_candidate_manifest.json`에는 이제 `freeze_contract_hash`, `promotion_shortlist_hash_verified`, `promotion_shortlist_modified_after_decision_date`, `canonical_holdout_contract_verified`가 함께 남아, 후보 고정과 holdout 경계가 실제 reject 규칙으로 연결되기 시작했다.
  - `holdout_auto_execute=true`일 때는 `final_candidate_manifest.json`의 champion만 CPU로 holdout을 실행하고, adequacy metric을 계산해 `holdout_manifest.json`에 기록할 수 있다.
  - 단, 이 자동 실행은 `holdout_start/end configured + champion_hard_gate_pass + final candidate CPU audit pass`일 때만 시도된다.
  - `research_start_date_robustness`는 `research_shortlist_path + research_anchor_start_dates`가 주어지면 frozen shortlist multi-anchor evaluation을 실행할 수 있다.
  - research lane은 `anchor_manifest.json`, `research_anchor_fold_metrics.csv`, `research_metric_distribution_summary.json`을 남기고 단일 합성 curve는 만들지 않는다.
  - `hard gate` 공식식과 `robust score` 공식식은 promotion selection contract v1으로 코드와 문서에 고정됐다.
  - holdout adequacy 기준은 이제 `holdout_adequacy_thresholds`와 `holdout_waiver_reason` 구조로 manifest에 연결된다.
  - holdout과 lane에는 이제 `approval_eligible(내부 승인 가능)`와 `external_claim_eligible(대외 설명 가능)`가 분리되어 기록된다.
    - waiver가 있는 approval-grade run은 내부적으로는 진행 가능할 수 있지만, `external_claim_eligible=false`로 남는다.
    - `internal_holdout_class(내부용 holdout 분류)`는 참고용 라벨이고, 대외 설명 가능 여부의 최종 판단은 반드시 `external_claim_eligible`를 본다.
  - promotion lane의 CPU 최종 판정 SSOT는 이제 `final candidate CPU audit(최종 후보 CPU 검산)`이다.
    - 예전 `cpu_certification(선택 단계 CPU 확인)`은 `selection_cpu_check_outcome`으로 따로 기록되는 선택 단계 품질 확인용 보조 정보다.
  - reserve 자동 승계는 이번 `#68` 범위에서 보류(defer, 이번 작업 범위에서는 보류)하고, reserve는 provenance(후보 기록) 용도로만 남긴다.
  - `promotion_ablation_summary.csv`는 현재 선택 로직을 다시 돌리는 파일이 아니라, “왜 이 champion이 뽑혔는지”를 설명하는 비교 리포트다.
  - 현재 설계 방향은 `promotion lane`과 `research lane`을 분리하는 것이다.
  - 추가 숙의 결과, 이 전략의 분할 진입 특성상 `1년 미만 holdout`은 최종 승인용으로 약하다는 쪽으로 기울었다.
  - 따라서 구현 목표는 `approval-grade holdout >= 24개월`을 기본값으로 두고, 현재 `2025-01-01 ~ 2025-11-30`는 `internal provisional holdout`으로 취급하는 것이다.
- Next action:
  - [WFO / OOS Lane 임시 합의안](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_12-wfo-oos-lane-provisional-review.md)을 기준으로 구현 규칙을 코드에 옮긴다.

## 1. 초심자용 한 페이지 설명
### 1-1. 이 문서를 한 줄로
- 이 문서는 `좋아 보이는 후보`를 `정말 버티는 후보`로 걸러내는 공식 구현 계획이다.

### 1-2. 지금 만들려는 흐름
1. 넓은 simulation으로 후보를 많이 찾는다.
2. `hard gate`로 먼저 탈락시킨다.
3. 살아남은 후보끼리만 `robust score`로 tie-break 한다.
4. `promotion lane`과 `research lane`은 다른 질문에 쓰이므로 결과를 따로 저장한다.
5. 마지막에는 별도로 남겨 둔 `final untouched holdout`으로 확인한다.

### 1-3. 자주 나오는 말을 쉽게 풀면
- `hard gate`
  - 최소한 이 정도는 버텨야 통과하는 1차 기준
- `robust score`
  - gate를 통과한 후보들끼리 비교할 때 쓰는 보조 점수
- `promotion lane`
  - 시간 전이 검증용 경로
- `research lane`
  - 시작 시점 민감도 관찰용 경로
- `holdout`
  - 마지막까지 따로 남겨 둔 최종 시험지
- `internal provisional holdout`
  - holdout처럼 따로 남겨 두었지만, 길이 또는 이력 때문에 `release-grade final proof`라고는 부르지 않는 구간
- `audit`
  - 다시 최적화가 아니라 CPU `SSOT` 기준 검산

## 2. 이번 구현에서 꼭 지켜야 할 큰 원칙
- `CPU=SSOT`
- `candidate_source_mode=tier`
- research 결과를 release-grade evidence처럼 보이게 만들지 않는다.
- `promotion lane`과 `research lane`의 역할을 섞지 않는다.
- `final untouched holdout` 날짜는 재사용하지 않는다.
- approval-grade holdout은 충분히 길어야 하고, 이 전략에서는 기본적으로 `24개월 이상`을 목표로 둔다.

## 3. 무엇을 구현할 문서인가
### 3-1. In scope
- `walk_forward_analyzer`의 `lane_mode` 분리
- `hard gate`와 `robust score` 역할 고정
- lane별 artifact 저장 방식 고정
- `manifest` 저장
- 최종 선택 근거와 gate 리포트 저장

### 3-2. Out of scope
- 체결 로직 변경
- Tier 데이터셋 자체 재설계
- 전략 규칙의 본질적인 의미 변경

## 4. 현재까지 잠긴 결정
### 4-1. lane 역할 분리
- `promotion_evaluation`
  - 질문:
    - `시간이 앞으로 가도 이 후보가 버티는가`
- `research_start_date_robustness`
  - 질문:
    - `시작 시점이 달라도 결과가 과하게 흔들리지 않는가`

### 4-2. research lane 규칙
- 권장 구조:
  - `multi-anchor Anchored WFO`
- 공식 research mode:
  - `frozen_shortlist_multi_anchor_eval`
- 핵심 규칙:
  - 각 fold는 동일 `initial_cash`로 시작
  - carry-over 금지
  - `single composite equity curve` 금지
  - 같은 `shortlist_hash`를 여러 anchor에서 반복 평가
- 제외:
  - `anchor_specific_refit_exploration`
  - 이유:
    - 공식 evidence lane이 아니라 별도 탐색 연구로 남겨야 하기 때문

### 4-3. promotion lane 규칙
- 권장 구조:
  - `single-anchor`, `non-overlap Anchored WFO`
- 핵심 규칙:
  - 더 엄격한 시간 전이 검증
  - CPU `audit`은 `pass/fail`
  - final holdout 전에는 여기서 shortlist를 다시 흔들지 않음
  - `promotion_shortlist_path` 없이 새 후보를 다시 찾지 않음

### 4-4. canonical holdout 경계
- current repository status:
  - `promotion_WFO_end = 2024-12-31`
  - `internal_provisional_holdout_start = 2025-01-01`
  - `internal_provisional_holdout_end = 2025-11-30`
- target policy:
  - `approval-grade final untouched holdout >= 24개월`
- 의미:
  - 현재 `2025` 구간은 최신으로 남겨 둔 구간이지만 너무 짧아서 `approval-grade final proof`로는 약하다.
  - 앞으로의 approval-grade holdout은 더 길게 잡아야 한다.
  - 다만 `2024-01-01 ~ 2025-12-31`를 지금 와서 untouched라고 다시 부르지는 않는다.

### 4-5. robust score의 역할
- `robust score`는 selector가 아니다.
- 순서는 아래처럼 고정한다.
  1. 먼저 `hard gate`를 통과했는지 본다.
  2. 통과한 후보들끼리만 `robust score`를 tie-break로 쓴다.

## 5. 구현 체크리스트
- [x] 노트북/비GPU 환경에서도 `walk_forward_analyzer` import 가능하도록 기반 정리
- [x] `hard gate` 공식식 고정
  - official formula:
    - `promotion_oos_is_calmar_ratio_median >= 0.60`
    - `promotion_fold_pass_rate >= 70%`
    - `promotion_oos_mdd_depth_p95 <= 25%`
  - per-fold gate:
    - `oos_calmar_ratio >= 0.0`
    - `oos_mdd_depth <= 25%`
- [x] `robust score` 공식식 고정
  - official formula:
    - `robust_score = (promotion_oos_calmar_mean - 0.50 * promotion_oos_calmar_std) * log1p(promotion_fold_count)`
  - 용도:
    - `hard gate` 통과 후보 안에서만 tie-break
- [x] `lane_mode` 분기 구현
  - `research_start_date_robustness`
  - `promotion_evaluation`
- [x] research lane guardrail 구현
  - carry-over 금지
  - `single composite equity curve` 금지
  - `frozen_shortlist_multi_anchor_eval`만 공식 mode로 허용
- [x] canonical holdout partition 구현
  - current provisional window:
    - `promotion_WFO_end = 2024-12-31`
    - `internal_provisional_holdout_start = 2025-01-01`
    - `internal_provisional_holdout_end = 2025-11-30`
  - target policy:
    - `approval-grade final untouched holdout >= 24개월`
    - shorter window는 `internal_provisional` 또는 waiver case로만 표기
- [x] shortlist freeze contract 구현
  - `research_data_cutoff <= 2024-12-31`
  - holdout 시작 후 후보 수정 금지
  - `promotion_shortlist_hash_verified`
  - `promotion_shortlist_modified_after_decision_date`
  - `freeze_contract_hash`
  - `canonical_holdout_contract_verified`
- [x] final candidate selection contract 구현
  - `promotion_candidate_fold_metrics.csv`
  - `promotion_candidate_summary.csv`
  - `final_candidate_manifest.json`
  - `single champion + reserve provenance`
  - holdout은 manifest에 봉인된 champion만 입력으로 받음
  - current v1:
    - `hard gate -> robust score tie-break -> single champion`
    - reserve는 holdout 비교 pack이 아니며, 이번 `#68`에서는 자동 승계를 구현하지 않고 provenance 용도로만 남김
    - `weighted super-score` selector는 아직 쓰지 않음
- [ ] 행동지표 feature 실험
  - `trade_count`
  - `closed_trade_count`
  - `avg_hold_days`
  - `distinct_entry_months`
  - `peak_slot_utilization`
  - `realized_split_depth`
  - `avg_invested_capital_ratio`
  - `cash_drag_ratio`
- [x] ablation 4축 비교 고정
  - `Legacy-Calmar`
  - `Robust-Score`
  - `Robust+Gate`
  - `Robust+Gate+Behavior`
- [x] 결과 저장 형식 고정
  - `fold_gate_report`
  - 최종 robust parameter CSV
  - `lane_manifest.json`
    - 현재 상태:
      - WFO 결과 폴더에 저장 시작
      - 아직 `legacy_wfo` 기준의 provisional evidence 기록
  - `holdout_manifest.json`
    - 현재 상태:
      - WFO 결과 폴더에 저장 시작
      - `holdout_auto_execute=true`일 때는 champion 기준 holdout 실행/adequacy 자동 기록 가능
      - holdout 상태는 `attempted / success / blocked`로 나눠 기록 시작
      - `holdout_adequacy_thresholds`가 있으면 approval-grade 강등 규칙까지 같이 반영
      - `holdout_waiver_reason`이 있으면 짧은 holdout/adequacy 미달을 `waived_reasons`로 남길 수 있음
  - `final_candidate_manifest.json`
    - 현재 상태:
      - promotion lane에서 champion/reserve 기록 artifact로 생성
      - `champion_params`, `reserve_candidates`, `final_candidate_hash`를 함께 남겨 holdout 입력 계약으로 사용 가능
      - `reserve_auto_succession_implemented=false`
      - `reserve_auto_succession_deferred=true`
  - `promotion_ablation_summary.csv`
    - 현재 상태:
      - `Legacy-Calmar`, `Robust-Score`, `Robust+Gate`, `Robust+Gate+Behavior` 4축 기준에서 어떤 후보가 선택되는지 요약
  - `promotion_explanation_report.json`
    - 현재 상태:
      - champion, reserve 정책, behavior evidence, ablation 4축 비교를 한 파일에서 설명용으로 요약
      - `executive_summary`, `runner_up_comparison`, `behavior_evidence.threshold_checks`까지 포함
  - `promotion_explanation_summary.md`
    - 현재 상태:
      - 사람(팀원/관리직)이 바로 읽을 수 있는 짧은 설명 요약
  - `anchor_manifest.json`
  - research lane의 `anchor/fold metric distribution summary`

## 6. 산출물은 왜 따로 저장해야 하나
### 6-1. Research lane 산출물
- 목적:
  - 시작 시점 민감도를 분포로 보여 주는 것
- 기본 산출물:
  - `anchor/fold metric distribution summary`
  - `lane_manifest.json`
  - `anchor_manifest.json`
- 만들면 안 되는 것:
  - `release-grade`처럼 읽히는 단일 합성 curve

### 6-2. Promotion lane 산출물
- 목적:
  - 시간 전이 검증 결과를 남기는 것
- 기본 산출물:
  - `fold_gate_report`
  - 최종 robust parameter CSV
  - `promotion_candidate_fold_metrics.csv`
  - `promotion_candidate_summary.csv`
  - `final_candidate_manifest.json`
  - `lane_manifest.json`
  - `holdout_manifest.json`

### 6-3. manifest는 왜 필요한가
- 실험 결과만 남기면 나중에 `어떤 규칙으로 만든 결과인지`를 증명하기 어렵다.
- 그래서 `manifest`로 아래를 함께 남긴다.
  - 어떤 lane이었는지
  - 어느 날짜까지 데이터를 봤는지
  - holdout을 침범하지 않았는지
  - 어떤 shortlist를 썼는지

## 7. Stop-the-line 규칙
- `parity mismatch > 0 => reject`
- `holdout date reuse detected => reject`
- `decision_date 이후 shortlist 변경 => reject`
- `approval-grade claim with holdout < 24 months and no adequacy waiver => reject`

쉽게 말하면:
- 계산이 안 맞거나
- 최종 시험지를 미리 봤거나
- freeze 뒤에 몰래 후보를 바꾸면
- 그 run은 승인 근거로 쓰지 않는다.
- 또 holdout이 너무 짧은데도 `release-grade final proof`처럼 부르면 안 된다.

## 8. Acceptance criteria
- robust mode ON/OFF가 같은 입력에서 재현 가능하다.
- legacy 대비 `OOS 안정성`이 좋아졌다는 근거가 남는다.
- 문제 발생 시 `legacy`로 즉시 rollback 가능하다.
- `#67 tier mode`와 `#56 parity`가 안정화된 뒤에도 같은 기준으로 판정 가능하다.
- research lane 결과가 `release-grade evidence`로 오해되지 않도록 artifact/label guardrail이 남는다.
- approval-grade run이라면 `promotion_WFO_end < approval_grade_holdout_start`가 문서와 manifest에서 동시에 증명된다.
- approval-grade run이라면 `holdout_length_days >= 730` 또는 명시적 waiver 사유가 남는다.
- 대외 설명 가능 run이라면 `external_claim_eligible=true`가 lane/holdout manifest 둘 다에서 확인된다.
- approval-grade run이라면 holdout adequacy 필드가 함께 남는다.
- holdout adequacy는 `미청산 비율`보다 `자본 배치 적정성` 중심으로 해석된다.
- `lane_manifest.json`에 아래 필드가 남는다.
  - `evidence_tier`
  - `approval_eligible`
  - `decision_date`
  - `shortlist_hash`
  - `engine_version_hash`
- `holdout_manifest.json`에 아래 필드가 남는다.
  - `holdout_start`
  - `holdout_end`
  - `wfo_end`
  - `holdout_date_reuse_forbidden`
  - `internal_holdout_class`
  - `holdout_length_days`
  - `trade_count`
  - `closed_trade_count`
  - `avg_hold_days`
  - `distinct_entry_months`
  - `avg_invested_capital_ratio`
  - `cash_drag_ratio`

## 9. 남아 있는 구현 리스크
- 문서와 현재 코드가 아직 다르다.
- 특히 아래가 구현에서 가장 먼저 막혀야 한다.
  - research lane에서 carry-over가 가능한 상태
  - research lane에서 composite curve가 생성될 수 있는 상태
  - CPU audit 결과가 downstream governance gate와 아직 연결되지 않은 상태
  - 짧은 holdout이 `approval-grade final proof`처럼 과장되어 읽히는 상태
- 따라서 코드를 먼저 guardrail 중심으로 나누고, 그다음 점수식과 리포트 템플릿을 고정하는 순서가 안전하다.

## 10. 참고 메모
- `#101`은 분포 기반 파라미터 선택 프레임 자체를 다룬다.
- `#68`은 현재 공식 경로 위에서 `robust score`와 verification layer를 강화하는 문서다.
- 초심자용 설계 메모:
  - [WFO / OOS Lane 임시 합의안](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_12-wfo-oos-lane-provisional-review.md)
- 운영용 단계별 안내:
  - [WFO Approval Workflow Runbook](/root/projects/Split_Investment_Strategy_Optimizer/docs/operations/2026-03-14-wfo-approval-runbook.md)
- 현재 세션에서 잠긴 핵심:
  - research lane 데이터는 `후보 발굴/필터링`에는 쓸 수 있지만 `final untouched OOS`라고 부를 수는 없다.
  - CPU certification의 목표는 `rerank`보다 `audit/pass-fail`에 가깝게 잠근다.
  - `Anchored WFO`는 `고정 출발점에서 시간 전이 검증`용이다.
  - `multi-anchor Anchored WFO`는 `시작 시점 민감도` 관찰용이다.
  - `approval-grade holdout`은 기본적으로 `24개월 이상`을 목표로 둔다.
  - 현재 `2025-01-01 ~ 2025-11-30`는 `internal provisional holdout`으로 취급한다.
  - holdout 적정성은 `미청산 비율` 하나보다 `자본 배치 적정성`과 `실제 회전/청산 커버리지`를 같이 보는 쪽이 맞다.
