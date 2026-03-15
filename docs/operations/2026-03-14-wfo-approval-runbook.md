# WFO(Walk-Forward Optimization, 과거로 공부하고 미래 구간으로 시험 보는 검증 방식) Approval Workflow(승인 절차) Runbook(운영 안내서)

> 작성일: 2026-03-14
> 상태: Draft
> 범위: Magic Split 전략의 `parameter simulation(후보를 넓게 찾는 단계) -> research WFO(연구용 검증) -> shortlist freeze(후보 고정) -> promotion WFO(승인 심사용 검증) -> CPU audit(CPU 기준 검산) -> holdout(끝까지 남겨 둔 최종 시험 구간)` 운영 순서
> 목적: 초심자도 `무엇을 먼저 하고`, `어디서 멈추고`, `어떤 말까지 할 수 있는지` 헷갈리지 않게 만드는 운영용 안내서

## 1. 한 줄 결론
- 이 전략은 `후보 찾기`와 `최종 승인`을 분리해서 봐야 한다.
- 즉, `simulation(후보를 넓게 찾는 단계)`으로 후보를 찾고, `research lane(연구용 검증 경로)`으로 흔들림을 보고, `promotion lane(승인 심사용 검증 경로)`으로 더 엄격하게 다시 보고, 마지막에 `holdout(끝까지 남겨 둔 최종 시험 구간)`으로 최종 확인한다.

## 2. 이 문서가 필요한 이유
- 설계 문서는 `왜 이렇게 하는가`를 설명한다.
- 하지만 실제 운영에서는 아래가 더 중요하다.
  - 지금 무엇을 돌려야 하나
  - 어떤 결과는 연구용인가
  - 어떤 결과부터 승인 근거가 되나
  - 어디서 `stop-the-line` 해야 하나
- 이 runbook(운영 안내서)은 그 질문에 답하는 문서다.

## 3. 먼저 큰 그림
### 3-1. 단계만 아주 간단히
1. `Parameter simulation(파라미터 시뮬레이션)`
   - 후보를 넓게 찾는다.
2. `Research lane(연구용 검증 경로)`
   - 시작 시점이 달라도 결과가 너무 흔들리지 않는지 본다.
3. `Shortlist freeze(후보 고정)`
   - 후보를 좁히고 고정한다.
4. `Promotion evaluation lane(승인 심사용 검증 경로)`
   - 더 엄격한 시간 전이 검증을 한다.
5. `CPU audit(CPU 기준 검산)`
   - GPU/선정 결과가 CPU 기준에서도 맞는지 검산한다.
6. `Holdout(끝까지 남겨 둔 최종 시험 구간)`
   - 마지막으로 따로 남겨 둔 구간에서 최종 확인한다.

### 3-2. 가장 쉬운 비유
- `simulation`은 문제집을 많이 푸는 단계다.
- `research lane(연구용 검증 경로)`은 여러 날짜에 본 모의고사다.
- `promotion lane(승인 심사용 검증 경로)`은 공식 모의평가다.
- `holdout(끝까지 남겨 둔 최종 시험 구간)`은 끝까지 안 풀고 남겨 둔 진짜 마지막 시험지다.

## 4. 현재 상태를 먼저 알고 시작하기
### 4-1. 이미 합의된 것
- `research lane(연구용 검증 경로)`과 `promotion lane(승인 심사용 검증 경로)`은 역할이 다르다.
- `research lane(연구용 검증 경로)`
  - 목적:
    - 시작 시점 민감도 관찰
  - 권장 구조:
    - `multi-anchor Anchored WFO(시작점을 여러 개 두고 반복 검증하는 방식)`
- `promotion lane(승인 심사용 검증 경로)`
  - 목적:
    - 시간 전이 검증
  - 권장 구조:
    - `single-anchor non-overlap Anchored WFO(출발점은 하나로 고정하고 시험 구간이 겹치지 않게 보는 방식)`
- `approval-grade holdout(승인용 최종 등급 holdout)`
  - 기본 목표:
    - `24개월 이상 untouched(끝까지 별도로 남겨 둔) 구간`

### 4-2. 아직 구현이 다 안 된 것
- 현재 `walk_forward_analyzer.py`는 lane 분리, final candidate(최종 후보) 봉인, holdout 자동 실행, holdout adequacy 강등까지 상당 부분 자동화했다.
- 하지만 아래는 아직 후속 구현 또는 최종 마감이 더 필요하다.
  - 행동지표(behavior metrics) / ablation(비교 실험) 리포트 고도화
  - approval-grade holdout(승인용 최종 등급 holdout)을 진짜 대외 설명 가능한 수준으로 닫는 작업
- 현재 반영된 것:
  - CPU 단계는 GPU가 고른 후보를 `pass/fail`로 검산하는 쪽으로 1차 정리됐다.
  - `promotion_evaluation(승인 심사용 실행 모드)`은 `promotion_shortlist_path`를 받아 frozen shortlist(미리 고정한 후보 목록) 기반 single-anchor non-overlap anchored WFO를 실행한다.
  - `research_start_date_robustness(시작 시점 민감도 연구 모드)`는 `research_shortlist_path + research_anchor_start_dates`가 주어지면 frozen shortlist multi-anchor evaluation을 실행할 수 있다.
  - `strict_only_governance(운영 승인 판단 규칙)` observation gate(관찰용 판정 단계)는 `lane_manifest.json`, `holdout_manifest.json`, `final_candidate_manifest.json`이 없으면 clean pass(문제 없음 판정)가 되지 않도록 막는다.
  - `final_candidate_manifest.json`에는 이제 `freeze_contract_hash`, `promotion_shortlist_hash_verified`, `promotion_shortlist_modified_after_decision_date`, `canonical_holdout_contract_verified`가 같이 남는다.
  - promotion lane은 이제 `decision_date(후보 고정 기준일)`가 없으면 실행 자체를 시작하지 않는다.
- 아직 남은 것:
  - 행동지표(behavior metrics) / ablation(비교 실험) 보고 형식 고도화
  - approval-grade holdout(승인용 최종 등급 holdout) 최종 마감
- 현재 guardrail:
  - promotion lane은 `promotion_mode=frozen_shortlist_single_anchor_eval`만 허용한다.
  - promotion lane은 `promotion_shortlist_path`가 없으면 실행되지 않는다.
  - promotion lane은 CPU audit이 `pass`가 아니면 clean approval lane으로 읽지 않는다.
  - promotion lane은 shortlist가 `decision_date(후보 고정 기준일)` 이후 수정됐거나 canonical holdout 계약과 어긋나면 holdout 자동 실행 전에 막힌다.
  - research lane은 carry-over를 쓰지 않고, 단일 합성 curve를 만들지 않는다.
  - research lane은 `research_mode=frozen_shortlist_multi_anchor_eval`만 허용한다.
  - research lane은 `research_shortlist_path`, `research_anchor_start_dates`가 없으면 실행되지 않는다.

### 4-3. 그래서 이 runbook의 성격
- 지금은 `완전 자동 운영 매뉴얼`이라기보다
- `사람이 실수하지 않도록 순서와 의미를 고정하는 운영 계약서`
에 가깝다.

## 5. 단계별 실행 순서
### 5-0. 자주 쓰는 명령어
- 작업 시작 전 상태 확인:
```bash
git status --short --branch
```
- WFO 관련 설정과 holdout 메타데이터(설정 정보) 확인:
```bash
rg -n "walk_forward_settings|lane_type|promotion_shortlist_path|research_shortlist_path|holdout_start|holdout_end|holdout_auto_execute" config/config.yaml
```
- GPU 파라미터 simulation 실행:
```bash
CONDA_NO_PLUGINS=true conda run -n rapids-env \
  python -m src.parameter_simulation_gpu
```
- WFO 실행:
```bash
CONDA_NO_PLUGINS=true conda run -n rapids-env \
  python -m src.walk_forward_analyzer
```
- 가장 최근 WFO 결과 폴더 확인:
```bash
ls -td results/wfo_run_* | head -n 1
```
- 최근 WFO run의 manifest(상태 기록 파일) 확인:
```bash
LATEST_DIR=$(ls -td results/wfo_run_* | head -n 1)
sed -n '1,220p' "$LATEST_DIR/lane_manifest.json"
sed -n '1,220p' "$LATEST_DIR/holdout_manifest.json"
sed -n '1,260p' "$LATEST_DIR/final_candidate_manifest.json"
```
- strict-only governance(운영 승인 판단 규칙)에 WFO manifest를 같이 넣어 보기:
```bash
LATEST_DIR=$(ls -td results/wfo_run_* | head -n 1)
CONDA_NO_PLUGINS=true conda run -n rapids-env \
  python -m src.strict_only_governance \
  --mode observation \
  --run-manifest-glob "results/run_*/run_manifest.json" \
  --lane-manifest-json "$LATEST_DIR/lane_manifest.json" \
  --holdout-manifest-json "$LATEST_DIR/holdout_manifest.json" \
  --final-candidate-manifest-json "$LATEST_DIR/final_candidate_manifest.json"
```
- WFO 관련 빠른 회귀 테스트:
```bash
CONDA_NO_PLUGINS=true conda run -n rapids-env \
  python -m unittest \
  tests.test_wfo_holdout_policy \
  tests.test_wfo_cpu_certification \
  tests.test_issue68_wfo_import_side_effects \
  tests.test_issue69_entrypoint_compat
```

### 5-1. Step 0. 준비
- 목적:
  - 데이터와 기본 설정이 승인 경로에 맞는지 먼저 확인한다.
- 확인할 것:
  - `strict_pit`
  - `candidate_source_mode=tier`
  - parity 관련 기존 가드가 깨져 있지 않은지
  - `walk_forward_settings.holdout_start`, `walk_forward_settings.holdout_end`를 실제로 넣었는지
- 지금 단계에서 하면 안 되는 것:
  - research용 설정을 promotion 경로에 섞기
  - 이미 오염된 구간을 holdout(끝까지 남겨 둔 최종 시험 구간)이라고 부르기

### 5-2. Step 1. Parameter simulation(파라미터 시뮬레이션)
- 목적:
  - 좋은 후보를 넓게 찾는다.
- 해석:
  - 여기서 중요한 것은 `무조건 1등 하나`가 아니라 `후보군(shortlist, 압축된 후보 목록)` 또는 `plateau(특정 한 점이 아니라 넓게 안정적인 구간)`다.
- 산출물:
  - GPU simulation 결과
  - 후보군
- 여기서 할 수 있는 말:
  - `이 구간에서 후보군이 이렇게 보인다`
- 여기서 하면 안 되는 말:
  - `이 결과만으로 승인 가능하다`

### 5-3. Step 2. Research lane(연구용 검증 경로)
- 목적:
  - 시작 시점이 바뀌어도 결과가 너무 흔들리지 않는지 본다.
- 권장 구조:
  - `multi-anchor Anchored WFO(시작점을 여러 개 두고 반복 검증하는 방식)`
- 중요한 규칙:
  - 각 fold는 같은 `initial_cash`로 시작
  - carry-over(이전 결과를 다음 실험으로 넘기는 방식) 금지
  - 단일 합성 equity curve(여러 결과를 평균 합성한 수익곡선) 금지
- 산출물:
  - `anchor별/fold별 metric 분포`
  - summary
- 여기서 할 수 있는 말:
  - `시작 시점 민감도가 상대적으로 낮았다`
- 여기서 하면 안 되는 말:
  - `release-grade final proof(대외적으로도 최종 승인 증거라고 부를 수 있는 수준)다`

### 5-4. Step 3. Shortlist freeze(후보 고정)
- 목적:
  - research 단계에서 후보를 좁힌 뒤 고정한다.
- 아주 쉽게 말하면:
  - 이제부터는 답안지를 더 고치지 않는 단계다.
- 남겨야 할 것:
  - 어떤 후보를 freeze 했는지
  - 언제 freeze 했는지
  - 어떤 데이터 cutoff(이 시점까지만 보고 판단했다는 경계)까지 보고 freeze 했는지
- 여기서 하면 안 되는 것:
  - freeze 이후 holdout을 보고 후보를 다시 수정하기

### 5-5. Step 4. Promotion evaluation lane(승인 심사용 검증 경로)
- 목적:
  - shortlist가 시간 전이 검증을 통과하는지 본다.
- 권장 구조:
  - `single-anchor non-overlap Anchored WFO(출발점은 하나로 고정하고 시험 구간이 겹치지 않게 보는 방식)`
- 해석:
  - 이 단계는 `재탐색`이 아니라 `승인 심사`다.
- 실행 전제:
  - `promotion_shortlist_path`로 freeze된 후보 CSV/JSON을 넘긴다.
  - promotion lane은 이 shortlist(압축된 후보 목록) 밖의 새 후보를 다시 찾지 않는다.
- 산출물:
  - fold별 결과
  - 선택/탈락 근거
  - `promotion_candidate_fold_metrics.csv`
  - `promotion_candidate_summary.csv`
  - `final_candidate_manifest.json`
- selection contract v1(후보 선정 계약 1차 버전):
  - 먼저 `hard gate(최소 통과 기준)`로 후보를 자른다.
  - 그다음 남은 후보 안에서 `robust score(통과 후보끼리 비교하는 보조 점수)`와 deterministic tie-break(항상 같은 순서가 나오게 하는 동점 정리 규칙)로 순위를 고정한다.
  - 최종적으로 `single champion(최종 1등 후보)` 1개만 holdout에 들어간다.
  - `reserve(예비 후보)`는 둘 수 있지만, 이것은 holdout에서 여러 후보를 비교하자는 뜻이 아니다.
  - 이번 `#68`에서는 reserve 자동 승계는 구현하지 않고, provenance(후보 기록) 용도로만 남긴다.
 - hard gate 공식식:
  - `promotion_oos_is_calmar_ratio_median >= 0.60`
  - `promotion_fold_pass_rate >= 70%`
  - `promotion_oos_mdd_depth_p95 <= 25%`
 - robust score 공식식:
  - `robust_score = (promotion_oos_calmar_mean - 0.50 * promotion_oos_calmar_std) * log1p(promotion_fold_count)`
  - 뜻:
    - 평균 OOS Calmar(수익 대비 낙폭 효율)가 높고
    - fold마다 흔들림(표준편차)이 작고
    - 같은 품질을 더 많은 fold에서 보여 줄수록
    - 점수가 올라간다.
- 현재 v1 tie-break:
  - `hard_gate_pass`
  - `robust_score`
  - `promotion_fold_pass_rate`
  - `promotion_oos_mdd_depth_worst`
  - `promotion_oos_cagr_median`
  - 마지막 동점이면 `candidate_signature`
- 현재 v1에서 일부러 하지 않는 것:
  - weighted super-score(여러 점수를 가중합해서 만든 단일 점수)로 한 숫자만 만들어 최종 선정하기
  - holdout에 후보 pack(여러 후보 묶음)을 넣고 나중에 더 좋은 것을 고르기
- 여기서 할 수 있는 말:
  - `고정 출발점 기준 시간 전이 검증을 통과했다`

### 5-6. Step 5. CPU audit(CPU 기준 검산)
- 목적:
  - 이미 고른 후보가 CPU 기준에서도 맞는지 검산한다.
- 중요한 점:
  - audit은 `다시 최적화`가 아니다.
  - `pass/fail(통과/실패)`에 가깝다.
  - `final_candidate_manifest.json`의 champion/reserve 순서를 바꾸는 용도로 쓰면 안 된다.
  - champion이 CPU audit에서 탈락하면 이번 `#68` 범위에서는 거기서 멈추고 원인을 조사한다.
  - reserve 자동 승계는 후속 운영 이슈로 보류(defer, 이번 작업 범위에서는 보류)한다.
- 여기서 하면 안 되는 것:
  - CPU 결과를 보고 사실상 새 후보를 다시 뽑기

### 5-7. Step 6. Holdout(끝까지 남겨 둔 최종 시험 구간)
- 목적:
  - 마지막으로 따로 남겨 둔 시험지에서 확인한다.
- 현재 정책:
- `approval-grade holdout(승인용 최종 등급 holdout)`의 기본 목표는 `24개월 이상 untouched(끝까지 따로 남겨 둔) 구간`
- 중요한 구분:
  - `approval_eligible(내부 승인 가능)`는 내부 절차상 다음 단계로 진행 가능한지 보는 값이다.
  - `external_claim_eligible(대외 설명 가능)`는 외부나 관리직에게 `이제 최종 승인 증거로 설명해도 된다`고 말할 수 있는지 보는 더 엄격한 값이다.
  - waiver(예외 승인 사유)가 들어간 holdout은 내부적으로는 진행할 수 있어도, `external_claim_eligible=false`로 남는다.
  - `holdout_class`는 내부 분류용 라벨이다. 최종 대외 설명 가능 여부는 반드시 `external_claim_eligible`로 판단해야 한다.
- 현재 구현 상태:
  - `holdout_start`, `holdout_end`를 config에 넣으면 `holdout_manifest.json(holdout 상태 기록 파일)`이 함께 저장된다.
  - `holdout_auto_execute=true(holdout 자동 실행 옵션)`를 같이 넣으면, promotion lane이 끝난 뒤 `final_candidate_manifest.json`의 champion만 CPU로 holdout을 실행한다.
  - 단, 이 자동 실행은 `holdout window(holdout 날짜 범위)가 설정되어 있고`, `champion이 hard gate(최소 통과 기준)를 통과했고`, `freeze contract(후보 고정 계약)`이 검증됐고, `final candidate CPU audit(최종 CPU 검산)`까지 pass인 경우에만 시도된다.
  - `holdout_auto_execute=false`면 holdout 정책 메타데이터(설정 정보)만 남기고, manifest에 `holdout_backtest_not_attempted`가 남는다.
  - `promotion_WFO_end < holdout_start` 여부도 manifest에 함께 남긴다.
  - `holdout_adequacy_thresholds(holdout 충분성 최소 기준)`가 설정되어 있으면, `trade_count`, `closed_trade_count`, `distinct_entry_months`, `avg_invested_capital_ratio`, `cash_drag_ratio` 같은 지표가 기준을 못 넘을 때 `approval_eligible=false`로 강등된다.
  - 다만 `holdout_waiver_reason(예외 승인 사유)`가 있으면, `holdout_too_short` 또는 adequacy threshold 미달은 `waived_reasons(예외 승인으로 넘긴 사유 목록)`으로 남기고 계속 진행할 수 있다.
  - `final_candidate_manifest.json`은 holdout 직전 champion/reserve 봉인용이고, `champion_params`와 `reserve_candidates`까지 포함해서 holdout 입력 계약을 self-contained(파일 하나만 봐도 필요한 정보가 다 들어 있는 상태)하게 남긴다.
  - 단, 이번 `#68`에서는 reserve는 자동 승계에 쓰지 않고 provenance(후보 기록) 용도로만 남긴다.
  - 이 파일에는 `promotion_shortlist_hash`, `freeze_contract_hash`, `canonical_holdout_*` 정보도 같이 남아서, 후보가 freeze 이후 바뀌지 않았는지와 holdout 경계가 정책과 맞는지를 나중에도 다시 확인할 수 있다.
  - `holdout_auto_execute=true`일 때는 `final candidate CPU audit -> holdout 실행 -> adequacy metric(holdout이 충분히 의미 있었는지 보는 지표) 계산`까지 이어지고, 결과가 manifest에 다시 반영된다.
  - holdout 결과는 이제 `attempted / success / blocked`를 나눠 기록한다. 그래서 “아예 안 돌렸다”, “돌렸는데 막혔다”, “성공적으로 끝났다”를 구분할 수 있다.
  - `holdout_manifest.json`에는 이제 `external_claim_eligible`와 `external_claim_reasons`도 같이 남는다. 그래서 `waiver가 있었는지`, `짧은 holdout이었는지`, `대외 설명용으로 닫혔는지`를 명확히 구분할 수 있다.
  - `lane_manifest.json`에도 `external_claim_eligible`가 같이 남아서, lane 전체가 내부 승인만 가능한 상태인지, 대외 설명까지 가능한 상태인지 구분할 수 있다.
  - promotion lane은 이제 `promotion_ablation_summary.csv`와 `promotion_explanation_report.json`도 함께 남긴다. 이 두 파일은 “왜 champion이 뽑혔는지”와 “behavior evidence(행동지표 근거)가 어땠는지”를 설명하는 보고용 산출물이다.
  - 중요한 점: `promotion_ablation_summary.csv`는 현재 `설명용 비교 리포트`에 가깝다. 아직 독립적인 새 선택기(selector, 후보를 다시 고르는 별도 로직)라고 읽으면 안 된다.
- 현재 저장소 상태:
- `2025-01-01 ~ 2025-11-30`는 `internal provisional holdout(임시 내부 검증용 holdout)`
- 아주 쉽게 말하면:
  - 지금 남겨 둔 최신 구간은 있긴 하지만, 너무 짧아서 최종 승인용으로는 약하다.
- 그래서 현재 가능한 표현:
  - `internal provisional check(임시 내부 검증)`
- 현재 하면 안 되는 말:
  - `2025-01-01 ~ 2025-11-30`는 이미 충분한 release-grade final proof(대외적으로도 최종 승인 증거라고 부를 수 있는 수준)다

## 6. Holdout은 어떻게 읽어야 하나
### 6-1. 왜 이 전략은 holdout이 길어야 하나
- 이 전략은 한 번에 풀인하지 않는다.
- 조금 사고, 더 떨어지면 추가매수하고, 시간이 걸린 뒤에야 빠져나올 수 있다.
- 그래서 너무 짧은 holdout은:
  - 아직 포트폴리오가 충분히 차기 전에 끝날 수 있고
  - 매수만 많이 하고 청산은 덜 본 채 끝날 수 있다

### 6-2. 그래서 무엇을 같이 보나
- 기간만 길다고 자동으로 충분한 것은 아니다.
- 아래도 같이 봐야 한다.
  - `trade_count`
  - `closed_trade_count`
  - `avg_hold_days`
  - `distinct_entry_months`
  - `peak_slot_utilization`
  - `realized_split_depth`
  - `avg_invested_capital_ratio`
  - `cash_drag_ratio`
- 운영 팁:
  - 이 숫자들은 `walk_forward_settings.holdout_adequacy_thresholds`로 최소 기준을 잠글 수 있다.
  - 예외 승인이라면 `holdout_waiver_reason`을 같이 남겨서, 왜 기준 미달을 이번에는 넘겼는지 기록해야 한다.

### 6-3. 해석의 중심
- 이 전략에서는
  - `안 판 주식이 많냐`
보다
  - `돈이 실제로 얼마나 적절히 배치되고 회전했느냐`
를 더 중요하게 본다.

## 7. 금지 사항
- research lane 결과를 `release-grade evidence`라고 부르기
- research lane(연구용 검증 경로) 결과를 `release-grade evidence(대외적으로도 최종 승인 증거라고 부를 수 있는 결과)`라고 부르기
- freeze 이후 후보를 슬쩍 수정하기
- holdout 날짜를 research나 promotion WFO에 다시 쓰기
- `final_candidate_manifest.json`을 만든 뒤 champion/reserve 순서를 손으로 다시 바꾸기
- parity/release-readiness에 쓴 구간을 untouched holdout이라고 부르기
- 짧은 holdout을 waiver 없이 `approval-grade final proof`처럼 말하기

## 8. Stop-the-line
- `parity mismatch > 0`
- `holdout date reuse detected`
- `decision_date 이후 shortlist 변경`
- `approval-grade claim(승인용 최종 등급 주장) with holdout < 24 months and no adequacy waiver(충분성 예외 승인 사유 없음)`
- `final_candidate_hash mismatch`
- `holdout after candidate reselection`

초심자 버전으로 말하면:
- 계산이 안 맞거나
- 최종 시험지를 미리 봤거나
- 답안지를 몰래 다시 고쳤으면
- 그 run은 승인 근거로 쓰지 않는다.

## 9. 현재 기준으로 할 수 있는 주장과 하면 안 되는 주장
### 9-1. 할 수 있는 주장
- `research lane(연구용 검증 경로)에서 시작 시점 민감도를 관찰했다`
- `promotion lane(승인 심사용 검증 경로)에서 시간 전이 검증을 수행했다`
- `현재 2025 구간은 internal provisional holdout(임시 내부 검증용 holdout)이다`

### 9-2. 하면 안 되는 주장
- `research lane 성과가 곧 최종 승인 성과다`
- `2024-01-01 ~ 2025-12-31는 지금 기준으로 untouched holdout(끝까지 따로 남겨 둔 최종 시험 구간)이다`
- `2025-01-01 ~ 2025-11-30는 이미 충분한 release-grade final proof(대외적으로도 최종 승인 증거라고 부를 수 있는 수준)다`

## 10. 현재 산출물 체크리스트
- simulation 결과
- shortlist
- selection audit
- `promotion_candidate_fold_metrics.csv`
- `promotion_candidate_summary.csv`
- `final_candidate_manifest.json`
- lane manifest(검증 경로 상태 기록 파일)
- holdout manifest(holdout 상태 기록 파일)
- `holdout_backtest_summary.json` (`holdout_auto_execute=true`일 때)
- `holdout_equity_curve_data.csv` (`holdout_auto_execute=true`일 때)
- research 분포 요약
- promotion fold 결과

## 11. 현재 저장소 문맥에서 기억할 것 3가지
1. `2025-01-01 ~ 2025-11-30`는 `internal provisional holdout(임시 내부 검증용 holdout)`이다.
2. `approval-grade holdout(승인용 최종 등급 holdout)`의 기본 목표는 `24개월 이상 untouched(끝까지 따로 남겨 둔) 구간`이다.
3. 이 전략에서는 `미청산 비율` 하나보다 `자본 배치 적정성`과 `실제 회전/청산 커버리지`가 더 중요하다.

## 12. 관련 문서
- [WFO / OOS Lane 임시 합의안](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_12-wfo-oos-lane-provisional-review.md)
- [Issue #68: Robust WFO / Ablation](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_02_09-issue68-robust-wfo-ablation.md)
- [Hybrid Release Gate Board](/root/projects/Split_Investment_Strategy_Optimizer/docs/operations/2026-03-06-hybrid-release-gate-board.md)
