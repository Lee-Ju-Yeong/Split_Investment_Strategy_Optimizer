# WFO Approval Workflow Runbook

> 작성일: 2026-03-14
> 상태: Draft
> 범위: Magic Split 전략의 `parameter simulation -> research WFO -> shortlist freeze -> promotion WFO -> CPU audit -> holdout` 운영 순서
> 목적: 초심자도 `무엇을 먼저 하고`, `어디서 멈추고`, `어떤 말까지 할 수 있는지` 헷갈리지 않게 만드는 운영용 안내서

## 1. 한 줄 결론
- 이 전략은 `후보 찾기`와 `최종 승인`을 분리해서 봐야 한다.
- 즉, `simulation`으로 후보를 찾고, `research lane`으로 흔들림을 보고, `promotion lane`으로 더 엄격하게 다시 보고, 마지막에 `holdout`으로 최종 확인한다.

## 2. 이 문서가 필요한 이유
- 설계 문서는 `왜 이렇게 하는가`를 설명한다.
- 하지만 실제 운영에서는 아래가 더 중요하다.
  - 지금 무엇을 돌려야 하나
  - 어떤 결과는 연구용인가
  - 어떤 결과부터 승인 근거가 되나
  - 어디서 `stop-the-line` 해야 하나
- 이 runbook은 그 질문에 답하는 문서다.

## 3. 먼저 큰 그림
### 3-1. 단계만 아주 간단히
1. `Parameter simulation`
   - 후보를 넓게 찾는다.
2. `Research lane`
   - 시작 시점이 달라도 결과가 너무 흔들리지 않는지 본다.
3. `Shortlist freeze`
   - 후보를 좁히고 고정한다.
4. `Promotion evaluation lane`
   - 더 엄격한 시간 전이 검증을 한다.
5. `CPU audit`
   - GPU/선정 결과가 CPU 기준에서도 맞는지 검산한다.
6. `Holdout`
   - 마지막으로 따로 남겨 둔 구간에서 최종 확인한다.

### 3-2. 가장 쉬운 비유
- `simulation`은 문제집을 많이 푸는 단계다.
- `research lane`은 여러 날짜에 본 모의고사다.
- `promotion lane`은 공식 모의평가다.
- `holdout`은 끝까지 안 풀고 남겨 둔 진짜 마지막 시험지다.

## 4. 현재 상태를 먼저 알고 시작하기
### 4-1. 이미 합의된 것
- `research lane`과 `promotion lane`은 역할이 다르다.
- `research lane`
  - 목적:
    - 시작 시점 민감도 관찰
  - 권장 구조:
    - `multi-anchor Anchored WFO`
- `promotion lane`
  - 목적:
    - 시간 전이 검증
  - 권장 구조:
    - `single-anchor non-overlap Anchored WFO`
- `approval-grade holdout`
  - 기본 목표:
    - `24개월 이상 untouched 구간`

### 4-2. 아직 구현이 다 안 된 것
- 현재 `walk_forward_analyzer.py`는 이 새 구조를 100% 자동으로 강제하지 않는다.
- 특히 아래는 아직 후속 구현이 더 필요하다.
  - research lane의 carry-over 차단
  - research lane의 composite curve 차단
  - shortlist freeze와 manifest 자동 연결
  - holdout adequacy 지표 자동 기록
- 현재 반영된 것:
  - CPU 단계는 GPU가 고른 후보를 `pass/fail`로 검산하는 쪽으로 1차 정리됐다.
  - `promotion_evaluation`은 `promotion_shortlist_path`를 받아 frozen shortlist 기반 single-anchor non-overlap anchored WFO를 실행한다.
  - `research_start_date_robustness`는 `research_shortlist_path + research_anchor_start_dates`가 주어지면 frozen shortlist multi-anchor evaluation을 실행할 수 있다.
  - `strict_only_governance` observation gate는 `lane_manifest.json`, `holdout_manifest.json`이 없으면 clean pass가 되지 않도록 막는다.
- 아직 남은 것:
  - WFO 실행 경로가 governance gate 입력을 자동으로 남기도록 end-to-end 연결
- 현재 guardrail:
  - promotion lane은 `promotion_mode=frozen_shortlist_single_anchor_eval`만 허용한다.
  - promotion lane은 `promotion_shortlist_path`가 없으면 실행되지 않는다.
  - promotion lane은 CPU audit이 `pass`가 아니면 clean approval lane으로 읽지 않는다.
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
- WFO 관련 설정과 holdout 메타데이터 확인:
```bash
rg -n "walk_forward_settings|lane_type|promotion_shortlist_path|research_shortlist_path|holdout_start|holdout_end" config/config.yaml
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
- 최근 WFO run의 manifest 확인:
```bash
LATEST_DIR=$(ls -td results/wfo_run_* | head -n 1)
sed -n '1,220p' "$LATEST_DIR/lane_manifest.json"
sed -n '1,220p' "$LATEST_DIR/holdout_manifest.json"
```
- strict-only governance에 WFO manifest를 같이 넣어 보기:
```bash
LATEST_DIR=$(ls -td results/wfo_run_* | head -n 1)
CONDA_NO_PLUGINS=true conda run -n rapids-env \
  python -m src.strict_only_governance \
  --mode observation \
  --run-manifest-glob "results/run_*/run_manifest.json" \
  --lane-manifest-json "$LATEST_DIR/lane_manifest.json" \
  --holdout-manifest-json "$LATEST_DIR/holdout_manifest.json"
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
  - 이미 오염된 구간을 holdout이라고 부르기

### 5-2. Step 1. Parameter simulation
- 목적:
  - 좋은 후보를 넓게 찾는다.
- 해석:
  - 여기서 중요한 것은 `무조건 1등 하나`가 아니라 `후보군(shortlist)` 또는 `plateau`다.
- 산출물:
  - GPU simulation 결과
  - 후보군
- 여기서 할 수 있는 말:
  - `이 구간에서 후보군이 이렇게 보인다`
- 여기서 하면 안 되는 말:
  - `이 결과만으로 승인 가능하다`

### 5-3. Step 2. Research lane
- 목적:
  - 시작 시점이 바뀌어도 결과가 너무 흔들리지 않는지 본다.
- 권장 구조:
  - `multi-anchor Anchored WFO`
- 중요한 규칙:
  - 각 fold는 같은 `initial_cash`로 시작
  - carry-over 금지
  - 단일 합성 equity curve 금지
- 산출물:
  - `anchor별/fold별 metric 분포`
  - summary
- 여기서 할 수 있는 말:
  - `시작 시점 민감도가 상대적으로 낮았다`
- 여기서 하면 안 되는 말:
  - `release-grade final proof다`

### 5-4. Step 3. Shortlist freeze
- 목적:
  - research 단계에서 후보를 좁힌 뒤 고정한다.
- 아주 쉽게 말하면:
  - 이제부터는 답안지를 더 고치지 않는 단계다.
- 남겨야 할 것:
  - 어떤 후보를 freeze 했는지
  - 언제 freeze 했는지
  - 어떤 데이터 cutoff까지 보고 freeze 했는지
- 여기서 하면 안 되는 것:
  - freeze 이후 holdout을 보고 후보를 다시 수정하기

### 5-5. Step 4. Promotion evaluation lane
- 목적:
  - shortlist가 시간 전이 검증을 통과하는지 본다.
- 권장 구조:
  - `single-anchor non-overlap Anchored WFO`
- 해석:
  - 이 단계는 `재탐색`이 아니라 `승인 심사`다.
- 실행 전제:
  - `promotion_shortlist_path`로 freeze된 후보 CSV/JSON을 넘긴다.
  - promotion lane은 이 shortlist 밖의 새 후보를 다시 찾지 않는다.
- 산출물:
  - fold별 결과
  - 선택/탈락 근거
  - `promotion_candidate_fold_metrics.csv`
  - `promotion_candidate_summary.csv`
  - `final_candidate_manifest.json`
- 여기서 할 수 있는 말:
  - `고정 출발점 기준 시간 전이 검증을 통과했다`

### 5-6. Step 5. CPU audit
- 목적:
  - 이미 고른 후보가 CPU 기준에서도 맞는지 검산한다.
- 중요한 점:
  - audit은 `다시 최적화`가 아니다.
  - `pass/fail`에 가깝다.
  - `final_candidate_manifest.json`의 champion/reserve 순서를 바꾸는 용도로 쓰면 안 된다.
- 여기서 하면 안 되는 것:
  - CPU 결과를 보고 사실상 새 후보를 다시 뽑기

### 5-7. Step 6. Holdout
- 목적:
  - 마지막으로 따로 남겨 둔 시험지에서 확인한다.
- 현재 정책:
- `approval-grade holdout`의 기본 목표는 `24개월 이상 untouched 구간`
- 현재 구현 상태:
  - `holdout_start`, `holdout_end`를 config에 넣으면 `holdout_manifest.json`이 함께 저장된다.
  - 아직 holdout 백테스트 자체가 자동 실행되는 것은 아니므로, manifest에 `holdout_backtest_not_executed`가 남고 `approval_eligible=false`로 유지된다.
  - `promotion_WFO_end < holdout_start` 여부도 manifest에 함께 남긴다.
  - `final_candidate_manifest.json`은 holdout 직전 champion/reserve 봉인용이고, 아직 final candidate CPU audit/holdout 실행이 안 끝났으면 `holdout_ready=false`로 남는다.
- 현재 저장소 상태:
  - `2025-01-01 ~ 2025-11-30`는 `internal provisional holdout`
- 아주 쉽게 말하면:
  - 지금 남겨 둔 최신 구간은 있긴 하지만, 너무 짧아서 최종 승인용으로는 약하다.
- 그래서 현재 가능한 표현:
  - `internal provisional check`
- 현재 하면 안 되는 말:
  - `2025-01-01 ~ 2025-11-30`는 이미 충분한 release-grade final proof다

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

### 6-3. 해석의 중심
- 이 전략에서는
  - `안 판 주식이 많냐`
보다
  - `돈이 실제로 얼마나 적절히 배치되고 회전했느냐`
를 더 중요하게 본다.

## 7. 금지 사항
- research lane 결과를 `release-grade evidence`라고 부르기
- freeze 이후 후보를 슬쩍 수정하기
- holdout 날짜를 research나 promotion WFO에 다시 쓰기
- `final_candidate_manifest.json`을 만든 뒤 champion/reserve 순서를 손으로 다시 바꾸기
- parity/release-readiness에 쓴 구간을 untouched holdout이라고 부르기
- 짧은 holdout을 waiver 없이 `approval-grade final proof`처럼 말하기

## 8. Stop-the-line
- `parity mismatch > 0`
- `holdout date reuse detected`
- `decision_date 이후 shortlist 변경`
- `approval-grade claim with holdout < 24 months and no adequacy waiver`
- `final_candidate_hash mismatch`
- `holdout after candidate reselection`

초심자 버전으로 말하면:
- 계산이 안 맞거나
- 최종 시험지를 미리 봤거나
- 답안지를 몰래 다시 고쳤으면
- 그 run은 승인 근거로 쓰지 않는다.

## 9. 현재 기준으로 할 수 있는 주장과 하면 안 되는 주장
### 9-1. 할 수 있는 주장
- `research lane에서 시작 시점 민감도를 관찰했다`
- `promotion lane에서 시간 전이 검증을 수행했다`
- `현재 2025 구간은 internal provisional holdout이다`

### 9-2. 하면 안 되는 주장
- `research lane 성과가 곧 최종 승인 성과다`
- `2024-01-01 ~ 2025-12-31는 지금 기준으로 untouched holdout이다`
- `2025-01-01 ~ 2025-11-30는 이미 충분한 release-grade final proof다`

## 10. 현재 산출물 체크리스트
- simulation 결과
- shortlist
- selection audit
- lane manifest
- holdout manifest
- final candidate manifest
- research 분포 요약
- promotion fold 결과

## 11. 현재 저장소 문맥에서 기억할 것 3가지
1. `2025-01-01 ~ 2025-11-30`는 `internal provisional holdout`이다.
2. `approval-grade holdout`의 기본 목표는 `24개월 이상 untouched 구간`이다.
3. 이 전략에서는 `미청산 비율` 하나보다 `자본 배치 적정성`과 `실제 회전/청산 커버리지`가 더 중요하다.

## 12. 관련 문서
- [WFO / OOS Lane 임시 합의안](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_12-wfo-oos-lane-provisional-review.md)
- [Issue #68: Robust WFO / Ablation](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_02_09-issue68-robust-wfo-ablation.md)
- [Hybrid Release Gate Board](/root/projects/Split_Investment_Strategy_Optimizer/docs/operations/2026-03-06-hybrid-release-gate-board.md)
