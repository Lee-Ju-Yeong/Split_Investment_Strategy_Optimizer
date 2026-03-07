# Issue #68: Robust WFO / Ablation

> Type: `implementation`
> Status: `planned`
> Priority: `P2`
> Last updated: 2026-03-07
> Related issues: `#68`, `#56`, `#67`, `#101`
> Gate status: `not started`

## 1. Summary
- What: 단일 최고점 파라미터가 아니라 OOS에서 다시 설명 가능한 robust 파라미터 선택 체계를 만듭니다.
- Why: 현재 WFO는 `calmar_ratio` 중심이라 plateau 후보와 열화 리스크를 충분히 설명하지 못합니다.
- Current status: import-safe 기반 작업은 끝났고, robust score와 hard gate 공식안은 아직 열지 않았습니다.
- Next action: robust score 식, hard gate, ablation 매트릭스를 공식안으로 고정합니다.

## 2. Scope And Constraints
- In scope:
  - `walk_forward_analyzer`의 robust score / gate / feature flag
  - `parameter_simulation_gpu` 결과 컬럼 보강
  - fold별 gate 리포트와 최종 선택 근거 저장
- Out of scope:
  - 체결 로직 변경
  - Tier v2 데이터셋 자체 설계 변경
- Constraints:
  - `CPU=SSOT`
  - `candidate_source_mode=tier` 전제 유지
  - official release 전까지 `deterministic baseline` + `seeded_stress` + `jackknife_drop_topN` 공통 검증 필요

## 3. Current Plan
- [x] 노트북/비GPU 환경에서도 `walk_forward_analyzer` import 가능하도록 기반 정리
- [ ] robust score 식 고정
  - 초안: `(mean - k*std) * log1p(cluster_size)`
- [ ] hard gate 고정
  - `median(OOS/IS) >= 0.60`
  - `fold_pass_rate >= 70%`
  - `OOS_MDD_p95 <= 25%`
- [ ] 행동지표 feature 실험
  - `trade_count`
  - `avg_hold_days`
- [ ] ablation 4축 비교 고정
  - `Legacy-Calmar`
  - `Robust-Score`
  - `Robust+Gate`
  - `Robust+Gate+Behavior`
- [ ] 결과 저장 형식 고정
  - fold별 gate report
  - 최종 robust parameter CSV

## 4. Evidence And Gate
- Existing evidence:
  - lazy import groundwork 완료
  - 테스트: `python -m unittest tests.test_issue68_wfo_import_side_effects`
- Acceptance criteria:
  - robust mode ON/OFF가 같은 입력에서 재현 가능
  - legacy 대비 OOS 안정성 개선 근거가 남음
  - 문제 발생 시 `legacy`로 즉시 rollback 가능
  - `#67 tier mode`와 `#56 parity`가 안정화된 뒤에도 같은 기준으로 판정 가능

## 5. Notes
- `#101`은 분포 기반 파라미터 선택 프레임 자체를 다룹니다.
- `#68`은 현재 공식 경로 위에서 robust score와 verification layer를 강화하는 문서입니다.
