# feat(strategy): 멀티팩터 랭킹 + Robust WFO/Ablation 실행안 (Issue #68)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/68`
- 작성일: 2026-02-09
- 목적: 단일 Peak 파라미터가 아닌 OOS 재현 가능한 Robust 파라미터 선택 체계로 전환

## 1. 배경
- 현재 WFO는 클러스터링 기반 강건 선택이 있으나, 점수가 Calmar 중심이라 변동성/열화 통제가 약함
- GPU 최적화 결과 정렬이 `calmar_ratio` 중심이라 plateau 후보가 밀릴 수 있음

## 2. 이번 이슈의 구현 범위
- `src/walk_forward_analyzer.py`
  - robust 점수 함수 분리 (`compute_robust_score`)
  - hard gate 도입 (`apply_robust_gates`)
  - feature flag (`robust_selection_enabled`) 기반 legacy/robust 분기
- `src/parameter_simulation_gpu.py`
  - robust 후처리에 필요한 결과 컬럼(행동지표 포함) 노출 보강
- `config/config.yaml`
  - robust 관련 threshold 설정 추가
- `results/` 산출물
  - universe mode 메타데이터(`weekly|hybrid_transition|tier`) 포함

## 3. 체크리스트
- [x] 노트북 개발 지원: `src/walk_forward_analyzer.py`가 GPU deps(`cupy`, `cudf`) 없이도 import 가능하도록 lazy import 적용
  - 브랜치: `feature/issue68-wfo-import-safe-no-gpu`
  - 커밋: `890aa30d5fe65ce8fb352b96106933898ab3ad65`
  - 테스트: `python -m unittest tests.test_issue68_wfo_import_side_effects`
- [ ] robust score 실험식 고정: `(mean - k*std) * log1p(cluster_size)`
- [ ] gate 고정: `median(OOS/IS) >= 0.60`, `fold_pass_rate >= 70%`, `OOS_MDD_p95 <= 25%`
- [ ] 행동지표 feature 실험: `trade_count`, `avg_hold_days`
- [ ] 민감도 분석: 선택 파라미터 ±10% perturbation 저하율(`<= 15%`) 측정
- [ ] Ablation 4축 비교: Legacy-Calmar / Robust-Score / Robust+Gate / Robust+Gate+Behavior
- [ ] Universe 모드 비교(최소): `hybrid_transition` vs `tier`
- [ ] 강건성 검증 프로토콜 고정:
  - 학습/최적화 기준 경로는 `deterministic baseline`으로 고정
  - 검증 경로는 `seeded_stress` + `jackknife_drop_topN` 별도 수행
  - `random-only` 선택 정책은 운영 기준으로 금지
- [ ] 집중도 리스크 지표 추가: `max_single_stock_contribution`, `HHI` (또는 동등 지표)
- [ ] 결과 산출물 저장: fold별 gate 리포트 + 최종 robust 파라미터 CSV

## 4. 브랜치 규칙 (A안 전환 연계)
- [ ] `main` 직접 수정 금지, 기능 브랜치에서 WFO 변경 수행
- [ ] 권장 브랜치: `feature/issue68-robust-wfo-a-universe`

## 5. 완료 기준
- robust 모드 ON/OFF가 동일 코드 경로에서 재현 가능
- legacy 대비 OOS 안정성 지표(분산/열화) 개선 근거 확보
- 문제 시 즉시 legacy 복귀 가능(rollback 절차 문서화)
- #67 tier 모드 전환 이후에도 gate 통과 재현 가능
- deterministic baseline + seeded_stress + jackknife 결과가 동일 승격 규칙으로 판정 가능

## 6. 제외 범위
- 전략 체결 로직(`strategy.py`, `execution.py`) 기능 변경
- Tier v2 데이터셋 신규 도입(#71 범위)
