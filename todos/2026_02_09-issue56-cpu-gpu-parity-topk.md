# test(parity): CPU/GPU 정합성 하네스 강화 - top-k 배치 검증 (Issue #56)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/56`
- 작성일: 2026-02-09
- 목적: 단일 파라미터 검증을 넘어, 최적화 상위 후보(top-k) 전수 parity 검증으로 회귀 리스크 차단

## 0. 진행 현황 (2026-02-09)
- 선행 완료(`#67` Phase A):
  - GPU 후보 기준일 `signal_date(T-1)` 정렬
  - ATR 조회 `as-of(<=)` 정렬
  - Tier preload를 `start 이전 latest 1행 + 기간 데이터`로 보강(30일 가정 제거)
- 현재 상태:
  - parity 하네스 본체(`top-k`, scenario pack, mismatch report)는 아직 미구현
  - 즉, `#56`은 선행 블로커 해소 완료, 본작업은 다음 단계

## 0-1. 진행 현황 업데이트 (2026-02-11)
- [x] `tier-only` parity gate 추가:
  - `python -m src.debug_gpu_single_run --parity-gate`
  - 절대 오차 `--parity-tolerance` 기준 mismatch 카운트 산출
  - mismatch `> 0`이면 즉시 `AssertionError`로 fail-fast
- [ ] top-k/scenario pack parity는 후속 구현 필요

## 1. 배경
- CPU는 SSOT, GPU는 동일 결과 보장 원칙
- 현재 단일 실행 비교만으로는 대규모 조합의 edge case drift를 놓칠 수 있음

## 2. 구현 범위
- `src/debug_gpu_single_run.py`
  - top-k 파라미터 배치 parity 모드 추가
- `src/parameter_simulation_gpu.py`
  - parity 입력 스냅샷 직렬화(후보군 모드 포함)
- `tests/`
  - `test_cpu_gpu_parity_topk.py` 추가
  - 스냅샷 메타데이터 검증 테스트 보강

## 3. 체크리스트
- [ ] top-k(권장 100+) 후보를 CPU/GPU 모두 실행해 일치 여부 검증
- [ ] `candidate_source_mode` 3종 parity 검증:
  - `weekly`
  - `hybrid_transition`
  - `tier`
- [ ] scenario pack parity 검증:
  - `baseline_deterministic`
  - `seeded_stress`(권장 50~100 seed)
  - `jackknife_drop_topN`(상위 기여 1~3종목 제거)
- [ ] mismatch 리포트 표준화: first mismatch index + cash/positions/value dump
- [ ] snapshot 메타데이터 저장: 기간, 파라미터, 코드 버전, 생성시각
- [ ] snapshot 메타데이터에 `candidate_source_mode`, `use_weekly_alpha_gate` 필드 추가
- [ ] snapshot 메타데이터에 `scenario_type`, `seed_id`, `drop_top_n` 필드 추가
- [ ] GPU 미사용 환경 skip 처리 유지
- [ ] CI/로컬 실행 명령 문서화

## 4. 브랜치 규칙 (A안 전환 연계)
- [ ] `main` 직접 수정 금지, 기능 브랜치에서 parity 하네스 변경 수행
- [ ] 권장 브랜치: `feature/issue56-parity-topk-universe-modes`

## 5. 완료 기준
- top-k parity mismatch `0건`만 통과
- scenario pack(`baseline_deterministic`, `seeded_stress`, `jackknife_drop_topN`) parity mismatch `0건`
- 스냅샷 갱신 기준/절차가 문서화
- 실패 시 재현 가능한 리포트 자동 생성

## 6. 제외 범위
- CPU/GPU 공통 코드로 강제 통합
- 전략 성능 개선 목적 로직 변경
