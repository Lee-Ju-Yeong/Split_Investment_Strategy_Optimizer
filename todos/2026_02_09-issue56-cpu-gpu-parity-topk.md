# test(parity): CPU/GPU 정합성 하네스 강화 - top-k 배치 검증 (Issue #56)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/56`
- 작성일: 2026-02-09
- 목적: 단일 파라미터 검증을 넘어, 최적화 상위 후보(top-k) 전수 parity 검증으로 회귀 리스크 차단

## 0. 진행 현황 (2026-02-09)
- 선행 완료(`#67` Phase A):
  - GPU 후보 기준일 `signal_date(T-1)` 정렬
  - ATR 조회 `as-of(<=)` 정렬
  - Tier preload를 `start 이전 latest 1행 + 기간 데이터`로 보강(30일 가정 제거)
- 최신 상태 (2026-02-17):
  - parity 하네스 본체 구현 완료: `src/cpu_gpu_parity_topk.py`
  - `tier` 경로 mismatch 수정 완료:
    - 신규 진입/추가매수/매도 신호의 `T-1` 기준 정합
    - 익절의 `T-1 신호` + `T0 체결 가능` 동시 조건 반영
  - 체결 정책 정리:
    - 단일 정책: `open-market` (T0 시가 체결 통일)
  - 회귀 테스트 추가:
    - `tests/test_backtest_strategy_gpu.py` (`TestIssue56TierSignalExecutionParity`)
  - 실행 증적:
    - `results/parity_topk_smoke_tier_5d_after_fix.json` (`total_mismatches=0`)
    - `results/parity_topk_tier_scenariopack_1d.json` (`baseline/seeded/jackknife`, `total_mismatches=0`)
    - `results/parity_topk_smoke_hybrid_1d.json` (`hybrid_transition`, `total_mismatches=0`)
    - `results/parity_topk_tier_top100_1d.json` (`top-k=100`, `total_mismatches=0`)

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
- [x] top-k(권장 100+) 후보를 CPU/GPU 모두 실행해 일치 여부 검증
- [ ] `candidate_source_mode` 3종 parity 검증:
  - [x] `weekly`
  - [x] `hybrid_transition`
  - [x] `tier`
- [x] scenario pack parity 검증:
  - `baseline_deterministic`
  - `seeded_stress`(권장 50~100 seed)
  - `jackknife_drop_topN`(상위 기여 1~3종목 제거)
- [x] mismatch 리포트 표준화: first mismatch index + cash/positions/value dump
- [x] snapshot 메타데이터 저장: 기간, 파라미터, 코드 버전, 생성시각
- [x] snapshot 메타데이터에 `candidate_source_mode`, `use_weekly_alpha_gate` 필드 추가
- [x] snapshot 메타데이터에 `scenario_type`, `seed_id`, `drop_top_n` 필드 추가
- [x] GPU 미사용 환경 skip 처리 유지
- [x] CI/로컬 실행 명령 문서화

## 4. 브랜치 규칙 (A안 전환 연계)
- [ ] `main` 직접 수정 금지, 기능 브랜치에서 parity 하네스 변경 수행
- [ ] 권장 브랜치: `feature/issue56-parity-topk-universe-modes`

## 5. 완료 기준
- top-k parity mismatch `0건`만 통과
- scenario pack(`baseline_deterministic`, `seeded_stress`, `jackknife_drop_topN`) parity mismatch `0건`
- 스냅샷 갱신 기준/절차가 문서화
- 실패 시 재현 가능한 리포트 자동 생성

### 완료 판정 (2026-02-17)
- `tier 범위`: **충족**
  - 5거래일 baseline: mismatch `0`
  - scenario pack(1일, top-k=2): mismatch `0`
- `Issue #56 전체`: **충족(현재 정의 기준)**
  - `hybrid_transition` 1일 증적 확보: mismatch `0`
  - `top-k 100` 증적 확보: mismatch `0`

## 7. 실행 명령 (로컬/CI)
```bash
# 1) 단일 모드(top-k + 전체 scenario pack)
python -m src.cpu_gpu_parity_topk \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --top-k 100 \
  --scenario all \
  --candidate-source-mode tier

# 2) 3모드 매트릭스 검증(weekly/hybrid_transition/tier)
python -m src.cpu_gpu_parity_topk \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --top-k 100 \
  --scenario all \
  --candidate-source-mode all

# 3) mismatch가 있어도 리포트만 남기고 종료
python -m src.cpu_gpu_parity_topk \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --top-k 20 \
  --scenario baseline_deterministic \
  --no-fail-on-mismatch

# 4) tier 전용 회귀 모니터(운영 전 체크, PASS/FAIL 한 줄 출력)
python -m src.tier_parity_monitor \
  --start-date 2021-01-04 \
  --end-date 2021-01-08 \
  --top-k 1 \
  --params-csv /tmp/parity_params_smoke.csv \
  --scenario baseline_deterministic \
  --config-path /tmp/config_parity_use_pure.yaml
```

## 6. 제외 범위
- CPU/GPU 공통 코드로 강제 통합
- 전략 성능 개선 목적 로직 변경
