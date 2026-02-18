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
  - `tier` 경로 주요 정합 이슈 수정:
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
  - 체크포인트(2026-02-17):
    - `strict_hysteresis_v1` 적용 직후 parity 스모크 실행을 보류하고 후속 슬롯에서 재검증
  - 재검증 결과(동일일자 5거래일):
    - `results/parity_topk_tier_top5_5d.json` (`total_mismatches=5`)
    - `results/parity_topk_tier_top5_5d_retry1.json` (`total_mismatches=5`, 재시도 동일)
    - 공통 first mismatch: `2021-01-05`, `abs_diff=6538.0`
  - 원인 추적/수정(2026-02-17):
    - 진단 도구 추가: `src/parity_sell_event_dump.py` (CPU/GPU 매수/매도 이벤트 1:1 비교)
    - 원인 확인: GPU 신규진입 루프가 `max_slots` 범위만 순회하여, 앞선 보유/쿨다운 후보로 인해 뒤쪽 유효 후보를 누락
    - 수정: `src/backtest/gpu/logic.py` 신규진입 후보 순회를 `range(num_candidates)`로 확장
  - 수정 후 재검증(동일일자 5거래일):
    - `results/parity_trade_events_param0_5d_after_slotloopfix.json`
      - `sell: 5 vs 5, mismatch=0`
      - `buy: 8 vs 8, mismatch=0`
    - `results/parity_topk_tier_param0_5d_after_slotloopfix.json` (`total_mismatches=0`)
    - `results/parity_topk_tier_top5_5d_after_slotloopfix.json` (`total_mismatches=0`)
  - 파이프라인 분리(2026-02-17):
    - `src.cpu_gpu_parity_topk`에 2단계 실행 추가
    - `--pipeline-stage gpu`: GPU 스냅샷만 생성
    - `--pipeline-stage cpu --snapshot-in <...>`: 스냅샷 기반 CPU strict parity만 재실행
    - CPU 병렬 옵션 `--cpu-workers` 추가(기본 1, 권장 2~4)

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
- [x] `candidate_source_mode` 3종 parity 검증:
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
- [x] 승격 게이트용 5거래일 이상 `tier top-k>=5` decision-level parity `0 mismatch` 충족(기준 시나리오/기간)
- [ ] mismatch 원인(class) 태깅 리포트 추가(선정 drift / 체결가 drift / 수치 오차)
- [ ] `Tier v2 deterministic mapping/sort` 정책 적용(Release 기본)
- [ ] `Tier v2` 운영 모니터링/롤백 지표 2주 관찰 통과
- [x] GPU/CPU 분리 실행 경로 문서화(`--pipeline-stage gpu/cpu`, snapshot replay)
- [ ] 장기 strict 게이트(최소 6개월, `top-k=1/5`) `0 mismatch` 증적 추가
- [ ] 전략/체결/후보선정 로직 변경 시 parity 재검증 트리거 규칙을 CI 문서/체크리스트에 연결

## 4. 브랜치 규칙 (A안 전환 연계)
- [ ] `main` 직접 수정 금지, 기능 브랜치에서 parity 하네스 변경 수행
- [ ] 권장 브랜치: `feature/issue56-parity-topk-universe-modes`

## 5. 완료 기준
- Research Gate (탐색 단계, GPU 가속 목적):
  - metric-level 비교 허용(성능/랭크 지표 기준)
  - top-k 후보 추출 및 scenario pack 실행 증적 확보
- Release Gate (승격/배포 단계, CPU=SSOT):
  - decision-level parity `0 mismatch` 필수
  - 비교 키: `date/ticker/side/qty/fill_price` + 현금/포지션 상태
  - 권장 최소 범위: `tier`, 5거래일 이상, `top-k>=5`
- 스냅샷 갱신 기준/절차 문서화
- 실패 시 재현 가능한 리포트 자동 생성

### 완료 판정 (2026-02-17 갱신)
- `Research Gate`: **충족**
  - `hybrid_transition` 1일 증적: mismatch `0`
  - `top-k 100` 1일 증적: mismatch `0`
  - scenario pack 1일 증적: mismatch `0`
- `Release Gate`: **단기 strict 기준 충족(조건부)**
  - `tier` 5거래일 `param0` mismatch `0`
  - `tier` 5거래일 `top-k=5` mismatch `0`
- `Issue #56 전체`: **부분 충족(지속 추적 유지)**
  - 이유: 로직 변경 시 parity 재발 가능성이 있어 장기 strict 증적 및 운영 트리거 관리 필요

## 6. Tier v2 매핑/정렬 정책 (2026-02-17 합의안)
목표: `tier` 경로의 decision-level parity 안정화와 ATR 영향 완화.

### 6-1. Release 기본 정책(결정론 고정)
- 시점 앵커:
  - `signal_date = T-1`
  - `tier_date = max(DailyStockTier.date <= signal_date)`
  - `snapshot_date = max(TickerUniverseSnapshot.snapshot_date <= signal_date)`
  - `mcap_date = max(MarketCapDaily.date <= signal_date)`
- 후보군:
  - `tier in (1,2)` + `tier@tier_date ∩ snapshot@snapshot_date` 교집합
- 정렬 키(고정):
  - primary: `tier_rank asc` (`tier1=0`, `tier2=1`)
  - secondary: `market_cap` 내림차순(정수 quantize)
  - tertiary: `atr_14_ratio` 내림차순(정수 quantize)
  - final tie-breaker: `stock_code asc`
- ATR 규칙:
  - `atr_14_ratio > 0`인 후보만 정렬 대상
  - cap은 적용하지 않음(이미 후순위)
- 결측/지연 처리:
  - `market_cap NULL -> 0`, `atr NULL or <=0 -> 제외`
  - `snapshot/tier` as-of staleness가 3 거래일 초과 시 fail-close
  - `ShortSellingDaily`는 lag 구조로 ranking key에서 제외
- 실행 모드:
  - `parity_mode=strict`: 체결 정산을 포지션(차수) 단위 floor로 계산(CPU 정산 정합 우선)
  - `parity_mode=fast`(기본): 체결 정산 합산 후 floor 1회(GPU 처리량 우선)

### 6-2. Research 실험 정책(선택)
- 정렬 키 실험 허용:
  - `tier_rank asc -> liquidity desc -> capped_atr desc -> stock_code asc`
- 단, Release 승격 전에는 반드시 Release 기본 정책으로 재검증하고 parity gate 통과 필요.

### 6-3. 구현 템플릿(Python key)
```python
tier_rank = 0 if tier == 1 else 1
mcap_q = int((market_cap or 0) // 1_000_000)
atr_q = int(round(atr_14_ratio * 10000))
sort_key = (tier_rank, -mcap_q, -atr_q, stock_code)
```

### 6-4. 롤백 트리거(운영)
- `parity mismatch_count > 0` (decision-level)
- `top-k ordered hash` CPU/GPU 불일치 발생
- `candidate coverage < 99%` (tier/snapshot/mcap 조인 기준)
- `as-of staleness > 3 trading days`
- `empty_entry_day_rate > 20%`가 3일 연속 발생

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
  --parity-mode fast \
  --no-fail-on-mismatch

# 4) tier 전용 회귀 모니터(대체: parity_topk 직접 실행)
python -m src.cpu_gpu_parity_topk \
  --start-date 2021-01-04 \
  --end-date 2021-01-08 \
  --top-k 1 \
  --params-csv /tmp/parity_params_smoke.csv \
  --scenario baseline_deterministic \
  --candidate-source-mode tier \
  --parity-mode strict \
  --cpu-workers 1 \
  --no-fail-on-mismatch \
  --out results/tier_parity_monitor_replacement_5d.json

# 5) 승격/배포 전 strict 검증(정산 정합 우선)
python -m src.cpu_gpu_parity_topk \
  --start-date 2021-01-04 \
  --end-date 2021-01-08 \
  --top-k 5 \
  --scenario baseline_deterministic \
  --candidate-source-mode tier \
  --parity-mode strict

# 6) 2단계 분리 실행 - 1단계(GPU snapshot 생성)
python -m src.cpu_gpu_parity_topk \
  --pipeline-stage gpu \
  --start-date 2021-01-04 \
  --end-date 2021-01-08 \
  --top-k 5 \
  --scenario baseline_deterministic \
  --candidate-source-mode tier \
  --parity-mode strict \
  --snapshot-out results/parity_topk_tier_top5_5d.snapshot.json \
  --out results/parity_topk_tier_top5_5d_gpu_stage.json

# 7) 2단계 분리 실행 - 2단계(CPU strict parity replay)
python -m src.cpu_gpu_parity_topk \
  --pipeline-stage cpu \
  --snapshot-in results/parity_topk_tier_top5_5d.snapshot.json \
  --cpu-workers 4 \
  --tolerance 1e-3 \
  --no-fail-on-mismatch \
  --out results/parity_topk_tier_top5_5d_cpu_stage.json
```

## 8. 제외 범위
- CPU/GPU 공통 코드로 강제 통합
- 전략 성능 개선 목적 로직 변경

## 9. 운영 추적 규칙 (Follow-up)
- 이 이슈는 완료 후 닫는 성격이 아니라, parity 회귀 리스크를 추적하는 운영 이슈로 유지한다.
- 아래 변경이 발생하면 `tier strict` parity를 재실행한다.
  - `src/backtest/cpu/*` 전략/체결 로직 변경
  - `src/backtest/gpu/engine.py`, `src/backtest/gpu/logic.py` 변경
  - 후보 선정/정렬 키(Tier/ATR/시총/stock_code) 또는 슬롯 배정 규칙 변경
  - 수수료/세금/호가/rounding/dtype(float32) 규칙 변경
  - 관련 DB join/as-of 매핑 규칙 변경
- 최소 재검증 세트:
  - 단기 스모크: 5거래일 `param0`, `top-k=5`, `parity-mode strict`
  - 장기 게이트: 6개월 이상 `top-k=1/5`, `parity-mode strict`

## 10. #98 Throughput 연계 규칙 (2026-02-18)
- #98 문서의 변경 분류를 그대로 적용:
  - `PC(Parity-Coupled)`: 결과 의사결정 영향 가능 변경
  - `PO(Perf-Only)`: 결과 불변 성능 최적화
- 실행 원칙:
  - PC는 CPU/GPU 동시 수정이 기본이며, 본 이슈의 strict gate를 통과해야만 병합 가능
  - PO는 단독 수정 가능하되, 병합 전 strict parity 최소 세트를 반드시 재실행
- #98 항목별 gate 매핑:
  - `P-005/P-006/P-008/P-009` -> Release Gate(결정 레벨) 필수
  - `P-001/P-002/P-003/P-004/P-007/P-010/P-011/P-012/P-013/P-014` -> Smoke + strict 재검증
- 표준 검증 순서:
  - 1) GPU snapshot 생성(`--pipeline-stage gpu`, `--parity-mode strict`)
  - 2) CPU replay(`--pipeline-stage cpu`, 동일 snapshot)
  - 3) `mismatch=0` 확인 후 성능 지표 비교
- 병합 차단 규칙:
  - decision-level mismatch 1건 이상 발생 시 즉시 병합 중지
  - 원인 태깅(후보선정 drift/체결 drift/수치 오차) 완료 전 재시도 금지
