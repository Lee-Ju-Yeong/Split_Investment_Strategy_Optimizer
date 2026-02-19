# perf(gpu): GPU Throughput 리팩토링 + 성능 저하 fallback 제거 (Issue #98)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/98`
- 작성일: 2026-02-17
- 최종 갱신: 2026-02-20
- 목적: GPU 처리량 병목과 fallback 유발 성능 저하를 제거하되, CPU=SSOT 원칙과 decision-level parity(`#56`)를 유지

## 0. 분리 원칙 (Issue #97/#56/#67 관계)
- #97: 레거시 자산 정리 거버넌스(삭제/아카이브/승인 게이트)
- #98: 성능 리팩토링(throughput, kernel launch, host-device sync, I/O)
- #56: parity hard gate(Release 기준 `mismatch=0`)
- #67: tier/hybrid 후보군 로직 및 host 병목 제거 Phase B/C
- 원칙:
  - 정책 정리(#97)와 성능 리팩토링(#98)을 같은 PR에 섞지 않는다.
  - 정합성 영향 변경은 CPU/GPU 동시 수정 + `#56 strict parity`를 필수로 통과한다.

## 1. 변경 분류 프레임 (필수)
| class | 정의 | 예시 | 필수 게이트 |
| --- | --- | --- | --- |
| `PC` (Parity-Coupled) | 결과 의사결정(후보군/정렬/체결/시점 앵커)에 영향 가능성이 있는 변경 | `signal_date`, 정렬 tie-break, top-k 슬롯 배정, 체결/rounding 규칙 | CPU/GPU 동시 수정 + `#56` strict `0 mismatch` |
| `PO` (Perf-Only) | 로직 결과를 바꾸지 않는 구현 최적화 | 캐시, host sync 제거, 로더 재사용, profiling 추가 | 단독 수정 가능 + strict parity 사후 검증 |

## 2. 성능 병목 전수 인벤토리 (2026-02-18)
| id | class | location | current_behavior | impact | recommendation | owner |
| --- | --- | --- | --- | --- | --- | --- |
| P-001 | PO | `src/optimization/gpu/kernel.py` | batch-size 계산 실패 시 `None` 반환 | kernel launch 과다, GPU util 저하 | 안전 기본치/heuristic + fail-fast 기준 명확화 | #98 |
| P-002 | PO | `src/optimization/gpu/parameter_simulation.py` | `get_optimal_batch_size` 실패 시 작은 batch fallback | 준-직렬 실행으로 throughput 급락 | 최소 batch-size 가드 + fallback 축소 | #98 |
| P-003 | PO | `src/pipeline/ohlcv_batch.py` | `--allow-legacy-fallback`로 universe 과대 확장 가능 | 텐서화/로딩 비용 급증 | 운영 경로 제거(비상 복구 분리) | #98/#97 |
| P-004 | PO | `src/data/collectors/financial_collector.py` | snapshot 부재 시 legacy universe fallback | downstream GPU 비용 상승 | snapshot 우선 강제 + 단계적 제거 | #98/#97 |
| P-005 | PC | `src/backtest/gpu/engine.py` | 일별 후보 선정에서 `tolist()/to_pylist()` + Python 루프/`.loc` 사용 | host-device sync 증가, GPU 벡터화 붕괴 | 후보군/랭킹 경로를 CuPy 텐서 기반으로 고정 | #98/#67/#56 |
| P-006 | PC | `src/backtest/gpu/data.py` | `_collect_candidate_rank_metrics_asof`가 cudf filter/concat/pylist 반복 | 일별 재할당/동기화 누적 | as-of metric tensor 사전 생성 후 gather | #98/#67/#56 |
| P-007 | PO | `src/backtest/gpu/logic.py` | `run_lengths.tolist()`로 device->host sync | 세그먼트 연산 오버헤드 누적 | CuPy-only segment/prefix 연산으로 치환 | #98 |
| P-008 | PC | `src/backtest/gpu/logic.py` | `_process_new_entry_signals_gpu`의 `for k in range(num_candidates)` 순차 루프 | kernel launch 과다 + 후보 수 증가 시 급격한 저하 | 후보 우선순위/자본차감의 결정론을 유지한 벡터화 단계 도입 | #98/#56 |
| P-009 | PC | `src/backtest/gpu/utils.py` | 후보 정렬이 Python `sorted` | CPU 정렬 hot path 편입 | `cp.lexsort` 기반 결정론 정렬 | #98/#56 |
| P-010 | PO | `src/backtest/cpu/strategy.py` | `get_stock_row_as_of` 티커별 반복 조회(N+1) | CPU 경로 I/O 바운드화 | 일자 단위 배치 조회/캐시 dict 도입 | #98/#57/#58 |
| P-011 | PO | `src/backtest/cpu/portfolio.py`, `src/backtest/cpu/execution.py` | `get_latest_price/get_ohlc_data_on_date/get_name_from_ticker` 반복 호출 | 거래/평가 루프 중복 조회 | 일자 스코프 캐시 주입 + 중복 집계 제거 | #98/#57/#58 |
| P-012 | PO | `src/data_handler.py` | `@lru_cache(maxsize=200)` + row copy/asof 반복 | universe 확장 시 캐시 미스 증가, DB thrashing | 캐시 정책 재설계(구간/배치 중심) | #98/#57/#58 |
| P-013 | PO | `src/optimization/gpu/data_loading.py` | 매 호출 `create_engine` + pandas round-trip | fold/재실행 시 로딩 지연 누적 | engine 재사용 + GPU 친화 로딩 경로 | #98/#58 |
| P-014 | PO | `tests/test_backtest_strategy_gpu.py` | 성능 회귀 가드 부재(`pass` placeholder) | 회귀 조기 탐지 불가 | 고정 데이터 benchmark/smoke budget 테스트 추가 | #98/#56 |
| P-015 | PO | `src/backtest/gpu/engine.py` | `signal_day_idx < 0` 분기에서 `cp.zeros` 4개를 일별 재할당 | 장기 구간에서 allocator churn/단편화 누적 | fallback zero tensor를 루프 밖에서 1회 생성 후 재사용 | #98 |

## 3. 비목표 (Out of Scope)
- 전략 규칙 변경(수익률 개선 목적 알고리즘 변경)
- 체결 규칙 변경(수수료/세금/호가 정책 변경)
- parity 기준 완화

## 4. 게이트
- Gate A (설계 동결):
  - 각 항목의 `class(PC/PO)`, 구현 범위, 롤백 경로 확정
  - PC 항목은 CPU/GPU 동시 변경 파일 목록을 사전 명시
- Gate B (검증):
  - 성능 지표 Before/After 동시 제출
  - `#56` strict parity(`decision-level`) `0 mismatch` 통과
- Gate C (반영):
  - 운영 배포 전 승인 + 롤백 절차 검증 완료
- 게이트 운영 정책 소스:
  - PR/Nightly two-tier 및 full-13 강제 조건은 `todos/2026_02_09-issue56-cpu-gpu-parity-topk.md`의 `11`장을 단일 소스로 따른다.

## 5. 측정/검증 기준 (반드시 Before/After)
- 동일 기간/모드/파라미터 고정
- 성능 지표:
  - wall-time
  - GPU kernel launch count
  - GPU util(가능 시)
  - peak memory
  - host-device transfer count(가능 시)
- 안정성 지표:
  - strict parity mismatch `0`
  - parity failure 발생 시 즉시 rollback 가능

## 6. 실행 체크리스트
- [ ] Gate A: `P-001~P-015`의 `PC/PO` 분류와 PR 매핑 확정
- [x] PR-98A: `P-001~P-004` fallback 축소/정리
- [ ] PR-98B(PC): `P-005/P-006/P-008/P-009` CPU/GPU 동시 수정 + parity 통과
- [ ] PR-98C(PO): `P-007/P-010/P-011/P-012/P-013/P-015` 캐시/동기화/I/O 최적화
- [ ] PR-98D: `P-014` 성능 회귀 가드/벤치마크 테스트 반영
- [ ] 성능 측정 결과 문서화(before/after)
- [ ] `#56` strict parity 재검증(`0 mismatch`)
- [ ] Gate B 승인 완료
- [ ] Gate C 승인 완료

## 7. PR 분할 제안
- PR-98A: batch-size + universe fallback 정리
  - `src/optimization/gpu/kernel.py`
  - `src/optimization/gpu/parameter_simulation.py`
  - `src/pipeline/ohlcv_batch.py`
  - `src/data/collectors/financial_collector.py`
- PR-98B (Parity-Coupled Hot Path): 후보선정/정렬/신규진입 처리량 리팩토링
  - `src/backtest/gpu/engine.py`
  - `src/backtest/gpu/data.py`
  - `src/backtest/gpu/logic.py`
  - `src/backtest/gpu/utils.py`
  - 필요 시 대응 CPU 파일(`src/backtest/cpu/strategy.py`, `src/backtest/cpu/execution.py`)
- PR-98C (Perf-Only Data Path): CPU I/O/캐시/로딩 최적화
  - `src/backtest/cpu/strategy.py`
  - `src/backtest/cpu/portfolio.py`
  - `src/backtest/cpu/execution.py`
  - `src/data_handler.py`
  - `src/optimization/gpu/data_loading.py`
- PR-98D: 성능/정합 게이트 자동화
  - `tests/test_backtest_strategy_gpu.py`
  - parity/perf 실행 문서 및 리포트 템플릿

## 8. 실행 순서 (파일 단위 상세)
### 8-1. Baseline 고정 (코드 변경 전 미수집 -> PR-98A 기준선 채택)
- [x] 성능 baseline 수집(B0):
  - 명령: `python -m src.parameter_simulation_gpu` + `/usr/bin/time -v`
  - 로그: `results/perf_baseline_strict_hyst_20260220_024023.log`
  - 결과 CSV: `results/standalone_simulation_results_20260220_024023.csv`
- [x] parity baseline 수집(B0):
  - `python -m src.cpu_gpu_parity_topk ... --parity-mode strict --params-csv <topk csv> --topk 3`
  - 결과: `summary.failed=0`, `summary.passed=3`
- [x] 기준선 정책:
  - 코드 변경 전 baseline이 없으므로, `PR-98A (P0) fallback 정리 완료 상태`를 공식 baseline(B0)으로 사용

### 8-2. PR-98A (PO) fallback 정리
- [x] `src/optimization/gpu/kernel.py`
  - `get_optimal_batch_size` 실패 시 fallback 경계/최소치 명시
  - 불능 상태는 로그 + fail-fast 기준 추가
- [x] `src/optimization/gpu/parameter_simulation.py`
  - batch-size fallback 축소 및 안전 기본치 적용
- [x] `src/pipeline/ohlcv_batch.py`
  - `--allow-legacy-fallback` 운영 경로 분리/제한
- [x] `src/data/collectors/financial_collector.py`
  - snapshot 미존재 fallback 축소
- [ ] 검증:
  - strict parity 재실행(`mismatch=0`)
  - throughput 지표 비교

### 8-3. PR-98B (PC) GPU 후보군 hot path
- [ ] `src/backtest/gpu/engine.py`
  - `tolist()/to_pylist()` 제거
  - 일별 weekly 스캔 경로 사전 인덱스화
- [ ] `src/backtest/gpu/data.py`
  - `_collect_candidate_rank_metrics_asof`의 일별 filter/concat 축소
  - tensor gather 중심으로 재구성
- [ ] `src/backtest/gpu/logic.py`
  - 신규진입 후보 처리에서 launch 과다 루프 축소
  - 결정론 순서/자본차감 semantics 보존
- [ ] `src/backtest/gpu/utils.py`
  - Python `sorted` -> `cp.lexsort` 전환
- [ ] 필요 시 동반 CPU 수정:
  - `src/backtest/cpu/strategy.py`
  - `src/backtest/cpu/execution.py`
- [ ] 검증:
  - `#56` strict parity(5거래일 + top-k gate) 필수 통과
  - mismatch 발생 시 즉시 롤백, 원인 태깅 후 재시도

### 8-4. PR-98C (PO) CPU I/O/캐시 + GPU 로딩 최적화
- [ ] `src/backtest/cpu/strategy.py`
  - 일자 단위 배치 조회 캐시
- [ ] `src/backtest/cpu/portfolio.py`
  - 평가/스냅샷 가격 조회 중복 제거
- [ ] `src/backtest/cpu/execution.py`
  - 주문 실행 경로 OHLC/이름 조회 캐시 주입
- [ ] `src/data_handler.py`
  - 캐시 정책(`maxsize`, 구간 단위 로딩) 재설계
- [ ] `src/optimization/gpu/data_loading.py`
  - `create_engine` 재사용
  - pandas round-trip 최소화
- [ ] 검증:
  - strict parity 재검증
  - CPU wall-time 개선 확인

### 8-5. PR-98D 게이트 자동화
- [ ] `tests/test_backtest_strategy_gpu.py`
  - `pass` placeholder 제거
  - 고정 입력 성능 스모크/예산 테스트 추가
- [ ] 문서/리포트 템플릿 정비
  - before/after 표준 포맷
  - parity 증적 링크 규격

## 9. 의사결정 규칙 (병합 차단)
- PC 변경에서 strict parity `mismatch>0`이면 병합 금지
- PO 변경에서 throughput 개선이 없고 리스크만 증가하면 병합 보류
- 측정 결과 누락 시 Gate B 미통과 처리

## 10. PR-98A 반영 내역 (2026-02-18)
- `src/optimization/gpu/kernel.py`
  - 가용 GPU 메모리 조회 경로를 `nvidia-smi` -> `cupy.runtime.memGetInfo` 순서로 확장
  - 메모리 조회 불가/부족 원인을 배치 크기 계산 로그에 명확히 출력
- `src/optimization/gpu/parameter_simulation.py`
  - batch-size 자동 계산 실패 시 `ctx.num_combinations` all-in fallback 제거
  - fallback 우선순위: `simulation_batch_size` -> adaptive safe default(목표 batch 수 기반, 상한/하한 적용)
- `src/data/collectors/financial_collector.py`
  - ticker universe 조회에 `allow_legacy_fallback`(default false) 도입
  - snapshot/history 비어 있을 때 기본 fail-fast, opt-in 시에만 legacy fallback 허용 + deprecated 경고
  - 실행 summary에 `universe_source`, `legacy_fallback_*` 지표 추가
- `src/pipeline/batch.py`
  - `--allow-financial-legacy-fallback` CLI 추가(기본 off, deprecated)
  - financial collector 호출에 legacy fallback 플래그 전달
- 테스트
  - `tests/test_pipeline_batch.py` (새 플래그 전달/파서 검증)
  - `tests/test_financial_collector_universe.py` (fail-fast/opt-in legacy/summary 검증)
  - `tests/test_gpu_parameter_batch_fallback.py` (배치 fallback 정책 검증)
  - `tests/test_gpu_kernel_batch_size.py` (`nvidia-smi`/runtime fallback 순서 검증)
  - 실행: `conda run -n rapids-env python -m unittest tests.test_pipeline_batch tests.test_ohlcv_batch tests.test_financial_collector_universe tests.test_gpu_parameter_batch_fallback tests.test_gpu_kernel_batch_size`

## 11. Tier-only 병목 검토 메모 (멀티에이전트, 2026-02-18)
- 검토 범위: `candidate_source_mode=tier` 경로
- 제약: CPU=SSOT, strict parity `mismatch=0`, 전략/체결 규칙 불변
- 관찰 요약:
  - 최근 실행에서 로딩 단계가 `12~20s`, 단일 배치 kernel은 `~0.61s`
  - Tier1 스캔 자체(`cp.where(signal_tiers == 1)`)는 상대적으로 경량
  - 병목은 후보 후처리의 host-device 왕복/정렬/순차 루프에 집중

### 11-1. 합의된 병목/리스크 항목
| id | class | location | issue | note |
| --- | --- | --- | --- | --- |
| T-001 | PO | `src/backtest/gpu/engine.py` | `all_data_gpu.reset_index()` 중복 materialize | 로딩 비용 증가, 의미 변화 없음 |
| T-002 | PO | `src/backtest/gpu/engine.py` | tier 모드에서도 `weekly_filtered_reset_idx` 선계산 | 미사용 연산 비용 |
| T-003 | PO | `src/backtest/gpu/data.py` | 텐서 생성 시 동일 인덱스 변환 반복 | D2D 준비 단계 오버헤드 |
| T-004 | PC | `src/backtest/gpu/engine.py`, `src/backtest/gpu/data.py` | `tolist()/to_pylist()/loc` 중심 후보 메트릭 처리 | host sync + parity 민감 구간 |
| T-005 | PC | `src/backtest/gpu/utils.py` | Python `sorted` 기반 랭킹 | tie-break drift 시 parity 위험 |
| T-006 | PC | `src/backtest/gpu/logic.py` | `_process_new_entry_signals_gpu`의 후보 순차 루프 | 후보 수 증가 시 throughput 급락 |

### 11-2. PR-98B 세분화 제안 (tier-only 우선)
- `PR-98B-1 (PO, 저위험)`:
  - `T-001`, `T-002`, `T-003` 우선 반영
  - 목표: 의미 불변 상태에서 로딩/준비 단계 비용 절감
- `PR-98B-2 (PC, 정렬/메트릭)`:
  - `T-004`, `T-005` 반영
  - 목표: 후보 메트릭/정렬 경로 GPU 중심 재구성 + 결정론 계약 고정
- `PR-98B-3 (PC, 신규진입 hot path)`:
  - `T-006` 반영
  - 목표: 순차 루프 축소하되 CPU와 동일한 슬롯/자본차감 semantics 유지

### 11-3. 각 단계 공통 게이트
- 성능:
  - 동일 기간/파라미터로 wall-time, kernel 시간, 준비 단계 시간 비교
- 정합:
  - `python -m src.cpu_gpu_parity_topk --pipeline-stage all --parity-mode strict --candidate-source-mode tier`
  - 합격 기준: decision-level `mismatch=0`
- 롤백:
  - 단계별 PR 분리 유지, parity 실패 시 해당 단계만 즉시 revert

### 11-4. GPU Tier Tensor PIT 정합성 버그 (2026-02-18)
- class: `PC` (Parity-Coupled)
- 위치: `src/optimization/gpu/data_loading.py` (`preload_tier_data_to_tensor`)
- 원인:
  - 기존 구현이 `reindex(trading_dates)` 후 `ffill`을 수행해 `start_date` 이전 tier 이력이 먼저 소실됨
  - 결과적으로 `latest tier <= signal_date` PIT 규칙이 깨져 일부 종목이 GPU에서 tier=0 처리됨
- 영향:
  - 재현 케이스(2026-01-08, strict parity, tier 모드)에서 `196170`이 GPU 후보군에서 탈락
  - `005490`가 top-k 슬롯으로 밀려 조기 신규진입되고 buy mismatch 연쇄 발생
- 수정:
  - `union_index = df_wide.index.union(trading_dates_pd)` 기준으로 먼저 reindex+ffill
  - 그 다음 `trading_dates_pd`로 slice하여 as-of tier를 보존
  - ticker 컬럼/축을 문자열로 정규화해 매핑 drift 방지
- 검증:
  - `python -m src.parity_sell_event_dump --start-date 2026-01-05 --end-date 2026-01-09 --params-csv results/parity_debug_param0.csv --param-id 0 --candidate-source-mode tier --parity-mode strict --out results/parity_sell_debug_after_tierfix_20260218_125948.json --gpu-log-out results/parity_sell_debug_after_tierfix_20260218_125948.gpu.log`
  - 결과: `sell_mismatched_pairs=0`, `buy_mismatched_pairs=0`
  - 회귀 테스트 추가: `tests/test_gpu_tier_tensor_pit.py`

### 11-5. 정합성 비포기 결정 및 단계형 최적화 원칙 (2026-02-18)
- 결론:
  - throughput 개선을 위해 parity 정합성을 완화하지 않는다.
  - CPU=SSOT 원칙과 strict parity `mismatch=0`는 유지한다.
- 판단 근거:
  - 현재 지연의 주 병목은 GPU 커널 자체보다 후보군 준비/정렬/신규진입의 host-device 왕복 및 Python 루프 구간에 집중됨
  - 즉, 정합성 규칙을 깨지 않고도 구현 경로 최적화로 개선 여지가 충분함

#### 11-5-1. Non-Negotiable Invariants
- `signal_date`/PIT(`as-of <= date`) 일치
- 정렬 키 일치(`tier -> market_cap -> atr -> ticker`)
- 신규진입 슬롯 배정/자본 차감 순서 일치
- 체결가/호가/수수료 floor 규칙 일치
- float32 연산 규약 일치
- 매도 -> 신규진입 -> 추가매수 처리 순서 일치

#### 11-5-2. 적용 순서 (Risk-Based)
- 1단계 (Low Risk, 우선):
  - `src/backtest/gpu/engine.py`의 `tolist()/to_pylist()/Python loc loop` 축소
  - 목표: host 왕복 비용 감소, 의미 불변
- 2단계 (Medium Risk):
  - `src/backtest/gpu/data.py`의 `_collect_candidate_rank_metrics_asof` 반복 필터/concat 배치화
  - 목표: 일별 as-of 조회 비용 절감, PIT 계약 유지
- 3단계 (High Risk, 마지막):
  - `src/backtest/gpu/logic.py` 신규진입 순차 루프 최적화
  - 목표: 후보 수 증가 구간 성능 개선, 슬롯/자본 semantics 보존

#### 11-5-3. 금지/주의 패턴
- parity 검증 없이 정렬/슬롯/체결 규칙 변경
- `float64` 혼용 또는 CPU와 다른 rounding 경로 도입
- GPU 후보 결과를 Python 리스트로 반복 왕복하며 임의 재정렬
- PIT 조회 기준(`as-of`) 변경 후 strict gate 미실행

### 11-6. PR-98B-1 (PO) 1차 반영: T-001/T-002/T-003 (2026-02-18)
- 반영 범위:
  - `src/backtest/gpu/engine.py`
    - `all_data_gpu.reset_index()` 결과를 텐서 빌더에 재사용(중복 materialize 제거)
    - `candidate_source_mode=tier` 경로에서는 `weekly_filtered_gpu.reset_index()` 선계산 생략
  - `src/backtest/gpu/data.py`
    - 텐서 생성 시 `day_idx`/`ticker_idx`를 컬럼 반복마다 변환하지 않고 1회 계산 후 재사용
- 의미/정합성:
  - 후보 선정/정렬/체결 규칙은 변경하지 않음 (PO 유지)
  - strict parity gate 대상 변경 없음
- 테스트:
  - 신규: `tests/test_gpu_engine_prep_path.py`
    - tier 모드에서 weekly reset 생략 확인
    - weekly 모드에서 weekly reset 유지 확인
    - all_data reset 결과 재사용 확인
  - 기존 회귀: `tests/test_gpu_tier_tensor_pit.py`

### 11-7. PR-98B-2 준비 리팩토링: T-004 일부 (2026-02-18)
- 반영 범위:
  - `src/backtest/gpu/data.py`
    - `build_ranked_candidate_payload` 추가
    - candidate metrics를 row/column 단위 `loc` 반복 조회하지 않고 한 번에 추출 후 필터/정렬 처리
    - `_collect_candidate_rank_metrics_asof`를 `date<=signal_date` + `ticker별 latest 1건` 형태로 단순화
  - `src/backtest/gpu/engine.py`
    - 기존 per-ticker `loc` 루프를 helper 호출로 치환
- 의도:
  - host-device 왕복을 유발하는 반복 접근을 축소
  - 정렬 규칙(`market_cap_q desc -> atr_q desc -> ticker asc`)과 필터 규칙은 유지
- 테스트:
  - 신규: `tests/test_gpu_candidate_payload_builder.py`
    - 필터링/정렬/동률 ticker tie-break 동작 검증
  - 신규: `tests/test_gpu_candidate_metrics_asof.py`
    - `latest <= signal_date` 선택, 미래 행 배제, empty 입력 동작 검증

### 11-8. 스크래치 재검토 델타 (코드 재점검, 2026-02-18)
- 실행 증상(실측 로그):
  - `python -m src.parameter_simulation_gpu` 장기 구간 실행에서
  - `[GPU_PROGRESS] 25/244 (10.2%) elapsed=1:35:20 eta=13:55:09`
  - 해석: GPU 커널 연산보다 일별 후보군 준비/정렬 오버헤드가 wall-time을 지배
- 코드 확인(현 상태):
  - `src/backtest/gpu/engine.py`
    - tier 후보 인덱스를 `candidate_indices.tolist()`로 Python 변환
    - weekly 경로에서 `to_arrow().to_pylist()` + Python list/set 교집합
  - `src/backtest/gpu/data.py`
    - `_collect_candidate_rank_metrics_asof`: `date<=signal_date` 조건의 일별 필터/정렬/중복제거 반복
    - `build_ranked_candidate_payload`: `to_pylist()` + Python `for` 루프 기반 payload 구성
  - `src/backtest/gpu/logic.py`
    - `run_lengths.tolist()`를 통한 device->host sync 잔존
  - `src/backtest/gpu/engine.py`
    - `signal_day_idx < 0`일 때 `cp.zeros` 4개 재할당(신규 식별, P-015)
- strict_hysteresis_v1 최적화 주의점(숨은 리스크):
  - tier 마스크(`Tier1 only`, fallback 규칙)를 랭킹 이전에 적용하지 않으면
    후보군 자체가 달라져 parity drift가 발생한다.
  - 즉, `mask -> rank -> slot` 순서는 고정해야 한다.

## 12. 현재 상태 업데이트 (2026-02-20, Baseline 고정 완료)
- 브랜치 상태:
  - `feature/issue98-pr98a` 최신 푸시 완료
  - 최근 반영 커밋:
    - `a56654f` `perf(issue98): trim gpu prep overhead in tier path`
    - `fe66cd2` `refactor(issue98): collapse candidate metric loc loops`
    - `f7beee8` `refactor(issue98): simplify as-of candidate metrics lookup`
- 진행 중 작업:
  - PR-98B/98C throughput 개선 항목 진행
- baseline 실행 이슈:
  - 기존 baseline 실행(2015-01-01~2020-12-31)에서 `adjust_price_up_gpu` float64 중간배열로 CUDA OOM 발생(`std::bad_alloc`, 169,128,000B 할당 실패)
  - 대응: `src/backtest/gpu/utils.py`의 가격 올림 연산을 in-place float64로 전환해 피크 메모리 축소, OOM 시 chunked fallback 경로 추가
  - 일관성: `src/backtest/gpu/logic.py`의 동일 함수도 공용 `utils` 구현을 사용하도록 정렬
- 검증:
  - `tests/test_gpu_price_adjustment.py` 추가
  - 실행: `conda run -n rapids-env python -m unittest tests.test_gpu_price_adjustment`
- baseline 대기 중 선반영 완료 항목:
  - PR-98B-1(T-001/T-002/T-003) 저위험 최적화 반영/테스트 완료
  - PR-98B-2(T-004) 준비 리팩토링 일부 반영/테스트 완료
- 아직 미완료:
  - 동일 조건 `after` 1회 재실행 및 before/after 비교표 작성
  - 개선 이후 strict parity gate 재실행 및 `mismatch=0` 증적 첨부
- 운영 원칙:
  - baseline 완료 전에도 PO/저위험 리팩토링은 진행 가능
  - merge 판단은 baseline + parity 증적 확보 후 수행

## 13. Baseline(B0) 공식 기록 (PR-98A / P0 fallback 정리 기준)
- 기준 선언:
  - 코드 변경 전 baseline 미수집 상태이므로, `PR-98A (P0) fallback 정리 완료` 시점을 baseline(B0)로 확정
- 성능(B0):
  - 측정 로그: `results/perf_baseline_strict_hyst_20260220_024023.log`
  - 결과 CSV: `results/standalone_simulation_results_20260220_024023.csv`
  - wall-clock: `5:11:28`
  - user/system CPU time: `16524.25s / 2119.53s`
  - max RSS: `1207796 KB`
  - exit status: `0`
- 정합(B0):
  - strict parity(top-k=3): `summary.failed=0`, `summary.passed=3`
  - strict 이벤트 덤프: `buy_mismatched_pairs=0`, `sell_mismatched_pairs=0`
- 해석 규칙:
  - 이후 PR-98B/98C의 throughput 비교는 B0 대비(`before=B0`)로 수행
  - parity gate는 B0와 동일한 strict 조건으로 유지
