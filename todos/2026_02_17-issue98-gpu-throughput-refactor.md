# Issue #98: GPU Throughput Refactor

> Type: `implementation`
> Status: `in_progress`
> Priority: `P1`
> Last updated: 2026-03-08
> Related issues: `#98`, `#56`, `#67`, `#97`
> Gate status: `runtime governance approved; canonical Jan-Feb throughput baseline fixed; ranking parity fixtures fixed; PR-98D artifact guard active; slice 2a retained locally (baseline promotion pending)`

## 1. One-Page Summary
- What: GPU 처리량 병목과 fallback 유발 성능 저하를 줄이는 문서입니다.
- Why: 긴 최적화/WFO 실행 시간을 줄여야 하지만, 의미론이 흔들리면 성능 개선이 무의미해집니다.
- Current status: `slice 2a`는 canonical remeasure에서 baseline보다 약간 느렸지만, `slice 1 only` rollback canonical 2-run이 더 크게 느려서 현재 작업 트리는 `slice 2a`를 유지합니다. 다만 baseline 대비 throughput win은 아직 증명되지 않아 승격은 보류입니다.
- Next action: `PR-98D`는 actual `summary.json` artifact 기반으로 고정했고, 다음 판단은 `large-batch / long-window + strict parity` 증적에서 내립니다.

## 2. 초심자용 현재 판단 (2026-03-08)
### 2-1. 지금 무엇이 끝났나
- `PR-98C`의 안전한 성능 개선 2개는 이미 들어갔습니다.
  - CPU 쪽: 같은 날 같은 종목 데이터를 반복 조회하던 낭비를 줄였습니다.
  - GPU 쪽: batch마다 같은 market-data tensor를 다시 만들지 않게 했습니다.
- 즉, “결과를 바꾸지 않는 안전한 최적화”는 먼저 깔아둔 상태입니다.

### 2-2. 지금 보류 중인 것은 무엇인가
- 보류 중인 것은 `remaining execution-loop hot-path reduction`입니다.
- 지금 남은 후보는 주로 `additional_buy`의 남은 rank loop 축소, 더 공격적인 `new-entry` loop 축소, 그리고 `PR-98D` 성능 회귀 가드입니다.
- 이 중 GPU execution loop는 속도 이득이 더 클 가능성이 있지만, 후보 순서와 자본 차감 규칙에 닿기 때문에 `PC (Parity-Coupled)`입니다.
- 쉬운 말로 하면:
  - 이미 끝난 것: “같은 계산을 두 번 하지 않게 정리”
  - 아직 보류한 것: “매수 실행 루프 자체를 더 공격적으로 줄이는 일”

### 2-3. 왜 아직 바로 안 들어가나
- 이 항목은 빨라질 가능성은 크지만, CPU와 GPU가 같은 후보를 같은 순서로 뽑는지까지 다시 증명해야 합니다.
- 지금 바로 넣으면 아래 두 질문이 한 번에 섞입니다.
  - 정말 빨라졌는가?
  - 결과 의미가 바뀌지 않았는가?
- 이 프로젝트는 CPU가 기준(`SSOT`)이므로, 두 질문을 분리해서 확인해야 합니다.

### 2-4. 그래서 현재 상태를 한 줄로 말하면
- `무기한 보류`가 아니라 `Ready-after-remeasure`입니다.
- 즉, “하지 않는 일”이 아니라 “이번 slice의 실제 개선폭을 먼저 확인한 뒤 여는 일”입니다.

### 2-5. 이번 단계 전제조건 2개
- 조건 1: `prepared_market_data` slice의 target-hardware canonical 2-run baseline
  - 상태: 완료
  - 증적: `results/issue98_measure/pr98c_slice2_janfeb2024_cov020_20260308_131130/summary.json`
  - 핵심 수치:
    - `median_kernel_s = 1106.09`
    - `median_wall_s = 1146.35` (`19분 06초`)
    - `batch_count = 4 / 4`
    - `oom_retry = false / false`
  - 왜 필요한가: 방금 넣은 안전한 최적화가 실제로 얼마만큼 도움이 되는지 먼저 고정해야, 다음 PC 변경의 효과를 분리해서 볼 수 있습니다.
- 조건 2: ranking parity fixture 보강
  - `direct composite-rank parity fixture`
  - `multi-sim active-set rerank parity`
  - 왜 필요한가: 후보 정렬/선정 경로를 바꿀 때 CPU 기준과 drift가 없는지 바로 잡아내기 위해서입니다.
  - 상태: 완료
  - 증적:
    - `tests/test_gpu_candidate_payload_builder.py::test_direct_composite_rank_parity_fixture_matches_cpu_history`
    - `tests/test_gpu_new_entry_signals.py::test_multi_sim_active_set_rerank_matches_python_reference`

## 3. Plain-Language Rule
- `PO (Perf-Only)`: 결과를 바꾸지 않는 최적화
- `PC (Parity-Coupled)`: 후보군, 정렬, 체결 의미론에 영향을 줄 수 있는 최적화
- `PC`는 반드시 `#56` strict parity 증적을 다시 통과해야 합니다.

## 4. Current Priority Bottlenecks
- [x] fixed-data VRAM blind spot
- [x] daily as-of ranking precompute
- [ ] ranking scratch memory estimate 보강
- [ ] strict fallback telemetry 고정
- [x] multi-sim active-set rerank parity
- [x] additional-buy run-owner host sync 제거
- [ ] CPU I/O / session cache / engine reuse 정리

## 5. Current Plan
- [x] PR-98A: batch-size fallback / legacy universe fallback 일부 정리
- [ ] PR-98B: candidate hot path (`PC`)
- [ ] PR-98C: CPU I/O / cache / data loading (`PO`)
- [x] PR-98D: perf regression guard / benchmark
- [ ] before/after 문서화
- [ ] `#56` strict parity 재검증

## 6. Key Evidence
- 이미 있는 것:
  - baseline log: `results/perf_baseline_strict_hyst_20260220_024023.log`
  - canonical local baseline summary: `results/issue98_measure/pr98c_slice2_janfeb2024_cov020_20260308_131130/summary.json`
  - `slice 2a` canonical remeasure summary: `results/issue98_measure/pr98b2_slice2a_janfeb2024_cov020_20260308_152446/summary.json`
  - `slice 1 only` rollback canonical remeasure summary: `results/issue98_measure/pr98b2_slice1only_janfeb2024_cov020_20260308_162509/summary.json`
  - release gate board: `docs/operations/2026-03-06-hybrid-release-gate-board.md`
- 아직 없는 것:
  - `PC` 변경 후 longer-window strict parity 증적

## 7. Reading Guide
- 이 문서는 “어떻게 더 빠르게 만들까”보다 먼저 “어떤 최적화가 결과 의미를 바꾸는가”를 분리하는 문서입니다.
- 세부 병목 인벤토리는 아래 본문을 필요할 때만 읽으세요.

## 8. Detailed History And Working Log
### 0. 분리 원칙 (Issue #97/#56/#67 관계)
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
  - PR/Nightly two-tier 및 full-13 강제 조건은 `todos/done_2026_02_09-issue56-cpu-gpu-parity-topk.md`의 `11`장을 단일 소스로 따른다.

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
- [x] PR-98D: `P-014` 성능 회귀 가드/벤치마크 테스트 반영
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
- [x] 검증:
  - strict parity 재실행(`mismatch=0`)
  - throughput 지표 비교

### 8-3. PR-98B (PC) GPU 후보군 hot path
- [x] `src/backtest/gpu/engine.py`
  - `tolist()/to_pylist()` 제거
  - 일별 weekly 스캔 경로 사전 인덱스화
- [x] `src/backtest/gpu/data.py`
  - `_collect_candidate_rank_metrics_asof`의 일별 filter/concat 축소
  - tensor gather 중심으로 재구성
- [x] `src/backtest/gpu/logic.py`
  - 신규진입 후보 처리에서 launch 과다 루프 축소
  - 결정론 순서/자본차감 semantics 보존
- [x] `src/backtest/gpu/utils.py`
  - Python `sorted` -> `cp.lexsort` 전환
- [ ] 필요 시 동반 CPU 수정:
  - `src/backtest/cpu/strategy.py`
  - `src/backtest/cpu/execution.py`
- [x] 검증:
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
- [x] `tests/test_backtest_strategy_gpu.py`
  - `pass` placeholder 제거
  - canonical profile actual `summary.json` artifact 기반 `median 2-run` 성능 예산 가드 테스트 추가
  - baseline / `slice2a` / `slice1only rollback` artifact smoke coverage 추가
- [ ] 문서/리포트 템플릿 정비
  - before/after 표준 포맷
  - parity 증적 링크 규격

## 9. 의사결정 규칙 (병합 차단)
- PC 변경에서 strict parity `mismatch>0`이면 병합 금지
- PO 변경에서 throughput 개선이 없고 리스크만 증가하면 병합 보류
- 측정 결과 누락 시 Gate B 미통과 처리

## 10. PR-98A 반영 내역 (2026-02-18)

## 11. PR-98C slice 1 반영 내역 (2026-03-08)
- What:
  - `DataHandler` runtime lookup cache 추가
  - CPU strategy의 신규 진입/추가 매수에서 tier batch query + cached multi-ticker row access path 추가
  - `Portfolio`의 cached multi-ticker latest-price access path 지원
  - GPU engine의 zero signal tensor 재사용
- Why:
  - 같은 날 같은 종목에 대한 `asof`/OHLC/price lookup 중복을 줄이고, 보유 종목별 tier query를 batch로 바꿔 CPU 경로의 Python/DB 오버헤드를 줄이기 위함
  - row/price lookup은 새 set-based SQL이 아니라 runtime cache 위의 multi-ticker access path 정리다
  - GPU 쪽은 루프 내부 `cp.zeros` 재할당을 없애 allocator churn을 줄이기 위함
- Files:
  - `src/data_handler.py`
  - `src/backtest/cpu/strategy.py`
  - `src/backtest/cpu/portfolio.py`
  - `src/backtest/cpu/backtester.py`
  - `src/backtest/gpu/engine.py`
  - `tests/test_data_handler_runtime_cache.py`
  - `tests/test_cpu_candidate_priority.py`
  - `tests/test_portfolio.py`
  - `tests/test_backtester_entry_metrics.py`
- Verification:
  - `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_data_handler_runtime_cache tests.test_cpu_candidate_priority tests.test_portfolio tests.test_backtester_entry_metrics tests.test_point_in_time tests.test_gpu_engine_prep_path -v`
- Notes:
  - 아직 `PR-98C` 전체 완료는 아님
  - 이 slice는 correctness-safe perf-only 정리이며, throughput improvement는 별도 before/after 계측으로 확정해야 한다
  - `tests.test_gpu_engine_prep_path`는 CUDA/OS 제약 환경에서 skip 될 수 있다
  - 다음 큰 GPU 개선 후보는 batch별 market-data prep 재사용(`prepared_market_data` bundle)이다

## 12. PR-98C slice 2 진행 현황 (2026-03-08)
- What:
  - optimizer worker에서 fixed market-data preparation을 1회만 수행하고 batch 간 재사용하는 `prepared_market_data` bundle 경로 추가
  - batch-size estimator의 fixed-data VRAM 계산에 prepared price tensor bytes를 반영
- Why:
  - 기존에는 같은 `all_data_gpu`/`all_tickers`/`trading_dates_pd`에 대해 batch마다 `reset_index`, `ticker_idx` 매핑, OHLC tensor build, strict as-of forward-fill을 반복했다
  - RTX 5060 class 장비에서는 이 재준비 비용보다 OOM 없이 안정적으로 batch를 유지하는 것이 더 중요하므로, reuse와 VRAM accounting을 함께 묶는 것이 안전하다
- Files:
  - `src/backtest/gpu/engine.py`
  - `src/optimization/gpu/kernel.py`
  - `src/optimization/gpu/parameter_simulation.py`
  - `tests/test_gpu_engine_prep_path.py`
  - `tests/test_gpu_parameter_simulation_orchestration.py`
- Verification:
  - `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_gpu_engine_prep_path tests.test_gpu_parameter_simulation_orchestration tests.test_gpu_kernel_batch_size -v`
  - `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest discover -s tests -p 'test_*.py'`
- Notes:
  - 이번 slice는 `PO`만 건드린다. candidate ranking/as-of precompute는 다음 `PC` slice로 분리한다
  - standalone baseline이 `1 batch`인 경우 wall-time 절감폭이 작을 수 있으므로, throughput 증적은 multi-batch 창에서도 같이 수집해야 한다
  - `prepared_market_data`는 optional 인자로 유지해 direct caller(`debug_gpu_single_run`, parity harness)와 호환성을 보존한다
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
    - 이후 `#97 step 3`에서 tier-only runtime의 `weekly_filtered_gpu` dead plumbing 자체를 제거
  - `src/backtest/gpu/data.py`
    - 텐서 생성 시 `day_idx`/`ticker_idx`를 컬럼 반복마다 변환하지 않고 1회 계산 후 재사용
- 의미/정합성:
  - 후보 선정/정렬/체결 규칙은 변경하지 않음 (PO 유지)
  - strict parity gate 대상 변경 없음
- 테스트:
  - 신규: `tests/test_gpu_engine_prep_path.py`
    - engine signature에서 weekly filtered frame 제거 확인
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

## 14. PR-98B-2 진행 현황 (2026-02-20)
- 반영 범위(이번 커밋 묶음):
  - `src/backtest/gpu/engine.py`
    - tier/hybrid 후보군 결합 경로를 CuPy 배열 중심으로 유지(`candidate_indices_gpu`, `weekly_indices_gpu`)
    - hybrid alpha-gate 교집합을 `cp.isin`으로 처리하고, 최종 후보 인덱스는 `cp.unique`로 정규화
    - candidate metrics 수집 호출을 ticker string 목록 대신 `ticker_idx` 기반으로 전환
  - `src/backtest/gpu/data.py`
    - `_collect_candidate_rank_metrics_asof`를 `ticker_idx` as-of latest 조회로 단순화
    - 후보 인덱스를 `cudf.Series(int32)`로 명시 변환 후 `isin` 적용
    - `build_ranked_candidate_payload`를 cuDF 필터/정렬 기반으로 재구성하고, host `for` 루프 제거
    - debug가 아닐 때 `ranked_records` materialize 생략
- 테스트:
  - `conda run --no-capture-output -n rapids-env python -m unittest tests.test_gpu_candidate_metrics_asof tests.test_gpu_candidate_payload_builder tests.test_gpu_engine_prep_path`
  - `conda run --no-capture-output -n rapids-env python -m unittest tests.test_cpu_gpu_parity_topk`
  - 결과: 모두 통과
- 추가 반영(PR-98B-3, 2026-02-20):
  - `src/backtest/gpu/logic.py`
    - `_process_new_entry_signals_gpu`에서 `(sim x candidate)` 대형 확장/`argsort` 경로 제거
    - 후보 우선순위 입력 순서를 유지한 채, 후보 축 순차 + 시뮬레이션 축 벡터화로 자본/슬롯 즉시 차감 semantics 보존
  - `src/backtest/gpu/utils.py`
    - 후보 정렬 helper를 `cp.lexsort` 기반으로 전환(`market_cap->atr->ticker`, `atr->ticker`)
  - 추가 테스트:
    - `tests/test_gpu_new_entry_signals.py`
    - `tests/test_gpu_candidate_sorting_utils.py`
  - 실행:
    - `conda run --no-capture-output -n rapids-env python -m unittest tests.test_gpu_new_entry_signals tests.test_gpu_candidate_sorting_utils`
    - `conda run --no-capture-output -n rapids-env python -m unittest tests.test_backtest_strategy_gpu`
    - `conda run --no-capture-output -n rapids-env python -m unittest tests.test_gpu_candidate_metrics_asof tests.test_gpu_candidate_payload_builder tests.test_gpu_engine_prep_path tests.test_cpu_gpu_parity_topk`
  - 결과: 모두 통과
- strict parity 재실행(실환경 GPU):
  - 명령: `python -m src.cpu_gpu_parity_topk --start-date 2026-01-05 --end-date 2026-01-09 --params-csv results/parity_single_param_issue56.csv --topk 3 --tolerance 1e-3 --parity-mode strict`
  - 증적: `results/parity_topk_strict_pr98b_20260221_082146.json`
  - 결과: `passed rows=1`, `failed=0`
- throughput(B0 대비) 측정:
  - baseline: `results/perf_baseline_strict_hyst_20260219_212900.log`
  - after(PR-98B): `results/perf_after_pr98b_20260220_063502.log`
  - wall-clock: `18688s -> 19350s` (`+662s`, `+3.54%`)
  - kernel time: `18633.54s -> 19296.47s` (`+662.93s`, `+3.56%`)
  - sims/sec: `0.01926 -> 0.01860`
- 후속 보정(PR-98B-4, 2026-02-21):
  - `src/backtest/gpu/logic.py`
    - `_process_new_entry_signals_gpu`에서 후보 루프 내부의 반복 비용 계산(`adjust_price_up`, `quantities`, `commissions`) 제거
    - 후보별 체결가/수량/총비용을 루프 외부에서 1회 벡터화 계산 후, 루프에서는 affordability/slot/cooldown 판정만 수행
  - 목적:
    - PR-98B 이후 관측된 kernel +3.56% 회귀 보정
    - 의사결정 순서/자본차감 semantics는 유지
  - 검증(로컬 단위):
    - `tests.test_gpu_new_entry_signals`
    - `tests.test_backtest_strategy_gpu`
    - `tests.test_gpu_candidate_metrics_asof`
    - `tests.test_gpu_candidate_payload_builder`
    - `tests.test_gpu_engine_prep_path`
    - `tests.test_cpu_gpu_parity_topk`
    - 결과: 모두 통과
- throughput 재측정(PR-98B-4 적용 후):
  - after(PR-98B-4): `results/perf_after_pr98b4_20260221_095532.log`
  - wall-clock(B0 대비): `18688s -> 18393s` (`-295s`, `-1.58%`)
  - kernel time(B0 대비): `18633.54s -> 18333.14s` (`-300.40s`, `-1.61%`)
  - sims/sec(B0 대비): `0.01926 -> 0.01957`
- strict parity 재검증(PR-98B-4):
  - 증적: `results/parity_topk_strict_pr98b4_20260221_150205.json`
  - 결과: `passed rows=1`, `failed=0`
- 남은 범위(PR-98B 미완료 항목):
  - 없음 (PR-98B 범위 완료)
- strict parity 실행 메모:
  - Codex 실행 샌드박스에서는 CUDA 디바이스 접근 불가(`cudaErrorOperatingSystem`)로 통합 parity 실행 불가
  - 운영 GPU 호스트에서 strict parity를 재실행해 증적 첨부 완료

## 15. PR-98C-1 진행 현황 (2026-02-21)
- 반영 범위(저위험 캐시/로딩 개선):
  - `src/optimization/gpu/data_loading.py`
    - DB 엔진 생성을 `_get_sql_engine`(LRU 캐시)로 통합해 동일 connection string에서 `create_engine` 재사용
    - `preload_all_data_to_gpu`, `preload_tier_data_to_tensor`를 공용 엔진 경로로 사용
    - 주간 후보 preload helper는 이후 strict-only cleanup에서 제거됐고, tier-only runtime hot path만 유지
  - `src/data_handler.py`
    - `load_stock_data`의 캐시 키를 날짜 정규화(`YYYY-MM-DD`) 후 private cached helper로 위임
    - 동일 범위를 문자열/`Timestamp` 등 서로 다른 타입으로 호출해도 cache hit 되도록 정규화
    - 캐시 초기화용 `clear_load_stock_data_cache` 추가
- 테스트:
  - 신규: `tests/test_gpu_data_loading_engine_cache.py`
  - 보강: `tests/test_data_handler.py` (`test_load_stock_data_cache_key_is_normalized`)
  - 실행:
    - `conda run --no-capture-output -n rapids-env python -m unittest tests.test_gpu_data_loading_engine_cache tests.test_data_handler.TestDataHandler.test_load_stock_data tests.test_data_handler.TestDataHandler.test_load_stock_data_empty tests.test_data_handler.TestDataHandler.test_load_stock_data_cache_key_is_normalized tests.test_gpu_new_entry_signals tests.test_gpu_candidate_sorting_utils tests.test_backtest_strategy_gpu tests.test_gpu_candidate_metrics_asof tests.test_gpu_candidate_payload_builder tests.test_gpu_engine_prep_path tests.test_cpu_gpu_parity_topk`
  - 결과: 통과(28 tests)
- 남은 PR-98C 범위:
  - `src/backtest/cpu/strategy.py` 일자/신호 row 캐시 도입
  - `src/backtest/cpu/portfolio.py`, `src/backtest/cpu/execution.py` lookup cache 주입
  - `src/data_handler.py` 캐시 정책(구간 단위/size) 후속 조정
  - `src/optimization/gpu/data_loading.py` pandas round-trip 최소화 검토

## 16. C1 원복 결정 및 재발방지 규칙 (2026-02-23)
- 결정:
  - `PR-98C-1` 상태를 기준선으로 복귀한다.
  - `PR-98C-2` 구간에서 throughput 회귀가 커서(최대 `+4.52%`) 현재 상태는 운영 기준으로 보류한다.
- 실측 근거(기록):
  - C2B 성능 로그: `results/perf_after_pr98c2b_20260222_131310.log`
  - `Batch 1 Kernel Execution Time: 19482.52s`
  - `Total GPU Kernel Execution Time: 19482.52s`
  - `Elapsed (wall clock): 5:25:33` (`19533s`)
  - 비교 기준(B0): `results/perf_baseline_strict_hyst_20260219_212900.log` (`18688s`)
- 실제 원복(완료):
  - `Revert "docs(issue98): record PR-98C2 memory fixes and validation"` (`1f02e35`)
  - `Revert "perf(cpu): cache strict-tier and execution lookups"` (`2104b3b`)
  - `Revert "perf(issue98): harden batch sizing and add OOM backoff"` (`1fe8cf0`)
  - `Revert "fix(gpu): cut temp mask allocations in sell/add-buy paths"` (`7252e63`)
  - 결과: 브랜치 HEAD는 C1 기준(= `c57f477`/`a3565f3` 계열)으로 복귀
- 왜 C1로 복귀했는가:
  - 정합성: C1 strict parity 증적이 이미 존재
  - 성능: C1 회귀는 소폭(`+0.60%`)이지만, C2B는 회귀 폭이 큼(`+4.52%`)
  - 운영성: C2는 OOM 안정화 장점이 있으나, 현재 throughput 손실이 큼
- 같은 반복 방지를 위한 규칙(강제):
  - Rule-1: hot path(`src/backtest/gpu/logic.py`) 수정은 기본적으로 `PC`로 취급하고, 단위테스트 통과만으로 병합하지 않는다.
  - Rule-2: OOM 완화 수정은 반드시 `exp/*` 브랜치에서 검증하고, 본선 브랜치에는 성능 게이트 통과 후에만 반영한다.
  - Rule-3: 성능 판정은 단일 run 금지. 동일 조건 2회 측정 후 `median kernel time`으로 비교한다.
  - Rule-4: 병합 조건은 아래 3개 동시 충족:
    - strict parity `mismatch=0`
    - OOM 미발생
    - C1 대비 kernel time 비열화(최소 `<= +1.0%`, 목표 `<= 0%`)
  - Rule-5: 위 조건 불충족 시 즉시 revert하고, 다음 시도는 변경 단위를 더 작게 쪼갠다(한 번에 1개 가설만).
- 다음 작업 순서:
  - 1) C1 기준 perf/parity 재확인(스냅샷 고정)
  - 2) C2 항목은 기능 단위로 분해해 실험 브랜치에서 재도입
  - 3) 항목별 gate 통과분만 순차 cherry-pick

## 17. C2 제외 잔여 개선안 + 예상 효과 (2026-02-23)
- 범위:
  - 아래 항목은 `PR-98C2`에서 revert된 변경(`5e93656`, `e0dc787`, `fe520d5`)을 재도입하지 않고 진행 가능한 후보만 포함한다.
  - 기준선은 C1(`c57f477`/`a3565f3`) + B0 측정 규칙을 따른다.
- 추정 방법:
  - 최근 실측에서 병목의 대부분이 kernel loop 구간(`~18.3k~19.5k sec`)에 집중됨
  - 예상치는 동일 조건(2024-01-01~2024-12-31, 360 sims)에서의 `kernel time` 변화 범위로 기재

### 17-1. 실행 후보(우선순위순)
| id | class | location | 변경 요지 | 예상 효과(커널) | 예상 효과(전체 wall) | 리스크 |
| --- | --- | --- | --- | --- | --- | --- |
| R-005 | PO | `tests/test_backtest_strategy_gpu.py` 외 | PR-98D 성능 회귀 가드(예산 테스트, median 2-run 규칙 자동화) | 직접 개선 없음 | 직접 개선 없음 | 낮음(운영 안전성↑) |
| R-002 | PC | `src/backtest/gpu/logic.py` | `run_lengths.tolist()` host sync 제거(CuPy-only segment prefix) | `-0.2% ~ -0.8%` | `-0.1% ~ -0.7%` | 중간(핫패스/정합성 민감) |
| R-003 | PO | `src/backtest/gpu/engine.py` | `signal_day_idx<0` 분기의 `cp.zeros` 4종을 루프 밖 1회 생성 후 재사용(P-015) | `0.0% ~ -0.3%` | `0.0% ~ -0.3%` | 낮음 |
| R-001 | PC | `src/backtest/gpu/data.py`, `src/backtest/gpu/engine.py` | as-of 후보 메트릭을 일별 `sort/drop_duplicates` 반복 대신 텐서 사전생성+gather로 전환 | `-1.0% ~ -3.0%` (stretch: `-4.0%`) | `-0.9% ~ -2.8%` | 중간(정렬/시점 parity 민감) |
| R-004 | PO | `src/optimization/gpu/analysis.py` | 결과 분석에서 불필요한 host copy/포맷팅 최소화(옵션화) | `~0.0%` | `0.0% ~ -0.3%` | 낮음(결과표시만 영향) |

### 17-2. 제외/보류 항목 (C2 범주)
- 아래는 본 섹션 범위에서 제외한다.
  - `perf(issue98): harden batch sizing and add OOM backoff` (`5e93656`)
  - `fix(gpu): cut temp mask allocations in sell/add-buy paths` (`e0dc787`)
  - `perf(cpu): cache strict-tier and execution lookups` (`fe520d5`)
- 사유:
  - C2 구간 실측에서 throughput 회귀가 커서(`+4.52%`) 현재 본선 기준으로는 재도입 보류
  - 재시도 시 `exp/*`에서 단일 가설 단위로 분리 검증 후 gate 통과분만 cherry-pick

### 17-3. 권장 실행 순서 (C2 제외)
- 1) `R-005` 먼저 반영(회귀 가드 선설치)
- 2) `R-003` 단독 반영 후 측정/게이트
- 3) `R-002` 단독 반영 후 측정/게이트
- 4) `R-001` 단독 반영 후 측정/게이트
- 5) `R-004` 마지막 반영(운영 결정 경로 비개입 항목)
- 원칙:
  - Rule-5에 따라 한 번에 1개 가설만 반영한다(묶음 반영 금지).

### 17-4. 합격 기준(동일 유지)
- strict parity `decision-level mismatch=0` (two-tier/full-13 정책은 `todos/done_2026_02_09-issue56-cpu-gpu-parity-topk.md` 11장 준수)
- OOM 미발생
- C1 대비 `kernel time <= +1.0%` (목표: `<= 0%`)
- 측정 규격:
  - warm-up 1회는 선택 사항이다
  - 판정용 run은 동일 입력 `run1`, `run2` 2회로 고정한다
  - 판정값은 `run1/run2`의 `median kernel time` 사용
  - branch hit-rate(`signal_day_idx<0`)와 host-sync 지표(`tolist()/to_pylist()`)를 함께 기록

### 17-5. 실행 체크리스트(신규 참여자용)
- [ ] Scope Lock: 이번 PR에 포함할 항목을 `R-*` 1개로 고정
- [ ] Baseline Lock: 비교 기준은 C1 + `results/perf_baseline_strict_hyst_20260219_212900.log`
- [ ] Environment Pin:
  - `conda run -n rapids-env python -V`
  - `conda run -n rapids-env python -c "import cupy,cudf; print(cupy.__version__, cudf.__version__)"`
  - `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`
- [ ] Input Snapshot:
  - `config/config.yaml` 해시, params CSV 경로, 기간, sim 수를 `results/sec17_input_<ts>.txt`에 저장
- [ ] Unit Gate: 변경 항목 연관 테스트 + 회귀 테스트 통과
- [ ] Parity Gate: strict parity 실행 후 JSON 증적 첨부(`mismatch=0`)
- [ ] Perf Gate: 동일 조건 2회 측정 후 median kernel time 비교
- [ ] Evidence Links: env/input/perf/parity 로그 경로를 PR 본문에 명시
- [ ] Rollback Trigger:
  - 조건 미충족 시 즉시 `git revert <candidate_commit_sha>`
  - 실패 로그 경로를 TODO/PR에 함께 남긴다

### 17-6. PC 항목 사전 명시(게이트 A 정합)
- `R-002` (`src/backtest/gpu/logic.py`):
  - 변경 전후 strict parity를 동일 params/date 범위로 필수 비교
  - 동점/동일 자본 상황에서 자본 차감 순서 회귀 케이스를 단위테스트로 고정
- `R-001` (`src/backtest/gpu/data.py`, `src/backtest/gpu/engine.py`):
  - GPU 후보군/정렬 변경이므로 CPU 대응 변경 필요 여부를 PR 시작 시 명시
  - CPU 코드 변경이 없다면 `정렬/시점 계약 불변 근거`를 PR 본문에 서술

## 18. 최신 정합성 재검증 + 머지 판단 (2026-02-25)
- 목적:
  - Issue #98 관련 최근 수정 상태에서 `2022-01-03 ~ 2022-09-30` 구간 strict parity 재확인
  - 문서 증적을 기준으로 `main` 머지 가능 여부를 명시
- 입력:
  - params: `results/best_param_20260224.csv` (`topk=1`)
  - mode: `candidate_source_mode=tier`
  - tolerance: `1e-3`
- strict parity(top-k) 결과:
  - 리포트: `results/parity_topk_strict_20220103_20220930_20260225_212904.json`
  - 요약: `passed rows=1`, `failed=0`
- strict sell/buy event dump 결과:
  - 리포트: `results/parity_sell_buy_dump_20220103_20220930_20260225_214329.json`
  - GPU 로그: `results/parity_sell_buy_gpu_20220103_20220930_20260225_214329.log`
  - 요약: `cpu_sell_events=72`, `gpu_sell_events=72`, `sell_mismatched_pairs=0`
  - 요약: `cpu_buy_events=163`, `gpu_buy_events=163`, `buy_mismatched_pairs=0`
- 판단:
  - Gate B(정합성)는 본 검증 범위에서 충족됨
  - Issue #98 범위는 문서/증적 기준으로 `main` 머지 진행 가능 상태
  - 단, throughput 개선 항목(`R-*`)은 별도 PR로 분리해 gate를 다시 적용

## 19. 현재 단계 측정 Runbook (2026-03-08)
### 19-1. 목적
- 현재 단계의 목적은 `PR-98C slice 2` 이후 성능 변화를 `target hardware`에서 반복 가능하게 측정하는 것이다.
- 이 runbook은 `release-grade coverage 검증`이 아니라 `throughput tracking`을 위한 것이다.
- 따라서 coverage gate drift가 성능 측정을 방해하지 않도록, canonical config는 `research-only`로 고정한다.

### 19-2. 초심자용 한 줄 설명
- 지금은 “전략이 더 좋아졌는지”를 보는 단계가 아니라, “같은 종류의 짧은 시험을 2번 돌려서 GPU가 얼마나 빨라졌는지 비교할 기준점을 만드는 단계”다.

### 19-3. canonical 측정 프로파일
- 기준 config:
  - `config/config.issue98_perf_multibatch_janfeb_2024_research020.yaml`
- 기준 기간:
  - `2024-01-01 ~ 2024-02-29`
- 기준 조합 수:
  - `360 sims`
- 기준 batch cap:
  - `simulation_batch_size=90`
  - 기대 batch 수: `4`
- coverage threshold:
  - `min_tier12_coverage_ratio=0.2`
  - 의미: `release gate`가 아니라 `throughput measurement`에 집중하기 위한 research-only 완화값이다.
- 현재 장비 기준 예상 시간:
  - 1회: 약 `25분 ~ 40분`
  - `run1 + run2`: 약 `50분 ~ 80분`
- 정리 원칙:
  - 이전 `perf_2024`, `smoke_q1`, `multibatch_2024`, `research043` config는 삭제한다.
  - 앞으로 `#98` 성능 비교는 이 canonical profile 하나로만 진행한다.
  - 과거 `results/perf_baseline_strict_hyst_20260219_212900.log`는 참고용 아카이브일 뿐, 이 짧은 창과 직접 delta 비교하지 않는다.

### 19-4. 산출물
- `results/issue98_measure/<label>_<timestamp>/env.txt`
- `results/issue98_measure/<label>_<timestamp>/input_snapshot.json`
- `results/issue98_measure/<label>_<timestamp>/run1.log`
- `results/issue98_measure/<label>_<timestamp>/run1.gpu.csv`
- `results/issue98_measure/<label>_<timestamp>/run2.log`
- `results/issue98_measure/<label>_<timestamp>/run2.gpu.csv`
- `results/issue98_measure/<label>_<timestamp>/summary.json`
- 첫 성공 측정본은 이후 slice 비교용 local baseline으로 유지한다.
- 현재 고정된 local baseline:
  - `results/issue98_measure/pr98c_slice2_janfeb2024_cov020_20260308_131130/summary.json`
  - `run1 kernel=1092.74s`, `run2 kernel=1119.44s`, `median_kernel_s=1106.09`
  - `run1 wall=18:52.13`, `run2 wall=19:20.57`, `median_wall_s=1146.35`
  - `oom_retry=false / false`

### 19-5. 실행 전 준비
- 전제:
  - `rapids-env` 사용
  - 다른 대형 GPU 작업 종료
  - `MAGICSPLIT_CONFIG_PATH`가 아래 canonical config를 가리켜야 한다
- 추천 label:
  - `pr98c_slice2_janfeb2024_cov020`
  - 이후 비교 run은 `pr98b2_janfeb2024_cov020`, `pr98b2_retry1` 같은 식으로 고정한다.

### 19-6. 복붙용 명령 세트
#### 19-6-0. 권장: `issue98_perf_measure` CLI
```bash
CONDA_NO_PLUGINS=true conda run -n rapids-env \
  python -m src.issue98_perf_measure \
  --label pr98c_slice2_janfeb2024_cov020
```

- 이 CLI는 아래 산출물을 자동으로 만든다.
  - `env.txt`
  - `input_snapshot.json`
  - `run1.log`, `run1.gpu.csv`
  - `run2.log`, `run2.gpu.csv`
  - `summary.json`
- 기본값:
  - `config_path`: `MAGICSPLIT_CONFIG_PATH` 또는 canonical config
  - `runs=2`
  - `gpu_sample_interval_sec=5`
- 예시:
```bash
CONDA_NO_PLUGINS=true conda run -n rapids-env \
  python -m src.issue98_perf_measure \
  --label pr98b2_slice2a_janfeb2024_cov020 \
  --config-path config/config.issue98_perf_multibatch_janfeb_2024_research020.yaml
```

#### 19-6-1. 수동 fallback 세팅
```bash
export MAGICSPLIT_CONFIG_PATH="config/config.issue98_perf_multibatch_janfeb_2024_research020.yaml"
export ISSUE98_LABEL="pr98c_slice2_janfeb2024_cov020"
export ISSUE98_TS="$(date +%Y%m%d_%H%M%S)"
export ISSUE98_OUTDIR="results/issue98_measure/${ISSUE98_LABEL}_${ISSUE98_TS}"
mkdir -p "$ISSUE98_OUTDIR"
```

#### 19-6-2. 환경/입력 스냅샷
```bash
{
  echo "timestamp=$ISSUE98_TS"
  echo "label=$ISSUE98_LABEL"
  echo "config_path=$MAGICSPLIT_CONFIG_PATH"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD)"
  CONDA_NO_PLUGINS=true conda run -n rapids-env python -V
  CONDA_NO_PLUGINS=true conda run -n rapids-env python -c "import cupy, cudf; print('cupy', cupy.__version__); print('cudf', cudf.__version__)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  sha256sum "$MAGICSPLIT_CONFIG_PATH"
} | tee "$ISSUE98_OUTDIR/env.txt"
```

```bash
CONDA_NO_PLUGINS=true conda run -n rapids-env python - <<'PY' > "$ISSUE98_OUTDIR/input_snapshot.json"
import hashlib
import json
import os
from pathlib import Path

import yaml

config_path = Path(os.environ["MAGICSPLIT_CONFIG_PATH"]).resolve()
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
backtest = config.get("backtest_settings", {})
strategy = config.get("strategy_params", {})

print(json.dumps({
    "config_path": str(config_path),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "start_date": backtest.get("start_date"),
    "end_date": backtest.get("end_date"),
    "initial_cash": backtest.get("initial_cash"),
    "simulation_batch_size": backtest.get("simulation_batch_size"),
    "candidate_source_mode": strategy.get("candidate_source_mode"),
    "tier_hysteresis_mode": strategy.get("tier_hysteresis_mode"),
    "price_basis": strategy.get("price_basis"),
    "min_tier12_coverage_ratio": strategy.get("min_tier12_coverage_ratio"),
}, ensure_ascii=False, indent=2))
PY
```

#### 19-6-3. 실행 함수
```bash
issue98_perf_run() {
  local run_tag="$1"
  (
    set -euo pipefail
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv,noheader -l 5 \
      > "$ISSUE98_OUTDIR/${run_tag}.gpu.csv" &
    local smi_pid=$!
    trap 'kill "$smi_pid" >/dev/null 2>&1 || true' EXIT

    /usr/bin/time -v \
      env MAGICSPLIT_CONFIG_PATH="$MAGICSPLIT_CONFIG_PATH" CONDA_NO_PLUGINS=true \
      conda run -n rapids-env python -m src.parameter_simulation_gpu \
      2>&1 | tee "$ISSUE98_OUTDIR/${run_tag}.log"

    kill "$smi_pid" >/dev/null 2>&1 || true
    wait "$smi_pid" 2>/dev/null || true
    trap - EXIT
  )
}
```

#### 19-6-4. run1 / run2
```bash
issue98_perf_run run1
issue98_perf_run run2
```

#### 19-6-5. summary 생성
```bash
cat > /tmp/issue98_make_summary.py <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

outdir = Path(sys.argv[1])

def parse_wall_to_seconds(value: str):
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return None

def parse_run_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    kernel_match = re.search(r"Total GPU Kernel Execution Time: ([0-9.]+)s", text)
    wall_match = re.search(r"Elapsed \\(wall clock\\) time .*: (.+)", text)
    batch_count = len(re.findall(r"--- Running Batch ", text))
    return {
        "path": str(path),
        "kernel_s": float(kernel_match.group(1)) if kernel_match else None,
        "wall_clock": wall_match.group(1).strip() if wall_match else None,
        "wall_clock_s": parse_wall_to_seconds(wall_match.group(1)) if wall_match else None,
        "batch_count": batch_count,
        "oom_retry": "[GPU_WARNING] OOM" in text,
    }

def parse_gpu_csv(path: Path):
    if not path.exists():
        return {"samples": 0, "gpu_util_median": None, "gpu_mem_used_max_mib": None}
    utils = []
    mem_used = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            utils.append(int(parts[1].replace(" %", "").replace("%", "").strip()))
            mem_used.append(int(parts[3].replace(" MiB", "").replace("MiB", "").strip()))
        except ValueError:
            continue
    return {
        "samples": len(utils),
        "gpu_util_median": statistics.median(utils) if utils else None,
        "gpu_mem_used_max_mib": max(mem_used) if mem_used else None,
    }

run1 = parse_run_log(outdir / "run1.log")
run2 = parse_run_log(outdir / "run2.log")
run1.update(parse_gpu_csv(outdir / "run1.gpu.csv"))
run2.update(parse_gpu_csv(outdir / "run2.gpu.csv"))

kernel_values = [value for value in [run1["kernel_s"], run2["kernel_s"]] if value is not None]
wall_values = [value for value in [run1["wall_clock_s"], run2["wall_clock_s"]] if value is not None]

summary = {
    "canonical_profile": "issue98_janfeb2024_multibatch_research020",
    "research_only": True,
    "run1": run1,
    "run2": run2,
    "median_kernel_s": statistics.median(kernel_values) if kernel_values else None,
    "median_wall_s": statistics.median(wall_values) if wall_values else None,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

CONDA_NO_PLUGINS=true conda run -n rapids-env python /tmp/issue98_make_summary.py "$ISSUE98_OUTDIR" > "$ISSUE98_OUTDIR/summary.json"

cat "$ISSUE98_OUTDIR/summary.json"
```

### 19-7. 판정 규칙
- 합격:
  - `run1/run2` 모두 exit `0`
  - `run1/run2` 모두 `oom_retry=false`
  - `run1/run2` 모두 `batch_count >= 4`
  - `summary.json` 생성 성공
- 재실행 권고:
  - `run1`과 `run2`의 `wall_clock_s` 차이가 `10%`를 넘는 경우
  - 한쪽만 비정상적으로 느리거나 `gpu_util_median`이 급락한 경우
- 기록 원칙:
  - 첫 성공 `summary.json`을 `#98`의 local baseline으로 본다.
  - 다음 slice부터는 같은 canonical profile로 다시 측정해 직접 비교한다.

### 19-7-1. 현재 고정된 baseline (2026-03-08)
- artifact:
  - `results/issue98_measure/pr98c_slice2_janfeb2024_cov020_20260308_131130/summary.json`
- 판정:
  - `run1`, `run2` 모두 exit `0`
  - `oom_retry=false / false`
  - `batch_count=4 / 4`
  - run 간 wall 편차 약 `2.5%`
- 해석:
  - 현재 canonical profile에서 `PR-98C slice 2`는 안정적으로 재현 가능하다.
  - 이 값은 이후 `PR-98B-2`의 직접 비교 기준으로 사용한다.

### 19-8. OOM stress는 별도 단계
- `1500 sims / batch 500` 같은 큰 workload는 이 runbook의 기본 단계가 아니다.
- 이유:
  - 목적이 `throughput tracking`이 아니라 `OOM/stability stress`이기 때문이다.
  - workload 자체가 커지면, 성능 변화와 workload 변화가 섞여 원인 분리가 어려워진다.
- 따라서 OOM stress는 나중의 `stability stress` 또는 `promotion 직전 soak` 단계에서 별도 config로 수행한다.

### 19-9. 다음 단계 연결
- 이 canonical baseline이 잡힌 뒤, 대형 stress로 바로 가지 않았다.
- 먼저 아래 2개를 고정했다.
  - `direct composite-rank parity fixture`
  - `multi-sim active-set rerank parity`
- 그 다음부터 `PR-98B-2` (`daily as-of ranking precompute + hot-path ranking/gather`)를 slice 단위로 진행한다.

### 19-10. PR-98B-2 slice 1 반영 (2026-03-08)
- 무엇을 했나:
  - `src/backtest/gpu/data.py`
    - candidate ranking용 `atr_14_ratio`, `flow5_mcap`, `cheap_score_effective`, `market_cap_q` tensor를 1회 생성하고 as-of forward-fill하는 helper를 추가했다.
    - 후보 subset에 대해서만 tensor gather로 small metrics frame을 만드는 경로를 추가했다.
    - `flow5_mcap NaN`이 ranking percentile을 왜곡하지 않도록 결측을 `0`으로 바꾸지 않고 유지하게 고쳤다.
    - ranking용 tensor는 `float64`를 유지하게 바꿔 CPU/legacy 경로와 rounding 경계가 어긋나지 않도록 맞췄다.
  - `src/backtest/gpu/engine.py`
    - ranking metrics 조회를 legacy cudf as-of filter 대신 `prepared_market_data["candidate_rank_tensors"]` 기반 gather로 우선 라우팅한다.
    - strict single-sim rerank 경로도 동일 helper를 사용하도록 맞췄다.
  - `src/optimization/gpu/parameter_simulation.py`
    - prepared market-data의 fixed memory 추정에 ranking tensor 4개 + retained prepared bundle footprint를 반영하도록 바꿨다.
    - reusable bundle 생성에서 GPU OOM이 나면 전체 run을 죽이지 않고 legacy per-batch preparation으로 후퇴하게 했다.
- 왜 중요한가:
  - 이전에는 거래일마다 후보군에 대해 cudf filter/sort/drop-duplicates를 다시 하면서 host-side 비용이 누적됐다.
  - 지금은 “미리 만든 tensor에서 필요한 후보만 꺼내는” 쪽으로 바뀌었기 때문에, 다음 slice에서 더 큰 hot-path 축소를 넣기 위한 바닥 공사가 끝났다.
- 이번 slice에서 고정한 테스트:
  - `tests/test_gpu_candidate_payload_builder.py::test_direct_composite_rank_parity_fixture_matches_cpu_history`
  - `tests/test_gpu_new_entry_signals.py::test_multi_sim_active_set_rerank_matches_python_reference`
  - `tests/test_gpu_candidate_metrics_asof.py::test_tensor_gather_matches_legacy_asof_ranking_payload`
  - `tests/test_gpu_candidate_metrics_asof.py::test_tensor_gather_preserves_nan_flow_ranking_semantics`
  - `tests/test_gpu_engine_prep_path.py::test_prepared_rank_tensors_bypass_legacy_asof_lookup`
  - `tests/test_gpu_parameter_simulation_orchestration.py::test_find_optimal_parameters_uses_tier_only_runtime_inputs`
- 검증:
  - `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_gpu_candidate_payload_builder tests.test_gpu_new_entry_signals tests.test_parity_sell_event_dump tests.test_cpu_candidate_priority tests.test_cpu_strategy_entry_context tests.test_gpu_candidate_metrics_asof tests.test_gpu_engine_prep_path tests.test_gpu_parameter_simulation_orchestration -v`
  - 결과: `57 tests OK`
- 현재 판단:
  - `fixture 2개 대기` 상태는 끝났다.
  - 이제 `PR-98B-2` 다음 slice에서 remaining hot-path reduction을 넣고, canonical baseline(`summary.json`)과 직접 비교하면 된다.

### 19-11. PR-98B-2 slice 2a 반영 (2026-03-08)
- 무엇을 했나:
  - `src/backtest/gpu/logic.py`
    - `_process_additional_buy_signals_gpu`에서 `run_lengths.tolist()` 기반 host sync를 제거했다.
    - `cp.unique(..., return_index=True)`로 얻은 `sim_start_indices`에 `cp.searchsorted`를 적용해, 각 candidate의 per-sim rank를 device 쪽에서 직접 계산하도록 바꿨다.
  - `tests/test_backtest_strategy_gpu.py`
    - 다중 시뮬레이션/다중 후보에서 추가매수 rank 처리와 순차 자본 차감 semantics가 유지되는 회귀 테스트를 추가했다.
- 왜 중요한가:
  - 현재 canonical baseline은 GPU kernel 자체보다 host orchestration 비중이 아직 높다.
  - 따라서 큰 `new-entry` 의미론 변경 전에, blast radius가 가장 작은 host sync 제거부터 넣는 것이 더 안전하다.
- 검증:
  - `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_backtest_strategy_gpu tests.test_gpu_new_entry_signals -v`
  - `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_parity_sell_event_dump -v`
  - 결과: `16 tests OK`, `17 tests OK`
- 다음 판단:
  - 지금의 다음 우선순위는 더 큰 구현이 아니라 canonical remeasure다.
  - `slice 2a`를 같은 Jan-Feb profile로 다시 재고, 개선폭이 충분하면 다음 execution-loop hot path로 간다.
  - 개선폭이 작으면 `PR-98D` perf regression guard를 먼저 여는 것이 더 합리적이다.

### 19-12. PR-98B-2 slice 2a canonical remeasure 결과 (2026-03-08)
- artifact:
  - baseline: `results/issue98_measure/pr98c_slice2_janfeb2024_cov020_20260308_131130/summary.json`
  - remeasure: `results/issue98_measure/pr98b2_slice2a_janfeb2024_cov020_20260308_152446/summary.json`
  - limitation: `slice2a` remeasure 디렉터리에는 현재 `summary.json`만 있고 `env.txt`/`input_snapshot.json`이 없다.
- run health:
  - baseline/remeasure 모두 `run1/run2 exit=0`
  - baseline/remeasure 모두 `batch_count=4/4`, `oom_retry=false/false`
- median 비교:
  - `median_kernel_s`: `1106.09s -> 1115.86s` (`+9.77s`, `-0.88%`)
  - `median_wall_s`: `1146.35s -> 1161.25s` (`+14.90s`, `-1.30%`)
- 판정:
  - 이번 `slice 2a`는 canonical 기준에서 throughput 개선이 확인되지 않았다.
  - 따라서 이 시점의 1차 판정은 `baseline 미승격 + PR-98D 우선`이다.
  - rollback 여부는 별도 canonical remeasure(`19-13`, `19-14`)로 최종 결정한다.

### 19-13. 롤백 실험 및 재측정 필요성 (2026-03-08)
- 실험 목적:
  - `slice 2a`를 뺀 `slice 1 only` 상태가 정말 더 나은지 canonical 2-run으로 확인한다.
- 왜 재측정이 필요한가:
  - 현재 고정 baseline artifact(`pr98c_slice2_janfeb2024_cov020_20260308_131130`)의 `git_head`는 `b4204bb`다.
  - rollback 재측정 artifact(`pr98b2_slice1only_janfeb2024_cov020_20260308_162509`)의 실제 `git_head`는 `5673cfe`다.
  - 즉 baseline과 rollback remeasure는 서로 다른 코드 상태다.
  - 따라서 과거 baseline 숫자를 그대로 재사용하면 안 되고, `slice 1 only` 상태에서 canonical 2-run을 새로 찍어야 공정 비교가 된다.
- 운영 규칙:
  - rollback 여부는 추정이 아니라 canonical 재측정 결과로 결정한다.

### 19-14. `slice 1 only` rollback canonical remeasure 결과 (2026-03-08)
- artifact:
  - rollback remeasure: `results/issue98_measure/pr98b2_slice1only_janfeb2024_cov020_20260308_162509/summary.json`
- run health:
  - `run1/run2 exit=0`
  - `batch_count=4/4`, `oom_retry=false/false`
- median 비교:
  - baseline(`pr98c_slice2`):
    - `median_kernel_s`: `1106.09s -> 1185.30s` (`+79.21s`, `-7.16%`)
    - `median_wall_s`: `1146.35s -> 1227.15s` (`+80.80s`, `-7.05%`)
  - `slice2a` remeasure:
    - `median_kernel_s`: `1115.86s -> 1185.30s` (`+69.44s`, `-6.22%`)
    - `median_wall_s`: `1161.25s -> 1227.15s` (`+65.90s`, `-5.68%`)
- 해석:
  - “원복했는데 재측정이 꼭 필요한가?”에 대한 실측 답은 “필요하다”이다.
  - 실제로 rollback 상태가 기존 기준보다 큰 폭으로 느려졌고, `slice2a`보다도 더 나빴다.
  - 즉, baseline 재사용만으로는 판단이 틀릴 수 있으며, 코드 상태별 canonical 재측정이 필요하다.
  - 따라서 현재 운영 후보는 rollback이 아니라 `slice2a`를 유지하는 편이 더 낫다.
- authoritative decision:
  - 현재 작업 트리는 `slice2a`를 유지한다.
  - 다만 `slice2a`는 baseline 대비 throughput win이 확인되지 않았으므로 아직 승격/병합 완료 상태는 아니다.
  - `PR-98D`는 actual artifact gate로 유지하고, 다음 승격 판단은 `large-batch / long-window + strict parity` 증적까지 모은 뒤 다시 내린다.

### 19-15. `issue98_perf_measure` candidate smoke 확인 (2026-03-08)
- 목적:
  - 새 측정 CLI가 실제 GPU 실행에서도 `env.txt` / `input_snapshot.json` / `run*.log` / `run*.gpu.csv` / `summary.json`을 한 세트로 남기는지 확인한다.
- artifact:
  - smoke run: `results/issue98_measure/pr98_smoke_candidate_janfeb2024_cov020_20260308_174843/summary.json`
- 결과:
  - 산출물: `env.txt`, `input_snapshot.json`, `run1.log`, `run1.gpu.csv`, `summary.json` 생성 확인
  - `run1 exit_code=0`
  - `kernel_s=1208.90s`
  - `wall_clock_s=1247.33s`
  - `batch_count=4`
  - `oom_retry=false`
- 해석:
  - smoke 목적이었던 “artifact 형식 검증”은 통과했다.
  - 이 수치는 1-run smoke이므로 baseline delta 판단이나 승격 근거로 사용하지 않는다.
- 다음 작업:
  - large-workload는 기존 canonical gate와 분리된 탐색 트랙으로 취급한다.
  - 초기 권장은 `long-window only` 또는 `large-batch only` 중 한 축만 바꾼 새 perf config를 만든 뒤 candidate 1-run smoke를 먼저 보는 것이었다.
  - 이후 사용자 결정으로 `long-window + large-batch combo`를 먼저 수행하는 research+screening lane을 채택했다.
  - combo config 초안: `config/config.issue98_perf_combo_2017_2021_research044.yaml`
  - 이 combo lane은 실제 투자 후보 파라미터 mining과 throughput screening을 겸하지만, canonical perf gate 승격 근거로 직접 사용하지 않는다.
  - strict parity는 existing parity config(`config.parity_research_043.yaml` 우선, 필요 시 `044`)로 별도 재검증한다.

### 19-16. long-window + large-batch combo mining report (2026-03-10)
- 목적:
  - `long-window + large-batch` candidate combo run을 performance stress + parameter mining lane으로 사용하고, 결과 CSV를 재현 가능한 report artifact로 고정한다.
- source artifacts:
  - combo measure summary: `results/issue98_measure/pr98_slice2a_combo_2017_2021_cov020_20260308_183948/summary.json`
  - standalone results CSV: `results/standalone_simulation_results_20260310_003636.csv`
  - generated mining report JSON: `results/issue98_measure/pr98_slice2a_combo_2017_2021_cov020_20260308_183948/mining_report/combo_mining_report.json`
  - generated mining report Markdown: `results/issue98_measure/pr98_slice2a_combo_2017_2021_cov020_20260308_183948/mining_report/combo_mining_report.md`
  - report generator: `python -m src.analysis.issue98_combo_mining_report`
- run health:
  - `run1 exit_code=0`
  - `median_kernel_s=93515.12s` (`25.98h`)
  - `median_wall_s=107808.0s` (`29.95h`)
  - `batch_count=31`
  - `oom_retry=true`
  - retry trace:
    - batch 1: `1200 -> 600`
    - batch 4: `600 -> 300 -> 150 -> 75`
- 판정:
  - 이 run은 `candidate combo workload` 완주에는 성공했지만, 공격적 cap(`1200`)이 실제 안정 batch가 아님이 확인됐다.
  - 따라서 본 artifact는 `perf proof`가 아니라 `research mining artifact`로 채택한다.
  - same-workload baseline 대비 throughput 입증은 여전히 별도 baseline combo run이 필요하다.
- report 재생성 명령:
  - `python -m src.analysis.issue98_combo_mining_report --csv-path results/standalone_simulation_results_20260310_003636.csv --report-dir results/issue98_measure/pr98_slice2a_combo_2017_2021_cov020_20260308_183948/mining_report`
- metric distribution (`calmar_ratio` 기준):
  - 전체 `3600` row
  - `calmar_ratio`: `min=-0.0483`, `p25=0.1249`, `median=0.1736`, `p75=0.2339`, `max=0.5106`
  - `cagr`: `min=-0.0286`, `median=0.0926`, `max=0.2161`
  - `mdd`: `min=-0.6468`, `median=-0.5327`, `max=-0.3804`
- main-effect 요약 (`mean calmar spread` 기준):
  - strongest:
    - `order_investment_ratio`: spread `0.1172` (`0.0350` best, `0.0150` worst)
    - `max_inactivity_period`: spread `0.0892` (`126` best, `504` worst)
    - `sell_profit_rate`: spread `0.0739` (`0.2500` best mean, `0.1600` worst mean)
  - mid:
    - `additional_buy_priority`: spread `0.0421` (`0` > `1`)
    - `additional_buy_drop_rate`: spread `0.0413` (`0.0500` best mean, `0.0900` worst mean)
  - weak:
    - `stop_loss_rate`: spread `0.0241` (`-0.9000` > `-0.6000`)
    - `max_splits_limit`: spread `0.0014` (`10` ~= `15`)
- top 5% concentration:
  - `order_investment_ratio`: `0.0350`가 `41.1%`, `0.0300`까지 합치면 `73.3%`
  - `max_inactivity_period`: `126`이 `85.0%`
  - `additional_buy_priority`: `0`이 `72.2%`
  - `additional_buy_drop_rate`: `0.05~0.07` 구간이 `87.2%`
  - `sell_profit_rate`: `0.25`가 최빈이지만 `0.10`도 `25.0%`로 강한 보조 peak를 보인다.
  - `stop_loss_rate`는 평균 기준으로는 `-0.9000`이 낫지만, top 5% 빈도는 `-0.6000`이 `55.0%`로 더 높다. 즉 단일 mean effect만으로 닫기보다 interaction 후보로 남겨야 한다.
- immediate shortlist for parity canary:
  - `A`: `order_investment_ratio=0.0350`, `additional_buy_drop_rate=0.0500`, `sell_profit_rate=0.2500`, `additional_buy_priority=0`, `stop_loss_rate=-0.9000`, `max_splits_limit=10`, `max_inactivity_period=252`
  - `B`: `order_investment_ratio=0.0250`, `additional_buy_drop_rate=0.0500`, `sell_profit_rate=0.2500`, `additional_buy_priority=0`, `stop_loss_rate=-0.9000`, `max_splits_limit=10`, `max_inactivity_period=252`
  - `C`: `order_investment_ratio=0.0300`, `additional_buy_drop_rate=0.0700`, `sell_profit_rate=0.1000`, `additional_buy_priority=0`, `stop_loss_rate=-0.9000`, `max_splits_limit=10`, `max_inactivity_period=126`
  - `D`: `order_investment_ratio=0.0300`, `additional_buy_drop_rate=0.0500`, `sell_profit_rate=0.1900`, `additional_buy_priority=0`, `stop_loss_rate=-0.6000`, `max_splits_limit=10`, `max_inactivity_period=126`
- why these four:
  - `A/B`는 top-calmar plateau를 대표한다.
  - `C`는 `126-day inactivity` + `0.10 sell` 쪽 strong local mode를 대표한다.
  - `D`는 top-5% 빈도에서 살아남은 `-0.6000` challenger를 남겨 mean-effect vs interaction tension을 검증하기 위한 후보다.
- continuation rule:
  - future combo CSV가 생기면 동일 스크립트로 `combo_mining_report.{json,md}`를 다시 생성한다.
  - report 해석은 `mining / sensitivity / parity shortlist` 용도로만 사용하고, `Issue #98 perf gate` 판단은 별도 baseline combo summary와 함께 내린다.
