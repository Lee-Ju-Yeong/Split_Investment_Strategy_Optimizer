# Issue #104: GPU Throughput Follow-up Hot Path

> Type: `implementation`
> Status: `done`
> Priority: `P1`
> Last updated: 2026-03-13
> Related issues: `#104`, `#98`, `#56`
> Gate status: `issue opened on GitHub; follow-up tranche created from #98 handoff; H-001 was implemented/tested/measured, then rolled back after canonical 4-run regression confirmation; H-003 was re-scoped onto the canonical live path, tested, measured, and then rolled back after canonical regression confirmation; H-004-a was also implemented/tested/measured, then rolled back after canonical regression confirmation; H-005-a / H-005-b-0 probes narrowed the true bottleneck to additional_buy last-trade dedup; H-005-b replaced cp.unique-based dedup with direct idempotent update; final canonical 2-run and strict parity both passed`

## 1. One-Page Summary
- What: `#98`에서 일부러 미뤄둔 더 공격적인 GPU hot-path 최적화를 별도 tranche로 진행하는 문서입니다.
- Why: `#98`은 current HEAD 기준 canonical 성능 개선과 strict parity 재확인까지 마치고 닫았습니다. 이제는 완료 문서를 오염시키지 않고, 다음 최적화만 분리해서 추적해야 합니다.
- Current status: GitHub issue `#104` 구현 tranche는 완료됐습니다. `H-001`, `H-003`, `H-004-a`는 모두 canonical 회귀로 롤백했고, `H-005-a` / `H-005-b-0` probe로 실제 병목을 additional_buy last-trade dedup으로 좁힌 뒤 `H-005-b` direct-update slice를 구현했습니다. 그 결과 current HEAD canonical 2-run과 strict parity를 모두 통과했습니다.
- Next action: follow-up backlog가 새로 생기기 전까지는 이 tranche를 다시 열지 않습니다. probe instrumentation은 `--kernel-breakdown` gated tool로 유지하고, 다음 hot-path tranche가 필요해질 때 재사용합니다.

## 2. 초심자용 현재 판단
### 2-1. 왜 새 문서로 시작하나
- `#98` 문서는 이미 `done`입니다.
- 그 문서는 “무엇을 했고, 어떤 증거로 닫았는가”를 보관하는 완료 기록입니다.
- 이번 follow-up은 새로운 실험과 새로운 측정이 들어가므로, 기존 완료 문서와 섞지 않는 편이 훨씬 이해하기 쉽습니다.

### 2-2. 이번 이슈에서 하고 싶은 일은 무엇인가
- `new-entry` 쪽에서 남아 있는 반복 스캔 비용을 더 줄입니다.
- `cp.lexsort` 기반 후보 정렬 비용을 줄일 수 있는지 검토합니다.
- 가능하면 CPU I/O / cache / engine reuse 계열도 `PO`로 분리해 성능 여지를 다시 확인합니다.

### 2-3. 무엇을 조심해야 하나
- CPU 백테스터가 정답(`SSOT`)입니다.
- 빨라져도 CPU와 결과가 달라지면 실패입니다.
- 그래서 이번 이슈도 매 slice마다 아래 두 질문을 따로 확인합니다.
  - 정말 빨라졌는가?
  - 결과 의미가 그대로인가?

## 3. Scope
- `P-008` follow-up: `new-entry` hot path 추가 pruning / vectorization
- `P-009` follow-up: GPU-side candidate ordering / sorting 비용 축소
- current HEAD 기준 canonical throughput 재측정
- parity-coupled slice의 strict parity 재검증

## 4. Non-Goals
- 전략 의미론 변경
- strict-only governance 정책 변경
- `#68` WFO / OOS 정책 확정
- `#98 done` 문서의 증적 구조 재작성

## 5. Guardrails
- CPU backtester is SSOT
- `candidate_source_mode='tier'` 유지
- `strict_pit` runtime governance 유지
- `PC` 변경은 strict parity 재검증 없이 승격하지 않음
- canonical lane과 combo lane의 의미를 섞지 않음

## 6. Current Plan
- [x] hot-path backlog를 `PO / PC`로 다시 분류
- [x] 첫 safe slice 선택
- [x] 코드 수정 + 회귀 테스트 추가
- [x] current HEAD 대비 canonical throughput 재측정
- [x] 다음 slice로 진행할지 / 이 tranche를 닫을지 판정
- [x] `H-001` rollback
- [x] `H-003` slice contract 고정 및 1차 구현 시도
- [x] `H-003` 1차 구현 rollback
- [x] canonical/live path (`data.py` / `engine.py`) 기준 재설계
- [x] live-path slice 구현 + engine-level regression test 추가
- [x] `H-003` canonical throughput 재측정
- [x] `H-003` rollback
- [x] `H-004` slice contract 고정
- [x] `H-004` 구현
- [x] `H-004` canonical 측정
- [x] `H-004` rollback

## 7. Follow-up Backlog Reclassification
### 7-1. 이번 문서에서 다시 고정한 규칙
- `PO (Perf-Only)`: 결과를 바꾸지 않는 최적화입니다. 캐시, workspace 재사용, 로더/세션 재사용처럼 의미론을 건드리지 않는 항목입니다.
- `PC (Parity-Coupled)`: 후보 순서, 매수 가능 여부, 자본 차감 타이밍처럼 CPU/GPU 정합성에 직접 닿는 항목입니다.
- 주의: `src/backtest/gpu/logic.py`의 `new-entry` 경로는 수학적으로 결과가 같아 보여도, 현재 프로젝트 규칙상 기본적으로 `PC`로 취급합니다.

### 7-2. 현재 backlog 재분류
| ID | Class | Area | What | Why it matters | Note |
| --- | --- | --- | --- | --- | --- |
| `H-001` | `PC` | `src/backtest/gpu/logic.py` | `new-entry`에서 전체 `quantities_matrix / total_costs_matrix`를 먼저 만들지 않고, 후보 `k`별로 active simulation subset만 계산 | 현재는 `(num_sims x num_candidates)` 전체 비용 행렬을 먼저 만들고 있어서, 실제로는 곧 inactive 될 simulation까지 upfront 비용을 낸다 | first safe slice 후보 |
| `H-002` | `PC` | `src/backtest/gpu/logic.py` | `new-entry` 순차 루프를 block 단위 또는 prefix 선택 방식으로 더 줄이기 | 후보 수가 많을수록 순차 스캔 비용이 커진다 | blast radius 큼, `H-001` 뒤로 미룸 |
| `H-003` | `PC` | `src/backtest/gpu/data.py`, `src/backtest/gpu/engine.py` | canonical/live path의 candidate payload 생성 구간에서 non-debug ticker string materialization을 피하고 numeric `ticker_rank` tie-break로 정렬 유지 | 현재 실제 runtime은 `logic.py`가 아니라 `data.py -> engine.py` 경로를 사용한다 | measured no-go; rolled back |
| `H-004` | `PO` | CPU loader / cache | CPU I/O / session cache / engine reuse 계열 정리 | GPU만 빠라도 CPU 보조 경로가 느리면 전체 반복 속도가 늘어진다 | next recommended slice |

### 7-3. 첫 safe slice 선정
- 선택: `H-001`
- 왜 이걸 먼저 하냐:
  - 전면 벡터화보다 범위가 작습니다.
  - 후보 순서는 그대로 둡니다.
  - 자본 차감 순서도 그대로 둡니다.
  - 바뀌는 것은 “언제 계산하느냐”입니다.
- 쉬운 말로 하면:
  - 지금은 모든 후보의 비용표를 한꺼번에 미리 만들고 있습니다.
  - 다음 slice는 “지금 실제로 검사 중인 후보”의 비용만 그때그때 계산해서, 쓸모없는 큰 표를 줄이는 방향입니다.
- 분류는 왜 `PC`냐:
  - 계산 시점을 바꾸더라도 결국 `매수 가능 여부`와 `commission 포함 total_cost` 판단 경로를 건드립니다.
  - 따라서 이번 프로젝트 규칙상 strict parity 재검증이 필요합니다.

## 8. First Slice Contract (`H-001`)
### 8-1. What
- `_process_new_entry_signals_gpu`에서 아래 full matrix upfront materialization을 제거 대상으로 봅니다.
  - `quantities_matrix`
  - `costs_matrix`
  - `commissions_matrix`
  - `total_costs_matrix`
- 대신 후보 `k` 루프 안에서 `active_sim_indices` 기준으로만 수량/비용을 계산합니다.

### 8-2. What must stay the same
- 후보 우선순위 순회 순서
- `cash < order_budget`일 때 더 이상 다음 후보를 보지 않는 contract
- `cash >= order_budget`이지만 `commission` 때문에 첫 후보를 못 사면, 더 싼 다음 후보는 계속 보는 contract
- cooldown / holding skip contract
- `temp_capital`과 `temp_available_slots`의 즉시 차감 contract

### 8-3. Verification plan
- 기존 `tests/test_gpu_new_entry_signals.py` 회귀셋 유지
- 특히 아래 케이스를 다시 핵심 증적으로 본다
  - multi-sim active set rerank parity
  - budget / slot exhaustion parity
  - commission-aware skip-then-buy parity
  - heterogeneous order budget + commission + active-set shrink parity
- 구현 후:
  - `tests.test_gpu_new_entry_signals`
  - `tests.test_backtest_strategy_gpu`
  - current HEAD canonical throughput remeasure
  - strict parity canary 재확인

## 9. References
- GitHub issue: `#104`
- GitHub issue URL: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/104`
- previous tranche: `todos/2026_02_17-issue98-gpu-throughput-refactor.md`
- current branch: `feature/issue98-followup-hotpath`

## 10. Working Log
- 2026-03-13: GitHub issue `#104` 생성. `#98` 완료 이후 follow-up hot-path tranche를 별도 문서로 분리 시작.
- 2026-03-13: follow-up backlog를 `PO / PC`로 다시 분류. 첫 safe slice는 `H-001`(`new-entry` full matrix upfront materialization 제거)로 선택.
- 2026-03-13: `H-001` 구현. `_process_new_entry_signals_gpu`에서 full upfront matrix(`quantities/costs/commissions/total_costs`)를 제거하고, 후보 `k`별 `eligible active subset` 기준 lazy 계산으로 전환.
- 2026-03-13: review gap 보강. `heterogeneous order budget + commission + active-set shrink` 케이스 회귀 테스트 추가.
- 2026-03-13: 회귀 검증 실행. `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_gpu_new_entry_signals tests.test_backtest_strategy_gpu -v` -> `27 tests OK`.
- 2026-03-13: canonical `Jan-Feb 2024` 2-run 재측정(`pr104_h001_janfeb2024_cov020_20260313_071910`) 결과, current-head baseline(`pr98_head_final_janfeb2024_cov020_20260312_223231`) 대비 성능 악화 확인.
  - `median_kernel_s`: `1011.80s -> 1095.26s` (`-8.25%`)
  - `median_wall_s`: `1049.52s -> 1138.31s` (`-8.46%`)
  - `oom_retry`: 둘 다 `false/false`, `batch_count`: 둘 다 `4/4`
  - 배치별 관찰: run2에서 `Batch 1`이 `712.13s -> 846.14s`로 크게 증가해 전체 회귀를 주도.
- 2026-03-13: 옵션 B(추가 canonical 2-run) 재판정 실행(`pr104_h001_janfeb2024_cov020_rerun_20260313_104228`).
  - 추가 2-run만 보면:
    - `median_kernel_s = 1022.13s`
    - `median_wall_s = 1058.25s`
  - 기존 2-run + 추가 2-run 합산(총 4-run median):
    - `median_kernel_s = 1033.315s` (`vs base 1011.795s`, `-2.13%`)
    - `median_wall_s = 1073.87s` (`vs base 1049.52s`, `-2.32%`)
  - 결론: 변동성 재판정 후에도 throughput 회귀가 남아서 `H-001` 승격은 계속 보류.
- 2026-03-13: `H-001` 코드 rollback 완료. `src/backtest/gpu/logic.py`는 baseline 계산 경로로 복귀.
- 2026-03-13: rollback 후 회귀 검증 재실행. `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_gpu_new_entry_signals tests.test_backtest_strategy_gpu -v` -> `27 tests OK`.
- 2026-03-13: `H-003` 1차 시도. `logic.py` legacy runner에 GPU argsort helper를 연결하는 방향으로 구현하고 타깃 회귀 검증을 통과시켰다.
- 2026-03-13: Codex blind-first + Gemini 사후 대조 결과, canonical runtime/perf path는 `logic.py`가 아니라 `data.py -> engine.py`를 사용한다는 점이 확인됐다. 따라서 1차 H-003은 live path에 닿지 않는 dead-code optimization으로 판정.
- 2026-03-13: `H-003` 1차 구현 rollback 완료. 다음 작업은 `build_ranked_candidate_payload(...)`와 그 호출부를 기준으로 실제 병목을 다시 잡는 것으로 재설계.
- 2026-03-13: `H-003` live-path 재설계 구현. `create_candidate_rank_tensors(...)`에 `ticker_rank` tensor를 추가하고, `collect_candidate_rank_metrics_from_tensors(...)`가 non-debug 경로에서는 ticker 문자열 대신 numeric `ticker_rank`를 싣도록 변경. `build_ranked_candidate_payload(...)`는 `ticker_rank`가 있으면 그걸 tie-break로 사용하고, debug 기록이 필요할 때만 `ticker` 문자열을 요구하도록 조정.
- 2026-03-13: engine helper `_collect_candidate_rank_metrics(...)`가 `include_ticker_strings=debug_mode`를 전달하도록 수정. 즉, 실제 canonical non-debug lane에서 `cp.asnumpy(filtered_indices).tolist()` 기반 ticker 문자열 materialization을 건너뛴다.
- 2026-03-13: live-path 회귀 검증 실행. `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_gpu_candidate_metrics_asof tests.test_gpu_candidate_payload_builder tests.test_backtest_strategy_gpu tests.test_issue67_tier_universe.TestIssue67GpuParityHelpers -v` -> `36 tests OK`.
- 2026-03-13: follow-up review에서 나온 backward-compat gap 보강. legacy/manual `prepared_market_data`에 `ticker_rank`가 없으면 tensor gather 시점에 1회 backfill 후 캐시하도록 수정.
- 2026-03-13: exact tie-case 테스트 추가. non-debug no-string 경로에서도 `entry_composite_score_q`와 `market_cap_q`가 동률일 때 `ticker_rank`가 마지막 tie-break로 동작하는지 직접 고정.
- 2026-03-13: compat + tie-case 포함 회귀 검증 실행. `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_gpu_candidate_metrics_asof tests.test_gpu_candidate_payload_builder tests.test_gpu_engine_prep_path tests.test_backtest_strategy_gpu tests.test_issue67_tier_universe.TestIssue67GpuParityHelpers -v` -> `48 tests OK`.
- 2026-03-13: canonical `Jan-Feb 2024` 2-run 재측정(`pr104_h003_janfeb2024_cov020_20260313_134103`) 결과, current-head baseline(`pr98_head_final_janfeb2024_cov020_20260312_223231`) 대비 성능 악화 확인.
  - `median_kernel_s`: `1011.80s -> 1107.565s` (`-9.47%`)
  - `median_wall_s`: `1049.52s -> 1148.33s` (`-9.41%`)
  - `oom_retry`: 둘 다 `false/false`, `batch_count`: 둘 다 `4/4`
- 2026-03-13: 해석 정리. `H-003`은 문자열 materialization을 줄이려던 의도는 맞았지만, canonical lane에서는 candidate rank tensor 준비 + cuDF payload/sort 경로 비용이 더 커서 총 throughput이 악화된 것으로 판단.
- 2026-03-13: `H-003` 코드 rollback 완료. 이번 tranche의 다음 우선순위는 parity 리스크가 낮은 `H-004`(`PO`)로 재설정.
- 2026-03-13: `H-004-a` 구현. `find_optimal_parameters(...)`에서 이미 로드한 `all_data_gpu`의 multi-index를 사용해 `trading_dates_pd`, `all_tickers`를 직접 구성하도록 변경. 같은 run 안에서 `create_engine(...) + pd.read_sql(...)`로 거래일 목록을 다시 읽는 CPU-side 중복 준비를 제거.
- 2026-03-13: `H-004-a` 회귀 검증 실행. `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_gpu_parameter_simulation_orchestration tests.test_gpu_parameter_batch_fallback -v` -> `19 tests OK`.
- 2026-03-13: canonical `Jan-Feb 2024` 2-run 재측정(`pr104_h004a_janfeb2024_cov020`) 결과, current-head baseline(`pr98_head_final_janfeb2024_cov020_20260312_223231`) 대비 성능 악화 확인.
  - `median_kernel_s`: `1011.80s -> 1091.59s` (`-7.89%`)
  - `median_wall_s`: `1049.52s -> 1121.56s` (`-6.86%`)
  - `oom_retry`: 둘 다 `false/false`, `batch_count`: 둘 다 `4/4`
- 2026-03-13: 해석 정리. `H-004-a`는 “DB 재조회 제거” 아이디어 자체는 맞았지만, canonical lane에서는 제거한 SQL 비용보다 `all_data_gpu.index`에서 날짜/티커 축을 다시 뽑아 `to_pandas()/sorted(...)`로 CPU 객체를 재구성하는 비용이 더 컸을 가능성이 높다고 판단.
- 2026-03-13: `H-004-a` 코드 rollback 완료. 이번 tranche는 세 slice 연속 canonical no-go가 나온 상태라, 다음 구현 전엔 prep/wall breakdown 기준으로 병목을 다시 좁혀서 재우선순위화하기로 함.
- 2026-03-13: breakdown 수집 경로 추가. `issue98_perf_measure`가 기존 로그에서 `all_data_load / tier_load / pit_mask_load / prepared_bundle / wide_tensor_build / analysis` 시간을 파싱해 `summary.json`의 `stage_breakdown_s`, `median_stage_breakdown_s`에 같이 남기도록 확장.
- 2026-03-13: `prepare_market_data_bundle(...)` 성공/실패 시간도 로그에 직접 남기도록 계측 추가. 다음부터는 “prep이 느린지, kernel이 느린지, analysis가 느린지”를 summary만 보고도 빠르게 구분할 수 있음.
- 2026-03-13: kernel breakdown probe 추가. `MAGICSPLIT_KERNEL_BREAKDOWN=1` 또는 `issue98_perf_measure --kernel-breakdown`일 때 `sell / candidate_select / candidate_payload / strict_rerank / new_entry / additional_buy / valuation` 누적 시간을 `[GPU_KERNEL_BREAKDOWN]` 한 줄로 남기고, 이를 `summary.json`의 `kernel_stage_breakdown_s`, `median_kernel_stage_breakdown_s`로 함께 집계하도록 확장.

## 11. H-001 Interim Decision
- 상태: `No-Go for promotion`
- 이유: 메모리 피크 이점은 있었지만 canonical throughput은 유의미하게 악화되었습니다.
- 해석: active set이 큰 초기 구간에서 후보별 lazy 계산의 gather/임시배열 비용이 상쇄 이상으로 커졌을 가능성이 큽니다.
- 후속 권장:
  - `Option A (recommended)`: `H-001` 코드 rollback 후 다음 slice(`H-003` 또는 `H-002`)로 이동
  - `Option B`: 동일 코드로 canonical 2-run 추가 측정 후(총 4-run) 변동성 재판정

## 12. H-001 Final Decision
- 최종 상태: `Rolled back`
- 이유: 총 4-run median 기준으로도 baseline 대비 회귀가 남았습니다.
  - `kernel`: `-2.13%`
  - `wall`: `-2.32%`
- 유지한 것:
  - `new-entry` parity 회귀 테스트 보강
  - `memory-lean` 아이디어와 측정 기록
- 되돌린 것:
  - `_process_new_entry_signals_gpu`의 lazy subset 계산 경로

## 13. Next Slice Selection
- 다음 slice: `H-003`
- 이유:
  - `H-002`보다 blast radius가 작습니다.
  - 다만 이제는 `logic.py`가 아니라 실제 live path인 `data.py` / `engine.py`를 대상으로 다시 잡아야 합니다.
  - `new-entry` 계산 계약을 흔들지 않고 candidate payload 생성 비용을 줄이는 쪽으로 좁힙니다.
- 초심자용 한 줄:
  - 이번엔 “정렬 helper를 예쁘게 바꾸는 것”보다, 실제로 성능을 재는 경로가 어디인지 먼저 맞추고 그 경로만 최적화합니다.

## 14. H-003 Current Status
- 상태: `Rolled back after measurement`
- 무엇을 바꿨나:
  - non-debug tensor gather 경로에서 ticker 문자열을 바로 만들지 않도록 바꿨습니다.
  - 대신 `ticker_rank` 숫자 tie-break를 만들고, 실제 payload 정렬은 그 숫자 키로 유지합니다.
  - debug 기록이 필요할 때만 ticker 문자열을 남깁니다.
- 측정 결과:
  - `kernel`: `-9.47%`
  - `wall`: `-9.41%`
- 왜 멈췄나:
  - 문자열 생성 비용을 줄인 것보다 candidate rank tensor 준비와 payload 정렬 비용이 더 크게 붙었습니다.
  - canonical 기준에서는 “조금 나쁨”이 아니라 “꽤 나쁨” 쪽이라 기본 경로 승격 근거가 부족합니다.
- 최종 판단:
  - 기본 경로에서는 rollback이 맞습니다.
  - `H-001`처럼 memory-lean 연구 후보로 남길 정도의 뚜렷한 보조 이점도 이번 slice에서는 확보되지 않았습니다.

## 15. Next Slice Reprioritization
- 다음 slice: `H-004`
- 왜 이제 `H-004`인가:
  - `H-001`, `H-003` 모두 `PC`였고 canonical 기준으로는 둘 다 승격 실패였습니다.
  - 다음엔 parity 리스크가 낮은 `PO`를 먼저 보는 편이 더 안전합니다.
  - CPU loader / cache / engine reuse는 전략 의미론을 건드리지 않으면서 전체 반복 시간을 줄일 여지가 있습니다.
- 초심자용 한 줄:
  - 이제는 “매수/정렬 로직을 더 건드리는 것”보다 “같은 일을 덜 준비하고 덜 다시 읽는 것”부터 보는 게 더 합리적입니다.

## 16. H-004 First Contract (`H-004-a`)
### 16-1. What
- 목표: `find_optimal_parameters(...)` 안의 CPU-side 준비 단계에서, 이미 메모리에 올린 정보로 대체 가능한 중복 작업을 줄입니다.
- 첫 구현 범위는 아주 좁게 잡습니다.
  - `all_data_gpu`를 이미 로드한 뒤 다시 `create_engine(...) + pd.read_sql(...)`로 거래일 목록을 읽는 경로를 검토합니다.
  - 같은 run 안에서 이미 확보한 `all_data_gpu`의 인덱스/내용으로 `trading_dates_pd`, `all_tickers`를 안정적으로 만들 수 있으면 그 경로를 우선합니다.
- 쉽게 말하면:
  - 이미 들고 있는 데이터에서 알 수 있는 정보를, DB에 한 번 더 물어보지 않도록 줄여보는 시도입니다.

### 16-2. What must stay the same
- `trading_dates_pd`의 정렬 순서와 값은 기존과 같아야 합니다.
- `all_tickers`는 기존처럼 정렬된 목록이어야 합니다.
- 기간 필터링 결과(`start_date ~ end_date`)가 바뀌면 안 됩니다.
- GPU kernel 입력(`all_data_gpu`, `tier_tensor`, `pit_universe_mask_tensor`, `prepared_market_data`) 의미론은 건드리지 않습니다.
- cross-run 전역 캐시나 프로세스 간 공유 상태는 이번 slice 범위에 넣지 않습니다.

### 16-3. Why this is a better next slice
- `H-001`, `H-003`은 둘 다 `PC`였고, canonical 기준으로는 둘 다 회귀였습니다.
- `H-004-a`는 `PO`입니다.
- 즉, 전략 결과를 바꾸지 않고도 wall time 쪽에서 개선 여지를 볼 수 있습니다.
- 이번 slice는 “GPU kernel을 더 빠르게”보다 “커널에 들어가기 전에 덜 준비하기”에 가깝습니다.

### 16-4. Verification plan
- 최소 단위 검증:
  - 기존 경로와 새 경로가 같은 `trading_dates_pd`, 같은 `all_tickers`를 만드는지 테스트
  - 기존 경로와 새 경로가 같은 `prepared_market_data` shape contract를 유지하는지 확인
- 구현 후 확인:
  - `tests.test_gpu_engine_prep_path`
  - 필요 시 `tests.test_gpu_candidate_metrics_asof`
  - canonical 2-run 재측정
- 기대값:
  - kernel time은 같거나 거의 같아도 괜찮습니다.
  - wall time이 줄어드는지가 이번 slice의 핵심입니다.

## 17. H-004 Final Decision
- 상태: `Rolled back after measurement`
- 무엇을 바꿨나:
  - `find_optimal_parameters(...)`가 이미 로드한 `all_data_gpu`에서 거래일 축과 티커 축을 직접 계산하도록 바꿨습니다.
  - 같은 run 안에서 별도 SQL engine을 만들고 `DailyStockPrice`에서 거래일 목록을 다시 읽는 경로를 제거했습니다.
- 측정 결과:
  - `kernel`: `-7.89%`
  - `wall`: `-6.86%`
- 왜 멈췄나:
  - 없앤 일은 `SELECT DISTINCT date ...` 한 번이었고, 전체 run에서 차지하는 비중이 생각보다 작았을 가능성이 큽니다.
  - 대신 새 경로는 `all_data_gpu.index`에서 `date/ticker` unique를 뽑고, `to_pandas()`와 `sorted(...)`로 CPU 객체를 다시 만드는 비용이 추가됐습니다.
  - 쉽게 말하면, “DB에 다시 안 물어보는” 대신 “이미 큰 장부에서 다시 목록을 정리하는” 비용이 생겨서 오히려 손해가 난 그림에 가깝습니다.
- 최종 판단:
  - 기본 경로에서는 rollback이 맞습니다.
  - `H-001`처럼 memory-lean 보조 가치가 있는 것도 아니고, `H-003`처럼 live-path 정렬 비용을 줄이는 연구 가설로 남길 정도의 신호도 부족했습니다.

## 18. Breakdown-Driven Reprioritization
- 현재까지 `H-001`, `H-003`, `H-004-a` 모두 canonical 기준 `No-Go`였습니다.
- 그래서 다음 단계는 “바로 다음 구현”보다 “어디가 진짜 wall 병목인지 다시 좁히기”가 더 중요했습니다.
- 2026-03-13 probe 결과:
  - `wall = 1128.75s`
  - `kernel = 1081.12s`
  - `pre_kernel_stage_s = 32.59s`
  - raw log 4배치 합산 기준 `additional_buy_s = 1048.95s`
  - 같은 기준 `candidate_payload_s = 7.23s`, `new_entry_s = 18.53s`, `sell_s = 5.54s`
- 해석:
  - kernel이 여전히 전체 시간의 대부분입니다.
  - 그 kernel 안에서도 거의 전부가 `additional_buy`에 몰려 있습니다.
  - 반대로 `candidate_payload`, `new_entry`, `sell`은 지금 canonical 기준으로는 1차 병목이 아닙니다.
- 측정 메모:
  - 첫 probe 시점의 `summary.json`은 `[GPU_KERNEL_BREAKDOWN]` 첫 줄만 집계했습니다.
  - raw `run1.log`를 다시 합산해 4배치 기준 누적치를 확인했고, 이후 `issue98_perf_measure` 파서를 여러 breakdown 줄 합산 방식으로 보정했습니다.
- 초심자용 한 줄:
  - 이제는 “뭔가 줄이면 빨라질 것 같은 곳”이 아니라, 실제로 시간이 거의 다 쓰이는 `additional_buy`만 집중해서 보는 게 맞습니다.

## 19. Next Slice Contract (`H-005-a`)
### 19-1. What
- 목표: `_process_additional_buy_signals_gpu(...)` 안에서 전체 시간 대부분을 쓰는 추가매수 경로를 바로 최적화하지 않고, 먼저 sub-stage probe로 더 잘게 쪼갭니다.
- 이번 `H-005-a`는 **probe-only**입니다.
  - 전체 의미론은 그대로 둡니다.
  - `mask_gen`, `candidate_extract`, `cost_priority`, `sort`, `rank_apply`, `state_update` 시간을 분리 계측합니다.
  - 아직 `cp.where(...)`, `cp.lexsort(...)`, `for rank in range(...)` 알고리즘 자체는 바꾸지 않습니다.

### 19-2. What must stay the same
- 추가매수 우선순위(`lowest_order` / `highest_drop`) 결과가 바뀌면 안 됩니다.
- `cash`, `temp_capital`, `available_slots` 차감 타이밍이 바뀌면 안 됩니다.
- 동일 simulation 안에서 rank별 추가매수 순서가 바뀌면 안 됩니다.
- CPU/GPU strict parity 계약은 유지되어야 합니다.

### 19-3. Why this is now the best next slice
- `H-001`, `H-003`, `H-004-a`는 모두 실제 큰 병목이 아닌 곳을 건드렸거나, canonical 기준으로는 비용이 더 커졌습니다.
- 이번 probe에서는 `additional_buy`가 kernel 시간 대부분을 차지하는 것이 확인됐습니다.
- 다만 `additional_buy` 안에서 **어느 연산이 제일 느린지**는 아직 확정되지 않았습니다.
- 그래서 다음엔 “바로 고치기”보다 “이미 느리다고 증명된 곳 안을 더 잘게 재기”가 먼저입니다.

### 19-4. Verification plan
- `H-005-a`는 계측만 추가하므로, probe 전후 결과 의미가 바뀌지 않아야 합니다.
- 구현 후 확인:
  - `tests.test_backtest_strategy_gpu`
  - `tests.test_issue98_perf_measure`
  - `tests.test_gpu_engine_prep_path`
  - `issue98_perf_measure --kernel-breakdown --runs 1` 재실행
  - `summary.json`에서 `additional_buy_*` sub-stage가 모두 집계되는지 확인

### 19-5. Implementation status
- 2026-03-13: `H-005-a` probe-only instrumentation 구현 완료.
- 추가된 sub-stage key:
  - `additional_buy_mask_gen_s`
  - `additional_buy_candidate_extract_s`
  - `additional_buy_cost_priority_s`
  - `additional_buy_sort_s`
  - `additional_buy_rank_apply_s`
  - `additional_buy_state_update_s`
- 엔진은 `kernel_stage_timing_enabled=True`일 때 위 키를 누적하고, `issue98_perf_measure`는 이를 `kernel_stage_breakdown_s` / `median_kernel_stage_breakdown_s`로 집계합니다.
- 구현 검증:
  - `tests.test_issue98_perf_measure`
  - `tests.test_gpu_engine_prep_path`
  - `tests.test_backtest_strategy_gpu`
  - 결과: `35 tests OK`

### 19-6. First probe result
- 측정 artifact:
  - `results/issue98_measure/pr104_h005a_probe_janfeb2024_cov020_20260313_200426/summary.json`
- 1-run canonical 결과:
  - `median_kernel_s = 1019.06`
  - `median_wall_s = 1055.52`
  - `oom_retry = false`
  - `batch_count = 4`
- `additional_buy` 내부 sub-stage:
  - `additional_buy_s = 980.78s`
  - `additional_buy_state_update_s = 969.18s`
  - `additional_buy_rank_apply_s = 6.80s`
  - `additional_buy_mask_gen_s = 2.67s`
  - `additional_buy_cost_priority_s = 1.07s`
  - `additional_buy_sort_s = 0.17s`
  - `additional_buy_candidate_extract_s = 0.08s`
- 해석:
  - `additional_buy` 안의 1차 병목은 `sort`, `cp.where`, `cost 계산`이 아니라 `state_update`입니다.
  - 즉, “무엇을 살지 결정하는 연산”보다 “결정된 추가매수를 큰 상태 배열에 반영하는 연산”이 거의 전부의 시간을 차지합니다.
  - 다음 실제 최적화는 `state_update`를 다시 더 잘게 쪼개는 쪽이 가장 안전합니다.

### 19-7. `H-005-b-0` State-Update Micro-Probe
- 2026-03-13: `state_update`를 아래 4개 세부 단계로 다시 쪼개는 probe-only 계측을 추가했습니다.
  - `additional_buy_state_final_compact_s`
  - `additional_buy_state_slot_lookup_s`
  - `additional_buy_state_position_write_s`
  - `additional_buy_state_last_trade_update_s`
- 목적:
  - `state_update`가 느리다는 사실은 확인됐으므로, 이제 그 안에서 무엇이 진짜 병목인지 한 단계 더 줄여 확인합니다.
  - 이 단계도 여전히 의미론은 바꾸지 않습니다.
- 검증:
  - `tests.test_issue98_perf_measure`
  - `tests.test_gpu_engine_prep_path`
  - `tests.test_backtest_strategy_gpu`
  - 결과: `35 tests OK`
- 다음 측정:
  - `issue98_perf_measure --kernel-breakdown --runs 1`
  - `summary.json`에서 위 4개 key의 상대 비중을 비교한 뒤 `H-005-b` 실제 최적화 방향을 고릅니다.

### 19-8. `H-005-b` Direct Last-Trade Update
- 2026-03-13: `cp.unique(cp.vstack([final_sims, final_stocks]), axis=1, return_index=True)`를 제거하고, `last_trade_day_idx_state[final_sims, final_stocks] = current_day_idx` direct update로 교체했습니다.
- Why:
  - `H-005-b-0` probe에서 `additional_buy_state_last_trade_update_s = 1039.60s`로, `state_update` 안의 거의 전부가 마지막 거래일 갱신 구간에 몰렸습니다.
  - 현재 additional-buy accepted pair는 원래 `(sim, stock)` 2D grid에서 나온 좌표를 정렬/필터한 결과라는 가설을 채택했습니다.
- Safety guard:
  - `debug_mode=True`일 때만 `(final_sims, final_stocks)` unique invariant를 확인하도록 넣었습니다.
  - `tests.test_backtest_strategy_gpu`에 `last_trade_day_idx_state` 전용 회귀 테스트를 추가했습니다.
- 구현 검증:
  - `tests.test_backtest_strategy_gpu`
  - `tests.test_issue98_perf_measure`
  - `tests.test_gpu_engine_prep_path`
  - 결과: `36 tests OK`
- 다음 측정:
  - `issue98_perf_measure --kernel-breakdown --runs 1`
  - `additional_buy_state_last_trade_update_s`
  - `additional_buy_state_update_s`
  - `additional_buy_s`
  - `median_kernel_s`, `median_wall_s`
- 최종 측정 결과:
  - probe 1-run:
    - `results/issue98_measure/pr104_h005b_janfeb2024_cov020_20260313_212214/summary.json`
    - `median_kernel_s = 50.23s`
    - `median_wall_s = 86.55s`
    - `additional_buy_state_last_trade_update_s = 0.04s`
  - canonical final 2-run:
    - `results/issue98_measure/pr104_h005b_final_janfeb2024_cov020_20260313_212906/summary.json`
    - `median_kernel_s = 51.015s`
    - `median_wall_s = 87.295s`
    - baseline(`pr98_head_final_janfeb2024_cov020_20260312_223231`) 대비:
      - kernel `+94.96%`
      - wall `+91.68%`
    - `oom_retry = false / false`
    - `batch_count = 4 / 4`
- parity reconfirmation:
  - `results/issue98_measure/parity_canary/issue98_combo_shortlist_ad_20260312_report_all.json`
  - `failed = 0`
  - `decision_level_parity_zero_mismatch = true`
  - `promotion_blocked = false`
- 최종 판정:
  - `H-005-b`는 keep
  - `#104` tranche 목표는 달성
  - probe instrumentation은 후속 hot-path tranche 재사용을 위해 gated 도구로 유지

## 21. Final Close Recommendation
- 2026-03-13 기준 `#104`는 close 권고입니다.
- 이유:
  - exploratory slice 세 개(`H-001`, `H-003`, `H-004-a`)는 모두 canonical 회귀로 정리되어 재시도 비용을 낮췄습니다.
  - `H-005-a` / `H-005-b-0` probe로 병목을 additional-buy last-trade dedup으로 좁혔습니다.
  - `H-005-b`가 canonical 2-run과 strict parity를 모두 통과했습니다.
- 후속 규칙:
  - `--kernel-breakdown` instrumentation은 유지합니다.
  - 추가 GPU hot-path 작업은 새 issue/tranche로 엽니다.
  - 이 문서는 done evidence로 유지하고, 새 실험 로그는 여기에 누적하지 않습니다.

## 20. Multi-Agent Direction Review (Codex blind-first + Gemini post-check)
- 2026-03-13: `multi-agent-with-codex-gemini`로 `H-005-a` 방향성을 구현 전에 다시 검토했습니다.
- Codex 1차 블라인드 의견 요약:
  - 한 관점은 `active-domain compaction`이 유망하다고 봤습니다.
  - 한 관점은 `rank loop`의 반복 재스캔 제거가 가장 작은 blast radius라고 봤습니다.
  - 한 관점은 `initial_buy_mask` 생성부의 2D broadcast/temp 축소가 가장 안전하다고 봤습니다.
- 하지만 세 의견 모두 공통으로 동의한 점은 같았습니다.
  - 지금은 `additional_buy`가 병목인 건 확실하다.
  - 다만 그 안에서 어떤 sub-stage가 1차 병목인지는 아직 모른다.
  - 따라서 첫 slice는 최적화가 아니라 fine-grained profiling이어야 한다.
- Gemini 사후 대조도 같은 결론을 줬습니다.
  - `Fine-grained Profiling First`
  - 이유: micro-benchmark 부재, parity risk, memory-bandwidth saturation 가능성
- 최종 통합 판정:
  - 상태: `실행`
  - 실행안: `H-005-a`를 probe-only slice로 진행
  - 다음 실제 최적화는 `H-005-b`로 분리하고, `H-005-a` 결과를 본 뒤 고릅니다.
