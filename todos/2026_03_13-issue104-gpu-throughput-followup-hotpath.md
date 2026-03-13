# Issue #104: GPU Throughput Follow-up Hot Path

> Type: `implementation`
> Status: `in_progress`
> Priority: `P1`
> Last updated: 2026-03-13
> Related issues: `#104`, `#98`, `#56`
> Gate status: `issue opened on GitHub; follow-up tranche created from #98 handoff; H-001 was implemented/tested/measured, then rolled back after canonical 4-run regression confirmation; H-003 was re-scoped onto the canonical live path in data.py/engine.py and targeted regression tests are green; throughput remeasure is pending`

## 1. One-Page Summary
- What: `#98`에서 일부러 미뤄둔 더 공격적인 GPU hot-path 최적화를 별도 tranche로 진행하는 문서입니다.
- Why: `#98`은 current HEAD 기준 canonical 성능 개선과 strict parity 재확인까지 마치고 닫았습니다. 이제는 완료 문서를 오염시키지 않고, 다음 최적화만 분리해서 추적해야 합니다.
- Current status: GitHub issue `#104`를 만들었고, 로컬 작업 브랜치 `feature/issue98-followup-hotpath`도 준비됐습니다. `H-001`은 구현, 테스트, canonical 재측정, 4-run 재판정까지 마친 뒤 rollback했습니다. 현재 worktree는 다시 baseline 경로로 돌아왔습니다.
- Next action: current HEAD 기준 canonical throughput 재측정을 진행합니다. 이번에는 non-debug live path에서 ticker string materialization을 건너뛰는 실제 경로가 측정 대상입니다.

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
- [ ] canonical throughput 재측정

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
| `H-003` | `PC` | `src/backtest/gpu/data.py`, `src/backtest/gpu/engine.py` | canonical/live path의 candidate payload 생성 구간에서 non-debug ticker string materialization을 피하고 numeric `ticker_rank` tie-break로 정렬 유지 | 현재 실제 runtime은 `logic.py`가 아니라 `data.py -> engine.py` 경로를 사용한다 | implemented; remeasure pending |
| `H-004` | `PO` | CPU loader / cache | CPU I/O / session cache / engine reuse 계열 정리 | GPU만 빠라도 CPU 보조 경로가 느리면 전체 반복 속도가 늘어진다 | hot-path 본체와 분리 가능 |

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
- 상태: `Implemented on live path, waiting for measurement`
- 무엇을 바꿨나:
  - non-debug tensor gather 경로에서 ticker 문자열을 바로 만들지 않도록 바꿨습니다.
  - 대신 `ticker_rank` 숫자 tie-break를 만들고, 실제 payload 정렬은 그 숫자 키로 유지합니다.
  - debug 기록이 필요할 때만 ticker 문자열을 남깁니다.
- 왜 이 방식이 안전한가:
  - canonical/live path에 직접 들어갑니다.
  - 정렬 우선순위는 여전히 `entry_composite_score_q desc -> market_cap_q desc -> ticker asc`와 동등합니다.
  - backward compatibility를 위해 기존 tensor helper 기본값은 유지하고, engine만 non-debug 경로에서 `include_ticker_strings=False`를 명시합니다.
  - legacy/manual prepared bundle에는 `ticker_rank`를 1회 backfill해서 late `KeyError` 없이 지나가게 했습니다.
- 다음 확인:
  - current HEAD canonical 2-run
  - 필요 시 strict parity canary 재실행
