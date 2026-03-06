# review: 성능/안정성 로직 재검토 멀티에이전트 숙의 메모

> 2026-03-06 Codex-only 멀티에이전트 숙의 결과를 임시 보관하는 로컬 메모입니다.
> 본 문서는 `#56`, `#67`, `#97`, `#98`의 우선순위와 게이트 판단을 연결하는 보조 기록이며,
> 이슈별 상세 구현 Source-of-truth를 대체하지 않습니다.

## 0. Meta
- **Type**: Research
- **Status**: Draft
- **Owner**: Codex session
- **Due**: TBD
- **Stakeholders / Audience**: repo maintainer
- **Priority**: P0
- **Links**:
  - `TODO.md`
  - `todos/2026_02_09-issue56-cpu-gpu-parity-topk.md`
  - `todos/2026_02_09-issue67-tier-universe-migration.md`
  - `todos/2026_02_17-issue97-legacy-code-audit-governance.md`
  - `todos/2026_02_17-issue98-gpu-throughput-refactor.md`

## 1. Goal & Success (Why)
- **Problem / Opportunity**:
  - 현재 로직을 32GB RAM + RTX 5060 + Ryzen 7 1700 환경에서 최대한 빠르고 안정적으로 업그레이드할 수 있는지 재검토가 필요했다.
- **Goal**:
  - 성능 최적화 착수 가능 여부를 parity/PIT/운영 안정성 기준으로 다시 판정한다.
- **Success Criteria (DoD)**:
  - `Go/Hold/No-Go` 판정
  - 우선순위 재정렬
  - 즉시 후속 액션 명시
- **Non-goals / Out of Scope**:
  - 이번 문서에서 직접 코드 수정은 하지 않는다.
  - Gemini/Web 의견은 포함하지 않는다.

## 2. Context & Constraints (What)
- **Background**:
  - Codex reviewer 4명 독립 리뷰(Round 1) 후, lead 요약 공유 및 반론 1회(Round 2), 마지막으로 red-team 1회 수행.
  - 환경:
    - RAM 32GB
    - GPU: RTX 5060
    - CPU: AMD Ryzen 7 1700 (8C/16T)
- **Constraints**:
  - CPU는 SSOT
  - decision-level parity `0 mismatch`가 승격 게이트
  - PIT/lookahead bias 금지
- **Dependencies**:
  - `#56` parity hard gate
  - `#67` PIT/coverage 정합
  - `#97` strict-only scope 정리
  - `#98` throughput 리팩토링
- **Risks**:
  - 잘못된 fast path 의미론을 성능 최적화로 고착
  - OOM fallback이 batch throughput을 장기 저하시킴
  - 32GB 환경에서 full preload 배치가 재시작 복구성을 해침

## 3. Requirements / Scope
- [x] GPU fast path와 CPU SSOT 간 의미론 차이 재검토
- [x] OOM fallback 및 host-side 병목 검토
- [x] PIT/T-1 규칙 및 same-day risk 재분류
- [x] 관련 TODO/테스트/코드 근거 수집
- [x] 후속 우선순위 도출

## 4. Strategy / Options (How)
- **Option A**: `#98` throughput 최적화 먼저 진행
  - 장점: 단기 체감 성능 개선 가능
  - 단점: parity 미해소 경로를 더 빠르게 실행할 위험이 큼
- **Option B**: `#56 -> #67 -> #97 -> #98` 순으로 진행
  - 장점: SSOT/parity/PIT 게이트를 먼저 닫고 성능 작업을 진행
  - 단점: 성능 개선 체감은 늦어짐
- **Design/Business Decision**:
  - **No-Go for immediate #98**
  - 우선순위는 `#56 Release parity 0 mismatch` -> `#67 PIT default path + coverage gate` -> `#97 strict-only scope 정리` -> `#98 throughput`

## 5. Open Questions
- Q) `same-day as-of`를 현재 주 리스크로 볼 것인가?
  - A안) 주 리스크로 본다
  - B안) 잔존 리스크로만 본다
- 결론/권고안):
  - hot path 기준으로는 **잔존 리스크**다.
  - 실제 1차 blocker는 `multi-sim fast path parity`, `strict scope 미완료`, `OOM fallback`, `host-side 병목`이다.

## 6. Action Items (Plan)
- [ ] `#56`: multi-sim 기준 strict parity release gate를 `0 mismatch`로 닫기
- [ ] `#56`: equity curve 외 `orders/fills/positions/pnl` diff report 추가
- [ ] `#67`: PIT default path + coverage gate TODO/코드 정합화
- [ ] `#97`: strict-only scope 강제와 legacy 입력 제거 범위 확정
- [ ] `#98`: OOM retry count, batch recovery, GPU util, H2D/D2H bytes 계측 스캐폴드 추가
- [ ] `#98`: batch 반감 고착 제거 전까지 성능 승격 보류
- **Next Action**:
  - `#56` release gate 문서와 하네스 기준을 먼저 고정
- **Owner**:
  - TBD

## 7. Notes & Evidence
- **Final Verdict**:
  - `No-Go`
  - 이유:
    - parity hard gate 미충족
    - GPU fast path와 CPU SSOT 의미론 차이 가능성
    - OOM fallback의 throughput 고착 저하
    - host-side Python orchestration 병목
- **Consensus**:
  - `#98`을 먼저 당기면 안 된다.
  - 성능 작업은 parity/PIT/strict scope 정리 이후 열어야 한다.
- **Dissent Resolved**:
  - `same-day as-of`는 주 리스크가 아니라 `current_day_idx` 없는 비표준 CPU 호출의 residual risk로 정리.

### Key Findings
- GPU strict 보정이 single-sim에만 한정됨:
  - [src/backtest/gpu/engine.py](/root/projects/Split_Investment_Strategy_Optimizer/src/backtest/gpu/engine.py#L313)
- GPU optimizer 기본 경로는 `fast` 의미론으로 흘러갈 수 있음:
  - [src/backtest/gpu/engine.py](/root/projects/Split_Investment_Strategy_Optimizer/src/backtest/gpu/engine.py#L90)
  - [src/optimization/gpu/parameter_simulation.py](/root/projects/Split_Investment_Strategy_Optimizer/src/optimization/gpu/parameter_simulation.py#L177)
- OOM 후 축소 batch size가 이후 배치에도 유지됨:
  - [src/optimization/gpu/parameter_simulation.py](/root/projects/Split_Investment_Strategy_Optimizer/src/optimization/gpu/parameter_simulation.py#L351)
- engine/data/logic에 host-side Python loop와 candidate 재조회가 남아 있음:
  - [src/backtest/gpu/engine.py](/root/projects/Split_Investment_Strategy_Optimizer/src/backtest/gpu/engine.py#L173)
  - [src/backtest/gpu/data.py](/root/projects/Split_Investment_Strategy_Optimizer/src/backtest/gpu/data.py#L85)
  - [src/backtest/gpu/logic.py](/root/projects/Split_Investment_Strategy_Optimizer/src/backtest/gpu/logic.py#L507)
  - [src/backtest/gpu/logic.py](/root/projects/Split_Investment_Strategy_Optimizer/src/backtest/gpu/logic.py#L669)
- Tier batch는 대형 preload 후 마지막에 commit:
  - [src/pipeline/daily_stock_tier_batch.py](/root/projects/Split_Investment_Strategy_Optimizer/src/pipeline/daily_stock_tier_batch.py#L1147)
  - [src/pipeline/daily_stock_tier_batch.py](/root/projects/Split_Investment_Strategy_Optimizer/src/pipeline/daily_stock_tier_batch.py#L1249)
- `same-day as-of`는 hot path 주 리스크 아님:
  - CPU는 일반 backtest 경로에서 T-1 사용:
    - [src/backtest/cpu/strategy.py](/root/projects/Split_Investment_Strategy_Optimizer/src/backtest/cpu/strategy.py#L127)
  - GPU도 T-1 helper 사용:
    - [src/backtest/gpu/utils.py](/root/projects/Split_Investment_Strategy_Optimizer/src/backtest/gpu/utils.py#L107)

### TODO / Gate Evidence
- `#98` 안정성 게이트:
  - [TODO.md](/root/projects/Split_Investment_Strategy_Optimizer/TODO.md#L162)
  - [TODO.md](/root/projects/Split_Investment_Strategy_Optimizer/TODO.md#L167)
- `#56` release gate 미충족:
  - [TODO.md](/root/projects/Split_Investment_Strategy_Optimizer/TODO.md#L189)
  - [TODO.md](/root/projects/Split_Investment_Strategy_Optimizer/TODO.md#L196)
  - [TODO.md](/root/projects/Split_Investment_Strategy_Optimizer/TODO.md#L200)

### Local Validation Executed
- Passed:
  - `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_gpu_parameter_simulation_orchestration tests.test_gpu_kernel_batch_size tests.test_cpu_gpu_parity_topk`
  - `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_cpu_candidate_priority tests.test_gpu_context_priority_validation tests.test_gpu_parameter_batch_fallback tests.test_backtest_strategy_gpu`
- Failed:
  - `CONDA_NO_PLUGINS=true conda run -n rapids-env python -m unittest tests.test_gpu_tier_tensor_pit tests.test_issue67_tier_universe`
  - 요약: `issue67` 테스트 일부가 현재 tier 강제 정책/유니버스 필터 동작과 불일치

## 8. Deliverables
- [x] [todos/2026_03_06-multi-agent-performance-stability-review.md](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_06-multi-agent-performance-stability-review.md): 멀티에이전트 숙의 결론 임시 저장

## 9. Change Log
- 2026-03-06: Codex-only 멀티에이전트 숙의 결과 최초 기록
