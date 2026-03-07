# Review Memo: Gemini + Codex 백테스트/최적화 재검토

> Type: `review`
> Status: `reference`
> Priority: `N/A`
> Last updated: 2026-03-07
> Related issues: `#54`, `#56`, `#67`, `#98`
> Gate status: `reference only`

## 1. Summary
- What: CPU 백테스트, GPU 백테스트, GPU optimizer를 다시 읽고 `TODO.md`에 없던 follow-up backlog를 추린 메모입니다.
- Why: 성능 병목과 정합성 리스크를 동시에 분리해서 관리해야 했기 때문입니다.
- Final verdict: immediate code change보다 backlog 분리와 gate 명확화가 먼저였습니다.
- Next action: 아래 follow-up은 각 owning issue 문서에서 실행합니다.

## 2. Key Conclusions
- GPU optimizer/runtime path가 CPU SSOT candidate gate를 아직 완전히 공유하지 않습니다.
- optimizer default path는 strict/as-of 의미론을 강제하지 못합니다.
- CPU 경로는 live SQL-in-loop와 작은 LRU cache 때문에 느리고, frozen manifest가 없습니다.
- GPU 경로는 preload 이후에도 일별 ranking 재계산과 batch별 재물질화가 남아 있습니다.
- CPU/GPU composite rank 직접 parity fixture가 부족합니다.

## 3. Follow-up Mapping
- `#67`
  - runtime candidate gate parity
  - frozen PIT candidate manifest
- `#54`
  - CPU backtest session cache 정리
- `#98`
  - fixed-data VRAM blind spot
  - daily as-of ranking precompute
  - ranking scratch memory estimate
  - strict fallback telemetry
  - multi-sim active-set rerank parity
- `#56`
  - optimizer strict-only gate
  - direct composite rank parity fixture
  - 신규 진입 계약 테스트 고정

## 4. Reading Rule
- 이 문서는 새 backlog 항목이 왜 생겼는지 설명하는 근거 메모입니다.
- 실제 실행 상태는 해당 issue 문서를 봅니다.
