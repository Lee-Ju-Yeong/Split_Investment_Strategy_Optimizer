# Review Memo: 성능/안정성 우선순위 재검토

> Type: `review`
> Status: `reference`
> Priority: `N/A`
> Last updated: 2026-03-07
> Related issues: `#56`, `#67`, `#97`, `#98`
> Gate status: `reference only`

## 1. Summary
- What: `#98`을 바로 열어도 되는지 다시 점검한 메모입니다.
- Why: 성능 최적화를 먼저 당기면 parity/PIT/strict scope가 흐려질 수 있기 때문입니다.
- Final verdict: immediate `#98` promotion은 `No-Go`였습니다.
- Next action: `#56 -> #67 -> #97 -> #98` 순서를 유지합니다.

## 2. Consensus
- 현재 주 blocker는 throughput 자체보다 `release parity`, `runtime PIT policy`, `strict-only scope`입니다.
- `OOM fallback`, host-side orchestration, candidate hot path가 병목이지만, 의미론이 고정되기 전에는 승격하면 안 됩니다.
- `same-day as-of`는 핵심 blocker보다는 잔존 리스크로 재분류되었습니다.

## 3. Follow-up Owner Docs
- `#56`: release-grade parity evidence
- `#67`: PIT / coverage / candidate policy
- `#97`: legacy strict-only governance
- `#98`: throughput refactor

## 4. Reading Rule
- 이 문서는 우선순위 판단 근거를 설명하는 참고 메모입니다.
- 실제 체크리스트와 증적은 각 issue 문서를 우선합니다.
