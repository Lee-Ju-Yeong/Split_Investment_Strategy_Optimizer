# Project Status Dashboard

> Last updated: 2026-03-13
> Role: 이 파일은 현재 진행 중인 일을 빠르게 파악하는 관제판입니다.
> Rule: 상세 계획, 작업 로그, 증적은 각 `todos/*.md` 문서가 담당합니다.

## Core Invariants
- `CPU=SSOT`: CPU 백테스터가 최종 기준입니다.
- `Parity First`: release-grade 변경은 CPU/GPU parity mismatch `0`이 기본 게이트입니다.
- `PIT / No Lookahead`: 미래 데이터 참조는 즉시 중단 사유입니다.
- `Evidence Before Promotion`: 중요한 판단에는 결과 파일, 테스트, 리포트 링크가 있어야 합니다.

## How To Read
1. `Active Focus`에서 지금 실제로 움직이는 항목을 먼저 봅니다.
2. 각 항목의 상세 계획과 로그는 링크된 `todos/*.md` 문서를 엽니다.
3. `Research & Review Notes`는 참고 자료입니다. 실행 SSOT가 아닙니다.
4. `todos/done_*.md`는 완료 아카이브입니다. 현재 우선순위 판단에는 쓰지 않습니다.

## Document Types
- `implementation`: 실제 실행 계획, 체크리스트, 로그, 증적을 가진 문서
- `research`: 아직 공식 경로가 아닌 설계/실험 문서
- `review`: 판단 근거와 결론을 요약해 둔 문서
- `reference`: 참고 메모, 이관 백로그, 개발 편의 노트

## Gate Board
| Gate | Current State | Meaning | Owner Doc |
| --- | --- | --- | --- |
| `#56 Release parity` | Done (synthetic) | config single-row + synthetic `top-k=20` 범위까지 decision-level 증적 확보, real optimizer/WFO CSV는 spot revalidation만 남음 | [#56](todos/done_2026_02_09-issue56-cpu-gpu-parity-topk.md) |
| `#67 Runtime PIT candidate policy` | Done (live evidence) | GPU runtime gate parity, strict frozen manifest `record/replay`, candidate order zero-mismatch, structured `pit_failure` artifact까지 실제 parity/certification run 증적 확보 | [#67](todos/done_2026_02_09-issue67-tier-universe-migration.md) |
| `ShortSellingDaily PIT lag` | Open | `sbv_ratio` same-date 반영의 PIT 의미를 확정해야 함 | [lag note](todos/2026_03_07-short-selling-publication-lag-pit.md) |
| `외부 출시 경계` | Open | 내부 검증 단계와 외부 출시 단계를 분리해서 관리해야 함 | [review](todos/2026_03_10-roadmap-commercialization-checkpoint.md) |
| `#97 Strict-only governance` | Done | Gate A/B/C 승인, step 2 synthetic sample pack 승인, step 3 active non-strict surface 제거까지 완료. Gate B 예외(`src.parameter_simulation_gpu_lib`, `src.main_script`, historical/archive docs)만 명시적으로 유지 | [#97](todos/done_2026_02_17-issue97-legacy-code-audit-governance.md) |
| `#98 Throughput promotion` | Done | PR `#103`으로 current HEAD canonical throughput win과 strict parity reconfirmation을 함께 반영했고, 이번 tranche 목표를 종료했다 | [#98](todos/2026_02_17-issue98-gpu-throughput-refactor.md) |

## Active Focus
| Priority | Item | Status | Why Now | Next Action | Detail |
| --- | --- | --- | --- | --- | --- |
| `P0` | `ShortSellingDaily` publication lag 정리 | In Progress | 공매도 데이터 same-date 반영은 PIT 리스크 후보이며, 비인증 KRX 경로는 현재 호스트에서 막혀 있다 | 임시 `lag=3 + same-date 금지` 정책은 코드 반영 완료. 이제 shadow diff / backfill 범위를 정리하고 영향 구간을 증적으로 남긴다 | [doc](todos/2026_03_07-short-selling-publication-lag-pit.md) |
| `P0` | `내부 검증 / 외부 출시 경계 정리` | Draft | 아직 외부 출시 전 단계인데 가능한 일과 불가능한 일이 섞여 보이면 혼선이 생김 | 내부 검증만 기본 허용, 외부는 `NDA + 읽기 전용 설명` 범위까지만 허용하는 문구로 정리 | [review](todos/2026_03_10-roadmap-commercialization-checkpoint.md) |
| `P1` | `#104` GPU throughput follow-up hot path | Done | `H-001`, `H-003`, `H-004-a`는 canonical 회귀로 정리했고, `H-005-b`가 `cp.unique` 기반 additional-buy last-trade dedup 제거 후 canonical 2-run `+94.96% / +91.68%`와 strict parity를 함께 통과했다 | 새 hot-path tranche가 생기기 전까지 재개하지 않는다. 필요 시 `--kernel-breakdown` probe를 재사용한다 | [doc](todos/2026_03_13-issue104-gpu-throughput-followup-hotpath.md) |
| `P2` | `#68` Robust WFO / Ablation | Planned | 공식 경로 안정화 후 전략 선택 계층을 고도화해야 함 | 임시 합의안 기준으로 `Anchored WFO`, `final untouched OOS`, `stress pack` 구조와 robust score / hard gate 공식안 고정 | [doc](todos/2026_02_09-issue68-robust-wfo-ablation.md) |

## Roadmap Checkpoint (2026-03-10)
- `2026 Q2`: 데이터 계약과 검증 규칙을 먼저 확정합니다. `ShortSellingDaily lag`, strict parity, manifest, `#68` 공식안을 이 구간에서 잠급니다.
- `2026 Q3`: 강건성 검증과 결과 리포트 표준화를 진행합니다. CPU certification 결과를 기본 산출물로 굳힙니다.
- `2026 Q4`: `#72` continuity, `#101` 편향 완화, k3s/GPU job 연구는 내부 검증용으로만 진행합니다.
- `2027 Q1`: 출시 기준을 모두 통과했을 때만 제한적 공개 또는 canary를 검토합니다.
- 기준: 외부 유료 파일럿, 실시간 신호 제공, 실계좌 연동은 출시 기준 통과 전까지 금지합니다.

## Backlog Summary

### Integrity & Runtime
- `#72` ticker continuity / position inheritance: 합병·분할·티커 변경 시 포지션 연속성 설계가 필요합니다. [doc](todos/2026_02_09-issue72-ticker-continuity-position-inheritance.md)
- `#54` DataPipeline modularization + CPU session cache: CPU backtest session cache와 레거시 스크립트 정리가 남아 있습니다.
- `#58` DB access layer standardization: connector/engine 표준화가 아직 남아 있습니다.
- `#57` domain model/cache integration: Position/CompanyInfo 캐시 통합은 아직 후순위입니다.
- `#71 carryover backlog`: issue #71에서 넘어온 후속 데이터/거버넌스 항목을 따로 모아둔 문서입니다. [doc](todos/2026_03_01-issue71-carryover-non71-backlog.md)

### Strategy & Research
- `#101` deterministic Tier1 bias 완화: 특정 종목 고착보다 `theta x scenario x fold` 분포를 기준으로 파라미터를 고르려는 연구 트랙입니다. [doc](todos/2026_02_22-issue101-feat-tier1.md)
- `GPU-native WFO v2`: 현재 공식 경로가 아닌 내부 연구용 문서입니다. [doc](todos/2026_03_06-gpu-native-wfo-v2-design.md)

### Supporting Notes
- 로컬 DB 없이 할 수 있는 개발 범위: [note](todos/2026_02_14-notebook-no-local-db-dev.md)
- 성능/안정성 우선순위 재검토 메모: [review](todos/2026_03_06-performance-stability-review.md)
- 백테스트/최적화 재검토 메모: [review](todos/2026_03_07-backtest-optimizer-review.md)
- 사업화 포함 로드맵 점검 메모: [review](todos/2026_03_10-roadmap-commercialization-checkpoint.md)
- 사업모델/시장성 검토 메모: [review](todos/2026_03_10-business-model-marketability-review.md)

## Research & Review Notes
- [로드맵/사업화 점검 메모](todos/2026_03_10-roadmap-commercialization-checkpoint.md)
  - 성격: `review`
  - 읽는 시점: 내부 검증 단계와 외부 출시 가능 시점을 다시 확인할 때
- [사업모델/시장성 검토 메모](todos/2026_03_10-business-model-marketability-review.md)
  - 성격: `review`
  - 읽는 시점: 어떤 BM을 먼저 잡아야 하는지 또는 공개레포에 남길 사업 문서의 범위를 정할 때
- [GPU-native WFO v2 design](todos/2026_03_06-gpu-native-wfo-v2-design.md)
  - 성격: `research`
  - 읽는 시점: 공식 경로 대체를 고민할 때
- [성능/안정성 재검토 메모](todos/2026_03_06-performance-stability-review.md)
  - 성격: `review`
  - 읽는 시점: `#98`을 왜 바로 열지 않는지 설명이 필요할 때
- [백테스트/최적화 재검토 메모](todos/2026_03_07-backtest-optimizer-review.md)
  - 성격: `review`
  - 읽는 시점: 최근 추가된 backlog 항목의 근거를 확인할 때
- [로컬 DB 없이 진행 가능한 작업 메모](todos/2026_02_14-notebook-no-local-db-dev.md)
  - 성격: `reference`
  - 읽는 시점: 노트북/비로컬 DB 환경에서 바로 작업 가능한 범위를 보고 싶을 때

## Recent Done / Archive
- 완료 문서는 `todos/done_*.md`에 둡니다.
- 최근 완료된 큰 묶음:
  - [#64 PIT / lookahead 방지](todos/done_2026_02_07-issue64-point-in-time-lookahead-bias.md)
  - [#65 schema/index 확장](todos/done_2026_02_07-issue65-financial-investor-tier-schema-index.md)
  - [#66 collector 분리 + tier batch](todos/done_2026_02_07-issue66-financial-investor-collector-tier-batch.md)
  - [#70 historical universe](todos/done_2026_02_08-issue70-historical-ticker-universe-delisted.md)
  - [#71 pykrx + Tier v2 roadmap](todos/done_2026_02_08-issue71-pykrx-tier-v2-data-roadmap.md)
  - [#69 package restructure](todos/done_2026_02_16-issue69-src-package-restructure-breakdown.md)
  - [#56 parity release gate](todos/done_2026_02_09-issue56-cpu-gpu-parity-topk.md)
  - [#67 PIT candidate policy](todos/done_2026_02_09-issue67-tier-universe-migration.md)
  - [#97 strict-only governance](todos/done_2026_02_17-issue97-legacy-code-audit-governance.md)
  - [#98 throughput refactor tranche](todos/2026_02_17-issue98-gpu-throughput-refactor.md)
- 완료 상태이지만 참고가 필요한 운영 문서:
  - [#93 wrapper deprecation/removal plan](todos/2026_02_16-issue93-wrapper-deprecation-removal-plan.md)

## Rule For Future Docs
- 새 active 문서는 반드시 `Status / Type / Why / Next Action / Gate Status / Evidence Links / Related Issues`를 문서 앞부분에 둡니다.
- `implementation` 문서는 체크리스트와 작업 로그를 함께 가져야 합니다.
- `research/review/reference` 문서는 배경 설명보다 `이 문서를 언제 읽어야 하는가`를 먼저 써야 합니다.
