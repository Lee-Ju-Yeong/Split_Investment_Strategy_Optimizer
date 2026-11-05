# 작업 기록 색인

[현재 할 일](../TODO.md) · [프로젝트 소개](../README.md) · [문서 안내](../docs/README.md)

문서의 **유형과 완료 상태는 별개**입니다. 연구·검토·참고 문서는 그 자체가 제품 미완료 목록이 아닙니다. done_은 해당 문서의 작업 범위 완료이며 전체 제품·최신 환경의 검증이나 전략 승인을 뜻하지 않습니다.
과거 시험의 입력·SHA·실행일은 개별 기록에서 확인합니다. 원본 results/logs가 Git에 없으면 공개 기록만 확인할 수 있으며, 새 시험을 실행한 것으로 세지 않습니다.

## 처음 읽는 순서

1. [구현 사례](../docs/engineering-evidence.md)에서 문제·해결·검사를 확인합니다.
2. [현재 상태판](../TODO.md)에서 남은 확인과 연구를 구분합니다.
3. 필요한 작업 기록만 아래에서 찾습니다. 과거 코드 경로는 [현재 경로 안내](../docs/refactoring/issue69-import-path-mapping.md)와 대조합니다.
4. 이번 정비의 모든 제안과 보류 사유는 [실행 원장](2026_11_06-project-maintenance-execution.md)에 있습니다.

## 전체 기록

| 문서 | 유형 | 읽을 때의 상태 |
| --- | --- | --- |
| [Issue #68: Robust WFO / Ablation](2026_02_09-issue68-robust-wfo-ablation.md) | implementation | 부분 구현·후속 확인 |
| [Issue #72: Ticker Continuity & Position Inheritance](2026_02_09-issue72-ticker-continuity-position-inheritance.md) | implementation | 부분 구현·후속 확인 |
| [Reference Note: 로컬 DB 없이 진행 가능한 작업](2026_02_14-notebook-no-local-db-dev.md) | reference | 참고·후속 연결 |
| [테스트 인터페이스 갱신 (test_integration.py) (Issue #59)](2026_02_15-issue59-tests-split-no-db-gpu.md) | implementation | 부분 완료·DB 검증 미완료 |
| [Issue #101: Tier1 선택 편향 완화와 분포 기반 강건 최적화](2026_02_22-issue101-feat-tier1.md) | research | 연구·제안 |
| [Reference Backlog: Issue #71 Carryover](2026_03_01-issue71-carryover-non71-backlog.md) | reference | 참고·후속 연결 |
| [GPU-native WFO v2 Design Note](2026_03_06-gpu-native-wfo-v2-design.md) | research | 연구·제안 |
| [Review Memo: 성능/안정성 우선순위 재검토](2026_03_06-performance-stability-review.md) | review | 당시 검토·참고 |
| [Review Memo: 백테스트/최적화 재검토](2026_03_07-backtest-optimizer-review.md) | review | 당시 검토·참고 |
| [Work Note: `ShortSellingDaily` Publication Lag / PIT](2026_03_07-short-selling-publication-lag-pit.md) | implementation | 부분 구현·후속 확인 |
| [Review: Business Model & Marketability](2026_03_10-business-model-marketability-review.md) | review | 당시 검토·참고 |
| [Review: Roadmap & Commercialization Checkpoint](2026_03_10-roadmap-commercialization-checkpoint.md) | review | 당시 검토·참고 |
| [WFO / OOS Lane 임시 합의안](2026_03_12-wfo-oos-lane-provisional-review.md) | review | 당시 검토·참고 |
| [WFO Shortlist Derivation Review](2026_03_16-wfo-shortlist-derivation-review.md) | review | 당시 검토·참고 |
| [N-Window Shortlist Contract v1](2026_03_19-n-window-shortlist-contract-v1.md) | research | 연구·제안 |
| [Cluster V1 Shortlist Diversification PoC](2026_03_24-cluster-v1-shortlist-diversification-poc.md) | research | 연구·제안 |
| [대표 프로젝트 정비: 다중 관점 탐색과 실행안](2026_11_06-project-maintenance-discovery.md) | review | 탐색 완료·원제안 보존 |
| [대표 프로젝트 정비 실행 기록](2026_11_06-project-maintenance-execution.md) | implementation | 진행 중 |
| [feat(data): Point-in-Time 규칙 명문화 및 룩어헤드 방지 테스트 추가](done_2026_02_07-issue64-point-in-time-lookahead-bias.md) | implementation | 완료·당시 범위 |
| [feat(db): FinancialData, InvestorTradingTrend, DailyStockTier 스키마 및 인덱스 추가](done_2026_02_07-issue65-financial-investor-tier-schema-index.md) | implementation | 완료·당시 범위 |
| [feat(pipeline): 재무/수급 수집기 분리 및 DailyStockTier 사전 계산 배치 도입](done_2026_02_07-issue66-financial-investor-collector-tier-batch.md) | implementation | 완료·당시 범위 |
| [feat(data): 상폐 포함 Historical Ticker Universe 구축](done_2026_02_08-issue70-historical-ticker-universe-delisted.md) | implementation | 완료·당시 범위 |
| [feat(planning): pykrx 확장 데이터셋 + Tier v2 로드맵](done_2026_02_08-issue71-pykrx-tier-v2-data-roadmap.md) | implementation | 완료·당시 범위 |
| [Issue #56: CPU/GPU Parity Release Gate](done_2026_02_09-issue56-cpu-gpu-parity-topk.md) | implementation | 완료·당시 범위 |
| [Issue #67: PIT Universe + DailyStockTier Runtime Standardization](done_2026_02_09-issue67-tier-universe-migration.md) | implementation | 완료·당시 범위 |
| [설정 소스 표준화 및 하드코딩 경로/플래그 제거 (Issue #53)](done_2026_02_15-issue53-config-source-standardization.md) | implementation | 완료·당시 범위 |
| [구조화된 로깅 도입 및 하드코딩 디버그 출력 제거 (Issue #55)](done_2026_02_15-issue55-structured-logging.md) | implementation | 완료·당시 범위 |
| [스크립트 import 부작용 제거 (parameter_simulation_gpu.py) (Issue #60)](done_2026_02_15-issue60-import-parameter-simulation-gpu-py.md) | implementation | 완료·당시 범위 |
| [임포트 스타일 통일(상대/절대/스크립트 실행) (Issue #61)](done_2026_02_15-issue61-import-style-standardization.md) | implementation | 완료·당시 범위 |
| [src 패키지 구조 재편 및 대형 모듈 브레이크다운(동작 동일 리팩터링) (Issue #69)](done_2026_02_16-issue69-src-package-restructure-breakdown.md) | implementation | 완료·당시 범위 |
| [Issue #93: Wrapper Deprecation / Removal Plan](done_2026_02_16-issue93-wrapper-deprecation-removal-plan.md) | reference | 완료·당시 범위 |
| [Issue #97: Legacy Audit And Strict-Only Governance](done_2026_02_17-issue97-legacy-code-audit-governance.md) | implementation | 완료·당시 범위 |
| [GPU 처리량 개선 기록 (기존 로컬 식별자 issue98)](done_2026_02_17-issue98-gpu-throughput-refactor.md) | implementation | 완료·당시 범위 |
| [Issue #104: GPU Throughput Follow-up Hot Path](done_2026_03_13-issue104-gpu-throughput-followup-hotpath.md) | implementation | 완료·당시 범위 |
| [무료 실행기로 CPU 기본 CI 도입](done_2026_11_06-cpu-checks-ci.md) | implementation | 완료·당시 범위 |
| [공개 저장소 안내 정비](done_2026_11_06-public-project-refresh.md) | implementation | 완료·당시 범위 |

## 과거 체크리스트를 읽을 때

- 빈 칸이 재사용 양식·나중에 대체된 계획·실제 후속 과제 중 무엇인지 본문과 이슈를 대조합니다. 완료 문서라는 이유로 일괄 체크하지 않습니다.
- #56은 당시 합성 자료 검증 범위이며 실제 최적화 결과 재검증과 구분합니다. #67의 과거 hybrid 계획은 strict-only 전환으로 대체됐고, #97의 승인된 유지 예외는 삭제 완료와 다릅니다.
- GPU 처리량 기록의 issue98은 로컬 식별자입니다. 실제 병합은 PR #103, 후속 성능 개선은 PR #105입니다.
- 찾을 수 없는 원본·참고 자료는 미확인으로 남깁니다. 자료가 없다는 이유로 비공개 자료나 유실 사고라고 단정하지 않습니다.
