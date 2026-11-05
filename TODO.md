# 현재 작업과 확인 범위

> 문서 대조일: 2026-11-06(Asia/Seoul). 실제 시험일·입력·후보 SHA는 각 기록에서 따로 확인합니다.

처음 방문했다면 [프로젝트 소개](README.md) → [구현 근거](docs/engineering-evidence.md) → [직접 확인하기](tests/README.md) 순서로 읽으세요.
모든 작업 문서는 [작업 색인](todos/README.md), 이번 56개 제안의 처리 결과는 [정비 실행 기록](todos/2026_11_06-project-maintenance-execution.md)에 연결합니다.

## 지금 진행하는 정비

| 작업 | 상태 | 다음 행동 |
| --- | --- | --- |
| 문서·이슈 상태 정리 | 진행 중 | 완료 기록·원래 이슈 요구·현재 코드 대조, 링크와 공개 상태 확인 |
| #59 DB 통합 검사 | 부분 완료 / 실제 DB 검증 미완료 | [기존 기록](todos/2026_02_15-issue59-tests-split-no-db-gpu.md)의 호출 불일치·건너뜀 처리를 보수하고 전용 임시 MySQL에서 검증 |
| 방문 동선·Pages | 계획 확정 | 코드 읽기 안내·한국어 제목·탐색·공개 설정 정비 |

매매 규칙·GPU 계산·운영 DB·전체 환경 이관은 이 정비의 범위가 아닙니다. DB 검사는 합성 자료와 실행마다 폐기하는 검사용 DB만 사용합니다.

## 구현됐지만 남은 확인

| 항목 | 확인된 범위 | 남은 확인 |
| --- | --- | --- |
| 공매도 데이터의 지연 반영 | 임시 3거래일 정책·회귀 검사·읽기 전용 영향 비교 구현 | 실제 제공처 가용 시점 확정, 기존 장기 재계산의 완료 증거·행 수·최신 날짜·샘플 확인. 이번에 재계산하지 않음. [기록](todos/2026_03_07-short-selling-publication-lag-pit.md) |
| 기간을 나눈 전략 평가 #68 | PR #106의 후보 고정·CPU 검산·holdout 보호·후보 추출 | 원래 ablation·강건성 평가와 최종 승인 조건. 열린 PR #75의 차이도 별도. [요구 대조](todos/2026_02_09-issue68-robust-wfo-ablation.md) |
| 전체 실행 환경 | CPU 최소 확인 경로와 명령 안내 제공 | Python 환경 파일·YAML/INI 설정 혼재, GPU 환경 준비 안내. [실행 조건](docs/running.md) |
| 과거 CPU·GPU 일치 및 성능 | 해당 기록의 당시 입력에서 확인 | 전체 입력·최신 GPU 환경을 검증한 결과가 아님. [측정 조건](docs/engineering-evidence.md#gpu) |

## 열린 이슈와 연구

현재 상태·근거·재개 조건은 이슈 본문에 기록합니다. 오래됐다는 이유로 닫거나 연구를 구현 성과로 소개하지 않습니다.

- 사용성·설명: [#19 통합 CLI](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/19), [#20 GPU 환경 안내](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/20), [#21 핵심 API 설명](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/21), [#30 결과 리포팅](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/30).
- 구현 후속: [#52 GPU 순차 처리·병목](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/52), [#54 파이프라인](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/54), [#57 모델·캐시](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/57), [#58 DB 접근](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/58), [#59 통합 검사](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/59).
- 평가·기준서: [#68 강건성 평가](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/68), [#94 전략 기준서](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/94). PR #75·#95는 차이와 재개 조건을 기록하고 유지합니다.
- 연구: [#35 동적 파라미터](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/35), [#72 기업 이벤트와 포지션 승계](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/72), [#101 분포 기반 최적화](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/101), [#107 ETF 전략](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/107).
- 다른 과제에서 이어진 항목은 [#71 후속 연결표](todos/2026_03_01-issue71-carryover-non71-backlog.md)에서 추적합니다.

## 완료 기록

| 완료한 범위 | 근거와 경계 |
| --- | --- |
| 후보군·시점 기준 전환 #67 | [당시 검증 기록](todos/done_2026_02_09-issue67-tier-universe-migration.md). 과거 모드의 모든 체크박스를 지금 실행했다는 뜻은 아님 |
| 레거시 조사와 승인 범위 정리 #97 | [분류·승인·유지 예외](todos/done_2026_02_17-issue97-legacy-code-audit-governance.md) |
| 호환 경로 정리 #93 | [PR #96 완료 기록](todos/done_2026_02_16-issue93-wrapper-deprecation-removal-plan.md) |
| GPU 처리량 개선 | [PR #103 기록](todos/done_2026_02_17-issue98-gpu-throughput-refactor.md). 기존 로컬 식별자 issue98은 실제 GitHub PR #98과 다름 |
| GPU 후속 개선 #104 | [PR #105 측정·원복 기록](todos/done_2026_03_13-issue104-gpu-throughput-followup-hotpath.md) |
| 공개 안내·작은 예제 | [PR #109 완료 기록](todos/done_2026_11_06-public-project-refresh.md) |
| CPU 기본 CI | [PR #110 완료 기록](todos/done_2026_11_06-cpu-checks-ci.md): 기존 검사 34개·예제·도구 자체 검사 22개 |

그 밖의 완료 기록과 연구·검토 문서는 [전체 색인](todos/README.md)에 있습니다.
[3월 사업화 검토](todos/2026_03_10-roadmap-commercialization-checkpoint.md)의 분기 일정·외부 출시 조건은 당시 제안입니다. 현재 서비스 출시 일정이나 채용용 코드 소개의 제한으로 읽지 않습니다.

## 기록을 갱신할 때

- 현재 할 일은 이 상태판, 이유·완료 조건·검증은 상세 작업 문서에 남깁니다.
- 문서 유형(구현·연구·검토·참고)과 상태(진행·완료·보류)는 별개입니다.
- 완료 문서는 결과·근거·남은 조건을 적고 알려진 링크를 고친 뒤 done_를 붙입니다. 시험 건너뜀은 통과가 아닙니다.
- 실제 자료·시험 근거 없이 완료·개인 기여·운영 현황·성능을 확대하지 않습니다.
