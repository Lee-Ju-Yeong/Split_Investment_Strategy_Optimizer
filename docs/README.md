# 문서 안내

처음 방문했다면 [프로젝트 소개](../README.md)에서 목적을 읽고, [세 가지 구현 근거](engineering-evidence.md)에서 코드·테스트·PR을 확인하세요. 직접 실행하려면 [외부 연결 없는 예제와 검사](../tests/README.md)부터 시작할 수 있습니다.

## 현재 구현을 이해할 때

| 궁금한 내용 | 문서 |
| --- | --- |
| 어떤 문제를 어떻게 해결했나 | [구현 근거](engineering-evidence.md) |
| 무엇을 직접 실행할 수 있나 | [CPU 예제·검사](../tests/README.md) · [격리 MySQL 검사](testing-mysql.md) · [전체 실행 조건](running.md) |
| 분할 매매는 어떤 규칙으로 계산하나 | [전략 원칙](MAGIC_SPLIT_STRATEGY_PRINCIPLES.md) |
| 데이터를 어떻게 저장하나 | [DB 스키마](database/schema.md) · [종목 등급 계산 규칙](database/daily_stock_tier_rules.md) |
| 코드는 어떤 순서로 읽나 | [src/ 읽기 안내](../src/README.md) |
| 실행 파일과 실제 구현 위치가 왜 다른가 | [소스 경로 안내](refactoring/issue69-import-path-mapping.md) · [호환 진입점 유지 정책](refactoring/wrapper-deprecation.md) |
| 구현된 것과 남은 일이 무엇인가 | [상태판](../TODO.md) |

## 개발·실험 이력을 읽을 때

| 묶음 | 읽는 방법 |
| --- | --- |
| [todos](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/tree/main/todos) | 개별 변경의 계획·판단·검증 기록. `done_`은 그 작업 범위의 완료이며 제품 전체 승인과 다름 |
| [WFO 설명](briefings/2026-03-14-wfo-journey-and-status.md) | 기간을 나눈 검증이 발전한 과정과 당시 판단. 문서의 날짜와 적용 범위를 함께 확인 |
| [운영 문서 안내](operations/README.md) | 수집·접속·검증 절차와 과거 운영 기록. 실제 연결값은 공개 문서에 보관하지 않음 |
| [과거 재적재 검증 절차](database/backfill_validation_runbook.md) | 준비된 DB·원천 데이터가 필요한 절차. 지금 실행 중이라는 증거로 읽지 않음 |
| [사업화 참고 문서](business/README.md) | 과거 검토·제안의 범위. 현재 고객·매출·출시 성과를 뜻하지 않음 |
| [노트북 안내](../notebook/README.md) | 수집·상관관계·설정 탐색의 과거 실험. 기본 실행 경로는 `src/` |
| [보관된 도구](../archive/legacy_tools/README.md) | 사용을 중단한 보조 도구와 대체 경로 |

## AI 작업 지침과 과거 설계 자료

`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`는 모두 [llm.md](../llm.md)를 가리킵니다. 도구가 읽는 위치를 유지하기 위해 루트에 둡니다. 문서 속 역할 설정은 AI 도구에 대한 지시이며 유지보수자의 직함·경력 소개가 아닙니다.

[llm-context 안내](../llm-context/README.md)는 2025년 설계·실험 배경을 읽기 위한 자료입니다. 당시 `completed` 표기, SQLite 설명, 오래된 모듈명은 현행 제품 상태와 구분합니다. 현재 상태는 코드, 검증 기록, [TODO](../TODO.md)를 대조합니다.

## 근거의 범위

문서에 적힌 과거 수치와 이번에 재실행한 검사는 구분합니다. `results/`·`logs/`에만 있던 산출물은 공개 저장소에서 열 수 없을 수 있습니다. 원본이 없는 수치는 원본을 재검증했다고 쓰지 않습니다.

최초 [소개 정비(PR #109)](../todos/done_2026_11_06-public-project-refresh.md)에서는 경로를 유지했습니다. 후속 정비에서는 완료 기록의 이름과 링크를 바로잡고 [전체 작업 색인](../todos/README.md)을 만들었습니다. 과거 로그·명령은 보존하며 [56개 제안별 결과](../todos/2026_11_06-project-maintenance-execution.md)에서 이번 반영·보류 범위를 확인할 수 있습니다.

문서 사이트는 같은 Markdown을 게시합니다. [문서 화면 검사와 유지 방법](pages-maintenance.md)에서 미리보기 범위와 배포의 차이를 확인할 수 있습니다.
