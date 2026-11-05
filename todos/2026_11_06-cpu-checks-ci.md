# 무료 실행기로 CPU 기본 CI 도입

- Status: Done (구현·도입 전 검증). 최종 병합·공개 확인 결과는 아래 PR에 기록.
- Type: implementation
- Why: 방문자가 실행하는 최소 CPU 검사와 예제를 PR·main에서도 같은 명령으로 확인한다.
- Next Action: 검사 대상·실행 코드가 달라지면 자체 검사와 해당 후보의 CI를 확인한다. 이번 도입의 최종 SHA·병합·main 실행·브랜치 정리 결과는 [PR #110](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/pull/110) 본문에 이어 기록한다.
- Gate Status: 원격 개발 환경·지정 리뷰·PR 구현 후보 CPU checks·Secret Scan 통과. 최종 병합은 해당 후보의 성공을 재확인한 뒤 진행한다. 병합 필수 검사 지정은 이번 범위 밖.
- Evidence Links: [workflow](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/.github/workflows/cpu-checks.yml), [검사 도구](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/tools/ci/run_cpu_checks.py), [실행 안내](../tests/README.md)
- Related Issues: [PR #110](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/pull/110). 사용자 승인 계획에 따른 CI 도입.

## 승인된 실행 계약

- 기준 main: `bfa584dcf7f25b597d3dacc664f872f9d57bfb0d`, 브랜치: `codex/cpu-checks-ci`.
- 무료 표준 GitHub 실행기 `ubuntu-24.04`, Python 3.12, 작업 1개. 새 가상환경에 기존 `requirements-review.txt` 설치와 `pip check`.
- main 대상 PR 생성·갱신·재개, main push, 수동 실행. 경로 필터 없음. 같은 PR·브랜치의 이전 실행 취소. 작업 10분, 모듈·예제 각각 2분 제한.
- `contents: read`, checkout 인증정보 지속 저장 해제. 공식 액션은 승인한 커밋 고정. 로그·작업 요약만 보관하고 artifact·Actions 캐시는 추가하지 않는다.
- 고정된 세 모듈과 예제를 최대 3개 독립 프로세스로 검사한다. 최초 제품 검사 기준은 8+7+19=34개이며 이후 모듈 내 추가 검사는 자동 포함한다.
- 수집·시작·종료 식별자와 결과 수를 대조한다. 가져오기 실패·0개·중복·누락·건너뜀·예상된 실패·결과 기록 누락·비정상 종료·시간 초과를 실패로 처리한다.
- 한 검사 실패 후 다른 결과도 수집한다. 시간 초과·취소 시 해당 실행의 자식 프로세스를 종료하고 회수한다.
- 임시 자료를 사용하는 도구 자체 검사를 먼저 실행한다. 제품 코드·기존 테스트 본문·스키마·환경 파일은 수정하지 않는다.
- `sol_reviewer` (`gpt-6.1-sol / xhigh`)가 이벤트·권한·액션 고정·실패 전파를 읽기 전용으로 검토한다.
- 최종 PR 후보 CPU checks·Secret Scan 성공 및 차단 결함 해소 후 일반 merge. main CPU checks·Secret Scan·Pages 확인 후 이번 작업 브랜치만 삭제한다.
- 실패하면 원인·실행 기록을 PR에 보존한다. 자체 서버·유료 실행기로 바꾸거나 실패를 무시해 병합하지 않는다.
- Split Trader·실행기 서버·계정·보호 설정·다른 브랜치는 변경하지 않는다. GPU·실제 DB·외부 수집·성능 재측정·문서 링크 자동 검사는 제외한다.

## 체크리스트

- [x] main·origin·미커밋 변경·작업 브랜치 확인
- [x] 검사 도구·실패 감지 자체 검사·workflow 구현
- [x] 안내 명령·검사 범위·배지 갱신
- [x] 원격 개발 환경의 도구 자체 검사·34개 제품 검사·예제·pip check
- [x] 지정 리뷰어 검토와 차단 결함 해소
- [x] PR 구현 후보의 실제 CPU checks·Secret Scan

최종 문서 커밋까지 포함한 PR 검사, 일반 병합 후 main CPU checks·Secret Scan·Pages·공개 안내 확인, 이번 작업 브랜치 정리는 **PR #110 본문의 최종 결과**에 기록한다. 병합 이후에 결정되는 SHA·실행 ID를 기록하기 위해 검증 완료한 소스를 다시 변경하지 않는다.

## 작업 로그

- 2026-11-06: 대상 저장소가 main 기준 깨끗한 상태임을 확인하고 별도 작업 브랜치를 생성했다. 기존 Secret Scan·Pages와 Split Trader는 유지한다. 이미 준비된 원격 개발 환경을 사용하며 사용자 PC에는 환경을 설치하지 않는다.
- 2026-11-06: Python 3.12.3, NumPy 2.2.6, pandas 2.2.3, Requests 2.32.5의 원격 개발 환경에서 아래 검증을 수행했다. 실제 GitHub 실행 결과는 PR에 이어 기록한다.
- 2026-11-06: `sol_reviewer` (`gpt-6.1-sol / xhigh`) 읽기 전용 검토 완료. 이벤트·권한·액션 SHA·실행 제한·불완전 결과 거부·독립 결과 집계·Linux 프로세스 그룹 종료를 대조했고 차단 결함 없음. 자체 검사 22개를 재실행해 통과했다. 실제 GitHub 후보 검증은 별도 게이트로 유지한다.

## 개발 환경 검증

| 명령 | 결과 |
| --- | --- |
| `.venv/bin/python -m unittest discover -s tools/ci/tests -p 'test_run_cpu_checks.py' -v` | 도구 자체 검사 22개 통과, 실패·건너뜀 0 |
| `.venv/bin/python tools/ci/run_cpu_checks.py` | 기존 검사 8+7+19=34개와 예제 통과, 실패·오류·건너뜀·예상된 실패 0, 종료 코드 0 |
| `.venv/bin/python -m pip check` | `No broken requirements found.` |
| `git diff --check` | 공백 오류 없음 |

자체 검사는 임시 자료에서 실패·오류·건너뜀·예상된 실패·예상 밖 성공, 모듈 가져오기 오류, 0개·중복·실행 누락, 종료 코드 0인 불완전 결과, 성공 결과와 비정상 종료의 불일치, 시간 초과·취소와 자식 프로세스 정리를 확인한다. 자체 검사에서 의도적으로 만든 FAIL 결과는 실제 CPU 검사 요약과 분리한다.

실행 코드와 계산 규칙은 그대로다. GPU·실제 DB·외부 수집·성능 재측정은 필요한 환경·인증이 이번 범위에 없으므로 실행하지 않았다. Ubuntu 외 운영체제의 프로세스 정리는 이번에 검증하지 않았다.

## 실제 GitHub 실행

- 구현 후보: `cf4988fbcd6da65f7c5c36de04daf42e304f90a5`.
- PR의 실제 checkout: `3e689ac0946fc458ade6e5ebe0ba54a5d778de41` (GitHub가 만든 main과의 임시 병합 커밋).
- [CPU checks](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/actions/runs/42112535646): 성공. 표준 GitHub 실행기 `ubuntu-24.04`, 실제 OS Ubuntu 24.04.5, Python 3.12.16. NumPy 2.2.6 / pandas 2.2.3 / Requests 2.32.5.
- 새 가상환경 설치·`pip check` 통과. 자체 검사 22개(2.744초), 기존 검사 8+7+19=34개, 예제 모두 통과. 실패·오류·건너뜀·예상된 실패·예상 밖 성공 0. 각 대상 종료 코드 0.
- [Secret Scan](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/actions/runs/42112535893): 성공.
- 위 기록 추가는 문서 변경이며 실행 코드·workflow·검사 본문은 동일하다. 최종 후보와 병합 후 증거는 [PR #110](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/pull/110)에 구분해 남긴다.
