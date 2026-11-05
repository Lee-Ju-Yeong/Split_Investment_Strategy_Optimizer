# Operations Docs Policy

이 디렉터리는 운영 절차와 당시의 검증 기록을 보관합니다. 처음 방문한 사람은 [전체 실행 안내](../running.md)부터 읽습니다. 날짜가 붙은 문서의 승인·성공·진행 상태는 해당 기록 시점에 한정됩니다.

## 문서 지도

- [수집·DB 터널 절차](krx_hotspot_db_tunnel_runbook.md): 공개 명령 템플릿과 접근 조건
- [로컬 설정 예제](krx_hotspot_db_tunnel_runbook.local.example.md): 실제 값 없이 보관하는 템플릿
- [기간별 평가와 승인 절차](2026-03-14-wfo-approval-runbook.md): 당시 평가 경계와 후속 구현 근거
- [CPU·GPU 비교 기록](2026-03-06-parity-gate-universe-mode-report.md), [당시 승인 항목표](2026-03-06-hybrid-release-gate-board.md): 특정 실행의 결과
- [강건한 파라미터 선정 계획](2026-03-01-robust-parameter-pert-cpm-plan.md), [종목 등급 파생 지표 검토](2026-02-28-tier-derived-features-decision-brief.md): 계획·판단 기록

현재 구현 요약은 [TODO](../../TODO.md), 대표 구현의 코드·테스트·변경 기록은 [구현 근거](../engineering-evidence.md)를 확인합니다.

## 공개 문서

공개 저장소에 커밋 가능한 내용:

1. 절차/명령 템플릿
2. 판정 기준(예: HTTP code별 조치)
3. 일반 보안 원칙

## 비공개 문서

공개 저장소에 커밋하면 안 되는 내용:

1. 실제 DB/SSH 계정, 비밀번호, 토큰
2. 내부 호스트명/사설 IP/터널 실주소
3. 운영 환경 고유 식별값

비공개 문서는 아래 경로/패턴으로만 저장합니다.

1. `docs/operations/*.local.md`
2. `docs/operations/private/`

해당 경로는 `.gitignore`로 제외되어야 합니다.

## Runbook 구성 원칙

1. 공개판: `krx_hotspot_db_tunnel_runbook.md`
2. 로컬 템플릿: `krx_hotspot_db_tunnel_runbook.local.example.md`
3. 실제 환경값은 템플릿을 복사한 `*.local.md`에만 기록

## 권장 로컬 파일명

1. `docs/operations/krx_hotspot_db_tunnel_runbook.local.md`
2. `docs/operations/private/krx_hotspot_db_tunnel_values.md`

두 파일 모두 `.gitignore` 패턴으로 커밋에서 제외됩니다.
