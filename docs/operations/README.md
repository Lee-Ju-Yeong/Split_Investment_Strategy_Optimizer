# Operations Docs Policy

이 디렉터리는 운영 절차 문서를 관리합니다.

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
