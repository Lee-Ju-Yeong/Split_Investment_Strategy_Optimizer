# KRX Hotspot + DB Tunnel Runbook (Local Example, Do Not Commit Secrets)

이 파일은 로컬 개인/팀 환경값을 채우는 템플릿입니다.
실파일은 `*.local.md` 이름으로 복사 후 사용하고 커밋하지 않습니다.

## 환경값 템플릿

1. SSH jump host: `<SSH_HOST>`
2. SSH user: `<SSH_USER>`
3. DB host (tunnel target): `<DB_HOST>`
4. DB port: `<DB_PORT>`
5. DB name: `<DB_NAME>`
6. DB user: `<DB_USER>`

## 터널 명령 템플릿

```bash
ssh -N -L 13306:<DB_HOST>:<DB_PORT> <SSH_USER>@<SSH_HOST>
```

## 로컬 설정 파일

`config/config.yaml` 또는 `config.ini`에 아래를 반영:

```yaml
database:
  host: "127.0.0.1"
  user: "<DB_USER>"
  password: "<DB_PASSWORD>"
  database: "<DB_NAME>"
```

주의: 비밀번호/토큰/내부 주소를 공개 문서에 복사하지 않습니다.
