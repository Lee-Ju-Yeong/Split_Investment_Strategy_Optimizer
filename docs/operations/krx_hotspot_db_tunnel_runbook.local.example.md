# KRX Hotspot + DB Tunnel Runbook (Local Example, Do Not Commit Secrets)

이 파일은 로컬 개인/팀 환경값을 채우는 템플릿입니다.
실파일은 `*.local.md` 이름으로 복사 후 사용하고 커밋하지 않습니다.

## 환경값 템플릿 (실값 기입)

1. Desktop Tailnet IP: `<DESKTOP_TAILSCALE_IP>`
2. SSH user: `<SSH_USER>`
3. Remote WSL DB IP: `<REMOTE_WSL_DB_IP>`
4. Remote WSL DB port: `3306`
5. Local tunnel port: `13306` (테스트), `3306` (파이프라인 기본)
6. DB name: `<DB_NAME>`
7. DB user: `<DB_USER>`
8. DB password: `<DB_PASSWORD>`

## 연결 경로 메모

1. 노트북 WSL -> 노트북 Windows: `<NOTEBOOK_WINDOWS_IP>:13306`
2. 노트북 Windows -> Desktop WSL DB: `<REMOTE_WSL_DB_IP>:3306`
3. SSH hop: `<SSH_USER>@<DESKTOP_TAILSCALE_IP>`

## 터널 명령 (노트북 Windows PowerShell)

```powershell
ssh -N -g -L 0.0.0.0:13306:<REMOTE_WSL_DB_IP>:3306 <SSH_USER>@<DESKTOP_TAILSCALE_IP>
```

파이프라인 실행을 위해 로컬 3306으로 여는 경우:

```powershell
ssh -N -g -L 0.0.0.0:3306:<REMOTE_WSL_DB_IP>:3306 <SSH_USER>@<DESKTOP_TAILSCALE_IP>
```

## 접속 확인 명령

노트북 WSL:

```bash
WIN_IP=$(ip route | awk '/default/ {print $3; exit}')
mysql --protocol=TCP -h "$WIN_IP" -P 13306 -u <DB_USER> -p \
  -e "SELECT USER(),CURRENT_USER(),@@hostname,@@port;"
```

## 로컬 설정 반영

`config/config.yaml` 또는 `config.ini`에 DB 계정을 반영합니다.
이 저장소에서는 `config/config.yaml`, `config.ini`가 gitignored 입니다.

```yaml
database:
  host: "127.0.0.1"
  user: "<DB_USER>"
  password: "<DB_PASSWORD>"
  database: "<DB_NAME>"
```

주의: 이 파일은 로컬 전용입니다. 실값을 넣은 뒤 절대 커밋하지 않습니다.
