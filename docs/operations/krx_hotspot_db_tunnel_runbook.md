# KRX Hotspot + DB Tunnel Runbook (Public)

## 목적

KRX `market_cap`/`short_selling` 수집 시 서버 egress 차단(예: `403`)이 발생할 때,
노트북 + 모바일 핫스팟 + DB 터널로 우회 실행하는 표준 절차입니다.

이 문서는 공개 저장용이며 실제 자격증명/실 IP/내부 호스트명은 포함하지 않습니다.

## 보안 원칙

1. 비밀번호/토큰/실 IP/내부 호스트명은 본문에 기록하지 않습니다.
2. 민감값은 `docs/operations/*.local.md` 또는 `docs/operations/private/`에만 기록합니다.
3. `config/config.yaml`, `config.ini`는 gitignored 상태를 유지합니다.
4. DB 비밀번호 변경 후에는 즉시 rotate + 로컬 설정 업데이트를 수행합니다.

## 사전 조건

1. 모바일 핫스팟 연결 완료
2. 노트북 실행 환경 준비 (`pykrx`, `pymysql`, `mysql` client)
3. 운영 DB 서버 SSH 접근 가능
4. `config/config.yaml` 또는 `config.ini` 로컬 설정 완료

## 1) 네트워크 판별 (IP/차단 여부)

### 1-1. egress IP 확인

```bash
curl -fsS https://api.ipify.org; echo
```

### 1-2. KRX endpoint probe

```bash
ENDPOINT='https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'

curl -sS -o /tmp/krx_market_cap_probe.out -w "market_cap http=%{http_code}\n" "$ENDPOINT" \
  -H 'User-Agent: Mozilla/5.0' \
  -H 'Referer: http://data.krx.co.kr/contents/MDC/MDI/mdiLoader' \
  -H 'Origin: http://data.krx.co.kr' \
  --data 'bld=dbms/MDC/STAT/standard/MDCSTAT01701&locale=ko_KR&mktId=ALL&trdDd=20260206&share=1&money=1&csvxls_isNo=false'

curl -sS -o /tmp/krx_short_probe.out -w "short_selling http=%{http_code}\n" "$ENDPOINT" \
  -H 'User-Agent: Mozilla/5.0' \
  -H 'Referer: http://data.krx.co.kr/contents/MDC/MDI/mdiLoader' \
  -H 'Origin: http://data.krx.co.kr' \
  --data 'bld=dbms/MDC/STAT/srt/MDCSTAT30001&locale=ko_KR&searchType=1&mktId=ALL&trdDd=20260206&csvxls_isNo=false'
```

### 1-3. 판정 기준

1. `403`: egress 차단 가능성 높음, 네트워크 변경 필요
2. `400 LOGOUT`: 세션 맥락 이슈 가능, pykrx 실호출로 최종 판단
3. `200` + JSON: 네트워크 통과

## 2) DB 접근 경로 준비

### 2-1. 기본 터널 (Linux/macOS 공통)

```bash
ssh -N -L 13306:127.0.0.1:3306 <SSH_USER>@<SSH_HOST>
mysql --protocol=TCP -h 127.0.0.1 -P 13306 -u <DB_USER> -p -e "SELECT NOW();"
```

### 2-2. Windows + WSL 혼합 경로 (실운영에서 자주 발생)

조건:
1. 운영 DB가 원격 Desktop의 WSL MySQL(`0.0.0.0:3306`)에 있음
2. 노트북은 WSL에서 파이프라인 실행, Windows PowerShell에서 SSH 터널 실행

원격 Desktop(운영DB 호스트) 준비:

```powershell
# SSH 서버 확인
Get-Service sshd

# WSL MySQL 포트 확인
wsl -e bash -lc "ss -lntp | grep 3306"
```

노트북 Windows PowerShell(관리자)에서 터널 실행:

```powershell
# WSL -> Windows 터널 접근 허용 (최초 1회)
New-NetFirewallRule -DisplayName "WSL-MySQL-Tunnel-13306" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 13306 -Profile Any

# 원격 Desktop WSL DB IP로 포워딩
ssh -N -g -L 0.0.0.0:13306:<REMOTE_WSL_DB_IP>:3306 <SSH_USER>@<DESKTOP_TAILSCALE_IP>
```

노트북 WSL에서 접속 확인:

```bash
WIN_IP=$(ip route | awk '/default/ {print $3; exit}')
mysql --protocol=TCP -h "$WIN_IP" -P 13306 -u <DB_USER> -p \
  -e "SELECT USER(),CURRENT_USER(),@@hostname,@@port;"
```

### 2-3. MySQL 계정 정리 템플릿

```sql
CREATE USER IF NOT EXISTS '<DB_USER>'@'localhost' IDENTIFIED BY '<STRONG_PASSWORD>';
CREATE USER IF NOT EXISTS '<DB_USER>'@'127.0.0.1' IDENTIFIED BY '<STRONG_PASSWORD>';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX ON <DB_NAME>.* TO '<DB_USER>'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX ON <DB_NAME>.* TO '<DB_USER>'@'127.0.0.1';
FLUSH PRIVILEGES;
```

### 2-4. 자주 보는 오류 코드

1. `ssh timeout` / `TCP22_BLOCKED`: Tailnet/SSH 경로 문제
2. `ERROR 2003 (111)`: 로컬 터널 포트 미리스닝 또는 터널 세션 종료
3. `ERROR 2003 (110)`: WSL->Windows 경로 또는 `-L` 대상 IP 오설정
4. `ERROR 1045`: 비밀번호 오입력 또는 계정 host 매핑(`localhost`/`127.0.0.1`) 불일치

## 3) 수집 실행

```bash
python -m src.pipeline_batch \
  --mode backfill --start-date 20131120 --end-date 20260206 \
  --run-marketcap --run-shortsell \
  --skip-financial --skip-investor --skip-tier
```

참고: `src/db_setup.py`는 `config.ini`의 `[mysql] port` 값을 읽으며, 미설정 시 기본 3306을 사용합니다.

## 4) 결과 검증

```sql
SELECT COUNT(*) AS market_cap_rows FROM MarketCapDaily;
SELECT COUNT(*) AS short_rows FROM ShortSellingDaily;
SELECT MIN(date), MAX(date) FROM MarketCapDaily;
SELECT MIN(date), MAX(date) FROM ShortSellingDaily;
```

## 5) 증적 기록 (운영 로그)

민감값을 제외하고 아래 항목만 이슈/로그에 남깁니다.

1. 실행 시각 (로컬 시간)
2. egress IP
3. KRX probe HTTP code
4. 실행 커맨드 (민감값 제외)
5. 적재 row 수 및 min/max date

## 6) 관련 문서

1. `docs/database/backfill_validation_runbook.md`
2. `docs/operations/README.md`
3. `docs/operations/krx_hotspot_db_tunnel_runbook.local.example.md`
4. `TODO.md` (Issue #71/#67 진행 상태)
