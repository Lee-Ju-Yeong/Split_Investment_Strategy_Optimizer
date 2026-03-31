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

### 1-4. 인증 세션 주입 smoke (2026-03-13 확인)

2026-02-27 이후 KRX 접근 정책 변경으로, 비인증 상태에서는 `400 LOGOUT`, `403`, 빈 DataFrame, 에러 HTML이 섞여 나올 수 있습니다.
이 호스트에서는 **비인증 collector preflight는 실패**했고, **로그인 세션을 pykrx에 주입하는 smoke는 성공**했습니다.

따라서 `ShortSellingDaily` 관련 증적을 모을 때는 아래 순서를 권장합니다.

1. 비인증 preflight 결과는 참고만 한다
2. 실제 접근성 판정은 **로그인 세션 주입 smoke**로 한다
3. smoke가 실패하면 해당 환경의 공매도 응답은 lag 근거로 쓰지 않는다

실행 스크립트:

```bash
cd /root/projects/Split_Investment_Strategy_Optimizer
CONDA_NO_PLUGINS=true conda run -n rapids-env \
  python -u tools/operations/check_pykrx_login_session.py \
  --date 20260227 --ticker 005930
```

성공 기준:

1. `login_ok=True`
2. `ETF ticker list` smoke 정상
3. `short status` smoke에서 rows가 1건 이상

참고:
- 로그인 성공 후에도 직접 `getJsonData.cmd` short endpoint는 `{"OutBlock_1":[],"CURRENT_DATETIME":...}` 형태의 빈 JSON이 올 수 있습니다.
- 현재 환경에서는 **직접 short endpoint probe보다, 로그인 세션 주입 후 pykrx smoke를 더 신뢰할 수 있는 판정 경로**로 취급합니다.
- 자격증명은 스크립트 prompt 입력 또는 `KRX_LOGIN_ID`, `KRX_LOGIN_PW` 환경변수로 주입합니다. 쉘 히스토리에 직접 남기지 않습니다.
- 외부 참고:
  - pykrx issue `#276`: https://github.com/sharebook-kr/pykrx/issues/276
  - pykrx issue `#278`: https://github.com/sharebook-kr/pykrx/issues/278

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
  --skip-financial --skip-investor --skip-tier \
  --collector-workers 1 \
  --collector-write-batch-size 500 \
  --collector-delay 4.0 \
  --collector-jitter-max-seconds 2.0 \
  --collector-macro-pause-every 40 \
  --collector-macro-pause-min-seconds 45 \
  --collector-macro-pause-max-seconds 70 \
  --shortsell-prefilter-enabled \
  --shortsell-prefilter-markets KOSPI,KOSDAQ \
  --shortsell-prefilter-min-hits 1
```

참고: `src/db_setup.py`는 `config.ini`의 `[mysql] port` 값을 읽으며, 미설정 시 기본 3306을 사용합니다.
참고: `--shortsell-prefilter-enabled`는 `start_date~end_date` 구간에서 연도별 마지막 거래일 앵커를 사용합니다.
참고: `ShortSellingDaily` 수집은 티커별 조회 구간을 내부적으로 2년 미만(최대 729일) 청크로 분할 호출합니다.
참고: `ShortSellingDaily` 청크는 최신→과거 순으로 호출되며, 데이터가 한 번 나온 뒤 `empty`가 연속 2회 발생하면 해당 티커를 종료합니다.
참고: 재실행 시 티커별 최소 적재일 + `ShortSellingBackfillCoverage(DONE_EMPTY)` 체크포인트를 함께 사용해 미수집 과거 구간만 이어서 백필합니다(resume).

## 4) 결과 검증

```sql
SELECT COUNT(*) AS market_cap_rows FROM MarketCapDaily;
SELECT COUNT(*) AS short_rows FROM ShortSellingDaily;
SELECT MIN(date), MAX(date) FROM MarketCapDaily;
SELECT MIN(date), MAX(date) FROM ShortSellingDaily;
```

## 5) 증적 기록 (운영 로그)

관측 시점:
- 최초 차단 확인 시점(레포 증적 기준): `2026-02-11` (`TODO.md`의 `external_blocked` 운영 결정 기록)
- 오늘 재시도 시점: `2026-02-28 18:12:54 KST` (`market_cap`, `short_selling` probe 모두 HTTP 403 재확인)
- 최신 재시도 시점: `2026-03-07 11:26:39 KST`
  - direct KRX probe: `market_cap http=403`, `short_selling http=403`
  - pykrx 실호출(`rapids-env`): `get_market_ticker_list(20260206/20260224/20260306)=0`, `get_market_cap/get_market_fundamental -> KeyError(empty columns)`, `get_shorting_status_by_date(..., 005930)=shape(0, 0)`, `get_stock_major_changes(005930/000660/035420)=shape(0, 0)`
  - 판정: 서버 egress 기준 차단 미해제. public 문서에는 실 egress IP를 남기지 않고, 로컬 운영 로그에만 기록
- 인증 세션 주입 smoke 확인 시점: `2026-03-13 23:06 KST`
  - 로그인 POST: `CD001 정상`
  - `stock.get_etf_ticker_list(20260227)` 정상 (`etf_count=1072`)
  - `stock.get_shorting_status_by_date(20260227, 20260227, 005930)` rows=`1`
  - direct short endpoint with login session: `http_status=200`, body head=`{\"OutBlock_1\":[],\"CURRENT_DATETIME\":...}`
  - 판정: 이 호스트에서는 **비인증 경로는 불가**, **로그인 세션 주입 후 pykrx smoke는 가능**

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
