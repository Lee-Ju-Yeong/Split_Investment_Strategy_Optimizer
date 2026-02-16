# KRX Hotspot + DB Tunnel Runbook (Public)

## 목적

KRX `market_cap`/`short_selling` 수집 시 서버 egress 차단(예: `403`)이 발생할 때,
노트북 + 모바일 핫스팟 + DB 터널로 안전하게 우회 실행하는 표준 절차입니다.

이 문서는 공개 저장용이며, 실제 자격증명/호스트/IP는 포함하지 않습니다.

## 보안 원칙

1. 비밀번호/토큰/실 IP/내부 호스트명은 문서 본문에 기록하지 않습니다.
2. 실행 파라미터의 민감값은 환경변수 또는 로컬 비공개 문서로만 관리합니다.
3. `config/config.yaml`, `config.ini`는 gitignored 상태를 유지합니다.
4. 로컬 메모는 `docs/operations/*.local.md` 또는 `docs/operations/private/`에만 저장합니다.

## 사전 조건

1. 모바일 핫스팟 연결 완료
2. 노트북에서 운영 DB 접근 가능(직접 또는 SSH 터널)
3. Python 실행 환경 준비(`pykrx`, `pymysql` 등)
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

1. `403`: egress 차단 가능성 높음 -> 네트워크 변경 필요
2. `400 LOGOUT`: 세션/요청 맥락 이슈 가능 -> pykrx 실제 호출로 최종 판단
3. `200` + JSON: 네트워크 통과

## 2) DB 접근 경로 준비

## 2-1. SSH 터널 권장

```bash
# 예시 (값은 로컬 비공개 문서에만 보관)
ssh -N -L 13306:127.0.0.1:3306 <ssh_user>@<ssh_host>
```

## 2-2. DB 연결 테스트

```bash
python - <<'PY'
from src.db_setup import get_db_connection
conn = get_db_connection()
with conn.cursor() as cur:
    cur.execute("SELECT NOW()")
    print(cur.fetchone())
conn.close()
PY
```

## 2-3. DB 비밀번호 회전(권장)

운영 DB 계정 비밀번호를 주기적으로 회전합니다. 아래 SQL은 템플릿입니다.

```sql
ALTER USER '<DB_USER>'@'127.0.0.1' IDENTIFIED BY '<NEW_STRONG_PASSWORD>';
FLUSH PRIVILEGES;
```

회전 후에는 로컬 `config/config.yaml` 또는 `config.ini`를 즉시 갱신합니다.
새 비밀번호는 공개 문서/이슈 코멘트에 기록하지 않습니다.

## 3) 수집 실행

```bash
python -m src.pipeline_batch \
  --mode backfill --start-date 20131120 --end-date 20260206 \
  --run-marketcap --run-shortsell \
  --skip-financial --skip-investor --skip-tier
```

## 4) 결과 검증

```sql
SELECT COUNT(*) AS market_cap_rows FROM MarketCapDaily;
SELECT COUNT(*) AS short_rows FROM ShortSellingDaily;
SELECT MIN(date), MAX(date) FROM MarketCapDaily;
SELECT MIN(date), MAX(date) FROM ShortSellingDaily;
```

## 5) 증적 기록 (운영 로그)

아래 항목을 이슈 코멘트 또는 비공개 운영 노트에 남깁니다.

1. 실행 시각(로컬 시간)
2. egress IP
3. KRX probe HTTP code
4. 실행 커맨드(민감값 제외)
5. 적재 row 수 및 min/max date

## 6) 실패 대응

1. 핫스팟에서도 `403`이면 다른 통신사/네트워크로 전환
2. DB 연결 실패면 SSH 터널/ACL 먼저 복구
3. Python 의존성 실패면 venv 재생성 후 최소 패키지 설치

## 7) 관련 문서

1. `docs/database/backfill_validation_runbook.md`
2. `TODO.md` (Issue #71/#67 진행 상태)
3. `docs/operations/krx_hotspot_db_tunnel_runbook.local.example.md`
