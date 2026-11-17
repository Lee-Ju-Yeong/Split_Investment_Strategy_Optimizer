# ETF M3 빠른 커널

> 유형: `implementation`
> 상태: `완료 — C++ 커널·M2 정확 대조·성능 측정·필수 리뷰 통과`
> 시작일: `2026-11-17` (이 환경 UTC)
> 완료일: `2026-11-17` (이 환경 UTC, 선행 M2 문서의 2026-11-18 기록은 원문 보존)
> 기준: M2 `e241371` / 엔진 구현 `d14af43`

## 목표와 설계

승인된 [연구 계획](2026_11_15-etf-vol-split-research-plan.md)의 M3를 수행한다.
현재 작업 폴더는 M1 일부인 `a22490d`였으므로 원격 연구 브랜치를 읽고, M2 완료본
`e241371`에서 `feat/etf-m3-fast-kernel` 브랜치를 만들었다. 기존 브랜치와 M1 자료는 보존했다.

- C++ 커널로 일봉의 매매 판단·최초 주문 고정·체결 후보를 계산한다. 호가 전수 순회와
  유리수 연산이 집중된 부분이며, 다중정밀도 정수비로 Decimal/Fraction 경계를 보존한다.
- Python의 UTC 사건 큐·현금·환전·결제·기업행사·세금 원장은 공유한다. 커널은 현금이나
  보유를 변경하지 않는다. 서로 다른 실행이 전역 함수를 교체하지 않도록 인스턴스로 연결한다.
- 호가·비용은 M2 시장 규칙에서 전달한다. 고정 소스와 빌드 정보·해시를 남기고 Linux/WSL에서
  명시적으로 컴파일한다. import나 일반 CPU 검사에서 컴파일하지 않는다.
- 기존 CPU/GPU 제품 계산·프로토콜·M1 원본·보류 분할·실제 거래 계정은 변경하지 않는다.

## 완료·검증 조건

1. 두 장중 경로, 세 시장의 갭/터치/호가 경계, 순수익률 반올림, 수량·비용·차수 빈칸,
   미체결 주문 교체 금지와 마이크로초 체결 시각을 기준 엔진과 정확히 대조한다.
2. M2의 독립 손계산·제품 고정 사례를 커널에도 적용한다. 혼합시장·환전·결제·월 예산·합성
   세금의 전체 EngineResult(거래·거절·평가·현금·보유·세금·경고)도 대조한다.
3. 고정 시드 임의 사례, 미래 자료 변경과 입력 거부를 확인한다. 허용 오차는 0이다.
4. 같은 합성 입력의 반복 실행으로 기준/커널/전체 실행 시간을 각각 기록한다. 컴파일 시간은
   실행 시간과 구분한다. 속도 향상을 실제 대규모 탐색이나 수익성 검증으로 확대하지 않는다.
5. 기존 CPU 전체 계약 검사와 실행기 검사를 통과하고 `sol_reviewer` 필수 리뷰의 차단 결함을
   해소한다. 위험 통합 지점과 최종 후보를 서로 다른 범위로 검토한다.

## 중단·재개와 경계

불일치가 나면 최초 입력·실패·실행 로그를 남기고 같은 고정 사례로 재현한다. M2 규칙을 바꾸거나
허용 오차를 늘려 맞추지 않는다. 계약 변경이 필요하면 근거와 범위를 재검토한다. 두 해결 시도가
실패하면 DIAGNOSE를 요청한다. 재개 시 실제 diff·소스 해시·로그를 먼저 대조한다.
상폐 청산·KR T+2·완결 세후 보고 등 [M2 이후 실행 조건](2026_11_15-etf-vol-split-research-plan.md#M2-이후-실행-조건)은
그대로 유지한다. 이번은 합성 자료 검증이며 실제 탐색/보류 평가를 시작하지 않는다.

## 진행과 근거

- 시작: 미커밋 변경 없음. 원격 M2 완료 기록과 코드 대조. Ubuntu 24.04 / Python 3.12.3 /
  g++ 13.3.0 / Boost 1.83.0을 준비하고 `.venv`에 두 고정 requirements를 설치했다. `pip check` 통과.
- `engine.py` 변경은 후보 함수를 인스턴스에 연결하는 3곳뿐이다. 기본은 기존 Python 함수다.
  C++ 계산과 연결·빌드·성능 도구, 별도 CI 작업과 엄격한 검사 실행기를 추가했다.
- 초기 후보는 native 145개 통과. 초기 리뷰가 찾은 4건과 사전에 확인한 수량 반올림 차이를 보완했다.
- r3의 추가 수량 검사 1건은 기대한 입력이 80자리 밖이 아니라 안에 있어 기준·커널 모두 1주였으나
  독립 기대값은 2주로 적힌 실패다. 기존 M2 기대값은 건드리지 않고, 의도한 80자리 밖 입력으로
  바로잡아 r4 149개와 엄격 실행기 149개를 통과했다. 최초 실패는 `native-r3.log`에 보존한다.
- 합성 세금의 결제·연말까지 전체 EngineResult 대조를 추가해 150개도 통과했다.
  최종 리뷰에서 추가 발견한 매수 수수료율 사전 반올림·호출자 반올림 방식 차이를 수정하고
  ABI 2로 새 빌드했다. 매수는 원래 수수료율을 전달하고, Decimal의 8가지 반올림 방식을 전달한다.
  수익률 판단 자체의 HALF_UP 규칙은 그대로다. 긴 수수료율 최초 반례는 `raw-fee-repro.log`에 보존한다.

## 리뷰와 조치

사용자 지침에 따라 `sol_reviewer` (`gpt-6.1-sol`, `xhigh`)를 읽기 전용으로 호출했다.
초기는 주문 판단 통합 지점, 후속은 보완 반례·새 정밀도 연산·빌드/CI·검증 근거를 검토했다.
마지막 보완 검토는 긴 수수료율·8가지 반올림·정확한 중간값을 직접 대조하고, 소스8개/빌드 해시와
최신153개 검사·성능 결과를 확인해 **이번 M3 범위에 남은 차단 결함 없음**을 보고했다.

| 발견 | 원인과 조치 | 회귀 |
|---|---|---|
| 가격 덧셈 반올림 | Fraction 완전 정확 덧셈과 M2 Decimal80 덧셈 차이. 각 단계의 Decimal80과 next_minus 재현 | `test_decimal80_step_rounding_changes_gap_qualification` |
| 시가 출력 자릿수 소실 | 분자/분모 Decimal 나눗셈이 원래 시가를 80자리로 줄임. 유한소수를 정확히 구성 | `test_manual_open_keeps_original_decimal_digits_and_lazy_date_lookup` |
| 불필요한 날짜표 조회 | 슬리피지 0 수동 시가 진입은 원래 표를 안 읽음. 필요한 경우만 조회 | 위 수동 진입 검사 |
| 관통·슬리피지 상한 추가 | 배열 수 검사용 상한이 옵션에도 적용됨. 옵션은 다중정밀도 정수로 처리 | `test_unused_manual_penetration_has_no_new_upper_bound` |
| 매수 수수료율 사전 반올림 | 단위 금액 수수료를 요율로 전달. 매수에는 원래 요율, 실제 금액 곱셈 뒤 반올림 | `test_raw_buy_commission_is_rounded_after_multiplying_notional` |
| 호출자 반올림 미보존 | HALF_EVEN 고정 대신 호출자의 Decimal 반올림 방식 전달 | `test_callers_decimal_rounding_mode_is_preserved`, `test_decimal_half_tie_follows_each_rounding_mode` |

## 검증 근거

로컬 증거 루트: `results/etf-m3-validation/` (Git 제외). 외부 계정·실제 자료·DB·GPU를 사용하지 않는다.

| 명령/대상 | 결과 | 증거 |
|---|---|---|
| `.venv/bin/python -m pip check` | PASS | 실행 출력, 의존성 고정 파일 |
| `.venv/bin/python tools/ci/run_cpu_checks.py` | CPU480개·예제3개 PASS, 실패/누락/건너뜀0 | `cpu-r1.log` |
| `.venv/bin/python -m unittest discover -s tools/ci/tests -p test_run_cpu_checks.py -v` | 실행기24개 PASS | `runner-r1.log` |
| ABI1 엄격 native 검사 | 149개 PASS, 합성세금 추가 후150개 PASS | `native-strict-r1.log`, `native-strict-final.log` |
| ABI2 빌드·긴 수수료율 회귀 | 빌드 성공·기준/커널 BUY1주 일치 | `build-v2/build.json`, 최초 실패 `raw-fee-repro.log` |
| ABI2 최종 엄격 native 검사 | 153개 PASS, 수집·시작·종료 일치, 실패/오류/건너뜀0 | `native-v2.log` |
| ABI2 성능 | 두 경로 후보·전체 결과 정확 일치, 3회 중앙값 | `benchmark-v2.json`, `benchmark-v2.log` |
| 최종 소스·보호 대상 대조 | 소스8개 해시 일치, M2 기준/세금/입력/프로토콜/시장표/분할/기대값7개 원문 불변 | `source-v2.json`, 기준 `e241371` |
| 변경 검사 | `git diff --check` PASS | 로컬 실행 출력 |

153개는 M2 오라클137개를 C++ 경로로 실행한 검사와 새16개다. 새 검사 안에서 고정 시드 후보360건,
호가 전환36건, 혼합시장 전체 원장4건, 합성세금 전체 원장2건과 8가지 반올림/중간값을 대조했다.
기존 CPU480개·예제3개는 이번 환경에서 직접 실행했다. 최종 보완은 C++ 전용 파일·검사만 바꿨고
기존 CPU 선택 대상과 실행기는 불변이므로 전체 CPU 검사를 불필요하게 반복하지 않고 그 근거를 재사용했다.

| 최종 합성 측정 | 기준 엔진 중앙값(초) | C++ 경로 중앙값(초) | 속도 향상 |
|---|---:|---:|---:|
| O-H-L-C 후보3개 | 0.204987 | 0.052933 | 3.87배 |
| O-L-H-C 후보3개 | 0.206333 | 0.050135 | 4.12배 |
| O-H-L-C 전체24일봉 | 1.115358 | 0.244851 | 4.56배 |
| O-L-H-C 전체24일봉 | 1.282850 | 0.298021 | 4.30배 |

이전 ABI1 r4 측정은 준비 실행 뒤 3회 중앙값에서 후보3.35~4.05배, 전체 포트폴리오3.71~4.82배였다.
같은 합성 입력의 결과는 두 경로 모두 허용 오차0으로 일치했다. 이는 최종 ABI2 측정과 구분한다.
컴파일 시간·실제 자료 탐색 처리량·투자 성과는 포함하지 않는다. 초기 측정·실패 로그는 덮어쓰지 않는다.

## 완료와 인계

- [사용·빌드·검사 안내](../docs/research/etf_m3_kernel.md). 커널은
  `results/etf-m3-validation/build-v2/`에 보존한다. ABI2 C++ 소스 SHA256은
  `7499e8d9c9ebc31d74592c08ba49f7b9132e4eb8e524380eb792ac64815168dd`다.
- 원격 CI 작업을 추가했으나 push나 GitHub CI 실행은 하지 않았다. 운영/실제 세후/수익성 승인이 아니다.
- 다음은 M4 리밸런싱·보유·프런티어 기준선이다. M5 전에 상장폐지 청산·국내 결제·자료 완전성과
  기존 실행 조건을 충족해야 한다. 보류 평가와 실제 자료 탐색은 시작하지 않았다.
- 새 규칙·시장표·Decimal 계산·커널·연결부가 바뀌면 같은 입력과 최신 빌드로 정확 대조를 재실행한다.

## 사용자 추가 요청: Opus 독립 리뷰

- 2026-11-17 사용자 요청으로 Opus 읽기 전용 검토를 제출했다. 기존 sol 리뷰와 별도 요청이며,
  Opus가 검토했다고 기록하거나 기존 결과로 대신하지 않는다.
- 요청서: `results/etf-m3-validation/opus-review-spec.md`. 최신 diff·untracked 파일·계약·검토 질문·
  실제 검증/실패 로그·8개 소스 해시와 보호 범위를 전달했다.
- Orca Run `run_8fe20dc790c3`, Task `task_2e81daa7de0f`, Dispatch `ctx_fe5da034de18`,
  agent `claude`, model `opus`, effort `xhigh`로 시작을 시도했다.
- **1차 시작 실패:** `agent_readiness`에서 `bash: claude: command not found`로 종료했다.
  `orca account list`도 Claude 계정/로그인 정보 없음으로 확인됐고, 연결된 다른 환경·호스트도 없다.
  단순 재시도로 해결되는 상태가 아니므로 같은 시작을 반복하지 않았다.
- 실패 receipt는 `results/etf-m3-validation/opus-start.json`에 보존했다. 남은 실패 작업 터미널은
  receipt 지시대로 `worker-release`하여 `state=released`, `processAction=closed_agent_terminal`을 확인했다.
- 사용자 후속 지시로 이 호스트에 공식 native stable 설치기로 Claude Code **2.1.311**을 설치했다.
  `/root/.local/bin/claude`가 PATH에서 확인되고 `claude doctor`는 `No installation issues found`다.
  설치·진단·바이너리 해시 기록은 저장소 밖 `/mnt/volume/workspace/.host-admin/claude-opus-setup/`에 둔다.
  기존 Claude 설정과 M3 코드는 변경하지 않았다.
- 설치 직후 기본 프로필의 `claude auth status`는 `loggedIn=false`, `authMethod=none`이었다. 사용자 계정 로그인을 위해
  Orca 터미널 `Claude 로그인 · Opus 리뷰`에서 `orca account add --agent claude --label opus-review`를 시작했다.
  터미널 handle `term_947d4349-c1cb-45f3-bfed-f790128f53b8`에서 공식 로그인 흐름을 열었다.
  터미널은 실행 중이나 focus 결과 `navigated=false`라 사용자 화면 전환까지 됐다고 주장하지 않는다.
  비밀번호·인증 코드를 채팅이나 저장소에 받지 않고 해당 공식 로그인 흐름에서만 입력한다.
- 사용자에게 전용 로그인 탭이 보이지 않아 기존 일반 `Terminal` 세션으로 로그인 절차를 옮겼다.
  기존 전용 로그인은 Ctrl+C 후 종료 코드130·빈 shell 프롬프트를 확인했다. 새 로그인은
  `term_8b37b154-5e31-48ed-a174-1a8dcf80e9ce`에서 공식 OAuth URL과 `Paste code here if prompted`를
  표시했다. 당시 로그인 URL/터미널 상태는 저장소 밖 관리 폴더의
  권한0600 `active-login.json`에만 보존한다. 계정 코드/토큰은 문서나 채팅으로 받지 않는다.
- 사용자 로그인 성공 후 Orca 관리 프로필에서 `loggedIn=true`, `authMethod=claude.ai`,
  Claude Max를 확인했다. 기본 프로필과 다르므로 관리 프로필의 `CLAUDE_CONFIG_DIR`를 유지한다.
- **2차 시작 실패:** 같은 Task의 재시도 Dispatch `ctx_4b36ccf0563b`는 Orca 기본 실행 명령의
  `--dangerously-skip-permissions`를 Claude가 root에서 거부해 종료했다. 실패 receipt는
  `results/etf-m3-validation/opus-retry-start.json`에 보존했고 실패 터미널은 `worker-release`했다.
- 서로 다른 시작 실패 2회에 따라 `sol_reviewer` DIAGNOSE를 수행했다. 실행 파일·인증·실행 옵션을
  대조했고, 권한 검사 생략 옵션 없이 준비한 터미널을 같은 Task에 연결하는 문서화된 경로를 확인했다.
  제품 코드·인증 파일을 수정하거나 실패 이력을 초기화하지 않았다.
- 저장소 밖 실행 스크립트에서 기존 Orca 플러그인·관리 프로필을 유지하고 `--permission-mode plan`,
  `--tools Read,Glob,Grep,Bash`로 실행했다. 화면에서 **Opus 5.5 / xhigh / Claude Max / plan mode on**과
  `tui-idle` 준비 상태를 확인했다. `--allowedTools`는 자동 허용 규칙이며 읽기 전용 검토 계약도 유지한다.
- 터미널 `term_b5d49464-a6a2-4666-8b53-fd7200f1eb20`을 같은 Run·Task에
  `--retry-of ctx_4b36ccf0563b --terminal ...`로 연결했다. **현재 Dispatch `ctx_1ef220491fda`,
  `state=ready`로 리뷰를 시작했으며 결과 대기 중이다.** 성공 receipt는
  `results/etf-m3-validation/opus-safe-start.json`에 보존했다. 리뷰 시작을 검토 완료·통과로 간주하지 않는다.
- **Opus 검토 완료:** Dispatch `ctx_1ef220491fda`가 `worker_done / succeeded`를 보고했다.
  Opus 5.5/xhigh가 M3 8개 소스·빌드 해시, 기존 로그·CI와 M2 대비 차이를 검토하고 기존 바이너리로
  세 시장·두 경로·8개 반올림 등의 합성 반례5,500건을 대조했다. **P1/P2 없음**이다.
  새 빌드·전체153/480 재검사·실제 자료 검증은 수행하지 않았고 M4 세무 변경은 제외했다.
- P3 두 건은 보완했다. Decimal 지수/후행0 문자열 표현은 보존하지 않을 수 있어 `etf_m3_kernel.md`에
  정확 일치를 수치 동치로 명시했다. `TODO.md`의 인증/리뷰 대기 상태는 완료로 갱신했다.
