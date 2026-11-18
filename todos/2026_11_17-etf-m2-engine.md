# ETF M2 제품 규칙 기준 엔진

> 유형: `implementation`
> 상태: `완료 — M2 기준 엔진·CPU480개·예제3개·Opus r3 차단 없음, M3 대조 기준 사용 가능`
> 시작일: `2026-11-17`
> 완료일: `2026-11-18`
> 근거: [연구 계획](2026_11_15-etf-vol-split-research-plan.md), `src/research/etf/protocol_v1.json` 초안

## 목표와 범위

split-trader V2를 일봉으로 근사한 Python 기준 엔진을 연구 전용으로 추가한다. 국내·미국 ETF와 KRX 금,
두 장중 경로, 원화·달러 현금, UTC 사건 순서와 사전 공표 환율, 비용·세금 원장을 검산할 수 있어야 한다.
기존 제품 CPU/GPU 엔진과 split-trader 저장소는 변경하지 않는다. M3 고속 커널, M4 기준선, M5 탐색,
보류 표본 성과 평가, 실거래·계정·운영 DB 접근은 이번 범위 밖이다.

사용자는 2026-11-17 Orca 멀티워커 병렬 구현과 마지막 Opus 검토를 요청했다.
Goal 생성 요청은 없으므로 이 작업은 일반 구현 작업으로 추적한다.

## 불변조건과 선택 이유

- 연구 기준 엔진은 Decimal로 금액과 비율을 계산한다. 제품 V2의 정확한 원가 비교와 수익률 반올림을
  재현하기 위한 선택이다. 기존 CPU/GPU float32 계약은 변경하지 않는다.
- 제품 근거는 split-trader 커밋 `aaa3bb9`의 `back/src/split_trader/kiwoom_integration/product_trading_policy.py`다.
  인접 저장소에는 다른 작업이 있으므로 현재 미커밋 파일 대신 이 커밋을 읽는다.
- 1차는 시작일 시가에 연구용 수동 진입을 한 번만 한다. 이후 1차를 자동으로 채우지 않는다.
  2..N 중 가장 낮은 빈 차수만 보고 바로 전 차수가 없으면 매수하지 않는다.
- 종목당 거래일 최대 주문 1건. 같은 시점이면 매도 우선. 여러 익절 후보는 제품의 SHA256(symbol:level) 순서.
- 비율은 소수 비율(5%=0.05), 금액은 해당 상품 통화다. 매수 예산은 수수료를 포함한다.
  현금·주문 참여율 부족 시 수량을 줄여 체결하지 않는다. 차수 예산 안의 정수 수량 산정과 구분한다.
- 미래 가격과 아직 공표되지 않은 환율은 판단·평가에 쓰지 않는다. 일봉의 장중 경로를 가정하더라도
  나중의 체결·종가가 앞선 원장이나 18:00 KST 평가에 반영되지 않아야 한다.
- OHLC 경로는 O-H-L-C와 O-L-H-C를 각각 실행한다. 장중 꼭짓점은 세션 길이의 1/3·2/3 시각,
  선형 구간의 최초 체결 가능 시각을 사용한다. 실제 장중 시각을 관측한 것으로 표현하지 않는다.
- 세후 실제 판정은 분배금 지급일·과표기준가 등 자료가 확보될 때까지 보류한다. 합성 사례로 세금
  계산기를 검산하되 자료 부족을 0세금으로 숨기지 않는다. 수정가격의 분배금·분할 이중 계산을 금지한다.
- M1 입력과 기존 탐색/보류 분할을 덮어쓰거나 재생성하지 않는다. M2 설정이 바뀌면 M1 분할의
  생성 프로토콜 해시는 원본 그대로 보존하고 향후 실행 설정과 구분한다.

## 공통 모듈 계약

공통 불변 입력은 `src/research/etf/engine_types.py`: Instrument, Bar, LevelRule, Position, Candidate,
FxRate, EngineResult다. 이 파일은 통합 담당만 수정한다. 추가 필드가 필요하면 coordinator에 질의한다.

### 워커 A — 시장 규칙

소유: `engine_market.py`, `market_rules_v1.json`, `tests/test_research_etf_engine_market.py`,
`docs/research/etf_execution_rules.md`.

- `tick_size(market: str, on: date, price: Decimal) -> Decimal`
- `round_price(market: str, on: date, price: Decimal, side: str) -> Decimal` (BUY 내림, SELL 올림)
- `sec_fee_rate(on: date, *, known_at: datetime | None=None) -> Decimal` (notional에 곱할 비율)
- `fee_amount(market, side, on, notional, commission_rate=None, sec_rate_override=None) -> Decimal`
  연구 수수료+US 매도 SEC 비용. 기본 수수료 KR .00015, US .0007, GOLD .003.
  공식 확인 범위 밖 SEC 요율은 추측하거나 0으로 채우지 않고 실패한다. GOLD/KRX_GOLD 별칭 지원.
- 비용은 모든 시장에서 정확한 Decimal 비례액이다. 증권사별 10원·1원·센트 절사/올림은 적용하지
  않는 연구 근사이며 제품의 비용 청구 반올림과 동일하다고 주장하지 않는다. 수익률 0.01% HALF_UP와
  세금 원장의 반올림은 각각 별도다. 제품 결정 대조는 같은 입력 비용이 성립하는 사례에서 수행한다.
- SEC의 on은 charge date(통상 결제일)이며 tick_size의 on은 거래일이다. C는 검증한 결제일의 요율을
  override로 넘긴다. 공식 공표일도 기록해 거래시각 이후에 발표된 미래 요율을 판단에 쓰지 않는다.
  known_at을 주면 당시 공표된 요율만 허용한다. 공표 날짜만 확인됐을 때는 미국 동부시간 공표일
  종료 뒤부터 사용 가능하다고 보수적으로 가정하고 그 가정을 명시한다.

### 워커 B — 매매 판단과 일봉 체결

소유: `engine_strategy.py`, `tests/test_research_etf_engine_strategy.py`.

- `candidate_for_bar(instrument: Instrument, bar: Bar, positions: list[Position], rules: tuple[LevelRule, ...], *,
  path='O-H-L-C', first_entry=False, penetration_ticks=1, slippage_ticks=0,
  commission_rate=None, sec_rate_override=None, product_sell_tax_rate=Decimal('0')) -> Candidate | None`
- 비용을 뺀 수익률을 0.01% 단위 HALF_UP 반올림하고 목표와 비교한다. 매수 조건은 수수료 제외
  직전 차수 원가와 정확히 비교한다. 시장가와 호가 보정 지정가 둘 다 확인한다.
- 갭은 시가, 터치는 1호가 관통, 수수료 포함 정수 수량. 거래량0·가격제한폭 고정일 거래 없음.
  사소한 OHLC 반올림 위반 허용은 프로토콜의 최대 .62% 안에서만, 심한 위반은 실패한다.
- Candidate.at은 경로상 가정한 체결 시각이다. 함수는 현금 원장을 바꾸지 않는다.

### 워커 C — 현금·사건 순서·포트폴리오

소유: `engine.py`, `tests/test_research_etf_engine.py`.

- `run_engine(instruments: list[Instrument], bars: list[Bar], rules: dict[str, tuple[LevelRule, ...]], *,
  initial_cash: dict[str, Decimal], fx_rates: list[FxRate]=(), evaluation_times: list[datetime]=(),
  conversion_times: list[datetime]=(), path='O-H-L-C', first_entry=True,
  initial_positions: list[Position]=(), first_level_fraction: Decimal | None=None,
  participation_rate=Decimal('.05'), fx_cost=Decimal('.001'), cash_floor_fraction=Decimal('0'),
  symbol_cap_fraction=Decimal('1'), tax_mode='pre_tax', **explicit_options) -> EngineResult`
- None first_level_fraction은 입력 고정 예산, 값이 있으면 월 첫 통합 거래일 직전 평가액×비율÷종목수로
  차수 예산을 갱신한다(보유 차수의 취득 원가와 익절 비율은 변경하지 않음).
- 최초 진입은 각 상품 최초 Bar 시가 한 번, 그 뒤 자동 재진입 금지. 통화별 현금, 비용, 보유,
  거절 사유, 평가액을 기록한다. 시가·체결·종가·환전·평가를 UTC로 정렬한다.
- 미국 매도 대금은 매매 재사용 현금과 환전 가능 결제현금을 구분하고 T+3/T+2/T+1 전환을 지킨다.
  휴일은 제공된 세션으로 계산하고 범위가 모자라면 추정하지 않는다.
- 추가 입력: `us_sessions: list[date]`, `settlement_session_times: dict[date, datetime]`.
  결제 가능 시각은 명시 제공된 aware 시각만 사용한다. 통합 어댑터는 M1 NYSE 마감 시각을 보수적
  결제 가용 시각 가정으로 공급한다. 부족하면 미결제 유지·경고로 두고 환전에 사용하지 않는다.
- 세금 연계 입력: `tax_basis_by_bar: dict[(symbol,date), Decimal]`,
  `initial_tax_basis: dict[(symbol,level), Decimal]`, `tax_payments: dict[datetime, Decimal]`.
  월별 갱신 차수 예산은 모든 차수 동일 금액이며 기존 차수 예산 비율은 유지하지 않는다(D5).
- 결과 행 공통 키: trades의 `side/level/quantity/price`, valuations의 `equity_krw`.
- 결제 세션이 부족하고 명시 SEC override도 없으면 US 비용 계산은 실패한다. 거래일 요율이나 0으로
  바꾸지 않는다. 지급일 없는 pre_tax 배당은 C 자체 미수금과 경고만 남기고 현금으로 쓰지 않는다.
  배당락 직전 전략 보유와 결제 세무 보유 수량이 다르면 after_tax는 실패하고 pre_tax는 자체 미수만 기록한다.
- raw 가격의 주식분할은 세무 원가원장까지 함께 바꿀 API가 없어 현 단계에서 명시 거부한다.
  M1의 `yahoo_split_adjusted_quote_with_cash_dividends` 등 분할반영 가격은 보유 수량에 다시 분할을 적용하지 않는다.
- 현금·종목 상한은 평가에 사용 가능한 과거 가격으로만 적용. 동일 시각 상품 순서는 입력 순서로 고정.
- tax_mode='pre_tax'가 주 결과. synthetic after-tax는 워커 D TaxLedger 연계. 연계에 미확정점이 있으면
  질의하고 필요한 옵션을 명시해 받는다. 무시하는 **옵션은 금지.

### 워커 D — 세금 원장

소유: `engine_tax.py`, `tests/test_research_etf_engine_tax.py`.

- `TaxLedger(mode='pre_tax', *, synthetic=False)` (after_tax는 synthetic=True인 합성 검산만 허용)
- `buy(instrument: Instrument, position: Position, *, fx_rate=Decimal('1'), tax_basis=None) -> None`
- `sell(instrument: Instrument, position: Position, price: Decimal, fee: Decimal, at: datetime, *,
  fx_rate=Decimal('1'), tax_basis=None) -> Decimal` (당장 원천징수할 자기 통화 세액)
- `dividend(instrument: Instrument, gross: Decimal, at: datetime) -> Decimal` (원천징수 세액)
- `snapshot() -> dict` (연간 US 원화 실현손익·세액·미납 세액 등)
- 세금 문서 정의에 따라 합성 자료로 검산한다. 국내 기타 ETF는 이동평균 원장, 과표기준가·분배 이월
  등 필요한 입력이 없으면 after_tax 계산을 거부한다. 기본 pre_tax도 원장은 남기되 실제 세후 판정은 아님.
- US 양도손익은 차수와 독립적으로 계좌·종목별 취득원가 원장을 유지한다. 연 250만원 공제 합산·22%,
  다음 해 납부와 미납 세액은 거래 비용과 분리한다. 지급 스케줄·추가 API는 문서에 명시하고 C에 공유한다.
- buy/sell의 `settled_at`은 선택 인자다. C는 결제 사건이 도달했을 때 그 시각에 알려진 환율로 세무
  buy/sell을 반영한다. 거래시각에 미래 결제일 환율을 가져오지 않는다. `advance(at)`,
  `pay_tax(amount_krw, at)`, `accrue_dividend(...)`, `dividend(..., receivable_id=...)`로 미수·지급을 분리한다.
  5/31·7/31 납부 예정일은 휴일조정 미확인 표시를 남기고 실제 납부는 명시 입력만 따른다.

### 워커 E — 독립 사례

소유: `tests/test_research_etf_engine_contract.py`, `tests/fixtures/etf_m2_cases.json`,
`docs/research/etf_m2_product_cases.md`.

제품의 고정 커밋과 손계산에서 기대값을 먼저 만든다. 구현을 복사한 기대값은 금지.
1차 빈칸·2차 막힘·중간 빈칸·매도 순서·갭/관통·경로 차이·수수료·현금 거절·미래 환율/종가 변경
불변성·월 예산·결제·환전·세금 중 독립 사례를 작성한다. 파일 미생성이나 API 문제는 질의로 해결한다.

## 통합·검증·검토

워커는 같은 현재 작업 트리에서 파일 소유권을 지킨다. 다른 워커의 파일·공통 계약·문서·CI를 수정하거나
커밋/리셋/체크아웃/삭제하지 않는다. 완료 시 변경 경로, 명령·결과, 제한과 질문을 worker_done으로 보고한다.
통합 담당은 M1 입력 어댑터·실행 예제, protocol M2 값 연결, CI 등록과 진행 기록을 맡는다.

완료 조건:

1. M2 모듈·독립 손계산·제품 사례·미래 정보 방지 검사 통과.
2. 각 워커의 전체 결과를 취합하고 필요한 수정을 해당 워커에 돌려 해결.
3. 최신 통합본의 `python tools/ci/run_cpu_checks.py`와 실행기 자체 검사 통과. 로컬 환경이 지원하지 않는
   필수 검사는 구체적 장애를 기록하고 통과로 바꾸지 않는다. GPU/DB 전체 검증으로 확대하지 않는다.
4. 코드·프로토콜·문서·M1 보류 분할을 함께 본 Opus의 읽기 전용 최종 검토. 차단 지적 해소 후 필요 범위 재검증.
5. 진행판에 최종 범위·검증·남은 한계·M3 인계를 남긴다. 수익성·세후 승인·실전 운용 완료로 표현하지 않는다.

## 실행 기록

- 2026-11-17: 시작 후보 `d11b5e1`, 미커밋 변경 없음. M1 작업 트리 6개 제거 후 현재 작업 트리 1개.
- 공통 입력 타입·작업 경계·검증 기준을 작성하고 타입 import를 확인했다.
- Orca Run: `run_c21877133c43` (현재 작업 트리 공유, 파일 소유권 분리).

| 작업 | Task | Dispatch | 상태 |
|---|---|---|---|
| A 시장 규칙 | `task_5372be440419` | `ctx_8c17069aa892` | succeeded, 시장 규칙 검사 24개 exit 0 |
| B 매매·체결 | `task_b24345bf4f8d` | `ctx_37015ae575cb` | succeeded, 경계·손계산 검사 35개 exit 0 |
| C 현금·UTC 엔진 | `task_750b7fbf8f52` | `ctx_876b232665ff` | succeeded, 엔진 검사 41개 exit 0 |
| D 세금 원장 | `task_a31078c9e930` | `ctx_376dd92dd6fe` | succeeded 보고, 독립 검사 29개 exit 0, 통합 대기 |
| E 독립 사례 | `task_a408cf0ea633` | `ctx_cac71fe9be14` | succeeded, 독립 계약 검사 37개 exit 0 |
| Opus 1차 검토 | `task_4f9121eb8385` | `ctx_50b4fde31856` | 검토 완료, HIGH 2·MEDIUM 2 및 LOW 지적, 후보 미승인 |
| B HIGH-1 보완 | `task_b13e9227a3f0` | `ctx_d7927698312a` | succeeded, 전략47개·독립44개 exit0 |
| C HIGH-2·평가 보완 | `task_f78adb742d7d` | `ctx_3ce90d8eb2c4` | succeeded, 자체46개 exit0, B 연계 실패2건은 보존 |
| D 추정세 보고 API | `task_a8d892cf7297` | `ctx_9f2de9e439c2` | succeeded, 자체35개 exit0 |
| E 독립 회귀 추가 | `task_4e94b1460236` | `ctx_2a6fd990aa3c` | succeeded, 44개 기대값 고정·B 수정 검증 출처 기록 |
| Opus 보완 재검토 | `task_480b4415b71f` | `ctx_145684e2c5bf` | 완료, M2/M3 대조 기준 남은 차단 없음, M5전 MEDIUM1·LOW3 |
| D 준비금 설명 정합화 | `task_024312792a95` | `ctx_7a1ec85108b5` | succeeded, 워커 보고: 문서 문자열만 변경·실행 AST 불변 |
| C 직접 호출 계약 | `task_3d2b9b2db3bc` | `ctx_4fe605a27632` | succeeded, run_engine docstring만 변경 |
| Opus 마지막 확인 | `task_404eaa192f4d` | `ctx_eae4c179ca8f` | 완료, 19개 해시·480개 검사 대조·새 차단 없음 |

- 워커 사양 파일: `C:/Users/leerj/Documents/Codex/2026-11-17/etf-m2-orchestration/specs/`.
- 다음: Orca 결과·질문 도착 시 대응, 모듈 취합·M1 어댑터·설정·검증 연결 후 Opus 검토.
- A·C·D 질의에 결제시각·동일 차수 예산·SEC charge date·합성 세후 모드·미래 세무환율 금지를 답변하고
  B·C·D·E에 연계 사항을 전달했다. 아직 구현 완료나 검사 통과를 확인한 단계는 아니다.
- C의 후속 연계 질의에 공표시점 조회 API, 결제일 부족 실패, 지급일 누락 미수·세후 권리수량 불일치 실패,
  raw split 거부와 M1 분할반영 가격 인식을 확정했다. E 결과 키 계약을 승인했다.
- E는 독립 기대값·검사 파일 3개를 작성했고, 최초 검사는 engine_market 미생성으로 실패했다고 보고했다.
  이는 구현 대기 상태이며 완료·통과 처리하지 않았다.
- D가 29개 검사 exit 0 완료를 보고했다. C에 snapshot·배당·납부 API를 공유했다.
- A가 시장 모듈·규칙 JSON 생성과 2011~2026 SEC 18구간·known_at 지원을 보고했고 파일 존재를 확인했다.
  B·C·E에 연계 검사 재개를 전달했다. A 자체 검증·문서는 아직 진행 중이다.
- A/B/C/E도 완료 보고를 받았다. 다섯 담당 모듈 검사 합계 166개이며 C의 E 검사 재실행은 중복 합산하지 않는다.
- 통합 담당은 M1 탐색 상품 연결부와 가상 3일 예제를 추가했다. 연결부 15개, 갱신 프로토콜 20개 검사 통과.
  실제 M1의 069500·QQQ·KRX 금에서 2026-11-02~04 일봉 9개 로딩을 확인했고 성과 실행·보류 실행은 하지 않았다.
- 프로토콜 `1.1.0-m2-draft`, 해시 `214742a641a58e6716d8113242d2e000eccb54718de82fe0f62f8fd6caf000f3`.
  M2 미정 3개를 시장 규칙 파일 해시에 연결했고 M5 미정 5개가 남는다. 기존 M1 분할은 변경하지 않았다.
- Python 3.12.4 검사용 가상환경의 초기 pip 설치는 한글 주석을 cp949로 읽어 실패했다.
  원본 파일은 바꾸지 않고 PYTHONUTF8=1로 다시 설치 중이며 전체 검사를 아직 통과 처리하지 않았다.
- UTF-8 모드로 Windows Python 3.12.4 의존성 설치와 pip check 성공. 선행 CPU 441개·예제 3개는 통과했다.
  이후 직전 20거래일 누락을 과거 행으로 메우지 않는 입력 검사 1개를 추가했고 입력 검사 16개가 통과했다.
- Windows 실행기 자체 검사 24개는 자식 프로세스 취소/SIGTERM 정리에서 실패 2·오류 2였다.
  검사나 요구를 완화하지 않고 Ubuntu WSL Python 3.12.3 및 동일 고정 의존성으로 전체 검사 재검증 중이다.
  이는 GitHub 원격 CI나 GPU/DB 검증이 아니다. 로그는 외부 validation 폴더에 보존한다.
- 위 Windows 검사에서 남은 빈 임시 폴더 `C:/Users/leerj/AppData/Local/Temp/tmpl427ooap`,
  `tmpyzyc0zc2`의 삭제는 자동 승인 검토가 `blocked by policy`로 거절했다. 우회 삭제하지 않고 유지한다.
- 후보 소스·검사·설정 19개 해시를 외부 `validation/code-fingerprints.json`에 고정했다.
- WSL의 첫 `/tmp` 가상환경은 후속 호출에서 사라져 검사 실행 전 exit 127이 났다. 해당 실패 로그도
  `*-wsl-env-missing.log`에 보존하고 `/root/.cache/split-etf-m2-20261117-review`에 환경을 다시 구성했다.
- **최신 Linux 검증 통과**: Ubuntu WSL Python 3.12.3, 고정 두 requirements와 pip check 성공,
  `python tools/ci/run_cpu_checks.py` 선택 CPU **442개·예제 3개 exit 0**, 실행기 자체 검사 **24개 exit 0**.
  실패·건너뜀·예상 실패 0. 로그: `validation/cpu-checks-wsl.log`, `validation/runner-tests-wsl.log`.
  검사 뒤 후보 19개 소스 해시가 그대로임을 대조했다. 다음은 Opus 읽기 전용 최종 검토다.
- Opus 검토를 같은 Run에서 시작했다. 차단 지적을 받으면 해당 구현 워커에 수정·검증을 돌리고
  최신 후보 전체 검사 및 필요한 Opus 재검토를 마친 뒤 완료 판단한다.

## Opus 1차 검토와 보완

Opus는 후보 19개 소스 해시와 Linux 검사 기록을 대조하고 고정 제품 근거와 메모리 내 반례로 확인했다.
검토 작업의 succeeded는 검토 수행 성공이며 M2 후보 승인이 아니다. HIGH 2건 때문에 M3 대조 기준으로 쓰지 않는다.

| 지적 | 원인·재현 | 담당·조치 |
|---|---|---|
| HIGH-1 | 최초 매도조건이 관통 미달로 미체결인데 뒤의 매수를 같은 날 체결. KR O10000/H10500/L9400/C9600, 1차10주@10000 take5%, 2차drop5%, 비용0, 관통1의 O-H-L-C | B: 최초 주문을 고정하고 그 주문만 기다림. E: 양방향 미체결 후 주문 교체 금지 독립 기대값 |
| HIGH-2 | 환전이 전일 평가의 보유량과 현재 현금을 섞어 밤사이 US 매수 후 USD 비중을 과도하게 늘림 | C: 환전 시각의 공개된 현금·보유·환율로 통일, 야간 매수 후 목표 비중 회귀 |
| MEDIUM-1 | 상장폐지/자료 끝의 누락이 마지막 가격 동결로 이어짐 | 통합 담당: 요청 기간 실제 세션 전부를 요구. 누락이면 명시 실패, 원본 보정·자동 청산 추정 없음. 입력20개 검사 통과 |
| MEDIUM-2 | 현재 연도에 이미 실현된 US 손익의 추정세를 별도 차감 보고하지 않음 | D/C: 가벼운 valuation_summary와 estimated_tax_krw/equity_after_estimated_tax_krw 추가 |

세금 문서 3.5의 기존 정책은 현재 연도 추정세를 보고에만 쓰고 판단·현금 가용액에 쓰지 않는 것이다.
최초 수정 지시의 월 예산 추정세 차감은 이를 대조해 철회했다. 장부 equity_krw·월 예산은 확정 미납만 차감하고,
추정세 반영 보고는 equity_after_estimated_tax_krw로 분리한다. 미래 연말 손익을 추정하거나 익절에 넣지 않는다.

LOW 지적은 KR 매도 즉시 결제현금 가정, US 세무 귀속·환율의 KST 가정, 최초 미진입 무경고,
평가마다 원장 deepcopy 비용, 제품 설정 검증의 허용 범위 차이, 거래량0/OHLC0 자료 오류 문맥 부족이다.
C는 최초 미진입 사유·가벼운 세금 조회를 보완하고 통합 담당은 나머지 가정·지원 경계를 문서화한다.
OHLC0 자료는 상품·날짜·거래량·ohlc_ok를 붙여 실패시키며 원본을 보정하지 않는다.

- C 보완은 자체46개 통과로 완료됐다. D는35개 통과. E는 독립 기대값44개를 고정하고 HIGH-1 관련
  순수 전략 양방향2건·통합1건 실패를 보존했으며 B 수정 완료 알림을 기다린다. E의 idle 알림은 이 의도적
  대기 상태이며 실패·유실로 간주하거나 새 워커로 교체하지 않는다.
- 통합 담당은 M1 연결부20개를 통과했고 프로토콜을 `1.1.1-m2-draft`로 갱신했다. 환전 동시점,
  추정세 보고 전용과 KR 결제·US 세무 시각·상장폐지·제품 설정 검증의 지원 경계를 명시했다.
- B는 최초 주문 고정으로 HIGH-1을 수정했다. 기존35개 기대값을 유지한 전략47개와 E 독립44개가 exit0.
  E는 처음 실패3건과 이후 B가 실행한 통과 결과의 출처를 구분해 기록하고 완료했다. 모든 보완 워커 결과를 취합했다.
- r2 당시 프로토콜 해시: `59554471ed33a45e4802a82b7e4b44af756566fc45f1ea8802d552e8acec79b4`.
  다음은 최신 통합 CPU476개·예제3개 검사와 Opus 재검토다.
- **보완 후보 r2 전체 검사 통과**: 동일 Ubuntu WSL Python3.12.3에서 선택 CPU476개·예제3개,
  exit0·실패/건너뜀/예상실패0. `validation/cpu-checks-wsl-r2.log`에 보존한다. 실행기 소스는 r1과 같은
  해시이고 자체 검사도 HEAD와 줄바꿈 정규화 후 같아 기존24개 통과 증거를 중복 실행 없이 재사용했다.
- r1 로그/소스 해시는 `*-r1.log`, `code-fingerprints-r1.json`에 보존했다. r2 당시19개 소스는
  현재 `code-fingerprints-r2.json`에 보존돼 있다. 원래 M1 분할 CSV 해시도
  `8823252647973af1e79a380341e86002bf44ceba222b723126a3e72d03e73f6b`로 유지됐다.
- Opus 보완 재검토를 시작했다. 검토 보고 전까지 M2를 완료나 M3 승인으로 표시하지 않는다.

## Opus r2 결과와 마지막 정리

Opus r2는 19개 해시·476개 검사·원래 검사 ID 비삭제와 메모리 반례를 확인해 M2 기준 엔진을
M3 대조에 쓰는 데 남은 HIGH/MEDIUM 차단이 없다고 판정했다. 실제 세후·실전·수익성 승인은 아니다.

- 남은 MEDIUM은 M5 전 상장폐지 청산 사건과 독립 사례가 필수라는 점이다. 거부한 폐지 상품을
  조용히 제외하고 실제 자료 탐색을 시작하지 않도록 상위 연구 계획에 실행 조건을 연결했다.
- LOW-1: 준비금의 실제 현금 구속·납부 부족 대응·종료 청산 곡선은 미구현임을 세금 모듈 설명과
  보고 문서에 명시했다. 주곡선은 장부 곡선, 실현분 추정세 차감은 보조 진단이며 완결 세후 보고가 아니다.
- LOW-2: 요청 종료일이 사용 시장 달력 범위를 넘으면 실패하도록 보완했다. US는 환전용 KRX와
  NYSE를 함께 확인한다. 확인 범위 초과/시장별 끝 차이 4개를 추가해 입력24개 검사가 통과했다.
- LOW-3: run_engine 직접 호출자는 전체 의도 기간·실제 세션 완전성을 사전 검증해야 하며 M3/M4도
  strict M1 연결부 또는 동등 검사를 거친다는 공개 계약을 추가했다. C/D 문구 수정은 실행문을 바꾸지 않았다.
- r1의 원본 fixture·검사 파일 전체 사본은 없으므로 기존 기대값 불변은 워커 보고와 검사 ID 비삭제
  수준으로만 확인했다. 해시만으로 이전 원문을 복원할 수 있다고 주장하지 않는다.
- 다음: 최신 후보480개·예제3개 검증, Opus 마지막 확인, 완료 범위와 M5 전 과제를 고정한 뒤
  M3 대조 기준을 현재 브랜치 로컬 커밋으로 보존한다.
- r3 통합 후보는 최신 소스19개를 고정한 뒤 Ubuntu WSL에서 CPU480개·예제3개 exit0,
  실패/건너뜀/예상실패0을 확인했다. 실행기 자체 소스가 그대로라24개 통과 증거를 재사용한다.
  `code-fingerprints-r2.json`/`cpu-checks-wsl-r2.log`에 이전 후보를 보존하고 최신 증거는 기존 파일명에 기록했다.
- 마지막 프로토콜 `1.1.2-m2-draft` 해시는
  `ec2f3f48e9c4ec5145261748dddfed33ff0859ca330b5f3c85330beb29fbe55c`다.
  r2→r3 변경은 입력 경계·그 검사·엔진/세금 docstring·프로토콜 설명의5개 소스 파일이다.

## 완료와 M3 인계

- Opus r3는 실제19개 해시,480개 검사와 예제3개, 프로토콜 해시·M5 미정5개·상장폐지 원래 요구 보존을
  대조했고 M2 기준 엔진을 M3 대조에 쓰는 범위에 새 차단이 없다고 판정했다. 남은 M5/세후 전 과제는
  상위 연구 계획의 실행 조건에 연결했다. GitHub 원격 CI·GPU/DB·실제 세후·실전·수익성은 확인 범위 밖이다.
- 최종 문서의476개·r2 파일명 표기를 정리했다. C/D의 docstring-only·AST 불변은 이전 원문이 없어
  워커 보고와480개 통과로만 뒷받침됨을 인정한다. 최종 소스에는 이후 대조용 docstring 제거 AST 해시를 별도 보존한다.
- 기존 ChatGPT `shared_account.cpp`(SHA256 `2358428eac6a0a8713d572362c6421248b099422322d2f7d2e5105532e3ddeb8`)를
  원본 변경 없이 별도 산출물 폴더에 컴파일했다. 공통 규칙 합성3사례의 거래6행·일별평가6행·비용·현금이
  M2와 허용오차1e-8 안에서 일치했다. 추가 매수 후 단일 차수 익절, 1차 매수·매도, 공유 현금 부족 거절이다.
  전일종가 신호/장중경로·1차 재진입·복수매도·비용/세금/환전/기업행사는 비교 범위에서 제외했다.
  증거: `validation/chatgpt_common_cases.py`, `validation/chatgpt-common/comparison.json`과 합성 입력/출력.
- Orca `run_c21877133c43`의14개 작업은 모두 completed임을 실제 task-list로 확인했다.
- 최종480개·예제3개 및 이전 실행기24개 통과 근거, r1/r2/r3 해시와 로그를 외부 validation 폴더에 보존했다.
  다음 구현은 M3이며 완료된 M2 소스를 기준으로 비교한다. 실제 자료 탐색은 상폐 청산·KR 결제 등 실행 조건을 먼저 충족한다.
- M2 구현과 완료 기록은 로컬 커밋 `d14af432acdfc128b5e44498925690a16dfa0adc`에 보존했다.
- 2026-11-18 재체크아웃 검수에서 Windows 자동 줄바꿈이 비용표의 원시 바이트 해시를 바꾸는 문제를
  재현했다. `.gitattributes`로 비용표 JSON만 LF로 고정했고 같은 `core.autocrlf=true` 체크아웃의 해시가
  프로토콜 `079bc678e16bacdc4f5ec65614b9b50dca5b229762e00ace7bb923d8ed392cbb`와 일치했다.
  이 보완은 검토받은19개 코드·설정·검사 바이트를 바꾸지 않아480개 검사를 반복하지 않고 Git 체크아웃으로 검증했다.
- 최종 소스의 docstring 제거 AST 해시16개는 `validation/code-ast-fingerprints-final.json`에 새로 남겼다.
  이는 이후 대조용이며 사본이 없는 과거 r2 원문을 소급 증명하지 않는다. Opus INFO의 환율 최대 경과일·
  공통 평가 달력 고정은 M5 실행 조건에 추가했다.
