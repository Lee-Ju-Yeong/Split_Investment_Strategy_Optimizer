# ETF M4 비교 기준선

같은 ETF 바스켓에서 분할매매가 단순히 현금을 많이 들고 있어서 낙폭이 작은지 구분한다.
주 비교는 **학습 구간에서 전략의 평균 투자 비중을 구한 뒤, 그 비중을 평가 구간에 고정한
월별 리밸런싱**이다. `baseline_policy.py`, `baseline_engine.py`, `baseline_report.py`에 구현한다.

## 실행

```bash
python -m examples.etf_m4_demo
python -m examples.etf_m4_demo --path O-L-H-C --report results/etf-m4-demo-second-path.json
python tools/ci/run_cpu_checks.py
```

예제는 메모리 합성 가격을 쓰며 외부 계정·DB·네트워크·GPU·C++ 빌드를 사용하지 않는다.
JSON 파일은 명시한 새 경로에만 쓴다. 학습 결과와 평가 결과는 별도 실행하며 모든 비용은 같다.
한 번의 예제는 주 기준선·단순 보유·프런티어 5개·사후 진단, 총 8개 기준을 반환한다.
예제의 수익률은 계산 확인용이며 실제 투자 성과가 아니다.

## 정책 고정과 사용

```python
from src.research.etf.baseline_policy import fit_baseline
from src.research.etf.baseline_report import run_baseline_suite, comparison_report

policy = fit_baseline(training_result, instruments,
    training_start=training_start, training_end=training_end,
    evaluation_start=evaluation_start)
baselines = run_baseline_suite(policy, instruments, bars,
    initial_cash=initial_cash, **common_execution_options)
report = comparison_report(strategy_result, baselines, instruments,
    initial_equity_krw=initial_wealth, started_at=evaluation_start, ended_at=evaluation_end)
```

- `training_result`는 학습 구간의 결과만 받는다. 창 밖·중복·역순 시각, 다른 바스켓, 비양수 자산,
  차입으로 100%를 넘는 투자 비중을 거부한다. 학습 종료는 평가 시작보다 빨라야 한다.
- 비중은 평가 시각별 `보유 평가액 / 장부 평가액`의 산술평균이다. `합계 투자액 / 합계 자산`이 아니다.
- 동일가중이 기본이다. 위험 예산 주문의 짝은 `volatilities={symbol: VolatilityEstimate(value, known_at)}`로
  명시한다. 모든 상품의 양수 추정값과 학습 종료 이전 정보 시각이 필요하다. 정의·추정 기간·최소 관측치는
  프로토콜의 변동성 계약을 호출자가 충족해야 한다. 평가 자료로 가중을 다시 계산하지 않는다.
- 정책은 불변 객체다. 학습 행·가중·시각의 출처 해시와 정책 전체 해시를 반환한다. M5 학습 구간은
  여기서 정하지 않으며 호출자가 명시한다. 사람이 새로 구성한 정책도 입력 출처의 진실성을 자동 인증하지는 않는다.
  정책·학습 행·평가 시각은 UTC로 정규화하므로 서머타임 전환과 같은 시각의 다른 표기에도 순서·경과시간이 같다.
  정책의 평균·가중·합계 검사는 Decimal 정밀도80·`ROUND_HALF_EVEN`으로 고정해 호출자의 반올림 설정에
  의존하지 않는다. 호출자 설정은 보존하며 매매·체결 원장은 기존 M2의 연산 규칙을 유지한다.
- `fixed_baseline`은 사전 지정 비중을 쓰는 참고군이다. `diagnostic_baseline`은 평가 구간 실현 비중을
  사후 매칭하며 `diagnostic`으로 표시한다. 이를 `primary`로 바꾸거나 주 비교 묶음의 출발 정책으로 쓰면 거부한다.

## 매매와 원장

월 첫 **통합 세션**에 직전 명시 평가액으로 상품별 원화 목표를 고정한다. 최초 목표는 초기 현금의
알려진 평가액을 쓴다. 각 상품은 그달 자기 시장 첫 세션의 시가에 목표와 보유의 차이만큼 정수로
거래한다. 휴장한 시장은 다음 실제 개장까지 기다린다. 시가와 당시에 공표된 환율로 수량을 계산하고
그 뒤 고가·저가·종가는 주문에 사용하지 않는다. 같은 시각은 매도 후 매수, 같은 방향은 상품 입력 순서다.

매수 수수료를 목표 예산에 포함한다. 매도 후 보유는 목표 금액을 넘지 않는 정수 수량으로 내림한다.
비용·정수 수량·시가 갭·주문 거절 때문에 실현 비중은 목표와 다를 수 있으며 실현 평균도 보고한다.
목표액 갱신에는 그달 개장 가격으로 아직 확정되지 않은 다른 시장의 미래 평가액을 쓰지 않는다.

시가 주문은 M2의 최초 시가 진입과 같은 체결 근사다. 지정가 터치 주문이 아니므로 관통 호가 수가
시가 체결을 막지 않는다. 불리한 슬리피지, 거래 없는 날, 가격제한폭, 거래대금 참여율, 현금 부족,
현금 하한·상품 상한은 공통 규칙이다. 거절된 주문은 줄이거나 같은 달에 다시 시도하지 않는다.

현금·환전·US 결제·분배금·세금은 M2 원장을 공유한다. 통화별 환전 목표는 M2처럼 **상품 수 비율**이며
주식 투자 비중이나 역변동성 가중과 별개다. 낮은 주식 비중의 US 바스켓도 USD 현금과 환전 비용이 있을 수 있다.
환전 시각은 호출자가 명시하며 미래 매도대금으로 과거 주문을 충당하지 않는다.
단순 보유는 최초 진입·최초 명시 환전만 수행하고 이후 분배금을 재투자하지 않는다.

취득 건은 FIFO 순서로 일부 줄인다. `Position.buy_fee`는 세무 취득 건의 원래 수수료 식별값을 보존한다.
세무 원가는 기존 종목별 이동평균이다. `TaxLedger.sell_lots`는 여러 취득분을 **한 주문**으로 묶어
수수료·국내 기타 ETF 원천징수의 원 미만 절사를 한 번만 적용한다. 별개 주문의 같은 날 복수 매도
제한은 유지한다. 실제 세후 판정 승인은 없으며 합성 세후는 기존 명시 옵션과 자료 조건을 요구한다.

실자료는 `load_m1_inputs(...).engine_kwargs()` 또는 동등한 검사를 거쳐 요청 전체 기간·실제 거래일·
가격 완전성을 먼저 검증해야 한다. 기준선 실행 함수도 마지막 입력 봉을 연구 종료일로 추정하지 않는다.
M5 전 상폐 청산·국내 T+2·환율 경과일 통제와 완결 세후 보고의 조건은
[연구 계획](../../todos/2026_11_15-etf-vol-split-research-plan.md#M2-이후-실행-조건)을 유지한다.

## 보고와 판정의 구분

`comparison_report`는 같은 초기 자산·바스켓·평가 시각을 전제로 다음을 계산한다.
실행 옵션의 동일성과 입력 출처는 호출자가 보장해야 한다. 평가 시각이 다르면 쌍대 비교를 거부한다.

| 필드 | 정의 |
| --- | --- |
| `cagr` | 초기 자산부터 종료 평가액까지 실제 경과일/365.25 기준 연복리. 소수 비율 |
| `mdd` | 초기 자산을 포함한 고점 대비 최대 하락률. 0 또는 음수 |
| `calmar` | CAGR / \|MDD\|. 낙폭 0이면 JSON null |
| `underwater_days` | 평가 표본의 고점부터 회복 또는 종료까지 최장 경과일 |
| `average_investment_fraction` | 각 평가 시각 투자액/자산의 산술평균 |
| `unrealized_price_pnl_krw` | 종료 보유분 가격 차이×수량×현재 환율. 매수 수수료·환차익·청산세 제외 |
| `delta_cagr` | 전략 − 기준선. 양수면 개선 |
| `delta_mdd` | \|기준선 MDD\| − \|전략 MDD\|. 양수면 개선 |

목표 비중 0·25·50·75·100%의 `performance.mdd`, `performance.cagr`가 프런티어 그림의 좌표다.
단순 보유·100% 월별 리밸런싱은 별도 행이다. 두 차이가 양수이면 해당 실행의 `joint_improvement`만
참으로 표시한다. 통계·보류 성공 판정을 수행하지 않고 `statistical_success_evaluated=false`로 반환한다.
현금 이자는 주 결과인 0%이며 금리 민감도·변동성 일치 기준선·매일 적응형 전략은 별도 후속 연구다.

[범위·검증·리뷰 기록](../../todos/done_2026_11_17-etf-m4-baselines.md)
