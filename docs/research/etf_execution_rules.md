# ETF M2 호가·비용 규칙

확인일: **2026-11-17**. 구현은 `src/research/etf/engine_market.py`, 실행 데이터는
`src/research/etf/market_rules_v1.json`이다. 이 표는 연구 시뮬레이션을 재현하기 위한 비용표다.
공식 시장 규칙과 실행 계약에서 정한 위탁수수료 가정을 구분하며, 실제 증권사 청구액과 같다고 주장하지 않는다.

## 날짜와 공개 함수

| 함수 | 날짜의 의미 | 동작 |
| --- | --- | --- |
| `tick_size(market, on, price)` | 해당 시장 거래일 | 자기 통화 가격에 맞는 호가단위를 Decimal로 반환 |
| `round_price(market, on, price, side)` | 해당 시장 거래일 | BUY 내림, SELL 올림; 내림 결과가 0이면 오류 |
| `sec_fee_rate(on, *, known_at=None)` | SEC의 **charge date**, 통상 결제일 | 거래대금에 곱할 소수 비율 반환; `known_at`으로 정보 가용 시점 검사 가능 |
| `fee_amount(market, side, on, notional, commission_rate=None, sec_rate_override=None)` | US SELL의 SEC 조회에는 **charge date** | 위탁수수료와 US SELL의 SEC 비례비용 합산 |

`on`은 `date`이며 `datetime`을 자동 변환하지 않는다. `known_at`은 시간대가 있는 `datetime`이어야 한다.
시장 이름은 `KR`, `US`, `GOLD`, `KRX_GOLD`를 지원하며 마지막 둘은 같은 금 현물 규칙을 사용한다.
공통 `Instrument`에서 사용할 이름과 별칭 정규화는 상위 입력 계약을 따른다.

SEC 공지는 charge date의 전환일을 제시한다. 이 날짜를 거래일에 그대로 적용하면 결제 전환 경계의
비용이 잘못될 수 있다. [SEC Rule 31(a)(3)와 Release 34-49928](https://www.sec.gov/rule-release/34-49928)을
기준으로 상위 엔진이 검증된 결제일을 계산해야 한다. 이 모듈은 휴일이나 결제일을 추정하지 않는다.

신호 생성 시에는 다음 순서로 호출한다. 결제일 또는 공표된 요율이 확인되지 않으면 오류를 전달하고,
0이나 거래일 요율로 바꾸지 않는다.

```python
rate = sec_fee_rate(verified_settlement_date, known_at=bar.open_at)
fee = fee_amount("US", "SELL", verified_settlement_date, notional,
                 sec_rate_override=rate)
```

SEC 원문에서 공표 **날짜**는 확인했으나 시각은 확인하지 못했다. 따라서 `known_at`을 주면 공표일의
미국 동부시간 하루가 모두 끝난 **다음 날 00:00 America/New_York부터** 사용할 수 있다고 보수적으로
가정한다. 서머타임을 포함해 IANA 시간대 자료로 UTC와 비교한다. 날짜가 바뀌기 전에는 그날 실제로
공개됐을 수 있어도 거부하며, 새 요율을 알 수 없다고 이전 요율로 되돌리지 않는다.
`known_at=None`은 사후 정적 비용표 조회로 정보 가용성 검사를 생략한다. **신호용 호출은 반드시
`known_at=bar.open_at` 등 실제 판단 시각을 지정해야 한다.** JSON의 각 SEC 행은 `announced_on`,
`effective_from`, `source_url`을 보존한다. Windows에서도 `America/New_York` IANA 자료가 필요하다.

## 국내 ETF와 금 현물 호가

| 시장 | 적용 거래일(양 끝 포함) | 가격 조건 | 호가단위 |
| --- | --- | --- | --- |
| KR ETF | 2011-01-01 ~ 2023-12-10 | 모든 가격 | 5원/주 |
| KR ETF | 2023-12-11 ~ 2026-11-17 | 2,000원 미만 | 1원/주 |
| KR ETF | 같은 기간 | 2,000원 이상 | 5원/주 |
| GOLD / KRX_GOLD | 2014-03-24 ~ 2026-11-17 | 모든 가격 | 10원/g |

과거 ETF의 전 가격 5원 규칙은 [KRX 유가증권시장 업무규정 시행세칙 제32조 제2항 단서와 개정
이력](https://law.krx.co.kr/las/LawBon.jsp?lawid=000111&pubdt=20151001&pubno=0000012330)에서 확인했다.
제624호(2010-07-29) 부칙상 제32조의 시행일은 2010-10-04이며, 이후 2014-09-02 개정 이력에도
ETF 5원 단서가 있다. v1의 확인·지원 시작은 2011-01-01로 한정하며 최초 ETF 도입 이후의 전 이력을
제공하지 않는다. 일반 주식의 2023-01-25 가격대별 호가 변경을 ETF에 잘못 적용하지 않는다.

저가 ETF 변경은 [KRX 시행세칙 제32조와 부칙 제2181호(2023-10-17 개정)](https://law.krx.co.kr/las/RefBon.jsp?lawid=000111&lawkd=B&pubdt=20231204&pubno=0000021920&reflinkchk=Y)의
최종 **2023-12-11 시행일**을 적용한다. 입법 예고 때 보도된 12월 18일 예정일은 사용하지 않는다.
확인일의 [KRX ETF 매매절차](https://regulation.krx.co.kr/contents/RGL/03/03060101/RGL03060101.jsp)도
2,000원 미만 1원, 이상 5원과 수량단위 1주를 명시한다. 과거 KRX 교육 페이지 일부에는 전 가격 5원
설명이 남아 있으므로 현행 규칙의 근거로 쓰지 않았다.

금은 [금융위원회의 2014-03-24 개장 발표](https://fsc.go.kr/po010101/71087)에서 개장일, 1g 수량,
10원 호가를 확인했다. [KRX 금 현물 거래제도](https://regulation.krx.co.kr/contents/RGL/04/04010104/RGL04010104.jsp)는
현재도 1g당 원화 표시와 10원 단위를 명시한다. 개장 전 날짜는 실패한다. 금 ETF와 금선물의 규칙을
이 금 현물 규칙에 섞지 않는다. 실물 인출과 그 비용·부가세는 이 모듈의 범위 밖이다.

## 미국 ETF 호가

미국 NMS ETF 일반 주문의 최소 가격단위는 $1 미만 $0.0001, $1 이상 $0.01로 구현했다.
지원 거래일은 2011-01-01 ~ 2026-11-17이다. [SEC Rule 612 시행 안내](https://www.sec.gov/newsroom/press-releases/2005-106-extension-compliance-date-rule-612-sub-penny-rule-regulation-nms)는
두 가격대의 단위와 2006-01-31 시행을 제시한다. 2024년에 채택한 일부 종목의 $0.005 단위는
[SEC 2026-06-11 연장 명령, Release 34-105656](https://www.sec.gov/files/rules/exorders/2026/34-105656.pdf)에서
2027년 11월 첫 영업일까지 적용이 미뤄졌다. v1은 그 미래까지 자동 연장하지 않는다.

이 값은 일반 주문의 최소 단위이며, 모든 거래소의 더 큰 주문 단위나 개별 면제, 중간가격·소수단위
가격개선 체결까지 재현하지 않는다. 프로토콜의 종전 US `0.01`은 **$1 이상 가격**에만 그대로 맞는다.

## SEC 매도 수수료 역사표

단위는 **매도 거래대금 USD 1백만당 수수료 USD**다. 실행 시 1,000,000으로 나눈 소수 비율을
Decimal로 곱한다. 아래 기간은 charge date 기준이며 양 끝을 포함한다. 2011년 첫 행은 2010-04-01부터
효력이 있었지만, v1은 2011-01-01부터만 조회를 허용한다.

| 시작 | 끝 | 백만 USD당 비용 | 공식 공표일 | SEC 1차 출처 |
| --- | --- | ---: | --- | --- |
| 2011-01-01 | 2011-01-20 | 16.90 | 2010-03-01 | [FY2010 #5](https://www.sec.gov/newsroom/press-releases/2010-29-fee-rate-advisory-5-fiscal-year-2010) |
| 2011-01-21 | 2012-02-20 | 19.20 | 2010-12-22 | [FY2011 #5](https://www.sec.gov/newsroom/press-releases/2010-255-fee-rate-advisory-5-fiscal-year-2011) |
| 2012-02-21 | 2012-03-31 | 18.00 | 2012-01-20 | [FY2012 #5](https://www.sec.gov/newsroom/press-releases/2012-2012-15htm) |
| 2012-04-01 | 2013-05-24 | 22.40 | 2012-03-01 | [FY2012 #6](https://www.sec.gov/newsroom/press-releases/2012-2012-35htm) |
| 2013-05-25 | 2014-03-17 | 17.40 | 2013-04-25 | [FY2013 #3](https://www.sec.gov/newsroom/press-releases/2013-2013-74htm) |
| 2014-03-18 | 2015-02-13 | 22.10 | 2014-02-12 | [FY2014 #3](https://www.sec.gov/newsroom/press-releases/2014-30) |
| 2015-02-14 | 2016-02-15 | 18.40 | 2015-01-15 | [FY2015 #3](https://www.sec.gov/newsroom/press-releases/2015-8) |
| 2016-02-16 | 2017-07-03 | 21.80 | 2016-01-07 | [FY2016 #3](https://www.sec.gov/newsroom/press-releases/2016-2) |
| 2017-07-04 | 2018-05-21 | 23.10 | 2017-05-31 | [FY2017 #3](https://www.sec.gov/newsroom/press-releases/2017-111) |
| 2018-05-22 | 2019-04-15 | 13.00 | 2018-04-17 | [FY2018 #3](https://www.sec.gov/newsroom/press-releases/2018-67) |
| 2019-04-16 | 2020-02-17 | 20.70 | 2019-03-12 | [FY2019 #2](https://www.sec.gov/newsroom/press-releases/2019-30) |
| 2020-02-18 | 2021-02-24 | 22.10 | 2020-01-09 | [FY2020 #2](https://www.sec.gov/newsroom/press-releases/2020-7) |
| 2021-02-25 | 2022-05-13 | 5.10 | 2021-01-15 | [FY2021 #2](https://www.sec.gov/newsroom/press-releases/2021-8) |
| 2022-05-14 | 2023-02-26 | 22.90 | 2022-04-08 | [FY2022 #2](https://www.sec.gov/newsroom/press-releases/2022-60) |
| 2023-02-27 | 2024-05-21 | 8.00 | 2023-01-23 | [FY2023 #2](https://www.sec.gov/newsroom/press-releases/2023-15) |
| 2024-05-22 | 2025-05-13 | 27.80 | 2024-04-17 | [FY2024 #2](https://www.sec.gov/newsroom/press-releases/2024-47) |
| 2025-05-14 | 2026-04-03 | 0.00 | 2025-04-08 | [FY2025 Section 31](https://www.sec.gov/rules-regulations/fee-rate-advisories/2025-2) |
| 2026-04-04 | 2026-11-17 | 20.60 | 2026-02-27 | [FY2026 Section 31](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2) |

2012년 초에 실제 시행되지 않은 예정 요율 $15.10은 포함하지 않았다. 이를 취소·대체한 사실은
[FY2012 #5](https://www.sec.gov/newsroom/press-releases/2012-2012-15htm)에 명시돼 있다.
등록 수수료인 Section 6(b)의 연도별 수치를 거래 수수료 Section 31에 사용하지 않는다.
[SEC의 공식 목록](https://www.sec.gov/rules-regulations/fee-rate-advisories)에서 두 종류를 구분해 대조했다.
2025-05-14 ~ 2026-04-03의 0은 공식 발표에 따른 실제 표 값이다.

**2011-01-01 ~ 2026-11-17 안에 미확인 SEC 구간은 없다.** 2011년 이전과 **2026-11-18 이후**, 즉
남은 2026년 날짜도 확인 범위 밖으로 거부한다. 마지막 요율이 계속 유효할 것이라고 가정하지 않는다.
기간을 늘릴 때는 공식 목록과 해당 공지·효력일을 다시 확인하고 데이터와 경계 검사를 함께 갱신한다.

## 비용과 명시적 대체값

위탁수수료는 실행 계약의 편도 연구 가정으로 KR `0.00015`, US `0.0007`, GOLD/KRX_GOLD `0.003`이다.
각 상품 자기 통화의 거래대금에 곱하며, US SELL에만 SEC 비율을 추가한다.

```text
fee = notional × commission_rate
US SELL일 때 fee += notional × sec_rate
```

모든 계산은 Decimal이고 10원·1원·센트 절사나 올림을 하지 않는다. 따라서 비용은 거래대금에 선형이며
청구단위 반올림 경계가 없다. 예를 들어 2024-06-03 charge date의 US 매도 $100.50은
`100.50 × (0.0007 + 0.0000278) = 0.07314390 USD`다. 실제 증권사의 센트 올림, 최소 청구액,
국내 위탁수수료 절사, 금 수수료 부가세, FINRA TAF, 거래소별 부대비용을 복원하지 않는다.
세금과 세금 원장의 반올림도 이 함수와 별도다.

`commission_rate`와 `sec_rate_override`를 지정하면 합성 사례의 명시적 요율로 사용할 수 있다.
특히 `sec_rate_override=Decimal("0")` 또는 정수 `0`은 역사표를 조회하지 않는다. `None`과 구분하므로
미확인 날짜에서도 명시적으로 비용을 정한 합성 검사를 할 수 있다. 대체값의 출처와 사용 사실은
상위 실행 기록에서 남긴다. US BUY와 다른 시장은 SEC 표를 조회하지 않으며, US SELL은 거래대금이
0이어도 대체값 없이 미확인 날짜를 조용히 통과시키지 않는다. 음수·무한대·NaN·bool 입력은 거부한다.

## 조정가격과 검증 범위

분배금·분할을 반영한 조정 주가에 당시 호가단위를 적용하면 **가상 호가**가 된다. 이는 가격 단위에
맞춘 시뮬레이션의 일관성을 위한 장치이며 실제 과거 주문가, 스프레드, 호가장 또는 체결 경로를 복원한
결과가 아니다. 조정가격으로 분배금·분할을 다시 계산하는 문제는 상위 엔진이 별도로 막아야 한다.

집중 검사:

```powershell
python -B -m unittest discover -s tests -p test_research_etf_engine_market.py -v
```

2026-11-17 실행 결과: **24개 통과, 실패 0, 건너뜀 0, exit 0**. 검사는 ETF 변경일과 가격대 경계,
금 개장일·별칭, BUY/SELL 방향, SEC 18개 기간의 전환일과 직전일, 공표일의 동부시간 가용성,
겨울/서머타임, 미확인 기간의 오류, 공식 0 요율과 명시적 0 대체값의 구분, 비용 선형성을 포함한다.
원격 자료의 실시간 재수집, 전체 M2 통합, DB·GPU·실제 증권사 청구액은 이 검사 결과에 포함하지 않는다.

통합 설정에는 역사표 경로·버전과 위 수수료 가정, KR 2023-12-11 호가 전환, GOLD 10원,
US $1 경계, SEC charge date와 `known_at` 정책을 연결한다. M1 원본 프로토콜 해시나 기존 보류 분할은
이 시장 규칙 파일을 추가했다는 이유로 변경하지 않는다.
