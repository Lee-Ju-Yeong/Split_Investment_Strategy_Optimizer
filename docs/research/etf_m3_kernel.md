# ETF M3 빠른 매매 후보 커널

M3는 M2의 매매 판단·일봉 체결 후보 계산을 C++17로 옮긴 연구 전용 경로다.
정수비로 판단 경계를 비교하며, M2가 Decimal 정밀도 80자리로 연산하는 가격·수량 계산은
그 반올림 순서를 재현한다. 현금·환전·결제·기업행사·세금은 M2 Python 원장을 공유한다.
호출자의 Decimal 반올림 방식도 전달하며, 수익률 판단의 HALF_UP 규칙은 M2처럼 별도로 적용한다.
일반 CPU/GPU 제품 엔진과 연구 프로토콜은 바뀌지 않는다.

## 준비와 검사

Linux/WSL에 Python 3.12, `g++`, Boost 헤더(`libboost-dev`)가 필요하다. C++ 공유 라이브러리는
저장소에 넣지 않는다. import와 기존 CPU 기본 검사는 컴파일을 시작하지 않는다.

```bash
sudo apt-get install g++ libboost-dev
python3 -m src.research.etf.fast_build --output-dir /tmp/etf-m3-build
python3 tools/ci/run_etf_m3_checks.py --build-dir /tmp/etf-m3-build
python3 -m src.research.etf.fast_benchmark \
  --build-dir /tmp/etf-m3-build --report results/etf-m3-benchmark.json --repeat 3
```

빌드·보고서 경로는 새 경로를 쓴다. 빌드는 C++ 소스 사본, 바이너리, 컴파일러·명령·해시를 담은
`build.json`을 남긴다. 소스나 바이너리 해시가 다르거나 ABI가 맞지 않으면 로드하지 않는다.
컴파일 실패·오류를 Python 실행으로 자동 대체하지 않는다. CI의 별도 `ETF M3 C++ 정확성 대조`
작업도 같은 빌드·검사 명령을 쓴다. 검사 수집·시작·종료를 대조하며 실패·누락·건너뜀은 실패다.
기존 CPU 기본 검사에는 고정 Python 의존성 설치가 필요하지만 이 C++ 대조 검사는 표준 라이브러리만 쓴다.

## 호출

```python
from src.research.etf.fast_kernel import FastKernel, run_fast_engine

kernel = FastKernel('/tmp/etf-m3-build')
candidate = kernel.candidate_for_bar(instrument, bar, positions, rules,
                                     path='O-H-L-C', sec_rate_override=verified_sec_rate)
result = run_fast_engine(kernel, instruments, bars, rules, **validated_engine_options)
```

인수·반환은 [M2](etf_m2_engine.md)와 같다. M2 입력 검사와 SHA256 매도 우선순위만 공유하고
후보의 판단·가격·수량·체결 시각은 C++에서 계산한다. 후보 함수는 엔진 인스턴스에 연결하고
전역 함수를 교체하지 않는다. 같은 커널을 여러 실행에서 사용할 수 있다.

이 문서의 정확 일치는 Decimal의 **수치 동치**를 뜻한다. `100.50`과 `100.5`처럼 값이 같아도
지수·후행 0은 보존하지 않을 수 있으므로 `str`·JSON 바이트·출력 바이트 해시의 동일성을 뜻하지 않는다.
그 비교가 필요한 소비자는 Decimal 표기를 먼저 정규화해야 한다.

실제 자료는 `load_m1_inputs(...).engine_kwargs()` 같은 엄격한 연결부를 거쳐 전체 의도 기간·실제
시장 세션·가격 완전성을 확인해야 한다. 커널의 수동 시가 진입은 M2처럼 슬리피지가 0이면 호가표를
읽지 않으므로, 후보 API만 호출해 날짜표·실제 자료의 완전성을 검사했다고 해석하면 안 된다.
연구 실행의 전체 검사는 별도 계약이다. 보류 분할을 재생성하거나 보류 성과를 평가하지 않는다.

## 대조와 성능의 범위

검사는 M2의 기존 제품·손계산 사례를 같은 기대값으로 네이티브 경로에도 실행하고, 세 시장·두 경로의
고정 시드 후보와 혼합시장 전체 원장을 허용 오차 0으로 대조한다. 소수 반올림, 호가 밴드 변경,
미체결 주문 유지, 수동 첫 진입, 차수 빈칸, 환전·결제·합성 세금과 미래 자료 변경을 포함한다.
M2 자체에 남은 공통 오류까지 없음을 증명하지는 않는다.

성능 도구는 고정된 합성 후보 6개와 24개 일봉의 혼합시장 포트폴리오를 두 경로로 실행한다.
결과가 정확히 같을 때만 준비 실행 후 반복 실행 중앙값을 기록한다. 컴파일 시간은 포함하지 않는다.
입력·실행 소스·시장 규칙·커널 해시와 모든 시간 표본을 JSON에 보존한다. 시가 진입이나 무보유처럼
M2 계산이 짧은 경우에는 연결 비용 때문에 C++이 느릴 수도 있다. 전체 엔진을 C++로 옮긴 결과나
10만 설정 규모의 탐색 처리량이 아니며, 향상 배수는 입력과 환경에 따라 달라진다.

로컬 검사153개와 기존 CPU480개·예제3개·실행기24개를 통과했고 필수 리뷰의 차단 결함을 해소했다.
합성24일봉 전체 실행은 두 경로에서4.30~4.56배 빨랐다. 검증·측정 조건은
[M3 완료 기록](../../todos/done_2026_11_17-etf-m3-fast-kernel.md)에 남긴다.
M4 기준선·M5 실제 자료 탐색·상장폐지 청산·KR T+2·완결 세후 보고는
[M2 이후 실행 조건](../../todos/2026_11_15-etf-vol-split-research-plan.md#M2-이후-실행-조건)을 따른다.
