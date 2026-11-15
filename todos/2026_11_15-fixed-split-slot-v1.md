# 고정 차수 규칙 `fixed_slot_v1` 명세

유형: implementation. 상태: CPU 구현·검사 완료, GPU 미구현(설정 시 GPU 경로는 거부). 기준: main `05bf3de37871e263306e601c09d95f4d1f4ae561`.

## 배경

CPU 추가 매수는 새 차수 번호를 `len(positions)+1`로 만든다([strategy.py:671](../src/backtest/cpu/strategy.py#L671)). 중간 차수가 먼저 익절된 뒤 다시 사면 번호가 겹친다. 예: 1·2·3차 보유 → 2차 익절 → 추가 매수가 다시 3차로 기록되어 보유 차수 `[1, 3, 3]`.

또 1차가 없으면 추가 매수를 멈춘다([strategy.py:624](../src/backtest/cpu/strategy.py#L624)). 그런데 신규 진입은 이미 보유한 종목을 제외한다([strategy.py:410](../src/backtest/cpu/strategy.py#L410)). 그래서 1차만 먼저 익절되고 2차가 남으면, 남은 차수가 팔리거나 손절·비활성 청산될 때까지 그 종목은 매수가 멈춘다.

GPU는 첫 빈칸을 채우고([logic.py:625-635](../src/backtest/gpu/logic.py#L625)), 가장 최근에 산 차수를 기준가로 쓴다([logic.py:390-400](../src/backtest/gpu/logic.py#L390)). CPU와 구조가 다른 규칙이다. 실제 CUDA 실행으로 결과 불일치를 확인하지는 않았다.

두 문제는 2026-11-15 외부 연구(ChatGPT 합성 재현기)가 먼저 보고했다. 이 저장소의 실제 엔진에서 같은 거래가 재현되는 것을 직접 확인했다.

## 결정 (사용자 승인 2026-11-15)

- 실제 제품(split-trader V2 `product_trading_policy.decide`, 커밋 `1bd2d32`)과 같은 차수 선택·기준가 규칙을 쓴다.
- 1차가 비어도 2차 이상 추가 매수는 계속한다. 1차는 신규 진입 조건이 다시 맞을 때 채운다.
- 설정 키 `split_slot_policy`를 추가한다. 값은 `legacy`(기본값)와 `fixed_slot_v1`이다. 기본값을 `legacy`로 두어 과거 WFO·GPU 결과를 그대로 재현할 수 있게 한다. 기본값 변경은 GPU 정합성 확인 뒤 따로 결정한다.

## `fixed_slot_v1` 규칙

N은 `max_splits_limit`이다.

1. **고정 번호**: 차수 번호는 1..N이고 종목마다 겹치지 않는다. 추가 매수·1차 재진입 신호를 만들 때 대상 차수가 이미 보유 중이면 `RuntimeError`로 즉시 멈춘다.
2. **추가 매수 차수**: 2..N 중 비어 있는 가장 낮은 번호 k를 산다. 빈칸이 없으면 추가 매수하지 않는다.
3. **기준가**: k−1 차수의 매입가. k−1 차수가 비어 있으면 추가 매수하지 않는다. 예: 1·2차 빈칸, 3차 보유 → k=2, 1차 없음 → 매수 없음.
4. **조건과 체결**: 트리거 가격 = 기준가 × (1 − `additional_buy_drop_rate`). 신호일(T−1) 저가 ≤ 트리거이면 T0 시가에 산다. 현행과 같다.
5. **1차가 빈 경우**: 2~4번에 따라 추가 매수를 계속한다.
6. **1차 재진입**: 보유 중이지만 1차가 빈 종목도 신규 진입 후보 목록에 들어갈 수 있다. 신규 진입과 같은 조건(Tier1, 후보 순위, 쿨다운, 현금)을 통과하면 1차를 다시 산다. 이미 보유 중인 종목이므로 `max_stocks` 자리를 새로 쓰지 않는다. 포트폴리오가 `max_stocks`만큼 차 있어도 가능하다. 재진입 후보는 새 종목 후보와 같은 순위 목록(신규 진입 점수·시가총액 순)에 섞여 현금을 차례로 쓴다. 새 종목만 남은 `max_stocks` 자리 수로 제한한다.
7. **같은 날 중복 금지**: 그날 같은 종목에서 매도나 매수가 있었으면 그 종목은 추가 매수도, 1차 재진입도 하지 않는다. `cooldown_period_days`가 0이어도 같다. 하루 처리 순서는 현행 `매도 → 신규 진입(1차 재진입 포함) → 추가 매수`를 유지한다.
8. **우선순위**: `lowest_order`는 보유 차수 개수가 적은 종목이 먼저다. `highest_drop`은 3번 기준가 대비 신호일 종가 하락률이 큰 종목이 먼저다.
9. **바꾸지 않는 것**: 익절·손절·비활성 청산, 주문 금액, 호가·수수료·세금, 후보 산정.

`legacy`는 현행 동작을 그대로 유지한다. 새 번호는 `len+1`, 기준가는 번호가 가장 큰 차수, 1차가 없으면 추가 매수를 멈춘다.

## 검증 계획

1. **현행 고정 테스트**(`legacy`, 지금 통과해야 함): `[1, 3, 3]` 중복, 1차가 먼저 팔린 뒤 추가 매수 중단.
2. **새 규칙 테스트**(`fixed_slot_v1`, 구현 전에는 실패): 중간 빈칸 채우기, 앞 차수 기준가, 1차 빈칸 중 추가 매수 계속, 1차 재진입과 `max_stocks` 미사용, 최대 차수, 같은 날 매도 후 매수 금지, 앞 차수 없음, 잘못된 설정값 오류, 모든 시점의 차수 번호 중복 없음.
3. **CPU 구현 뒤**: `python tools/ci/run_cpu_checks.py`로 전체를 돌리고, `legacy` 결과가 바뀌지 않았는지 확인한다(`examples.cpu_split_demo` 기대값 포함).
4. **GPU 구현 뒤**: GPU 테스트(`test_backtest_strategy_gpu.py`의 빈칸·기준가 테스트 갱신)와 CPU-GPU 결과 비교. Windows 쪽에는 CuPy가 없지만, WSL Ubuntu의 기존 `rapids-env`(CUDA 12.9, CuPy 13.5.1)에서 GPU 테스트 34개 중 31개 통과를 확인했다(2026-11-15, 별도 브랜치 `claude/zealous-khorana-25e2bf`의 기록).
5. **제품 교차 확인**: 같은 보유 상태·가격을 split-trader V2 `decide`에 넣었을 때 같은 차수와 기준가가 나오는지 사례로 대조한다.

## CPU 구현 기록 (2026-11-15)

- 설정: `strategy_params.split_slot_policy` (`legacy` 기본). 정규화·GPU 거부는 [split_slot_policy.py](../src/split_slot_policy.py). `main_backtest`가 키를 전략에 넘긴다.
- GPU 차단: GPU는 `legacy` 커널만 있으므로 `fixed_slot_v1`이면 다음 단계에서 멈춘다. ① 공통 실행 설정 구성 `build_gpu_execution_params`(WFO 고정 후보 평가 `_build_runtime_contract`와 GPU 단일 실행이 사용), ② GPU 엔진 `run_magic_split_strategy_on_gpu` 진입(실행 설정에 키가 담겨 온 경우), ③ 설정 파일을 직접 읽는 GPU 탐색(`optimization/gpu/context.py`)·대조 도구(`cpu_gpu_parity_topk.py`, `parity_sell_event_dump.py`, `debug_gpu_single_run.py`). Codex 반론 검토(2026-11-15)가 처음에 ③만 있던 점을 지적해 ①②를 추가했다. WFO 후보 평가에 `fixed_slot_v1`을 넣으면 오류로 멈추는 것을 직접 확인했다. 변경 뒤 WSL `rapids-env`에서 GPU 검사 34개를 다시 돌려 31개 통과, 3개 실패로 변경 전 기록과 같다. 실패 3개는 성능 측정 원본 파일(`results/issue98_measure/`)이 없어서 나는 기존 실패다. ③의 GPU 스크립트(`debug_gpu_single_run.py` 등)는 DB가 필요해 직접 실행하지 않았다.
- 추가 매수: [strategy.py](../src/backtest/cpu/strategy.py)의 `generate_additional_buy_signals`가 정책에 따라 (대상 차수, 기준 차수)를 고른다. `legacy`는 `len+1`·마지막 차수로 기존 계산을 그대로 둔다.
- 1차 재진입: `generate_new_entry_signals`가 1차가 빈 보유 종목을 후보에 포함하고, 재진입은 `max_stocks` 자리 수에 세지 않는다. 거래 사유는 `1차 재진입`.
- 검사: [test_cpu_fixed_split_slot.py](../tests/test_cpu_fixed_split_slot.py) 15개. `legacy` 고정 4개, `fixed_slot_v1` 9개(여러 종목 재진입 1개 포함), GPU 거부 2개(정책 함수, 공통 실행 설정). `run_cpu_checks.py`에 추가해 전체 87개·예제 2개 통과(Windows·Python 3.11.9, 2026-11-15). 기존 `test_gpu_execution_policy.py` 2개도 통과. 실행 도구 자체 검사 24개 중 프로세스 신호·자식 정리 관련 2개(실패 2·오류 2)는 이 Windows PC에서 main 기준으로도 똑같이 실패해 이번 변경과 무관하다. CI(ubuntu)에서 다시 확인한다.
- main 병합(2026-11-15): #117(매도 체결가 반올림 설정 `sell_price_tick_rounding`)과 import·검사 개수가 겹쳐 둘 다 살려 합쳤다. 병합 뒤 CPU 검사 99개·예제 2개 통과, WSL GPU 검사 34개 중 31개 통과(기존 실패 3개 동일).
- 확인하지 않은 것: 규칙 8(`highest_drop` 우선순위) 전용 테스트, 실제 DB 기간 백테스트에서 `legacy`와 `fixed_slot_v1`의 성과 차이, GPU.

## 이번 단계 범위 밖

GPU 구현, 기본값 전환, ETF 확장, 매도 체결가 호가 올림([execution.py:164](../src/backtest/cpu/execution.py#L164)) 점검.
