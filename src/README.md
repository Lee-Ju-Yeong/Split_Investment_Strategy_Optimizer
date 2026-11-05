# 코드 읽기 안내

처음에는 [프로젝트 목적](../README.md)과 [세 가지 구현 사례](../docs/engineering-evidence.md)를 읽고 아래 순서로 코드를 확인하세요. 표의 검사는 각 구현의 중요한 경계를 보여줍니다. GPU·외부 수집·전체 제품을 모두 검증하는 묶음은 아닙니다.

## 추천 읽기 순서

| 순서와 관심사 | 실제 구현 | 함께 읽을 검사·기록 |
| --- | --- | --- |
| 1. 일부 수집이 실패했을 때 | [short_selling_collector.py](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/src/short_selling_collector.py): `_fetch_and_normalize_short_selling`, `run_short_selling_batch` | [test_short_selling_collector.py](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/tests/test_short_selling_collector.py), [PR #100](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/pull/100) |
| 2. 데이터가 언제 사용되는지 | [pipeline/daily_stock_tier_batch.py](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/src/pipeline/daily_stock_tier_batch.py): `build_daily_stock_tier_frame` | [test_daily_stock_tier_batch.py](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/tests/test_daily_stock_tier_batch.py), [여섯 행 예제](../tests/README.md#example) |
| 3. 거래와 현금 상태가 어떻게 연결되는지 | [backtest/cpu](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/tree/main/src/backtest/cpu): 엔진 → 전략 → 체결 → 포트폴리오 | [test_portfolio.py](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/tests/test_portfolio.py), [실제 MySQL·CPU 합성 검사](../docs/testing-mysql.md) |
| 4. 반복 계산을 어떻게 개선했는지 | [backtest/gpu/logic.py](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/src/backtest/gpu/logic.py): 추가 매수·마지막 거래일 갱신 | [test_backtest_strategy_gpu.py](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/tests/test_backtest_strategy_gpu.py), [측정 조건·한계](../docs/engineering-evidence.md#gpu) |

## 구현 위치

| 묶음 | 역할 |
| --- | --- |
| `pipeline/` | 데이터 수집·배치·당시 후보 구성·종목 등급 계산 |
| `data_handler.py` | SQL 조회, 가격 기준·후보·캐시 제공 |
| `db_setup.py` | 테이블 생성. 실제 DB를 변경하므로 읽기 안내로 먼저 확인 |
| `backtest/cpu/` | 하루씩 진행하는 기준 계산과 매매·자산 상태 |
| `backtest/gpu/` | 배열 기반 병렬 계산과 메모리·배치 처리 |
| `analysis/` | 기간을 나눈 평가, 후보 목록과 최종 판정 근거 |

`pipeline_batch.py`, `walk_forward_analyzer.py` 등 최상위 파일 일부는 기존 실행 명령을 유지하는 연결 파일입니다. [정확한 경로 대응](../docs/refactoring/issue69-import-path-mapping.md)과 [호환 정책](../docs/refactoring/wrapper-deprecation.md)을 참고하세요. `notebook/`·`archive/legacy_tools/`·`llm-context/`는 별도의 연구·과거 자료입니다.

외부 연결 없는 확인은 [CPU 기본 검사](../tests/README.md), 임시 MySQL까지 사용하는 확인은 [DB 통합 검사](../docs/testing-mysql.md)로 나뉩니다. 자동 검사의 성공 범위를 실제 전략의 유효성이나 운영 성과로 확대하지 않습니다.
