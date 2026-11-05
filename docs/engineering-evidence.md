# 코드와 기록으로 보는 세 가지 구현

[프로젝트 소개](../README.md) → 이 문서 → 각 구현의 코드·테스트·변경 기록 순서로 읽을 수 있습니다.

개인 프로젝트의 설계·구현·테스트·운영을 주도했습니다. 여기서는 소스와 공개 기록에서 확인할 수 있는 문제 해결 과정을 설명합니다. 코드·테스트 링크는 조사 기준 커밋 `ea5683405a5c2e0e55a2f2365f4a1e0ff6022dd3`에 고정해 이후 변경에도 당시 구현을 확인할 수 있게 했습니다.

<a id="collection"></a>
## 1. 수집 중 일부가 실패해도 성공한 데이터와 실패 원인을 남기기

### 문제와 구현

여러 해의 데이터를 한 번에 수집하면 일부 요청이 실패하거나 데이터가 없는 구간을 반복 조회할 수 있습니다. 공매도 수집기는 조회 기간을 최대 729일 단위로 나누고 최신 구간부터 처리합니다. 뒤쪽 요청이 실패했을 때 이미 받은 행을 `partial_error` 상태와 함께 반환하고, 배치에서는 저장 행 수와 실패 건수를 별도로 집계합니다.

빈 구간은 `ShortSellingBackfillCoverage`에 `DONE_EMPTY`로 기록합니다. 재실행 때는 상장일, 기존 적재 범위, 빈 구간 기록을 참고해 조회 범위를 좁힙니다. 이 기록은 수집기가 빈 응답으로 판정한 구간을 뜻합니다. 접근 실패와 실제 데이터 부재를 구분하는 운영 확인도 필요합니다.

### 직접 확인할 근거

| 근거 | 확인할 내용 |
| --- | --- |
| [수집 함수](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/ea5683405a5c2e0e55a2f2365f4a1e0ff6022dd3/src/short_selling_collector.py#L690) | `_fetch_and_normalize_short_selling`: 성공한 행과 뒤쪽 실패를 함께 반환 |
| [배치 처리](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/ea5683405a5c2e0e55a2f2365f4a1e0ff6022dd3/src/short_selling_collector.py#L863) | `run_short_selling_batch`: 저장, 실패 유형별 집계, 진행 요약 |
| [부분 실패 테스트](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/ea5683405a5c2e0e55a2f2365f4a1e0ff6022dd3/tests/test_short_selling_collector.py#L128) | 뒤쪽 요청 실패 후에도 앞서 받은 1행 보존, 저장·오류 수 분리 |
| [빈 구간 테스트](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/ea5683405a5c2e0e55a2f2365f4a1e0ff6022dd3/tests/test_short_selling_collector.py#L217) | 재실행에 사용할 빈 구간 기록 작성 |
| [PR #100](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/pull/100) | 수집 제어 옵션, 재실행 처리, 스키마·문서·테스트를 함께 반영한 변경 기록 |

테스트는 외부 API와 DB를 모의 객체로 대체합니다. [실행 안내](../tests/README.md)의 공매도 수집기 검사로 확인할 수 있습니다. 실제 제공처 접속 성공이나 수집 시간 단축률을 증명하는 검사는 아닙니다.

<a id="gpu"></a>
## 2. 병목을 측정하고 결과 일치를 확인하며 GPU 계산 시간 줄이기

### 문제와 구현

GPU는 많은 설정 조합을 동시에 계산하지만 준비·정렬·상태 갱신 비용이 크면 전체 작업이 느려집니다. 구간별 측정 기능을 추가해 추가 매수 후 마지막 거래일을 갱신하는 과정의 중복 제거 계산에 시간이 집중된 것을 확인했습니다.

매수 대상으로 확정된 좌표에 직접 날짜를 기록하도록 바꾸고, 디버그 모드에서는 좌표 중복 여부를 확인합니다. 성능이 나빠진 앞선 세 실험은 되돌렸으며 그 결과도 기록에 남아 있습니다.

### 직접 확인할 근거

| 근거 | 확인할 내용 |
| --- | --- |
| [GPU 상태 갱신](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/ea5683405a5c2e0e55a2f2365f4a1e0ff6022dd3/src/backtest/gpu/logic.py#L650) | 중복 제거를 반복하던 갱신을 직접 대입으로 변경, 디버그 검사 유지 |
| [갱신 대상 회귀 테스트](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/ea5683405a5c2e0e55a2f2365f4a1e0ff6022dd3/tests/test_backtest_strategy_gpu.py#L759) | 허용된 매수 대상만 마지막 거래일이 바뀌는지 확인 |
| [측정 도구](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/ea5683405a5c2e0e55a2f2365f4a1e0ff6022dd3/src/issue98_perf_measure.py) | 실행 시간·구간별 시간·환경·입력 정보를 측정 산출물로 기록 |
| [PR #105](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/pull/105) | 개선 결과와 CPU·GPU 일치 검사 요약 |
| [실험·원복·최종 판정 기록](../todos/done_2026_03_13-issue104-gpu-throughput-followup-hotpath.md) | 효과가 없었던 변경과 채택한 변경을 함께 추적 |

### 당시 측정 결과와 조건

아래는 2026-03-13 기록의 개선 전후 비교입니다. CPU 대비 속도가 아니라 **같은 GPU 경로의 변경 전후 실행 시간**입니다.

| 항목 | 조건 또는 결과 |
| --- | --- |
| 입력 기간 | 2024-01-01 ~ 2024-02-29 |
| 설정 조합 | 360개, 묶음당 90개, 총 4묶음 |
| 측정 규격 | `issue98_janfeb2024_multibatch_research020`, 각 2회 실행한 중앙값 |
| 데이터 포함 기준 | `min_tier12_coverage_ratio=0.2`; 계산 시간 측정용 연구 설정으로, 전략 승인 조건과 다름 |
| GPU 계산 구간 | 1,011.80초 → 51.02초, 약 94.96% 감소 |
| 전체 실행 시간 | 1,049.52초 → 87.30초, 약 91.68% 감소 |
| 실행 상태 | 두 실행 모두 메모리 부족 재시도 없음, 각 4묶음 처리 |

측정 규격은 [#98 기록의 19-3절](../todos/done_2026_02_17-issue98-gpu-throughput-refactor.md), 최종 수치는 [#104 기록의 19-8절](../todos/done_2026_03_13-issue104-gpu-throughput-followup-hotpath.md)과 PR #105에 있습니다. 비교 식은 `(변경 전 시간 - 변경 후 시간) / 변경 전 시간`입니다. 기록에 적힌 `+94.96%` 등을 이 문서에서는 시간 감소율로 풀어 썼습니다.

기준 실행 ID는 `pr98_head_final_janfeb2024_cov020_20260312_223231`, 변경 후 실행 ID는 `pr104_h005b_final_janfeb2024_cov020_20260313_212906`입니다. 원본 `summary.json`·환경 기록은 Git에 포함되지 않은 `results/` 경로를 가리킵니다. **정확한 장비·드라이버와 원본 로그는 공개 자료만으로 확인되지 않았고 이번 정비에서 GPU 재측정하지 않았습니다.**

CPU·GPU 일치 검사도 기록된 검증 사례에서 실패 0건, 매매 결정 불일치 0건으로 보고된 범위에 한정합니다. 모든 입력에서의 일치나 투자 성과를 보장하는 수치로 쓰지 않습니다.

<a id="availability"></a>
## 3. 데이터 반영 시점을 바꾸기 전에 영향 범위 확인하기

### 문제와 구현

공매도 데이터에는 거래 기준일과 실제로 알 수 있는 시점의 차이가 있습니다. 기준일 당일에 그 값을 넣으면 당시 알 수 없던 정보를 과거 시뮬레이션에서 사용하게 될 수 있습니다.

현재 코드는 임시 정책으로 **3거래일 뒤에 반영**합니다. 거래일은 입력 가격 데이터에 포함된 날짜를 사용합니다. 데이터가 실제로 언제 공개되는지에 대한 확정값으로 3을 주장하지 않습니다.

정책 변경 전후를 같은 입력으로 계산하는 읽기 전용 비교 도구를 추가하고, 공매도 관련 비율과 최종 종목 등급이 얼마나 달라지는지 확인했습니다. 재계산 대상으로 결정한 것은 파생 결과인 `DailyStockTier`이며, 원본 공매도 데이터의 재수집과 구분합니다.

### 직접 확인할 근거

| 근거 | 확인할 내용 |
| --- | --- |
| [반영 날짜 계산](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/ea5683405a5c2e0e55a2f2365f4a1e0ff6022dd3/src/pipeline/daily_stock_tier_batch.py#L589) | 입력 거래일 기준으로 반영 날짜 이동 |
| [반영 시점 테스트](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/ea5683405a5c2e0e55a2f2365f4a1e0ff6022dd3/tests/test_daily_stock_tier_batch.py#L1016) | 2거래일 지연을 지정한 사례에서 당일 미반영·가용일 반영 확인 |
| [영향 비교 도구](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/ea5683405a5c2e0e55a2f2365f4a1e0ff6022dd3/src/pipeline/daily_stock_tier_shadow.py) | 같은 입력에 지연값 0과 3을 적용한 차이 요약 |
| [정책 변경 커밋](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/commit/0f5bf06cd3b69efe45333e4149aa99dbdb721a66) · [비교 도구 커밋](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/commit/b1cd7cd0b32f7a21322782f4e46a04aa8e2c09ce) | PR #106으로 병합된 구현 이력 |
| [영향 분석·재계산 결정 기록](../todos/2026_03_07-short-selling-publication-lag-pit.md) | 비교 결과와 당시 재계산 진행 상황 |

### 당시 영향 분석

2026-03-13 기록에서 비교한 구간은 **2025-12-01 ~ 2026-02-06**입니다.

| 항목 | 기록된 결과 |
| --- | --- |
| 비교 행 | 130,562행 |
| 영향 행 | 70,045행, 47개 날짜 |
| 공매도 관련 비율의 값 변경 | 61,641행 |
| 최종 종목 등급 변경 | 302행 |
| 등급 사유 변경 | 453행 |

비율이 새로 생기거나 사라진 행도 있어 영향 행 수와 값 변경 행 수는 다릅니다. 이 수치는 투자 수익률의 변화가 아닙니다. 원본 결과 파일은 Git에 포함되지 않았으며 이번에 운영 DB로 재실행하지 않았습니다.

**코드·테스트·영향 분석은 반영됐지만, 장기 데이터 재계산 전체 완료는 공개 기록으로 확인되지 않습니다.** 2026-03-14 기록은 2017-04-18부터 재개하는 지점까지 남아 있습니다. 현재 실행 중이라는 뜻으로 읽지 않습니다.

이 동작의 작은 부분은 [가상 데이터 CPU 예제](../tests/README.md#example)로 직접 확인할 수 있습니다.
