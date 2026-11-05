# 연구 노트북

초기 데이터 수집과 탐색에 사용한 노트북을 보존합니다. 저장된 출력은 당시 실험의 결과이며, 현재 환경에서 다시 실행한 검증 결과가 아닙니다. 기본 실행 경로는 [작은 CPU 예제와 검사](../tests/README.md)입니다.

| 노트북 | 내용과 경계 |
| --- | --- |
| [correlation_of_stocks.ipynb](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/notebook/correlation_of_stocks.ipynb) | 주가 상관관계 탐색 기록 |
| [stock_etf_data_collector.ipynb](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/notebook/stock_etf_data_collector.ipynb) | 주식·ETF 데이터 수집 실험. 외부 연결 조건은 실행 전에 확인 필요 |
| [test.ipynb](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/blob/main/notebook/test.ipynb) | 작은 탐색용 코드. 자동 회귀 테스트 모음은 아님 |

현재 제품 구현은 `src/`, 자동 검사는 `tests/`에 있습니다. 연구 노트북의 용량이나 GitHub 대표 언어 표시만으로 구현의 중심을 판단하기보다 [세 구현의 코드·테스트·변경 기록](../docs/engineering-evidence.md)을 확인할 수 있습니다. 이번 정비에서는 노트북 코드와 출력을 수정하거나 재실행하지 않았습니다.
