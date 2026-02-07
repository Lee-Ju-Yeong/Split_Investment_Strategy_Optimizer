# Tests Guide

`tests/` 디렉토리의 실행 방법과 의존성을 정리한 문서입니다.
프로젝트 루트(`Split_Investment_Strategy_Optimizer/`)에서 실행하세요.

## 1. 기본 실행

### 1-1. 전체 테스트 실행
```bash
python -m unittest discover -s tests
```

### 1-2. 권장 환경(rapids-env)에서 실행
```bash
conda run -n rapids-env python -m unittest discover -s tests
```

## 2. 테스트 분류

### 2-1. 빠른 단위 테스트(로컬 의존성 낮음)
- `tests/test_data_handler.py`
- `tests/test_data_handler_tier.py`
- `tests/test_point_in_time.py`
- `tests/test_db_setup.py`
- `tests/test_pipeline_batch.py`
- `tests/test_daily_stock_tier_batch.py`
- `tests/test_indicator_calculator.py`
- `tests/test_portfolio.py`

예시:
```bash
conda run -n rapids-env python -m unittest \
  tests.test_data_handler \
  tests.test_data_handler_tier \
  tests.test_point_in_time \
  tests.test_db_setup -v
```

### 2-2. DB 연동 통합 테스트
- `tests/test_integration.py`

필수 조건:
- 로컬 MySQL 접근 가능
- `config.ini`에 유효한 DB 접속 정보 존재
- 테스트가 `WeeklyFilteredStocks`, `DailyStockPrice`, `CalculatedIndicators`에 테스트 데이터를 삽입/삭제

실행:
```bash
conda run -n rapids-env python -m unittest tests.test_integration -v
```

### 2-3. GPU 의존 테스트
- `tests/test_backtest_strategy_gpu.py`

필수 조건:
- CUDA 사용 가능 환경
- `cupy`, `cudf` 설치

실행:
```bash
conda run -n rapids-env python -m unittest tests.test_backtest_strategy_gpu -v
```

## 3. 이슈 #66 관련 신규 테스트

- `tests/test_pipeline_batch.py`: 배치 오케스트레이터 인자/호출 검증
- `tests/test_daily_stock_tier_batch.py`: Tier 계산(유동성 + financial risk) 검증
- `tests/test_data_handler_tier.py`: Tier as-of 조회 API 검증

권장 실행:
```bash
conda run -n rapids-env python -m unittest \
  tests.test_pipeline_batch \
  tests.test_daily_stock_tier_batch \
  tests.test_data_handler_tier \
  tests.test_point_in_time \
  tests.test_db_setup -v
```

## 4. 운영 팁

- 테스트 실패 시 먼저 환경 확인:
  - `pandas`, `pymysql`, `mysql-connector-python`, `pykrx`
  - GPU 테스트의 경우 `cupy`, `cudf`
- DB 연동 테스트 전후로 테스트 데이터 정리 여부를 확인하세요.
- CI/자동화에서는 GPU 테스트를 별도 job으로 분리하는 것을 권장합니다.
