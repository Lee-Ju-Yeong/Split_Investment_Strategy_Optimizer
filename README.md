# 분할 투자 전략 최적화 (Split Investment Strategy Optimizer)

분할 매수/분할 매도(Magic Split) 전략을 데이터 파이프라인, CPU/GPU 백테스터, WFO 분석으로 검증하는 프로젝트입니다.

## 현재 상태 (2026-02 기준)
- `PIT(Point-in-Time)` 보호 로직 적용: T-1 신호/룩어헤드 방지 (`src/data_handler.py`, `src/strategy.py`)
- 재무/수급/Tier 스키마 확장 완료: `FinancialData`, `InvestorTradingTrend`, `DailyStockTier` (`src/db_setup.py`)
- 배치 오케스트레이터 도입: 백필/일배치 분리 실행 (`src/pipeline_batch.py`)
- Historical Universe Phase 1 추가: `TickerUniverseSnapshot`/`TickerUniverseHistory` 배치 (`src/ticker_universe_batch.py`)
- 상세 로드맵과 진행 상태는 `TODO.md`를 기준으로 관리

## 아키텍처
1. **Data Pipeline**: 종목/주가/지표 수집 (`src/main_script.py`)
2. **Batch Precompute**: 재무/수급 수집 + Tier 사전 계산 (`src/pipeline_batch.py`)
3. **Backtest (CPU/GPU)**: 전략 검증/파라미터 탐색 (`src/main_backtest.py`, `src/parameter_simulation_gpu.py`)
4. **Robustness (WFO/Clustering)**: 강건 파라미터 분석 (`src/walk_forward_analyzer.py`)

## 디렉토리
- `src/`: 핵심 코드
- `tests/`: 단위/통합/GPU 테스트 (`tests/README.md` 참고)
- `config/`: `config.yaml` 템플릿
- `docs/`: 전략/스키마 문서
- `todos/`: 이슈 단위 작업 문서

## 빠른 시작

### 1) 환경 구성
```bash
# conda 권장 (CPU 기본 환경: environment.yml)
conda env create -f environment.yml
conda activate stock_optimizer_env

# GPU 실행 시 (별도 RAPIDS 환경)
conda activate rapids-env

# 또는 pip
pip install -r requirement.txt
```

### 2) 설정 파일
- `config/config.example.yaml` -> `config/config.yaml` 복사 후 수정
- 단일 설정 소스: `config/config.yaml` (필요 시 `MAGICSPLIT_CONFIG_PATH`로 override)
  - `database`: DB 접속 정보
  - `data_pipeline`: `src/main_script.py` 경로/플래그 (하드코딩 제거)

### 3) DB 스키마 생성/갱신
```bash
python -c "from src.db_setup import get_db_connection, create_tables; conn=get_db_connection(); create_tables(conn); conn.close()"
```

## 실행 명령

> Last Verified: 2026-02-08

### 데이터 파이프라인(OHLCV/지표)
```bash
python -m src.main_script
```

### 재무·수급·Tier 배치 (신규)
```bash
# 초기 백필
python -m src.pipeline_batch --mode backfill --start-date 20150101 --end-date <YYYYMMDD>

# 일배치
python -m src.pipeline_batch --mode daily --end-date <YYYYMMDD>
```

### Historical Universe 배치 (신규, 상폐 포함 유니버스)
```bash
# Phase 1 백필 (권장: workers=1부터 시작)
python -m src.ticker_universe_batch --mode backfill --start-date 20100101 --end-date <YYYYMMDD> --step-days 7 --workers 1

# 일배치 (당일 스냅샷 + history 갱신)
python -m src.ticker_universe_batch --mode daily --end-date <YYYYMMDD>
```

### 백테스트 / 최적화 / 분석
```bash
# CPU 백테스트
python -m src.main_backtest

# GPU 파라미터 시뮬레이션
python -m src.parameter_simulation_gpu

# WFO 분석
python -m src.walk_forward_analyzer
```

### Web UI
```bash
python -m src.app
```

## 데이터 스키마 핵심
- 가격: `DailyStockPrice`
- 지표: `CalculatedIndicators`
- 유니버스: `WeeklyFilteredStocks`, `CompanyInfo`
- PIT 유니버스(신규): `TickerUniverseSnapshot`, `TickerUniverseHistory`
- 재무: `FinancialData`
- 수급: `InvestorTradingTrend`
- 사전 계산 Tier: `DailyStockTier`

스키마 상세: `docs/database/schema.md`

## 테스트
```bash
python -m unittest discover -s tests
```

권장(rapids-env):
```bash
conda run -n rapids-env python -m unittest discover -s tests
```

테스트 분류/의존성: `tests/README.md`

## 로드맵
최신 우선순위는 `TODO.md`를 단일 소스로 유지합니다.

### P0 (신뢰도/데이터 정합성)
- [x] #64 PIT 규칙 및 룩어헤드 방지
- [x] #65 스키마/인덱스 확장
- [ ] #66 재무·수급 수집기 분리 + Tier 사전 계산 배치 운영 적용(백필/일배치)
- [x] #70 상폐 포함 Historical Universe 구축 (Phase 1/2 코드 반영 및 운영 검증 완료)

### P1 (운영 안정화)
- [ ] #67 Tier fallback/PIT 조인 고도화
- [ ] #53 설정 소스 표준화
- [ ] #54 데이터 파이프라인 모듈화
- [ ] #55 구조화 로깅

### P2 (전략 고도화)
- [ ] #68 멀티팩터 랭킹 + WFO/Ablation
- [ ] #56 CPU/GPU Parity 하네스 강화

## 문서 맵
- 전략 원칙: `docs/MAGIC_SPLIT_STRATEGY_PRINCIPLES.md`
- DB 스키마: `docs/database/schema.md`
- 작업 백로그: `TODO.md`
- 이슈 작업 문서: `todos/`
