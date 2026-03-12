# 매직스플릿 투자 전략 최적화 (Split Investment Strategy Optimizer)

**매직스플릿(Magic Split) 분할 투자 전략**의 백테스팅 및 GPU 기반 파라미터 최적화 시스템입니다. 한국 주식 시장의 방대한 데이터를 활용하여 과최적화를 방지하고, Walk-Forward Optimization(WFO)을 통해 실제 시장에서 작동하는 강건한 파라미터를 도출하는 것을 목표로 합니다.

---

## 🚀 프로젝트 핵심 가치

- **CPU=SSOT (Source of Truth)**: 모든 전략 승격 판단은 CPU 백테스터를 기준으로 하며, GPU 경로는 parity evidence로 검증합니다.
- **Strict PIT-Oriented Runtime**: 핵심 runtime은 엄격한 시점별 유니버스와 수정주가 gate를 사용하며, lag-sensitive 데이터 정책은 별도 gate로 관리합니다.
- **GPU Throughput Refactor**: 대규모 파라미터 탐색을 더 빠르게 만들기 위해 벡터화 hot path와 측정 도구를 지속적으로 개선하고 있습니다.
- **Cluster-Oriented Design**: k3s/Kubernetes 기반 GPU job 운영은 연구 및 인프라 로드맵으로 관리하고 있습니다.

---

## 📅 현재 상태 (2026-03 기준)

- **거버넌스 확정 (Issue #97)**: 운영 및 승격용 런타임 정책(Strict-only)을 확정하여 전략의 신뢰성을 확보했습니다.
- **GPU 성능 개선 진행 (Issue #98)**: 랭킹 텐서 사전 계산 및 실행 루프 최적화를 중심으로 시뮬레이션 처리량(Throughput)을 계속 측정하고 개선하고 있습니다.
- **데이터 인프라 운영 중**: 상폐 종목을 포함한 Historical Universe(PIT), 재무/수급/Tier 사전 계산 배치를 운영하고 있습니다.
- **인프라 현대화**: 로컬 PC를 넘어 k3s 클러스터 기반의 서버급 GPU 자원 활용 연구가 진행 중입니다.

---

## 🛠 아키텍처 (5-Stage Pipeline)

1.  **Data Pipeline**: pykrx 기반 OHLCV 및 기술적 지표 ETL (`src/main_script.py`)
2.  **Batch Precompute**: PIT 유니버스, 재무/수급, 종목 Tier 사전 계산 (`src/pipeline_batch.py`)
3.  **CPU Backtester**: OOP 기반의 정밀한 시뮬레이션 및 결과 검증 (`src/main_backtest.py`)
4.  **GPU Optimizer**: 다차원 배열(State Arrays) 기반의 대규모 파라미터 시뮬레이션 (`src/parameter_simulation_gpu.py`)
5.  **WFO Analysis**: Walk-Forward Optimization을 통한 전략 강건성 검증 (`src/walk_forward_analyzer.py`)

---

## ⚙️ 빠른 시작

### 1) 환경 구성
```bash
# CPU 환경 (Standard)
conda env create -f environment.yml
conda activate stock_optimizer_env

# GPU 환경 (RAPIDS/CUDA 필수, 별도 수동 준비)
conda activate rapids-env
```

`rapids-env`는 이 저장소의 `environment.yml`로 생성되지 않습니다. CUDA/RAPIDS가 설치된 호스트에서 별도로 준비된 GPU 환경을 사용합니다.

### 2) 설정 및 DB 초기화
```bash
# 설정 파일 복사 및 수정
cp config/config.example.yaml config/config.yaml

# DB 스키마 생성
python -c "from src.db_setup import get_db_connection, create_tables; conn=get_db_connection(); create_tables(conn); conn.close()"
```

### 3) 주요 실행 명령
```bash
# 데이터 파이프라인 및 지표 계산
python -m src.main_script

# 재무·수급·Tier 배치
python -m src.pipeline_batch --mode backfill --start-date <YYYYMMDD> --end-date <YYYYMMDD>
python -m src.pipeline_batch --mode daily --end-date <YYYYMMDD>

# PIT 유니버스 배치
python -m src.ticker_universe_batch --mode backfill --start-date <YYYYMMDD> --end-date <YYYYMMDD> --step-days 7 --workers 1
python -m src.ticker_universe_batch --mode daily --end-date <YYYYMMDD>

# CPU 백테스트
python -m src.main_backtest

# GPU 파라미터 최적화
python -m src.parameter_simulation_gpu

# GPU 시뮬레이션 성능 측정 (`--label` 필수)
python -m src.issue98_perf_measure --label issue98_baseline --runs 3

# WFO 분석 파이프라인 실행
python -m src.walk_forward_analyzer

# 테스트 / 웹 UI
python -m unittest discover -s tests -p 'test_*.py'
python -m src.app
```

---

## 📋 현재 포커스

최신 우선순위와 상태는 `TODO.md`를 단일 소스(SSOT)로 관리합니다. README에는 현재 읽기 시작할 때 필요한 포커스만 요약합니다.

### 지금 먼저 볼 것
- `ShortSellingDaily publication lag`: PIT 리스크가 남아 있어 정책 확정이 먼저 필요합니다.
- `#98 GPU throughput refactor`: canonical baseline 재측정과 다음 hot-path 판단이 진행 중입니다.
- `#68 Robust WFO / Ablation`: 공식 스코어링 로직을 고정하는 다음 단계입니다.

### 참고
- `#72`, `#101`, `GPU-native WFO v2` 같은 backlog/research 항목은 README에서 고정 우선순위를 다시 쓰지 않고 `TODO.md`에서만 추적합니다.

---

## 📚 주요 문서
- **전략 원칙**: [MAGIC_SPLIT_STRATEGY_PRINCIPLES.md](docs/MAGIC_SPLIT_STRATEGY_PRINCIPLES.md)
- **DB 스키마**: [schema.md](docs/database/schema.md)
- **테스트 가이드**: [tests/README.md](tests/README.md)
- **작업 백로그**: [TODO.md](TODO.md)

---
**Maintainer**: 퀀트-J (Senior Quant System Developer)
