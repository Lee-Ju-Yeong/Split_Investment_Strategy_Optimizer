# src 패키지 구조 재편 및 대형 모듈 브레이크다운(동작 동일 리팩터링) (Issue #69)
(현재 파일 이름은 YYYY_MM_DD-issue<이슈번호>-<issue_name_only_english>.md 로 지정)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/69`
- 요약:
  - `src`의 평면 구조/대형 모듈을 **기능 변경 없이** 단계적으로 정리
  - 대형 모듈을 책임 단위로 분해(헬퍼/서비스/러너)
  - 기존 실행 커맨드 호환을 위해 엔트리포인트는 thin wrapper 유지
  - import 경로 변경 가이드 문서화

- 완료 조건(DoD):
  - 대형 모듈이 책임 단위로 분해되어 파일 경계가 명확함
  - 기존 주요 실행 경로가 동일하게 동작:
    - `python -m src.main_backtest`
    - `python -m src.pipeline_batch`
    - `python -m src.main_script`
  - 최소 스모크 테스트/단위 테스트 통과
  - 변경 가이드(이전 import 경로 -> 신규 경로) 문서화

## 1. 배경(현재 이슈의 대략적인 이전 맥락)
- 현황:
  - `src`에 대형 파일과 중복 구현이 혼재되어 리뷰/유지보수 비용이 높음
  - 대형 모듈(예시):
    - `src/backtest_strategy_gpu.py` (약 895 LOC)
    - `src/parameter_simulation_gpu.py` (약 386 LOC)
    - `src/walk_forward_analyzer.py` (약 333 LOC)
    - `src/daily_stock_tier_batch.py` (약 325 LOC)
    - `src/ohlcv_batch.py` (약 324 LOC)
    - `src/strategy.py` (약 320 LOC)
  - 중복/분산 구현(예시):
    - `Position` 중복 정의 (`src/strategy.py`, `src/portfolio.py`)
    - CompanyInfo 캐시 관리 지점 분산 (`src/data_handler.py`, `src/company_info_manager.py`)
    - GPU preload/bootstrap 로직 중복 (`src/parameter_simulation_gpu.py`, `src/debug_gpu_single_run.py`)

- 선행/연관 이슈:
  - 제외 범위(별도 이슈에서 처리): #54, #57, #58, #60, #61

#### 1-1. 핵심 원칙/제약
- 이 이슈에서는 **전략/체결/신호의 기능 로직 변경 금지**
- 기존 실행 커맨드 호환성 유지(thin wrapper)
- 노트북 환경(no DB / no GPU)에서도 import-safe, 테스트 skip-safe 유지 (가능한 범위에서)
---

## 2. 요구사항(구현하고자 하는 필요한 기능)
### 2-1. `src` 하위 패키지 구조 도입
- 예시 구조(초안):
  - `src/backtest/cpu`
  - `src/backtest/gpu`
  - `src/data/collectors`
  - `src/data/pipeline`
  - `src/common`

### 2-2. 대형 모듈을 책임 단위로 분해
- 대상 후보:
  - `src/backtest_strategy_gpu.py`
  - `src/parameter_simulation_gpu.py`
  - `src/walk_forward_analyzer.py`
  - `src/daily_stock_tier_batch.py`, `src/ohlcv_batch.py`
  - `src/strategy.py`
- 분해 방향:
  - entrypoint는 thin wrapper로 남기고, 내부 구현을 패키지 하위로 이동
  - 중복 구현은 가능한 범위에서 공통 모듈로 정리(단, 의미/기능 변경 금지)

### 2-3. 기존 실행 커맨드/호환성 유지
- 아래 커맨드는 유지되어야 함:
  - `python -m src.main_backtest`
  - `python -m src.pipeline_batch`
  - `python -m src.main_script`

### 2-4. 스모크/단위 테스트 통과
- 최소 요구:
  - 기존 유닛 테스트 통과
  - 노트북 환경(no DB / no GPU)에서 스킵되는 테스트는 명시적으로 skip 처리 유지

### 2-5. 문서화
- "이전 import 경로 -> 신규 경로" 매핑/가이드 작성

#### 2-5-1. 우선적으로 참조할 파일
- `TODO.md`
- `src/backtest_strategy_gpu.py`
- `src/parameter_simulation_gpu.py`
- `src/walk_forward_analyzer.py`
- `src/main_backtest.py`
- `src/main_script.py`
- `src/pipeline_batch.py` (존재 시)

---

(하위 부분은 사람이 작성하는게 아니라 AI 가 작성하는 부분)

# AI 결과

## 3. (AI가 확인한) 기존 코드/구현의 핵심내용들/의도들
- 엔트리포인트(호환성 중요):
  - `src/main_backtest.py`, `src/pipeline_batch.py`, `src/main_script.py`는 `python -m ...` 뿐 아니라 `python src/<file>.py` 직접 실행도 지원하기 위해 **BOOTSTRAP 블록**을 포함함
  - 따라서 (1) 파일을 그대로 유지하거나 (2) 동일한 thin wrapper + BOOTSTRAP 패턴을 유지해야 함

- import 규칙(이슈 #61):
  - `tests/test_issue61_import_style_standardization.py`는 `src/*.py` 내부에서 `src.<module>` 절대 import 및 top-level 모듈 절대 import를 금지하고, 상대 import를 강제함
  - `if __name__ == "__main__":`가 있는 엔트리포인트는 `# BOOTSTRAP:` 주석이 상대 import보다 위에 있어야 함

- import-side-effects 규칙(이슈 #60/#68):
  - `tests/test_issue60_import_side_effects.py`: `import src.parameter_simulation_gpu`가 stdout 출력 없이 성공해야 하고, `find_optimal_parameters` 심볼을 제공해야 함
  - `tests/test_issue68_wfo_import_side_effects.py`: `import src.walk_forward_analyzer`가 GPU deps(`cupy/cudf`) 없이도 성공해야 함

- GPU/DB 의존성 경계(노트북 환경 고려):
  - `src/backtest_strategy_gpu.py`는 module import 시점에 `cupy/cudf`를 import함(무GPU 환경에서 import 불가)
  - 현재는 `src.parameter_simulation_gpu_lib._ensure_gpu_deps()`가 GPU deps 확인 후에만 `src.backtest_strategy_gpu`를 import하도록 설계되어, CPU-only 환경에서도 `parameter_simulation_gpu` import가 안전함
  - 리팩터링 과정에서도 GPU 모듈은 CPU/WFO 엔트리포인트에서 직접 import되지 않도록 경계를 유지해야 함

---

## 4. 생각한 수정 방안들 (ai 가 생각하기에) 구현에 필요한 핵심 변경점
(수정 방안 최소 3가지, PR 단위 분할 가능해야 함)

### 4-1. A안: 레이어드/도메인 우선 “빅뱅” 재배치 + 상단 thin wrapper 유지
- 패키지 구조(예시):
  - `src/common/` (config, logging, utils)
  - `src/data/` (collectors, db)
  - `src/backtest/cpu/` (engine/strategy/portfolio/execution)
  - `src/backtest/gpu/` (tensors/kernels/runner; GPU deps는 lazy)
  - `src/optimization/gpu/` (parameter search)
  - `src/analysis/` (wfo, robust scoring)
- 상단 엔트리포인트는 유지:
  - `src/main_backtest.py`: `from .entrypoints.main_backtest import main` 형태로 delegate
  - `src/pipeline_batch.py`, `src/main_script.py`도 동일
- 장점:
  - 구조가 가장 “깨끗하게” 정리되고 경계가 명확해짐
- 리스크:
  - 한 번에 이동 범위가 커서 import 경로 회귀/충돌/리뷰 난이도가 큼
- 테스트/검증:
  - 기존 테스트 + 신규 `tests/test_issue69_entrypoint_compat.py`(import + 심볼 존재) 추가

### 4-2. B안: Wrapper-first 점진 리팩터링(가장 PR 친화적, 권장 후보)
- 핵심 아이디어:
  - 기존 `src/*.py` 파일명/엔트리포인트는 그대로 두고, 구현을 `src/<subpkg>/...`로 점진적으로 “빼내기”
  - 각 PR은 1~2개 대형 모듈만 분해하고, 상단 wrapper는 delegate/re-export만 수행
- 단계적 진행(예시):
  - PR-0(안전망): `tests/test_issue69_entrypoint_compat.py` 추가 + 폴더 스켈레톤만 생성
  - PR-1: `src/backtest_strategy_gpu.py` 내부를 `src/backtest/gpu/*`로 분해(가능하면 GPU import lazy), wrapper는 `run_magic_split_strategy_on_gpu`만 노출
  - PR-2: `src/walk_forward_analyzer.py` 내부 분석/robust 로직을 `src/analysis/wfo/*`로 이동(무GPU import 보장 유지)
  - PR-3: 파이프라인 배치(`src/pipeline_batch.py`, `src/ohlcv_batch.py`, `src/ticker_universe_batch.py`, `src/daily_stock_tier_batch.py`)를 `src/pipeline/*`로 이동
  - PR-4: CPU 백테스터(`src/backtester.py`, `src/strategy.py`, `src/portfolio.py`, `src/execution.py`)를 `src/backtest/cpu/*`로 이동
- 장점:
  - 노트북(no DB / no GPU) 환경에서도 “import-safe”를 유지하면서 점진적으로 진행 가능
  - 충돌/리뷰 부담이 작고, PR 실패 시 롤백도 쉬움
- 리스크:
  - 중간 단계에서 wrapper가 많아 “구조가 잠시 더 복잡”해 보일 수 있음(하지만 완료 시 정리 가능)
- 테스트/검증(필수):
  - 기존: `tests/test_issue61_import_style_standardization.py`, `tests/test_issue60_import_side_effects.py`, `tests/test_issue68_wfo_import_side_effects.py`
  - 신규: `tests/test_issue69_entrypoint_compat.py`
    - import: `src.main_backtest`, `src.pipeline_batch`, `src.main_script`, `src.ohlcv_batch`, `src.walk_forward_analyzer`, `src.parameter_simulation_gpu`
    - 심볼 존재: `main`/`run_*`/`find_optimal_parameters` 등

### 4-3. C안: `src/legacy/` shim 도입(리스크 최소형, 기술부채 증가)
- 접근:
  - 현행 구현을 `src/legacy/`로 이동하고, 상단 `src/*.py`는 `from .legacy.<module> import ...` 형태로 re-export
  - 새 구조는 `src/core/`, `src/runtime/`, `src/adapters/` 같은 형태로 별도 구축
- 장점:
  - 외부/내부 import 경로 호환성이 가장 안전하게 유지됨
- 리스크:
  - legacy 경계가 오래 남을 가능성이 큼(정리 비용이 뒤로 밀림)
  - 동일 심볼(예: `Position`)의 “단일 SSOT”가 흐려질 수 있음
- 테스트/검증:
  - B안과 동일 + “중복 심볼 금지” 체크(예: `__all__` 고정)

---

## 5. 최종 결정된 수정 방안 (AI 가 자동 진행하면 안되고 **무조건**/**MUST** 사람에게 선택/결정을 맡겨야 한다)
결정: **B안(Wrapper-first 점진 리팩터링)** 로 진행한다.

- 요약:
  - 기존 `src/*.py` 엔트리포인트/파일명은 유지하고(thin wrapper), 내부 구현을 신규 서브패키지로 단계적으로 이동한다.
  - PR 당 1~2개 대형 모듈만 대상으로 하여 리스크/충돌을 최소화한다.
  - no DB/no GPU 환경에서도 import-safe, 테스트 skip-safe 규칙을 유지한다.

### 5-1. 결정 이유
- 리팩터링 범위가 크므로 한 번에 이동(A안)하면 import 경로/의존성 회귀 리스크가 큼
- 엔트리포인트 호환성(BOOTSTRAP 포함)과 import-side-effects 규칙(#60/#68)을 PR 단위로 안전하게 고정 가능
- 노트북 환경(no DB/no GPU)에서도 “개발 가능한 상태”를 유지하며 병렬로 정리 가능

### 5-2. PR 단계(초안)
- PR-0(안전망): 신규 테스트 + 패키지 스켈레톤(빈 `__init__.py`)만 추가
- PR-1: `src/pipeline_batch.py` 구현을 `src/pipeline/batch.py`로 이동(엔트리포인트 wrapper 유지)
- PR-2: `src/ticker_universe_batch.py` 구현을 `src/pipeline/ticker_universe_batch.py`로 이동(엔트리포인트 wrapper 유지)
- PR-3: `src/ohlcv_batch.py` 구현을 `src/pipeline/ohlcv_batch.py`로 이동(엔트리포인트 wrapper 유지)
- PR-4: `src/daily_stock_tier_batch.py` 구현을 `src/pipeline/daily_stock_tier_batch.py`로 이동(필요 시 엔트리포인트 wrapper)
- PR-5: `src/walk_forward_analyzer.py`의 분석/robust 로직을 `src/analysis/*`로 이동(무GPU import 보장 유지)
- PR-6: `src/backtest_strategy_gpu.py`를 `src/backtest/gpu/*`로 분해(가능하면 GPU deps lazy), 상단 wrapper 유지
- PR-7: CPU 백테스터(core) 계층을 `src/backtest/cpu/*`로 이동(기능 변경 금지)

---

## 6. 코드 수정 요약
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
(최종 방안이 결정되면 checkbox 로 checklist 를 먼저 작성한 후 코드 수정이 진행되면 경과를 기록한다)
(작성 시: 파일경로:라인 + 무엇을 + 어떻게. 코드 전체 복사 금지)
- <어떻게 고쳤다 한 줄>
### 6-1. <수정한 기능1>
- [x] `tests/test_issue69_entrypoint_compat.py`: 엔트리포인트 import/호환성 가드 테스트 추가(PR-0)
- [x] `src/<new_pkg>/__init__.py`: 신규 서브패키지 스켈레톤 추가(PR-0)
- [x] `python -m unittest tests.test_issue60_import_side_effects tests.test_issue61_import_style_standardization tests.test_issue68_wfo_import_side_effects tests.test_issue69_entrypoint_compat`: 통과 확인(PR-0)

### 6-2. PR-1: `pipeline_batch` 구현을 `src/pipeline`로 이동
- [x] `src/pipeline/batch.py`: 기존 `src/pipeline_batch.py` 구현 이동(BOOTSTRAP 제거 + 상위 모듈 상대 import로 변경)
- [x] `src/pipeline_batch.py`: entrypoint 호환 wrapper로 재생성(`python -m src.pipeline_batch` 유지)
- [x] `tests/test_pipeline_batch.py`: patch 타겟을 `src.pipeline.batch.*`로 갱신
- [x] 테스트 통과:
  - `python -m unittest tests.test_issue60_import_side_effects tests.test_issue61_import_style_standardization tests.test_issue68_wfo_import_side_effects tests.test_issue69_entrypoint_compat tests.test_pipeline_batch -v`

### 6-3. PR-2: `ticker_universe_batch` 구현을 `src/pipeline`로 이동
- [x] `src/pipeline/ticker_universe_batch.py`: 기존 `src/ticker_universe_batch.py` 구현 이동(BOOTSTRAP 제거 + 상위 모듈 상대 import로 변경)
- [x] `src/ticker_universe_batch.py`: entrypoint 호환 wrapper로 재생성(`python -m src.ticker_universe_batch` 유지)
- [x] `src/pipeline/batch.py`: 내부 import를 wrapper 대신 `.ticker_universe_batch` 구현으로 전환
- [x] `tests/test_ticker_universe_batch.py`: patch 타겟을 `src.pipeline.ticker_universe_batch.*`로 갱신
- [x] 테스트 통과:
  - `python -m unittest tests.test_issue60_import_side_effects tests.test_issue61_import_style_standardization tests.test_issue68_wfo_import_side_effects tests.test_issue69_entrypoint_compat tests.test_pipeline_batch tests.test_ticker_universe_batch -v`

### 6-4. PR-3: `ohlcv_batch` 구현을 `src/pipeline`로 이동
- [x] `src/pipeline/ohlcv_batch.py`: 기존 `src/ohlcv_batch.py` 구현 이동(BOOTSTRAP 제거 + 상위 모듈 상대 import로 변경)
- [x] `src/ohlcv_batch.py`: entrypoint 호환 wrapper로 재생성(`python -m src.ohlcv_batch` 유지 + 주요 심볼 re-export)
- [x] `src/ohlcv_collector.py`: `pykrx` import를 lazy로 전환(import-safe; 호출 시에만 의존)
- [x] `tests/test_ohlcv_batch.py`: patch 타겟을 `src.pipeline.ohlcv_batch.*`로 갱신
- [x] `tests/test_issue69_entrypoint_compat.py`: `src.ohlcv_batch` import + 심볼(`run_ohlcv_batch`) 가드 추가
- [x] 테스트 통과:
  - `python -m unittest tests.test_issue60_import_side_effects tests.test_issue61_import_style_standardization tests.test_issue68_wfo_import_side_effects tests.test_issue69_entrypoint_compat tests.test_pipeline_batch tests.test_ticker_universe_batch tests.test_ohlcv_batch -v`

---

## 7. 문제 해결에 참고
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
(문제 해결에 참고했던 issue 번호가 포함된 링크 or commit hash)
- issue: <이슈링크1>
- issue: <이슈링크2>
- ...
- commit: <commit hash1>
- commit: <commit hash2>
- ...
