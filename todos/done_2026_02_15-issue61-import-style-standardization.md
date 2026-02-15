# 임포트 스타일 통일(상대/절대/스크립트 실행) (Issue #61)
(현재 파일 이름은 YYYY_MM_DD-issue<이슈번호>-<issue_name_only_english>.md 로 지정)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/61`
- `src/` 내부 모듈 간 import 방식이 `from src...` / `from .` / 직접 import 형태로 섞여 있어,
  - 실행 방식(`python -m src.<module>` vs `python src/<module>.py`)에 따라 ImportError가 발생하고
  - 개발자가 실행 방법을 헷갈리기 쉬움
- 목표: **패키지 기준 상대 import로 통일**하고, **패키지 실행/단독 실행 모두 동작**하도록 정리

## 1. 배경(현재 이슈의 대략적인 이전 맥락)
- 이 프로젝트는 `src/`를 패키지로 두고 `python -m src.<module>` 형태로 오케스트레이터/스크립트를 실행하는 구성이 많음
- 하지만 일부 파일은 `from src...`(절대) / `from .`(상대) / 직접 import가 혼재되어 있어, 실행 컨텍스트에 따라 import가 깨질 수 있음
- 노트북/로컬 개발 환경에서는 “파일을 바로 실행”하는 경우가 많아(`python src/<file>.py`) 단독 실행 호환성이 특히 중요함

## 1. 현재 이슈 및 현상, 디버그 했던 내용
### 1-1. 현상: 실행 방식에 따라 import 실패
- `src/` 내부에서 상대 import(`from .foo import ...`)를 사용하는 파일은 `python src/<file>.py`로 직접 실행 시:
  - `ImportError: attempted relative import with no known parent package`가 발생 가능
- `from src.foo import ...` 형태의 절대 import를 사용하는 파일은 `python src/<file>.py`로 직접 실행 시:
  - `ModuleNotFoundError: No module named 'src'`가 발생 가능(실행 시 `sys.path[0]`가 `.../src`로 잡히기 때문)

### 1-2. 현상: 코드베이스 내 import 스타일 혼재
- `src/debug_gpu_single_run.py`, `src/indicator_calculator.py` 등 일부는 `from src...`를 사용
- 다수의 오케스트레이터/배치 스크립트는 `from .` 형태의 상대 import를 사용

### 1-3. 디버그: 현재 발견된 `from src...` 사용처
- `src/indicator_calculator.py:12` `from src.indicator_calculator_gpu ...`
- `src/debug_gpu_single_run.py:15` `from src.config_loader ...` 등
- 그 외 대부분의 내부 모듈 참조는 `from .` 상대 import 형태

---

## 2. 목표(해결하고자 하는 목표)

목표 핵심 정리
- `src/` 내부 모듈 간 import를 **패키지 기준 상대 import(`from .xxx import ...`)로 통일**
- 오케스트레이터/스크립트(직접 실행되는 파일)는 다음 2가지 실행 방식 모두에서 import가 깨지지 않게 보장
  - 패키지 실행: `python -m src.<module>`
  - 단독 실행: `python src/<module>.py`
- 영향을 받는 파일을 점검하고, 변경 범위를 최소화하되 일관성은 강제(혼재 방지)

### 2-1. (사람이 생각하기에) 우선적으로 참조할 파일 (이 파일들 이외에 자율적으로 더 찾아봐야 함)
- `src/main_backtest.py`, `src/walk_forward_analyzer.py` (대표 오케스트레이터)
- `src/main_script.py`, `src/pipeline_batch.py` (배치/파이프라인)
- `src/app.py` (Flask 엔트리)
- `src/parameter_simulation_gpu.py`, `src/debug_gpu_single_run.py` (GPU 엔트리)
- `src/indicator_calculator.py` (절대 import 사용 중)

---

(하위 부분은 사람이 작성하는게 아니라 AI 가 작성하는 부분)

# AI 결과

## 3. (AI가 생각하기에) 이슈의 원인으로 의심되는 부분들
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
- `src/` 내부 import 스타일이 혼재
  - `from src.xxx import ...` (패키지 루트가 `sys.path`에 있어야 동작)
  - `from .xxx import ...` (패키지 컨텍스트에서만 동작)
  - `from db_setup import ...` 처럼 “src 내부 모듈을 로컬 스크립트처럼” 임포트(실행 CWD/`sys.path`에 따라 동작이 달라짐)
- 결과적으로 실행 방식에 따라 동일 코드가 ImportError를 내거나, 개발자가 실행 방식을 헷갈리기 쉬움

## 4. (AI가 진행한) 디버그 과정
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
- `rg`로 `from src.` / `import src.` 사용처 탐색
- AST로 `src/` 내부에서 “내부 모듈을 비-상대 import”하는 케이스 탐색
- `if __name__ == "__main__"` 엔트리포인트 목록을 확보(직접 실행 지원 대상)

## 5. (AI가) 파악한 이슈의 원인
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
- Python에서 `python src/<file>.py`로 직접 실행하면 `__package__`가 비어 **상대 import가 동작하지 않는 구조적 제약**이 있음
- 동시에 `src/` 내부에서 `from src.xxx ...` 또는 `from db_setup ...` 같은 스타일이 섞여 있어, 실행 컨텍스트에 따라 import 성공 여부가 달라짐

---

## 6. 생각한 수정 방안들
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
(수정 방안 최소 3가지)
(작성 시: 파일경로:위치 + 무엇을 + 어떻게 + 왜. 코드 전체 복사 금지)
### 6-1. (권장) `src/` 내부는 상대 import로 통일 + 엔트리포인트에 “단독 실행 bootstrap” 추가
- 대상: `if __name__ == "__main__":`가 있는 스크립트성 모듈들
- 각 엔트리포인트 최상단에 아래 bootstrap을 넣어 `python src/<file>.py`에서도 패키지 컨텍스트를 갖도록 처리
  - `sys.path`에 repo root를 추가하고 `__package__ = "src"` 설정
- 내부 모듈 참조는 `from .xxx import ...`로 통일(`from src.xxx ...` 제거)
- 장점: 이슈 완료 조건(패키지 실행/단독 실행 모두 동작)을 가장 정확히 만족
- 단점: 여러 엔트리포인트 파일에 반복 코드가 들어감(하지만 변경 폭은 예측 가능)

### 6-2. `python -m src.<module>` 실행만 “공식 지원”으로 고정(문서화/가드 최소)
- `src/` 내부 import를 상대 import로 통일하되, 단독 실행(`python src/<file>.py`)은 지원하지 않음
- README/AGENTS/Quick Commands에 실행 규칙을 명확히 적고, 잘못 실행 시 친절한 에러 메시지를 출력하도록만 보강
- 장점: 코드 변경 범위 최소
- 단점: 이슈의 완료 조건(단독 실행 포함)을 충족하기 어려움

### 6-3. `scripts/` (repo root) 래퍼 엔트리포인트를 추가해 단독 실행 UX 제공
- `src/`는 “패키지 전용(상대 import)”으로 정리하고, 직접 실행은 `scripts/*.py`에서만 수행
- `scripts/*.py`는 repo root에서 실행되므로 `import src...`가 안정적
- 장점: `src/` 내부에서 bootstrap 반복을 피할 수 있음
- 단점: 실행 경로가 분산되고, 기존에 `src/*.py` 직접 실행하던 습관과 충돌 가능

---

## 7. 최종 결정된 수정 방안 (AI 가 자동 진행하면 안되고 **무조건**/**MUST** 사람에게 선택/결정을 맡겨야 한다)
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
(작성 시: 파일경로:위치 + 무엇을 + 어떻게 + 왜. 코드 전체 복사 금지)
- 선택(사용자 결정): **6-1. `src/` 내부 상대 import 통일 + 엔트리포인트 bootstrap으로 단독 실행 지원**
  - `src/` 내부 모듈 간 import는 `from .xxx import ...`로 통일 (`from src.xxx ...`, `from db_setup ...` 제거)
  - `if __name__ == "__main__":` 엔트리포인트 파일에는, 상대 import보다 위에서 다음 bootstrap을 실행:
    - `python -m src.<module>`: 기존대로 동작
    - `python src/<module>.py`: repo root를 `sys.path`에 추가하고 `__package__='src'`로 설정해 상대 import가 동작하도록 보장
  - 회귀 방지: `src/` 내부에 `from src.` / `import src.` / `from db_setup` 같은 패턴이 다시 들어오지 않도록 정적 테스트(또는 CI grep)를 추가

### 7-1. 최종 결정 이유
- 요구사항이 “패키지 실행 + 단독 실행”을 동시에 요구하므로, bootstrap shim 없이 상대 import를 직접 실행에서 동작시키는 것은 불가능
- bootstrap + 상대 import 통일은 변경 범위가 예측 가능하고, import 오류를 근본적으로 제거
- 회귀 방지(정적 체크)를 같이 두면 혼재가 재발하는 문제를 막을 수 있음

---

## 8. 코드 수정 요약
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
(최종 방안이 결정되면 checkbox 로 checklist 를 먼저 작성한 후 코드 수정이 진행되면 경과를 기록한다)
(작성 시: 파일경로:라인 + 무엇을 + 어떻게. 코드 전체 복사 금지)
- `src/` 내부 import를 상대 import로 통일하고, 엔트리포인트에 bootstrap을 추가해 단독 실행에서도 ImportError가 발생하지 않도록 수정.

### 8-1. `src/` 내부 import 통일
- [x] `src/debug_gpu_single_run.py` `from src...` -> `from . ...`로 변경
- [x] `src/indicator_calculator.py` `from src...` -> `from . ...`로 변경
- [x] `src/data_pipeline.py`, `src/ticker_collector.py`, `src/stock_data_collector.py`, `src/etf_data_collector.py`의 `from db_setup ...` 등 로컬 import -> 상대 import로 변경

### 8-2. 엔트리포인트 단독 실행 bootstrap 추가
- [x] `src/*.py` 중 `if __name__ == "__main__":` 엔트리포인트에 bootstrap shim 추가(상대 import보다 먼저 실행)
  - `src/app.py`, `src/company_info_manager.py`, `src/corporate_event_collector.py`, `src/data_pipeline.py`
  - `src/debug_gpu_single_run.py`, `src/filtered_stock_loader.py`, `src/main_backtest.py`, `src/main_script.py`
  - `src/ohlcv_adjusted_updater.py`, `src/ohlcv_batch.py`, `src/parameter_simulation_gpu.py`, `src/performance_analyzer.py`
  - `src/pipeline_batch.py`, `src/ticker_universe_batch.py`, `src/walk_forward_analyzer.py`

### 8-3. 회귀 방지 테스트/체크 추가
- [x] `tests/test_issue61_import_style_standardization.py` 추가: “src 내부에서 `from src.`/`import src.`/로컬 import 금지” 정적 테스트

---

## 9. 문제 해결에 참고
(사람이 작성하는 시점엔 TODO 로만 남겨놓는다)
(문제 해결에 참고했던 issue 번호가 포함된 링크 or commit hash)
- issue: https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/61
- PR: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/pull/78`
- merged commit (main): `bed91ea`
