# 스크립트 import 부작용 제거 (parameter_simulation_gpu.py) (Issue #60)
(현재 파일 이름은 YYYY_MM_DD-issue<이슈번호>-<issue_name_only_english>.md 로 지정)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/60`
- 문제: `src/parameter_simulation_gpu.py`가 **import 시점에** 설정 로딩 + CuPy 파라미터 그리드 생성 + 로그 출력 등을 수행함
- 영향: WFO 오케스트레이터(`src/walk_forward_analyzer.py`) 등에서 안전하게 import 하기 어렵고, 노트북/무DB 개발에서 불필요한 부작용(시간/메모리/출력)을 유발

## 1. 배경(현재 이슈의 대략적인 이전 맥락)
- 프로젝트 아키텍쳐나 구성 요소를 짐작할 수 있는 배경
- `src/walk_forward_analyzer.py`는 `from .parameter_simulation_gpu import find_optimal_parameters` 형태로 워커 함수를 import하여 WFO 파이프라인에서 사용함
- GPU 최적화 스크립트는 단독 실행(`python -m src.parameter_simulation_gpu`)도 지원해야 함
- 원칙: import는 “정의만” 수행하고, 실행은 `if __name__ == "__main__"` 또는 명시적 함수 호출에서만 수행

## 1. 현재 이슈 및 현상, 디버그 했던 내용
### 1-1. 현상: import 시점에 실행이 발생
- 표면적으로만 나타나는 보이는 대로 서술한 내용
- `import src.parameter_simulation_gpu`만 해도 아래가 실행됨
- `load_config()` 호출(파일 IO)
- `cp.meshgrid` 기반 파라미터 조합 생성(GPU 메모리/시간 사용 가능)
- `print("✅ Dynamically generated ...")` 출력 발생
```
python -c "import src.parameter_simulation_gpu"
```

---

## 2. 목표(해결하고자 하는 목표)

목표 핵심 정리:
- import 시 실행되는 사이드 이펙트를 제거한다(설정 로딩/파라미터 생성/출력/실행 금지)
- 단독 실행(`python -m src.parameter_simulation_gpu`)은 기존과 동일하게 동작한다
- 외부 모듈(WFO 등)에서 안전하게 import 가능해야 한다

### 2-1. (사람이 생각하기에) 우선적으로 참조할 파일 (이 파일들 이외에 자율적으로 더 찾아봐야 함)
- `src/parameter_simulation_gpu.py`
- `src/walk_forward_analyzer.py`
- `src/config_loader.py`
- `config/config.example.yaml` (parameter_space, backtest_settings 등 구조 확인)

---

(하위 부분은 사람이 작성하는게 아니라 AI 가 작성하는 부분)

# AI 결과

## 3. (AI가 생각하기에) 이슈의 원인으로 의심되는 부분들
- `src/parameter_simulation_gpu.py` 모듈 최상단에서 실행되는 전역 초기화 블록
  - `config = load_config()` (파일 IO)
  - `param_space_config` 기반 `cp.meshgrid`/`cp.vstack`로 `param_combinations` 생성 (GPU 컨텍스트/메모리 사용)
  - import 시점 `print(...)` 실행
- `src/walk_forward_analyzer.py`가 해당 모듈을 import 하기만 해도 위 전역 코드가 실행되는 구조

## 4. (AI가 진행한) 디버그 과정
- `src/walk_forward_analyzer.py`에서 `find_optimal_parameters`를 import 하는지 확인
- `src/parameter_simulation_gpu.py`에서 import 시점 실행되는 코드(전역 변수 초기화/print)를 확인
- `config/config.example.yaml`에서 `parameter_space`가 실제로 `cp.meshgrid`로 조합 생성되는 형태인지 확인

## 5. (AI가) 파악한 이슈의 원인
원인: “라이브러리(import) 경로”와 “실행(standalone) 경로”가 분리되지 않아, 실행에 필요한 초기화가 모듈 import 시점에 수행되고 있음.

---

## 6. 생각한 수정 방안들
(수정 방안 최소 3가지)
(작성 시: 파일경로:위치 + 무엇을 + 어떻게 + 왜. 코드 전체 복사 금지)
### 6-1. A안(최소 변경): 전역 초기화를 `find_optimal_parameters()` 내부로 이동
- `src/parameter_simulation_gpu.py`: 전역 `config/db_connection_str/param_combinations/num_combinations` 제거
- `find_optimal_parameters()` 시작 시점에 `load_config()` + DB conn string 생성 + param grid 생성
- 장점: 구현이 단순, 전역 상태 없음
- 단점: WFO에서 fold마다 `find_optimal_parameters()`를 여러 번 호출하면, 파라미터 조합을 매번 재생성(시간/VRAM 부담)

### 6-2. B안(권장): Lazy init + 1회 캐시(첫 호출 때만 초기화)
- `src/parameter_simulation_gpu.py`: `_init_context()` 같은 private 함수 도입
- `_context`(dict/dataclass)에 `config`, `execution_params`, `strategy_params`, `db_connection_str`, `param_combinations`, `num_combinations`을 1회만 적재
- `find_optimal_parameters()`는 매 호출마다 `_init_context()`로 컨텍스트를 얻고 나머지 로직 수행
- 장점: import 시 사이드 이펙트 0 + 반복 호출 성능 유지
- 단점: 전역 캐시(뮤터블) 도입 → 테스트/디버그 시 초기화 타이밍을 의식해야 함

### 6-3. C안(구조 개선): “라이브러리 모듈”과 “CLI 엔트리”를 분리
- 신규: `src/gpu_optimizer.py` 같은 라이브러리 모듈로 로직 이동(순수 함수/클래스)
- `src/parameter_simulation_gpu.py`는 CLI wrapper로만 유지(`main()` + argparse 등)
- 장점: import 안정성 + 구조 명확(향후 `#69` 패키지 구조 재편에도 유리)
- 단점: 파일 이동/임포트 경로 변경 등 변경 폭이 커서, 이번 이슈(#60) 범위를 넘어갈 수 있음

---

## 7. 최종 결정된 수정 방안 (AI 가 자동 진행하면 안되고 **무조건**/**MUST** 사람에게 선택/결정을 맡겨야 한다)
(작성 시: 파일경로:위치 + 무엇을 + 어떻게 + 왜. 코드 전체 복사 금지)
- 결정: **C안(라이브러리 모듈/CLI 엔트리 분리)**로 진행
- `src/parameter_simulation_gpu.py`: “진입점/호환 레이어”로 축소하고, import 시 전역 실행이 없도록 정리
- 신규 `src/parameter_simulation_gpu_lib.py`(이름 확정 필요): 기존 최적화 로직을 라이브러리로 이동
- 외부 사용처(WFO): 기존대로 `from .parameter_simulation_gpu import find_optimal_parameters`가 동작하도록 wrapper/re-export 유지
- 단독 실행: `python -m src.parameter_simulation_gpu`가 내부적으로 lib의 `main()`을 호출해 기존 동작 유지

### 7-1. 최종 결정 이유
- 구조적으로 “import 경로”와 “실행 경로”를 분리해 재발을 막는다(단순 위치 이동보다 근본적)
- 향후 `#69(src 패키지 구조 재편)`과도 자연스럽게 이어지는 방향이다
- WFO/단독 실행 모두에 대해 API/엔트리포인트 호환을 유지하면서 부작용을 제거할 수 있다

### 7-2. 현재 상태(2026-02-15, 노트북 환경 메모)
- 현재 작업 환경에서 `cupy`가 설치되어 있지 않아, GPU 경로의 E2E 실행 검증은 제한적임
- 대신 아래는 노트북 환경에서도 확실히 검증 가능
  - import 시 `load_config()`/파라미터 그리드 생성/print 등 사이드 이펙트가 “0”인지
  - `src.walk_forward_analyzer`가 GPU 환경 없이도 import 가능한지(분석/후처리 로직 개발 목적)

---

## 8. 코드 수정 요약
(최종 방안이 결정되면 checkbox 로 checklist 를 먼저 작성한 후 코드 수정이 진행되면 경과를 기록한다)
(작성 시: 파일경로:라인 + 무엇을 + 어떻게. 코드 전체 복사 금지)
- import-safe(무부작용) 구조로 전환: “라이브러리 모듈” + “얇은 wrapper/엔트리” 분리 + lazy import
### 8-1. 라이브러리/엔트리 분리(Approach C)
- [x] `src/parameter_simulation_gpu.py` 를 thin wrapper로 교체
  - `find_optimal_parameters`, `main`만 re-export하고 `__main__`에서만 실행
- [x] `src/parameter_simulation_gpu_lib.py` 신규 추가
  - 기존 로직을 lib로 이동하고, import 시점에 `load_config()`/param grid 생성/print가 발생하지 않도록 정리
  - `cupy/cudf/backtest_strategy_gpu`는 실행 시점에만 import(lazy)하도록 변경
  - 컨텍스트 초기화(`_get_context`)는 첫 호출 시점에만 수행되도록 cache 적용

### 8-2. 회귀 방지 테스트
- [x] `tests/test_issue60_import_side_effects.py` 추가
  - `import src.parameter_simulation_gpu`가 stdout 출력 없이 성공하는지 검증
  - (노트북 환경에서 `config/config.yaml`/GPU deps가 없어도 import 가능해야 함)

---

## 9. 문제 해결에 참고
(문제 해결에 참고했던 issue 번호가 포함된 링크 or commit hash)
- issue: https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/60
- merged commit (main): `eaf5bfe`
