# 설정 소스 표준화 및 하드코딩 경로/플래그 제거 (Issue #53)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/53`
- 작성일: 2026-02-15

## 1. 배경
- `config/config.yaml`(gitignored) 기반으로 백테스트/최적화 설정은 이미 표준화되어 있음(`src/config_loader.py`).
- 하지만 일부 스크립트는 여전히 `config.ini`(configparser) 또는 환경 의존 하드코딩(Windows 경로, 실행 플래그)을 사용하고 있어:
  - 노트북/CPU-only 환경에서 재현성이 떨어지고
  - 실행/테스트/문서가 서로 다른 “설정 소스”를 가리키는 문제가 발생.

## 2. 현재 이슈 및 현상
### 2-1. `src/main_script.py`의 하드코딩
- `CONDITION_SEARCH_FILES_FOLDER`가 특정 Windows 절대경로(`E:/...`)로 하드코딩
- 파이프라인 실행 플래그(`USE_GPU`, `COLLECT_OHLCV_DATA` 등)가 코드 상수로 존재

### 2-2. `config.ini` 의존 경로 존재
- `src/db_setup.py`: DB 연결 정보를 `config.ini`에서 읽음
- `src/filtered_stock_loader.py`: DB 엔진 생성 정보를 `config.ini`에서 읽음
- `tests/test_integration.py`, `tests/README.md`, `README.md` 일부 문서가 `config.ini`를 전제로 안내

### 2-3. (노트북 환경) no DB/no GPU에서 import/테스트 불편
- DB 드라이버가 없는 환경에서 `pymysql` 등 모듈 import가 선행되면, “실행하지 않는 코드”까지 import 실패로 이어질 수 있음.

## 3. 목표(요구사항)
- `config/config.yaml`을 설정의 단일 표준으로 승격
  - env override(`MAGICSPLIT_CONFIG_PATH`) 지원
- `src/main_script.py`의 경로/플래그를 `config.yaml`에서 읽도록 전환(하드코딩 제거)
- `configparser`로 `config.ini`를 직접 읽는 코드 경로 제거/축소
- README/AGENTS(= `llm.md`) 및 테스트 문서/게이트를 최신 설정 소스로 정리

### 3-1. 우선 참조 파일
- `src/config_loader.py`
- `src/main_script.py`
- `src/db_setup.py`
- `src/filtered_stock_loader.py`
- `config/config.example.yaml`
- `README.md`, `llm.md`
- `tests/test_integration.py`, `tests/README.md`

---

# AI 결과

## 4. 원인(정리)
- `config.yaml` 도입 이후 레거시 스크립트 일부가 `config.ini`/하드코딩을 유지하면서 설정 소스가 분기됨.
- “실행 플래그”가 config 대신 코드에 남아 있어, 실행 재현성/문서 일치성 문제가 지속됨.

## 5. 생각한 수정 방안들
### 5-1. A안: `config.yaml` 단일화 + `data_pipeline` 섹션 추가 (권장)
- `config/config.example.yaml`에 `data_pipeline.paths`/`data_pipeline.flags` 추가
- `src/main_script.py`는 `load_config()`로 읽어 경로/플래그를 결정
- `src/db_setup.py`, `src/filtered_stock_loader.py`는 `config.ini` 대신 `config.yaml`의 `database`를 사용
- 장점: 설정 SSOT 확립, 문서/테스트 정합성 상승
- 단점: 사용자 로컬 `config.yaml`에 섹션 추가 필요

### 5-2. B안: `main_script.py`를 argparse 기반 CLI로 전환(플래그/경로를 인자로)
- 장점: 실행 단위별로 인자 오버라이드 쉬움
- 단점: 변경량 증가, 문서/운영 커맨드 갱신 범위 확대

### 5-3. C안: env var 중심(경로/플래그 모두 env) + yaml fallback
- 장점: 컨테이너/배치 환경에서 유연
- 단점: 설정이 분산되어 다시 SSOT가 흔들릴 가능성 큼

## 6. 교차 검증(Gemini 리뷰) 메모
- 큰 방향은 `config.yaml` 단일화가 맞고, 가능한 한 “명시적 의존성 전달(Dependency Injection)”이 테스트/가독성에 유리.
- 다만 `get_db_connection()` 같은 공용 util은 시그니처 변경이 넓게 번질 수 있어, optional arg로 점진 전환이 현실적.

## 7. 최종 결정된 수정 방안
- A안 채택: `config.yaml` 단일화 + `data_pipeline` 섹션 도입
- 구현 방식은 “Hybrid”:
  - `get_db_connection(db_config=None)`, `get_db_engine(db_config=None)`처럼 optional arg로 유지해 기존 호출부 변경을 최소화
  - 내부적으로는 `config.yaml`의 `database` 섹션을 사용

## 8. 코드 수정 요약
- [x] `src/config_loader.py`: `MAGICSPLIT_CONFIG_PATH` env override + project-root 기반 path resolve 도입
- [x] `config/config.example.yaml`: `data_pipeline.paths|flags` 추가(경로/플래그 SSOT)
- [x] `src/main_script.py`: Windows 절대경로/플래그 하드코딩 제거, `config.yaml` 기반으로 실행
- [x] `src/db_setup.py`: `config.ini` 제거, `database` 섹션 사용. `pymysql` lazy import로 no-DB import 안전성 개선
- [x] `src/filtered_stock_loader.py`: `config.ini` 제거, `database` 섹션 사용
- [x] `src/indicator_calculator.py`: `use_gpu=True`일 때만 GPU 모듈 lazy import(노트북/CPU-only에서 import 안전성 개선)
- [x] `tests/test_integration.py`: `config/config.yaml` 기반으로 전환(+ `MAGICSPLIT_CONFIG_PATH` 허용)
- [x] `README.md`, `tests/README.md`, `llm.md`: 설정 소스/키 정리 업데이트

## 9. 참고
- issue: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/53`
