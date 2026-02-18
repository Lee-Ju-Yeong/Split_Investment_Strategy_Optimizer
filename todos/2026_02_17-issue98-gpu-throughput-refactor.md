# perf(gpu): GPU Throughput 리팩토링 + 성능 저하 fallback 제거 (Issue #98)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/98`
- 작성일: 2026-02-17
- 최종 갱신: 2026-02-18
- 목적: GPU 처리량 병목과 fallback 유발 성능 저하를 제거하되, CPU=SSOT 원칙과 decision-level parity(`#56`)를 유지

## 0. 분리 원칙 (Issue #97/#56/#67 관계)
- #97: 레거시 자산 정리 거버넌스(삭제/아카이브/승인 게이트)
- #98: 성능 리팩토링(throughput, kernel launch, host-device sync, I/O)
- #56: parity hard gate(Release 기준 `mismatch=0`)
- #67: tier/hybrid 후보군 로직 및 host 병목 제거 Phase B/C
- 원칙:
  - 정책 정리(#97)와 성능 리팩토링(#98)을 같은 PR에 섞지 않는다.
  - 정합성 영향 변경은 CPU/GPU 동시 수정 + `#56 strict parity`를 필수로 통과한다.

## 1. 변경 분류 프레임 (필수)
| class | 정의 | 예시 | 필수 게이트 |
| --- | --- | --- | --- |
| `PC` (Parity-Coupled) | 결과 의사결정(후보군/정렬/체결/시점 앵커)에 영향 가능성이 있는 변경 | `signal_date`, 정렬 tie-break, top-k 슬롯 배정, 체결/rounding 규칙 | CPU/GPU 동시 수정 + `#56` strict `0 mismatch` |
| `PO` (Perf-Only) | 로직 결과를 바꾸지 않는 구현 최적화 | 캐시, host sync 제거, 로더 재사용, profiling 추가 | 단독 수정 가능 + strict parity 사후 검증 |

## 2. 성능 병목 전수 인벤토리 (2026-02-18)
| id | class | location | current_behavior | impact | recommendation | owner |
| --- | --- | --- | --- | --- | --- | --- |
| P-001 | PO | `src/optimization/gpu/kernel.py` | batch-size 계산 실패 시 `None` 반환 | kernel launch 과다, GPU util 저하 | 안전 기본치/heuristic + fail-fast 기준 명확화 | #98 |
| P-002 | PO | `src/optimization/gpu/parameter_simulation.py` | `get_optimal_batch_size` 실패 시 작은 batch fallback | 준-직렬 실행으로 throughput 급락 | 최소 batch-size 가드 + fallback 축소 | #98 |
| P-003 | PO | `src/pipeline/ohlcv_batch.py` | `--allow-legacy-fallback`로 universe 과대 확장 가능 | 텐서화/로딩 비용 급증 | 운영 경로 제거(비상 복구 분리) | #98/#97 |
| P-004 | PO | `src/data/collectors/financial_collector.py` | snapshot 부재 시 legacy universe fallback | downstream GPU 비용 상승 | snapshot 우선 강제 + 단계적 제거 | #98/#97 |
| P-005 | PC | `src/backtest/gpu/engine.py` | 일별 후보 선정에서 `tolist()/to_pylist()` + Python 루프/`.loc` 사용 | host-device sync 증가, GPU 벡터화 붕괴 | 후보군/랭킹 경로를 CuPy 텐서 기반으로 고정 | #98/#67/#56 |
| P-006 | PC | `src/backtest/gpu/data.py` | `_collect_candidate_rank_metrics_asof`가 cudf filter/concat/pylist 반복 | 일별 재할당/동기화 누적 | as-of metric tensor 사전 생성 후 gather | #98/#67/#56 |
| P-007 | PO | `src/backtest/gpu/logic.py` | `run_lengths.tolist()`로 device->host sync | 세그먼트 연산 오버헤드 누적 | CuPy-only segment/prefix 연산으로 치환 | #98 |
| P-008 | PC | `src/backtest/gpu/logic.py` | `_process_new_entry_signals_gpu`의 `for k in range(num_candidates)` 순차 루프 | kernel launch 과다 + 후보 수 증가 시 급격한 저하 | 후보 우선순위/자본차감의 결정론을 유지한 벡터화 단계 도입 | #98/#56 |
| P-009 | PC | `src/backtest/gpu/utils.py` | 후보 정렬이 Python `sorted` | CPU 정렬 hot path 편입 | `cp.lexsort` 기반 결정론 정렬 | #98/#56 |
| P-010 | PO | `src/backtest/cpu/strategy.py` | `get_stock_row_as_of` 티커별 반복 조회(N+1) | CPU 경로 I/O 바운드화 | 일자 단위 배치 조회/캐시 dict 도입 | #98/#57/#58 |
| P-011 | PO | `src/backtest/cpu/portfolio.py`, `src/backtest/cpu/execution.py` | `get_latest_price/get_ohlc_data_on_date/get_name_from_ticker` 반복 호출 | 거래/평가 루프 중복 조회 | 일자 스코프 캐시 주입 + 중복 집계 제거 | #98/#57/#58 |
| P-012 | PO | `src/data_handler.py` | `@lru_cache(maxsize=200)` + row copy/asof 반복 | universe 확장 시 캐시 미스 증가, DB thrashing | 캐시 정책 재설계(구간/배치 중심) | #98/#57/#58 |
| P-013 | PO | `src/optimization/gpu/data_loading.py` | 매 호출 `create_engine` + pandas round-trip | fold/재실행 시 로딩 지연 누적 | engine 재사용 + GPU 친화 로딩 경로 | #98/#58 |
| P-014 | PO | `tests/test_backtest_strategy_gpu.py` | 성능 회귀 가드 부재(`pass` placeholder) | 회귀 조기 탐지 불가 | 고정 데이터 benchmark/smoke budget 테스트 추가 | #98/#56 |

## 3. 비목표 (Out of Scope)
- 전략 규칙 변경(수익률 개선 목적 알고리즘 변경)
- 체결 규칙 변경(수수료/세금/호가 정책 변경)
- parity 기준 완화

## 4. 게이트
- Gate A (설계 동결):
  - 각 항목의 `class(PC/PO)`, 구현 범위, 롤백 경로 확정
  - PC 항목은 CPU/GPU 동시 변경 파일 목록을 사전 명시
- Gate B (검증):
  - 성능 지표 Before/After 동시 제출
  - `#56` strict parity(`decision-level`) `0 mismatch` 통과
- Gate C (반영):
  - 운영 배포 전 승인 + 롤백 절차 검증 완료

## 5. 측정/검증 기준 (반드시 Before/After)
- 동일 기간/모드/파라미터 고정
- 성능 지표:
  - wall-time
  - GPU kernel launch count
  - GPU util(가능 시)
  - peak memory
  - host-device transfer count(가능 시)
- 안정성 지표:
  - strict parity mismatch `0`
  - parity failure 발생 시 즉시 rollback 가능

## 6. 실행 체크리스트
- [ ] Gate A: `P-001~P-014`의 `PC/PO` 분류와 PR 매핑 확정
- [ ] PR-98A: `P-001~P-004` fallback 축소/정리
- [ ] PR-98B(PC): `P-005/P-006/P-008/P-009` CPU/GPU 동시 수정 + parity 통과
- [ ] PR-98C(PO): `P-007/P-010/P-011/P-012/P-013` 캐시/동기화/I/O 최적화
- [ ] PR-98D: `P-014` 성능 회귀 가드/벤치마크 테스트 반영
- [ ] 성능 측정 결과 문서화(before/after)
- [ ] `#56` strict parity 재검증(`0 mismatch`)
- [ ] Gate B 승인 완료
- [ ] Gate C 승인 완료

## 7. PR 분할 제안
- PR-98A: batch-size + universe fallback 정리
  - `src/optimization/gpu/kernel.py`
  - `src/optimization/gpu/parameter_simulation.py`
  - `src/pipeline/ohlcv_batch.py`
  - `src/data/collectors/financial_collector.py`
- PR-98B (Parity-Coupled Hot Path): 후보선정/정렬/신규진입 처리량 리팩토링
  - `src/backtest/gpu/engine.py`
  - `src/backtest/gpu/data.py`
  - `src/backtest/gpu/logic.py`
  - `src/backtest/gpu/utils.py`
  - 필요 시 대응 CPU 파일(`src/backtest/cpu/strategy.py`, `src/backtest/cpu/execution.py`)
- PR-98C (Perf-Only Data Path): CPU I/O/캐시/로딩 최적화
  - `src/backtest/cpu/strategy.py`
  - `src/backtest/cpu/portfolio.py`
  - `src/backtest/cpu/execution.py`
  - `src/data_handler.py`
  - `src/optimization/gpu/data_loading.py`
- PR-98D: 성능/정합 게이트 자동화
  - `tests/test_backtest_strategy_gpu.py`
  - parity/perf 실행 문서 및 리포트 템플릿

## 8. 실행 순서 (파일 단위 상세)
### 8-1. Baseline 고정 (코드 변경 전)
- [ ] 성능 baseline 수집:
  - `python -m src.parameter_simulation_gpu` (동일 config/기간)
  - wall-time, kernel launch, peak memory, GPU util 기록
- [ ] parity baseline 수집:
  - `python -m src.cpu_gpu_parity_topk --pipeline-stage gpu ... --parity-mode strict`
  - `python -m src.cpu_gpu_parity_topk --pipeline-stage cpu --snapshot-in ... --parity-mode strict`
  - 결과: decision-level `mismatch=0` 확인

### 8-2. PR-98A (PO) fallback 정리
- [ ] `src/optimization/gpu/kernel.py`
  - `get_optimal_batch_size` 실패 시 fallback 경계/최소치 명시
  - 불능 상태는 로그 + fail-fast 기준 추가
- [ ] `src/optimization/gpu/parameter_simulation.py`
  - batch-size fallback 축소 및 안전 기본치 적용
- [ ] `src/pipeline/ohlcv_batch.py`
  - `--allow-legacy-fallback` 운영 경로 분리/제한
- [ ] `src/data/collectors/financial_collector.py`
  - snapshot 미존재 fallback 축소
- [ ] 검증:
  - strict parity 재실행(`mismatch=0`)
  - throughput 지표 비교

### 8-3. PR-98B (PC) GPU 후보군 hot path
- [ ] `src/backtest/gpu/engine.py`
  - `tolist()/to_pylist()` 제거
  - 일별 weekly 스캔 경로 사전 인덱스화
- [ ] `src/backtest/gpu/data.py`
  - `_collect_candidate_rank_metrics_asof`의 일별 filter/concat 축소
  - tensor gather 중심으로 재구성
- [ ] `src/backtest/gpu/logic.py`
  - 신규진입 후보 처리에서 launch 과다 루프 축소
  - 결정론 순서/자본차감 semantics 보존
- [ ] `src/backtest/gpu/utils.py`
  - Python `sorted` -> `cp.lexsort` 전환
- [ ] 필요 시 동반 CPU 수정:
  - `src/backtest/cpu/strategy.py`
  - `src/backtest/cpu/execution.py`
- [ ] 검증:
  - `#56` strict parity(5거래일 + top-k gate) 필수 통과
  - mismatch 발생 시 즉시 롤백, 원인 태깅 후 재시도

### 8-4. PR-98C (PO) CPU I/O/캐시 + GPU 로딩 최적화
- [ ] `src/backtest/cpu/strategy.py`
  - 일자 단위 배치 조회 캐시
- [ ] `src/backtest/cpu/portfolio.py`
  - 평가/스냅샷 가격 조회 중복 제거
- [ ] `src/backtest/cpu/execution.py`
  - 주문 실행 경로 OHLC/이름 조회 캐시 주입
- [ ] `src/data_handler.py`
  - 캐시 정책(`maxsize`, 구간 단위 로딩) 재설계
- [ ] `src/optimization/gpu/data_loading.py`
  - `create_engine` 재사용
  - pandas round-trip 최소화
- [ ] 검증:
  - strict parity 재검증
  - CPU wall-time 개선 확인

### 8-5. PR-98D 게이트 자동화
- [ ] `tests/test_backtest_strategy_gpu.py`
  - `pass` placeholder 제거
  - 고정 입력 성능 스모크/예산 테스트 추가
- [ ] 문서/리포트 템플릿 정비
  - before/after 표준 포맷
  - parity 증적 링크 규격

## 9. 의사결정 규칙 (병합 차단)
- PC 변경에서 strict parity `mismatch>0`이면 병합 금지
- PO 변경에서 throughput 개선이 없고 리스크만 증가하면 병합 보류
- 측정 결과 누락 시 Gate B 미통과 처리
