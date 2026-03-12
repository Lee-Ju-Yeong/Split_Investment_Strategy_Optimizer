# Issue #97: Legacy Audit And Strict-Only Governance

> Type: `implementation`
> Status: `done`
> Priority: `P1`
> Last updated: 2026-03-07
> Related issues: `#97`, `#93`, `#98`
> Gate status: `Gate A/B/C approved, step 2 synthetic approved, step 3 done`

## 1. One-Page Summary
- What: 레거시, fallback, 호환 경로를 무작정 지우지 않고 승인 게이트를 거쳐 정리하는 문서입니다.
- Why: 실행 경로를 단순화해야 parity와 문서 가독성도 같이 좋아지지만, 잘못 지우면 운영 계약이 깨집니다.
- Current status: runtime strict-only fail-fast(`tier_hysteresis_mode`, `candidate_source_mode`, `use_weekly_alpha_gate`)와 retained wrapper drift 정리까지 반영됐습니다. `src.pipeline_batch`와 `src.ohlcv_batch`는 canonical module thin wrapper로 재정렬했고, Gate A/B/C 승인 기준도 충족했습니다. step 2 historical synthetic sample pack도 helper 기준으로 승인됐습니다. step 3에서는 tier-only runtime에서 더 이상 쓰지 않는 `weekly_filtered_gpu` dead plumbing, compat alias, Financial/OHLCV legacy universe fallback surface를 제거했습니다.
- Next action: `#98` throughput 증적 수집으로 진행합니다. Gate B 승인 예외(`src.parameter_simulation_gpu_lib` facade, `src.main_script` legacy/general orchestrator, historical/archive docs)는 유지 이유가 확정된 상태입니다.

## 2. Fixed Rules
- 근거 없는 즉시 삭제는 금지합니다.
- `entrypoint wrapper`와 핵심 실행 경로는 `#93` 정책과 충돌 없이 다뤄야 합니다.
- 성능 튜닝은 `#98`, 정책 삭제/축소는 `#97`이 담당합니다.

## 3. Current Plan
- [x] 레거시 인벤토리 1차 작성
- [x] 저위험 archive 이동
- [x] fallback 축소 1차 반영
- [x] strict-only step 1: `strict_hysteresis_v1` + `candidate_source_mode='tier'` + `use_weekly_alpha_gate=False`만 허용
- [x] Gate A 승인
- [x] Gate B 승인
- [x] Gate C 승인 기준 고정
- [x] strict-only step 2 synthetic 집계 규칙 고정
- [x] Gate C 승인
- [x] strict-only step 2: synthetic sample pack(서로 다른 window 10개 + matched parity sample 1개 이상)
- [x] strict-only step 3: active non-strict 코드/테스트/문서 삭제

## 4. Key Evidence
- 이미 끝난 것:
  - archive 이동 3건
  - `allow_legacy_fallback`과 tier fallback 일부 축소
  - `tier_hysteresis_mode=legacy`는 strict validator로 거부
  - CPU strategy/shared helper, tests/docs/config는 `candidate_source_mode='tier'`, `use_weekly_alpha_gate=False` strict-only spec으로 정렬
- 이번 라운드에서 닫은 것:
  - candidate runtime shim 제거
  - retained wrapper drift 재정렬(`src.pipeline_batch`, `src.ohlcv_batch`)
  - real entrypoint regression test 보강(`--help`, wrapper->canonical symbol identity)
- 이번 라운드에서 닫은 것:
  - Gate C 승인 기록 저장
  - step 2 synthetic sample pack 결과 JSON 저장
  - synthetic sample pack window와 직접 매칭되는 parity artifact 1개 확보
- 이번 라운드에서 닫은 것:
  - tier-only runtime에서 `weekly_filtered_gpu` dead plumbing 제거
  - `resolve_tier_hysteresis_mode` compat alias 제거
  - `config/config.example.yaml` strict-only surface 정리
- 이번 라운드에서 닫은 것:
  - Financial batch legacy universe fallback 제거
  - `run_financial_batch` / `run_pipeline_batch`에서 `allow_legacy_fallback` surface 제거
  - `--allow-financial-legacy-fallback` CLI 제거 및 관련 summary/test/doc 정렬
- 이번 라운드에서 닫은 것:
  - OHLCV batch legacy universe fallback 제거
  - `run_ohlcv_batch` / `src.ohlcv_batch`에서 `allow_legacy_fallback` surface 제거
  - OHLCV summary/test/runbook을 `history|explicit` strict-only surface로 정렬
- 이번 라운드에서 닫은 것:
  - `src.optimization.gpu.data_loading`의 dead weekly preload helper 제거
  - `src.parameter_simulation_gpu_lib` compat facade에서 non-strict weekly GPU helper export 제거
- 이번 라운드에서 닫은 것:
  - `DataHandler` non-PIT candidate helper(`get_candidates_with_tier_fallback`, `get_candidates_with_tier_fallback_pit`, `get_filtered_stock_codes_with_tier`) 제거
  - CPU strategy가 `get_candidates_with_tier_fallback_pit_gated(...)`만 사용하도록 정렬
  - `tests/test_issue67_tier_universe.py`, `tests/test_data_handler_tier.py`, `tests/test_cpu_candidate_priority.py`, `tests/README.md` strict-only 기준 정렬
- 최종 readonly review:
  - Banach: `no findings`
  - Heisenberg: DataHandler/Strategy non-PIT helper와 관련 테스트/README를 blocker로 지적했고, 해당 범위를 제거 후 회귀 통과

## 5. Reading Guide
- 이 문서는 “무엇을 지워도 되는가”보다 “무엇은 왜 아직 남겨야 하는가”를 먼저 설명합니다.
- 세부 인벤토리는 아래 표를 필요할 때만 읽으세요.

## 6. Detailed History And Working Log
### 0. 진행 현황 (2026-02-17)
- [x] 1차 인벤토리 작성(핵심 실행경로/호환 wrapper/fallback/유휴 스크립트)
- [x] 저위험 archive 이동 반영
  - `reproduce_issue.py` -> `archive/legacy_tools/reproduce_issue.py`
  - `reproduce_issue_v2.py` -> `archive/legacy_tools/reproduce_issue_v2.py`
  - `src/tier_parity_monitor.py` -> `archive/legacy_tools/tier_parity_monitor.py`
- [x] fallback 축소 1차 반영
  - `L-003`: strict 경로에서 Tier2 fallback 비사용(`allow_tier2_fallback=False`)
  - `L-005`: `TickerUniverseHistory` 미구축 시 strict fail-fast 고정
- [x] Gate A: 인벤토리 확정 승인
- [x] Gate B: 제거/축소 대상 항목별 승인
- [x] Gate C: 배포 전 최종 승인

## 1. 이번 이슈의 핵심 원칙 (현업 기준)
- 원칙 1: `종목선정`과 `매매 우선순위/체결`은 전략 신뢰성 핵심 경로이므로 임의 삭제 금지
- 원칙 2: `entrypoint 호환 wrapper`는 #93 정책과 충돌 없이 유지(성급한 삭제 금지)
- 원칙 3: 실행경로 불명 항목은 `즉시 삭제`보다 `archive 격리`를 기본값으로 사용
- 원칙 4: 삭제는 “근거 + 롤백 경로 + 승인” 3요건 충족 시에만 수행
- 원칙 5: GPU 처리량 최적화(throughput) 작업은 #98로 분리해 관리

## 2. 삭제 vs 정리/이동 판단 기준
| 분류 | 조건 | 조치 |
| --- | --- | --- |
| 유지(`keep`) | 핵심 실행경로, 정합성/패리티 영향, 운영 커맨드 계약 포함 | 코드 유지, 테스트 보강 |
| 축소(`shrink`) | 기능은 필요하나 legacy fallback/분기 과다 | fallback 축소 PR 별도 진행 |
| 아카이브(`archive`) | 기본 실행경로 미사용, 수동 재현/운영보조 성격 | `archive/` 또는 `tools/repro/`로 이동 |
| 제거(`delete`) | 참조 없음 + 호환계약 없음 + 대체경로 확정 | 승인 후 삭제 |

## 3. 1차 인벤토리 (핵심 실행경로 중심)
| id | location | category | current_behavior | risk | removal_cost | recommendation | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L-001 | `src/backtest/cpu/strategy.py` | 종목선정/우선순위 | runtime strict-only: `candidate_source_mode='tier'`, `use_weekly_alpha_gate=False`, `additional_buy_priority`로 신호 순서 결정 | 변경 시 CPU/GPU parity 및 종목선정 결과 변동 | High | 유지 | strict validator + entry ranking 분기 존재 |
| L-002 | `src/backtest/cpu/execution.py` | 체결/정산 | 한국시장 호가 반올림 + 매수/매도 체결 정산 SSOT | 체결가/수량/손익 왜곡 | High | 유지 | `_adjust_price_up`, `_execute_buy/_execute_sell` |
| L-003 | `src/data_handler.py` | 후보군 fallback | `get_candidates_with_tier_fallback` (tier1 우선, tier<=2 fallback) | 후보군 공백/의도치 않은 전략 드리프트 | Medium | 축소 | #67 정책 전환 단계 fallback 핵심 |
| L-004 | `src/backtest/gpu/engine.py` | 모드 호환 fallback | strict-only spec에서 legacy candidate mode와 `use_weekly_alpha_gate=True`를 즉시 거부 | 설정 오류를 조용히 숨길 위험 | Medium | 축소 완료 | `normalize_runtime_candidate_policy` fail-fast 반영 |
| L-005 | `src/pipeline/ohlcv_batch.py` | 데이터 fallback | `TickerUniverseHistory`가 비어 있으면 즉시 실패하고 legacy 유니버스는 더 이상 허용하지 않음 | 데이터 소스 일관성 저하 방지 | Medium | 축소 완료 | 2026-03-07 `allow_legacy_fallback` runtime/CLI 제거 |
| L-016 | `src/data/collectors/financial_collector.py`, `src/pipeline/batch.py` | Financial batch fallback | legacy universe fallback surface 제거 완료, snapshot/history 미구축 시 strict fail-fast | 제거 누락 시 strict-only 거버넌스 의미 약화 | Medium | 축소 완료 | 2026-03-07 step 3 cleanup 반영 |
| L-006 | `src/main_script.py` | legacy orchestrator | 주간 필터 CSV/DB 경로 기반 legacy/general 파이프라인 | 신규 배치 체계와 중복 운영 리스크 | Medium | 축소 | `legacy/general data pipeline` 명시 |
| L-007 | `src/pipeline_batch.py` | entrypoint wrapper | `src.pipeline.batch` thin wrapper | 운영 커맨드 계약 파손 | Medium | 유지 | 2026-03-07 thin-forward 재정렬 완료 |
| L-008 | `src/ticker_universe_batch.py` | entrypoint wrapper | `src.pipeline.ticker_universe_batch` thin wrapper | 운영 스크립트 호환 파손 | Medium | 유지 | #93 wrapper policy keep 목록 |
| L-009 | `src/ohlcv_batch.py` | entrypoint wrapper | `src.pipeline.ohlcv_batch` thin wrapper | 운영 스크립트 호환 파손 | Medium | 유지 | 2026-03-07 thin-forward 재정렬 완료 |
| L-010 | `src/walk_forward_analyzer.py` | entrypoint wrapper | `src.analysis.walk_forward_analyzer` thin wrapper | 운영/문서 커맨드 호환 파손 | Medium | 유지 | #93 wrapper policy keep 목록 |
| L-011 | `src/parameter_simulation_gpu.py` | entrypoint wrapper | import-safe wrapper(`#60`) | public API/실행 커맨드 호환 파손 | Medium | 유지 | #93 keep 목록 |
| L-012 | `src/parameter_simulation_gpu_lib.py` | legacy import compat | broad compatibility facade (`src.optimization.gpu.*` re-export) | legacy import 깨짐 | Medium | 유지(예외) | thin wrapper는 아니지만 Gate B에서 명시적 예외로 유지 |
| L-013 | `src/tier_parity_monitor.py` | 운영보조 wrapper | parity 명령 래핑 + PASS/FAIL 출력 | 즉시 삭제 시 운영 모니터링 편의 하락 | Low | 아카이브 후보 | `todos/done_2026_02_09-issue56-cpu-gpu-parity-topk.md`에서 사용 |
| L-014 | `reproduce_issue.py` | 유휴 재현 스크립트 | 단발성 pykrx 재현 코드(레포 참조 없음) | 삭제 리스크 낮음 | Low | 아카이브 후보 | 코드/문서 참조 검색 결과 없음 |
| L-015 | `reproduce_issue_v2.py` | 유휴 재현 스크립트 | 단발성 pykrx 재현 코드(레포 참조 없음) | 삭제 리스크 낮음 | Low | 아카이브 후보 | 코드/문서 참조 검색 결과 없음 |

## 4. Gate 기반 실행 계획
- Gate A 산출물: 위 인벤토리의 `recommendation` 동결
- Gate B 산출물: 제거/축소 승인 목록 확정(항목별)
- Gate C 산출물: PR 반영 결과 + 롤백 경로 + 최종 승인 기록

## 5. 실행 체크리스트
- [x] 레거시 인벤토리 1차 작성
- [x] 저위험 archive 이동(3건) 반영
- [x] fallback 축소 1차 반영(`L-003`, `L-005`)
- [x] Strict-only 전환 1단계: `strict_hysteresis_v1` + `candidate_source_mode='tier'`/`use_weekly_alpha_gate=False`만 허용(legacy 입력 시 fail-fast)
- [x] Strict-only 전환 2단계: synthetic sample pack 관찰(서로 다른 window 10개 + matched parity sample 1개)
- [ ] Strict-only 전환 3단계: non-strict 코드/테스트/문서 완전 삭제
- [x] Gate A 승인 완료
- [x] 제거/축소 대상 확정안 작성
- [x] Gate B 승인 완료
- [x] 승인 범위만 PR로 반영
- [x] Gate C 승인 완료
- [x] `TODO.md`/`todos/` 동기화
- [x] Financial batch legacy fallback surface 제거(`allow_legacy_fallback`, `--allow-financial-legacy-fallback`)
- [x] OHLCV batch legacy fallback surface 제거(`allow_legacy_fallback`, `--allow-legacy-fallback`)

## 6. 완료 기준
- [x] 전수조사 인벤토리 완료(근거 포함)
- [x] 유지/축소/제거/아카이브 분류 확정
- [x] 사용자 승인 게이트 A/B/C 기록 완료

## 6-1. 2026-03-07 strict-only step 1 / wrapper drift 반영
- 구현:
  - `src/tier_hysteresis_policy.py` 추가
  - `src/candidate_runtime_policy.py` 추가
  - `src/gpu_execution_policy.py` 추가
  - CPU/GPU/optimizer/debug/parity가 동일 strict validator를 사용
  - `src/pipeline_batch.py`, `src/ohlcv_batch.py`를 canonical module thin wrapper로 재정렬
- 테스트/문서:
  - `tests/test_cpu_strategy_entry_context.py`
  - `tests/test_gpu_engine_prep_path.py`
  - `tests/test_gpu_parameter_simulation_orchestration.py`
  - `tests/test_gpu_parameter_batch_fallback.py`
  - `tests/test_gpu_execution_policy.py`
  - `tests/test_issue67_tier_universe.py`
  - `tests/test_issue97_retained_wrapper_drift.py`
  - `tests/test_parity_sell_event_dump.py`
  - `tests/test_pipeline_batch.py`
  - `tests/test_ohlcv_batch.py`
  - `tests/test_wrapper_usage.py`
  - `tests/test_issue69_entrypoint_compat.py`
  - `tests/test_issue69_cpu_backtest_wrapper_compat.py`
  - `config/config.example.yaml`
- 해석:
  - strict-only step 1은 닫혔다.
  - retained wrapper drift도 `pipeline_batch`/`ohlcv_batch` 기준으로 닫혔다.
  - 남은 것은 step 3 cleanup이다.

## 7. 다음 PR 제안 (분리 권장)
- PR-97A: 인벤토리/정책 문서 고정 (코드 변경 없음)
- PR-97B: 저위험 아카이브 이동 (`reproduce_issue*.py`)
- PR-97C: 축소 대상 fallback 제거(`ohlcv`/`financial` legacy fallback 제거, mode fallback fail-fast 전환)

## 7-1. 명시적 제외 범위
- 성능 튜닝(배치 크기 최적화, kernel launch 최소화, GPU util 개선)은 #98에서 수행한다.
- #97은 삭제/축소 거버넌스와 승인 게이트 관리에 집중한다.

## 8. Strict-only 전환 계획 (합의안)
목표: `tier_hysteresis_mode`와 runtime candidate policy의 non-strict 분기를 단계적으로 제거해 정책/코드 경로를 단순화한다.

### Step 1. strict만 허용 (fail-fast)
- 적용 대상: CPU/GPU 공통 실행 경로
- 정책:
  - `tier_hysteresis_mode != strict_hysteresis_v1` 입력 시 즉시 오류 반환
  - `candidate_source_mode != tier` 또는 `use_weekly_alpha_gate=True` 입력 시 즉시 오류 반환
- 목적: non-strict 경로 신규 사용을 원천 차단하고 운영 정책을 단일화

### Step 2. Synthetic sample pack 관찰
- SSOT:
  - `run_manifest.json`
  - strict parity/certification JSON(선택 수집)
  - helper: `src.strict_only_governance`
- 관찰 지표:
  - `empty_entry_day_rate`
  - `tier1_coverage`
  - `source_lookup_error_days`
  - `source_missing_days`
  - `source_unknown_days`
  - `metrics_cast_error_count`
  - `pit_failure_days_by_code`
  - `pit_failure_days_by_stage`
  - `parity_mismatch_runs` (strict parity artifact가 있을 때만 집계)
- 통과 기준:
  - 최소 `10`개 strict-only run manifest 확보
  - 최소 `10`개 서로 다른 `backtest_window(start_date,end_date)` 확보
  - 최소 `1`개 strict parity/certification JSON 확보
  - `failed_runs = 0`
  - `promotion_blocked_runs = 0`
  - `degraded_runs = 0`
  - `source_lookup_error_days = 0`
  - `source_missing_days = 0`
  - `source_unknown_days = 0`
  - `metrics_cast_error_count = 0`
  - `pit_failure_days = 0`
  - `fatal_pit_failure_runs = 0`
  - `parity_mismatch_runs = 0`
  - `P95(empty_entry_day_rate) <= 0.20`
  - `median(tier1_coverage) >= 0.55`
- 산출물:
  - `issue97_observation_summary.json`
  - helper 출력 JSON과 원본 `run_manifest.json` 목록

### Step 3. non-strict 완전 삭제
- 삭제 범위:
  - non-strict 분기 코드
  - 관련 테스트 케이스
  - 문서/설정 예시(`legacy` 모드 표기)
- 산출물:
  - 제거 PR + 롤백 메모 + 운영 반영 기록

## 9. 2026-03-07 후속 체크리스트 초안 (retained wrapper contract)
- [x] retained entrypoint wrapper drift 재점검
  - `src.pipeline_batch.py`, `src.ohlcv_batch.py`는 thin wrapper로 재정렬
  - `src.ticker_universe_batch.py`, `src.walk_forward_analyzer.py`, `src.parameter_simulation_gpu.py`는 기존 thin wrapper 유지 확인
  - `src.parameter_simulation_gpu_lib.py`는 Gate B 승인 예외(compat facade)로 유지
- [x] wrapper는 thin forwarder만 허용하도록 기준 명문화
  - canonical 구현 모듈만 import
  - CLI/help/argparse는 canonical과 동일 동작 유지
- [x] real entrypoint regression test 보강
  - `python -m src.pipeline_batch --help`
  - `python -m src.ohlcv_batch --help`
  - wrapper import/export뿐 아니라 canonical delegation까지 검증
- [x] `src.pipeline_batch.py`와 `src.pipeline.batch`의 SSOT 위치 재확정
  - canonical 구현은 `src.pipeline.*`, `src.*` wrapper는 entrypoint thin forwarder로 고정
- [x] thin wrapper 유지 항목은 삭제 대상이 아니라 drift 방지 대상으로 분류 업데이트

## 10. 2026-03-07 Gate C / Step 2 기준 고정
- 구현:
  - `src.strict_only_governance.py`
  - `tests/test_strict_only_governance.py`
  - `src.main_backtest._write_run_manifest()`에 `config.use_weekly_alpha_gate` 기록 추가
- Gate C 승인 기준:
  - 입력: strict parity/certification JSON 2개 이상
  - 필수 모드:
    - `record_strict`
    - `replay_strict`
  - 통과 조건:
    - `summary.policy_ready_for_release_gate = true`
    - `summary.curve_level_parity_zero_mismatch = true`
    - `summary.decision_level_parity_zero_mismatch = true`
    - `summary.promotion_blocked = false`
    - `summary.decision_evidence_release_fields_complete = true`
    - `summary.decision_evidence_pit_failure_rows = 0`
    - 모든 evidence row에서 `candidate_order_zero_mismatch = true`
    - 모든 evidence row에서 `pit_failure_code = null`
  - 출력:
    - `approved`
    - `seen_frozen_manifest_modes`
    - `reasons`
- 실행 명령:
```bash
python -m src.strict_only_governance \
  --mode gate-c \
  --parity-json results/issue67_record_parity_retry.json \
  --parity-json results/issue67_replay_parity_retry.json \
  --out results/issue97_gate_c_summary.json
```
- Step 2 synthetic sample pack 수집 명령:
```bash
python -m src.strict_only_governance \
  --mode observation \
  --run-manifest-glob 'results/issue97_observation/run_*/run_manifest.json' \
  --parity-json results/issue67_record_parity_retry.json \
  --parity-json results/issue67_replay_parity_retry.json \
  --out results/issue97_observation_summary.json
```
- 해석:
  - `Gate C`는 correctness/stability 승인 게이트다. 성능 승인 게이트가 아니다.
  - 성능 회귀 판정은 `#98` 계측에서 따로 닫는다.
  - helper CLI는 `approved=false`면 exit code `1`을 반환한다.
  - `step 2`는 live 14일 관찰이 아니라 historical synthetic sample pack 관찰로 재정의했다.
  - synthetic sample pack의 parity artifact는 최소 1개 이상이 pack 내부 backtest window와 직접 매칭되어야 한다.
- 승인 증적:
  - `results/issue97_gate_c_summary.json`
  - `results/issue97_observation_summary_synthetic.json`
  - `results/issue97_observation/issue97_parity_record_20260105_20260130.json`
