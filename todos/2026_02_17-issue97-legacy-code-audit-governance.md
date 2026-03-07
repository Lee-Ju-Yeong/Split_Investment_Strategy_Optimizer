# Issue #97: Legacy Audit And Strict-Only Governance

> Type: `implementation`
> Status: `in progress`
> Priority: `P1`
> Last updated: 2026-03-07
> Related issues: `#97`, `#93`, `#98`
> Gate status: `Gate A/B/C pending`

## 1. One-Page Summary
- What: 레거시, fallback, 호환 경로를 무작정 지우지 않고 승인 게이트를 거쳐 정리하는 문서입니다.
- Why: 실행 경로를 단순화해야 parity와 문서 가독성도 같이 좋아지지만, 잘못 지우면 운영 계약이 깨집니다.
- Current status: 1차 인벤토리, 저위험 아카이브 이동, fallback 축소 1차는 끝났습니다.
- Next action: Gate A/B 승인과 strict-only step 1을 고정합니다.

## 2. Fixed Rules
- 근거 없는 즉시 삭제는 금지합니다.
- `entrypoint wrapper`와 핵심 실행 경로는 `#93` 정책과 충돌 없이 다뤄야 합니다.
- 성능 튜닝은 `#98`, 정책 삭제/축소는 `#97`이 담당합니다.

## 3. Current Plan
- [x] 레거시 인벤토리 1차 작성
- [x] 저위험 archive 이동
- [x] fallback 축소 1차 반영
- [ ] Gate A 승인
- [ ] Gate B 승인
- [ ] Gate C 승인
- [ ] strict-only step 1: `strict_hysteresis_v1`만 허용
- [ ] strict-only step 2: 운영 지표 최소 2주 관찰
- [ ] strict-only step 3: non-strict 코드/테스트/문서 삭제

## 4. Key Evidence
- 이미 끝난 것:
  - archive 이동 3건
  - `allow_legacy_fallback`과 tier fallback 일부 축소
- 아직 필요한 것:
  - 승인 기록
  - retained wrapper drift 재점검
  - real entrypoint regression test 보강

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
  - `L-005`: `--allow-legacy-fallback` deprecated 명시(문구/경고)
- [ ] Gate A: 인벤토리 확정 승인
- [ ] Gate B: 제거/축소 대상 항목별 승인
- [ ] Gate C: 배포 전 최종 승인

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
| L-001 | `src/backtest/cpu/strategy.py` | 종목선정/우선순위 | `candidate_source_mode`(`weekly/tier/hybrid`)와 `additional_buy_priority`로 신호 순서 결정 | 변경 시 CPU/GPU parity 및 종목선정 결과 변동 | High | 유지 | tier/hybrid/fallback 및 strict hysteresis 분기 존재 |
| L-002 | `src/backtest/cpu/execution.py` | 체결/정산 | 한국시장 호가 반올림 + 매수/매도 체결 정산 SSOT | 체결가/수량/손익 왜곡 | High | 유지 | `_adjust_price_up`, `_execute_buy/_execute_sell` |
| L-003 | `src/data_handler.py` | 후보군 fallback | `get_candidates_with_tier_fallback` (tier1 우선, tier<=2 fallback) | 후보군 공백/의도치 않은 전략 드리프트 | Medium | 축소 | #67 정책 전환 단계 fallback 핵심 |
| L-004 | `src/backtest/gpu/engine.py` | 모드 호환 fallback | invalid `candidate_source_mode` 시 `weekly` fallback | 설정 오류가 조용히 숨겨질 위험 | Medium | 축소 | warning 후 fallback 처리 |
| L-005 | `src/pipeline/ohlcv_batch.py` | 데이터 fallback | `--allow-legacy-fallback`로 history 비어있을 때 legacy 유니버스 사용 | 데이터 소스 일관성 저하 가능성 | Medium | 축소 | TODO/P0 후속에서 제거 예정으로 명시 |
| L-006 | `src/main_script.py` | legacy orchestrator | 주간 필터 CSV/DB 경로 기반 legacy/general 파이프라인 | 신규 배치 체계와 중복 운영 리스크 | Medium | 축소 | `legacy/general data pipeline` 명시 |
| L-007 | `src/pipeline_batch.py` | entrypoint wrapper | `src.pipeline.batch` thin wrapper | 운영 커맨드 계약 파손 | Medium | 유지 | #93 wrapper policy keep 목록 |
| L-008 | `src/ticker_universe_batch.py` | entrypoint wrapper | `src.pipeline.ticker_universe_batch` thin wrapper | 운영 스크립트 호환 파손 | Medium | 유지 | #93 wrapper policy keep 목록 |
| L-009 | `src/ohlcv_batch.py` | entrypoint wrapper | `src.pipeline.ohlcv_batch` thin wrapper | 운영 스크립트 호환 파손 | Medium | 유지 | #93 wrapper policy keep 목록 |
| L-010 | `src/walk_forward_analyzer.py` | entrypoint wrapper | `src.analysis.walk_forward_analyzer` thin wrapper | 운영/문서 커맨드 호환 파손 | Medium | 유지 | #93 wrapper policy keep 목록 |
| L-011 | `src/parameter_simulation_gpu.py` | entrypoint wrapper | import-safe wrapper(`#60`) | public API/실행 커맨드 호환 파손 | Medium | 유지 | #93 keep 목록 |
| L-012 | `src/parameter_simulation_gpu_lib.py` | legacy import compat | `src.optimization.gpu.*` re-export | legacy import 깨짐 | Medium | 유지 | #93 keep 목록 + compat 테스트 |
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
- [ ] Strict-only 전환 1단계: `strict_hysteresis_v1`만 허용(legacy 입력 시 fail-fast)
- [ ] Strict-only 전환 2단계: 운영 지표 관찰(최소 2주)
- [ ] Strict-only 전환 3단계: non-strict 코드/테스트/문서 완전 삭제
- [ ] Gate A 승인 완료
- [ ] 제거/축소 대상 확정안 작성
- [ ] Gate B 승인 완료
- [ ] 승인 범위만 PR로 반영
- [ ] Gate C 승인 완료
- [ ] `TODO.md`/`todos/`/이슈 코멘트 동기화

## 6. 완료 기준
- [ ] 전수조사 인벤토리 완료(근거 포함)
- [ ] 유지/축소/제거/아카이브 분류 확정
- [ ] 사용자 승인 게이트 A/B/C 기록 완료
- [ ] 승인 범위 제거 PR 병합 완료

## 7. 다음 PR 제안 (분리 권장)
- PR-97A: 인벤토리/정책 문서 고정 (코드 변경 없음)
- PR-97B: 저위험 아카이브 이동 (`reproduce_issue*.py`)
- PR-97C: 축소 대상 fallback 제거(`ohlcv` legacy fallback, mode fallback fail-fast 전환)

## 7-1. 명시적 제외 범위
- 성능 튜닝(배치 크기 최적화, kernel launch 최소화, GPU util 개선)은 #98에서 수행한다.
- #97은 삭제/축소 거버넌스와 승인 게이트 관리에 집중한다.

## 8. Strict-only 전환 계획 (합의안)
목표: `tier_hysteresis_mode`의 non-strict(`legacy`) 분기를 단계적으로 제거해 정책/코드 경로를 단순화한다.

### Step 1. strict만 허용 (fail-fast)
- 적용 대상: CPU/GPU 공통 실행 경로
- 정책: `tier_hysteresis_mode != strict_hysteresis_v1` 입력 시 즉시 오류 반환
- 목적: non-strict 경로 신규 사용을 원천 차단하고 운영 정책을 단일화

### Step 2. 운영 지표 관찰 (최소 2주)
- 관찰 지표:
  - `empty_entry_day_rate` (Tier1 후보 부재로 신규진입이 없는 일 비율)
  - `tier1_coverage` (거래일별 Tier1 후보 존재율)
  - `parity_mismatch_count` (strict 기준)
- 통과 기준:
  - parity mismatch `0` 유지
  - `empty_entry_day_rate`가 운영 허용 범위 내
  - 장애/성능 회귀 없음

### Step 3. non-strict 완전 삭제
- 삭제 범위:
  - non-strict 분기 코드
  - 관련 테스트 케이스
  - 문서/설정 예시(`legacy` 모드 표기)
- 산출물:
  - 제거 PR + 롤백 메모 + 운영 반영 기록

## 9. 2026-03-07 후속 체크리스트 초안 (retained wrapper contract)
- [ ] retained entrypoint wrapper drift 재점검
  - `src.pipeline_batch.py`
  - `src.ohlcv_batch.py`
  - `src.ticker_universe_batch.py`
- [ ] wrapper는 thin forwarder만 허용하도록 기준 명문화
  - canonical 구현 모듈만 import
  - CLI/help/argparse는 canonical과 동일 동작 유지
- [ ] real entrypoint regression test 보강
  - `python -m src.pipeline_batch --help`
  - `python -m src.ohlcv_batch --help`
  - wrapper import/export뿐 아니라 canonical delegation까지 검증
- [ ] `src.pipeline_batch.py`와 `src.pipeline.batch`의 SSOT 위치 재확정
  - 테스트 경로와 운영 경로가 다른 파일을 보지 않도록 정리
- [ ] thin wrapper 유지 항목은 삭제 대상이 아니라 drift 방지 대상으로 분류 업데이트
