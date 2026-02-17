# 리팩토링 TODO

이 파일은 상위 수준의 리팩토링 목표를 정리합니다. 상세 범위와 논의는 GitHub 이슈에서 진행합니다.
이슈가 닫히면 이 목록도 함께 갱신하세요.
상세 실행 메모는 `todos/YYYY_MM_DD-issue<N>-<name>.md` 패턴으로 관리합니다.

## 이슈별 상세 TODO 문서
- [x] 이슈 #64 PIT/룩어헤드 방지: `todos/done_2026_02_07-issue64-point-in-time-lookahead-bias.md`
- [x] 이슈 #65 스키마/인덱스 확장: `todos/done_2026_02_07-issue65-financial-investor-tier-schema-index.md`
- [x] 이슈 #66 수집기 분리/사전계산 배치: `todos/done_2026_02_07-issue66-financial-investor-collector-tier-batch.md`
- [x] 이슈 #70 상폐 포함 Historical Universe: `todos/done_2026_02_08-issue70-historical-ticker-universe-delisted.md`
- [x] 이슈 #59 테스트 인터페이스 갱신: `todos/done_2026_02_15-issue59-tests-split-no-db-gpu.md`
- [x] 이슈 #60 스크립트 import 부작용 제거: `todos/done_2026_02_15-issue60-import-parameter-simulation-gpu-py.md`
- [x] 이슈 #61 임포트 스타일 통일: `todos/done_2026_02_15-issue61-import-style-standardization.md`
- [x] 이슈 #69 `src` 패키지 구조 재편/브레이크다운: `todos/done_2026_02_16-issue69-src-package-restructure-breakdown.md`
- [x] 이슈 #93 Wrapper deprecation/removal 계획: `todos/2026_02_16-issue93-wrapper-deprecation-removal-plan.md`
- [ ] 이슈 #71 pykrx 확장 데이터셋 + Tier v2 로드맵: `todos/2026_02_08-issue71-pykrx-tier-v2-data-roadmap.md`
- [ ] 이슈 #67 PIT 조인 확장 + A안 전환(Tier universe): `todos/2026_02_09-issue67-tier-universe-migration.md`
- [ ] 이슈 #68 멀티팩터 + Robust WFO/Ablation: `todos/2026_02_09-issue68-robust-wfo-ablation.md`
- [x] 이슈 #56 CPU/GPU Parity 하네스(top-k): `todos/2026_02_09-issue56-cpu-gpu-parity-topk.md`

## 전체 우선순위 (Global Backlog)

### P0 (신뢰도/데이터 정합성 선행)
- [x] Point-in-Time 규칙 명문화 및 룩어헤드 방지 테스트 추가 (이슈 #64): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/64
- [x] `FinancialData`/`InvestorTradingTrend`/`DailyStockTier` 스키마 및 인덱스 추가 (이슈 #65): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/65
- [x] 재무·수급 수집기 분리 + Tier 사전계산 배치(백필/일배치) 도입 (이슈 #66): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/66
  - [x] InvestorTradingTrend 포함 Tier v1 read-only 튜닝 및 임계값 확정안 도출
- [x] 상폐 포함 `TickerUniverseSnapshot`/`TickerUniverseHistory` 구축 (이슈 #70): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/70
  - [x] Phase 1 코드 반영: 스키마/인덱스 + `ticker_universe_batch.py` + `pipeline_batch --run-universe` 옵션 추가
  - [x] 운영 검증: 스냅샷/히스토리 백필 1회 실행 및 샘플 상폐 종목 검증
  - [x] Phase 2 코드 반영: `ohlcv_batch` history 소스 + 수집 기간 교집합 적용
- [x] 운영 DB 스키마 반영 실행 (`create_tables`) 및 테이블/인덱스 검증 (`FinancialData`, `InvestorTradingTrend`, `DailyStockTier`, `TickerUniverseSnapshot`, `TickerUniverseHistory`)
- [x] 초기 1회 백필 실행 (`python -m src.pipeline_batch --mode backfill --start-date <YYYYMMDD> --end-date <YYYYMMDD>`) 후 일배치 전환
- [x] `InvestorTradingTrend` 전기간 백필(1995~현재) 재실행
  - 실행 정책: `TickerUniverseHistory`(상폐 포함) 기준 유니버스 사용, 기존 적재 구간의 최소일 이전만 API 호출
  - 실행 중(2026-02-09): `python -m src.pipeline_batch --mode backfill --start-date 19950101 --end-date <today> --skip-financial --skip-tier --log-interval 20`
  - 로그: `logs/investor_backfill_1995_*.log`
  - 결과(2026-02-10): 완료 (processed=5216, rows_saved=9,261,970)
- [x] `DailyStockPrice` 전기간 재적재: KRX raw(`adjusted=False`)를 SSOT로 재정렬
  - 실행 커맨드: `python -m src.ohlcv_batch --start-date 19950101 --end-date <today> --log-interval 20`
  - 운영 원칙: `resume=True` 유지(중단 후 동일 명령 재실행), `allow_legacy_fallback=False` 유지
  - 검증 결과(2026-02-08): `rows_total=14,750,953`, `tickers_total=4,795`, `min_date=1995-05-08`, `max_date=2026-02-06`, `duplicate_like_rows=0`, `future_rows=0`
- [x] `adj_close`/`adj_ratio` 파생 계산 배치 추가: raw OHLCV 적재 이후 보정계수 산출 및 업데이트 배치 구현
  - [x] Step 1: `CorporateMajorChanges` 스키마 추가 및 `DailyStockPrice` 컬럼 검증
  - [x] Step 2: `corporate_event_collector.py` 구현 및 백필 준비
  - [x] Step 2-1: `corporate_event_collector.py` 관측성/동시성 보강
    - fetch 결과를 `fetch_errors | empty_results | nonempty_results | normalize_empty`로 분리 계측
    - write buffer flush 경로에서 중첩 lock 제거(잠재 deadlock 리스크 완화)
  - [x] Step 2-2: pykrx 원천 응답 정상화 확인(운영 블로커 종료)
    - 2026-02-10 full run: `processed=5216`, `saved_rows=0`, `fetch_errors=0`, `empty=5216`
    - 상태 해석: 수집기 오류가 아니라 원천(`get_stock_major_changes`) 데이터 부재로 판단
    - 결정(2026-02-11): 해당 원천 복구 전까지 `CorporateMajorChanges` 확장 수집은 `external_blocked`로 종료 처리
    - 운영 가드: preflight health-check(`ticker_count`, 샘플 major_changes) fail 시 배치 skip/blocked 처리 유지
  - [x] Step 3: `ohlcv_adjusted_updater.py` 구현 및 테스트(5종) 통과
  - [x] Step 4: 백테스트 보장 구간 기준 종료(지표 계산은 raw `close_price` SSOT 유지)
    - 근거: `indicator_calculator`는 `DailyStockPrice.close_price` 기반 계산 경로를 사용
    - 정책: `adj_close/adj_ratio`는 보정 참조용 파생 컬럼으로 유지하고, 지표 재계산의 강제 게이트에서 분리
  - 운영 실측(2026-02-11):
    - 전체: `total=14,750,953`, `adj_close_not_null=8,417,573`, `adj_ratio_not_null=8,417,559`, `ratio_mismatch=0`
    - 연도 기준: `2014~2026` 각 연도에서 `close_price>0` 행 대비 `adj_close/adj_ratio` 100% (`ratio_mismatch=0`)
    - 일자 기준 최초 완전 커버 시작일: `2013-11-20` (해당 일자 이후 `close_price>0` 행 `adj_*` 누락 0)
    - 결론: 장기 백테스트 보장 구간을 `2013-11-20` 이후로 정의하고, 이전 레거시 구간 null은 허용
- [x] `FinancialData`/`InvestorTradingTrend` 0값 의미 정리(수집기 정책 반영)
  - `FinancialData`: `per<=0`, `pbr<=0`은 `NULL`로 정규화(기존 누적 데이터 포함)
  - `InvestorTradingTrend`: 컬럼 미탐지/미관측은 `NULL`, all-zero 무의미 row는 저장 제외
- [x] `CalculatedIndicators` 재계산 완료 및 검증 통과
  - 기준: `docs/database/backfill_validation_runbook.md`의 6장(전체/구간 기준, 롤백 기준) 준수
- [x] `DailyStockTier` 재계산 완료 및 커버리지 검증
  - 기준: runbook 3장 후속 순서 + 9장(`flow_impact_pct`, `churn_pct`) 게이트 점검
  - 적용 구간: 운영 기준 `2024-01-01 ~ 2026-02-08` (`rows_total=1,329,758`, `invalid_tier_rows=0`)
- [x] P0 Exit Gate 증적 저장(SQL 통과 스냅샷)
  - 필수 증적: `rows_total`, `tickers_total`, `min_date/max_date`, `duplicate_like_rows=0`, `future_rows=0`
  - 저장 위치: 운영 이슈 코멘트 또는 `todos/` 실행 로그 섹션에 명령/결과 함께 기록
  - 최신 증적(2026-02-10):
    - `DailyStockPrice`: `rows_total=14,750,953`, `tickers_total=4,795`, `min_date=1995-05-08`, `max_date=2026-02-06`
    - `FinancialData`: `rows_total=11,792,617`, `tickers_total=3,785`, `min_date=1995-05-02`, `max_date=2026-02-06`
    - `InvestorTradingTrend`: `rows_total=11,065,013`, `tickers_total=4,397`, `min_date=1999-01-04`, `max_date=2026-02-06`
    - `DailyStockTier`: `rows_total=1,329,758`, `tickers_total=2,721`, `min_date=2024-01-02`, `max_date=2026-02-06`
    - `CalculatedIndicators`: `rows_total=14,748,703`, `tickers_total=4,792`, `min_date=1995-05-08`, `max_date=2026-02-06`
    - `duplicates=0`, `future_rows=0` (All tables ok)

### P1 (실행 경로/운영 안정화)
- [ ] 데이터 플로우 의사결정(2026-02-09): 최종 `A안(KRX PIT + DailyStockTier)`로 전환
  - [x] Phase 1(전환기): `C안(Hybrid)` 허용 (`weekly`는 선택 alpha gate로만 사용)
  - [ ] Phase 2(고정): `A안`을 기본값으로 승격, `weekly` 기본 후보군 경로 제거
  - [x] 브랜치 규칙: `A안 전환 코드는 main 직접 작업 금지`, 기능 브랜치에서만 진행
  - [x] 권장 브랜치명: `feature/issue67-a-universe-tier-phase1`, `feature/issue67-a-universe-tier-phase2`
- [ ] pykrx 확장 데이터셋 도입 로드맵 실행 (이슈 #71): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/71
  - Phase P0: `MarketCapDaily`, `ShortSellingDaily` 우선 적재 및 PIT 규칙 반영
  - Phase P1: `ForeignOwnershipDaily`, `SectorClassificationHistory` 추가
  - Phase P2: `IndexDaily`, `IndexConstituentHistory` 및 Tier v2 멀티팩터 고도화
  - Phase P3: `Optuna` 후행 적용(범위 제한)
    - Allowed: Tier v2 수치 임계값/가중치, `#68` robust selection 계층 수치 파라미터
    - Forbidden: `candidate_source_mode`, execution/fee/tick/lag 규칙, WFO fold 분할 규칙
    - Preflight: `#67` 모드 고정 + `#56` parity mismatch `0건` + `#68` hard gate 설정 완료
    - Promotion: feature flag canary 통과 + 재현성(seed/기간/모드) 증적 확보 시에만 운영 반영
  - [ ] P0 테이블 DDL/인덱스 확정
  - [ ] 수집 배치 엔트리(`pipeline_batch`) 확장(일/주/월)
  - [ ] Tier v2 read-only 실험 스크립트 추가
  - [ ] PIT/왜곡 방지 검증 항목 테스트화
  - [ ] pykrx source health-check 유틸/가드 추가(전량 empty 패턴 fail-fast)
  - [ ] Optuna 실험 스크립트/설정 추가(`robust_score` objective, seed 고정)
  - [ ] Optuna 전제조건 체크(후보군 모드/Parity/데이터 커버리지) 자동 가드 추가
  - [ ] Optuna 산출물 저장 규격 정의(`trial params`, `score`, `gate pass/fail`, `metadata`)
  - [ ] Optuna invalid trial 기준 명문화(`INVALID_REPRO`, `INVALID_PARITY`, `INVALID_DATA`)
  - [ ] Optuna run manifest 저장(`config hash`, `data hash`, `env fingerprint`, `git sha`)
  - [ ] mode 전환(`hybrid_transition` -> `tier`) 시 study 분리 강제(혼합 비교 금지)
  - [ ] `docs/database/schema.md` 및 `TODO.md` 동기화
- [ ] `DataHandler` PIT 조인 확장 + `tier=1 -> tier<=2` fallback 조회 적용 (이슈 #67): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/67
  - [x] 후보군 조회 정책 플래그 도입: `weekly | tier | hybrid_transition`
  - [x] `tier=1` 우선, 빈 경우 `tier<=2` fallback 규칙을 CPU/GPU 동일 로직으로 고정
  - [x] 후보군 선택은 결정론 baseline 고정(`random-only` 금지), 동점/정렬 규칙 명문화
  - [x] Phase A parity hardening: GPU `signal_date(T-1)` + ATR as-of + Tier preload(30일 가정 제거) 반영
  - [ ] `#52` 연계: GPU 후보군/신규진입 경로 host 병목 제거(Phase B/C 우선 완료, top-level 중복 이슈로 분리하지 않음)
  - [x] Entry/Hold hysteresis 규칙 반영/문서화(`tier_hysteresis_mode`):
    - `legacy`: `Entry=tier1 -> empty면 tier<=2 fallback`
    - `strict_hysteresis_v1`: `Entry=tier1 only`, `Hold/Add=tier<=2`, `tier3` 강제청산
  - [x] 현재 동작 기준 hysteresis 문서/테스트 보강(2026-02-17): `docs/MAGIC_SPLIT_STRATEGY_PRINCIPLES.md`, `tests/test_issue67_tier_universe.py` (`invalid mode`, `ATR tie-break`, `T-1 호출 인자`, `T+0/T+1`)
  - [ ] `TickerUniverseSnapshot/History` 기반 PIT 후보군 조회를 기본 경로로 구현
  - [ ] `WeeklyFilteredStocks`는 `use_weekly_alpha_gate`가 `true`일 때만 교집합/가중치로 사용
  - [ ] `DailyStockTier` 커버리지 게이트(구간별) 미달 시 실패 처리/리포트 추가
- [ ] 유동성 필터(일평균 거래대금 하한) config 기반 적용 및 회귀 테스트 (이슈 #67 범위 포함)
- [x] 설정 소스 표준화 및 하드코딩 경로/플래그 제거 (이슈 #53): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/53
- [ ] 데이터 파이프라인 모듈화(DataPipeline) 및 레거시 스크립트 정리 (이슈 #54): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/54
- [x] `src` 패키지 구조 재편 및 대형 모듈 브레이크다운(동작 동일 리팩터링) (이슈 #69): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/69
- [x] Wrapper deprecation/removal 단계적 정리 (이슈 #93): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/93
  - [x] Deprecation 정책/일정 문서화
  - [x] 조건부 wrapper 사용처 탐지 규칙(CI/테스트) 추가
  - [x] 조건부 wrapper 제거 패치
  - [x] wrapper compat 테스트/문서 정리
  - [x] 지정 테스트 통과 기록
- [ ] `ohlcv_batch` legacy fallback 제거 (운영 1~2주 fallback 0회 확인 후) (이슈 #70 후속)
- [x] 구조화된 로깅 도입 및 하드코딩 디버그 출력 제거 (이슈 #55): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/55
- [ ] DB 접근 계층 표준화(connector/engine) (이슈 #58): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/58

### P2 (전략 고도화/개선)
- [ ] ATR 단일 랭킹을 멀티팩터 랭킹으로 전환 + WFO/Ablation 검증 (이슈 #68): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/68
  - [ ] `walk_forward_analyzer` 강건 점수 함수 도입: `robust_score = (mean - k*std) * log1p(cluster_size)` 형태 실험/고정
  - [ ] WFO 하드 게이트 도입: `median(OOS/IS) >= 0.60`, `fold pass rate >= 70%`, `OOS MDD p95 <= 25%`
  - [ ] 검증 체계 고정: `deterministic baseline` + `seeded_stress` + `jackknife_drop_topN`
  - [ ] 집중도 리스크 지표(`max_single_stock_contribution`, `HHI`)를 승격 게이트에 포함
  - [ ] 클러스터링 feature 확장: 파라미터 4종 + 행동지표(`trade_count`, `avg_hold_days`) 비교
  - [ ] 민감도 테스트 추가: 선택 파라미터 주변(±10%) perturbation 성능 저하율 측정(목표 `<= 15%`)
  - [ ] `robust_selection_enabled` feature flag와 `legacy` rollback 경로 추가
  - [ ] Ablation 매트릭스 고정: `Legacy-Calmar`, `Robust-Score`, `Robust+Gate`, `Robust+Gate+BehaviorFeature`
- [x] CPU/GPU 결과 정합성(Parity) 테스트 하네스 추가 (이슈 #56): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/56
  - [x] `#67` Phase A parity blocker 선반영: GPU `signal_date(T-1)` + ATR as-of + Tier preload(30일 가정 제거)
  - [x] top-k(권장 100+) 배치 parity 검증 루틴 추가 (`src/cpu_gpu_parity_topk.py`)
  - [x] scenario pack parity 추가(`baseline_deterministic`, `seeded_stress`, `jackknife_drop_topN`)
  - [x] 스냅샷 메타데이터 강화: 파라미터/기간/코드 버전/생성시각 저장
  - [x] 스냅샷 메타데이터에 `scenario_type`, `seed_id`, `drop_top_n` 필드 추가
  - [x] 불일치 리포트 표준화: first mismatch 인덱스 + cash/positions/value 덤프
  - [x] 하드 게이트: parity mismatch `0건`만 pass
  - [x] `candidate_source_mode`별(`weekly`, `hybrid_transition`, `tier`) parity 배치 검증 추가
- [ ] 도메인 모델/캐시 통합(Position, CompanyInfo 캐시) (이슈 #57): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/57
- [x] 테스트 인터페이스 갱신(test_integration.py) (이슈 #59): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/59
- [x] 스크립트 import 부작용 제거(parameter_simulation_gpu.py) (이슈 #60): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/60
- [x] 임포트 스타일 통일(상대/절대/스크립트 실행) (이슈 #61): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/61

### P2-Notes (2026-02-09, 멀티에이전트 합의)
- 최적점(peak) 단일 선택보다 OOS 일관성 중심의 robust cluster 정책을 우선 적용
- 초기 적용 범위는 `선정/검증 계층`에 한정하고, `strategy/execution` 체결 로직은 변경하지 않음
- 운영 반영은 `feature flag`로 점진 전환(문제 시 즉시 legacy 복귀)
- 데이터 플로우 최종안은 `A안(KRX PIT + Tier)`, 전환기에는 `C안(Hybrid)`를 제한적으로 허용
- `A안 전환 구현/병합은 main 직접 작업 금지`, 반드시 별도 기능 브랜치 + PR로 진행

### P2-Notes (2026-02-09, Optuna Scope v1 합의)
- Optuna는 `파라미터 탐색 엔진`으로만 사용하고, 전략 로직/체결 규칙 생성 도구로 사용하지 않음
- Tier는 safety gate 역할을 유지하고, 수익 최적화 objective는 `#68 robust score` 계층에 한정
- `random-only` 선택은 운영 기준으로 금지, 결정론적 baseline + seeded stress test로 강건성 검증
- Trial 채택 하드게이트: parity(`#56`) + OOS/IS + fold pass rate + OOS MDD p95 동시 통과
- 하나라도 fail이면 `STOP` (채택 금지), 모두 pass일 때만 `GO` (canary 후 승격)

### Parking Backlog (조건부 재검토)
- [ ] `#19` CLI 단일 엔트리(`main.py`) 도입: 현재는 모듈별 CLI 체계 유지. 재검토 조건은 P1 핵심(`#54/#58/#93`) 안정화 이후 UX 관점 통합 필요성이 명확할 때
- [ ] `#20` README 상세 문서화(legacy 스코프 재정의 필요): 기존 이슈 본문의 `config.ini` 전제가 현재 정책(`config.yaml` SSOT)과 불일치. 재검토 시 “README-현재상태 정합성 보수”로 재작성
- [ ] `#21` 핵심 모듈 Docstring 확장: low-priority 문서 보강 작업. 재검토 조건은 P1/P2 기능 변경 안정화 이후 API/동작이 고정됐을 때
- [ ] `#30` GPU 결과 리포팅 가독성 개선: 현재 출력 경로는 개편됨. 재검토 조건은 최신 실행 로그 기준 지표 표기 오류가 재현될 때
- [ ] `#35` 최초 현금 고갈 이벤트 기반 동적 파라미터: 전략 확장 범위가 커서 parity/robust gate 선행 과제 완료 후 별도 실험 트랙에서 검토

### Issue Hygiene (2026-02-17)
- [x] `#28` 다양한 종목 선정 우선순위 전략: 사용자 수동 close 완료(2026-02-17)
