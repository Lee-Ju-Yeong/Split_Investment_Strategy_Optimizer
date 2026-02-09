# 리팩토링 TODO

이 파일은 상위 수준의 리팩토링 목표를 정리합니다. 상세 범위와 논의는 GitHub 이슈에서 진행합니다.
이슈가 닫히면 이 목록도 함께 갱신하세요.
상세 실행 메모는 `todos/YYYY_MM_DD-issue<N>-<name>.md` 패턴으로 관리합니다.

## 이슈별 상세 TODO 문서
- [x] 이슈 #64 PIT/룩어헤드 방지: `todos/done_2026_02_07-issue64-point-in-time-lookahead-bias.md`
- [x] 이슈 #65 스키마/인덱스 확장: `todos/done_2026_02_07-issue65-financial-investor-tier-schema-index.md`
- [ ] 이슈 #66 수집기 분리/사전계산 배치: `todos/2026_02_07-issue66-financial-investor-collector-tier-batch.md`
- [x] 이슈 #70 상폐 포함 Historical Universe: `todos/done_2026_02_08-issue70-historical-ticker-universe-delisted.md`
- [ ] 이슈 #71 pykrx 확장 데이터셋 + Tier v2 로드맵: `todos/2026_02_08-issue71-pykrx-tier-v2-data-roadmap.md`
- [ ] 이슈 #67 PIT 조인 확장 + A안 전환(Tier universe): `todos/2026_02_09-issue67-tier-universe-migration.md`
- [ ] 이슈 #68 멀티팩터 + Robust WFO/Ablation: `todos/2026_02_09-issue68-robust-wfo-ablation.md`
- [ ] 이슈 #56 CPU/GPU Parity 하네스(top-k): `todos/2026_02_09-issue56-cpu-gpu-parity-topk.md`

## 전체 우선순위 (Global Backlog)

### P0 (신뢰도/데이터 정합성 선행)
- [x] Point-in-Time 규칙 명문화 및 룩어헤드 방지 테스트 추가 (이슈 #64): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/64
- [x] `FinancialData`/`InvestorTradingTrend`/`DailyStockTier` 스키마 및 인덱스 추가 (이슈 #65): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/65
- [ ] 재무·수급 수집기 분리 + Tier 사전계산 배치(백필/일배치) 도입 (이슈 #66): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/66
  - [x] InvestorTradingTrend 포함 Tier v1 read-only 튜닝 및 임계값 확정안 도출
- [x] 상폐 포함 `TickerUniverseSnapshot`/`TickerUniverseHistory` 구축 (이슈 #70): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/70
  - [x] Phase 1 코드 반영: 스키마/인덱스 + `ticker_universe_batch.py` + `pipeline_batch --run-universe` 옵션 추가
  - [x] 운영 검증: 스냅샷/히스토리 백필 1회 실행 및 샘플 상폐 종목 검증
  - [x] Phase 2 코드 반영: `ohlcv_batch` history 소스 + 수집 기간 교집합 적용
- [ ] 운영 DB 스키마 반영 실행 (`create_tables`) 및 테이블/인덱스 검증 (`FinancialData`, `InvestorTradingTrend`, `DailyStockTier`, `TickerUniverseSnapshot`, `TickerUniverseHistory`)
- [ ] 초기 1회 백필 실행 (`python -m src.pipeline_batch --mode backfill --start-date <YYYYMMDD> --end-date <YYYYMMDD>`) 후 일배치 전환
- [ ] `InvestorTradingTrend` 전기간 백필(1995~현재) 재실행
  - 실행 정책: `TickerUniverseHistory`(상폐 포함) 기준 유니버스 사용, 기존 적재 구간의 최소일 이전만 API 호출
  - 실행 중(2026-02-09): `python -m src.pipeline_batch --mode backfill --start-date 19950101 --end-date <today> --skip-financial --skip-tier --log-interval 20`
  - 로그: `logs/investor_backfill_1995_*.log`
- [x] `DailyStockPrice` 전기간 재적재: KRX raw(`adjusted=False`)를 SSOT로 재정렬
  - 실행 커맨드: `python -m src.ohlcv_batch --start-date 19950101 --end-date <today> --log-interval 20`
  - 운영 원칙: `resume=True` 유지(중단 후 동일 명령 재실행), `allow_legacy_fallback=False` 유지
  - 검증 결과(2026-02-08): `rows_total=14,750,953`, `tickers_total=4,795`, `min_date=1995-05-08`, `max_date=2026-02-06`, `duplicate_like_rows=0`, `future_rows=0`
- [ ] `adj_close`/`adj_ratio` 파생 계산 배치 추가(보류): raw OHLCV 적재 이후 보정계수 산출 및 업데이트 배치 구현
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
  - 최신 증적(2026-02-08):
    - `DailyStockPrice`: `rows_total=14,750,953`, `tickers_total=4,795`, `min_date=1995-05-08`, `max_date=2026-02-06`, `duplicate_like_rows=0`, `future_rows=0`
    - `DailyStockTier`: `rows_total=1,329,758`, `tickers_total=2,721`, `min_date=2024-01-02`, `max_date=2026-02-06`, `invalid_tier_rows=0`
    - `CalculatedIndicators`: `rows_total=14,748,703`, `tickers_total=4,792`, `min_date=1995-05-08`, `max_date=2026-02-06`, `duplicate_like_rows=0`

### P1 (실행 경로/운영 안정화)
- [ ] 데이터 플로우 의사결정(2026-02-09): 최종 `A안(KRX PIT + DailyStockTier)`로 전환
  - [ ] Phase 1(전환기): `C안(Hybrid)` 허용 (`weekly`는 선택 alpha gate로만 사용)
  - [ ] Phase 2(고정): `A안`을 기본값으로 승격, `weekly` 기본 후보군 경로 제거
  - [ ] 브랜치 규칙: `A안 전환 코드는 main 직접 작업 금지`, 기능 브랜치에서만 진행
  - [ ] 권장 브랜치명: `feature/issue67-a-universe-tier-phase1`, `feature/issue67-a-universe-tier-phase2`
- [ ] pykrx 확장 데이터셋 도입 로드맵 실행 (이슈 #71): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/71
  - Phase P0: `MarketCapDaily`, `ShortSellingDaily` 우선 적재 및 PIT 규칙 반영
  - Phase P1: `ForeignOwnershipDaily`, `SectorClassificationHistory` 추가
  - Phase P2: `IndexDaily`, `IndexConstituentHistory` 및 Tier v2 멀티팩터 고도화
- [ ] `DataHandler` PIT 조인 확장 + `tier=1 -> tier<=2` fallback 조회 적용 (이슈 #67): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/67
  - [ ] 후보군 조회 정책 플래그 도입: `weekly | tier | hybrid_transition`
  - [ ] `tier=1` 우선, 빈 경우 `tier<=2` fallback 규칙을 CPU/GPU 동일 로직으로 고정
  - [ ] `TickerUniverseSnapshot/History` 기반 PIT 후보군 조회를 기본 경로로 구현
  - [ ] `WeeklyFilteredStocks`는 `use_weekly_alpha_gate`가 `true`일 때만 교집합/가중치로 사용
  - [ ] `DailyStockTier` 커버리지 게이트(구간별) 미달 시 실패 처리/리포트 추가
- [ ] 유동성 필터(일평균 거래대금 하한) config 기반 적용 및 회귀 테스트 (이슈 #67 범위 포함)
- [ ] 설정 소스 표준화 및 하드코딩 경로/플래그 제거 (이슈 #53): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/53
- [ ] 데이터 파이프라인 모듈화(DataPipeline) 및 레거시 스크립트 정리 (이슈 #54): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/54
- [ ] `src` 패키지 구조 재편 및 대형 모듈 브레이크다운(동작 동일 리팩터링) (이슈 #69): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/69
- [ ] `ohlcv_batch` legacy fallback 제거 (운영 1~2주 fallback 0회 확인 후) (이슈 #70 후속)
- [ ] 구조화된 로깅 도입 및 하드코딩 디버그 출력 제거 (이슈 #55): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/55
- [ ] DB 접근 계층 표준화(connector/engine) (이슈 #58): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/58

### P2 (전략 고도화/개선)
- [ ] ATR 단일 랭킹을 멀티팩터 랭킹으로 전환 + WFO/Ablation 검증 (이슈 #68): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/68
  - [ ] `walk_forward_analyzer` 강건 점수 함수 도입: `robust_score = (mean - k*std) * log1p(cluster_size)` 형태 실험/고정
  - [ ] WFO 하드 게이트 도입: `median(OOS/IS) >= 0.60`, `fold pass rate >= 70%`, `OOS MDD p95 <= 25%`
  - [ ] 클러스터링 feature 확장: 파라미터 4종 + 행동지표(`trade_count`, `avg_hold_days`) 비교
  - [ ] 민감도 테스트 추가: 선택 파라미터 주변(±10%) perturbation 성능 저하율 측정(목표 `<= 15%`)
  - [ ] `robust_selection_enabled` feature flag와 `legacy` rollback 경로 추가
  - [ ] Ablation 매트릭스 고정: `Legacy-Calmar`, `Robust-Score`, `Robust+Gate`, `Robust+Gate+BehaviorFeature`
- [ ] CPU/GPU 결과 정합성(Parity) 테스트 하네스 추가 (이슈 #56): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/56
  - [ ] top-k(권장 100+) 배치 parity 검증 루틴 추가 (`debug_gpu_single_run` 기반)
  - [ ] 스냅샷 메타데이터 강화: 파라미터/기간/코드 버전/생성시각 저장
  - [ ] 불일치 리포트 표준화: first mismatch 인덱스 + cash/positions/value 덤프
  - [ ] 하드 게이트: parity mismatch `0건`만 pass
  - [ ] `candidate_source_mode`별(`weekly`, `hybrid_transition`, `tier`) parity 배치 검증 추가
- [ ] 도메인 모델/캐시 통합(Position, CompanyInfo 캐시) (이슈 #57): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/57
- [ ] 테스트 인터페이스 갱신(test_integration.py) (이슈 #59): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/59
- [ ] 스크립트 import 부작용 제거(parameter_simulation_gpu.py) (이슈 #60): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/60
- [ ] 임포트 스타일 통일(상대/절대/스크립트 실행) (이슈 #61): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/61

### P2-Notes (2026-02-09, 멀티에이전트 합의)
- 최적점(peak) 단일 선택보다 OOS 일관성 중심의 robust cluster 정책을 우선 적용
- 초기 적용 범위는 `선정/검증 계층`에 한정하고, `strategy/execution` 체결 로직은 변경하지 않음
- 운영 반영은 `feature flag`로 점진 전환(문제 시 즉시 legacy 복귀)
- 데이터 플로우 최종안은 `A안(KRX PIT + Tier)`, 전환기에는 `C안(Hybrid)`를 제한적으로 허용
- `A안 전환 구현/병합은 main 직접 작업 금지`, 반드시 별도 기능 브랜치 + PR로 진행
