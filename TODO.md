# 리팩토링 TODO

이 파일은 상위 수준의 리팩토링 목표를 정리합니다. 상세 범위와 논의는 GitHub 이슈에서 진행합니다.
이슈가 닫히면 이 목록도 함께 갱신하세요.
상세 실행 메모는 `todos/YYYY_MM_DD-issue<N>-<name>.md` 패턴으로 관리합니다.

## 이슈별 상세 TODO 문서
- [x] 이슈 #64 PIT/룩어헤드 방지: `todos/done_2026_02_07-issue64-point-in-time-lookahead-bias.md`
- [x] 이슈 #65 스키마/인덱스 확장: `todos/done_2026_02_07-issue65-financial-investor-tier-schema-index.md`
- [ ] 이슈 #66 수집기 분리/사전계산 배치: `todos/2026_02_07-issue66-financial-investor-collector-tier-batch.md`
- [x] 이슈 #70 상폐 포함 Historical Universe: `todos/2026_02_08-issue70-historical-ticker-universe-delisted.md`
- [ ] 이슈 #71 pykrx 확장 데이터셋 + Tier v2 로드맵: `todos/2026_02_08-issue71-pykrx-tier-v2-data-roadmap.md`

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
- [ ] `DailyStockPrice` 전기간 재적재(진행중): KRX raw(`adjusted=False`)를 SSOT로 재정렬
  - 실행 커맨드: `python -m src.ohlcv_batch --start-date 19950101 --end-date <today> --log-interval 20`
  - 운영 원칙: `resume=True` 유지(중단 후 동일 명령 재실행), `allow_legacy_fallback=False` 유지
  - 완료 후 필수: `docs/database/backfill_validation_runbook.md`의 SQL/커버리지 점검 실행
- [ ] `adj_close`/`adj_ratio` 파생 계산 배치 추가(보류): raw OHLCV 적재 이후 보정계수 산출 및 업데이트 배치 구현

### P1 (실행 경로/운영 안정화)
- [ ] pykrx 확장 데이터셋 도입 로드맵 실행 (이슈 #71): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/71
  - Phase P0: `MarketCapDaily`, `ShortSellingDaily` 우선 적재 및 PIT 규칙 반영
  - Phase P1: `ForeignOwnershipDaily`, `SectorClassificationHistory` 추가
  - Phase P2: `IndexDaily`, `IndexConstituentHistory` 및 Tier v2 멀티팩터 고도화
- [ ] `DataHandler` PIT 조인 확장 + `tier=1 -> tier<=2` fallback 조회 적용 (이슈 #67): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/67
- [ ] 유동성 필터(일평균 거래대금 하한) config 기반 적용 및 회귀 테스트 (이슈 #67 범위 포함)
- [ ] 설정 소스 표준화 및 하드코딩 경로/플래그 제거 (이슈 #53): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/53
- [ ] 데이터 파이프라인 모듈화(DataPipeline) 및 레거시 스크립트 정리 (이슈 #54): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/54
- [ ] `src` 패키지 구조 재편 및 대형 모듈 브레이크다운(동작 동일 리팩터링) (이슈 #69): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/69
- [ ] `ohlcv_batch` legacy fallback 제거 (운영 1~2주 fallback 0회 확인 후) (이슈 #70 후속)
- [ ] 구조화된 로깅 도입 및 하드코딩 디버그 출력 제거 (이슈 #55): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/55
- [ ] DB 접근 계층 표준화(connector/engine) (이슈 #58): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/58

### P2 (전략 고도화/개선)
- [ ] ATR 단일 랭킹을 멀티팩터 랭킹으로 전환 + WFO/Ablation 검증 (이슈 #68): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/68
- [ ] CPU/GPU 결과 정합성(Parity) 테스트 하네스 추가 (이슈 #56): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/56
- [ ] 도메인 모델/캐시 통합(Position, CompanyInfo 캐시) (이슈 #57): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/57
- [ ] 테스트 인터페이스 갱신(test_integration.py) (이슈 #59): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/59
- [ ] 스크립트 import 부작용 제거(parameter_simulation_gpu.py) (이슈 #60): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/60
- [ ] 임포트 스타일 통일(상대/절대/스크립트 실행) (이슈 #61): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/61
