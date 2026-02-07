# 리팩토링 TODO

이 파일은 상위 수준의 리팩토링 목표를 정리합니다. 상세 범위와 논의는 GitHub 이슈에서 진행합니다.
이슈가 닫히면 이 목록도 함께 갱신하세요.
상세 실행 메모는 `todos/YYYY_MM_DD-issue<N>-<name>.md` 패턴으로 관리합니다.

## 이슈별 상세 TODO 문서
- [ ] 이슈 #64 PIT/룩어헤드 방지: `todos/2026_02_07-issue64-point-in-time-lookahead-bias.md`

## 전체 우선순위 (Global Backlog)

### P0 (신뢰도/데이터 정합성 선행)
- [ ] Point-in-Time 규칙 명문화 및 룩어헤드 방지 테스트 추가 (이슈 #64): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/64
- [ ] `FinancialData`/`InvestorTradingTrend`/`DailyStockTier` 스키마 및 인덱스 추가 (이슈 #65): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/65
- [ ] 재무·수급 수집기 분리 + Tier 사전계산 배치(백필/일배치) 도입 (이슈 #66): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/66

### P1 (실행 경로/운영 안정화)
- [ ] `DataHandler` PIT 조인 확장 + `tier=1 -> tier<=2` fallback 조회 적용 (이슈 #67): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/67
- [ ] 유동성 필터(일평균 거래대금 하한) config 기반 적용 및 회귀 테스트 (이슈 #67 범위 포함)
- [ ] 설정 소스 표준화 및 하드코딩 경로/플래그 제거 (이슈 #53): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/53
- [ ] 데이터 파이프라인 모듈화(DataPipeline) 및 레거시 스크립트 정리 (이슈 #54): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/54
- [ ] 구조화된 로깅 도입 및 하드코딩 디버그 출력 제거 (이슈 #55): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/55
- [ ] DB 접근 계층 표준화(connector/engine) (이슈 #58): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/58

### P2 (전략 고도화/개선)
- [ ] ATR 단일 랭킹을 멀티팩터 랭킹으로 전환 + WFO/Ablation 검증 (이슈 #68): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/68
- [ ] CPU/GPU 결과 정합성(Parity) 테스트 하네스 추가 (이슈 #56): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/56
- [ ] 도메인 모델/캐시 통합(Position, CompanyInfo 캐시) (이슈 #57): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/57
- [ ] 테스트 인터페이스 갱신(test_integration.py) (이슈 #59): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/59
- [ ] 스크립트 import 부작용 제거(parameter_simulation_gpu.py) (이슈 #60): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/60
- [ ] 임포트 스타일 통일(상대/절대/스크립트 실행) (이슈 #61): https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/61
