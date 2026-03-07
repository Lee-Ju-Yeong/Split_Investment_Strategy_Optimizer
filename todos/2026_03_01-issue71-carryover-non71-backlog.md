# chore(planning): issue71 범위 외 미완료 항목 이관 백로그
- source: `todos/done_2026_02_08-issue71-pykrx-tier-v2-data-roadmap.md`
- 목적: 이슈 #71(close) 문서에 남아 있던 미완료 항목 중 범위 외/후속 과제를 분리 추적
- 분리 원칙:
  - 이슈 #71 완료 범위를 넘는 확장 과제는 본 문서로 이관
  - 구현/검증 트랙은 기존 연관 이슈(#56/#67/#68/#101) 문서와 동기화

## 1) 데이터 파이프라인/스키마 확장 (후속 이슈 트랙)
- [ ] pykrx source health-check preflight 유틸/가드 공통화(전량 empty fail-fast)
- [ ] P0 테이블 DDL/인덱스 잔여 확정 항목 정리
- [ ] 수집 배치 엔트리(`pipeline_batch`) 일/주/월 운영 시나리오 확장
- [ ] `MarketCapDaily` 적재 시 Common Stock only 제외 규칙 고정(ETF/ETN/ELW/SPAC 제외)
- [ ] `MarketCapDaily` gap-audit/checkpoint 또는 replay command 추가
  - 목표: `MIN/MAX(date)`만으로 놓치는 중간 hole 검출
  - 산출물: coverage summary 또는 재실행 대상 window 추출
- [ ] 거래정지/비정상 거래일 파생 플래그(`halt`, `zero-volume`) 정책 정의
- [ ] (선택) `FundamentalDaily` 병행 수집 여부 결정 (`get_market_fundamental(date)`)
- [ ] `SectorClassificationHistory` 스냅샷 + SCD Type2 적재 워커 추가
- [ ] `short_volume/short_value/short_balance` 수집 정상화 + 컬럼 매핑 drift guard

## 2) Tier/Ranking 고도화 및 연구 트랙
- [ ] Tier v2 read-only shadow 실험 스크립트/리포트 파이프라인 정리
- [ ] 공매도 랭킹 반영(`shadow -> gated -> default`, 런타임 재조인 금지)
- [ ] `ShortSellingDaily` publication lag 보정 이후 `sbv_ratio` shadow/backfill 재검증
- [ ] `DailyStockTier` 계산 경로의 shadow 결과 검증(분포/이동률/안정성)
- [ ] MVP 파생변수 8~10개 `DailyStockTier` 저장
- [ ] Tier 행동 규칙 반영 정리(Tier1=신규, Tier2=추매 가능, Tier3=추매 유보)
- [ ] 결측 처리 공통 규칙 코드화(`safe_div`, `missing_flag`, `confidence`, coverage gate)
- [ ] CPU/GPU 공통 조회/랭킹 경로에 신규 변수 연결
- [ ] parity + PIT + coverage 회귀 테스트 확장
- [ ] 백필/검증 리포트 저장 후 `shadow -> gated -> default` 전환 절차 정식화

## 3) Optuna / Robust WFO / 거버넌스 트랙
- [ ] Optuna 실험 스크립트/설정 추가(`robust_score`, seed 고정)
- [ ] Optuna 전제조건 체크 자동 가드(후보군 모드/Parity/커버리지)
- [ ] Optuna 산출물 저장 규격 정의(`trial params`, `score`, `gate pass/fail`, `metadata`)
- [ ] Optuna invalid trial 기준 명문화(`INVALID_REPRO`, `INVALID_PARITY`, `INVALID_DATA`)
- [ ] Optuna run manifest 저장(`config hash`, `data hash`, `env fingerprint`, `git sha`)
- [ ] 모드 전환 시 study 분리 강제(`hybrid_transition` -> `tier`, 혼합 비교 금지)
- [ ] 문서 동기화(`docs/database/schema.md`, `TODO.md`)

## 4) 연관 TODO 문서(실행 대상)
- `todos/2026_02_09-issue56-cpu-gpu-parity-topk.md`
- `todos/2026_02_09-issue67-tier-universe-migration.md`
- `todos/2026_02_09-issue68-robust-wfo-ablation.md`
- `todos/2026_02_22-issue101-feat-tier1.md`

## 5) 즉시 실행 우선순위(제안)
1. `#67`: 운영 모드 정책/유니버스 일관성 마감
2. `#56`: parity 회귀 게이트 강화
3. `#68/#101`: robust score/Optuna 연구 트랙 분리 실행
4. 데이터 확장(섹터/공매도 detail/fundamental daily)은 별도 배치 안정성 검증 후 도입
