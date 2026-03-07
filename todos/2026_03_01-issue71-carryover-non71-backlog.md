# Reference Backlog: Issue #71 Carryover

> Type: `reference`
> Status: `parking-lot`
> Priority: `N/A`
> Last updated: 2026-03-07
> Related issues: `#71`, `#67`, `#68`, `#101`
> Gate status: `N/A`

## 1. Summary
- What: 이미 닫힌 `#71` 문서에서 범위를 넘어 남겨진 후속 과제를 따로 모아둔 백로그입니다.
- Why: `#71`을 닫았더라도, 후속 데이터/거버넌스 작업이 사라지지 않도록 하기 위해 필요합니다.
- Current status: active implementation 문서가 아니라 parking lot입니다.
- Next action: 실제로 손대는 항목이 생기면 반드시 owning issue 문서로 옮긴 뒤 작업합니다.

## 2. Buckets

### Data Pipeline / Schema
- pykrx source health-check guard
- P0 table DDL / index 잔여 항목
- `pipeline_batch` 운영 시나리오 확장
- `MarketCapDaily` gap audit / replay command
- 거래정지 / 비정상 거래일 파생 플래그
- `SectorClassificationHistory` 스냅샷

### Tier / Ranking
- Tier v2 shadow 실험 파이프라인
- 공매도 랭킹 반영
- `ShortSellingDaily` publication lag 보정 후 `sbv_ratio` 재검증
- `DailyStockTier` shadow 결과 검증
- CPU/GPU 공통 조회/랭킹 경로에 신규 변수 연결

### Optuna / Governance
- Optuna run script / config / manifest
- invalid trial 기준
- gate pass/fail 저장 규격
- mode 전환 시 study 분리 강제
- 문서 동기화

## 3. Related Active Docs
- [#56 parity](done_2026_02_09-issue56-cpu-gpu-parity-topk.md)
- [#67 tier universe](2026_02_09-issue67-tier-universe-migration.md)
- [#68 robust WFO](2026_02_09-issue68-robust-wfo-ablation.md)
- [#101 tier1 bias](2026_02_22-issue101-feat-tier1.md)

## 4. Reading Rule
- 이 문서 자체는 실행 SSOT가 아닙니다.
- 작업을 시작할 때는 반드시 owning issue 문서로 이관한 뒤 체크리스트를 만듭니다.
