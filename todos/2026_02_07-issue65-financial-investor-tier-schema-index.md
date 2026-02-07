# feat(db): FinancialData, InvestorTradingTrend, DailyStockTier 스키마 및 인덱스 추가
(현재 파일 이름은 YYYY_MM_DD-issue<이슈번호>-<issue_name_only_english>.md 로 지정)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/65`
- 가격/지표 중심 구조에 재무/수급/티어 데이터를 분리 저장할 수 있도록 DB 스키마를 확장
- 백필/일배치에서 재사용할 조인 키/조회 인덱스를 준비

## 1. 배경(현재 이슈의 대략적인 이전 맥락)
- `DailyStockPrice`/`CalculatedIndicators`는 일봉 및 파생지표 중심으로 구성되어 있고, 재무/수급 데이터 저장소가 분리되어 있지 않음
- `#64`에서 PIT 가드와 T-1 신호 기준을 적용했으므로, 다음 단계는 확장 데이터 저장 스키마를 안전하게 준비하는 것
- 배치 파이프라인(#66), DataHandler 조인 확장(#67), 멀티팩터 랭킹(#68)의 선행조건은 스키마/인덱스 확정임

#### 1-1. 기타 참고해야할 로직/원칙/세부사항/구현의도1
- `create_tables()`는 반복 호출될 수 있으므로 인덱스 생성은 멱등성이 필요함
- MySQL 버전 호환성을 고려해 `CREATE INDEX IF NOT EXISTS` 대신 `INFORMATION_SCHEMA` 확인 후 생성 방식 사용

---

## 2. 요구사항(구현하고자 하는 필요한 기능)
### 2-1. 재무 데이터 저장 테이블
- `FinancialData(stock_code, date)` 복합 PK
- `per/pbr/eps/bps/dps/div_yield/roe/source/updated_at` 컬럼 포함
- 날짜 기준 조인 성능 확보를 위한 `(date, stock_code)` 인덱스 제공

### 2-2. 수급 데이터 저장 테이블
- `InvestorTradingTrend(stock_code, date)` 복합 PK
- `individual/foreigner/institution/total_net_buy` 컬럼 포함
- 날짜 조회 및 랭킹 조회를 위한 인덱스 제공

### 2-3. Tier 사전계산 결과 테이블
- `DailyStockTier(date, stock_code)` 복합 PK
- `tier/reason/liquidity_20d_avg_value/computed_at` 컬럼 포함
- 백테스트 조회 패턴(`date,tier`) 최적화 인덱스 제공

#### 2-3-1. (사람이 생각하기에) 우선적으로 참조할 파일
- `src/db_setup.py`
- `tests/test_db_setup.py`
- `TODO.md`

---

# AI 결과

## 3. (AI가 확인한) 기존 코드/구현의 핵심내용들/의도들
- `src/db_setup.py`의 `create_tables()`는 기존 핵심 테이블을 한 번에 생성하는 단일 진입점
- 기존 구조에는 재무/수급/티어 테이블이 없고, 인덱스 생성 멱등 로직도 별도로 존재하지 않음
- 테스트는 DB setup 직접 검증이 부족해, 스키마 확장 시 회귀를 잡기 어려운 상태

---

## 4. 생각한 수정 방안들 (ai 가 생각하기에) 구현에 필요한 핵심 변경점
### 4-1. 방안 A: 단순 `CREATE TABLE` + `CREATE INDEX` 추가
- 장점: 구현 단순
- 단점: `create_tables()` 재호출 시 인덱스 중복 오류 발생 가능

### 4-2. 방안 B: `INFORMATION_SCHEMA` 조회 후 인덱스 생성 (채택)
- 장점: 멱등성 확보, MySQL 호환성 높음
- 단점: 쿼리 수 증가

### 4-3. 방안 C: 애플리케이션 마이그레이션 도구 도입
- 장점: 장기적으로 스키마 관리 용이
- 단점: 현재 레포 구조에서 도입 범위가 큼(이번 이슈 범위 초과)

---

## 5. 최종 결정된 수정 방안
- 방안 B 채택: `create_tables()` 내부에서 `ensure_index()`를 통해 인덱스 존재 여부를 확인하고 없을 때만 생성

### 5-1. 최종 결정 이유1
- 반복 실행 안전성이 확보되어 로컬/CI/테스트 환경에서 재실행 실패를 방지
- 현재 아키텍처 변경 없이 이슈 #65 요구사항을 충족

### 5-2. 최종 결정 이유2
- 후속 이슈(#66~#68)에서 바로 사용할 수 있는 키/인덱스를 최소 변경으로 제공

---

## 6. 코드 수정 요약
- [x] `src/db_setup.py`에서 `FinancialData`, `InvestorTradingTrend`, `DailyStockTier` 테이블 생성 SQL 추가
- [x] `src/db_setup.py`에 `ensure_index()` 헬퍼 추가 및 인덱스 멱등 생성 적용
- [x] `tests/test_db_setup.py` 신규 추가: 신규 테이블/인덱스 생성 SQL 포함 여부 검증
- [x] `TODO.md`에 이슈 #65 상세 TODO 링크 추가

---

## 7. 문제 해결에 참고
- issue: https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/65
- issue: https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/64
- commit: a06bd04
