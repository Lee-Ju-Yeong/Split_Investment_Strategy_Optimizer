# feat: 결정론적 Tier1 선정 편향 완화 및 분포 기반 강건 최적화 프레임 도입 (Issue #101)
(현재 파일 이름은 YYYY_MM_DD-issue<이슈번호>-<issue_name_only_english>.md 로 지정)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/101`
- 목표: 특정 종목 고착이 아닌 `조건(theta)`의 일반화 성능 분포를 기준으로 파라미터를 선택하는 최적화 프레임으로 전환
- 핵심 아이디어: 단일 백테스트 최고점 대신 `theta x scenario(omega) x fold` 분포 점수 + hard gate 기반 승격
- 전제조건: PIT/no-lookahead 유지, `candidate_source_mode=tier` 유지, CPU SSOT/GPU batch 원칙 유지

## 1. 배경(현재 이슈의 대략적인 이전 맥락)
- 현재 신규 진입 정렬은 결정론(`ATR -> market_cap -> ticker`)이며, 동일 조건에서 특정 종목이 반복 선택될 수 있음
- 이 방식은 재현성은 높지만 \"조건 자체의 강건성\"보다 \"종목 포함/제외\"에 성과가 과민해질 위험이 있음
- 최근 운영 지표(`empty_entry_day_rate`, `tier1_coverage`, `tier2_fallback_rate`)가 추가되어 강건성 gate 입력으로 활용 가능
- `ShortSellingDaily`는 현재 `short_balance_value` 중심 커버리지여서 공매도 신호는 `sbv_ratio = short_balance_value / market_cap` 단일 경로를 유지 중
#### 1-1. 기타 참고해야할 원칙/구현의도
- 고정해야 하는 원칙
  - `candidate_source_mode=tier` 유지
  - `signal_date=T-1` 및 PIT/as-of 조회 규칙 유지
  - execution/fee/tick/rounding 규칙은 탐색 대상에서 제외
- 운영 안전장치
  - `shadow -> gated -> default` 단계 전환
  - parity mismatch, coverage/entry 지표 악화 시 즉시 rollback 가능해야 함
---

## 2. 요구사항(구현하고자 하는 필요한 기능)
### 2-1. 분포 기반 평가 단위 도입
- 파라미터 단위를 단일 실행 점수가 아닌 `theta x scenario(omega) x fold` 분포 점수로 평가
- `theta`: 전략/티어 조건 파라미터 세트
- `omega`: PIT-safe 불확실성 시나리오(tie-break jitter, sub-universe sampling, cost shock 등)
- `fold`: WFO 기간 분할
### 2-2. robust score + hard gate 기반 선발
- 평균 성과만이 아니라 하방 리스크/안정성/운영 가능성을 함께 반영한 robust score 계산
- 예시 hard gate(초안)
  - `median(OOS/IS) >= 0.60`
  - `fold_pass_rate >= 70%`
  - `OOS_MDD_p95 <= 25%`
  - `P95(empty_entry_day_rate) <= 0.20`
  - `median(tier1_coverage) >= 0.55`
#### 2-2-1. 구현 시 신경써야 하는 중요한 부분
- 시나리오 주입이 lookahead/PIT 위반을 만들지 않도록 보장해야 함
- CPU/GPU 경로에서 동일 `theta, omega` 입력에 대해 parity strict 기준 유지 필요
- 운영 전환 전 최소 2주 shadow 증적 필요
#### 2-2-2. 우선 참조 파일
- `src/backtest/cpu/strategy.py`
- `src/backtest/gpu/engine.py`
- `src/backtest/cpu/backtester.py`
- `src/analysis/walk_forward_analyzer.py`
- `src/optimization/gpu/parameter_simulation.py`
- `TODO.md`
- `todos/done_2026_02_08-issue71-pykrx-tier-v2-data-roadmap.md`

---

(하위 부분은 사람이 작성하는게 아니라 AI 가 작성하는 부분)

# AI 결과

## 3. (AI가 확인한) 기존 코드/구현의 핵심내용들/의도들
- `src/backtest/cpu/strategy.py`
  - 신규 진입 후보는 `get_candidates_with_tier_fallback(_pit)` 결과를 사용하고,
    보유/쿨다운 제외 후 `ATR -> market_cap -> ticker`로 결정론 정렬됨.
  - `last_entry_context`에 후보군 카운트(`raw/active/ranked`)와 진입 소스가 기록됨.
- `src/data_handler.py`
  - 후보군 기본 정책은 `tier=1 우선`, 비어 있으면 `tier<=2 fallback`.
  - PIT 경로는 `get_candidates_with_tier_fallback_pit_gated`에서 coverage gate를 함께 적용 가능.
- `src/backtest/cpu/backtester.py`
  - 운영 지표 `empty_entry_day_rate`, `tier1_coverage`, `tier2_fallback_rate`가 이미 집계됨.
  - 즉, #101 hard gate 입력 데이터는 기존 런타임 경로에서 바로 재사용 가능.
- `src/pipeline/daily_stock_tier_batch.py`
  - Tier 테이블에 `cheap_score` 계열(`pbr_discount`, `per_discount`, `div_premium`, `cheap_score`, `cheap_score_version`, `cheap_score_confidence`) 저장이 추가됨.
  - `div_yield <= 0`인 경우 Tier1 강등 규칙이 적용됨.

---

## 4. 생각한 수정 방안들 (ai 가 생각하기에) 구현에 필요한 핵심 변경점
### 4-1. 방안 A: 결정론 유지 + 후보군 편향 완화(저비용)
- 핵심 변경점
  - `src/backtest/cpu/strategy.py`: 정렬 이전에 시총 구간 quota(대/중/소형) 또는 상위 cap 제한 적용
  - `src/backtest/gpu/engine.py`: 동일 규칙 parity 반영
  - `src/backtest/cpu/backtester.py`: quota 적용 전/후 지표 비교 로깅
- 장점: 구현/연산 부담이 작고 운영 반영이 빠름
- 단점: 여전히 단일 경로 평가라 분포 강건성 문제를 근본 해결하지 못함

### 4-2. 방안 B: 시나리오(omega) 러너 도입 + 분포 스코어 계산(중간비용)
- 핵심 변경점
  - `src/analysis/walk_forward_analyzer.py`: `theta x omega x fold` 평가 루프 추가
  - `src/optimization/gpu/parameter_simulation.py`: seed 고정 시나리오 배치 실행
  - 신규 결과 테이블(또는 결과 파일): `theta_id, omega_id, fold_id, metrics` 저장
- 장점: 사용자 요구인 \"조건의 일반적 성능 분포\"를 직접 측정 가능
- 단점: 계산량 증가, 결과 스키마/리포트 설계가 필요

### 4-3. 방안 C: hard gate 우선 도입 + 단계 승격(shadow->gated->default) (권고)
- 핵심 변경점
  - `src/analysis/walk_forward_analyzer.py`: robust score + hard gate 계산기 추가
  - `src/backtest/cpu/backtester.py`: 기존 운영 지표를 gate 입력으로 연결
  - `docs/` runbook: rollback trigger/승격 조건 문서화
- 장점: 실무 적용성이 높고, 분포평가를 운영 의사결정과 직접 연결 가능
- 단점: gate 임계값 튜닝과 shadow 관찰 기간이 필요

### 4-4. 공통 선행 과제(필수)
- CPU/GPU parity strict 유지(결정/체결 결과 mismatch=0)
- PIT-safe 보장(`signal_date=T-1`, as-of join 고정)
- 계산비 제약을 고려해 시나리오 수(`omega`)와 fold 수 상한을 먼저 고정

---

## 5. 최종 결정된 수정 방안 (AI 가 자동 진행하면 안되고 **무조건**/**MUST** 사람에게 선택/결정을 맡겨야 한다)
- 사용자 최종 선택 대기
- 후보: `A(저비용 완화)`, `B(분포평가 중심)`, `C(hard gate 중심 운영형)` 또는 `A+B/C` 혼합
- 승인 전에는 구현 코드 변경을 진행하지 않음

---

## 6. 코드 수정 요약
- [x] 이슈 #101 상세 TODO 생성 및 요구사항/대안 정리
- [ ] 최종 방안 선택 후 구현 체크리스트 확정
- [ ] 구현/검증/리포트 반영

---

## 7. 문제 해결에 참고
- issue: https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/101
- issue: https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/71
- file: `TODO.md`
- file: `todos/done_2026_02_08-issue71-pykrx-tier-v2-data-roadmap.md`
