# GPU-native WFO v2 Design Note

> Type: `research`
> Status: `draft`
> Priority: `P2`
> Last updated: 2026-03-07
> Related issues: `#56`, `#67`, `#98`
> Gate status: `shadow only`

## 1. One-Page Summary
- What: CPU를 최대한 흉내 내는 GPU 경로가 아니라, WFO 자체를 GPU에 맞게 다시 설계하는 별도 연구 문서입니다.
- Why: 진짜 속도 개선은 fold별 재로딩 제거, ranked candidate precompute, device-side metric reduction 같은 구조적 변화에서 나올 가능성이 큽니다.
- Current status: 설계 초안 단계입니다. 공식 경로 승격 대상이 아닙니다.
- Next action: `fold manifest`, `super-range tensor cache`, `shadow acceptance` 같은 연구 전제부터 고정합니다.

## 2. Important Warning
- 이 문서는 현재 공식 경로를 대체하지 않습니다.
- official path는 여전히 CPU/hybrid 기준입니다.
- `#56 -> #67 -> #97 -> #98` 선행 조건 없이 승격하지 않습니다.

## 3. What This Note Is For
- GPU-native WFO가 실제로 가치가 있는지 판단
- 어떤 검증 계약이 CPU parity 대신 필요한지 정리
- Research -> Shadow -> Official 승격 단계를 설계

## 4. Reading Guide
- 지금 공식 경로를 고치는 중이라면 이 문서는 바로 읽지 않아도 됩니다.
- `#56/#67/#98`가 정리된 뒤 “별도 엔진 프로젝트”를 검토할 때 읽으면 됩니다.

## 5. Detailed Design Notes
### 0. 이 문서가 답하려는 질문
- 질문 1: `CPU에 최대한 맞추는 GPU`가 아니라, `GPU에 맞게 처음부터 다시 설계한 WFO 엔진`이 필요한가?
- 질문 2: 그 엔진이 지금 하드웨어(`Ryzen 7 1700`, `32GB RAM`, `RTX 5060`)에서 실제로 더 빠를 가능성이 있는가?
- 질문 3: CPU parity 없이도 믿을 수 있는 결과를 내는지 무엇으로 검증할 것인가?

## 1. 한 줄 결론
- **방향은 맞다.**
- 다만 `지금 당장 공식 경로로 승격`하는 안은 아니다.
- `GPU-native WFO v2`는 **기존 GPU 경로의 연장선**이 아니라 **새 엔진 프로젝트**로 다뤄야 한다.

## 2. 왜 새 엔진이 필요한가

### 2-1. 현재 GPU 경로의 한계
- 현재 GPU 경로는 여전히 `CPU 결과와 맞추는 것`을 중요한 목표로 둔다.
- 이 때문에 일부 hot path가 GPU에 최적화되기보다 parity 유지 쪽으로 설계돼 있다.
- 현재 병목도 GPU 커널 계산 자체보다 아래 항목에 많이 남아 있다.
  - fold마다 DB read와 tensor build를 반복
  - host-driven month/day loop
  - 일별 후보 metric 재조회
  - `num_combinations x num_days` 곡선을 host로 복사
  - OOM 발생 후 작은 batch 크기로 고착

### 2-2. 왜 발상의 전환이 필요한가
- `GPU를 CPU의 복제 엔진`으로만 보면, GPU가 잘하는 방식으로 아키텍처를 바꾸기 어렵다.
- 진짜 속도 혁신은 보통 아래에서 나온다.
  - 데이터 적재를 한 번만 하고 fold마다 재사용
  - 후보군 정렬을 매일 다시 하지 않고 전처리해서 재사용
  - 일별 순차성은 유지하되 시뮬레이션 축만 크게 병렬화
  - 메트릭 계산을 host가 아니라 device에서 끝냄

### 2-3. 그러나 바로 갈아타면 안 되는 이유
- 현재 repo의 공식 규칙은 아직 `CPU=Source of Truth`다.
- `GPU-native v2`를 바로 기본 경로로 올리면, 빨라져도 “왜 이 결과를 믿어도 되는가?”를 설명할 공통 기준이 없다.
- 그래서 이 엔진은 먼저 **Research -> Shadow -> Official** 3단계로 밟아야 한다.

## 3. 초심자용 개념 정리
- `CPU official path`
  - 현재 기준 정답 엔진
  - 운영 결과를 설명할 때 가장 보수적인 기준
- `Hybrid path`
  - GPU가 많은 후보를 찾고 CPU가 상위 후보를 다시 검사
  - 과도기 안전장치
- `GPU-native WFO v2`
  - CPU에 맞추는 것이 1순위가 아닌 별도 GPU 전용 엔진
  - 목표는 `탐색 throughput`과 `WFO 전체 처리량`을 근본적으로 높이는 것

## 4. 이 엔진에서 바뀌는 것과 안 바뀌는 것

### 4-1. 바뀌는 것
- fold마다 다시 만드는 tensor/DB read를 줄인다.
- 후보군 정렬을 런타임이 아니라 전처리 단계로 많이 이동한다.
- full equity curve 전체를 CPU로 복사하지 않고 GPU에서 metric을 줄여서 반환한다.
- CPU는 hot path 계산이 아니라 제어/감사 역할로 축소한다.

### 4-2. 안 바뀌는 것
- WFO 자체는 그대로 유지된다.
  - IS에서 찾고
  - OOS에서 검증하고
  - fold를 반복하는 구조는 동일
- 시간 순서 제약은 유지된다.
  - `sell -> new_entry -> add_buy -> mark_to_market`
- PIT/no-lookahead 금지는 유지된다.
- 운영 승격 전까지 CPU는 audit/reference 용도로 남는다.

## 5. 목표 하드웨어 가정
- CPU: `AMD Ryzen 7 1700` (8C/16T)
- RAM: `32GB`
- GPU: `RTX 5060`

### 5-1. 이 하드웨어에서 기대할 수 있는 점
- 대규모 파라미터 탐색은 CPU-only보다 GPU 쪽이 훨씬 유리할 가능성이 크다.
- 다만 Ryzen 7 1700은 host orchestration이 무거우면 병목이 되기 쉽다.
- RAM 32GB는 `super-range tensor cache`를 위한 여지는 있지만, 전체 범위를 무조건 크게 올리면 OOM/스왑/장시간 run 변동성이 생길 수 있다.

### 5-2. 그래서 설계 원칙이 달라져야 한다
- GPU는 계산면(data plane)을 담당한다.
- CPU는 제어면(control plane)을 담당한다.
- 메모리는 “무조건 전부 preload”가 아니라 “재사용 가치가 큰 범위만 cache” 관점으로 본다.

## 6. 제안 아키텍처

### 6-1. 상위 구조
```text
fold manifest
  -> super-range tensor cache
  -> GPU proposal engine
  -> shortlist / audit artifact
  -> shadow diff / optional CPU certify
  -> OOS stitcher
  -> final report
```

### 6-2. 컴포넌트 설명

#### A. `fold manifest`
- 각 WFO run이 어떤 데이터와 어떤 규칙으로 돌아가는지 기록하는 파일
- 최소 포함 항목:
  - 기간(`start_date`, `end_date`)
  - trading date set 또는 hash
  - `price_basis`
  - `universe_mode`
  - `candidate_source_mode`
  - `tier_hysteresis_mode`
  - 전략 파라미터
  - ranking policy version
  - seed / scenario

#### B. `super-range tensor cache`
- 전체 WFO 기간의 union 범위를 한 번만 적재하고, fold에서는 slice view만 사용한다.
- 예시:
  - `price[day, ticker, field]`
  - `tier[day, ticker]`
  - `pit_mask[day, ticker]`
  - `rank_features[day, ticker, feature]`
- 핵심은 “fold마다 다시 로딩하지 않기”다.

#### C. `daily ranked candidate table`
- 후보 종목을 매일 런타임에서 다시 정렬하지 않고, fold 시작 전에 `day x topM` 형태로 정리한다.
- 런타임에서는 이 테이블을 순서대로 보며 skip/apply만 한다.
- 주의:
  - 이 방식은 빠르지만, 현재 strict parity의 `active subset 재정렬`과 의미가 달라질 수 있다.
  - 그래서 초반에는 CPU finalist certification 또는 shadow diff가 필요하다.

#### D. `GPU proposal engine`
- IS 구간 전체를 대상으로 `theta x scenario x perturbation` 타일 단위로 실행한다.
- 시간축(day)은 순차 유지
- 시뮬레이션 축(parameter/scenario)은 최대한 벡터화
- host로 넘기는 것은 전체 곡선이 아니라 shortlist와 audit metric 중심으로 최소화한다.

#### E. `OOS stitcher`
- OOS는 fold 간 cash chaining이 있으므로 완전 병렬화하지 않는다.
- 대신 다음 fold 데이터 prefetch만 겹친다.

### 6-3. CPU 역할 재정의
- CPU는 아래만 맡는다.
  - fold 생성
  - run ledger 기록
  - optional finalist certification
  - OOS stitching
  - shadow diff
- CPU가 더 이상 일별 후보 계산 hot path를 돌지 않는 것이 핵심이다.

## 7. 이 엔진에서도 반드시 지켜야 하는 데이터/PIT 계약

### 7-1. 가장 중요한 규칙
- `signal_date`는 항상 `trade_date`의 직전 거래일이어야 한다.
- 모든 의사결정 입력은 `row_date <= signal_date`를 만족해야 한다.
- `row_date > signal_date`가 한 건이라도 나오면 즉시 실패해야 한다.

### 7-2. 유니버스/티어 계약
- 유니버스는 `latest snapshot <= signal_date`를 우선 사용한다.
- snapshot이 없을 때만 `history active(as-of)` fallback을 쓴다.
- legacy weekly fallback을 기본 경로로 두면 안 된다.
- Tier는 `latest DailyStockTier <= signal_date`만 허용한다.
- 시작일 이전 seed row를 보존한 뒤 forward-fill 해야 한다.

### 7-3. 후보 메트릭 계약
- ATR 등 후보 메트릭은 ticker별 `latest <= signal_date` 1건만 읽어야 한다.
- future row는 절대 포함되면 안 된다.
- 정렬은 deterministic해야 하며 최종 tie-breaker가 필요하다.

### 7-4. degraded mode 규칙
- `strict_pit`가 기본이다.
- `candidate lookup error skip` 같은 degraded mode는 연구용으로만 허용하고, 공식 run이나 승격 증적으로 쓰면 안 된다.

## 8. CPU parity 대신 필요한 새 검증 계약

### 8-1. 왜 새 계약이 필요한가
- `GPU-native v2`는 CPU와 똑같은 내부 의미를 일부러 보장하지 않을 수 있다.
- 그러면 기존의 `CPU와 100% 같은가?`만으로는 pass/fail을 정할 수 없다.
- 대신 아래 5개 계약이 필요하다.

### 8-2. 최소 검증 계약 5종

#### 1. `PIT / no-lookahead`
- 미래 데이터 참조 0건
- snapshot/tier/history 모두 as-of 규칙 준수

#### 2. `deterministic replay`
- 같은 manifest
- 같은 seed
- 같은 input snapshot
- 같은 code version
- 이 네 가지가 같으면 결과가 재현돼야 한다.

#### 3. `accounting invariants`
- 현금 음수 drift 금지
- 포지션 수량/평균단가 보존
- 청산 후 잔여 수량 0
- 수익 계산 규칙 일관성

#### 4. `execution invariants`
- `sell -> new_entry -> add_buy` 순서 보존
- 호가/rounding/수수료/세금 규칙 일관성
- 종료 시 정리 규칙 일관성

#### 5. `GPU-only WFO acceptance`
- WFO 결과가 최소 acceptance를 만족해야 한다.
- 예시:
  - fold pass rate
  - OOS/IS ratio
  - MDD upper bound
  - shortlist stability
  - 주변 파라미터 민감도

## 9. 승격 단계

### 9-1. Stage 1. `Research`
- 목적: 실험
- 권한: 없음
- 특징:
  - 공식 파라미터 선택권 없음
  - 성능, 재현성, invariant 위반 탐지 중심

#### Research 진입 조건
- v1/hybrid와 코드 경로가 분리돼 있을 것
- 데이터 규칙과 체결 규칙 문서화 완료
- manifest/replay artifact 설계 완료

#### Research 종료 조건
- PIT 위반 0
- deterministic replay 통과
- accounting invariant 통과
- 최소 GPU-only WFO acceptance 충족
- 기준 하드웨어에서 의미 있는 성능 증적 확보

### 9-2. Stage 2. `Shadow`
- 목적: 공식 경로와 병렬 비교
- 권한: 결과 기록만, 실제 채택 금지
- 특징:
  - 동일 기간/동일 fold에서 current official path와 나란히 실행
  - diff를 축적

#### Shadow 종료 조건
- 연속 run에서 치명적 silent failure 0
- diff 리포트가 설명 가능한 범위 내
- OOM/backoff/retry 안정성 허용 범위 내
- 운영자가 diff artifact만으로 원인 분석 가능

### 9-3. Stage 3. `Official`
- 목적: 기본 선택 엔진 승격
- 권한: 공식 파라미터 제안 가능
- 특징:
  - parity gate는 중심이 아니라 audit 용도로 후퇴
  - 하지만 CPU reference/audit는 완전히 제거하지 않음

## 10. 성능 주장을 하기 전에 반드시 계측할 것

### 10-1. 반드시 남길 지표
- wall-time
- GPU util
- CPU util
- VRAM peak
- host RAM peak
- OOM retry count
- batch recovery 여부
- degraded mode 진입 여부
- H2D/D2H bytes 또는 대체 가능한 전송 지표

### 10-2. 먼저 검증할 병목 가설
- fold마다 DB read/tensor build가 가장 큰가?
- host day loop가 가장 큰가?
- 후보 랭킹/조회가 가장 큰가?
- full curve host copy가 가장 큰가?
- OOM fallback 고착이 가장 큰가?

### 10-3. 금지할 약속
- “무조건 더 빠르다”
- “기존 공식 경로를 곧 대체한다”
- “CPU와 직접 비교 가능한 같은 전략이다”
- “OOM/안정성 문제까지 자동 해결된다”

## 11. 실패 조건(Kill Criteria)
- 동일 fold/기간 A/B에서 wall-time 개선이 작거나 음수
- GPU util도 낮고 CPU 병목도 그대로
- `future reference > 0`
- accounting invariant 위반 1건 이상
- manifest 없이 재현 불가
- OOM/backoff로 장시간 WFO 재현성이 무너짐
- 경로가 늘어 유지보수 비용이 성능 이익보다 커짐

## 12. 단계별 구현 제안

### Phase 1. `fold manifest + super-range tensor cache`
- 목표:
  - fold별 중복 DB/tensor build 제거
- 아직 현행 엔진 재사용
- 가장 안전한 첫 단계

### Phase 2. `device-side metric reduction`
- 목표:
  - full equity curve host copy 제거
- shortlist와 audit metric만 host로 전달

### Phase 3. `daily ranked candidate table`
- 목표:
  - 후보 선별/정렬을 런타임이 아니라 전처리 단계로 이동
- 주의:
  - 의미론 drift 가능성이 있으므로 shadow diff 필수

### Phase 4. `persistent/fused kernels`
- 목표:
  - host day loop 최소화
- 이 단계부터는 진짜 GPU-native 성격이 강해진다.

### Phase 5. `shadow promotion`
- 목표:
  - official path와 병렬 비교
- 조건 충족 시에만 승격 판단

## 13. 이 문서 기준의 현재 판정
- `Direction`: Go
- `Implementation now`: Limited Go
- `Official promotion now`: No-Go
- 요약:
  - 방향은 맞다.
  - 문서화와 계측 없이 바로 rewrite 하는 것은 틀렸다.
  - 먼저 `새 검증 계약`과 `작은 프로토타입`으로 시작해야 한다.

## 14. 바로 실행할 실무 액션
- [ ] `GPU-native WFO v2`를 별도 연구 트랙으로 TODO에 등록
- [ ] Stage 1 범위 문서화: `manifest`, `tensor cache`, `telemetry`
- [ ] 승격 계약 문서화: `PIT`, `determinism`, `accounting`, `execution`, `WFO acceptance`
- [ ] 프로토타입 계측 범위 확정:
  - fold tensor rebuild 시간
  - host day loop 시간
  - ranking/candidate time
  - host copy time
- [ ] small-scale prototype 작성 후 A/B 측정
- [ ] Shadow 전까지 official path 변경 금지

## 15. 현재 공식 경로와의 관계
- 현재 공식 경로:
  - CPU/hybrid
- `GPU-native v2`:
  - 별도 연구 트랙
- 원칙:
  - 둘을 같은 PR에 섞지 않는다.
- `#98` throughput 리팩토링과 `v2 신규 엔진`을 같은 작업으로 취급하지 않는다.
- `#56` parity gate는 v1/hybrid release 기준으로 유지한다.
- v2는 새 validation contract로 승격한다.

## 16. Backlog Freeze / Reopen 체크리스트

### 16-1. 지금 당장 재개하지 않는 이유
- 현재 상태는 `Implementation now: Limited Go`, `Official promotion now: No-Go`다.
- 즉, 설계 문서는 만들었지만 바로 구현을 크게 열 단계는 아니다.
- 당분간은 `frozen research track`으로 두고, 하이브리드 공식 경로 안정화가 우선이다.

### 16-2. Reopen 전에 반드시 닫혀 있어야 하는 선행 조건
- `#56` release parity가 backlog 상태를 벗어날 것
- `#67` PIT default path + coverage gate가 backlog 상태를 벗어날 것
- `#97` strict-only 승인/관찰 게이트가 backlog 상태를 벗어날 것
- `#98`은 적어도 low-risk throughput 계측과 병목 분해가 확보될 것
- v2 작업은 반드시 별도 연구 이슈/별도 PR로 분리할 것

### 16-3. Reopen 최소 조건
- 범위는 `Research` 단계로만 재개한다.
- 1차 범위는 아래까지만 허용한다.
  - `fold manifest`
  - `super-range tensor cache`
  - `telemetry`
- 아래 계약이 문서로 고정돼 있어야 한다.
  - `PIT/no-lookahead`
  - `deterministic replay`
  - `accounting invariants`
  - `execution invariants`
  - `GPU-only WFO acceptance`

### 16-4. Reopen에 필요한 증적
- 동일 manifest 기준 small-scale A/B에서 `end-to-end fold wall-time` 개선이 확인될 것
- `GPU util`, `CPU util`, `VRAM peak`, `host RAM peak`, `OOM retry`, `batch recovery`가 함께 기록될 것
- `DB/tensor build`, `host day loop`, `candidate ranking`, `GPU kernel`, `H2D/D2H copy` 시간 분해가 있을 것
- 동일 `manifest/seed/input/code`에서 replay hash가 재현될 것
- `future reference=0`, invariant 위반 `0`, silent failure `0`일 것
- WFO acceptance 숫자 기준이 잠겨 있을 것
  - `median(OOS/IS) >= 0.60`
  - `fold_pass_rate >= 70%`
  - `OOS_MDD_p95 <= 25%`
  - `deterministic baseline / seeded_stress / jackknife` 통과

### 16-5. Reopen 금지 신호
- kernel 시간만 빨라지고 end-to-end wall-time이 안 좋아지는 경우
- OOM 한 번 뒤 줄어든 batch size가 계속 고착되는 경우
- full equity curve를 계속 host로 복사하는데도 `GPU-native`라고 부르는 경우
- manifest 없이 ad-hoc 로그만으로 판단하는 경우
- degraded mode, legacy fallback, random-only 결과를 근거로 삼는 경우

### 16-6. 지금의 권고
- 당분간 `v2`는 frozen backlog로 둔다.
- 실제 개발은 `CPU/hybrid` 경로에서 진행한다.
- v2는 위 체크리스트를 만족할 때만 다시 꺼낸다.

## 17. 후속 문서/이슈 연결
- `TODO.md`
- `todos/2026_03_06-multi-agent-performance-stability-review.md`
- `todos/2026_02_09-issue56-cpu-gpu-parity-topk.md`
- `todos/2026_02_17-issue98-gpu-throughput-refactor.md`
