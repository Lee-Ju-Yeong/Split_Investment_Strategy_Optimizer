# Review: Roadmap & Commercialization Checkpoint

> Type: `review`
> Status: `reference`
> Priority: `N/A`
> Last updated: 2026-03-10
> Related issues: `#98`, `#68`, `#72`, `#101`, `#67`
> Gate status: `recommendation issued (approval pending)`

## 1. Summary
- What: 기술 로드맵과 사업화 순서를 쉽게 다시 정리한 점검 문서입니다.
- Why: 지금은 "만들 수 있는 것"과 "밖에 내보내도 되는 것"이 다르기 때문에, 순서를 잘못 잡으면 리스크가 커집니다.
- Current status: 당장은 외부 출시보다 내부 검증을 먼저 해야 한다는 쪽으로 정리되었습니다.
- Next action: `TODO.md`에서 우선순위와 분기별 목표를 이 문서 기준으로 맞춥니다.

## 2. Easy Conclusion
- 최종 상태: `추가 검증`
- 지금 바로 해도 되는 것:
  - 내부 검증용 실행
  - 결과 리포트와 증적 묶음 정리
  - 성능/정합성 게이트 보강
- 아직 하면 안 되는 것:
  - 외부 유료 신호/API 제공
  - 실계좌 주문 연동
  - "출시 준비 완료"처럼 들리는 홍보 문구

## 3. Easy Rule
- `내부 검증 경로`
  - 팀 안에서 먼저 돌려보는 단계입니다.
  - 속도, 정합성, 재현성, 복구 가능성을 확인합니다.
- `외부 출시 경로`
  - 밖에 보여주거나 판매할 수 있는 단계입니다.
  - 내부 검증을 통과한 뒤에만 열 수 있습니다.

## 4. Why This Order
- `CPU=SSOT`, `Parity First`, `PIT/no-lookahead`는 단순 기술 규칙이 아니라 신뢰의 기준입니다.
- `ShortSellingDaily lag`와 데이터 정합성은 성능 개선보다 먼저 닫아야 합니다.
- `#98`은 빨라지는 것도 중요하지만, CPU와 같은 결과를 유지하는 것이 더 중요합니다.
- `#68` robust score/hard gate가 고정되어야 "왜 이 전략을 선택했는지" 설명할 수 있습니다.
- `#101`과 `GPU-native WFO v2`는 당장 운영 기본 경로가 아니라 연구 경로로 관리하는 편이 안전합니다.

## 5. What We Can Do Now
- 내부용 백테스트/최적화 실행
- 내부용 품질 게이트 점검
- `NDA` 기반의 제한적 설명 데모
  - 조건: 과거 데이터 또는 replay 결과만 사용
  - 조건: 읽기 전용 결과만 보여줌
  - 조건: 투자 조언, 자동매매, 성과 판매처럼 보이면 안 됨

## 6. What We Must Not Do Yet
- 외부 유료 파일럿 시작
- 실시간 또는 준실시간 매수/매도 신호 제공
- 실계좌 주문 연결
- 게이트 미통과 상태에서 출시 문구 사용

## 7. 12-Month Roadmap
- `2026 Q2`
  - `ShortSellingDaily publication lag` 정책 확정
  - `DailyStockPrice` 정합성 재확인
  - `#98` canonical 성능 재측정 + strict parity 재검증
  - `#68` robust score / hard gate 공식안 고정
- `2026 Q3`
  - robust WFO / ablation 실행
  - CPU certification 결과를 기본 산출물로 고정
  - 내부 검증 결과를 보기 쉬운 형태로 정리
- `2026 Q4`
  - `#72` continuity 설계/fixture 정리
  - `#101` 내부 검증용 평가 시작
  - k3s/GPU job 운영 실험은 내부 용도로만 진행
- `2027 Q1`
  - 아래 출시 기준을 모두 통과했는지 확인
  - 통과 시 제한적 canary 또는 B2B 검증 서비스 검토

## 8. Release Checklist
- `decision-level parity mismatch = 0`
- `future_reference = 0`
- `degraded_run = false`
- `CPU certification CSV + parity diff + PIT/coverage report + manifest/replay`를 한 번에 남길 수 있어야 함
- `2주 내부 검증` 안정성 통과

## 9. 30/60/90
- `D+30`
  - lag 정책 문서 확정
  - 내부 검증 경로 / 외부 출시 경계 문구 고정
- `D+60`
  - `#98` 성능 측정과 strict parity 재검증 완료
  - `#68` 공식안 고정
- `D+90`
  - 내부 검증 실행 시작
  - 외부에 보여줄 수 있는 범위를 다시 판정

## 10. Reading Rule
- 이 문서는 큰 방향을 쉽게 이해하기 위한 요약 문서입니다.
- 실제 체크리스트, 로그, 증적은 `#98`, `#68`, `lag` 문서를 먼저 봅니다.
