# 이슈 #94: 전략/파라미터 기준서 문서화

## 목표

- 종목 선정 기준, Tier 규칙, 매수/매도 기준, 파라미터 범위, 승격 게이트를 한 문서에서 확인 가능하게 정리
- 논의된 의사결정을 `TODO.md`와 `docs/`에 동기화

## 배경

- 전략 규칙은 `docs/MAGIC_SPLIT_STRATEGY_PRINCIPLES.md`에 상세히 있으나,
  최적화/승격 관점의 실행 기준(게이트, 시나리오, parity 범위)을 한눈에 보기 어려움
- 2026-02-16 사용자 합의:
  - 데이터 시작일: `2013-11-20+`
  - 목적함수: `Calmar` 통일
  - 승격 게이트: `OOS MDD p95 <= 30%`
  - parity 검증 범위: `Top-100`
  - 후보군: `tier` 고정
  - 초기 범위: shared 파라미터 우선

## 산출물

1. `docs/strategy/strategy_parameter_playbook.md` 신설
2. `TODO.md`에 #94 트랙/합의 노트 반영
3. `README.md` 문서 맵에 신규 기준서 추가
4. `docs/MAGIC_SPLIT_STRATEGY_PRINCIPLES.md`에서 playbook 링크 제공

## 수용 기준 (Acceptance Criteria)

- [ ] 신규 기준서에서 아래 항목을 모두 확인 가능
  - [ ] 종목 선정/Tier fallback 규칙
  - [ ] 매수/매도 핵심 트리거 및 실행 순서
  - [ ] 탐색 대상/고정 파라미터 범위
  - [ ] 비용 시나리오(S0/S1/S2)
  - [ ] 승격 게이트(특히 `MDD p95 <= 30%`, Top-100 parity)
- [ ] `TODO.md`와 문서 간 링크가 일관됨
- [ ] 기존 전략/체결 로직 변경 없음 (문서화 전용)

## 비고

- 본 이슈는 문서화/의사결정 정렬 이슈이며, 코드 동작 변경 이슈가 아님
- 향후 #68/#56 구현 진행 시 본 문서를 기준으로 수치/용어를 동기화
