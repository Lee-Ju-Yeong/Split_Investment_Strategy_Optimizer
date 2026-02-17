# chore(legacy): 레거시 코드 전수조사 및 단계적 제거 거버넌스 (Issue #97)
- 이슈 주소: `https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/97`
- 작성일: 2026-02-17
- 목적: 레거시/우회/호환 로직을 전수조사하고, 사용자 승인 게이트를 거쳐 안전하게 정리

## 1. 배경
- 실행 경로별로 과거 호환 로직, 과도한 fallback, 중복 래퍼가 일부 잔존
- 성능 저하/디버깅 난이도 증가/정합성 리스크를 유발할 수 있음
- 무차별 삭제가 아니라, 근거 기반 분류와 승인 절차가 필요

## 2. 범위
- `src/` 실행 경로: 백테스트/Parity/데이터 배치/오케스트레이터
- 데이터 접근 계층: `DataHandler` 및 DB 조회 경로
- 설정/플래그: `config/`, CLI 옵션, 환경변수 분기
- wrapper/compat/fallback 로직
- 문서와 실제 동작 불일치 구간

## 3. 조사 산출물
- 레거시 인벤토리 표(필수 컬럼):
  - `id`, `location(file:line)`, `category`, `current_behavior`, `risk`, `removal_cost`, `recommendation(유지/축소/제거)`, `evidence`
- 우선순위:
  - Quick Win(저위험 즉시 가능)
  - Medium(검증 필요)
  - High Risk(아키텍처 영향)

## 4. 사용자 확인 게이트 (필수)
- Gate A: 전수조사 인벤토리 승인
- Gate B: 제거 대상 목록 승인(항목별)
- Gate C: 배포 전 최종 승인

## 5. 실행 체크리스트
- [ ] 레거시 인벤토리 1차 작성
- [ ] Gate A 승인 완료
- [ ] 제거/축소 대상 확정안 작성
- [ ] Gate B 승인 완료
- [ ] 승인 범위만 PR로 반영
- [ ] Gate C 승인 완료
- [ ] `TODO.md`/`todos/`/이슈 코멘트 동기화

## 6. 완료 기준
- [ ] 전수조사 인벤토리 완료(근거 포함)
- [ ] 유지/축소/제거 분류 확정
- [ ] 사용자 승인 게이트 A/B/C 기록 완료
- [ ] 승인 범위 제거 PR 병합 완료
