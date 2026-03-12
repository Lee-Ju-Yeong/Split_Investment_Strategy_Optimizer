# Reference Note: 로컬 DB 없이 진행 가능한 작업

> Type: `reference`
> Status: `reference`
> Priority: `N/A`
> Last updated: 2026-03-07
> Related issues: `#56`, `#68`, `#69`
> Gate status: `N/A`

## 1. Summary
- What: 로컬 MySQL 없이도 바로 진행 가능한 작업 범위를 빠르게 확인하기 위한 메모입니다.
- Why: 노트북 환경이나 임시 개발 환경에서는 DB가 없어도 할 수 있는 작업과 없는 작업을 구분해야 합니다.
- Current status: 참고 문서입니다. 구현 SSOT가 아닙니다.
- Next action: `TODO.md`의 active 항목을 보기 전에, 현재 환경 제약을 먼저 확인할 때 사용합니다.

## 2. DB 없이 가능한 작업
- 문서 정리
- import-safe 리팩터링
- parity harness의 스냅샷/리포트/메타데이터 보강
- WFO 후처리/분석 로직 정리
- 테스트 인터페이스, 패키지 구조, 로깅 계층 정리

## 3. DB 없이는 끝내기 어려운 작업
- 실제 PIT candidate 검증
- Tier / universe coverage 실측
- 운영 데이터 backfill 검증
- CPU/GPU parity의 실데이터 수치 확인

## 4. Reading Rule
- 현재 활성 작업과 직접 연결된 판단은 항상 해당 issue 문서를 우선합니다.
- 이 문서는 “환경 제약 때문에 무엇을 지금 해도 되는가”를 보는 참고 노트입니다.
