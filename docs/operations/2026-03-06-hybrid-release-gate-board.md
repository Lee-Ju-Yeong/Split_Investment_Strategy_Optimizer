# Hybrid Release Gate Board

> 작성일: 2026-03-06
> 상태: Active
> 범위: `GPU proposes, CPU certifies finalists` 하이브리드 경로의 현재 허용 범위와 승격 게이트를 분리 관리
> 목적: `implementation-allowed lane`과 `release/promotion gate`를 같은 TODO 체크박스로 오해하지 않도록 단일 판단판을 제공한다.

## 1. 한 줄 결론
- 현재 hybrid 경로는 **구현 가능 / 실험 가능** 상태다.
- 하지만 아직 **기본 승인 경로 / 승격 가능** 상태는 아니다.

## 2. 현재 판정
| lane | 상태 | 의미 |
| --- | --- | --- |
| Implementation lane | `OPEN` | 하이브리드 경로 기반 구현, 계측, fallback 정리, finalist CPU certification 경로 보강 가능 |
| Research lane | `OPEN` | strict/pit/parity/replay 증적 수집 가능 |
| Default approval lane | `BLOCKED` | 운영 기본 승인 트랙으로 간주하면 안 됨 |
| Promotion lane | `BLOCKED` | release gate 미충족 상태 |

## 3. 지금 이미 구현된 것
- WFO에서 `GPU shortlist -> CPU certification -> CPU OOS` 경로가 존재한다.
- config example에 CPU certification 설정이 들어가 있다.
- CPU certification shortlist 로직과 artifact suppression 테스트가 있다.

## 4. 지금 허용되는 작업
- `#98`의 low-risk/perf-only 작업
  - telemetry 추가
  - batch fallback 계측
  - host/device transfer 관측
  - artifact/replay 보강
- finalist CPU certification 경로 보강
- strict PIT 승격 조건을 닫기 위한 parity/PIT 테스트 보강

## 5. 지금 금지되는 판단
- hybrid를 `release-ready`라고 부르는 것
- hybrid를 `default approved path`로 승격하는 것
- `strict_pit` 재검증 없이 example config를 운영 승인 예시처럼 쓰는 것
- `decision-level parity 0 mismatch` 없이 promotion 판단을 내리는 것

## 6. Release Gate

### 6-1. 필수 게이트
- `strict_pit` 승인 트랙에서 재검증 통과
- finalist CPU certification 통과
- decision-level parity `0 mismatch`
- `future reference=0`
- degraded run `false`
- `promotion_blocked=false`

### 6-2. 필수 증적
- run manifest
- parity diff report
- CPU certification CSV
- PIT/coverage report
- wall-time / GPU util / OOM retry / batch recovery 측정

## 7. 현재 미충족 항목
- `#56` synthetic/research parity gate는 종결되었고, 실제 optimizer/WFO CSV가 생기면 spot revalidation만 남음
- `#67`은 CPU opt-in strict frozen manifest까지 반영됐지만, candidate order direct validation과 PIT failure/log standardization이 미완료
- `#97` strict-only cleanup 미완료
- release-safe / research-only example 분리는 반영됐지만, 운영 승인 판단은 여전히 별도 증적이 필요

## 8. 현재 예시 config 해석
- `config/config.example.yaml`은 지금 `release-safe / promotion candidate`와 `research-only / non-promotion` 예시를 함께 담는다.
- 더 정확한 해석은:
  - 기본 strategy example은 `strict_pit + tier + strict_hysteresis_v1 + raise`
  - `cpu_certification_*`는 hybrid 기능 예시
  - `optimistic_survivor` / `legacy`는 연구/비승격 트랙 예시
- 즉 example config만으로 운영 승인 결론을 내리면 안 되고, 여전히 parity/PIT/future reference 증적이 함께 있어야 한다.

## 9. TODO 해석 규칙
- `#98`이 P1에 있다는 이유만으로 promotion 가능하다는 뜻은 아니다.
- `#56`이 P2에 있다는 이유만으로 구현 lane에서 완전히 뒤라는 뜻도 아니다.
- 해석 원칙은 아래다.
  - backlog order: 구현/정리 우선순위
  - release gate: 승격 판단 순서
- 승격 판단 순서는 별도로 `#56 -> #67 -> #97 -> #98`를 따른다.

## 10. 다음 액션
- `TODO.md`에서 hybrid 구현 완료와 release gate 미충족을 분리 표기
- example config의 승격 가능 의미 오해를 줄이는 문구 추가
- `#56`, `#67`, `#97`, `#98`의 증적 제출 위치를 이 문서 기준으로 연결
