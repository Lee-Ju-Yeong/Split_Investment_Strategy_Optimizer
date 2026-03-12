# Review: Business Model & Marketability

> Type: `review`
> Status: `reference`
> Priority: `N/A`
> Last updated: 2026-03-10
> Related issues: `#98`, `#68`, `#67`, `#101`
> Gate status: `recommendation issued (approval pending)`

## 1. Summary
- What: 현재 프로젝트에 가장 맞는 사업모델과 시장 진입 순서를 정리한 공개용 검토 문서입니다.
- Why: 제품 성숙도보다 앞서는 사업 약속을 막고, 공개 레포에서도 안전하게 공유할 수 있는 기준을 남기기 위해 작성했습니다.
- Current status: 향후 12개월 기본 모델은 `B2B 신뢰 기반 퀀트 검증 플랫폼`으로 정리되었고, 외부 제품 판매는 아직 보수적으로 다뤄야 합니다.
- Next action: `lag`, `#98`, `#68`, legal/compliance redline이 닫힌 뒤 첫 외부 오퍼 범위를 다시 판정합니다.

## 2. Easy Conclusion
- 최종 상태: `추가 검증`
- 가장 맞는 기본 모델:
  - 운영 프레임은 `E. 단계형 하이브리드`
  - 실질 primary는 `C. B2B 신뢰 기반 퀀트 검증 플랫폼`
- 가장 가까운 확장 모델:
  - `D. B2B 최적화/검증 API`
- 아직 이른 모델:
  - `A. B2C 구독형 신호/추천`
  - `B. 자동매매/실계좌 연동`

## 3. Why This Fits Best
- 현재 강점은 `알파 생산`보다 `검증 체계`입니다.
  - `CPU=SSOT`, `Parity First`, `PIT / No Lookahead`, `Evidence Before Promotion`이 이미 control plane 원칙으로 고정되어 있습니다.
- 현재 문서 상태와도 맞습니다.
  - `ShortSellingDaily PIT lag`는 아직 `Open`이고, `#98 Throughput promotion`은 `In Progress`, `#68 Robust WFO / Ablation`은 `Planned`입니다.
  - 즉, 지금은 실시간 판매형 제품보다 `검증 가능한 엔진` 포지셔닝이 맞습니다.
- 외부 환경도 이 방향과 맞닿아 있습니다.
  - 금융권은 AI 개인화와 검증 인프라에 관심이 커지고 있지만, 자동매매/자문 경계는 여전히 보수적으로 봐야 합니다.

## 4. Model Ranking
- `1위`: `C. B2B 검증 플랫폼`
  - 이유: 현재 강점과 가장 잘 맞고, 제품 성숙도와도 충돌이 적습니다.
- `2위`: `D. B2B 최적화/검증 API`
  - 이유: 장기 확장성은 좋지만, 먼저 `#68`과 `#98`을 닫아야 합니다.
- `3위`: `A. 제한적 B2C 리서치/리포트`
  - 이유: 시장성은 있으나 규제 경계와 기대 관리가 어렵습니다.
- `4위`: `B. 자동매매/실계좌 연동`
  - 이유: 현재 gate 상태와 가장 멀고 규제 리스크가 큽니다.

## 5. Hard Boundaries
- 지금 해도 되는 것:
  - 내부 검증 실행
  - evidence pack 정리
  - `NDA` 기반 read-only 설명 데모
- 아직 하면 안 되는 것:
  - 외부 유료 신호/API 판매
  - 실시간 또는 준실시간 주문/신호 제공
  - 실계좌 연동
  - `release-ready`처럼 들리는 홍보 문구
- 조건부로만 다시 검토할 수 있는 것:
  - fixed-fee 기반의 아주 좁은 `controls-assessment` 성격 외부 용역
  - 전제: legal/compliance sign-off, 금지 출력 redline, 고정 계약 문구, read-only artifact 중심

## 6. First External Offer Shape
- 추천 이름:
  - `Backtest Controls Assessment`
  - `Research Reproducibility Audit`
- 공개 문서에서 피할 표현:
  - `alpha`
  - `deployable`
  - `release-ready`
  - `recommended parameter`
  - `go-live`
- 공개 문서에 남길 산출물:
  - `PIT / lookahead audit`
  - `CPU-GPU parity report`
  - `replay / manifest evidence pack`
  - `data lineage summary`

## 7. 30/60/90
- `D+30`
  - `ShortSellingDaily lag` 정책 확정
  - 외부 문구와 금지 출력 redline 확정
  - 공개/비공개 문서 경계 정리
- `D+60`
  - `#98` strict parity 재검증
  - `#68` robust score / hard gate 공식안 고정
  - evidence pack 템플릿 정리
- `D+90`
  - 외부 제품 출시가 아니라 `NDA read-only demo` 가능 여부 재판정
  - legal/compliance sign-off 이후에만 제한적 외부 유료 범위 검토

## 8. Public Repo Hygiene
- 이 문서는 공개 레포에 남겨도 되는 수준으로만 작성합니다.
- 아래 정보는 공개 문서에 두지 않습니다.
  - 고객명
  - 세부 가격표
  - 계약 문안
  - 로펌 의견서 원문
  - 세일즈 파이프라인
- 상세 상업/법무 초안은 `docs/business/private/`, `docs/legal/private/` 같은 ignore 경로로 분리합니다.

## 9. Source Notes
- 내부 근거:
  - [TODO.md](/root/projects/Split_Investment_Strategy_Optimizer/TODO.md)
  - [로드맵/사업화 점검 메모](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_10-roadmap-commercialization-checkpoint.md)
  - [Hybrid Release Gate Board](/root/projects/Split_Investment_Strategy_Optimizer/docs/operations/2026-03-06-hybrid-release-gate-board.md)
- 외부 근거:
  - FSC Korea Fintech Week 2025: <https://www.fsc.go.kr/eng/pr010101/85503>
  - FSC 금융권 AI 플랫폼 공개 (2025-12-22): <https://www.fsc.go.kr/no010101/85908>
  - 자본시장법 제6조: <https://www.law.go.kr/LSW/lsSideInfoP.do?docCls=jo&joBrNo=00&joNo=0006&lsiSeq=283193&urlMode=lsScJoRltInfoR>
  - 자본시장법 제18조: <https://www.law.go.kr/LSW/lsSideInfoP.do?docCls=jo&joBrNo=00&joNo=0018&lsiSeq=283193&urlMode=lsScJoRltInfoR>
  - 자본시장법 제101조: <https://www.law.go.kr/LSW/lsSideInfoP.do?docCls=jo&joBrNo=00&joNo=0101&lsiSeq=284145&urlMode=lsScJoRltInfoR>
