# WFO / OOS Lane 임시 합의안

> Type: `review`
> Status: `draft`
> Priority: `P2`
> Last updated: `2026-03-12`
> Related issues: `#68`, `#98`, `#56`
> Gate status: `temporary design consensus only; not implemented`

## Summary
- What:
  - parameter simulation, Walk-Forward Optimization(WFO), Out-of-Sample(OOS) 검증을 어떤 순서와 규칙으로 운영할지에 대한 임시 합의안을 정리한다.
- Why:
  - 현재 `#68` 공식안이 아직 고정되지 않았고, `#98` combo mining / parity 결과를 어디까지 연구용으로 쓰고 어디부터 승인용으로 쓸지 기준이 필요하다.
- Current status:
  - Codex multi-agent 1차 검토와 Gemini 2nd opinion을 대조한 결과, `Anchored WFO + final untouched OOS + separate stress pack` 구조가 가장 현실적인 절충안으로 정리되었다.
- Next action:
  - `#68`에서 `hard gate`, `non-overlap WFO`, `CPU audit 역할`, `holdout 범위`를 공식안으로 잠근다.

## 1. 초심자용 현재 결론
### 1-1. 한 줄 결론
- 먼저 후보를 넓게 찾고, 그다음 시간을 앞으로 걸어가며(WFO) 검증하고, 마지막에 진짜 안 건드린 최신 구간(OOS)으로 최종 시험을 본다.

### 1-2. 지금 바로 기억할 4줄
- `parameter simulation`은 “좋은 후보를 많이 찾는 단계”다.
- `WFO`는 “과거에서 고른 후보가 미래 구간에서도 계속 괜찮은지 보는 단계”다.
- `OOS`는 “마지막으로 남겨둔 안 본 시험지”다.
- `2020 폭락장`은 너무 중요해서 꼭 검증해야 하지만, 최종 안 본 시험지라고 부르면 안 된다.

### 1-3. 용어를 아주 쉽게 풀면
- `IS (In-Sample)`:
  - 후보를 고를 때 보는 학습용 구간
- `OOS (Out-of-Sample)`:
  - 후보를 고른 뒤 처음 보는 검증 구간
- `WFO (Walk-Forward Optimization)`:
  - 시간을 앞으로 조금씩 옮기면서 `IS -> OOS`를 여러 번 반복하는 방식
- `Anchored WFO`:
  - 시작일을 고정하고, 학습 구간을 점점 길게 늘리는 방식
- `Stress pack`:
  - 폭락장처럼 특별히 힘든 구간만 따로 모아 보는 추가 시험
- `Plateau`:
  - 1등 한 점만 좋은 게 아니라, 주변 설정들도 함께 괜찮은 넓은 구간

## 2. 한 페이지 결론
- `Research mining lane`과 `promotion evaluation lane`을 분리한다.
- parameter simulation은 넓게 돌려도 되지만, 목적은 `top 1` 선정보다 `parameter family / plateau`를 찾는 데 둔다.
- WFO는 `Anchored WFO`를 채택할 수 있다.
  - 뜻: 시작일을 고정하고 IS 기간을 점점 늘리면서 다음 구간 OOS를 검증한다.
- 다만 release-grade 경로에서는 `non-overlap only`를 강제한다.
  - `OOS_Start > IS_End`
  - 중복 OOS 날짜 평균 합성 금지
- `2020-03` crash는 반드시 검증한다.
  - 다만 `final untouched OOS`라고 부르지 않고, `WFO validation fold + stress-labeled regime`로 다룬다.
- 최종 승인용 OOS는 별도 최신 구간으로 남긴다.

## 3. 이번 합의의 핵심
### 3-1. parameter simulation은 어떻게 쓰나
- 넓은 시뮬레이션은 허용한다.
- 다만 결과 해석은 아래 순서를 따른다.
  1. coarse grid로 대략적인 민감도와 plateau를 찾는다.
  2. fine grid로 plateau 주변을 다시 촘촘히 본다.
  3. shortlist만 CPU SSOT audit 대상으로 넘긴다.
- 즉, full-grid 결과는 `가설 생성`용이고, 최종 승인 근거는 아니다.

### 3-2. robust candidate는 무엇을 뜻하나
- 단일 최고점이 아니라, 주변 파라미터도 함께 괜찮은 `넓은 고원(plateau)`의 대표 후보를 뜻한다.
- robust score는 사용할 수 있지만, hard gate를 통과한 후보 안에서 tie-break로만 쓰는 것이 더 안전하다.
- cluster size, 내부 분산, fold별 OOS 전이 성능을 함께 봐야 한다.

### 3-3. WFO는 어떤 구조가 맞나
- 임시 권고안은 `Anchored WFO`다.
- 이유:
  - 사용 가능한 adjusted/PIT window가 아주 길다고 보긴 어렵다.
  - 과거 데이터를 버리지 않고 누적해서 쓰는 쪽이 안정적이다.
- 단, 현재 구현처럼 overlap이 허용되고 OOS curve를 평균 합성하는 경로는 release-grade 증거로 쓰면 안 된다.

### 3-4. OOS는 어떻게 남기나
- `WFO OOS`와 `final untouched OOS`를 구분한다.
- `WFO OOS`:
  - 모델/선택 규칙이 시간 순서대로 버티는지 보는 중간 검증
- `final untouched OOS`:
  - freeze 이후 한 번만 보는 마지막 시험

## 4. 임시 권고 프로토콜
### 4-1. lane 분리
- `Research mining lane`
  - 넓은 parameter simulation
  - plateau 탐색
  - shortlist freeze
  - stress/regime 관찰
- `Promotion evaluation lane`
  - strict_pit
  - candidate_source_mode=tier
  - non-overlap Anchored WFO
  - CPU audit(pass/fail)
  - final untouched OOS

### 4-2. 왜 lane을 나누나
- 초심자용으로 말하면:
  - 연구 단계에서는 “어떤 후보가 좋아 보이는지” 자유롭게 탐색해도 된다.
  - 하지만 승인 단계에서는 “정해진 규칙대로 정말 버티는지”만 봐야 한다.
- 이 둘을 섞으면:
  - 연구 중에 본 데이터가 최종 시험에 다시 들어가서
  - 성적이 실제보다 좋아 보일 수 있다.

### 4-3. 2020 crash 취급 원칙
- `2020-03`은 빼지 않는다.
- 다만 아래처럼 부른다.
  - `WFO validation fold`
  - `stress-labeled regime`
- 아래 주장은 금지한다.
  - `2020년 OOS는 final untouched holdout이다`
  - `2020년 WFO OOS와 stress 결과를 합산한 KPI가 최종 승인 성능이다`

### 4-4. 날짜안
- 일반 권고안:
  - `Development / WFO`: `2013-11-20 ~ 2023-12-31`
  - `Final untouched holdout`: `2024-01-01 ~ 2025-12-31`
- 현재 저장소 문맥에서 더 보수적인 대안:
  - `Final untouched holdout`: `2025-01-01 ~ 2025-11-30`
  - 이유:
    - `2025-12-01 ~ 2026-01-31`는 parity canary에 사용된 적이 있어 strict claim이 약해질 수 있다.

### 4-5. Anchored WFO 예시
| Fold | IS Period | OOS Period | 비고 |
| --- | --- | --- | --- |
| 1 | `2014-01-01 ~ 2018-12-31` | `2019-01-01 ~ 2019-12-31` | 평시 검증 |
| 2 | `2014-01-01 ~ 2019-12-31` | `2020-01-01 ~ 2020-12-31` | crash / 팬데믹 |
| 3 | `2014-01-01 ~ 2020-12-31` | `2021-01-01 ~ 2021-12-31` | 유동성 장세 |
| 4 | `2014-01-01 ~ 2021-12-31` | `2022-01-01 ~ 2022-12-31` | 긴축 하락장 |
| 5 | `2014-01-01 ~ 2022-12-31` | `2023-01-01 ~ 2023-12-31` | 회복기 |
| 6 | `2014-01-01 ~ 2023-12-31` | `2024-01-01 ~ 2024-12-31` | 최신 regime |

## 5. hard gate 초안
- 아래는 아직 `#68` 공식안이 아니라 임시 출발점이다.
- 후보 gate:
  - `median(OOS/IS) >= 0.60`
  - `fold_pass_rate >= 70%`
  - `OOS_MDD_p95 <= 25%`
- 추가 stress reject 예시:
  - `2020` 또는 `2022` fold에서 `MDD > 30%`
  - `Recovery Factor < 1.0`

## 6. 지금 구현에 대한 경고
- 현재 WFO 구현은 overlap 경로를 허용할 수 있다.
  - [walk_forward_analyzer.py](/root/projects/Split_Investment_Strategy_Optimizer/src/analysis/walk_forward_analyzer.py:542)
- 현재 최종 curve는 중복 OOS 날짜를 평균 합성할 수 있다.
  - [walk_forward_analyzer.py](/root/projects/Split_Investment_Strategy_Optimizer/src/analysis/walk_forward_analyzer.py:691)
- 현재 CPU certification 일부는 audit보다 same-IS rerank 성격이 남아 있다.
  - [walk_forward_analyzer.py](/root/projects/Split_Investment_Strategy_Optimizer/src/analysis/walk_forward_analyzer.py:620)
- 따라서 이 문서는 `바로 실행 승인`이 아니라 `공식 설계 잠금 전 임시 합의안`이다.

## 7. 다음 공식화 후보
1. `#68` 문서에 `Anchored WFO`와 `non-overlap only`를 공식 제안으로 승격
2. `hard gate` 식을 코드와 문서에서 하나로 고정
3. `CPU certification`을 rerank가 아니라 audit/pass-fail로 역할 고정
4. `fold_gate_report.csv`, `holdout_manifest.json` 같은 산출물 포맷 정의

## 8. 참고 문서
- [#68 Robust WFO / Ablation](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_02_09-issue68-robust-wfo-ablation.md)
- [#98 GPU Throughput Refactor](/root/projects/Split_Investment_Strategy_Optimizer/todos/done_2026_02_17-issue98-gpu-throughput-refactor.md)
- [WFO analysis context](/root/projects/Split_Investment_Strategy_Optimizer/llm-context/04_wfo_analysis.md)
