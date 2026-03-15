# WFO(Walk-Forward Optimization, 과거로 공부하고 미래 구간으로 시험 보는 검증 방식) 여정과 현재 상태

> 작성일: 2026-03-14
> 대상 독자: 처음 이 프로젝트를 접하는 팀원, 관리직 임원, 협업자
> 목적: Magic Split 전략의 WFO 개선이 왜 필요했는지, 지금 어디까지 왔는지, 다음에 무엇을 해야 하는지 쉽게 공유하는 문서

## 1. 한 줄 요약
- 우리는 `좋아 보이는 파라미터`를 찾는 단계와 `정말 운영해도 되는지 확인하는 단계`를 분리하는 쪽으로 WFO 체계를 다시 만들고 있다.
- 지금은 그 구조의 큰 골격과 핵심 guardrail(실수 방지 장치)은 대부분 들어왔고, 남은 것은 `행동지표/ablation(비교 실험) 고도화`와 `approval-grade holdout(승인용 최종 등급 holdout)의 최종 마감` 같은 마지막 마감 작업에 가깝다.
- 다만 `마지막 마감`이라는 뜻이 `이제 아무 검증도 안 남았다`는 뜻은 아니다. 특히 `freeze contract(후보 고정 계약)`과 `holdout 차단 경로`는 계속 보수적으로 잠그는 중이다.

## 2. 이 문서가 필요한 이유
- 기존 WFO는 연구용 결과와 승인용 결과가 섞여 읽힐 여지가 있었다.
- 그러면 팀 내부에서는 물론, 나중에 관리직이나 외부 이해관계자에게도 이런 오해가 생길 수 있다.
  - `이 수익곡선이 진짜 최종 성적표인가?`
  - `이 결과는 후보 탐색용인가, 운영 승인용인가?`
  - `이 전략은 언제 시작해도 안정적인가, 아니면 특정 구간에서만 좋아 보였는가?`
- 그래서 지금은 단순히 코드 몇 줄을 고친 것이 아니라, `전략 검증의 의미` 자체를 더 명확히 다시 세우는 작업을 하고 있다.
- 최근에는 `decision_date(후보 고정 기준일)`가 없으면 promotion lane이 아예 시작되지 않도록 바꿔서, “언제 후보를 고정했는가”를 더 엄격하게 다루기 시작했다.

## 3. 먼저 아주 쉽게: WFO가 뭐냐
- WFO는 `Walk-Forward Optimization`의 줄임말이다.
- 아주 쉽게 말하면:
  - 과거로 공부하고
  - 그다음 미래 구간으로 시험 보고
  - 이걸 여러 번 반복해서
  - 전략이 시간이 지나도 버티는지 확인하는 방식이다.

비유로 보면:
- `파라미터 시뮬레이션`은 문제집을 많이 푸는 단계다.
- `WFO`는 모의고사를 여러 번 보는 단계다.
- `holdout`은 끝까지 안 풀고 남겨 둔 마지막 시험지다.

## 4. 왜 이번에 WFO를 다시 손봤나
### 4-1. 예전 방식의 핵심 문제
- 예전 구조는 `연구용 관찰`과 `승인용 검증`이 충분히 분리되어 있지 않았다.
- 특히 아래 같은 문제가 쟁점이 됐다.
  - 같은 날짜가 여러 검증에 겹쳐 들어갈 수 있었다.
  - OOS 결과를 평균 합성한 curve가 실제 운용 계좌처럼 읽힐 수 있었다.
  - 시작 시점이 달라도 안정적인지와, 시간이 지나도 버티는지가 같은 질문처럼 섞였다.
  - 최신 holdout 구간도 길이가 짧아서 `최종 승인용 마지막 증거`라고 부르기에는 약했다.

### 4-2. 왜 이게 중요한가
- Magic Split 전략은 한 번에 풀인하지 않는다.
- 조금 사고, 더 떨어지면 추가로 사고, 시간이 지난 뒤 빠져나오는 구조다.
- 그래서 단기 성과 1위보다 더 중요한 질문은 아래에 가깝다.
  - `언제 시작해도 너무 심하게 흔들리지 않는가`
  - `시간이 지나도 다시 설명 가능한가`
  - `실제 자본이 적절히 배치되고 회전했는가`

즉, 이 전략은 단순히 `가장 높은 수익률 하나`를 고르는 게임이 아니라, `오래 버티고 다시 설명 가능한 후보`를 고르는 문제에 가깝다.

## 5. 우리가 새로 잡은 큰 구조
### 5-1. 두 개의 lane(검증 경로)으로 분리
- 이제 WFO는 크게 두 갈래로 본다.

1. `Research lane(연구용 검증 경로)`
- 질문:
  - `시작 시점이 달라도 결과가 너무 흔들리지 않는가`
- 성격:
  - 연구용
  - 시작 시점 민감도 관찰용
- 현재 권장 방식:
  - `multi-anchor Anchored WFO(시작점을 여러 개 두고 반복 검증하는 방식)`

2. `Promotion lane(승인 심사용 검증 경로)`
- 질문:
  - `시간이 앞으로 가도 이 후보가 버티는가`
- 성격:
  - 승인 심사용
  - 더 엄격한 시간 전이 검증
- 현재 권장 방식:
  - `single-anchor non-overlap Anchored WFO(출발점은 하나로 고정하고, 시험 구간이 겹치지 않게 보는 방식)`

### 5-2. 왜 이렇게 나눴나
- 두 질문은 비슷해 보이지만 사실 다르다.
- `Research lane(연구용 검증 경로)`은:
  - `이 파라미터가 특정 시작일 운빨이 아닌가?`
  - 를 보는 쪽이다.
- `Promotion lane(승인 심사용 검증 경로)`은:
  - `이 후보를 진짜 공식 검증 기준으로 봐도 통과하나?`
  - 를 보는 쪽이다.

한 줄로 정리하면:
- research는 `후보를 이해하는 단계`
- promotion은 `후보를 심사하는 단계`

## 6. 그동안 어떤 결정을 했나
### 6-1. 연구용 결과를 최종 승인용처럼 보이게 하지 않기
- research lane(연구용 검증 경로)에서는 단일 합성 수익곡선을 최종 성적표처럼 쓰지 않기로 했다.
- 대신 `anchor별/fold별 분포`를 보는 쪽으로 방향을 정리했다.
- 이유는, 연구용 결과는 어디까지나 `흔들림 관찰`이지 `최종 승인 증거`가 아니기 때문이다.

### 6-2. 후보 선정 방식을 더 보수적으로 바꾸기
- holdout(끝까지 남겨 둔 최종 시험 구간) 직전에 여러 후보를 다시 비교하지 않기로 했다.
- 대신:
  - 먼저 후보를 좁히고
  - `single champion(최종 1등 후보)` 1개를 고정하고
  - `reserve(예비 후보)`는 provenance(후보 기록) 용도로만 남기고, 이번 `#68`에서는 자동 승계를 구현하지 않기로 했다.

즉:
- `holdout(끝까지 남겨 둔 최종 시험 구간)에서 여러 후보를 비교해서 더 좋은 것 고르기`는 금지
- `holdout 전에 champion(최종 1등 후보)을 봉인`하는 쪽으로 간다

### 6-3. holdout(끝까지 남겨 둔 최종 시험 구간)을 더 엄격하게 보기
- 이 전략 특성상 `1년 미만 holdout`은 너무 짧다는 공감대가 생겼다.
- 현재 정책 목표는:
  - `approval-grade holdout(승인용 최종 등급 holdout) >= 24개월`
- 다만 현재 저장소 기준 최신 구간은 아직 짧아서:
  - `2025-01-01 ~ 2025-11-30`
  - 는 `internal provisional holdout(임시 내부 검증용 holdout)`으로 부른다.

쉽게 말하면:
- `최종 시험지`는 따로 남겨두고 있지만
- 아직 충분히 길지 않아서
- `완전한 최종 승인 증거`라고 부르지는 않는 상태다.

## 7. 지금까지 실제로 구현된 것
### 7-1. lane(검증 경로) 분리의 뼈대
- `promotion_evaluation(승인 심사용 실행 모드)`
  - frozen shortlist 기반으로 동작
  - single-anchor non-overlap 방식으로 실행
- `research_start_date_robustness(시작 시점 민감도 연구 모드)`
  - frozen shortlist 기반 multi-anchor evaluation 경로가 생김

즉, 이제 코드도 예전처럼 “다 같은 WFO”가 아니라, 어느 정도 역할을 구분해 실행할 수 있게 됐다.

### 7-2. 결과를 더 정직하게 남기는 manifest(실행 계약/상태 기록 파일) 체계
- 결과 폴더에 아래 같은 artifact(실행 결과 파일)가 남기 시작했다.
  - `lane_manifest.json(검증 경로 상태 기록 파일)`
  - `holdout_manifest.json(holdout 상태 기록 파일)`
  - `promotion_candidate_summary.csv(후보 요약표)`
  - `final_candidate_manifest.json(최종 후보 봉인 기록 파일)`
- 이 파일들의 목적은 단순 저장이 아니라:
  - 어떤 lane이었는지
  - 어떤 후보를 썼는지
  - holdout을 건드렸는지
  - approval-grade(승인용 최종 등급)로 읽어도 되는지
  - 를 나중에 다시 증명할 수 있게 만드는 것이다.
- 그리고 이제 `strict_only_governance(운영 승인 판단 규칙)`도 이 파일들을 직접 읽어서:
  - `최종 후보가 hard gate(최소 통과 기준)를 통과했는지`
  - `CPU audit(CPU 기준 검산)이 pass(통과)인지`
  - `holdout 실행 상태가 어떤지`
  - 를 이유(reason)로 직접 반영할 수 있게 됐다.
- 여기에 더해 `final_candidate_manifest.json(최종 후보 봉인 기록 파일)` 안에:
  - `freeze_contract_hash(후보 고정 계약 해시값)`
  - `promotion_shortlist_hash_verified(후보 목록 파일 해시 일치 여부)`
  - `promotion_shortlist_modified_after_decision_date(후보 고정 기준일 이후 파일 수정 여부)`
  - `canonical_holdout_contract_verified(공식 holdout 경계 계약 일치 여부)`
  - 도 같이 남도록 바뀌었다.

### 7-3. final candidate(최종 후보) 계약
- promotion lane(승인 심사용 검증 경로)이 끝나면:
  - 후보 요약표를 만들고
  - `single champion(최종 1등 후보)`을 정하고
  - `final_candidate_manifest.json(최종 후보 봉인 기록 파일)`에 봉인한다.
- 여기에는:
  - champion 파라미터
  - reserve(예비 후보)
  - candidate hash(후보 내용이 바뀌지 않았는지 확인하는 고유 지문값)
  - tie-break 기준(점수가 비슷할 때 순서를 정하는 규칙)
  - 이 들어간다.
- 그리고 지금은 여기서 끝나지 않고:
  - `hard gate(최소 통과 기준)`를 먼저 통과했는지 보고
  - 통과한 후보 안에서만 `robust score(통과 후보끼리 비교하는 보조 점수)`로 tie-break(동점 정리)를 하도록
  - 코드와 문서의 공식 계약이 맞춰졌다.
- 또 `freeze contract(후보 고정 계약)`도 들어가서:
  - shortlist(후보 목록) 파일이 `decision_date(후보 고정 기준일)` 이후 바뀌었는지
  - holdout 경계가 현재 공식 계약과 맞는지
  - 를 자동 실행 전에 다시 확인하게 됐다.
- 즉 이제 `왜 이 후보가 champion(최종 1등 후보)이 되었는가`를 예전보다 훨씬 다시 설명하기 쉬워졌다.

### 7-4. holdout 자동 실행과 adequacy(충분성) guardrail(실수 방지 장치)
- `holdout_auto_execute=true(holdout 자동 실행 옵션)`일 때
  - champion이 준비되어 있고
  - holdout window가 설정되어 있고
  - final CPU audit(CPU 기준 최종 검산)까지 pass일 때만
  - holdout을 자동으로 실행하게 만들었다.
- 또 holdout 상태를 더 정직하게 남기기 위해:
  - `attempted`
  - `success`
  - `blocked`
  - 를 구분해서 기록하도록 정리했다.
- 여기에 더해, holdout이 너무 형식적으로만 끝나지 않도록:
  - `trade_count(거래 횟수)`
  - `closed_trade_count(완결된 거래 횟수)`
  - `distinct_entry_months(서로 다른 진입 월 수)`
  - `avg_invested_capital_ratio(평균 투자 자본 비율)`
  - `cash_drag_ratio(현금이 놀고 있던 비율)`
- 같은 adequacy 지표도 함께 본다.
- 이 adequacy 기준을 못 넘으면 `approval_eligible(승인 가능 여부)=false`로 강등되고, 정말 예외가 필요할 때만 `waiver(예외 승인 사유)`를 남기는 구조가 들어왔다.
- 그리고 이제는 `approval_eligible(내부 승인 가능)`와 `external_claim_eligible(대외 설명 가능)`를 따로 기록한다.
  - 즉, 내부적으로는 예외 승인으로 넘길 수 있어도
  - 대외적으로는 `아직 최종 승인 증거로 말하면 안 된다`를 코드가 따로 남겨 준다.

## 8. 지금 상태를 한 문장으로 말하면
- `WFO의 의미, 실행 경로, 최종 후보 선정 방식, freeze 계약, holdout 충분성 판단 방식까지 큰 줄기는 대부분 정리되었다.`
- 다만 `행동지표/ablation 고도화`와 `approval-grade holdout의 최종 경계 고정`은 아직 마지막으로 남아 있다.

## 9. 아직 남은 핵심 작업
### 9-1. 행동지표(behavior metrics)와 비교 리포트를 더 발전시키기
- 지금은 holdout adequacy 기준이 들어왔지만, 이것을 더 설득력 있는 비교 리포트로 키우는 작업이 남아 있다.
- 예를 들면:
  - `promotion_ablation_summary.csv`
  - `promotion_explanation_report.json`
  - 더 풍부한 행동지표(feature) 실험
- 이 단계가 완성되면 팀이 나중에 `왜 이번 버전이 더 낫다고 보는지`를 더 쉽게 설명할 수 있다.

## 10. 지금 진행률을 경영 관점에서 보면
### 10-1. 이미 해결한 것
- 후보 탐색과 승인 심사를 같은 성격으로 보던 문제를 구조적으로 분리했다.
- 연구용 결과가 승인용처럼 보일 수 있는 위험을 많이 줄였다.
- 결과를 나중에 추적·감사할 수 있도록 artifact(실행 결과 파일) 체계를 만들었다.
- 최종 후보를 봉인하는 계약이 생겼다.
- 최종 후보 선정 규칙(`hard gate + robust score + tie-break`)도 이제 코드와 문서에 같이 묶였다.
- holdout이 짧거나, 너무 의미 없이 끝났을 때 자동으로 강등시키는 adequacy 판단도 들어왔다.
- freeze 이후 shortlist가 바뀌었는지와 holdout 경계가 공식 계약과 맞는지도 이제 코드가 직접 확인한다.

### 10-2. 아직 남은 것
- 행동지표/ablation을 더 설득력 있는 비교 리포트로 끌어올리는 작업
- approval-grade holdout을 진짜 대외 설명 가능한 수준으로 닫기 위한 마지막 마감 작업

### 10-3. 지금 이 작업의 의미
- 이건 단순 성능 튜닝이 아니다.
- 전략 검증을 `사람 기억과 해석`에 덜 의존하고, `재현 가능한 운영 체계`로 옮기는 작업이다.
- 쉽게 말하면:
  - `좋아 보인다`를
  - `왜 좋은지 설명 가능하고, 나중에도 같은 기준으로 다시 검증 가능하다`
  - 로 바꾸는 과정이다.

## 11. 앞으로 팀에 이렇게 설명하면 된다
- `예전에는 연구용과 승인용 결과가 섞여 보일 위험이 있었다.`
- `지금은 research lane / promotion lane / holdout을 분리해서, 후보 탐색과 최종 승인 절차를 다르게 운영하도록 바꾸고 있다.`
- `현재는 final candidate manifest, governance 연결, hard gate/robust score, freeze contract, holdout adequacy guardrail까지 들어와서 큰 골격은 거의 잡혔다.`
- `남은 것은 행동지표/ablation 고도화와 approval-grade holdout 마감 작업이다.`

## 12. 참고 문서
- [WFO Approval Workflow Runbook](/root/projects/Split_Investment_Strategy_Optimizer/docs/operations/2026-03-14-wfo-approval-runbook.md)
- [Issue #68: Robust WFO / Ablation](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_02_09-issue68-robust-wfo-ablation.md)
- [WFO / OOS Lane 임시 합의안](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_12-wfo-oos-lane-provisional-review.md)
