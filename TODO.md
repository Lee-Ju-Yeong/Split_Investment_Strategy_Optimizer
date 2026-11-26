# 현재 작업과 확인 범위

## 분할매매 연구 마감 · 2026-11-26

비용을 뺀 장기 수익률을 높이는 목적에서 **ETF 분할매매 연구는 종료하고, 개별주 분할매매는 초과수익 미입증으로 보류한다.** KODEX 200 단순 보유를 비교 기준으로 남긴다. 개별주 연구는 가설과 판정 기준을 먼저 정하고, 미사용 평가 자료와 분할 효과를 가릴 대조 계좌를 갖췄을 때 다시 검토한다. 현재 추가 실험은 없다. [ETF·개별주 최종 판단과 정식 결과 위치](todos/done_2026_11_26-split-strategy-review.md)

### 개별주 선행 연구와 자료 수집 · 2026-11-22 당시 기록

목표는 시장 ETF와 비슷한 최대낙폭에서 비용 차감 후 초과수익을 낼 수 있는지 보는 것이다. 종목 선정과 분할매매를 나누어 평가한다.

- **2단계.** 종목 선정만 한 계좌는 KODEX 200 대비 0/5였다. 현행 분할 계좌는 원래 기준 2/5, 결과를 본 뒤 추가한 칼마 보완 기준 3/5다.
- **3단계.** 수익성 하락 시 추가매수 보류(C1)는 사전 해석 규칙상 1/5로 후속 후보가 아니다. 추가매수를 줄이면 낙폭이 얕아져 ETF 기준 통과가 늘지만, 무작위로 보류해도 비슷했다.

11-22에 연구 DB·코드·산출물을 myworks-ws의 영속 볼륨(`/mnt/volume/workspace/split-research`)으로 옮겼다. DB 16개 표의 체크섬과 3월 보류 구간·3단계 산출물의 해시가 일치했다. FinancialData EPS·BPS의 주식 수 보정은 설계 검토 뒤 보류하고, 연속 계좌와 후보군 대 지수 비교를 먼저 마쳤다. [결과와 판정 기준](todos/2026_11_21-stock-selection-and-additional-buys.md)

- **끊지 않은 연속 계좌 진단 완료(11-22).** 원래 기준은 미통과 유지, 칼마 보완은 배당·시작일에 따라 갈려 미확정이다. 2년 재시작은 연복리를 연 1.1~6.8%p 높였다. 연속 계좌는 2025년까지 세 시작일 모두 KODEX 200보다 낮고 낙폭이 깊다. 수정주가 단위 정수 체결과 처분 불가 보유분 처리는 해석상 한계로 남긴다. [결과](todos/2026_11_21-stock-selection-and-additional-buys.md)
- **1등급 후보군 대 시장 지수 진단 완료(11-22).** 같은 구성의 시장보다 낫다는 판정은 결론 못 냄(교체일에 따라 갈림)이다. KODEX 200에는 연 1.8~2.8%p 지고 낙폭은 16%p 깊다. 격차는 주로 동일 비중·중소형 바스켓 자체에서 나타났다. [결과](todos/2026_11_21-stock-selection-and-additional-buys.md)
- **개별주 선행 연구 종료(11-22 사용자 결정).** 당시 1등급 후보와 분할 규칙 조합은 24개월 확인 대상으로 넘기지 않고 ETF 연구로 옮겼다. [당시 판단](todos/2026_11_21-stock-selection-and-additional-buys.md)
- **일일 수집 복구는 보류(11-22 사용자 결정).** KRX 차단 위험을 줄이려고 2026-02-09 이후 갱신을 하지 않는다. 다음 갱신용으로 날짜별 일괄 수집 경로(하루 KRX 12회)를 만들었지만 3차 리뷰에서 결함이 남아 **사용 금지 상태로 보관**한다. [상태와 남은 일](docs/operations/krx_daily_bulk_update_runbook.md)

## ETF M5 정식 탐색 R1 종료 · 후보 승격 보류

780개 단위를 모두 처리했고 연도별 선택 검증과 과최적화 진단을 완료했다. 2016~2025년 동시 개선 바스켓
비율은 연도 중앙값 13.22%, 최고 40%였다. 후보 128번은 M6로 승격하지 않고 M7 보류 자료를 보존한다.
실행기와 분석기 수정은 PR #125로 병합했다. [결과와 판단](docs/research/etf_m5_formal_r1.md) · [실행 기록](todos/2026_11_17-etf-m5-readiness.md).

후속 표본 24개로 전량 청산 후 월초 재진입을 비교했다. 두 장중 경로에서 수익률과 낙폭이 함께 개선된 바스켓은
11/24개였다. [청산 뒤 대기·재진입 비교와 판단](todos/2026_11_20-etf-reentry-diagnostic.md).

아래 문단들은 그 전 기록이다.

### 이전 단계: 정식 탐색 계획·실행기 구현·실자료 연결 실행 완료

PR #123을 병합했다(`6960031`). 주 기준선이 사실상 현금이던 원인(재진입 없는 규칙+2년 연속 학습)을 확인하고
정식 탐색 계획(`m5_search`)·실행기(`m5_runner`)를 구현했다. 주 기준선 비중은 직전 두 해의 1년 실행에서 구한다.
전액 환전 반올림으로 결제현금이 음수가 되던 엔진 결함을 고쳤다. CPU599개·예제5개·native180개·실행기25개 통과,
실자료 연결 실행 단위67/67 완료(설정 선택 없음). 정식 설정·선택/PBO 분석·정식 실행은 미완료다.
아래 두 문단은 그 전 기록이다.

PR #122를 `main`에 병합했고(`28b33ca`) 실자료 작은 실행4/4가 통과했다(069500+QQQ, 연결 검사).
환율 장기 휴장 정책(7일 초과는 KRX 세션이 없고12일 이내일 때만 허용)을 구현했고 로컬에서
CPU589개·예제5개·native179개·실행기25개가 통과했다. PR #123에서 독립 리뷰의 P2 1건을 보완했다.
아래 문단은 병합 전 기록이다.

사용자가 제안한 순서 전체를 승인했다. M3·M4 완료본을 `a2b101b`로 고정했고 상폐 청산·국내 결제·
입력 보호와 M5 조건 고정·작은 실행 연결을 구현했다. Opus의 M3 CI 의존성·US 단독 학습 구간 P2를 수정했고
최신CPU583개·예제5개·표준 라이브러리 native173개·실행기25개·US 단독 합성4/4가 통과했다.
SOL·Opus의 차단 해소 확인을 마쳤다. 남은 검사 시간 지적도 US 단독 검사를 분리해 보완했고
29모듈 전체583개·예제5개가 통과했다. 실행 준비 구현·합성 검증 범위를 완료했다.
실자료는 재택 PC에 있어 퇴근 후 전송이 필요하다. 정식 전체 탐색·중첩 검증·보류 판정은 미실행이다.
[M5 계획·계약·현재 상태](todos/2026_11_17-etf-m5-readiness.md).

### 선행 M4 완료

월별 리밸런싱·단순 보유·투자 비중 프런티어를 구현했다. 학습 구간에서 비중·가중을 고정하고 M2 원장을 공유한다.
최신CPU538개·예제4개, 실행기24개·M3대조153개가 통과했고 필수 SOL 리뷰의 지적3건을 해소했다.
Opus 최종 리뷰의 P1/P2는 없으며 P3 두 건도 수정·회귀·SOL 확인을 마쳤다.
[M4 범위·검증·완료 기록](todos/done_2026_11_17-etf-m4-baselines.md).
다음 M5는 상폐 청산·국내 결제 등 기존 선행 조건을 충족한 뒤 시작한다. 실자료 탐색·보류 평가는 아직 하지 않았다.

M2 완료본 `e241371`에서 별도 브랜치로 C++ 매매·체결 후보 커널을 추가했다. Python 현금·환전·세금
원장을 공유하며, M3 검사153개·CPU480개·예제3개·실행기24개가 통과했다. 필수 리뷰의 차단 결함을
해소했고 합성24일봉 전체 실행은 두 경로에서4.30~4.56배 빨랐다. M4의 선행 검증으로 유지한다.
[M3 범위·검증·측정](todos/done_2026_11_17-etf-m3-fast-kernel.md). 실제 자료 탐색·수익성 판정은 이후 단계다.
사용자가 추가 요청한 Opus 독립 리뷰는 완료됐고 P1/P2 차단 결함이 없다.
추가 합성5,500건의 M2 수치 일치를 확인했으며 Decimal 문자열 표현 차이는 문서에 명시한다.

### 선행 M2 완료 기록 (2026-11-18 기록)

Orca 워커5개의 M2 Python 기준 엔진을 통합하고 Opus 지적을 보완했다. 최종 후보는 Linux Python3.12에서
선택 CPU480개·예제3개를 통과했고 동일 실행기24개 통과 증거를 재사용한다. Opus r3는 M3 대조 기준으로
남은 차단이 없다고 확인했다. ChatGPT 공통 규칙 합성3사례도 일치했다. 이후 M3는 위 완료 기록을 따른다.
상장폐지 청산·국내 결제·완결 세후 보고 등 M5 전 실행 조건은 상세 기록에 남긴다.
[실행 범위·진행·검증](todos/2026_11_17-etf-m2-engine.md).
아래 11월 6일 정비 기록과 별도이며, M3 이후 최적화·수익성 판단은 아직 진행하지 않았다.

> 문서 대조일: 2026-11-06(Asia/Seoul). 실제 시험일·입력·후보 SHA는 각 기록에서 따로 확인합니다.

처음 방문했다면 [프로젝트 소개](README.md) → [구현 근거](docs/engineering-evidence.md) → [직접 확인하기](tests/README.md) 순서로 읽으세요.
모든 작업 문서는 [작업 색인](todos/README.md), 이번 56개 제안의 처리 결과는 [정비 실행 기록](todos/2026_11_06-project-maintenance-execution.md)에 연결합니다.

## 이번 단계: CPU 합성 시나리오 재현

**CPU 구현·후보 검증 완료.** [완료 범위·기대 거래·검증 근거](todos/done_2026_11_06-cpu-split-demo.md)에 따라 실제 CPU 엔진·전략·체결·자산 계산을 가상 7거래일로 재현합니다. 거래 4건, 최종 현금·자산 1,020,551원, 수수료 60원·세금 389원, CPU 72개·예제 2개·실행기 자체 24개를 확인했습니다. 최종 공개 후보와 병합 후 CI·Pages 확인은 [PR #115](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/pull/115)에 연결합니다. 그 확인까지 마치면 이번 정비 단계를 마감합니다.

#21의 CPU 경로 설명을 보강했고 DataHandler·GPU의 남은 설명 요구는 보존합니다. GPU 환경 안내·DB 표준화는 이 단계의 완료 조건이 아닙니다.

<a id="다음에-할-일"></a>
## 그 뒤에 할 일

이전 우선순위 정리에서 고른 후속 작업입니다. #21의 CPU 설명만 이번 단계에 포함하며, #20·#58은 미착수입니다. 아래 항목 전부를 이번 단계에서 끝내야 한다는 뜻은 아닙니다.

| 순서 | 작업 | 바로 다음 행동 | 완료를 확인하는 기준 |
| --- | --- | --- | --- |
| 1 | [#21 핵심 코드 설명 보강](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/21) | CPU 경로 반영 후 남은 DataHandler·GPU 진입점의 입력·반환·날짜·오류 설명을 대조 | 원래 요구 대비 CPU 완료·남은 대상을 기록. 나머지 요구를 마친 뒤 이슈 종료 판단 |
| 2 | [#20 GPU 환경 준비·실행 안내](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/20) | 검증 가능한 GPU 환경과 공개 가능한 환경 기록 확보 | OS·장비·드라이버·CUDA·Python·패키지 조합, 설치부터 최소 실행·관련 검사까지 실제 명령·결과 기록. 통합 CLI 구현은 #19에 유지 |
| 3 | [#58 DB 접근 방식 정리](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/58) | 호출자·설정 경로·연결 생성/반환/종료를 조사하고 호환·전환·복구 계획 검토 | 원래 요구인 런타임 DB 접근 통일·중복 설정 제거·문서 갱신을 별도 구현과 격리 DB 회귀 검증으로 확인. 현황 조사만으로 완료 처리하지 않음 |

#21은 외부 장비 없이 설명 대조부터 시작할 수 있습니다. #20은 GPU 환경 준비가 필요하며, 준비를 기다리는 동안 #58의 읽기 전용 현황 조사처럼 독립적인 준비를 먼저 할 수 있습니다. 제품 DB 변경은 별도 설계·검증 범위를 확정한 뒤 진행합니다.

이슈 목록에서는 기존 `priority-high`(#21), `priority-medium`(#20·#58) 라벨로 구분합니다. 선정 이유와 이번 정리 범위는 [후속 기록](todos/2026_11_06-project-maintenance-execution.md#후속-정리-다음에-할-일의-우선순위)에 있습니다.

## 이번 정비 결과

| 작업 | 상태 | 다음 행동 |
| --- | --- | --- |
| 문서·이슈 상태 정리 | PR #111 반영 완료 | 이슈 17개 갱신, #67·#97 근거로 종료, PR #75·#95 유지 |
| #59 DB 통합 검사 | PR #112 병합·main 검증 완료 | [결과](todos/done_2026_02_15-issue59-tests-split-no-db-gpu.md): DB 3·연결 제한 14 통과. main CI 확인 후 이슈 completed 종료 |
| 방문 동선·Pages | PR #113 구현·검증 기록 | [최종 후보·병합 후 공개 확인](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/pull/113), 홈페이지·Topics 반영 |

매매 규칙·GPU 계산·운영 DB·전체 환경 이관은 이 정비의 범위가 아닙니다. DB 검사는 합성 자료와 실행마다 폐기하는 검사용 DB만 사용합니다.

## 구현됐지만 남은 확인

| 항목 | 확인된 범위 | 남은 확인 |
| --- | --- | --- |
| 공매도 데이터의 지연 반영 | 임시 3거래일 정책·회귀 검사·읽기 전용 영향 비교 구현 | 실제 제공처 가용 시점 확정, 기존 장기 재계산의 완료 증거·행 수·최신 날짜·샘플 확인. 이번에 재계산하지 않음. [기록](todos/2026_03_07-short-selling-publication-lag-pit.md) |
| 기간을 나눈 전략 평가 #68 | PR #106의 후보 고정·CPU 검산·holdout 보호·후보 추출 | 원래 ablation·강건성 평가와 최종 승인 조건. 열린 PR #75의 차이도 별도. [요구 대조](todos/2026_02_09-issue68-robust-wfo-ablation.md) |
| 전체 실행 환경 | CPU 최소 확인 경로와 명령 안내 제공 | Python 환경 파일·YAML/INI 설정 혼재, GPU 환경 준비 안내. [실행 조건](docs/running.md) |
| 과거 CPU·GPU 일치 및 성능 | 해당 기록의 당시 입력에서 확인 | 전체 입력·최신 GPU 환경을 검증한 결과가 아님. [측정 조건](docs/engineering-evidence.md#gpu) |

## 그 밖의 후속 개선과 연구

현재 상태·근거·재개 조건은 이슈 본문에 기록합니다. 오래됐다는 이유로 닫거나 연구를 구현 성과로 소개하지 않습니다.

- 사용성·설명: [#19 통합 CLI](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/19), [#30 결과 리포팅](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/30).
- 구현 후속: [#52 GPU 순차 처리·병목](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/52), [#54 파이프라인](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/54), [#57 모델·캐시](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/57).
- 평가·기준서: [#68 강건성 평가](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/68), [#94 전략 기준서](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/94). PR #75·#95는 차이와 재개 조건을 기록하고 유지합니다.
- 연구: [#35 동적 파라미터](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/35), [#72 기업 이벤트와 포지션 승계](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/72), [#101 분포 기반 최적화](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/101), [#107 ETF 전략](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/107).
- 다른 과제에서 이어진 항목은 [#71 후속 연결표](todos/2026_03_01-issue71-carryover-non71-backlog.md)에서 추적합니다.

## 완료 기록

| 완료한 범위 | 근거와 경계 |
| --- | --- |
| 후보군·시점 기준 전환 #67 | [당시 검증 기록](todos/done_2026_02_09-issue67-tier-universe-migration.md). 과거 모드의 모든 체크박스를 지금 실행했다는 뜻은 아님 |
| 레거시 조사와 승인 범위 정리 #97 | [분류·승인·유지 예외](todos/done_2026_02_17-issue97-legacy-code-audit-governance.md) |
| 호환 경로 정리 #93 | [PR #96 완료 기록](todos/done_2026_02_16-issue93-wrapper-deprecation-removal-plan.md) |
| GPU 처리량 개선 | [PR #103 기록](todos/done_2026_02_17-issue98-gpu-throughput-refactor.md). 기존 로컬 식별자 issue98은 실제 GitHub PR #98과 다름 |
| GPU 후속 개선 #104 | [PR #105 측정·원복 기록](todos/done_2026_03_13-issue104-gpu-throughput-followup-hotpath.md) |
| 공개 안내·작은 예제 | [PR #109 완료 기록](todos/done_2026_11_06-public-project-refresh.md) |
| 격리 DB 통합 검사 #59 | [PR #112 완료 기록](todos/done_2026_02_15-issue59-tests-split-no-db-gpu.md): 실제 MySQL 3개·연결 제한 14개 통과 |
| CPU 기본 CI | [PR #110 완료 기록](todos/done_2026_11_06-cpu-checks-ci.md): 기존 검사 34개·예제·도구 자체 검사 22개 |
| CPU 합성 시나리오 재현 | [PR #115 CPU 검증 기록](todos/done_2026_11_06-cpu-split-demo.md): 7거래일·4거래·CSV, CPU 72·예제 2·실행기 자체 24 |

그 밖의 완료 기록과 연구·검토 문서는 [전체 색인](todos/README.md)에 있습니다.
[3월 사업화 검토](todos/2026_03_10-roadmap-commercialization-checkpoint.md)의 분기 일정·외부 출시 조건은 당시 제안입니다. 현재 서비스 출시 일정이나 채용용 코드 소개의 제한으로 읽지 않습니다.

## 기록을 갱신할 때

- 현재 할 일은 이 상태판, 이유·완료 조건·검증은 상세 작업 문서에 남깁니다.
- 문서 유형(구현·연구·검토·참고)과 상태(진행·완료·보류)는 별개입니다.
- 완료 문서는 결과·근거·남은 조건을 적고 알려진 링크를 고친 뒤 done_를 붙입니다. 시험 건너뜀은 통과가 아닙니다.
- 실제 자료·시험 근거 없이 완료·개인 기여·운영 현황·성능을 확대하지 않습니다.
