# WFO(Walk-Forward Optimization, 과거로 공부하고 미래 구간으로 시험 보는 검증 방식) 여정과 현재 상태

> 작성일: 2026-03-14
> 대상 독자: 처음 이 프로젝트를 접하는 팀원, 관리직 임원, 협업자
> 목적: Magic Split 전략의 WFO 개선이 왜 필요했는지, `simulation(후보를 넓게 찾는 단계)` 이후 실제로 무엇을 하는지, 지금 어디까지 왔는지를 초심자도 이해하기 쉽게 공유하는 문서

## 1. 한 줄 요약
- 우리는 `좋아 보이는 파라미터(설정값 조합)`를 찾는 단계와 `정말 운영해도 되는지 확인하는 단계`를 분리하는 쪽으로 WFO 체계를 다시 만들고 있다.
- 쉽게 말하면, `문제집 많이 풀기`와 `진짜 시험 보기`를 구분하기 시작한 것이다.
- 지금은 큰 뼈대와 핵심 guardrail(실수 방지 장치)은 대부분 들어왔고, 남은 것은 `설명용 보고서 고도화`와 `approval-grade holdout(승인용 최종 등급 holdout, 정말 마지막 최종 시험지)` 마감에 가깝다.

## 2. 가장 먼저 이해해야 할 것
- 이 프로젝트에서 `simulation(후보를 넓게 찾는 단계)`은 `좋아 보이는 후보를 많이 찾는 과정`이다.
- 하지만 `simulation 결과가 좋다`와 `실제로 운영 승인할 수 있다`는 전혀 다른 말이다.
- 그래서 지금 구조는 아래처럼 나뉜다.

1. `simulation(후보를 넓게 찾는 단계)`
   - 후보를 많이 찾는다.
2. `research WFO(연구용 WFO, 시작 시점 흔들림을 보는 단계)`
   - 언제 시작하느냐에 따라 결과가 너무 크게 달라지지 않는지 본다.
3. `shortlist freeze(후보 고정)`
   - 후보를 좁히고, 이제부터는 답을 바꾸지 않겠다고 정한다.
4. `promotion WFO(승인 심사용 WFO, 더 엄격한 검증 단계)`
   - 시간이 앞으로 흘러가도 이 후보가 버티는지 다시 본다.
5. `CPU audit(CPU 기준 검산, 다시 최적화가 아니라 계산 확인)`
   - GPU가 고른 후보가 CPU 기준에서도 이상 없는지 확인한다.
6. `holdout(끝까지 남겨 둔 최종 시험 구간)`
   - 마지막으로 따로 남겨 둔 구간에서 최종 확인한다.

## 3. 아주 쉬운 비유
- `simulation`은 문제집을 많이 푸는 단계다.
- `research WFO`는 여러 날짜에 본 모의고사다.
- `promotion WFO`는 공식 모의평가다.
- `holdout`은 끝까지 안 풀고 남겨 둔 진짜 마지막 시험지다.

이 비유에서 중요한 점은 하나다.
- `문제집에서 1등` 했다고
- `진짜 시험에서도 통과`한 것은 아니다.

현재 길이 기준(2026-03-16 기준):
- `research WFO`
  - 현재 기본 OOS는 `거래일 기준 378일(약 18개월)`로 본다.
- `promotion WFO`
  - 현재 기본 OOS도 `거래일 기준 378일(약 18개월)`로 맞춘다.
- `holdout`
  - `24개월 이상`, 즉 `거래일 기준 약 504일 이상`을 목표로 둔다.
- 예전 `1년(365일)` OOS
  - 빠른 탐색용 최소선으로는 이해할 수 있지만, 현재 기본 운영값으로는 권장하지 않는다.

## 4. 초심자가 먼저 알아두면 좋은 실행 감각
- 이 프로젝트에서 자주 쓰는 실행 명령은 생각보다 단순하다.
- 핵심은 `명령어가 많이 다른 것`이 아니라, `어떤 설정 파일(config, 실행 설정 파일)을 주느냐`가 다르다는 점이다.

가장 자주 쓰는 기본 형태는 이것이다.

```bash
MAGICSPLIT_CONFIG_PATH=/root/projects/Split_Investment_Strategy_Optimizer/config/원하는설정파일.yaml \
CONDA_NO_PLUGINS=true conda run -n rapids-env \
python -m src.walk_forward_analyzer
```

초심자 식으로 풀면:
- `MAGICSPLIT_CONFIG_PATH`
  - 지금 어떤 설정 파일로 실행할지 정하는 환경변수(실행 전에 잠깐 넣는 값)다.
- `conda run -n rapids-env`
  - GPU 관련 라이브러리가 들어 있는 실행 환경으로 들어가서 실행하겠다는 뜻이다.
- `python -m src.walk_forward_analyzer`
  - WFO 엔진을 실행하는 명령이다.

즉:
- `research WFO`도 이 명령을 쓰고
- `promotion WFO`도 이 명령을 쓰고
- 차이는 대부분 `config 안의 lane_type(어떤 검증 경로인지)`과 각종 설정값에 있다.

## 5. simulation 이후에 실제로 무엇을 하나
이 부분이 초심자에게 가장 중요하다.  
아래 순서가 `simulation 다음 실제 WFO 흐름`이다.

### 5-1. Step 1. simulation 결과를 바로 믿지 않는다
- `simulation`은 넓게 보는 단계라서, 잘 보면 항상 좋아 보이는 후보가 몇 개는 나온다.
- 하지만 그중 일부는:
  - 특정 시작일에만 좋았을 수 있고
  - 특정 구간에서만 운 좋게 맞았을 수 있고
  - 실제 운영 기준으로는 너무 흔들릴 수 있다.
- 그래서 `simulation 직후`에 바로 운영 승인으로 가지 않는다.

이 단계에서 주로 쓰는 명령어:

```bash
CONDA_NO_PLUGINS=true conda run -n rapids-env \
python -m src.parameter_simulation_gpu
```

이 단계에서 특히 중요한 설정값:
- `backtest_settings.start_date`, `backtest_settings.end_date`
  - 어떤 기간을 놓고 후보를 찾을지 정한다.
- `backtest_settings.initial_cash`
  - 시작 자금(처음 넣는 돈)을 정한다.
- `parameter_space`
  - 어떤 파라미터 조합을 탐색할지 정한다.
- `strategy_params.price_basis`
  - 수정주가(adjusted, 권리락/액면분할 등을 반영한 가격) 기준인지 정한다.
- `strategy_params.universe_mode`
  - 어떤 종목 집합으로 시험할지 정한다.

초심자 포인트:
- 여기서는 `최종 1등`을 뽑는 것이 아니라
- `후보를 넓게 찾는 것`이 목적이다.

### 5-2. Step 2. research WFO로 "시작 시점 운빨"을 먼저 본다
- 여기서 묻는 질문은 이것이다.
- `이 후보가 특정 날짜에만 잘 맞는 것 아닌가?`
- 방법은 여러 시작점(anchor, 출발 날짜)을 두고 비슷한 검증을 반복하는 것이다.
- 이 단계는 `연구용`이다.
- 즉, 여기서 할 수 있는 말은:
  - `시작 시점이 달라도 비교적 덜 흔들렸다`
- 여기서 하면 안 되는 말은:
  - `이제 최종 승인됐다`

이 단계에서 주로 쓰는 명령어:

```bash
MAGICSPLIT_CONFIG_PATH=config/config.issue98_research_family_ab_top8_20260314_141954.yaml \
CONDA_NO_PLUGINS=true conda run -n rapids-env \
python -m src.walk_forward_analyzer
```

이 단계에서 특히 중요한 설정값:
- `walk_forward_settings.lane_type: research_start_date_robustness`
  - 연구용 WFO로 돌리겠다는 뜻이다.
- `walk_forward_settings.research_mode: frozen_shortlist_multi_anchor_eval`
  - 이미 만든 shortlist를 여러 시작점으로 반복 검증하겠다는 뜻이다.
- `walk_forward_settings.research_shortlist_path`
  - 어떤 shortlist 파일을 검증할지 적는다.
- `walk_forward_settings.research_anchor_start_dates`
  - 어떤 시작 날짜들로 반복 검증할지 적는다.
- `walk_forward_settings.decision_date`
  - 이 후보를 언제 기준으로 고정해 보기 시작했는지 남기는 날짜다.
- `walk_forward_settings.period_length_days`
  - 각 fold의 OOS 길이다.
  - 현재 Magic Split 기본값은 `period_length_basis=trading_days` 기준 `378일(약 18개월)`이다.
- `walk_forward_settings.period_length_basis`
  - `period_length_days`를 달력일로 읽을지, 거래일로 읽을지 정하는 값이다.
  - 현재 기본값은 `trading_days`다.

실행 후 자주 보는 파일:

```bash
LATEST_DIR=$(ls -td results/wfo_run_* | head -n 1)
sed -n '1,120p' "$LATEST_DIR/lane_manifest.json"
sed -n '1,120p' "$LATEST_DIR/anchor_manifest.json"
```

초심자 포인트:
- research WFO는 `좋은 후보를 더 이해하는 단계`다.
- `최종 승인 증거`를 만드는 단계가 아니다.
- 이 전략은 천천히 사고 천천히 정리하는 편이라, 현재는 `1년 OOS`보다 `18개월 OOS`를 기본값으로 두는 쪽이 더 자연스럽다.

### 5-3. Step 3. shortlist freeze로 후보를 고정한다
- research WFO를 보고 나면 후보를 줄인다.
- 이 줄인 목록을 `shortlist(압축된 후보 목록)`라고 부른다.
- 그리고 `freeze(고정)`를 한다는 뜻은:
  - 이제부터는 holdout 결과를 보고 후보를 다시 바꾸지 않겠다는 뜻이다.
- 아주 쉽게 말하면:
  - 답안지를 제출한 뒤에는 다시 고치지 않는 단계다.

이 단계에서 실제로 확인하는 것:

```bash
sed -n '1,20p' results/research_shortlists/SHORTLIST.csv
```

이 단계에서 중요한 설정값:
- `walk_forward_settings.promotion_shortlist_path`
  - promotion WFO에서 쓸 최종 shortlist 경로다.
- `walk_forward_settings.decision_date`
  - 이 shortlist를 언제 기준으로 고정했는지 적는다.
- `walk_forward_settings.shortlist_hash`
  - shortlist 내용이 나중에 바뀌지 않았는지 확인하는 데 쓰는 값이다.
- `walk_forward_settings.period_length_days`
  - promotion fold의 OOS 길이다.
  - 현재 기본 정책은 `period_length_basis=trading_days` 기준 `378일(약 18개월)`이다.

초심자 포인트:
- freeze는 단순 저장이 아니다.
- `이제부터는 답을 바꾸지 않겠다`는 운영 계약에 가깝다.

### 5-4. Step 4. promotion WFO로 "시간 전이"를 본다
- 이제 질문이 바뀐다.
- 여기서는:
  - `이 후보가 시간이 지나도 버티는가?`
  - `이 후보를 공식 심사 기준으로 다시 봐도 통과하는가?`
  - 를 본다.
- 이 단계는 `연구`가 아니라 `심사`에 가깝다.
- 그래서 promotion WFO는:
  - 새 후보를 다시 찾는 단계가 아니고
  - 이미 freeze된 shortlist만 놓고 다시 심사하는 단계다.

이 단계에서 주로 쓰는 명령어:

```bash
MAGICSPLIT_CONFIG_PATH=/root/projects/Split_Investment_Strategy_Optimizer/config/promotion_wfo_config.yaml \
CONDA_NO_PLUGINS=true conda run -n rapids-env \
python -m src.walk_forward_analyzer
```

이 단계에서 특히 중요한 설정값:
- `walk_forward_settings.lane_type: promotion_evaluation`
  - 승인 심사용 WFO로 돌리겠다는 뜻이다.
- `walk_forward_settings.promotion_mode: frozen_shortlist_single_anchor_eval`
  - freeze된 shortlist만 놓고 보겠다는 뜻이다.
- `walk_forward_settings.promotion_shortlist_path`
  - 심사에 쓸 후보 목록 파일이다.
- `walk_forward_settings.decision_date`
  - 후보 고정 기준일이다.
- `walk_forward_settings.selection_contract`
  - 어떤 기준으로 후보를 자르고 순위를 정할지 적는 계약이다.

실행 후 자주 보는 파일:

```bash
LATEST_DIR=$(ls -td results/wfo_run_* | head -n 1)
sed -n '1,40p' "$LATEST_DIR/promotion_candidate_summary.csv"
sed -n '1,220p' "$LATEST_DIR/final_candidate_manifest.json"
```

초심자 포인트:
- promotion WFO는 `다시 탐색`이 아니다.
- `이미 고른 후보를 공식 기준으로 심사`하는 단계다.

### 5-5. Step 5. CPU audit으로 계산을 다시 확인한다
- GPU는 빠르지만, 최종 후보는 CPU 기준에서도 한 번 더 확인해야 한다.
- 여기서 중요한 점은:
  - `CPU audit`은 다시 후보를 뽑는 단계가 아니다.
  - `계산이 제대로 되었는가`를 확인하는 단계다.
- 쉽게 말하면:
  - `답이 좋으냐`를 보는 게 아니라
  - `채점이 틀리지 않았느냐`를 보는 단계다.

이 단계에서 중요한 설정값:
- `walk_forward_settings.cpu_certification_enabled`
  - 켤지 끌지 정한다.
- `walk_forward_settings.cpu_certification_top_n`
  - CPU로 확인할 상위 후보 개수를 정한다.
- `walk_forward_settings.cpu_certification_metric`
  - 어떤 기준 점수를 중심으로 확인할지 정한다.

초심자 포인트:
- 최근 정리 방향은 `CPU가 GPU를 다시 이겨서 순위를 뒤집는 것`이 아니라
- `GPU가 고른 후보가 CPU 기준에서도 계산상 문제 없는지 확인`하는 쪽이다.

### 5-6. Step 6. holdout에서 마지막 시험을 본다
- holdout은 `끝까지 안 보고 남겨 둔 마지막 구간`이다.
- 여기서 확인하는 것은:
  - 이 후보가 정말 마지막 시험지에서도 버티는가
  - 너무 형식적으로만 좋은 숫자가 나온 것은 아닌가
  - 거래 수나 자금 투입 같은 기본 행동이 너무 빈약하지는 않은가
- 이 단계까지 와야 비로소 `최종 승인`에 가까운 이야기를 할 수 있다.

이 단계에서 주로 쓰는 명령어:

```bash
MAGICSPLIT_CONFIG_PATH=/root/projects/Split_Investment_Strategy_Optimizer/config/promotion_wfo_config.yaml \
CONDA_NO_PLUGINS=true conda run -n rapids-env \
python -m src.walk_forward_analyzer
```

초심자에게 헷갈릴 수 있지만:
- holdout도 보통 `같은 WFO 명령`으로 돈다.
- 대신 config 안에서 아래 값이 채워져 있어야 자동으로 이어진다.

특히 중요한 설정값:
- `walk_forward_settings.holdout_start`
  - holdout 시작일
- `walk_forward_settings.holdout_end`
  - holdout 종료일
- `walk_forward_settings.holdout_auto_execute`
  - promotion WFO가 끝난 뒤 holdout을 자동 실행할지
- `walk_forward_settings.canonical_holdout_start`
  - 공식 holdout 계약상 시작일
- `walk_forward_settings.canonical_holdout_end`
  - 공식 holdout 계약상 종료일
- `walk_forward_settings.holdout_adequacy_thresholds`
  - 거래 수나 자금 투입 같은 기본 충분성 기준이다.

실행 후 자주 보는 파일:

```bash
LATEST_DIR=$(ls -td results/wfo_run_* | head -n 1)
sed -n '1,240p' "$LATEST_DIR/holdout_manifest.json"
sed -n '1,240p' "$LATEST_DIR/final_candidate_manifest.json"
```

초심자 포인트:
- holdout은 `좋은 숫자 하나 더 뽑는 곳`이 아니다.
- `끝까지 남겨 둔 마지막 시험지`다.

## 6. baseline(비교 기준)은 어떻게 봐야 하나
- WFO와 holdout은 `이 전략 자체가 시험을 통과했는지` 보는 단계다.
- baseline은 `그 성적이 시장 대비 어느 정도 의미가 있는지`를 보는 비교표다.
- 즉:
  - `WFO/holdout = 시험`
  - `baseline = 반 평균과 비교`

현재 권장 baseline 역할 분담:
- `KOSDAQ buy-and-hold(같은 기간 코스닥 지수를 그냥 들고 있는 비교 기준)`
  - 주 baseline(주 비교 기준)
  - 이유:
    - 실제로 전략이 주로 코스닥 종목을 많이 사기 때문
- `KOSPI buy-and-hold(같은 기간 코스피 지수를 그냥 들고 있는 비교 기준)`
  - 보조 baseline(보조 비교 기준)
  - 이유:
    - 관리직이나 외부 설명에서 가장 익숙한 넓은 시장 기준이기 때문

중요한 원칙:
- baseline은 `설명용`으로는 매우 중요하다.
- 하지만 현재 v1에서는 `champion(최종 1등 후보)`을 다시 뽑는 공식 선택 규칙으로 쓰지는 않는다.
- 쉽게 말하면:
  - `후보 선정은 WFO 계약대로`
  - `의미 설명은 코스닥/코스피 비교표로`
  - 간다.

초심자 포인트:
- 주로 코스닥을 사는 전략이면 `코스피만` 비교하면 조금 억울할 수 있다.
- 그래서 `코스닥`이 주 기준이고, `코스피`는 넓은 시장과 비교하는 보조 기준으로 두는 것이 자연스럽다.

## 7. 왜 research WFO와 promotion WFO를 나눴나
- 두 단계가 묻는 질문이 다르기 때문이다.

### 7-1. research WFO가 묻는 질문
- `언제 시작하든 너무 심하게 흔들리지 않는가?`
- 즉, `시작 시점 민감도(언제 시작했느냐에 따라 결과가 달라지는 정도)`를 보는 쪽이다.

### 7-2. promotion WFO가 묻는 질문
- `시간이 앞으로 흐를 때도 이 후보가 버티는가?`
- 즉, `시간 전이 강건성(시간이 바뀌어도 버티는 정도)`을 보는 쪽이다.

### 7-3. 한 줄 정리
- `research`는 후보를 이해하는 단계다.
- `promotion`은 후보를 심사하는 단계다.

## 8. 왜 이 전략에서는 이 구분이 특히 더 중요하나
- Magic Split 전략은 한 번에 전액 투자하는 전략이 아니다.
- 조금 사고, 더 떨어지면 더 사고, 시간이 지나면 빠져나오는 구조다.
- 그래서 짧은 구간에서 잠깐 좋아 보이는 성과만 보면 실제 성격을 오해할 수 있다.

이 전략에서 더 중요한 질문은 이런 것들이다.
- `언제 시작해도 너무 심하게 흔들리지 않는가`
- `시간이 지나도 설명 가능한가`
- `실제 자본이 적절히 배치되고 회전했는가`

즉, 이 전략은 `가장 높은 숫자 하나`를 찾는 게임이 아니라, `오래 버티고 다시 설명 가능한 후보`를 찾는 문제에 더 가깝다.

## 9. 예전 방식에서 무엇이 문제였나
- 예전 구조는 `연구용 관찰`과 `승인용 검증`이 충분히 분리되어 있지 않았다.
- 그래서 아래 같은 오해가 생길 수 있었다.

- 같은 날짜가 여러 검증에 겹쳐 들어갈 수 있었다.
- OOS(Out-Of-Sample, 공부에 쓰지 않고 시험만 보는 구간) 결과를 평균 합성한 곡선이 실제 운용 계좌처럼 읽힐 수 있었다.
- `시작 시점이 달라도 안정적인가`와 `시간이 지나도 버티는가`가 같은 질문처럼 섞였다.
- holdout도 길이가 짧으면 `최종 승인용 마지막 증거`라고 부르기 어렵다.

## 10. 지금까지 실제로 구현된 것
### 10-1. lane(검증 경로) 분리의 뼈대
- `promotion_evaluation(승인 심사용 실행 모드)` 경로가 정리됐다.
- `research_start_date_robustness(시작 시점 민감도 연구 모드)` 경로가 정리됐다.
- 즉, 이제 코드도 예전처럼 “다 같은 WFO”가 아니라, 역할을 구분해 실행할 수 있게 됐다.

### 10-2. research WFO가 더 빨라졌다
- 최근에는 research WFO 안에서 `shortlist 후보를 하나씩 따로 GPU 실행`하던 구조를 줄였다.
- 이제 같은 fold(검증 한 구간) 안에서는 `후보 여러 개를 single GPU batch(한 번에 묶어서 처리)`로 평가하고, `market data(시장 데이터)` 준비도 재사용한다.
- 쉽게 말하면:
  - 예전에는 같은 문제지를 후보마다 매번 다시 펼쳐 봤다면
  - 지금은 한 번 펼쳐 놓고 후보들을 한꺼번에 채점하는 쪽으로 개선됐다.
- 이 변화는 `후보 선택 의미`를 바꾸는 변경이 아니라, `같은 의미를 더 효율적으로 계산`하는 쪽에 가깝다.

### 10-3. 결과를 더 정직하게 남기는 manifest(상태 기록 파일) 체계
- 결과 폴더에는 이제 아래 파일들이 남는다.
- `lane_manifest.json(이번 실행이 어떤 경로였는지 적는 파일)`
- `holdout_manifest.json(holdout 상태를 적는 파일)`
- `promotion_candidate_summary.csv(후보 요약표)`
- `final_candidate_manifest.json(최종 후보를 봉인한 기록 파일)`

이 파일들이 중요한 이유는:
- 어떤 lane으로 돌렸는지
- 어떤 후보를 썼는지
- holdout을 건드렸는지
- 외부에 설명해도 되는 단계인지
- 를 나중에 다시 증명할 수 있게 해 주기 때문이다.

### 10-4. final candidate(최종 후보) 선정 계약이 생겼다
- promotion WFO가 끝나면:
  - 후보 요약표를 만들고
  - `single champion(최종 1등 후보)` 1개를 정하고
  - `final_candidate_manifest.json`에 봉인한다.
- 이때 중요한 원칙은:
  - holdout에 여러 후보를 넣고 나중에 더 좋은 것을 고르지 않는다는 점이다.
- 쉽게 말하면:
  - 마지막 시험지로 답을 다시 고르는 일을 막는 구조다.

### 10-5. freeze contract(후보 고정 계약)가 들어갔다
- 이제는 후보를 고른 뒤 아래를 다시 확인한다.
- shortlist 파일이 `decision_date(후보 고정 기준일)` 이후 바뀌지 않았는가
- holdout 경계가 공식 계약과 맞는가
- 맞지 않으면 holdout 자동 실행 전에 막는다.

즉, 단순히 `좋아 보이는 숫자`만 저장하는 것이 아니라,
- `후보를 언제 고정했는가`
- `중간에 몰래 바꾸지 않았는가`
- 도 함께 관리하기 시작했다.

### 10-6. holdout guardrail(holdout 실수 방지 장치)이 강화됐다
- `holdout_auto_execute=true(holdout 자동 실행 옵션)`일 때도 아무 때나 holdout이 도는 것이 아니다.
- champion(최종 1등 후보)이 준비되어 있고
- CPU audit이 통과했고
- freeze 계약과 holdout 계약이 맞을 때만
- holdout을 자동 실행하게 정리했다.

또 holdout 결과도 더 정직하게 기록한다.
- `attempted(실행 시도했는가)`
- `success(끝까지 성공했는가)`
- `blocked(규칙 때문에 막혔는가)`

그리고 holdout이 너무 형식적으로 끝나지 않도록 아래도 본다.
- `trade_count(거래 횟수)`
- `closed_trade_count(완결된 거래 횟수)`
- `distinct_entry_months(서로 다른 진입 월 수)`
- `avg_invested_capital_ratio(평균 투자 자본 비율)`
- `cash_drag_ratio(현금이 놀고 있던 비율)`

쉽게 말하면:
- 수익률 숫자만 보는 것이 아니라
- `실제로 전략이 의미 있게 움직였는가`도 같이 보기 시작한 것이다.

## 11. 지금 상태를 가장 쉽게 말하면
- `WFO의 큰 흐름은 많이 정리됐다.`
- `후보 탐색`, `연구용 검증`, `승인 심사`, `최종 시험`이 예전보다 훨씬 분리되었다.
- `후보 고정`, `최종 후보 봉인`, `holdout 차단`, `기본 행동 충분성 확인`도 코드에 많이 들어왔다.
- 하지만 아직 `정말 대외 설명 가능한 마지막 승인 보고서`까지는 완전히 닫히지 않았다.

## 11-1. 최신 실행 스냅샷 (2026-03-24)
기준 실행 폴더:
- `results/wfo_run_20260323_215211`

이번 실행에서 확인된 것:
- `promotion WFO`에서 `candidate 6`이 최종 후보(champion)로 선택되었다.
핵심 심사 기준(이번 실행):
- `fold 통과율`: `3개 중 2개 통과(2/3)`
- `MDD P95`: `0.409` (완화 기준 `<= 0.41` 충족)
- `CPU audit(계산 검산)`은 `pass`로 통과했다.
- `holdout 자동 실행`도 실제로 수행되었고, 기술적으로는 성공했다(`attempted=true`, `success=true`).

하지만 아직 최종 승인(approval-grade)이 아닌 이유:
- 현재 holdout 길이가 `221 거래일`이라서
- 정책 최소치 `504 거래일(약 24개월)`에 못 미친다.
- 그래서 현재 등급은 `internal_provisional(내부 잠정)`이며
- `approval_eligible=false`, `external_claim_eligible=false` 상태다.

초심자 한 줄 요약:
- `엔진은 제대로 동작했고 후보도 골랐지만, 마지막 시험지 길이가 짧아서 아직 공식 합격은 아니다.`

## 11-2. 최신 승인 스냅샷 (2026-03-24, update)
기준 실행 폴더:
- `results/wfo_run_20260324_125326`

이번 실행에서 확인된 것:
- `promotion WFO`에서 `candidate 3`이 최종 후보(champion)로 선택되었다.
- 하드게이트 통과 후보가 실제로 존재했고(`candidate 3`, `candidate 1`), tie-break 규칙으로 최종 1개가 확정되었다.
- `CPU audit(계산 검산)`은 `pass`였다.
- `holdout`은 실제 실행되었고(`attempted=true`, `success=true`) 차단 없이 완료되었다.

최종 판정:
- `lane_manifest`: `approval_eligible=true`, `external_claim_eligible=true`
- `holdout_manifest`: `internal_approval_ready`, `approval_eligible=true`, `external_claim_eligible=true`
- 즉, 이번 run은 현재 계약 기준으로 `approval-grade(승인 등급)` 판정을 받았다.

다만 성과 해석 주의:
- holdout 지표는 `CAGR 약 1.32%`, `MDD 약 -34.4%`, `Calmar 약 0.038`로, 절차 합격과 투자 매력도는 별개로 해석해야 한다.

## 12. 아직 남은 핵심 작업
### 12-1. 행동지표(behavior metrics, 전략이 실제로 어떻게 움직였는지 보여 주는 지표) 보고서 고도화
- 지금도 기본 행동지표는 보지만, 이것을 더 설득력 있는 설명용 보고서로 키우는 작업이 남아 있다.
- 예를 들면:
  - 어떤 후보가 왜 더 안정적인지
  - 어느 지점에서 자금 배치가 더 나았는지
  - 파라미터를 조금 바꿨을 때도 성격이 유지되는지
- 를 더 쉽게 보여 주는 비교 보고서가 필요하다.

### 12-2. approval-grade holdout 마감
- 이번 최신 run(`wfo_run_20260324_125326`)에서 현재 계약 기준 `approval-grade`를 달성했다.
- 즉, `절차 합격` 관점에서는 holdout 마감이 완료된 상태다.
- 다만 남은 과제는 `성과 품질 기준(절대 수익/위험 기준)`을 정책에 추가할지 여부다.

## 13. 경영 관점에서 보면 지금 어디까지 왔나
### 13-1. 이미 해결한 것
- 후보 탐색과 승인 심사를 같은 것으로 보던 문제를 구조적으로 분리했다.
- 연구용 결과가 승인용처럼 읽힐 위험을 많이 줄였다.
- 결과를 나중에 추적하고 감사할 수 있도록 상태 기록 파일을 남기기 시작했다.
- 최종 후보를 봉인하는 절차가 생겼다.
- holdout이 짧거나, 너무 형식적으로 끝났을 때 자동으로 경고하거나 강등하는 구조가 들어왔다.

### 13-2. 아직 남은 것
- 행동지표와 비교 실험(ablation, 일부 조건을 바꿔 보며 설명력을 높이는 실험) 보고서 고도화
- 절차 합격 이후의 성과 품질 기준(예: 최소 Calmar/CAGR) 정책화

### 13-3. 이 작업의 의미
- 이건 단순히 성능 숫자를 조금 올리는 작업이 아니다.
- `좋아 보인다`를
- `왜 좋은지 다시 설명할 수 있고, 나중에도 같은 기준으로 다시 검증할 수 있다`
- 로 바꾸는 작업이다.

## 14. 팀에 아주 짧게 설명하면
- `simulation은 후보를 찾는 단계다.`
- `research WFO는 시작일 운빨이 아닌지 보는 단계다.`
- `promotion WFO는 시간 전이 검증을 하는 심사 단계다.`
- `CPU audit은 계산 검산 단계다.`
- `holdout은 끝까지 남겨 둔 마지막 시험지다.`
- `우리는 지금 이 다섯 단계를 섞지 않도록 구조를 다시 세우는 중이다.`

## 15. 참고 문서
- [WFO Approval Workflow Runbook](/root/projects/Split_Investment_Strategy_Optimizer/docs/operations/2026-03-14-wfo-approval-runbook.md)
- [Issue #68: Robust WFO / Ablation](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_02_09-issue68-robust-wfo-ablation.md)
- [WFO / OOS Lane 임시 합의안](/root/projects/Split_Investment_Strategy_Optimizer/todos/2026_03_12-wfo-oos-lane-provisional-review.md)

## 16. approval-grade로 닫는 실행 체크리스트 (2년 holdout, 초심자용)
아래 순서는 `내 터미널(WSL)`에서 그대로 따라 하기 쉽게 정리한 것이다.

### 16-1. 먼저 확인: 2년 holdout 거래일 수(거래가 실제 있었던 날짜 수)
```bash
MYSQL_PWD='@waren2ss' mysql -uroot -h127.0.0.1 stocks -Nse \
"SELECT COUNT(DISTINCT date) AS trading_days \
 FROM DailyStockPrice \
 WHERE date BETWEEN '2024-01-01' AND '2025-12-31';"
```

해석:
- 출력 숫자가 `holdout_min_length_days` 이상이어야 `holdout_too_short`를 피할 수 있다.
- 현재 기본 정책(`trading_days`)은 보통 `504`일을 기준으로 본다.

### 16-2. 실행용 config(이미 생성됨)
이미 아래 파일이 준비되어 있다:
- `config/config.issue98_promotion_dual_window_core6_approval_holdout2y_20260324.yaml`

필요하면 이 파일만 열어서 값 점검:
- `backtest_settings.end_date: '2023-12-31'` (promotion WFO 종료일과 맞춰야 holdout 분리가 성립)
- `walk_forward_settings.research_data_cutoff: '2023-12-31'`
- `walk_forward_settings.promotion_data_cutoff: '2023-12-31'`
- `walk_forward_settings.canonical_promotion_wfo_end: '2023-12-31'`
- `walk_forward_settings.canonical_holdout_start: '2024-01-01'`
- `walk_forward_settings.canonical_holdout_end: '2025-12-31'`
- `walk_forward_settings.holdout_start: '2024-01-01'`
- `walk_forward_settings.holdout_end: '2025-12-31'`
- `walk_forward_settings.holdout_min_length_days: 486` (이 DB 기준 2년 거래일 하한)
- `walk_forward_settings.selection_contract.min_oos_is_calmar_ratio_median: 0.5`
- `walk_forward_settings.selection_contract.max_oos_mdd_depth_p95: 0.435`

주의:
- `holdout_contaminated_ranges(오염 구간)`이 holdout과 겹치면 자동 차단될 수 있다.
- 이 값은 팀 정책(거버넌스, 운영 규칙) 결정 후에만 수정한다.

### 16-3. 실행
```bash
MAGICSPLIT_CONFIG_PATH=/root/projects/Split_Investment_Strategy_Optimizer/config/config.issue98_promotion_dual_window_core6_approval_holdout2y_20260324.yaml \
CONDA_NO_PLUGINS=true conda run -n rapids-env \
python -m src.walk_forward_analyzer
```

### 16-4. 실행 직후 확인(가장 중요한 4줄)
```bash
LATEST_DIR=$(ls -td results/wfo_run_* | head -n 1)
echo "$LATEST_DIR"
sed -n '1,220p' "$LATEST_DIR/final_candidate_manifest.json"
sed -n '1,220p' "$LATEST_DIR/holdout_manifest.json"
```

통과 판단(초심자 기준):
- `final_candidate_manifest.json`
- `cpu_audit_outcome == "pass"`
- `holdout_execution_status == "executed"`
- `holdout_success == true`
- `holdout_manifest.json`
- `approval_eligible == true`
- `external_claim_eligible == true`
- `reasons`가 비어 있거나, 승인 차단 사유가 없어야 한다.

### 16-5. 실패 시 바로 보는 포인트
- `holdout_too_short=...`
- holdout 기간(또는 최소 길이 정책) 재검토 필요
- `holdout_range_contaminated`
- holdout 구간과 canary/오염 구간이 겹친 상태
- `missing_adequacy_fields=...` 또는 adequacy 실패
- 거래/자본 투입 충분성 지표가 비어 있거나 기준 미달
