# 공개 저장소 안내 정비

> Type: `implementation`
> Status: `documentation and example validated; publication tracked in PR #109`
> Last updated: 2026-11-06 (Asia/Seoul)
> Why: 처음 방문한 사람이 프로젝트 목적, 개인 기여, 구현과 검증 근거를 순서대로 확인하도록 한다.
> Next Action: 공개 반영의 최종 SHA·자동 검사·About·Pages 결과는 아래 PR에서 추적한다.
> Gate Status: 예제·기존 검사 34개·문서 링크 확인 완료. 제품 전체 검증·운영 승인을 뜻하지 않는다.
> Evidence Links: 아래 실행 기록과 [PR #109](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/pull/109)
> Related Issues: 기존 구현 근거 #100, #104/#105, #106. 이 정비는 해당 제품 작업을 재개하지 않는다.

## 승인한 실행 계약

- 기준 main: `ea5683405a5c2e0e55a2f2365f4a1e0ff6022dd3`.
- 목적: 국내 주식 분할투자 시뮬레이션이라는 본래 목적과 개발 이력을 살려 목적 → 기여 → 구현 → 코드·테스트·변경 기록의 방문 동선을 만든다.
- 범위: README·문서·상태·링크, 기존 검사 실행 안내, 외부 연결 없는 작은 CPU 예제와 실행 의존성.
- 개인 기여: 사용자가 설계·구현·테스트·운영을 주도했다고 공개하는 범위를 확인했다. 직함·연차는 추정하지 않는다.
- 공개 권한: 검증 후 PR 일반 merge와 기존 Pages 공개 반영, About 설명 변경. Bio는 문구만 준비한다.
- 재량: 되돌릴 수 있는 문구·배치·링크·파일명·예제 출력 형식은 목적에 맞춰 결정한다.
- 제외: 제품 매매 규칙·GPU 커널·DB 스키마·운영 데이터·공개 범위·보안 설정 변경, 이력 재작성, 다른 작업 덮어쓰기, 새 CI 추가.
- 필수 확인: 안내 명령의 실제 실행, 예제 결과, 문서 링크/상태/수치, 연결 파일 보존, 최종 diff, Secret Scan, 병합 후 공개 동선과 Pages 배포.
- 중단·재개: 인증·검사·충돌로 한 항목이 막혀도 독립 작업은 계속한다. 승인 밖 제품 수정은 보류하고, 원인·완료 범위·재개 방법을 남긴다. 병합 불가 시 Draft PR/브랜치, 푸시 불가 시 로컬 커밋에 보존한다.
- 완료: 승인 범위의 구현·검증·공개 반영과 결과 보고. 외부 환경 미확인이나 실행 중단을 제품 검증 성공으로 쓰지 않는다.

## 공개 문구

- About: 국내 주식 분할투자 전략 시뮬레이션 · Python/MySQL 데이터 수집·검증 · CPU 기준 검증과 GPU 병렬 계산
- Bio 준비 문구: AIOps와 운영 자동화에 관심이 있습니다.
- Bio는 계정에 적용하지 않는다.

## 변경 순서와 확인 기준

1. [x] 별도 checkout의 origin/main/SHA/미커밋 상태와 지침 확인.
2. [x] README, 문서 지도, 세 구현 근거, 실행 안내 작성.
3. [x] TODO·AI 참고 문서·과거 문서의 상태와 링크 정비. 기존 경로 유지.
4. [x] 가상 데이터 예제와 확인용 의존성 추가. 기존 제품 함수 호출, DB/API/GPU 사용 없음.
5. [x] 세 기존 검사와 예제 실행, 링크·수치·연결 파일·diff 확인.
6. PR 생성 완료. 최종 후보의 Secret Scan과 일반 merge 결과는 PR #109에 기록한다.
7. About·Pages·로그아웃 방문 동선의 게시 후 확인 결과도 PR #109에 기록한다. 이 파일의 오프라인 검사 결과와 구분한다.

## 실행 기록

- 2026-11-06: main이 조사 기준 SHA와 같고 checkout이 깨끗함을 확인. 세 AI 지침 파일은 `llm.md` 심볼릭 링크다. 별도 `codex/public-project-refresh` 브랜치에서 시작했다.
- 검증 환경: 원격 Linux, Python 3.12.3. 이 정비 전에는 제품 검사를 실행하지 않았다.
- 기존 자동 검사는 Secret Scan과 Pages 배포다. 제품 테스트 전체를 수행하는 CI가 아니다.
- 내용·예제 후보 커밋: `6323d986f6fcc91c4a2225c8fb6ccafaca422fe2`. 이후 공개 결과 링크를 추가한 문서 변경은 예제·검사·의존성을 바꾸지 않는다.

### 안내 명령 검증

저장소 루트에서 `python3.12 -m venv .venv`와 `.venv/bin/python -m pip install -r requirements-review.txt`로 독립 환경을 만들었다. 아래 `python`은 이 가상환경의 실행 파일이다. 예제와 세 검사는 독립 프로세스로 병렬 실행했다.

| 명령 | 결과 | 종료 코드 |
| --- | --- | --- |
| `python -m examples.short_selling_lag_demo` | 두 반영 방식의 날짜·값 기대 검사 통과. 2024-01-02 값 0.03이 지연 모드에서 2024-01-05에만 반영 | 0 |
| `python -m unittest discover -s tests -p 'test_portfolio.py' -v` | 8개 통과, 실패·건너뜀 0 | 0 |
| `python -m unittest discover -s tests -p 'test_short_selling_collector.py' -v` | 7개 통과, 실패·건너뜀 0. 실패 상황의 로그는 모의 요청으로 의도적으로 발생 | 0 |
| `python -m unittest discover -s tests -p 'test_daily_stock_tier_batch.py' -v` | 19개 통과, 실패·건너뜀 0 | 0 |
| `python -m pip check` | 의존성 충돌 없음 | 0 |

실제 설치 조합은 다음과 같다. 상위 세 라이브러리는 확인용 의존성 파일에 고정했으며, 이 목록은 당시 간접 의존성까지 포함한 기록이다.

```text
certifi==2026.10.23
charset-normalizer==3.5.2
idna==3.20
numpy==2.2.6
pandas==2.2.3
python-dateutil==4.0.0
pytz==2026.6
requests==2.32.5
tzdata==2026.6
urllib3==2.8.0
```

### 문서·변경 경계 확인

- 문서 65개의 Markdown 링크 232개에서 로컬 대상·문서 앵커와 저장소 코드 경로를 검사했다. 끊어진 대상 0개. 인터넷의 모든 외부 문서를 재검증했다는 뜻은 아니다.
- 작성자 PC를 가리키던 탐색 링크 37개를 상대 문서 링크나 기준 커밋의 GitHub 코드 링크로 고쳤다. 과거 명령·로그의 경로는 일괄 치환하지 않았다.
- 대표 수치는 PR #105·관련 측정 기록, 공매도 영향 분석 기록과 대조했다. 원본 결과 파일·장비 정보의 공개 여부와 재실행하지 않은 범위를 명시했다.
- `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`는 `llm.md`를 가리키는 심볼릭 링크(`120000`)를 유지한다.
- `src/`, 기존 `tests/*.py`, `config/`, 기존 환경 파일과 `.github/`의 diff는 없다. 새 실행 파일은 `examples/`와 `requirements-review.txt`뿐이다.
- `git diff --check` 통과. 문서와 기존 함수를 호출하는 예제 범위여서 별도 제품 로직 리뷰는 수행하지 않았다.

### 발견했지만 이번에 수정하지 않은 항목

- `environment.yml`은 Python 3.9를 지정하지만 현재 일부 코드의 문법은 더 새로운 Python을 요구한다. 확인용 Python 3.12 환경과 구분해 안내했다.
- YAML 설정과 `src.db_setup`의 INI 설정이 다르다. 각 설정 경로를 실행 안내에 분리했다.
- `tests/test_integration.py`의 `get_db_connection(cls.db_config)`는 인자를 받지 않는 실제 함수와 맞지 않는다. 준비 단계 예외가 건너뜀으로 바뀔 수 있어 검증 완료로 소개하지 않았다. 제품·기존 테스트 수정은 승인 범위 밖이다.
- GPU·DB·인증이 필요한 작업은 아래 조건을 갖춘 별도 작업에서 재개한다. 이 제한으로 문서·오프라인 예제 작업을 중단하지 않았다.

## 미실행 범위

- GPU 전체 검사·성능 재측정: CUDA/RAPIDS·대상 GPU와 당시 입력/설정/원본 로그가 필요하다.
- KRX 실수집·MySQL 통합 검사·장기 데이터 재계산: 별도 인증·검사용 DB·원천 데이터가 필요하다. 이번 승인 범위에는 운영 연결·데이터 변경이 없다.
- Windows/macOS 실행: 이번 실행 환경에서 확인하지 않는다.
