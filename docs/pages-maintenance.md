# 문서 사이트 확인과 유지 방법

[공개 문서 사이트](https://lee-ju-yeong.github.io/Split_Investment_Strategy_Optimizer/)는 저장소의 Markdown을 Jekyll로 게시합니다. README를 별도 홈페이지 원고로 복사하지 않습니다. main/root를 게시하는 기존 GitHub Pages 설정과 자동 배포를 유지합니다.

## 변경 위치

| 바꿀 내용 | 원본 |
| --- | --- |
| 소개·구현 사례 | [README](../README.md), [구현 근거](engineering-evidence.md) |
| 한국어·사이트 제목·기본 주소 | `_config.yml` |
| 공통 메뉴·본문 바로가기·문서 제목 구조 | `_layouts/default.html` |
| 글 폭·색·표·작은 화면·초점 표시 | `assets/css/project.css` |
| 데이터·계산 흐름 그림 | `assets/images/data-flow.svg`와 README의 텍스트 설명 |

본문 중심의 흰 바탕과 짙은 글자, 청색 링크, 녹색 키보드 초점을 사용합니다. 본문은 최대 54rem로 제한하고 한국어 시스템 글꼴을 사용해 외부 글꼴 요청을 추가하지 않습니다. 메뉴는 작은 화면에서 줄을 바꾸며 긴 표·코드는 해당 영역 안에서 가로로 읽습니다. 한 문서의 제목 H1은 하나만 표시합니다.

## Docs preview가 확인하는 범위

[실행 기록](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/actions/workflows/docs-preview.yml)에서 후보 SHA·실제 checkout SHA·Python·Playwright·Chromium 버전과 결과를 확인할 수 있습니다.

- main 대상 PR·main push 중 Markdown, Pages 설정·레이아웃·자료, 문서 검사 코드·의존성·workflow가 바뀔 때 실행합니다. 수동 실행도 가능합니다.
- 무료 Ubuntu 24.04에서 기존 Pages와 같은 Jekyll 빌드 액션을 사용합니다. 공식 액션은 커밋으로 고정합니다.
- 소개·구현 근거·직접 확인·문서 안내·소스 안내·전체 실행 안내·DB 검사 안내의 **7개 문서**를 확인합니다.
- Chromium의 **1280px·390px** 화면에서 내부 링크·앵커·자료, 한국어 메타데이터, H1·제목 단계, 공통 메뉴, 전체 화면 가로 넘침을 확인합니다. Tab 이동, 초점 표시, 본문 바로가기도 확인합니다.
- main에서는 기존 Pages 배포가 이번 병합 SHA를 게시할 때까지 최대 90초 기다린 후 실제 공개 주소에도 같은 검사를 수행합니다. 배포 지연·접속 실패는 성공으로 바꾸지 않습니다.
- 작업은 10분, 화면 검사 단계는 각각 3분 제한입니다. 배포 권한·별도 미리보기 사이트·캐시·산출물 업로드는 사용하지 않습니다.

전용 환경에서 Jekyll 생성물이 준비됐을 때의 실제 명령입니다. 사용자 PC에 브라우저를 설치하는 것을 기본 절차로 삼지 않습니다.

```bash
python tools/ci/check_docs.py --site-dir _site
```

전체 과거 문서·외부 GitHub 링크·모든 브라우저·화면낭독기의 전수 검사는 아닙니다. 코드·CPU·DB·GPU 검사와 문서 화면 검사를 구분하며 실제 게시 성공은 기존 `pages build and deployment` 기록도 함께 확인합니다.

## 실패할 때

Jekyll 빌드 실패는 원본 설정·문법을, 화면 검사 실패는 해당 URL·화면 폭·오류를 확인합니다. 배포가 늦으면 공개 페이지의 `siso-revision` 메타데이터와 Pages 실행을 확인합니다. 이전 배포 화면을 새 후보의 성공으로 기록하지 않습니다. 화면 변경을 검증하지 못하면 PR에 보존하고, 병합 후 회귀가 확인되면 관련 변경을 되돌립니다.
