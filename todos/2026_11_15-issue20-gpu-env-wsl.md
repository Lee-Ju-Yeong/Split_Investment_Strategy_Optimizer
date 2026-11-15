# #20 GPU 검사 환경 확인: WSL2 + GeForce RTX 5060

> Type: `implementation`
> Status: `GPU 검사 2개 모듈 실행 완료 (34개 중 31개 통과, 3개는 성능 측정 원본 파일 부재로 실패)`
> Last updated: 2026-11-15 (Asia/Seoul)
> Why: CPU·GPU 결과 100% 일치 원칙을 지키려면 GPU 쪽 수정도 실제 GPU에서 검사해야 한다. 고정 차수 규칙(`fixed_slot_v1`, 별도 브랜치 `feat/fixed-split-slot-spec`의 `todos/2026_11_15-fixed-split-slot-v1.md`) 작업의 GPU 정합성 확인에 쓸 환경을 확보한다.
> Next Action: 성능 측정 원본이 없는 환경에서 3개 검사를 어떻게 다룰지(건너뜀 처리 또는 원본 공개) 결정. GPU 검사 명령을 `tests/README.md`에 반영할지 결정.
> Gate Status: GPU 계산 검사는 실제 GPU에서 통과. 성능 회귀 기준 검사 3개는 이 환경에서 판정 불가. 전체 GPU 탐색·WFO·성능 재측정은 실행하지 않음.
> Related Issues: [#20](https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer/issues/20)

## 결론

- **새로 설치한 것은 없다.** WSL Ubuntu 안에 이미 `rapids-env`(conda 환경)가 있었고, CUDA 12.9 런타임과 CuPy 13.5.1·cuDF 25.06이 RTX 5060에서 그대로 동작했다.
- `tests/test_backtest_strategy_gpu.py`: 21개 중 18개 통과, 3개 실패. 실패 3개는 모두 Git에서 제외된 `results/issue98_measure/.../summary.json` 성능 측정 원본이 없어서 생긴 것이다. GPU 계산 결과가 기대값과 달라서 실패한 검사는 없다.
- `tests/test_cpu_gpu_parity_topk.py`: 13개 모두 통과. CuPy 장치를 실제로 잡는 검사도 건너뛰지 않고 실행됐다.
- 두 모듈 모두 DB 연결이 필요한 검사는 없다. 건너뛴(`skipped`) 검사는 0개다.

## 장비와 소프트웨어

| 항목 | 값 | 확인 방법 |
| --- | --- | --- |
| OS | Windows 11 Pro 10.0.26100 + WSL2 `Ubuntu` 24.04.2 LTS, 커널 6.6.87.2-microsoft-standard-WSL2 | `wsl.exe -l -v`, `/etc/os-release`, `uname -r` |
| GPU | NVIDIA GeForce RTX 5060, 8 GB (데스크톱 화면이 약 1.6 GB 사용 중) | WSL 안 `nvidia-smi` |
| 드라이버 | Windows 드라이버 581.80 (WSL 안 `nvidia-smi` 580.105.07 표기), 드라이버가 지원하는 최대 CUDA 13.0 | `nvidia-smi` |
| compute capability | 12.0 (`sm_120`, Blackwell) | `cupy.cuda.Device(0).compute_capability` → `120` |
| conda | Miniconda `conda 25.5.1`, 환경 이름 `rapids-env` | `conda env list` |
| Python | 3.10.18 (conda-forge) | `python --version` |
| CUDA 런타임 | 12.9 (`cuda-version 12.9`, `runtimeGetVersion()` → `12090`) | `conda list`, CuPy |
| 주요 패키지 | CuPy 13.5.1, cuDF 25.06.00, numba 0.61.2, numba-cuda 0.11.0, numpy 1.26.4, pandas 2.2.3, PyYAML 6.0.2, PyMySQL 1.1.1, SQLAlchemy 2.0.42 | `conda list -n rapids-env` |

전체 목록은 [패키지 스냅샷](../docs/environments/rapids-env-wsl-rtx5060.yml)에 있다. 이 스냅샷으로 새 환경을 만들어 본 적은 없으므로, 그대로 재생성된다고 보장하지 않는다.

`conda env export --from-history`로 본 최초 설치 요청은 `python=3.10`, `cupy`, `cudf`, `dask-cuda`, `mysql-connector-python`, `seaborn`이다. 채널은 `rapidsai`, `conda-forge`, `defaults` 순이다. 이 환경을 언제 어떤 명령으로 만들었는지는 기록이 없다.

### RTX 5060에서 이 조합이 맞는 근거

- NVIDIA [CUDA 12.8 Release Notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html): CUDA 12.8부터 Blackwell 대상 `sm_100`·`sm_101`·`sm_120` 컴파일을 지원한다. 이 환경은 12.9라 조건을 만족한다.
- NVIDIA [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus): GeForce RTX 50 시리즈는 compute capability 12.0이다.
- 실제 확인: CuPy `ElementwiseKernel`과 직접 작성한 `RawKernel`을 이 GPU에서 컴파일·실행해 기대값을 얻었고, cuDF `Series.sum()`도 동작했다. 드라이버 측 CUDA 버전(13.0)이 런타임(12.9)보다 높으므로 하위 호환 조건도 맞는다.

## 실행 명령과 결과

검사 대상 커밋: `05bf3de` (main과 같은 내용의 작업 트리). Windows 쪽 저장소 폴더를 WSL에서 `/mnt/c/...`로 열어 실행했다. 두 모듈은 `unittest` 기반이라 `pytest` 없이 실행했다(이 환경에는 `pytest`가 없다).

```bash
# Windows에서 WSL 진입
wsl.exe -d Ubuntu
```

```bash
# WSL 안, 저장소 루트에서
cd <저장소 경로>
/root/miniconda3/envs/rapids-env/bin/python -m unittest -v tests.test_backtest_strategy_gpu
/root/miniconda3/envs/rapids-env/bin/python -m unittest -v tests.test_cpu_gpu_parity_topk
```

`conda activate rapids-env` 후 `python -m unittest ...`로 실행해도 같다. 이 PC의 WSL 기본 셸에는 conda가 PATH에 잡혀 있지 않아 위처럼 환경의 Python을 직접 지정했다.

| 모듈 | 실행 | 통과 | 실패 | 건너뜀 | 소요 |
| --- | --- | --- | --- | --- | --- |
| `tests.test_backtest_strategy_gpu` | 21 | 18 | 3 | 0 | 검사 6.1초 (프로세스 12.6초) |
| `tests.test_cpu_gpu_parity_topk` | 13 | 13 | 0 | 0 | 검사 1.2초 (프로세스 3.3초) |

### 실패 3개: 성능 측정 원본 부재

`TestIssue98PerfRegressionGuard`의 아래 3개 검사는 #98 성능 측정 때 만든 `results/issue98_measure/` 아래 `summary.json`을 읽는다.

- `test_pr98d_budget_guard_accepts_baseline_artifact`
- `test_pr98d_budget_guard_detects_slice2a_artifact_regression`
- `test_pr98d_budget_guard_detects_slice1only_rollback_artifact_regression`

```text
AssertionError: False is not true : missing perf artifact: <저장소>/results/issue98_measure/pr98c_slice2_janfeb2024_cov020_20260308_131130/summary.json
```

`results/`는 `.gitignore` 대상이고, [실행 안내](../docs/running.md)에도 당시 `results/` 원본이 공개 저장소에 없다고 적혀 있다. 이 PC의 원본 저장소 폴더에도 해당 파일이 없다. 따라서 이 3개는 **GPU 계산 문제가 아니라 입력 파일이 없는 환경 문제**이며, 성능 회귀 기준은 이 환경에서 판정하지 못했다. 같은 클래스의 나머지 2개(원본 파일 없이 만든 값으로 판정하는 검사)는 통과했다.

검사 코드는 이번에 바꾸지 않았다. 원본이 없을 때 `skipTest`로 처리할지, 원본 일부를 공개할지는 별도 결정이 필요하다.

## 이번에 하지 않은 것

- 패키지 설치·업그레이드, 새 환경 생성, `rapids-env` 변경.
- 전략·체결 코드(`src/backtest/`)와 검사 코드 수정.
- DB가 필요한 GPU 경로(`src.debug_gpu_single_run`, `src.parameter_simulation_gpu`, `src.walk_forward_analyzer`), 전체 `test_*.py` 실행, 성능 재측정.
- 스냅샷 파일로 새 환경을 만들어 재현되는지 확인.

## Windows에서 WSL 명령을 넘길 때 주의

Claude Code 등 Windows 쪽 도구에서 WSL로 명령을 넘길 때 겪은 문제다.

- `wsl.exe -- bash -lc '... $변수 ...'`는 WSL 기본 셸이 한 번 더 해석해 `$변수`가 빈 값이 될 수 있다. 셸 스크립트 파일을 만들어 `wsl.exe -d Ubuntu --exec bash /mnt/c/.../script.sh`로 실행하면 안전하다.
- Git Bash에서 `/mnt/c/...` 경로를 넘기면 Windows 경로로 바뀐다. `MSYS_NO_PATHCONV=1`을 앞에 붙인다.
