# Parity Gate Report by Universe Mode (2026-03-06)

## 목적
- `strict_pit`와 `optimistic_survivor`를 분리 트랙으로 검증하고, CPU-GPU parity 게이트 증적을 남긴다.

## 실행 조건
- 환경: `conda activate rapids-env`
- 구간: `2021-01-01 ~ 2021-03-31`
- parity mode: `strict`
- top-k 기준 CSV: `results/standalone_simulation_results_20260302_012840.csv` (`sort-by cagr`)

## 실행 명령
```bash
# optimistic_survivor
MAGICSPLIT_UNIVERSE_MODE=optimistic_survivor python -m src.cpu_gpu_parity_topk \
  --start-date 20210101 --end-date 20210331 --topk 1 --parity-mode strict \
  --out /tmp/parity_report_top1_optimistic_survivor_latest.json

MAGICSPLIT_UNIVERSE_MODE=optimistic_survivor python -m src.cpu_gpu_parity_topk \
  --start-date 20210101 --end-date 20210331 \
  --params-csv results/standalone_simulation_results_20260302_012840.csv \
  --sort-by cagr --topk 5 --parity-mode strict \
  --out /tmp/parity_report_top5_optimistic_survivor_latest.json

MAGICSPLIT_UNIVERSE_MODE=optimistic_survivor python -m src.parity_sell_event_dump \
  --start-date 20210101 --end-date 20210331 \
  --params-csv results/standalone_simulation_results_20260302_012840.csv \
  --parity-mode strict --candidate-source-mode tier \
  --out /tmp/parity_events_optimistic_survivor_latest.json

# strict_pit
MAGICSPLIT_UNIVERSE_MODE=strict_pit python -m src.cpu_gpu_parity_topk \
  --start-date 20210101 --end-date 20210331 --topk 1 --parity-mode strict \
  --out /tmp/parity_report_top1_strictpit_latest.json

MAGICSPLIT_UNIVERSE_MODE=strict_pit python -m src.cpu_gpu_parity_topk \
  --start-date 20210101 --end-date 20210331 \
  --params-csv results/standalone_simulation_results_20260302_012840.csv \
  --sort-by cagr --topk 5 --parity-mode strict \
  --out /tmp/parity_report_top5_strictpit_latest.json

MAGICSPLIT_UNIVERSE_MODE=strict_pit python -m src.parity_sell_event_dump \
  --start-date 20210101 --end-date 20210331 \
  --params-csv results/standalone_simulation_results_20260302_012840.csv \
  --parity-mode strict --candidate-source-mode tier \
  --out /tmp/parity_events_strictpit_latest_afterfix.json
```

## 결과 요약

| Track | top1 parity | top5 parity | Event parity (sell/buy mismatch) |
|---|---:|---:|---:|
| `optimistic_survivor` | `failed=0, passed=1` | `failed=0, passed=5` | `0 / 0` |
| `strict_pit` | `failed=0, passed=1` | `failed=0, passed=5` | `0 / 0` |

## 세부 수치
- `optimistic_survivor`
  - top1: `tolerance=0.001`, `parity_mode=strict`, `universe_mode=optimistic_survivor`, `failed=0`, `passed=1`
  - top5: `tolerance=0.001`, `parity_mode=strict`, `universe_mode=optimistic_survivor`, `failed=0`, `passed=5`
  - event: `cpu_sell_events=42`, `gpu_sell_events=42`, `cpu_buy_events=72`, `gpu_buy_events=72`, `sell_mismatched_pairs=0`, `buy_mismatched_pairs=0`
- `strict_pit`
  - top1: `tolerance=0.001`, `parity_mode=strict`, `universe_mode=strict_pit`, `failed=0`, `passed=1`
  - top5: `tolerance=0.001`, `parity_mode=strict`, `universe_mode=strict_pit`, `failed=0`, `passed=5`
  - event: `cpu_sell_events=41`, `gpu_sell_events=41`, `cpu_buy_events=67`, `gpu_buy_events=67`, `sell_mismatched_pairs=0`, `buy_mismatched_pairs=0`

## 판정
- 두 트랙 모두 현재 범위(`2021Q1`, `strict`, top-k/event-level)에서 parity gate 통과.
- 운영 게이트는 `strict_pit`를 기본 승인 트랙으로 유지하고, `optimistic_survivor`는 별도 연구/회귀 트랙으로 병행 관리.
