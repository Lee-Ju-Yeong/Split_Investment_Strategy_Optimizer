"""명령. 실행: python3 -B -m src.research.etf.markets_mix <명령> --out DIR [--data-root … --manifest …]

selftest     합성 시험 묶음을 돌리고 이 판본의 통과 기록을 남긴다(실자료를 쓰지 않는다).
baseline     공개값 재현(기준점 802항목과 229200·169950의 공개값, 허용오차 0). --formal-result가 필요하다.
eligibility  자격 집계(이력·거래대금 한도의 통과 여부와 사유의 종류만. 장부를 열지 않는다).
run          계획한 단위를 병렬 실행해 봉인한다(진행 로그에는 단위 이름·시간·기술 상태만).
assemble     계획한 단위가 모두 봉인됐을 때만 열어 혼합을 조립하고 판정과 표를 만든다.
eta          진행 로그의 기술적 완료시간만으로 남은 시간을 다시 잡는다(--workers).

run·assemble은 같은 판본의 selftest 통과 기록과 baseline 재현 증거가 --out 아래에 있어야 시작한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..kodex_core.config import load_config
from ..kodex_grid import progress
from ..kodex_grid.grid import load_grid
from ..kodex_grid.sealing import dump
from . import baseline, batch, eligibility, permit
from .definition import load_definition, plan_units
from .identity import MixSource, open_context
from .records import STAGE


LOCAL = ("selftest", "eta")  # 실자료 묶음을 열지 않는 명령
COMMANDS = ("selftest", "baseline", "eligibility", "run", "assemble", "eta")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -B -m src.research.etf.markets_mix", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--formal-result", help="baseline: 정식 결과 result.json 경로")
    return parser


def _selftest(args) -> int:
    record = permit.run_selftest(args.out)
    print(f"selftest: passed tests_run={record['tests_run']} seal={record['seal']}")
    return 0


def _eta(args) -> int:
    planned = len(plan_units(load_definition(), load_grid()))
    log = Path(args.out) / STAGE / "progress.log"
    print(dump(progress.remaining(log, planned=planned, workers=args.workers)))
    return 0


def _baseline(ctx, args) -> int:
    if not args.formal_result:
        raise SystemExit("baseline needs --formal-result")
    record = baseline.run_baseline(ctx, args.formal_result, args.out, workers=max(args.workers, 1))
    print(baseline.markdown(json.loads(dump(record))))
    return 0 if record["main_run_allowed"] else 1


def _eligibility(ctx, args) -> int:
    print(eligibility.markdown(eligibility.run_eligibility(ctx, args.out)))
    return 0


def _assemble(ctx, args) -> int:
    done = batch.assemble(ctx, args.out)["conclusion"]
    print(f"assembled: conclusion={done['kind']} rows={done['rows']} counts={done['counts']}")
    return 0


def _staged(ctx, args) -> int:
    """실자료 명령. 본실행 허가는 run·assemble 안의 공용 진입점이 확인한다."""
    if args.command == "baseline":
        return _baseline(ctx, args)
    if args.command == "eligibility":
        return _eligibility(ctx, args)
    if args.command == "assemble":
        return _assemble(ctx, args)
    batch.run(ctx, args.out, workers=args.workers)
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command in LOCAL:
        return _selftest(args) if args.command == "selftest" else _eta(args)
    if not args.data_root or not args.manifest:
        raise SystemExit(f"{args.command} needs --data-root and --manifest")
    base = load_config()
    ctx = open_context(MixSource(args.data_root, args.manifest, base, symbols=load_definition().symbols()))
    return _staged(ctx, args)
