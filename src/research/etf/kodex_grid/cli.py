"""탐색 실행기 명령. 실행: python3 -B -m src.research.etf.kodex_grid <명령> --out DIR [--data-root … --manifest …]

selftest  합성 시험 묶음을 돌리고 이 판본의 통과 기록을 남긴다(실자료를 쓰지 않는다).
baseline  공개 기준점 대조(정식 결과와 허용오차 0). 통과 기록이 있어야 하고 --formal-result가 필요하다.
stage1    주 시작일 36개를 설정 단위로 병렬 실행해 봉인한다(진행 로그에는 번호·시간·기술 상태만).
assemble1 36개가 모두 봉인됐을 때만 열어 판정·영역·선택·요약을 만들고 최초 기록을 고정한다.
stage2    선택된 영역이 있을 때만 재확인(8점 × 2시작일)과 확장(8점 × 2상품)을 실행한다.
assemble2 재확인과 확장을 별도 칸으로 조립하고 결론 문장을 남긴다.
eta       진행 로그의 기술적 완료시간만으로 남은 시간을 다시 잡는다(--stage, --workers).

stage1·assemble1·stage2·assemble2는 같은 판본의 selftest 통과 기록과 baseline 대조 증거가 --out 아래에 있어야 시작한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..kodex_core.config import load_config
from . import baseline, permit, progress, stages
from .grid import load_grid
from .identity import BundleSource, open_context
from .sealing import dump


LOCAL = ("selftest", "eta")  # 실자료 묶음을 열지 않는 명령
COMMANDS = ("selftest", "baseline", "stage1", "assemble1", "stage2", "assemble2", "eta")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -B -m src.research.etf.kodex_grid", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--formal-result", help="baseline: 정식 결과 result.json 경로")
    parser.add_argument("--stage", choices=(stages.STAGE1, stages.STAGE2), default=stages.STAGE1)
    return parser


def _selftest(args) -> int:
    record = permit.run_selftest(args.out)
    print(f"selftest: passed tests_run={record['tests_run']} seal={record['seal']}")
    return 0


def _eta(args) -> int:
    spec = load_grid()
    planned = len(spec.settings()) if args.stage == stages.STAGE1 else 4 * spec.region["points"]
    log = Path(args.out) / args.stage / "progress.log"
    print(dump(progress.remaining(log, planned=planned, workers=args.workers)))
    return 0


def _baseline(ctx, args) -> int:
    if not args.formal_result:
        raise SystemExit("baseline needs --formal-result")
    record = baseline.run_baseline(ctx, args.formal_result, args.out)
    print(baseline.markdown(json.loads(dump(record))))
    return 0 if record["main_run_allowed"] else 1


def _assembled(result) -> int:
    selection = result["selection"]
    print(f"assembled: selection={selection['state']} stage2={selection['stage2']}")
    return 0


def _staged(ctx, args) -> int:
    """네 단계 명령. 본실행 허가는 단계 함수 안의 공용 진입점이 확인한다."""
    if args.command == "stage1":
        stages.run_stage1(ctx, args.out, workers=args.workers)
    elif args.command == "stage2":
        print(f"stage2: {stages.run_stage2(ctx, args.out, workers=args.workers)}")
    elif args.command == "assemble1":
        return _assembled(stages.assemble1(ctx, args.out))
    else:
        result = stages.assemble2(ctx, args.out)
        print(f"assembled: stage2={result['stage2']} conclusion={result['conclusion']['sentence']}")
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command in LOCAL:
        return _selftest(args) if args.command == "selftest" else _eta(args)
    if not args.data_root or not args.manifest:
        raise SystemExit(f"{args.command} needs --data-root and --manifest")
    ctx = open_context(BundleSource(args.data_root, args.manifest, load_config()))
    return _baseline(ctx, args) if args.command == "baseline" else _staged(ctx, args)
