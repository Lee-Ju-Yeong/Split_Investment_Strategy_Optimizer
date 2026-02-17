"""
tier_parity_monitor.py

Tier-only parity monitor wrapper.
- Runs `src.cpu_gpu_parity_topk` in tier mode.
- Prints deterministic PASS/FAIL line.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tier-only parity monitor wrapper.")
    parser.add_argument("--start-date", default="2021-01-04")
    parser.add_argument("--end-date", default="2021-01-08")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--params-csv", required=True)
    parser.add_argument("--scenario", default="baseline_deterministic")
    parser.add_argument("--seeded-stress-count", type=int, default=2)
    parser.add_argument("--jackknife-max-drop", type=int, default=1)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--out", default=None)
    return parser


def _default_out_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("results", f"tier_parity_monitor_{ts}.json")


def main() -> None:
    args = _build_parser().parse_args()
    out_path = args.out or _default_out_path()

    cmd = [
        sys.executable,
        "-m",
        "src.cpu_gpu_parity_topk",
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--top-k",
        str(args.top_k),
        "--params-csv",
        args.params_csv,
        "--scenario",
        args.scenario,
        "--seeded-stress-count",
        str(args.seeded_stress_count),
        "--jackknife-max-drop",
        str(args.jackknife_max_drop),
        "--candidate-source-mode",
        "tier",
        "--no-fail-on-mismatch",
        "--out",
        out_path,
    ]

    env = os.environ.copy()
    if args.config_path:
        env["MAGICSPLIT_CONFIG_PATH"] = args.config_path

    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        print(f"[tier_parity_monitor] status=FAIL reason=runner_exit_code code={proc.returncode} report={out_path}")
        raise SystemExit(proc.returncode)

    with open(out_path, "r", encoding="utf-8") as fp:
        report = json.load(fp)
    mismatches = int(report.get("total_mismatches", -1))
    skipped = bool(report.get("skipped", False))

    if skipped:
        print(f"[tier_parity_monitor] status=FAIL reason=skipped report={out_path}")
        raise SystemExit(3)
    if mismatches != 0:
        print(f"[tier_parity_monitor] status=FAIL mismatches={mismatches} report={out_path}")
        raise SystemExit(2)

    print(f"[tier_parity_monitor] status=PASS mismatches=0 report={out_path}")


if __name__ == "__main__":
    main()

