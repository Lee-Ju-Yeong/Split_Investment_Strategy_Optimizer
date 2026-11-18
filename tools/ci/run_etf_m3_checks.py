"""명시 빌드한 M3 커널을 CPU 실행기의 엄격한 성공 조건으로 검사한다."""
import argparse
import os
from pathlib import Path

from run_cpu_checks import Runner, Task, publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["MAGICSPLIT_ETF_M3_BUILD"] = str(args.build_dir.resolve())
    root = Path(__file__).resolve().parents[2]
    outcomes = Runner(root, timeout=180).run([Task("tests", "test_research_etf_fast_kernel.py")])
    return publish(outcomes)


if __name__ == "__main__":
    raise SystemExit(main())
