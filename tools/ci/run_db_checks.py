"""전용 임시 MySQL의 통합 검사. CPU 기본 검사 범위는 변경하지 않는다."""

from pathlib import Path
import json
import os
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.ci.db_fixture import required_config
from tools.ci.run_cpu_checks import Runner, Task, interrupted


def publish(outcomes):
    rows = ["## DB 통합 검사", "", "| 대상 | 결과 | 검사 수 | 종료 코드 |",
            "| --- | --- | ---: | ---: |"]
    for outcome in outcomes:
        status = "FAIL" if outcome.problem else "PASS"
        count = (outcome.report or {}).get("tests_run", "미완료")
        rows.append(f"| {outcome.task.target} | {status} | {count} | {outcome.returncode} |")
        print(outcome.output, flush=True)
        if outcome.problem:
            print(outcome.problem, flush=True)
        if outcome.report:
            print(json.dumps(outcome.report, ensure_ascii=False), flush=True)
    rows.extend(["", "임시 MySQL·합성 자료·시험용 전략의 CPU 통합 범위입니다.",
                 "운영 DB·실제 MagicSplitStrategy 전체·GPU·외부 수집은 검사하지 않습니다."])
    summary = "\n".join(rows) + "\n"
    print(summary, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(summary)
    return int(not outcomes or any(outcome.problem for outcome in outcomes))


def main():
    if len(sys.argv) != 1:
        print("사용법: python tools/ci/run_db_checks.py", file=sys.stderr)
        return 2
    try:
        required_config()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return publish(Runner(ROOT, timeout=120).run((Task("tests", "test_integration.py"),)))
    except KeyboardInterrupt:
        print("취소: DB 검사 프로세스를 종료했습니다.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
