"""외부 DB/GPU 없이 실행하는 고정 CPU 검사 묶음 (Python 3.12)."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import json
import os
import runpy
import signal
import subprocess
import sys
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[2]
COUNTERS = ("failures", "errors", "skipped", "expectedFailures", "unexpectedSuccesses")


@dataclass(frozen=True)
class Task:
    kind: str
    target: str


TASKS = (
    Task("tests", "test_portfolio.py"),
    Task("tests", "test_short_selling_collector.py"),
    Task("tests", "test_daily_stock_tier_batch.py"),
    Task("example", "examples.short_selling_lag_demo"),
)


class TrackedResult(unittest.TextTestResult):
    def __init__(self, *args):
        super().__init__(*args)
        self.started = []
        self.finished = []

    def startTest(self, test):
        self.started.append(test.id())
        super().startTest(test)

    def stopTest(self, test):
        self.finished.append(test.id())
        super().stopTest(test)


def test_ids(suite):
    if isinstance(suite, unittest.TestSuite):
        return [item for child in suite for item in test_ids(child)]
    return [suite.id()]


def run_test_module(target):
    directory = Path.cwd() / "tests"
    if not (directory / target).is_file():
        raise FileNotFoundError(target)
    loader = unittest.TestLoader()
    suite = loader.discover(str(directory), pattern=target)
    discovered = test_ids(suite)
    result = unittest.TextTestRunner(verbosity=2, resultclass=TrackedResult).run(suite)
    return {
        "discovered": discovered,
        "started": result.started,
        "finished": result.finished,
        "tests_run": result.testsRun,
        "loader_errors": loader.errors,
        "counts": {name: len(getattr(result, name)) for name in COUNTERS},
    }


def validate_report(task, report):
    if not isinstance(report, dict) or report.get("task") != [task.kind, task.target]:
        return "결과 기록의 대상이 다릅니다."
    if report.get("completed") is not True or "error" in report:
        return "자식 프로세스가 검사를 완료하지 못했습니다."
    if task.kind == "example":
        return ""
    return validate_test_counts(report)


def validate_test_counts(report):
    ids = [report.get(key) for key in ("discovered", "started", "finished")]
    for entries in ids:
        if not isinstance(entries, list) or not entries:
            return "검사 0개 또는 검사 식별자 기록 누락입니다."
        if any(not isinstance(item, str) or not item for item in entries):
            return "잘못된 검사 식별자입니다."
        if len(entries) != len(set(entries)):
            return "중복 검사 식별자입니다."
    if not (set(ids[0]) == set(ids[1]) == set(ids[2])):
        return "수집·시작·종료한 검사가 일치하지 않습니다."
    if type(report.get("tests_run")) is not int or report["tests_run"] != len(ids[0]):
        return "실제 검사 수가 수집한 수와 다릅니다."
    if report.get("loader_errors") != []:
        return "검사 모듈 가져오기 오류입니다."
    counts = report.get("counts")
    if not isinstance(counts, dict) or set(counts) != set(COUNTERS):
        return "검사 결과 집계가 없습니다."
    if any(type(value) is not int or value != 0 for value in counts.values()):
        return "실패·오류·건너뜀·예상된 실패·예상 밖 성공은 허용하지 않습니다."
    return ""


def worker(task, report_path):
    sys.path.insert(0, str(Path.cwd()))
    report = {"task": [task.kind, task.target], "completed": False}
    try:
        if task.kind == "tests":
            report.update(run_test_module(task.target))
        else:
            runpy.run_module(task.target, run_name="__main__")
        report["completed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    problem = validate_report(task, report)
    if problem:
        print(problem, report.get("error", ""), file=sys.stderr)
    return int(bool(problem))


def stop_process(process):
    """각 작업의 새 세션만 종료한다. 손자 프로세스도 같은 그룹으로 회수한다."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


@dataclass
class Outcome:
    task: Task
    problem: str
    output: str = ""
    report: dict | None = None
    returncode: int | None = None


class Runner:
    def __init__(self, root, timeout=120):
        self.root = Path(root)
        self.timeout = timeout
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.processes = set()

    def cancel(self):
        # 시작과 취소를 같은 잠금으로 보호하여 취소 뒤 새 자식이 생기지 않게 한다.
        with self.lock:
            self.cancelled.set()
            for process in self.processes:
                stop_process(process)

    def start(self, command, directory):
        with self.lock:
            if self.cancelled.is_set():
                return None
            env = dict(os.environ, PYTHONUTF8="1", TMPDIR=str(directory))
            process = subprocess.Popen(
                command, cwd=self.root, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                start_new_session=(os.name == "posix"),
            )
            self.processes.add(process)
            return process

    def execute(self, task):
        with tempfile.TemporaryDirectory(prefix="cpu-check-") as directory:
            path = Path(directory) / "result.json"
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--worker", task.kind, task.target, str(path)]
            try:
                process = self.start(command, directory)
            except OSError as exc:
                return Outcome(task, f"프로세스 시작 실패: {exc}")
            if process is None:
                return Outcome(task, "취소됨")
            try:
                return self.collect(task, (process, path))
            finally:
                with self.lock:
                    stop_process(process)
                    self.processes.remove(process)

    def collect(self, task, execution):
        process, path = execution
        try:
            output, _ = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            stop_process(process)
            output, _ = process.communicate()
            return Outcome(task, f"시간 초과 ({self.timeout}초)", output,
                           returncode=process.returncode)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return Outcome(task, f"결과 기록 누락/손상: {exc}", output,
                           returncode=process.returncode)
        problem = validate_report(task, report)
        if process.returncode != 0:
            problem = f"종료 코드 {process.returncode}: {problem}"
        if self.cancelled.is_set():
            problem = "취소됨"
        return Outcome(task, problem, output, report if isinstance(report, dict) else None,
                       process.returncode)

    def run(self, tasks):
        executor = ThreadPoolExecutor(max_workers=3)
        try:
            futures = [executor.submit(self.execute, task) for task in tasks]
            return [future.result() for future in futures]
        finally:
            self.cancel()
            executor.shutdown(wait=True, cancel_futures=True)


def publish(outcomes):
    rows = ["## CPU 기본 검사", "", "| 대상 | 결과 | 검사 수 | 종료 코드 |",
            "| --- | --- | ---: | ---: |"]
    for outcome in outcomes:
        status = "FAIL" if outcome.problem else "PASS"
        count = (outcome.report or {}).get("tests_run", "예제")
        rows.append(f"| `{outcome.task.target}` | {status} | {count} | {outcome.returncode} |")
        print(f"\n=== {outcome.task.target}: {status} ===\n{outcome.output}", flush=True)
        if outcome.problem:
            print(outcome.problem, flush=True)
        if outcome.report:
            print(json.dumps(outcome.report, ensure_ascii=False), flush=True)
    rows.extend(["", "PASS는 검사 누락·건너뜀·예상된 실패가 없음을 포함합니다.",
                 "도구 자체 검사는 별도 단계입니다. GPU·실제 DB·외부 수집은 검사하지 않습니다."])
    summary = "\n".join(rows) + "\n"
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(summary)
    return int(any(outcome.problem for outcome in outcomes))


def interrupted(_signum, _frame):
    raise KeyboardInterrupt


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        return worker(Task(sys.argv[2], sys.argv[3]), Path(sys.argv[4]))
    if len(sys.argv) != 1:
        print("사용법: python tools/ci/run_cpu_checks.py", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return publish(Runner(ROOT).run(TASKS))
    except KeyboardInterrupt:
        print("취소: 자식 프로세스를 종료했습니다.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
