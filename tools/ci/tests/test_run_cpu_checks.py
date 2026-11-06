"""제품 검사를 바꾸지 않고 실행기의 실패 판정을 검증한다."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest.mock import patch

from tools.ci import run_cpu_checks as ci


GOOD_TEST = """
import unittest
class Check(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(2 + 2, 4)
"""

SLEEPING_EXAMPLE = """
import os
from pathlib import Path
import subprocess
import sys
import time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path('pids').write_text(f'{os.getpid()} {child.pid}')
time.sleep(30)
"""


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tests").mkdir()
        summary_env = patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(self.root / "fixture-summary.md")})
        summary_env.start()
        self.addCleanup(summary_env.stop)

    def fixture(self, source, name="test_fixture.py"):
        (self.root / "tests" / name).write_text(textwrap.dedent(source), encoding="utf-8")
        return ci.Task("tests", name)

    def example(self, source):
        (self.root / "sample.py").write_text(textwrap.dedent(source), encoding="utf-8")
        return ci.Task("example", "sample")

    def run_tasks(self, tasks, timeout=10):
        runner = ci.Runner(self.root, timeout)
        outcomes = runner.run(tasks)
        self.assertFalse(runner.processes)
        return outcomes

    def assert_rejected(self, task):
        outcome = self.run_tasks([task])[0]
        self.assertTrue(outcome.problem, outcome)
        with redirect_stdout(StringIO()):
            self.assertEqual(ci.publish([outcome]), 1)
        return outcome

    def test_normal_tests_and_example(self):
        tasks = [self.fixture(GOOD_TEST), self.example("print('example passed')")]
        outcomes = self.run_tasks(tasks)
        self.assertTrue(all(not outcome.problem for outcome in outcomes), outcomes)
        self.assertEqual(outcomes[0].report["tests_run"], 1)
        self.assertIn("example passed", outcomes[1].output)
        with redirect_stdout(StringIO()):
            self.assertEqual(ci.publish(outcomes), 0)

    def test_added_test_is_included(self):
        task = self.fixture(GOOD_TEST + "\n    def test_added(self): pass\n")
        outcome = self.run_tasks([task])[0]
        self.assertFalse(outcome.problem)
        self.assertEqual(outcome.report["tests_run"], 2)

    def test_failure_and_error_are_rejected(self):
        for statement in ("self.fail('failure')", "raise RuntimeError('error')"):
            with self.subTest(statement=statement):
                self.assert_rejected(self.fixture(GOOD_TEST.replace("self.assertEqual(2 + 2, 4)", statement)))

    def test_skips_and_expected_failures_are_rejected(self):
        variants = (
            "@unittest.skip('not available')\n    def test_ok(self): pass",
            "@unittest.expectedFailure\n    def test_ok(self): self.fail('expected')",
            "@unittest.expectedFailure\n    def test_ok(self): pass",
        )
        for definition in variants:
            with self.subTest(definition=definition):
                self.assert_rejected(self.fixture("import unittest\nclass Check(unittest.TestCase):\n    " + definition))

    def test_subtest_failure_is_rejected(self):
        self.assert_rejected(self.fixture("""
            import unittest
            class Check(unittest.TestCase):
                def test_subtest(self):
                    with self.subTest(value=1):
                        self.fail('subtest failed')
        """))

    def test_class_setup_skip_and_error_are_rejected(self):
        for exception in ("unittest.SkipTest('missing')", "RuntimeError('setup')"):
            with self.subTest(exception=exception):
                source = GOOD_TEST + "\n    @classmethod\n    def setUpClass(cls): raise " + exception
                self.assert_rejected(self.fixture(source))

    def test_zero_and_missing_modules_are_rejected(self):
        self.assert_rejected(self.fixture("# no tests"))
        self.assert_rejected(ci.Task("tests", "test_missing.py"))

    def test_import_failure_is_rejected(self):
        self.assert_rejected(self.fixture("raise ImportError('fixture unavailable')"))

    def test_duplicate_ids_are_rejected(self):
        task = self.fixture(GOOD_TEST + """
def load_tests(loader, tests, pattern):
    return unittest.TestSuite([Check('test_ok'), Check('test_ok')])
""")
        self.assertIn("중복", self.assert_rejected(task).problem)

    def test_discovered_but_unexecuted_test_is_rejected(self):
        task = self.fixture(GOOD_TEST + """
class IncompleteSuite(unittest.TestSuite):
    def run(self, result, debug=False):
        list(self)[0](result)
        return result
class Another(Check):
    pass
def load_tests(loader, tests, pattern):
    return IncompleteSuite([Check('test_ok'), Another('test_ok')])
""")
        self.assertIn("일치", self.assert_rejected(task).problem)

    def test_exit_zero_without_report_is_rejected(self):
        for task in (self.fixture("import os; os._exit(0)"), self.example("import os; os._exit(0)")):
            with self.subTest(kind=task.kind):
                outcome = self.assert_rejected(task)
                self.assertEqual(outcome.returncode, 0)
                self.assertIn("기록 누락", outcome.problem)

    def test_exit_zero_with_mismatched_report_is_rejected(self):
        report = {"task": ["tests", "test_fixture.py"], "completed": True,
                  "discovered": ["a", "b"], "started": ["a"], "finished": ["a"],
                  "tests_run": 1, "loader_errors": [], "counts": dict.fromkeys(ci.COUNTERS, 0)}
        task = self.fixture("import os, sys\nfrom pathlib import Path\n"
                            f"Path(sys.argv[-1]).write_text({json.dumps(report)!r})\nos._exit(0)")
        outcome = self.assert_rejected(task)
        self.assertEqual(outcome.returncode, 0)
        self.assertIn("일치", outcome.problem)

    def test_nonzero_exit_with_success_report_is_rejected(self):
        report = {"task": ["example", "sample"], "completed": True}
        task = self.example("import os\nfrom pathlib import Path\n"
                            f"(Path(os.environ['TMPDIR']) / 'result.json').write_text({json.dumps(report)!r})\nos._exit(7)")
        outcome = self.assert_rejected(task)
        self.assertEqual(outcome.returncode, 7)
        self.assertEqual(outcome.report, report)

    def test_example_receives_only_its_own_cli_arguments(self):
        task = self.example("import argparse, sys\n"
                            "argparse.ArgumentParser().parse_args()\n"
                            "assert sys.argv == ['sample'], sys.argv")
        outcome = self.run_tasks([task])[0]
        self.assertFalse(outcome.problem, outcome)

    def test_example_arguments_are_restored_after_success_and_error(self):
        original = ["runner", "--worker", "private-report-path"]
        for error in (False, True):
            with self.subTest(error=error):
                report_path = self.root / "argv-report.json"
                def invoke(*_args, **_kwargs):
                    self.assertEqual(sys.argv, ["sample"])
                    if error:
                        raise RuntimeError("fixture failure")
                with patch.object(sys, "argv", original), patch.object(ci.runpy, "run_module", invoke):
                    with redirect_stdout(StringIO()):
                        code = ci.worker(ci.Task("example", "sample"), report_path)
                    self.assertIs(sys.argv, original)
                self.assertEqual(code, int(error))

    def test_example_failure(self):
        self.assert_rejected(self.example("raise RuntimeError('example failure')"))

    def test_example_timeout_and_process_group_cleanup(self):
        outcome = self.run_tasks([self.example(SLEEPING_EXAMPLE)], timeout=1)[0]
        self.assertIn("시간 초과", outcome.problem)
        self.assert_processes_stopped()

    def test_module_timeout(self):
        task = self.fixture("import time; time.sleep(30)")
        outcome = self.run_tasks([task], timeout=0.3)[0]
        self.assertIn("시간 초과", outcome.problem)

    def test_failure_does_not_hide_other_results(self):
        tasks = [self.fixture("raise ImportError('broken')"),
                 self.fixture(GOOD_TEST, "test_good.py"), self.example("print('finished')")]
        outcomes = self.run_tasks(tasks)
        self.assertEqual(len(outcomes), 3)
        self.assertTrue(outcomes[0].problem)
        self.assertTrue(all(not item.problem for item in outcomes[1:]), outcomes)

    def test_cancel_reaps_owned_children(self):
        runner = ci.Runner(self.root)
        task = self.example(SLEEPING_EXAMPLE)
        results = []
        thread = threading.Thread(target=lambda: results.extend(runner.run([task])))
        thread.start()
        try:
            self.wait_for_pids()
            runner.cancel()
        finally:
            runner.cancel()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(runner.processes)
        self.assertTrue(results[0].problem)
        self.assert_processes_stopped()

    def test_cancel_before_start(self):
        runner = ci.Runner(self.root)
        runner.cancel()
        outcome = runner.run([self.example("raise RuntimeError('must not run')")])[0]
        self.assertEqual(outcome.problem, "취소됨")
        self.assertFalse(runner.processes)

    def test_sigterm_cleans_up_and_exits_nonzero(self):
        self.example(SLEEPING_EXAMPLE)
        source = f"""
import signal
from pathlib import Path
from tools.ci import run_cpu_checks as ci
ci.ROOT = Path({str(self.root)!r})
ci.TASKS = (ci.Task('example', 'sample'),)
raise SystemExit(ci.main())
"""
        process = subprocess.Popen([sys.executable, "-c", source], cwd=ci.ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.wait_for_pids()
            process.send_signal(signal.SIGTERM)
            output, errors = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 130, output + errors)
            self.assert_processes_stopped()
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()

    def test_summary_records_scope_and_exit_code(self):
        outcome = self.run_tasks([self.fixture(GOOD_TEST)])[0]
        summary = self.root / "summary.md"
        with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}), redirect_stdout(StringIO()):
            self.assertEqual(ci.publish([outcome]), 0)
        text = summary.read_text(encoding="utf-8")
        self.assertIn("PASS | 1 | 0", text)
        self.assertIn("GPU·실제 DB·외부 수집은 검사하지 않습니다", text)

    def wait_for_pids(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (self.root / "pids").exists():
                return
            time.sleep(0.02)
        self.fail("child processes did not start")

    def assert_processes_stopped(self):
        for pid in (self.root / "pids").read_text().split():
            status = Path(f"/proc/{pid}/stat")
            # 종료된 손자 프로세스는 init이 회수하기 전 잠시 zombie로 보일 수 있다.
            deadline = time.monotonic() + 2
            while status.exists() and time.monotonic() < deadline:
                try:
                    if status.read_text().split()[2] == "Z":
                        break
                except FileNotFoundError:
                    break
                time.sleep(0.02)
            else:
                self.assertFalse(status.exists(), f"process {pid} still running")


class ReportTests(unittest.TestCase):
    def test_invalid_result_records_are_rejected(self):
        task = ci.Task("tests", "test_fixture.py")
        valid = {"task": [task.kind, task.target], "completed": True,
                 "discovered": ["Check.test_ok"], "started": ["Check.test_ok"],
                 "finished": ["Check.test_ok"], "tests_run": 1, "loader_errors": [],
                 "counts": dict.fromkeys(ci.COUNTERS, 0)}
        self.assertFalse(ci.validate_report(task, valid))
        replacements = {"task": ["tests", "wrong.py"], "completed": False,
                        "discovered": [], "started": ["different"], "finished": [],
                        "tests_run": 2, "loader_errors": ["import error"], "counts": {}}
        for key, value in replacements.items():
            with self.subTest(key=key):
                broken = deepcopy(valid)
                broken[key] = value
                self.assertTrue(ci.validate_report(task, broken))
                del broken[key]
                self.assertTrue(ci.validate_report(task, broken))
        for malformed in (None, [], "success", {}):
            with self.subTest(malformed=malformed):
                self.assertTrue(ci.validate_report(task, malformed))


if __name__ == "__main__":
    unittest.main()
