"""임시 DB 제한과 실패 판정. 실제 MySQL 자료를 사용하지 않는다."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import importlib.util
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tools.ci import db_fixture
from tools.ci import run_db_checks as db
from tools.ci.run_cpu_checks import Outcome, Runner, Task


ENV = {
    "SISO_RUN_DB_TESTS": "1", "SISO_TEST_MYSQL_HOST": "127.0.0.1",
    "SISO_TEST_MYSQL_PORT": "33060", "SISO_TEST_MYSQL_USER": "siso_test",
    "SISO_TEST_MYSQL_PASSWORD": "ephemeral-test-only",
    "SISO_TEST_MYSQL_DATABASE": "siso_test_integration",
}


class DatabaseGuardTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, ENV, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.connect, self.create = MagicMock(), MagicMock()
        self.connection = self.connect.return_value
        self.cursor = self.connection.cursor.return_value.__enter__.return_value
        self.cursor.fetchone.side_effect = [("siso_test_integration", "siso_test@%", "8.4"), (0,)]

    def prepare(self):
        return db_fixture.prepare_database(self.connect, self.create)

    def test_valid_explicit_config_and_empty_db(self):
        config, conn, version = self.prepare()
        self.assertEqual(config["port"], 33060)
        self.assertIs(conn, self.connection)
        self.assertEqual(version, "8.4")
        self.create.assert_called_once_with(conn)
        self.connection.close.assert_not_called()
        self.assertTrue(all(call.args[0].startswith("SELECT") for call in self.cursor.execute.call_args_list))

    def test_invalid_settings_never_connect(self):
        bad_values = {
            "SISO_RUN_DB_TESTS": ("", "0"),
            "SISO_TEST_MYSQL_HOST": ("localhost", "192.0.2.1", ""),
            "SISO_TEST_MYSQL_USER": ("root", "", "siso_test@%"),
            "SISO_TEST_MYSQL_DATABASE": ("production", "siso_test_other", ""),
            "SISO_TEST_MYSQL_PORT": ("", "0", "65536", "port", "3306.0"),
            "SISO_TEST_MYSQL_PASSWORD": ("",),
        }
        for name, values in bad_values.items():
            for value in values:
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                    with self.assertRaises(ValueError):
                        self.prepare()
        self.connect.assert_not_called()
        self.create.assert_not_called()

    def test_missing_fields_do_not_fall_back_to_product_settings(self):
        for key in ENV:
            env = dict(ENV, MAGICSPLIT_CONFIG_PATH="must-not-read.yaml", MYSQL_HOST="127.0.0.1")
            del env[key]
            with self.subTest(key=key), patch.dict(os.environ, env, clear=True):
                with patch("builtins.open", side_effect=AssertionError("No product config reads")):
                    with self.assertRaises(ValueError):
                        self.prepare()
        self.connect.assert_not_called()

    def test_wrong_actual_identity_or_existing_tables_never_write(self):
        for database, user, tables in (("other", "siso_test@%", 0),
                                       ("siso_test_integration", "root@localhost", 0),
                                       ("siso_test_integration", "siso_test@%", 1)):
            with self.subTest(database=database, user=user, tables=tables):
                self.cursor.fetchone.side_effect = [(database, user, "8.4"), (tables,)]
                with self.assertRaises(ValueError):
                    self.prepare()
        self.create.assert_not_called()
        self.assertTrue(all(call.args[0].startswith("SELECT") for call in self.cursor.execute.call_args_list))
        self.assertEqual(self.connection.close.call_count, 3)

    def test_connect_errors_propagate(self):
        self.connect.side_effect = RuntimeError("connection unavailable")
        with self.assertRaisesRegex(RuntimeError, "connection unavailable"):
            self.prepare()
        self.create.assert_not_called()

    def test_read_errors_close_connection_before_ddl(self):
        self.cursor.execute.side_effect = RuntimeError("preflight failed")
        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
            self.prepare()
        self.create.assert_not_called()
        self.connection.close.assert_called_once()

    def test_schema_failure_closes_connection_and_propagates(self):
        self.create.side_effect = RuntimeError("DDL failed")
        with self.assertRaisesRegex(RuntimeError, "DDL failed"):
            self.prepare()
        self.connection.close.assert_called_once()

    def integration_class(self):
        path = db.ROOT / "tests/test_integration.py"
        spec = importlib.util.spec_from_file_location("db_integration_contract", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TestBacktestingIntegration

    def test_unrequested_discovery_skips_without_reading_files(self):
        case = self.integration_class()
        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.open", side_effect=AssertionError("No config reads")):
                with self.assertRaisesRegex(unittest.SkipTest, "SISO_RUN_DB_TESTS"):
                    case.setUpClass()

    def test_requested_missing_driver_is_error_not_skip(self):
        case = self.integration_class()
        with patch.dict("sys.modules", {"pymysql": None}):
            with self.assertRaises(ModuleNotFoundError):
                case.setUpClass()

    def test_requested_setup_failure_is_error_not_skip(self):
        case = self.integration_class()
        with patch.object(db_fixture, "prepare_database", side_effect=RuntimeError("setup failed")):
            with self.assertRaisesRegex(RuntimeError, "setup failed"):
                case.setUpClass()


class DatabaseRunnerTests(unittest.TestCase):
    def test_command_rejects_missing_config_without_starting_worker(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(db, "Runner") as runner:
            with patch("sys.argv", ["run_db_checks.py"]):
                self.assertEqual(db.main(), 2)
        runner.assert_not_called()

    def test_worker_skip_zero_tests_and_timeout_are_failures(self):
        sources = (
            "import unittest\nclass Check(unittest.TestCase):\n @unittest.skip('unavailable')\n def test_skip(self): pass\n",
            "import unittest\n",
            "import unittest,time\nclass Check(unittest.TestCase):\n def test_wait(self): time.sleep(10)\n",
        )
        for source in sources:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "tests").mkdir()
                (root / "tests/test_integration.py").write_text(source)
                outcomes = Runner(root, timeout=0.5).run((Task("tests", "test_integration.py"),))
                self.assertTrue(outcomes[0].problem)
                with redirect_stdout(StringIO()), patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(db.publish(outcomes), 1)

    def test_summary_is_db_specific_and_failed_exit_propagates(self):
        task = Task("tests", "test_integration.py")
        for problem, code in (("", 0), ("missing result", 1)):
            stream = StringIO()
            with redirect_stdout(stream), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(db.publish([Outcome(task, problem, report={"tests_run": 3}, returncode=code)]), code)
            self.assertIn("DB 통합 검사", stream.getvalue())
            self.assertNotIn("CPU 기본 검사", stream.getvalue())

    def test_empty_summary_is_failure(self):
        with redirect_stdout(StringIO()), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(db.publish([]), 1)


if __name__ == "__main__":
    unittest.main()
