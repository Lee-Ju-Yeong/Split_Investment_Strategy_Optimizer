import importlib
import os
import subprocess
import sys
import unittest


class TestIssue97RetainedWrapperDrift(unittest.TestCase):
    def test_pipeline_batch_wrapper_exports_canonical_symbols(self):
        wrapper = importlib.import_module("src.pipeline_batch")
        canonical = importlib.import_module("src.pipeline.batch")
        self.assertIs(wrapper.run_pipeline_batch, canonical.run_pipeline_batch)
        self.assertIs(wrapper._build_arg_parser, canonical._build_arg_parser)
        self.assertIs(wrapper.main, canonical.main)

    def test_ohlcv_batch_wrapper_exports_canonical_symbols(self):
        wrapper = importlib.import_module("src.ohlcv_batch")
        canonical = importlib.import_module("src.pipeline.ohlcv_batch")
        self.assertIs(wrapper.run_ohlcv_batch, canonical.run_ohlcv_batch)
        self.assertIs(
            wrapper.get_ohlcv_ticker_universe,
            canonical.get_ohlcv_ticker_universe,
        )
        self.assertIs(wrapper._build_arg_parser, canonical._build_arg_parser)
        self.assertIs(wrapper.main, canonical.main)

    def test_pipeline_batch_help_succeeds(self):
        self._assert_help("src.pipeline_batch", "Run batch pipeline")

    def test_ohlcv_batch_help_succeeds(self):
        self._assert_help("src.ohlcv_batch", "Run resume-capable OHLCV batch")

    def _assert_help(self, module_name, expected_phrase):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env = dict(os.environ)
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"{module_name} --help failed.\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n",
        )
        self.assertIn(expected_phrase, result.stdout)


if __name__ == "__main__":
    unittest.main()
