import importlib
import os
import subprocess
import sys
import unittest


class TestIssue69EntrypointCompat(unittest.TestCase):
    def test_core_entrypoints_import_without_gpu_deps(self):
        """
        Issue #69 safety net (PR-0):
        Ensure key orchestrators remain importable even when GPU deps are unavailable.
        """
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        script = r"""
import importlib
import importlib.abc
import os
import sys

sys.path.insert(0, os.getcwd())

BLOCK = {"cupy", "cudf"}

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] in BLOCK:
            raise ModuleNotFoundError(f"Blocked import: {fullname}")
        return None

sys.meta_path.insert(0, _Blocker())

for name in [
    "src.walk_forward_analyzer",
    "src.parameter_simulation_gpu",
    "src.pipeline_batch",
    "src.ticker_universe_batch",
    "src.ohlcv_batch",
    "src.daily_stock_tier_batch",
]:
    importlib.import_module(name)
"""

        env = dict(os.environ)
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg="Import should succeed without GPU deps.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n",
        )

    def test_entrypoints_export_expected_symbols(self):
        pipeline_batch = importlib.import_module("src.pipeline_batch")
        self.assertTrue(hasattr(pipeline_batch, "run_pipeline_batch"))

        ticker_universe_batch = importlib.import_module("src.ticker_universe_batch")
        self.assertTrue(hasattr(ticker_universe_batch, "run_ticker_universe_batch"))

        ohlcv_batch = importlib.import_module("src.ohlcv_batch")
        self.assertTrue(hasattr(ohlcv_batch, "run_ohlcv_batch"))

        tier_batch = importlib.import_module("src.daily_stock_tier_batch")
        self.assertTrue(hasattr(tier_batch, "run_daily_stock_tier_batch"))

        wfo = importlib.import_module("src.walk_forward_analyzer")
        self.assertTrue(hasattr(wfo, "run_walk_forward_analysis"))

        sim = importlib.import_module("src.parameter_simulation_gpu")
        self.assertTrue(hasattr(sim, "find_optimal_parameters"))
