import importlib
import sys
import unittest
from pathlib import Path


class TestIssue69CpuBacktestWrapperCompat(unittest.TestCase):
    def test_cpu_backtest_modules_import_in_package_and_legacy_modes(self):
        """
        Issue #69:
        - After moving CPU backtest core under `src/backtest/cpu/*`,
          top-level wrappers must keep working.

        Legacy mode is used by some unit tests that add `src/` to sys.path and then:
        - `import strategy`
        - `import portfolio`
        """

        # Package imports (preferred)
        strategy_pkg = importlib.import_module("src.strategy")
        self.assertTrue(hasattr(strategy_pkg, "MagicSplitStrategy"))
        self.assertTrue(hasattr(strategy_pkg, "Strategy"))

        portfolio_pkg = importlib.import_module("src.portfolio")
        self.assertTrue(hasattr(portfolio_pkg, "Portfolio"))
        self.assertTrue(hasattr(portfolio_pkg, "Trade"))

        execution_pkg = importlib.import_module("src.execution")
        self.assertTrue(hasattr(execution_pkg, "BasicExecutionHandler"))

        backtester_pkg = importlib.import_module("src.backtester")
        self.assertTrue(hasattr(backtester_pkg, "BacktestEngine"))

        # Legacy imports (src/ on sys.path)
        repo_root = Path(__file__).resolve().parent.parent
        src_dir = repo_root / "src"
        sys.path.insert(0, str(src_dir))
        try:
            strategy_legacy = importlib.import_module("strategy")
            self.assertTrue(hasattr(strategy_legacy, "MagicSplitStrategy"))
            self.assertTrue(hasattr(strategy_legacy, "Strategy"))

            portfolio_legacy = importlib.import_module("portfolio")
            self.assertTrue(hasattr(portfolio_legacy, "Portfolio"))
            self.assertTrue(hasattr(portfolio_legacy, "Trade"))

            execution_legacy = importlib.import_module("execution")
            self.assertTrue(hasattr(execution_legacy, "BasicExecutionHandler"))

            backtester_legacy = importlib.import_module("backtester")
            self.assertTrue(hasattr(backtester_legacy, "BacktestEngine"))
        finally:
            # Clean up path injection for other tests.
            try:
                sys.path.remove(str(src_dir))
            except ValueError:
                pass

            # Keep cleanup minimal and targeted.
            for name in [
                "strategy",
                "portfolio",
                "execution",
                "backtester",
                "backtest",
                "backtest.cpu",
                "backtest.cpu.strategy",
                "backtest.cpu.portfolio",
                "backtest.cpu.execution",
                "backtest.cpu.backtester",
            ]:
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()

