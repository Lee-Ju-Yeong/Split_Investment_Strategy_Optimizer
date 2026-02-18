import importlib
import unittest


class TestIssue69CpuBacktestWrapperCompat(unittest.TestCase):
    def test_cpu_backtest_modules_import_from_canonical_paths(self):
        strategy_pkg = importlib.import_module("src.backtest.cpu.strategy")
        self.assertTrue(hasattr(strategy_pkg, "MagicSplitStrategy"))
        self.assertTrue(hasattr(strategy_pkg, "Strategy"))

        portfolio_pkg = importlib.import_module("src.backtest.cpu.portfolio")
        self.assertTrue(hasattr(portfolio_pkg, "Portfolio"))
        self.assertTrue(hasattr(portfolio_pkg, "Trade"))

        execution_pkg = importlib.import_module("src.backtest.cpu.execution")
        self.assertTrue(hasattr(execution_pkg, "BasicExecutionHandler"))

        backtester_pkg = importlib.import_module("src.backtest.cpu.backtester")
        self.assertTrue(hasattr(backtester_pkg, "BacktestEngine"))

    def test_removed_wrapper_modules_are_not_importable(self):
        for module_name in [
            "src.backtester",
            "src.strategy",
            "src.portfolio",
            "src.execution",
            "src.backtest_strategy_gpu",
            "src.daily_stock_tier_batch",
            "src.financial_collector",
            "src.investor_trading_collector",
        ]:
            with self.subTest(module_name=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
