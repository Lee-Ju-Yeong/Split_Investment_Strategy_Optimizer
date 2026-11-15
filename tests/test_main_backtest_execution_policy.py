"""CPU 진입점이 매도 호가 반올림 설정을 전달·기록하는지 확인한다 (전체 런타임 의존성 필요: yaml)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import main_backtest


class TestMainBacktestExecutionPolicy(unittest.TestCase):
    def test_cpu_entry_point_forwards_the_key(self):
        self.assertIn("sell_price_tick_rounding", main_backtest._EXECUTION_PARAM_KEYS)

    def test_run_manifest_records_applied_value(self):
        with tempfile.TemporaryDirectory() as result_dir:
            path = main_backtest._write_run_manifest(
                result_dir=result_dir,
                strategy_params={},
                backtest_settings={"start_date": "2024-01-02", "end_date": "2024-01-03"},
                universe_mode="optimistic_survivor",
                price_basis="adjusted",
                adjusted_gate="2013-11-20",
                sell_price_tick_rounding="down",
            )
            with open(path, encoding="utf-8") as f:
                manifest = json.load(f)
        self.assertEqual(manifest["execution_policy"], {"sell_price_tick_rounding": "down"})


if __name__ == "__main__":
    unittest.main()
