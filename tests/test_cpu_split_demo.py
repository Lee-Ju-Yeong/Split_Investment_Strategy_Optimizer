"""실제 CPU 경로의 고정 합성 거래·현금·입출력 계약을 검사한다."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import csv
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from examples import cpu_split_demo as demo


ROOT = Path(__file__).resolve().parents[1]


class CpuSplitDemoTests(unittest.TestCase):
    def run_cli(self, *arguments, code=None):
        command = [sys.executable, "-m", "examples.cpu_split_demo", *arguments]
        if code is not None:
            command = [sys.executable, "-c", code]
        env = dict(os.environ, MAGICSPLIT_CONFIG_PATH="/missing/demo-must-not-read.yaml")
        return subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                              text=True, encoding="utf-8", timeout=20)

    def test_exact_trades_and_all_daily_balances(self):
        result = demo.run_simulation()
        demo.verify_result(result)
        # 독립적인 승인 표. 구현의 EXPECTED_*로 기대값을 만들지 않는다.
        expected_trades = [
            ("2024-01-03", "2024-01-02", "BUY", 1, 10, 10000, 15, 0, 899985),
            ("2024-01-04", "2024-01-03", "BUY", 2, 10, 9400, 14, 0, 805971),
            ("2024-01-08", "2024-01-05", "SELL", 2, 10, 10400, 15, 188, 909768),
            ("2024-01-09", "2024-01-08", "SELL", 1, 10, 11100, 16, 201, 1020551),
        ]
        keys = ("trade_date", "signal_date", "side", "split", "quantity", "price",
                "commission", "tax", "cash_after")
        self.assertEqual([tuple(row[key] for key in keys) for row in result["trades"]], expected_trades)
        self.assertTrue(all(row["ticker"] == "SYNTH001" for row in result["trades"]))
        self.assertEqual([row["cash"] for row in result["daily_equity"]],
                         [1000000, 899985, 805971, 805971, 909768, 1020551, 1020551])
        self.assertEqual([row["quantity"] for row in result["daily_equity"]], [0, 10, 20, 20, 10, 0, 0])
        self.assertEqual([row["total_value"] for row in result["daily_equity"]],
                         [1000000, 994985, 993971, 1013971, 1020768, 1020551, 1020551])
        self.assertEqual(result["summary"], dict(initial_cash=1000000, final_cash=1020551,
                         final_value=1020551, total_commission=60, total_tax=389, net_pnl=20551))

    def test_cash_and_asset_accounting_includes_every_cost(self):
        result = demo.run_simulation()
        cash = 1_000_000
        for row in result["trades"]:
            gross = row["quantity"] * row["price"]
            cash += (-gross - row["commission"] if row["side"] == "BUY"
                     else gross - row["commission"] - row["tax"])
            self.assertEqual(cash, row["cash_after"])
        for row in result["daily_equity"]:
            self.assertEqual(row["cash"] + row["market_value"], row["total_value"])
        self.assertEqual(cash - 1_000_000, 20_551)

    def test_first_day_and_cooldown_have_no_trades(self):
        result = demo.run_simulation()
        dates = [row["trade_date"] for row in result["trades"]]
        self.assertNotIn("2024-01-02", dates)
        self.assertNotIn("2024-01-10", dates)
        self.assertEqual(result["daily_equity"][-1]["quantity"], 0)

    def test_max_splits_prevents_third_buy(self):
        prices = demo.make_price_rows()
        prices[2]["low_price"] = 8400  # 3차 매수 조건에 해당해도 최대 2회만 허용.
        result = demo.run_simulation(prices=prices)
        self.assertEqual([row["split"] for row in result["trades"] if row["side"] == "BUY"], [1, 2])
        demo.verify_result(result)

    def test_tier_can_block_additional_buy_without_changing_strategy(self):
        tiers = demo.make_tier_rows()
        # 첫 신규 매수의 기준일만 등급 1. 이후에도 추가 매수 자격을 주지 않는다.
        for row in tiers[1:]:
            row["tier"] = 3
        result = demo.run_simulation(tiers=tiers)
        self.assertEqual([row["side"] for row in result["trades"]], ["BUY", "SELL"])
        self.assertEqual(result["daily_equity"][2]["quantity"], 10)

    def test_valid_noneligible_tiers_preserve_all_cash(self):
        for tier in (2, 3):
            with self.subTest(tier=tier):
                tiers = [dict(date=row["date"], tier=tier) for row in demo.make_tier_rows()]
                result = demo.run_simulation(tiers=tiers)
                self.assertEqual(result["trades"], [])
                self.assertTrue(all(row["quantity"] == 0 and row["cash"] == row["total_value"] == 1_000_000
                                    for row in result["daily_equity"]))

    def test_future_prices_cannot_change_past_results(self):
        baseline = demo.run_simulation()
        prices = demo.make_price_rows()
        for row in prices[4:]:
            for field in demo.PRICE_FIELDS:
                row[field] *= 2
        changed = demo.run_simulation(prices=prices)
        self.assertEqual(changed["daily_equity"][:4], baseline["daily_equity"][:4])
        self.assertEqual([r for r in changed["trades"] if r["trade_date"] <= "2024-01-05"],
                         [r for r in baseline["trades"] if r["trade_date"] <= "2024-01-05"])

    def test_asof_and_previous_trading_date_never_use_future_rows(self):
        handler = demo.InMemoryDataHandler(demo.make_price_rows(), demo.make_tier_rows())
        dates = handler.get_trading_dates("2024-01-02", "2024-01-10")
        self.assertIsNone(handler.get_previous_trading_date(dates, 0))
        self.assertEqual(handler.get_previous_trading_date(dates, 4), pd.Timestamp("2024-01-05"))
        row = handler.get_stock_row_as_of("SYNTH001", "2024-01-07", dates[0], dates[-1])
        self.assertEqual(row.name, pd.Timestamp("2024-01-05"))
        self.assertEqual(row["close_price"], 10400)
        handler.get_stock_tier_as_of("SYNTH001", "2024-01-07")
        self.assertTrue(all(source <= requested for requested, source in handler.lookups))

    def test_every_price_day_is_required_even_final_untraded_day(self):
        for index, date in enumerate(demo.TRADING_DATES):
            with self.subTest(date=date):
                prices = demo.make_price_rows()
                del prices[index]
                with self.assertRaisesRegex(ValueError, date):
                    demo.run_simulation(prices=prices)

    def test_missing_tier_and_duplicate_dates_fail_before_run(self):
        tiers = demo.make_tier_rows()[:-1]
        with self.assertRaisesRegex(ValueError, "2024-01-10"):
            demo.run_simulation(tiers=tiers)
        prices = demo.make_price_rows()
        prices.append(dict(prices[-1]))
        with self.assertRaisesRegex(ValueError, "중복"):
            demo.run_simulation(prices=prices)

    def test_bad_values_and_missing_fields_are_rejected(self):
        for field, value in (("close_price", float("nan")), ("open_price", 0),
                             ("high_price", 9000), ("atr_14_ratio", -1),
                             ("flow5_mcap", float("inf")), ("cheap_score", 2)):
            with self.subTest(field=field, value=value):
                prices = demo.make_price_rows()
                prices[0][field] = value
                with self.assertRaisesRegex(ValueError, "SYNTH001.*2024-01-02"):
                    demo.run_simulation(prices=prices)
        prices = demo.make_price_rows()
        for row in prices:
            del row["close_price"]
        with self.assertRaisesRegex(ValueError, "close_price"):
            demo.run_simulation(prices=prices)
        tiers = demo.make_tier_rows()
        tiers[-1]["tier"] = None
        with self.assertRaisesRegex(ValueError, "2024-01-10"):
            demo.run_simulation(tiers=tiers)

    def test_real_cpu_path_does_not_open_database_or_network(self):
        targets = ("src.data_handler.DataHandler.__init__", "mysql.connector.connect",
                   "mysql.connector.pooling.MySQLConnectionPool", "socket.create_connection",
                   "socket.socket.connect")
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"MAGICSPLIT_CONFIG_PATH": "/missing/no-settings.yaml"}))
            guards = [stack.enter_context(patch(name, side_effect=AssertionError(name))) for name in targets]
            result = demo.run_simulation()
            demo.verify_result(result)
            for guard in guards:
                guard.assert_not_called()

    def test_repeated_cli_runs_and_results_are_identical(self):
        self.assertEqual(demo.run_simulation(), demo.run_simulation())
        first, second = self.run_cli(), self.run_cli()
        self.assertEqual((first.returncode, second.returncode), (0, 0), first.stderr + second.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(first.stderr + second.stderr, "")
        self.assertIn("최종 현금·자산 1,020,551원", first.stdout)

    def test_cli_help_and_bad_argument_exit_codes(self):
        help_result, bad = self.run_cli("--help"), self.run_cli("--unknown")
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--output-dir", help_result.stdout)
        self.assertEqual(bad.returncode, 2)
        self.assertNotIn("확인 완료", bad.stdout)

    def test_cli_mismatch_has_nonzero_exit_and_no_success(self):
        result = self.run_cli(code="from examples import cpu_split_demo as d; "
                              "d.STRATEGY_SETTINGS['order_investment_ratio']=0.2; d.main()")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("기대값", result.stderr)
        self.assertNotIn("확인 완료", result.stdout)

    def test_result_verifier_rejects_a_changed_cash_value(self):
        result = deepcopy(demo.run_simulation())
        result["daily_equity"][-1]["cash"] += 1
        with self.assertRaisesRegex(RuntimeError, "기대값"):
            demo.verify_result(result)

    def test_csv_matches_actual_result_and_is_byte_repeatable(self):
        result = demo.run_simulation()
        with tempfile.TemporaryDirectory() as temp:
            directories = [Path(temp) / name for name in ("first", "second")]
            for directory in directories:
                cli = self.run_cli("--output-dir", str(directory))
                self.assertEqual(cli.returncode, 0, cli.stderr)
                self.assertIn("CSV 저장 완료", cli.stdout)
                for name, key in (("trades.csv", "trades"), ("daily_equity.csv", "daily_equity")):
                    path = directory / name
                    self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
                    with path.open(encoding="utf-8-sig", newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(rows, [{k: str(v) for k, v in row.items()} for row in result[key]])
            for name in ("trades.csv", "daily_equity.csv"):
                self.assertEqual((directories[0] / name).read_bytes(), (directories[1] / name).read_bytes())

    def test_existing_directory_is_preserved_and_cli_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            sentinel = directory / "keep.txt"
            sentinel.write_text("original", encoding="utf-8")
            cli = self.run_cli("--output-dir", str(directory))
            self.assertNotEqual(cli.returncode, 0)
            self.assertNotIn("확인 완료", cli.stdout)
            self.assertEqual(sentinel.read_text(), "original")
            self.assertEqual(list(directory.iterdir()), [sentinel])

    def test_write_failure_cleans_only_own_files_and_has_no_success(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "failed"
            def fail(_rows):
                (directory / "keep.txt").write_text("unrelated", encoding="utf-8")
                raise OSError("simulated write failure")
            output = StringIO()
            with patch.object(demo.csv.DictWriter, "writerows", side_effect=fail):
                with redirect_stdout(output), self.assertRaisesRegex(OSError, "simulated"):
                    demo.main(["--output-dir", str(directory)])
            self.assertNotIn("확인 완료", output.getvalue())
            self.assertEqual([p.name for p in directory.iterdir()], ["keep.txt"])

    def test_cancellation_cleans_created_files(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "cancelled"
            with patch.object(demo.csv.DictWriter, "writerows", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    demo.save_csv(demo.run_simulation(), directory)
            self.assertFalse(directory.exists())


if __name__ == "__main__":
    unittest.main()
