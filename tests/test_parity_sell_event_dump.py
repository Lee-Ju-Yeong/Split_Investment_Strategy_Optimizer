import unittest
from unittest.mock import patch

from src.parity_sell_event_dump import BuyEvent, SellEvent, collect_trade_event_parity_report


class TestParitySellEventDump(unittest.TestCase):
    @patch("src.parity_sell_event_dump._run_gpu_and_collect_sell_events")
    @patch("src.parity_sell_event_dump._run_cpu_and_collect_trade_events")
    def test_collect_trade_event_parity_report_marks_zero_mismatch_when_all_pairs_match(
        self,
        mock_cpu_runner,
        mock_gpu_runner,
    ):
        cpu_sell = [
            SellEvent(
                source="cpu",
                date="2024-01-03",
                ticker="005930",
                quantity=10,
                execution_price=70000.0,
                net_revenue=699000.0,
                reason="수익 실현",
                order=1,
                trigger_price=70000.0,
            )
        ]
        cpu_buy = [
            BuyEvent(
                source="cpu",
                date="2024-01-02",
                ticker="005930",
                quantity=10,
                execution_price=68000.0,
                total_cost=680100.0,
                reason="신규 매수",
                order=1,
                trigger_price=68000.0,
            )
        ]
        gpu_sell = [
            SellEvent(
                source="gpu",
                date="2024-01-03",
                ticker="005930",
                quantity=10,
                execution_price=70000.0,
                net_revenue=699000.0,
                reason="Profit-Taking",
                split=0,
                trigger_price=70000.0,
            )
        ]
        gpu_buy = [
            BuyEvent(
                source="gpu",
                date="2024-01-02",
                ticker="005930",
                quantity=10,
                execution_price=68000.0,
                total_cost=680100.0,
                reason="신규 매수",
                split=0,
                trigger_price=68000.0,
            )
        ]
        mock_cpu_runner.return_value = (cpu_sell, cpu_buy)
        mock_gpu_runner.return_value = (gpu_sell, gpu_buy, "[GPU_DEBUG]")

        payload = collect_trade_event_parity_report(
            config={"execution_params": {}, "strategy_params": {}, "database": {}},
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_cash=1_000_000.0,
            params={"max_stocks": 10},
            candidate_source_mode="tier",
            use_weekly_alpha_gate=False,
            parity_mode="strict",
            universe_mode="strict_pit",
        )

        self.assertTrue(payload["decision_level_zero_mismatch"])
        self.assertEqual(payload["comparison_scope"], "structured_trade_events")
        self.assertFalse(payload["release_decision_fields_complete"])
        self.assertEqual(payload["sell_mismatched_pairs"], 0)
        self.assertEqual(payload["buy_mismatched_pairs"], 0)
        self.assertEqual(payload["cpu_sell_events_count"], 1)
        self.assertEqual(payload["gpu_buy_events_count"], 1)
        self.assertEqual(payload["sell_pairs"][0]["matched"], True)
        self.assertEqual(payload["buy_pairs"][0]["matched"], True)
        self.assertEqual(payload["gpu_log_text"], "[GPU_DEBUG]")

    @patch("src.parity_sell_event_dump._run_gpu_and_collect_sell_events")
    @patch("src.parity_sell_event_dump._run_cpu_and_collect_trade_events")
    def test_collect_trade_event_parity_report_counts_sell_and_buy_mismatches(
        self,
        mock_cpu_runner,
        mock_gpu_runner,
    ):
        mock_cpu_runner.return_value = (
            [
                SellEvent(
                    source="cpu",
                    date="2024-01-03",
                    ticker="005930",
                    quantity=10,
                    execution_price=70000.0,
                    net_revenue=699000.0,
                    reason="수익 실현",
                    order=2,
                    trigger_price=70200.0,
                )
            ],
            [
                BuyEvent(
                    source="cpu",
                    date="2024-01-02",
                    ticker="005930",
                    quantity=10,
                    execution_price=68000.0,
                    total_cost=680100.0,
                    reason="추가 매수(하락)",
                    order=2,
                    trigger_price=67500.0,
                )
            ],
        )
        mock_gpu_runner.return_value = (
            [
                SellEvent(
                    source="gpu",
                    date="2024-01-03",
                    ticker="005930",
                    quantity=10,
                    execution_price=70000.0,
                    net_revenue=699000.0,
                    reason="Stop-Loss",
                    split=0,
                    trigger_price=70000.0,
                )
            ],
            [
                BuyEvent(
                    source="gpu",
                    date="2024-01-02",
                    ticker="005930",
                    quantity=10,
                    execution_price=68000.0,
                    total_cost=680100.0,
                    reason="신규 매수",
                    split=0,
                    trigger_price=68000.0,
                )
            ],
            "",
        )

        payload = collect_trade_event_parity_report(
            config={"execution_params": {}, "strategy_params": {}, "database": {}},
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_cash=1_000_000.0,
            params={"max_stocks": 10},
            candidate_source_mode="tier",
            use_weekly_alpha_gate=False,
            parity_mode="strict",
            universe_mode="strict_pit",
        )

        self.assertFalse(payload["decision_level_zero_mismatch"])
        self.assertEqual(payload["sell_mismatched_pairs"], 1)
        self.assertEqual(payload["buy_mismatched_pairs"], 1)
        self.assertFalse(payload["sell_pairs"][0]["matched"])
        self.assertFalse(payload["buy_pairs"][0]["matched"])
        self.assertFalse(payload["sell_pairs"][0]["diff"]["reason_same"])
        self.assertFalse(payload["buy_pairs"][0]["diff"]["reason_same"])

    @patch("src.parity_sell_event_dump._run_gpu_and_collect_sell_events")
    @patch("src.parity_sell_event_dump._run_cpu_and_collect_trade_events")
    def test_collect_trade_event_parity_report_rejects_add_buy_when_cpu_order_is_first_split(
        self,
        mock_cpu_runner,
        mock_gpu_runner,
    ):
        mock_cpu_runner.return_value = (
            [],
            [
                BuyEvent(
                    source="cpu",
                    date="2024-01-02",
                    ticker="005930",
                    quantity=10,
                    execution_price=68000.0,
                    total_cost=680100.0,
                    reason="추가 매수(하락)",
                    order=1,
                    trigger_price=67500.0,
                )
            ],
        )
        mock_gpu_runner.return_value = (
            [],
            [
                BuyEvent(
                    source="gpu",
                    date="2024-01-02",
                    ticker="005930",
                    quantity=10,
                    execution_price=68000.0,
                    total_cost=680100.0,
                    reason="추가 매수(하락)",
                    split=None,
                    trigger_price=67500.0,
                )
            ],
            "",
        )

        payload = collect_trade_event_parity_report(
            config={"execution_params": {}, "strategy_params": {}, "database": {}},
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_cash=1_000_000.0,
            params={"max_stocks": 10},
            candidate_source_mode="tier",
            use_weekly_alpha_gate=False,
            parity_mode="strict",
            universe_mode="strict_pit",
        )

        self.assertFalse(payload["decision_level_zero_mismatch"])
        self.assertFalse(payload["buy_pairs"][0]["matched"])
        self.assertFalse(payload["buy_pairs"][0]["diff"]["order_match"])

    @patch("src.parity_sell_event_dump._run_gpu_and_collect_sell_events")
    @patch("src.parity_sell_event_dump._run_cpu_and_collect_trade_events")
    def test_collect_trade_event_parity_report_avoids_cascading_mismatch_when_gpu_has_extra_event(
        self,
        mock_cpu_runner,
        mock_gpu_runner,
    ):
        cpu_sell = [
            SellEvent(
                source="cpu",
                date="2024-01-03",
                ticker="005930",
                quantity=10,
                execution_price=70000.0,
                net_revenue=699000.0,
                reason="수익 실현",
                order=1,
                trigger_price=70000.0,
            ),
            SellEvent(
                source="cpu",
                date="2024-01-03",
                ticker="005930",
                quantity=5,
                execution_price=71000.0,
                net_revenue=354500.0,
                reason="손절매 (평균가 기준)",
                order=2,
                trigger_price=71000.0,
            ),
        ]
        gpu_sell = [
            SellEvent(
                source="gpu",
                date="2024-01-03",
                ticker="005930",
                quantity=10,
                execution_price=70000.0,
                net_revenue=699000.0,
                reason="Profit-Taking",
                split=0,
                trigger_price=70000.0,
            ),
            SellEvent(
                source="gpu",
                date="2024-01-03",
                ticker="005930",
                quantity=3,
                execution_price=70500.0,
                net_revenue=211500.0,
                reason="Profit-Taking",
                split=1,
                trigger_price=70500.0,
            ),
            SellEvent(
                source="gpu",
                date="2024-01-03",
                ticker="005930",
                quantity=5,
                execution_price=71000.0,
                net_revenue=354500.0,
                reason="Stop-Loss",
                split=1,
                trigger_price=71000.0,
            ),
        ]
        mock_cpu_runner.return_value = (cpu_sell, [])
        mock_gpu_runner.return_value = (gpu_sell, [], "")

        payload = collect_trade_event_parity_report(
            config={"execution_params": {}, "strategy_params": {}, "database": {}},
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_cash=1_000_000.0,
            params={"max_stocks": 10},
            candidate_source_mode="tier",
            use_weekly_alpha_gate=False,
            parity_mode="strict",
            universe_mode="strict_pit",
        )

        self.assertEqual(payload["sell_matched_pairs"], 2)
        self.assertEqual(payload["sell_mismatched_pairs"], 1)
        self.assertTrue(payload["sell_pairs"][0]["matched"])
        self.assertTrue(payload["sell_pairs"][1]["matched"])
        self.assertIsNone(payload["sell_pairs"][2]["cpu"])
        self.assertEqual(payload["sell_pairs"][2]["gpu"]["quantity"], 3)

    @patch("src.parity_sell_event_dump._run_gpu_and_collect_sell_events")
    @patch("src.parity_sell_event_dump._run_cpu_and_collect_trade_events")
    def test_collect_trade_event_parity_report_avoids_cascade_when_gpu_has_extra_sell_event(
        self,
        mock_cpu_runner,
        mock_gpu_runner,
    ):
        mock_cpu_runner.return_value = (
            [
                SellEvent(
                    source="cpu",
                    date="2024-01-03",
                    ticker="005930",
                    quantity=10,
                    execution_price=70000.0,
                    net_revenue=699000.0,
                    reason="수익 실현",
                    order=1,
                    trigger_price=70000.0,
                ),
                SellEvent(
                    source="cpu",
                    date="2024-01-05",
                    ticker="000660",
                    quantity=5,
                    execution_price=120000.0,
                    net_revenue=598500.0,
                    reason="수익 실현",
                    order=1,
                    trigger_price=120000.0,
                ),
            ],
            [],
        )
        mock_gpu_runner.return_value = (
            [
                SellEvent(
                    source="gpu",
                    date="2024-01-03",
                    ticker="005930",
                    quantity=10,
                    execution_price=70000.0,
                    net_revenue=699000.0,
                    reason="Profit-Taking",
                    split=0,
                    trigger_price=70000.0,
                ),
                SellEvent(
                    source="gpu",
                    date="2024-01-04",
                    ticker="035420",
                    quantity=3,
                    execution_price=200000.0,
                    net_revenue=598000.0,
                    reason="Profit-Taking",
                    split=0,
                    trigger_price=200000.0,
                ),
                SellEvent(
                    source="gpu",
                    date="2024-01-05",
                    ticker="000660",
                    quantity=5,
                    execution_price=120000.0,
                    net_revenue=598500.0,
                    reason="Profit-Taking",
                    split=0,
                    trigger_price=120000.0,
                ),
            ],
            [],
            "",
        )

        payload = collect_trade_event_parity_report(
            config={"execution_params": {}, "strategy_params": {}, "database": {}},
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_cash=1_000_000.0,
            params={"max_stocks": 10},
            candidate_source_mode="tier",
            use_weekly_alpha_gate=False,
            parity_mode="strict",
            universe_mode="strict_pit",
        )

        self.assertEqual(payload["sell_matched_pairs"], 2)
        self.assertEqual(payload["sell_mismatched_pairs"], 1)
        self.assertEqual(sum(1 for row in payload["sell_pairs"] if row["matched"]), 2)
        unmatched_rows = [row for row in payload["sell_pairs"] if not row["matched"]]
        self.assertEqual(len(unmatched_rows), 1)
        self.assertIsNone(unmatched_rows[0]["cpu"])
        self.assertEqual(unmatched_rows[0]["gpu"]["ticker"], "035420")


if __name__ == "__main__":
    unittest.main()
