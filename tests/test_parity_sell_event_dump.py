import unittest
from unittest.mock import patch

from src.data_handler import PitRuntimeError
from src.parity_sell_event_dump import (
    BuyEvent,
    CandidateRankRow,
    DailySnapshot,
    PositionSnapshot,
    SellEvent,
    _parse_gpu_buy_events,
    _parse_gpu_candidate_rank_rows,
    _parse_gpu_daily_snapshots,
    _parse_gpu_position_snapshots,
    collect_trade_event_parity_report,
)


def _daily_snapshots():
    cpu_rows = [
        DailySnapshot(source="cpu", date="2024-01-02", total_value=1_000_000.0, cash=319_900.0, stock_count=1),
        DailySnapshot(source="cpu", date="2024-01-03", total_value=1_019_000.0, cash=1_019_000.0, stock_count=0),
    ]
    gpu_rows = [
        DailySnapshot(source="gpu", date="2024-01-02", total_value=1_000_000.0, cash=319_900.0, stock_count=1),
        DailySnapshot(source="gpu", date="2024-01-03", total_value=1_019_000.0, cash=1_019_000.0, stock_count=0),
    ]
    return cpu_rows, gpu_rows


def _position_snapshots():
    cpu_rows = [
        PositionSnapshot(
            source="cpu",
            date="2024-01-02",
            ticker="005930",
            holdings=1,
            quantity=10,
            avg_buy_price=68000.0,
            current_price=68000.0,
            total_value=680000.0,
        )
    ]
    gpu_rows = [
        PositionSnapshot(
            source="gpu",
            date="2024-01-02",
            ticker="005930",
            holdings=1,
            quantity=10,
            avg_buy_price=68000.0,
            current_price=68000.0,
            total_value=680000.0,
        )
    ]
    return cpu_rows, gpu_rows


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
                reason="신규 진입",
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
        cpu_daily, gpu_daily = _daily_snapshots()
        cpu_positions, gpu_positions = _position_snapshots()
        mock_cpu_runner.return_value = (cpu_sell, cpu_buy, cpu_daily, cpu_positions)
        mock_gpu_runner.return_value = (gpu_sell, gpu_buy, gpu_daily, gpu_positions, "[GPU_DEBUG]")

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
        self.assertEqual(payload["comparison_scope"], "structured_trade_and_state_snapshots")
        self.assertTrue(payload["release_decision_fields_complete"])
        self.assertEqual(payload["sell_mismatched_pairs"], 0)
        self.assertEqual(payload["buy_mismatched_pairs"], 0)
        self.assertEqual(payload["daily_snapshot_mismatched_pairs"], 0)
        self.assertEqual(payload["position_snapshot_mismatched_pairs"], 0)
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
            _daily_snapshots()[0],
            _position_snapshots()[0],
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
            _daily_snapshots()[1],
            _position_snapshots()[1],
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
            _daily_snapshots()[0],
            _position_snapshots()[0],
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
            _daily_snapshots()[1],
            _position_snapshots()[1],
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
        mock_cpu_runner.return_value = (cpu_sell, [], _daily_snapshots()[0], _position_snapshots()[0])
        mock_gpu_runner.return_value = (gpu_sell, [], _daily_snapshots()[1], _position_snapshots()[1], "")

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
            _daily_snapshots()[0],
            _position_snapshots()[0],
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
            _daily_snapshots()[1],
            _position_snapshots()[1],
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

    @patch("src.parity_sell_event_dump._run_gpu_and_collect_sell_events")
    @patch("src.parity_sell_event_dump._run_cpu_and_collect_trade_events")
    def test_collect_trade_event_parity_report_flags_snapshot_cash_mismatch(
        self,
        mock_cpu_runner,
        mock_gpu_runner,
    ):
        cpu_daily, gpu_daily = _daily_snapshots()
        cpu_positions, gpu_positions = _position_snapshots()
        gpu_daily = [
            DailySnapshot(source="gpu", date="2024-01-02", total_value=1_000_000.0, cash=300_000.0, stock_count=1),
            gpu_daily[1],
        ]
        mock_cpu_runner.return_value = ([], [], cpu_daily, cpu_positions)
        mock_gpu_runner.return_value = ([], [], gpu_daily, gpu_positions, "")

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
        self.assertTrue(payload["release_decision_fields_complete"])
        self.assertEqual(payload["daily_snapshot_mismatched_pairs"], 1)
        self.assertEqual(payload["daily_snapshot_pairs"][0]["diff"]["cash_diff"], 19900.0)

    def test_parse_gpu_snapshot_markers(self):
        log_text = "\n".join(
            [
                "[PARITY_SNAPSHOT_DAILY] date=2024-01-02|total_value=1000000.000000|cash=319900.000000|stock_count=1",
                "[PARITY_SNAPSHOT_POSITION] date=2024-01-02|ticker=005930|holdings=1|quantity=10|avg_buy_price=68000.000000|current_price=68000.000000|total_value=680000.000000",
            ]
        )

        daily_rows = _parse_gpu_daily_snapshots(log_text)
        position_rows = _parse_gpu_position_snapshots(log_text)

        self.assertEqual(len(daily_rows), 1)
        self.assertEqual(daily_rows[0].cash, 319900.0)
        self.assertEqual(len(position_rows), 1)
        self.assertEqual(position_rows[0].quantity, 10)
        self.assertEqual(position_rows[0].ticker, "005930")

    def test_parse_gpu_candidate_rank_markers(self):
        log_text = "\n".join(
            [
                "[GPU_CANDIDATE_RANK] trade_date=2024-01-02|signal_date=2024-01-01|rank=1|ticker=005930|entry_composite_score_q=8123|flow_score_q=7000|atr_score_q=6000|market_cap_q=400000|atr_14_ratio=0.123400",
                "[GPU_CANDIDATE_RANK] trade_date=2024-01-02|signal_date=2024-01-01|rank=2|ticker=000660|entry_composite_score_q=7123|flow_score_q=6500|atr_score_q=5500|market_cap_q=300000|atr_14_ratio=0.113400",
            ]
        )

        rows = _parse_gpu_candidate_rank_rows(log_text)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].ticker, "005930")
        self.assertEqual(rows[0].rank, 1)
        self.assertEqual(rows[1].entry_composite_score_q, 7123)

    def test_parse_gpu_new_buy_marker_reads_target_price(self):
        log_text = (
            "[GPU_NEW_BUY_CALC] 1, Sim 0, Stock 0(005930) | "
            "Target: 68000.00 | Invest: 680,000 / ExecPrice: 68,000 = Qty: 10"
        )

        rows = _parse_gpu_buy_events(
            log_text,
            trading_dates=["2024-01-01", "2024-01-02"],
            buy_commission_rate=0.0,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].ticker, "005930")
        self.assertEqual(rows[0].trigger_price, 68000.0)

    def test_parse_gpu_add_buy_marker_reads_split_and_target_price(self):
        log_text = "\n".join(
            [
                "[GPU_ADD_BUY_SUMMARY] Day 2, Sim 0 | Buys: 1 | Capital After: 800,000",
                "  └─ Stock 0(005930) | Split: 1 | Target: 67,500.00 | Qty: 10 @ 68,000",
            ]
        )

        rows = _parse_gpu_buy_events(
            log_text,
            trading_dates=["2024-01-01", "2024-01-02", "2024-01-03"],
            buy_commission_rate=0.0,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].ticker, "005930")
        self.assertEqual(rows[0].split, 1)
        self.assertEqual(rows[0].trigger_price, 67500.0)

    @patch("src.parity_sell_event_dump._run_gpu_and_collect_sell_events")
    @patch("src.parity_sell_event_dump._run_cpu_and_collect_trade_events")
    def test_collect_trade_event_parity_report_allows_small_avg_buy_price_snapshot_delta(
        self,
        mock_cpu_runner,
        mock_gpu_runner,
    ):
        cpu_daily, gpu_daily = _daily_snapshots()
        cpu_positions, gpu_positions = _position_snapshots()
        gpu_positions = [
            PositionSnapshot(
                source="gpu",
                date="2024-01-02",
                ticker="005930",
                holdings=1,
                quantity=10,
                avg_buy_price=68000.0015,
                current_price=68000.0,
                total_value=680000.0,
            )
        ]
        mock_cpu_runner.return_value = ([], [], cpu_daily, cpu_positions)
        mock_gpu_runner.return_value = ([], [], gpu_daily, gpu_positions, "")

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

        self.assertTrue(payload["position_snapshot_pairs"][0]["matched"])
        self.assertEqual(payload["position_snapshot_mismatched_pairs"], 0)

    @patch("src.parity_sell_event_dump._run_gpu_and_collect_sell_events")
    @patch("src.parity_sell_event_dump._run_cpu_and_collect_trade_events")
    def test_collect_trade_event_parity_report_captures_candidate_order_matches(
        self,
        mock_cpu_runner,
        mock_gpu_runner,
    ):
        candidate_rows = [
            CandidateRankRow(
                source="cpu",
                trade_date="2024-01-02",
                signal_date="2024-01-01",
                rank=1,
                ticker="005930",
                entry_composite_score_q=8123,
                flow_score_q=7000,
                atr_score_q=6000,
                market_cap_q=400000,
                atr_14_ratio=0.1234,
            )
        ]
        gpu_candidate_rows = [
            CandidateRankRow(
                source="gpu",
                trade_date="2024-01-02",
                signal_date="2024-01-01",
                rank=1,
                ticker="005930",
                entry_composite_score_q=8123,
                flow_score_q=7000,
                atr_score_q=6000,
                market_cap_q=400000,
                atr_14_ratio=0.12340001,
            )
        ]
        mock_cpu_runner.return_value = ([], [], _daily_snapshots()[0], _position_snapshots()[0], candidate_rows)
        mock_gpu_runner.return_value = ([], [], _daily_snapshots()[1], _position_snapshots()[1], gpu_candidate_rows, "")

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

        self.assertEqual(payload["candidate_order_paired_count"], 1)
        self.assertEqual(payload["candidate_order_mismatched_pairs"], 0)
        self.assertTrue(payload["candidate_order_zero_mismatch"])
        self.assertTrue(payload["candidate_order_pairs"][0]["matched"])

    @patch("src.parity_sell_event_dump._run_gpu_and_collect_sell_events")
    @patch("src.parity_sell_event_dump._run_cpu_and_collect_trade_events")
    def test_collect_trade_event_parity_report_flags_candidate_order_mismatch_without_affecting_decision_level(
        self,
        mock_cpu_runner,
        mock_gpu_runner,
    ):
        candidate_rows = [
            CandidateRankRow(
                source="cpu",
                trade_date="2024-01-02",
                signal_date="2024-01-01",
                rank=1,
                ticker="005930",
                entry_composite_score_q=8123,
                flow_score_q=7000,
                atr_score_q=6000,
                market_cap_q=400000,
                atr_14_ratio=0.1234,
            )
        ]
        gpu_candidate_rows = [
            CandidateRankRow(
                source="gpu",
                trade_date="2024-01-02",
                signal_date="2024-01-01",
                rank=1,
                ticker="000660",
                entry_composite_score_q=8123,
                flow_score_q=7000,
                atr_score_q=6000,
                market_cap_q=400000,
                atr_14_ratio=0.1234,
            )
        ]
        mock_cpu_runner.return_value = ([], [], _daily_snapshots()[0], _position_snapshots()[0], candidate_rows)
        mock_gpu_runner.return_value = ([], [], _daily_snapshots()[1], _position_snapshots()[1], gpu_candidate_rows, "")

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

        self.assertEqual(payload["candidate_order_mismatched_pairs"], 1)
        self.assertFalse(payload["candidate_order_zero_mismatch"])
        self.assertFalse(payload["candidate_order_pairs"][0]["matched"])
        self.assertTrue(payload["decision_level_zero_mismatch"])

    @patch("src.parity_sell_event_dump._run_gpu_and_collect_sell_events")
    @patch("src.parity_sell_event_dump._run_cpu_and_collect_trade_events")
    def test_collect_trade_event_parity_report_embeds_candidate_order_and_frozen_manifest_artifacts(
        self,
        mock_cpu_runner,
        mock_gpu_runner,
    ):
        candidate_rows = [
            CandidateRankRow(
                source="cpu",
                trade_date="2024-01-02",
                signal_date="2024-01-01",
                rank=1,
                ticker="005930",
                entry_composite_score_q=8123,
                flow_score_q=7000,
                atr_score_q=6000,
                market_cap_q=400000,
                atr_14_ratio=0.1234,
            )
        ]
        gpu_candidate_rows = [
            CandidateRankRow(
                source="gpu",
                trade_date="2024-01-02",
                signal_date="2024-01-01",
                rank=1,
                ticker="005930",
                entry_composite_score_q=8123,
                flow_score_q=7000,
                atr_score_q=6000,
                market_cap_q=400000,
                atr_14_ratio=0.1234,
            )
        ]
        mock_cpu_runner.return_value = (
            [],
            [],
            _daily_snapshots()[0],
            _position_snapshots()[0],
            candidate_rows,
            {
                "candidate_lookup_summary": {"error_count": 0, "policy": "raise"},
                "run_metrics": {"pit_failure_days_by_code": {}},
                "frozen_candidate_manifest": {
                    "mode": "record_strict",
                    "path": "results/frozen_manifest.json",
                    "sha256": "abc123",
                },
            },
        )
        mock_gpu_runner.return_value = (
            [],
            [],
            _daily_snapshots()[1],
            _position_snapshots()[1],
            gpu_candidate_rows,
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

        self.assertEqual(payload["cpu_candidate_lookup_summary"]["error_count"], 0)
        self.assertEqual(payload["frozen_candidate_manifest"]["mode"], "record_strict")
        self.assertTrue(payload["candidate_order_artifact"]["zero_mismatch"])
        self.assertTrue(payload["certification_artifact"]["candidate_order"]["zero_mismatch"])
        self.assertEqual(
            payload["certification_artifact"]["frozen_candidate_manifest"]["path"],
            "results/frozen_manifest.json",
        )
        self.assertIsNone(payload["certification_artifact"]["pit_failure"])

    @patch("src.parity_sell_event_dump._run_cpu_and_collect_trade_events")
    def test_collect_trade_event_parity_report_returns_structured_pit_failure_payload(
        self,
        mock_cpu_runner,
    ):
        mock_cpu_runner.side_effect = PitRuntimeError(
            "tier12_coverage_gate_failed",
            "coverage gate failed",
            stage="tier12_coverage_gate",
            details={"date": "2024-01-02"},
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
        self.assertFalse(payload["release_decision_fields_complete"])
        self.assertEqual(payload["comparison_scope"], "pit_failure")
        self.assertEqual(payload["pit_failure"]["code"], "tier12_coverage_gate_failed")
        self.assertEqual(payload["certification_artifact"]["pit_failure"]["stage"], "tier12_coverage_gate")
        self.assertEqual(payload["candidate_order_artifact"]["paired_count"], 0)

    @patch("src.parity_sell_event_dump._run_gpu_and_collect_sell_events")
    @patch("src.parity_sell_event_dump._run_cpu_and_collect_trade_events")
    def test_collect_trade_event_parity_report_preserves_cpu_manifest_artifact_on_gpu_pit_failure(
        self,
        mock_cpu_runner,
        mock_gpu_runner,
    ):
        mock_cpu_runner.return_value = (
            [],
            [],
            _daily_snapshots()[0],
            _position_snapshots()[0],
            [],
            {
                "candidate_lookup_summary": {"error_count": 0, "policy": "raise"},
                "run_metrics": {"pit_failure_days_by_code": {}},
                "frozen_candidate_manifest": {
                    "mode": "replay_strict",
                    "path": "results/frozen_manifest.json",
                    "sha256": "deadbeef",
                },
            },
        )
        mock_gpu_runner.side_effect = PitRuntimeError(
            "frozen_manifest_sha_mismatch",
            "sha mismatch",
            stage="strict_frozen_manifest_replay",
            details={"path": "results/frozen_manifest.json"},
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

        self.assertEqual(payload["pit_failure"]["code"], "frozen_manifest_sha_mismatch")
        self.assertEqual(payload["frozen_candidate_manifest"]["mode"], "replay_strict")
        self.assertEqual(
            payload["certification_artifact"]["frozen_candidate_manifest"]["path"],
            "results/frozen_manifest.json",
        )


if __name__ == "__main__":
    unittest.main()
