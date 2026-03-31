import unittest

from src.analysis.promotion_shortlist_recompression import (
    DEFAULT_FAMILY_KEYS,
    DEFAULT_RESULT_EQUALITY_KEYS,
    select_family_representatives,
)


def _base_row(candidate_id: str, stop_loss: str) -> dict[str, str]:
    return {
        "shortlist_candidate_id": candidate_id,
        "max_stocks": "20",
        "order_investment_ratio": "0.022",
        "additional_buy_drop_rate": "0.065",
        "sell_profit_rate": "0.21",
        "additional_buy_priority": "lowest_order",
        "stop_loss_rate": stop_loss,
        "max_splits_limit": "10",
        "max_inactivity_period": "63",
        "promotion_fold_pass_rate": "1.0",
        "promotion_oos_cagr_median": "0.10",
        "promotion_oos_calmar_median": "0.40",
        "promotion_oos_mdd_depth_worst": "0.41",
        "promotion_oos_is_calmar_ratio_median": "1.20",
        "hard_gate_pass": "True",
        "hard_gate_fail_reasons": "",
        "robust_score": "0.30",
    }


class PromotionShortlistRecompressionTest(unittest.TestCase):
    def test_same_results_keep_smallest_stop_loss(self):
        rows = [
            _base_row("17", "-0.9"),
            _base_row("18", "-0.8"),
            _base_row("19", "-0.7"),
        ]

        representatives = select_family_representatives(
            rows,
            family_keys=DEFAULT_FAMILY_KEYS,
            result_keys=DEFAULT_RESULT_EQUALITY_KEYS,
        )

        self.assertEqual(len(representatives), 1)
        self.assertEqual(representatives[0].row["shortlist_candidate_id"], "17")

    def test_different_results_keep_multiple_representatives(self):
        rows = [
            _base_row("17", "-0.9"),
            _base_row("18", "-0.8"),
            _base_row("19", "-0.7"),
        ]
        rows[2]["promotion_oos_cagr_median"] = "0.12"

        representatives = select_family_representatives(
            rows,
            family_keys=DEFAULT_FAMILY_KEYS,
            result_keys=DEFAULT_RESULT_EQUALITY_KEYS,
        )

        selected_ids = [item.row["shortlist_candidate_id"] for item in representatives]
        self.assertEqual(len(representatives), 2)
        self.assertIn("17", selected_ids)
        self.assertIn("19", selected_ids)

    def test_non_hard_gate_rows_are_excluded(self):
        rows = [
            _base_row("17", "-0.9"),
            _base_row("18", "-0.8"),
        ]
        rows[1]["hard_gate_pass"] = "False"

        representatives = select_family_representatives(
            rows,
            family_keys=DEFAULT_FAMILY_KEYS,
            result_keys=DEFAULT_RESULT_EQUALITY_KEYS,
        )

        self.assertEqual(len(representatives), 1)
        self.assertEqual(representatives[0].row["shortlist_candidate_id"], "17")


if __name__ == "__main__":
    unittest.main()
