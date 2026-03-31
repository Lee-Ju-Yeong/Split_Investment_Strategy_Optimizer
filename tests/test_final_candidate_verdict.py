import unittest

import pandas as pd

from src.analysis.final_candidate_verdict import _reorder_candidate_summary


class FinalCandidateVerdictTest(unittest.TestCase):
    def test_reorder_candidate_summary_promotes_requested_candidate(self):
        summary_df = pd.DataFrame(
            [
                {
                    "shortlist_candidate_id": 19,
                    "hard_gate_pass": True,
                    "selection_rank": 1,
                    "selection_role": "champion",
                },
                {
                    "shortlist_candidate_id": 17,
                    "hard_gate_pass": True,
                    "selection_rank": 2,
                    "selection_role": "reserve",
                },
                {
                    "shortlist_candidate_id": 20,
                    "hard_gate_pass": True,
                    "selection_rank": 3,
                    "selection_role": "reserve",
                },
            ]
        )

        ordered = _reorder_candidate_summary(
            summary_df,
            champion_candidate_id=17,
            reserve_count=2,
        )

        self.assertEqual(int(ordered.iloc[0]["shortlist_candidate_id"]), 17)
        self.assertEqual(str(ordered.iloc[0]["selection_role"]), "champion")
        self.assertEqual(str(ordered.iloc[1]["selection_role"]), "reserve")
        self.assertEqual(str(ordered.iloc[2]["selection_role"]), "reserve")

    def test_reorder_candidate_summary_rejects_non_passing_candidate(self):
        summary_df = pd.DataFrame(
            [
                {
                    "shortlist_candidate_id": 19,
                    "hard_gate_pass": False,
                },
                {
                    "shortlist_candidate_id": 17,
                    "hard_gate_pass": True,
                },
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "must already pass the promotion hard gate",
        ):
            _reorder_candidate_summary(
                summary_df,
                champion_candidate_id=19,
                reserve_count=2,
            )


if __name__ == "__main__":
    unittest.main()
