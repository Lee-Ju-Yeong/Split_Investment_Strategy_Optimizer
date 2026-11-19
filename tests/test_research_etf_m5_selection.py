"""정식 M5 선택 규칙·중첩 순차검증·PBO의 손계산 검사. 엔진을 실행하지 않고 결과표만 만든다."""
from decimal import Decimal as D
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest

from src.research.etf.m5_parameters import grid_candidates, load_config
from src.research.etf.m5_selection import (
    analyse, basket_value, load_table, neighbours, overfit_probability, select, setting_scores, walk_forward)


GRID = {"kg": ["1", "2", "3"]}
RULE = {"outer_train_years": 3, "embargo_starts": 1, "pbo_blocks": 8}
STARTS = [f"{year}-{month}-02" for year in range(2019, 2024) for month in ("01", "07")]


def plan(baskets, starts=STARTS, paths=("A", "B")):
    return {"schema": "etf-m5-plan-v1", "scope": "smoke_not_formal", "synthetic": True,
            "candidates": [{"kg": value} for value in GRID["kg"]], "paths": list(paths),
            "baskets": [{"id": name, "evaluations": list(starts)} for name in baskets]}


def write_units(root, value, deltas):
    """deltas(basket, start, index, path) → (ΔCAGR, ΔMDD) 또는 통과하지 못한 비교의 상태 문자열."""
    for basket in value["baskets"]:
        directory = Path(root) / "units" / basket["id"]
        directory.mkdir(parents=True)
        for start in basket["evaluations"]:
            trials = []
            for index in range(len(value["candidates"])):
                for path in value["paths"]:
                    delta = deltas(basket["id"], start, index, path)
                    comparison = ({"status": delta} if isinstance(delta, str) else
                                  {"status": "passed", "matched": {"delta_cagr": str(delta[0]), "delta_mdd": str(delta[1])}})
                    trials.append({"candidate": index, "path": path, "comparison": comparison})
            (directory / f"{start}.json").write_text(json.dumps({"trials": trials}), encoding="utf-8")


def score(share, values=()):
    return {"share": F(share), "improved": 0, "baskets": 1, "valued": len(values), "values": list(values)}


class SelectionRuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def table(self, value, deltas):
        write_units(self.root, value, deltas)
        return load_table(value, self.root)

    def test_basket_value_is_the_start_median_on_the_worse_path_per_metric(self):
        starts = STARTS[:3]
        rows = {"A": [(1, 1), (3, -1), (2, 2)], "B": [("0.5", 3)] * 3}
        value = plan(["b1"], starts)
        table = self.table(value, lambda b, s, i, p: rows[p][starts.index(s)] if i == 0 else "rejected")
        self.assertEqual(basket_value(table, "b1", starts, 0, ["A", "B"]), (D("0.5"), D(1)))
        self.assertEqual(basket_value(table, "b1", starts, 0, ["A"]), (D(2), D(1)))
        self.assertIsNone(basket_value(table, "b1", starts, 1, ["A", "B"]))

    def test_rejected_or_failed_comparisons_count_as_not_improved_in_the_denominator(self):
        value = plan(["b1", "b2", "b3"], STARTS[:2])
        value["baskets"][2]["evaluations"] = STARTS[4:6]

        def deltas(basket, start, index, path):
            if basket == "b2" and start == STARTS[1] and index == 0:
                return "rejected"
            return (1, 1) if index == 0 else (1, -1) if index == 1 else "failed"
        scores = setting_scores(self.table(value, deltas), value, set(STARTS[:2]))
        self.assertEqual([(row["share"], row["improved"], row["baskets"], row["valued"]) for row in scores],
                         [(F(1, 2), 1, 2, 1), (F(0), 0, 2, 2), (F(0), 0, 2, 0)])
        with self.assertRaisesRegex(ValueError, "no planned evaluations"):
            setting_scores(load_table(value, self.root), value, {"2030-01-02"})

    def test_neighbours_differ_by_one_grid_step_in_one_parameter(self):
        self.assertEqual(neighbours([{"kg": value} for value in GRID["kg"]], GRID), [[1], [0, 2], [1]])
        config = load_config()
        candidates = grid_candidates(config)
        near = neighbours(candidates, config["grid"])
        self.assertEqual((len(near), min(map(len, near)), max(map(len, near))), (144, 6, 8))
        for index, others in enumerate(near):
            for other in others:
                self.assertEqual(sum(candidates[index][key] != candidates[other][key] for key in config["grid"]), 1)
                self.assertIn(index, near[other])

    def test_selection_prefers_a_flat_region_over_an_isolated_peak(self):
        line = [[1], [0, 2], [1, 3], [2, 4], [3]]
        scores = [score(1), score(0), score("7/10"), score("7/10"), score("7/10")]
        best, smoothed = select(scores, line)
        self.assertEqual(smoothed, [F(1, 2), F(17, 40), F(21, 40), F(7, 10), F(7, 10)])
        self.assertEqual(best, 3)
        scores[4] = score("7/10", [(D(1), D("0.2")), (D(1), D("0.4"))])
        scores[3] = score("7/10", [(D(1), D("0.1")), (D(1), D("0.9"))])
        self.assertEqual(select(scores, line)[0], 4)
        # 가장자리 설정이 좋은 이웃 하나 때문에 뽑히지 않는다.
        self.assertEqual(select([score(0), score(1, [(D(1), D(1))]), score(0)], [[1], [0, 2], [1]])[0], 1)

    def test_walk_forward_selects_on_three_prior_years_with_an_embargoed_start(self):
        value = plan(["b1", "b2"])
        good = {2022: 1, 2023: 2}

        def deltas(basket, start, index, path):
            year = int(start[:4])
            if year < 2022:
                return (1, 1) if index == (0 if year < 2020 else 2) else (-1, -1)
            return (1, 1) if index == good[year] else (-1, -1)
        folds, skipped = walk_forward(self.table(value, deltas), value, neighbours(value["candidates"], GRID), RULE)
        self.assertEqual([row["test_year"] for row in skipped], [2019, 2020, 2021])
        first, second = folds
        self.assertEqual((first["test_year"], first["train_starts"], first["test_starts"]),
                         (2022, STARTS[:5], STARTS[6:8]))
        self.assertEqual((second["train_starts"], second["test_starts"]), (STARTS[2:7], STARTS[8:10]))
        # 2019~2021-01: 설정0이 앞 2개, 설정2가 뒤 3개 시작일에서 양수라 중앙값은 설정2만 양수다.
        self.assertEqual((first["selected"], first["train"]["joint_improvement_share"], first["train_smoothed_share"]),
                         (2, F(1), F(1, 2)))
        self.assertEqual((first["test"]["joint_improvement_share"], first["test_rank"]), (F(0), 2.5))
        self.assertEqual(first["test"]["median_delta_mdd"], D(-1))
        self.assertEqual((first["inner"]["train_starts"], first["inner"]["test_starts"], first["inner"]["selected"]),
                         (STARTS[:3], STARTS[4:5], 0))
        self.assertFalse(first["inner"]["same_as_outer"])
        self.assertEqual(first["inner"]["test"]["joint_improvement_share"], F(0))

    def test_pbo_is_zero_for_a_stable_winner_and_one_for_a_reversing_winner(self):
        starts = STARTS[:8]
        stable = plan(["b1"], starts, paths=("A",))
        near = neighbours(stable["candidates"], GRID)
        table = self.table(stable, lambda b, s, i, p: (1, 1) if i == 0 else (-1, -1))
        result = overfit_probability(table, stable, near, 8)
        self.assertEqual((result["status"], result["splits"], result["pbo"]), ("computed", 70, F(0)))
        self.assertEqual(overfit_probability(table, stable, near, 12)["status"], "not_computed")

        def reversing(basket, start, index, path):
            early = starts.index(start) < 4
            return (1, 1) if (index == 0 and early) or (index == 2 and not early) else (-1, -1)
        other = self.root / "reversing"
        write_units(other, stable, reversing)
        self.assertEqual(overfit_probability(load_table(stable, other), stable, near, 8)["pbo"], F(1))

    def test_analysis_refuses_an_incomplete_run_and_never_claims_a_verdict(self):
        value = plan(["b1"], paths=("A",))
        write_units(self.root, value, lambda b, s, i, p: (1, 1) if i == 1 else "rejected")
        report = analyse(value, self.root, grid=GRID)
        self.assertEqual((report["comparisons_planned"], report["comparisons_passed"]), (30, 10))
        self.assertEqual([fold["selected"] for fold in report["folds"]], [1, 1])
        self.assertEqual((report["overfit"]["status"], report["overfit"]["pbo"]), ("computed", F(0)))
        self.assertFalse(report["statistical_success_evaluated"] or report["holdout_evaluated"])
        (self.root / "units" / "b1" / f"{STARTS[3]}.json").unlink()
        with self.assertRaisesRegex(ValueError, "run is incomplete"):
            analyse(value, self.root, grid=GRID)


if __name__ == "__main__":
    unittest.main()
