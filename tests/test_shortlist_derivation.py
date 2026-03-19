import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.shortlist_derivation import derive_shortlist  # noqa: E402


HEADER = [
    "max_stocks",
    "order_investment_ratio",
    "additional_buy_drop_rate",
    "sell_profit_rate",
    "additional_buy_priority",
    "stop_loss_rate",
    "max_splits_limit",
    "max_inactivity_period",
    "period_start",
    "period_end",
    "initial_value",
    "final_value",
    "final_cumulative_returns",
    "cagr",
    "annualized_volatility",
    "mdd",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "universe_mode",
    "is_experimental",
]


def _row(
    *,
    order_ratio: str,
    drop_rate: str,
    sell_rate: str,
    stop_loss: str,
    inactivity: str,
    calmar: str,
    cagr: str,
    mdd: str,
    sharpe: str,
    period_start: str,
    period_end: str,
) -> dict[str, str]:
    return {
        "max_stocks": "20.0000",
        "order_investment_ratio": order_ratio,
        "additional_buy_drop_rate": drop_rate,
        "sell_profit_rate": sell_rate,
        "additional_buy_priority": "0.0000",
        "stop_loss_rate": stop_loss,
        "max_splits_limit": "10.0000",
        "max_inactivity_period": inactivity,
        "period_start": period_start,
        "period_end": period_end,
        "initial_value": "10000000.0000",
        "final_value": "15000000.0000",
        "final_cumulative_returns": "0.5000",
        "cagr": cagr,
        "annualized_volatility": "0.2000",
        "mdd": mdd,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sharpe,
        "calmar_ratio": calmar,
        "universe_mode": "optimistic_survivor",
        "is_experimental": "True",
    }


class TestShortlistDerivation(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> str:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(rows)
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_bundle_manifest(self, path: Path, windows: list[dict], *, optional_fail_budget: int = 0) -> None:
        payload = {
            "bundle_id": "bundle_test_v1",
            "source_mode": "n_window_consensus_mining",
            "decision_date": "2026-03-19",
            "research_data_cutoff": "2024-12-31",
            "promotion_data_cutoff": "2024-12-31",
            "holdout_start": "2025-01-01",
            "holdout_end": "2025-12-31",
            "governance_gates": {
                "max_rank_percentile": 100.0,
                "optional_fail_budget": optional_fail_budget,
                "minimum_criteria": {
                    "calmar_ratio_min": 0.30,
                },
            },
            "selection_contract": {
                "selection_metric": "calmar_ratio",
                "shortlist_size": 3,
                "family_excluded_parameters": ["stop_loss_rate"],
            },
            "windows": windows,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def test_derive_shortlist_writes_provenance_and_actual_row_family_rep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_a = tmp / "window_a.csv"
            csv_b = tmp / "window_b.csv"

            rows_a = [
                _row(
                    order_ratio="0.0280",
                    drop_rate="0.0650",
                    sell_rate="0.1600",
                    stop_loss="-0.9000",
                    inactivity="63.0000",
                    calmar="0.9000",
                    cagr="0.1500",
                    mdd="-0.1700",
                    sharpe="0.8500",
                    period_start="2015-01-02",
                    period_end="2019-12-30",
                ),
                _row(
                    order_ratio="0.0280",
                    drop_rate="0.0650",
                    sell_rate="0.1600",
                    stop_loss="-0.7000",
                    inactivity="63.0000",
                    calmar="0.8500",
                    cagr="0.1450",
                    mdd="-0.1800",
                    sharpe="0.8200",
                    period_start="2015-01-02",
                    period_end="2019-12-30",
                ),
                _row(
                    order_ratio="0.0280",
                    drop_rate="0.0650",
                    sell_rate="0.1600",
                    stop_loss="-0.5000",
                    inactivity="63.0000",
                    calmar="0.8000",
                    cagr="0.1400",
                    mdd="-0.1900",
                    sharpe="0.7900",
                    period_start="2015-01-02",
                    period_end="2019-12-30",
                ),
                _row(
                    order_ratio="0.0300",
                    drop_rate="0.0700",
                    sell_rate="0.2200",
                    stop_loss="-0.7000",
                    inactivity="126.0000",
                    calmar="0.7000",
                    cagr="0.1300",
                    mdd="-0.2200",
                    sharpe="0.7100",
                    period_start="2015-01-02",
                    period_end="2019-12-30",
                ),
            ]
            rows_b = [
                _row(
                    order_ratio="0.0280",
                    drop_rate="0.0650",
                    sell_rate="0.1600",
                    stop_loss="-0.9000",
                    inactivity="63.0000",
                    calmar="0.8800",
                    cagr="0.1520",
                    mdd="-0.1800",
                    sharpe="0.8300",
                    period_start="2018-01-02",
                    period_end="2022-12-29",
                ),
                _row(
                    order_ratio="0.0280",
                    drop_rate="0.0650",
                    sell_rate="0.1600",
                    stop_loss="-0.7000",
                    inactivity="63.0000",
                    calmar="0.8600",
                    cagr="0.1480",
                    mdd="-0.1850",
                    sharpe="0.8150",
                    period_start="2018-01-02",
                    period_end="2022-12-29",
                ),
                _row(
                    order_ratio="0.0280",
                    drop_rate="0.0650",
                    sell_rate="0.1600",
                    stop_loss="-0.5000",
                    inactivity="63.0000",
                    calmar="0.8100",
                    cagr="0.1410",
                    mdd="-0.1950",
                    sharpe="0.7800",
                    period_start="2018-01-02",
                    period_end="2022-12-29",
                ),
                _row(
                    order_ratio="0.0300",
                    drop_rate="0.0700",
                    sell_rate="0.2200",
                    stop_loss="-0.7000",
                    inactivity="126.0000",
                    calmar="0.6900",
                    cagr="0.1310",
                    mdd="-0.2400",
                    sharpe="0.7000",
                    period_start="2018-01-02",
                    period_end="2022-12-29",
                ),
            ]
            hash_a = self._write_csv(csv_a, rows_a)
            hash_b = self._write_csv(csv_b, rows_b)

            manifest_path = tmp / "window_bundle_manifest.json"
            self._write_bundle_manifest(
                manifest_path,
                windows=[
                    {
                        "window_id": "window_a",
                        "csv_path": str(csv_a),
                        "expected_hash": hash_a,
                        "window_role": "mandatory",
                    },
                    {
                        "window_id": "window_b",
                        "csv_path": str(csv_b),
                        "expected_hash": hash_b,
                        "window_role": "mandatory",
                    },
                ],
            )

            out_dir = tmp / "out"
            outputs = derive_shortlist(
                bundle_manifest_path=manifest_path,
                out_dir=out_dir,
            )

            with outputs["shortlist_csv"].open(encoding="utf-8") as handle:
                shortlist_rows = list(csv.DictReader(handle))
            self.assertEqual(len(shortlist_rows), 2)
            self.assertEqual(shortlist_rows[0]["stop_loss_rate"], "-0.7000")
            self.assertIn("candidate_signature", shortlist_rows[0])
            self.assertEqual(shortlist_rows[0]["aggregation_rule_version"], "n_window_rank_robust_v1")

            source_manifest = json.loads(outputs["source_manifest"].read_text(encoding="utf-8"))
            self.assertFalse(source_manifest["approval_evidence_allowed"])
            self.assertEqual(source_manifest["claim_ceiling"], "research_only")
            self.assertEqual(source_manifest["source_window_count"], 2)
            self.assertTrue(source_manifest["bundle_manifest_hash"])

    def test_optional_fail_budget_allows_one_optional_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rows_good = [
                _row(
                    order_ratio="0.0280",
                    drop_rate="0.0650",
                    sell_rate="0.1600",
                    stop_loss="-0.7000",
                    inactivity="63.0000",
                    calmar="0.8200",
                    cagr="0.1450",
                    mdd="-0.1800",
                    sharpe="0.8200",
                    period_start="2015-01-02",
                    period_end="2019-12-30",
                )
            ]
            rows_bad_optional = [
                _row(
                    order_ratio="0.0280",
                    drop_rate="0.0650",
                    sell_rate="0.1600",
                    stop_loss="-0.7000",
                    inactivity="63.0000",
                    calmar="0.1000",
                    cagr="0.0200",
                    mdd="-0.4500",
                    sharpe="0.2000",
                    period_start="2020-01-02",
                    period_end="2024-12-30",
                )
            ]

            windows = []
            for index, rows in enumerate([rows_good, rows_good, rows_bad_optional], start=1):
                path = tmp / f"window_{index}.csv"
                digest = self._write_csv(path, rows)
                windows.append(
                    {
                        "window_id": f"window_{index}",
                        "csv_path": str(path),
                        "expected_hash": digest,
                        "window_role": "mandatory" if index == 1 else "optional",
                    }
                )

            manifest_path = tmp / "window_bundle_manifest.json"
            self._write_bundle_manifest(
                manifest_path,
                windows=windows,
                optional_fail_budget=1,
            )

            out_dir = tmp / "out"
            outputs = derive_shortlist(bundle_manifest_path=manifest_path, out_dir=out_dir)
            with outputs["shortlist_csv"].open(encoding="utf-8") as handle:
                shortlist_rows = list(csv.DictReader(handle))
            self.assertEqual(len(shortlist_rows), 1)

    def test_approval_compatible_rejects_more_than_three_windows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            windows = []
            for index in range(4):
                path = tmp / f"window_{index}.csv"
                digest = self._write_csv(
                    path,
                    [
                        _row(
                            order_ratio="0.0280",
                            drop_rate="0.0650",
                            sell_rate="0.1600",
                            stop_loss="-0.7000",
                            inactivity="63.0000",
                            calmar="0.8200",
                            cagr="0.1450",
                            mdd="-0.1800",
                            sharpe="0.8200",
                            period_start="2015-01-02",
                            period_end="2019-12-30",
                        )
                    ],
                )
                windows.append(
                    {
                        "window_id": f"window_{index}",
                        "csv_path": str(path),
                        "expected_hash": digest,
                        "window_role": "mandatory" if index == 0 else "optional",
                    }
                )

            manifest_path = tmp / "window_bundle_manifest.json"
            self._write_bundle_manifest(manifest_path, windows=windows, optional_fail_budget=1)

            with self.assertRaisesRegex(ValueError, "approval-compatible mode only supports"):
                derive_shortlist(
                    bundle_manifest_path=manifest_path,
                    out_dir=tmp / "out",
                    approval_compatible=True,
                )


if __name__ == "__main__":
    unittest.main()
