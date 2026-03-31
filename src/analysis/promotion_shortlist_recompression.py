"""
Recompress promotion compare results into a smaller frozen shortlist.

This module is intentionally narrow. It consumes:

- `promotion_candidate_summary.csv` from a promotion compare run
- the source frozen shortlist csv that fed that compare run

and produces a smaller shortlist artifact for the next promotion run.

Policy v1:
- start from hard-gate-pass candidates only
- group candidates by family keys
- when rows differ only by stop_loss and promotion results are identical,
  keep the smallest stop_loss representative
- when promotion results differ, keep one representative per distinct result
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PARAMETER_KEYS = (
    "max_stocks",
    "order_investment_ratio",
    "additional_buy_drop_rate",
    "sell_profit_rate",
    "additional_buy_priority",
    "stop_loss_rate",
    "max_splits_limit",
    "max_inactivity_period",
)
DEFAULT_FAMILY_KEYS = (
    "order_investment_ratio",
    "additional_buy_drop_rate",
    "sell_profit_rate",
    "max_splits_limit",
    "max_inactivity_period",
)
DEFAULT_RESULT_EQUALITY_KEYS = (
    "promotion_fold_pass_rate",
    "promotion_oos_cagr_median",
    "promotion_oos_calmar_median",
    "promotion_oos_mdd_depth_worst",
    "promotion_oos_is_calmar_ratio_median",
    "hard_gate_pass",
    "hard_gate_fail_reasons",
)


@dataclass(frozen=True)
class Representative:
    row: dict[str, str]
    rank_score: float
    family_signature: str
    result_signature: str


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _hash_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_priority(value: Any) -> str:
    raw = str(value).strip()
    if raw in {"lowest_order", "highest_drop"}:
        return raw
    try:
        return "highest_drop" if int(float(raw)) == 1 else "lowest_order"
    except (TypeError, ValueError):
        return "lowest_order"


def _normalize_numeric(value: Any) -> str:
    return f"{float(value):.10f}"


def _normalize_param_value(key: str, value: Any) -> str:
    if key == "additional_buy_priority":
        return _normalize_priority(value)
    if key in {
        "max_stocks",
        "max_splits_limit",
        "max_inactivity_period",
    }:
        return str(int(float(value)))
    return _normalize_numeric(value)


def _candidate_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(_normalize_param_value(key, row.get(key)) for key in PARAMETER_KEYS)


def _family_signature(row: dict[str, str], family_keys: tuple[str, ...]) -> str:
    return "|".join(
        f"{key}={_normalize_param_value(key, row.get(key))}" for key in family_keys
    )


def _result_signature(
    row: dict[str, str], result_keys: tuple[str, ...], *, precision: int = 10
) -> str:
    parts: list[str] = []
    for key in result_keys:
        value = row.get(key, "")
        try:
            rendered = f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            rendered = str(value).strip()
        parts.append(f"{key}={rendered}")
    return "|".join(parts)


def _smallest_stop_loss_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda row: float(row["stop_loss_rate"]))


def select_family_representatives(
    summary_rows: list[dict[str, str]],
    *,
    family_keys: tuple[str, ...] = DEFAULT_FAMILY_KEYS,
    result_keys: tuple[str, ...] = DEFAULT_RESULT_EQUALITY_KEYS,
) -> list[Representative]:
    hard_gate_rows = [row for row in summary_rows if row.get("hard_gate_pass") == "True"]
    family_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in hard_gate_rows:
        family_groups[_family_signature(row, family_keys)].append(row)

    representatives: list[Representative] = []
    for family_sig, family_rows in family_groups.items():
        result_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in family_rows:
            result_groups[_result_signature(row, result_keys)].append(row)
        for result_sig, result_rows in result_groups.items():
            representative_row = _smallest_stop_loss_row(result_rows)
            rank_score = max(float(row["robust_score"]) for row in result_rows)
            representatives.append(
                Representative(
                    row=representative_row,
                    rank_score=rank_score,
                    family_signature=family_sig,
                    result_signature=result_sig,
                )
            )

    return sorted(representatives, key=lambda item: item.rank_score, reverse=True)


def build_recompressed_shortlist_rows(
    source_shortlist_rows: list[dict[str, str]],
    representatives: list[Representative],
) -> list[dict[str, str]]:
    source_by_candidate = {_candidate_key(row): dict(row) for row in source_shortlist_rows}
    shortlist_rows: list[dict[str, str]] = []

    for rank, item in enumerate(representatives, start=1):
        candidate_key = _candidate_key(item.row)
        if candidate_key not in source_by_candidate:
            raise KeyError(
                "source shortlist is missing representative candidate: "
                f"{item.row.get('shortlist_candidate_id')}"
            )
        shortlist_row = dict(source_by_candidate[candidate_key])
        shortlist_row["shortlist_rank"] = str(rank)
        shortlist_row["promotion_recompression_rank"] = str(rank)
        shortlist_row["promotion_recompression_family_signature"] = item.family_signature
        shortlist_row["promotion_recompression_result_signature"] = item.result_signature
        shortlist_row["promotion_recompression_rank_score"] = str(item.rank_score)
        shortlist_row["promotion_recompression_source_candidate_id"] = item.row[
            "shortlist_candidate_id"
        ]
        shortlist_row["promotion_recompression_source_robust_score"] = item.row[
            "robust_score"
        ]
        shortlist_row["promotion_recompression_source_fold_pass_rate"] = item.row[
            "promotion_fold_pass_rate"
        ]
        shortlist_row["promotion_recompression_source_cagr_median"] = item.row[
            "promotion_oos_cagr_median"
        ]
        shortlist_row["promotion_recompression_source_calmar_median"] = item.row[
            "promotion_oos_calmar_median"
        ]
        shortlist_row["promotion_recompression_source_mdd_worst"] = item.row[
            "promotion_oos_mdd_depth_worst"
        ]
        shortlist_rows.append(shortlist_row)
    return shortlist_rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    *,
    source_summary_path: Path,
    summary_rows: list[dict[str, str]],
    representatives: list[Representative],
) -> None:
    hard_gate_count = sum(row.get("hard_gate_pass") == "True" for row in summary_rows)
    lines = [
        "# Promotion Recompression Summary",
        "",
        f"- Source summary: `{source_summary_path}`",
        f"- Source candidate count: `{len(summary_rows)}`",
        f"- Hard-gate pass candidate count: `{hard_gate_count}`",
        f"- Recompressed canonical shortlist count: `{len(representatives)}`",
        (
            "- Recompression rule: hard-gate pass only, group by family keys, "
            "and when only stop_loss differs while promotion results are equal, "
            "keep the smallest stop_loss representative."
        ),
        "",
        "## Selected Representatives",
    ]
    for item in representatives:
        row = item.row
        lines.append(
            f"- id `{row['shortlist_candidate_id']}`: "
            f"stop `{row['stop_loss_rate']}`, "
            f"pass_rate `{row['promotion_fold_pass_rate']}`, "
            f"CAGR_med `{float(row['promotion_oos_cagr_median']):.4f}`, "
            f"Calmar_med `{float(row['promotion_oos_calmar_median']):.4f}`, "
            f"MDD_worst `{float(row['promotion_oos_mdd_depth_worst']):.4f}`, "
            f"robust_score `{float(row['robust_score']):.4f}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(
    path: Path,
    *,
    source_summary_path: Path,
    source_shortlist_path: Path,
    output_shortlist_path: Path,
    representatives: list[Representative],
    family_keys: tuple[str, ...],
    result_keys: tuple[str, ...],
) -> None:
    manifest = {
        "artifact_type": "promotion_recompressed_shortlist",
        "created_at": _utc_now_iso(),
        "source_summary_path": str(source_summary_path),
        "source_summary_hash": _hash_file_sha256(source_summary_path),
        "source_shortlist_path": str(source_shortlist_path),
        "source_shortlist_hash": _hash_file_sha256(source_shortlist_path),
        "selection_contract": {
            "source_stage": "promotion_compare",
            "include_only_hard_gate_pass": True,
            "family_grouping_parameters": list(family_keys),
            "result_equality_keys": list(result_keys),
            "representative_rule": (
                "smallest_stop_loss_when_results_equal_else_keep_each_result_group"
            ),
            "sort_order": "result_group_max_robust_score_desc",
        },
        "selected_count": len(representatives),
        "selected_candidate_ids": [
            int(item.row["shortlist_candidate_id"]) for item in representatives
        ],
        "shortlist_csv_path": str(output_shortlist_path),
        "shortlist_hash": _hash_file_sha256(output_shortlist_path),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def recompress_promotion_shortlist(
    *,
    summary_csv: Path,
    source_shortlist_csv: Path,
    out_dir: Path,
    family_keys: tuple[str, ...] = DEFAULT_FAMILY_KEYS,
    result_keys: tuple[str, ...] = DEFAULT_RESULT_EQUALITY_KEYS,
) -> dict[str, Any]:
    summary_rows = _load_csv_rows(summary_csv)
    source_shortlist_rows = _load_csv_rows(source_shortlist_csv)
    representatives = select_family_representatives(
        summary_rows, family_keys=family_keys, result_keys=result_keys
    )
    shortlist_rows = build_recompressed_shortlist_rows(
        source_shortlist_rows, representatives
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    shortlist_path = out_dir / "shortlist_candidates.csv"
    selected_summary_path = out_dir / "promotion_recompression_selected_summary.csv"
    summary_path = out_dir / "promotion_recompression_summary.md"
    manifest_path = out_dir / "promotion_recompression_manifest.json"

    _write_csv(shortlist_path, shortlist_rows)
    _write_csv(selected_summary_path, [item.row for item in representatives])
    _write_summary(
        summary_path,
        source_summary_path=summary_csv,
        summary_rows=summary_rows,
        representatives=representatives,
    )
    _write_manifest(
        manifest_path,
        source_summary_path=summary_csv,
        source_shortlist_path=source_shortlist_csv,
        output_shortlist_path=shortlist_path,
        representatives=representatives,
        family_keys=family_keys,
        result_keys=result_keys,
    )

    return {
        "selected_count": len(representatives),
        "selected_candidate_ids": [
            int(item.row["shortlist_candidate_id"]) for item in representatives
        ],
        "shortlist_path": str(shortlist_path),
        "shortlist_hash": _hash_file_sha256(shortlist_path),
    }


def _parse_key_list(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None or str(raw).strip() == "":
        return default
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompress promotion compare results into a smaller shortlist."
    )
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--source-shortlist-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--family-keys", default=",".join(DEFAULT_FAMILY_KEYS))
    parser.add_argument(
        "--result-equality-keys",
        default=",".join(DEFAULT_RESULT_EQUALITY_KEYS),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = recompress_promotion_shortlist(
        summary_csv=Path(args.summary_csv),
        source_shortlist_csv=Path(args.source_shortlist_csv),
        out_dir=Path(args.out_dir),
        family_keys=_parse_key_list(args.family_keys, DEFAULT_FAMILY_KEYS),
        result_keys=_parse_key_list(
            args.result_equality_keys, DEFAULT_RESULT_EQUALITY_KEYS
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
