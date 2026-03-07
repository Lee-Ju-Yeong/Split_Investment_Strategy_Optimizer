"""
tier_coverage_report.py

Issue #67 follow-up:
Report Tier coverage (tier=1 / tier<=2) within PIT universe over time.

Usage (recommended):
  PYTHONPATH=$PWD conda run --no-capture-output -n rapids-env \
    python -m src.tier_coverage_report --start-date 20131120 --end-date 20141231 --step-days 7 --out /tmp/tier_coverage.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys

import pandas as pd

# BOOTSTRAP: allow direct execution (`python src/tier_coverage_report.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from .config_loader import load_config
from .data_handler import DataHandler

DEFAULT_MIN_TIER12_COVERAGE_RATIO = 0.45


def _parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _iter_dates(start: date, end: date, step_days: int):
    step = max(int(step_days), 1)
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=step)
    if cursor - timedelta(days=step) != end:
        yield end


def _filter_min_liquidity(tier_map: dict, min_liquidity_20d_avg_value: int | None) -> dict:
    if min_liquidity_20d_avg_value is None or int(min_liquidity_20d_avg_value) <= 0:
        return tier_map

    threshold = int(min_liquidity_20d_avg_value)
    filtered = {}
    for code, info in tier_map.items():
        liq = info.get("liquidity_20d_avg_value")
        if liq is None or pd.isna(liq):
            continue
        if int(liq) >= threshold:
            filtered[code] = info
    return filtered


def build_tier_coverage_rows(
    handler: DataHandler,
    start_date: date,
    end_date: date,
    step_days: int,
    min_liquidity_20d_avg_value: int | None,
    min_tier12_coverage_ratio: float | None = None,
) -> list[dict]:
    rows: list[dict] = []
    threshold = float(
        DEFAULT_MIN_TIER12_COVERAGE_RATIO
        if min_tier12_coverage_ratio is None
        else min_tier12_coverage_ratio
    )
    for d in _iter_dates(start_date, end_date, step_days=step_days):
        pit_codes, pit_source = handler.get_pit_universe_codes_as_of(d)
        pit_size = len(pit_codes)

        tier12_map = handler.get_tiers_as_of(
            as_of_date=d,
            tickers=pit_codes,
            allowed_tiers=[1, 2],
        )
        tier12_map = _filter_min_liquidity(tier12_map, min_liquidity_20d_avg_value)

        tier12_count = len(tier12_map)
        tier1_count = sum(1 for info in tier12_map.values() if int(info.get("tier") or 0) == 1)
        ratio = float(tier12_count) / float(pit_size) if pit_size > 0 else 0.0
        gate_pass = pit_size <= 0 or threshold <= 0.0 or ratio >= threshold

        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "pit_source": pit_source,
                "universe_count": pit_size,
                "tier1_count": int(tier1_count),
                "tier12_count": int(tier12_count),
                "tier12_ratio": round(ratio, 6),
                "required_tier12_ratio": round(threshold, 6),
                "coverage_gate_pass": bool(gate_pass),
                "min_liquidity_20d_avg_value": int(min_liquidity_20d_avg_value or 0),
            }
        )
    return rows


def summarize_tier_coverage_rows(
    rows: list[dict],
    *,
    min_tier12_coverage_ratio: float | None = None,
) -> dict:
    threshold = float(
        DEFAULT_MIN_TIER12_COVERAGE_RATIO
        if min_tier12_coverage_ratio is None
        else min_tier12_coverage_ratio
    )
    if not rows:
        return {
            "sampled_days": 0,
            "min_tier12_coverage_ratio": threshold,
            "coverage_gate_pass": True,
            "failed_days": 0,
            "first_failed_date": None,
            "min_observed_tier12_ratio": None,
            "avg_observed_tier12_ratio": None,
        }

    ratios = [float(row.get("tier12_ratio", 0.0) or 0.0) for row in rows]
    failed_rows = [
        row for row in rows
        if not bool(row.get("coverage_gate_pass", True))
    ]
    return {
        "sampled_days": int(len(rows)),
        "min_tier12_coverage_ratio": threshold,
        "coverage_gate_pass": len(failed_rows) == 0,
        "failed_days": int(len(failed_rows)),
        "first_failed_date": failed_rows[0]["date"] if failed_rows else None,
        "min_observed_tier12_ratio": round(min(ratios), 6),
        "avg_observed_tier12_ratio": round(sum(ratios) / len(ratios), 6),
    }


def write_csv_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Tier coverage within PIT universe.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD.")
    parser.add_argument("--step-days", type=int, default=7, help="Sampling interval in days.")
    parser.add_argument("--min-liquidity-20d-avg-value", type=int, default=0)
    parser.add_argument(
        "--min-tier12-coverage-ratio",
        type=float,
        default=None,
        help="Override coverage threshold. If omitted, use config strategy_params.min_tier12_coverage_ratio.",
    )
    parser.add_argument("--out", default=None, help="Output CSV path. If omitted, prints table.")
    parser.add_argument("--summary-out", default=None, help="Optional JSON summary path.")
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        default=False,
        help="Raise when sampled coverage violates --min-tier12-coverage-ratio.",
    )
    args = parser.parse_args()

    start_date = _parse_yyyymmdd(args.start_date)
    end_date = _parse_yyyymmdd(args.end_date)
    if start_date > end_date:
        raise ValueError(f"Invalid range: start_date({start_date}) > end_date({end_date})")
    if args.fail_on_gate and int(args.step_days) != 1:
        raise ValueError("--fail-on-gate requires --step-days 1 for release-grade daily coverage checks.")

    config = load_config()
    handler = DataHandler(db_config=config["database"])
    strategy_cfg = dict(config.get("strategy_params", {}))
    configured_min_ratio = float(
        strategy_cfg.get("min_tier12_coverage_ratio", DEFAULT_MIN_TIER12_COVERAGE_RATIO)
        or DEFAULT_MIN_TIER12_COVERAGE_RATIO
    )
    min_ratio = configured_min_ratio if args.min_tier12_coverage_ratio is None else float(args.min_tier12_coverage_ratio)

    rows = build_tier_coverage_rows(
        handler=handler,
        start_date=start_date,
        end_date=end_date,
        step_days=int(args.step_days),
        min_liquidity_20d_avg_value=int(args.min_liquidity_20d_avg_value or 0),
        min_tier12_coverage_ratio=min_ratio,
    )
    summary = summarize_tier_coverage_rows(
        rows,
        min_tier12_coverage_ratio=min_ratio,
    )

    if args.out:
        write_csv_rows(args.out, rows)
        print(f"[tier_coverage_report] saved rows={len(rows)} path={args.out}")
    if args.summary_out:
        write_json(args.summary_out, summary)
        print(f"[tier_coverage_report] saved summary path={args.summary_out}")
    if args.fail_on_gate and not summary["coverage_gate_pass"]:
        raise ValueError(
            "Tier coverage gate failed: "
            f"first_failed_date={summary['first_failed_date']} "
            f"threshold={summary['min_tier12_coverage_ratio']:.4f}"
        )
    if args.out:
        return

    df = pd.DataFrame(rows)
    if df.empty:
        print("[tier_coverage_report] empty")
        return
    print(df.to_string(index=False))
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
