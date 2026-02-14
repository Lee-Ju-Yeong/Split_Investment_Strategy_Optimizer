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

import pandas as pd

from .config_loader import load_config
from .data_handler import DataHandler


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
) -> list[dict]:
    rows: list[dict] = []
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

        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "pit_source": pit_source,
                "universe_count": pit_size,
                "tier1_count": int(tier1_count),
                "tier12_count": int(tier12_count),
                "tier12_ratio": round(ratio, 6),
                "min_liquidity_20d_avg_value": int(min_liquidity_20d_avg_value or 0),
            }
        )
    return rows


def write_csv_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Tier coverage within PIT universe.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD.")
    parser.add_argument("--step-days", type=int, default=7, help="Sampling interval in days.")
    parser.add_argument("--min-liquidity-20d-avg-value", type=int, default=0)
    parser.add_argument("--out", default=None, help="Output CSV path. If omitted, prints table.")
    args = parser.parse_args()

    start_date = _parse_yyyymmdd(args.start_date)
    end_date = _parse_yyyymmdd(args.end_date)
    if start_date > end_date:
        raise ValueError(f"Invalid range: start_date({start_date}) > end_date({end_date})")

    config = load_config()
    handler = DataHandler(db_config=config["database"])

    rows = build_tier_coverage_rows(
        handler=handler,
        start_date=start_date,
        end_date=end_date,
        step_days=int(args.step_days),
        min_liquidity_20d_avg_value=int(args.min_liquidity_20d_avg_value or 0),
    )

    if args.out:
        write_csv_rows(args.out, rows)
        print(f"[tier_coverage_report] saved rows={len(rows)} path={args.out}")
        return

    df = pd.DataFrame(rows)
    if df.empty:
        print("[tier_coverage_report] empty")
        return
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

