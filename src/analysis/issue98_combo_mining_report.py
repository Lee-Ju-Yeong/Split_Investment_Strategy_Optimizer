"""
Issue #98 combo mining report generator.

This module turns a standalone parameter sweep CSV into reproducible
JSON/Markdown artifacts so the combo-research lane can be resumed later
without redoing ad-hoc notebook work.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


METRIC_FIELDS = (
    "calmar_ratio",
    "cagr",
    "mdd",
    "sharpe_ratio",
    "sortino_ratio",
    "annualized_volatility",
)

PARAMETER_FIELDS = (
    "order_investment_ratio",
    "additional_buy_drop_rate",
    "sell_profit_rate",
    "additional_buy_priority",
    "stop_loss_rate",
    "max_splits_limit",
    "max_inactivity_period",
)

DISPLAY_FIELDS = (
    "calmar_ratio",
    "cagr",
    "mdd",
    "sharpe_ratio",
    "stop_loss_rate",
    "max_splits_limit",
    "max_inactivity_period",
    "sell_profit_rate",
    "additional_buy_drop_rate",
    "order_investment_ratio",
    "additional_buy_priority",
)


def _parse_float(value: str) -> float:
    return float(str(value).strip())


def _load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    index = int((len(sorted_values) - 1) * ratio)
    return sorted_values[index]


def _metric_distribution(rows: list[dict], field: str) -> dict:
    values = sorted(_parse_float(row[field]) for row in rows)
    return {
        "count": len(values),
        "min": min(values),
        "p25": _percentile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p75": _percentile(values, 0.75),
        "max": max(values),
    }


def _mean_metric_by_value(rows: list[dict], parameter: str, metric: str) -> list[dict]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row[parameter], []).append(_parse_float(row[metric]))

    summary = []
    for value, values in grouped.items():
        summary.append(
            {
                "value": value,
                "count": len(values),
                "mean_metric": statistics.fmean(values),
                "median_metric": statistics.median(values),
            }
        )
    return sorted(summary, key=lambda item: _parse_float(item["value"]))


def _parameter_importance(main_effects: dict[str, list[dict]]) -> list[dict]:
    ranked = []
    for parameter, rows in main_effects.items():
        means = [item["mean_metric"] for item in rows]
        ranked.append(
            {
                "parameter": parameter,
                "mean_spread": max(means) - min(means),
                "best_value": max(rows, key=lambda item: item["mean_metric"])["value"],
                "worst_value": min(rows, key=lambda item: item["mean_metric"])["value"],
            }
        )
    return sorted(ranked, key=lambda item: item["mean_spread"], reverse=True)


def _top_rows(rows: list[dict], metric: str, limit: int) -> list[dict]:
    sorted_rows = sorted(rows, key=lambda row: _parse_float(row[metric]), reverse=True)
    return [{field: row[field] for field in DISPLAY_FIELDS} for row in sorted_rows[:limit]]


def _top_subset(rows: list[dict], metric: str, top_percent: float) -> list[dict]:
    sorted_rows = sorted(rows, key=lambda row: _parse_float(row[metric]), reverse=True)
    count = max(1, int(len(sorted_rows) * (top_percent / 100.0)))
    return sorted_rows[:count]


def _top_subset_frequencies(rows: list[dict], top_percent: float, metric: str) -> dict:
    top_rows = _top_subset(rows, metric, top_percent)
    output = {
        "top_percent": top_percent,
        "row_count": len(top_rows),
        "parameter_value_frequency": {},
    }
    for parameter in PARAMETER_FIELDS:
        counts: dict[str, int] = {}
        for row in top_rows:
            counts[row[parameter]] = counts.get(row[parameter], 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], _parse_float(item[0])))
        output["parameter_value_frequency"][parameter] = [
            {
                "value": value,
                "count": count,
                "share": count / len(top_rows),
            }
            for value, count in ranked
        ]
    return output


def build_report(rows: list[dict], *, top_percent: float, metric: str, shortlist_size: int) -> dict:
    main_effects = {
        parameter: _mean_metric_by_value(rows, parameter, metric)
        for parameter in PARAMETER_FIELDS
    }
    return {
        "row_count": len(rows),
        "metric_distributions": {
            field: _metric_distribution(rows, field)
            for field in METRIC_FIELDS
        },
        "main_effects": main_effects,
        "parameter_importance": _parameter_importance(main_effects),
        "top_rows": _top_rows(rows, metric, shortlist_size),
        "top_subset_summary": _top_subset_frequencies(rows, top_percent, metric),
    }


def _format_number(value: float) -> str:
    return f"{value:.4f}"


def _render_markdown(*, csv_path: Path, report: dict, metric: str) -> str:
    lines = [
        "# Issue #98 Combo Mining Report",
        "",
        f"- Source CSV: `{csv_path}`",
        f"- Row count: `{report['row_count']}`",
        f"- Ranking metric: `{metric}`",
        "",
        "## Metric Distribution",
        "",
    ]

    for field, stats in report["metric_distributions"].items():
        lines.append(
            "- "
            f"`{field}`: min={_format_number(stats['min'])}, "
            f"p25={_format_number(stats['p25'])}, "
            f"median={_format_number(stats['median'])}, "
            f"p75={_format_number(stats['p75'])}, "
            f"max={_format_number(stats['max'])}"
        )

    lines.extend(
        [
            "",
            "## Parameter Importance",
            "",
        ]
    )
    for item in report["parameter_importance"]:
        lines.append(
            "- "
            f"`{item['parameter']}` spread={_format_number(item['mean_spread'])} "
            f"(best `{item['best_value']}`, worst `{item['worst_value']}`)"
        )

    lines.extend(
        [
            "",
            "## Main Effects",
            "",
        ]
    )
    for parameter, effects in report["main_effects"].items():
        lines.append(f"### `{parameter}`")
        lines.append("")
        for item in effects:
            lines.append(
                "- "
                f"`{item['value']}`: mean={_format_number(item['mean_metric'])}, "
                f"median={_format_number(item['median_metric'])}, "
                f"n={item['count']}"
            )
        lines.append("")

    top_subset = report["top_subset_summary"]
    lines.extend(
        [
            f"## Top {top_subset['top_percent']:.1f}% Frequency",
            "",
        ]
    )
    for parameter, items in top_subset["parameter_value_frequency"].items():
        if not items:
            continue
        top_item = items[0]
        lines.append(
            "- "
            f"`{parameter}` top value `{top_item['value']}` "
            f"(count={top_item['count']}, share={top_item['share']:.2%})"
        )

    lines.extend(
        [
            "",
            "## Shortlist",
            "",
        ]
    )
    for index, row in enumerate(report["top_rows"], start=1):
        rendered = ", ".join(f"{key}={row[key]}" for key in DISPLAY_FIELDS)
        lines.append(f"{index}. {rendered}")

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- This report is exploratory. The rows are a parameter sweep over one historical window, not independent samples.",
            "- Use this report for shortlist mining, sensitivity review, and parity/WFO candidate selection.",
            "- Do not treat these statistics alone as final investment proof or canonical Issue #98 perf evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report_files(*, csv_path: Path, report_dir: Path, report: dict, metric: str) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "combo_mining_report.json"
    md_path = report_dir / "combo_mining_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(csv_path=csv_path, report=report, metric=metric) + "\n", encoding="utf-8")
    return json_path, md_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Issue #98 combo mining report artifacts.")
    parser.add_argument("--csv-path", required=True, help="Path to standalone_simulation_results_*.csv")
    parser.add_argument("--report-dir", required=True, help="Directory where JSON/Markdown artifacts are written")
    parser.add_argument("--metric", default="calmar_ratio", choices=METRIC_FIELDS, help="Ranking metric")
    parser.add_argument("--top-percent", type=float, default=5.0, help="Top percentile slice for frequency analysis")
    parser.add_argument("--shortlist-size", type=int, default=10, help="Number of top rows kept in shortlist")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    csv_path = Path(args.csv_path).resolve()
    report_dir = Path(args.report_dir).resolve()
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError(f"empty csv: {csv_path}")
    report = build_report(
        rows,
        top_percent=float(args.top_percent),
        metric=args.metric,
        shortlist_size=int(args.shortlist_size),
    )
    report["source_csv"] = str(csv_path)
    report["ranking_metric"] = args.metric
    report["top_percent"] = float(args.top_percent)
    json_path, md_path = write_report_files(
        csv_path=csv_path,
        report_dir=report_dir,
        report=report,
        metric=args.metric,
    )
    print(f"[issue98_combo_mining_report] saved json={json_path}")
    print(f"[issue98_combo_mining_report] saved md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
