"""외부 계정·파일·DB 없이 학습 고정 기준선과 프런티어를 계산한다."""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path

from src.research.etf.baseline_policy import fit_baseline
from src.research.etf.baseline_report import comparison_report, run_baseline_suite
from src.research.etf.engine import run_engine
from src.research.etf.engine_types import Bar, Instrument, LevelRule


D = Decimal


def _at(day, hour=0):
    return datetime.combine(day, time(hour), timezone.utc)


def _bar(day, opening, closing):
    start, end = D(opening), D(closing)
    return Bar("SYNTHETIC_KR", day, _at(day), _at(day, 6), start, max(start, end),
               min(start, end), end, D("1000000"), adv20=D("1000000000"))


def demo(path="O-H-L-C"):
    instruments = [Instrument("SYNTHETIC_KR", "KR", "KRW", "국내주식형")]
    rules = {"SYNTHETIC_KR": (LevelRule(1, D("3000"), D("0"), D(".15")),
                               LevelRule(2, D("3000"), D(".1"), D(".15")))}
    options = dict(initial_cash={"KRW": D("10000")}, commission_rate=D(".001"), path=path)
    training_bars = [_bar(date(2024, 5, 2), "100", "100"), _bar(date(2024, 5, 3), "90", "90")]
    training = run_engine(instruments, training_bars, rules,
                          evaluation_times=[_at(row.session_date, 9) for row in training_bars], **options)
    frozen = fit_baseline(training, instruments, training_start=_at(date(2024, 5, 1)),
                          training_end=_at(date(2024, 5, 31)), evaluation_start=_at(date(2024, 6, 3)))
    bars = [_bar(date(2024, 6, 3), "100", "110"), _bar(date(2024, 6, 4), "110", "100"),
            _bar(date(2024, 7, 1), "80", "90"), _bar(date(2024, 8, 1), "120", "100")]
    evaluations = [_at(row.session_date, 9) for row in bars]
    strategy = run_engine(instruments, bars, rules, evaluation_times=evaluations, **options)
    suite = run_baseline_suite(frozen, instruments, bars, strategy_result=strategy,
                               diagnostic_end=evaluations[-1], evaluation_times=evaluations, **options)
    report = comparison_report(strategy, suite, instruments, initial_equity_krw=D("10000"),
                               started_at=frozen.evaluation_start, ended_at=evaluations[-1])
    assert len(report["baselines"]) == 8
    assert report["baselines"]["matched"]["role"] == "primary"
    assert report["baselines"]["posthoc_diagnostic"]["role"] == "diagnostic"
    report["synthetic_only"] = True
    report["path"] = path
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="새 JSON 경로. 기존 파일은 덮어쓰지 않음")
    parser.add_argument("--path", choices=("O-H-L-C", "O-L-H-C"), default="O-H-L-C")
    args = parser.parse_args()
    report = demo(args.path)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
