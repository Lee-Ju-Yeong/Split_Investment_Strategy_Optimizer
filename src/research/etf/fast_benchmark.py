"""고정 합성 입력에서 M2/M3의 정확한 일치와 실행 시간을 함께 기록한다."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
import time

from .engine import run_engine
from .engine_strategy import candidate_for_bar
from .engine_types import Bar, FxRate, Instrument, LevelRule, Position
from .fast_build import fingerprint
from .fast_kernel import FastKernel, run_fast_engine

OPEN = datetime(2024, 6, 3, tzinfo=timezone.utc)


def candidate_cases():
    cases = []
    for market, price, width in (("KR", "2000", "100"), ("US", "100", "10"), ("GOLD", "100000", "2000")):
        instrument = Instrument(market, market, "USD" if market == "US" else "KRW", "국내주식형")
        value, spread = D(price), D(width)
        bar = Bar(market, OPEN.date(), OPEN, OPEN + timedelta(hours=6), value,
                  value + spread, value - spread, value, D(1000000), adv20=D(1000000000))
        rules = tuple(LevelRule(n, value * 100, D(".05"), D(".05")) for n in (1, 2, 3))
        position = Position(market, 1, 10, value, D(0), D(".05"), OPEN - timedelta(days=1))
        for path in ("O-H-L-C", "O-L-H-C"):
            cases.append(((instrument, bar, [position], rules), {"path": path, "sec_rate_override": D(0)}))
    return cases


def portfolio_case(cases):
    selected = cases[::2]
    instruments = [item[0][0] for item in selected]
    bars = []
    for index in range(8):
        for (instrument, original, _, _), _ in selected:
            at = OPEN + timedelta(days=index, hours=13 if instrument.market == "US" else 0)
            bars.append(Bar(instrument.symbol, at.date(), at, at + timedelta(hours=6), original.open,
                            original.high, original.low, original.close, original.volume, adv20=original.adv20))
    rules = {item[0][0].symbol: item[0][3] for item in selected}
    options = dict(initial_cash={"KRW": D(10000000), "USD": D(10000)},
                   initial_positions=[item[0][2][0] for item in selected], first_entry=False,
                   fx_rates=[FxRate(OPEN - timedelta(days=1), D(1300))], sec_rate_override=D(0),
                   evaluation_times=[OPEN + timedelta(days=n, hours=9) for n in range(9)])
    return (instruments, bars, rules), options


def _timed(function, repeat):
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return {"samples_seconds": samples, "median_seconds": statistics.median(samples)}


def compare_timing(reference, fast, repeat):
    expected = reference()
    if fast() != expected:
        raise AssertionError("M2/M3 불일치: 속도 결과를 승인하지 않습니다")
    baseline, compiled = _timed(reference, repeat), _timed(fast, repeat)
    return {"exact_match": True, "reference": baseline, "native": compiled,
            "speedup": baseline["median_seconds"] / compiled["median_seconds"]}


def benchmark(kernel: FastKernel, repeat=3):
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat는 양의 정수여야 합니다")
    cases = candidate_cases()
    args, options = portfolio_case(cases)
    reports = {}
    for path in ("O-H-L-C", "O-L-H-C"):
        selected = [case for case in cases if case[1]["path"] == path]
        reports[path] = {
            "candidates": compare_timing(lambda: [candidate_for_bar(*a, **k) for a, k in selected],
                                          lambda: [kernel.candidate_for_bar(*a, **k) for a, k in selected], repeat),
            "portfolio": compare_timing(lambda: run_engine(*args, **options, path=path),
                                         lambda: run_fast_engine(kernel, *args, **options, path=path), repeat)}
    return _report(kernel, cases, reports, repeat)


def _report(kernel, cases, reports, repeat):
    inputs = [{"instrument": asdict(a[0]), "bar": asdict(a[1]), "positions": [asdict(p) for p in a[2]],
               "rules": [asdict(r) for r in a[3]], "options": options} for a, options in cases]
    encoded = json.dumps(inputs, default=str, sort_keys=True).encode()
    sources = {path.name: fingerprint(path) for path in Path(__file__).parent.glob("engine*.py")}
    sources.update({name: fingerprint(Path(__file__).with_name(name))
                    for name in ("fast_kernel.py", "fast_benchmark.py", "market_rules_v1.json")})
    return {"schema_version": 1, "synthetic_only": True, "repeat": repeat, "tolerance": "0",
            "python": platform.python_version(), "platform": platform.platform(), "build": kernel.manifest,
            "source_sha256": sources, "input_sha256": sha256(encoded).hexdigest(), "inputs": inputs,
            "portfolio_bar_count": 24, "results": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = benchmark(FastKernel(args.build_dir), args.repeat)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, default=str, ensure_ascii=False, indent=2)
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
