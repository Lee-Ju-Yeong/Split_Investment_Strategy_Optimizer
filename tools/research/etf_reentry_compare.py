"""같은 표본과 원장에서 월초 재진입만 바꿔 비교한다."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from decimal import Decimal
from functools import partial
from pathlib import Path
from statistics import median

from tools.research.etf_reentry_sample import (FastKernel, _sha, _strategy, arguments, inputs_for,
                                metrics, protocol, read, sample, verify, write)
from src.research.etf.m5_parameters import CostGuardRejection


def month_number(day):
    return day.year * 12 + day.month


def month_opens(days):
    first = {}
    for day in sorted(days):
        first.setdefault(month_number(day), day)
    return first


class MonthlyReentry:
    """다음 개장 때 실제 잔고를 관측해 직전 거래일의 전량 청산을 확인한다."""

    def __init__(self, kernel, inputs):
        self.kernel = kernel
        domestic = month_opens(inputs.kr_sessions)
        self.opens = {"KR": domestic, "GOLD": domestic, "US": month_opens(inputs.us_sessions)}
        self.last_held = {}
        self.due = {}
        self.attempts = {}

    def candidate_for_bar(self, instrument, bar, positions, rules, **options):
        symbol, day = bar.symbol, bar.session_date
        previous_held_day = self.last_held.pop(symbol, None)
        if positions:
            self.last_held[symbol] = day
        elif previous_held_day is not None:
            self.due[symbol] = self.opens[instrument.market].get(month_number(previous_held_day) + 1)
        due = self.due.get(symbol)
        retry = due == day and not positions
        if due is not None and due <= day:
            del self.due[symbol]
        if retry:
            options["first_entry"] = True
        order = self.kernel.candidate_for_bar(instrument, bar, positions, rules, **options)
        if retry:
            self.attempts[symbol, bar.open_at] = order is not None
            if order is not None:
                if (order.side, order.level, order.at) != ("BUY", 1, bar.open_at):
                    raise ValueError("monthly entry must buy level one at market open")
                return replace(order, reason="monthly_reentry")
        return order

    def counts(self, result):
        fills = {(r["symbol"], r["generated_at"]) for r in result.trades if r["reason"] == "monthly_reentry"}
        rejected = {(r["symbol"], r["generated_at"]) for r in result.rejections} & self.attempts.keys()
        sent = {key for key, submitted in self.attempts.items() if submitted}
        if fills & rejected or fills | rejected != sent:
            raise ValueError("monthly orders differ from the execution ledger")
        return dict(attempted=len(self.attempts), filled=len(fills), rejected=len(rejected),
                    unavailable=len(self.attempts) - len(sent))


def compare_unit(unit, *, args, plan):
    if unit["status"] == "r1_failed":
        return dict(basket=unit["basket"], start=unit["start"], status="r1_failed",
                    paths={p: dict(status="r1_failed", error=unit["error"]) for p in plan["paths"]})
    inputs, estimates = inputs_for(args, unit, plan)
    kernel = FastKernel(args.build)
    capital = Decimal(protocol.load()["portfolio"]["initial_capital_krw"])
    paths = {}
    for path in plan["paths"]:
        control = unit["paths"][path]
        variant = MonthlyReentry(kernel, inputs)
        try:
            result = _strategy(inputs, estimates, plan["candidates"][unit["candidate"]], path, capital, variant)
        except CostGuardRejection as error:
            if control != dict(status="rejected", error=str(error)):
                raise
            paths[path] = control
            continue
        if control["status"] != "passed":
            raise ValueError("A and B cost guard statuses differ")
        stats = metrics(result, inputs, capital)
        diff = {k: Decimal(stats[k]) - Decimal(control["metrics"][k])
                for k in ("cagr", "mdd", "average_investment_fraction")}
        diff["trades"] = len(result.trades) - control["trades"]
        paths[path] = dict(status="passed", delta=diff, metrics=stats, trades=len(result.trades),
                           reentry=variant.counts(result), improved=diff["cagr"] > 0 and diff["mdd"] > 0)
    return dict(basket=unit["basket"], start=unit["start"], status="compared", paths=paths)


def summarize(rows, paths):
    report = {}
    for path in paths:
        values = [r["paths"].get(path, {"status": "r1_failed"}) for r in rows]
        passed = [v for v in values if v["status"] == "passed"]
        report[path] = dict(counts=dict(Counter(v["status"] for v in values)),
                            improved=sum(v["improved"] for v in passed),
                            median_delta={k: median(v["delta"][k] for v in passed) if passed else None
                                          for k in ("cagr", "mdd", "average_investment_fraction", "trades")},
                            reentry={k: sum(v["reentry"][k] for v in passed)
                                     for k in ("attempted", "filled", "rejected", "unavailable")})
    both = sum(all(r["paths"].get(p, {}).get("improved", False) for p in paths) for r in rows)
    return dict(planned=len(rows), improved_both_paths=both, paths=report)


def control_rows(args, plan):
    selected = sample(plan, read(args.r1 / "selection-r1.json"))
    rows = read(args.out / "original.json")
    if len(rows) != len(selected) or any(any(row[k] != v for k, v in unit.items())
                                         for unit, row in zip(selected, rows)):
        raise ValueError("control rows differ from the pinned sample")
    for path in plan["paths"]:
        strata = {r["stratum"] for r in rows for s in r["paths"].get(path, {}).get("symbols", [])
                  if s["last_full_exit"] and 2 * s["remaining_sessions"] >= s["sessions"]}
        if len(strata) < 3:
            raise ValueError("early exit criterion not met on both paths")
    return rows


def main():
    args = arguments()
    plan = verify(args)
    artifacts = {name: args.out / name for name in ("original.json", "sample.json", "independent-exit-diagnosis.json")}
    sources = {name: Path(__file__).with_name(name) for name in ("etf_reentry_compare.py", "etf_reentry_sample.py")}
    watched = {**artifacts, **sources}
    hashes = {name: _sha(path) for name, path in watched.items()}
    files = {name: hashes[name] for name in ("original.json", "sample.json")}
    diagnosis = read(args.out / "independent-exit-diagnosis.json")
    if files != diagnosis["inputs"] or not diagnosis["exploration_criterion"]["met"]:
        raise ValueError("independent diagnosis does not verify these control files")
    controls = control_rows(args, plan)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(partial(compare_unit, args=args, plan=plan), controls))
    verify(args)
    if any(_sha(path) != hashes[name] for name, path in watched.items()):
        raise ValueError("diagnosis, control inputs or code changed during comparison")
    write(args.out / "monthly-comparison.json", dict(sha256=hashes,
          summary=summarize(rows, plan["paths"]), units=rows))


if __name__ == "__main__":
    main()
