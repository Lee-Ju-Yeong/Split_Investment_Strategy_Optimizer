"""KODEX 200 주문 크기 탐색. 기존 28번의 분할 주문액만 바꾼다.

python3 -B -m tools.research.kodex_order_size baseline|run --out RUN/full
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path
import time
from traceback import print_exc

from src.research.etf.engine_market import fee_amount
from src.research.etf.kodex_core import ledgers
from src.research.etf.kodex_core.config import load_config
from src.research.etf.kodex_grid import judge, reading, runner, sealing
from src.research.etf.kodex_grid.grid import setting_config
from src.research.etf.kodex_grid.identity import BundleSource, open_context


ROOT = Path("/mnt/volume/workspace/split-research")
DATA = ROOT / "etf/data/etf-m5-r1"
REFERENCE = ROOT / "runs/kodex-grid-r1/r3/stage1/units/setting-28.json"
REFERENCE_HASH = "e11a17f62a23cb35fe119b209435e1c9b23d05c5591702beb98c8e8822497b43"
RATES = (Fraction(1, 9), Fraction(3, 20), Fraction(1, 5), Fraction(1, 4), Fraction(1, 3))
ONE_THIRD = RATES[-1]
IMPROVEMENT = Decimal("0.0025")
SCHEMA = "kodex-order-size-unit-v1"


def decimal_fraction(rate: Fraction) -> Decimal:
    with localcontext(prec=80):
        return Decimal(rate.numerator) / rate.denominator


def budget_amount(value, rate):
    # 1/9 기준점은 기존 엔진의 연산 순서를 재현한다. 새 1/3은 나눗셈으로 정확한 체결 경계를 지킨다.
    return value / 3 if rate == ONE_THIRD else value * decimal_fraction(rate)


class OneThirdEngine(ledgers._SplitEngine):
    def _refresh_budget(self, at, payload):
        event_index = len(self.result.cash_events)
        super()._refresh_budget(at, payload)
        event = self.result.cash_events[event_index]
        if event["kind"] != "budget_refresh":
            raise ValueError("budget refresh event missing from the approved engine")
        symbol, = self.instruments
        exact = budget_amount(event["source_total_krw"], ONE_THIRD)
        self.rules[symbol] = tuple(replace(rule, budget=exact) for rule in self.rules[symbol])
        event["budgets"][symbol] = exact


def run_sized_split(inputs, setting, rate):
    """승인 장부와 같은 진입·결측 처리를 쓰고 바로 그 장부에 주문율을 전달한다."""
    config, first = setting.config, inputs.bars[0]
    plan = ledgers.volatility_plan(inputs, inputs.start, config.volatility_mode)
    if any(row["date"] == plan["start"] for row in plan["unavailable"]):
        return ledgers.LedgerRun(None, "volatility_unavailable", {"unavailable": plan["unavailable"]})

    def rates(session, estimate):
        return ledgers.split_rates(setting, inputs, session, estimate)

    try:
        drop, take = rates(first, inputs.volatility[plan["start"]])
    except ledgers.CostGuardRejection as error:
        return ledgers.LedgerRun(None, "cost_guard_start", {"reason": str(error)})
    ended = ledgers._observation_end(inputs, plan) if plan["unavailable"] else None
    observed = inputs if ended is None else ledgers._until(inputs, ended["last_valid_session"])
    if ended is not None:
        plan = {**plan, "refreshes": [day for day in plan["refreshes"] if day <= ended["last_valid_session"]]}
    return run_split_engine(observed, setting, (plan, rates, drop, take, ended, rate))


def run_split_engine(inputs, setting, params):
    plan, rates, drop, take, ended, rate = params
    config = setting.config
    fraction = decimal_fraction(rate)
    rules = {inputs.instrument.symbol: tuple(ledgers.LevelRule(n, ledgers.ONE, drop, take)
                                             for n in range(1, config.levels + 1))}
    hooks = ledgers._split_hooks(inputs, setting, rates, plan)
    options = ledgers._options(setting, first_entry=True, first_level_fraction=fraction)
    engine_inputs = dict(ledgers._defaults(ledgers.run_engine),
                         **ledgers._common(inputs, setting, config.capital["split"]),
                         rules=rules, first_level_fraction=fraction, _reentry_provider=hooks["schedule"])
    with localcontext(prec=80):
        engine_type = OneThirdEngine if rate == ONE_THIRD else ledgers._SplitEngine
        engine = engine_type(engine_inputs, options, hooks)
        result = engine.run()
        check_budgets(result, inputs.bars, BuyBudget(config.capital["split"], rate, setting.commission))
    detail = {"rates": (drop, take), "volatility_start": plan["start"], "refreshes": engine.refresh_log,
              "reentries": [] if hooks["schedule"] is None else hooks["schedule"].attempts}
    if any(row["kind"] == "first_entry_unavailable" for row in result.cash_events):
        return ledgers.LedgerRun(result, "start_failed", detail)
    if ended is not None:
        return ledgers.LedgerRun(result, "observation_ended", {**detail, **ended})
    return ledgers.LedgerRun(result, "ok", detail)


@dataclass
class SizeJob(runner._Job):
    rate: Fraction

    def ledgers(self, cost, path):
        accounts = {name: self.hold(cost, path, name) for name in ("core", "hold")}
        setting = ledgers.RunSetting(self.config, path, self.config.cost(cost), self.mode)
        accounts[self.mode] = run_sized_split(self.data, setting, self.rate)
        return accounts


@dataclass(frozen=True)
class BuyBudget:
    capital: Decimal
    rate: Fraction
    commission: Decimal


def monthly_opens(bars):
    opens = {}
    for bar in bars:
        opens.setdefault((bar.session_date.year, bar.session_date.month), bar.open_at)
    return list(opens.values())


def check_fill(event, limit, commission):
    price, shares, day = event["price"], event["quantity"], event["session_date"]
    gross, next_gross = price * shares, price * (shares + 1)
    fee = fee_amount("KR", "BUY", day, gross, commission_rate=commission)
    next_fee = fee_amount("KR", "BUY", day, next_gross, commission_rate=commission)
    if limit is None or shares <= 0 or event["fee"] != fee or not gross + fee <= limit < next_gross + next_fee:
        raise ValueError("buy quantity or fee differs from the active order budget")


def check_budgets(result, bars, order):
    valuations = result.valuations
    times = [row["at"] for row in valuations]
    refreshes = [event["at"] for event in result.cash_events if event["kind"] == "budget_refresh"]
    if refreshes != monthly_opens(bars):
        raise ValueError("order budgets were not reset exactly at the registered monthly opens")
    budget = None
    for event in result.cash_events:
        if event["kind"] == "budget_refresh":
            prior = bisect_left(times, event["at"]) - 1
            source = order.capital if prior < 0 else valuations[prior]["total_krw"]
            budget, = event["budgets"].values()
            if event["source_total_krw"] != source or budget != budget_amount(source, order.rate):
                raise ValueError("order budget differs from the preceding valuation and chosen fraction")
        elif event["kind"] == "trade" and event["side"] == "BUY":
            check_fill(event, budget, order.commission)
    return len(refreshes)


def order_stats(run):
    if run.result is None:
        return {"status": run.status, "budget_refreshes": None, "buy_levels": None,
                "highest_buy_level": None, "buy_rejections": None}
    events = run.result
    levels = Counter(trade["level"] for trade in events.trades if trade["side"] == "BUY")
    rejections = Counter(row["reason"] for row in events.rejections if row["side"] == "BUY")
    return {"status": run.status, "budget_refreshes": sum(row["kind"] == "budget_refresh" for row in events.cash_events),
            "buy_levels": {str(n): levels[n] for n in sorted(levels)},
            "highest_buy_level": max(levels, default=0), "buy_rejections": dict(rejections)}


def run_point(job, start):
    head = runner._head(job, start, None)
    with localcontext(prec=80):
        runs = {cost: {path: job.ledgers(cost, path) for path in job.config.paths} for cost in runner.COSTS}
        orders = {path: {cost: order_stats(runs[cost][path][job.mode])
                         for cost in runner.COSTS} for path in job.config.paths}
        result = runner.assemble_point(runner.PointLedgers(job, head, runs, job.split_holds()))
    return result, orders


def compare_baseline(value, reference):
    """두 경로의 장부 지표·거래 상태·분할 몫 집계를 기존 28번과 그대로 대조한다."""
    if json.loads(sealing.dump(value)) != reference:
        raise ValueError("1/9 result differs from the archived setting 28")


def reference_result():
    if sealing.file_digest(REFERENCE) != REFERENCE_HASH:
        raise ValueError("archived setting 28 hash differs")
    return json.loads(REFERENCE.read_text())["result"]


def identity(context, out):
    return {**context.identity, "study": SCHEMA, "source_sha256": sealing.file_digest(Path(__file__)),
            "design_sha256": sealing.file_digest(out.parent / "DESIGN-v1.md"),
            "reference_sha256": REFERENCE_HASH, "rates": [str(rate) for rate in RATES]}


def unit_path(out, rate):
    return out / "sealed" / f"{rate.numerator}-{rate.denominator}.json"


def read_unit(path, expected, rate):
    row = sealing.read_sealed(path, SCHEMA)
    if row["identity"] != expected or row["rate"] != str(rate):
        raise ValueError(f"{path.name} belongs to a different input, source or order rate")
    return row


def run_unit(work, rate):
    out, identity_, data, config, bench, reference = work
    path = unit_path(out, rate)
    if path.exists():
        row = read_unit(path, identity_, rate)
        if rate == RATES[0]:
            compare_baseline(row["result"], reference)
        print(f"{rate} skipped", flush=True)
        return row
    started = time.monotonic()
    job = SizeJob(data, config, bench, rate)
    failure = None
    with sealing.captured_output(out / "sealed" / f"{path.stem}.log"):
        try:
            result, orders = run_point(job, data.start)
            if rate == RATES[0]:
                compare_baseline(result, reference)
            row = {"schema": SCHEMA, "identity": identity_, "rate": str(rate), "result": result, "orders": orders}
            sealing.write_once(path, sealing.dump(sealing.with_seal(row)), mode=sealing.SEALED)
        except Exception as error:
            print_exc()
            failure = sealing.error_code(error)
    if failure:
        raise SystemExit(f"{rate} failed {failure}")
    print(f"{rate} ok {time.monotonic() - started:.1f}s", flush=True)
    return read_unit(path, identity_, rate)


def improvement(point, reference):
    gap = {}
    for path, row in point["paths"].items():
        base = reference["paths"][path]
        gap[path] = {cost: reading.exact_sub(row[cost]["delta_cagr"], base[cost]["delta_cagr"])
                     if judge.cost_valid(row, cost) and judge.cost_valid(base, cost) else None
                     for cost in runner.COSTS}
    values = [value for path in gap.values() for value in path.values()]
    passed = None if None in values else all(row["base"] >= IMPROVEMENT and row["stress"] > 0
                                              for row in gap.values())
    return {"deltas": gap, "passes": passed}


def summarize(units, spec, config):
    frame = reading.Frame(config.paths, tuple(segment.label for segment in config.segments), config.thresholds)
    points = [reading.point_input(unit["result"], frame) for unit in units]
    results = []
    for unit, point in zip(units, points, strict=True):
        results.append({"rate": unit["rate"], "verdict": judge.setting_verdict(point, spec.thresholds),
                        "metrics": reading.point_display(unit["result"], frame), "orders": unit["orders"],
                        "vs_reference": improvement(point, points[0])})
    adjacent = [[a["rate"], b["rate"]] for a, b in zip(results, results[1:])
                if a["verdict"]["status"] == b["verdict"]["status"] == judge.PASS]
    possible = any(judge.FAIL not in (a["verdict"]["status"], b["verdict"]["status"])
                   for a, b in zip(results, results[1:]))
    return {"schema": "kodex-order-size-result-v1", "exploration": True, "formal_run": False,
            "identity": units[0]["identity"], "adjacent_passes": adjacent,
            "extension_status": judge.PASS if adjacent else judge.OPEN if possible else judge.FAIL, "rates": results}


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}"


def markdown(result):
    lines = ["# KODEX 200 회당 주문 크기 결과", "",
             "같은 ETF 전액 보유 대비 연복리 차는 %p, 투자 비중은 %다. 주문율의 분모는 분할 몫이다.", "",
             "| 주문율 | 장중 순서 | 기본 비용 차 | 높은 비용 차 | 평균 투자 비중 | 현금 부족 거절 | 도달 차수 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["rates"]:
        for path, values in row["metrics"]["paths"].items():
            base, stress = values["base"] or {}, values["stress"] or {}
            orders = row["orders"][path]["base"]
            rejections = orders["buy_rejections"]
            cash = "—" if rejections is None else rejections.get("insufficient_cash", 0)
            level = "—" if orders["highest_buy_level"] is None else orders["highest_buy_level"]
            lines.append(f"| {row['rate']} | {path} | {percent(base.get('total_delta'))} | "
                         f"{percent(stress.get('total_delta'))} | {percent(base.get('exposure_mean'))} | {cash} | {level} |")
    improves = ", ".join(row["rate"] for row in result["rates"] if row["vs_reference"]["passes"]) or "없음"
    labels = {judge.PASS: "통과", judge.FAIL: "미달", judge.OPEN: "미확정"}
    adjacent = ", ".join("·".join(pair) for pair in result["adjacent_passes"]) or labels[result["extension_status"]]
    return "\n".join([*lines, "", f"- 기준점 대비 진단선 통과 {improves}",
                      f"- 전액 보유 대비 기준을 통과한 인접 주문율 {adjacent}",
                      "- 분할 몫 수익과 비중을 맞춘 계좌 대비 수익 차는 result.json에 있다.", ""])


def pinned_context():
    return open_context(BundleSource(DATA, DATA / "m5-inputs.json", load_config()))


def stage(out, stop_after_baseline):
    context = pinned_context()
    spec = context.spec
    config = setting_config(context.base, spec, spec.setting(28))
    if spec.baseline_number() != 28:
        raise ValueError("the registered baseline setting changed")
    expected = identity(context, out)
    reference = reference_result()
    data = runner._from(context.source.load(spec.symbol, spec.primary_start), spec.primary_start)
    with sealing.writer_lock(out):
        sealing.write_once(out / "identity.json", sealing.dump(expected))
        work = (out, expected, data, config, runner.Bench(), reference)
        baseline = run_unit(work, RATES[0])
        if stop_after_baseline:
            return
        units = [baseline] + [run_unit(work, rate) for rate in RATES[1:]]
        if identity(pinned_context(), out) != expected:
            raise ValueError("the input bundle, design or execution code changed during the run")
        reference_result()
        result = summarize(units, spec, config)
        sealing.write_once(out / "result.json", sealing.dump(sealing.with_seal(result)))
        sealing.write_once(out / "summary.md", markdown(result))
        print("assembled 5/5", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baseline", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    stage(args.out.resolve(), args.stage == "baseline")


if __name__ == "__main__":
    main()
