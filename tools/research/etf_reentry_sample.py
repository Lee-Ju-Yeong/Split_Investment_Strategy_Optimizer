"""R1 탐색 표본을 고정하고 원래 전략을 재현해 종목별 청산 뒤 거래일을 기록한다."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import date
from decimal import Decimal
from functools import partial
from hashlib import sha256
import json
from pathlib import Path

from src.research.etf import protocol
from src.research.etf.baseline_report import performance
from src.research.etf.engine_inputs import _sha
from src.research.etf.fast_kernel import FastKernel
from src.research.etf.m5_inputs import load_m5_inputs, read_manifest
from src.research.etf.m5_parameters import CostGuardRejection, load_volatilities
from src.research.etf.m5_pilot import _encode, _strategy, _window_start
from src.research.etf.m5_runner import _pin


SEED = "etf-reentry-r1"
PLAN_SHA = "d4765a37d3764e16462192c48cf4f8c061ccce743d0f98fcdef7b6efedf660ee"
PROVENANCE_SHA = "0d1d2c76241223d056aac4089a8680db57b30f0c41a0b49358278fef2b9aabaa"
UNIT_SET_SHA = "15c76c5e6804bc18ac90e9cb928598a9932b0ac75d4a549d59b501532e269790"
SELECTION_SHA = "600eca64a31f9b4ef72350bfafcef12aa7ce50a1e0cc33733f90ce3b90a3722b"


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=_encode)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def rank(*parts):
    return sha256("|".join((SEED, *parts)).encode()).hexdigest()


def sample(plan, selections):
    folds = {s: f["selected"] for f in selections["folds"] for s in f["test_starts"]}
    strata = defaultdict(list)
    for basket in plan["baskets"]:
        starts = set(basket["evaluations"]).intersection(folds)
        if starts:
            start = min(starts, key=lambda s: rank("date", basket["id"], s))
            end = next(r["end"] for r in basket["runs"] if r["start"] == start)
            strata[basket["stratum"]].append(dict(basket=basket["id"], symbols=basket["symbols"],
                start=start, end=end, stratum=basket["stratum"], candidate=folds[start]))
    if len(strata) != 6 or any(len(baskets) < 4 for baskets in strata.values()):
        raise ValueError("expected six populated strata")
    return [row for key in sorted(strata) for row in sorted(strata[key],
            key=lambda r: rank("basket", key, r["basket"]))[:4]]


def history(result, bars, symbol):
    days = sorted(bar.session_date for bar in bars if bar.symbol == symbol)
    trades = [row for row in result.trades if row["symbol"] == symbol]
    quantity, first, exits = 0, None, []
    for row in trades:
        if row["side"] == "BUY":
            first = first or row["session_date"]
            quantity += row["quantity"]
        else:
            quantity -= row["quantity"]
            if quantity == 0:
                exits.append(row["session_date"])
        if quantity < 0:
            raise ValueError("negative ledger position")
    if quantity != sum(position.quantity for position in result.positions if position.symbol == symbol):
        raise ValueError("terminal quantity differs from trades")
    last = exits[-1] if exits and quantity == 0 else None
    remaining = sum(day > last for day in days) if last else 0
    return dict(symbol=symbol, sessions=len(days), first_buy=first, full_exits=exits, terminal_quantity=quantity,
                last_full_exit=last, remaining_sessions=remaining if first else None)


def inputs_for(args, unit, plan):
    manifest = args.data / "m5-inputs.json"
    span = [date.fromisoformat(unit[k]) for k in ("start", "end")]
    inputs = load_m5_inputs(args.data, manifest, unit["symbols"], *span,
        max_fx_age_days=plan["fx"]["max_age_days"], fx_closure_age_days=plan["fx"]["closure_age_days"])
    estimates = load_volatilities(args.data, manifest, unit["symbols"], span[0])
    return inputs, estimates


def metrics(result, inputs, capital):
    stats = performance(result, inputs.instruments, initial_equity_krw=capital,
                        started_at=_window_start(inputs), ended_at=max(inputs.evaluation_times))
    return json.loads(json.dumps(asdict(stats), default=_encode))


def original_unit(unit, *, args, plan):
    old = read(args.r1 / "run/units" / unit["basket"] / (unit["start"] + ".json"))
    if (old["basket"], old["start"], old["end"], old["symbols"]) != (
            unit["basket"], unit["start"], unit["end"], unit["symbols"]):
        raise ValueError("R1 unit identity mismatch")
    if old["status"] == "failed":
        return dict(unit, status="r1_failed", error=old["error"], paths={})
    inputs, estimates = inputs_for(args, unit, plan)
    kernel = FastKernel(args.build)
    capital = Decimal(protocol.load()["portfolio"]["initial_capital_krw"])
    paths = {}
    for name in plan["paths"]:
        ref, = [r for r in old["trials"] if (r["candidate"], r["path"]) == (unit["candidate"], name)]
        try:
            result = _strategy(inputs, estimates, plan["candidates"][unit["candidate"]], name, capital, kernel)
        except CostGuardRejection as error:
            if (ref["status"], ref["error"]) != ("rejected", str(error)):
                raise
            paths[name] = dict(status="rejected", error=str(error))
            continue
        stats = metrics(result, inputs, capital)
        if ref["status"] != "passed" or stats != ref["strategy"] or len(result.trades) != ref["trades"]:
            raise ValueError(f"R1 strategy replay differs: {unit['basket']} {unit['start']} {name}")
        paths[name] = dict(status="passed", metrics=stats, trades=len(result.trades),
            symbols=[history(result, inputs.bars, s) for s in unit["symbols"]])
    return dict(unit, status="replayed", paths=paths)


def unit_set_hash(directory, plan):
    digest = sha256()
    for basket in plan["baskets"]:
        for run in basket["runs"]:
            key = f"{basket['id']}/{run['start']}"
            digest.update(f"{key} {_sha(directory / (key + '.json'))}\n".encode())
    return digest.hexdigest()


def verify(args):
    src = args.r1 / "plan.json"
    if _sha(src) != PLAN_SHA or _sha(args.r1 / "selection-r1.json") != SELECTION_SHA:
        raise ValueError("R1 plan or selection digest differs")
    plan = read(src)
    if unit_set_hash(args.r1 / "run/units", plan) != UNIT_SET_SHA:
        raise ValueError("R1 unit files differ from the published closeout digest")
    manifest = args.data / "m5-inputs.json"
    read_manifest(args.data, manifest, [])
    provenance = args.r1 / "run/provenance.json"
    if _sha(provenance) != PROVENANCE_SHA or _pin(manifest, src, plan, args.build) != read(provenance):
        raise ValueError("use the R1 pinned worktree, source, inputs and build")
    return plan


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("r1", "data", "build", "out"):
        p.add_argument("--" + option, type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    return p.parse_args()


def main():
    args = arguments()
    source = _sha(Path(__file__))
    plan = verify(args)
    units = sample(plan, read(args.r1 / "selection-r1.json"))
    write(args.out / "sample.json", dict(seed=SEED, plan_sha=PLAN_SHA, selection_sha=SELECTION_SHA, units=units))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(partial(original_unit, args=args, plan=plan), units))
    verify(args)
    if _sha(Path(__file__)) != source:
        raise ValueError("replay script changed during execution")
    write(args.out / "original.json", rows)


if __name__ == "__main__":
    main()
