"""정식 M5 탐색 실행기: 계획의 모든 실행 단위를 재개 가능하게 실행한다. 선택·PBO·보류 판정 없음.

실행 단위는 바스켓×시작일의 1년 실행이다. 같은 실행이 그 시작일의 평가이자 뒤 두 해의 기준선 학습이다.
실패한 단위·시도·비교도 계획 횟수에 남기며, 빠진 단위가 있으면 통과로 표시하지 않는다.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import json
import os
from pathlib import Path
import time
import traceback

from . import protocol
from .baseline_engine import run_baseline
from .baseline_policy import BaselinePolicy, _exposure_rows, _hash, _weights, diagnostic_baseline
from .baseline_report import baseline_policies, paired_difference, performance
from .engine import _utc
from .engine_inputs import _sha
from .engine_types import ZERO, decimal
from .fast_kernel import FastKernel
from .m5_inputs import load_m5_inputs, read_manifest
from .m5_parameters import CONFIG_PATH, load_config, load_volatilities
from .m5_pilot import _common, _encode, _provenance, _strategy, _unchanged, _window_start
from .m5_search import FORMAL_PATH, SCHEMA as PLAN_SCHEMA


UNIT_SCHEMA = "etf-m5-unit-v1"
TAIL_ROWS = 3


def _dump(path, value):
    """새 파일만 만든다. 중간에 끊긴 쓰기가 완성된 단위로 보이지 않게 임시 파일에서 옮긴다."""
    partial = path.with_name(path.name + ".partial")
    with partial.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, default=_encode)
        stream.write("\n")
    if path.exists():
        raise FileExistsError(path)
    os.replace(partial, path)


def _exposure(result, instruments, window):
    """학습에 쓸 투자 비중 요약. 다음 평가 시작 뒤의 값을 뺄 수 있게 마지막 몇 개는 따로 남긴다."""
    with localcontext() as arithmetic:
        arithmetic.prec, arithmetic.rounding = 80, ROUND_HALF_EVEN
        rows = _exposure_rows(result, tuple(item.symbol for item in instruments), *window)
        return {"sum": sum((row[3] for row in rows), ZERO), "count": len(rows),
                "tail": [[row[0], row[3]] for row in rows[-TAIL_ROWS:]]}


def matched_policy(records, instruments, cutoff):
    """직전 두 1년 실행의 평가 시각별 투자 비중 평균. 평가 구간 시작 시각 이후의 값은 쓰지 않는다."""
    cutoff = _utc(cutoff)
    with localcontext() as arithmetic:
        arithmetic.prec, arithmetic.rounding = 80, ROUND_HALF_EVEN
        total, count, known = ZERO, 0, None
        for record in records:
            tail = [(_utc(datetime.fromisoformat(str(at))), decimal(value)) for at, value in record["tail"]]
            late = [value for at, value in tail if at >= cutoff]
            if len(late) == len(tail):
                raise ValueError("learning exposure does not end before the evaluation start")
            total += decimal(record["sum"]) - sum(late, ZERO)
            count += record["count"] - len(late)
            known = max(at for at, _ in tail if at < cutoff)
        symbols = tuple(item.symbol for item in instruments)
        return BaselinePolicy(total / count, _weights(symbols, None, known), known, cutoff, _hash(records))


def _compare(policy, strategy, metrics, inputs, options, capital, window):
    result = run_baseline(policy, inputs.instruments, inputs.bars, **options)
    if [_utc(row["at"]) for row in result.valuations] != [_utc(row["at"]) for row in strategy.valuations]:
        raise ValueError("paired results must have exactly the same evaluation times")
    base = performance(result, inputs.instruments, initial_equity_krw=capital, started_at=window[0], ended_at=window[1])
    delta = paired_difference(metrics, base)
    return {"investment_fraction": policy.investment_fraction, "known_at": policy.known_at,
            "policy_sha256": policy.sha256, "performance": asdict(base), **delta,
            "joint_improvement": delta["delta_cagr"] > ZERO and delta["delta_mdd"] > ZERO}


def _references(inputs, options, capital, window):
    """설정과 무관한 참고 기준선(보유·투자 비중 0~100%). 현금 하한·종목 상한·경로가 같으면 한 번만 실행한다."""
    symbols = tuple(item.symbol for item in inputs.instruments)
    known = window[0] - timedelta(seconds=1)
    anchor = BaselinePolicy(ZERO, _weights(symbols, None, known), known, window[0], "0" * 64, role="reference")
    rows = {}
    for name, policy in baseline_policies(anchor).items():
        if name == "matched":
            continue
        try:
            result = run_baseline(policy, inputs.instruments, inputs.bars, **options)
            rows[name] = dict(asdict(performance(result, inputs.instruments, initial_equity_krw=capital,
                                                 started_at=window[0], ended_at=window[1])), status="passed")
        except Exception as error:
            rows[name] = _failure(error)
    return rows


def _learning(directory, basket, start):
    """같은 달 직전 두 시작일의 단위 파일. 없거나 실패했으면 비교를 실패로 남긴다."""
    units = []
    for years in (2, 1):
        name = next((run["start"] for run in basket["runs"]
                     if (date.fromisoformat(run["start"]).year, date.fromisoformat(run["start"]).month)
                     == (start.year - years, start.month)), None)
        path = directory / f"{name}.json"
        if name is None or not path.is_file():
            raise ValueError(f"learning unit {start.year - years}-{start.month:02d} is missing")
        unit = json.loads(path.read_text(encoding="utf-8"))
        if unit["status"] != "completed":
            raise ValueError(f"learning unit {name} did not complete")
        units.append({(row["candidate"], row["path"]): row for row in unit["trials"]})
    return units


def _comparison(learned, key, strategy, metrics, inputs, options, capital, window, posthoc):
    if isinstance(learned, Exception):
        raise learned
    rows = [unit[key] for unit in learned]
    if any(row["status"] != "passed" for row in rows):
        raise ValueError("a learning run for this setting failed")
    policy = matched_policy([row["exposure"] for row in rows], inputs.instruments, window[0])
    result = {"status": "passed", "matched": _compare(policy, strategy, metrics, inputs, options, capital, window)}
    if posthoc:
        diagnostic = diagnostic_baseline(strategy, policy, evaluation_end=window[1])
        result["posthoc_diagnostic"] = _compare(diagnostic, strategy, metrics, inputs, options, capital, window)
    return result


def _failure(error):
    return {"status": "failed", "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc()}


def _trial(context, index, candidate, path):
    inputs, capital, window = context["inputs"], context["capital"], context["window"]
    row = {"candidate": index, "path": path}
    try:
        strategy = _strategy(inputs, context["estimates"], candidate, path, capital, context["kernel"])
        metrics = performance(strategy, inputs.instruments, initial_equity_krw=capital,
                              started_at=window[0], ended_at=window[1])
        options = _common(inputs, candidate, path, capital)
        reference = "|".join((path, candidate["cash_floor_fraction"], candidate["symbol_cap_fraction"]))
        if reference not in context["references"]:
            context["references"][reference] = _references(inputs, options, capital, window)
        row.update(status="passed", strategy=asdict(metrics), reference=reference, trades=len(strategy.trades),
                   warnings=len(strategy.warnings), exposure=_exposure(strategy, inputs.instruments, window))
    except Exception as error:
        return dict(row, **_failure(error))
    if context["learned"] is not None:
        try:
            row["comparison"] = _comparison(context["learned"], (index, path), strategy, metrics, inputs,
                                            options, capital, window, context["posthoc"])
        except Exception as error:
            row["comparison"] = _failure(error)
    return row


def _unit(root, manifest_path, plan, basket, run, directory, kernel, pinned):
    start, end = date.fromisoformat(run["start"]), date.fromisoformat(run["end"])
    symbols, started = basket["symbols"], time.perf_counter()
    record = {"schema": UNIT_SCHEMA, "basket": basket["id"], "stratum": basket["stratum"], "symbols": symbols,
              "start": run["start"], "end": run["end"], "evaluable": run["start"] in basket["evaluations"]}
    try:
        _unchanged(pinned, root, manifest_path, symbols)
        inputs = load_m5_inputs(root, manifest_path, symbols, start, end, max_fx_age_days=plan["fx"]["max_age_days"],
                                fx_closure_age_days=plan["fx"]["closure_age_days"])
        learned = None
        if record["evaluable"]:
            try:
                learned = _learning(directory, basket, start)
            except Exception as error:
                learned = error
        context = {"inputs": inputs, "estimates": load_volatilities(root, manifest_path, symbols, start),
                   "capital": Decimal(protocol.load()["portfolio"]["initial_capital_krw"]),
                   "window": (_window_start(inputs), max(inputs.evaluation_times)), "kernel": kernel,
                   "learned": learned, "references": {}, "posthoc": plan["posthoc_diagnostic"]}
        trials = [_trial(context, index, candidate, path)
                  for index, candidate in enumerate(plan["candidates"]) for path in plan["paths"]]
        _unchanged(pinned, root, manifest_path, symbols)
        record.update(status="completed", trials=trials, references=context["references"],
                      input=inputs.provenance, volatility=context["estimates"])
    except Exception as error:
        record.update(_failure(error))
    record["elapsed_seconds"] = time.perf_counter() - started
    return record


def _run_basket(job):
    root, manifest_path, plan, basket, run_dir, build_dir, pinned, starts = job
    root, manifest_path = Path(root), Path(manifest_path)
    kernel = None if build_dir is None else FastKernel(build_dir)
    directory = Path(run_dir) / "units" / basket["id"]
    directory.mkdir(parents=True, exist_ok=True)
    for run in basket["runs"]:
        path = directory / f"{run['start']}.json"
        if run["start"] in starts and not path.exists():
            _dump(path, _unit(root, manifest_path, plan, basket, run, directory, kernel, pinned))
    return basket["id"]


def _pin(manifest_path, plan_path, plan, build_dir):
    """실행을 시작한 소스·입력·계획·빌드. 재개할 때 하나라도 다르면 같은 폴더에 이어 쓰지 않는다."""
    kernel = None if build_dir is None else FastKernel(build_dir)
    pinned = _provenance(manifest_path, load_config(), kernel)
    pinned.update(schema="etf-m5-run-v1", plan_sha256=_sha(Path(plan_path)), formal_sha256=_sha(FORMAL_PATH),
                  scope=plan["scope"], synthetic=plan["synthetic"])
    for name, current in (("protocol_sha256", pinned["protocol_sha256"]), ("formal_sha256", pinned["formal_sha256"]),
                          ("configuration_sha256", _sha(CONFIG_PATH)),
                          ("input_manifest_sha256", pinned["input_manifest_sha256"])):
        if plan[name] != current:
            raise ValueError(f"plan was built for a different {name}")
    return pinned


def _pending(plan, directory, max_units):
    pending = [(basket["id"], run["start"]) for basket in plan["baskets"] for run in basket["runs"]
               if not (directory / "units" / basket["id"] / f"{run['start']}.json").exists()]
    return pending if max_units is None else pending[:max_units]


def status(plan, run_dir):
    """단위 파일에서 다시 센다. 계획한 횟수를 분모로 쓰고 빠진 단위는 통과가 아니다."""
    per_unit = len(plan["candidates"]) * len(plan["paths"])
    counts = {"units_planned": plan["counts"]["units"], "units_completed": 0, "units_failed": 0,
              "trials_planned": plan["counts"]["trials"], "trials_passed": 0,
              "comparisons_planned": plan["counts"]["comparisons"], "comparisons_passed": 0,
              "references_failed": 0}
    elapsed = 0.0
    for basket in plan["baskets"]:
        for run in basket["runs"]:
            path = Path(run_dir) / "units" / basket["id"] / f"{run['start']}.json"
            if not path.is_file():
                continue
            unit = json.loads(path.read_text(encoding="utf-8"))
            elapsed += unit["elapsed_seconds"]
            if unit["status"] != "completed" or len(unit["trials"]) != per_unit:
                counts["units_failed"] += 1
                continue
            counts["units_completed"] += 1
            counts["trials_passed"] += sum(row["status"] == "passed" for row in unit["trials"])
            counts["comparisons_passed"] += sum(row.get("comparison", {}).get("status") == "passed"
                                                for row in unit["trials"])
            counts["references_failed"] += sum(row["status"] != "passed" for rows in unit["references"].values()
                                               for row in rows.values())
    missing = counts["units_planned"] - counts["units_completed"] - counts["units_failed"]
    passed = (missing == 0 and counts["units_failed"] == 0 and counts["trials_passed"] == counts["trials_planned"]
              and counts["comparisons_passed"] == counts["comparisons_planned"]
              and counts["references_failed"] == 0)
    return dict(counts, schema="etf-m5-run-v1", scope=plan["scope"], synthetic=plan["synthetic"],
                units_missing=missing, status="passed" if passed else "incomplete" if missing else "failed",
                unit_seconds=elapsed, statistical_success_evaluated=False, holdout_evaluated=False,
                nested_validation_performed=False, actual_after_tax_approved=False)


def run_plan(data_root, manifest_path, plan_path, run_dir, *, build_dir=None, workers=1, max_units=None):
    """같은 계획·소스·입력이면 같은 폴더에서 이어 실행한다. 완성된 단위는 다시 실행하거나 덮어쓰지 않는다."""
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("invalid M5 plan")
    read_manifest(data_root, manifest_path, [])
    if not plan["synthetic"] and build_dir is None:
        raise ValueError("real-data search requires an explicit verified M3 build")
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pinned = json.loads(json.dumps(_pin(manifest_path, plan_path, plan, build_dir), default=_encode))
    record = directory / "provenance.json"
    if not record.exists():
        _dump(record, pinned)
    elif json.loads(record.read_text(encoding="utf-8")) != pinned:
        raise ValueError("run directory belongs to a different source, input, plan or build")
    pending = _pending(plan, directory, max_units)
    jobs = [(str(data_root), str(manifest_path), plan, basket, str(directory), build_dir, pinned,
             {start for name, start in pending if name == basket["id"]})
            for basket in plan["baskets"] if any(name == basket["id"] for name, _ in pending)]
    if workers == 1:
        for job in jobs:
            _run_basket(job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_run_basket, jobs))
    return status(plan, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True, help="m5_search가 만든 계획 JSON")
    parser.add_argument("--run-dir", type=Path, required=True, help="실행 폴더. 같은 계획이면 이어 실행")
    parser.add_argument("--build-dir", type=Path, help="검증된 M3 빌드 폴더, 실자료 필수")
    parser.add_argument("--workers", type=int, default=1, help="동시에 실행할 바스켓 수")
    parser.add_argument("--max-units", type=int, help="이번 호출에서 실행할 단위 수 상한")
    args = parser.parse_args()
    result = run_plan(args.data_root, args.manifest, args.plan, args.run_dir, build_dir=args.build_dir,
                      workers=args.workers, max_units=args.max_units)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
