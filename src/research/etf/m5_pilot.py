"""고정한 두 설정×두 경로의 작은 탐색 연결 검사. 선택·중첩 검증·보류 판정 없음."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import time
import traceback

from . import protocol
from .baseline_policy import fit_baseline
from .baseline_report import comparison_report, run_baseline_suite
from .engine import run_engine
from .engine_inputs import _rows, _sha
from .fast_kernel import FastKernel, run_fast_engine
from .m5_inputs import load_m5_inputs, read_manifest
from .m5_parameters import CONFIG_PATH, candidate_rules, load_config, load_volatilities


def _encode(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, default=_encode)
        stream.write("\n")


def _event(directory, value):
    value = dict(value, recorded_at=datetime.now(timezone.utc))
    with (directory / "trials.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=_encode) + "\n")


def source_hashes():
    root = Path(__file__).parent
    return {path.relative_to(root).as_posix(): _sha(path) for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".json", ".cpp"}}


def _basket(data_root, manifest, symbols, config):
    root = Path(data_root)
    members = {row["symbol"]: row for row in _rows(root / manifest["split_csv"])}
    counts = dict(Counter(members[s]["market"] for s in symbols))
    if counts not in config["market_mix_strata"] or len(symbols) not in config["k_values"]:
        raise ValueError("pilot basket must match a pinned market stratum")
    groups = [members[s].get("group") for s in symbols if members[s]["market"] != "GOLD"]
    if any(not group for group in groups) or len(groups) != len(set(groups)):
        raise ValueError("pilot requires distinct frozen index-family groups")
    metadata = {(row["market"], row["symbol"]): row for row in _rows(root / "metadata" / "etf_metadata.csv")}
    for symbol in symbols:
        market = members[symbol]["market"]
        if market != "GOLD" and metadata[(market, symbol)].get("leverage_multiple") not in {"1", "1.0"}:
            raise ValueError("small operational pilot supports confirmed x1_only baskets")


def _period(data_root, manifest_path, symbols, pilot, period):
    start, end = (date.fromisoformat(pilot[f"{period}_{key}"]) for key in ("start", "end"))
    inputs = load_m5_inputs(data_root, manifest_path, symbols, start, end,
                            max_fx_age_days=pilot["max_fx_age_days"])
    estimates = load_volatilities(data_root, manifest_path, symbols, start)
    return inputs, estimates


def _common(inputs, candidate, path, capital):
    return dict(inputs.engine_kwargs(), initial_cash={"KRW": capital}, path=path,
                cash_floor_fraction=Decimal(candidate["cash_floor_fraction"]),
                symbol_cap_fraction=Decimal(candidate["symbol_cap_fraction"]))


def _strategy(inputs, estimates, candidate, path, capital, kernel):
    rules = candidate_rules(candidate, inputs, estimates)
    options = _common(inputs, candidate, path, capital)
    options["first_level_fraction"] = Decimal(candidate["first_level_fraction"])
    if kernel is None:
        return run_engine(inputs.instruments, inputs.bars, rules, **options)
    return run_fast_engine(kernel, inputs.instruments, inputs.bars, rules, **options)


def _trial(training, evaluation, candidate, path, capital, kernel):
    learned_inputs, learned_vol = training
    inputs, estimates = evaluation
    learned = _strategy(learned_inputs, learned_vol, candidate, path, capital, kernel)
    start = _window_start(inputs)
    frozen = fit_baseline(learned, inputs.instruments,
                          training_start=_window_start(learned_inputs),
                          training_end=max(learned_inputs.evaluation_times), evaluation_start=start)
    strategy = _strategy(inputs, estimates, candidate, path, capital, kernel)
    suite = run_baseline_suite(frozen, inputs.instruments, inputs.bars, **_common(inputs, candidate, path, capital))
    report = comparison_report(strategy, suite, inputs.instruments, initial_equity_krw=capital,
                               started_at=start, ended_at=max(inputs.evaluation_times))
    return {"comparison": report, "frozen_policy": frozen,
            "training_volatility": learned_vol, "evaluation_volatility": estimates,
            "training_input": learned_inputs.provenance, "evaluation_input": inputs.provenance,
            "training_ledger": learned, "strategy_ledger": strategy,
            "baseline_ledgers": {name: result for name, (_, result) in suite.items()}}


def _window_start(inputs):
    return min([bar.open_at for bar in inputs.bars] + inputs.conversion_times)


def _provenance(manifest_path, config, kernel):
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return {"schema": "etf-m5-pilot-v1", "source_git_sha": revision, "source_sha256": source_hashes(),
            "configuration": config, "configuration_sha256": _sha(CONFIG_PATH),
            "protocol_sha256": protocol.protocol_hash(protocol.load()),
            "input_manifest_sha256": _sha(manifest_path), "native_build": None if kernel is None else kernel.manifest,
            "scope": "operational_only_no_selection_no_nested_validation",
            "statistical_success_evaluated": False, "actual_after_tax_approved": False}


def run_pilot(data_root, manifest_path, symbols, run_dir, *, build_dir=None):
    """실패한 시도도 보존한다. 새 run_dir만 허용하며 중단된 실행을 성공으로 보지 않는다."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    records, synthetic, planned = [], None, 4
    try:
        config = load_config()
        planned = len(config["pilot"]["candidates"]) * len(config["pilot"]["paths"])
        manifest = read_manifest(data_root, manifest_path, symbols)
        synthetic = manifest["synthetic"]
        if not synthetic and build_dir is None:
            raise ValueError("real-data pilot requires an explicit verified M3 build")
        kernel = None if build_dir is None else FastKernel(build_dir)
        pinned = _provenance(manifest_path, config, kernel)
        _write(directory / "provenance.json", dict(pinned, symbols=symbols, synthetic=synthetic))
        _basket(data_root, manifest, symbols, config)
        training = _period(data_root, manifest_path, symbols, config["pilot"], "training")
        evaluation = _period(data_root, manifest_path, symbols, config["pilot"], "evaluation")
        capital = Decimal(protocol.load()["portfolio"]["initial_capital_krw"])
        for index, candidate in enumerate(config["pilot"]["candidates"]):
            for path in config["pilot"]["paths"]:
                identity = f"candidate-{index + 1:02d}-{path}"
                records.append(_attempt(directory, identity, candidate, path, training, evaluation, capital,
                                        kernel, pinned, data_root, manifest_path, symbols))
    except Exception as error:
        _event(directory, {"event": "preflight_failed", "error_type": type(error).__name__,
                           "error": str(error), "traceback": traceback.format_exc(), "symbols": symbols,
                           **_failure_evidence(manifest_path)})
    succeeded = sum(row["status"] == "passed" for row in records)
    summary = {"schema": "etf-m5-pilot-v1", "synthetic": synthetic, "planned": planned,
               "completed": len(records), "succeeded": succeeded, "failed": len(records) - succeeded,
               "status": "passed" if len(records) == succeeded == planned else "failed",
               "elapsed_seconds": time.perf_counter() - started, "trials": records,
               "statistical_success_evaluated": False, "holdout_evaluated": False,
               "actual_after_tax_approved": False, "nested_validation_performed": False}
    _write(directory / "summary.json", summary)
    return summary


def _attempt(directory, identity, candidate, path, training, evaluation, capital,
             kernel, pinned, data_root, manifest_path, symbols):
    started = time.perf_counter()
    record = {"trial_id": identity, "candidate": candidate, "path": path}
    _event(directory, dict(record, event="started"))
    try:
        _unchanged(pinned, data_root, manifest_path, symbols)
        result = _trial(training, evaluation, candidate, path, capital, kernel)
        _unchanged(pinned, data_root, manifest_path, symbols)
        _write(directory / f"{identity}.json", result)
        record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    record["elapsed_seconds"] = time.perf_counter() - started
    _event(directory, dict(record, event="finished"))
    return record


def _unchanged(pinned, data_root, manifest_path, symbols):
    read_manifest(data_root, manifest_path, symbols)
    if source_hashes() != pinned["source_sha256"] or _sha(manifest_path) != pinned["input_manifest_sha256"]:
        raise ValueError("source or input manifest changed during the run")


def _failure_evidence(manifest_path):
    evidence, errors = {}, {}
    probes = {"source_sha256": source_hashes, "configuration_sha256": lambda: _sha(CONFIG_PATH),
              "input_manifest_sha256": lambda: _sha(Path(manifest_path))}
    for name, probe in probes.items():
        try:
            evidence[name] = probe()
        except Exception as error:
            evidence[name] = None
            errors[name] = f"{type(error).__name__}: {error}"
    return dict(evidence, hash_errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", required=True, help="원본 explore의 1배 상품 코드, 순서 고정")
    parser.add_argument("--run-dir", type=Path, required=True, help="새 실행 폴더")
    parser.add_argument("--build-dir", type=Path, help="검증된 M3 빌드 폴더, 실자료 필수")
    args = parser.parse_args()
    result = run_pilot(args.data_root, args.manifest, args.symbols, args.run_dir, build_dir=args.build_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_encode))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
