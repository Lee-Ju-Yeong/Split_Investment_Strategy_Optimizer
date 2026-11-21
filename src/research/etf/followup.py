"""Run the phased ETF follow-up from the development sample through forward verification."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

from . import protocol
from .engine_inputs import _sha
from .fast_kernel import FastKernel
from .followup_plan import MIXES, PATHS, PHASES, SCHEMA, make_plan
from .followup_report import jobs_for, phase_settings, summarize
from .followup_run import StudyData, execute_unit, prepare_worker
from .m5_inputs import read_manifest
from .m5_pilot import _encode, _provenance
from .m5_runner import _dump


GIB = 1024 ** 3


def _read(path):
    return json.loads(path.read_bytes())


def _write_or_verify(path, contents):
    encoded = json.loads(json.dumps(contents, default=_encode))
    if path.exists():
        if _read(path) != encoded:
            raise ValueError(f"{path} differs from the pinned execution")
    else:
        _dump(path, encoded)
    return _read(path)


@contextmanager
def _exclusive(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runner.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _pins(args, plan):
    native = FastKernel(args.build_dir)
    pinned = _provenance(args.manifest, {}, native)
    pinned["research_schema"] = SCHEMA
    pinned["plan_sha256"] = _sha(args.run_dir / "plan.json")
    if (pinned["protocol_sha256"] != plan["protocol_sha256"]
            or pinned["input_manifest_sha256"] != plan["manifest_sha256"]
            or protocol.protocol_hash(protocol.load()) != plan["protocol_sha256"]):
        raise ValueError("plan belongs to a different protocol or input manifest")
    return pinned


def _expected(job):
    return [(setting, path) for setting in job["settings"] for path in PATHS]


def _unit_digest(row):
    encoded = json.loads(json.dumps(row, default=_encode))
    return hashlib.sha256(json.dumps(encoded, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _read_unit(path, job):
    record = _read(path)
    row = record.get("unit", {})
    if any(row.get(k) != v for k, v in job.items()):
        raise ValueError(f"unit identity differs: {path}")
    actual = [(entry["setting"], entry["path"]) for entry in row["trials"]]
    if actual != _expected(job):
        raise ValueError(f"unit trial list differs: {path}")
    if _unit_digest(row) != record.get("sha256"):
        raise ValueError(f"unit contents differ from their checksum: {path}")
    return row


def _unit_rows(directory, jobs):
    for job in jobs:
        path = directory / "units" / (job["id"] + ".json")
        if not path.is_file():
            raise ValueError(f"missing planned unit {path}")
        yield path, _read_unit(path, job)


def _index(directory, jobs):
    digest = hashlib.sha256()
    rows = []
    for path, row in _unit_rows(directory, jobs):
        digest.update(f"{path.name} {_sha(path)}\n".encode())
        rows.append(row)
    return rows, digest.hexdigest()


def _pilot_report(rows):
    result = {}
    for mix in MIXES:
        group = [row for row in rows if row["mix"] == mix]
        passes = [row for row in group if all(t["status"] == "passed" for t in row["trials"])]
        result[mix] = {"units": len(group), "complete_strategy_units": len(passes),
                       "elapsed_seconds": [row["elapsed_seconds"] for row in group],
                       "peak_rss_kib": max(row["peak_rss_kib"] for row in group)}
    return result


def _avail():
    with Path("/proc/meminfo").open() as file:
        return next(int(row.split()[1]) * 1024 for row in file if row.startswith("MemAvailable:"))


def _cpu_idle():
    with Path("/proc/stat").open() as file:
        fields = list(map(int, file.readline().split()[1:9]))
    return sum(fields), fields[3] + fields[4]


def _idle_cores():
    total, idle = _cpu_idle()
    time.sleep(1)
    end, idle_end = _cpu_idle()
    return len(os.sched_getaffinity(0)) * (idle_end - idle) / max(1, end - total)


def _workers(args, phase, pilot):
    if phase == "pilot":
        return min(args.workers, 3)
    peak = max(row["peak_rss_kib"] for row in pilot.values()) * 1024
    requested = min(args.workers, len(os.sched_getaffinity(0)) - 2)
    idle = _idle_cores()
    feasible = min(requested, max(0, int(idle + 0.1) - 2), max(0, (_avail() - 8 * GIB) // max(1, peak)))
    if feasible < 1:
        raise RuntimeError("resource pause: keep 2 CPU cores and 8 GiB available before resuming")
    return feasible


def _run(args, study, phase, jobs, workers):
    folder = args.run_dir / phase / "units"
    folder.mkdir(parents=True, exist_ok=True)
    pending = []
    for job in jobs:
        path = folder / (job["id"] + ".json")
        if path.exists():
            _read_unit(path, job)
        else:
            pending.append(job)
    if args.max_units is not None:
        pending = pending[:args.max_units]
    if not pending:
        return
    print(f"{phase}: {len(jobs)} planned, scheduling {len(pending)} with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"),
                             initializer=prepare_worker, initargs=(study, args.build_dir)) as pool:
        done = 0
        for start in range(0, len(pending), workers):
            if _avail() < 8 * GIB or shutil.disk_usage(folder).free < 8 * GIB:
                raise RuntimeError("resource pause: preserve available memory and disk for other jobs")
            chunk = pending[start:start + workers]
            futures = [pool.submit(execute_unit, job) for job in chunk]
            for future in as_completed(futures):
                row = future.result()
                _dump(folder / (row["id"] + ".json"), {"unit": row, "sha256": _unit_digest(row)})
                done += 1
                print(f"{phase}: +{done}/{len(pending)} {row['id']} {row['elapsed_seconds']:.1f}s", flush=True)


def _phase_report(args, plan, phase, previous, *, study=None, pilot=None):
    directory = args.run_dir / phase
    plan_file = args.run_dir / ("forward-plan.json" if phase == "forward" else "plan.json")
    jobs = list(jobs_for(plan, phase, phase_settings(phase, previous)))
    plan_sha = _sha(plan_file)
    execution_sha = _unit_digest([_sha(args.run_dir / "provenance.json"), plan_sha])
    for job in jobs:
        job["execution_sha256"] = execution_sha
    if study is not None:
        _run(args, study, phase, jobs, _workers(args, phase, pilot))
        read_manifest(args.data_root, args.manifest, [])
        _write_or_verify(args.run_dir / "provenance.json", _pins(args, plan))
    rows, unit_digest = _index(directory, jobs)
    report = _pilot_report(rows) if phase == "pilot" else summarize(phase, rows, previous)
    sealed = dict(phase=phase, plan_sha256=plan_sha, unit_set_sha256=unit_digest, report=report)
    output = directory / "summary.json"
    if study is None and not output.is_file():
        raise ValueError(f"{phase} is not complete")
    _write_or_verify(output, sealed)
    return _read(output)["report"]


def _plan(args):
    output = args.run_dir / "plan.json"
    if output.exists() and (args.run_dir / "provenance.json").exists():
        return _read(output)
    return _write_or_verify(output, make_plan(args.data_root, args.manifest))


def _forward_plan(args, development):
    output = args.run_dir / "forward-plan.json"
    selection = _sha(args.run_dir / "reentry" / "summary.json")
    basis = {"development_sha256": _sha(args.run_dir / "plan.json"), "selection_sha256": selection}
    if output.exists():
        sealed = _read(output)
        plan = sealed["plan"]
        if sealed["basis"] != basis or sealed["sha256"] != _unit_digest(plan):
            raise ValueError("forward plan or its frozen choice changed during resume")
    else:
        plan = make_plan(args.data_root, args.manifest, forward=True)
        sealed = _write_or_verify(output, dict(basis=basis, plan=plan, sha256=_unit_digest(plan)))
        plan = sealed["plan"]
    if any(plan[k] != development[k] for k in ("protocol_sha256", "manifest_sha256", "windows", "forward")):
        raise ValueError("forward plan does not match the development assumptions")
    return plan


def execute(args):
    with _exclusive(args.run_dir):
        plan = _plan(args)
        _write_or_verify(args.run_dir / "provenance.json", _pins(args, plan))
        if args.stage == "plan":
            print(f"frozen: {len(plan['families'])} basket families, {len(plan['development'])} development, {len(plan['forward'])} forward starts")
            return
        study = None
        if args.stage == "pilot":
            study = StudyData(args.data_root, args.manifest, plan)
        pilot = _phase_report(args, plan, "pilot", None, study=study)
        if args.stage == "pilot":
            print(json.dumps(pilot, ensure_ascii=False))
            return
        if not all(row["complete_strategy_units"] > 0 for row in pilot.values()):
            raise ValueError("resource pilot must complete a real strategy unit in each mix")
        previous = None
        for phase in PHASES:
            active = args.stage in (phase, "all")
            if phase == "forward":
                plan = _forward_plan(args, plan)
            if active and (study is None or phase == "forward"):
                study = None
                study = StudyData(args.data_root, args.manifest, plan, forward=phase == "forward")
            previous = _phase_report(args, plan, phase, previous, study=study if active else None, pilot=pilot)
            if args.stage == phase:
                break
        print(f"{args.stage}: sealed results in {args.run_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("plan", "pilot", "all", *PHASES))
    for name in ("data-root", "manifest", "build-dir", "run-dir"):
        parser.add_argument("--" + name, required=True, type=lambda v: Path(v).resolve())
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-units", type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6 or (args.max_units is not None and args.max_units < 1):
        parser.error("workers must be 1–6 and max-units must be positive")
    try:
        execute(args)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"R2 stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
