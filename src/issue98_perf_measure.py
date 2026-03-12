"""
issue98_perf_measure.py

Issue #98 canonical throughput measurement runner.

Recommended usage:
  CONDA_NO_PLUGINS=true conda run -n rapids-env \
    python -m src.issue98_perf_measure --label pr98c_slice2_janfeb2024_cov020
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

import yaml

# BOOTSTRAP: allow direct execution (`python src/issue98_perf_measure.py`)
# while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

DEFAULT_CANONICAL_PROFILE = "issue98_janfeb2024_multibatch_research020"
DEFAULT_CONFIG_PATH = "config/config.issue98_perf_multibatch_janfeb_2024_research020.yaml"
DEFAULT_OUTDIR_ROOT = "results/issue98_measure"
DEFAULT_GPU_SAMPLE_INTERVAL_SEC = 5
DEFAULT_RUN_COUNT = 2


def _parse_wall_to_seconds(value: str):
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def _module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - defensive only
        return f"unavailable ({exc.__class__.__name__})"
    return str(getattr(module, "__version__", "unknown"))


def build_env_snapshot_lines(
    *,
    timestamp: str,
    label: str,
    config_path: Path,
    git_head: str,
    git_branch: str,
    python_version: str,
    cupy_version: str,
    cudf_version: str,
    gpu_info: str,
) -> list[str]:
    return [
        f"timestamp={timestamp}",
        f"label={label}",
        f"config_path={config_path}",
        f"git_head={git_head}",
        f"git_branch={git_branch}",
        f"Python {python_version}",
        "",
        f"cupy {cupy_version}",
        f"cudf {cudf_version}",
        "",
        gpu_info,
        f"{_sha256_file(config_path)}  {config_path}",
    ]


def collect_env_snapshot(*, timestamp: str, label: str, config_path: Path) -> str:
    lines = build_env_snapshot_lines(
        timestamp=timestamp,
        label=label,
        config_path=config_path,
        git_head=_safe_command_output(["git", "rev-parse", "HEAD"]),
        git_branch=_safe_command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        python_version=sys.version.split()[0],
        cupy_version=_module_version("cupy"),
        cudf_version=_module_version("cudf"),
        gpu_info=_safe_command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
    )
    return "\n".join(lines) + "\n"


def build_input_snapshot(
    *,
    config_path: Path,
    label: str,
    timestamp: str,
    canonical_profile: str,
    run_count: int,
    gpu_sample_interval_sec: int,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    backtest = dict(config.get("backtest_settings", {}))
    strategy = dict(config.get("strategy_params", {}))
    return {
        "label": label,
        "timestamp": timestamp,
        "canonical_profile": canonical_profile,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "start_date": backtest.get("start_date"),
        "end_date": backtest.get("end_date"),
        "initial_cash": backtest.get("initial_cash"),
        "simulation_batch_size": backtest.get("simulation_batch_size"),
        "candidate_source_mode": strategy.get("candidate_source_mode"),
        "tier_hysteresis_mode": strategy.get("tier_hysteresis_mode"),
        "price_basis": strategy.get("price_basis"),
        "min_tier12_coverage_ratio": strategy.get("min_tier12_coverage_ratio"),
        "measurement_run_count": int(run_count),
        "gpu_sample_interval_sec": int(gpu_sample_interval_sec),
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _spawn_gpu_sampler(csv_path: Path, interval_sec: int):
    handle = csv_path.open("w", encoding="utf-8")
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used",
        "--format=csv,noheader",
        "-l",
        str(interval_sec),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        handle.close()
        csv_path.touch()
        return None, None
    return process, handle


def _stop_process(process, handle) -> None:
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive only
            process.kill()
            process.wait(timeout=5)
    if handle is not None:
        handle.close()


def _stream_output(process: subprocess.Popen, log_handle) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        sys.stdout.write(line)
        log_handle.write(line)
    sys.stdout.flush()
    log_handle.flush()


def run_single_measurement(
    *,
    run_tag: str,
    outdir: Path,
    config_path: Path,
    gpu_sample_interval_sec: int,
) -> int:
    log_path = outdir / f"{run_tag}.log"
    csv_path = outdir / f"{run_tag}.gpu.csv"
    sampler, sampler_handle = _spawn_gpu_sampler(csv_path, gpu_sample_interval_sec)
    env = os.environ.copy()
    env["MAGICSPLIT_CONFIG_PATH"] = str(config_path)
    env["CONDA_NO_PLUGINS"] = "true"
    command = ["/usr/bin/time", "-v", sys.executable, "-m", "src.parameter_simulation_gpu"]

    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            _stream_output(process, log_handle)
            return process.wait()
    finally:
        _stop_process(sampler, sampler_handle)


def parse_run_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    kernel_match = re.search(r"Total GPU Kernel Execution Time: ([0-9.]+)s", text)
    wall_match = re.search(r"Elapsed \(wall clock\) time .*: (.+)", text)
    exit_match = re.search(r"Exit status: ([0-9]+)", text)
    return {
        "path": str(path),
        "kernel_s": float(kernel_match.group(1)) if kernel_match else None,
        "wall_clock": wall_match.group(1).strip() if wall_match else None,
        "wall_clock_s": _parse_wall_to_seconds(wall_match.group(1)) if wall_match else None,
        "batch_count": len(re.findall(r"--- Running Batch ", text)),
        "oom_retry": "[GPU_WARNING] OOM" in text,
        "exit_code": int(exit_match.group(1)) if exit_match else None,
    }


def parse_gpu_csv(path: Path) -> dict:
    if not path.exists():
        return {"samples": 0, "gpu_util_median": None, "gpu_mem_used_max_mib": None}

    utils = []
    mem_used = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            utils.append(int(parts[1].replace("%", "").strip()))
            mem_used.append(int(parts[3].replace("MiB", "").replace(" ", "").strip()))
        except ValueError:
            continue
    return {
        "samples": len(utils),
        "gpu_util_median": statistics.median(utils) if utils else None,
        "gpu_mem_used_max_mib": max(mem_used) if mem_used else None,
    }


def build_summary(*, outdir: Path, canonical_profile: str, research_only: bool) -> dict:
    run_tags = sorted(
        path.stem
        for path in outdir.glob("run*.log")
        if path.stem.startswith("run")
    )
    summary = {
        "canonical_profile": canonical_profile,
        "research_only": bool(research_only),
    }
    kernel_values = []
    wall_values = []

    for run_tag in run_tags:
        run_summary = parse_run_log(outdir / f"{run_tag}.log")
        run_summary.update(parse_gpu_csv(outdir / f"{run_tag}.gpu.csv"))
        summary[run_tag] = run_summary
        if run_summary["kernel_s"] is not None:
            kernel_values.append(run_summary["kernel_s"])
        if run_summary["wall_clock_s"] is not None:
            wall_values.append(run_summary["wall_clock_s"])

    summary["median_kernel_s"] = statistics.median(kernel_values) if kernel_values else None
    summary["median_wall_s"] = statistics.median(wall_values) if wall_values else None
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Issue #98 canonical throughput measurement runner.")
    parser.add_argument("--label", required=True, help="Measurement label, e.g. pr98b2_slice2a_janfeb2024_cov020")
    parser.add_argument(
        "--config-path",
        default=os.environ.get("MAGICSPLIT_CONFIG_PATH", DEFAULT_CONFIG_PATH),
        help="Config path. Defaults to $MAGICSPLIT_CONFIG_PATH or the Issue #98 canonical config.",
    )
    parser.add_argument(
        "--canonical-profile",
        default=DEFAULT_CANONICAL_PROFILE,
        help="Canonical profile identifier stored into summary.json.",
    )
    parser.add_argument(
        "--outdir-root",
        default=DEFAULT_OUTDIR_ROOT,
        help="Root directory for measurement artifacts.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUN_COUNT,
        help="Number of measurement runs. Default=2.",
    )
    parser.add_argument(
        "--gpu-sample-interval-sec",
        type=int,
        default=DEFAULT_GPU_SAMPLE_INTERVAL_SEC,
        help="nvidia-smi sampling interval in seconds. Default=5.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    if int(args.runs) <= 0:
        raise ValueError("--runs must be >= 1")
    if int(args.gpu_sample_interval_sec) <= 0:
        raise ValueError("--gpu-sample-interval-sec must be >= 1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir_root) / f"{args.label}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=False)

    env_snapshot = collect_env_snapshot(
        timestamp=timestamp,
        label=args.label,
        config_path=config_path,
    )
    (outdir / "env.txt").write_text(env_snapshot, encoding="utf-8")

    input_snapshot = build_input_snapshot(
        config_path=config_path,
        label=args.label,
        timestamp=timestamp,
        canonical_profile=args.canonical_profile,
        run_count=int(args.runs),
        gpu_sample_interval_sec=int(args.gpu_sample_interval_sec),
    )
    write_json(outdir / "input_snapshot.json", input_snapshot)

    print(f"[issue98_perf_measure] outdir={outdir}")
    run_exit_codes = []
    for index in range(1, int(args.runs) + 1):
        run_tag = f"run{index}"
        print(f"[issue98_perf_measure] starting {run_tag}")
        exit_code = run_single_measurement(
            run_tag=run_tag,
            outdir=outdir,
            config_path=config_path,
            gpu_sample_interval_sec=int(args.gpu_sample_interval_sec),
        )
        run_exit_codes.append(exit_code)
        print(f"[issue98_perf_measure] finished {run_tag} exit_code={exit_code}")
        if exit_code != 0:
            break

    summary = build_summary(
        outdir=outdir,
        canonical_profile=args.canonical_profile,
        research_only=True,
    )
    write_json(outdir / "summary.json", summary)
    print(f"[issue98_perf_measure] saved summary path={outdir / 'summary.json'}")

    return 0 if run_exit_codes and all(code == 0 for code in run_exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
