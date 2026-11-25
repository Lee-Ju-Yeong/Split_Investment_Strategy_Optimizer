"""본실행 허가(설계 v2.1 8.3·8.6). 단계 함수(실행·재개·조립)의 공용 진입점이 부른다.

실제 묶음의 실행·재개·조립은 같은 판본에서 (1) 합성 시험이 통과했다는 기록과 (2) 공개 기준점 대조 증거가
모두 확인될 때만 허가한다. 합성 실행은 증거가 필요 없지만 등록한 실제 묶음을 받지 않는다.
두 기록은 필수 항목과 자기 해시를 검사하므로 몇 칸만 손으로 적은 파일로는 통과하지 않는다.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import platform
import time
import unittest

from ..kodex_core.data import source_identity
from . import identity as identity_module
from .grid import GRID_PATH, setting_config
from .identity import Context, IdentityError, check_frozen, check_synthetic
from .sealing import captured_output, dump, file_digest, read_sealed, with_seal, write_once, writer_lock


SELFTEST_SCHEMA = "kodex-grid-selftest-v1"
BASELINE_SCHEMA = "kodex-grid-baseline-v2"
TEST_MODULES = ("tests.test_research_etf_kodex_grid", "tests.test_research_etf_kodex_core")
REQUIRED_ITEMS = ("head", "item_status", "start_orders", "base", "stress", "hold_metrics", "reentries",
                  "buys_closed_years", "matched_baseline", "verdict", "observation_end")
REQUIRED_CHECKS = ("verdict_equals_formal_only_expectation", "path_cost_checks_equal_formal_only_expectation",
                   "baseline_unit_passes_record_checks", "applied_config_equals_registered",
                   "strategy_ending_equals_core_plus_split", "split_share_equals_formal_arithmetic",
                   "same_capital_hold_equals_formal_arithmetic", "trade_counts_equal_ledger_events",
                   "reentry_counts_equal_formal_lists", "matched_rerun_equal")
CLEAN = {"failures": 0, "errors": 0, "skipped": 0, "unexpected_successes": 0}


class PermitError(IdentityError):
    """본실행 허가 증거가 없거나 이 판본·이 입력의 것이 아니다. 실제 묶음을 실행·조립하지 않는다."""


def test_file(module: str) -> Path:
    found = importlib.util.find_spec(module)
    if found is None or not found.origin:
        raise PermitError(f"test module {module} is not found")
    return Path(found.origin)


def count_tests(path) -> int:
    """시험 파일에 적힌 시험 메서드 수(실행하지 않고 센다)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return sum(isinstance(item, ast.FunctionDef) and item.name.startswith("test")
               for node in tree.body if isinstance(node, ast.ClassDef) for item in node.body)


def version() -> dict:
    """이 판본: 승인 소스, 탐색 소스, 격자 정의 파일, 시험 파일의 해시."""
    return {"approved_source_sha256": source_identity()["source_sha256"],
            "exploration_source_sha256": identity_module.exploration_source()["exploration_source_sha256"],
            "grid_file_sha256": file_digest(GRID_PATH),
            "test_files": {module: file_digest(test_file(module)) for module in TEST_MODULES}}


def _run_suites(folder: Path) -> tuple:
    """시험을 실제로 돌린다. 러너와 시험의 출력은 봉인 로그에만 남긴다."""
    suites = {module: unittest.defaultTestLoader.loadTestsFromName(module) for module in TEST_MODULES}
    with captured_output(folder / "sealed" / "output.log") as sink:
        result = unittest.TextTestRunner(stream=sink, verbosity=1).run(unittest.TestSuite(suites.values()))
    return suites, result


def _selftest_body(before: dict, suites, result) -> dict:
    counts = {module: suite.countTestCases() for module, suite in suites.items()}
    outcome = {"failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
               "unexpected_successes": len(result.unexpectedSuccesses)}
    passed = outcome == CLEAN and result.testsRun == sum(counts.values()) and version() == before
    return {"schema": SELFTEST_SCHEMA, "exploration": True, "formal_run": False, "version": before,
            "modules": counts, "tests_run": result.testsRun, **outcome, "passed": passed,
            "python": platform.python_version()}


def run_selftest(out) -> dict:
    """합성 시험 묶음을 돌리고 통과 여부를 기록한다. 이미 기록이 있으면 다시 돌리지 않고 그 기록을 검사한다."""
    folder = Path(out) / "selftest"
    with writer_lock(out):
        if (folder / "record.json").exists():
            return check_selftest(out)
        before, started = version(), time.perf_counter()
        suites, result = _run_suites(folder)
        body = {**_selftest_body(before, suites, result), "seconds": time.perf_counter() - started}
        write_once(folder / "record.json", dump(with_seal(body)))
    return check_selftest(out)


def _read(path: Path, schema: str) -> dict:
    if not path.is_file():
        raise PermitError(f"{path.parent.name}/{path.name} is missing; the main run is not started")
    try:
        return read_sealed(path, schema)
    except (ValueError, OSError) as error:
        raise PermitError(f"{path.parent.name}/{path.name} is not an intact {schema} record") from error


def _selftest_problems(record: dict) -> list:
    counts = {module: count_tests(test_file(module)) for module in TEST_MODULES}
    problems = [] if record.get("version") == version() else [
        "approved source, exploration source, grid file or test files differ from this version"]
    if record.get("modules") != counts or record.get("tests_run") != sum(counts.values()) or not sum(counts.values()):
        problems.append("recorded test counts differ from the test files")
    if {key: record.get(key) for key in CLEAN} != CLEAN or record.get("passed") is not True:
        problems.append("the recorded run did not pass cleanly")
    return problems


def check_selftest(out) -> dict:
    """같은 판본의 합성 시험 통과 기록(탐색 소스·시험 파일 해시, 시험 수, 결과)."""
    record = _read(Path(out) / "selftest" / "record.json", SELFTEST_SCHEMA)
    problems = _selftest_problems(record)
    if problems:
        raise PermitError("synthetic test record is not valid for this version: " + "; ".join(problems))
    return record


def expected_baseline(ctx: Context) -> dict:
    """기준점(28번)이 가져야 할 설정과 실제 적용 해시. 적용 해시는 설정 번호로 다시 만든다."""
    spec = ctx.spec
    setting = spec.setting(spec.baseline_number())
    return {**setting.describe(), "symbol": spec.symbol, "start": spec.primary_start.isoformat(), "mode": spec.mode,
            "applied_config_sha256": setting_config(ctx.base, spec, setting).applied_sha256()}


def _target_problems(ctx: Context, record: dict) -> list:
    """어느 판본·어느 입력·어느 정식 결과·어느 설정의 대조인지."""
    spec, expected, problems = ctx.spec, expected_baseline(ctx), []
    if (record.get("exploration"), record.get("formal_run")) != (True, False):
        problems.append("not an exploration record")
    if record.get("identity") != json.loads(dump(ctx.identity)):
        problems.append("identity (approved source, exploration source, grid, inputs) differs from this run")
    if (record.get("formal_result") or {}).get("sha256") != spec.approved["formal_result_sha256"]:
        problems.append("formal result hash is not the registered one")
    if record.get("baseline") != expected or expected["applied_config_sha256"] != spec.approved[
            "baseline_applied_config_sha256"]:
        problems.append("baseline setting or its applied configuration hash is not the registered one")
    return problems


def _comparison_problems(comparison: dict) -> list:
    """필수 대조 항목이 모두 있고, 불일치 0건이며, 일치한 잎이 정식 기록의 잎 전부인지."""
    by_item, problems = comparison.get("by_item") or {}, []
    if comparison.get("mismatches") != []:
        problems.append("comparison has mismatches or no mismatch list")
    if any(type(by_item.get(name)) is not int or by_item[name] <= 0 for name in REQUIRED_ITEMS):
        problems.append("a required comparison item is missing")
    totals = {comparison.get("matched"), comparison.get("formal_leaves"), comparison.get("exploration_leaves"),
              sum(value for value in by_item.values() if type(value) is int)}
    if len(totals) != 1 or not comparison.get("matched"):
        problems.append("matched leaves do not cover the whole formal record")
    return problems


def _evidence_problems(out, record: dict) -> list:
    """필수 검산이 모두 참인지, 이 판본의 시험 통과 기록 뒤에 잰 것인지, 봉인한 단위가 그대로인지."""
    checks, problems = record.get("self_checks") or {}, []
    if any(checks.get(name) is not True for name in REQUIRED_CHECKS):
        problems.append("a required self-check is missing or not true")
    if record.get("selftest_seal") != check_selftest(out)["seal"]:
        problems.append("baseline was not taken after this version's synthetic test record")
    sealed = Path(out) / "baseline" / "sealed" / "unit.json"
    if not sealed.is_file() or record.get("sealed_unit_sha256") != file_digest(sealed):
        problems.append("sealed baseline unit is missing or changed")
    if record.get("main_run_allowed") is not True:
        problems.append("the comparison did not allow the main run")
    return problems


def check_baseline(ctx: Context, out) -> dict:
    """공개 기준점 대조 증거(정식 결과 해시, 28번의 실제 적용 해시, 필수 대조 항목, 불일치 0건, 필수 검산)."""
    record = _read(Path(out) / "baseline" / "compare.json", BASELINE_SCHEMA)
    problems = (_target_problems(ctx, record) + _comparison_problems(record.get("comparison") or {})
                + _evidence_problems(out, record))
    if problems:
        raise PermitError("baseline comparison does not permit the main run: " + "; ".join(problems))
    return record


def check_current(ctx: Context) -> None:
    """문맥의 코드 식별값이 지금 코드·격자 정의·설정으로 다시 만든 값과 같은지."""
    current = identity_module.code_identity(ctx.spec, ctx.base)
    if any(ctx.identity.get(key) != value for key, value in current.items()):
        raise PermitError("the context identity does not describe the current code and definitions")


def enter(ctx: Context, out) -> None:
    """공용 진입점. 실제 묶음이면 고정 정의·시험 통과 기록·기준점 대조 증거를 모두 확인하고,
    합성 입력이면 실제 묶음이 아님을 확인한다."""
    check_current(ctx)
    if not ctx.registered:
        check_synthetic(ctx)
        return
    check_frozen(ctx)
    check_selftest(out)
    check_baseline(ctx, out)
