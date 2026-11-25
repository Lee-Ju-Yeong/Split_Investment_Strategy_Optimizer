"""본실행 허가(설계 v1.1 7절). run·assemble의 공용 진입점이 부른다.

실제 묶음의 실행·재개·조립은 같은 판본에서 (1) 합성 시험이 통과했다는 기록과 (2) 공개값 재현 증거(기준점 802항목과
229200·169950의 공개값, 불일치 0, 필수 검산 전부 참)가 모두 확인될 때만 허가한다. 합성 실행은 증거가 필요 없지만
등록한 실제 묶음을 받지 않는다. 두 기록은 필수 항목과 자기 해시를 검사한다.
"""
from __future__ import annotations

import json
from pathlib import Path
import platform
import time
import unittest

from ..kodex_core.data import source_identity
from ..kodex_grid import identity as grid_identity
from ..kodex_grid.grid import GRID_PATH
from ..kodex_grid.permit import CLEAN, PermitError, count_tests, test_file
from ..kodex_grid.sealing import captured_output, dump, file_digest, read_sealed, with_seal, write_once, writer_lock
from . import identity as identity_module
from .definition import DEFINITION_PATH
from .identity import MixContext, check_current, check_frozen, check_synthetic


SELFTEST_SCHEMA = "markets-mix-selftest-v1"
BASELINE_SCHEMA = "markets-mix-baseline-v1"
TEST_MODULES = ("tests.test_research_etf_markets_mix", "tests.test_research_etf_kodex_grid",
                "tests.test_research_etf_kodex_core")
REQUIRED_ITEMS = ("head", "item_status", "start_orders", "base", "stress", "hold_metrics", "reentries",
                  "buys_closed_years", "matched_baseline", "verdict", "observation_end")
REQUIRED_CHECKS = ("applied_config_equals_registered", "unit_values_equal_formal", "unit_reasons_equal_formal",
                   "unit_hold_equals_formal", "unit_start_orders_equal_formal", "unit_counts_equal_formal",
                   "trade_counts_equal_ledger_events", "series_reproduce_metrics", "one_share_mix_equals_solo",
                   "unit_passes_record_checks")


FOREIGN = ("stage1", "stage2")  # kodex_grid 결과 폴더의 단계 이름


def check_folder(out) -> None:
    """이 조사의 출력 폴더는 kodex_grid의 결과 폴더와 섞지 않는다(그 폴더에는 아무것도 쓰지 않는다)."""
    found = [name for name in FOREIGN if (Path(out) / name).exists()]
    if found:
        raise PermitError(f"{Path(out).name} holds kodex_grid results {found}; use a separate output folder")


def version() -> dict:
    """이 판본: 승인 소스, kodex_grid 소스, 이 조사의 소스, 격자·정의 파일, 시험 파일의 해시."""
    return {"approved_source_sha256": source_identity()["source_sha256"],
            "kodex_grid_source_sha256": grid_identity.exploration_source()["exploration_source_sha256"],
            "mix_source_sha256": identity_module.mix_source()["mix_source_sha256"],
            "grid_file_sha256": file_digest(GRID_PATH), "definition_file_sha256": file_digest(DEFINITION_PATH),
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
    check_folder(out)
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
        "approved source, kodex_grid source, this package, definitions or test files differ from this version"]
    if record.get("modules") != counts or record.get("tests_run") != sum(counts.values()) or not sum(counts.values()):
        problems.append("recorded test counts differ from the test files")
    if {key: record.get(key) for key in CLEAN} != CLEAN or record.get("passed") is not True:
        problems.append("the recorded run did not pass cleanly")
    return problems


def check_selftest(out) -> dict:
    """같은 판본의 합성 시험 통과 기록."""
    record = _read(Path(out) / "selftest" / "record.json", SELFTEST_SCHEMA)
    problems = _selftest_problems(record)
    if problems:
        raise PermitError("synthetic test record is not valid for this version: " + "; ".join(problems))
    return record


def expected_points(ctx: MixContext) -> list:
    """재현해야 하는 공개값: 기준점(주 상품의 정식 주 시작일)과 확장 두 상품. 모두 민감도 설정(28번)이다."""
    primary, extension = ctx.mix.baseline["primary"], ctx.mix.baseline["extension"]
    return [f"{symbol}@{start.isoformat()}" for symbol, start in (primary, *extension)]


def comparison_problems(comparison: dict) -> list:
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


def _registered_problems(ctx: MixContext, name, point: dict) -> list:
    """그 점의 등록값: 정식 기록의 잎 수(069500@2011 802, 169950@2017 712, 229200@2017 596), 28번, 두 비용의 봉인 단위."""
    problems, leaves = [], ctx.mix.baseline["leaves"].get(name)
    if leaves is None or (point.get("comparison") or {}).get("matched") != leaves:
        problems.append(f"matched leaves are not the registered {leaves}")
    if point.get("number") != ctx.mix.sensitivity["number"]:
        problems.append("the reproduced setting is not the registered number")
    if set(point.get("sealed_units") or {}) != set(ctx.mix.costs["one_way"]):
        problems.append("sealed baseline units of both costs are required")
    return problems


def point_problems(ctx: MixContext, out, point: dict) -> list:
    """한 재현 점: 등록값, 대조, 필수 검산, 봉인한 단위가 그대로인지."""
    name = f"{point.get('symbol')}@{point.get('start')}"
    problems = _registered_problems(ctx, name, point) + comparison_problems(point.get("comparison") or {})
    checks = point.get("self_checks") or {}
    if any(checks.get(check) is not True for check in REQUIRED_CHECKS):
        problems.append("a required self-check is missing or not true")
    for cost, sealed in (point.get("sealed_units") or {"": None}).items():
        path = Path(out) / "baseline" / "sealed" / f"{name}-{cost}.unit.json"
        if not path.is_file() or sealed != file_digest(path):
            problems.append(f"sealed baseline unit {cost} is missing or changed")
    return [f"{name}: {problem}" for problem in problems]


def _target_problems(ctx: MixContext, out, record: dict) -> list:
    """어느 판본·어느 입력·어느 정식 결과의 재현인지, 그리고 이 판본의 시험 통과 기록 뒤에 잰 것인지."""
    problems = []
    if (record.get("exploration"), record.get("formal_run")) != (True, False):
        problems.append("not an exploration record")
    if record.get("identity") != json.loads(dump(ctx.identity)):
        problems.append("identity (sources, definitions, inputs) differs from this run")
    if (record.get("formal_result") or {}).get("sha256") != ctx.mix.approved["formal_result_sha256"]:
        problems.append("formal result hash is not the registered one")
    if record.get("selftest_seal") != check_selftest(out)["seal"]:
        problems.append("baseline was not taken after this version's synthetic test record")
    if record.get("main_run_allowed") is not True:
        problems.append("the reproduction did not allow the main run")
    return problems


def check_baseline(ctx: MixContext, out) -> dict:
    """공개값 재현 증거(정식 결과 해시, 세 재현 점의 대조와 필수 검산)."""
    record = _read(Path(out) / "baseline" / "compare.json", BASELINE_SCHEMA)
    points = record.get("points") or {}
    problems = _target_problems(ctx, out, record)
    if sorted(points) != sorted(expected_points(ctx)):
        problems.append("the reproduced points are not the registered ones")
    for name, point in points.items():
        stated = f"{point.get('symbol')}@{point.get('start')}"
        problems += [f"{name}: point is filed under another name"] if stated != name else []
        problems += point_problems(ctx, out, point)
    if problems:
        raise PermitError("baseline reproduction does not permit the main run: " + "; ".join(problems))
    return record


def enter(ctx: MixContext, out) -> None:
    """공용 진입점. 실제 묶음이면 고정 정의·시험 통과 기록·재현 증거를 모두 확인하고, 합성 입력이면 실제 묶음이 아님을
    확인한다."""
    check_folder(out)
    check_current(ctx)
    if not ctx.registered:
        check_synthetic(ctx)
        return
    check_frozen(ctx)
    check_selftest(out)
    check_baseline(ctx, out)
    check_eligibility(ctx, out)


ELIGIBILITY_SCHEMA = "markets-mix-eligibility-v2"


def check_eligibility(ctx: MixContext, out) -> dict:
    """같은 판본·입력의 자격 검산 기록이 있고, 묶음 시작일 규칙과 민감도 시작일(+5·+20)이 모두 성립했는지."""
    record = _read(Path(out) / "eligibility" / "record.json", ELIGIBILITY_SCHEMA)
    rules, problems = record.get("start_rules"), []
    if record.get("identity") != json.loads(dump(ctx.identity)):
        problems.append("identity (sources, definitions, inputs) differs from this run")
    declared = [cohort.id for cohort in ctx.mix.cohorts if isinstance(cohort.start_rule, dict)]
    if not isinstance(rules, list) or [row.get("cohort") for row in rules] != declared:
        problems.append("start rules of the declared cohorts are missing")
    elif any(row.get("holds") is not True for row in rules):
        problems.append("a cohort start does not satisfy its start rule")
    if record.get("sensitivity_starts_are_offsets") is not True:
        problems.append("sensitivity starts are not the registered session offsets")
    if problems:
        raise PermitError("eligibility record does not permit the main run: " + "; ".join(problems))
    return record
