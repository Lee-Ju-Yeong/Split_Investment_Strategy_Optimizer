"""단위 실행·봉인·재개·조립(설계 v1.1 7·8절).

- run과 assemble은 같은 진입점을 거친다: 본실행 허가 확인 → 출력 폴더의 작성자 잠금. 실제 묶음은 같은 판본의 합성 시험
  통과 기록과 공개값 재현 증거가 있어야 들어온다. 입력 적재·단위 계산·봉인·열람은 모두 가장 안쪽 경계 Plan.guard를 거친다.
- 단위 결과는 한 번만 쓰고, 그 바이트의 해시를 실행 당시의 별도 봉인 기록으로 남긴다. 재개와 조립은 같은 검사를 쓴다.
- 조립은 계획한 단위가 모두 봉인됐을 때만 연다. 그 전에 공개되는 것은 단위 이름·소요시간·기술 상태 코드뿐이다.
- 최초 조립이 단위별 내용 해시·실행 식별값·산출물 해시를 고정하고, 이후 조립은 그 기록과 같을 때만 통과한다.
- 결과 폴더는 --out 아래의 run/ 하나다(kodex_grid의 stage1/·stage2/와 이름도 기록 꼴도 다르다).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import json
import multiprocessing
import re
from pathlib import Path
import time
import traceback

from ..kodex_core import data as loader
from ..kodex_core import study
from ..kodex_core.inputs import DataHold
from ..kodex_core.metrics import plain
from ..kodex_grid.identity import IdentityError
from ..kodex_grid.progress import DONE, SKIPPED, Progress
from ..kodex_grid.records import AssemblyError
from ..kodex_grid.runner import Bench, LedgerCache
from ..kodex_grid.sealing import (SEALED, OverwriteError, bytes_digest, captured_output, clear_temporaries, digest,
                                  dump, error_code, flush_standard, listed, read_sealed, with_seal, write_marker,
                                  write_once, writer_lock, writer_token)
from ..kodex_grid.stages import _timing
from . import permit, report, summary
from .definition import Unit, plan_units
from .identity import MixContext, check_loaded
from .permit import PermitError
from .records import STAGE, UNIT_SCHEMA, Seat, check_record
from .rows import Table, reproduced
from .unit import held_unit, run_unit


SEAL_SCHEMA = "markets-mix-seal-v1"
HEX64 = re.compile(r"[0-9a-f]{64}")  # 봉인한 본문 해시의 꼴(SHA-256 16진수)
LOCK_SCHEMA = "markets-mix-lock-v1"
RESULT_SCHEMA = "markets-mix-result-v1"
LOCK_FILE = "lock.json"


class RunError(RuntimeError):
    """단위 계산의 기술 오류. 같은 입력으로 복구해 다시 실행한 뒤에만 조립한다."""


def compact(record) -> str:
    """단위 파일의 글(일별 계열이 길어 줄바꿈 없이 쓴다). 봉인 해시는 이 바이트로 약속한다."""
    return json.dumps(plain(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Plan:
    """작업 묶음: 계획한 단위, 봉인 폴더, 적재한 입력(또는 보류 사유), 보유·코어 장부 캐시."""

    def __init__(self, ctx: MixContext, out, units=None):
        self.ctx, self.out, self.folder = ctx, Path(out), Path(out) / STAGE
        self.units = tuple(plan_units(ctx.mix, ctx.spec) if units is None else units)
        self.inputs, self.cache, self.admitted = {}, LedgerCache(), None

    def guard(self) -> None:
        """가장 안쪽 공용 경계: 이 출력 폴더의 본실행 허가와, 이 호출 흐름(또는 그것이 포크한 작업 자식)의 작성자 잠금.
        같은 잠금 구간 안에서는 한 번 확인한 것으로 충분하다."""
        token = writer_token(self.out)
        if token is not None and token == self.admitted:
            return
        permit.enter(self.ctx, self.out)
        if token is None:
            raise PermitError(f"the writer lock of {self.out} is not held by this call flow")
        self.admitted = token

    def seat(self, unit: Unit) -> Seat:
        return Seat(self.ctx, unit)

    def path(self, unit: Unit) -> Path:
        return self.folder / "units" / f"{unit.name}.json"

    def seal_path(self, unit: Unit) -> Path:
        return self.folder / "sealed" / "seals" / f"{unit.name}.json"

    def log_path(self, unit: Unit) -> Path:
        return self.folder / "sealed" / "logs" / f"{unit.name}.log"

    def error_path(self, unit: Unit) -> Path:
        return self.folder / "errors" / f"{unit.name}.error.txt"

    def _load(self, key):
        """한 상품·시작일 입력. 예상한 자료 문제만 보류 사유(문자열)로 남기고, 합성 실행에 실제 묶음이 섞이면 멈춘다."""
        try:
            inputs = self.ctx.source.load(*key)
        except DataHold as error:
            return str(error)
        check_loaded(self.ctx, inputs)
        return inputs

    def load_inputs(self) -> None:
        """단위들이 쓰는 (상품, 시작일) 입력을 한 번씩 읽는다."""
        self.guard()
        loader._source.cache_clear()
        try:
            for key in sorted({(unit.symbol, unit.start) for unit in self.units} - set(self.inputs)):
                self.inputs[key] = self._load(key)
        finally:
            loader._source.cache_clear()

    def compute(self, unit: Unit, bench: Bench) -> dict:
        self.guard()
        loaded, seat = self.inputs[(unit.symbol, unit.start)], self.seat(unit)
        if isinstance(loaded, str):
            result = held_unit(seat.config(), unit.cost, loaded)
        else:
            result = run_unit(loaded, seat.config(), unit.start, cost=unit.cost, nominal_floor=seat.floor(), bench=bench)
        return {"schema": UNIT_SCHEMA, "exploration": True, "formal_run": False, "identity": seat.identity(),
                "setting": seat.setting(), "result": result}

    def seal(self, unit: Unit, started) -> None:
        """단위를 계산해 한 번만 쓰고, 쓴 바이트의 해시와 실행 시간을 별도 봉인 기록으로 남긴다."""
        self.guard()
        bench = Bench(cache=self.cache)
        text = compact(self.compute(unit, bench))
        write_once(self.path(unit), text, mode=SEALED)
        seal = {"schema": SEAL_SCHEMA, "unit": unit.name, "sha256": digest(text), "timing": _timing(bench, started)}
        write_once(self.seal_path(unit), dump(with_seal(seal)), mode=SEALED)

    def sealed_hash(self, unit: Unit) -> str:
        """실행 당시 봉인한 SHA-256. 봉인 기록만 읽고 단위 본문은 열지 않는다."""
        self.guard()
        try:
            seal = read_sealed(self.seal_path(unit), SEAL_SCHEMA)
        except (OSError, ValueError) as error:
            raise AssemblyError(f"{unit.name}: seal record is missing or damaged") from error
        if seal.get("unit") != unit.name or not isinstance(seal.get("sha256"), str) or not HEX64.fullmatch(seal["sha256"]):
            raise AssemblyError(f"{unit.name}: seal record names another unit or has no hash")
        return seal["sha256"]

    def opened(self, unit: Unit, sealed: str) -> dict:
        """단위 본문을 읽어 봉인 해시와 맞춘 뒤에만 해석하고 기록 검사를 한다."""
        self.guard()
        raw = self.path(unit).read_bytes()
        if bytes_digest(raw) != sealed:
            raise AssemblyError(f"{unit.name}: content differs from the hash sealed when it was computed")
        record = json.loads(raw)
        check_record(self.seat(unit), record)
        return record

    def verified(self, unit: Unit) -> tuple:
        """봉인된 단위 하나의 (기록, 바이트 해시)."""
        sealed = self.sealed_hash(unit)
        return self.opened(unit, sealed), sealed


_ACTIVE = {}  # 포크한 자식 프로세스가 그대로 물려받는 현재 작업 묶음(적재한 입력 포함)


def _attempt(plan: Plan, unit: Unit, started) -> str:
    """단위 하나를 계산해 봉인한다. 작업 내부의 출력과 예외 전문은 봉인 로그에만 남기고 기술 코드만 돌려준다."""
    plan.error_path(unit).unlink(missing_ok=True)
    with captured_output(plan.log_path(unit)) as sink:
        try:
            plan.seal(unit, started)
            return DONE
        except Exception as error:  # 기술 오류: 표시가 남아 있는 동안 조립을 막는다
            traceback.print_exc(file=sink)
            code = error_code(error)
    write_marker(plan.error_path(unit), f"{unit.name} {code}\n")
    return code


def _work(unit: Unit) -> tuple:
    plan, started = _ACTIVE["plan"], time.perf_counter()
    code = _attempt(plan, unit, started)
    return unit.name, code, time.perf_counter() - started


def _dispatch(pending, workers):
    """단위 병렬 실행. 포크한 작업 프로세스가 죽으면 멈춘 채 기다리지 않고 오류로 끝난다."""
    if workers <= 1 or len(pending) <= 1:
        yield from map(_work, pending)
        return
    flush_standard()
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(min(workers, len(pending)), mp_context=context) as pool:
        for future in as_completed([pool.submit(_work, unit) for unit in pending]):
            yield future.result()


def _reusable(plan: Plan, unit: Unit) -> bool:
    """재개 때 이미 봉인된 단위인지. 봉인된 단위는 조립과 같은 검사를 통과해야 다시 쓴다."""
    has_unit, has_seal = plan.path(unit).exists(), plan.seal_path(unit).exists()
    if has_seal and not has_unit:
        raise IdentityError(f"{unit.name}: sealed unit file is missing; it is not recomputed silently")
    if not has_seal:
        plan.path(unit).unlink(missing_ok=True)  # 봉인까지 가지 못한 쓰기는 결과가 아니다
        return False
    try:
        plan.verified(unit)
    except AssemblyError as error:
        raise IdentityError(f"sealed unit is not reused: {error}") from error
    plan.error_path(unit).unlink(missing_ok=True)
    return True


def _pending(plan: Plan, log) -> list:
    pending = []
    for unit in plan.units:
        if _reusable(plan, unit):
            log(unit.name, SKIPPED, 0.0)
        else:
            pending.append(unit)
    return pending


def _run_pending(plan: Plan, pending, *, workers, log) -> list:
    _ACTIVE["plan"] = plan
    try:
        failed = []
        for name, code, seconds in _dispatch(pending, workers):
            log(name, code, seconds)
            failed += [name] if code != DONE else []
        return failed
    finally:
        _ACTIVE.clear()


def _check_starts(plan: Plan) -> None:
    """묶음마다 민감도 시작일이 주 시작일의 +5·+20 국내 거래일인지 적재한 달력으로 확인한다."""
    mix = plan.ctx.mix
    calendars = [item.calendar for item in plan.inputs.values() if not isinstance(item, str)]
    for cohort in mix.cohorts if calendars else ():
        study.check_starts(calendars[0], cohort.starts(), (0, *mix.sensitivity["offsets"]))


def run_units(plan: Plan, workers: int, log) -> None:
    """끝나지 않은 단위만 병렬 실행하고 봉인한다. 허가와 잠금을 확인하기 전에는 입력을 읽지도, 임시 파일을 정리하지도,
    단위를 쓰지도 않는다."""
    plan.guard()
    clear_temporaries(plan.folder)
    pending = _pending(plan, log)
    if pending:
        plan.load_inputs()
        _check_starts(plan)
    failed = _run_pending(plan, pending, workers=workers, log=log)
    if failed:
        raise RunError(f"{len(failed)} unit(s) ended with a technical error: {sorted(failed)}")


def _check_folder(plan: Plan) -> None:
    """계획한 단위 파일만 있어야 한다. 누락·여분·기술 오류 표시가 있으면 조립하지 않는다."""
    expected, present = {plan.path(unit).name for unit in plan.units}, listed(plan.folder / "units")
    if len(expected) != len(plan.units):
        raise AssemblyError("duplicate units in the plan")
    problems = ([f"missing {name}" for name in sorted(expected - present)]
                + [f"unexpected {name}" for name in sorted(present - expected)]
                + [f"error marker {name}" for name in sorted(listed(plan.folder / "errors"))])
    if problems:
        raise AssemblyError("; ".join(problems))


def read_units(plan: Plan) -> tuple:
    """봉인된 단위를 모두 연다: ({단위: 기록}, {단위 이름: 바이트 해시}). 허가와 잠금을 확인한 뒤에만 연다."""
    plan.guard()
    _check_folder(plan)
    seals = {unit: plan.sealed_hash(unit) for unit in plan.units}  # 모든 봉인을 먼저 확인한 뒤에만 본문을 연다
    opened = {unit: (plan.opened(unit, sealed), sealed) for unit, sealed in seals.items()}
    records = {unit: pair[0] for unit, pair in opened.items()}
    broken = [unit.name for unit, record in records.items() if not reproduced(plan.ctx, record)]
    if broken:
        raise AssemblyError(f"sealed daily series do not reproduce the sealed metrics: {broken}")
    return records, {unit.name: pair[1] for unit, pair in opened.items()}


@contextmanager
def _entered(ctx: MixContext, out):
    """run과 assemble의 공용 진입점: 본실행 허가를 확인한 뒤 출력 폴더의 작성자 잠금을 잡는다."""
    permit.enter(ctx, out)
    with writer_lock(out):
        yield


def run(ctx: MixContext, out, *, workers=1, echo=print) -> None:
    with _entered(ctx, out):
        plan = Plan(ctx, out)
        run_units(plan, workers, Progress(plan.folder, STAGE, echo))


def _cohort_frame(mix, cohort) -> dict:
    return {"id": cohort.id, "start": cohort.start.isoformat(), "solo": list(cohort.solo),
            "mixes": [[row.id, list(row.legs)] for row in cohort.mixes], "periods": [row.label for row in cohort.periods],
            "slots": [[start.isoformat(), cost] for start, cost in mix.slots(cohort)]}


def _envelope(ctx: MixContext) -> dict:
    """결과의 머리: 탐색 표시, 실행 식별값, 표의 틀."""
    mix = ctx.mix
    frame = {"paths": list(ctx.base.paths), "products": [list(row) for row in mix.products],
             "cohorts": [_cohort_frame(mix, cohort) for cohort in mix.cohorts], "thresholds": mix.thresholds,
             "settings": [row.describe() for row in ctx.spec.settings()]}
    return {"schema": RESULT_SCHEMA, "exploration": True, "formal_run": False, "identity": ctx.identity,
            "frame": frame}


def _result(plan: Plan) -> dict:
    """봉인 단위에서 결과를 만든다(파일을 쓰지 않는다). 단위별 내용 해시도 결과에 싣는다."""
    ctx = plan.ctx
    records, hashes = read_units(plan)
    table = Table(ctx, records)
    rows = summary.primary_rows(ctx, table)
    result = {**_envelope(ctx), "conclusion": summary.conclusion(ctx, rows),
              "subjects": summary.subject_views(ctx, rows), "rows": rows,
              "sensitivity": summary.sensitivity_views(ctx, table), "units": hashes}
    return result


def _first_record(plan: Plan, result, texts) -> dict:
    """최초 조립 때 한 번 고정하는 기록: 실행 식별값, 단위별 내용 해시, 결론의 종류, 산출물 해시."""
    chosen = {key: result["conclusion"][key] for key in ("kind", "rows", "counts")}
    return with_seal({"schema": LOCK_SCHEMA, "exploration": True, "formal_run": False, "stage": STAGE,
                      "identity": plan.ctx.identity, "units": result["units"], "chosen": chosen,
                      "outputs": {name: digest(text) for name, text in texts.items()}})


def _publish(folder: Path, record, texts) -> None:
    """최초 기록을 먼저 고정한 뒤 산출물을 쓴다. 이미 있으면 내용이 같을 때만 통과하고 덮어쓰지 않는다."""
    try:
        write_once(folder / LOCK_FILE, dump(record))
        for name, text in texts.items():
            write_once(folder / name, text)
    except OverwriteError as error:
        raise IdentityError(f"{folder.name} differs from its first record and is not overwritten: {error}") from error


def assemble(ctx: MixContext, out) -> dict:
    """계획한 단위가 모두 봉인됐을 때만 연다. 최초 조립이 기록을 고정하고, 이후 조립은 그 기록과 같을 때만 통과한다."""
    with _entered(ctx, out):
        plan = Plan(ctx, out)
        result = _result(plan)
        texts = {"result.json": dump(result), "summary.md": report.markdown(json.loads(dump(result)))}
        _publish(plan.folder, _first_record(plan, result, texts), texts)
        return result
