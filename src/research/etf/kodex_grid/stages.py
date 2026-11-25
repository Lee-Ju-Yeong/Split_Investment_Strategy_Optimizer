"""단계 실행·봉인·재개·조립(설계 v2.1 8절).

- 네 단계 함수(run_stage1·assemble1·run_stage2·assemble2)는 같은 진입점을 거친다: 본실행 허가 확인 → 출력 폴더의
  작성자 잠금. 실제 묶음은 같은 판본의 합성 시험 통과 기록과 공개 기준점 대조 증거가 있어야 들어온다.
- 묶음(Plan)이 입력을 적재하거나 단위를 계산·봉인하거나 봉인 단위를 여는 모든 공용 함수는 가장 안쪽 경계
  Plan.guard를 거친다: 그 출력 폴더의 허가, 이 호출 흐름의 작성자 잠금, (2단계) 최초 선택과의 연결.
- 설정별 결과는 한 번만 쓰고, 그 바이트의 해시를 실행 당시의 별도 봉인 기록으로 남긴다. 재개와 조립은 같은 검사를 쓴다.
- 최초 조립이 단위별 내용 해시·실행 식별값·선택(순위와 8개 번호)·산출물 해시를 고정한다. 이후 조립과 2단계는
  그 기록과 같을 때만 통과하고, 다른 내용으로 덮어쓰지 않는다.
- 조립 전에 공개되는 것은 단위 이름·소요시간·기술 상태 코드뿐이다. 작업 내부의 출력과 예외 전문은 봉인 위치에 둔다.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import json
import multiprocessing
from pathlib import Path
import resource
import time
import traceback

from ..kodex_core import data as loader
from ..kodex_core import study
from ..kodex_core.inputs import DataHold
from . import judge, permit, reading, report, summary
from .grid import segment_labels
from .identity import Context, IdentityError, check_loaded
from .permit import PermitError
from .progress import DONE, SKIPPED, Progress
from .records import UNIT_SCHEMA, AssemblyError, Seat, Unit, check_record, primary_unit
from .runner import Bench, LedgerCache, held_point, run_point
from .sealing import (SEALED, OverwriteError, bytes_digest, captured_output, clear_temporaries, digest, dump,
                      encoded, error_code, flush_standard, listed, read_sealed, with_seal, write_marker, write_once,
                      writer_lock, writer_token)


SEAL_SCHEMA = "kodex-grid-seal-v1"
LOCK_SCHEMA = "kodex-grid-lock-v1"
LOCK_FILE = "lock.json"
STAGE1, STAGE2 = "stage1", "stage2"
NOT_RUN, RUN = "not_run", "run"
CHOSEN_KEYS = ("state", "region", "blocked_by_rank", "candidate_if_reconfirmed", "stage2")


class StageError(RuntimeError):
    """단위 계산의 기술 오류. 같은 입력으로 복구해 다시 실행한 뒤에만 조립한다."""


@dataclass(frozen=True)
class _Held:
    reason: str


def _timing(bench: Bench, started) -> dict:
    return {"seconds": time.perf_counter() - started, "by_kind_seconds": dict(bench.clock.seconds),
            "by_kind_counts": dict(bench.clock.counts),
            "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


class Plan:
    """한 단계의 작업 묶음: 단위 목록, 봉인 폴더, 적재한 입력, 보유·코어 장부 캐시."""

    def __init__(self, ctx: Context, folder, units, *, binding=None):
        self.ctx, self.folder, self.units = ctx, Path(folder), tuple(units)
        self.stage, self.binding = self.folder.name, dict(binding or {})
        self.out, self.admitted = self.folder.parent, None
        self.inputs, self.cache = {}, LedgerCache()

    def guard(self) -> None:
        """가장 안쪽 공용 경계. 이 출력 폴더의 본실행 허가, 이 호출 흐름(또는 그것이 포크한 작업 자식)의 작성자 잠금,
        2단계면 최초 선택과의 연결을 확인한다. 같은 잠금 구간 안에서는 한 번 확인한 것으로 충분하다."""
        token = writer_token(self.out)
        if token is not None and token == self.admitted:
            return
        permit.enter(self.ctx, self.out)
        if token is None:
            raise PermitError(f"the writer lock of {self.out} is not held by this call flow")
        _check_link(self)
        self.admitted = token

    def seat(self, unit: Unit) -> Seat:
        return Seat(self.ctx, self.stage, unit, self.binding)

    def path(self, unit: Unit) -> Path:
        return self.folder / "units" / f"{unit.name}.json"

    def seal_path(self, unit: Unit) -> Path:
        return self.folder / "sealed" / "seals" / f"{unit.name}.json"

    def log_path(self, unit: Unit) -> Path:
        return self.folder / "sealed" / "logs" / f"{unit.name}.log"

    def error_path(self, unit: Unit) -> Path:
        return self.folder / "errors" / f"{unit.name}.error.txt"

    def _load(self, key):
        """한 종목·시작일 입력. 예상한 자료 문제만 보류 사유로 남기고, 합성 실행에 실제 묶음이 섞이면 멈춘다."""
        try:
            inputs = self.ctx.source.load(*key)
        except DataHold as error:
            return _Held(str(error))
        check_loaded(self.ctx, inputs)
        return inputs

    def load_inputs(self) -> None:
        """단위들이 쓰는 (종목, 시작일) 입력을 한 번씩 읽는다."""
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
        if isinstance(loaded, _Held):
            result = held_point(seat.config(), loaded.reason)
        else:
            result = run_point(loaded, seat.config(), unit.start, nominal_floor=seat.nominal_floor(), bench=bench)
        return {"schema": UNIT_SCHEMA, "exploration": True, "formal_run": False, "identity": seat.identity(),
                "setting": self.ctx.spec.setting(unit.number).describe(), "result": result}

    def seal(self, unit: Unit, started) -> None:
        """단위를 계산해 한 번만 쓰고, 쓴 바이트의 해시와 실행 시간을 별도 봉인 기록으로 남긴다."""
        self.guard()
        bench = Bench(cache=self.cache)
        text = dump(self.compute(unit, bench))
        write_once(self.path(unit), text, mode=SEALED)
        seal = {"schema": SEAL_SCHEMA, "unit": unit.name, "sha256": digest(text), "timing": _timing(bench, started)}
        write_once(self.seal_path(unit), dump(with_seal(seal)), mode=SEALED)

    def verified(self, unit: Unit) -> tuple:
        """봉인된 단위의 (기록, 바이트 해시). 저장된 바이트를 먼저 실행 당시 봉인한 SHA-256과 맞추고, 통과한 바이트만
        JSON으로 해석해 식별값·네 자리·구간 검사를 한다."""
        self.guard()
        raw = self.path(unit).read_bytes()
        try:
            seal = read_sealed(self.seal_path(unit), SEAL_SCHEMA)
        except (OSError, ValueError) as error:
            raise AssemblyError(f"{unit.name}: seal record is missing or damaged") from error
        if (seal.get("unit"), seal.get("sha256")) != (unit.name, bytes_digest(raw)):
            raise AssemblyError(f"{unit.name}: content differs from the hash sealed when it was computed")
        record = json.loads(raw)
        check_record(self.seat(unit), record)
        return record, seal["sha256"]


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
    """설정 단위 병렬 실행. 포크한 작업 프로세스가 죽으면 멈춘 채 기다리지 않고 오류로 끝난다."""
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
    plan.error_path(unit).unlink(missing_ok=True)  # 봉인까지 끝난 단위에 남은 지난 시도의 오류 표시는 지운다
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


def run_units(plan: Plan, workers: int, log) -> None:
    """끝나지 않은 단위만 설정 단위로 병렬 실행하고 봉인한다. 허가와 잠금을 확인하기 전에는 입력을 읽지도,
    임시 파일을 정리하지도, 단위를 쓰지도 않는다."""
    plan.guard()
    clear_temporaries(plan.folder)
    pending = _pending(plan, log)
    if pending:
        plan.load_inputs()
    failed = _run_pending(plan, pending, workers=workers, log=log)
    if failed:
        raise StageError(f"{len(failed)} unit(s) ended with a technical error: {sorted(failed)}")


def _check_folder(plan: Plan) -> None:
    """예정한 단위 파일만 있어야 한다. 누락·여분·기술 오류 표시가 있으면 조립하지 않는다."""
    expected, present = {plan.path(unit).name for unit in plan.units}, listed(plan.folder / "units")
    problems = ([f"missing {name}" for name in sorted(expected - present)]
                + [f"unexpected {name}" for name in sorted(present - expected)]
                + [f"error marker {name}" for name in sorted(listed(plan.folder / "errors"))])
    if problems:
        raise AssemblyError("; ".join(problems))


def _check_unique(records) -> None:
    """같은 설정·종목·시작일이 두 번 들어오면 한 결과를 두 번 세게 되므로 멈춘다."""
    keys = [(json.dumps(record["setting"], sort_keys=True), record["identity"]["symbol"],
             record["identity"]["start"]) for record in records.values()]
    if len(set(keys)) != len(keys):
        raise AssemblyError("duplicate setting/symbol/start among the sealed units")


def read_units(plan: Plan) -> tuple:
    """봉인된 단위를 모두 연다: ({단위: 기록}, {단위 이름: 바이트 해시}). 허가와 잠금을 확인한 뒤에만 연다."""
    plan.guard()
    _check_folder(plan)
    opened = {unit: plan.verified(unit) for unit in plan.units}
    records = {unit: pair[0] for unit, pair in opened.items()}
    _check_unique(records)
    return records, {unit.name: pair[1] for unit, pair in opened.items()}


def _row_head(record) -> dict:
    return {**record["setting"], "symbol": record["identity"]["symbol"], "start": record["identity"]["start"],
            "exploration": True, "formal_run": False,
            "applied_config_sha256": record["identity"]["applied_config_sha256"]}


def _judged(plan: Plan, unit: Unit, record) -> dict:
    """봉인 결과 하나의 4절 판정과 '함께 표기' 항목."""
    frame = plan.seat(unit).frame()
    verdict = judge.setting_verdict(reading.point_input(record["result"], frame), plan.ctx.spec.thresholds)
    return {**_row_head(record), "verdict": verdict, "display": reading.point_display(record["result"], frame)}


def _extension_row(plan: Plan, unit: Unit, record) -> dict:
    """확장 행에는 확장 점의 판정만 싣는다. 읽는 것은 명목 한도, 기본 비용 사유, 두 기본 경로의 전체 차뿐이다."""
    frame, threshold = plan.seat(unit).frame(), plan.ctx.spec.extension["base_edge_gt"]
    point = judge.extension_point(reading.extension_input(record["result"], frame), threshold)
    return {**_row_head(record), "extension_point": point,
            "display": reading.extension_display(record["result"], frame)}


def _envelope(ctx: Context, schema: str) -> dict:
    """모든 결과의 머리: 탐색 표시, 실행 식별값, 표의 틀(등록한 경로와 주 시작일의 구간)."""
    labels = segment_labels(ctx.spec.segments, ctx.spec.primary_start, ctx.base.end)
    return {"schema": schema, "exploration": True, "formal_run": False, "identity": ctx.identity,
            "frame": {"paths": list(ctx.base.paths), "labels": list(labels)}}


def _first_record(plan: Plan, hashes, texts, *, chosen) -> dict:
    """최초 조립 때 한 번 고정하는 기록: 실행 식별값, 단위별 내용 해시, 선택·결론, 산출물 해시."""
    return with_seal({"schema": LOCK_SCHEMA, "exploration": True, "formal_run": False, "stage": plan.stage,
                      "identity": plan.ctx.identity, "binding": plan.binding, "units": hashes, "chosen": chosen,
                      "outputs": {name: digest(text) for name, text in texts.items()}})


def _publish(folder: Path, record, texts) -> None:
    """최초 기록을 먼저 고정한 뒤 산출물을 쓴다. 이미 있으면 내용이 같을 때만 통과하고 덮어쓰지 않는다."""
    try:
        write_once(folder / LOCK_FILE, dump(record))
        for name, text in texts.items():
            write_once(folder / name, text)
    except OverwriteError as error:
        raise IdentityError(f"{folder.name} differs from its first record and is not overwritten: {error}") from error


@contextmanager
def _entered(ctx: Context, out):
    """네 단계 함수의 공용 진입점: 본실행 허가를 확인한 뒤 출력 폴더의 작성자 잠금을 잡는다."""
    permit.enter(ctx, out)
    with writer_lock(out):
        yield


# ---------------------------------------------------------------- 1단계: 주 시작일 36개

def stage1_plan(ctx: Context, out) -> Plan:
    units = [primary_unit(ctx.spec, row.number) for row in ctx.spec.settings()]
    return Plan(ctx, Path(out) / STAGE1, units)


def run_stage1(ctx: Context, out, *, workers=1, echo=print) -> None:
    with _entered(ctx, out):
        plan = stage1_plan(ctx, out)
        run_units(plan, workers, Progress(plan.folder, plan.stage, echo))


def _selection(ctx: Context, table) -> dict:
    """영역 선택과 그때 잠기는 8점·순위. '영역 없음'과 '선택 미확정'을 구별해 남긴다."""
    chosen = judge.select_region(table)
    region = chosen["region"]
    candidate = None if region is None else ctx.spec.candidate_number(ctx.spec.regions()[region["rank"] - 1])
    regions = [{key: row[key] for key in ("rank", "core", "kg", "kp", "numbers", "status", "counts")}
               for row in table]
    return {**_envelope(ctx, "kodex-grid-selection-v2"), **chosen, "candidate_if_reconfirmed": candidate,
            "stage2": RUN if chosen["state"] == judge.SELECTED else NOT_RUN,
            "conclusion": judge.conclusion(ctx.spec.conclusions, chosen), "regions": regions}


def _stage1(plan: Plan) -> tuple:
    """봉인 단위에서 1단계 결과를 만든다(파일을 쓰지 않는다): (결과, 단위별 내용 해시)."""
    ctx = plan.ctx
    records, hashes = read_units(plan)
    rows = [_judged(plan, unit, records[unit]) for unit in plan.units]
    statuses = {row["number"]: row["verdict"]["status"] for row in rows}
    table = judge.regions_table([region.describe() for region in ctx.spec.regions()], statuses)
    result = {**_envelope(ctx, "kodex-grid-stage1-v2"), "settings": rows, "regions": table,
              "isolated_passes": judge.isolated_passes(table, statuses), "selection": _selection(ctx, table),
              "summary": summary.describe(rows, ctx.base.paths),
              "units": {unit.name: records[unit] for unit in plan.units}}
    return result, hashes


def _stage1_texts(result) -> dict:
    return {"result.json": dump(result), "selection.json": dump(result["selection"]),
            "summary.md": report.stage1_markdown(json.loads(dump(result)))}


def _stage1_record(plan: Plan, result, hashes) -> tuple:
    texts = _stage1_texts(result)
    chosen = {key: result["selection"][key] for key in CHOSEN_KEYS}
    return _first_record(plan, hashes, texts, chosen=chosen), texts


def assemble1(ctx: Context, out) -> dict:
    """36개가 모두 봉인됐을 때만 연다. 최초 조립이 기록을 고정하고, 이후 조립은 그 기록과 같을 때만 통과한다."""
    with _entered(ctx, out):
        plan = stage1_plan(ctx, out)
        result, hashes = _stage1(plan)
        _publish(plan.folder, *_stage1_record(plan, result, hashes))
        return result


# ---------------------------------------------------------------- 2단계: 재확인과 확장

def _stored(path: Path) -> bytes:
    if not path.is_file():
        raise IdentityError("stage1 has no first record; assemble stage1 before stage2")
    return path.read_bytes()


def locked_selection(ctx: Context, out) -> tuple:
    """1단계 최초 기록과 지금의 봉인 단위·산출물이 약속한 바이트 그대로인지 대조한다: (선택, 최초 기록의 바이트 해시)."""
    plan = stage1_plan(ctx, out)
    plan.guard()
    stored = _stored(plan.folder / LOCK_FILE)
    record, texts = _stage1_record(plan, *_stage1(plan))
    different = [LOCK_FILE] if encoded(dump(record)) != stored else [
        name for name, text in texts.items() if _stored(plan.folder / name) != encoded(text)]
    if different:
        raise IdentityError(f"stage1 differs from its first record in {different}; stage2 does not proceed")
    return json.loads(texts["selection.json"]), bytes_digest(stored)


def stage2_plan(ctx: Context, out, locked) -> Plan:
    """locked: (선택, 1단계 최초 기록의 해시). 2단계 단위의 식별값은 그 해시에 묶인다."""
    spec, extension, numbers = ctx.spec, ctx.spec.extension, (locked[0]["region"] or {}).get("numbers", ())
    units = [Unit("reconfirm", number, spec.symbol, start) for start in spec.reconfirm["starts"]
             for number in numbers]
    units += [Unit("extension", number, symbol, extension["start"]) for symbol in extension["symbols"]
              for number in numbers]
    return Plan(ctx, Path(out) / STAGE2, units, binding={"stage1_lock_sha256": locked[1]})


def _check_link(plan: Plan) -> None:
    """2단계 묶음은 1단계 최초 기록에 잠긴 선택(순위와 8개 번호)과 그 기록의 해시에서 나온 것이어야 한다."""
    if plan.stage != STAGE2:
        return
    expected = stage2_plan(plan.ctx, plan.out, locked_selection(plan.ctx, plan.out))
    if (plan.units, plan.binding) != (expected.units, expected.binding):
        raise IdentityError("stage2 units are not the ones locked by the first stage1 record")


def _check_reconfirm_starts(plan: Plan) -> None:
    """재확인 시작일이 주 시작일의 +5·+20 국내 거래일인지 적재한 달력으로 확인한다."""
    spec = plan.ctx.spec
    loaded = [plan.inputs[(spec.symbol, start)] for start in spec.reconfirm["starts"]]
    calendars = [item.calendar for item in loaded if not isinstance(item, _Held)]
    if calendars:
        study.check_starts(calendars[0], (spec.primary_start, *spec.reconfirm["starts"]),
                           (0, *spec.reconfirm["offsets"]))


def _stage2_texts(result) -> dict:
    return {"result.json": dump(result), "summary.md": report.stage2_markdown(json.loads(dump(result)))}


def _not_run(ctx: Context, out, locked) -> dict:
    """영역 없음·선택 미확정: 2단계를 실행하지 않고 그 구별과 결론 문장을 남긴다."""
    selection, plan = locked[0], stage2_plan(ctx, out, locked)
    result = {**_envelope(ctx, "kodex-grid-stage2-v2"), "stage2": NOT_RUN, "reason": selection["state"],
              "selection": selection, "conclusion": selection["conclusion"]}
    texts = _stage2_texts(result)
    chosen = {"stage2": NOT_RUN, "reason": selection["state"]}
    _publish(plan.folder, _first_record(plan, {}, texts, chosen=chosen), texts)
    return result


def run_stage2(ctx: Context, out, *, workers=1, echo=print) -> str:
    """선택된 영역이 있을 때만, 최초 기록에 잠긴 같은 8점으로 재확인(2시작일)과 확장(2상품)을 실행한다."""
    with _entered(ctx, out):
        locked = locked_selection(ctx, out)
        if locked[0]["state"] != judge.SELECTED:
            _not_run(ctx, out, locked)
            return NOT_RUN
        plan = stage2_plan(ctx, out, locked)
        plan.load_inputs()
        _check_reconfirm_starts(plan)
        run_units(plan, workers, Progress(plan.folder, plan.stage, echo))
        return RUN


def _pair(row) -> str:
    return f"{row['number']:02d}@{row['start']}"


def _reconfirm(plan: Plan, records) -> dict:
    """KODEX 16쌍. 각 쌍 안에서 4절을 그대로 적용한다."""
    units = [unit for unit in plan.units if unit.kind == "reconfirm"]
    rows = [_judged(plan, unit, records[unit]) for unit in units]
    expected = [f"{unit.number:02d}@{unit.start.isoformat()}" for unit in units]
    verdict = judge.reconfirm_verdict({_pair(row): row["verdict"]["status"] for row in rows}, expected)
    return {"verdict": verdict, "rows": rows}


def _extension(plan: Plan, records) -> dict:
    """확장 두 상품. 상품마다 분모 8을 유지하고 KODEX 재확인과 별도 칸에 기록한다."""
    spec, products, rows = plan.ctx.spec, {}, {}
    for symbol in spec.extension["symbols"]:
        units = [unit for unit in plan.units if unit.kind == "extension" and unit.symbol == symbol]
        rows[symbol] = [_extension_row(plan, unit, records[unit]) for unit in units]
        products[symbol] = judge.extension_product([row["extension_point"]["status"] for row in rows[symbol]],
                                                   points=spec.region["points"],
                                                   points_min=spec.extension["points_min"])
    return {"symbols": list(spec.extension["symbols"]), "rows": rows,
            "verdict": judge.extension_verdict(products, expected=spec.extension["symbols"])}


def _stage2(plan: Plan, selection) -> tuple:
    """봉인 단위에서 2단계 결과를 만든다(파일을 쓰지 않는다): (결과, 단위별 내용 해시)."""
    ctx = plan.ctx
    records, hashes = read_units(plan)
    reconfirm, extension = _reconfirm(plan, records), _extension(plan, records)
    conclusion = judge.conclusion(ctx.spec.conclusions, selection, reconfirm=reconfirm["verdict"],
                                  candidate=selection["candidate_if_reconfirmed"])
    result = {**_envelope(ctx, "kodex-grid-stage2-v2"), "stage2": RUN, "selection": selection,
              "kodex_reconfirm": reconfirm, "extension": extension, "conclusion": conclusion,
              "units": {unit.name: records[unit] for unit in plan.units}}
    return result, hashes


def _stage2_chosen(result) -> dict:
    return {"stage2": RUN, "kodex_reconfirm": result["kodex_reconfirm"]["verdict"]["status"],
            "extension": result["extension"]["verdict"]["status"], "candidate": result["conclusion"]["candidate"]}


def assemble2(ctx: Context, out) -> dict:
    """재확인 16쌍과 확장 상품별 8점이 모두 봉인됐을 때만 연다. 두 결과는 별도 칸에 기록한다."""
    with _entered(ctx, out):
        locked = locked_selection(ctx, out)
        if locked[0]["state"] != judge.SELECTED:
            return _not_run(ctx, out, locked)
        plan = stage2_plan(ctx, out, locked)
        result, hashes = _stage2(plan, locked[0])
        texts = _stage2_texts(result)
        _publish(plan.folder, _first_record(plan, hashes, texts, chosen=_stage2_chosen(result)), texts)
        return result
