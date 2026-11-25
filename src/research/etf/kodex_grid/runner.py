"""한 설정·한 시작일 실행기(탐색 전용). 승인된 장부·지표·조립 함수(kodex_core)를 그대로 부른다.

'run_start'는 같은 설정의 study.run_start와 같은 값이고, 'exploration'은 판정에 쓰지 않는 관찰이다.
보유·코어 장부는 입력 묶음·종목·시작일·종료일·비용·경로·원금이 모두 같을 때만 캐시에서 다시 쓴다.
분할 장부는 정수 주·비용·한도 때문에 배수로 환산하지 않고 설정마다 따로 실행한다.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import localcontext
import hashlib
import time

from ..kodex_core import study
from ..kodex_core.config import StudyConfig
from ..kodex_core.inputs import EtfInputs, check_volatility, starting_at, volatility_digest
from ..kodex_core.ledgers import LedgerRun, RunSetting, run_hold, run_split
from ..kodex_core.verdict import HOLD, candidates, item_status


COSTS = (study.BASE, study.STRESS)
SPLIT_HOLD = "split_hold"
CLOCK_KINDS = {"core": "core", "hold": "hold", "split": SPLIT_HOLD}
PROVENANCE_KEYS = ("input_manifest_sha256", "daily_sha256", "metadata_sha256", "calendar_sha256", "price_bundle",
                   "market_rules_sha256", "source_sha256")


def inputs_digest(data: EtfInputs) -> str:
    """입력 묶음·종목·시작일·종료일을 한꺼번에 식별한다(출처 식별값, 상품, 모든 일봉, 달력, 연초 변동성).
    보유 장부는 변동성을 읽지 않지만, 입력이 조금이라도 다르면 같은 묶음으로 보지 않는다."""
    provenance = {key: data.provenance.get(key) for key in PROVENANCE_KEYS}
    digest = hashlib.sha256()
    for part in (provenance, data.instrument, data.bars, sorted(data.calendar.times.items()), data.distributions,
                 data.data_end, volatility_digest(data.volatility)):
        digest.update(repr(part).encode())
    return digest.hexdigest()


class LedgerCache:
    """보유·코어 장부 캐시. 열쇠가 모두 같은 장부만 다시 쓴다(돌려준 장부는 읽기만 한다)."""

    def __init__(self):
        self.runs, self.hits, self.misses = {}, 0, 0

    def get(self, key, compute) -> LedgerRun:
        if key in self.runs:
            self.hits += 1
        else:
            self.misses += 1
            self.runs[key] = compute()
        return self.runs[key]


class NoCache(LedgerCache):
    """캐시 없이 매번 새로 계산한다(캐시 사용 결과와의 대조용)."""

    def get(self, key, compute) -> LedgerRun:
        self.misses += 1
        return compute()


class Clock:
    """장부 종류별 경과 시간(초)과 횟수. 결과 값에는 넣지 않는다."""

    def __init__(self):
        self.seconds, self.counts = {}, {}

    @contextmanager
    def __call__(self, kind):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[kind] = self.seconds.get(kind, 0.0) + time.perf_counter() - started
            self.counts[kind] = self.counts.get(kind, 0) + 1


@dataclass
class Bench:
    cache: LedgerCache = field(default_factory=LedgerCache)
    clock: Clock = field(default_factory=Clock)


@dataclass
class _Job:
    data: EtfInputs
    config: StudyConfig
    bench: Bench

    def __post_init__(self):
        if len(self.config.modes) != 1:
            raise ValueError("the exploration runner takes exactly one reentry mode")
        self.mode = self.config.modes[0]
        self.digest = inputs_digest(self.data)

    def hold(self, cost, path, name) -> LedgerRun:
        """원금 이름(core·hold·split)만큼 시작 시가에 최대 정수 주를 사서 보유하는 장부."""
        setting, capital = RunSetting(self.config, path, self.config.cost(cost)), self.config.capital[name]
        key = (self.digest, path, setting.commission, capital, self.config.participation_rate,
               self.config.penetration_ticks, self.config.slippage_ticks)
        with self.bench.clock(CLOCK_KINDS[name]):
            return self.bench.cache.get(key, lambda: run_hold(self.data, setting, capital))

    def ledgers(self, cost, path) -> dict:
        """study._ledgers와 같은 장부 셋. 코어·단독 보유만 캐시를 거친다."""
        runs = {"core": self.hold(cost, path, "core"), "hold": self.hold(cost, path, "hold")}
        with self.bench.clock("split"):
            runs[self.mode] = run_split(self.data, RunSetting(self.config, path, self.config.cost(cost), self.mode))
        return runs

    def split_holds(self) -> dict:
        """분할 몫 대조용: 실제 분할 원금만큼의 단독 보유. 코어 50%에서는 열쇠가 같은 코어 장부가 다시 쓰인다."""
        return {cost: {path: self.hold(cost, path, "split") for path in self.config.paths} for cost in COSTS}

    def cumulative(self, run: LedgerRun, name) -> dict:
        """만기 순자산 ÷ 시작 원금 − 1. 만기까지 성립하지 않은 장부는 수치를 비우고 상태만 남긴다."""
        capital = self.config.capital[name]
        if self.data.data_end is not None or not study._valid(run):
            return {"status": run.status, "capital_krw": capital, "ending_equity_krw": None,
                    "cumulative_return": None}
        ending = run.result.valuations[-1]["equity_krw"]
        return {"status": run.status, "capital_krw": capital, "ending_equity_krw": ending,
                "cumulative_return": ending / capital - 1}


def _trade_counts(run: LedgerRun) -> dict:
    if run.result is None:
        return {"status": run.status, "buys": None, "sells": None}
    sides = [row["side"] for row in run.result.trades]
    return {"status": run.status, "buys": sides.count("BUY"), "sells": sides.count("SELL")}


def _reentry_counts(run: LedgerRun) -> dict:
    attempts = run.detail.get("reentries")
    if attempts is None:
        return {"status": run.status, "attempts": None, "filled": None, "unfilled": None}
    outcomes = [row["outcome"] for row in attempts]
    unfilled = {name: outcomes.count(name) for name in sorted(set(outcomes)) if name != "filled"}
    return {"status": run.status, "attempts": len(outcomes), "filled": outcomes.count("filled"), "unfilled": unfilled}


def _observations(job: _Job, group, holds) -> dict:
    """group[cost] = 한 경로의 장부 셋, holds[cost] = 그 경로의 분할 원금 보유 장부."""
    failures, start_holds, later, _ = study._flags(job.data, job.config, group, job.mode)
    return {"flags": {"failures": failures, "holds": start_holds, "later": later},
            "split_ledger": {cost: _trade_counts(group[cost][job.mode]) for cost in COSTS},
            "reentry_counts": {cost: _reentry_counts(group[cost][job.mode]) for cost in COSTS},
            "core_share": {cost: job.cumulative(group[cost]["core"], "core") for cost in COSTS},
            "split_share": {cost: {"split": job.cumulative(group[cost][job.mode], "split"),
                                   "hold_same_capital": job.cumulative(holds[cost], "split")} for cost in COSTS}}


@dataclass
class PointLedgers:
    """한 설정·시작일의 머리 항목과 장부 묶음. 명목 한도 미달이면 장부를 열지 않는다(runs=None)."""

    job: _Job
    head: dict
    runs: dict | None = None
    split_hold: dict | None = None


def _head(job: _Job, start, nominal_floor) -> dict:
    data, config = job.data, job.config
    last = data.calendar.evaluation_times(data.start, data.end)[-1]
    return {"symbol": data.instrument.symbol, "start": start, "end": data.end, "last_evaluation": last,
            "data_end": data.data_end, "basis": data.basis_label(config.labels),
            "capacity": study._capacity(data, config, nominal_floor), "input_provenance": dict(data.provenance)}


def _result(job: _Job, run_start, extras) -> dict:
    """탐색 결과의 꼴. 본문에도 실제로 쓴 적용 설정 해시를 남겨 머리(식별값)와 대조할 수 있게 한다."""
    return {"mode": job.mode, "held_reason": None, "held_slots": None,
            "applied_config_sha256": job.config.applied_sha256(), "run_start": run_start, "exploration": extras}


def _held_by_capacity(ledgers: PointLedgers) -> dict:
    """확장 명목 한도 미달: 승인 코드와 같이 장부를 열지 않고 비교 보류로 남긴다."""
    job = ledgers.job
    items = study._held_items(job.config, study.CAPACITY_HOLD)
    run_start = {**ledgers.head, "items": items, "candidates": candidates({mode: HOLD for mode in items}),
                 "warnings": []}
    flags = {"failures": [], "holds": [study.CAPACITY_HOLD], "later": []}
    empty = {"flags": flags, "split_ledger": None, "reentry_counts": None, "core_share": None, "split_share": None}
    return _result(job, run_start, {path: dict(empty) for path in job.config.paths})


def _warnings(runs) -> list:
    return sorted({w for c in runs.values() for p in c.values() for r in p.values()
                   if r.result is not None for w in r.result.warnings})


def _assemble(ledgers: PointLedgers) -> dict:
    """경로별 조립은 승인 코드의 study._path_result를 그대로 부른다."""
    job, runs, holds = ledgers.job, ledgers.runs, ledgers.split_hold
    mode, config = job.mode, job.config
    groups = {path: {cost: runs[cost][path] for cost in COSTS} for path in config.paths}
    with job.bench.clock("assemble"):
        paths = {path: study._path_result(job.data, config, groups[path], mode, path, False)
                 for path in config.paths}
    items = {mode: {"status": item_status(row["verdict"]["status"] for row in paths.values()), "paths": paths}}
    extras = {path: _observations(job, groups[path], {cost: holds[cost][path] for cost in COSTS})
              for path in config.paths}
    run_start = {**ledgers.head, "items": items, "candidates": candidates({mode: items[mode]["status"]}),
                 "warnings": _warnings(runs)}
    return _result(job, run_start, extras)


def _from(inputs: EtfInputs, start) -> EtfInputs:
    """그 시작일부터의 입력. study.run_start와 같이 사용 직전에 연초 변동성을 다시 검사한다(미래 정보 차단)."""
    data = starting_at(inputs, start)
    try:
        check_volatility(data.calendar, data.volatility)
    except ValueError as error:
        raise study.InputIdentityError(f"{data.instrument.symbol} {start}: {error}") from error
    return data


def point_ledgers(inputs: EtfInputs, config: StudyConfig, start, *, nominal_floor=None, bench=None) -> PointLedgers:
    """study.run_start와 같은 순서(변동성 재검사 → 명목 한도 → 장부)로 장부까지만 연다."""
    job = _Job(_from(inputs, start), config, bench or Bench())
    head = _head(job, start, nominal_floor)
    if not head["capacity"]["ok"]:
        return PointLedgers(job, head)
    with localcontext(prec=80):
        runs = {cost: {path: job.ledgers(cost, path) for path in config.paths} for cost in COSTS}
        return PointLedgers(job, head, runs, job.split_holds())


def assemble_point(ledgers: PointLedgers) -> dict:
    """연 장부를 경로별로 조립한다(명목 한도 미달이면 비교 보류 행)."""
    if ledgers.runs is None:
        return _held_by_capacity(ledgers)
    with localcontext(prec=80):
        return _assemble(ledgers)


def open_ledgers(inputs: EtfInputs, config: StudyConfig, start, *, bench=None) -> dict:
    """두 비용 × 두 경로의 장부 셋(코어·단독 보유·분할)과 분할 원금 보유 장부를 그대로 돌려준다."""
    ledgers = point_ledgers(inputs, config, start, bench=bench)
    return {"runs": ledgers.runs, SPLIT_HOLD: ledgers.split_hold}


def run_point(inputs: EtfInputs, config: StudyConfig, start, *, nominal_floor=None, bench=None) -> dict:
    """한 설정·시작일 실행: 장부를 열고 경로별로 조립한다."""
    return assemble_point(point_ledgers(inputs, config, start, nominal_floor=nominal_floor, bench=bench))


def held_point(config: StudyConfig, reason) -> dict:
    """예상한 자료 문제(DataHold)로 적재하지 못한 시작일: 두 경로 × 두 비용 네 자리에 빈 수치와 사유를 남긴다."""
    slots = {path: {cost: {"values": None, "reason": reason} for cost in COSTS} for path in config.paths}
    return {"mode": config.modes[0], "held_reason": reason, "held_slots": slots,
            "applied_config_sha256": config.applied_sha256(), "run_start": None, "exploration": None}
