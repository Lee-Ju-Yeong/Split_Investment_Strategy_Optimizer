"""실행 식별값과 두 진입점. kodex_grid의 문맥(승인 소스·탐색 소스·격자·입력 대조)에 이 조사의 소스 해시와
정의 파일 해시·실제 적용 정의 해시를 더한다.

- open_context: 등록한 실제 묶음 전용(재현·자격 집계·본실행). 고정 정의 파일과 kodex_grid r3 판본만 쓴다.
- open_synthetic_context: 합성 입력 전용(시험). 등록한 실제 묶음은 받지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from ..engine_inputs import _sha
from ..kodex_core import data as loader
from ..kodex_core import study
from ..kodex_core.config import StudyConfig
from ..kodex_grid import identity as grid_identity
from ..kodex_grid.grid import GridSpec
from ..kodex_grid.identity import BundleSource, Context, IdentityError
from .definition import MixSpec, check_definition, load_definition


STUDY = "markets-mix"


def frozen_definition() -> MixSpec:
    """고정 정의(파일 그대로)."""
    return load_definition()


def mix_source() -> dict:
    """이 패키지의 실행용 .py 전부의 상대 경로와 내용으로 만든 소스 해시."""
    here = Path(__file__).resolve().parent
    files = sorted(here.glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.relative_to(here.parent)} {_sha(path)}\n".encode())
    return {"mix_source_sha256": digest.hexdigest(), "mix_source_files": [str(path.relative_to(here.parent)) for path in files]}


def mix_identity(mix: MixSpec) -> dict:
    return {"study": STUDY, "mix_source_sha256": mix_source()["mix_source_sha256"],
            "mix_definition_file_sha256": mix.sha256, "mix_definition_applied_sha256": mix.applied_sha256()}


class MixSource(BundleSource):
    """고정 M5 묶음 입력기. 승인된 사전 대조에 더해, 고정 설정의 명단에 없는 상품(나스닥100·회사채)의 가격 기준도
    첫 장부 전에 확인한다. 탐색 명단이 아닌 상품은 가격 행을 읽기 전에 입력 명세 검사에서 멈춘다."""

    def __init__(self, data_root, manifest, base: StudyConfig, *, symbols, registered=None):
        super().__init__(data_root, manifest, base, registered=registered)
        self.symbols = tuple(symbols)

    def load_until(self, symbol, start):
        """그 시작일 시가에 알 수 있는 입력만: 그날까지의 일봉(직전 251거래일 이력·직전 20거래일 거래대금 포함).
        시작일 뒤의 행은 읽지 않는다. 승인 입력기 그대로이고, 출처는 사전 대조한 묶음과 같아야 한다."""
        inputs = loader.load_etf(self.root, self.manifest, symbol, start, start)
        self._check_provenance(inputs.provenance)
        return inputs

    def identity(self) -> dict:
        found = super().identity()
        basis = dict(found["price_basis_by_symbol"])
        for symbol in self.symbols:
            basis.setdefault(symbol, self._basis(symbol))
        return {**found, "price_basis_by_symbol": basis}

    def _basis(self, symbol) -> str:
        try:
            rows = loader._source(self.root, self.manifest, symbol, self.base.end)[2]
            bases = sorted({str(row.get("price_basis")) for row in rows})
        finally:
            loader._source.cache_clear()
        if bases != [self.base.price_basis]:
            raise study.InputIdentityError(f"{symbol}: price basis {bases} differs from the registered one")
        return bases[0]


@dataclass(frozen=True)
class MixContext:
    """이 조사의 실행 문맥: 정의, kodex_grid 문맥(격자·고정 설정·입력기), 합친 식별값."""

    mix: MixSpec
    grid: Context
    identity: dict

    @property
    def spec(self) -> GridSpec:
        return self.grid.spec

    @property
    def base(self) -> StudyConfig:
        return self.grid.base

    @property
    def source(self):
        return self.grid.source

    @property
    def registered(self) -> bool:
        return self.grid.registered


def _context(grid: Context, mix: MixSpec) -> MixContext:
    check_definition(mix, grid.spec, grid.base)
    covered = set(grid.identity["inputs"].get("price_basis_by_symbol", mix.symbols()))
    if not set(mix.symbols()) <= covered:
        raise IdentityError("the input identity does not cover every product of the definition")
    return MixContext(mix, grid, {**grid.identity, **mix_identity(mix)})


def check_frozen(ctx: MixContext) -> None:
    """실제 묶음 실행은 고정 정의 파일, 등록한 kodex_grid 판본, 이 조사의 입력기만 쓴다."""
    grid_identity.check_frozen(ctx.grid)
    frozen, problems = frozen_definition(), []
    if (ctx.mix.sha256, ctx.mix.applied_sha256()) != (frozen.sha256, frozen.applied_sha256()):
        problems.append("mix definition is not the frozen file")
    if not isinstance(ctx.source, MixSource) or set(ctx.source.symbols) != set(ctx.mix.symbols()):
        problems.append("source is not the pinned loader of the declared products")
    approved = ctx.mix.approved
    if ctx.identity["exploration_source_sha256"] != approved["kodex_grid_source_sha256"]:
        problems.append("kodex_grid source is not the registered r3 version")
    if (ctx.identity["grid_file_sha256"], ctx.identity["approved_source_sha256"]) != (
            approved["grid_file_sha256"], approved["source_sha256"]):
        problems.append("grid file or approved source differs from the registered hashes")
    if problems:
        raise IdentityError("a registered run needs the frozen definitions: " + "; ".join(problems))


def check_synthetic(ctx: MixContext) -> None:
    grid_identity.check_synthetic(ctx.grid)


def open_context(source) -> MixContext:
    """등록한 실제 묶음 전용 실행 문맥."""
    ctx = _context(grid_identity.open_context(source), frozen_definition())
    check_frozen(ctx)
    return ctx


def open_synthetic_context(source, *, mix=None, spec=None, base=None) -> MixContext:
    """합성 입력 전용 실행 문맥(시험). 정의·격자·설정을 바꿔 넣을 수 있지만 실제 묶음은 받지 않는다."""
    grid = grid_identity.open_synthetic_context(source, spec=spec, base=base)
    return _context(grid, mix or frozen_definition())


def check_current(ctx: MixContext) -> None:
    """문맥의 코드 식별값이 지금 코드·정의로 다시 만든 값과 같은지."""
    current = {**grid_identity.code_identity(ctx.spec, ctx.base), **mix_identity(ctx.mix)}
    if any(ctx.identity.get(key) != value for key, value in current.items()):
        raise IdentityError("the context identity does not describe the current code and definitions")


def check_loaded(ctx: MixContext, inputs) -> None:
    grid_identity.check_loaded(ctx.grid, inputs)
