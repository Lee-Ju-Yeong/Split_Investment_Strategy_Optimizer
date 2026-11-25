"""실행 식별값과 두 진입점. 승인 소스 해시와 별도로 탐색 소스 해시·격자 정의 해시·실제 입력을 남기고,
재개·조립 때 같은 판본·같은 입력인지 대조한다.

- open_context: 등록한 실제 묶음 전용(본실행). 고정 격자 정의·고정 설정만 쓴다.
- open_synthetic_context: 합성 입력 전용(시험). 등록한 실제 묶음은 받지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from ..engine_inputs import _sha
from ..kodex_core import formal, study
from ..kodex_core.config import StudyConfig, load_config
from ..kodex_core.data import load_etf, source_identity
from .grid import GridSpec, check_base, load_grid


PROVENANCE_SAME = ("input_manifest_sha256", "daily_sha256", "metadata_sha256", "calendar_sha256", "price_bundle")
REAL_MARKS = ("input_manifest_sha256", "daily_sha256", "price_bundle")  # 등록한 실제 묶음을 가리키는 값
REGISTERED, SYNTHETIC = "registered", "synthetic"


class IdentityError(RuntimeError):
    """승인 소스·탐색 소스·격자 정의·입력 식별값이 기대와 다르다. 다른 판본의 결과를 섞지 않도록 멈춘다."""


def registered_inputs() -> dict:
    """등록한 실제 묶음의 식별값(고정 설정 파일)."""
    return dict(load_config().registered)


def frozen_grid() -> GridSpec:
    """고정 격자 정의(파일 그대로)."""
    return load_grid()


def exploration_source() -> dict:
    """이 패키지의 실행용 .py 전부의 상대 경로와 내용으로 만든 탐색 소스 해시."""
    here = Path(__file__).resolve().parent
    files = sorted(here.glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.relative_to(here.parent)} {_sha(path)}\n".encode())
    return {"exploration_source_sha256": digest.hexdigest(),
            "exploration_source_files": [str(path.relative_to(here.parent)) for path in files]}


def code_identity(spec: GridSpec, base: StudyConfig) -> dict:
    """승인 소스가 등록값과 같을 때만 식별값을 만든다."""
    approved = source_identity()
    if approved["source_sha256"] != spec.approved["source_sha256"]:
        raise IdentityError("kodex_core source hash differs from the approved value; exploration is not run")
    return {"approved_source_sha256": approved["source_sha256"],
            "market_rules_sha256": approved["market_rules_sha256"],
            "exploration_source_sha256": exploration_source()["exploration_source_sha256"],
            "grid_file_sha256": spec.sha256, "grid_applied_sha256": spec.applied_sha256(),
            "base_config_file_sha256": base.sha256, "base_config_applied_sha256": base.applied_sha256()}


def carries_real_marks(values: dict) -> list:
    """등록한 실제 묶음과 같은 식별값(명세·일봉 파일·가격 묶음 이름)."""
    registered = registered_inputs()
    return [key for key in REAL_MARKS if values.get(key) is not None and values.get(key) == registered.get(key)]


class BundleSource:
    """고정 M5 묶음 입력기. 장부를 열기 전에 승인된 사전 대조(formal.preflight)를 그대로 통과해야 한다.
    registered를 넘기면(합성 묶음 시험용) 그 값으로 대조한다. 등록값과 같은 묶음만 실제 묶음으로 표시된다."""

    def __init__(self, data_root, manifest, base: StudyConfig, *, registered=None):
        self.root, self.manifest = Path(data_root).resolve(), Path(manifest).resolve()
        self.base, self.registered = base, registered
        self.checked = None

    def identity(self) -> dict:
        self.checked = formal.preflight(self.root, self.manifest, self.base, registered=self.registered)
        actual = {row["item"]: row["actual"] for row in self.checked["inputs"]}
        return {"actual": actual, "registered": dict(self.checked["registered"]),
                "price_basis_by_symbol": dict(self.checked["price_basis_by_symbol"]),
                "registered_inputs_frozen": self.checked["registered"] == registered_inputs()}

    def load(self, symbol, start):
        """한 종목·시작일 입력. 예상한 자료 문제(DataHold)는 호출자가 보류로 남기고, 식별 불일치는 멈춘다."""
        inputs = load_etf(self.root, self.manifest, symbol, start, self.base.end)
        study.verify_identity(inputs, symbol, start, self.base)
        self._check_provenance(inputs.provenance)
        return inputs

    def _check_provenance(self, provenance) -> None:
        actual = {row["item"]: row["actual"] for row in self.checked["inputs"]}
        problems = [key for key in PROVENANCE_SAME if provenance.get(key) != actual[key]]
        if provenance.get("manifest_synthetic") is not False:
            problems.append("manifest_synthetic")
        if (provenance.get("extra_inputs") or {}).get("present", True):
            problems.append("extra_inputs")
        if provenance.get("price_basis") != self.base.price_basis:
            problems.append("price_basis")
        if problems:
            raise study.InputIdentityError(f"loaded input differs from the checked bundle: {problems}")


@dataclass(frozen=True)
class Context:
    spec: GridSpec
    base: StudyConfig
    source: object  # identity() -> dict, load(symbol, start) -> EtfInputs
    identity: dict

    @property
    def registered(self) -> bool:
        """등록한 실제 묶음의 실행인지. 표시가 아니라 입력 식별값의 내용으로 본다."""
        inputs = self.identity["inputs"]
        return inputs.get("registered_inputs_frozen") is True or bool(carries_real_marks(inputs.get("actual", {})))


def _context(source, spec: GridSpec, base: StudyConfig) -> Context:
    check_base(spec, base)
    inputs = source.identity()
    scope = REGISTERED if inputs.get("registered_inputs_frozen") is True else SYNTHETIC
    identity = {"exploration": True, "formal_run": False, "scope": scope, **code_identity(spec, base),
                "inputs": inputs}
    return Context(spec, base, source, identity)


def check_frozen(ctx: Context) -> None:
    """실제 묶음 실행은 고정 입력기와 고정 격자 정의·고정 설정만 쓴다."""
    spec, base = frozen_grid(), load_config()
    problems = [] if isinstance(ctx.source, BundleSource) else ["source is not the pinned bundle loader"]
    if (ctx.spec.sha256, ctx.spec.applied_sha256()) != (spec.sha256, spec.applied_sha256()):
        problems.append("grid definition is not the frozen file")
    if ctx.base.applied_sha256() != base.applied_sha256():
        problems.append("base configuration is not the frozen file")
    if ctx.identity.get("scope") != REGISTERED:
        problems.append("inputs are not the registered bundle")
    if problems:
        raise IdentityError("a registered run needs the frozen definitions: " + "; ".join(problems))


def check_synthetic(ctx: Context) -> None:
    """합성 전용 진입점은 등록한 실제 묶음을 받지 않는다."""
    if ctx.registered or ctx.identity.get("scope") != SYNTHETIC:
        raise IdentityError("the synthetic entry does not take the registered bundle; use the registered entry")


def open_context(source) -> Context:
    """등록한 실제 묶음 전용 실행 문맥. 격자 정의·고정 설정·승인 소스·입력을 대조한다."""
    ctx = _context(source, frozen_grid(), load_config())
    check_frozen(ctx)
    return ctx


def open_synthetic_context(source, *, spec=None, base=None) -> Context:
    """합성 입력 전용 실행 문맥(시험). 격자 정의와 설정을 바꿔 넣을 수 있지만 실제 묶음은 받지 않는다."""
    ctx = _context(source, spec or load_grid(), base or load_config())
    check_synthetic(ctx)
    return ctx


def check_loaded(ctx: Context, inputs) -> None:
    """합성 실행에 실제 묶음에서 읽은 입력이 섞이지 않았는지 적재 직후에 본다."""
    if not ctx.registered and carries_real_marks(dict(inputs.provenance)):
        raise IdentityError("an input loaded from the registered bundle reached a synthetic run")
