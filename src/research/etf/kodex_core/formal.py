"""정식 실행 진입점(B1·B4·B7). 장부를 열기 전에 등록 입력을 대조하고, 등록 파일에서 직접 입력을 구성한다.

- 등록 입력 명세 전체 SHA-256, synthetic=false, 가격 묶음·국내 일봉·분류표·달력·시장 규칙 해시를 대조한다.
- 명세 불일치·파일 손상·해시 오류는 InputIntegrityError로 멈춘다(자료 보류로 흡수하지 않음).
- 추가 입력(제한폭 확인값·분배금·외부 변동성)은 받지 않는다. 그런 실행은 study.run_plan으로만 가능하며
  formal_run=False로 남는다. study.run_plan은 사전 대조표를 받지 않는다(D2).
- 고정 설정·명단·시작일·만기·가격 기준이 다르면 장부를 열기 전에 InputIdentityError(D3).
- 호출마다 입력 캐시를 비우고 파일에서 다시 읽은 읽기 전용 사본을 쓴다(D1).
- 등록 입력 확인 뒤의 자료 부족(예: 2004년 달력)은 시작일별 보류로 남긴다.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from ..engine_inputs import _calendar, _sha, _snapshot
from ..m5_inputs import read_manifest
from . import data as _data
from .calendar import KrCalendar
from .config import StudyConfig, frozen_identity, load_config
from .data import load_etf, source_identity
from .inputs import InputIntegrityError
from .study import InputIdentityError, _context, _sources, check_starts, execute_plan


IDENTITY_KEYS = ("price_bundle", "input_manifest_sha256", "synthetic", "daily_sha256", "metadata_sha256",
                 "calendar_sha256", "market_rules_sha256", "extra_inputs")


def _symbols(config: StudyConfig) -> list:
    return sorted({symbol for row in config.rows for symbol in row.symbols})


def actual_inputs(root: Path, manifest_path: Path, config: StudyConfig) -> dict:
    """등록 항목의 실제 값. 명세 파일 해시 검사(read_manifest) 실패는 무결성 오류."""
    try:
        manifest = json.loads(manifest_path.read_bytes())
        read_manifest(root, manifest_path, _symbols(config))
        record, _, _, bundle = _snapshot(root, root / manifest["split_csv"], _symbols(config))
    except (ValueError, OSError, KeyError) as error:
        raise InputIntegrityError(f"pinned input manifest check failed: {error}") from error
    rules = source_identity()["market_rules_sha256"]
    return {"price_bundle": bundle.name, "input_manifest_sha256": _sha(manifest_path),
            "synthetic": manifest.get("synthetic"), "daily_sha256": record["inputs"]["daily_sha256"]["KR"],
            "metadata_sha256": record["inputs"]["metadata_sha256"],
            "calendar_sha256": _sha(root / "fx" / "calendar_krx.csv"), "market_rules_sha256": rules,
            "extra_inputs": "none"}


def identity_table(registered: dict, actual: dict) -> list:
    return [{"item": key, "registered": registered.get(key), "actual": actual.get(key),
             "match": registered.get(key) == actual.get(key)} for key in IDENTITY_KEYS]


def _git(root: Path) -> dict:
    """기준 커밋과 미커밋 변경 목록(소스 해시와 함께 실행 코드를 고정)."""
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                                  check=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"head": run("rev-parse", "HEAD"), "status": run("status", "--porcelain")}


def preflight(data_root, manifest_path, config: StudyConfig, *, registered=None) -> dict:
    """장부를 열기 전의 대조. 하나라도 다르면 InputIntegrityError."""
    root, manifest_path = Path(data_root).resolve(), Path(manifest_path).resolve()
    registered = dict(config.registered if registered is None else registered)
    actual = actual_inputs(root, manifest_path, config)
    table = identity_table(registered, actual)
    mismatched = [row["item"] for row in table if not row["match"]]
    if mismatched or actual["synthetic"] is not False:
        raise InputIntegrityError(f"registered inputs differ: {mismatched or ['synthetic']}")
    frozen = frozen_identity(config)
    _check_frozen_plan(root, config, frozen)
    basis = _check_price_basis(root, manifest_path, config)
    source = source_identity()
    here = Path(__file__).resolve().parent
    return {"inputs": table, "registered": registered, "price_basis_by_symbol": basis, "config": frozen, "source_sha256": source["source_sha256"],
            "source_files": source["source_files"], "git": _git(here)}


def _check_frozen_plan(root, config, frozen):
    """D3: 고정 설정(명단·시작일·만기·가격 기준 포함)과 다르거나 시작일이 등록 달력의 오프셋과 다르면
    첫 장부를 열기 전에 멈춘다."""
    if frozen["config_modified"]:
        fixed = load_config()
        diffs = [name for name in ("rows", "end", "price_basis", "start_offsets", "costs", "paths", "modes")
                 if getattr(fixed, name) != getattr(config, name)]
        raise InputIdentityError(f"formal run requires the frozen configuration; differs in {diffs or ['other']}")
    calendar = KrCalendar(_calendar(root / "fx" / "calendar_krx.csv"))
    if config.end not in calendar.times:
        raise InputIdentityError(f"registered end {config.end} is not a session of the registered calendar")
    for row in config.rows:
        try:
            check_starts(calendar, row.starts, config.start_offsets)
        except ValueError as error:
            raise InputIdentityError(f"{row.id}: {error}") from error


def _check_price_basis(root, manifest_path, config) -> dict:
    """E1: 등록 명단 모든 종목의 만기까지 전체 일봉(준비기간 포함) 가격 기준이 고정값 하나뿐인지,
    첫 장부 전에 확인한다. 단일 원가격·혼합 기준 모두 입력 식별 오류."""
    found = {}
    try:
        for symbol in _symbols(config):
            _, _, rows, _, _ = _data._source(root, manifest_path, symbol, config.end)
            found[symbol] = sorted({str(row.get("price_basis")) for row in rows})
    finally:
        _data._source.cache_clear()
    wrong = {symbol: bases for symbol, bases in found.items() if bases != [config.price_basis]}
    if wrong:
        raise InputIdentityError(f"price basis differs from the registered {config.price_basis}: {wrong}")
    return {symbol: bases[0] for symbol, bases in found.items()}


def _formal_identity(config, output, checked) -> dict:
    """정식 실행 표시: 이 모듈의 입력기로 읽은 시작일만 대상. 출처는 사전 대조의 실제 값과도 같아야 한다."""
    sources = _sources(output)
    actual = {row["item"]: row["actual"] for row in checked["inputs"]}
    pinned = checked["registered"]
    frozen = frozen_identity(config)
    same = ("input_manifest_sha256", "daily_sha256", "metadata_sha256", "calendar_sha256", "price_bundle")
    checks = {"via_formal_entrypoint": True,
              "config_unmodified": not frozen["config_modified"],
              "registered_inputs_frozen": pinned == load_config().registered,
              "loaded_starts": bool(sources),
              "manifest_registered": bool(sources) and all(
                  p.get("input_manifest_sha256") == pinned.get("input_manifest_sha256") for p in sources),
              "sources_match_preflight": all(p.get(key) == actual.get(key) for p in sources for key in same),
              "manifest_not_synthetic": bool(sources) and all(p.get("manifest_synthetic") is False for p in sources),
              "no_extra_inputs": all(not (p.get("extra_inputs") or {}).get("present", True) for p in sources),
              "price_basis_registered": all(p.get("price_basis") == config.price_basis for p in sources)}
    return {**frozen, "checks": checks, "formal_run": all(checks.values()), "preflight": checked}


def run_formal(data_root, manifest_path, config: StudyConfig, *, registered=None) -> dict:
    """등록 파일에서 직접 입력을 구성해 실행한다(외부에서 만든 EtfInputs·입력기를 받지 않는다).
    registered를 넘기면(합성 시험용) 대조는 그 값으로 하지만 registered_inputs_frozen=False라 정식이 아니다."""
    checked = preflight(data_root, manifest_path, config, registered=registered)
    _data._source.cache_clear()  # D1: 앞선 호출의 캐시·반환 객체를 다음 정식 실행에 쓰지 않는다

    def load(symbol, row, start):
        return load_etf(data_root, manifest_path, symbol, start, config.end)

    try:
        output = execute_plan(load, config, context=_context(config, checked["registered"]))
    finally:
        _data._source.cache_clear()
    output["run_identity"] = _formal_identity(config, output, checked)
    return output
