"""ETF 연구 프로토콜(M0 숫자 고정) 읽기·해시·검증.

프로토콜 JSON은 탐색 전에 정한 숫자·규칙이다. 해시는 키 순서와 공백에 상관없이 같은 내용이면 같게 나오도록
정규화한 JSON(UTF-8, 키 정렬)으로 계산한다. 아직 정하지 않은 값은 "decide_by:<단계>" 문자열로 두고,
규칙 고정(M6) 때는 남은 값이 없어야 한다.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterator

DEFAULT_PATH = Path(__file__).with_name("protocol_v1.json")
OPEN_PREFIX = "decide_by:"
REQUIRED_SECTIONS = ("universe", "portfolio", "baskets", "schedule", "fx", "execution", "cash", "tax",
                     "baseline", "search", "success", "holdout")
_MISSING = object()
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# (경로, 하한, 상한, 정수 여부). 닫힌 구간.
NUMBERS = (
    ("universe.min_history_years_before_start", 0, 30, False),
    ("universe.liquidity.window_sessions", 1, 250, True),
    ("universe.liquidity.floor.KR", 0, 1e12, False),
    ("universe.liquidity.floor.US", 0, 1e10, False),
    ("portfolio.initial_capital_krw", 1e6, 1e12, False),
    ("baskets.leverage_mix.mixed_max_x2_share", 0, 1, False),
    ("baskets.leverage_mix.mixed_min_x2_count", 1, 100, True),
    ("baskets.generation_seed", 0, 2**32 - 1, True),
    ("fx.conversion.cost_one_way", 0, 0.05, False),
    ("fx.conversion.cost_stress_one_way", 0, 0.05, False),
    ("fx.max_publication_age_days", 0, 30, True),
    ("fx.max_closure_publication_age_days", 0, 30, True),
    ("execution.participation.max_order_to_adv20", 1e-6, 1, False),
    ("execution.fill.penetration_ticks", 0, 10, True),
    ("execution.fill.slippage_ticks", 0, 10, True),
    ("execution.volatility.min_observations", 20, 2500, True),
    ("execution.costs_one_way.KR_commission", 0, 0.05, False),
    ("execution.costs_one_way.US_commission", 0, 0.05, False),
    ("execution.costs_one_way.KRX_GOLD_commission", 0, 0.05, False),
    ("execution.costs_one_way.stress_one_way_all", 0, 0.05, False),
    ("cash.interest_primary", 0, 0.2, False),
    ("success.joint_improvement_share_min", 0, 1, False),
    ("success.min_effect.dCAGR_pp_per_year", 0, 100, False),
    ("success.min_effect.dMDD_pp", 0, 100, False),
    ("success.bootstrap.block_sessions", 1, 1000, True),
    ("success.bootstrap.resamples", 100, 1_000_000, True),
    ("success.bootstrap.seed", 0, 2**32 - 1, True),
    ("success.forward_test_months_min", 1, 120, True),
    ("holdout.fraction", 0.05, 0.95, False),
    ("holdout.seed", 0, 2**32 - 1, True),
    ("holdout.min_groups_per_stratum", 1, 1000, True),
    ("holdout.min_holdout_groups_for_subgroup_report", 1, 1000, True),
    ("holdout.correlation_merge.threshold", 0.5, 1, False),
    ("holdout.correlation_merge.sensitivity_report_threshold", 0.5, 1, False),
    ("holdout.correlation_merge.min_overlap_sessions", 20, 10000, True),
)
# 판정·실행 정의가 빠지면 나중에 결과를 보고 정하게 되므로 반드시 있어야 하는 글 항목.
TEXTS = (
    "success.comparison", "success.unit", "success.population", "success.joint_improvement_share_definition",
    "success.min_effect_applies_to", "success.intraday_paths_rule", "success.stress_rule",
    "success.bootstrap.method", "success.bootstrap.statistic", "baseline.primary", "baseline.exposure_target",
    "baseline.weights", "baseline.rebalance", "baseline.learning_window", "fx.conversion.initial",
    "fx.conversion.settlement", "portfolio.currency_cash_target", "execution.volatility.estimator",
    "search.grid", "holdout.unit", "holdout.freeze",
)
ENUMS = {
    "status": {"draft", "frozen"},
    "universe.gold_family_etfs": {"excluded_from_baskets", "allowed"},
    "tax.judgment": {"pre_tax_only_until_d7_data", "after_tax"},
    "fx.conversion.frequency": {"monthly", "quarterly"},
}
LISTS = ("execution.intraday_paths", "success.bootstrap.block_sensitivity", "success.stress_applies")


def load(path: str | Path = DEFAULT_PATH) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    validate(data)
    return data


def _reject_constant(name: str):
    raise ValueError(f"JSON에 {name}를 쓸 수 없습니다")


def canonical_bytes(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def protocol_hash(data: dict) -> str:
    return hashlib.sha256(canonical_bytes(data)).hexdigest()


def _walk(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk(value, f"{path}[{i}]")
    else:
        yield path, node


def open_items(data: dict) -> dict[str, str]:
    """아직 정하지 않은 값: {경로: 정할 단계와 설명}."""
    return {p: v[len(OPEN_PREFIX):].strip() for p, v in _walk(data)
            if isinstance(v, str) and v.startswith(OPEN_PREFIX)}


def _get(data: dict, dotted: str):
    node: Any = data
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(data: dict) -> None:
    """구조·자료형·범위·관계 검사. 값이 연구적으로 옳은지는 판단하지 않는다."""
    if not isinstance(data, dict):
        raise ValueError("프로토콜은 JSON 객체여야 합니다")
    missing = [s for s in REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ValueError(f"프로토콜 절이 없습니다: {missing}")
    for path, value in _walk(data):
        if value is None:
            raise ValueError(f"{path}가 비어 있습니다. 미정 값은 '{OPEN_PREFIX}<단계>'로 적습니다")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path}가 유한한 숫자가 아닙니다")
        if isinstance(value, str):
            if not value.strip():
                raise ValueError(f"{path}가 빈 문자열입니다")
            if value.startswith(OPEN_PREFIX) and not value[len(OPEN_PREFIX):].strip():
                raise ValueError(f"{path}: 미정 값에 정할 단계가 없습니다")
    for dotted, low, high, integer in NUMBERS:
        value = _get(data, dotted)
        if value is _MISSING:
            raise ValueError(f"프로토콜에 {dotted}가 없습니다")
        if not _is_number(value) or not (low <= value <= high) or (integer and not float(value).is_integer()):
            kind = "정수" if integer else "숫자"
            raise ValueError(f"{dotted}={value!r}는 [{low}, {high}] 범위의 {kind}여야 합니다")
    for dotted in TEXTS:
        if not isinstance(_get(data, dotted), str):
            raise ValueError(f"프로토콜에 글 항목 {dotted}가 없습니다")
    for dotted, allowed in ENUMS.items():
        value = _get(data, dotted)
        if value not in allowed:
            raise ValueError(f"{dotted}={value!r}는 {sorted(allowed)} 중 하나여야 합니다")
    if not _HHMM.match(str(_get(data, "schedule.evaluation_time_kst"))):
        raise ValueError("schedule.evaluation_time_kst는 HH:MM 형식이어야 합니다")
    window = _get(data, "holdout.correlation_merge.window")
    if not (isinstance(window, dict) and all(_DATE.match(str(window.get(k, ""))) for k in ("start", "end"))
            and window["start"] < window["end"]):
        raise ValueError("holdout.correlation_merge.window는 start < end인 YYYY-MM-DD 쌍이어야 합니다")
    for dotted in LISTS:
        value = _get(data, dotted)
        if not isinstance(value, list) or not value:
            raise ValueError(f"{dotted}는 비어 있지 않은 목록이어야 합니다")
    for dotted in _get(data, "success.stress_applies"):
        if not isinstance(dotted, str) or not _is_number(_get(data, dotted)):
            raise ValueError(f"success.stress_applies의 {dotted!r}가 숫자 항목을 가리키지 않습니다")
    if _get(data, "fx.conversion.cost_stress_one_way") < _get(data, "fx.conversion.cost_one_way"):
        raise ValueError("환전 스트레스 비용이 주 비용보다 작습니다")
    if _get(data, "fx.max_closure_publication_age_days") < _get(data, "fx.max_publication_age_days"):
        raise ValueError("장기 휴장 환율 한도가 기본 한도보다 작습니다")
    commissions = [_get(data, f"execution.costs_one_way.{k}")
                   for k in ("KR_commission", "US_commission", "KRX_GOLD_commission")]
    if _get(data, "execution.costs_one_way.stress_one_way_all") < max(commissions):
        raise ValueError("거래비용 스트레스가 주 수수료보다 작습니다")
    if (_get(data, "holdout.correlation_merge.sensitivity_report_threshold")
            > _get(data, "holdout.correlation_merge.threshold")):
        raise ValueError("민감도 보고 상관 기준은 주 기준 이하여야 합니다")
    if data.get("engine_schema") == "etf-m2-v1":
        _validate_market_reference(data)
    if "configuration_file" in data["search"]:
        _validate_search_reference(data)
    if data["status"] == "frozen" and open_items(data):
        raise ValueError(f"고정된 프로토콜에 미정 값이 남았습니다: {sorted(open_items(data))}")


def _validate_market_reference(data: dict) -> None:
    reference = _get(data, "execution.market_rules")
    if not isinstance(reference, dict) or reference.get("file") != "market_rules_v1.json":
        raise ValueError("M2 시장 규칙 파일 참조가 잘못됐습니다")
    path = Path(__file__).with_name(reference["file"])
    if reference.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("M2 시장 규칙 해시가 프로토콜과 다릅니다")
    rules = json.loads(path.read_text(encoding="utf-8"))
    for market, key in (("KR", "KR"), ("US", "US"), ("KRX_GOLD", "GOLD")):
        if str(_get(data, f"execution.costs_one_way.{market}_commission")) != rules["commission_rates"][key]:
            raise ValueError("프로토콜과 M2 시장 규칙의 수수료가 다릅니다")


def _validate_search_reference(data):
    if data["search"]["configuration_file"] != "m5_search_v1.json":
        raise ValueError("M5 탐색 설정 파일 참조가 잘못됐습니다")
    path = Path(__file__).with_name("m5_search_v1.json")
    if data["search"].get("configuration_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("M5 탐색 설정 해시가 프로토콜과 다릅니다")
    config = json.loads(path.read_text(encoding="utf-8"))
    if any(data["baskets"][key] != config[key] for key in ("market_mix_strata", "k_values")):
        raise ValueError("M5 시장 구성과 프로토콜이 다릅니다")
    if (data["schedule"]["start_dates"]["horizon_years"] != config["outer_evaluation_years"]
            or data["fx"]["max_publication_age_days"] != config["pilot"]["max_fx_age_days"]):
        raise ValueError("M5 기간/환율 제한과 프로토콜이 다릅니다")


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="프로토콜 검사와 해시 출력")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PATH))
    args = parser.parse_args(argv)
    data = load(args.path)
    print(json.dumps({"version": data.get("version"), "status": data["status"], "sha256": protocol_hash(data),
                      "open_items": open_items(data)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
