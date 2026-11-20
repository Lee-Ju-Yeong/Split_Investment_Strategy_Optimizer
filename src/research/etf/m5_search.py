"""정식 M5 탐색 계획: 시작일·시점별 후보군·고정 시드 바스켓·실행 단위. 성과를 읽지 않는다."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import date
import hashlib
import json
from pathlib import Path

from . import protocol
from .engine_inputs import _calendar, _history_ok, _rows, _sha, _snapshot, _stamp
from .engine_types import ZERO, decimal
from .m5_inputs import read_manifest
from .m5_parameters import CONFIG_PATH, grid_candidates, load_config


FORMAL_PATH = Path(__file__).with_name("m5_formal_v1.json")
SCHEMA = "etf-m5-plan-v1"
LIQUIDITY_SESSIONS = 20
DRAW_ATTEMPTS = 50


def load_formal():
    formal = json.loads(FORMAL_PATH.read_text(encoding="utf-8"))
    count = formal.get("baskets_per_stratum")
    if (formal.get("schema") != "etf-m5-formal-v1" or type(count) is not int or count < 1
            or formal.get("status") not in {"draft_preperformance", "frozen"}
            or formal.get("selection") != "random" or formal.get("leverage_mix") != "x1_only"
            or formal.get("learning_runs") != 2 or type(formal.get("posthoc_diagnostic")) is not bool):
        raise ValueError("invalid M5 formal configuration")
    rule, config = formal.get("selection_rule") or {}, load_config()
    if (any(type(rule.get(name)) is not int or rule[name] < minimum for name, minimum in
            (("outer_train_years", 1), ("embargo_starts", 0), ("pbo_blocks", 2)))
            or rule["pbo_blocks"] % 2 or any(rule.get(name) != config[key] for name, key in (
                ("outer_train_years", "outer_train_years"), ("outer_test_years", "outer_evaluation_years"),
                ("inner_train_years", "inner_train_years"), ("inner_test_years", "inner_evaluation_years")))):
        raise ValueError("invalid M5 selection rule")
    return formal


def start_windows(union, first, data_end):
    """매 1월·7월 첫 통합 거래일부터 다음 해 같은 달 시작일 직전 거래일까지. 구간은 겹치지 않는다."""
    firsts = {}
    for day in union:
        if day.month in (1, 7):
            firsts.setdefault((day.year, day.month), day)
    windows = []
    for (year, month), start in sorted(firsts.items()):
        following = firsts.get((year + 1, month))
        if (year, month) < first or following is None:
            continue
        end = union[bisect_left(union, following) - 1]
        if end <= data_end:
            windows.append((start, end))
    return windows


def _month(value):
    year, month = value.split("-")
    return int(year), int(month)


def _previous(start, starts, years):
    return next((day for day in starts if (day.year, day.month) == (start.year - years, start.month)), None)


def _members(root, manifest, metadata, gold_symbol):
    """원본 탐색 분할의 확인된 1배 상품만. 보류 상품은 이름도 계획에 넣지 않는다."""
    pool = {"KR": {}, "US": {}, "GOLD": {}}
    for row in _rows(root / manifest["split_csv"]):
        market, symbol = row["market"], row["symbol"]
        if market == "GOLD" and row["split"] == "both" and symbol == gold_symbol:
            pool["GOLD"][symbol] = row["group"]
            continue
        detail = metadata.get((market, symbol), {})
        if (row["split"] == "explore" and market in ("KR", "US") and row.get("group")
                and detail.get("d1_eligible") == "True" and detail.get("needs_review") != "True"
                and detail.get("leverage_multiple") in {"1", "1.0"}):
            pool[market][symbol] = row["group"]
    return pool


def _needed(starts, sessions):
    days = set()
    for start in starts:
        index = bisect_left(sessions, start)
        days.update(day.isoformat() for day in sessions[max(0, index - LIQUIDITY_SESSIONS):index])
    return days


def _scan(root, bundle, pool, needed):
    """시작 전 20세션과 최초 가격일만 남긴다. 시작 이후 가격은 후보 판정에 쓰지 않는다."""
    seen = {market: {symbol: {"first": None, "rows": {}} for symbol in names} for market, names in pool.items()}
    for market, names in pool.items():
        if not names:
            continue
        path = (root / "krx_gold" / "gold_9999_1kg_daily.csv" if market == "GOLD"
                else bundle / f"{market.lower()}_daily.csv.gz")
        for row in _rows(path):
            symbol = next(iter(names)) if market == "GOLD" else row["symbol"]
            if symbol not in names:
                continue
            item, day = seen[market][symbol], row["date"]
            if item["first"] is None or day < item["first"]:
                item["first"] = day
            if day in needed[market]:
                item["rows"][day] = (row["close"], row["volume"])
    return seen


def eligible(item, start, sessions, floor):
    """연결부와 같은 시작 전 조건: 3년 이력, 직전 20세션 완비, 평균 거래대금 하한(금은 하한 없음)."""
    index = bisect_left(sessions, start)
    prior = [day.isoformat() for day in sessions[max(0, index - LIQUIDITY_SESSIONS):index]]
    if item["first"] is None or len(prior) != LIQUIDITY_SESSIONS or any(day not in item["rows"] for day in prior):
        return False
    if not _history_ok(date.fromisoformat(item["first"]), start):
        return False
    if floor is None:
        return True
    traded = sum((decimal(item["rows"][day][0]) * decimal(item["rows"][day][1]) for day in prior), ZERO)
    return traded / LIQUIDITY_SESSIONS >= floor


def _rank(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def draw_basket(pool, stratum, seed, key, gold_symbol):
    """고정 시드 순위로 시장별 개수를 채운다. 같은 지수 묶음은 바스켓에 한 번만 쓴다."""
    chosen, groups = [], set()
    for market in ("KR", "US"):
        picked = []
        for symbol, group in sorted(pool[market], key=lambda item: _rank(seed, key, item[0])):
            if len(picked) == stratum.get(market, 0):
                break
            if group not in groups:
                picked.append(symbol)
                groups.add(group)
        if len(picked) != stratum.get(market, 0):
            return None
        chosen.extend(picked)
    return chosen + ([gold_symbol] if stratum.get("GOLD") else [])


def _stratum_key(stratum):
    return "+".join(f"{market}{stratum[market]}" for market in ("KR", "US", "GOLD") if stratum.get(market))


class _Universe:
    def __init__(self, seen, pool, calendars, floors):
        self.seen, self.pool, self.calendars, self.floors = seen, pool, calendars, floors
        self.cache = {}
        self.market = {symbol: name for name, members in pool.items() for symbol in members}

    def ok(self, market, symbol, start):
        key = (market, symbol, start)
        if key not in self.cache:
            self.cache[key] = eligible(self.seen[market][symbol], start, self.calendars[market], self.floors[market])
        return self.cache[key]

    def pool_at(self, starts):
        return {market: [(symbol, group) for symbol, group in sorted(self.pool[market].items())
                         if all(self.ok(market, symbol, start) for start in starts)] for market in ("KR", "US")}

    def basket_ok(self, symbols, start):
        return all(self.ok(self.market[symbol], symbol, start) for symbol in symbols)


def _allowed(stratum, windows, proto):
    first = date(*_month(proto["schedule"]["start_dates"]["kr_containing_first"]), 1)
    gold = date.fromisoformat(proto["baskets"]["gold_first_eligible"])
    return [(start, end) for start, end in windows
            if (not (stratum.get("KR") or stratum.get("GOLD")) or start >= first)
            and (not stratum.get("GOLD") or start >= gold)]


def _stratum_baskets(stratum, windows, universe, proto, count):
    key, seed = _stratum_key(stratum), proto["baskets"]["generation_seed"]
    gold = proto["baskets"]["gold_symbol"]
    allowed = _allowed(stratum, windows, proto)
    starts = [start for start, _ in allowed]
    anchors = [start for start in starts if _previous(start, starts, 1) and _previous(start, starts, 2)]
    if stratum.get("GOLD"):
        anchors = [start for start in anchors if gold in universe.pool["GOLD"] and all(
            universe.ok("GOLD", gold, day) for day in (start, _previous(start, starts, 1), _previous(start, starts, 2)))]
    baskets, taken = [], set()
    for draw in range(count if anchors else 0):
        anchor = anchors[draw * len(anchors) // count]
        pool = universe.pool_at([anchor, _previous(anchor, starts, 1), _previous(anchor, starts, 2)])
        for attempt in range(DRAW_ATTEMPTS):
            symbols = draw_basket(pool, stratum, seed, f"{key}|{anchor}|{draw}|{attempt}", gold)
            if symbols is None:
                break
            if frozenset(symbols) not in taken:
                taken.add(frozenset(symbols))
                # 기준일 뒤의 정보로 고른 바스켓을 그 전 구간 평가에 쓰지 않는다. 학습 두 해부터만 실행한다.
                earliest = _previous(anchor, starts, 2)
                runs = [(start, end) for start, end in allowed
                        if start >= earliest and universe.basket_ok(symbols, start)]
                days = [start for start, _ in runs]
                baskets.append({"stratum": key, "symbols": symbols, "anchor": anchor.isoformat(),
                                "runs": [{"start": start.isoformat(), "end": end.isoformat()} for start, end in runs],
                                "evaluations": [start.isoformat() for start in days
                                                if _previous(start, days, 1) and _previous(start, days, 2)]})
                break
    return baskets


def _data_end(root, calendars):
    published = max(_stamp(row["available_at_utc"]) for row in _rows(root / "fx" / "fx_usdkrw_smbs_mar.csv"))
    return min(published.date(), max(calendars["KR"]), max(calendars["US"]))


def build_plan(data_root, manifest_path, *, first_start=None, last_start=None,
               pilot_candidates=False, baskets_per_stratum=None):
    """계획만 만든다. 후보 판정은 각 시작일 전 가격·거래량만 쓰고 전략을 실행하지 않는다."""
    root = Path(data_root).resolve()
    manifest = read_manifest(root, manifest_path, [])
    proto, config, formal = protocol.load(), load_config(), load_formal()
    _, _, metadata, bundle = _snapshot(root, root / manifest["split_csv"], [])
    krx, nyse = (_calendar(root / "fx" / f"calendar_{name}.csv") for name in ("krx", "nyse"))
    calendars = {"KR": sorted(krx), "US": sorted(nyse), "GOLD": sorted(krx)}
    schedule = proto["schedule"]["start_dates"]
    if schedule["horizon_years"] != 1:
        raise ValueError("the formal planner supports one-year runs only")
    windows = start_windows(sorted(set(krx) | set(nyse)), _month(schedule["first"]), _data_end(root, calendars))
    windows = [(start, end) for start, end in windows
               if (first_start is None or start >= first_start) and (last_start is None or start <= last_start)]
    pool = _members(root, manifest, metadata, proto["baskets"]["gold_symbol"])
    starts = [start for start, _ in windows]
    needed = {market: _needed(starts, calendars[market]) for market in pool}
    floors = {"KR": decimal(str(proto["universe"]["liquidity"]["floor"]["KR"])),
              "US": decimal(str(proto["universe"]["liquidity"]["floor"]["US"])), "GOLD": None}
    universe = _Universe(_scan(root, bundle, pool, needed), pool, calendars, floors)
    count = formal["baskets_per_stratum"] if baskets_per_stratum is None else baskets_per_stratum
    baskets, shortfalls = [], {}
    for stratum in config["market_mix_strata"]:
        made = _stratum_baskets(stratum, windows, universe, proto, count)
        if len(made) != count:
            shortfalls[_stratum_key(stratum)] = {"requested": count, "generated": len(made)}
        baskets.extend(made)
    for index, basket in enumerate(baskets, 1):
        basket["id"] = f"b{index:03d}"
    candidates = config["pilot"]["candidates"] if pilot_candidates else grid_candidates(config)
    overrides = first_start is not None or last_start is not None or pilot_candidates or baskets_per_stratum is not None
    trials = len(candidates) * len(config["pilot"]["paths"])
    # 확정 전 초안 설정이나 연결 검사용 축소 계획은 정식 실행으로 표시하지 않는다.
    scope = "smoke_not_formal" if overrides else "formal" if formal["status"] == "frozen" else "draft_not_formal"
    return {"schema": SCHEMA, "scope": scope,
            "synthetic": manifest["synthetic"], "protocol_sha256": protocol.protocol_hash(proto),
            "configuration_sha256": _sha(CONFIG_PATH), "formal_sha256": _sha(FORMAL_PATH),
            "input_manifest_sha256": _sha(Path(manifest_path)),
            "candidates": candidates, "paths": config["pilot"]["paths"],
            "fx": {"max_age_days": proto["fx"]["max_publication_age_days"],
                   "closure_age_days": proto["fx"]["max_closure_publication_age_days"]},
            "posthoc_diagnostic": formal["posthoc_diagnostic"],
            "windows": [{"start": start.isoformat(), "end": end.isoformat()} for start, end in windows],
            "baskets": baskets, "shortfalls": shortfalls,
            "counts": {"baskets": len(baskets), "units": sum(len(b["runs"]) for b in baskets),
                       "evaluations": sum(len(b["evaluations"]) for b in baskets),
                       "trials": trials * sum(len(b["runs"]) for b in baskets),
                       "comparisons": trials * sum(len(b["evaluations"]) for b in baskets)},
            "statistical_success_evaluated": False, "holdout_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="새 계획 JSON 경로")
    parser.add_argument("--first-start", type=date.fromisoformat, help="연결 검사용: 이 날짜 이후 시작일만")
    parser.add_argument("--last-start", type=date.fromisoformat, help="연결 검사용: 이 날짜 이전 시작일만")
    parser.add_argument("--pilot-candidates", action="store_true", help="연결 검사용: 고정한 두 설정만")
    parser.add_argument("--baskets-per-stratum", type=int, help="연결 검사용: 층별 바스켓 수")
    args = parser.parse_args()
    plan = build_plan(args.data_root, args.manifest, first_start=args.first_start, last_start=args.last_start,
                      pilot_candidates=args.pilot_candidates, baskets_per_stratum=args.baskets_per_stratum)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"plan": str(args.output), "sha256": _sha(args.output), "scope": plan["scope"],
                      "counts": plan["counts"], "shortfalls": plan["shortfalls"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
