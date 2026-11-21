"""Freeze R2 samples and dates using only price availability before formation."""
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path

from . import protocol
from .engine_inputs import _calendar, _sha, _snapshot
from .m5_inputs import read_manifest
from .m5_search import _Universe, _data_end, _members, _needed, _rank, _scan


SEED = "etf-followup-r2-v1"
SCHEMA = "etf-followup-r2-v1"
PATHS = ("O-H-L-C", "O-L-H-C")
DEPTHS = (3, 5, 6, 7, 8, 9, 10)
SIZES = (4, 6, 8)
SLEEVES = {
    "kr_equity": ("KR", "국내주식", "한국"),
    "us_equity": ("US", "해외주식", "미국"),
    "us_bond": ("US", "채권", "미국"),
}
MIXES = {
    "kr_equity": {"kr_equity": "1"},
    "us_equity": {"us_equity": "1"},
    "kr_us_equity": {"kr_equity": "0.5", "us_equity": "0.5"},
    "kr_us_balanced": {"kr_equity": "0.5", "us_equity": "0.25", "us_bond": "0.25"},
}
PHASES = ("sizing", "spacing", "reentry", "forward")


@dataclass(frozen=True, order=True)
class Setting:
    size: int
    levels: int
    kg: int = 3
    kp: int = 3
    entry: str = "monthly_fixed"
    budget: str = "0.80"
    stress: bool = False


def quarter_windows(calendars):
    firsts = {}
    for day in sorted(calendars["KR"]):
        if day.month in (1, 4, 7, 10):
            firsts.setdefault((day.year, day.month), day)
    union = sorted(set(calendars["KR"]) | set(calendars["US"]))
    windows = {}
    for (year, month), start in firsts.items():
        if 2018 <= year <= 2025 and (year + 1, month) in firsts:
            successor = firsts[(year + 1, month)]
            windows[f"{year}-{month:02}"] = (start, union[bisect_left(union, successor) - 1])
    return windows


def sleeve_counts(mix, size, family):
    if len(MIXES[mix]) != 3:
        return dict.fromkeys(MIXES[mix], size // len(MIXES[mix]))
    equity = 2 if size == 8 or (size == 6 and family % 2 == 0) else 1
    return {"kr_equity": size // 2, "us_equity": equity, "us_bond": size // 2 - equity}


def weighted_basket(mix, parent, size, family):
    weights = {}
    for sleeve, count in sleeve_counts(mix, size, family).items():
        symbols = parent[sleeve][:count]
        with localcontext(prec=40):
            allocation = Decimal(MIXES[mix][sleeve])
            piece = allocation / count
            weights.update(dict.fromkeys(symbols[:-1], str(piece)))
            weights[symbols[-1]] = str(allocation - piece * (count - 1))
    return weights


def draw_parent(eligible, metadata, year, mix, family, attempt):
    seen, parent = set(), {}
    for sleeve, count in sleeve_counts(mix, 8, family).items():
        market, asset, region = SLEEVES[sleeve]
        ordered = sorted(eligible[market], key=lambda row: _rank(SEED, year, mix, family, attempt, sleeve, row[0]))
        picks = []
        for symbol, group in ordered:
            detail = metadata[market, symbol]
            if group in seen or (detail["asset_class"], detail["region"]) != (asset, region):
                continue
            picks.append(symbol)
            seen.add(group)
            if len(picks) == count:
                break
        if len(picks) != count:
            raise ValueError(f"{year}/{mix}/{sleeve}: insufficient distinct eligible index groups")
        parent[sleeve] = picks
    return parent


def draw_families(eligible, metadata, year, mix):
    seen = set()
    for number in range(4):
        for attempt in range(50):
            parent = draw_parent(eligible, metadata, year, mix, number, attempt)
            baskets = {str(size): weighted_basket(mix, parent, size, number) for size in SIZES}
            identities = {frozenset(basket) for basket in baskets.values()}
            if not identities & seen:
                seen.update(identities)
                yield {"id": f"{year}-{mix}-{number + 1}", "year": year, "mix": mix, "baskets": baskets}
                break
        else:
            raise ValueError(f"{year}/{mix}: four distinct nested families required")


def make_plan(root, manifest_path):
    manifest = read_manifest(root, manifest_path, [])
    proto = protocol.load()
    _, _, metadata, bundle = _snapshot(root, root / manifest["split_csv"], [])
    calendars = {m: _calendar(root / "fx" / f"calendar_{name}.csv")
                 for m, name in (("KR", "krx"), ("US", "nyse"))}
    dates = {m: sorted(series) for m, series in calendars.items()}
    windows = quarter_windows(dates)
    if len(windows) != 32 or max(end for _, end in windows.values()) > _data_end(root, dates):
        raise ValueError("pinned data does not cover the final comparison")
    pool = _members(root, manifest, metadata, proto["baskets"]["gold_symbol"])
    del pool["GOLD"]
    anchors = [windows[f"{year}-01"][0] for year in range(2018, 2026)]
    needed = {market: _needed(anchors, dates[market]) for market in pool}
    floors = {market: Decimal(str(proto["universe"]["liquidity"]["floor"][market])) for market in pool}
    universe = _Universe(_scan(root, bundle, pool, needed), pool, dates, floors)
    families = []
    for year in range(2020, 2026):
        eligible = universe.pool_at([windows[f"{year - lag}-01"][0] for lag in range(3)])
        for mix in MIXES:
            families.extend(draw_families(eligible, metadata, year, mix))
    return {"schema": SCHEMA, "seed": SEED, "synthetic": manifest["synthetic"],
            "manifest_sha256": _sha(manifest_path), "protocol_sha256": protocol.protocol_hash(proto),
            "families": families, "windows": {key: [s.isoformat() for s in span] for key, span in windows.items()},
            "development": [key for key in windows if "2020-01" <= key <= "2023-01"],
            "forward": [key for key in windows if "2024-01" <= key]}
