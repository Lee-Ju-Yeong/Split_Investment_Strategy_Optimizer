"""탐색용·보류용 ETF 분할(계획 3.2절 1단계).

결과를 보기 전에 ETF 단위로 나눈다. 같은 기초지수나 거의 같은 노출의 ETF는 한 묶음으로 보고 묶음째 한쪽에만 둔다.
1. 분류표의 `index_family`에서 출발한다(국내·미국 상장을 가로지른 묶음 포함).
2. 같은 시장 안에서 일간 수익률 상관이 기준 이상(겹치는 거래일 하한 충족)인 상품의 묶음을 합친다.
3. 시장 묶음·자산군·이력 길이로 층을 나누고, 고정 시드로 층마다 정해진 비율을 보류용으로 뽑는다.
KRX 금현물은 상품이 하나라 나누지 않고 양쪽에서 쓴다(both).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import protocol as protocol_mod

SPLIT_COLUMNS = ["market", "symbol", "name", "index_family", "group", "stratum", "first_date", "split"]
EXPLORE, HOLDOUT, BOTH = "explore", "holdout", "both"


# ---------------------------------------------------------------- 후보
def eligible(metadata: pd.DataFrame, proto: dict) -> pd.DataFrame:
    """가격이 있고 D1을 통과한 상품. 금 묶음 ETF와 단일종목 레버리지는 바스켓 후보에서 뺀다(프로토콜)."""
    frame = metadata[metadata["has_price_history"].astype(bool) & metadata["d1_eligible"].astype(bool)].copy()
    if proto["universe"]["gold_family_etfs"] == "excluded_from_baskets":
        frame = frame[frame["index_family"] != "GOLD"]
    lev = pd.to_numeric(frame["leverage_multiple"], errors="coerce")
    frame = frame[~(frame["is_single_stock"].astype(bool) & (lev > 1))]
    if frame["index_family"].isna().any() or (frame["index_family"].astype(str).str.strip() == "").any():
        raise ValueError("index_family가 빈 후보가 있습니다")
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------- 상관
def daily_returns(daily: pd.DataFrame, symbols, start=None, end=None) -> pd.DataFrame:
    """종가 일간 수익률(날짜×종목).

    날짜 축은 각 종목의 원래 행을 그대로 쓴다. 거래량 0이거나 가격이 0 이하인 날은 종가를 비워 두므로,
    그날과 그다음 날 수익률은 다른 종목 구성과 상관없이 항상 결측이다. 무한대 수익률도 결측으로 둔다.
    """
    part = daily[daily["symbol"].isin(set(symbols))]
    if start is not None:
        part = part[part["date"] >= pd.Timestamp(start)]
    if end is not None:
        part = part[part["date"] <= pd.Timestamp(end)]
    part = part.assign(close=part["close"].where((part["volume"] > 0) & (part["close"] > 0)))
    pieces = {}
    for symbol, rows in part.sort_values("date").groupby("symbol", sort=True):
        series = rows.drop_duplicates("date", keep="last").set_index("date")["close"]
        pieces[symbol] = series.pct_change(fill_method=None)
    returns = pd.DataFrame(pieces).sort_index()
    return returns.replace([np.inf, -np.inf], np.nan)


MIN_PAIR_STD = 1e-8   # 짝별 표준편차가 이보다 작으면(사실상 상수) 상관을 정의하지 않는다


def pairwise_correlation(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """결측을 짝마다 따로 빼는 피어슨 상관과 겹친 표본 수. 행렬 곱으로 한 번에 계산한다.

    제곱 평균에서 평균 제곱을 빼는 식은 값이 거의 같을 때 반올림 오차가 커지므로, 먼저 각 열을 자기 전체
    평균으로 옮겨 놓고 계산한다. 짝별 분산이 사실상 0이면 결측으로 둔다.
    """
    x = returns.to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    mask = ~np.isnan(x)
    counts = mask.sum(axis=0)
    center = np.where(counts > 0, np.where(mask, x, 0.0).sum(axis=0) / np.maximum(counts, 1), 0.0)
    x = x - center
    x = np.where(mask, x, 0.0)
    m = mask.astype(float)
    n = m.T @ m
    sx = x.T @ m            # sx[i, j]: j가 있는 날의 i 합
    sxx = (x * x).T @ m
    sxy = x.T @ x
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_i = sx / n
        mean_j = sx.T / n
        cov = sxy / n - mean_i * mean_j
        var_i = sxx / n - mean_i ** 2
        var_j = sxx.T / n - mean_j ** 2
        corr = cov / np.sqrt(var_i * var_j)
    flat = (var_i < MIN_PAIR_STD ** 2) | (var_j < MIN_PAIR_STD ** 2)
    corr[~np.isfinite(corr) | flat] = np.nan
    return corr, n


class _Union:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        lo, hi = sorted((ra, rb))
        self.parent[hi] = lo
        return True


def correlated_pairs(returns: pd.DataFrame, threshold: float, min_overlap: int) -> list[tuple[str, str, float, int]]:
    corr, n = pairwise_correlation(returns)
    cols = list(returns.columns)
    hits = np.argwhere(np.triu((corr >= threshold) & (n >= min_overlap), k=1))
    return [(cols[i], cols[j], float(corr[i, j]), int(n[i, j])) for i, j in hits]


def merge_families(candidates: pd.DataFrame, pairs) -> tuple[pd.Series, list[dict]]:
    """상관이 높은 짝의 index_family를 합친다. 반환: 종목별 묶음 id, 합친 기록.

    묶음 id는 병합 순서와 무관하게 연결 성분의 구성원만으로 정한다: 종목이 가장 많은 원래 묶음 이름
    (같으면 이름순 앞). 기록은 짝을 정렬한 순서로 남긴다.
    """
    family_of = dict(zip(candidates["symbol"], candidates["index_family"]))
    sizes = candidates["index_family"].value_counts().to_dict()
    uf = _Union(sorted(set(family_of.values())))
    log = []
    for a, b, rho, overlap in sorted(pairs, key=lambda p: (min(p[0], p[1]), max(p[0], p[1]))):
        fa, fb = family_of[a], family_of[b]
        if uf.union(fa, fb):
            log.append({"a": a, "b": b, "family_a": fa, "family_b": fb, "corr": round(rho, 6), "overlap": overlap})
    members: dict[str, list[str]] = {}
    for family in sorted(set(family_of.values())):
        members.setdefault(uf.find(family), []).append(family)
    label = {}
    for component in members.values():
        name = sorted(component, key=lambda f: (-sizes[f], f))[0]
        for family in component:
            label[family] = name
    return candidates["index_family"].map(label), log


# ---------------------------------------------------------------- 층과 배정
def history_bucket(first: pd.Timestamp) -> str:
    """프로토콜 holdout.history_buckets와 같은 구간: long ≤2012, mid 2013–2018, short ≥2019."""
    year = first.year
    if year <= 2012:
        return "long"
    if year <= 2018:
        return "mid"
    return "short"


def strata(frame: pd.DataFrame, proto: dict) -> pd.Series:
    """묶음 단위 층. 가장 세밀한 층은 시장 묶음 | 대표 자산군(최빈, 동률이면 이름순) | 이력 구간이다.

    묶음 수가 holdout.min_groups_per_stratum보다 적은 층은 결과를 보기 전에 정한 순서대로 거칠게 합친다
    (이력 구간 → 자산군을 차례로 뺀다). 끝까지 작은 층은 그대로 두고 요약에서 표시한다.
    """
    minimum = proto["holdout"]["min_groups_per_stratum"]
    levels = {}
    for group, part in frame.groupby("group", sort=True):
        markets = "+".join(sorted(set(part["market"])))
        counts = Counter(part["asset_class"].astype(str))
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        bucket = history_bucket(part["first_date"].min())
        levels[group] = (f"{markets}|{top}|{bucket}", f"{markets}|{top}|*", f"{markets}|*|*")
    out = {}
    for level in range(3):
        pending = {g: lv[level] for g, lv in levels.items() if g not in out}
        counts = Counter(pending.values())
        for group, name in pending.items():
            if counts[name] >= minimum or level == 2:
                out[group] = name
    return pd.Series(out, name="stratum").sort_index()


def assign(stratum_of_group: pd.Series, fraction: float, seed: int) -> pd.Series:
    """층마다 묶음을 섞어 floor(n·비율)개 + 나머지 비율만큼의 확률로 1개를 보류용으로 뽑는다(확률적 반올림)."""
    rng = np.random.default_rng(seed)
    out = {}
    for stratum in sorted(stratum_of_group.unique()):
        groups = sorted(stratum_of_group[stratum_of_group == stratum].index)
        order = rng.permutation(len(groups))
        exact = len(groups) * fraction
        k = int(np.floor(exact)) + int(rng.random() < exact - np.floor(exact))
        for rank, idx in enumerate(order):
            out[groups[idx]] = HOLDOUT if rank < k else EXPLORE
    return pd.Series(out, name="split")


def split(candidates: pd.DataFrame, first_dates: pd.Series, returns_by_market: dict[str, pd.DataFrame],
          proto: dict) -> tuple[pd.DataFrame, dict]:
    cfg = proto["holdout"]
    merge_cfg = cfg["correlation_merge"]
    frame = candidates.copy()
    frame["first_date"] = frame["symbol"].map(first_dates)
    if frame["first_date"].isna().any():
        missing = frame.loc[frame["first_date"].isna(), "symbol"].tolist()[:10]
        raise ValueError(f"가격 첫 거래일이 없는 후보: {missing}")

    def pairs_at(threshold):
        found = []
        for market, returns in sorted(returns_by_market.items()):
            window = merge_cfg["window"]
            returns = returns.loc[(returns.index >= pd.Timestamp(window["start"]))
                                  & (returns.index <= pd.Timestamp(window["end"]))]
            cols = [c for c in returns.columns if c in set(frame.loc[frame["market"] == market, "symbol"])]
            found += correlated_pairs(returns[cols], threshold, merge_cfg["min_overlap_sessions"])
        return found

    main_pairs = pairs_at(merge_cfg["threshold"])
    frame["group"], merge_log = merge_families(frame, main_pairs)
    sens_group, _ = merge_families(frame, pairs_at(merge_cfg["sensitivity_report_threshold"]))

    stratum_of_group = strata(frame, proto)
    frame["stratum"] = frame["group"].map(stratum_of_group)
    frame["split"] = frame["group"].map(assign(stratum_of_group, cfg["fraction"], cfg["seed"]))

    gold = pd.DataFrame([{"market": "GOLD", "symbol": proto["baskets"]["gold_symbol"], "name": "KRX 금현물 99.99 1kg",
                          "index_family": "KRX_GOLD", "group": "KRX_GOLD", "stratum": "GOLD",
                          "first_date": pd.Timestamp("2014-03-24"), "split": BOTH}])
    result = pd.concat([frame[SPLIT_COLUMNS], gold], ignore_index=True)
    result = result.sort_values(["market", "symbol"]).reset_index(drop=True)

    def group_stats(groups: pd.Series) -> dict:
        sizes = groups.value_counts()
        return {"groups": int(sizes.size), "largest_group_symbols": int(sizes.max()),
                "largest_group": str(sizes.idxmax())}

    by_split = result.groupby(["split", "market"]).size().unstack(fill_value=0)
    split_of_group = frame.drop_duplicates("group").set_index("group")["split"]
    strata_summary = {s: {"groups": int((stratum_of_group == s).sum()),
                          "holdout_groups": int((split_of_group.reindex(stratum_of_group[stratum_of_group == s].index)
                                                 == HOLDOUT).sum())}
                      for s in sorted(stratum_of_group.unique())}
    summary = {
        "candidates": int(len(frame)),
        "families_before_merge": int(frame["index_family"].nunique()),
        "merges": len(merge_log),
        "merge_log": merge_log,
        "after_merge": group_stats(frame["group"]),
        "at_sensitivity_threshold": {"threshold": merge_cfg["sensitivity_report_threshold"], **group_stats(sens_group)},
        "symbols_by_split_market": {s: {m: int(v) for m, v in row.items()} for s, row in by_split.iterrows()},
        "groups_by_split": {k: int(v) for k, v in frame.drop_duplicates("group")["split"].value_counts().items()},
        "strata": strata_summary,
        "strata_below_report_minimum": sorted(s for s, v in strata_summary.items()
                                              if v["holdout_groups"] < cfg["min_holdout_groups_for_subgroup_report"]),
    }
    return result, summary


# ---------------------------------------------------------------- 실제 자료 실행
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _load_daily(bundle: Path, market: str) -> pd.DataFrame:
    with gzip.open(bundle / f"{market.lower()}_daily.csv.gz", "rt", encoding="utf-8") as handle:
        return pd.read_csv(handle, dtype={"symbol": str}, usecols=["symbol", "date", "close", "volume"],
                           parse_dates=["date"])


def _verify_price_bundle(bundle: Path) -> dict:
    """가격 묶음 manifest에 적힌 일봉 해시와 실제 파일이 같은지 확인한다."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    hashes = {}
    for market in manifest.get("markets", []):
        actual = _sha256(bundle / market["daily_file"])
        if actual != market["daily_sha256"]:
            raise RuntimeError(f"{bundle / market['daily_file']}: 해시가 manifest와 다릅니다")
        hashes[market["market"]] = actual
    if set(hashes) != {"KR", "US"}:
        raise RuntimeError(f"{bundle}: manifest에 KR·US 일봉 해시가 모두 있어야 합니다")
    return hashes


def _write_once(path: Path, data: bytes) -> None:
    """같은 이름에 다른 내용이 이미 있으면 덮어쓰지 않는다(고정된 분할 보호)."""
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"{path}가 이미 다른 내용으로 있습니다. 고정된 분할은 덮어쓰지 않습니다")
        return
    _atomic_write(path, data)


def build(metadata_csv: Path, price_bundle: Path, out_dir: Path, proto_path: Path = protocol_mod.DEFAULT_PATH) -> dict:
    proto = protocol_mod.load(proto_path)
    daily_hashes = _verify_price_bundle(price_bundle)
    metadata = pd.read_csv(metadata_csv, dtype={"symbol": str}, encoding="utf-8-sig")
    candidates = eligible(metadata, proto)
    first_dates, returns = {}, {}
    for market in ("KR", "US"):
        daily = _load_daily(price_bundle, market)
        symbols = candidates.loc[candidates["market"] == market, "symbol"]
        traded = daily[daily["symbol"].isin(set(symbols)) & (daily["volume"] > 0)]
        first_dates.update(traded.groupby("symbol")["date"].min().to_dict())
        returns[market] = daily_returns(daily, symbols)
    result, summary = split(candidates, pd.Series(first_dates), returns, proto)

    digest = protocol_mod.protocol_hash(proto)
    inputs = {"metadata_sha256": _sha256(metadata_csv), "daily_sha256": daily_hashes,
              "code_sha256": {name: _sha256(Path(__file__).with_name(name))
                              for name in ("holdout.py", "protocol.py")}}
    input_digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()
    stem = f"holdout_split_{digest[:12]}_{input_digest[:12]}"
    csv_bytes = result.assign(first_date=result["first_date"].dt.strftime("%Y-%m-%d")).to_csv(
        index=False, lineterminator="\n").encode("utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_once(out_dir / f"{stem}.csv", csv_bytes)
    manifest = {
        "protocol_version": proto.get("version"), "protocol_sha256": digest, "protocol_status": proto["status"],
        "inputs": {"metadata_csv": str(metadata_csv), "price_bundle": str(price_bundle),
                   "price_manifest_sha256": _sha256(price_bundle / "manifest.json"), **inputs},
        "inputs_sha256": input_digest,
        "split_csv": f"{stem}.csv", "split_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        **summary,
    }
    _write_once(out_dir / f"{stem}.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    return manifest


def main(argv=None) -> None:
    import argparse
    from . import paths, price_store
    parser = argparse.ArgumentParser(description="탐색용·보류용 ETF 분할")
    parser.add_argument("--protocol", default=str(protocol_mod.DEFAULT_PATH))
    args = parser.parse_args(argv)
    root = paths.data_root()
    manifest = build(root / paths.METADATA_DIR / "etf_metadata.csv",
                     price_store.current_bundle(root / paths.PRICES_DIR),
                     paths.output_dir(paths.PROTOCOL_DIR), Path(args.protocol))
    brief = {k: manifest[k] for k in ("protocol_sha256", "candidates", "families_before_merge", "merges", "after_merge",
                                      "at_sensitivity_threshold", "symbols_by_split_market", "groups_by_split")}
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
