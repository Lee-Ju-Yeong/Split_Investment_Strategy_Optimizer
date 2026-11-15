"""국내·미국 ETF 일봉을 연구용 한 형식으로 정규화하고 품질 표시를 붙인다(M1).

입력은 ChatGPT 연구 묶음의 가격 자료(읽기 전용)이고, 출력은 저장소 밖
`<MAGICSPLIT_ETF_DATA_ROOT>/prices/`다. 값을 고치거나 행을 지우지 않는다. 문제가 있는
행·종목은 표시만 하고, 쓸지 말지는 프로토콜(D1)에서 정한다.

가격 기준(이 차이 때문에 시장 간 총수익 비교에 주의가 필요하다):
- KR: Naver 차트 수정주가. 분배금 원장이 없고 분배금 반영 여부도 검증되지 않았다.
- US: Yahoo 분할 반영 시세(OHLC), 분배금은 배당락일에 1주당 현금으로 별도 열에 둔다.
  `adj_close`로 총수익을 다시 더하지 않는다(분배금 이중 계산 방지).

기업행사(US): 빈 칸은 "사건 없음"(분배금 0, 분할 1)이다. 숫자가 아니거나 음수·0 같은
잘못된 값은 0·1로 바꾸지 않고 NaN으로 두고 `corp_action_ok=False`로 표시한다. 원본 문자열은
원천 파일(inventory의 `source_sha256`)에 그대로 남는다.

저장 구조(실행 중 실패해도 이전 묶음이 섞이지 않게):
  prices/runs/<run_id>/{kr,us}_daily.csv.gz, {kr,us}_inventory.csv, manifest.json
  prices/CURRENT   ← 확정 묶음의 run_id 한 줄. 읽는 쪽은 `current_bundle()`을 쓴다.
한 실행은 `runs/.tmp-<run_id>`에 모든 시장 파일과 manifest를 쓰고 검증한 뒤 이름을 바꾸고,
마지막에 CURRENT를 원자적으로 교체한다. 실패하면 임시 폴더를 지우고 CURRENT는 그대로다.
시장·종목 일부만 돈 실행은 묶음은 남기되 CURRENT를 바꾸지 않는다.

실행: python -m src.research.etf.price_store [--markets KR US] [--symbols 069500 SPY]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import PRICES_DIR, output_dir, raw_root

KR_SUBDIR = Path("kr_etf_price_history_2026-11-15") / "history_kr" / "prices"
US_SUBDIR = Path("us_etf_price_history_2026-11-15") / "history_us" / "prices"
REFERENCE_SYMBOL = {"KR": "069500", "US": "SPY"}  # 거래일 대조용(공식 달력은 fx/calendars 모듈)
LARGE_RETURN = 0.35
ALL_MARKETS = ("KR", "US")
RUNS_DIR = "runs"
CURRENT_POINTER = "CURRENT"
MANIFEST = "manifest.json"

COLUMNS = [
    "market", "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "dividend", "split_ratio", "ohlc_ok", "corp_action_ok", "currency", "price_basis",
]
PRICE_BASIS = {
    "KR": "naver_provider_adjusted_distribution_unverified",
    "US": "yahoo_split_adjusted_quote_with_cash_dividends",
}
CURRENCY = {"KR": "KRW", "US": "USD"}


@dataclass(frozen=True)
class MarketSource:
    market: str
    prices_dir: Path


def market_sources(root: Path) -> dict[str, MarketSource]:
    return {"KR": MarketSource("KR", root / KR_SUBDIR), "US": MarketSource("US", root / US_SUBDIR)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dividend(text: str) -> float:
    """1주당 현금 분배금. 빈 칸 → 0(사건 없음). 숫자가 아니거나 음수·무한대 → NaN(표시 대상)."""
    text = text.strip()
    if not text:
        return 0.0
    try:
        value = float(text)
    except ValueError:
        return float("nan")
    return value if np.isfinite(value) and value >= 0 else float("nan")


def _split_ratio(text: str) -> float:
    """'2:1' → 2.0(1주가 2주로). 빈 칸 → 1(사건 없음). 형식이 틀리거나 0·음수 → NaN(표시 대상)."""
    text = text.strip()
    if not text:
        return 1.0
    parts = text.split(":")
    if len(parts) != 2:
        return float("nan")
    try:
        new, old = float(parts[0]), float(parts[1])
    except ValueError:
        return float("nan")
    if not (np.isfinite(new) and np.isfinite(old) and new > 0 and old > 0):
        return float("nan")
    return new / old


def _corp_action_text(path: Path, rows: int) -> pd.DataFrame:
    """분배금·분할 열을 문자열 그대로 읽는다. pandas 기본 결측 처리('N/A', 'null' 등 → 빈 칸)를
    끄기 위해 따로 읽는다. 열 자체가 없으면 NaN 문자열을 넣어 잘못된 값으로 표시되게 한다."""
    text = pd.read_csv(path, usecols=lambda c: c in ("dividend", "split"), dtype=str, keep_default_na=False)
    if text.columns.empty:  # 두 열이 모두 없으면 pandas가 0행을 돌려준다
        text = pd.DataFrame(index=range(rows))
    for column in ("dividend", "split"):
        if column not in text:
            text[column] = "nan"
    if len(text) != rows:
        raise ValueError(f"{path}: 기업행사 열 행 수가 가격 행 수와 다릅니다({len(text)} != {rows})")
    return text


def load_symbol(market: str, path: Path, symbol: str) -> pd.DataFrame:
    """원천 CSV 한 개를 공통 열로 바꾼다. 행을 지우거나 값을 고치지 않는다."""
    raw = pd.read_csv(path, dtype={"date": str})
    out = pd.DataFrame({
        "market": market,
        "symbol": symbol,
        "date": pd.to_datetime(raw["date"], format="%Y-%m-%d"),
        "open": raw["open"].astype(float),
        "high": raw["high"].astype(float),
        "low": raw["low"].astype(float),
        "close": raw["close"].astype(float),
        "volume": raw["volume"].astype(float),
    })
    if market == "US":
        corp = _corp_action_text(path, len(raw))
        out["dividend"] = corp["dividend"].map(_dividend).to_numpy()
        out["split_ratio"] = corp["split"].map(_split_ratio).to_numpy()
        out["corp_action_ok"] = out["dividend"].notna() & out["split_ratio"].notna()
    else:
        out["dividend"] = 0.0
        out["split_ratio"] = 1.0
        out["corp_action_ok"] = True
    prices = out[["open", "high", "low", "close"]]
    out["ohlc_ok"] = (
        np.isfinite(prices).all(axis=1)
        & (prices > 0).all(axis=1)
        & (out["low"] <= out[["open", "close"]].min(axis=1))
        & (out["high"] >= out[["open", "close"]].max(axis=1))
    )
    # 거래대금 근사: 종가 × 거래량(KR은 수정주가 기준이라 과거 값은 근사).
    out["turnover"] = out["close"] * out["volume"]
    out["currency"] = CURRENCY[market]
    out["price_basis"] = PRICE_BASIS[market]
    return out[COLUMNS]


def symbol_quality(frame: pd.DataFrame, reference_dates: pd.DatetimeIndex | None) -> dict:
    """종목 하나의 품질 지표. 자동으로 제외하지 않고 숫자로만 남긴다."""
    dates = frame["date"]
    closes = frame["close"].where(frame["close"] > 0)
    returns = closes.pct_change().abs()
    span_missing = None
    if reference_dates is not None and len(frame):
        span = reference_dates[(reference_dates >= dates.min()) & (reference_dates <= dates.max())]
        span_missing = int(len(span.difference(pd.DatetimeIndex(dates))))
    return {
        "rows": int(len(frame)),
        "first_date": dates.min().strftime("%Y-%m-%d") if len(frame) else None,
        "last_date": dates.max().strftime("%Y-%m-%d") if len(frame) else None,
        "duplicate_dates": int(dates.duplicated().sum()),
        "dates_sorted": bool(dates.is_monotonic_increasing),
        "bad_ohlc_rows": int((~frame["ohlc_ok"]).sum()),
        "zero_volume_days": int((frame["volume"] <= 0).sum()),
        "large_return_days": int((returns > LARGE_RETURN).sum()),
        "max_abs_daily_return": float(returns.max()) if returns.notna().any() else None,
        "dividend_events": int((frame["dividend"] > 0).sum()),
        "split_events": int((frame["split_ratio"].notna() & (frame["split_ratio"] != 1.0)).sum()),
        "bad_corp_action_rows": int((~frame["corp_action_ok"]).sum()),
        "missing_reference_days_in_span": span_missing,
    }


def build_market(source: MarketSource, out: Path, symbols: list[str] | None = None) -> dict:
    files = sorted(source.prices_dir.glob("*.csv"))
    if symbols:
        wanted = set(symbols)
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise FileNotFoundError(f"{source.market}: 가격 파일이 없습니다: {source.prices_dir}")
    reference_file = source.prices_dir / f"{REFERENCE_SYMBOL[source.market]}.csv"
    reference_dates = None
    if reference_file.exists():
        reference_dates = pd.DatetimeIndex(load_symbol(source.market, reference_file, reference_file.stem)["date"])
    frames, inventory = [], []
    for path in files:
        frame = load_symbol(source.market, path, path.stem)
        frames.append(frame)
        inventory.append({"market": source.market, "symbol": path.stem, "source_sha256": _sha256(path),
                          **symbol_quality(frame, reference_dates)})
    daily = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"], kind="stable")
    daily_path = out / f"{source.market.lower()}_daily.csv.gz"
    daily.to_csv(daily_path, index=False, date_format="%Y-%m-%d", compression={"method": "gzip", "mtime": 0})
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(out / f"{source.market.lower()}_inventory.csv", index=False)
    return {
        "market": source.market,
        "symbols": len(files),
        "rows": int(len(daily)),
        "first_date": daily["date"].min().strftime("%Y-%m-%d"),
        "last_date": daily["date"].max().strftime("%Y-%m-%d"),
        "symbols_with_bad_ohlc": int((inventory_frame["bad_ohlc_rows"] > 0).sum()),
        "symbols_with_large_returns": int((inventory_frame["large_return_days"] > 0).sum()),
        "symbols_with_reference_gaps": int((inventory_frame["missing_reference_days_in_span"].fillna(0) > 0).sum()),
        "symbols_with_bad_corp_actions": int((inventory_frame["bad_corp_action_rows"] > 0).sum()),
        "price_basis": PRICE_BASIS[source.market],
        "reference_symbol": REFERENCE_SYMBOL[source.market],
        "daily_file": daily_path.name,
        "daily_sha256": _sha256(daily_path),
    }


def current_bundle(out: Path) -> Path:
    """CURRENT가 가리키는 확정 묶음 폴더. 읽는 쪽은 이 함수로만 위치를 얻는다."""
    pointer = out / CURRENT_POINTER
    if not pointer.is_file():
        raise FileNotFoundError(f"확정된 가격 묶음이 없습니다: {pointer}")
    bundle = out / RUNS_DIR / pointer.read_text(encoding="utf-8").strip()
    if not (bundle / MANIFEST).is_file():
        raise FileNotFoundError(f"CURRENT가 가리키는 묶음에 {MANIFEST}가 없습니다: {bundle}")
    return bundle


def _verify_bundle(bundle: Path, summary: dict) -> None:
    """manifest에 적힌 파일이 모두 있고 해시가 맞는지 확인한다."""
    for market in summary["markets"]:
        daily = bundle / market["daily_file"]
        if _sha256(daily) != market["daily_sha256"]:
            raise RuntimeError(f"{daily}: 해시가 manifest와 다릅니다")
        if not (bundle / f"{market['market'].lower()}_inventory.csv").is_file():
            raise RuntimeError(f"{bundle}: {market['market']} inventory가 없습니다")
    if json.loads((bundle / MANIFEST).read_text(encoding="utf-8")) != summary:
        raise RuntimeError(f"{bundle / MANIFEST}: 내용이 실행 결과와 다릅니다")


def _replace_pointer(out: Path, run_id: str) -> None:
    tmp = out / f".{CURRENT_POINTER}.{run_id}.tmp"
    tmp.write_text(run_id + "\n", encoding="utf-8")
    os.replace(tmp, out / CURRENT_POINTER)  # 같은 폴더 안 교체는 원자적이다(POSIX·Windows)


def build(root: Path, out: Path, markets=ALL_MARKETS, symbols: list[str] | None = None) -> dict:
    """새 묶음을 runs/<run_id>에 만들고, 전체 실행이면 CURRENT를 그 묶음으로 바꾼다.

    반환값은 manifest 내용에 `bundle_dir`(묶음 폴더)와 `promoted`(CURRENT 교체 여부)를 더한 것이다.
    """
    sources = market_sources(root)
    markets = list(markets)
    complete = sorted(markets) == sorted(ALL_MARKETS) and not symbols
    created = datetime.now(timezone.utc)
    run_id = f"{created:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    runs = out / RUNS_DIR
    staging = runs / f".tmp-{run_id}"
    staging.mkdir(parents=True)
    try:
        summary = {
            "run_id": run_id,
            "created_utc": created.isoformat(timespec="seconds"),
            "raw_root": str(root),
            "scope": {"markets": markets, "symbols": list(symbols) if symbols else None, "complete": complete},
            "markets": [build_market(sources[m], staging, symbols) for m in markets],
            "notes": [
                "행을 지우거나 값을 고치지 않았다. bad_ohlc·large_return·reference gap은 표시만 했다.",
                "US 분배금·분할의 빈 칸은 사건 없음(0·1)이다. 잘못된 값은 NaN·corp_action_ok=False로 표시했다.",
                "turnover = close × volume 근사. KR은 수정주가 기준이라 과거 거래대금은 근사다.",
                "현재 상장 종목만 있다(생존 편향).",
            ],
        }
        (staging / MANIFEST).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _verify_bundle(staging, summary)
        bundle = runs / run_id
        staging.rename(bundle)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if complete:
        _replace_pointer(out, run_id)
    return {**summary, "bundle_dir": str(bundle), "promoted": complete}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markets", nargs="+", default=list(ALL_MARKETS), choices=list(ALL_MARKETS))
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args(argv)
    summary = build(raw_root(), output_dir(PRICES_DIR), args.markets, args.symbols)
    print(json.dumps(summary["markets"], ensure_ascii=False, indent=2))
    state = "CURRENT 교체" if summary["promoted"] else "일부 실행이라 CURRENT 유지"
    print(f"묶음: {summary['bundle_dir']} ({state})")


if __name__ == "__main__":
    main()
