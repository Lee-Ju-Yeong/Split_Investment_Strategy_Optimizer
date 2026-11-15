"""국내·미국 ETF 일봉을 연구용 한 형식으로 정규화하고 품질 표시를 붙인다(M1).

입력은 ChatGPT 연구 묶음의 가격 자료(읽기 전용)이고, 출력은 저장소 밖
`<MAGICSPLIT_ETF_DATA_ROOT>/prices/`다. 값을 고치거나 행을 지우지 않는다. 문제가 있는
행·종목은 표시만 하고, 쓸지 말지는 프로토콜(D1)에서 정한다.

가격 기준(이 차이 때문에 시장 간 총수익 비교에 주의가 필요하다):
- KR: Naver 차트 수정주가. 분배금 원장이 없고 분배금 반영 여부도 검증되지 않았다.
- US: Yahoo 분할 반영 시세(OHLC), 분배금은 배당락일에 1주당 현금으로 별도 열에 둔다.
  `adj_close`로 총수익을 다시 더하지 않는다(분배금 이중 계산 방지).

실행: python -m src.research.etf.price_store [--markets KR US] [--symbols 069500 SPY]
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

COLUMNS = [
    "market", "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "dividend", "split_ratio", "ohlc_ok", "currency", "price_basis",
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


def _split_ratio(value) -> float:
    """'2:1' → 2.0(1주가 2주로). 비어 있으면 1."""
    if not isinstance(value, str) or ":" not in value:
        return 1.0
    new, old = value.split(":", 1)
    return float(new) / float(old)


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
        out["dividend"] = pd.to_numeric(raw.get("dividend", 0.0), errors="coerce").fillna(0.0)
        out["split_ratio"] = raw.get("split", pd.Series([None] * len(raw))).map(_split_ratio)
    else:
        out["dividend"] = 0.0
        out["split_ratio"] = 1.0
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
        "split_events": int((frame["split_ratio"] != 1.0).sum()),
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
        "price_basis": PRICE_BASIS[source.market],
        "reference_symbol": REFERENCE_SYMBOL[source.market],
        "daily_file": daily_path.name,
        "daily_sha256": _sha256(daily_path),
    }


def build(root: Path, out: Path, markets=("KR", "US"), symbols: list[str] | None = None) -> dict:
    sources = market_sources(root)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_root": str(root),
        "markets": [build_market(sources[m], out, symbols) for m in markets],
        "notes": [
            "행을 지우거나 값을 고치지 않았다. bad_ohlc·large_return·reference gap은 표시만 했다.",
            "turnover = close × volume 근사. KR은 수정주가 기준이라 과거 거래대금은 근사다.",
            "현재 상장 종목만 있다(생존 편향).",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markets", nargs="+", default=["KR", "US"], choices=["KR", "US"])
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args(argv)
    summary = build(raw_root(), output_dir(PRICES_DIR), args.markets, args.symbols)
    print(json.dumps(summary["markets"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
