"""국내·미국 ETF 일봉을 연구용 한 형식으로 정규화하고 품질 표시를 붙인다(M1).

입력은 ChatGPT 연구 묶음의 가격 자료(읽기 전용)와, 국내는 폐지 ETF 원천(아래)이다. 출력은 저장소 밖
`<MAGICSPLIT_ETF_DATA_ROOT>/prices/`다. 값을 고치거나 행을 지우지 않는다. 문제가 있는
행·종목은 표시만 하고, 쓸지 말지는 프로토콜(D1)에서 정한다.

가격 기준(이 차이 때문에 시장 간 총수익 비교에 주의가 필요하다):
- KR: Naver 차트(fchart `sise.nhn`, `requestType=1`) 수정주가. 생존 ETF(ChatGPT 묶음)와 폐지 ETF가 같은
  주소·같은 요청값으로 받은 같은 형식이다. KRX 월말 스냅샷의 실제 종가와 대조하면 Naver 종가는 과거로
  갈수록 낮아지고(비율 < 1) 최근 날짜에서는 같아진다. 분배금만큼 과거 가격을 낮추는 역방향 수정과 같은
  모양이고, KRX 종가보다 높은 경우는 없었다. 분배금 원장과 금액까지 맞춰 보지는 않았으므로 기준 이름은
  `..._distribution_unverified`를 그대로 쓴다. 대조 결과는 inventory의 `krx_close_*` 열과 manifest의
  `krx_close_check`에 남는다.
- US: Yahoo 분할 반영 시세(OHLC), 분배금은 배당락일에 1주당 현금으로 별도 열에 둔다.
  `adj_close`로 총수익을 다시 더하지 않는다(분배금 이중 계산 방지).

기업행사(US): 빈 칸은 "사건 없음"(분배금 0, 분할 1)이다. 숫자가 아니거나 음수·0 같은
잘못된 값은 0·1로 바꾸지 않고 NaN으로 두고 `corp_action_ok=False`로 표시한다. 원본 문자열은
원천 파일(inventory의 `source_sha256`)에 그대로 남는다.

폐지 국내 ETF(KR): `<MAGICSPLIT_ETF_DATA_ROOT>/kr_universe/`의 이력표(`kr_etf_history_scoped.csv`의
`delisted=True` 행)와 Naver 일봉 XML(`naver_daily/<code>.xml`)을 읽는다.
- XML은 외부에서 받은 신뢰할 수 없는 자료다. DOCTYPE·엔티티 선언·외부 참조를 만나면 바로 거부하는
  expat 처리기로 읽는다(엔티티 확장 없음).
- 마지막 행(시가·고가·저가·거래량이 0, 종가만 직전 값)은 거래일이 아니라 폐지 표시 행이다. 일봉에서 빼고
  `kr_delisting_markers.csv`에 따로 두며, 그 날짜를 `delisting_date`로 쓴다. 같은 모양의 행이 중간에 있으면
  거래 없는 날이므로 생존 ETF와 똑같이 일봉에 두고 `ohlc_ok=False`로 표시한다.
- 빈 응답·파싱 실패·코드 불일치 종목은 넣지 않고 `kr_delisted_sources.csv`에 사유와 해시를 남긴다.
- 모든 행에 `listing_status`(listed/delisted)와 `delisting_date`가 붙는다. US는 현재 상장 종목뿐이다(한계).

저장 구조(실행 중 실패해도 이전 묶음이 섞이지 않게):
  prices/runs/<run_id>/{kr,us}_daily.csv.gz, {kr,us}_inventory.csv, manifest.json
  (KR 폐지 원천을 읽은 실행이면) kr_delisted_sources.csv, kr_delisting_markers.csv
  prices/CURRENT   ← 확정 묶음의 run_id 한 줄. 읽는 쪽은 `current_bundle()`을 쓴다.
한 실행은 `runs/.tmp-<run_id>`에 모든 시장 파일과 manifest를 쓰고 검증한 뒤 이름을 바꾸고,
마지막에 CURRENT를 원자적으로 교체한다. 실패하면 임시 폴더를 지우고 CURRENT는 그대로다.
시장·종목 일부만 돈 실행은 묶음은 남기되 CURRENT를 바꾸지 않는다.

실행: python -m src.research.etf.price_store [--markets KR US] [--symbols 069500 SPY]
  CLI는 KR을 돌릴 때 `$MAGICSPLIT_ETF_DATA_ROOT/kr_universe`(폐지 원천)가 없으면 멈춘다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.parsers import expat

import numpy as np
import pandas as pd

from .paths import PRICES_DIR, data_root, output_dir, raw_root

KR_SUBDIR = Path("kr_etf_price_history_2026-11-15") / "history_kr" / "prices"
US_SUBDIR = Path("us_etf_price_history_2026-11-15") / "history_us" / "prices"
REFERENCE_SYMBOL = {"KR": "069500", "US": "SPY"}  # 거래일 대조용(공식 달력은 fx/calendars 모듈)
LARGE_RETURN = 0.35
ALL_MARKETS = ("KR", "US")
RUNS_DIR = "runs"
CURRENT_POINTER = "CURRENT"
MANIFEST = "manifest.json"

KR_UNIVERSE_DIR = "kr_universe"
KR_DELISTED_HISTORY = "kr_etf_history_scoped.csv"
NAVER_DAILY_SUBDIR = "naver_daily"
KRX_SNAPSHOT_SUBDIR = "snapshots"
DELISTED_SOURCES_FILE = "kr_delisted_sources.csv"
DELISTING_MARKERS_FILE = "kr_delisting_markers.csv"
LISTED, DELISTED = "listed", "delisted"
SOURCE_LABEL = {"KR": "chatgpt_naver_survivors", "US": "chatgpt_yahoo_survivors"}
DELISTED_SOURCE_LABEL = "kr_universe_naver_delisted"

COLUMNS = [
    "market", "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "dividend", "split_ratio", "ohlc_ok", "corp_action_ok", "currency", "price_basis",
    "listing_status", "delisting_date",
]
PRICE_BASIS = {
    "KR": "naver_provider_adjusted_distribution_unverified",
    "US": "yahoo_split_adjusted_quote_with_cash_dividends",
}
CURRENCY = {"KR": "KRW", "US": "USD"}


@dataclass(frozen=True)
class DelistedKrSource:
    """폐지 국내 ETF 원천: 이력표, Naver 일봉 XML 폴더, (대조용) KRX 월말 스냅샷 폴더."""
    history_csv: Path
    naver_dir: Path
    snapshots_dir: Path | None = None

    @classmethod
    def from_dir(cls, kr_universe: Path) -> "DelistedKrSource":
        history = kr_universe / KR_DELISTED_HISTORY
        naver = kr_universe / NAVER_DAILY_SUBDIR
        if not history.is_file() or not naver.is_dir():
            raise FileNotFoundError(f"폐지 국내 ETF 원천이 없습니다: {history}, {naver}")
        snapshots = kr_universe / KRX_SNAPSHOT_SUBDIR
        return cls(history, naver, snapshots if snapshots.is_dir() else None)


@dataclass(frozen=True)
class MarketSource:
    market: str
    prices_dir: Path
    delisted: DelistedKrSource | None = None  # KR만 쓴다. None이면 생존 ETF만 읽는다.


def market_sources(root: Path, kr_delisted: DelistedKrSource | None = None) -> dict[str, MarketSource]:
    return {"KR": MarketSource("KR", root / KR_SUBDIR, kr_delisted), "US": MarketSource("US", root / US_SUBDIR)}


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


def _base_frame(market: str, symbol: str, dates: pd.Series, raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "market": market,
        "symbol": symbol,
        "date": dates,
        "open": raw["open"].astype(float),
        "high": raw["high"].astype(float),
        "low": raw["low"].astype(float),
        "close": raw["close"].astype(float),
        "volume": raw["volume"].astype(float),
    })


def _finish(out: pd.DataFrame, market: str, listing_status: str = LISTED, delisting_date=pd.NaT) -> pd.DataFrame:
    """품질 표시와 파생 열을 붙인다. 생존·폐지 원천이 같은 규칙을 쓰도록 한곳에 둔다."""
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
    out["listing_status"] = listing_status
    out["delisting_date"] = pd.Series(pd.Timestamp(delisting_date), index=out.index, dtype="datetime64[ns]")
    return out[COLUMNS]


def load_symbol(market: str, path: Path, symbol: str) -> pd.DataFrame:
    """원천 CSV 한 개(생존 ETF)를 공통 열로 바꾼다. 행을 지우거나 값을 고치지 않는다."""
    raw = pd.read_csv(path, dtype={"date": str})
    out = _base_frame(market, symbol, pd.to_datetime(raw["date"], format="%Y-%m-%d"), raw)
    if market == "US":
        corp = _corp_action_text(path, len(raw))
        out["dividend"] = corp["dividend"].map(_dividend).to_numpy()
        out["split_ratio"] = corp["split"].map(_split_ratio).to_numpy()
        out["corp_action_ok"] = out["dividend"].notna() & out["split_ratio"].notna()
    else:
        out["dividend"] = 0.0
        out["split_ratio"] = 1.0
        out["corp_action_ok"] = True
    return _finish(out, market)


# ---------------------------------------------------------------------------
# 폐지 국내 ETF: Naver 일봉 XML
# ---------------------------------------------------------------------------
_XML_DECL = re.compile(rb'^<\?xml[^>]*?encoding="([A-Za-z0-9_-]+)"[^>]*\?>')
_NAVER_ENCODINGS = {"EUC-KR": "cp949", "CP949": "cp949", "UTF-8": "utf-8"}
NAVER_FIELDS = ("date", "open", "high", "low", "close", "volume")


class UnsafeXmlError(ValueError):
    """DOCTYPE·엔티티 선언·외부 참조처럼 신뢰할 수 없는 XML 구성."""


@dataclass(frozen=True)
class NaverChart:
    symbol: str | None
    rows: list[tuple[str, ...]]


def parse_naver_chart(raw: bytes) -> NaverChart:
    """Naver fchart 응답을 안전하게 읽는다.

    - 앞 공백을 걷어 내고, XML 선언의 인코딩으로 엄격하게 해독한다(EUC-KR은 상위 집합인 CP949로).
    - expat 처리기가 DOCTYPE·엔티티 선언·외부 엔티티 참조를 만나면 바로 실패한다(엔티티 확장 차단).
    - `<chartdata>`는 0개 또는 1개여야 하고, `<item data="날짜|시|고|저|종|거래량">`만 값으로 쓴다.
    `<chartdata>`가 없으면(빈 응답) `symbol=None, rows=[]`.
    """
    body = raw.lstrip(b" \t\r\n")
    match = _XML_DECL.match(body)
    if not match:
        raise ValueError("encoding이 적힌 XML 선언이 맨 앞에 없습니다")
    encoding = _NAVER_ENCODINGS.get(match.group(1).decode("ascii").upper())
    if encoding is None:
        raise ValueError(f"알 수 없는 인코딩: {match.group(1)!r}")
    text = body[match.end():].decode(encoding)  # 깨진 바이트가 있으면 UnicodeDecodeError(ValueError)

    charts: list[str | None] = []
    rows: list[tuple[str, ...]] = []

    def start(tag, attrs):
        if tag == "chartdata":
            charts.append(attrs.get("symbol"))
        elif tag == "item":
            if len(charts) != 1:
                raise ValueError("item이 chartdata 밖에 있습니다")
            parts = tuple(attrs.get("data", "").split("|"))
            if len(parts) != len(NAVER_FIELDS):
                raise ValueError(f"item 필드 수가 다릅니다: {attrs.get('data')!r}")
            rows.append(parts)

    def refuse(*_args):
        raise UnsafeXmlError("DOCTYPE·엔티티 선언이나 외부 참조가 있는 XML은 읽지 않습니다")

    parser = expat.ParserCreate(encoding="UTF-8")  # 선언을 떼고 UTF-8로 다시 넣으므로 인코딩을 고정한다
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    parser.StartDoctypeDeclHandler = refuse
    parser.EntityDeclHandler = refuse
    parser.UnparsedEntityDeclHandler = refuse
    parser.ExternalEntityRefHandler = refuse
    parser.StartElementHandler = start
    parser.Parse(text.encode("utf-8"), True)
    if len(charts) > 1:
        raise ValueError(f"chartdata가 {len(charts)}개입니다")
    return NaverChart(charts[0] if charts else None, rows)


def _is_marker(row: tuple[str, ...]) -> bool:
    """Naver 폐지 표시 행 모양: 시가·고가·저가·거래량이 0이고 종가만 직전 값."""
    _date, open_, high, low, _close, volume = row
    try:
        return float(open_) == 0 and float(high) == 0 and float(low) == 0 and float(volume) == 0
    except ValueError:
        return False


def load_naver_delisted(chart: NaverChart, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """폐지 ETF 한 종목의 (거래일 일봉, 폐지 표시 행, 정보). 마지막 행만 폐지 표시 행으로 본다.

    행을 지우거나 값을 고치지 않는다. 폐지 표시 행은 지우는 것이 아니라 별도 파일로 옮긴다.
    """
    rows = list(chart.rows)
    marker_rows = [rows.pop()] if rows and _is_marker(rows[-1]) else []
    delisting_date = pd.to_datetime(marker_rows[0][0], format="%Y%m%d") if marker_rows else pd.NaT

    def frame(items):
        raw = pd.DataFrame(items, columns=list(NAVER_FIELDS))
        out = _base_frame("KR", symbol, pd.to_datetime(raw["date"], format="%Y%m%d"), raw)
        out["dividend"] = 0.0
        out["split_ratio"] = 1.0
        out["corp_action_ok"] = True
        return _finish(out, "KR", DELISTED, delisting_date)

    daily = frame(rows)
    markers = frame(marker_rows)
    if marker_rows and len(daily) and not delisting_date > daily["date"].max():
        raise ValueError(f"{symbol}: 폐지 표시 행 날짜({delisting_date:%Y-%m-%d})가 마지막 거래일 뒤가 아닙니다")
    info = {
        "delisting_marker_found": bool(marker_rows),
        "delisting_date": None if pd.isna(delisting_date) else delisting_date.strftime("%Y-%m-%d"),
    }
    return daily, markers, info


def read_delisted_history(path: Path) -> pd.DataFrame:
    """이력표에서 폐지 ETF 행만 돌려준다. 코드는 문자열(앞자리 0·영문), UTF-8(BOM) 엄격 해독."""
    history = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    missing = {"code", "last_name", "delisted", "last_seen"} - set(history.columns)
    if missing:
        raise ValueError(f"{path}: 필요한 열이 없습니다: {sorted(missing)}")
    if history["code"].duplicated().any():
        raise ValueError(f"{path}: 종목코드가 중복됩니다")
    if not history["delisted"].isin(["True", "False"]).all():
        raise ValueError(f"{path}: delisted 값은 True/False여야 합니다")
    return history[history["delisted"] == "True"].reset_index(drop=True)


def _load_delisted(source: DelistedKrSource, symbols: list[str] | None):
    """(일봉 목록, 폐지 표시 행 목록, 원천 기록 목록). 넣지 못한 종목은 기록에 사유를 남긴다."""
    history = read_delisted_history(source.history_csv)
    if symbols:
        history = history[history["code"].isin(set(symbols))]
    frames, markers, records = [], [], []
    for item in history.to_dict("records"):
        code = item["code"]
        path = source.naver_dir / f"{code}.xml"
        record = {"symbol": code, "name": item["last_name"], "last_seen_krx": item["last_seen"],
                  "xml_file": path.name, "xml_sha256": None, "status": "excluded", "reason": "",
                  "detail": "", "rows": 0, "first_date": None, "last_date": None,
                  "delisting_marker_found": None, "delisting_date": None}
        records.append(record)
        if not path.is_file():
            record["reason"] = "naver_file_missing"
            continue
        raw = path.read_bytes()
        record["xml_sha256"] = hashlib.sha256(raw).hexdigest()
        try:
            chart = parse_naver_chart(raw)
        except UnsafeXmlError as exc:
            record.update(reason="unsafe_xml", detail=str(exc))
            continue
        except (ValueError, expat.ExpatError) as exc:
            record.update(reason="parse_error", detail=f"{type(exc).__name__}: {exc}")
            continue
        if chart.symbol is None or not chart.rows:
            record["reason"] = "naver_empty_response"
            continue
        if chart.symbol != code:
            record.update(reason="symbol_mismatch", detail=f"chartdata symbol={chart.symbol!r}")
            continue
        try:
            daily, marker, info = load_naver_delisted(chart, code)
        except ValueError as exc:
            record.update(reason="invalid_rows", detail=str(exc))
            continue
        if daily.empty:
            record["reason"] = "no_trading_rows"
            continue
        record.update(status="included", rows=int(len(daily)), **info,
                      first_date=daily["date"].min().strftime("%Y-%m-%d"),
                      last_date=daily["date"].max().strftime("%Y-%m-%d"))
        frames.append(daily)
        markers.append(marker)
    return frames, markers, records


# ---------------------------------------------------------------------------
# 품질 표시
# ---------------------------------------------------------------------------
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


def _read_krx_closes(snapshots_dir: Path) -> pd.DataFrame:
    """KRX 월말 스냅샷의 실제 종가. 휴장일 스냅샷의 가격 칸('-')은 뺀다."""
    frames = []
    for path in sorted(snapshots_dir.glob("*.csv")):
        snap = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False,
                           usecols=["trade_date", "ISU_SRT_CD", "TDD_CLSPRC"])
        frames.append(snap[snap["TDD_CLSPRC"].str.fullmatch(r"[\d,]+")])
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "krx_close"])
    snap = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({
        "symbol": snap["ISU_SRT_CD"],
        "date": pd.to_datetime(snap["trade_date"], format="%Y-%m-%d"),
        "krx_close": snap["TDD_CLSPRC"].str.replace(",", "", regex=False).astype(float),
    }).drop_duplicates(["symbol", "date"])


def krx_close_check(daily: pd.DataFrame, snapshots_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Naver 종가와 KRX 실제 종가(월말 스냅샷)를 같은 날짜끼리 대조한다(가격 기준 근거).

    수정주가라면 과거 Naver 종가가 KRX 종가보다 낮거나 같고(비율 ≤ 1), 원래 가격이면 같다.
    """
    merged = daily[["symbol", "date", "close", "listing_status"]].merge(
        _read_krx_closes(snapshots_dir), on=["symbol", "date"])
    merged["ratio"] = merged["close"] / merged["krx_close"]
    merged["equal"] = np.isclose(merged["close"], merged["krx_close"], rtol=0, atol=1e-6)
    per_symbol = merged.groupby("symbol").agg(
        krx_close_compared=("ratio", "size"), krx_close_equal=("equal", "sum"),
        krx_close_min_ratio=("ratio", "min"), krx_close_max_ratio=("ratio", "max")).reset_index()
    summary = {}
    for status, group in merged.groupby("listing_status"):
        below = group["ratio"] < 1 - 1e-9
        above = group["ratio"] > 1 + 1e-9
        summary[status] = {
            "symbols_compared": int(group["symbol"].nunique()),
            "rows_compared": int(len(group)),
            "rows_equal": int(group["equal"].sum()),
            "rows_naver_below_krx": int(below.sum()),
            "rows_naver_above_krx": int(above.sum()),
            "symbols_with_naver_below_krx": int(group.loc[below, "symbol"].nunique()),
            "min_ratio": float(group["ratio"].min()),
            "max_ratio": float(group["ratio"].max()),
        }
    return per_symbol, summary


def build_market(source: MarketSource, out: Path, symbols: list[str] | None = None) -> dict:
    files = sorted(source.prices_dir.glob("*.csv"))
    if source.delisted is not None:
        delisted_codes = set(read_delisted_history(source.delisted.history_csv)["code"])
        if not delisted_codes:
            raise ValueError(f"{source.delisted.history_csv}: 폐지 ETF 행이 없습니다(원천 확인 필요)")
        overlap = sorted({f.stem for f in files} & delisted_codes)
        if overlap:
            raise ValueError(f"{source.market}: 생존 가격 파일과 폐지 목록에 같은 코드가 있습니다: {overlap[:5]}")
    if symbols:
        wanted = set(symbols)
        files = [f for f in files if f.stem in wanted]
    delisted_frames, marker_frames, delisted_records = ([], [], [])
    if source.delisted is not None:
        delisted_frames, marker_frames, delisted_records = _load_delisted(source.delisted, symbols)
    if not files and not delisted_frames:
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
                          **symbol_quality(frame, reference_dates),
                          "source": SOURCE_LABEL[source.market], "listing_status": LISTED,
                          "delisting_date": None, "delisting_marker_found": None})
    included = {r["symbol"]: r for r in delisted_records if r["status"] == "included"}
    for frame in delisted_frames:
        record = included[frame["symbol"].iloc[0]]
        frames.append(frame)
        inventory.append({"market": source.market, "symbol": record["symbol"], "source_sha256": record["xml_sha256"],
                          **symbol_quality(frame, reference_dates),
                          "source": DELISTED_SOURCE_LABEL, "listing_status": DELISTED,
                          "delisting_date": record["delisting_date"],
                          "delisting_marker_found": record["delisting_marker_found"]})
    daily = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"], kind="stable")
    prefix = source.market.lower()
    written: list[Path] = []
    daily_path = out / f"{prefix}_daily.csv.gz"
    daily.to_csv(daily_path, index=False, date_format="%Y-%m-%d", compression={"method": "gzip", "mtime": 0})
    written.append(daily_path)

    inventory_frame = pd.DataFrame(inventory)
    krx_check = None
    snapshots_dir = source.delisted.snapshots_dir if source.delisted is not None else None
    if source.market == "KR" and snapshots_dir is not None:
        per_symbol, krx_check = krx_close_check(daily, snapshots_dir)
        inventory_frame = inventory_frame.merge(per_symbol, on="symbol", how="left")
    inventory_frame = inventory_frame.sort_values("symbol", kind="stable")
    inventory_path = out / f"{prefix}_inventory.csv"
    inventory_frame.to_csv(inventory_path, index=False)
    written.append(inventory_path)

    delisted_summary = None
    if source.delisted is not None:
        sources_path = out / DELISTED_SOURCES_FILE
        pd.DataFrame(delisted_records, columns=[
            "symbol", "name", "last_seen_krx", "xml_file", "xml_sha256", "status", "reason", "detail", "rows",
            "first_date", "last_date", "delisting_marker_found", "delisting_date",
        ]).to_csv(sources_path, index=False, encoding="utf-8")
        markers = (pd.concat(marker_frames, ignore_index=True) if marker_frames
                   else pd.DataFrame(columns=COLUMNS)).sort_values(["symbol", "date"], kind="stable")
        markers_path = out / DELISTING_MARKERS_FILE
        markers.to_csv(markers_path, index=False, date_format="%Y-%m-%d")
        written += [sources_path, markers_path]
        excluded = Counter(r["reason"] for r in delisted_records if r["status"] != "included")
        delisted_summary = {
            "history_csv": str(source.delisted.history_csv),
            "history_sha256": _sha256(source.delisted.history_csv),
            "naver_dir": str(source.delisted.naver_dir),
            "candidates": len(delisted_records),
            "included": len(included),
            "excluded": dict(sorted(excluded.items())),
            "without_delisting_marker": sum(1 for r in included.values() if not r["delisting_marker_found"]),
            "marker_rows": int(len(markers)),
            "sources_file": sources_path.name,
            "markers_file": markers_path.name,
        }

    listed = inventory_frame["listing_status"] == LISTED
    return {
        "market": source.market,
        "symbols": int(len(inventory_frame)),
        "symbols_listed": int(listed.sum()),
        "symbols_delisted": int((~listed).sum()),
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
        "delisted_source": delisted_summary,
        "krx_close_check": krx_check,
        "files": {path.name: _sha256(path) for path in written},
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
        for name, digest in market.get("files", {}).items():
            path = bundle / name
            if not path.is_file() or _sha256(path) != digest:
                raise RuntimeError(f"{path}: 없거나 해시가 manifest와 다릅니다")
    if json.loads((bundle / MANIFEST).read_text(encoding="utf-8")) != summary:
        raise RuntimeError(f"{bundle / MANIFEST}: 내용이 실행 결과와 다릅니다")


def _replace_pointer(out: Path, run_id: str) -> None:
    tmp = out / f".{CURRENT_POINTER}.{run_id}.tmp"
    tmp.write_text(run_id + "\n", encoding="utf-8")
    os.replace(tmp, out / CURRENT_POINTER)  # 같은 폴더 안 교체는 원자적이다(POSIX·Windows)


def build(root: Path, out: Path, markets=ALL_MARKETS, symbols: list[str] | None = None,
          kr_delisted: DelistedKrSource | None = None) -> dict:
    """새 묶음을 runs/<run_id>에 만들고, 전체 실행이면 CURRENT를 그 묶음으로 바꾼다.

    `kr_delisted`를 주면 KR에 폐지 국내 ETF를 함께 넣는다. 전체 실행이란 두 시장 전 종목이면서 폐지 원천도
    읽은 실행이다. 폐지 원천 없이 돈 실행은 묶음만 남기고 CURRENT를 바꾸지 않는다(생존 편향 재유입 방지).
    반환값은 manifest 내용에 `bundle_dir`(묶음 폴더)와 `promoted`(CURRENT 교체 여부)를 더한 것이다.
    """
    sources = market_sources(root, kr_delisted)
    markets = list(markets)
    complete = sorted(markets) == sorted(ALL_MARKETS) and not symbols and kr_delisted is not None
    created = datetime.now(timezone.utc)
    run_id = f"{created:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    runs = out / RUNS_DIR
    staging = runs / f".tmp-{run_id}"
    staging.mkdir(parents=True)
    with_delisted = kr_delisted is not None and "KR" in markets
    try:
        summary = {
            "run_id": run_id,
            "created_utc": created.isoformat(timespec="seconds"),
            "raw_root": str(root),
            "scope": {"markets": markets, "symbols": list(symbols) if symbols else None, "complete": complete,
                      "kr_delisted": with_delisted},
            "markets": [build_market(sources[m], staging, symbols) for m in markets],
            "notes": [
                "행을 지우거나 값을 고치지 않았다. bad_ohlc·large_return·reference gap은 표시만 했다.",
                "US 분배금·분할의 빈 칸은 사건 없음(0·1)이다. 잘못된 값은 NaN·corp_action_ok=False로 표시했다.",
                "turnover = close × volume 근사. KR은 수정주가 기준이라 과거 거래대금은 근사다.",
                ("KR은 폐지 ETF를 포함한다(Naver 일봉이 있는 것만, listing_status=delisted). "
                 "폐지 표시 행은 일봉에서 빼 kr_delisting_markers.csv에 두었다. US는 현재 상장 종목만 있다(생존 편향)."
                 if with_delisted else "현재 상장 종목만 있다(생존 편향)."),
                "KR Naver 종가는 KRX 실제 종가보다 과거로 갈수록 낮다(분배금 역방향 수정 모양). krx_close_check 참고.",
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
    kr_delisted = DelistedKrSource.from_dir(data_root() / KR_UNIVERSE_DIR) if "KR" in args.markets else None
    summary = build(raw_root(), output_dir(PRICES_DIR), args.markets, args.symbols, kr_delisted)
    print(json.dumps(summary["markets"], ensure_ascii=False, indent=2))
    state = "CURRENT 교체" if summary["promoted"] else "일부 실행이라 CURRENT 유지"
    print(f"묶음: {summary['bundle_dir']} ({state})")


if __name__ == "__main__":
    main()
