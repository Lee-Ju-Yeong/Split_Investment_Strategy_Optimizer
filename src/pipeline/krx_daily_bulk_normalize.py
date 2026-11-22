"""
krx_daily_bulk_normalize.py

날짜별 전종목 KRX 응답(원 필드명)을 기존 종목별 수집기와 같은 열·단위·제외 규칙으로 바꾸고,
응답이 완전한지 검사한다. 이 모듈은 DB·pykrx·네트워크를 쓰지 않는 순수 함수만 담는다.

완전성 규칙(리뷰 합의 4):
- 시세: KOSPI·KOSDAQ 둘 다 있어야 하고, 숫자 해석 실패(손상)는 휴장으로 보지 않는다.
- 투자자: 주체 응답이 비면 즉시 중단. 일부 주체에만 있는 종목(누락 vs 0 구분 불가)은 세고 기본 0건 허용.
- 공매도: '-'는 0이 아니라 미집계(NaN)로 본다.
- 종목코드는 6자리 영숫자만 받는다(ISIN 자르기 같은 추정 변환 없음).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd

from ..data.collectors.financial_collector import normalize_fundamental_df
from ..data.collectors.investor_trading_collector import normalize_investor_df
from ..market_cap_collector import normalize_market_cap_df
from ..short_selling_collector import normalize_short_selling_df


MARKET_ID_TO_TYPE = {"STK": "KOSPI", "KSQ": "KOSDAQ"}
SHORT_MARKET_CODES = {"KOSPI": 1, "KOSDAQ": 2}
INVESTOR_CODES = {"institution": 7050, "individual": 8000, "foreigner": 9000, "other_foreigner": 9001}
PRICE_FIELDS = {
    "TDD_OPNPRC": "open_price", "TDD_HGPRC": "high_price", "TDD_LWPRC": "low_price",
    "TDD_CLSPRC": "close_price", "ACC_TRDVOL": "volume",
}
OHLC_COLUMNS = ("open_price", "high_price", "low_price", "close_price")
TICKER_PATTERN = re.compile(r"^[0-9A-Z]{6}$")
BASE_PRICE_TOLERANCE = 0.5


class BulkResponseError(RuntimeError):
    """응답이 비었거나 불완전·손상돼 그 날짜를 저장하면 안 되는 경우."""


@dataclass(frozen=True)
class ParsedField:
    """KRX 문자열 숫자 열을 값·'-'(또는 공란)·손상(해석 불가·결측)으로 나눈 결과."""

    values: pd.Series
    dash: pd.Series
    damaged: pd.Series


def parse_krx_field(values: pd.Series) -> ParsedField:
    missing = values.isna()
    text = values.astype(str).str.replace(",", "", regex=False).str.strip()
    dash = text.isin(["-", ""]) & ~missing
    numbers = pd.to_numeric(text.mask(dash | missing, "nan"), errors="coerce")
    damaged = numbers.isna() & ~dash
    return ParsedField(numbers, dash, damaged)


def parse_krx_number(values: pd.Series, dash_as_missing: bool = False) -> pd.Series:
    """KRX 문자열 숫자. '-'/빈칸은 pykrx처럼 0, dash_as_missing이면 NaN. 해석 불가는 NaN."""
    parsed = parse_krx_field(values)
    return parsed.values if dash_as_missing else parsed.values.mask(parsed.dash, 0)


def _require_not_damaged(parsed: ParsedField, label: str) -> None:
    if bool(parsed.damaged.any()):
        raise BulkResponseError(f"{label}: {int(parsed.damaged.sum())} unparsable values")


def validate_ticker_codes(values: pd.Series, label: str) -> pd.Series:
    codes = values.astype(str).str.strip()
    bad = ~codes.str.match(TICKER_PATTERN) | values.isna()
    if bool(bad.any()):
        raise BulkResponseError(f"{label}: {int(bad.sum())} invalid ticker codes")
    if bool(codes.duplicated().any()):
        raise BulkResponseError(f"{label}: duplicated ticker codes")
    return codes


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise BulkResponseError(f"{label}: empty response")
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise BulkResponseError(f"{label}: missing columns {missing}")


def _apply_ticker_normalizer(normalizer, frame: pd.DataFrame, codes: pd.Series) -> pd.DataFrame:
    """종목별 정규화 함수를 여러 종목 한 날짜 프레임에 적용한다(행 위치로 종목코드를 되돌린다)."""
    indexed = frame.reset_index(drop=True).set_index("날짜")
    normalized = normalizer(indexed, "000000")
    if normalized is None or normalized.empty:
        return pd.DataFrame()
    output = normalized.copy()
    output["stock_code"] = codes.reset_index(drop=True).loc[output.index].to_numpy()
    return output.reset_index(drop=True)


# ---- 시세(전종목시세 1회 → 시세·시총·명단) ---------------------------------

@dataclass
class MarketDaySnapshot:
    day: str
    prices: pd.DataFrame
    market_caps: pd.DataFrame
    universe: pd.DataFrame
    market_counts: Dict[str, int] = field(default_factory=dict)
    is_holiday: bool = False

    @property
    def codes(self) -> Set[str]:
        return set(self.universe["stock_code"]) if not self.universe.empty else set()


def _market_frame(raw: pd.DataFrame) -> pd.DataFrame:
    require_columns(raw, ["ISU_SRT_CD", "MKT_ID", "CMPPREVDD_PRC", "MKTCAP", "LIST_SHRS", *PRICE_FIELDS], "market_prices")
    frame = raw.loc[raw["MKT_ID"].astype(str).isin(MARKET_ID_TO_TYPE)].reset_index(drop=True)
    counts = frame["MKT_ID"].astype(str).map(MARKET_ID_TO_TYPE).value_counts().to_dict()
    missing = [m for m in MARKET_ID_TO_TYPE.values() if counts.get(m, 0) == 0]
    if missing:
        raise BulkResponseError(f"market_prices: market missing {missing}")
    out = pd.DataFrame({"stock_code": validate_ticker_codes(frame["ISU_SRT_CD"], "market_prices")})
    out["market_type"] = frame["MKT_ID"].astype(str).map(MARKET_ID_TO_TYPE)
    dash_any = pd.Series(False, index=frame.index)
    for source, target in PRICE_FIELDS.items():
        parsed = parse_krx_field(frame[source])
        _require_not_damaged(parsed, f"market_prices:{source}")
        if target == "close_price" and bool(parsed.dash.any()):
            raise BulkResponseError(f"market_prices: {int(parsed.dash.sum())} rows without close price")
        out[target] = parsed.values.mask(parsed.dash, 0)  # 시고저·거래량 '-'는 pykrx처럼 0(거래정지)
        dash_any |= parsed.dash
    for source, target in (("MKTCAP", "market_cap"), ("LIST_SHRS", "shares_outstanding")):
        parsed = parse_krx_field(frame[source])
        _require_not_damaged(parsed, f"market_prices:{source}")
        if bool(parsed.dash.any()):
            raise BulkResponseError(f"market_prices: {int(parsed.dash.sum())} rows with '-' in {source}")
        out[target] = parsed.values
    change = parse_krx_field(frame["CMPPREVDD_PRC"])
    _require_not_damaged(change, "market_prices:CMPPREVDD_PRC")
    out["change_vs_prev"] = change.values  # '-'는 0으로 바꾸지 않고 NaN(수정주가 판정에서 격리)
    out.attrs["price_dash_rows"] = int(dash_any.sum())
    return out


def normalize_market_day(raw: pd.DataFrame, day: str) -> MarketDaySnapshot:
    """휴장은 모든 시고저·종가·거래량이 숫자 0일 때만이다('-'·공란만으로 휴장을 확정하지 않음)."""
    frame = _market_frame(raw)
    counts = frame["market_type"].value_counts().to_dict()
    if bool((frame[list(OHLC_COLUMNS) + ["volume"]] == 0).all(axis=None)):
        if frame.attrs.get("price_dash_rows", 0):
            raise BulkResponseError("market_prices: zero response contains '-' values (not a clean holiday)")
        empty = pd.DataFrame()
        return MarketDaySnapshot(day, empty, empty, empty, counts, is_holiday=True)
    if bool((frame["close_price"] <= 0).any()):
        raise BulkResponseError(f"market_prices: {int((frame['close_price'] <= 0).sum())} rows without close price")
    iso = pd.Timestamp(day).strftime("%Y-%m-%d")
    prices = frame[["stock_code", *OHLC_COLUMNS, "volume", "change_vs_prev"]].assign(date=iso)
    caps_input = pd.DataFrame({"날짜": pd.Timestamp(day), "시가총액": frame["market_cap"], "상장주식수": frame["shares_outstanding"]})
    caps = _apply_ticker_normalizer(normalize_market_cap_df, caps_input, frame["stock_code"])
    return MarketDaySnapshot(day, prices, caps, frame[["stock_code", "market_type"]].copy(), counts)


@dataclass(frozen=True)
class UniverseChangeLimit:
    min_rows: int = 30
    max_ratio: float = 0.02


def check_universe_change(today: Set[str], previous: Set[str], limit: UniverseChangeLimit = UniverseChangeLimit()) -> dict:
    """전일 명단 대비 추가·제외 종목. 허용치를 넘으면 잘린 응답으로 보고 중단한다."""
    added, removed = sorted(today - previous), sorted(previous - today)
    allowed = max(limit.min_rows, int(limit.max_ratio * max(len(previous), 1)))
    if previous and len(added) + len(removed) > allowed:
        raise BulkResponseError(f"universe_change_exceeds_limit(added={len(added)}, removed={len(removed)}, allowed={allowed})")
    return {"added": added, "removed": removed, "previous_size": len(previous), "size": len(today)}


# ---- 재무 -------------------------------------------------------------------

def normalize_fundamental_day(raw: pd.DataFrame, day: str, universe_codes: Iterable[str]):
    """반환: (행 프레임, 정보). 명단 밖 종목(KONEX 등)은 버리고 수를 센다."""
    require_columns(raw, ["ISU_SRT_CD", "BPS", "PER", "PBR", "EPS", "DVD_YLD", "DPS"], "fundamentals")
    codes = validate_ticker_codes(raw["ISU_SRT_CD"], "fundamentals")
    keep = codes.isin(set(universe_codes))
    frame = pd.DataFrame({"날짜": pd.Timestamp(day)}, index=raw.index[keep])
    for source, target in {"PER": "PER", "PBR": "PBR", "EPS": "EPS", "BPS": "BPS", "DPS": "DPS", "DVD_YLD": "DIV"}.items():
        frame[target] = parse_krx_number(raw.loc[keep, source])
    if bool(frame.drop(columns="날짜").isna().any(axis=None)):
        raise BulkResponseError("fundamentals: unparsable values")
    out = _apply_ticker_normalizer(normalize_fundamental_df, frame, codes[keep])
    missing = sorted(set(universe_codes) - set(codes))
    return out, {"outside_universe": int((~keep).sum()), "missing_from_response": len(missing),
                 "missing_sample": missing[:20], "rows": int(len(out))}


# ---- 투자자 -----------------------------------------------------------------

def investor_response_series(raw: pd.DataFrame, label: str) -> pd.Series:
    """응답 하나를 받자마자 검사한다(빈 응답이면 다음 요청을 보내기 전에 중단)."""
    require_columns(raw, ["ISU_SRT_CD", "NETBID_TRDVAL"], label)
    codes = validate_ticker_codes(raw["ISU_SRT_CD"], label)
    parsed = parse_krx_field(raw["NETBID_TRDVAL"])
    _require_not_damaged(parsed, label)  # 숫자 손상은 누락과 구분해 즉시 중단
    return pd.Series(parsed.values.to_numpy(), index=codes.to_numpy())


def normalize_investor_day(series_by_key: Mapping[str, List[pd.Series]], day: str, universe_codes: Iterable[str]):
    """
    외국인 = 외국인(9000) + 기타외국인(9001): 종목별 일반 조회의 '외국인합계'와 같은 정의.
    네 주체 응답에 모두 값이 있는 종목만 행을 만든다. 일부 주체에만 있는 종목은 누락인지 0인지
    알 수 없으므로 partial로 세고 저장하지 않는다.
    """
    merged = {}
    for key in INVESTOR_CODES:
        joined = pd.concat(series_by_key[key])
        if joined.index.duplicated().any():
            raise BulkResponseError(f"investor:{key}: duplicated ticker codes across markets")
        merged[key] = joined
    wide = pd.DataFrame(merged)
    universe = set(universe_codes)
    outside = ~wide.index.isin(universe)
    wide = wide.loc[~outside]
    complete = wide.notna().all(axis=1)
    partial = sorted(wide.index[wide.notna().any(axis=1) & ~complete])
    dash_only = sorted(wide.index[wide.isna().all(axis=1)])
    unobserved = sorted(universe - set(wide.index))
    full = wide.loc[complete]
    all_zero = int((full.fillna(0) == 0).all(axis=1).sum())
    frame = pd.DataFrame({
        "날짜": pd.Timestamp(day),
        "기관합계": full["institution"].to_numpy(),
        "개인": full["individual"].to_numpy(),
        "외국인합계": (full["foreigner"] + full["other_foreigner"]).to_numpy(),
    })
    out = _apply_ticker_normalizer(normalize_investor_df, frame, pd.Series(full.index.to_numpy())) if len(full) else pd.DataFrame()
    info = {"outside_universe": int(outside.sum()), "partial": partial, "unobserved": unobserved,
            "dash_only": dash_only, "all_zero_excluded": all_zero, "rows": int(len(out))}
    return out, info


# ---- 공매도 -----------------------------------------------------------------

def short_response_frame(raw: pd.DataFrame, label: str) -> pd.DataFrame:
    require_columns(raw, ["ISU_CD", "BAL_AMT"], label)
    codes = validate_ticker_codes(raw["ISU_CD"], label)
    parsed = parse_krx_field(raw["BAL_AMT"])
    _require_not_damaged(parsed, label)
    return pd.DataFrame({"stock_code": codes.to_numpy(), "잔고금액": parsed.values.to_numpy()})


def normalize_short_balance_day(frames: Sequence[pd.DataFrame], day: str, universe_codes: Iterable[str]):
    """과거 저장분과 같게 잔고금액(short_balance_value)만 채우고 나머지 세 열은 NULL로 둔다."""
    merged = pd.concat(frames, ignore_index=True)
    if merged["stock_code"].duplicated().any():
        raise BulkResponseError("short_balance: duplicated ticker codes")
    universe = set(universe_codes)
    inside = merged["stock_code"].isin(universe)
    unpublished = sorted(merged.loc[inside & merged["잔고금액"].isna(), "stock_code"])
    kept = merged.loc[inside & merged["잔고금액"].notna()].reset_index(drop=True)
    kept["날짜"] = pd.Timestamp(day)
    out = _apply_ticker_normalizer(normalize_short_selling_df, kept[["날짜", "잔고금액"]], kept["stock_code"]) if len(kept) else pd.DataFrame()
    info = {
        "outside_universe": int((~inside).sum()),
        "unpublished": unpublished,
        "missing_from_response": sorted(universe - set(merged["stock_code"])),
        "rows": int(len(out)),
    }
    return out, info


# ---- 수정주가: 단위가 확인된 종목만 이어 쓰기 -----------------------------

@dataclass(frozen=True)
class PriorPriceState:
    date: date
    close_price: float
    adj_close: Optional[float]
    adj_ratio: Optional[float]


@dataclass
class AdjustedPriceResult:
    rows: pd.DataFrame
    quarantined: Dict[str, str] = field(default_factory=dict)


def _is_unit_ratio(state: PriorPriceState) -> bool:
    """직전 저장 행이 수정값 = 원시값(비율 1)인지. 이 경우에만 저장 수정종가를 정확히 재현할 수 있다."""
    return (
        state.adj_ratio is not None and float(state.adj_ratio) == 1.0
        and state.adj_close is not None and float(state.adj_close) == float(state.close_price)
    )


def classify_adjusted_row(row, prior: Mapping[str, PriorPriceState], previous_trading_date: Optional[date]) -> Optional[str]:
    """None이면 이어 쓰기 가능, 아니면 격리 사유."""
    state = prior.get(row["stock_code"])
    if state is None:
        return "no_prior_row"
    if previous_trading_date is None or state.date != previous_trading_date:
        return "gap_since_prior_row"
    if not _is_unit_ratio(state):
        return "prior_ratio_not_one"
    change = row["change_vs_prev"]
    if change is None or not np.isfinite(float(change)):
        return "unparsable_change"
    if abs(float(row["close_price"]) - float(change) - float(state.close_price)) > BASE_PRICE_TOLERANCE:
        return "base_price_mismatch"
    return None


def apply_adjusted_carry_forward(prices: pd.DataFrame, prior: Mapping[str, PriorPriceState], previous_trading_date: Optional[date]) -> AdjustedPriceResult:
    """
    비율 1로 저장돼 온 종목이 사건 없이(기준가 = 직전 저장 종가) 이어지면 수정값 = 원시값으로 쓴다.
    그 외(사건 의심·비율≠1·신규·공백·해석 불가)는 격리해 이 경로로 원본에 넣지 않는다.
    NULL 수정값을 만들지 않으므로 다음 날로 NULL 비율이 이어지지 않는다(격리 종목은 다음 날 공백 사유로 다시 격리).
    """
    quarantined, rows = {}, []
    for _, row in prices.iterrows():
        reason = classify_adjusted_row(row, prior, previous_trading_date)
        if reason:
            quarantined[row["stock_code"]] = reason
            continue
        record = row.to_dict()
        record.update({
            "adj_close": float(row["close_price"]), "adj_ratio": 1.0,
            "adj_open": float(row["open_price"]), "adj_high": float(row["high_price"]), "adj_low": float(row["low_price"]),
        })
        rows.append(record)
    columns = list(prices.columns) + ["adj_close", "adj_ratio", "adj_open", "adj_high", "adj_low"]
    return AdjustedPriceResult(pd.DataFrame(rows, columns=columns), quarantined)


def next_prior_state(written_prices: pd.DataFrame, day: date, prior: Dict[str, PriorPriceState]) -> Dict[str, PriorPriceState]:
    """같은 실행 안에서 실제로 쓴 행만 다음 거래일 판정의 직전 상태가 된다."""
    updated = dict(prior)
    for row in written_prices.itertuples(index=False):
        updated[row.stock_code] = PriorPriceState(day, float(row.close_price), float(row.adj_close), float(row.adj_ratio))
    return updated
