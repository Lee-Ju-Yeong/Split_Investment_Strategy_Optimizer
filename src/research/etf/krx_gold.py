"""KRX 금현물 일별 시세 수집(수동 다운로드 파일 적재)·정규화·품질 검사.

출처: KRX 정보데이터시스템(data.krx.co.kr) → 기본 통계 → 일반상품 → 금 → 개별종목 시세 추이.
2026-11-15 확인 기준으로 이 화면의 조회는 로그인한 세션에서만 되고(비로그인 요청은 ``LOGOUT``),
홈페이지 이용약관(2026-08-29 시행) 제10조 ②는 자동화 수단을 이용한 무단 수집·복제·배포를,
제12조 ②는 사전 허락 없는 복사·복제·배포·전송을 금지한다. 그래서 이 모듈은 사이트에 접속하지 않는다.
사용자가 로그인해 직접 내려받은 CSV를 읽어 원본을 보존하고, 정규화·품질 검사만 한다.
자료는 저장소 밖 ``MAGICSPLIT_ETF_DATA_ROOT/krx_gold``에만 두고 커밋·배포하지 않는다.

내려받은 CSV(cp949)의 열: 일자, 종가, 대비, 등락률, 시가, 고가, 저가, 거래량, 거래대금.
가격 단위는 원/g, 거래량 단위는 g, 거래대금 단위는 원이다(거래량 × 가격 ≈ 거래대금으로 확인).
파일에는 상품명이 없으므로 상품은 호출하는 쪽이 지정한다.

사용 예(PowerShell):
    $env:MAGICSPLIT_ETF_DATA_ROOT = "C:\\...\\split-research-data\\etf_m1"
    $env:MAGICSPLIT_ETF_RAW_ROOT = "C:\\...\\split-research-data\\2026-11-15-chatgpt\\extracted"
    python -m src.research.etf.krx_gold --inbox C:\\...\\split-research-data --pattern "data_*.csv" --move
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.research.etf.paths import FX_DIR, KRX_GOLD_DIR, data_root, output_dir, raw_root

SOURCE_NAME = "KRX 정보데이터시스템 일반상품>금>개별종목 시세 추이 (로그인 후 수동 CSV 다운로드)"
SOURCE_URL = "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd"
TERMS_NOTE = (
    "KRX Data Marketplace 홈페이지 이용약관(2026-08-29 시행) 제10조②: 자동화 수단을 이용한 정보 무단 "
    "수집·복제·배포 금지. 제12조②: 거래소 사전 허락 없는 복사·복제·배포·전송·공중송신 금지. "
    "개인 연구용으로 저장소 밖에만 보관하고 커밋·공유하지 않는다. 2026-11-15 기준 조회에 로그인 필요."
)

# 상품 키 → 표시 이름. KRX 금시장 상장 상품(1kg은 2014-03-24 개장일부터, 미니금 100g은 이후 상장).
PRODUCTS = {
    "gold_9999_1kg": "금 99.99_1kg",
    "gold_9999_100g": "미니금 99.99_100g",
}
DEFAULT_PRODUCT = "gold_9999_1kg"

# 원천 열 이름(한국어) → 정규화 열 이름
HEADER_MAP = {
    "일자": "date",
    "종가": "close",
    "대비": "change",
    "등락률": "change_pct",
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "거래량": "volume",
    "거래대금": "value",
}
REQUIRED_SOURCE_COLUMNS = tuple(HEADER_MAP)

NORMALIZED_COLUMNS = [
    "date", "product", "product_name",
    "open", "high", "low", "close",          # 원/g
    "volume",                                # g
    "value",                                 # 원
    "change", "change_pct",                  # 원천의 전일 대비(원/g)·등락률(%)
    "price_unit", "volume_unit",
    "source", "source_file", "source_sha256",
    "downloaded_at_utc",                     # 원천 파일 수정 시각(다운로드 시각 근사)
    "ingested_at_utc",                       # 이 모듈이 적재한 시각
]
PRICE_COLUMNS = ("open", "high", "low", "close")
DEFAULT_JUMP_THRESHOLD = 0.05   # |일간 로그수익률| 5% 초과를 급변일로 표시
PRICE_LIMIT = 0.10              # KRX 금시장 일일 가격제한폭(기준가격 대비 ±10%)
ETF_GOLD_SYMBOL = "411060"      # ACE KRX금현물(2021-12-15 상장)
ETF_PRICE_RELPATH = Path("kr_etf_price_history_2026-11-15/history_kr/prices") / f"{ETF_GOLD_SYMBOL}.csv"


class KrxGoldFormatError(ValueError):
    """원천 CSV 형식이 예상과 다를 때."""


@dataclass(frozen=True)
class SourceFile:
    name: str
    sha256: str
    size_bytes: int
    downloaded_at_utc: str
    raw_path: Path


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------

def decode_bytes(data: bytes) -> str:
    """KRX CSV는 cp949다. UTF-8(BOM 포함)로 저장된 사본도 받아들인다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise KrxGoldFormatError("CSV 인코딩을 cp949/UTF-8로 읽을 수 없습니다.")


def _to_number(text: str, *, column: str, line: int) -> float:
    cleaned = (text or "").strip().replace(",", "")
    if cleaned in ("", "-"):
        return math.nan
    try:
        return float(cleaned)
    except ValueError as exc:
        raise KrxGoldFormatError(f"{line}행 {column} 값이 숫자가 아닙니다: {text!r}") from exc


def parse_krx_gold_csv(data: bytes) -> pd.DataFrame:
    """KRX '개별종목 시세 추이' CSV 한 개를 원천 열 그대로(이름만 영문) 읽는다."""
    text = decode_bytes(data)
    reader = csv.reader(io.StringIO(text))
    try:
        header = [h.strip() for h in next(reader)]
    except StopIteration as exc:
        raise KrxGoldFormatError("빈 CSV입니다.") from exc
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in header]
    if missing:
        raise KrxGoldFormatError(f"필수 열이 없습니다: {missing} (받은 열: {header})")
    index = {name: header.index(name) for name in REQUIRED_SOURCE_COLUMNS}

    records = []
    for line_no, row in enumerate(reader, start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) < len(header):
            raise KrxGoldFormatError(f"{line_no}행 열 개수가 부족합니다: {row}")
        raw_date = row[index["일자"]].strip()
        try:
            date = datetime.strptime(raw_date.replace("-", "/"), "%Y/%m/%d").date()
        except ValueError as exc:
            raise KrxGoldFormatError(f"{line_no}행 일자 형식이 다릅니다: {raw_date!r}") from exc
        record = {"date": date}
        for source_col, target_col in HEADER_MAP.items():
            if target_col == "date":
                continue
            record[target_col] = _to_number(row[index[source_col]], column=source_col, line=line_no)
        records.append(record)
    frame = pd.DataFrame.from_records(records, columns=list(HEADER_MAP.values()))
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


# ---------------------------------------------------------------------------
# 원본 보존과 정규화
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_iso(ts: float | None = None) -> str:
    moment = datetime.now(timezone.utc) if ts is None else datetime.fromtimestamp(ts, timezone.utc)
    return moment.replace(microsecond=0).isoformat()


def preserve_raw(source: Path, raw_dir: Path, *, move: bool = False) -> SourceFile:
    """원본 바이트를 그대로 raw_dir에 복사(또는 이동)한다. 같은 이름·다른 내용이면 덮어쓰지 않고 멈춘다."""
    data = source.read_bytes()
    digest = _sha256(data)
    downloaded_at = _utc_iso(source.stat().st_mtime)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / source.name
    if target.exists():
        if _sha256(target.read_bytes()) != digest:
            raise FileExistsError(f"raw에 같은 이름의 다른 파일이 있습니다: {target}")
    else:
        shutil.copy2(source, target)   # 수정 시각 보존
        if _sha256(target.read_bytes()) != digest:
            raise IOError(f"복사본 해시가 원본과 다릅니다: {target}")
    if move and source.resolve() != target.resolve():
        source.unlink()
    return SourceFile(source.name, digest, len(data), downloaded_at, target)


def normalize(
    parsed: Sequence[tuple[SourceFile, pd.DataFrame]],
    *,
    product: str = DEFAULT_PRODUCT,
    ingested_at_utc: str | None = None,
) -> pd.DataFrame:
    """여러 기간 파일을 합쳐 날짜 오름차순 일봉으로 만든다.

    같은 날짜가 여러 파일에 있으면 값이 모두 같을 때만 하나로 줄이고, 다르면 오류를 낸다.
    """
    if product not in PRODUCTS:
        raise ValueError(f"알 수 없는 상품: {product} (가능: {sorted(PRODUCTS)})")
    ingested_at_utc = ingested_at_utc or _utc_iso()
    frames = []
    for source, frame in parsed:
        f = frame.copy()
        f["source_file"] = source.name
        f["source_sha256"] = source.sha256
        f["downloaded_at_utc"] = source.downloaded_at_utc
        frames.append(f)
    if not frames:
        raise ValueError("적재할 파일이 없습니다.")
    combined = pd.concat(frames, ignore_index=True)

    value_cols = [c for c in HEADER_MAP.values() if c != "date"]
    dup_mask = combined.duplicated("date", keep=False)
    if dup_mask.any():
        conflicts = (
            combined[dup_mask]
            .groupby("date")[value_cols]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        bad = conflicts[conflicts].index
        if len(bad):
            dates = ", ".join(d.strftime("%Y-%m-%d") for d in bad[:10])
            raise KrxGoldFormatError(f"같은 날짜에 서로 다른 값이 있습니다: {dates}")
    combined = (
        combined.sort_values(["date", "source_file"])
        .drop_duplicates("date", keep="first")
        .reset_index(drop=True)
    )
    combined["product"] = product
    combined["product_name"] = PRODUCTS[product]
    combined["price_unit"] = "KRW/g"
    combined["volume_unit"] = "g"
    combined["source"] = SOURCE_NAME
    combined["ingested_at_utc"] = ingested_at_utc
    return combined[NORMALIZED_COLUMNS]


# ---------------------------------------------------------------------------
# 품질 검사
# ---------------------------------------------------------------------------

def load_trading_calendar(path: Path) -> pd.DatetimeIndex:
    """국내 거래일 목록(``date`` 열). fx 단계 산출물 ``fx/calendar_krx.csv``를 읽기 전용으로 쓴다."""
    cal = pd.read_csv(path, usecols=["date"])
    return pd.DatetimeIndex(pd.to_datetime(cal["date"])).sort_values()


def _dates(values: Iterable[pd.Timestamp]) -> list[str]:
    return [pd.Timestamp(v).strftime("%Y-%m-%d") for v in values]


def quality_checks(
    daily: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex | None = None,
    jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
) -> tuple[dict, pd.DataFrame]:
    """정규화 일봉의 품질 요약(dict)과 급변일 표를 돌려준다. 자료를 고치지 않고 표시만 한다."""
    d = daily.sort_values("date").reset_index(drop=True)
    report: dict = {
        "rows": int(len(d)),
        "first_date": _dates([d["date"].iloc[0]])[0] if len(d) else None,
        "last_date": _dates([d["date"].iloc[-1]])[0] if len(d) else None,
        "duplicate_dates": int(d["date"].duplicated().sum()),
        "dates_sorted": bool(daily["date"].is_monotonic_increasing),
    }

    numeric = ["open", "high", "low", "close", "volume", "value"]
    missing = d[d[numeric].isna().any(axis=1)]
    report["missing_value_dates"] = _dates(missing["date"])
    nonpositive_price = d[(d[list(PRICE_COLUMNS)] <= 0).any(axis=1)]
    report["nonpositive_price_dates"] = _dates(nonpositive_price["date"])
    negative_volume = d[(d[["volume", "value"]] < 0).any(axis=1)]
    report["negative_volume_or_value_dates"] = _dates(negative_volume["date"])
    zero_volume = d[d["volume"] == 0]
    report["zero_volume_dates"] = _dates(zero_volume["date"])

    hi_bad = d["high"] < d[["open", "close", "low"]].max(axis=1)
    lo_bad = d["low"] > d[["open", "close"]].min(axis=1)
    report["ohlc_inconsistent_dates"] = _dates(d.loc[hi_bad | lo_bad, "date"])

    # 원천의 '대비'(전일 종가 대비)와 실제 전일 종가 차이가 맞는지: 행 누락·순서 오류를 잡는다.
    prev_close = d["close"].shift(1)
    change_gap = (d["close"] - prev_close - d["change"]).abs()
    change_bad = d.index[(d.index > 0) & (change_gap > 0.5)]
    report["change_mismatch_dates"] = _dates(d.loc[change_bad, "date"])

    # 거래대금 ÷ 거래량이 그날 저가~고가 범위 안인지(단위 확인). 1원 반올림 여유를 둔다.
    # 실제 자료에서는 범위 밖이 1% 안팎으로 드물게 나온다. 협의대량거래가 거래량·거래대금에 포함된
    # 것으로 보여 경고로만 센다(차단 문제 아님).
    traded = d[d["volume"] > 0]
    vwap = traded["value"] / traded["volume"]
    vwap_bad = traded[(vwap < traded["low"] - 1) | (vwap > traded["high"] + 1)]
    report["vwap_outside_range_dates"] = _dates(vwap_bad["date"])

    if calendar is not None and len(d):
        start, end = d["date"].iloc[0], d["date"].iloc[-1]
        window = calendar[(calendar >= start) & (calendar <= end)]
        have = pd.DatetimeIndex(d["date"])
        report["calendar_trading_days_in_range"] = int(len(window))
        report["missing_vs_calendar_dates"] = _dates(window.difference(have))
        report["extra_vs_calendar_dates"] = _dates(have.difference(window))
    else:
        report["calendar_trading_days_in_range"] = None
        report["missing_vs_calendar_dates"] = None
        report["extra_vs_calendar_dates"] = None

    ratio = (d["close"] / prev_close).where(lambda r: r > 0)
    log_ret = np.log(ratio)
    gap_days = d["date"].diff().dt.days
    # KRX 금시장 일일 가격제한폭은 기준가격 ±10%다. 전일 종가 대비 ±10%에 닿은 날을 표시한다
    # (기준가격이 전일 종가와 다를 수 있어 근사이며, 호가 단위 반올림 여유 0.05%p를 둔다).
    simple_ret = d["close"] / prev_close - 1
    at_limit = simple_ret.abs() >= PRICE_LIMIT - 0.0005
    jumps = d.assign(
        prev_close=prev_close, log_return=log_ret, calendar_gap_days=gap_days, at_price_limit=at_limit
    )
    jumps = jumps[jumps["log_return"].abs() > jump_threshold][
        ["date", "prev_close", "open", "high", "low", "close", "log_return",
         "calendar_gap_days", "at_price_limit", "volume"]
    ].reset_index(drop=True)
    report["jump_threshold_abs_log_return"] = jump_threshold
    report["jump_dates"] = _dates(jumps["date"])
    report["price_limit_close_dates"] = _dates(d.loc[at_limit.fillna(False), "date"])
    report["daily_log_return_stats"] = {
        "mean": float(log_ret.mean()) if len(d) > 1 else None,
        "std": float(log_ret.std()) if len(d) > 2 else None,
        "min": float(log_ret.min()) if len(d) > 1 else None,
        "max": float(log_ret.max()) if len(d) > 1 else None,
    }
    issues = (
        report["missing_value_dates"]
        + report["nonpositive_price_dates"]
        + report["negative_volume_or_value_dates"]
        + report["ohlc_inconsistent_dates"]
        + report["change_mismatch_dates"]
        + (report["missing_vs_calendar_dates"] or [])
        + (report["extra_vs_calendar_dates"] or [])
    )
    report["blocking_issue_count"] = len(issues) + report["duplicate_dates"] + int(not report["dates_sorted"])
    report["warning_count"] = len(report["vwap_outside_range_dates"]) + len(report["zero_volume_dates"])
    return report, jumps


def compare_with_etf(daily: pd.DataFrame, etf: pd.DataFrame) -> dict:
    """금현물 종가와 금현물 추종 ETF 종가의 겹치는 기간 비교(참고용).

    ETF 가격은 받은 자료의 공급자 수정가격이며 보수·분배 처리가 확인되지 않았다.
    """
    g = daily[["date", "close"]].rename(columns={"close": "gold"})
    e = etf[["date", "close"]].rename(columns={"close": "etf"})
    e = e.assign(date=pd.to_datetime(e["date"]))
    merged = g.merge(e, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if len(merged) < 3:
        return {"overlap_days": int(len(merged))}
    rets = merged[["gold", "etf"]].pct_change().dropna()
    years = (merged["date"].iloc[-1] - merged["date"].iloc[0]).days / 365.25
    total = merged.iloc[-1][["gold", "etf"]] / merged.iloc[0][["gold", "etf"]]
    cagr = total ** (1 / years) - 1 if years > 0 else total * math.nan
    diff = rets["etf"] - rets["gold"]
    return {
        "overlap_days": int(len(merged)),
        "first_date": _dates([merged["date"].iloc[0]])[0],
        "last_date": _dates([merged["date"].iloc[-1]])[0],
        "daily_return_correlation": float(rets["gold"].corr(rets["etf"])),
        "total_return_gold": float(total["gold"] - 1),
        "total_return_etf": float(total["etf"] - 1),
        "cagr_gold": float(cagr["gold"]),
        "cagr_etf": float(cagr["etf"]),
        "tracking_difference_cagr": float(cagr["etf"] - cagr["gold"]),
        "tracking_error_annualized": float(diff.std() * math.sqrt(252)),
        "note": "ETF 종가는 받은 자료(provider_adjusted, 분배·보수 처리 미확인)이며 참고용이다.",
    }


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def ingest(
    files: Sequence[Path],
    *,
    product: str = DEFAULT_PRODUCT,
    move: bool = False,
    calendar_path: Path | None = None,
    etf_path: Path | None = None,
    jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
) -> dict:
    out = output_dir(KRX_GOLD_DIR)
    raw_dir = out / "raw" / product
    ingested_at = _utc_iso()

    # 일봉은 매번 raw에 보존된 모든 파일 + 새 파일로 다시 만든다(일부 기간만 다시 넣어도 이력이 줄지 않는다).
    # 형식 검사·이름 충돌·중복 날짜 충돌 검사를 모두 통과한 뒤에만 raw로 복사·이동한다.
    def _describe(path: Path) -> tuple[SourceFile, pd.DataFrame]:
        data = path.read_bytes()
        src = SourceFile(path.name, _sha256(data), len(data), _utc_iso(path.stat().st_mtime), raw_dir / path.name)
        return src, parse_krx_gold_csv(data)

    existing = {p.name: _describe(p) for p in sorted(raw_dir.glob("*.csv"))} if raw_dir.exists() else {}
    staged = []
    for path in sorted(files):
        src, frame = _describe(path)
        if src.name in existing and existing[src.name][0].sha256 != src.sha256:
            raise FileExistsError(f"raw에 같은 이름의 다른 파일이 있습니다: {raw_dir / src.name}")
        staged.append((path, src, frame))
    combined = dict(existing)
    for _, src, frame in staged:
        combined.setdefault(src.name, (src, frame))
    daily = normalize(list(combined.values()), product=product, ingested_at_utc=ingested_at)
    for path, _, _ in staged:
        preserve_raw(path, raw_dir, move=move)
    parsed = list(combined.values())
    daily_path = out / f"{product}_daily.csv"
    out_frame = daily.assign(date=daily["date"].dt.strftime("%Y-%m-%d"))
    out_frame.to_csv(daily_path, index=False, encoding="utf-8")

    calendar = None
    if calendar_path is None:
        default_cal = data_root() / FX_DIR / "calendar_krx.csv"
        calendar_path = default_cal if default_cal.exists() else None
    if calendar_path is not None:
        calendar = load_trading_calendar(calendar_path)
    report, jumps = quality_checks(daily, calendar=calendar, jump_threshold=jump_threshold)
    jumps_path = out / f"{product}_jumps.csv"
    jumps.assign(date=jumps["date"].dt.strftime("%Y-%m-%d")).to_csv(jumps_path, index=False)

    if etf_path is None:
        try:
            candidate = raw_root() / ETF_PRICE_RELPATH
            etf_path = candidate if candidate.exists() else None
        except RuntimeError:
            etf_path = None
    etf_cmp = compare_with_etf(daily, pd.read_csv(etf_path)) if etf_path is not None else None

    quality = {
        "product": product,
        "product_name": PRODUCTS[product],
        "generated_at_utc": ingested_at,
        "calendar_path": calendar_path,
        "checks": report,
        "etf_comparison": {"symbol": ETF_GOLD_SYMBOL, "path": etf_path, **(etf_cmp or {})} if etf_cmp else None,
    }
    quality_path = out / f"{product}_quality.json"
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    manifest = {
        "product": product,
        "product_name": PRODUCTS[product],
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "terms": TERMS_NOTE,
        "ingested_at_utc": ingested_at,
        "rows": int(len(daily)),
        "first_date": report["first_date"],
        "last_date": report["last_date"],
        "outputs": {"daily": daily_path, "quality": quality_path, "jumps": jumps_path},
        "raw_files": [
            {
                "name": s.name,
                "sha256": s.sha256,
                "bytes": s.size_bytes,
                "downloaded_at_utc": s.downloaded_at_utc,
                "raw_path": s.raw_path,
                "rows": int(len(f)),
                "first_date": _dates([f["date"].min()])[0] if len(f) else None,
                "last_date": _dates([f["date"].max()])[0] if len(f) else None,
            }
            for s, f in parsed
        ],
    }
    manifest_path = out / f"{product}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return {"manifest": manifest, "quality": quality}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KRX 금현물 수동 다운로드 CSV 적재·정규화·품질 검사")
    parser.add_argument("files", nargs="*", type=Path, help="KRX에서 내려받은 CSV 파일들")
    parser.add_argument("--inbox", type=Path, help="CSV를 찾을 폴더")
    parser.add_argument("--pattern", default="*.csv", help="--inbox 안에서 찾을 파일 패턴")
    parser.add_argument("--product", default=DEFAULT_PRODUCT, choices=sorted(PRODUCTS))
    parser.add_argument("--move", action="store_true", help="raw 보존 후 원래 파일을 지운다(해시 확인 후)")
    parser.add_argument("--calendar", type=Path, help="국내 거래일 CSV(기본: <DATA_ROOT>/fx/calendar_krx.csv)")
    parser.add_argument("--etf", type=Path, help="411060 가격 CSV(기본: <RAW_ROOT>/kr_etf_price_history_.../411060.csv)")
    parser.add_argument("--jump-threshold", type=float, default=DEFAULT_JUMP_THRESHOLD)
    args = parser.parse_args(argv)

    files = list(args.files)
    if args.inbox:
        files += sorted(args.inbox.glob(args.pattern))
    if not files:
        parser.error("적재할 CSV가 없습니다.")
    result = ingest(
        files,
        product=args.product,
        move=args.move,
        calendar_path=args.calendar,
        etf_path=args.etf,
        jump_threshold=args.jump_threshold,
    )
    m, q = result["manifest"], result["quality"]["checks"]
    print(f"{m['product_name']}: {m['rows']}행 {m['first_date']}~{m['last_date']}, "
          f"차단 문제 {q['blocking_issue_count']}건, 경고 {q['warning_count']}건, "
          f"급변일 {len(q['jump_dates'])}일, 가격제한폭 종가 {len(q['price_limit_close_dates'])}일")
    print(f"출력: {m['outputs']['daily']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
