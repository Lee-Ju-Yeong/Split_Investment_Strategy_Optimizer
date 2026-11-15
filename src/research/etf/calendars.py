"""국내(KRX)·미국(NYSE) 거래일과 정규장 시각(UTC).

계획 D2 시각 명세: 모든 사건은 UTC 시각 순서로 처리하고, 미국 서머타임과
조기 폐장은 NYSE 거래소 달력을 따른다.

생성(build)과 사용(load)을 나눈다.
- 생성: 공개 라이브러리 `exchange_calendars`(버전 고정, PINNED_LIBRARY)로
  2004~2027 달력을 만들고, 라이브러리가 놓친 공지 사항을 KRX_OVERRIDES로 보정해
  `MAGICSPLIT_ETF_DATA_ROOT/fx/calendar_{krx,nyse}.csv`로 저장한다.
- 사용: 저장된 CSV만 읽는다. 엔진·테스트는 라이브러리나 네트워크가 필요 없다.

CSV 열(한 행 = 한 거래일):
  date               거래일(현지 날짜, YYYY-MM-DD)
  market             "KRX" | "NYSE"
  open_local/close_local  현지 정규장 시작·마감 시각(HH:MM)
  open_utc/close_utc      같은 시각의 UTC(ISO 8601, +00:00)
  utc_offset_hours   그날 현지 시각의 UTC 차이(KST +9, 뉴욕 −5/−4)
  is_dst             미국 서머타임 적용일(KRX는 항상 False)
  early_close        정상보다 일찍 마감(NYSE 13:00 등)
  late_open          정상보다 늦게 개장(KRX 새해 첫날·수능일 10:00)
  late_close         정상보다 늦게 마감(KRX 수능일)
  source             "exchange_calendars==<버전>" 또는 "override:<근거>"
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

PINNED_LIBRARY = "exchange_calendars==4.13.3"
CALENDAR_START = "2004-01-01"
CALENDAR_END = "2027-12-31"

MARKETS = {
    # market: (exchange_calendars 이름, 시간대)
    "KRX": ("XKRX", "Asia/Seoul"),
    "NYSE": ("XNYS", "America/New_York"),
}

# 정상 시각. KRX 정규장 마감은 2016-08-01부터 15:00 → 15:30으로 30분 연장됐다.
KRX_CLOSE_EXTENSION_DATE = date(2016, 8, 1)


def regular_hours(market: str, day: date) -> tuple[time, time]:
    """특별 사정이 없는 날의 정규장 시작·마감(현지 시각)."""
    if market == "KRX":
        close = time(15, 30) if day >= KRX_CLOSE_EXTENSION_DATE else time(15, 0)
        return time(9, 0), close
    if market == "NYSE":
        return time(9, 30), time(16, 0)
    raise ValueError(f"알 수 없는 시장: {market}")


@dataclass(frozen=True)
class Override:
    market: str
    day: date
    kind: str                 # "closed" | "hours"
    open_local: time | None
    close_local: time | None
    reason: str


# 라이브러리(4.13.3)가 반영하지 못한 한국거래소 공지. 근거는 보고서·manifest에 남긴다.
KRX_OVERRIDES: tuple[Override, ...] = (
    Override("KRX", date(2026, 7, 17), "closed", None, None,
             "제헌절 공휴일 재지정(2026-01-29 국회 통과, 2026-05-11 시행). "
             "한국거래소 2026-05-20 휴장 공지. 069500 실제 자료에도 거래 없음"),
    Override("KRX", date(2026, 11, 19), "hours", time(10, 0), time(16, 30),
             "2027학년도 수능일. 한국거래소 2026-11-05 공지: 정규장 10:00~16:30"),
)

# 확인하지 못해 라이브러리 값을 그대로 둔 날. 보고서에 남긴다.
UNVERIFIED_DAYS: dict[str, dict[date, str]] = {
    "KRX": {
        date(2017, 11, 16): "포항 지진으로 수능이 11-23로 연기됨. 라이브러리는 11-16·11-23 모두 "
                            "10:00~16:30로 둔다. 11-16 실제 운영 시각은 공지를 찾지 못함",
    },
}

# 아직 공지되지 않아 달력에 반영할 수 없는 일정(생성 시점 2026-11-15 기준).
KNOWN_GAPS = (
    "KRX 2027-11 수능일(2028학년도) 시간 변경은 미공지라 반영하지 않음",
    "KRX·NYSE 2027년 임시 휴장(선거·국가 애도일 등)은 라이브러리 규칙 기반 추정",
)


def local_to_utc(day: date, at: time, tz: str) -> datetime:
    """현지 날짜·시각을 UTC로 바꾼다(서머타임 포함)."""
    local = datetime.combine(day, at, tzinfo=ZoneInfo(tz))
    return local.astimezone(timezone.utc)


def _row(market: str, day: date, open_local: time, close_local: time, source: str) -> dict:
    tz = MARKETS[market][1]
    open_utc = local_to_utc(day, open_local, tz)
    close_utc = local_to_utc(day, close_local, tz)
    reg_open, reg_close = regular_hours(market, day)
    offset = datetime.combine(day, open_local, tzinfo=ZoneInfo(tz)).utcoffset()
    dst = datetime.combine(day, open_local, tzinfo=ZoneInfo(tz)).dst()
    return {
        "date": day.isoformat(),
        "market": market,
        "open_local": open_local.strftime("%H:%M"),
        "close_local": close_local.strftime("%H:%M"),
        "open_utc": open_utc.isoformat(),
        "close_utc": close_utc.isoformat(),
        "utc_offset_hours": offset.total_seconds() / 3600,
        "is_dst": bool(dst and dst.total_seconds() > 0),
        "early_close": close_local < reg_close,
        "late_open": open_local > reg_open,
        "late_close": close_local > reg_close,
        "source": source,
    }


def _as_date(x) -> date:
    return pd.Timestamp(x).date()


def apply_overrides(rows: list[dict], overrides: Iterable[Override],
                    start: str | date | None = None, end: str | date | None = None) -> list[dict]:
    """공지 보정을 적용한다. closed는 행 삭제, hours는 시각 교체(없던 날이면 추가).

    start/end를 주면 그 기간(양끝 포함) 안의 보정만 적용한다. 기간 밖 보정이 행을 추가하지 않게 한다.
    """
    lo = _as_date(start) if start is not None else date.min
    hi = _as_date(end) if end is not None else date.max
    by_day = {r["date"]: r for r in rows}
    for ov in overrides:
        if not lo <= ov.day <= hi:
            continue
        key = ov.day.isoformat()
        if ov.kind == "closed":
            by_day.pop(key, None)
        elif ov.kind == "hours":
            by_day[key] = _row(ov.market, ov.day, ov.open_local, ov.close_local,
                               f"override:{ov.reason}")
        else:
            raise ValueError(f"알 수 없는 보정 종류: {ov.kind}")
    return [by_day[k] for k in sorted(by_day)]


def build_calendar(market: str, start: str = CALENDAR_START, end: str = CALENDAR_END) -> pd.DataFrame:
    """exchange_calendars로 달력을 만든다(생성 단계에서만 호출, 라이브러리 필요)."""
    import exchange_calendars as xc  # 생성 단계 전용 의존성

    name, tz = MARKETS[market]
    version = getattr(xc, "__version__", "unknown")
    if f"exchange_calendars=={version}" != PINNED_LIBRARY:
        raise RuntimeError(f"{PINNED_LIBRARY}로 고정했는데 설치된 버전은 {version}입니다.")
    # 라이브러리는 start < end를 요구하므로 앞쪽을 넓혀 받은 뒤 요청 기간으로 자른다.
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    if hi < lo:
        raise ValueError(f"end({end})가 start({start})보다 빠릅니다.")
    sched = xc.get_calendar(name, start=lo - pd.Timedelta(days=7), end=hi).schedule
    sched = sched[(sched.index >= lo) & (sched.index <= hi)]
    rows = []
    for session, rec in sched.iterrows():
        day = session.date()
        open_local = rec["open"].tz_convert(tz).time()
        close_local = rec["close"].tz_convert(tz).time()
        rows.append(_row(market, day, open_local, close_local, PINNED_LIBRARY))
    overrides = [o for o in KRX_OVERRIDES if o.market == market]
    out = apply_overrides(rows, overrides, start, end)
    check_within_range(out, start, end)
    return pd.DataFrame(out)


def check_within_range(rows: list[dict], start, end) -> None:
    """결과의 모든 거래일이 요청 기간(양끝 포함) 안에 있는지 확인한다."""
    lo, hi = _as_date(start), _as_date(end)
    outside = [r["date"] for r in rows if not lo <= date.fromisoformat(r["date"]) <= hi]
    if outside:
        raise ValueError(f"요청 기간 {lo}~{hi} 밖의 거래일이 있습니다: {outside[:5]}")


def calendar_path(out_dir: Path, market: str) -> Path:
    return Path(out_dir) / f"calendar_{market.lower()}.csv"


def write_calendars(out_dir: Path) -> dict:
    """두 달력을 CSV로 저장하고 manifest를 남긴다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "library": PINNED_LIBRARY,
        "range": [CALENDAR_START, CALENDAR_END],
        "overrides": [
            {"market": o.market, "date": o.day.isoformat(), "kind": o.kind,
             "open_local": o.open_local.strftime("%H:%M") if o.open_local else None,
             "close_local": o.close_local.strftime("%H:%M") if o.close_local else None,
             "reason": o.reason}
            for o in KRX_OVERRIDES
        ],
        "unverified_days": {m: {d.isoformat(): r for d, r in v.items()} for m, v in UNVERIFIED_DAYS.items()},
        "known_gaps": list(KNOWN_GAPS),
        "files": {},
    }
    for market in MARKETS:
        df = build_calendar(market)
        path = calendar_path(out_dir, market)
        df.to_csv(path, index=False, lineterminator="\n")
        manifest["files"][path.name] = {
            "rows": len(df),
            "first": df["date"].iloc[0],
            "last": df["date"].iloc[-1],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (out_dir / "calendar_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_calendar(path: Path) -> pd.DataFrame:
    """저장된 달력 CSV를 읽는다. date는 datetime64, *_utc는 UTC tz-aware."""
    df = pd.read_csv(path, dtype={"open_local": str, "close_local": str})
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open_utc", "close_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    for col in ("is_dst", "early_close", "late_open", "late_close"):
        df[col] = df[col].astype(bool)
    if df["date"].duplicated().any():
        raise ValueError(f"{path}: 중복 거래일이 있습니다.")
    if not df["date"].is_monotonic_increasing:
        raise ValueError(f"{path}: 거래일이 정렬되어 있지 않습니다.")
    if (df["close_utc"] <= df["open_utc"]).any():
        raise ValueError(f"{path}: 마감이 개장보다 빠른 행이 있습니다.")
    return df


def crosscheck_sessions(calendar_dates: Iterable, observed_dates: Iterable) -> dict:
    """달력 거래일과 실제 가격 자료의 거래일을 겹치는 기간에서 비교한다."""
    cal = pd.DatetimeIndex(pd.to_datetime(list(calendar_dates))).normalize()
    obs = pd.DatetimeIndex(pd.to_datetime(list(observed_dates))).normalize()
    lo, hi = max(cal.min(), obs.min()), min(cal.max(), obs.max())
    cal_set = set(cal[(cal >= lo) & (cal <= hi)])
    obs_set = set(obs[(obs >= lo) & (obs <= hi)])
    return {
        "overlap": (lo.date().isoformat(), hi.date().isoformat()),
        "calendar_sessions": len(cal_set),
        "observed_sessions": len(obs_set),
        "calendar_only": sorted(d.date().isoformat() for d in cal_set - obs_set),
        "observed_only": sorted(d.date().isoformat() for d in obs_set - cal_set),
    }


def combined_sessions(krx: pd.DataFrame, nyse: pd.DataFrame) -> pd.DataFrame:
    """국내·미국 거래일 합집합(계획 D9의 통합 거래일). 날짜별 어느 시장이 열렸는지 표시."""
    k = set(krx["date"])
    n = set(nyse["date"])
    days = sorted(k | n)
    return pd.DataFrame({
        "date": pd.DatetimeIndex(days),
        "krx_open": [d in k for d in days],
        "nyse_open": [d in n for d in days],
    })


def session_events_utc(*calendars: pd.DataFrame) -> pd.DataFrame:
    """여러 시장의 개장·마감 사건을 UTC 시각 순서로 펼친다(엔진 사건 큐의 뼈대)."""
    parts = []
    for cal in calendars:
        for kind in ("open", "close"):
            parts.append(pd.DataFrame({
                "at_utc": cal[f"{kind}_utc"],
                "market": cal["market"],
                "session_date": cal["date"],
                "event": kind,
            }))
    events = pd.concat(parts, ignore_index=True)
    # 같은 시각이면 마감을 개장보다 먼저 둔다(마감 확정 → 다음 판단).
    events["_order"] = (events["event"] == "open").astype(int)
    events = events.sort_values(["at_utc", "_order", "market"], kind="mergesort")
    return events.drop(columns="_order").reset_index(drop=True)


def crosscheck_price_files(cal: pd.DataFrame, reference_csv: Path, price_glob_dir: Path | None = None,
                           until: str | None = None) -> dict:
    """달력을 기준 ETF(069500·SPY) 거래일, 그리고 선택적으로 폴더 안 모든 ETF 거래일과 대조한다.

    until을 주면 그 날짜까지만 비교한다(자료 수집일 이후 미래 거래일 제외).
    """
    cal_dates = cal["date"]
    if until:
        cal_dates = cal_dates[cal_dates <= pd.Timestamp(until)]
    ref = pd.read_csv(reference_csv, usecols=["date"])["date"]
    report = {"reference": Path(reference_csv).name, **crosscheck_sessions(cal_dates, ref)}
    if price_glob_dir is not None:
        cal_set = set(pd.DatetimeIndex(cal_dates))
        lo, hi = min(cal_set), max(cal_set)
        extra: dict[str, list[str]] = {}
        files = sorted(Path(price_glob_dir).glob("*.csv"))
        for f in files:
            d = pd.to_datetime(pd.read_csv(f, usecols=["date"])["date"])
            d = d[(d >= lo) & (d <= hi)]
            for x in sorted(set(d) - cal_set):
                extra.setdefault(x.date().isoformat(), []).append(f.stem)
        report["all_files_checked"] = len(files)
        report["dates_in_some_file_not_in_calendar"] = extra
    return report


def last_session_on_or_before(cal: pd.DataFrame, at_utc: pd.Timestamp, need: str = "close") -> pd.Series | None:
    """주어진 UTC 시각까지 마감(또는 개장)이 끝난 마지막 거래일 행."""
    at_utc = pd.Timestamp(at_utc)
    if at_utc.tzinfo is None:
        raise ValueError("at_utc는 시간대가 있어야 합니다.")
    done = cal[cal[f"{need}_utc"] <= at_utc]
    return None if done.empty else done.iloc[-1]


def main(argv: list[str] | None = None) -> None:
    """생성·대조 실행: python -m src.research.etf.calendars [--raw-root 경로] [--until YYYY-MM-DD]."""
    import argparse

    from . import paths

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-root", help="ChatGPT 원천 자료 extracted 폴더(기본: MAGICSPLIT_ETF_RAW_ROOT)")
    p.add_argument("--until", default=None, help="대조 마지막 날짜(기본: 기준 자료 마지막 날)")
    args = p.parse_args(argv)

    out = paths.output_dir(paths.FX_DIR)
    manifest = write_calendars(out)
    raw = Path(args.raw_root) if args.raw_root else paths.raw_root()
    kr_dir = raw / "kr_etf_price_history_2026-11-15" / "history_kr" / "prices"
    us_dir = raw / "us_etf_price_history_2026-11-15" / "history_us" / "prices"
    checks = {}
    for market, ref, folder in (("KRX", kr_dir / "069500.csv", kr_dir), ("NYSE", us_dir / "SPY.csv", us_dir)):
        cal = load_calendar(calendar_path(out, market))
        until = args.until or pd.read_csv(ref, usecols=["date"])["date"].max()
        checks[market] = crosscheck_price_files(cal, ref, folder, until=until)
    (out / "calendar_crosscheck.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest["files"], "crosscheck": {
        m: {k: v for k, v in c.items() if k != "dates_in_some_file_not_in_calendar"}
        | {"extra_dates": len(c.get("dates_in_some_file_not_in_calendar", {}))}
        for m, c in checks.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
