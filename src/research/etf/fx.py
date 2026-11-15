"""원/달러 환율과 현금 금리 대용 지표, 그리고 각 값을 실제로 쓸 수 있는 시각(UTC).

계획 D2 시각 명세: 주문 판단에는 그 시각까지 실제로 알려진 환율만 쓴다.
계획 D6: 현금 이자는 0%가 주 결과, 실제 예탁금 이용료는 민감도. 정책금리를 그대로 넣지 않는다.

모든 표는 같은 꼴을 따른다.
  <관측 열>          관측일(값이 가리키는 날짜 또는 월)
  value              값(환율은 원/달러, 금리는 연 %)
  available_at_utc   이 값을 처음 쓸 수 있는 UTC 시각(이 시각 이후에만 사용)
  rule               available_at_utc를 정한 규칙 이름(아래 RULES)

엔진은 `asof(table, at_utc)`로만 값을 꺼낸다. 관측일로 꺼내면 미래 정보가 섞인다.

자료 원천(키 없음):
- FRED DEXKOUS: 뉴욕 정오 매수율. 연준 H.10 주간 공표(월요일 16:15, 월요일이 연방 공휴일이면 다음 영업일).
- 서울외국환중개(SMBS) 매매기준율(시장평균환율, MAR): 전 영업일 09:00~15:30 거래의 거래량 가중평균.
  그래프용 XML 주소(`StdExRate_xml.jsp`)로 받는다. 비공식 주소라 지속성은 보장되지 않는다.
- FRED DTB3(미국 3개월 국채, 일별), EFFR(연방기금 실효금리, 일별).
- FRED/OECD 한국 월별 금리: IR3TCD01KRM156N(CD 3개월, 2023-12에서 끊김),
  IR3TIB01KRM156N(3개월 은행간, OECD 정의), IRSTCI01KRM156N(콜금리).
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
SEOUL = ZoneInfo("Asia/Seoul")

RULES = {
    "h10_weekly": (
        "FRED DEXKOUS: 관측일이 속한 주(월~일)의 다음 월요일 16:15 ET에 공표. 월요일이 연방 공휴일이면 "
        "다음 영업일. 2009-01-01 이전에는 연준이 매일 공표했지만 정확한 시각 자료가 없어 같은 주간 규칙을 "
        "보수적으로(실제보다 늦게) 적용한다. 2015년 이후는 ALFRED 과거 시점 자료로 검증."
    ),
    "smbs_mar_prev_close": (
        "SMBS 매매기준율: 적용일 d의 값은 전 영업일(표의 바로 앞 행) 09:00~15:30 KST 거래로 정해진다. "
        "정확한 고시 시각 문서를 찾지 못해 전 영업일 18:00 KST부터 쓸 수 있다고 보수적으로 둔다."
    ),
    "next_us_business_day_1615et": (
        "미국 일별 금리(DTB3·EFFR): 관측일 다음 미국 영업일 16:15 ET(연준 H.15 공표 시각)부터 사용."
    ),
    "oecd_monthly_plus_two": (
        "OECD 월별 평균 금리: 해당 월이 끝나야 정해지고 공표가 늦으므로, 관측월의 다음다음 달 1일 "
        "00:00 KST부터 사용."
    ),
}

H10_RELEASE_TIME = time(16, 15)
H10_WEEKLY_SINCE = date(2009, 1, 1)
SMBS_AVAILABLE_TIME_KST = time(18, 0)


# ---------------------------------------------------------------------------
# 미국 연방 공휴일과 공표일
# ---------------------------------------------------------------------------

def us_federal_holidays(start: str | date, end: str | date) -> set[date]:
    """연준 휴무일(연방 공휴일). 일요일 공휴일은 월요일로 옮긴다.

    pandas USFederalHolidayCalendar는 토요일 공휴일을 금요일로 옮기지만 연준은 그 금요일에
    영업한다. 공표일 계산은 월~금 중 '다음 영업일'만 쓰므로, 토요일→금요일 이동분은 뺀다.
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar

    cal = USFederalHolidayCalendar()
    raw = cal.holidays(start=pd.Timestamp(start) - pd.Timedelta(days=7),
                       end=pd.Timestamp(end) + pd.Timedelta(days=7), return_name=True)
    out = set(d for d in EXTRA_FED_CLOSURES)
    for ts in raw.index:
        d = ts.date()
        # 금요일로 옮겨진 토요일 고정일 공휴일(예: 2021-12-31 ← 2022-01-01)은 연준 영업일이다.
        sat = d + timedelta(days=1)
        if d.weekday() == 4 and (sat.month, sat.day) in _FIXED_DATE_HOLIDAYS:
            continue
        out.add(d)
    return out


# 날짜가 고정된 연방 공휴일(월, 일): 새해, 준틴스(2021~), 독립기념일, 재향군인의 날, 성탄절.
_FIXED_DATE_HOLIDAYS = {(1, 1), (6, 19), (7, 4), (11, 11), (12, 25)}

# 대통령령·국가 애도일로 연방기관이 쉰 날(공휴일 달력에 없음). 공표가 다음 영업일로 밀린다.
# 2018-12-24는 ALFRED에서 공표 지연을 확인했다(월요일 공표가 12-26로 밀림). 나머지는 같은 성격이라
# 보수적으로(공표를 늦게 보는 쪽) 넣는다. 주간 공표 계산에는 월요일인 날만 영향을 준다.
EXTRA_FED_CLOSURES = (
    date(2004, 6, 11),   # 레이건 국가 애도일
    date(2007, 1, 2),    # 포드 국가 애도일
    date(2007, 12, 24),  # 성탄 전날 대통령령
    date(2012, 12, 24),  # 성탄 전날 대통령령(월요일)
    date(2014, 12, 26),  # 성탄 다음날 대통령령
    date(2018, 12, 5),   # 부시 국가 애도일
    date(2018, 12, 24),  # 성탄 전날 대통령령(월요일, ALFRED 확인)
    date(2019, 12, 24),  # 성탄 전날 대통령령
    date(2020, 12, 24),  # 성탄 전날 대통령령
    date(2024, 12, 24),  # 성탄 전날 대통령령
    date(2025, 1, 9),    # 카터 국가 애도일
    date(2025, 12, 24),  # 성탄 전날 대통령령
    date(2025, 12, 26),  # 성탄 다음날 대통령령
)


def next_business_day(d: date, holidays: set[date]) -> date:
    """d 다음 미국 영업일(토·일·연방 공휴일 제외)."""
    x = d + timedelta(days=1)
    while x.weekday() >= 5 or x in holidays:
        x += timedelta(days=1)
    return x


def h10_release_date(obs: date, holidays: set[date]) -> date:
    """DEXKOUS 관측일 obs가 실린 H.10 주간 공표일."""
    monday = obs + timedelta(days=7 - obs.weekday())  # 관측 주 다음 월요일
    release = monday
    while release.weekday() >= 5 or release in holidays:
        release += timedelta(days=1)
    return release


def _at(d: date, t: time, tz: ZoneInfo) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, t, tzinfo=tz)).tz_convert("UTC")


# ---------------------------------------------------------------------------
# 원천 파일 읽기
# ---------------------------------------------------------------------------

def read_fred_csv(path: Path) -> pd.DataFrame:
    """FRED fredgraph.csv → [date, value]. 빈 값·'.'(휴일)은 버린다."""
    df = pd.read_csv(path, dtype=str)
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    value_col = [c for c in df.columns if c != date_col][0]
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "value": pd.to_numeric(df[value_col].replace({".": None, "": None}), errors="coerce"),
    })
    return out.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)


_SMBS_SET = re.compile(r"label='(\d{2})\.(\d{2})\.(\d{2})'\s+value='([0-9.]+)'")


def parse_smbs_xml(raw: bytes | str) -> pd.DataFrame:
    """SMBS StdExRate_xml.jsp 응답 → [date, value]. 두 자리 연도: 90~99는 19xx, 그 외 20xx."""
    text = raw.decode("euc-kr", errors="replace") if isinstance(raw, bytes) else raw
    rows = []
    for yy, mm, dd, val in _SMBS_SET.findall(text):
        y = int(yy)
        year = 1900 + y if y >= 90 else 2000 + y
        rows.append((pd.Timestamp(year, int(mm), int(dd)), float(val)))
    df = pd.DataFrame(rows, columns=["date", "value"])
    if df["date"].duplicated().any():
        raise ValueError("SMBS 응답에 중복 날짜가 있습니다.")
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 이용 가능 시각 붙이기
# ---------------------------------------------------------------------------

def dexkous_table(df: pd.DataFrame) -> pd.DataFrame:
    """DEXKOUS 관측값에 관측 시각(뉴욕 정오)과 공표 시각을 붙인다."""
    if df.empty:
        raise ValueError("빈 DEXKOUS 자료")
    hol = us_federal_holidays(df["date"].min().date(), df["date"].max().date() + timedelta(days=30))
    obs = [d.date() for d in df["date"]]
    release = [h10_release_date(d, hol) for d in obs]
    out = pd.DataFrame({
        "observation_date": df["date"].dt.strftime("%Y-%m-%d"),
        "value": df["value"],
        "observed_at_utc": [_at(d, time(12, 0), NY) for d in obs],
        "release_date": [r.isoformat() for r in release],
        "available_at_utc": [_at(r, H10_RELEASE_TIME, NY) for r in release],
        "rule": "h10_weekly",
        "daily_release_era": [d < H10_WEEKLY_SINCE for d in obs],
    })
    out["lag_days"] = (pd.to_datetime(out["release_date"]) - pd.to_datetime(out["observation_date"])).dt.days
    return out


def smbs_table(df: pd.DataFrame) -> pd.DataFrame:
    """SMBS 매매기준율에 기준 거래일(전 영업일)과 이용 가능 시각을 붙인다. 첫 행은 기준일을 몰라 뺀다."""
    df = df.sort_values("date").reset_index(drop=True)
    prev = df["date"].shift(1)
    out = pd.DataFrame({
        "value_date": df["date"].dt.strftime("%Y-%m-%d"),
        "value": df["value"],
        "based_on_trading_date": prev.dt.strftime("%Y-%m-%d"),
        "available_at_utc": [None if pd.isna(p) else _at(p.date(), SMBS_AVAILABLE_TIME_KST, SEOUL) for p in prev],
        "rule": "smbs_mar_prev_close",
    })
    return out.iloc[1:].reset_index(drop=True)


def us_daily_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    """미국 일별 금리(연 %)에 다음 영업일 16:15 ET 공표 시각을 붙인다."""
    hol = us_federal_holidays(df["date"].min().date(), df["date"].max().date() + timedelta(days=30))
    obs = [d.date() for d in df["date"]]
    return pd.DataFrame({
        "observation_date": df["date"].dt.strftime("%Y-%m-%d"),
        "value": df["value"],
        "available_at_utc": [_at(next_business_day(d, hol), H10_RELEASE_TIME, NY) for d in obs],
        "rule": "next_us_business_day_1615et",
    })


def monthly_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    """OECD 월별 평균 금리(연 %)에 관측월+2개월 1일 00:00 KST를 붙인다."""
    months = df["date"].dt.to_period("M")
    avail = [(m + 2).to_timestamp().date() for m in months]
    return pd.DataFrame({
        "observation_month": months.astype(str),
        "value": df["value"],
        "available_at_utc": [_at(a, time(0, 0), SEOUL) for a in avail],
        "rule": "oecd_monthly_plus_two",
    })


# ---------------------------------------------------------------------------
# 사용
# ---------------------------------------------------------------------------

def load_table(path: Path) -> pd.DataFrame:
    """fx 폴더의 정리된 표를 읽는다. available_at_utc는 UTC tz-aware."""
    df = pd.read_csv(path)
    df["available_at_utc"] = pd.to_datetime(df["available_at_utc"], utc=True)
    if "observed_at_utc" in df.columns:
        df["observed_at_utc"] = pd.to_datetime(df["observed_at_utc"], utc=True)
    return df


def asof(table: pd.DataFrame, at_utc, value_col: str = "value"):
    """at_utc 시각까지 실제로 공표된 값 중 가장 최근 관측값 행(없으면 None).

    공표 시각이 같으면 관측일이 늦은 쪽(주간 공표에 함께 실린 마지막 날)을 쓴다.
    """
    at = pd.Timestamp(at_utc)
    if at.tzinfo is None:
        raise ValueError("at_utc는 시간대가 있어야 합니다(미래 정보 방지).")
    known = table[table["available_at_utc"] <= at]
    if known.empty:
        return None
    key = [c for c in ("observation_date", "value_date", "observation_month") if c in known.columns][0]
    return known.sort_values(["available_at_utc", key], kind="mergesort").iloc[-1]


def asof_series(table: pd.DataFrame, at_utc: pd.Series | pd.DatetimeIndex, value_col: str = "value") -> pd.Series:
    """여러 시각에 대해 asof를 한꺼번에(merge_asof). 결과 인덱스는 입력 순서."""
    key = [c for c in ("observation_date", "value_date", "observation_month") if c in table.columns][0]
    t = table.sort_values(["available_at_utc", key], kind="mergesort")
    t = t.drop_duplicates("available_at_utc", keep="last")[["available_at_utc", value_col]]
    q = pd.DataFrame({"at": pd.DatetimeIndex(pd.to_datetime(at_utc, utc=True))})
    q["_i"] = range(len(q))
    merged = pd.merge_asof(q.sort_values("at"), t, left_on="at", right_on="available_at_utc",
                           direction="backward")
    return merged.sort_values("_i")[value_col].reset_index(drop=True)


def compare_fx(dex: pd.DataFrame, smbs: pd.DataFrame, start: str = "2004-01-01") -> dict:
    """DEXKOUS(뉴욕 정오 d)와 SMBS 매매기준율 비교.

    같은 날짜 비교와, 시장 정보 시점을 맞춘 비교(SMBS d+1 값 = 서울 d 거래 평균 ↔ 뉴욕 d 정오)를 함께 낸다.
    """
    d = dex.assign(date=pd.to_datetime(dex["observation_date"]))[["date", "value"]]
    s = smbs.assign(date=pd.to_datetime(smbs["value_date"]),
                    based=pd.to_datetime(smbs["based_on_trading_date"]))
    d = d[d["date"] >= start]
    res = {}
    for label, sk in (("same_label_date", "date"), ("seoul_trading_date", "based")):
        m = d.merge(s[[sk, "value"]].rename(columns={sk: "date", "value": "smbs"}), on="date")
        pct = (m["smbs"] / m["value"] - 1) * 100
        res[label] = {
            "n": int(len(m)),
            "mean_pct": round(float(pct.mean()), 4),
            "mean_abs_pct": round(float(pct.abs().mean()), 4),
            "p95_abs_pct": round(float(pct.abs().quantile(0.95)), 4),
            "max_abs_pct": round(float(pct.abs().max()), 4),
            "max_abs_date": m.loc[pct.abs().idxmax(), "date"].strftime("%Y-%m-%d"),
        }
    res["dexkous_dates_without_smbs"] = int((~d["date"].isin(s["date"])).sum())
    res["smbs_dates_without_dexkous"] = int((~s.loc[s["date"] >= start, "date"].isin(d["date"])).sum())
    return res


def alfred_vintage_check(dex: pd.DataFrame, vintage_dir: Path) -> dict:
    """이미 받아 둔 ALFRED 과거 시점 파일(v_YYYY-MM-DD.csv)로 공표 규칙을 검증한다(네트워크 없음).

    vintage V에 실린 마지막 관측일이, 규칙상 공표일 ≤ V인 마지막 관측일과 같아야 한다.
    """
    release = pd.to_datetime(dex["release_date"])
    rows, mismatches = [], []
    for f in sorted(Path(vintage_dir).glob("v_*.csv")):
        if f.stat().st_size == 0:
            continue
        v = pd.Timestamp(f.stem[2:])
        actual = read_fred_csv(f)["date"].max().strftime("%Y-%m-%d")
        predicted = dex.loc[release <= v, "observation_date"].max()
        rows.append(v.strftime("%Y-%m-%d"))
        if actual != predicted:
            mismatches.append({"vintage": v.strftime("%Y-%m-%d"), "actual_last": actual, "predicted_last": predicted})
    return {"vintages_checked": len(rows), "first": rows[0] if rows else None,
            "last": rows[-1] if rows else None, "mismatches": mismatches}


def compare_fred_copies(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """ChatGPT 묶음의 DEXKOUS 사본과 새로 받은 자료가 겹치는 기간에 같은지."""
    m = old.merge(new, on="date", how="outer", suffixes=("_old", "_new"), indicator=True)
    both = m[m["_merge"] == "both"]
    diff = both[(both["value_old"] - both["value_new"]).abs() > 1e-9]
    lo, hi = max(old["date"].min(), new["date"].min()), min(old["date"].max(), new["date"].max())
    inside = m[(m["date"] >= lo) & (m["date"] <= hi)]
    return {
        "overlap": [lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")],
        "both": int(len(both)),
        "value_differences": diff[["date", "value_old", "value_new"]].astype(str).values.tolist()[:20],
        "only_old_in_overlap": int((inside["_merge"] == "left_only").sum()),
        "only_new_in_overlap": int((inside["_merge"] == "right_only").sum()),
    }


# ---------------------------------------------------------------------------
# 생성
# ---------------------------------------------------------------------------

FRED_FILES = {
    # 출력 이름: (원천 파일, 표 함수)
    "fx_usdkrw_fred_dexkous.csv": ("fred_DEXKOUS.csv", dexkous_table),
    "rate_us_dtb3_daily.csv": ("fred_DTB3.csv", us_daily_rate_table),
    "rate_us_effr_daily.csv": ("fred_EFFR.csv", us_daily_rate_table),
    "rate_kr_cd3m_oecd_monthly.csv": ("fred_IR3TCD01KRM156N.csv", monthly_rate_table),
    "rate_kr_interbank3m_oecd_monthly.csv": ("fred_IR3TIB01KRM156N.csv", monthly_rate_table),
    "rate_kr_call_oecd_monthly.csv": ("fred_IRSTCI01KRM156N.csv", monthly_rate_table),
}
SMBS_RAW = "smbs_usd_1990_2026.xml"
SMBS_OUT = "fx_usdkrw_smbs_mar.csv"

# 원천 주소(키 없음, 2026-11-15 수신). 받기는 사람이 하고, 이 모듈은 받은 파일만 읽는다.
SOURCES = {
    "fred_*.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>",
    "alfred_DEXKOUS/v_<날짜>.csv": "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=DEXKOUS&vintage_date=<날짜>",
    SMBS_RAW: "http://www.smbs.biz/ExRate/StdExRate_xml.jsp?arr_value=USD_1990-01-01_2026-11-15",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_all(raw_dir: Path, out_dir: Path, chatgpt_dexkous: Path | None = None) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    manifest: dict = {"rules": RULES, "sources": SOURCES,
                      "extra_fed_closures": [d.isoformat() for d in EXTRA_FED_CLOSURES],
                      "raw": {}, "outputs": {}, "checks": {}}
    for f in sorted(raw_dir.glob("*")):
        if f.is_file():
            manifest["raw"][f.name] = {"bytes": f.stat().st_size, "sha256": _sha(f)}
    tables = {}
    for name, (src, fn) in FRED_FILES.items():
        tables[name] = fn(read_fred_csv(raw_dir / src))
    tables[SMBS_OUT] = smbs_table(parse_smbs_xml((raw_dir / SMBS_RAW).read_bytes()))
    for name, t in tables.items():
        path = out_dir / name
        t.to_csv(path, index=False, lineterminator="\n")
        key = t.columns[0]
        manifest["outputs"][name] = {"rows": len(t), "first": str(t[key].iloc[0]),
                                     "last": str(t[key].iloc[-1]), "rule": t["rule"].iloc[0],
                                     "sha256": _sha(path)}
    manifest["checks"]["dexkous_vs_smbs"] = compare_fx(tables["fx_usdkrw_fred_dexkous.csv"], tables[SMBS_OUT])
    dex = tables["fx_usdkrw_fred_dexkous.csv"]
    manifest["checks"]["dexkous_lag_days"] = {
        "since_2009": dex.loc[~dex["daily_release_era"], "lag_days"].value_counts().sort_index().to_dict(),
    }
    if (raw_dir / "alfred_DEXKOUS").is_dir():
        manifest["checks"]["dexkous_alfred_vintages"] = alfred_vintage_check(dex, raw_dir / "alfred_DEXKOUS")
    if chatgpt_dexkous is not None and Path(chatgpt_dexkous).exists():
        manifest["checks"]["dexkous_chatgpt_copy"] = compare_fred_copies(
            read_fred_csv(chatgpt_dexkous), read_fred_csv(raw_dir / "fred_DEXKOUS.csv"))
    (out_dir / "fx_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    """정리 실행: python -m src.research.etf.fx (원천은 <fx>/raw, 네트워크 없음)."""
    import argparse

    from . import paths

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chatgpt-dexkous", default=None,
                   help="비교용 DEXKOUS 사본(기본: RAW_ROOT/magic_split.../input/DEXKOUS_fred.csv)")
    args = p.parse_args(argv)
    out = paths.output_dir(paths.FX_DIR)
    old = args.chatgpt_dexkous
    if old is None:
        try:
            old = paths.raw_root() / "magic_split_reproducible_comparison_2026-11-15" / "input" / "DEXKOUS_fred.csv"
        except RuntimeError:
            old = None
    manifest = build_all(out / "raw", out, old)
    print(json.dumps({"outputs": manifest["outputs"], "checks": manifest["checks"]},
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
