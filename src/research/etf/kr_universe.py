"""국내 ETF 상장·폐지 이력(생존 편향 보정용 유니버스).

KRX 정보데이터시스템 [13101] ETF 전종목 시세(MDCSTAT04301)를 날짜별로 찍어
그날 거래소에 있던 ETF 전체를 모은다. pykrx `get_etf_ticker_list(date)`는
"지금 상장된 ETF 중 상장일 <= date"만 돌려주므로 폐지 ETF가 빠진다. 그래서
날짜별 전종목 시세를 쓴다.

KRX 로그인이 필요하다. pykrx>=1.2가 환경 변수 KRX_ID·KRX_PW로 로그인한다.
이 모듈은 계정 정보를 읽거나 기록하지 않고, 두 변수가 있는지만 확인한다.

실행(저장소 루트, pykrx>=1.2 환경):
    python -m src.research.etf.kr_universe collect --start 2002-10 --end 2026-11
    python -m src.research.etf.kr_universe build

출력: $MAGICSPLIT_ETF_DATA_ROOT/kr_universe/
    snapshots/YYYYMMDD.csv   날짜별 원자료(재실행 시 있는 날짜는 건너뜀)
    collect_manifest.json    수집 기록
    kr_etf_history.csv       종목별 처음·마지막 확인일, 이름, 폐지 여부
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from src.research.etf.paths import output_dir

KR_UNIVERSE_DIR = "kr_universe"
SNAPSHOT_SUBDIR = "snapshots"
HISTORY_FILE = "kr_etf_history.csv"
MANIFEST_FILE = "collect_manifest.json"

# 로그인 확인용: 2021-03-25 전종목 시세에 있어야 하는, 이후 상장폐지된 ETF(ARIRANG 200동일가중).
CANARY_DATE = "20210325"
CANARY_DELISTED_CODE = "295820"

NUMERIC_COLUMNS = ("TDD_CLSPRC", "ACC_TRDVOL", "ACC_TRDVAL", "MKTCAP", "LIST_SHRS", "NAV")
MAX_HOLIDAY_STEPBACK_DAYS = 10


def month_end_dates(start_month: str, end_month: str) -> list[dt.date]:
    """'YYYY-MM' 두 개 사이의 매월 말일(달력 기준)."""
    sy, sm = (int(x) for x in start_month.split("-"))
    ey, em = (int(x) for x in end_month.split("-"))
    if (sy, sm) > (ey, em):
        raise ValueError(f"시작 월이 끝 월보다 늦습니다: {start_month} > {end_month}")
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(dt.date(y, m, calendar.monthrange(y, m)[1]))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _require_login_env() -> None:
    missing = [k for k in ("KRX_ID", "KRX_PW") if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            "KRX 로그인 환경 변수가 없습니다: " + ", ".join(missing)
            + "\n이 터미널에서 직접 설정한 뒤 다시 실행하세요(값은 저장소·파일에 쓰지 마세요)."
        )


def fetch_all_etf_quotes(trade_date: str) -> pd.DataFrame:
    """그날 ETF 전종목 시세. 휴장일이면 빈 표."""
    from pykrx.website.krx.etx.core import 전종목시세_ETF

    df = 전종목시세_ETF().fetch(trade_date)
    if df is None:
        return pd.DataFrame()
    return df


def _check_canary(fetch=fetch_all_etf_quotes) -> None:
    try:
        df = fetch(CANARY_DATE)
    except Exception as exc:  # pykrx는 로그인 실패 시 KeyError 등을 낸다
        raise SystemExit(f"KRX 시세 조회 실패(로그인 실패 가능): {type(exc).__name__}: {exc}") from exc
    if df.empty or "ISU_SRT_CD" not in df.columns:
        raise SystemExit("KRX가 빈 결과를 돌려줬습니다. 로그인 상태와 pykrx>=1.2 설치를 확인하세요.")
    if CANARY_DELISTED_CODE not in set(df["ISU_SRT_CD"]):
        raise SystemExit(
            f"{CANARY_DATE} 전종목 시세에 폐지 ETF {CANARY_DELISTED_CODE}가 없습니다. "
            "이 조회가 과거 시점 목록을 주지 않는 것으로 보여 수집을 멈춥니다."
        )


def collect(start_month: str, end_month: str, out: Path, delay: float = 1.0,
            fetch=fetch_all_etf_quotes, check_canary: bool = True) -> dict:
    """월말마다(휴장이면 직전 거래일) 전종목 시세를 받아 snapshots/에 저장한다."""
    snap_dir = out / SNAPSHOT_SUBDIR
    snap_dir.mkdir(parents=True, exist_ok=True)
    if check_canary:
        _check_canary(fetch)

    records = []
    for target in month_end_dates(start_month, end_month):
        existing = sorted(snap_dir.glob(f"{target:%Y%m}*.csv"))
        if existing:
            records.append({"target": target.isoformat(), "file": existing[-1].name, "status": "cached"})
            continue
        found = None
        for back in range(MAX_HOLIDAY_STEPBACK_DAYS):
            day = target - dt.timedelta(days=back)
            if day.month != target.month:
                break
            df = fetch(day.strftime("%Y%m%d"))
            if delay:
                time.sleep(delay)
            if not df.empty:
                found = (day, df)
                break
        if found is None:
            records.append({"target": target.isoformat(), "status": "no_trading_day_found"})
            print(f"{target}: 거래일 자료 없음", flush=True)
            continue
        day, df = found
        df = df.copy()
        df.insert(0, "trade_date", day.isoformat())
        name = f"{day:%Y%m%d}.csv"
        df.to_csv(snap_dir / name, index=False, encoding="utf-8-sig")
        records.append({"target": target.isoformat(), "file": name, "rows": len(df), "status": "downloaded"})
        print(f"{day}: ETF {len(df)}개", flush=True)

    manifest = {
        "source": "KRX data.krx.co.kr MDCSTAT04301 ([13101] ETF 전종목 시세) via pykrx",
        "pykrx_version": _pykrx_version(),
        "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "start_month": start_month,
        "end_month": end_month,
        "rule": "매월 말일, 휴장이면 같은 달 안에서 직전 거래일",
        "snapshots": records,
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _pykrx_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("pykrx")
    except Exception:
        return None


def _to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).replace({"-": None, "": None}),
                         errors="coerce")


def load_snapshots(snap_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype=str, encoding="utf-8-sig") for p in sorted(snap_dir.glob("*.csv"))]
    if not frames:
        raise SystemExit(f"스냅샷이 없습니다: {snap_dir}")
    return pd.concat(frames, ignore_index=True)


def build_history(snapshots: pd.DataFrame) -> pd.DataFrame:
    """종목코드별 처음·마지막 확인일과 이름. 마지막 스냅샷에 없으면 폐지로 본다.

    폐지일 정밀도는 스냅샷 간격(한 달)이다. 정확한 마지막 거래일은 일봉으로 보정한다.
    """
    df = snapshots.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = _to_number(df[col])
    df = df.sort_values(["ISU_SRT_CD", "trade_date"])
    last_snapshot = df["trade_date"].max()
    all_dates = sorted(df["trade_date"].unique())

    rows = []
    for code, g in df.groupby("ISU_SRT_CD", sort=True):
        first, last = g.iloc[0], g.iloc[-1]
        seen = set(g["trade_date"])
        span = [d for d in all_dates if first["trade_date"] <= d <= last["trade_date"]]
        rows.append({
            "code": code,
            "first_seen": first["trade_date"].isoformat(),
            "last_seen": last["trade_date"].isoformat(),
            "n_snapshots": len(g),
            "gap_snapshots": len(span) - len(seen),
            "delisted": last["trade_date"] < last_snapshot,
            "first_name": first.get("ISU_ABBRV"),
            "last_name": last.get("ISU_ABBRV"),
            "name_changed": first.get("ISU_ABBRV") != last.get("ISU_ABBRV"),
            "last_index_name": last.get("IDX_IND_NM"),
            "last_close": last.get("TDD_CLSPRC"),
            "last_mktcap_krw": last.get("MKTCAP"),
            "last_trdval_krw": last.get("ACC_TRDVAL"),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="월말 전종목 시세 스냅샷 수집(KRX 로그인 필요)")
    c.add_argument("--start", default="2002-10", help="YYYY-MM (ETF 시장 개설 2002-10)")
    c.add_argument("--end", default=dt.date.today().strftime("%Y-%m"), help="YYYY-MM")
    c.add_argument("--delay", type=float, default=1.0, help="요청 사이 대기 초")
    sub.add_parser("build", help="스냅샷에서 상장·폐지 이력 생성(로그인 불필요)")
    args = parser.parse_args(argv)

    out = output_dir(KR_UNIVERSE_DIR)
    if args.cmd == "collect":
        _require_login_env()
        m = collect(args.start, args.end, out, delay=args.delay)
        counts = {}
        for r in m["snapshots"]:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        print("수집 결과:", counts)
        return 0
    hist = build_history(load_snapshots(out / SNAPSHOT_SUBDIR))
    hist.to_csv(out / HISTORY_FILE, index=False, encoding="utf-8-sig")
    print(f"종목 {len(hist)}개, 폐지 {int(hist['delisted'].sum())}개 → {out / HISTORY_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
