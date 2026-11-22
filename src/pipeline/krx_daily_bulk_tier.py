"""
krx_daily_bulk_tier.py

새 경로의 Tier 계산 보조와 재계산 대조 도구(읽기 전용).

리뷰 합의 3:
- 여러 날을 한 번에 계산해도 각 날짜의 시장 전체를 쓴다: 종료일 명단 하나가 아니라
  계산 구간에 가격이 있는 모든 종목(기간 중 상장폐지 종목 포함)을 넘긴다.
- 5년 위치는 1,260개 가격 관측치가 필요하므로 가격 이력을 시작일 전 충분한 거래일부터 읽는다.
- 이미 저장된 구간(예: 2026-01-05~02-06)을 같은 방식으로 메모리에서 재계산해
  computed_at을 뺀 모든 열을 원본과 대조한다(쓰기 0, 외부 호출 0).

예) python -m src.pipeline.krx_daily_bulk_tier --start 20260105 --end 20260206 --expect-database stocks --report-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .new_key_guard import TABLE_SPECS, as_date, current_database


TIER_POSITION_OBSERVATIONS = 1260
HISTORY_MARGIN_TRADING_DAYS = 40
TIER_COMPARE_COLUMNS = TABLE_SPECS["DailyStockTier"].columns[2:]
TIER_FLOAT_COLUMNS = ("sbv_ratio", "flow5_mcap", "pbr_discount", "per_discount", "div_premium", "cheap_score", "cheap_score_confidence")


def tier_universe_codes(conn, start: date, end: date) -> List[str]:
    """구간 안에 가격 행이 있는 모든 종목(기간 중 상장폐지 종목 포함)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT stock_code FROM DailyStockPrice WHERE date BETWEEN %s AND %s ORDER BY stock_code",
            (start, end),
        )
        return [row[0] for row in cur.fetchall()]


def price_history_start(conn, start: date, observations: int = TIER_POSITION_OBSERVATIONS + HISTORY_MARGIN_TRADING_DAYS) -> Optional[date]:
    """start 이전 observations번째 거래일. 그만큼 이력이 없으면 가장 이른 거래일."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT date FROM DailyStockPrice WHERE date < %s ORDER BY date DESC LIMIT %s",
            (start, int(observations)),
        )
        rows = cur.fetchall()
    return as_date(rows[-1][0]) if rows else None


def run_tier_window(conn, start: date, end: date, lag: int, writer: Callable, legacy_inputs: bool = False) -> dict:
    """legacy_inputs=True: 기존 daily 방식(종료일 명단 하나, 기본 가격 조회 시작일) — 비교용."""
    from .daily_stock_tier_batch import run_daily_stock_tier_batch

    extra = {} if legacy_inputs else {
        "ticker_codes": tier_universe_codes(conn, start, end),
        "price_history_start_date": price_history_start(conn, start),
    }
    return run_daily_stock_tier_batch(
        conn,
        mode="daily",
        start_date_str=start.strftime("%Y%m%d"),
        end_date_str=end.strftime("%Y%m%d"),
        short_selling_publication_lag_trading_days=int(lag),
        row_writer=writer,
        **extra,
    )


def expected_tier_key_gaps(conn, rows_df: pd.DataFrame, start: date, end: date) -> dict:
    """날짜별: 가격 행은 있는데 Tier 행이 없는 종목 수(시장 전체 재구성 확인용)."""
    with conn.cursor() as cur:
        cur.execute("SELECT date, stock_code FROM DailyStockPrice WHERE date BETWEEN %s AND %s", (start, end))
        price_keys = {(as_date(d), c) for d, c in cur.fetchall()}
    tier_keys = {(as_date(d), c) for d, c in zip(rows_df["date"], rows_df["stock_code"])}
    missing: Dict[str, int] = {}
    for day, _ in price_keys - tier_keys:
        missing[day.isoformat()] = missing.get(day.isoformat(), 0) + 1
    return {"price_keys": len(price_keys), "tier_keys": len(tier_keys), "missing_by_date": dict(sorted(missing.items()))}


def _tier_select(column: str) -> str:
    # FLOAT 일반 SELECT는 6자리로 반올림된다. 저장 비트를 그대로 복원하도록 DOUBLE로 읽는다.
    if column in TIER_FLOAT_COLUMNS:
        return f"CAST(`{column}` AS DOUBLE) AS `{column}`"
    return f"`{column}`"


def _db_tier_frame(conn, start: date, end: date, with_computed_at: bool = False) -> pd.DataFrame:
    columns = ["date", "stock_code", *TIER_COMPARE_COLUMNS] + (["computed_at"] if with_computed_at else [])
    sql = f"SELECT {', '.join(_tier_select(c) for c in columns)} FROM DailyStockTier WHERE date BETWEEN %s AND %s"
    with conn.cursor() as cur:
        cur.execute(sql, (start, end))
        rows = cur.fetchall()
    return pd.DataFrame([list(r) for r in rows], columns=columns, dtype=object)


def _same_values(new: pd.Series, old: pd.Series, column: str) -> pd.Series:
    if column in TIER_FLOAT_COLUMNS:
        a = pd.to_numeric(new, errors="coerce").astype("float64").astype(np.float32)
        b = pd.to_numeric(old, errors="coerce").astype("float64").astype(np.float32)
        return pd.Series(((a.isna() & b.isna()) | (a == b)).to_numpy(), index=new.index)
    left = np.array([_canonical(v) for v in new], dtype=object)
    right = np.array([_canonical(v) for v in old], dtype=object)
    return pd.Series(left == right, index=new.index)


def _canonical(value):
    """정수·문자 열 비교용 표준형. NULL은 None, 정수값 실수는 정수 문자열."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _float_difference_stats(new: pd.Series, old: pd.Series) -> dict:
    """불일치 행의 크기: 한쪽만 NULL인 수, 절대·상대 차이 분포."""
    a = pd.to_numeric(new, errors="coerce").astype("float64")
    b = pd.to_numeric(old, errors="coerce").astype("float64")
    one_null = int((a.isna() ^ b.isna()).sum())
    both_values = a.notna() & b.notna()
    diff = (a[both_values] - b[both_values]).abs()
    rel = diff / b[both_values].abs().clip(lower=1e-12)
    if diff.empty:
        return {"one_side_null": one_null}
    return {
        "one_side_null": one_null,
        "abs_diff_median": float(diff.median()), "abs_diff_max": float(diff.max()),
        "rel_diff_median": float(rel.median()), "rel_diff_p99": float(rel.quantile(0.99)),
    }


def _tier_transitions(both: pd.DataFrame) -> dict:
    """저장 Tier → 재계산 Tier 건수(달라진 것만)."""
    changed = both[both["tier_new"].astype("float64") != both["tier_db"].astype("float64")]
    counts = changed.groupby([changed["tier_db"].astype(int), changed["tier_new"].astype(int)]).size()
    return {f"{old}->{new}": int(n) for (old, new), n in counts.items()}


def compare_tier_frames(new: pd.DataFrame, stored: pd.DataFrame) -> dict:
    """computed_at을 뺀 전 열 대조. FLOAT 열은 저장 정밀도(float32)로, 나머지는 정확히."""
    left = new.assign(date=new["date"].map(as_date))
    right = stored.assign(date=stored["date"].map(as_date))
    merged = left.merge(right, on=["date", "stock_code"], how="outer", suffixes=("_new", "_db"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    result = {
        "recomputed": int(len(new)), "stored": int(len(stored)),
        "only_recomputed": int((merged["_merge"] == "left_only").sum()),
        "only_stored": int((merged["_merge"] == "right_only").sum()),
        "only_stored_sample": [f"{d}:{c}" for d, c in merged.loc[merged["_merge"] == "right_only", ["date", "stock_code"]].head(10).itertuples(index=False)],
        "mismatch": {},
    }
    for column in TIER_COMPARE_COLUMNS:
        same = _same_values(both[f"{column}_new"], both[f"{column}_db"], column)
        sample = both.loc[~same, ["date", "stock_code"]].head(5)
        entry = {"count": int((~same).sum()), "sample": [f"{d}:{c}" for d, c in sample.itertuples(index=False)]}
        if column in TIER_FLOAT_COLUMNS:
            entry.update(_float_difference_stats(both.loc[~same, f"{column}_new"], both.loc[~same, f"{column}_db"]))
        result["mismatch"][column] = entry
    result["tier_transitions"] = _tier_transitions(both)
    result["identical"] = (
        result["only_recomputed"] == 0 and result["only_stored"] == 0
        and all(v["count"] == 0 for v in result["mismatch"].values())
    )
    return result


def recompute_check(conn, start: date, end: date, lag: int, legacy_inputs: bool = False) -> dict:
    captured: List[pd.DataFrame] = []

    def _capture(_conn, rows_df):
        captured.append(rows_df.copy())
        return len(rows_df)

    summary = run_tier_window(conn, start, end, lag, _capture, legacy_inputs)
    recomputed = pd.concat(captured, ignore_index=True) if captured else pd.DataFrame(columns=["date", "stock_code", *TIER_COMPARE_COLUMNS])
    return {
        "start": start.isoformat(), "end": end.isoformat(), "lag": lag, "legacy_inputs": legacy_inputs,
        "price_history_start": str(price_history_start(conn, start)),
        "tier_summary": {k: v for k, v in summary.items() if k != "rows_saved"},
        "comparison": compare_tier_frames(recomputed, _db_tier_frame(conn, start, end)),
        "key_gaps": expected_tier_key_gaps(conn, recomputed, start, end),
    }


# ---- 2026-03-14 당시 계산 방식 재현 ---------------------------------------------
# 근거: todos/2026_03_07-short-selling-publication-lag-pit.md §11. 2017-04-18부터 2026-02-06까지
# src.tier_backfill_window --chunk-days 365(mode=backfill, 기본 인자)로 다시 계산했다.
MARCH14_CHUNK_START = date(2017, 4, 18)
MARCH14_CHUNK_END = date(2026, 2, 6)
MARCH14_CHUNK_DAYS = 365


def march14_windows(eval_start: date, eval_end: date) -> List[tuple]:
    """평가 구간과 겹치는 당시 365일 묶음(tier_backfill_window._iter_windows와 같은 경계)."""
    windows, cursor = [], MARCH14_CHUNK_START
    while cursor <= MARCH14_CHUNK_END:
        chunk_end = min(cursor + timedelta(days=MARCH14_CHUNK_DAYS - 1), MARCH14_CHUNK_END)
        if chunk_end >= eval_start and cursor <= eval_end:
            windows.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return windows


def march14_query_starts(window_start: date) -> dict:
    """run_daily_stock_tier_batch 기본값으로 계산되는 실제 조회 시작일(가격·공매도·수급·시총 / 재무)."""
    from .daily_stock_tier_batch import (
        DEFAULT_CHEAP_SCORE_DIV_LOOKBACK_YEARS, DEFAULT_CHEAP_SCORE_PBR_LOOKBACK_YEARS,
        DEFAULT_CHEAP_SCORE_PER_LOOKBACK_YEARS, DEFAULT_FINANCIAL_LAG_DAYS, DEFAULT_LOOKBACK_DAYS,
        DEFAULT_TIER1_POSITION_LOOKBACK_DAYS, DEFAULT_TIER1_POSITION_MIN_PERIODS_DAYS,
    )
    history = max(DEFAULT_TIER1_POSITION_LOOKBACK_DAYS, DEFAULT_TIER1_POSITION_MIN_PERIODS_DAYS)
    query_start = window_start - timedelta(days=max(DEFAULT_LOOKBACK_DAYS + DEFAULT_FINANCIAL_LAG_DAYS + 5, history + 5))
    years = max(DEFAULT_CHEAP_SCORE_PBR_LOOKBACK_YEARS, DEFAULT_CHEAP_SCORE_PER_LOOKBACK_YEARS, DEFAULT_CHEAP_SCORE_DIV_LOOKBACK_YEARS)
    financial_start = query_start - timedelta(days=max(int(years) * 366, 1) + 30)
    return {"price_short_investor_mcap": query_start.isoformat(), "financial": financial_start.isoformat()}


def _mismatch_rows(new: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
    left = new.assign(date=new["date"].map(as_date))
    right = stored.assign(date=stored["date"].map(as_date))
    merged = left.merge(right, on=["date", "stock_code"], how="outer", suffixes=("_new", "_db"), indicator=True)
    differs = merged["_merge"] != "both"
    for column in TIER_COMPARE_COLUMNS:
        differs |= ~_same_values(merged[f"{column}_new"], merged[f"{column}_db"], column).fillna(False).astype(bool)
    return merged[differs]


def _computed_at_breakdown(mismatches: pd.DataFrame, stored: pd.DataFrame) -> dict:
    """불일치가 어느 계산 배치(저장 computed_at 날짜)에서 왔는지."""
    stamp = lambda s: s.map(lambda v: str(v)[:10] if v is not None else None)
    all_rows = stamp(stored["computed_at"]).value_counts().to_dict()
    bad = stamp(mismatches["computed_at"].where(mismatches["computed_at"].notna(), None)).value_counts().to_dict() if len(mismatches) else {}
    return {"stored_rows_by_computed_date": all_rows, "mismatch_rows_by_computed_date": bad}


def reproduce_march14(conn, eval_start: date, eval_end: date) -> Tuple[dict, pd.DataFrame]:
    from .daily_stock_tier_batch import run_daily_stock_tier_batch

    captured: List[pd.DataFrame] = []
    summaries = []
    for window_start, window_end in march14_windows(eval_start, eval_end):
        summary = run_daily_stock_tier_batch(
            conn, mode="backfill",
            start_date_str=window_start.strftime("%Y%m%d"), end_date_str=window_end.strftime("%Y%m%d"),
            row_writer=lambda _c, rows: captured.append(rows.copy()) or len(rows),
        )
        summaries.append({"window": [window_start.isoformat(), window_end.isoformat()],
                          "query_starts": march14_query_starts(window_start),
                          **{k: v for k, v in summary.items() if k != "rows_saved"}})
    recomputed = pd.concat(captured, ignore_index=True) if captured else pd.DataFrame(columns=["date", "stock_code", *TIER_COMPARE_COLUMNS])
    days = recomputed["date"].map(as_date)
    recomputed = recomputed[(days >= eval_start) & (days <= eval_end)]
    stored = _db_tier_frame(conn, eval_start, eval_end, with_computed_at=True)
    comparison = compare_tier_frames(recomputed, stored.drop(columns="computed_at"))
    mismatches = _mismatch_rows(recomputed, stored)
    comparison["computed_at"] = _computed_at_breakdown(mismatches, stored)
    return {"eval": [eval_start.isoformat(), eval_end.isoformat()], "windows": summaries, "comparison": comparison}, mismatches


def main(argv=None) -> int:
    from ..db_setup import get_db_connection
    from .krx_daily_bulk import RunRecorder

    p = argparse.ArgumentParser(description="Tier 재계산 대조(읽기 전용)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--lag", type=int, default=3)
    p.add_argument("--expect-database", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--legacy-inputs", action="store_true", help="비교용: 기존 daily 방식 입력으로 재계산")
    p.add_argument("--reproduce-march14", action="store_true",
                   help="2026-03-14 당시 365일 묶음 backfill(시작 2017-04-18) 방식으로 재현해 평가 구간만 대조")
    args = p.parse_args(argv)
    recorder = RunRecorder(args.report_dir)
    try:
        conn = get_db_connection()
    except Exception as exc:  # 연결 예외 원문(호스트·계정)은 남기지 않는다
        recorder.write_json("report.json", {"stopped": f"db_connect_failed:{type(exc).__name__}"})
        return 3
    try:
        database = current_database(conn)
        if database != args.expect_database:
            recorder.write_json("report.json", {"stopped": "target_database_mismatch"})
            return 3
        start = datetime.strptime(args.start, "%Y%m%d").date()
        end = datetime.strptime(args.end, "%Y%m%d").date()
        if args.reproduce_march14:
            result, mismatches = reproduce_march14(conn, start, end)
            recorder.write_csv("mismatch_rows.csv", mismatches)
        else:
            result = recompute_check(conn, start, end, args.lag, args.legacy_inputs)
        result["database"] = database
        recorder.write_json("report.json", result)
        print(json.dumps(result["comparison"], ensure_ascii=False, default=str)[:4000])
        return 0 if result["comparison"]["identical"] else 4
    except Exception as exc:
        recorder.write_json("report.json", {"stopped": f"unexpected:{type(exc).__name__}"})
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
