from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.db_setup import get_db_connection
from src.pipeline.daily_stock_tier_batch import (
    DEFAULT_CHEAP_SCORE_DIV_LOOKBACK_YEARS,
    DEFAULT_CHEAP_SCORE_MIN_OBS_DAYS,
    DEFAULT_CHEAP_SCORE_PBR_LOOKBACK_YEARS,
    DEFAULT_CHEAP_SCORE_PER_LOOKBACK_YEARS,
    DEFAULT_CHEAP_SCORE_WEIGHT_DIV,
    DEFAULT_CHEAP_SCORE_WEIGHT_PBR,
    DEFAULT_CHEAP_SCORE_WEIGHT_PER,
    DEFAULT_DANGER_LIQUIDITY,
    DEFAULT_ENABLE_INVESTOR_V1_WRITE,
    DEFAULT_ENABLE_SBV_TIER_OVERLAY,
    DEFAULT_FINANCIAL_LAG_DAYS,
    DEFAULT_INVESTOR_FLOW5_THRESHOLD,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_PRIME_LIQUIDITY,
    DEFAULT_SBV_TIER1_DEMOTE_THRESHOLD,
    DEFAULT_SBV_TIER3_CONSECUTIVE_DAYS,
    DEFAULT_SBV_TIER3_THRESHOLD,
    DEFAULT_SBV_VALID_COVERAGE_THRESHOLD,
    DEFAULT_SHORT_SELLING_PUBLICATION_LAG_TRADING_DAYS,
    DEFAULT_TIER1_GROWTH_BPS_MIN,
    DEFAULT_TIER1_GROWTH_ROE_MIN,
    DEFAULT_TIER1_POSITION_GATE_MAX_PCT,
    DEFAULT_TIER1_POSITION_GATE_START_DATE,
    DEFAULT_TIER1_POSITION_LOOKBACK_DAYS,
    DEFAULT_TIER1_POSITION_MIN_PERIODS_DAYS,
    DEFAULT_TIER1_QUALITY_LOOKBACK_DAYS,
    DEFAULT_TIER3_FLOW20_CONSECUTIVE_DAYS,
    DEFAULT_TIER3_FLOW20_QUANTILE,
    DEFAULT_TIER3_FLOW20_VALID_COVERAGE_THRESHOLD,
    _to_calendar_lookback_days,
    build_daily_stock_tier_frame,
    fetch_financial_history,
    fetch_investor_history,
    fetch_market_cap_history,
    fetch_price_history,
    fetch_short_balance_ratio_inputs,
    get_tier_ticker_universe,
)


def _parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only shadow diff for DailyStockTier SBV lag policy.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD.")
    parser.add_argument("--lag-days", type=int, default=DEFAULT_SHORT_SELLING_PUBLICATION_LAG_TRADING_DAYS)
    parser.add_argument("--base-lag-days", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        default="results/daily_stock_tier_shadow_diff",
        help="Directory where summary/report artifacts are saved.",
    )
    return parser


def _query_start(start_date: date) -> date:
    history_days = max(int(DEFAULT_TIER1_POSITION_LOOKBACK_DAYS), int(DEFAULT_TIER1_POSITION_MIN_PERIODS_DAYS))
    return start_date - timedelta(days=max(DEFAULT_LOOKBACK_DAYS + DEFAULT_FINANCIAL_LAG_DAYS + 5, history_days + 5))


def _financial_query_start(query_start: date) -> date:
    financial_lookback_days = _to_calendar_lookback_days(
        max(
            DEFAULT_CHEAP_SCORE_PBR_LOOKBACK_YEARS,
            DEFAULT_CHEAP_SCORE_PER_LOOKBACK_YEARS,
            DEFAULT_CHEAP_SCORE_DIV_LOOKBACK_YEARS,
        )
    )
    return query_start - timedelta(days=financial_lookback_days + 30)


def _load_inputs(conn, start_date: date, end_date: date) -> dict[str, Any]:
    ticker_codes = get_tier_ticker_universe(conn, end_date=end_date, mode="backfill")
    query_start = _query_start(start_date)
    financial_query_start = _financial_query_start(query_start)
    return {
        "ticker_codes": ticker_codes,
        "price_df": fetch_price_history(conn, query_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), ticker_codes=ticker_codes),
        "financial_df": fetch_financial_history(conn, end_date=end_date.strftime("%Y-%m-%d"), start_date=financial_query_start.strftime("%Y-%m-%d"), ticker_codes=ticker_codes),
        "investor_df": fetch_investor_history(conn, query_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), ticker_codes=ticker_codes),
        "market_cap_df": fetch_market_cap_history(conn, query_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), ticker_codes=ticker_codes),
        "short_balance_df": fetch_short_balance_ratio_inputs(conn, query_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), ticker_codes=ticker_codes),
    }


def _build_view(inputs: dict[str, Any], lag_days: int) -> pd.DataFrame:
    frame = build_daily_stock_tier_frame(
        price_df=inputs["price_df"],
        financial_df=inputs["financial_df"],
        investor_df=inputs["investor_df"],
        market_cap_df=inputs["market_cap_df"],
        short_balance_df=inputs["short_balance_df"],
        short_selling_publication_lag_trading_days=lag_days,
        lookback_days=DEFAULT_LOOKBACK_DAYS,
        financial_lag_days=DEFAULT_FINANCIAL_LAG_DAYS,
        danger_liquidity=DEFAULT_DANGER_LIQUIDITY,
        prime_liquidity=DEFAULT_PRIME_LIQUIDITY,
        enable_investor_v1_write=DEFAULT_ENABLE_INVESTOR_V1_WRITE,
        investor_flow5_threshold=DEFAULT_INVESTOR_FLOW5_THRESHOLD,
        cheap_score_pbr_lookback_years=DEFAULT_CHEAP_SCORE_PBR_LOOKBACK_YEARS,
        cheap_score_per_lookback_years=DEFAULT_CHEAP_SCORE_PER_LOOKBACK_YEARS,
        cheap_score_div_lookback_years=DEFAULT_CHEAP_SCORE_DIV_LOOKBACK_YEARS,
        cheap_score_weight_pbr=DEFAULT_CHEAP_SCORE_WEIGHT_PBR,
        cheap_score_weight_per=DEFAULT_CHEAP_SCORE_WEIGHT_PER,
        cheap_score_weight_div=DEFAULT_CHEAP_SCORE_WEIGHT_DIV,
        cheap_score_min_obs_days=DEFAULT_CHEAP_SCORE_MIN_OBS_DAYS,
        enable_sbv_tier_overlay=DEFAULT_ENABLE_SBV_TIER_OVERLAY,
        sbv_tier3_threshold=DEFAULT_SBV_TIER3_THRESHOLD,
        sbv_tier1_demote_threshold=DEFAULT_SBV_TIER1_DEMOTE_THRESHOLD,
        sbv_valid_coverage_threshold=DEFAULT_SBV_VALID_COVERAGE_THRESHOLD,
        sbv_tier3_consecutive_days=DEFAULT_SBV_TIER3_CONSECUTIVE_DAYS,
        tier1_growth_roe_min=DEFAULT_TIER1_GROWTH_ROE_MIN,
        tier1_growth_bps_min=DEFAULT_TIER1_GROWTH_BPS_MIN,
        tier1_quality_lookback_days=DEFAULT_TIER1_QUALITY_LOOKBACK_DAYS,
        tier1_position_gate_max_pct=DEFAULT_TIER1_POSITION_GATE_MAX_PCT,
        tier1_position_gate_start_date=DEFAULT_TIER1_POSITION_GATE_START_DATE,
        tier1_position_lookback_days=DEFAULT_TIER1_POSITION_LOOKBACK_DAYS,
        tier1_position_min_periods_days=DEFAULT_TIER1_POSITION_MIN_PERIODS_DAYS,
        tier3_flow20_quantile=DEFAULT_TIER3_FLOW20_QUANTILE,
        tier3_flow20_valid_coverage_threshold=DEFAULT_TIER3_FLOW20_VALID_COVERAGE_THRESHOLD,
        tier3_flow20_consecutive_days=DEFAULT_TIER3_FLOW20_CONSECUTIVE_DAYS,
    )
    return frame


def _filter_in_range(frame: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"]).dt.date
    mask = (dates >= start_date) & (dates <= end_date)
    return frame.loc[mask].copy()


def build_shadow_diff_frame(base_frame: pd.DataFrame, lagged_frame: pd.DataFrame) -> pd.DataFrame:
    merged = base_frame.merge(
        lagged_frame,
        on=["date", "stock_code"],
        how="outer",
        suffixes=("_base", "_lagged"),
        indicator=True,
    )
    base_ratio = pd.to_numeric(merged["sbv_ratio_base"], errors="coerce")
    lagged_ratio = pd.to_numeric(merged["sbv_ratio_lagged"], errors="coerce")
    base_reason = merged["reason_base"].fillna("")
    lagged_reason = merged["reason_lagged"].fillna("")

    merged["row_presence_changed"] = merged["_merge"] != "both"
    merged["sbv_appeared"] = base_ratio.isna() & lagged_ratio.notna()
    merged["sbv_disappeared"] = base_ratio.notna() & lagged_ratio.isna()
    merged["sbv_changed_value"] = (
        base_ratio.notna()
        & lagged_ratio.notna()
        & ~np.isclose(base_ratio, lagged_ratio, atol=1e-12, rtol=1e-9)
    )
    merged["tier_changed"] = merged["tier_base"] != merged["tier_lagged"]
    merged["reason_changed"] = base_reason != lagged_reason
    merged["affected"] = merged[
        [
            "row_presence_changed",
            "sbv_appeared",
            "sbv_disappeared",
            "sbv_changed_value",
            "tier_changed",
            "reason_changed",
        ]
    ].any(axis=1)
    return merged


def summarize_shadow_diff(diff_frame: pd.DataFrame, base_lag_days: int, lag_days: int) -> dict[str, Any]:
    affected = diff_frame.loc[diff_frame["affected"]].copy()
    affected_dates = sorted(pd.to_datetime(affected["date"]).dt.strftime("%Y-%m-%d").unique().tolist()) if not affected.empty else []
    summary = {
        "base_lag_days": int(base_lag_days),
        "lag_days": int(lag_days),
        "rows_compared": int(len(diff_frame)),
        "affected_rows": int(len(affected)),
        "affected_dates": len(affected_dates),
        "first_affected_date": affected_dates[0] if affected_dates else None,
        "last_affected_date": affected_dates[-1] if affected_dates else None,
        "sbv_non_null_base": int(pd.to_numeric(diff_frame["sbv_ratio_base"], errors="coerce").notna().sum()),
        "sbv_non_null_lagged": int(pd.to_numeric(diff_frame["sbv_ratio_lagged"], errors="coerce").notna().sum()),
        "sbv_appeared_rows": int(diff_frame["sbv_appeared"].sum()),
        "sbv_disappeared_rows": int(diff_frame["sbv_disappeared"].sum()),
        "sbv_changed_value_rows": int(diff_frame["sbv_changed_value"].sum()),
        "tier_changed_rows": int(diff_frame["tier_changed"].sum()),
        "reason_changed_rows": int(diff_frame["reason_changed"].sum()),
    }
    return summary


def _daily_impact_frame(diff_frame: pd.DataFrame) -> pd.DataFrame:
    affected = diff_frame.loc[diff_frame["affected"]].copy()
    if affected.empty:
        return pd.DataFrame(columns=["date", "affected_rows", "tier_changed_rows", "sbv_appeared_rows", "sbv_disappeared_rows", "sbv_changed_value_rows"])
    return (
        affected.assign(date=pd.to_datetime(affected["date"]).dt.strftime("%Y-%m-%d"))
        .groupby("date", as_index=False)
        .agg(
            affected_rows=("stock_code", "count"),
            tier_changed_rows=("tier_changed", "sum"),
            sbv_appeared_rows=("sbv_appeared", "sum"),
            sbv_disappeared_rows=("sbv_disappeared", "sum"),
            sbv_changed_value_rows=("sbv_changed_value", "sum"),
        )
        .sort_values("date")
    )


def _tier_transition_frame(diff_frame: pd.DataFrame) -> pd.DataFrame:
    changed = diff_frame.loc[diff_frame["tier_changed"]].copy()
    if changed.empty:
        return pd.DataFrame(columns=["tier_base", "tier_lagged", "rows"])
    return (
        changed.groupby(["tier_base", "tier_lagged"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", "tier_base", "tier_lagged"], ascending=[False, True, True])
    )


def write_shadow_reports(diff_frame: pd.DataFrame, summary: dict[str, Any], output_dir: Path, sample_limit: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    affected = diff_frame.loc[diff_frame["affected"]].copy()
    affected = affected.sort_values(["date", "stock_code"]).head(max(int(sample_limit), 1))
    daily_impact = _daily_impact_frame(diff_frame)
    tier_transition = _tier_transition_frame(diff_frame)

    summary_path = output_dir / "summary.json"
    sample_path = output_dir / "affected_rows_sample.csv"
    daily_path = output_dir / "daily_impact.csv"
    transition_path = output_dir / "tier_transition.csv"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    affected.to_csv(sample_path, index=False)
    daily_impact.to_csv(daily_path, index=False)
    tier_transition.to_csv(transition_path, index=False)
    return {
        "summary": str(summary_path),
        "affected_rows_sample": str(sample_path),
        "daily_impact": str(daily_path),
        "tier_transition": str(transition_path),
    }


def run_shadow_diff(start_date: date, end_date: date, base_lag_days: int, lag_days: int, output_dir: Path, sample_limit: int) -> dict[str, Any]:
    conn = get_db_connection()
    try:
        inputs = _load_inputs(conn, start_date=start_date, end_date=end_date)
    finally:
        conn.close()

    base_frame = _filter_in_range(_build_view(inputs, lag_days=base_lag_days), start_date, end_date)
    lagged_frame = _filter_in_range(_build_view(inputs, lag_days=lag_days), start_date, end_date)
    diff_frame = build_shadow_diff_frame(base_frame, lagged_frame)
    summary = summarize_shadow_diff(diff_frame, base_lag_days=base_lag_days, lag_days=lag_days)
    artifact_paths = write_shadow_reports(diff_frame, summary, output_dir=output_dir, sample_limit=sample_limit)
    return {"summary": summary, "artifacts": artifact_paths}


def main() -> None:
    args = _build_arg_parser().parse_args()
    start_date = _parse_yyyymmdd(args.start_date)
    end_date = _parse_yyyymmdd(args.end_date)
    if start_date > end_date:
        raise ValueError(f"Invalid range: start_date({start_date}) > end_date({end_date})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.output_dir) / f"sbv_lag_shadow_{args.start_date}_{args.end_date}_{ts}"
    result = run_shadow_diff(
        start_date=start_date,
        end_date=end_date,
        base_lag_days=int(args.base_lag_days),
        lag_days=int(args.lag_days),
        output_dir=outdir,
        sample_limit=int(args.sample_limit),
    )
    print("[daily_stock_tier_shadow] outdir=", outdir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("[daily_stock_tier_shadow] artifacts=", json.dumps(result["artifacts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
