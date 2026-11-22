"""
new_key_guard.py

과거 행 보호용 INSERT 경로. 표별 기존 MAX(date)보다 늦은 날짜의 새 기본키만 넣는다.
ON DUPLICATE KEY UPDATE / REPLACE / INSERT IGNORE를 쓰지 않으므로 기존 키와 겹치면
DB가 오류를 내고 호출자가 롤백한다(조용히 덮어쓰거나 건너뛰지 않는다).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class NewKeyViolation(RuntimeError):
    """새 경로가 보호 구간(MAX(date) 이하) 또는 중복 키를 쓰려고 한 경우."""


@dataclass(frozen=True)
class TableSpec:
    table: str
    date_column: str
    key_columns: Tuple[str, ...]
    columns: Tuple[str, ...]


TABLE_SPECS: Dict[str, TableSpec] = {
    spec.table: spec
    for spec in (
        TableSpec("DailyStockPrice", "date", ("stock_code", "date"), (
            "stock_code", "date", "open_price", "high_price", "low_price", "close_price",
            "adj_close", "adj_ratio", "adj_open", "adj_high", "adj_low", "volume")),
        TableSpec("MarketCapDaily", "date", ("stock_code", "date"), (
            "stock_code", "date", "market_cap", "shares_outstanding", "source")),
        TableSpec("TickerUniverseSnapshot", "snapshot_date", ("snapshot_date", "stock_code"), (
            "snapshot_date", "stock_code", "market_type", "company_name", "source")),
        TableSpec("FinancialData", "date", ("stock_code", "date"), (
            "stock_code", "date", "per", "pbr", "eps", "bps", "dps", "div_yield", "roe", "source")),
        TableSpec("InvestorTradingTrend", "date", ("stock_code", "date"), (
            "stock_code", "date", "individual_net_buy", "foreigner_net_buy",
            "institution_net_buy", "total_net_buy")),
        TableSpec("ShortSellingDaily", "date", ("stock_code", "date"), (
            "stock_code", "date", "short_volume", "short_value", "short_balance",
            "short_balance_value", "source")),
        TableSpec("CalculatedIndicators", "date", ("stock_code", "date"), (
            "stock_code", "date", "ma_5", "ma_20", "atr_14_ratio",
            "price_vs_5y_low_pct", "price_vs_10y_low_pct")),
        TableSpec("DailyStockTier", "date", ("date", "stock_code"), (
            "date", "stock_code", "tier", "reason", "liquidity_20d_avg_value", "sbv_ratio",
            "flow5_mcap", "pbr_discount", "per_discount", "div_premium", "cheap_score",
            "cheap_score_version", "cheap_score_confidence")),
    )
}


def as_date(value) -> Optional[date]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    text = str(value).strip()
    fmt = "%Y%m%d" if len(text) == 8 and text.isdigit() else "%Y-%m-%d"
    return datetime.strptime(text[:10], fmt).date()


ORIGIN_DATABASE = "stocks"
ORIGIN_MIN_WRITE_FLOOR = date(2026, 2, 9)  # 합의: 원본에는 2026-02-09 이후 새 키만(2/4~6 공매도는 사본 전용)


def protected_until(watermark: Optional[date], write_floor: Optional[date]) -> Optional[date]:
    """이 날짜 이하는 쓰지 않는다: 표 MAX와 (쓰기 하한 - 1일) 중 늦은 날."""
    floor_guard = write_floor - timedelta(days=1) if write_floor is not None else None
    candidates = [d for d in (watermark, floor_guard) if d is not None]
    return max(candidates) if candidates else None


def current_database(conn) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT DATABASE()")
        row = cur.fetchone()
    return str(row[0]) if row and row[0] is not None else ""


def assert_target_database(conn, expected: str, write_floor: Optional[date], writes: bool) -> str:
    """
    연결된 DB가 운영자가 적은 이름과 같아야 한다. 원본(stocks)에 쓰는 실행은 쓰기 하한이 필수다
    (예: 공매도 2026-02-04~06은 표 MAX 이후지만 사본 전용 경계 입력이라 원본에는 넣지 않는다).
    """
    actual = current_database(conn)
    if not expected or actual != expected:
        raise NewKeyViolation(f"target_database_mismatch(expected={expected!r}, actual={actual!r})")
    if writes and actual == ORIGIN_DATABASE and write_floor is None:
        raise NewKeyViolation("origin_write_requires_write_floor")
    if writes and actual == ORIGIN_DATABASE and write_floor < ORIGIN_MIN_WRITE_FLOOR:
        raise NewKeyViolation(f"origin_write_floor_before_{ORIGIN_MIN_WRITE_FLOOR.isoformat()}")
    return actual


def fetch_table_watermark(conn, table: str) -> Optional[date]:
    spec = TABLE_SPECS[table]
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(`{spec.date_column}`) FROM `{spec.table}`")
        row = cur.fetchone()
    return as_date(row[0]) if row and row[0] is not None else None


def fetch_table_watermarks(conn, tables: Iterable[str]) -> Dict[str, Optional[date]]:
    return {table: fetch_table_watermark(conn, table) for table in tables}


def _to_db_value(value):
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def frame_to_rows(frame: pd.DataFrame, table: str) -> list:
    spec = TABLE_SPECS[table]
    missing = [col for col in spec.columns if col not in frame.columns]
    if missing:
        raise NewKeyViolation(f"{table}: rows missing columns {missing}")
    return [tuple(_to_db_value(v) for v in row) for row in frame[list(spec.columns)].itertuples(index=False, name=None)]


def assert_rows_after_watermark(table: str, rows: Sequence[tuple], watermark: Optional[date]) -> None:
    """watermark: 보호 상한(protected_until 결과). 이 날짜 이하 행은 거부한다."""
    spec = TABLE_SPECS[table]
    date_pos = spec.columns.index(spec.date_column)
    key_pos = [spec.columns.index(col) for col in spec.key_columns]
    seen = set()
    for row in rows:
        row_date = as_date(row[date_pos])
        if watermark is not None and row_date <= watermark:
            raise NewKeyViolation(
                f"{table}: refusing to write {row_date} (protected through {watermark})"
            )
        key = tuple(row[pos] for pos in key_pos)
        if key in seen:
            raise NewKeyViolation(f"{table}: duplicated key in batch {key}")
        seen.add(key)


def insert_new_keys(conn, table: str, rows: Sequence[tuple], watermark: Optional[date], batch_size: int = 5000) -> int:
    """커밋하지 않는다. 날짜 단위 트랜잭션은 호출자가 commit/rollback 한다."""
    if not rows:
        return 0
    assert_rows_after_watermark(table, rows, watermark)
    spec = TABLE_SPECS[table]
    column_sql = ", ".join(f"`{col}`" for col in spec.columns)
    placeholders = ", ".join(["%s"] * len(spec.columns))
    sql = f"INSERT INTO `{spec.table}` ({column_sql}) VALUES ({placeholders})"
    with conn.cursor() as cur:
        for offset in range(0, len(rows), batch_size):
            cur.executemany(sql, list(rows[offset: offset + batch_size]))
    return len(rows)
