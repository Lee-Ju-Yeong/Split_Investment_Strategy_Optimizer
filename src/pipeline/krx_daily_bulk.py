"""
krx_daily_bulk.py

날짜별 전종목 조회로 일일 갱신을 하는 추가 경로(기존 종목별 backfill 경로는 그대로 둔다).

- 거래일 1일당 KRX 12회: 전종목시세 1(시세·시총·종목 명단), PER/PBR 1,
  투자자별 순매수 8(KOSPI·KOSDAQ × 기관합계·개인·외국인·기타외국인), 공매도 잔고 2.
- 과거 행 보호: 표별로 max(MAX(date), --write-floor - 1일) 이하는 쓰지 않는다. 새 기본키만 INSERT.
  원본(stocks)에 쓰는 실행은 --write-floor가 필수이고, 연결 DB 이름은 --expect-database와 같아야 한다.
- 공매도: 전체 주가 달력에서 이번 처리의 마지막 거래일 기준 3거래일 전까지만 원천일로 수집.
- 수정주가: 비율 1로 저장돼 온 종목이 사건 없이 이어질 때만 수정값=원시값으로 쓴다. 나머지는 격리.
- 지표: 새 시세가 있는 종목만 과거 전체로 계산하고 새 날짜만 저장.
- Tier(선택): 시작일 고정, 구간의 시장 전체 종목, 5년 위치용 가격 이력 확보, 같은 공매도 지연값.
- --dry-run: 외부 호출 없이 DB만 읽어 대상 날짜·표·예상 호출 수를 출력.
- --verify-date: 저장된 날(기본 2026-02-02)을 다시 받아 대조만(쓰기 0, KRX 정확히 15회 예산).

예) python -m src.pipeline.krx_daily_bulk --end-date 20261120 --expect-database stocks_stage --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .krx_daily_bulk_normalize import (
    INVESTOR_CODES,
    SHORT_MARKET_CODES,
    BulkResponseError,
    PriorPriceState,
    UniverseChangeLimit,
    apply_adjusted_carry_forward,
    check_universe_change,
    investor_response_series,
    next_prior_state,
    normalize_fundamental_day,
    normalize_investor_day,
    normalize_market_day,
    normalize_short_balance_day,
    short_response_frame,
)
from .new_key_guard import (
    NewKeyViolation,
    TABLE_SPECS,
    as_date,
    assert_target_database,
    fetch_table_watermarks,
    frame_to_rows,
    insert_new_keys,
    protected_until,
)


PRICE_CALL_TABLES = ("DailyStockPrice", "MarketCapDaily", "TickerUniverseSnapshot")
DAILY_FETCH_TABLES = PRICE_CALL_TABLES + ("FinancialData", "InvestorTradingTrend")
ALL_TABLES = DAILY_FETCH_TABLES + ("ShortSellingDaily", "CalculatedIndicators", "DailyStockTier")
INVESTOR_MARKET_IDS = ("STK", "KSQ")
CALLS_MARKET_PRICES = 1
CALLS_FUNDAMENTALS = 1
CALLS_INVESTOR = len(INVESTOR_MARKET_IDS) * len(INVESTOR_CODES)
CALLS_SHORT = len(SHORT_MARKET_CODES)
CALLS_LOGIN = 3
CALLS_PER_TRADING_DAY = CALLS_MARKET_PRICES + CALLS_FUNDAMENTALS + CALLS_INVESTOR + CALLS_SHORT
VERIFY_KRX_BUDGET = CALLS_LOGIN + CALLS_PER_TRADING_DAY
DEFAULT_VERIFY_DATE = "20260202"
PER_TICKER_TABLE_CALLS = 5  # 기존 경로: 시세·재무·투자자·시총·공매도 종목별 1회 이상
DEFAULT_SHORT_LAG_TRADING_DAYS = 3
ADJUST_EVENT_POLICIES = ("quarantine", "stop")
EARLIEST = date(1900, 1, 1)


class BulkStopError(RuntimeError):
    """갱신을 멈춰야 하는 조건(전제 미충족, 정책 정지 등)."""


class VerifyMismatch(RuntimeError):
    """대조 결과가 기존 저장값과 다르다."""


# ---- 외부 조회(pykrx 원 필드) ----------------------------------------------

class PykrxDailyBulkSource:
    """한 메서드 호출 = KRX 요청 1회. 세션·속도 제한은 KrxSession.installed()가 담당."""

    def market_prices(self, ymd: str) -> pd.DataFrame:
        from pykrx.website.krx.market.core import 전종목시세
        return 전종목시세().fetch(ymd, "ALL")

    def fundamentals(self, ymd: str) -> pd.DataFrame:
        from pykrx.website.krx.market.core import PER_PBR_배당수익률_전종목
        return PER_PBR_배당수익률_전종목().fetch(ymd, "ALL")

    def investor_net_values(self, ymd: str, market_id: str, investor_code: int) -> pd.DataFrame:
        from pykrx.website.krx.market.core import 투자자별_순매수상위종목
        return 투자자별_순매수상위종목().fetch(ymd, ymd, market_id, investor_code)

    def short_balances(self, ymd: str, market_code: int) -> pd.DataFrame:
        from pykrx.website.krx.market.core import 전종목_공매도_잔고
        return 전종목_공매도_잔고().fetch(ymd, market_code)


class NoExternalSource:
    """외부 호출이 계획에 없을 때 쓰는 원천. 호출되면 즉시 멈춘다."""

    def __getattr__(self, name):
        def _refuse(*_args, **_kwargs):
            raise BulkStopError("external_call_not_planned")
        return _refuse


# ---- 증거 기록 --------------------------------------------------------------

class RunRecorder:
    """실행별 새 폴더에 날짜별 완료 증거를 남긴다. 기존 파일은 절대 덮어쓰지 않는다."""

    def __init__(self, directory: str):
        self.path = Path(directory)
        if self.path.exists() and (not self.path.is_dir() or any(self.path.iterdir())):
            raise BulkStopError("report_dir_not_empty")
        self.path.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload) -> Path:
        target = self.path / name
        with target.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
        return target

    def write_csv(self, name: str, frame: pd.DataFrame) -> Path:
        target = self.path / name
        with target.open("x", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
        return target


def codes_digest(codes: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(codes)).encode("utf-8")).hexdigest()


# ---- 계획 -------------------------------------------------------------------

@dataclass(frozen=True)
class BulkOptions:
    end_date: date
    expected_database: str
    write_floor: Optional[date] = None
    max_dates: Optional[int] = None
    holidays: FrozenSet[date] = frozenset()
    short_lag_trading_days: int = DEFAULT_SHORT_LAG_TRADING_DAYS
    run_short: bool = True
    run_indicators: bool = True
    run_tier: bool = False
    adjust_event_policy: str = "quarantine"
    universe_limit: UniverseChangeLimit = UniverseChangeLimit()
    max_partial_investor: int = 0
    max_unobserved_investor: int = 0
    max_short_unpublished: int = 0
    max_short_missing: int = 0
    tier_allow_quarantined: bool = False
    tier_max_short_missing: int = 0

    def __post_init__(self):
        if self.max_dates is not None and self.max_dates < 1:
            raise ValueError("max_dates_must_be_positive")
        if self.short_lag_trading_days < 0:
            raise ValueError("short_lag_must_be_non_negative")
        if self.adjust_event_policy not in ADJUST_EVENT_POLICIES:
            raise ValueError("unknown_adjust_event_policy")

    def required_tables(self) -> Tuple[str, ...]:
        derived = [t for t, on in (("ShortSellingDaily", self.run_short),
                                   ("CalculatedIndicators", self.run_indicators),
                                   ("DailyStockTier", self.run_tier)) if on]
        return DAILY_FETCH_TABLES + tuple(derived)


@dataclass(frozen=True)
class DatePlan:
    day: date
    tables: Tuple[str, ...]
    trading_known: bool

    @property
    def needs_price_call(self) -> bool:
        return any(table in PRICE_CALL_TABLES for table in self.tables)

    @property
    def expected_calls(self) -> int:
        calls = CALLS_MARKET_PRICES if self.needs_price_call else 0
        calls += CALLS_FUNDAMENTALS if "FinancialData" in self.tables else 0
        calls += CALLS_INVESTOR if "InvestorTradingTrend" in self.tables else 0
        return calls


@dataclass
class BulkPlan:
    watermarks: Dict[str, Optional[date]]
    protected: Dict[str, Optional[date]]
    date_plans: List[DatePlan]
    run_end: date
    short_source_dates: List[date]
    indicator_dates: List[date]
    tier_dates: List[date]
    per_ticker_universe_size: int = 0

    @property
    def needs_external_calls(self) -> bool:
        return bool(self.date_plans or self.short_source_dates)

    @property
    def expected_krx_calls(self) -> int:
        if not self.needs_external_calls:
            return 0
        return CALLS_LOGIN + sum(p.expected_calls for p in self.date_plans) + CALLS_SHORT * len(self.short_source_dates)

    def summary(self) -> dict:
        fmt = lambda values: [d.isoformat() for d in values]
        iso = lambda mapping: {k: (v.isoformat() if v else None) for k, v in mapping.items()}
        size = self.per_ticker_universe_size
        return {
            "watermarks": iso(self.watermarks),
            "protected_through": iso(self.protected),
            "run_end": self.run_end.isoformat(),
            "target_dates": [{"date": p.day.isoformat(), "tables": list(p.tables), "calls": p.expected_calls} for p in self.date_plans],
            "short_source_dates": fmt(self.short_source_dates),
            "indicator_dates": fmt(self.indicator_dates),
            "tier_dates": fmt(self.tier_dates),
            "step_through": {
                "short": str(step_through(self.short_source_dates, self.run_end)),
                "indicators": str(step_through(self.indicator_dates, self.run_end)),
                "tier": str(step_through(self.tier_dates, self.run_end)),
            },
            "needs_login": self.needs_external_calls,
            "expected_krx_calls_estimate": self.expected_krx_calls,
            "per_ticker_path_estimate": {"universe_size": size, "krx_calls_per_update_run": size * PER_TICKER_TABLE_CALLS + 2, "naver_calls_per_update_run": size},
            "note": "estimate counts weekday candidates as trading days; a holiday costs 1 call and skips the rest; duplicate login adds 1; a cooldown retry adds login+1",
        }


def _price_dates_between(conn, after: Optional[date], through: date) -> List[date]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT date FROM DailyStockPrice WHERE date > %s AND date <= %s ORDER BY date",
            (after or EARLIEST, through),
        )
        return [as_date(row[0]) for row in cur.fetchall()]


def _latest_snapshot_size(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM TickerUniverseSnapshot WHERE snapshot_date = "
            "(SELECT MAX(snapshot_date) FROM TickerUniverseSnapshot)"
        )
        row = cur.fetchone()
    return int(row[0]) if row and row[0] else 0


def _weekdays(start: date, end: date, holidays: FrozenSet[date]) -> List[date]:
    days, cursor = [], start
    while cursor <= end:
        if cursor.weekday() < 5 and cursor not in holidays:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def short_lag_cap(trading_dates: Sequence[date], lag: int) -> Optional[date]:
    """기존 short_selling_collector._cap_end_date_by_short_selling_lag와 같은 규칙.
    trading_dates는 마지막 거래일까지의 **전체** 달력(최소 lag+1개)이어야 한다."""
    ordered = sorted(set(trading_dates), reverse=True)
    if not ordered:
        return None
    if lag <= 0:
        return ordered[0]
    return ordered[lag] if len(ordered) > lag else ordered[-1]


def _require_tables(watermarks, options: BulkOptions) -> None:
    empty = [t for t in options.required_tables() if watermarks.get(t) is None]
    if empty:
        raise BulkStopError(f"empty_tables_need_initial_backfill:{empty}")


def _require_no_floor_gap(watermarks, options: BulkOptions) -> None:
    """하한이 표 MAX보다 한참 뒤면 그 사이 거래일이 영구히 비게 된다."""
    if options.write_floor is None:
        return
    for table in DAILY_FETCH_TABLES:
        gap = _weekdays(watermarks[table] + timedelta(days=1), options.write_floor - timedelta(days=1), options.holidays)
        if gap and table != "TickerUniverseSnapshot":
            raise BulkStopError(f"write_floor_leaves_gap:{table}:{gap[0].isoformat()}")


def _date_plan(day: date, protected, price_max: date, db_dates: set) -> Optional[DatePlan]:
    tables = tuple(t for t in DAILY_FETCH_TABLES if day > protected[t])
    if not tables:
        return None
    known = day <= price_max
    if known and day not in db_dates:
        return None
    return DatePlan(day, tables, known)


def build_plan(conn, options: BulkOptions) -> BulkPlan:
    watermarks = fetch_table_watermarks(conn, ALL_TABLES)
    _require_tables(watermarks, options)
    _require_no_floor_gap(watermarks, options)
    protected = {t: protected_until(watermarks[t], options.write_floor) for t in ALL_TABLES}
    start = min(protected[t] for t in DAILY_FETCH_TABLES) + timedelta(days=1)
    db_dates = set(_price_dates_between(conn, start - timedelta(days=1), options.end_date))
    plans = [p for d in _weekdays(start, options.end_date, options.holidays)
             if (p := _date_plan(d, protected, watermarks["DailyStockPrice"], db_dates))]
    truncated = options.max_dates is not None and len(plans) > options.max_dates
    plans = plans[: options.max_dates] if truncated else plans
    run_end = plans[-1].day if truncated else options.end_date
    floor = min(v for v in protected.values() if v is not None) - timedelta(days=40)
    calendar = sorted(set(_price_dates_between(conn, floor, run_end)) | {p.day for p in plans if p.day <= run_end})
    cut = _step_limiter(options.max_dates)
    return BulkPlan(
        watermarks=watermarks, protected=protected, date_plans=plans, run_end=run_end,
        short_source_dates=cut(_short_dates(calendar, protected["ShortSellingDaily"], options)),
        indicator_dates=cut(_after(calendar, protected["CalculatedIndicators"])) if options.run_indicators else [],
        tier_dates=cut(_after(calendar, protected["DailyStockTier"])) if options.run_tier else [],
        per_ticker_universe_size=_latest_snapshot_size(conn),
    )


def _step_limiter(max_dates: Optional[int]):
    """--max-dates는 단계마다(일반 수집·공매도·지표·Tier) 실제 대상 날짜 수에 각각 적용한다."""
    return (lambda dates: list(dates)[:max_dates]) if max_dates else list


def step_through(planned: Sequence[date], run_end: date) -> Optional[date]:
    """실행 단계의 상한: 계획한 마지막 날짜와 run_end 중 이른 날. 계획이 비면 그 단계는 하지 않는다."""
    return min(planned[-1], run_end) if planned else None


def _after(dates: Sequence[date], bound: Optional[date]) -> List[date]:
    return [d for d in dates if bound is None or d > bound]


def _short_dates(calendar: Sequence[date], bound: Optional[date], options: BulkOptions) -> List[date]:
    if not options.run_short:
        return []
    cap = short_lag_cap(calendar, options.short_lag_trading_days)
    return [d for d in _after(calendar, bound) if cap and d <= cap]


# ---- 실행 -------------------------------------------------------------------

@dataclass
class RunContext:
    conn: object
    source: object
    options: BulkOptions
    plan: BulkPlan
    recorder: Optional[RunRecorder] = None
    protected: Dict[str, Optional[date]] = field(default_factory=dict)
    prior: Optional[Dict[str, PriorPriceState]] = None
    previous_price_day: Optional[date] = None
    previous_universe: Optional[Set[str]] = None
    report: dict = field(default_factory=dict)

    def __post_init__(self):
        self.protected = self.protected or dict(self.plan.protected)
        for key in ("days", "holidays", "short", "quarantined"):
            self.report.setdefault(key, [] if key != "quarantined" else {})

    def evidence(self, name: str, payload) -> None:
        if self.recorder is not None:
            self.recorder.write_json(name, payload)


def load_prior_price_state(conn, before: date) -> Dict[str, PriorPriceState]:
    sql = (
        "SELECT p.stock_code, p.date, p.close_price, p.adj_close, p.adj_ratio FROM DailyStockPrice p "
        "JOIN (SELECT stock_code, MAX(date) AS d FROM DailyStockPrice WHERE date < %s "
        "GROUP BY stock_code) m ON p.stock_code = m.stock_code AND p.date = m.d"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (before,))
        rows = cur.fetchall()
    to_float = lambda v: float(v) if v is not None else None
    return {code: PriorPriceState(as_date(d), float(close), to_float(adj), to_float(ratio)) for code, d, close, adj, ratio in rows}


def _last_price_date_before(conn, day: date) -> Optional[date]:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM DailyStockPrice WHERE date < %s", (day,))
        row = cur.fetchone()
    return as_date(row[0]) if row and row[0] is not None else None


def universe_on(conn, day: date) -> Set[str]:
    """그날 명단: 그날 스냅샷이 있으면 스냅샷, 없으면 그날 주가 종목."""
    with conn.cursor() as cur:
        cur.execute("SELECT stock_code FROM TickerUniverseSnapshot WHERE snapshot_date = %s", (day,))
        codes = {row[0] for row in cur.fetchall()}
        if not codes:
            cur.execute("SELECT stock_code FROM DailyStockPrice WHERE date = %s", (day,))
            codes = {row[0] for row in cur.fetchall()}
    return codes


def _previous_universe(conn, day: date) -> Set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT stock_code FROM TickerUniverseSnapshot WHERE snapshot_date = "
            "(SELECT MAX(snapshot_date) FROM TickerUniverseSnapshot WHERE snapshot_date < %s)",
            (day,),
        )
        return {row[0] for row in cur.fetchall()}


def _fetch_investor_series(source, ymd: str) -> Dict[str, List[pd.Series]]:
    """한 응답씩 받자마자 검사한다. 빈 응답이면 나머지 요청을 보내지 않고 멈춘다."""
    series: Dict[str, List[pd.Series]] = {key: [] for key in INVESTOR_CODES}
    for key, code in INVESTOR_CODES.items():
        for market_id in INVESTOR_MARKET_IDS:
            raw = source.investor_net_values(ymd, market_id, code)
            series[key].append(investor_response_series(raw, f"investor:{key}:{market_id}"))
    return series


def _fetch_short_frames(source, ymd: str) -> List[pd.DataFrame]:
    frames = []
    for market, code in SHORT_MARKET_CODES.items():
        frames.append(short_response_frame(source.short_balances(ymd, code), f"short_balance:{market}"))
    return frames


def _adjusted_prices(ctx: RunContext, snapshot, day: date):
    if ctx.prior is None:
        ctx.prior = load_prior_price_state(ctx.conn, day)
        ctx.previous_price_day = _last_price_date_before(ctx.conn, day)
    result = apply_adjusted_carry_forward(snapshot.prices, ctx.prior, ctx.previous_price_day)
    if result.quarantined and ctx.options.adjust_event_policy == "stop":
        record_quarantine(ctx.recorder, day, result.quarantined, snapshot.prices)
        raise BulkStopError(f"adjusted_unit_unverified:{day.isoformat()}:{len(result.quarantined)}")
    return result


def _price_frames(ctx: RunContext, plan: DatePlan, snapshot, info: dict) -> Dict[str, pd.DataFrame]:
    frames = {
        "MarketCapDaily": snapshot.market_caps,
        "TickerUniverseSnapshot": snapshot.universe.assign(snapshot_date=plan.day.isoformat(), company_name=None, source="pykrx"),
    }
    if "DailyStockPrice" in plan.tables:
        adjusted = _adjusted_prices(ctx, snapshot, plan.day)
        frames["DailyStockPrice"] = adjusted.rows
        info["quarantined"] = adjusted.quarantined
        info["raw_prices"] = snapshot.prices
    return frames


def _collect_day_frames(ctx: RunContext, plan: DatePlan, info: dict) -> Optional[Dict[str, pd.DataFrame]]:
    ymd = plan.day.strftime("%Y%m%d")
    frames: Dict[str, pd.DataFrame] = {}
    universe = None
    if plan.needs_price_call:
        snapshot = normalize_market_day(ctx.source.market_prices(ymd), ymd)
        if snapshot.is_holiday:
            if plan.trading_known:
                raise BulkResponseError(f"holiday_response_for_stored_trading_day:{plan.day.isoformat()}")
            return None
        previous = ctx.previous_universe if ctx.previous_universe is not None else _previous_universe(ctx.conn, plan.day)
        info["universe_change"] = check_universe_change(snapshot.codes, previous, ctx.options.universe_limit)
        info["market_counts"] = snapshot.market_counts
        universe = snapshot.codes
        frames.update(_price_frames(ctx, plan, snapshot, info))
    if universe is None:
        universe = universe_on(ctx.conn, plan.day)
    info["universe_size"], info["universe_sha256"] = len(universe), codes_digest(universe)
    if "FinancialData" in plan.tables:
        frames["FinancialData"], info["fundamentals"] = normalize_fundamental_day(ctx.source.fundamentals(ymd), ymd, universe)
    if "InvestorTradingTrend" in plan.tables:
        frame, investor_info = normalize_investor_day(_fetch_investor_series(ctx.source, ymd), ymd, universe)
        if len(investor_info["partial"]) > ctx.options.max_partial_investor:
            raise BulkResponseError(f"investor_partial_rows:{plan.day.isoformat()}:{len(investor_info['partial'])}")
        unobserved = len(investor_info["unobserved"]) + len(investor_info["dash_only"])
        if unobserved > ctx.options.max_unobserved_investor:
            raise BulkResponseError(f"investor_unobserved_rows:{plan.day.isoformat()}:{unobserved}")
        frames["InvestorTradingTrend"], info["investor"] = frame, investor_info
    ctx.previous_universe = universe if plan.needs_price_call else ctx.previous_universe
    return {table: frame for table, frame in frames.items() if table in plan.tables}


def _write_frames(ctx: RunContext, frames: Dict[str, pd.DataFrame], before_commit=None) -> Dict[str, int]:
    """before_commit(counts): DB 커밋 직전에 증거(격리 원시값·예정 건수)를 먼저 파일로 남긴다.
    그 기록이 실패하면 커밋하지 않고 롤백한다."""
    counts = {}
    try:
        for table, frame in frames.items():
            counts[table] = insert_new_keys(ctx.conn, table, frame_to_rows(frame, table), ctx.protected[table])
        if before_commit is not None:
            before_commit(counts)
        ctx.conn.commit()
    except Exception:
        ctx.conn.rollback()
        raise
    return counts


def record_quarantine(recorder: Optional[RunRecorder], day: date, quarantined: Dict[str, str], prices: pd.DataFrame) -> None:
    """격리 종목의 원시 시세와 사유. DB 커밋(또는 stop 정책의 중단) 전에 쓴다."""
    if not quarantined or recorder is None:
        return
    rows = prices[prices["stock_code"].isin(quarantined)].assign(reason=lambda f: f["stock_code"].map(quarantined))
    recorder.write_csv(f"quarantine-{day.isoformat()}.csv", rows)


def _pending_writer(ctx: RunContext, name: str, payload: dict, extra=None):
    """커밋 직전: 예정 건수를 *.pending.json으로 남긴다. 커밋 후 같은 이름의 완료 파일을 쓴다.
    둘 사이에서 멈추면 다음 실행의 --previous-report-dir 점검이 DB 상태로 판정한다."""
    def _before_commit(counts):
        if extra is not None:
            extra()
        ctx.evidence(f"{name}.pending.json", {**payload, "rows": counts})
    return _before_commit


def _process_day(ctx: RunContext, plan: DatePlan) -> None:
    info: dict = {}
    frames = _collect_day_frames(ctx, plan, info)
    if frames is None:
        ctx.report["holidays"].append(plan.day.isoformat())
        ctx.evidence(f"holiday-{plan.day.isoformat()}.json", {"date": plan.day.isoformat()})
        return
    key = plan.day.isoformat()
    entry = {"date": key, "tables": list(frames), **{k: v for k, v in info.items() if k != "raw_prices"}}
    quarantine = lambda: record_quarantine(ctx.recorder, plan.day, info.get("quarantined") or {}, info["raw_prices"]) if "raw_prices" in info else None
    counts = _write_frames(ctx, frames, _pending_writer(ctx, f"day-{key}", entry, quarantine))
    for table in frames:
        ctx.protected[table] = plan.day
    if "DailyStockPrice" in frames:
        ctx.prior = next_prior_state(frames["DailyStockPrice"], plan.day, ctx.prior or {})
        ctx.previous_price_day = plan.day
    if info.get("quarantined"):
        ctx.report["quarantined"][key] = len(info["quarantined"])
    ctx.evidence(f"day-{key}.json", {**entry, "rows": counts})
    ctx.report["days"].append({"date": key, "rows": counts})


def db_short_cap(conn, through: date, lag: int) -> Optional[date]:
    """마지막 거래일까지의 전체 주가 달력에서 lag 거래일 전(기존 수집기와 같은 쿼리)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT date FROM DailyStockPrice WHERE date <= %s ORDER BY date DESC LIMIT %s",
            (through, int(lag) + 1),
        )
        dates = [as_date(row[0]) for row in cur.fetchall()]
    return short_lag_cap(dates, lag)


def run_short_step(ctx: RunContext) -> List[dict]:
    lag = ctx.options.short_lag_trading_days
    through = step_through(ctx.plan.short_source_dates, ctx.plan.run_end)
    cap = db_short_cap(ctx.conn, ctx.plan.run_end, lag)
    limit = min(cap, through) if cap and through else None
    dates = _price_dates_between(ctx.conn, ctx.protected["ShortSellingDaily"], limit) if limit else []
    done = []
    for day in dates:
        ymd = day.strftime("%Y%m%d")
        frame, info = normalize_short_balance_day(_fetch_short_frames(ctx.source, ymd), ymd, universe_on(ctx.conn, day))
        if len(info["unpublished"]) > ctx.options.max_short_unpublished:
            raise BulkResponseError(f"short_unpublished_rows:{day.isoformat()}:{len(info['unpublished'])}")
        if len(info["missing_from_response"]) > ctx.options.max_short_missing:
            raise BulkResponseError(f"short_missing_rows:{day.isoformat()}:{len(info['missing_from_response'])}")
        pending = {"date": day.isoformat(), "tables": ["ShortSellingDaily"], **info}
        counts = _write_frames(ctx, {"ShortSellingDaily": frame}, _pending_writer(ctx, f"short-{day.isoformat()}", pending))
        ctx.protected["ShortSellingDaily"] = day
        entry = {**pending, "rows": counts["ShortSellingDaily"]}
        ctx.evidence(f"short-{day.isoformat()}.json", entry)
        ctx.report["short"].append({"date": day.isoformat(), "rows": counts["ShortSellingDaily"]})
        done.append(entry)
    return done


def _indicator_rows_for(conn, ticker: str, after: date, through: date) -> pd.DataFrame:
    from ..indicator_calculator import calculate_indicators, get_ohlcv_from_db

    prices = get_ohlcv_from_db(conn, ticker)
    if prices.empty:
        raise BulkStopError(f"indicators_price_history_unavailable:{ticker}")
    indicators = calculate_indicators(prices)
    dates = indicators.index.date
    fresh = indicators.loc[(dates > after) & (dates <= through)].copy()
    fresh["stock_code"] = ticker
    fresh["date"] = fresh.index.strftime("%Y-%m-%d")
    return fresh


def run_indicator_step(ctx: RunContext) -> dict:
    """모든 대상 종목을 계산한 뒤 한 트랜잭션으로 넣는다(중간 실패 시 표 MAX가 어긋나지 않게)."""
    after = ctx.protected["CalculatedIndicators"]
    through = step_through(ctx.plan.indicator_dates, ctx.plan.run_end)
    if through is None:
        return {"tickers": 0, "rows": 0, "skipped": "no_planned_dates"}
    with ctx.conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT stock_code FROM DailyStockPrice WHERE date > %s AND date <= %s ORDER BY stock_code",
            (after, through),
        )
        tickers = [row[0] for row in cur.fetchall()]
    if not tickers:
        return {"tickers": 0, "rows": 0}
    frame = pd.concat([_indicator_rows_for(ctx.conn, t, after, through) for t in tickers], ignore_index=True)
    counts = _write_frames(ctx, {"CalculatedIndicators": frame})
    ctx.protected["CalculatedIndicators"] = as_date(frame["date"].max())
    result = {"tickers": len(tickers), "rows": counts["CalculatedIndicators"], "through": str(ctx.protected["CalculatedIndicators"])}
    ctx.evidence("indicators.json", result)
    return result


def _short_source_dates_for(conn, tier_dates: Sequence[date], lag: int) -> Dict[date, Optional[date]]:
    """각 Tier 날짜가 쓰는 공매도 원천일(lag 거래일 전)."""
    if not tier_dates:
        return {}
    calendar = _price_dates_between(conn, tier_dates[0] - timedelta(days=lag * 7 + 30), tier_dates[-1])
    index = {d: i for i, d in enumerate(calendar)}
    return {d: (calendar[index[d] - lag] if index[d] - lag >= 0 else None) for d in tier_dates}


def _require_short_inputs(ctx: RunContext, tier_dates: Sequence[date]) -> None:
    sources = _short_source_dates_for(ctx.conn, tier_dates, ctx.options.short_lag_trading_days)
    wanted = sorted({s for s in sources.values() if s is not None})
    counts = {}
    if wanted:
        placeholders = ", ".join(["%s"] * len(wanted))
        with ctx.conn.cursor() as cur:
            cur.execute(f"SELECT date, COUNT(*) FROM ShortSellingDaily WHERE date IN ({placeholders}) GROUP BY date", tuple(wanted))
            counts = {as_date(d): int(n) for d, n in cur.fetchall()}
    missing = sorted({s for s in sources.values() if s is None or counts.get(s, 0) == 0}, key=lambda d: d or EARLIEST)
    if missing:
        raise BulkStopError(f"tier_short_inputs_missing:{[str(d) for d in missing][:10]}")


def _codes_with_price(conn, day: date) -> Set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT stock_code FROM DailyStockPrice WHERE date = %s", (day,))
        return {row[0] for row in cur.fetchall()}


def _codes_with_short(conn, day: date) -> Set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT stock_code FROM ShortSellingDaily WHERE date = %s AND short_balance_value IS NOT NULL", (day,))
        return {row[0] for row in cur.fetchall()}


def audit_tier_date(conn, day: date, produced: Set[str], short_source: Optional[date], options: BulkOptions) -> dict:
    """
    저장 전 날짜별 대조: 명단(그날 스냅샷)·가격 키·공매도 원천일 유효 키·산출 키.
    설명되지 않는 차이(계산 누락, 가격 없는 산출, 명단에 있는데 가격 없는 종목=격리 등,
    공매도 원천일 유효 행 누락)가 있으면 ok=False.
    """
    roster, priced = universe_on(conn, day), _codes_with_price(conn, day)
    source_priced = _codes_with_price(conn, short_source) if short_source else set()
    short_valid = _codes_with_short(conn, short_source) if short_source else set()
    audit = {
        "date": day.isoformat(), "roster": len(roster), "priced": len(priced), "produced": len(produced),
        "missing_from_output": sorted(priced - produced), "output_without_price": sorted(produced - priced),
        "roster_without_price": sorted(roster - priced),
        "short_source": str(short_source), "short_missing": sorted(source_priced - short_valid),
    }
    problems = [k for k in ("missing_from_output", "output_without_price") if audit[k]]
    if audit["roster_without_price"] and not options.tier_allow_quarantined:
        problems.append("roster_without_price")
    if short_source is None or len(audit["short_missing"]) > options.tier_max_short_missing:
        problems.append("short_missing")
    audit["problems"], audit["ok"] = problems, not problems
    for key in ("missing_from_output", "output_without_price", "roster_without_price", "short_missing"):
        audit[f"{key}_count"] = len(audit[key])
        audit[key] = audit[key][:20]
    return audit


def _guarded_tier_writer(ctx: RunContext, audits: List[dict], expected_days: Sequence[date]):
    """날짜 순서로 대조해 통과한 앞부분만 쓴다. 첫 실패 날짜부터는 저장하지 않고 멈춘다."""
    sources_cache: Dict[date, Optional[date]] = {}

    def _write(_conn, rows_df):
        frame = rows_df.assign(_day=rows_df["date"].map(as_date))
        days = sorted(set(frame["_day"]) | set(expected_days))  # 산출이 통째로 빠진 날짜도 대조
        sources_cache.update(_short_source_dates_for(ctx.conn, days, ctx.options.short_lag_trading_days))
        passed = []
        for day in days:
            produced = set(frame.loc[frame["_day"] == day, "stock_code"])
            audit = audit_tier_date(ctx.conn, day, produced, sources_cache.get(day), ctx.options)
            audits.append(audit)
            if not audit["ok"]:
                break
            passed.append(day)
        good = frame[frame["_day"].isin(passed)].drop(columns="_day")
        count = _write_frames(ctx, {"DailyStockTier": good})["DailyStockTier"] if len(good) else 0
        if passed:
            ctx.protected["DailyStockTier"] = passed[-1]
        if len(passed) < len(days):
            bad = audits[-1]
            raise BulkStopError(f"tier_audit_failed:{bad['date']}:{bad['problems']}")
        return count
    return _write


def run_tier_step(ctx: RunContext) -> dict:
    from .krx_daily_bulk_tier import run_tier_window

    through = step_through(ctx.plan.tier_dates, ctx.plan.run_end)
    dates = _price_dates_between(ctx.conn, ctx.protected["DailyStockTier"], through) if through else []
    if not dates:
        return {"rows_saved": 0}
    _require_short_inputs(ctx, dates)
    audits: List[dict] = []
    try:
        summary = run_tier_window(ctx.conn, dates[0], dates[-1], ctx.options.short_lag_trading_days, _guarded_tier_writer(ctx, audits, dates))
    finally:
        ctx.evidence("tier-audit.json", audits)
    result = {**summary, "audited_dates": len(audits), "through": str(ctx.protected["DailyStockTier"])}
    ctx.evidence("tier.json", result)
    return result


def run_daily_bulk(conn, source, options: BulkOptions, plan: BulkPlan,
                   recorder: Optional[RunRecorder] = None, report: Optional[dict] = None) -> dict:
    """report를 넘기면 중간에 멈춰도 호출자가 그때까지의 진행 기록을 볼 수 있다."""
    ctx = RunContext(conn, source, options, plan, recorder, report=report if report is not None else {})
    for date_plan in plan.date_plans:
        _process_day(ctx, date_plan)
    if options.run_short:
        run_short_step(ctx)
    if options.run_indicators:
        ctx.report["indicators"] = run_indicator_step(ctx)
    if options.run_tier:
        ctx.report["tier"] = run_tier_step(ctx)
    return ctx.report


# ---- 대조(쓰기 없음) --------------------------------------------------------

VERIFY_SPECS = {
    "DailyStockPrice": {"open_price": "dec5", "high_price": "dec5", "low_price": "dec5", "close_price": "dec5", "volume": "int"},
    "DailyStockPrice:adjusted": {"adj_close": "dec5", "adj_ratio": "dec10", "adj_open": "dec5", "adj_high": "dec5", "adj_low": "dec5"},
    "MarketCapDaily": {"market_cap": "int", "shares_outstanding": "int"},
    "TickerUniverseSnapshot": {"market_type": "text"},
    "FinancialData": {"per": "float32", "pbr": "float32", "eps": "float32", "bps": "float32", "dps": "float32", "div_yield": "float32", "roe": "float32"},
    "InvestorTradingTrend": {"individual_net_buy": "int", "foreigner_net_buy": "int", "institution_net_buy": "int", "total_net_buy": "int"},
    "ShortSellingDaily": {"short_balance_value": "int"},
}
DECIMAL_QUANTA = {"dec5": Decimal("0.00001"), "dec10": Decimal("0.0000000001")}


def _canonical(value, kind: str):
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if kind == "int":
        number = Decimal(str(value))
        if number != number.to_integral_value():
            return f"non-integer:{number}"
        return int(number)
    if kind in DECIMAL_QUANTA:
        return Decimal(str(value)).quantize(DECIMAL_QUANTA[kind])
    if kind == "float32":
        return float(np.float32(float(value)))
    return str(value)


def _as_object_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """비교 전에 object로 바꿔 정수(2**53 초과 포함)·NULL이 float로 뭉개지지 않게 한다."""
    keep = ["stock_code", *columns]
    out = frame[keep].astype(object) if len(frame) else pd.DataFrame(columns=keep, dtype=object)
    return out.where(out.notna(), None)


def _canonical_array(values: pd.Series, kind: str) -> np.ndarray:
    return np.array([_canonical(v, kind) for v in values], dtype=object)


def compare_with_db(fetched: pd.DataFrame, stored: pd.DataFrame, specs: Dict[str, str]) -> dict:
    """양쪽 NULL은 일치, 한쪽 NULL은 불일치. BIGINT는 정수로 정확히, DECIMAL은 저장 자릿수로,
    FLOAT는 DB에서 CAST AS DOUBLE로 읽은 저장값과 float32 비트 기준으로 비교한다."""
    fetched, stored = _as_object_frame(fetched, list(specs)), _as_object_frame(stored, list(specs))
    merged = fetched.merge(stored, on="stock_code", how="outer", suffixes=("_new", "_db"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    only_new = sorted(merged.loc[merged["_merge"] == "left_only", "stock_code"])
    only_db = sorted(merged.loc[merged["_merge"] == "right_only", "stock_code"])
    result = {"fetched": int(len(fetched)), "stored": int(len(stored)),
              "only_fetched_count": len(only_new), "only_fetched": only_new[:10],
              "only_stored_count": len(only_db), "only_stored": only_db[:10], "mismatch": {}}
    for column, kind in specs.items():
        new = _canonical_array(both[f"{column}_new"], kind)
        old = _canonical_array(both[f"{column}_db"], kind)
        differs = np.array([a != b for a, b in zip(new, old)], dtype=bool)
        result["mismatch"][column] = {"count": int(differs.sum()), "sample": sorted(both.loc[differs, "stock_code"])[:5]}
    result["identical"] = not only_new and not only_db and all(v["count"] == 0 for v in result["mismatch"].values())
    return result


def _select_expression(column: str, kind: Optional[str]) -> str:
    # FLOAT 일반 SELECT는 6자리로 반올림돼 보이므로 저장 비트를 복원할 수 있게 DOUBLE로 읽는다.
    return f"CAST(`{column}` AS DOUBLE) AS `{column}`" if kind == "float32" else f"`{column}`"


def _db_frame(conn, table: str, day: date, columns: Sequence[str], codes: Optional[Set[str]] = None,
              kinds: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    spec = TABLE_SPECS[table]
    kinds = kinds or {}
    column_sql = ", ".join(["`stock_code`", *(_select_expression(c, kinds.get(c)) for c in columns)])
    with conn.cursor() as cur:
        cur.execute(f"SELECT {column_sql} FROM `{table}` WHERE `{spec.date_column}` = %s", (day,))
        rows = cur.fetchall()
    frame = pd.DataFrame([list(r) for r in rows], columns=["stock_code", *columns], dtype=object)
    return frame[frame["stock_code"].isin(codes)] if codes is not None else frame


def _require_stored_day(conn, day: date) -> None:
    missing = [t for t in ("DailyStockPrice", "MarketCapDaily", "TickerUniverseSnapshot", "FinancialData",
                           "InvestorTradingTrend", "ShortSellingDaily") if _db_frame(conn, t, day, []).empty]
    if missing:
        raise BulkStopError(f"verify_day_without_stored_rows:{missing}")


def _verify_fetch(conn, source, day: date) -> Tuple[Dict[str, pd.DataFrame], dict]:
    ymd = day.strftime("%Y%m%d")
    snapshot = normalize_market_day(source.market_prices(ymd), ymd)
    if snapshot.is_holiday:
        raise BulkResponseError("verify_day_holiday_response")
    universe = snapshot.codes
    adjusted = apply_adjusted_carry_forward(snapshot.prices, load_prior_price_state(conn, day), _last_price_date_before(conn, day))
    fundamentals, _ = normalize_fundamental_day(source.fundamentals(ymd), ymd, universe)
    investor, investor_info = normalize_investor_day(_fetch_investor_series(source, ymd), ymd, universe)
    short, short_info = normalize_short_balance_day(_fetch_short_frames(source, ymd), ymd, universe_on(conn, day))
    fetched = {
        "DailyStockPrice": snapshot.prices, "DailyStockPrice:adjusted": adjusted.rows,
        "MarketCapDaily": snapshot.market_caps, "TickerUniverseSnapshot": snapshot.universe,
        "FinancialData": fundamentals, "InvestorTradingTrend": investor, "ShortSellingDaily": short,
    }
    reasons: Dict[str, int] = {}
    for reason in adjusted.quarantined.values():
        reasons[reason] = reasons.get(reason, 0) + 1
    extra = {"market_counts": snapshot.market_counts, "quarantined": adjusted.quarantined,
             "adjusted_check": {"total": int(len(snapshot.prices)), "checked": int(len(adjusted.rows)),
                                "quarantined": len(adjusted.quarantined), "reasons": reasons},
             "investor_partial": investor_info["partial"],
             "investor_unobserved": investor_info["unobserved"] + investor_info["dash_only"],
             "short_unpublished": short_info["unpublished"],
             "short_missing_from_response": short_info["missing_from_response"]}
    return fetched, extra


def verify_day(conn, source, day: date) -> dict:
    """이미 저장된 날짜를 날짜별 방식으로 다시 받아 기존 종목별 결과와 대조한다(쓰기 0)."""
    _require_stored_day(conn, day)
    fetched, extra = _verify_fetch(conn, source, day)
    tables = {}
    for name, frame in fetched.items():
        table = name.split(":")[0]
        codes = set(frame["stock_code"]) if name.endswith(":adjusted") else None
        specs = VERIFY_SPECS[name]
        tables[name] = compare_with_db(frame, _db_frame(conn, table, day, list(specs), codes, specs), specs)
    unverified = [k for k in ("investor_partial", "investor_unobserved", "short_unpublished", "short_missing_from_response") if extra[k]]
    if extra["adjusted_check"]["quarantined"]:
        unverified.append("adjusted_quarantined")
    identical = not unverified and all(t["identical"] for t in tables.values())
    return {"date": day.isoformat(), "identical": identical, "unverified": unverified, "tables": tables,
            **{k: (v if not isinstance(v, (list, dict)) or len(v) <= 50 else f"{len(v)} items") for k, v in extra.items()},
            "quarantined_count": len(extra["quarantined"])}


# ---- CLI --------------------------------------------------------------------

def _parse_ymd(text: str) -> date:
    return datetime.strptime(text, "%Y%m%d").date()


def _read_holidays(path: Optional[str]) -> FrozenSet[date]:
    if not path:
        return frozenset()
    tokens = Path(path).read_text(encoding="utf-8").split()
    return frozenset(_parse_ymd(x.replace("-", "")) for x in tokens if x.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    from ..common.krx_session import KrxThrottlePolicy

    defaults = KrxThrottlePolicy()
    p = argparse.ArgumentParser(description="KRX 날짜별 전종목 일일 갱신(새 키만 INSERT)")
    p.add_argument("--end-date", help="YYYYMMDD, 마지막 수집 거래일(대조 모드 외 필수)")
    p.add_argument("--expect-database", required=True, help="연결 DB 이름이 이것과 같아야 실행")
    p.add_argument("--write-floor", help="YYYYMMDD, 이 날짜 전은 쓰지 않음(원본 stocks 쓰기 시 필수)")
    p.add_argument("--dry-run", action="store_true", help="외부 호출 없이 계획만 출력")
    p.add_argument("--verify-date", nargs="?", const=DEFAULT_VERIFY_DATE,
                   help=f"저장된 날짜를 다시 받아 대조만(쓰기 0, KRX {VERIFY_KRX_BUDGET}회). 값 생략 시 {DEFAULT_VERIFY_DATE}")
    p.add_argument("--max-dates", type=int, help="앞에서부터 이 개수의 거래일 후보만 처리(smoke)")
    p.add_argument("--max-krx-calls", type=int, help="KRX 호출 예산. 넘으려 하면 보내기 전에 중단")
    p.add_argument("--credentials-file", help="KRX_LOGIN_ID/KRX_LOGIN_PW가 든 600 권한 파일")
    p.add_argument("--lock-file", help="실행 간 단일 잠금 파일(기본 KRX_SESSION_LOCK 또는 /tmp/split-krx-session.lock)")
    p.add_argument("--holidays-file", help="건너뛸 휴장일 목록(YYYYMMDD, 공백/줄 구분)")
    p.add_argument("--skip-short", action="store_true")
    p.add_argument("--skip-indicators", action="store_true")
    p.add_argument("--run-tier", action="store_true", help="새 거래일 Tier 계산(시작일 고정)")
    p.add_argument("--adjust-event-policy", choices=ADJUST_EVENT_POLICIES, default="quarantine")
    p.add_argument("--short-lag-trading-days", type=int, default=DEFAULT_SHORT_LAG_TRADING_DAYS)
    p.add_argument("--max-partial-investor", type=int, default=0)
    p.add_argument("--max-short-unpublished", type=int, default=0)
    p.add_argument("--max-short-missing", type=int, default=0)
    p.add_argument("--universe-change-min", type=int, default=UniverseChangeLimit().min_rows)
    p.add_argument("--universe-change-ratio", type=float, default=UniverseChangeLimit().max_ratio)
    p.add_argument("--delay", type=float, default=defaults.base_delay_seconds)
    p.add_argument("--jitter-max", type=float, default=defaults.jitter_max_seconds)
    p.add_argument("--macro-pause-every", type=int, default=defaults.macro_pause_every)
    p.add_argument("--block-cooldown", type=float, default=defaults.block_cooldown_seconds)
    p.add_argument("--max-block-events", type=int, default=defaults.max_block_events,
                   help="403·429·5xx·LOGOUT 몇 번째에 중단할지(기본 1=첫 응답에서 즉시 중단, 사용자 결정 11-22)")
    p.add_argument("--max-unobserved-investor", type=int, default=0)
    p.add_argument("--tier-allow-quarantined", action="store_true", help="명단에 있으나 주가가 격리된 종목이 있어도 Tier 저장(사본 진단용)")
    p.add_argument("--tier-max-short-missing", type=int, default=0)
    p.add_argument("--previous-report-dir", help="직전 실행 증거 폴더: 커밋했지만 완료 표시가 없는 날짜를 DB로 판정")
    p.add_argument("--report-dir", help="실행 증거를 남길 새(비어 있는) 폴더. dry-run 외 필수")
    return p


def validate_mode_args(args) -> None:
    """세션·DB를 열기 전에 사용 오류를 거른다(예: 모의 실행과 대조를 함께 주면 실제 조회로 바뀌지 않게)."""
    if args.dry_run and args.verify_date:
        raise BulkStopError("usage:dry_run_and_verify_are_exclusive")
    if not args.verify_date and not args.end_date:
        raise BulkStopError("usage:end_date_required")
    if args.verify_date and args.max_krx_calls not in (None, VERIFY_KRX_BUDGET):
        raise BulkStopError(f"verify_budget_is_fixed_at_{VERIFY_KRX_BUDGET}")
    if args.max_block_events < 1:
        raise BulkStopError("usage:max_block_events_must_be_positive")


def options_from_args(args) -> BulkOptions:
    end = _parse_ymd(args.end_date) if args.end_date else (_parse_ymd(args.verify_date) if args.verify_date else None)
    if end is None:
        raise BulkStopError("end_date_required")
    return BulkOptions(
        end_date=end,
        expected_database=args.expect_database,
        write_floor=_parse_ymd(args.write_floor) if args.write_floor else None,
        max_dates=args.max_dates,
        holidays=_read_holidays(args.holidays_file),
        short_lag_trading_days=args.short_lag_trading_days,
        run_short=not args.skip_short,
        run_indicators=not args.skip_indicators,
        run_tier=args.run_tier,
        adjust_event_policy=args.adjust_event_policy,
        universe_limit=UniverseChangeLimit(args.universe_change_min, args.universe_change_ratio),
        max_partial_investor=args.max_partial_investor,
        max_unobserved_investor=args.max_unobserved_investor,
        max_short_unpublished=args.max_short_unpublished,
        max_short_missing=args.max_short_missing,
        tier_allow_quarantined=args.tier_allow_quarantined,
        tier_max_short_missing=args.tier_max_short_missing,
    )


def policy_from_args(args, max_krx_calls: Optional[int] = None):
    from ..common.krx_session import KrxThrottlePolicy

    return replace(
        KrxThrottlePolicy(),
        base_delay_seconds=args.delay, jitter_max_seconds=args.jitter_max,
        macro_pause_every=args.macro_pause_every, block_cooldown_seconds=args.block_cooldown,
        max_block_events=args.max_block_events,
        max_krx_calls=max_krx_calls if max_krx_calls is not None else args.max_krx_calls,
    )


def _open_session(args, max_krx_calls: Optional[int] = None):
    from ..common.krx_session import KrxRunLock, KrxSession, load_krx_credentials

    return KrxSession(load_krx_credentials(args.credentials_file), policy_from_args(args, max_krx_calls),
                      run_lock=KrxRunLock(args.lock_file))


def _run_update(conn, args, options: BulkOptions, plan: BulkPlan, result: dict, recorder) -> None:
    from ..common.krx_session import default_collector_modules

    result["report"] = {}
    if not plan.needs_external_calls:
        run_daily_bulk(conn, NoExternalSource(), options, plan, recorder, result["report"])
        result["krx_session"] = {"krx_calls": 0, "logins": 0}
        return
    session = _open_session(args)
    try:
        session.login()
        with session.installed(default_collector_modules()):
            run_daily_bulk(conn, PykrxDailyBulkSource(), options, plan, recorder, result["report"])
    finally:
        result["krx_session"] = session.stats()
        session.close()


def _run_verify(conn, args, result: dict) -> int:
    from ..common.krx_session import default_collector_modules

    day = _parse_ymd(args.verify_date)
    _require_stored_day(conn, day)  # 기준일 자료가 없으면 로그인 전에 멈춘다
    session = _open_session(args, VERIFY_KRX_BUDGET)
    try:
        session.login()
        with session.installed(default_collector_modules()):
            result["verify"] = verify_day(conn, PykrxDailyBulkSource(), day)
    finally:
        result["krx_session"] = session.stats()
        session.close()
    return 0 if result["verify"]["identical"] else 4


def _execute(conn, args, options: BulkOptions, result: dict, recorder) -> int:
    writes = not (args.dry_run or args.verify_date)
    result["database"] = assert_target_database(conn, options.expected_database, options.write_floor, writes)
    if args.verify_date:
        return _run_verify(conn, args, result)
    if args.previous_report_dir:
        result["previous_run_check"] = reconcile_previous_report(conn, args.previous_report_dir)
    plan = build_plan(conn, options)
    result["plan"] = plan.summary()
    if recorder is not None:
        recorder.write_json("plan.json", result["plan"])
    if args.dry_run:
        result["krx_calls_made"] = 0
        return 0
    try:
        _run_update(conn, args, options, plan, result, recorder)
    finally:
        _record_watermarks_after(conn, result)
    return 0


def reconcile_previous_report(conn, directory: str) -> dict:
    """직전 실행에서 *.pending.json만 있고 완료 파일이 없는 날짜(커밋과 완료 표시 사이 중단)를 DB로 판정한다.
    보존된 격리 파일 목록도 함께 돌려준다. 외부 조회는 하지 않는다."""
    folder = Path(directory)
    if not folder.is_dir():
        raise BulkStopError("previous_report_dir_missing")
    findings = []
    for pending in sorted(folder.glob("*.pending.json")):
        done = pending.with_name(pending.name.replace(".pending.json", ".json"))
        if done.exists():
            continue
        payload = json.loads(pending.read_text(encoding="utf-8"))
        day = as_date(payload["date"])
        status = {t: _table_has_date(conn, t, day) for t in payload.get("tables", [])}
        state = "committed" if status and all(status.values()) else "not_committed" if not any(status.values()) else "partial"
        findings.append({"file": pending.name, "date": day.isoformat(), "tables": status, "state": state})
    if any(f["state"] == "partial" for f in findings):
        raise BulkStopError("previous_run_partial_commit")
    return {"pending_without_done": findings, "quarantine_files": sorted(p.name for p in folder.glob("quarantine-*.csv"))}


def _table_has_date(conn, table: str, day: date) -> bool:
    column = TABLE_SPECS[table].date_column
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM `{table}` WHERE `{column}` = %s", (day,))
        row = cur.fetchone()
    return bool(row and row[0])


def _record_watermarks_after(conn, result: dict) -> None:
    """보고서의 마지막 날짜는 메모리가 아니라 DB를 다시 읽은 값이다(중단 시에도)."""
    try:
        conn.rollback()
        marks = fetch_table_watermarks(conn, ALL_TABLES)
        result["watermarks_after_db"] = {k: (v.isoformat() if v else None) for k, v in marks.items()}
    except Exception as exc:
        result["watermarks_after_db"] = f"unavailable:{type(exc).__name__}"


def _classify(exc: BaseException) -> Tuple[int, str]:
    from ..common.krx_session import KrxSessionError

    if isinstance(exc, KrxSessionError):
        return 2, f"{type(exc).__name__}:{exc}"
    if isinstance(exc, (BulkStopError, BulkResponseError, NewKeyViolation)):
        return 3, f"{type(exc).__name__}:{exc}"
    return 1, f"unexpected:{type(exc).__name__}"


def _finish(result: dict, recorder) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(text[:20000], flush=True)
    if recorder is not None:
        recorder.write_json("report.json", result)
    if "stopped" in result:
        print(f"[krx_daily_bulk] STOP: {result['stopped']}", file=sys.stderr, flush=True)


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    result: dict = {"mode": "dry-run" if args.dry_run else "verify" if args.verify_date else "run"}
    recorder = None
    code = 1
    try:
        validate_mode_args(args)
        if not args.dry_run and not args.report_dir:
            raise BulkStopError("report_dir_required")
        recorder = RunRecorder(args.report_dir) if args.report_dir else None
        try:
            options = options_from_args(args)
        except ValueError as exc:
            raise BulkStopError(f"invalid_options:{exc}") from None
        from ..db_setup import get_db_connection

        conn = get_db_connection()
        try:
            code = _execute(conn, args, options, result, recorder)
        finally:
            conn.close()
    except Exception as exc:  # 원문 대신 고정 분류명만 남긴다
        code, result["stopped"] = _classify(exc)
    _finish(result, recorder)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
