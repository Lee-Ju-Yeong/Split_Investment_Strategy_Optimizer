"""고정 M1 묶음에서 명시한 탐색용 상품만 M2 입력으로 읽는다. 원본에는 쓰지 않는다.

보류 상품 요청은 가격 파일을 읽기 전에 거부한다. 생성 당시 프로토콜과 현재 실행 프로토콜의
해시는 별도로 보존한다. 수정가격·지급일 미확인 배당 자료를 실제 세후 입력으로 승격하지 않는다.
"""
from __future__ import annotations

import csv
from bisect import bisect_left
import gzip
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from . import protocol
from .engine_types import Bar, FxRate, Instrument, aware, decimal
from .engine_lifecycle import DelistingEvent, validate_delistings
from .price_bundle import current_bundle


KST = ZoneInfo("Asia/Seoul")
ZERO = Decimal("0")


@dataclass
class M1Inputs:
    instruments: list[Instrument]
    bars: list[Bar]
    fx_rates: list[FxRate]
    evaluation_times: list[datetime]
    conversion_times: list[datetime]
    us_sessions: list[date]
    settlement_session_times: dict[date, datetime]
    provenance: dict
    warnings: list[str] = field(default_factory=list)
    kr_sessions: list[date] = field(default_factory=list)
    kr_settlement_session_times: dict[date, datetime] = field(default_factory=dict)
    domestic_settlement: str = "legacy_same_day"
    max_fx_age_days: int | None = None
    delistings: tuple[DelistingEvent, ...] = ()

    def engine_kwargs(self) -> dict:
        return {name: getattr(self, name) for name in (
            "fx_rates", "evaluation_times", "conversion_times", "us_sessions", "settlement_session_times",
            "kr_sessions", "kr_settlement_session_times", "domestic_settlement", "max_fx_age_days", "delistings")}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _stamp(value: str) -> datetime:
    return aware(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _check_hash(path: Path, expected: str) -> str:
    actual = _sha(path)
    if actual != expected:
        raise ValueError(f"M1 고정 입력 해시가 다릅니다: {path.name}")
    return actual


def _snapshot(root, split_csv, symbols):
    split_csv = Path(split_csv)
    record = json.loads(split_csv.with_suffix(".json").read_text(encoding="utf-8"))
    _check_hash(split_csv, record["split_sha256"])
    if record["split_csv"] != split_csv.name:
        raise ValueError("분할 기록과 CSV 이름이 다릅니다")
    selected = {}
    for row in _rows(split_csv):
        if row["symbol"] not in symbols:
            continue
        if row["symbol"] in selected:
            raise ValueError("여러 시장에 같은 상품 코드가 있습니다")
        if row["split"] != "explore" and not (row["market"] == "GOLD" and row["split"] == "both"):
            raise ValueError(f"M2에서는 보류 상품을 실행할 수 없습니다: {row['symbol']}")
        selected[row["symbol"]] = row
    if set(selected) != set(symbols):
        raise ValueError("요청 상품이 고정 탐색/보류 분할에 없습니다")
    metadata_path = root / "metadata" / "etf_metadata.csv"
    _check_hash(metadata_path, record["inputs"]["metadata_sha256"])
    metadata = {(r["market"], r["symbol"]): r for r in _rows(metadata_path)}
    bundle = current_bundle(root / "prices")
    if not bundle.resolve().is_relative_to((root / "prices" / "runs").resolve()):
        raise ValueError("가격 묶음이 지정 자료 루트 밖을 가리킵니다")
    _check_hash(bundle / "manifest.json", record["inputs"]["price_manifest_sha256"])
    return record, selected, metadata, bundle


def _calendar(path):
    result = {}
    previous = None
    for row in _rows(path):
        day = date.fromisoformat(row["date"])
        opened, closed = _stamp(row["open_utc"]), _stamp(row["close_utc"])
        if day in result or (previous is not None and day <= previous) or closed <= opened:
            raise ValueError("거래 달력의 중복·순서·시각이 잘못됐습니다")
        result[day] = (opened, closed)
        previous = day
    return result


def _prices(root, bundle, record, selected, end):
    output = {symbol: [] for symbol in selected}
    for market in ("KR", "US", "GOLD"):
        names = {symbol for symbol, row in selected.items() if row["market"] == market}
        if not names:
            continue
        path = (root / "krx_gold" / "gold_9999_1kg_daily.csv" if market == "GOLD"
                else bundle / f"{market.lower()}_daily.csv.gz")
        if market != "GOLD":
            _check_hash(path, record["inputs"]["daily_sha256"][market])
        for row in _rows(path):
            symbol = "KRX_GOLD_9999_1KG" if market == "GOLD" else row["symbol"]
            if symbol in names and date.fromisoformat(row["date"]) <= end:
                output[symbol].append(row)
    return output


def _history_ok(first: date, start: date) -> bool:
    try:
        boundary = start.replace(year=start.year - 3)
    except ValueError:
        boundary = start.replace(year=start.year - 3, day=28)
    return first <= boundary


def _symbol_bars(instrument, rows, calendar, start, end, payment_dates, locked_status, delisting=None):
    raw_days = [row["date"] for row in rows]
    if len(raw_days) != len(set(raw_days)):
        raise ValueError(f"{instrument.symbol}: 중복 가격이 있습니다")
    if delisting is not None:
        if not start <= delisting.last_session <= end:
            raise ValueError("explicit delisting must lie inside the requested run")
        for row in rows:
            day = date.fromisoformat(row["date"])
            if delisting.last_session < day <= end and not _delisting_marker(row, day, delisting):
                raise ValueError("source prices contradict the explicit last normal delisting session")
        end = delisting.last_session
        rows = [row for row in rows if date.fromisoformat(row["date"]) <= end]
    rows = sorted(rows, key=lambda row: row["date"])
    days = [date.fromisoformat(row["date"]) for row in rows]
    if not days or not _history_ok(days[0], start):
        raise ValueError(f"{instrument.symbol}: 중복 가격 또는 시작 전 3년 이력이 부족합니다")
    output, previous = [], None
    dated = dict(zip(days, rows))
    sessions = sorted(calendar)
    expected = [day for day in sessions if start <= day <= end]
    missing = [day for day in expected if day not in dated]
    if missing:
        raise ValueError(f"{instrument.symbol}: 요청 기간 세션 누락 {missing[0]}..{missing[-1]}; "
                         "상장폐지·거래정지·자료 말미 누락을 확인해야 하며 가격 동결 평가는 허용하지 않습니다")
    for row, day in zip(rows, days):
        close, volume = decimal(row["close"]), decimal(row["volume"])
        if close <= 0 or volume < 0:
            raise ValueError("종가·거래량이 잘못됐습니다")
        if start <= day <= end:
            index = bisect_left(sessions, day)
            prior = sessions[max(0, index - 20):index]
            if day not in calendar or len(prior) != 20 or any(stamp not in dated for stamp in prior):
                raise ValueError("실제 거래 달력 또는 직전 20거래일 거래대금이 없습니다")
            adv = sum((decimal(dated[stamp]["close"]) * decimal(dated[stamp]["volume"])
                       for stamp in prior), ZERO) / 20
            lower = Decimal("1000000") if instrument.market == "US" else Decimal("100000000")
            if not output and instrument.market != "GOLD" and adv < lower:
                raise ValueError(f"{instrument.symbol}: 시작 전 거래대금 하한 미달")
            if row.get("corp_action_ok", "True") != "True":
                raise ValueError("잘못된 기업행사 자료는 실행할 수 없습니다")
            ohlc = [decimal(row[name]) for name in ("open", "high", "low", "close")]
            if min(ohlc) <= 0:
                raise ValueError(f"{instrument.symbol} {day}: OHLC 0/음수 자료는 실행할 수 없습니다 "
                                 f"(volume={volume}, ohlc_ok={row.get('ohlc_ok', 'unknown')}); "
                                 "직전 가격으로 자동 보정하지 않습니다")
            flat = len(set(ohlc)) == 1
            key = (instrument.symbol, day)
            if flat and volume > 0 and key not in locked_status:
                raise ValueError("단일 가격 일봉은 가격제한폭 고정 여부의 명시 확인이 필요합니다")
            dividend = decimal(row.get("dividend") or "0")
            ratio = decimal(row.get("split_ratio") or "1")
            opened, closed = calendar[day]
            output.append(Bar(instrument.symbol, day, opened, closed, *ohlc, volume,
                              previous_close=previous, adv20=adv, limit_locked=locked_status.get(key, False),
                              dividend=dividend, dividend_pay_at=payment_dates.get(key), split_ratio=ratio))
        previous = close
    if not output:
        raise ValueError(f"{instrument.symbol}: 요청 기간의 가격이 없습니다")
    return output


def _delisting_marker(row, day, event):
    # M1의 폐지 표시행은 정상 일봉이 아니다. 명시 폐지일의 0 OHLC/거래량만 허용한다.
    return (day == event.delisting_date and decimal(row["close"]) > ZERO
            and all(decimal(row[name]) == ZERO for name in ("open", "high", "low", "volume")))


def load_m1_inputs(data_root, symbols: list[str], start: date, end: date, split_csv, *,
                   payment_dates=None, locked_status=None, delistings=(), input_hashes=None,
                   domestic_settlement="legacy_same_day", max_fx_age_days=None) -> M1Inputs:
    """명시한 탐색 상품을 읽는다. 성과 탐색·보류 실행·원본 갱신은 수행하지 않는다.

    locked_status[(symbol,date)]는 단일 가격 일봉의 제한폭 고정 여부를 확인한 bool이다.
    미확인 단일 가격 일봉을 정상 체결 가능일로 추정하지 않는다. 지급일은 확인값만 받는다.
    """
    if type(start) is not date or type(end) is not date or start > end:
        raise ValueError("start/end는 순서가 맞는 date여야 합니다")
    if not symbols or len(set(symbols)) != len(symbols) or any(not isinstance(s, str) or not s for s in symbols):
        raise ValueError("고유한 상품 코드 목록이 필요합니다")
    root = Path(data_root)
    delistings = tuple(delistings)
    payment_dates, locked_status = payment_dates or {}, locked_status or {}
    for mapping in (payment_dates, locked_status):
        for key in mapping:
            if (not isinstance(key, tuple) or len(key) != 2 or key[0] not in symbols
                    or type(key[1]) is not date or not start <= key[1] <= end):
                raise ValueError("기업행사 확인 입력은 요청한 상품·기간의 (symbol,date)여야 합니다")
    if any(type(value) is not bool for value in locked_status.values()):
        raise ValueError("제한폭 확인값은 bool이어야 합니다")
    for at in payment_dates.values():
        aware(at)
    record, selected, metadata, bundle = _snapshot(root, split_csv, symbols)
    notices = _notice_map(delistings, selected)
    extra_paths = _extra_paths(root, any(row["market"] == "GOLD" for row in selected.values()))
    if input_hashes is not None:
        if not set(extra_paths) <= set(input_hashes):
            raise ValueError("M5 needs pinned calendar/FX/gold hashes before loading data")
        for name, path in extra_paths.items():
            _check_hash(path, input_hashes[name])
    krx = _calendar(root / "fx" / "calendar_krx.csv")
    nyse = _calendar(root / "fx" / "calendar_nyse.csv")
    required_calendars = {"KRX": krx}  # 초기·월별 환전 일정에도 국내 달력이 필요하다.
    if any(row["market"] == "US" for row in selected.values()):
        required_calendars["NYSE"] = nyse
    for market, calendar in required_calendars.items():
        if not calendar or end > max(calendar):
            raise ValueError(f"{market}: 요청 종료일 {end}이 저장된 달력의 확인 범위를 넘습니다")
    if start not in set(krx) | set(nyse):
        raise ValueError("시작일은 실제 통합 거래일이어야 합니다")
    if any(row["market"] == "US" for row in selected.values()) and start not in krx:
        raise ValueError("국내 휴장일의 초기 환전 정의는 미확정입니다. 국내 개장일로 시작하세요")
    daily = _prices(root, bundle, record, selected, end)
    instruments, bars, warnings = [], [], []
    for symbol in symbols:
        market = selected[symbol]["market"]
        if market not in {"KR", "US", "GOLD"}:
            raise ValueError("지원하지 않는 시장입니다")
        row = metadata.get((market, symbol), {})
        if market != "GOLD" and row.get("d1_eligible") != "True":
            raise ValueError("D1 상품 유형 필터를 통과하지 못했습니다")
        basis = "raw" if market == "GOLD" else daily[symbol][0]["price_basis"] if daily[symbol] else ""
        if any(r.get("price_basis", basis) != basis for r in daily[symbol]):
            raise ValueError("상품 가격 기준이 중간에 바뀌었습니다")
        instrument = Instrument(symbol, market, "USD" if market == "US" else "KRW",
                                row.get("kr_tax_class") or "unknown", basis)
        instruments.append(instrument)
        bars.extend(_symbol_bars(instrument, daily[symbol], nyse if market == "US" else krx,
                                 start, end, payment_dates, locked_status, notices.get(symbol)))
        if market == "KR":
            warnings.append(f"{symbol}: 분배금 반영 미확인 수정가격, 호가는 가상 근사이며 실제 세후 판정 불가")
    rates = [FxRate(_stamp(row["available_at_utc"]), decimal(row["value"]))
             for row in _rows(root / "fx" / "fx_usdkrw_smbs_mar.csv")]
    validate_delistings(delistings, {item.symbol: item for item in instruments}, bars)
    if domestic_settlement == "t_plus_2":
        _check_settlement_tail(instruments, bars, {"KR": krx, "US": nyse})
    union = sorted(day for day in set(krx) | set(nyse) if start <= day <= end)
    evaluations = [datetime.combine(day, time(18), KST) for day in union]
    final_close = max(bar.close_at for bar in bars).astimezone(KST)
    terminal = datetime.combine(final_close.date(), time(18), KST)
    if terminal < final_close:
        terminal += timedelta(days=1)
    if terminal not in evaluations:
        evaluations.append(terminal)
    first_kr = {}
    for day in sorted(krx):
        if start <= day <= end:
            first_kr.setdefault((day.year, day.month), day)
    conversions = [datetime.combine(day, time(10), KST) for day in first_kr.values()]
    initial = datetime.combine(start, time(10), KST)
    if initial not in conversions:
        conversions.insert(0, initial)
    provenance = {"m1_split_protocol_sha256": record["protocol_sha256"],
                  "run_protocol_sha256": protocol.protocol_hash(protocol.load()),
                  "split_sha256": record["split_sha256"], "price_manifest_sha256": record["inputs"]["price_manifest_sha256"],
                  "metadata_sha256": record["inputs"]["metadata_sha256"],
                  "daily_sha256": dict(record["inputs"]["daily_sha256"]),
                  "additional_inputs_sha256": {name: _sha(path) for name, path in extra_paths.items()},
                  "symbols": list(symbols), "start": start.isoformat(), "end": end.isoformat(),
                  "payment_dates": [{"symbol": key[0], "date": key[1].isoformat(), "at": value.isoformat()}
                                    for key, value in sorted(payment_dates.items())],
                  "locked_status": [{"symbol": key[0], "date": key[1].isoformat(), "locked": value}
                                    for key, value in sorted(locked_status.items())],
                  "market_rules_sha256": _sha(Path(__file__).with_name("market_rules_v1.json")),
                  "settlement_availability_assumption": "명시한 시장 close_utc를 사용한 보수적 가정, 실제 결제시각을 관측한 값이 아님",
                  "evaluation_calendar_sha256": hashlib.sha256("\n".join(at.isoformat() for at in evaluations).encode()).hexdigest(),
                  "domestic_settlement": domestic_settlement, "max_fx_age_days": max_fx_age_days,
                  "delisting_sources_sha256": {event.symbol: event.source_sha256 for event in delistings}}
    if any(bar.dividend and bar.dividend_pay_at is None for bar in bars):
        warnings.append("미국 분배금 지급일 미확인: 미수금으로만 기록하며 현금 재투자에 사용할 수 없음")
    return M1Inputs(instruments, sorted(bars, key=lambda b: (b.open_at, symbols.index(b.symbol))), rates,
                    evaluations, conversions, list(nyse), {day: pair[1] for day, pair in nyse.items()}, provenance, warnings,
                    list(krx), {day: pair[1] for day, pair in krx.items()}, domestic_settlement, max_fx_age_days, tuple(delistings))


def _extra_paths(root, has_gold):
    result = {name: root / "fx" / name for name in
              ("calendar_krx.csv", "calendar_nyse.csv", "fx_usdkrw_smbs_mar.csv")}
    if has_gold:
        result["gold_9999_1kg_daily.csv"] = root / "krx_gold" / "gold_9999_1kg_daily.csv"
    return result


def _notice_map(events, selected):
    notices = {}
    for event in events:
        if not isinstance(event, DelistingEvent) or event.symbol not in selected or event.symbol in notices:
            raise ValueError("delistings must identify unique requested instruments")
        if selected[event.symbol]["market"] != "KR":
            raise ValueError("delisting liquidation supports KR ETFs only")
        notices[event.symbol] = event
    return notices


def _check_settlement_tail(instruments, bars, calendars):
    markets = {item.symbol: item.market for item in instruments}
    schedules = {market: sorted(calendar) for market, calendar in calendars.items()}
    for bar in bars:
        market = markets[bar.symbol]
        if market == "GOLD":
            continue
        sessions = schedules[market]
        lag = 2 if market == "KR" else 1 if bar.session_date >= date(2024, 5, 28) else 2 if bar.session_date >= date(2017, 9, 5) else 3
        index = bisect_left(sessions, bar.session_date)
        if index + lag >= len(sessions):
            raise ValueError(f"{bar.symbol}: actual settlement calendar tail is incomplete")
