"""고정 M5 묶음(etf-m5-r1)에서 단일 국내 ETF 입력을 읽는다. 원본에는 쓰지 않는다.

R2 load_m1_inputs의 '시작 전 상장 이력 3년'·'첫날 거래대금 하한' 대신 이 연구에 실제로 필요한
기준만 둔다(Astra v2 1절): 각 해 첫 거래일 직전 251거래일(변동성), 각 일봉 직전 20거래일(거래대금).
R2 입력기 자체는 바꾸지 않는다. 평가 일정은 국내 거래일 18:00 KST만 쓴다.
"""
from __future__ import annotations

from bisect import bisect_left
import copy
from datetime import date
from decimal import Decimal, localcontext
from functools import lru_cache
from types import MappingProxyType
import hashlib
from pathlib import Path

from ..engine_inputs import _calendar, _prices, _sha, _snapshot
from ..engine_types import Bar, Instrument, decimal
from ..m5_inputs import read_manifest
from ..m5_parameters import estimate_volatility
from .calendar import KrCalendar
from ..engine_strategy import _normalized_extremes
from .inputs import (DataHold, EtfInputs, InputIntegrityError, VolatilityUnavailable, ZeroVolatility, prepare,
                     volatility_digest)


ZERO = Decimal(0)


def _instrument(metadata, symbol, rows):
    row = metadata.get(("KR", symbol))
    if row is None or row.get("d1_eligible") != "True":
        raise DataHold(f"{symbol}: not an eligible domestic ETF in the pinned metadata")
    bases = {r["price_basis"] for r in rows}
    if len(bases) != 1:
        raise DataHold(f"{symbol}: price basis changes inside the history")
    return Instrument(symbol, "KR", "KRW", row.get("kr_tax_class") or "unknown", bases.pop())


def _dated(rows):
    dated = {}
    for row in rows:
        day = date.fromisoformat(row["date"])
        if day in dated:
            raise DataHold(f"{row.get('symbol', '?')} {day}: duplicate price rows")
        dated[day] = row
    return dated


def field_value(row, name, day, *, positive, symbol=None) -> Decimal:
    """입력 경계: 빈·비수치·무한·음수 값을 종목·날짜·필드가 담긴 자료 오류(DataHold)로 바꾼다."""
    symbol = symbol or row.get("symbol", "?")
    try:
        value = decimal(row[name])
    except (KeyError, TypeError, ValueError, ArithmeticError):
        raise DataHold(f"{symbol} {day} {name}: missing or non-numeric value {row.get(name)!r}") from None
    if value < 0 or (positive and value == 0):
        raise DataHold(f"{symbol} {day} {name}: {'nonpositive' if positive else 'negative'} value {value}")
    return value


def _adv20(dated, sessions, index):
    """직전 20거래일의 종가·거래량을 모두 검증한 뒤 평균 거래대금을 만든다."""
    prior = sessions[max(0, index - 20):index]
    if len(prior) != 20 or any(day not in dated for day in prior):
        raise DataHold(f"{sessions[index]}: 20 prior domestic sessions are required for the liquidity limit")
    with localcontext(prec=80):  # D7: 호출자 정밀도와 무관하게 1억 원 문턱 경계를 정확히 비교한다.
        return sum((field_value(dated[d], "close", d, positive=True) * field_value(dated[d], "volume", d, positive=False)
                    for d in prior), ZERO) / 20


def _bar(instrument, row, day, times, adv, previous, locked):
    symbol = instrument.symbol
    if row.get("corp_action_ok", "True") != "True":
        raise DataHold(f"{symbol} {day} corp_action_ok: invalid corporate-action row")
    ohlc = [field_value(row, name, day, positive=True, symbol=symbol) for name in ("open", "high", "low", "close")]
    volume = field_value(row, "volume", day, positive=False, symbol=symbol)
    if ohlc[1] < ohlc[2]:  # E5: 고가 < 저가는 입력 경계의 자료 오류
        raise DataHold(f"{symbol} {day} high/low: high {ohlc[1]} is below low {ohlc[2]}")
    if len(set(ohlc)) == 1 and volume > 0 and day not in locked:
        raise DataHold(f"{symbol} {day} ohlc: single-price bar needs explicit limit-lock confirmation")
    opened, closed = times[day]
    bar = Bar(symbol, day, opened, closed, *ohlc, volume, previous_close=previous, adv20=adv,
              limit_locked=locked.get(day, False))
    try:  # R2 체결 단계의 OHLC 허용 범위(0.62%) 검사를 입력 경계로 당긴다.
        _normalized_extremes(bar)
    except ValueError as error:
        raise DataHold(f"{symbol} {day} high/low: {error}") from error
    return bar


def _one_bar(instrument, dated, times, sessions, day, locked):
    if day not in dated:
        raise DataHold(f"{instrument.symbol} {day}: price row missing for a domestic session")
    index = bisect_left(sessions, day)
    before = dated.get(sessions[index - 1]) if index else None
    previous = None if before is None else field_value(before, "close", sessions[index - 1], positive=True,
                                                         symbol=instrument.symbol)
    return _bar(instrument, dated[day], day, times, _adv20(dated, sessions, index), previous, locked)


def _bars(instrument, dated, times, start, end, locked):
    """시작일 일봉(직전 20일 창 포함)의 자료 문제는 시작일 보류(DataHold). 그 뒤 첫 자료 문제에서는
    관측을 끊고 마지막 유효일·사유를 돌려준다(E2: 시작 판정·첫 주문 보존)."""
    sessions = sorted(times)
    window = [day for day in sessions if start <= day <= end]
    if not window:
        raise DataHold(f"no domestic sessions between {start} and {end}; judgment held")
    output, gap = [], None
    for day in window:
        try:
            output.append(_one_bar(instrument, dated, times, sessions, day, locked))
        except DataHold as error:
            if not output:
                raise
            gap = {"last_valid_session": output[-1].session_date, "first_bad_session": day,
                   "reason": str(error), "requested_end": end}
            break
    return output, gap


def _off_calendar(dated, times, since, end):
    """E3: 준비기간(251·20거래일)부터 만기까지 일봉은 있는데 달력에 없는 날."""
    return sorted(day for day in dated if since <= day <= end and day not in times)


def _history_problem(dated, sessions, symbol=None):
    """251일 모두 유한한 양의 종가·0 이상 거래량. 사유에 종목·날짜·필드를 남긴다."""
    if len(sessions) != 251:
        return "volatility requires 251 complete pre-start sessions"
    for day in sessions:
        row = dated.get(day)
        if row is None:
            return f"{symbol or '?'} {day}: price row missing in the 251-session volatility window"
        try:
            field_value(row, "close", day, positive=True, symbol=symbol)
            field_value(row, "volume", day, positive=False, symbol=symbol)
        except DataHold as error:
            return f"volatility window: {error}"
    return None


def _valid_ratios(dated, sessions):
    return [decimal(dated[b]["close"]) / decimal(dated[a]["close"]) for a, b in zip(sessions, sessions[1:])
            if decimal(dated[a]["volume"]) > 0 and decimal(dated[b]["volume"]) > 0]


def _estimate(rows, dated, times, day, symbol=None):
    """자료 문제 → VolatilityUnavailable, 유효 수익률이 모두 같아 표준편차가 0 → ZeroVolatility.
    그 밖의 R2 추정기 오류는 예상 밖 오류로 보고 전파한다(D6)."""
    sessions = sorted(d for d in times if d < day)[-251:]
    problem = _history_problem(dated, sessions, symbol)
    if problem is not None:
        return VolatilityUnavailable(problem)
    with localcontext(prec=80):
        ratios = _valid_ratios(dated, sessions)
    if len(ratios) < 200:
        return VolatilityUnavailable("volatility needs at least 200 nonzero-volume return observations")
    if len(set(ratios)) == 1:
        return ZeroVolatility(times[sessions[-1]][1])
    return estimate_volatility(rows, times, day)


def _volatility(rows, times, days, symbol=None):
    dated = _dated(rows)
    return {day: _estimate(rows, dated, times, day, symbol) for day in days}


@lru_cache(maxsize=16)
def _source(root: Path, manifest_path: Path, symbol: str, end: date):
    """같은 ETF의 시작일별 적재가 파일을 다시 읽지 않도록 고정 입력만 캐시한다."""
    try:
        manifest = read_manifest(root, manifest_path, [symbol])
        record, selected, metadata, bundle = _snapshot(root, root / manifest["split_csv"], [symbol])
        if selected[symbol]["market"] != "KR":
            raise ValueError("domestic ETF only")
        rows = tuple(MappingProxyType(dict(row)) for row in _prices(root, bundle, record, selected, end)[symbol])
        times = MappingProxyType(_calendar(root / "fx" / "calendar_krx.csv"))
    except (ValueError, OSError, KeyError) as error:
        raise InputIntegrityError(f"{symbol}: pinned input manifest/file check failed: {error}") from error
    provenance = {"input_manifest_sha256": _sha(manifest_path), "manifest_synthetic": manifest["synthetic"],
                  "price_bundle": bundle.name, "split": manifest["split_csv"],
                  "daily_sha256": record["inputs"]["daily_sha256"]["KR"],
                  "metadata_sha256": record["inputs"]["metadata_sha256"],
                  "calendar_sha256": _sha(root / "fx" / "calendar_krx.csv"), **source_identity()}
    return manifest, metadata, rows, times, provenance


def source_identity() -> dict:
    """호가표와 연결부·R2 엔진 소스 식별값."""
    here = Path(__file__).resolve().parent
    files = sorted(here.glob("*.py")) + [here / "kodex_core_v2.json"] + sorted(here.parent.glob("engine*.py")) + [
        here.parent / name for name in SOURCE_DEPENDENCIES]
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.relative_to(here.parent)} {_sha(path)}\n".encode())
    return {"market_rules_sha256": _sha(here.parent / "market_rules_v1.json"), "source_sha256": digest.hexdigest(),
            "source_files": [str(path.relative_to(here.parent)) for path in files]}


# 연결부가 실제로 import·사용하는 R2 파일(입력 명세 검사·가격 묶음 선택·프로토콜 포함).
SOURCE_DEPENDENCIES = ("baseline_engine.py", "baseline_policy.py", "baseline_report.py", "m5_parameters.py",
                       "m5_inputs.py", "price_bundle.py", "protocol.py", "protocol_v1.json", "market_rules_v1.json")


def _extras_identity(locked_status, distributions) -> dict:
    """외부에서 적용한 추가 입력(제한폭 확인값·분배금)의 내용 식별값과 대상 날짜."""
    locked = sorted((day.isoformat(), bool(value)) for day, value in (locked_status or {}).items())
    events = sorted((e.ex_date.isoformat(), str(e.per_share), e.pay_at.isoformat(), e.source) for e in distributions)
    payload = repr((locked, events)).encode()
    return {"locked_status": locked, "distributions": events, "sha256": hashlib.sha256(payload).hexdigest(),
            "present": bool(locked or events)}


def load_etf(data_root, manifest_path, symbol: str, start: date, end: date, *, locked_status=None,
             distributions=()) -> EtfInputs:
    """한 시작일의 입력. 그 시작일부터 end까지의 일봉, 시작 해부터 각 해 첫 거래일의 변동성.
    자료 오류는 이 시작일만 보류된다(다른 시작일은 따로 적재)."""
    manifest, metadata, rows, times, provenance = _source(Path(data_root).resolve(), Path(manifest_path).resolve(),
                                                          symbol, end)
    provenance = copy.deepcopy(dict(provenance))  # D1: 캐시와 공유하지 않는 사본
    times = dict(times)
    if end not in times:
        raise DataHold(f"fixed end session {end} is not in the domestic calendar; judgment held")
    calendar = KrCalendar(times)
    first = calendar.first_session_of_year(start.year)
    if first is None or start not in times:
        raise DataHold(f"domestic calendar has no sessions for start {start} (year {start.year}); judgment held")
    days = [first] + calendar.year_firsts(start, end)
    index = calendar.sessions.index(first)
    since = calendar.sessions[max(0, index - 251)]
    off = _off_calendar(_dated(rows), times, since, end)
    if off:
        raise DataHold(f"{symbol}: price rows on {[str(x) for x in off[:3]]} are not domestic calendar sessions "
                       f"(checked from preparation start {since}); judgment held")
    volatility = _volatility(list(rows), times, days, symbol)
    if isinstance(volatility[days[0]], VolatilityUnavailable):
        raise DataHold(f"calendar/history insufficient for {days[0]} volatility: {volatility[days[0]].reason}")
    missing_years = calendar.missing_years(start, end)
    if missing_years:
        raise DataHold(f"domestic calendar has no sessions in {missing_years} between {start} and {end}; judgment held")
    instrument = _instrument(metadata, symbol, rows)
    bars, gap = _bars(instrument, _dated(rows), times, start, end, dict(locked_status or {}))
    extras = _extras_identity(locked_status, tuple(distributions))
    if gap is not None:
        distributions = tuple(e for e in distributions if e.ex_date <= gap["last_valid_session"])
    return prepare(instrument, bars, calendar, volatility, distributions=distributions, data_end=gap,
                   provenance={**provenance, "price_basis": instrument.price_basis, "extra_inputs": extras,
                               "volatility_sha256": volatility_digest(volatility)})


def describe(inputs: EtfInputs) -> dict:
    """자료 로딩 점검용: 날짜 범위와 행 수만. 가격 행 내용·성과는 출력하지 않는다."""
    unavailable = [day.isoformat() for day, value in inputs.volatility.items() if isinstance(value, VolatilityUnavailable)]
    return {"symbol": inputs.instrument.symbol, "price_basis": inputs.instrument.price_basis,
            "first_session": inputs.start.isoformat(), "last_session": inputs.end.isoformat(),
            "bar_rows": len(inputs.bars),
            "domestic_sessions_in_window": len(inputs.calendar.window(inputs.start, inputs.end)),
            "volatility_days": len(inputs.volatility), "volatility_unavailable_days": unavailable,
            "basis": inputs.basis}
