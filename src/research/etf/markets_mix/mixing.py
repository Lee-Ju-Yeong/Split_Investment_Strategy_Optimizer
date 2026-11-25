"""혼합 조립(설계 v1.1 4절): 두 몫의 일별 원화 계열을 경로 짝마다 더해 혼합 전략·혼합 보유 계좌를 만든다.

1억 원 결과를 반으로 나누지 않는다: 각 몫은 실제 5천만 원으로 주문한 단위의 계열이다. 이체·재조정은 없으므로 합산은
같은 평가 시각의 순자산·현금·미수금·ETF 평가액을 그대로 더하는 것뿐이다. 지표는 승인된 지표 함수(kodex_core.metrics)로 구한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, localcontext
from itertools import product

from ..engine_types import ONE, ZERO, Instrument
from ..kodex_core.metrics import account_metrics, segment_metrics


SUMMED = ("equity_krw", "cash_krw", "receivables_krw")
PAIR = "|"


@dataclass(frozen=True)
class Frame:
    """한 계좌의 지표 틀: 시작 원금, 첫 시가 시각, 고정 설정의 구간, 종전 비중 한도(표기용)."""

    initial: Decimal
    started_at: datetime
    segments: tuple
    exposure_min: Decimal


def moment(value) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def rows_of(at, block, symbol) -> list:
    """봉인한 일별 계열(문자열)을 지표 함수가 읽는 평가 행으로 되돌린다. 길이가 다르면 오류다."""
    columns = zip(at, block["equity"], block["cash"], block["receivables"], block["holdings"], strict=True)
    return [{"at": moment(when), "equity_krw": Decimal(equity), "cash_krw": Decimal(cash),
             "receivables_krw": Decimal(receivables), "holdings_krw": {symbol: Decimal(holdings)},
             "positions": [], "prices": {}, "fx_rate": ONE} for when, equity, cash, receivables, holdings in columns]


def _sum(rows) -> dict:
    holdings = {}
    for row in rows:
        for symbol, value in row["holdings_krw"].items():
            holdings[symbol] = holdings.get(symbol, ZERO) + value
    return {"at": rows[0]["at"], **{key: sum((row[key] for row in rows), ZERO) for key in SUMMED},
            "holdings_krw": holdings, "positions": [], "prices": {}, "fx_rate": ONE}


def combine(legs) -> list:
    """같은 평가 시각의 몫들을 더한다. 평가 시각이 하나라도 다르면 더하지 않고 멈춘다."""
    times = [[row["at"] for row in leg] for leg in legs]
    if not legs or any(found != times[0] for found in times[1:]):
        raise ValueError("shares of a mix must be valued at the same times")
    with localcontext(prec=80):
        return [_sum(rows) for rows in zip(*legs)]


def metrics(rows, frame: Frame) -> dict:
    """study._metrics와 같은 계산(전체 기간 지표 + 구간 지표)을 평가 행에 적용한다."""
    instruments = [Instrument(symbol, "KR", "KRW") for symbol in rows[0]["holdings_krw"]]
    options = dict(initial=frame.initial, started_at=frame.started_at, exposure_min=frame.exposure_min)
    with localcontext(prec=80):
        return {**account_metrics(rows, instruments, **options),
                "segments": segment_metrics(rows, instruments, segments=frame.segments, **options)}


def pair_keys(paths, legs: int) -> tuple:
    """경로 짝의 이름과 몫별 경로: 두 몫 × 두 경로면 네 경우."""
    return tuple((PAIR.join(pair), pair) for pair in product(paths, repeat=legs))


def account_rows(legs, name) -> list:
    """legs: ((상품, 그 경로의 계열), ...). name: 'strategy' 또는 'hold'."""
    return combine([rows_of(series["at"], series[name], symbol) for symbol, series in legs])
