"""장부 합산·투자 비중·연복리·낙폭(v1 5절)과 구간 잘라 보기(v2 1절, 판정 아님)."""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, localcontext

from ..baseline_report import YEAR_DAYS, performance
from ..engine import _utc
from ..engine_types import EngineResult, ZERO, decimal
from .calendar import KST


def combined_rows(core: EngineResult, split: EngineResult) -> list[dict]:
    """같은 평가 시각의 두 장부 현금·ETF 평가액·확정 미수금을 합친다. 수익률을 80:20 평균내지 않는다."""
    if [row["at"] for row in core.valuations] != [row["at"] for row in split.valuations]:
        raise ValueError("core and split must share evaluation times")
    rows = []
    for left, right in zip(core.valuations, split.valuations):
        holdings = {s: left["holdings_krw"][s] + right["holdings_krw"][s] for s in left["holdings_krw"]}
        rows.append({"at": left["at"], "equity_krw": left["equity_krw"] + right["equity_krw"],
                     "cash_krw": left["cash_krw"] + right["cash_krw"],
                     "receivables_krw": left["receivables_krw"] + right["receivables_krw"],
                     "holdings_krw": holdings, "positions": left["positions"] + right["positions"],
                     "prices": {**left["prices"], **right["prices"]}, "fx_rate": left["fx_rate"]})
    return rows


def _exposures(rows):
    return [sum((decimal(v) for v in row["holdings_krw"].values()), ZERO) / decimal(row["equity_krw"])
            for row in rows]


def account_metrics(rows, instruments, *, initial, started_at, exposure_min) -> dict:
    """시작 직전 자산을 첫 고점으로 포함. 연복리 = (끝/처음)^(365.25/경과일) - 1, 종가 평가·청산비용 없음."""
    with localcontext(prec=80):
        result = EngineResult(valuations=rows)
        summary = performance(result, instruments, initial_equity_krw=initial,
                              started_at=started_at, ended_at=rows[-1]["at"])
        exposures = _exposures(rows)
        return {"cagr": summary.cagr, "mdd": summary.mdd, "exposure_mean": summary.average_investment_fraction,
                "days_below_exposure_min": sum(value < exposure_min for value in exposures),
                "evaluation_days": len(rows), "ending_equity_krw": summary.ending_equity_krw}


def _peaks(rows, initial):
    peak, output = initial, []
    for row in rows:
        peak = max(peak, decimal(row["equity_krw"]))
        output.append(peak)
    return output


def segment_metrics(rows, instruments, *, initial, started_at, segments, exposure_min) -> list[dict]:
    """같은 연속 계좌를 잘라 본다(판정 아님). 수익은 직전 구간 마지막 종가 순자산부터(최초 구간은 매수 직전 자산),
    낙폭은 계좌 개설 후 최고점 대비 그 구간 중 가장 깊은 하락."""
    peaks, output = _peaks(rows, initial), []
    with localcontext(prec=80):
        for segment in segments:
            index = [i for i, row in enumerate(rows)
                     if segment.start <= _utc(row["at"]).astimezone(KST).date() <= segment.end]
            if not index:
                continue
            first, last = index[0], index[-1]
            base, since = (rows[first - 1]["equity_krw"], rows[first - 1]["at"]) if first else (initial, started_at)
            days = Decimal((_utc(rows[last]["at"]) - _utc(since)).total_seconds()) / Decimal(86400)
            exposures = _exposures(rows[first:last + 1])
            output.append({"segment": segment.label, "judgment": False,
                           "cagr": (decimal(rows[last]["equity_krw"]) / decimal(base)) ** (YEAR_DAYS / days) - 1,
                           "mdd_from_inception_peak": min(decimal(rows[i]["equity_krw"]) / peaks[i] - 1 for i in index),
                           "exposure_mean": sum(exposures, ZERO) / len(exposures),
                           "days_below_exposure_min": sum(value < exposure_min for value in exposures),
                           "evaluation_days": len(index)})
    return output


def ledger_rows(result: EngineResult, symbol: str) -> list[dict]:
    """독립 검산 도구에 넘길 원장 행: 현금·수량·종가·미수금과 엔진이 보고한 평가액."""
    return [{"at": row["at"], "cash": row["cash_krw"], "receivable": row["receivables_krw"],
             "quantity": sum(p.quantity for p in row["positions"]), "price": row["prices"].get(symbol, ZERO),
             "equity": row["equity_krw"], "holdings": row["holdings_krw"][symbol]} for row in result.valuations]


def plain(value):
    """보고용 직렬화(Decimal은 문자열)."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return plain(asdict(value))
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)
